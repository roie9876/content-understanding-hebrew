#!/usr/bin/env python3
"""
Hebrew handwriting OCR helper pipeline.

Given a scanned PDF/image containing Hebrew printed text + Hebrew handwriting,
it prepares the best possible images for a Vision LLM and runs a multi-pass,
multi-version OCR with reconciliation (voting) to improve handwriting accuracy.

It does NOT rely on a single OCR pass. For the cropped handwriting region it
creates several enhanced versions, splits them into overlapping chunks, sends
every image to a Vision LLM (GPT-5.4 on Azure), then reconciles the readings.

Handwriting recognition engine: Azure OpenAI GPT vision (default deployment
`gpt-5.4`), authenticated with Microsoft Entra ID (run `az login` first).
Tesseract is intentionally NOT used for handwriting (weak for Hebrew).

Usage
-----
# 1) No crop yet -> render the page so you can find the handwriting rectangle:
python handwriting_pipeline.py --input "sample/doc.pdf" --page 1 --dpi 400 \
    --output ./output_handwriting

# 2) Re-run with the crop rectangle "x,y,w,h" (pixels on the rendered page):
python handwriting_pipeline.py --input "sample/doc.pdf" --page 1 --dpi 400 \
    --crop "300,650,1600,220" --output ./output_handwriting
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
import time
from html import escape
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import cv2
import fitz  # PyMuPDF
import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
from azure.identity import AzureCliCredential

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
GPT_ENDPOINT = os.environ.get(
    "GPT_ENDPOINT",
    "https://medical-summary.services.ai.azure.com/openai/v1/",
).rstrip("/") + "/"
GPT_DEPLOYMENT = os.environ.get("GPT_DEPLOYMENT", "gpt-5.4")
GPT_REASONING = os.environ.get("GPT_REASONING", "medium")  # low | medium | high
SCOPE = "https://cognitiveservices.azure.com/.default"
MAX_SEND_SIDE = int(os.environ.get("MAX_SEND_SIDE", "3000"))  # cap px sent to LLM
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))  # parallel GPT calls in flight

# Speed knobs (OPT-IN). Defaults preserve full quality: per-image OCR keeps the
# same reasoning/resolution as before. Set OCR_REASONING=low and/or OCR_MAX_SIDE
# smaller to trade some accuracy for speed on easy documents. (Measured: low
# reasoning noticeably degrades hard Hebrew handwriting, so it is NOT the default.)
# The always-on free win is CU running concurrently with region OCR (run_pipeline).
OCR_REASONING = os.environ.get("OCR_REASONING", GPT_REASONING)       # per-image OCR effort
OCR_MAX_SIDE = int(os.environ.get("OCR_MAX_SIDE", str(MAX_SEND_SIDE)))  # px cap for OCR images

# Content Understanding (for the printed-text layer used in the merged Markdown).
CU_ENDPOINT = os.environ.get(
    "CU_ENDPOINT",
    "https://ai-content-understanding-resource-2416.cognitiveservices.azure.com/",
).rstrip("/") + "/"
CU_API_VERSION = os.environ.get("CU_API_VERSION", "2025-11-01")

# Domain-agnostic: the handwriting OCR uses NO built-in vocabulary and makes no
# assumption about document type. Optionally, a short printed-text context string
# (from the separate Content Understanding pipeline) may be passed as a SOFT hint
# only — it never overrides the visible handwriting strokes.

_credential = AzureCliCredential()
_token_cache: dict[str, tuple[float, str]] = {}
_token_lock = threading.Lock()


def _bearer() -> str:
    with _token_lock:
        exp, tok = _token_cache.get(SCOPE, (0.0, ""))
        if time.time() < exp - 60:
            return tok
        at = _credential.get_token(SCOPE)
        _token_cache[SCOPE] = (at.expires_on, at.token)
        return at.token


# --------------------------------------------------------------------------- #
# Image helpers (PIL <-> OpenCV)
# --------------------------------------------------------------------------- #
def _pil_to_gray_cv(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("L"))


def _gray_cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


def _to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _to_data_uri(image: Image.Image, max_side: int = MAX_SEND_SIDE) -> str:
    """PNG data URI, downscaled if larger than max_side (keeps payload sane)."""
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    b64 = base64.b64encode(_to_png_bytes(image)).decode()
    return f"data:image/png;base64,{b64}"


# --------------------------------------------------------------------------- #
# 1. Input / rendering
# --------------------------------------------------------------------------- #
def render_pdf_page(pdf_path: str, page_number: int, dpi: int) -> Image.Image:
    """Render a 1-based PDF page to a PIL image at the given DPI."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


# --------------------------------------------------------------------------- #
# 2. Region selection
# --------------------------------------------------------------------------- #
def crop_region(image: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    W, H = image.size
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    return image.crop((x0, y0, x1, y1))


# --------------------------------------------------------------------------- #
# 3. Image preprocessing -> multiple enhanced versions
# --------------------------------------------------------------------------- #
def enhance_upscaled_contrast(image: Image.Image, scale: int = 4) -> Image.Image:
    g = image.convert("L")
    g = g.resize((g.width * scale, g.height * scale), Image.LANCZOS)
    g = ImageOps.autocontrast(g, cutoff=1)
    g = ImageEnhance.Contrast(g).enhance(1.4)
    g = g.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))
    return g


def enhance_adaptive_threshold(image: Image.Image) -> Image.Image:
    gray = _pil_to_gray_cv(image)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=25, C=15,
    )
    return _gray_cv_to_pil(th)


def enhance_denoised(image: Image.Image) -> Image.Image:
    gray = _pil_to_gray_cv(image)
    # Light denoise so small-but-important Hebrew strokes survive.
    den = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    return _gray_cv_to_pil(den)


def enhance_thickened_strokes(image: Image.Image) -> Image.Image:
    gray = _pil_to_gray_cv(image)
    # Otsu: strokes become white on black, dilate slightly, invert back.
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dil = cv2.dilate(bw, np.ones((2, 2), np.uint8), iterations=1)
    return _gray_cv_to_pil(cv2.bitwise_not(dil))


def optional_remove_horizontal_lines(image: Image.Image) -> Image.Image:
    gray = _pil_to_gray_cv(image)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cols = bw.shape[1]
    horizontal_size = max(20, cols // 15)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=1)
    out = gray.copy()
    out[lines == 255] = 255  # lighten detected long horizontal lines
    return _gray_cv_to_pil(out)


def build_versions(crop: Image.Image) -> dict[str, Image.Image]:
    """Return the ordered set of enhanced versions of the handwriting crop."""
    return {
        "01_original_crop.png": crop,
        "02_upscaled_contrast.png": enhance_upscaled_contrast(crop, scale=4),
        "03_adaptive_threshold.png": enhance_adaptive_threshold(crop),
        "04_denoised.png": enhance_denoised(crop),
        "05_thickened_strokes.png": enhance_thickened_strokes(crop),
        "06_optional_line_removed.png": optional_remove_horizontal_lines(crop),
    }


# --------------------------------------------------------------------------- #
# 4. Chunking
# --------------------------------------------------------------------------- #
def split_into_overlapping_chunks(
    image: Image.Image, chunk_width: int = 900, overlap: int = 150
) -> list[Image.Image]:
    W, H = image.size
    if W <= chunk_width:
        return [image]
    step = max(1, chunk_width - overlap)
    chunks = []
    x = 0
    while x < W:
        x1 = min(x + chunk_width, W)
        chunks.append(image.crop((x, 0, x1, H)))
        if x1 >= W:
            break
        x += step
    return chunks


# --------------------------------------------------------------------------- #
# 7. Output helpers
# --------------------------------------------------------------------------- #
def save_outputs(output_dir: str, images: dict[str, Image.Image]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, img in images.items():
        img.save(out / name)


# --------------------------------------------------------------------------- #
# Printed-text context (soft hint only; comes from the separate CU pipeline)
# --------------------------------------------------------------------------- #
def load_printed_context(printed_context: str | None,
                         printed_context_file: str | None,
                         max_chars: int = 1000) -> str:
    """Return a short printed-text context string (SOFT hint only).

    Uses the direct string if given, else reads the file. Whitespace is
    normalized and the result is trimmed to max_chars. Empty when no context.
    """
    ctx = ""
    if printed_context and printed_context.strip():
        ctx = printed_context
    elif printed_context_file:
        try:
            ctx = Path(printed_context_file).read_text(encoding="utf-8")
        except Exception as exc:
            print(f"[context] could not read {printed_context_file}: {exc}", file=sys.stderr)
            ctx = ""
    ctx = " ".join(ctx.split())  # normalize whitespace
    return ctx[:max_chars].strip()


def build_context_block(printed_context: str) -> str:
    """A clearly-labeled soft-context block, or '' when no context is provided."""
    if not printed_context or not printed_context.strip():
        return ""
    return (
        "\nOptional nearby printed context. Use ONLY as soft context to resolve "
        "ambiguous strokes. Do NOT copy it as handwriting and NEVER let it override "
        "the visible handwriting:\n"
        f'"""\n{printed_context.strip()}\n"""\n'
    )


# --------------------------------------------------------------------------- #
# 5. Vision-LLM OCR
# --------------------------------------------------------------------------- #
def build_ocr_prompt(printed_context: str = "", italic: bool = False) -> str:
    context_block = build_context_block(printed_context)
    written = "handwritten or italic/cursive-printed" if italic else "handwritten"
    italic_rule = (
        "\n- The region may be ITALIC/CURSIVE PRINTED Hebrew; beware italic letter "
        "confusions: ז vs ל, ע vs צ, ד vs ר, and final letters."
        if italic else ""
    )
    return f"""You are reading Hebrew {written} text from a scanned document.

Rules:
- Read Hebrew right-to-left.
- Transcribe only the {written} text.
- Ignore printed body text, form lines, stamps, signatures, and black redactions.
- The attached image is the source of truth.
- Use printed context only to resolve ambiguous handwriting when the visible strokes support it.
- Do not copy printed context as handwriting.
- Do not invent words.
- Do not use any predefined domain vocabulary.
- If uncertain, mark the word with [?].
- Provide alternatives for uncertain words.{italic_rule}
{context_block}
Return only JSON:
{{
  "full_transcription": "...",
  "alternative_transcriptions": ["...", "..."],
  "words": [
    {{"text": "word", "confidence": 0.0, "alternatives": ["alt1", "alt2"], "reason": "short visual reason"}}
  ],
  "notes": []
}}
"""


def _gpt_output_text(resp: dict) -> str:
    if isinstance(resp.get("output_text"), str) and resp["output_text"].strip():
        return resp["output_text"]
    chunks = []
    for item in resp.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") in ("output_text", "text") and c.get("text"):
                    chunks.append(c["text"])
    return "\n".join(chunks)


def _loads_lenient(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b != -1:
        text = text[a : b + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"_parse_error": str(e), "_raw": text}


def _responses_text(content: list[dict], retries: int = 4,
                    max_tokens: int = 6000, reasoning: str | None = None) -> str:
    """POST to the Responses API and return the raw output text.

    Robust to timeouts / 429 / 5xx / empty output: retries with exponential
    backoff + jitter so a transient burst never loses the call.
    """
    payload = {
        "model": GPT_DEPLOYMENT,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": max_tokens,
        "reasoning": {"effort": reasoning or GPT_REASONING},
    }
    url = f"{GPT_ENDPOINT}responses"
    last_err = ""
    for attempt in range(retries + 1):
        backoff = min(30.0, 2.0 ** attempt) + random.uniform(0, 1.5)
        try:
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {_bearer()}", "Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=180,
            )
        except requests.exceptions.RequestException as exc:  # timeout / connection reset
            last_err = f"request error: {exc}"
            if attempt < retries:
                time.sleep(backoff)
                continue
            raise RuntimeError(f"[GPT] {last_err}")

        if r.status_code == 400 and "reasoning" in r.text.lower():
            payload.pop("reasoning", None)
            continue  # retry immediately without the reasoning param
        if r.status_code in (408, 409, 429, 500, 502, 503, 504):
            last_err = f"HTTP {r.status_code}"
            if attempt < retries:
                wait = float(r.headers.get("retry-after", backoff))
                time.sleep(wait)
                continue
            raise RuntimeError(f"[GPT] {r.status_code} after retries: {r.text[:200]}")
        if r.status_code != 200:
            raise RuntimeError(f"[GPT] {r.status_code}: {r.text[:400]}")
        text = _gpt_output_text(r.json())
        if not text.strip() and attempt < retries:
            # 200 but empty output (reasoning consumed the whole budget) -> retry
            # with a larger output budget.
            last_err = "empty output"
            payload["max_output_tokens"] = min(payload.get("max_output_tokens", max_tokens) + 4000, 32000)
            time.sleep(backoff)
            continue
        return text

    raise RuntimeError(f"[GPT] failed after retries: {last_err}")


def _responses_call(content: list[dict], retries: int = 2,
                    reasoning: str | None = None, max_tokens: int = 6000) -> dict:
    return _loads_lenient(_responses_text(content, retries, max_tokens=max_tokens, reasoning=reasoning))


def ocr_image(image: Image.Image, prompt: str) -> dict:
    # Raw per-image OCR: low reasoning + smaller image. The final visual-reconcile
    # pass keeps full reasoning/resolution, so accuracy is preserved there.
    return _responses_call([
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": _to_data_uri(image, max_side=OCR_MAX_SIDE)},
    ], reasoning=OCR_REASONING)


# --------------------------------------------------------------------------- #
# Content Understanding printed-text OCR (for the merged Markdown)
# --------------------------------------------------------------------------- #
def cu_ocr_markdown(input_path: str, timeout_s: int = 180, retries: int = 4) -> str:
    """Run Content Understanding (prebuilt-documentSearch) on the file and return
    its printed-text markdown. Uses the same Entra ID token as GPT. Retries the
    submit and tolerates transient errors while polling."""
    ext = Path(input_path).suffix.lower()
    mime = {".pdf": "application/pdf", ".png": "image/png",
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "application/octet-stream")
    data = base64.b64encode(Path(input_path).read_bytes()).decode()
    hdr = {"Authorization": f"Bearer {_bearer()}", "Content-Type": "application/json"}
    url = f"{CU_ENDPOINT}contentunderstanding/analyzers/prebuilt-documentSearch:analyze?api-version={CU_API_VERSION}"
    body = json.dumps({"inputs": [{"data": data, "mimeType": mime}]})

    r = None
    last_err = ""
    for attempt in range(retries + 1):
        backoff = min(30.0, 2.0 ** attempt) + random.uniform(0, 1.5)
        try:
            r = requests.post(url, headers=hdr, data=body, timeout=120)
        except requests.exceptions.RequestException as exc:  # timeout / DNS / reset
            last_err = str(exc)
            if attempt < retries:
                time.sleep(backoff)
                continue
            raise RuntimeError(f"[CU] submit error after retries: {last_err}")
        if r.status_code == 202:
            break
        if r.status_code in (408, 429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(float(r.headers.get("retry-after", backoff)))
            continue
        raise RuntimeError(f"[CU] analyze submit failed {r.status_code}: {r.text[:300]}")

    op = r.headers["Operation-Location"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            g = requests.get(op, headers=hdr, timeout=30).json()
        except requests.exceptions.RequestException:  # transient poll error -> keep polling
            time.sleep(2)
            continue
        st = (g.get("status") or "").lower()
        if st == "succeeded":
            contents = g.get("result", {}).get("contents", [])
            return contents[0].get("markdown", "") if contents else ""
        if st == "failed":
            raise RuntimeError(f"[CU] analysis failed: {json.dumps(g)[:300]}")
        time.sleep(2)
    raise TimeoutError("[CU] analysis timed out")


def build_merged_markdown(output_dir: str, page_name: str, input_path: str,
                          cu_markdown: str, regions: list[dict]) -> Path:
    """Fuse CU printed text + GPT handwriting/italic into ONE clean Markdown, with
    no duplication. GPT does the semantic merge using the page image as the
    authority (corrects garbled italic, inserts handwriting in red)."""
    out = Path(output_dir)
    page = Image.open(out / page_name).convert("RGB")
    hw_list = "\n".join(
        f"- [{r.get('label', '')}]: {(r.get('final_transcription') or '').strip()}"
        for r in regions if (r.get("final_transcription") or "").strip()
    ) or "(none)"

    instruction = f"""You are merging OCR outputs of ONE scanned Hebrew document into a single,
clean Markdown transcription of the WHOLE page in natural reading order.

Inputs:
A) PRINTED-TEXT OCR (below). Reads regular printed Hebrew well, but GARBLES
   italic/cursive printed Hebrew (confuses ז/ל, ע/צ) and leaves handwriting blank.
B) HANDWRITING/ITALIC values (below), extracted separately by a vision model.
C) The page image (attached) - it is the AUTHORITY. Use it to fix garbled italic
   and to place handwriting correctly.

Rules:
- Output the full document text ONCE, in reading order. NO duplication - never
  repeat the same sentence from both sources.
- Use source A for regular printed text.
- Fix garbled italic printed Hebrew using the image + context.
- Insert each handwritten value at its correct place, wrapped in
  <span style="color:#d00">…</span> so handwriting shows in red.
- Keep [?] markers on genuinely uncertain words.
- Output ONLY the merged Markdown (Hebrew, right-to-left). No preamble, no notes.

=== A) PRINTED-TEXT OCR ===
{cu_markdown}

=== B) HANDWRITING / ITALIC VALUES (red) ===
{hw_list}
"""
    text = _responses_text([
        {"type": "input_text", "text": instruction},
        {"type": "input_image", "image_url": _to_data_uri(page)},
    ], max_tokens=16000, reasoning="low").strip()
    if not text:
        raise RuntimeError("merge returned empty output (raise --max_tokens or check quota)")
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    path = out / "merged.md"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 6. Reconciliation (text-only fallback + final visual adjudication)
# --------------------------------------------------------------------------- #
def _attach_source_images(final: dict, usable: list[dict]) -> None:
    """For each final word, record which candidate readings contained it."""
    for w in final.get("words", []) or []:
        txt = str(w.get("text", "")).replace("[?]", "").strip()
        w["source_images"] = [
            r["source"] for r in usable
            if txt and txt in (r.get("ocr") or {}).get("full_transcription", "")
        ]


def reconcile_ocr_results(results: list[dict], printed_context: str = "") -> dict:
    """Text-only reconciliation of candidate readings (the fallback path).

    `results` items: {"source": "<image name>", "ocr": {full_transcription, words, ...}}.
    Printed context is a SOFT hint only and is never copied as handwriting.
    """
    usable = [r for r in results if (r.get("ocr") or {}).get("full_transcription")]
    if not usable:
        return {"final_transcription": "", "words": [], "notes": ["no OCR output"]}

    candidates = "\n\n".join(
        f"[{r['source']}]\n{(r.get('ocr') or {}).get('full_transcription', '')}" for r in usable
    )
    context_block = build_context_block(printed_context)
    prompt = f"""These are multiple OCR readings of the SAME Hebrew handwriting,
each produced from a differently enhanced image or image chunk. Reconcile them
into the single most accurate Hebrew transcription.

Rules:
- Read right-to-left.
- Prefer a word candidate that appears in several readings.
- Keep uncertainty explicit: mark uncertain words with [?] and give alternatives.
- Do not invent words that no reading supports.
- Do not use any predefined domain vocabulary.
- Printed context (if any) is a soft hint only; never copy it as handwriting.
{context_block}
Candidate readings:
{candidates}

Return ONLY a single JSON object:
{{
  "final_transcription": "best reconciled Hebrew text, right-to-left",
  "words": [
    {{"text": "word", "confidence": 0.0, "alternatives": ["alt1"]}}
  ],
  "notes": ["explanation of any uncertainty"]
}}
"""
    final = _responses_call([{"type": "input_text", "text": prompt}])
    if "final_transcription" not in final:
        # Parse failure -> longest candidate as a last resort.
        best = max(usable, key=lambda r: len(r["ocr"].get("full_transcription", "")))
        return {
            "final_transcription": best["ocr"].get("full_transcription", ""),
            "words": best["ocr"].get("words", []),
            "notes": ["text reconcile parse failed; used longest candidate"],
        }
    _attach_source_images(final, usable)
    return final


# Representative images for the final visual adjudication (never send them all).
_VISUAL_PREFERRED = [
    "01_original_crop.png", "02_upscaled_contrast.png", "04_denoised.png",
    "05_thickened_strokes.png", "06_optional_line_removed.png",
]


def _select_visual_images(
    image_tasks: list[tuple[str, Image.Image]], results: list[dict], max_images: int = 7,
) -> list[tuple[str, Image.Image]]:
    """Pick a small, representative image set for visual reconciliation: the key
    enhanced versions + a few chunks (chunks with non-empty OCR text first)."""
    text_by_source = {r["source"]: (r.get("ocr") or {}).get("full_transcription", "") for r in results}
    imgs_by_name = dict(image_tasks)
    selected = [(n, imgs_by_name[n]) for n in _VISUAL_PREFERRED if n in imgs_by_name]
    chunk_names = [n for n, _ in image_tasks if n.startswith("chunk_")]
    chunk_names.sort(key=lambda n: (not text_by_source.get(n, "").strip(), n))  # non-empty first
    for n in chunk_names:
        if len(selected) >= max_images:
            break
        selected.append((n, imgs_by_name[n]))
    return selected[:max_images]


def visual_reconcile_ocr_results(
    image_tasks: list[tuple[str, Image.Image]], results: list[dict],
    printed_context: str = "",
) -> dict:
    """Final VISUAL adjudication: re-show the best images + candidate readings to
    the Vision LLM and let it decide from the strokes. The images are the source
    of truth; candidate readings and printed context are hints only.
    """
    selected = _select_visual_images(image_tasks, results)
    if not selected:
        raise RuntimeError("visual reconcile: no images to show")
    usable = [r for r in results if (r.get("ocr") or {}).get("full_transcription")]

    candidates = "\n\n".join(
        f"[{r['source']}]\n{(r.get('ocr') or {}).get('full_transcription', '')}" for r in usable
    ) or "(none)"
    context_block = build_context_block(printed_context)
    instruction = f"""You are doing final visual adjudication of Hebrew handwriting OCR.

You will receive:
1. Several enhanced image versions of the SAME handwritten region.
2. Candidate OCR readings from previous passes.
3. Optional nearby printed context.

Rules:
- The images are the source of truth.
- Candidate readings are hints only.
- Printed context is a hint only.
- Do not majority-vote blindly.
- Look at the handwriting strokes visually.
- Read Hebrew right-to-left.
- Ignore printed text unless it helps understand the form field.
- Do not copy printed context as handwriting.
- Do not invent unsupported words.
- Mark uncertain words with [?].
- Return JSON only.
{context_block}
Candidate OCR readings (hints only):
{candidates}

Return ONLY a single JSON object:
{{
  "final_transcription": "best Hebrew transcription, right-to-left",
  "alternative_transcriptions": ["..."],
  "words": [
    {{"text": "word", "confidence": 0.0, "alternatives": ["alt1"], "reason": "short visual reason"}}
  ],
  "notes": []
}}
"""
    content: list[dict] = [{"type": "input_text", "text": instruction}]
    for name, img in selected:
        content.append({"type": "input_text", "text": f"Image: {name}"})
        content.append({"type": "input_image", "image_url": _to_data_uri(img)})

    final = _responses_call(content)
    if "final_transcription" not in final:
        raise RuntimeError("visual reconcile: no final_transcription in output")
    final["_selected_images"] = [n for n, _ in selected]
    _attach_source_images(final, usable)
    return final


# --------------------------------------------------------------------------- #
# Region detection + per-region processing
# --------------------------------------------------------------------------- #
def refine_box_to_ink(
    page_img: Image.Image, box: list[int], vpad_frac: float = 0.35,
) -> list[int]:
    """Expand an approximate box to cover the actual handwriting strokes.

    Vision-LLM boxes often under-cover. Within the box's vertically-padded band
    we find ink connected components (ignoring long printed lines) and chain the
    ones separated by small word-gaps, starting from components that overlap the
    original box. Isolated fields (numbers) stay tight; long note lines fill out.
    """
    W, H = page_img.size
    x, y, w, h = box
    vpad = int(h * vpad_frac)
    y0, y1 = max(0, y - vpad), min(H, y + h + vpad)
    gray = np.array(page_img.convert("L"))[y0:y1, :]
    if gray.size == 0:
        return box

    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Drop long horizontal printed lines (form underlines) and tiny speckle.
    hsize = max(40, W // 12)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (hsize, 1))
    ink = cv2.subtract(ink, cv2.morphologyEx(ink, cv2.MORPH_OPEN, hk))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    n, _lbl, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    min_area = max(25, int((y1 - y0) * 0.6))
    comps = [
        (int(stats[j, cv2.CC_STAT_LEFT]),
         int(stats[j, cv2.CC_STAT_LEFT] + stats[j, cv2.CC_STAT_WIDTH]))
        for j in range(1, n) if stats[j, cv2.CC_STAT_AREA] >= min_area
    ]
    if not comps:
        return box

    # Seed with components overlapping the original box; else the nearest one.
    seeds = [c for c in comps if c[0] < x + w and c[1] > x]
    if not seeds:
        cx = x + w // 2
        seeds = [min(comps, key=lambda c: abs((c[0] + c[1]) // 2 - cx))]
    cur_left = min(c[0] for c in seeds)
    cur_right = max(c[1] for c in seeds)

    gap = max(25, int(0.7 * h))       # bridges word spaces, not big printed gaps
    max_w = min(W, int(w * 4) + 40)   # cap growth: generous for lines, tight for fields
    changed = True
    while changed:
        changed = False
        for cl, cr in comps:
            if cr < cur_left and (cur_left - cr) <= gap and (cur_right - cl) <= max_w:
                cur_left, changed = cl, True
            elif cl > cur_right and (cl - cur_right) <= gap and (cr - cur_left) <= max_w:
                cur_right, changed = cr, True
            elif cl < cur_left and cr > cur_right - gap \
                    and (max(cur_right, cr) - min(cur_left, cl)) <= max_w:
                cur_left, cur_right, changed = min(cur_left, cl), max(cur_right, cr), True

    cur_left = max(0, min(cur_left, x) - 8)
    cur_right = min(W, max(cur_right, x + w) + 8)
    return [cur_left, y0, cur_right - cur_left, y1 - y0]


def detect_handwriting_regions(page_img: Image.Image, det_max_side: int = 2000) -> list[dict]:
    """Ask the Vision LLM to locate every handwritten region on the page.

    Returns a list of {"label": str, "box": [x, y, w, h]} in full-res pixels.
    Each approximate box is then refined to the actual handwriting ink.
    """
    W, H = page_img.size
    scale = min(1.0, det_max_side / max(W, H))
    det_w, det_h = int(W * scale), int(H * scale)
    det_img = page_img.resize((det_w, det_h), Image.LANCZOS) if scale < 1.0 else page_img

    prompt = f"""This image is a scanned Hebrew document, {det_w}x{det_h} pixels.
Find EVERY distinct region that contains true HANDWRITTEN text filled in by a
person with a pen (irregular strokes, uneven baseline, often blue/black ink
entered into a blank field or margin). Include short handwritten entries (dates,
apartment/building numbers, ID numbers, phone numbers, emails, names) AND longer
handwritten notes. Treat each separate handwritten line or field as its own region.

DO NOT select machine-printed text, even when it is styled to look cursive:
- Ignore ITALIC / slanted / cursive PRINTED text. Printed text has a uniform
  typeface, consistent stroke width, perfectly even baseline and letter spacing,
  and usually forms full justified paragraphs or legal clauses. This is NOT
  handwriting even if it slants — it belongs to the printed body of the document.
- Also ignore normal upright printed text, logos, printed form lines, headings,
  stamps, signatures, and black redaction boxes.
If unsure whether a block is handwriting or an italic printed paragraph, treat it
as printed and DO NOT include it.

Return ONLY a single JSON object with PIXEL coordinates in THIS image size:
{{"regions": [{{"label": "short english description", "box": [x, y, width, height]}}]}}
Each box should tightly cover the handwriting with a small margin."""

    resp = _responses_call([
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": _to_data_uri(det_img)},
    ])

    inv = 1.0 / scale
    pad_x, pad_y = W * 0.008, H * 0.006
    regions: list[dict] = []
    for r in (resp.get("regions") or []):
        box = r.get("box")
        if not (isinstance(box, list) and len(box) == 4):
            continue
        try:
            x, y, w, h = (float(v) * inv for v in box)
        except (TypeError, ValueError):
            continue
        bx, by = max(0, int(x - pad_x)), max(0, int(y - pad_y))
        bw, bh = int(w + 2 * pad_x), int(h + 2 * pad_y)
        bw, bh = min(bw, W - bx), min(bh, H - by)
        if bw <= 10 or bh <= 10:
            continue
        refined = refine_box_to_ink(page_img, [bx, by, bw, bh])
        regions.append({"label": str(r.get("label") or "handwriting"), "box": refined})
    return regions


def draw_regions_overlay(page_img: Image.Image, regions: list[dict]) -> Image.Image:
    """Draw numbered rectangles over the page so the user can verify regions."""
    arr = cv2.cvtColor(np.array(page_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    for i, r in enumerate(regions, 1):
        x, y, w, h = r["box"]
        cv2.rectangle(arr, (x, y), (x + w, y + h), (0, 0, 255), 4)
        cv2.putText(arr, str(i), (x + 6, max(30, y + 42)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _slug(label: str, maxlen: int = 24) -> str:
    s = "".join(c if (c.isascii() and (c.isalnum() or c in "-_")) else "_" for c in label.strip())
    s = "_".join(filter(None, s.split("_")))
    return (s or "region")[:maxlen]


def build_merged_html(
    output_dir: str, page_name: str, regions: list[dict],
    display_width: int = 1400, cu_markdown: str = "",
) -> Path:
    """Self-contained HTML: the printed page (black) with extracted handwriting
    overlaid in RED at each region's position, plus a red transcription list."""
    out = Path(output_dir)
    img = Image.open(out / page_name).convert("RGB")
    W, H = img.size
    scale = display_width / W
    disp = img.resize((display_width, int(H * scale)), Image.LANCZOS)
    b64 = base64.b64encode(_to_png_bytes(disp)).decode()

    boxes, rows = [], []
    for i, r in enumerate(regions, 1):
        x, y, w, h = r["box"]
        text = escape((r.get("final_transcription") or "").strip())
        lx, ly, lw, lh = x * scale, y * scale, w * scale, h * scale
        fs = max(11, min(int(lh * 0.5), 24))
        boxes.append(
            f'<div class="hw" style="left:{lx:.0f}px;top:{ly:.0f}px;'
            f'width:{lw:.0f}px;min-height:{lh:.0f}px;font-size:{fs}px" '
            f'title="region {i}: {escape(str(r.get("label", "")))}">{text}</div>'
        )
        rows.append(f'<tr><td>{i}</td><td>{escape(str(r.get("label", "")))}</td>'
                    f'<td class="hwtext">{text}</td></tr>')

    printed = ""
    if cu_markdown.strip():
        printed = ('<h3>Printed text (Content Understanding, black)</h3>'
                   f'<div class="printed">{escape(cu_markdown)}</div>')

    html = f"""<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<title>Merged document - handwriting in red</title>
<style>
 body{{font-family:Arial,Helvetica,sans-serif;margin:20px;background:#eee;color:#111}}
 .legend{{margin:0 0 12px;font-size:15px}}
 .legend b{{color:#d00}}
 .page{{position:relative;width:{display_width}px;background:#fff;box-shadow:0 0 10px #888}}
 .page>img{{display:block;width:{display_width}px;height:auto}}
 .hw{{position:absolute;color:#d00;font-weight:700;direction:rtl;text-align:right;
      background:rgba(255,255,255,.55);border:1px solid rgba(221,0,0,.45);
      border-radius:3px;padding:0 3px;line-height:1.15;box-sizing:border-box;overflow:visible}}
 table{{border-collapse:collapse;margin-top:20px;background:#fff}}
 td,th{{border:1px solid #bbb;padding:5px 10px;text-align:right}}
 .hwtext{{color:#d00;font-weight:700;direction:rtl}}
 .printed{{white-space:pre-wrap;direction:rtl;background:#fff;padding:16px;
           margin-top:8px;border:1px solid #bbb;max-width:{display_width}px}}
</style></head><body>
<div class="legend"><b>Red</b> = handwriting extracted by GPT vision &nbsp;|&nbsp;
 black background image = original printed document.</div>
<div class="page"><img src="data:image/png;base64,{b64}" alt="page">{''.join(boxes)}</div>
<h3>Extracted handwriting (red)</h3>
<table><tr><th>#</th><th>Region</th><th>Handwriting</th></tr>{''.join(rows)}</table>
{printed}
</body></html>"""
    path = out / "merged.html"
    path.write_text(html, encoding="utf-8")
    return path


_HEBREW_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Hebrew.ttc",
]


def _hebrew_font(size: int) -> ImageFont.FreeTypeFont:
    for p in _HEBREW_FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def build_merged_jpg(output_dir: str, page_name: str, regions: list[dict], quality: int = 90) -> Path:
    """Raster (JPG) version of the merged doc: the printed page with the extracted
    handwriting drawn in RED (bidi-reordered) at each region's position."""
    try:
        from bidi.algorithm import get_display
    except Exception:
        def get_display(t):  # fallback: no reordering
            return t

    out = Path(output_dir)
    base = Image.open(out / page_name).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pad = 8

    for r in regions:
        try:
            x, y, w, h = r["box"]
            # Collapse any newlines/extra spaces -> single line (PIL can't measure
            # or anchor multiline text).
            text = " ".join((r.get("final_transcription") or "").split())
            if not text:
                continue
            disp = get_display(text)
            # Fit font to box: cap by height, then shrink to fit width.
            fs = max(14, min(int(h * 0.6), 90))
            font = _hebrew_font(fs)
            tw = draw.textlength(disp, font=font)
            if tw > (w - 2 * pad) and tw > 0:
                fs = max(12, int(fs * (w - 2 * pad) / tw))
                font = _hebrew_font(fs)
            # Semi-transparent white panel + red border over the region.
            draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255, 150),
                           outline=(210, 0, 0, 230), width=3)
            # Red text, right-aligned (RTL) and vertically centered.
            draw.text((x + w - pad, y + h / 2), disp, font=font,
                      fill=(210, 0, 0, 255), anchor="rm")
        except Exception as exc:
            print(f"[merge-jpg] skipped a region: {exc}", file=sys.stderr)

    # Legend banner (top-left).
    lf = _hebrew_font(34)
    draw.rectangle([20, 20, 760, 78], fill=(255, 255, 255, 220), outline=(210, 0, 0, 230), width=2)
    draw.text((32, 30), "Red = extracted handwriting  |  black = printed document",
              font=_hebrew_font(26), fill=(210, 0, 0, 255))

    merged = Image.alpha_composite(base, overlay).convert("RGB")
    path = out / "merged.jpg"
    merged.save(path, "JPEG", quality=quality)
    return path


def prepare_region_images(
    page_img: Image.Image, region: dict, out_dir: str,
    ocr_mode: str, chunk_width: int, overlap: int,
) -> list[tuple[str, Image.Image]]:
    """Crop + enhance + chunk a region, save the images, and return the list of
    (source_name, image) OCR tasks to run (order preserved)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x, y, w, h = region["box"]
    crop_img = crop_region(page_img, x, y, w, h)

    versions = build_versions(crop_img)
    save_outputs(out_dir, versions)
    raw_chunks = split_into_overlapping_chunks(crop_img, chunk_width, overlap)
    chunk_images = {
        f"chunk_{i}.png": enhance_upscaled_contrast(c, scale=3)
        for i, c in enumerate(raw_chunks)
    }
    save_outputs(out_dir, chunk_images)

    tasks: list[tuple[str, Image.Image]] = []
    if ocr_mode in ("full", "both"):
        tasks += list(versions.items())
    if ocr_mode in ("chunks", "both"):
        tasks += list(chunk_images.items())
    return tasks


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_pipeline(
    input_path: str, page: int, dpi: int,
    crops: list[tuple[int, int, int, int]] | None,
    output_dir: str, chunk_width: int, overlap: int, ocr_mode: str,
    auto_detect: bool = False, max_regions: int | None = None,
    printed_context: str = "", italic: bool = False, merge_md: bool = False,
) -> int:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- load / render ---------------------------------------------------- #
    ext = Path(input_path).suffix.lower()
    if ext == ".pdf":
        page_img = render_pdf_page(input_path, page, dpi)
    elif ext in (".png", ".jpg", ".jpeg"):
        page_img = load_image(input_path)
    else:
        print(f"Unsupported input type: {ext}", file=sys.stderr)
        return 1

    page_name = f"page_{page}_render.png"
    page_img.save(out / page_name)
    print(f"Rendered page -> {out / page_name}  (size {page_img.width}x{page_img.height} px)")

    # ---- kick off Content Understanding early (overlaps region OCR) -------- #
    cu_executor = None
    cu_future = None
    if merge_md:
        print("[CU] starting Content Understanding OCR in background (overlaps region OCR) ...")
        cu_executor = ThreadPoolExecutor(max_workers=1)
        cu_future = cu_executor.submit(cu_ocr_markdown, input_path)

    # ---- determine regions ------------------------------------------------ #
    regions: list[dict] = []
    if auto_detect:
        print(f"[detect] locating handwriting regions with {GPT_DEPLOYMENT} ...")
        regions = detect_handwriting_regions(page_img)
        print(f"[detect] found {len(regions)} handwriting region(s).")
    elif crops:
        regions = [{"label": f"manual_{i + 1}", "box": list(c)} for i, c in enumerate(crops)]

    if not regions:
        print("\nNo regions to process. Open the rendered page and either:")
        print('  - re-run with one or more --crop "x,y,w,h"  (repeat the flag per box), or')
        print("  - re-run with --auto-detect  to locate all handwriting automatically.")
        return 0

    if max_regions:
        regions = regions[:max_regions]

    # ---- region map overlay ----------------------------------------------- #
    overlay_name = f"page_{page}_regions.png"
    draw_regions_overlay(page_img, regions).save(out / overlay_name)
    print(f"Region map (numbered boxes) -> {out / overlay_name}")

    # ---- prepare every region's images (CPU, fast) ------------------------ #
    prompt = build_ocr_prompt(printed_context, italic=italic)
    region_dirs = [out / f"region_{i:02d}_{_slug(r.get('label', ''))}"
                   for i, r in enumerate(regions, 1)]
    region_tasks = [
        prepare_region_images(page_img, region, str(region_dirs[i]),
                              ocr_mode, chunk_width, overlap)
        for i, region in enumerate(regions)
    ]

    # ---- run ALL vision calls in parallel across ALL regions -------------- #
    flat = [(ri, name, img)
            for ri, tasks in enumerate(region_tasks)
            for name, img in tasks]
    region_results: list[list[dict]] = [[] for _ in regions]
    print(f"[OCR] {len(flat)} vision calls across {len(regions)} region(s), "
          f"{MAX_WORKERS} in parallel ...")

    def _ocr_batch(tasks: list[tuple[int, str, Image.Image]]) -> list[tuple[int, str, Image.Image]]:
        """Run a batch of (region_idx, name, image) OCR tasks in parallel.
        Appends successes to region_results; returns the tasks that failed."""
        failed: list[tuple[int, str, Image.Image]] = []
        workers = max(1, min(MAX_WORKERS, len(tasks)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_meta = {ex.submit(ocr_image, img, prompt): (ri, name, img) for ri, name, img in tasks}
            for fut in as_completed(fut_meta):
                ri, name, img = fut_meta[fut]
                try:
                    region_results[ri].append({"source": name, "ocr": fut.result()})
                except Exception as exc:
                    failed.append((ri, name, img))
                    print(f"   region {ri + 1} {name} failed: {exc}", file=sys.stderr)
        return failed

    # First pass + up to 2 extra retry rounds so a timeout burst never drops a region.
    failures = _ocr_batch(flat)
    for rnd in range(1, 3):
        if not failures:
            break
        print(f"[OCR] retry round {rnd}: re-running {len(failures)} failed call(s) ...")
        time.sleep(3 * rnd)
        failures = _ocr_batch(failures)

    # Safety net: any region that still has NO successful reading -> retry its images once more.
    stranded = [ri for ri in range(len(regions))
                if not any((r.get("ocr") or {}).get("full_transcription") for r in region_results[ri])]
    if stranded:
        retry_tasks = [(ri, name, img) for ri in stranded for name, img in region_tasks[ri]]
        print(f"[OCR] {len(stranded)} region(s) still empty -> final retry of {len(retry_tasks)} call(s) ...")
        time.sleep(5)
        _ocr_batch(retry_tasks)
    if failures:
        print(f"[OCR] WARNING: {len(failures)} call(s) still failing after all retries", file=sys.stderr)

    # ---- final reconciliation per region (visual -> text -> per-image) ----- #
    print("[reconcile] final visual adjudication per region ...")

    def _reconcile_region(ri: int) -> dict:
        imgs = region_tasks[ri]
        res = region_results[ri]
        try:  # 1) visual adjudication with images (primary)
            final = visual_reconcile_ocr_results(imgs, res, printed_context)
            final.setdefault("notes", []).append("visual reconciliation")
            return final
        except Exception as vexc:
            try:  # 2) text-only reconciliation (fallback)
                final = reconcile_ocr_results(res, printed_context)
                final.setdefault("notes", []).append(f"text reconciliation (visual failed: {vexc})")
                return final
            except Exception as texc:  # 3) best non-empty per-image reading
                usable = [r for r in res if (r.get("ocr") or {}).get("full_transcription")]
                if usable:
                    best = max(usable, key=lambda r: len(r["ocr"].get("full_transcription", "")))
                    return {"final_transcription": best["ocr"].get("full_transcription", ""),
                            "words": best["ocr"].get("words", []),
                            "notes": [f"per-image fallback (text failed: {texc})"]}
                return {"final_transcription": "", "words": [], "notes": ["all reconciliation failed"]}

    finals: list[dict] = [{} for _ in regions]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut_ri = {ex.submit(_reconcile_region, ri): ri for ri in range(len(regions))}
        for fut in as_completed(fut_ri):
            ri = fut_ri[fut]
            try:
                finals[ri] = fut.result()
            except Exception as exc:
                finals[ri] = {"final_transcription": "", "words": [],
                              "notes": [f"reconcile failed: {exc}"]}

    # ---- write per-region debug + combined -------------------------------- #
    all_results = []
    for i, region in enumerate(regions, 1):
        rdir = region_dirs[i - 1]
        res = region_results[i - 1]
        final = finals[i - 1] or {}

        per_image = [
            {"source": r["source"],
             "full_transcription": (r.get("ocr") or {}).get("full_transcription", ""),
             "words": (r.get("ocr") or {}).get("words", [])}
            for r in res
        ]
        (rdir / "per_image_ocr.json").write_text(
            json.dumps(per_image, ensure_ascii=False, indent=2), encoding="utf-8")

        sel = _select_visual_images(region_tasks[i - 1], res)
        vinput = {
            "printed_context_used": printed_context,
            "selected_images": [n for n, _ in sel],
            "candidate_readings": [
                {"source": r["source"], "text": (r.get("ocr") or {}).get("full_transcription", "")}
                for r in res
            ],
        }
        (rdir / "visual_reconcile_input.json").write_text(
            json.dumps(vinput, ensure_ascii=False, indent=2), encoding="utf-8")

        result_debug = {
            "label": region.get("label"),
            "box": region["box"],
            "printed_context_used": printed_context,
            "per_image": per_image,
            "final_transcription": final.get("final_transcription", ""),
            "alternative_transcriptions": final.get("alternative_transcriptions", []),
            "words": final.get("words", []),
            "notes": final.get("notes", []),
        }
        (rdir / "result.json").write_text(
            json.dumps(result_debug, ensure_ascii=False, indent=2), encoding="utf-8")

        all_results.append({
            "region": i, "folder": rdir.name,
            "label": region.get("label"), "box": region["box"],
            "final_transcription": final.get("final_transcription"),
            "words": final.get("words"), "notes": final.get("notes"),
        })
        print(f"  [{i}] {region.get('label')}: {final.get('final_transcription', '')!r}")

    combined = {
        "source": input_path, "page": page,
        "region_map": overlay_name, "regions": all_results,
    }
    (out / "handwriting_result.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- merged doc: handwriting (red) over printed page ------------------- #
    cu_markdown = ""
    if merge_md and cu_future is not None:
        try:
            print("[CU] collecting background Content Understanding result ...")
            cu_markdown = cu_future.result()
        except Exception as exc:
            print(f"[CU] OCR failed: {exc}", file=sys.stderr)
        finally:
            if cu_executor is not None:
                cu_executor.shutdown(wait=False)
    if not cu_markdown:
        cu_json = Path("out/result.json")
        if cu_json.exists():
            try:
                cu_markdown = json.loads(cu_json.read_text(encoding="utf-8")) \
                    .get("content_understanding", {}).get("markdown", "")
            except Exception:
                cu_markdown = ""

    merged = build_merged_html(str(out), page_name, all_results, cu_markdown=cu_markdown)
    try:
        merged_jpg = build_merged_jpg(str(out), page_name, all_results)
    except Exception as exc:
        print(f"[merge] jpg failed: {exc}", file=sys.stderr)
        merged_jpg = None

    merged_md = None
    if merge_md and cu_markdown:
        try:
            print("[merge] fusing CU printed text + GPT handwriting into merged.md ...")
            merged_md = build_merged_markdown(str(out), page_name, input_path, cu_markdown, all_results)
        except Exception as exc:
            print(f"[merge] markdown failed: {exc}", file=sys.stderr)

    # ---- summary ---------------------------------------------------------- #
    print("\n" + "=" * 70)
    print(f"ALL HANDWRITING REGIONS ({len(all_results)})")
    print("=" * 70)
    for r in all_results:
        print(f"  [{r['region']}] {r.get('label')}: {r.get('final_transcription')}")
    print(f"\nRegion map:    {out / overlay_name}")
    print(f"Per-region:    {out}/region_*/  (images + result.json)")
    print(f"Combined JSON: {out / 'handwriting_result.json'}")
    print(f"Merged doc:    {merged}  (handwriting in RED over the printed page)")
    if merged_jpg:
        print(f"Merged JPG:    {merged_jpg}")
    if merged_md:
        print(f"Merged MD:     {merged_md}  (CU printed + GPT handwriting, de-duplicated)")
    return 0


def _parse_crop(s: str | None) -> tuple[int, int, int, int] | None:
    if not s:
        return None
    parts = [int(p.strip()) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError('crop must be "x,y,w,h"')
    return tuple(parts)  # type: ignore[return-value]


def main() -> int:
    global MAX_WORKERS
    ap = argparse.ArgumentParser(description="Hebrew handwriting OCR helper pipeline")
    ap.add_argument("--input", required=True, help="PDF or image path")
    ap.add_argument("--page", type=int, default=1, help="PDF page number (1-based)")
    ap.add_argument("--crop", action="append", default=None,
                    help='handwriting rectangle "x,y,w,h" in px; repeat for multiple regions')
    ap.add_argument("--auto-detect", action="store_true",
                    help="auto-locate ALL handwriting regions with the Vision LLM")
    ap.add_argument("--max-regions", type=int, default=None, help="cap number of regions processed")
    ap.add_argument("--output", default="./output_handwriting", help="output folder")
    ap.add_argument("--dpi", type=int, default=400, help="PDF render DPI (300-600)")
    ap.add_argument("--chunk-width", type=int, default=900)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--ocr-mode", choices=["full", "chunks", "both"], default="both")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS,
                    help="parallel GPT vision calls in flight (use 1 for easy debugging)")
    ap.add_argument("--printed-context", default=None,
                    help="short printed-text context string (SOFT hint only)")
    ap.add_argument("--printed-context-file", default=None,
                    help="file with short printed-text context (SOFT hint only)")
    ap.add_argument("--context-mode", choices=["none", "nearby", "summary"], default="none",
                    help="how to use printed context (default: none = ignore it)")
    ap.add_argument("--italic", action="store_true",
                    help="treat regions as possibly ITALIC/CURSIVE printed Hebrew (not just handwriting)")
    ap.add_argument("--merge-md", action="store_true",
                    help="also produce merged.md: CU printed text + GPT handwriting, de-duplicated")
    args = ap.parse_args()

    MAX_WORKERS = max(1, args.workers)

    # Printed context is optional and used only as a SOFT hint (never as truth).
    printed_context = ""
    if args.context_mode != "none":
        max_chars = 1000 if args.context_mode == "summary" else 600
        printed_context = load_printed_context(args.printed_context, args.printed_context_file, max_chars)
        if printed_context:
            print(f"Using printed context ({args.context_mode}, {len(printed_context)} chars, soft hint only)")

    crops = [_parse_crop(c) for c in args.crop] if args.crop else None

    return run_pipeline(
        input_path=args.input, page=args.page, dpi=args.dpi,
        crops=crops, output_dir=args.output,
        chunk_width=args.chunk_width, overlap=args.overlap, ocr_mode=args.ocr_mode,
        auto_detect=args.auto_detect, max_regions=args.max_regions,
        printed_context=printed_context, italic=args.italic, merge_md=args.merge_md,
    )


if __name__ == "__main__":
    raise SystemExit(main())
