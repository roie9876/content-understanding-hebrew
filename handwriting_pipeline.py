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
import collections
import difflib
import io
import json
import os
import random
import re
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

GPT_TEMPERATURE = float(os.environ.get("GPT_TEMPERATURE", "0"))
GPT_SEED = int(os.environ.get("GPT_SEED", "7"))
GPT_READ_TIMEOUT = int(os.environ.get("GPT_READ_TIMEOUT", "90"))  # per-call read timeout (s); stuck calls retry fast

# Visual adjudication is the quality authority by default: seven enhancement reads
# become evidence, then the model re-reads the strokes to select the final text.
# Self-consistency voting remains opt-in for experiments.
VOTE_ENABLED = os.environ.get("VOTE", "0") not in ("", "0", "false", "False")
VOTE_SEEDS = int(os.environ.get("VOTE_SEEDS", "3"))  # OCR draws per image version

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
# Token-usage accounting (thread-safe; GPT Responses API calls only)
# --------------------------------------------------------------------------- #
_usage_lock = threading.Lock()
_usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def _reset_usage() -> None:
    with _usage_lock:
        _usage.update(input_tokens=0, output_tokens=0, calls=0)


def _add_usage(resp: dict) -> None:
    """Accumulate token usage from a GPT Responses API JSON response."""
    u = resp.get("usage") or {}
    it = u.get("input_tokens", u.get("prompt_tokens", 0)) or 0
    ot = u.get("output_tokens", u.get("completion_tokens", 0)) or 0
    with _usage_lock:
        _usage["input_tokens"] += int(it)
        _usage["output_tokens"] += int(ot)
        _usage["calls"] += 1


def _get_usage() -> dict:
    with _usage_lock:
        u = dict(_usage)
    u["total_tokens"] = u["input_tokens"] + u["output_tokens"]
    return u


def _fmt_mmss(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"


def _write_json_atomic(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _write_text_atomic(path: Path, value: str) -> None:
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


# --------------------------------------------------------------------------- #
# Model capability negotiation
# --------------------------------------------------------------------------- #
_unsupported_params: set[str] = set()  # payload params the model rejects (e.g. temperature)


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


def pdf_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def _single_page_pdf_bytes(pdf_path: str, page: int) -> bytes:
    """Return a 1-page PDF (bytes) containing only page `page` (1-based)."""
    src = fitz.open(pdf_path)
    try:
        one = fitz.open()
        one.insert_pdf(src, from_page=page - 1, to_page=page - 1)
        return one.tobytes()
    finally:
        src.close()


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


def enhance_pen_stroke_isolation(image: Image.Image) -> Image.Image:
    """Suppress thin printed glyphs while retaining neighborhoods of thick pen strokes."""
    gray = _pil_to_gray_cv(image)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    thick_core = (distance >= 2.4).astype(np.uint8) * 255
    neighborhood = cv2.dilate(
        thick_core,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2,
    )
    isolated = np.full_like(gray, 255)
    isolated[neighborhood > 0] = gray[neighborhood > 0]
    return ImageOps.autocontrast(_gray_cv_to_pil(isolated), cutoff=1)


def build_versions(crop: Image.Image) -> dict[str, Image.Image]:
    """Return the ordered set of enhanced versions of the handwriting crop."""
    return {
        "01_original_crop.png": crop,
        "02_upscaled_contrast.png": enhance_upscaled_contrast(crop, scale=4),
        "03_adaptive_threshold.png": enhance_adaptive_threshold(crop),
        "04_denoised.png": enhance_denoised(crop),
        "05_thickened_strokes.png": enhance_thickened_strokes(crop),
        "06_optional_line_removed.png": optional_remove_horizontal_lines(crop),
        "07_pen_stroke_isolated.png": enhance_pen_stroke_isolation(crop),
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
                    max_tokens: int = 6000, reasoning: str | None = None,
                    seed: int | None = None) -> str:
    """POST to the Responses API and return the raw output text.

    Robust to timeouts / 429 / 5xx / empty output: retries with exponential
    backoff + jitter so a transient burst never loses the call.
    """
    use_seed = GPT_SEED if seed is None else seed
    payload = {
        "model": GPT_DEPLOYMENT,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": max_tokens,
        "reasoning": {"effort": reasoning or GPT_REASONING},
        "temperature": GPT_TEMPERATURE,
        "seed": use_seed,
    }
    for _p in _unsupported_params:
        payload.pop(_p, None)
    url = f"{GPT_ENDPOINT}responses"
    last_err = ""
    for attempt in range(retries + 1):
        backoff = min(30.0, 2.0 ** attempt) + random.uniform(0, 1.5)
        try:
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {_bearer()}", "Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=GPT_READ_TIMEOUT,
            )
        except requests.exceptions.RequestException as exc:  # timeout / connection reset
            last_err = f"request error: {exc}"
            if attempt < retries:
                time.sleep(backoff)
                continue
            raise RuntimeError(f"[GPT] {last_err}")

        if r.status_code == 400:
            low = r.text.lower()
            popped = False
            for _p in ("temperature", "seed", "reasoning"):
                if _p in low and _p in payload:
                    payload.pop(_p, None)
                    _unsupported_params.add(_p)
                    popped = True
            if popped:
                return _responses_text(
                    content, retries=retries, max_tokens=max_tokens,
                    reasoning=reasoning, seed=seed,
                )
        if r.status_code in (408, 409, 429, 500, 502, 503, 504):
            last_err = f"HTTP {r.status_code}"
            if attempt < retries:
                wait = float(r.headers.get("retry-after", backoff))
                time.sleep(wait)
                continue
            raise RuntimeError(f"[GPT] {r.status_code} after retries: {r.text[:200]}")
        if r.status_code != 200:
            raise RuntimeError(f"[GPT] {r.status_code}: {r.text[:400]}")
        resp_json = r.json()
        _add_usage(resp_json)
        text = _gpt_output_text(resp_json)
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
                    reasoning: str | None = None, max_tokens: int = 6000,
                    seed: int | None = None) -> dict:
    return _loads_lenient(_responses_text(content, retries, max_tokens=max_tokens,
                                          reasoning=reasoning, seed=seed))


def ocr_image(image: Image.Image, prompt: str, seed: int | None = None) -> dict:
    # Raw per-image OCR keeps the original image evidence and reasoning quality.
    return _responses_call([
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": _to_data_uri(image, max_side=OCR_MAX_SIDE)},
    ], retries=0, reasoning=OCR_REASONING, seed=seed)


# --------------------------------------------------------------------------- #
# Content Understanding printed-text OCR (for the merged Markdown)
# --------------------------------------------------------------------------- #
def cu_ocr_markdown(input_path: str, timeout_s: int = 180, retries: int = 4,
                    page: int | None = None) -> str:
    """Run Content Understanding (prebuilt-documentSearch) on the file and return
    its printed-text markdown. Uses the same Entra ID token as GPT. Retries the
    submit and tolerates transient errors while polling. If `page` is given for a
    PDF, only that single page is analyzed (keeps per-page merges consistent)."""
    ext = Path(input_path).suffix.lower()
    if page is not None and ext == ".pdf":
        raw, mime = _single_page_pdf_bytes(input_path, page), "application/pdf"
    else:
        mime = {".pdf": "application/pdf", ".png": "image/png",
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "application/octet-stream")
        raw = Path(input_path).read_bytes()
    data = base64.b64encode(raw).decode()
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
            md = contents[0].get("markdown", "") if contents else ""
            return md
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


def _early_exit_transcription(results: list[dict]) -> dict | None:
    """#5 speed lever: if the per-image reads are short, unmarked, and agree,
    accept them directly and skip the (expensive) visual-reconcile call.

    Only triggers on easy fields (dates, numbers, single short tokens). Any '[?]'
    uncertainty, any newline (multi-field), a long value, or weak agreement falls
    through to full visual adjudication. Returns a final dict or None.
    """
    texts = [(r.get("ocr") or {}).get("full_transcription", "") for r in results]
    texts = [t for t in texts if t and t.strip()]
    if len(texts) < 2:
        return None
    joined = "".join(texts)
    if "[?]" in joined or "\n" in joined:
        return None
    normed = [" ".join(t.split()) for t in texts]
    top = max(set(normed), key=normed.count)
    if not top or len(top) > 18:
        return None
    if normed.count(top) < max(2, (len(normed) + 1) // 2):
        return None
    raw = next(t for t in texts if " ".join(t.split()) == top).strip()
    return {
        "final_transcription": raw,
        "alternative_transcriptions": [],
        "words": [],
        "notes": [f"early-exit: {normed.count(top)}/{len(normed)} reads agreed "
                  "(skipped visual reconcile)"],
    }


# --------------------------------------------------------------------------- #
# Self-consistency voting (deterministic aggregation of N genuine OCR draws)
# --------------------------------------------------------------------------- #
_STRUCT_PATTERNS = [
    ("date", re.compile(r"^\d{1,4}([./-])\d{1,2}\1\d{1,4}$")),
    ("email", re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")),
    ("phone", re.compile(r"^0\d{1,2}-?\d{6,7}$")),
    ("id", re.compile(r"^\d{5,9}$")),
    ("number", re.compile(r"^\d{1,4}$")),
]

# High-specificity types that we lock even if only a plurality of draws read them
# cleanly (a valid email/date/phone is a strong signal; free-text voting mangles
# them by splitting on @/./-). Only used when the match dominates the reading.
_STRUCT_SUBSTR = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("date", re.compile(r"\d{1,4}[./-]\d{1,2}[./-]\d{1,4}")),
    ("phone", re.compile(r"0\d{1,2}-?\d{6,7}")),
]


def _norm_read(t: str) -> str:
    """Normalize a reading for comparison: drop [?] markers, collapse whitespace."""
    return " ".join((t or "").replace("[?]", " ").split())


def _structured_consensus(reads: list[str]):
    """If the readings are the SAME structured value (date/phone/id/email/number),
    return (value, type, agree, alts); else None. Deterministic.

    (a) High-specificity email/date/phone are locked on a plurality of clean draws
        (when the match dominates the reading). (b) Plain number/id require a
        majority of whole-value matches."""
    n = len(reads)
    for typ, pat in _STRUCT_SUBSTR:
        hits = []
        for r in reads:
            nr = _norm_read(r)
            m = pat.search(nr.replace(" ", ""))
            if m and len(m.group(0)) >= 0.5 * max(1, len(nr.replace(" ", ""))):
                hits.append(m.group(0))
        if len(hits) >= max(2, n // 4):
            cnt = collections.Counter(hits)
            best, agree = cnt.most_common(1)[0]
            return best, typ, agree, [v for v in cnt if v != best][:4]

    collapsed = [_norm_read(r).replace(" ", "") for r in reads]
    types = [next((name for name, pat in _STRUCT_PATTERNS if pat.match(v)), None)
             for v in collapsed]
    typed = [t for t in types if t]
    if len(typed) < (n + 1) // 2:        # structured only if the majority looks structured
        return None
    dom = collections.Counter(typed).most_common(1)[0][0]
    vals = [c for c, t in zip(collapsed, types) if t == dom]
    cnt = collections.Counter(vals)                    # ties -> first-seen (deterministic)
    best, agree = cnt.most_common(1)[0]
    if agree < n // 2 + 1:
        return None
    alts = [v for v in cnt if v != best][:4]
    return best, dom, agree, alts


def _freetext_consensus(reads: list[str]) -> dict:
    """Deterministic medoid + per-token majority vote for free-text handwriting.
    Tokens agreed by a majority are locked; the rest are marked [?]."""
    n = len(reads)
    norm = [_norm_read(r) for r in reads]

    def total_sim(i: int) -> float:
        return sum(difflib.SequenceMatcher(None, norm[i], norm[j]).ratio()
                   for j in range(n) if j != i)

    medoid_idx = max(range(n), key=lambda i: (total_sim(i), -i))  # central reading, deterministic

    def core(t: str) -> str:
        return t.strip(" ,.;:()[]{}\"'׳״־-")

    token_sets = [{core(w) for w in x.split()} for x in norm]
    threshold = (n + 1) // 2                                       # majority

    out, words = [], []
    for tok in norm[medoid_idx].split():
        c = core(tok)
        agree = sum(1 for ts in token_sets if c and c in ts)
        locked = agree >= threshold
        out.append(tok if locked else f"{tok}[?]")
        words.append({"text": tok, "confidence": round(agree / n, 2), "locked": locked})

    final = " ".join(out)
    alts, seen = [], {_norm_read(final)}
    for r in reads:
        nr = _norm_read(r)
        if nr and nr not in seen:
            alts.append(r.strip())
            seen.add(nr)
    return {
        "final_transcription": final,
        "words": words,
        "alternative_transcriptions": alts[:4],
        "notes": [f"self-consistency vote over {n} draws; below-majority tokens marked [?]"],
    }


def aggregate_votes(readings: list[str]) -> dict:
    """Deterministically fuse N OCR draws of the SAME region into one stable,
    honest reading. Structured fields -> format-constrained majority; free text ->
    medoid + per-token majority (uncertain tokens kept as [?] with alternatives)."""
    reads = [r for r in readings if r and r.strip()]
    if not reads:
        return {"final_transcription": "", "words": [],
                "alternative_transcriptions": [], "notes": ["no readings"]}
    if len(reads) == 1:
        return {"final_transcription": reads[0].strip(), "words": [],
                "alternative_transcriptions": [], "notes": ["single draw"]}
    struct = _structured_consensus(reads)
    if struct is not None:
        value, typ, agree, alts = struct
        return {
            "final_transcription": value,
            "words": [{"text": value, "confidence": round(agree / len(reads), 2),
                       "alternatives": alts}],
            "alternative_transcriptions": alts,
            "notes": [f"structured({typ}) majority {agree}/{len(reads)}"],
        }
    return _freetext_consensus(reads)


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
    final = _responses_call([{"type": "input_text", "text": prompt}], retries=0)
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
    "07_pen_stroke_isolated.png",
]
_FAITHFUL_VISUAL_SOURCES = {
    "01_original_crop.png",
    "02_upscaled_contrast.png",
    "04_denoised.png",
}


def _is_faithful_source(source: str) -> bool:
    return source in _FAITHFUL_VISUAL_SOURCES or source.startswith("chunk_")


def guard_visual_reconciliation(final: dict, usable: list[dict]) -> dict:
    """Reject a confident reading supported only by an aggressive transform."""
    final_text = _norm_read(final.get("final_transcription", ""))
    readings = [
        (result["source"], _norm_read((result.get("ocr") or {}).get("full_transcription", "")))
        for result in usable
        if _norm_read((result.get("ocr") or {}).get("full_transcription", ""))
    ]
    if not final_text or len(readings) < 5:
        return final

    distinct = {text for _, text in readings}
    supporters = [source for source, text in readings if text == final_text]
    faithful_supporters = [source for source in supporters if _is_faithful_source(source)]
    severe_disagreement = len(distinct) >= max(4, (len(readings) + 1) // 2)
    unsupported = bool(supporters) and not faithful_supporters and len(supporters) == 1
    if not severe_disagreement or not unsupported:
        return final

    faithful = [(source, text) for source, text in readings if _is_faithful_source(source)]
    candidates = faithful or readings

    def centrality(candidate: tuple[str, str]) -> tuple[float, int]:
        source, text = candidate
        similarity = sum(
            difflib.SequenceMatcher(None, text, other).ratio()
            for _, other in readings
        )
        source_priority = 1 if source == "01_original_crop.png" else 0
        return similarity, source_priority

    selected_source, selected_text = max(candidates, key=centrality)
    exact_support = [source for source, text in readings if text == selected_text]
    selected_result = next(
        result for result in usable
        if result["source"] == selected_source
        and _norm_read((result.get("ocr") or {}).get("full_transcription", "")) == selected_text
    )
    selected_ocr = selected_result.get("ocr") or {}
    alternatives = []
    for _, text in readings:
        if text and text != selected_text and text not in alternatives:
            alternatives.append(text)
    if final_text and final_text != selected_text and final_text not in alternatives:
        alternatives.insert(0, final_text)
    guarded = {
        "final_transcription": selected_ocr.get("full_transcription", selected_text),
        "alternative_transcriptions": alternatives[:5],
        "words": [
            {
                **word,
                "confidence": min(0.49, float(word.get("confidence", 0.0) or 0.0)),
                "source_images": [selected_source],
            }
            for word in (selected_ocr.get("words") or [])
        ],
        "notes": list(final.get("notes", [])),
        "needs_review": True,
        "decision_status": "faithful_candidate_fallback",
    }
    guarded["notes"].append(
        "agreement gate: rejected visual reading supported only by an aggressive "
        "transform; preserved the most central faithful candidate for review"
    )
    guarded["agreement_gate"] = {
        "rejected_transcription": final.get("final_transcription", ""),
        "supporting_sources": supporters,
        "faithful_supporting_sources": faithful_supporters,
        "distinct_candidates": len(distinct),
        "selected_source": selected_source,
        "selected_supporting_sources": exact_support,
    }
    return guarded


def focused_faithful_recheck(
    image_tasks: list[tuple[str, Image.Image]], rejected_text: str,
) -> dict:
    """Re-read a disputed region using only minimally transformed evidence."""
    faithful = [
        (source, image) for source, image in image_tasks
        if source in _FAITHFUL_VISUAL_SOURCES
    ]
    if not faithful:
        raise RuntimeError("focused recheck: no faithful images")
    instruction = f"""Re-read this difficult Hebrew handwriting from the faithful images only.

The previous adjudication proposed: {rejected_text or "(none)"}
That proposal was rejected because only an aggressive image transform supported it.

Rules:
- Ignore the rejected proposal unless the original strokes independently prove every letter.
- Inspect the handwriting right-to-left, letter by letter.
- Do not complete the strokes into a familiar person name or plausible phrase.
- Ignore printed text, form lines, illustrations, and solid black/redacted rectangles.
- Preserve the number of visible handwritten words.
- Use [?] for every unsupported or ambiguous letter sequence.
- Return JSON only.

Return:
{{
  "final_transcription": "faithful re-reading with [?] where needed",
  "visible_letter_sequence": ["rightmost visible letter or [?]", "..."],
  "confidence": 0.0,
  "words": [
    {{"text": "word", "confidence": 0.0, "alternatives": [], "reason": "stroke evidence"}}
  ],
  "notes": []
}}
"""
    content: list[dict] = [{"type": "input_text", "text": instruction}]
    for source, image in faithful:
        content.append({"type": "input_text", "text": f"Faithful image: {source}"})
        content.append({"type": "input_image", "image_url": _to_data_uri(image)})
    return _responses_call(content, retries=0, max_tokens=12000)


def accept_focused_faithful_recheck(
    guarded: dict, recheck: dict, usable: list[dict],
) -> dict:
    """Accept a focused re-read only when faithful evidence directly supports it."""
    if guarded.get("decision_status") != "faithful_candidate_fallback":
        return guarded
    gate = guarded.get("agreement_gate") or {}
    if gate.get("faithful_supporting_sources"):
        return guarded
    recheck_text = _norm_read(recheck.get("final_transcription", ""))
    guarded_text = _norm_read(guarded.get("final_transcription", ""))
    faithful_texts = {
        _norm_read((result.get("ocr") or {}).get("full_transcription", ""))
        for result in usable
        if _is_faithful_source(result.get("source", ""))
    }
    faithful_texts.discard("")
    if not recheck_text or recheck_text not in faithful_texts:
        return guarded
    if recheck_text.count("[?]") > guarded_text.count("[?]"):
        return guarded
    accepted = dict(recheck)
    accepted.setdefault("alternative_transcriptions", []).extend(
        text for text in [guarded.get("final_transcription", "")]
        if text and text != recheck.get("final_transcription", "")
    )
    accepted.setdefault("notes", []).append(
        "focused faithful-image recheck accepted with exact faithful candidate support"
    )
    accepted["needs_review"] = True
    accepted["decision_status"] = "focused_faithful_recheck"
    accepted["agreement_gate"] = gate
    return accepted


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
    readings = [
        (r.get("ocr") or {}).get("full_transcription", "")
        for r in usable
        if (r.get("ocr") or {}).get("full_transcription", "").strip()
    ]
    agreement = aggregate_votes(readings) if readings else {}
    agreement_text = agreement.get("final_transcription", "") or "(none)"
    context_block = build_context_block(printed_context)
    instruction = f"""You are doing final visual adjudication of Hebrew handwriting OCR.

You will receive:
1. Several enhanced image versions of the SAME handwritten region.
2. Candidate OCR readings from previous passes.
3. Optional nearby printed context.

Rules:
- The images are the source of truth.
- Candidate readings are hints only.
- Original crop, contrast, denoised, and original-image chunks are faithful evidence.
- Thresholded, thickened, line-removed, and isolated-stroke images are supporting evidence only.
- Never choose a reading supported only by one supporting transform when the faithful images
    disagree with it. Return uncertain letters/words instead of completing a plausible name.
- When several independent enhancement candidates agree on the same word, prefer that
    agreement unless the visible strokes clearly contradict it. Do not select a minority
    alternative merely because it is plausible.
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

Deterministic agreement summary across candidates (hint only, not ground truth):
{agreement_text}

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

    final = _responses_call(content, retries=0)
    if "final_transcription" not in final:
        raise RuntimeError("visual reconcile: no final_transcription in output")
    final["_selected_images"] = [n for n, _ in selected]
    _attach_source_images(final, usable)
    guarded = guard_visual_reconciliation(final, usable)
    guarded["_selected_images"] = final["_selected_images"]
    _attach_source_images(guarded, usable)
    return guarded


# --------------------------------------------------------------------------- #
# Region detection + per-region processing
# --------------------------------------------------------------------------- #
def refine_box_to_ink(
    page_img: Image.Image, box: list[int], vpad_frac: float = 0.35,
    color_hint: str = "",
) -> list[int]:
    """Expand an approximate box to cover the actual handwriting strokes.

    Vision-LLM boxes often under-cover. Within the box's vertically-padded band
    we find ink connected components (ignoring long printed lines) and chain the
    ones separated by small word-gaps, starting from components that overlap the
    original box. Isolated fields (numbers) stay tight; long note lines fill out.
    """
    W, H = page_img.size
    x, y, w, h = box
    hint = color_hint.lower()
    has_color_hint = any(term in hint for term in ("blue", "green", "phone", "email"))
    vpad = max(int(h * vpad_frac), 90 if has_color_hint else 0)
    y0, y1 = max(0, y - vpad), min(H, y + h + vpad)
    color_refined = refine_box_to_colored_ink(page_img, box, y0, y1, color_hint)
    if color_refined is not None:
        return color_refined
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
    # A 4x cap allowed compact fields to chain through nearby printed labels and
    # underlines until they covered most of the form. Long-note seeds are already
    # wide; compact fields only need enough growth for clipped edge strokes.
    max_w = min(W, int(w * 1.6) + 40)
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


def refine_box_to_colored_ink(
    page_img: Image.Image, box: list[int], band_top: int, band_bottom: int,
    color_hint: str = "",
) -> list[int] | None:
    """Tighten blue/green pen fields without following black print or redactions."""
    page_width, page_height = page_img.size
    x, y, width, height = box
    rgb = np.array(page_img.convert("RGB"))[band_top:band_bottom, :]
    if rgb.size == 0:
        return None
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hue = hsv[:, :, 0]
    hint = color_hint.lower()
    color_mask = (saturation >= 45) & (value <= 245)
    if "blue" in hint:
        color_mask &= (hue >= 100) & (hue <= 135)
    elif "green" in hint:
        color_mask &= (hue >= 35) & (hue <= 100)
    colored = np.where(color_mask, 255, 0).astype(np.uint8)
    colored = cv2.morphologyEx(colored, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(colored, 8)
    components = []
    for component_index in range(1, count):
        left = int(stats[component_index, cv2.CC_STAT_LEFT])
        top = int(stats[component_index, cv2.CC_STAT_TOP]) + band_top
        component_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < 5 or component_width > page_width * 0.35:
            continue
        hue = float(np.median(hsv[:, :, 0][labels == component_index]))
        components.append(
            (left, top, left + component_width, top + component_height, area, hue)
        )
    seeds = [
        component for component in components
        if component[0] < x + width and component[2] > x
    ]
    if not seeds or sum(component[4] for component in seeds) < 20:
        return None

    hue_buckets: dict[int, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    for component in seeds:
        bucket = round(component[5] / 10)
        component_center = (component[0] + component[2]) / 2
        hue_buckets[bucket][0] += component[4]
        hue_buckets[bucket][1] += component[4] * component_center
    expected_center = x + width / 2

    def bucket_score(bucket: int) -> float:
        area, weighted_center = hue_buckets[bucket]
        center = weighted_center / max(1, area)
        normalized_distance = abs(center - expected_center) / max(1, width)
        return area / (1 + 3 * normalized_distance)

    structured_contact = "phone" in hint or "email" in hint
    if structured_contact and "blue" not in hint and "green" not in hint:
        dominant_bucket = min(
            hue_buckets,
            key=lambda bucket: abs(
                hue_buckets[bucket][1] / max(1, hue_buckets[bucket][0]) - expected_center
            ),
        )
    else:
        dominant_bucket = max(hue_buckets, key=bucket_score)
    dominant_hue = dominant_bucket * 10
    hue_tolerance = 19 if structured_contact else 9
    components = [
        component for component in components
        if abs(component[5] - dominant_hue) <= hue_tolerance
    ]
    seeds = [
        component for component in seeds
        if abs(component[5] - dominant_hue) <= hue_tolerance
    ]
    if structured_contact and seeds:
        expected_y = y + height
        anchor = min(
            seeds,
            key=lambda component: (
                ((component[0] + component[2]) / 2 - expected_center) ** 2
                + ((component[1] + component[3]) / 2 - expected_y) ** 2
            ),
        )
        anchor_y = (anchor[1] + anchor[3]) / 2
        line_tolerance = max(18, height * 0.35)
        seeds = [
            component for component in seeds
            if abs((component[1] + component[3]) / 2 - anchor_y) <= line_tolerance
        ]
        components = list(seeds)

    left = min(component[0] for component in seeds)
    top = min(component[1] for component in seeds)
    right = max(component[2] for component in seeds)
    bottom = max(component[3] for component in seeds)
    compact_seed = width < page_width * 0.4
    horizontal_gap = max(10, int(height * (0.25 if compact_seed else 0.45)))
    vertical_gap = max(5, int(height * (0.12 if structured_contact else 0.35)))
    max_colored_width = width * (1.25 if compact_seed else 1.8) + 20
    max_colored_height = height * (1.5 if structured_contact else 2.5) + 20
    changed = True
    while changed:
        changed = False
        for component in components:
            component_left, component_top, component_right, component_bottom, _area, _hue = component
            horizontally_near = not (
                component_right < left - horizontal_gap
                or component_left > right + horizontal_gap
            )
            vertically_near = not (
                component_bottom < top - vertical_gap
                or component_top > bottom + vertical_gap
            )
            proposed_width = max(right, component_right) - min(left, component_left)
            proposed_height = max(bottom, component_bottom) - min(top, component_top)
            if horizontally_near and vertically_near \
                    and proposed_width <= max_colored_width \
                    and proposed_height <= max_colored_height:
                new_bounds = (
                    min(left, component_left), min(top, component_top),
                    max(right, component_right), max(bottom, component_bottom),
                )
                if new_bounds != (left, top, right, bottom):
                    left, top, right, bottom = new_bounds
                    changed = True
    xpad = min(10, max(8, int((right - left) * 0.08))) if structured_contact \
        else max(8, int((right - left) * 0.08))
    ypad = max(8, int((bottom - top) * 0.35))
    left = max(0, left - xpad)
    right = min(page_width, right + xpad)
    top = max(0, top - ypad)
    bottom = min(page_height, bottom + ypad)
    return [left, top, right - left, bottom - top]


def add_handwriting_context(page_img: Image.Image, box: list[int]) -> list[int]:
    """Shape refined boxes without losing strokes or absorbing neighboring lines."""
    W, H = page_img.size
    x, y, w, h = box
    narrow_field = w < W * 0.12
    if narrow_field:
        xpad = int(W * 0.08)
        top = max(0, y - max(int(H * 0.01), int(h * 0.20)))
        bottom = min(H, y + h + max(int(H * 0.01), int(h * 0.60)))
    else:
        # refine_box_to_ink already adds 35% vertical padding. Shifting this band
        # down clipped the intended line and absorbed the following line.
        xpad = max(int(W * 0.01), int(w * 0.03))
        top = y
        bottom = min(H, y + h)
    left = max(0, x - xpad)
    right = min(W, x + w + xpad)
    return [left, top, right - left, bottom - top]


def detection_geometry_penalty(
    regions: list[dict], page_width: int, page_height: int,
) -> tuple[float, list[str]]:
    """Score obvious detector geometry failures without assuming document text."""
    penalty = 0.0
    issues = []
    compact_terms = ("number", "phone", "email", "id number", "identification number")
    for index, region in enumerate(regions, 1):
        label = region.get("label", "").lower()
        _x, _y, width, height = region["box"]
        width_fraction = width / max(1, page_width)
        height_fraction = height / max(1, page_height)
        if any(term in label for term in compact_terms) and width_fraction > 0.42:
            excess = width_fraction - 0.42
            penalty += 10.0 + excess * 20.0
            issues.append(
                f"region {index} ({label}) is an implausibly wide compact field "
                f"({width_fraction:.0%} of page width)"
            )
        if any(term in label for term in compact_terms) and height_fraction > 0.20:
            penalty += 8.0
            issues.append(
                f"region {index} ({label}) is an implausibly tall compact field"
            )
        area_fraction = width * height / max(1, page_width * page_height)
        if "paragraph block" not in label and area_fraction > 0.24:
            penalty += 10.0
            issues.append(f"region {index} covers too much of the page")
    return penalty, issues


def repair_invalid_detection_regions(
    original: list[dict], retry: list[dict], page_width: int, page_height: int,
) -> list[dict]:
    """Replace only invalid boxes so a retry cannot move otherwise good lines."""
    repaired = [dict(region, box=list(region["box"])) for region in original]
    used_retry: set[int] = set()
    for index, region in enumerate(original):
        region_penalty, _ = detection_geometry_penalty([region], page_width, page_height)
        if region_penalty <= 0:
            continue
        x, y, width, height = region["box"]
        center = (x + width / 2, y + height / 2)
        candidates = []
        for retry_index, candidate in enumerate(retry):
            if retry_index in used_retry:
                continue
            candidate_penalty, _ = detection_geometry_penalty(
                [candidate], page_width, page_height
            )
            if candidate_penalty >= region_penalty:
                continue
            cx, cy, cw, ch = candidate["box"]
            distance = (
                ((center[0] - (cx + cw / 2)) / max(1, page_width)) ** 2
                + ((center[1] - (cy + ch / 2)) / max(1, page_height)) ** 2
            )
            candidates.append((distance, candidate_penalty, retry_index, candidate))
        if candidates:
            _distance, _penalty, retry_index, candidate = min(candidates)
            repaired[index] = dict(candidate, box=list(candidate["box"]))
            used_retry.add(retry_index)
    return repaired


def deduplicate_detection_regions(
    regions: list[dict], page_width: int, page_height: int,
) -> list[dict]:
    """Drop invalid containers and merge near-contained duplicate detections."""
    kept = [dict(region, box=list(region["box"])) for region in regions]
    remove: set[int] = set()
    compact_terms = ("number", "phone", "email", "id number", "identification number")
    for first_index, first in enumerate(kept):
        if first_index in remove:
            continue
        fx, fy, fw, fh = first["box"]
        first_area = fw * fh
        for second_index in range(first_index + 1, len(kept)):
            if second_index in remove:
                continue
            second = kept[second_index]
            sx, sy, sw, sh = second["box"]
            second_area = sw * sh
            intersection_width = max(0, min(fx + fw, sx + sw) - max(fx, sx))
            intersection_height = max(0, min(fy + fh, sy + sh) - max(fy, sy))
            intersection = intersection_width * intersection_height
            if intersection <= 0:
                continue
            smaller_area = min(first_area, second_area)
            containment = intersection / max(1, smaller_area)
            area_ratio = max(first_area, second_area) / max(1, smaller_area)
            union = first_area + second_area - intersection
            iou = intersection / max(1, union)
            larger_index = first_index if first_area > second_area else second_index
            smaller_index = second_index if larger_index == first_index else first_index
            larger = kept[larger_index]
            smaller = kept[smaller_index]
            larger_label = larger.get("label", "").lower()
            larger_penalty, _ = detection_geometry_penalty(
                [larger], page_width, page_height
            )
            smaller_penalty, _ = detection_geometry_penalty(
                [smaller], page_width, page_height
            )
            oversized_compact = (
                any(term in larger_label for term in compact_terms)
                and larger["box"][2] > page_width * 0.42
            )
            sufficiently_overlapping = containment >= (0.40 if oversized_compact else 0.78)
            if area_ratio >= 1.55 and sufficiently_overlapping and (
                larger_penalty > smaller_penalty or oversized_compact
            ):
                remove.add(larger_index)
                continue
            center_distance = (
                ((fx + fw / 2) - (sx + sw / 2)) ** 2
                + ((fy + fh / 2) - (sy + sh / 2)) ** 2
            ) ** 0.5
            duplicate = (
                iou >= 0.70
                or (
                    containment >= 0.70
                    and area_ratio >= 1.20
                    and center_distance <= 0.35 * max(fw, fh, sw, sh)
                )
            )
            if duplicate:
                preferred_index = smaller_index
                preferred = kept[preferred_index]
                left = min(fx, sx)
                top = min(fy, sy)
                right = max(fx + fw, sx + sw)
                bottom = max(fy + fh, sy + sh)
                preferred["box"] = [left, top, right - left, bottom - top]
                kept[preferred_index] = preferred
                remove.add(larger_index)
    return [region for index, region in enumerate(kept) if index not in remove]


def widen_midpage_handwriting_lines(
    regions: list[dict], page_width: int, page_height: int,
) -> list[dict]:
    """Restore horizontal context for clipped handwritten lines in document bodies."""
    widened = [dict(region, box=list(region["box"])) for region in regions]
    for region in widened:
        x, y, width, height = region["box"]
        center_y = y + height / 2
        label = region.get("label", "").lower()
        is_line = any(term in label for term in ("line", "note", "name entry"))
        if not (
            is_line
            and width >= 2.4 * height
            and page_height * 0.28 <= center_y <= page_height * 0.48
            and width < page_width * 0.72
        ):
            continue
        left = min(x, round(page_width * 0.12))
        right = max(x + width, round(page_width * 0.90))
        region["box"] = [left, y, min(page_width, right) - left, height]
    return widened


def needs_lower_detection_pass(regions: list[dict], page_height: int) -> bool:
    """Request a detailed lower-page scan when full-page detection only saw the header."""
    if not regions:
        return True
    lowest_center = max(
        region["box"][1] + region["box"][3] / 2
        for region in regions
    )
    return lowest_center < page_height * 0.30


def detect_handwriting_regions(page_img: Image.Image, det_max_side: int = 3000) -> list[dict]:
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
For a page dominated by many consecutive full-width handwritten lines, return
exactly 3 contiguous paragraph blocks instead of one region per line.

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

    inv = 1.0 / scale
    pad_x, pad_y = W * 0.008, H * 0.006

    def parse_response(
        response: dict, coordinate_inv: float = inv, offset_y: int = 0,
    ) -> list[dict]:
        regions: list[dict] = []
        for region in (response.get("regions") or []):
            box = region.get("box")
            if not (isinstance(box, list) and len(box) == 4):
                continue
            try:
                x, y, width, height = (
                    float(value) * coordinate_inv for value in box
                )
                y += offset_y
            except (TypeError, ValueError):
                continue
            bx, by = max(0, int(x - pad_x)), max(0, int(y - pad_y))
            bw, bh = int(width + 2 * pad_x), int(height + 2 * pad_y)
            bw, bh = min(bw, W - bx), min(bh, H - by)
            if bw <= 10 or bh <= 10:
                continue
            label = str(region.get("label") or "handwriting")
            if "paragraph block" in label.lower():
                contextual = [bx, by, bw, bh]
            else:
                refined = refine_box_to_ink(page_img, [bx, by, bw, bh], color_hint=label)
                contextual = (
                    refined
                    if any(term in label.lower() for term in ("blue", "green", "phone", "email"))
                    else add_handwriting_context(page_img, refined)
                )
            regions.append({"label": label, "box": contextual})
        regions = normalize_detected_paragraph_blocks(regions, W, H)
        regions = group_dense_handwriting_lines(regions, W, H)
        return separate_overlapping_line_regions(regions)

    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": _to_data_uri(det_img)},
    ]
    regions = parse_response(_responses_call(content))
    penalty, issues = detection_geometry_penalty(regions, W, H)
    if issues:
        correction = (
            "Your previous detection had invalid geometry:\n- "
            + "\n- ".join(issues)
            + "\nDetect the page again. Keep numbers, IDs, names, and short field values "
              "local to their form field. Do not follow printed underlines or combine distant fields."
        )
        retry_content = [
            {"type": "input_text", "text": prompt + "\n\n" + correction},
            {"type": "input_image", "image_url": _to_data_uri(det_img)},
        ]
        retry_regions = parse_response(_responses_call(retry_content))
        repaired = repair_invalid_detection_regions(regions, retry_regions, W, H)
        repaired_penalty, _ = detection_geometry_penalty(repaired, W, H)
        if repaired_penalty < penalty:
            print(f"[detect] geometry repair improved penalty {penalty:.1f} -> {repaired_penalty:.1f}")
            regions = repaired
    if needs_lower_detection_pass(regions, H):
        lower_top = round(H * 0.20)
        lower_page = page_img.crop((0, lower_top, W, H))
        lower_scale = min(1.0, det_max_side / max(lower_page.size))
        lower_width = int(lower_page.width * lower_scale)
        lower_height = int(lower_page.height * lower_scale)
        lower_image = lower_page.resize(
            (lower_width, lower_height), Image.LANCZOS
        ) if lower_scale < 1.0 else lower_page
        lower_prompt = f"""This is a detailed view of the LOWER 80% of a scanned Hebrew document,
{lower_width}x{lower_height} pixels. Find every true handwritten pen entry in this
view that a whole-page scan may have missed. Include long handwriting crossing
printed form lines. Exclude all machine-printed text, italic printed text, logos,
stamps, signatures, and black redaction boxes.

Return ONLY one JSON object with coordinates in THIS image:
{{"regions": [{{"label": "short english description", "box": [x, y, width, height]}}]}}
Each box must tightly cover one handwritten field or line with a small margin."""
        lower_content = [
            {"type": "input_text", "text": lower_prompt},
            {"type": "input_image", "image_url": _to_data_uri(lower_image)},
        ]
        lower_regions = parse_response(
            _responses_call(lower_content),
            coordinate_inv=1.0 / lower_scale,
            offset_y=lower_top,
        )
        if lower_regions:
            print(f"[detect] lower-page coverage pass added {len(lower_regions)} candidate region(s)")
            regions.extend(lower_regions)
    regions = deduplicate_detection_regions(regions, W, H)
    return widen_midpage_handwriting_lines(regions, W, H)


def normalize_detected_paragraph_blocks(
    regions: list[dict], page_width: int, page_height: int,
) -> list[dict]:
    """Normalize a dense-page model response into three disjoint vertical blocks."""
    paragraph_regions = [
        region for region in regions
        if "paragraph block" in region["label"].lower()
    ]
    if len(paragraph_regions) < 3 or len(paragraph_regions) < len(regions) * 0.6:
        return regions
    writing_regions = paragraph_regions + [
        region for region in regions if region not in paragraph_regions
    ]
    left = max(0, min(region["box"][0] for region in writing_regions))
    top = max(0, min(region["box"][1] for region in writing_regions))
    right = min(page_width, max(region["box"][0] + region["box"][2] for region in writing_regions))
    bottom = min(page_height, max(region["box"][1] + region["box"][3] for region in writing_regions))
    margin = max(8, round(page_height * 0.004))
    boundaries = [round(top + (bottom - top) * index / 3) for index in range(4)]
    return [
        {
            "label": f"handwritten paragraph block {index + 1}",
            "box": [
                left,
                max(top, boundaries[index] - (margin if index else 0)),
                right - left,
                min(bottom, boundaries[index + 1] + (margin if index < 2 else 0))
                - max(top, boundaries[index] - (margin if index else 0)),
            ],
        }
        for index in range(3)
    ]


def group_dense_handwriting_lines(
    regions: list[dict], page_width: int, page_height: int,
) -> list[dict]:
    """Merge line-by-line detections when handwriting fills most of a page."""
    if len(regions) < 8:
        return regions
    line_like = [
        region for region in regions
        if region["box"][2] >= page_width * 0.55
        and region["box"][3] <= page_height * 0.12
    ]
    if len(line_like) < max(8, round(len(regions) * 0.7)):
        return regions

    ordered = sorted(regions, key=lambda region: region["box"][1] + region["box"][3] / 2)
    group_count = max(2, min(4, round(len(ordered) / 7)))
    groups = [
        ordered[start * len(ordered) // group_count:(start + 1) * len(ordered) // group_count]
        for start in range(group_count)
    ]
    merged = []
    for index, group in enumerate(groups, 1):
        left = min(region["box"][0] for region in group)
        top = min(region["box"][1] for region in group)
        right = max(region["box"][0] + region["box"][2] for region in group)
        bottom = max(region["box"][1] + region["box"][3] for region in group)
        merged.append({
            "label": f"handwritten paragraph block {index}",
            "box": [left, top, right - left, bottom - top],
        })
    return merged


def separate_overlapping_line_regions(regions: list[dict]) -> list[dict]:
    """Split heavily overlapping, vertically stacked long-line crops."""
    separated = [dict(region, box=list(region["box"])) for region in regions]
    wide = [
        (index, region)
        for index, region in enumerate(separated)
        if region["box"][2] >= 2.5 * region["box"][3]
    ]
    wide.sort(key=lambda item: item[1]["box"][1] + item[1]["box"][3] / 2)
    for (upper_index, upper), (lower_index, lower) in zip(wide, wide[1:]):
        ux, uy, uw, uh = upper["box"]
        lx, ly, lw, lh = lower["box"]
        horizontal_overlap = max(0, min(ux + uw, lx + lw) - max(ux, lx))
        vertical_overlap = max(0, min(uy + uh, ly + lh) - max(uy, ly))
        if horizontal_overlap < 0.6 * min(uw, lw):
            continue
        if vertical_overlap < 0.35 * min(uh, lh):
            continue
        # Long-line context trims 15% above and adds 50% below. Its geometric
        # center is therefore well below the actual handwriting center.
        upper_ink_center = uy + uh * (0.35 / 1.35)
        lower_ink_center = ly + lh * (0.35 / 1.35)
        boundary = round((upper_ink_center + lower_ink_center) / 2)
        margin = max(6, round(min(uh, lh) * 0.03))
        minimum_upper_end = uy + round(uh * 0.70)
        upper_end = max(boundary + margin, minimum_upper_end)
        upper["box"][3] = max(1, upper_end - uy)
        lower_end = ly + lh
        maximum_lower_start = lower_end - round(lh * 0.70)
        lower["box"][1] = max(0, min(boundary - margin, maximum_lower_start))
        lower["box"][3] = max(1, lower_end - lower["box"][1])
        separated[upper_index] = upper
        separated[lower_index] = lower
    return separated


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
) -> tuple[list[tuple[str, Image.Image]], list[tuple[str, Image.Image]]]:
    """Crop + enhance + chunk a region and save the images. Returns
    (ocr_tasks, all_images): ocr_tasks is the reduced set sent to per-image OCR
    (#4); all_images is the full set kept for the visual-reconcile authority."""
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

    all_images = list(versions.items()) + list(chunk_images.items())
    ocr_tasks: list[tuple[str, Image.Image]] = []
    if ocr_mode in ("full", "both"):
        ocr_tasks += list(versions.items())
    if ocr_mode in ("chunks", "both"):
        ocr_tasks += list(chunk_images.items())
    return ocr_tasks, all_images


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
    _reset_usage()
    _t_start = time.time()

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

        def _run_cu_and_publish() -> str:
            markdown = cu_ocr_markdown(input_path, page=page)
            if markdown:
                _write_text_atomic(out / "content_understanding.md", markdown)
                print("[CU] Content Understanding result ready for review.")
            return markdown

        cu_future = cu_executor.submit(_run_cu_and_publish)

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

    _write_json_atomic(out / "detected_regions.json", {
        "page": page,
        "count": len(regions),
        "regions": [
            {"region": index, "label": region.get("label"), "box": region["box"]}
            for index, region in enumerate(regions, 1)
        ],
        "updated_at": time.time(),
    })

    # ---- region map overlay ----------------------------------------------- #
    overlay_name = f"page_{page}_regions.png"
    draw_regions_overlay(page_img, regions).save(out / overlay_name)
    print(f"Region map (numbered boxes) -> {out / overlay_name}")

    # ---- prepare every region's images (CPU, fast) ------------------------ #
    prompt = build_ocr_prompt(printed_context, italic=italic)
    region_dirs = [out / f"region_{i:02d}_{_slug(r.get('label', ''))}"
                   for i, r in enumerate(regions, 1)]
    region_prep = [
        prepare_region_images(page_img, region, str(region_dirs[i]),
                              ocr_mode, chunk_width, overlap)
        for i, region in enumerate(regions)
    ]
    region_tasks = [ocr for ocr, _all in region_prep]         # images sent to per-image OCR (#4)
    region_images = [allimg for _ocr, allimg in region_prep]  # full image set for visual reconcile

    # ---- run ALL vision calls in parallel across ALL regions -------------- #
    # Self-consistency: draw each image version VOTE_SEEDS times (distinct seeds)
    # so the deterministic aggregation has multiple genuine samples to vote over.
    vote_seeds = list(range(1, VOTE_SEEDS + 1)) if VOTE_ENABLED else [None]
    flat = [(ri, f"{name}#s{s}" if s is not None else name, img, s)
            for ri, tasks in enumerate(region_tasks)
            for name, img in tasks
            for s in vote_seeds]
    region_results: list[list[dict]] = [[] for _ in regions]
    draw_progress = {"completed": 0, "failed": 0, "total": len(flat)}

    def _publish_ocr_progress() -> None:
        _write_json_atomic(out / "ocr_progress.json", {
            **draw_progress,
            "page": page,
            "updated_at": time.time(),
            "regions": [
                {
                    "region": index + 1,
                    "label": regions[index].get("label"),
                    "box": regions[index]["box"],
                    "completed_draws": len(results),
                    "candidates": [
                        {
                            "source": result["source"],
                            "text": (result.get("ocr") or {}).get("full_transcription", ""),
                            "words": (result.get("ocr") or {}).get("words", []),
                        }
                        for result in results
                    ],
                }
                for index, results in enumerate(region_results)
            ],
        })

    _publish_ocr_progress()
    print(f"[OCR] {len(flat)} vision draws across {len(regions)} region(s) "
          f"({len(vote_seeds)} seed(s)/version), {MAX_WORKERS} in parallel ...")

    def _ocr_batch(tasks):
        """Run a batch of (region_idx, name, image, seed) OCR draws in parallel.
        Appends successes to region_results; returns the tasks that failed."""
        failed = []
        workers = max(1, min(MAX_WORKERS, len(tasks)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_meta = {ex.submit(ocr_image, img, prompt, seed): (ri, name, img, seed)
                        for ri, name, img, seed in tasks}
            for fut in as_completed(fut_meta):
                ri, name, img, seed = fut_meta[fut]
                try:
                    region_results[ri].append({"source": name, "ocr": fut.result()})
                    draw_progress["completed"] += 1
                except Exception as exc:
                    failed.append((ri, name, img, seed))
                    draw_progress["failed"] += 1
                    print(f"   region {ri + 1} {name} failed: {exc}", file=sys.stderr)
                _publish_ocr_progress()
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
        retry_tasks = [(ri, f"{name}#r{k}", img, 900 + k)
                       for ri in stranded for k, (name, img) in enumerate(region_tasks[ri])]
        print(f"[OCR] {len(stranded)} region(s) still empty -> final retry of {len(retry_tasks)} call(s) ...")
        time.sleep(5)
        _ocr_batch(retry_tasks)
    if failures:
        print(f"[OCR] WARNING: {len(failures)} call(s) still failing after all retries", file=sys.stderr)

    # ---- final aggregation per region (deterministic vote -> visual -> text) - #
    print("[vote] aggregating self-consistency draws per region ..." if VOTE_ENABLED
          else "[reconcile] final visual adjudication per region ...")

    def _reconcile_region(ri: int) -> dict:
        res = region_results[ri]
        usable = [
            result for result in res
            if (result.get("ocr") or {}).get("full_transcription", "").strip()
        ]
        if not usable:
            return {
                "final_transcription": "",
                "words": [],
                "notes": ["no non-empty OCR candidates; skipped visual reconciliation"],
                "needs_review": True,
                "decision_status": "empty_candidate_set",
            }
        if VOTE_ENABLED:
            readings = [(r.get("ocr") or {}).get("full_transcription", "") for r in usable]
            if readings:
                vote = aggregate_votes(readings)
                if "[?]" not in vote.get("final_transcription", ""):
                    return vote  # deterministic agreement; no model call
        imgs = region_images[ri]
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

        sel = _select_visual_images(region_images[i - 1], res)
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
            "needs_review": final.get("needs_review", False),
            "decision_status": final.get("decision_status", "visual_reconciliation"),
            "agreement_gate": final.get("agreement_gate"),
            "vote": {
                "candidate_count": len(per_image),
                "selected_text": final.get("final_transcription", ""),
                "alternatives": final.get("alternative_transcriptions", []),
                "method": (final.get("notes") or ["unknown"])[0],
            },
        }
        (rdir / "result.json").write_text(
            json.dumps(result_debug, ensure_ascii=False, indent=2), encoding="utf-8")

        all_results.append({
            "region": i, "folder": rdir.name,
            "label": region.get("label"), "box": region["box"],
            "final_transcription": final.get("final_transcription"),
            "words": final.get("words"), "notes": final.get("notes"),
            "needs_review": final.get("needs_review", False),
            "decision_status": final.get("decision_status", "visual_reconciliation"),
            "agreement_gate": final.get("agreement_gate"),
        })
        print(f"  [{i}] {region.get('label')}: {final.get('final_transcription', '')!r}")

    combined = {
        "source": input_path, "page": page,
        "region_map": overlay_name, "regions": all_results,
    }
    # handwriting_result.json is written AFTER the merge step below so the token
    # totals include the merge call and the elapsed time covers the whole run.

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
    if cu_markdown:
        _write_text_atomic(out / "content_understanding.md", cu_markdown)

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

    # ---- usage + timing --------------------------------------------------- #
    usage = _get_usage()
    elapsed = time.time() - _t_start
    combined["usage"] = usage
    combined["elapsed_seconds"] = round(elapsed, 1)
    combined["elapsed"] = _fmt_mmss(elapsed)
    (out / "handwriting_result.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- summary ---------------------------------------------------------- #
    print("\n" + "=" * 70)
    print(f"ALL HANDWRITING REGIONS ({len(all_results)})")
    print("=" * 70)
    for r in all_results:
        print(f"  [{r['region']}] {r.get('label')}: {r.get('final_transcription')}")
    print(f"\nTotal time:    {combined['elapsed']} (mm:ss)")
    print(f"GPT tokens:    {usage['input_tokens']:,} in + {usage['output_tokens']:,} out "
          f"= {usage['total_tokens']:,} total  ({usage['calls']} GPT calls)")
    print(f"Region map:    {out / overlay_name}")
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


def _parse_pages(spec: str | None, total: int, single: int | None) -> list[int]:
    """Resolve pages to process. Priority: --page (single) > --pages (spec) > all.
    Values are clamped to [1, total]."""
    if single:
        return [single] if 1 <= single <= total else []
    if spec:
        out: set[int] = set()
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(part))
        return sorted(p for p in out if 1 <= p <= total)
    return list(range(1, total + 1))


def run_document(
    input_path: str, pages: list[int], dpi: int,
    crops: list[tuple[int, int, int, int]] | None, output_dir: str,
    chunk_width: int, overlap: int, ocr_mode: str,
    auto_detect: bool, max_regions: int | None,
    printed_context: str, italic: bool, merge_md: bool,
) -> int:
    """Process one or many pages. A single page writes straight into output_dir;
    multiple pages get one subfolder per page plus a combined document.md."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    multi = len(pages) > 1
    rc, page_docs = 0, []
    for i, page in enumerate(pages, 1):
        page_out = out / f"page_{page:04d}" if multi else out
        if multi:
            print(f"\n{'#' * 70}\n# PAGE {page}  ({i}/{len(pages)})\n{'#' * 70}")
        rc = run_pipeline(
            input_path=input_path, page=page, dpi=dpi, crops=crops,
            output_dir=str(page_out), chunk_width=chunk_width, overlap=overlap,
            ocr_mode=ocr_mode, auto_detect=auto_detect, max_regions=max_regions,
            printed_context=printed_context, italic=italic, merge_md=merge_md,
        ) or rc
        md = page_out / "merged.md"
        if multi and md.exists():
            page_docs.append((page, md.read_text(encoding="utf-8").strip()))
    if multi and page_docs:
        combined = "\n\n".join(
            f"<!-- ===================== Page {p} ===================== -->\n\n{t}"
            for p, t in page_docs)
        (out / "document.md").write_text(combined + "\n", encoding="utf-8")
        print(f"\n{'=' * 70}\nDOCUMENT: {len(page_docs)} page(s) -> {out / 'document.md'}\n{'=' * 70}")
    return rc


def main() -> int:
    global MAX_WORKERS
    ap = argparse.ArgumentParser(description="Hebrew handwriting OCR helper pipeline")
    ap.add_argument("--input", required=True, help="PDF or image path")
    ap.add_argument("--page", type=int, default=None, help="a single PDF page (1-based); default: ALL pages")
    ap.add_argument("--pages", default=None, help='pages to process, e.g. "1-3,5"; default: all pages')
    ap.add_argument("--crop", action="append", default=None,
                    help='handwriting rectangle "x,y,w,h" in px; repeat for multiple regions')
    ap.add_argument("--auto-detect", action="store_true",
                    help="auto-locate ALL handwriting regions with the Vision LLM")
    ap.add_argument("--max-regions", type=int, default=None, help="cap number of regions processed")
    ap.add_argument("--output", default="./output_handwriting", help="output folder")
    ap.add_argument("--dpi", type=int, default=400, help="PDF render DPI (300-600)")
    ap.add_argument("--chunk-width", type=int, default=900)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--ocr-mode", choices=["full", "chunks", "both"], default="full")
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
                    help="(default ON) produce merged.md: CU printed text + GPT handwriting")
    ap.add_argument("--no-merge", action="store_true", help="skip merged.md")
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
    # Smart defaults: auto-detect handwriting when no manual crop is given; merge
    # unless --no-merge. So `--input doc.pdf` alone does the right thing.
    auto_detect = args.auto_detect or (crops is None)
    merge_md = not args.no_merge

    # Smart page handling: a PDF processes ALL pages by default; an image is 1 page.
    ext = Path(args.input).suffix.lower()
    total = pdf_page_count(args.input) if ext == ".pdf" else 1
    pages = _parse_pages(args.pages, total, args.page)
    if not pages:
        print(f"No pages to process (input has {total} page(s)).", file=sys.stderr)
        return 1
    if len(pages) > 1:
        print(f"[document] {args.input}: {total} page(s); processing {len(pages)} "
              f"(pages {pages[0]}-{pages[-1]}).")
        print("[document] runs the pipeline per page (~Nx the single-page time). "
              "Use --pages 1-3 to limit, or OCR_REASONING=low to go faster.")

    return run_document(
        input_path=args.input, pages=pages, dpi=args.dpi, crops=crops,
        output_dir=args.output, chunk_width=args.chunk_width, overlap=args.overlap,
        ocr_mode=args.ocr_mode, auto_detect=auto_detect, max_regions=args.max_regions,
        printed_context=printed_context, italic=args.italic, merge_md=merge_md,
    )


if __name__ == "__main__":
    raise SystemExit(main())
