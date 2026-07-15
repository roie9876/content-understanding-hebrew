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
import atexit
import base64
import collections
import difflib
import io
import json
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
import uuid
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
from dotenv import load_dotenv
from segmented_ocr import (
    PROMPT_A as SEGMENTED_PROMPT_A,
    PROMPT_B as SEGMENTED_PROMPT_B,
    blue_pen_mask,
    build_line_view_set,
    build_word_view_sets,
    segment_handwritten_lines,
    segment_word_groups,
)

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
GPT_ENDPOINT = os.environ.get(
    "GPT_ENDPOINT",
    "https://example.invalid/openai/v1/",
).rstrip("/") + "/"
GPT_DEPLOYMENT = os.environ.get("GPT_DEPLOYMENT", "gpt-5.4")
GPT_TARGETS_JSON = os.environ.get("GPT_TARGETS", "")
GPT_REASONING = os.environ.get("GPT_REASONING", "medium")  # low | medium | high
MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "azure-responses")
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
OCR_MAX_TOKENS = max(1800, int(os.environ.get("OCR_MAX_TOKENS", "10000")))
OCR_FINAL_ORIGINAL_RESCUE = os.environ.get("OCR_FINAL_ORIGINAL_RESCUE", "1") != "0"
DETECTION_REASONING = os.environ.get("DETECTION_REASONING", GPT_REASONING)
DETECTION_MAX_TOKENS = max(1000, int(os.environ.get("DETECTION_MAX_TOKENS", "6000")))
OCR_STRATEGY = os.environ.get("OCR_STRATEGY", "standard")
ISOLATED_MULTIVIEW_STRATEGY = "isolated-multiview"
ISOLATED_MULTIVIEW_WORKERS = max(1, int(os.environ.get("ISOLATED_MULTIVIEW_WORKERS", "2")))
ISOLATED_MULTIVIEW_MAX_TOKENS = max(
    1800, int(os.environ.get("ISOLATED_MULTIVIEW_MAX_TOKENS", "1800"))
)

# Content Understanding (for the printed-text layer used in the merged Markdown).
CU_ENDPOINT = os.environ.get(
    "CU_ENDPOINT",
    "https://example.invalid/",
).rstrip("/") + "/"
CU_API_VERSION = os.environ.get("CU_API_VERSION", "2025-11-01")

GPT_TEMPERATURE = float(os.environ.get("GPT_TEMPERATURE", "0"))
GPT_SEED = int(os.environ.get("GPT_SEED", "7"))
GPT_READ_TIMEOUT = int(os.environ.get("GPT_READ_TIMEOUT", "90"))  # per-call read timeout (s); stuck calls retry fast
IVRIT_OCR_URL = os.environ.get("IVRIT_OCR_URL", "http://127.0.0.1:8000").rstrip("/")
IVRIT_OCR_TIMEOUT = max(1, int(os.environ.get("IVRIT_OCR_TIMEOUT", "60")))
IVRIT_OCR_ENABLED = os.environ.get("IVRIT_OCR_ENABLED", "1") not in (
    "", "0", "false", "False",
)
IVRIT_OCR_WORKERS = max(1, int(os.environ.get("IVRIT_OCR_WORKERS", "1")))
IVRIT_OCR_SOURCE = "ivrit_hash_v1"
IVRIT_OCR_MIN_AGREEMENT = float(
    os.environ.get("IVRIT_OCR_MIN_AGREEMENT", "0.40")
)
IVRIT_OCR_MIN_PREPROCESS_AGREEMENT = float(
    os.environ.get("IVRIT_OCR_MIN_PREPROCESS_AGREEMENT", "0.55")
)
CONTEXTUAL_RECHECK_MIN_CONFIDENCE = float(
    os.environ.get("CONTEXTUAL_RECHECK_MIN_CONFIDENCE", "0.60")
)

# Visual adjudication is the quality authority by default: seven enhancement reads
# become evidence, then the model re-reads the strokes to select the final text.
# Self-consistency voting remains opt-in for experiments.
VOTE_ENABLED = os.environ.get("VOTE", "0") not in ("", "0", "false", "False")
VOTE_SEEDS = int(os.environ.get("VOTE_SEEDS", "3"))  # OCR draws per image version

CORE_OCR_VIEWS = (
    "01_original_crop.png",
    "02_upscaled_contrast.png",
    "04_denoised.png",
    "06_optional_line_removed.png",
)
RETRYABLE_OCR_VIEWS = {
    "01_original_crop.png",
    "04_denoised.png",
}
MULTI_IMAGE_OCR = os.environ.get("OCR_MULTI_IMAGE", "1") not in ("", "0", "false", "False")
SPATIAL_NUMERIC_CELLS = os.environ.get("OCR_SPATIAL_NUMERIC_CELLS", "0") not in (
    "", "0", "false", "False"
)
NEARBY_CU_CONTEXT = os.environ.get("OCR_NEARBY_CU_CONTEXT", "0") not in (
    "", "0", "false", "False"
)

# Domain-agnostic: the handwriting OCR uses NO built-in vocabulary and makes no
# assumption about document type. Optionally, a short printed-text context string
# (from the separate Content Understanding pipeline) may be passed as a SOFT hint
# only — it never overrides the visible handwriting strokes.

_credential = AzureCliCredential()
_token_cache: dict[str, tuple[float, str]] = {}
_token_lock = threading.Lock()


def _responses_url(endpoint: str) -> str:
    normalized = endpoint.rstrip("/")
    return normalized if normalized.endswith("/responses") else f"{normalized}/responses"


def _load_gpt_targets() -> list[dict[str, str]]:
    if GPT_TARGETS_JSON:
        raw_targets = json.loads(GPT_TARGETS_JSON)
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ValueError("GPT_TARGETS must be a non-empty JSON array")
        targets = []
        for index, raw in enumerate(raw_targets, 1):
            if not isinstance(raw, dict) or not raw.get("endpoint"):
                raise ValueError(f"GPT_TARGETS entry {index} requires an endpoint")
            targets.append({
                "name": str(raw.get("name") or f"endpoint-{index}"),
                "url": _responses_url(str(raw["endpoint"])),
                "deployment": str(raw.get("deployment") or GPT_DEPLOYMENT),
            })
        return targets
    targets = [{
        "name": "primary",
        "url": _responses_url(GPT_ENDPOINT),
        "deployment": GPT_DEPLOYMENT,
    }]
    return targets


GPT_TARGETS = _load_gpt_targets()
_target_lock = threading.Lock()
_target_cursor = 0


def _ordered_gpt_targets() -> list[dict[str, str]]:
    """Round-robin the first target while preserving deterministic failover order."""
    global _target_cursor
    with _target_lock:
        start = _target_cursor % len(GPT_TARGETS)
        _target_cursor += 1
    return GPT_TARGETS[start:] + GPT_TARGETS[:start]


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
_usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "targets": {}}


def _reset_usage() -> None:
    with _usage_lock:
        _usage.update(input_tokens=0, output_tokens=0, calls=0, targets={})


def _add_usage(resp: dict, target_name: str) -> None:
    """Accumulate token usage from a GPT Responses API JSON response."""
    u = resp.get("usage") or {}
    it = u.get("input_tokens", u.get("prompt_tokens", 0)) or 0
    ot = u.get("output_tokens", u.get("completion_tokens", 0)) or 0
    with _usage_lock:
        _usage["input_tokens"] += int(it)
        _usage["output_tokens"] += int(ot)
        _usage["calls"] += 1
        targets = _usage["targets"]
        targets[target_name] = targets.get(target_name, 0) + 1


def _add_call_usage(target_name: str) -> None:
    """Record a model call when its SDK does not expose token counts."""
    with _usage_lock:
        _usage["calls"] += 1
        targets = _usage["targets"]
        targets[target_name] = targets.get(target_name, 0) + 1


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


def nearby_printed_context(
    cu_markdown: str, box: list[int], page_height: int, window: int = 4,
) -> str:
    """Select a soft CU reading-order window near a region's vertical position."""
    lines = []
    for raw_line in (cu_markdown or "").splitlines():
        line = raw_line.strip()
        if (
            not line
            or line.startswith("<!--")
            or line.startswith("![")
            or re.fullmatch(r"</?(?:table|tr|thead|tbody)>", line, re.I)
        ):
            continue
        line = re.sub(r"</?(?:td|th)(?:\s[^>]*)?>", " ", line, flags=re.I)
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = " ".join(line.split())
        if line:
            lines.append(line)
    if not lines or page_height <= 0:
        return ""
    _x, y, _width, height = box
    position = min(1.0, max(0.0, (y + height / 2) / page_height))
    center = min(len(lines) - 1, round(position * (len(lines) - 1)))
    return "\n".join(lines[max(0, center - window):center + window + 1])


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
- Read every numeric sequence left-to-right. Preserve the physical left-to-right order of
    separate numeric cells or groups; never reverse numeric groups because the form is Hebrew.
- Transcribe only the {written} text.
- Ignore printed body text, form lines, stamps, and black redactions.
- Describe visible X/checkbox marks in the structured mark fields.
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
    "marks": ["x"],
    "digits": ["1"],
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


class _CopilotSdkBridge:
    """One local Copilot CLI runtime shared by concurrent OCR sessions."""

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.start_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending: dict[str, object] = {}

    def _start(self) -> None:
        with self.start_lock:
            if self.process is not None and self.process.poll() is None:
                return
            self.process = subprocess.Popen(
                ["node", str(Path(__file__).with_name("copilot_sdk_bridge.mjs"))],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
                bufsize=1,
                cwd=str(Path(__file__).resolve().parent),
                start_new_session=True,
            )
            ready_line = self.process.stdout.readline() if self.process.stdout else ""
            if not ready_line:
                raise RuntimeError("GitHub Copilot SDK bridge exited during startup")
            ready = json.loads(ready_line)
            models = {model["id"]: model for model in ready.get("models", [])}
            model = models.get(GPT_DEPLOYMENT)
            if ready.get("type") != "ready" or not model or not model.get("vision") \
                    or model.get("policy") == "disabled":
                self.stop()
                raise RuntimeError(
                    f"Copilot model {GPT_DEPLOYMENT!r} is unavailable or lacks vision support"
                )
            threading.Thread(target=self._read_responses, daemon=True).start()

    def _read_responses(self) -> None:
        if not self.process or not self.process.stdout:
            return
        for line in self.process.stdout:
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            with self.pending_lock:
                destination = self.pending.get(response.get("id"))
            if destination:
                destination.put(response)

    def call(self, content: list[dict], timeout_seconds: int) -> str:
        import queue

        self._start()
        request_id = uuid.uuid4().hex
        destination = queue.Queue(maxsize=1)
        with self.pending_lock:
            self.pending[request_id] = destination
        try:
            request = {
                "id": request_id,
                "model": GPT_DEPLOYMENT,
                "content": content,
                "timeoutSeconds": timeout_seconds,
            }
            with self.write_lock:
                if not self.process or not self.process.stdin:
                    raise RuntimeError("GitHub Copilot SDK bridge is not running")
                self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                self.process.stdin.flush()
            try:
                response = destination.get(timeout=timeout_seconds + 15)
            except queue.Empty as exc:
                raise RuntimeError("GitHub Copilot SDK request timed out") from exc
            if not response.get("ok"):
                raise RuntimeError(response.get("error") or "GitHub Copilot SDK request failed")
            _add_call_usage("github-copilot-sdk")
            return str(response.get("text") or "")
        finally:
            with self.pending_lock:
                self.pending.pop(request_id, None)

    def stop(self) -> None:
        process, self.process = self.process, None
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, 15)
            except OSError:
                process.terminate()


_copilot_bridge = _CopilotSdkBridge()
atexit.register(_copilot_bridge.stop)


def _responses_text(content: list[dict], retries: int = 4,
                    max_tokens: int = 6000, reasoning: str | None = None,
                    seed: int | None = None,
                    endpoint_limit: int | None = None) -> str:
    """POST to the Responses API and return the raw output text.

    Robust to timeouts / 429 / 5xx / empty output: retries with exponential
    backoff + jitter so a transient burst never loses the call.
    """
    if MODEL_BACKEND == "github-copilot-sdk":
        last_error = ""
        for attempt in range(retries + 1):
            try:
                text = _copilot_bridge.call(content, GPT_READ_TIMEOUT)
                if text.strip():
                    return text
                last_error = "empty output"
            except Exception as exc:
                last_error = str(exc)
            if attempt < retries:
                time.sleep(min(4.0, 2.0 ** attempt) + random.uniform(0, 0.5))
        raise RuntimeError(f"[Copilot SDK] {last_error}")

    use_seed = GPT_SEED if seed is None else seed
    payload = {
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": max_tokens,
        "reasoning": {"effort": reasoning or GPT_REASONING},
        "temperature": GPT_TEMPERATURE,
        "seed": use_seed,
    }
    for _p in _unsupported_params:
        payload.pop(_p, None)
    last_err = ""
    for attempt in range(retries + 1):
        backoff = min(30.0, 2.0 ** attempt) + random.uniform(0, 1.5)
        retry_after = backoff
        ordered_targets = _ordered_gpt_targets()
        if endpoint_limit is not None:
            ordered_targets = ordered_targets[:max(1, endpoint_limit)]
        for target in ordered_targets:
            target_payload = {**payload, "model": target["deployment"]}
            try:
                r = requests.post(
                    target["url"],
                    headers={
                        "Authorization": f"Bearer {_bearer()}",
                        "Content-Type": "application/json",
                    },
                    data=json.dumps(target_payload),
                    timeout=GPT_READ_TIMEOUT,
                )
            except requests.exceptions.RequestException as exc:
                last_err = f"{target['name']} request error: {exc}"
                continue

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
                        endpoint_limit=endpoint_limit,
                    )
            if r.status_code in (408, 409, 429, 500, 502, 503, 504):
                last_err = f"{target['name']} HTTP {r.status_code}"
                retry_after = max(
                    retry_after,
                    float(r.headers.get("retry-after", backoff)),
                )
                continue
            if r.status_code != 200:
                raise RuntimeError(
                    f"[GPT] {target['name']} HTTP {r.status_code}: {r.text[:400]}"
                )
            resp_json = r.json()
            _add_usage(resp_json, target["name"])
            text = _gpt_output_text(resp_json)
            if text.strip():
                return text
            last_err = f"{target['name']} empty output"
            payload["max_output_tokens"] = min(
                payload.get("max_output_tokens", max_tokens) + 4000,
                32000,
            )
        if attempt < retries:
            time.sleep(retry_after)
            continue
        raise RuntimeError(f"[GPT] all endpoints failed: {last_err}")

    raise RuntimeError(f"[GPT] failed after retries: {last_err}")


def _responses_call(content: list[dict], retries: int = 2,
                    reasoning: str | None = None, max_tokens: int = 6000,
                    seed: int | None = None,
                    endpoint_limit: int | None = None) -> dict:
    return _loads_lenient(_responses_text(content, retries, max_tokens=max_tokens,
                                          reasoning=reasoning, seed=seed,
                                          endpoint_limit=endpoint_limit))


def ocr_image_with_ivrit(image: Image.Image) -> dict:
    """Send one region crop to the optional HASH v1 OCR sidecar."""
    image_bytes = io.BytesIO()
    image.convert("RGB").save(image_bytes, format="PNG")
    response = requests.post(
        f"{IVRIT_OCR_URL}/v1/ocr",
        files={"file": ("region.png", image_bytes.getvalue(), "image/png")},
        timeout=(3, IVRIT_OCR_TIMEOUT),
    )
    response.raise_for_status()
    payload = response.json()
    lines = []
    for page in payload.get("pages") or []:
        for line in page.get("lines") or []:
            text = str(line.get("text") or "").strip()
            if not text:
                continue
            lines.append({
                "text": text,
                "confidence": float(line.get("confidence") or 0.0),
                "box": dict(line.get("box") or {}),
            })
    text = str(payload.get("text") or "").strip()
    if not text:
        text = "\n".join(line["text"] for line in lines)
    confidence = (
        sum(line["confidence"] for line in lines) / len(lines)
        if lines else 0.0
    )
    return {
        "status": "complete",
        "source": IVRIT_OCR_SOURCE,
        "model": "HASH v1",
        "text": text,
        "confidence": confidence,
        "lines": lines,
        "raw_response": payload,
    }


def ivrit_ocr_service_status() -> tuple[bool, str]:
    """Check the optional sidecar once so absent Docker never fails a page."""
    if not IVRIT_OCR_ENABLED:
        return False, "disabled by IVRIT_OCR_ENABLED"
    try:
        response = requests.get(f"{IVRIT_OCR_URL}/health", timeout=(1, 2))
        response.raise_for_status()
        status = str((response.json() or {}).get("status") or "")
        if status != "ok":
            return False, f"service reported {status or 'unknown status'}"
    except (requests.RequestException, ValueError, TypeError) as exc:
        return False, str(exc)
    return True, "HASH v1 ready"


def prepare_ivrit_blue_only_image(image: Image.Image) -> Image.Image | None:
    """Remove black form print while preserving the original blue pen pixels."""
    rgb = np.array(image.convert("RGB"))
    work = rgb.astype(np.int16)
    red, green, blue = np.moveaxis(work, -1, 0)
    pen = (
        (blue - red >= 24)
        & (blue - green >= 17)
        & (blue <= 245)
    )
    rows, columns = np.where(pen)
    if len(columns) < max(40, round(image.width * image.height * 0.002)):
        return None
    left = max(0, int(columns.min()) - 12)
    right = min(image.width, int(columns.max()) + 13)
    top = max(0, int(rows.min()) - 10)
    bottom = min(image.height, int(rows.max()) + 11)
    isolated = np.full_like(rgb, 255)
    isolated[pen] = rgb[pen]
    return Image.fromarray(isolated[top:bottom, left:right])


def _ivrit_core_result(vision_results: list[dict]) -> dict | None:
    readings = [
        result for result in vision_results
        if result.get("source") != IVRIT_OCR_SOURCE
        and (result.get("ocr") or {}).get("full_transcription", "").strip()
    ]
    if not readings:
        return None
    return next(
        (
            result for result in readings
            if result.get("source", "").split("#", 1)[0] == "core4_multi"
        ),
        max(
            readings,
            key=lambda result: len(
                _information_text(
                    (result.get("ocr") or {}).get("full_transcription", "")
                )
            ),
        ),
    )


def _ivrit_lines_overlap(lines: list[dict]) -> bool:
    for first_index, first in enumerate(lines):
        first_box = first.get("box") or {}
        first_top = int(first_box.get("y") or 0)
        first_height = int(first_box.get("height") or 0)
        for second in lines[first_index + 1:]:
            second_box = second.get("box") or {}
            second_top = int(second_box.get("y") or 0)
            second_height = int(second_box.get("height") or 0)
            overlap = max(
                0,
                min(first_top + first_height, second_top + second_height)
                - max(first_top, second_top),
            )
            if overlap >= 0.50 * max(1, min(first_height, second_height)):
                return True
    return False


def should_try_ivrit_blue_only(
    evidence: dict, vision_results: list[dict], region: dict,
) -> bool:
    """Retry only when one Core4 line became overlapping HASH line fragments."""
    if evidence.get("status") != "complete":
        return False
    lines = evidence.get("lines") or []
    if len(lines) < 2 or not _ivrit_lines_overlap(lines):
        return False
    core_result = _ivrit_core_result(vision_results)
    if core_result is None:
        return False
    core_text = str(
        (core_result.get("ocr") or {}).get("full_transcription") or ""
    ).strip()
    if "\n" in core_text or re.search(r"\d", core_text):
        return False
    if len(_hebrew_letter_sequence(core_text)) < 3:
        return False
    return not re.search(
        r"\b(?:date|phone|email|account|reference|identifier|signature|"
        r"mark|checkbox|amount|total)\b",
        str(region.get("label") or ""),
        re.I,
    )


def select_ivrit_preprocessing_variant(
    original: dict, blue_only: dict, vision_results: list[dict], region: dict,
) -> dict:
    """Use a clean single blue line when two HASH preprocessors still agree."""
    variants = {
        "original": {
            key: original.get(key)
            for key in ("status", "text", "confidence", "lines", "detail")
            if original.get(key) is not None
        },
        "blue_only": {
            key: blue_only.get(key)
            for key in ("status", "text", "confidence", "lines", "detail")
            if blue_only.get(key) is not None
        },
    }
    selected = dict(original)
    selected["selected_variant"] = "original"
    selected["variants"] = variants
    if blue_only.get("status") != "complete" or len(blue_only.get("lines") or []) != 1:
        return selected

    original_letters = _hebrew_letter_sequence(str(original.get("text") or ""))
    blue_letters = _hebrew_letter_sequence(str(blue_only.get("text") or ""))
    if len(original_letters) < 3 or len(blue_letters) < 3:
        return selected
    agreement = difflib.SequenceMatcher(
        None, original_letters, blue_letters
    ).ratio()
    length_overlap = min(len(original_letters), len(blue_letters)) / max(
        len(original_letters), len(blue_letters)
    )
    if (
        agreement < IVRIT_OCR_MIN_PREPROCESS_AGREEMENT
        or length_overlap < 0.80
    ):
        return selected
    selected = dict(blue_only)
    selected["selected_variant"] = "blue_only"
    selected["variants"] = variants
    selected["preprocessing_consensus"] = {
        "eligible": True,
        "character_agreement": round(agreement, 4),
        "length_overlap": round(length_overlap, 4),
        "reason": "overlapping_hash_lines_collapsed_by_blue_isolation",
    }
    return selected


def _hebrew_letter_sequence(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(
        character for character in normalized
        if "\u05d0" <= character <= "\u05ea"
    )


def ivrit_ocr_advisory_gate(
    evidence: dict, vision_results: list[dict], region: dict,
) -> dict:
    """Offer HASH to GPT only as a nearby spelling hypothesis."""
    hash_text = str(evidence.get("text") or "").strip()
    if evidence.get("status") != "complete" or not hash_text:
        return {"eligible": False, "reason": "no_complete_hash_reading"}

    core_result = _ivrit_core_result(vision_results)
    if core_result is None:
        return {"eligible": False, "reason": "no_vision_reading_to_correlate"}
    core_ocr = core_result.get("ocr") or {}
    core_text = str(core_ocr.get("full_transcription") or "").strip()
    if core_ocr.get("classification") in {
        "mark_or_checkbox", "signature_non_text", "no_handwriting",
    }:
        return {"eligible": False, "reason": "non_text_region"}

    label = str(region.get("label") or "")
    structured_label = re.search(
        r"\b(?:date|phone|email|account|reference|"
        r"identifier|signature|mark|checkbox|amount|total)\b",
        label,
        re.I,
    )
    if structured_label or re.search(r"\d", core_text) or re.search(r"\d", hash_text):
        return {"eligible": False, "reason": "structured_or_numeric_region"}
    if "\n" in core_text or "\n" in hash_text or len(evidence.get("lines") or []) != 1:
        return {"eligible": False, "reason": "multiline_reading"}

    core_letters = _hebrew_letter_sequence(core_text)
    hash_letters = _hebrew_letter_sequence(hash_text)
    if len(core_letters) < 3 or len(hash_letters) < 3:
        return {"eligible": False, "reason": "insufficient_hebrew_letters"}
    if core_letters == hash_letters:
        return {
            "eligible": False,
            "reason": "duplicate_of_vision_reading",
            "vision_source": core_result.get("source"),
            "character_agreement": 1.0,
            "length_overlap": 1.0,
        }
    preprocessing_consensus = evidence.get("preprocessing_consensus") or {}
    if preprocessing_consensus.get("eligible"):
        return {
            "eligible": True,
            "reason": "preprocessing_consensus",
            "vision_source": core_result.get("source"),
            "character_agreement": round(
                difflib.SequenceMatcher(None, core_letters, hash_letters).ratio(),
                4,
            ),
            "length_overlap": round(
                min(len(core_letters), len(hash_letters)) / max(
                    len(core_letters), len(hash_letters)
                ),
                4,
            ),
            "preprocessing_consensus": preprocessing_consensus,
        }
    length_overlap = min(len(core_letters), len(hash_letters)) / max(
        len(core_letters), len(hash_letters)
    )
    agreement = difflib.SequenceMatcher(
        None, core_letters, hash_letters
    ).ratio()
    gate = {
        "eligible": (
            agreement >= IVRIT_OCR_MIN_AGREEMENT and length_overlap >= 0.60
        ),
        "reason": "related_compact_hebrew" if (
            agreement >= IVRIT_OCR_MIN_AGREEMENT and length_overlap >= 0.60
        ) else "unrelated_to_vision_reading",
        "vision_source": core_result.get("source"),
        "character_agreement": round(agreement, 4),
        "length_overlap": round(length_overlap, 4),
    }
    return gate


def ivrit_ocr_candidate(evidence: dict) -> dict | None:
    """Convert raw HASH evidence into a weak candidate for GPT adjudication."""
    text = str(evidence.get("text") or "").strip()
    if evidence.get("status") != "complete" or not text:
        return None
    words = [
        {
            "text": line.get("text", ""),
            "alternatives": [],
            "reason": "HASH v1 CNN-CTC advisory reading; confidence is uncalibrated",
        }
        for line in evidence.get("lines") or []
        if line.get("text")
    ]
    return {
        "source": IVRIT_OCR_SOURCE,
        "ocr": {
            "full_transcription": text,
            "words": words,
            "advisory": True,
            "advisory_gate": evidence.get("advisory_gate"),
        },
    }


def should_recheck_hebrew_numeric_script_conflict(
    final: dict, evidence: dict, region: dict,
) -> bool:
    """Escalate a numeric read when independent OCR sees Hebrew in a text slot."""
    if final.get("classification") in {"signature_non_text", "no_handwriting"}:
        return False
    text = str(final.get("final_transcription") or "").replace("[?]", "")
    compact = re.sub(r"[\s,./:+-]", "", text)
    if len(compact) < 2 or not compact.isdigit():
        return False
    if evidence.get("status") != "complete":
        return False
    if len(_hebrew_letter_sequence(str(evidence.get("text") or ""))) < 4:
        return False
    label = str(region.get("label") or "")
    if re.search(
        r"\b(?:date|phone|email|account|reference|identifier|signature|"
        r"mark|checkbox|amount|total|number|code|id)\b",
        label,
        re.I,
    ):
        return False
    strong_text_slot = re.search(
        r"\b(?:name|acknowledg\w*|instruction\w*|recipient|beneficiary|"
        r"holder|owner)\b",
        label,
        re.I,
    )
    generic_multiline_conflict = (
        label.strip().lower() in {
            "handwriting", "colored handwriting", "blue handwriting",
        }
        and str(final.get("field_type") or "").lower() != "date"
        and len(evidence.get("lines") or []) >= 3
    )
    return bool(strong_text_slot or generic_multiline_conflict)


def ocr_image(image: Image.Image, prompt: str, seed: int | None = None) -> dict:
    # Raw per-image OCR keeps the original image evidence and reasoning quality.
    return _responses_call([
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": _to_data_uri(image, max_side=OCR_MAX_SIDE)},
     ], retries=0, reasoning=OCR_REASONING, seed=seed,
         max_tokens=OCR_MAX_TOKENS, endpoint_limit=1)


def ocr_image_set(
    images: list[tuple[str, Image.Image]], prompt: str, seed: int | None = None,
) -> dict:
    """Read one region once from several labeled renderings of the same strokes."""
    instruction = prompt + """

All attached images show the SAME region. Compare them as complementary evidence and
return one transcription, not one answer per image. The original crop is authoritative;
enhancements may clarify strokes but must not introduce characters. Preserve numeric
cell/group order exactly as it appears spatially in the original crop.
"""
    content: list[dict] = [{"type": "input_text", "text": instruction}]
    for name, image in images:
        content.append({"type": "input_text", "text": f"Image view: {name}"})
        content.append({
            "type": "input_image",
            "image_url": _to_data_uri(image, max_side=OCR_MAX_SIDE),
        })
    result = _responses_call(
        content, retries=0, reasoning=OCR_REASONING, seed=seed,
        max_tokens=OCR_MAX_TOKENS,
        endpoint_limit=1,
    )
    result["evidence_views"] = [name for name, _image in images]
    return result


def _segmented_unit_content(
    prompt: str, views: list[tuple[str, Image.Image]],
) -> list[dict]:
    content: list[dict] = [{
        "type": "input_text",
        "text": prompt + """

All attached views are faithful renderings of the SAME isolated handwriting unit.
They contain no neighboring word or line. Compare only their shared strokes; a form line or
image enhancement is not an additional character and does not provide semantic context.
""",
    }]
    for view_name, image in views:
        content.append({"type": "input_text", "text": f"View: {view_name}"})
        content.append({"type": "input_image", "image_url": _to_data_uri(image)})
    return content


def _segmented_read_text(result: dict) -> str:
    return " ".join(str(result.get("transcription", "")).strip().split())


def _align_segmented_readings(first: str, second: str) -> tuple[str, float]:
    matcher = difflib.SequenceMatcher(None, first, second)
    consensus: list[str] = []
    for operation, first_start, first_end, _second_start, _second_end in matcher.get_opcodes():
        if operation == "equal":
            consensus.append(first[first_start:first_end])
        elif not consensus or consensus[-1] != "[?]":
            consensus.append("[?]")
    return "".join(consensus).strip() or "[?]", round(matcher.ratio(), 4)


def prepare_isolated_multiview_plan(
    original_crop: Image.Image, region_dir: Path,
) -> dict:
    lines, line_metadata, mask = segment_handwritten_lines(original_crop)
    if not 2 <= len(lines) <= 6:
        raise ValueError(f"requires 2-6 lined handwriting rows, detected {len(lines)}")
    Image.fromarray(mask * 255).save(region_dir / "10_segmented_blue_mask.png")

    plan_lines = []
    artifact = {"strategy": ISOLATED_MULTIVIEW_STRATEGY, "lines": []}
    for line_index, (line, metadata) in enumerate(zip(lines, line_metadata), start=1):
        words, word_metadata = segment_word_groups(line)
        if 2 <= len(words) <= 4:
            mode = "word_groups"
            view_sets = build_word_view_sets(
                original_crop, metadata, words, word_metadata,
            )
            unit_metadata = word_metadata
        else:
            mode = "whole_line"
            view_sets = [build_line_view_set(original_crop, line, metadata)]
            unit_metadata = [{
                "word_rtl": 1,
                "box": [0, 0, line.width, line.height],
                "blue_pixels": int(blue_pen_mask(line).sum()),
            }]

        units = []
        artifact_line = {"line": line_index, "mode": mode, "box": metadata["box"], "units": []}
        for unit_index, (views, unit_info) in enumerate(
            zip(view_sets, unit_metadata), start=1,
        ):
            saved_views = []
            for view_name, image in views:
                file_name = (
                    f"seg_line_{line_index:02d}_unit_{unit_index:02d}_{view_name}.png"
                )
                image.save(region_dir / file_name)
                saved_views.append(file_name)
            units.append({
                "unit_rtl": unit_index,
                "views": views,
                "metadata": unit_info,
            })
            artifact_line["units"].append({
                "unit_rtl": unit_index,
                "box": unit_info["box"],
                "views": saved_views,
            })
        plan_lines.append({"line": line_index, "mode": mode, "units": units})
        artifact["lines"].append(artifact_line)

    _write_json_atomic(region_dir / "segmented_multiview_plan.json", artifact)
    return {"lines": plan_lines, "draws": sum(len(line["units"]) * 2 for line in plan_lines)}


def run_isolated_multiview_plan(
    plan: dict, on_draw=None,
) -> tuple[dict, list[dict], dict]:
    line_a, line_b, line_consensus = [], [], []
    final_words = []
    workers = min(ISOLATED_MULTIVIEW_WORKERS, max(1, int(plan.get("draws", 1))))
    details = {
        "strategy": ISOLATED_MULTIVIEW_STRATEGY,
        "concurrency": workers,
        "max_output_tokens": ISOLATED_MULTIVIEW_MAX_TOKENS,
        "lines": [],
    }
    successful_calls = 0

    draw_specs = []
    ordered_lines = sorted(
        enumerate(plan["lines"]),
        key=lambda item: (item[1]["mode"] != "word_groups", item[0]),
    )
    for line_index, line in ordered_lines:
        for unit_index, unit in enumerate(line["units"]):
            for pass_name, prompt, seed in (
                ("a", SEGMENTED_PROMPT_A, 101),
                ("b", SEGMENTED_PROMPT_B, 202),
            ):
                draw_specs.append((line_index, unit_index, pass_name, prompt, seed, unit))

    responses: dict[tuple[int, int, str], dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_specs = {
            executor.submit(
                _responses_call,
                _segmented_unit_content(prompt, unit["views"]),
                0,
                OCR_REASONING,
                ISOLATED_MULTIVIEW_MAX_TOKENS,
                seed,
            ): (line_index, unit_index, pass_name)
            for line_index, unit_index, pass_name, prompt, seed, unit in draw_specs
        }
        for future in as_completed(future_specs):
            line_index, unit_index, pass_name = future_specs[future]
            try:
                responses[(line_index, unit_index, pass_name)] = future.result()
                successful_calls += 1
                if on_draw:
                    on_draw(True)
            except Exception as exc:
                responses[(line_index, unit_index, pass_name)] = {
                    "transcription": "",
                    "error": str(exc),
                }
                if on_draw:
                    on_draw(False)

    for line_index, line in enumerate(plan["lines"]):
        units_a, units_b, units_consensus = [], [], []
        detail_line = {"line": line["line"], "mode": line["mode"], "units": []}
        for unit_index, unit in enumerate(line["units"]):
            first = responses[(line_index, unit_index, "a")]
            second = responses[(line_index, unit_index, "b")]
            reads = [
                ("a", first, _segmented_read_text(first)),
                ("b", second, _segmented_read_text(second)),
            ]

            first_text, second_text = reads[0][2], reads[1][2]
            consensus, similarity = _align_segmented_readings(first_text, second_text)
            units_a.append(first_text or "[?]")
            units_b.append(second_text or "[?]")
            units_consensus.append(consensus)
            alternatives = [text for text in (first_text, second_text) if text and text != consensus]
            final_words.append({
                "text": consensus,
                "confidence": 1.0 if first_text == second_text and first_text else similarity,
                "alternatives": list(dict.fromkeys(alternatives)),
                "reason": "agreement across two candidate-blind isolated multi-view reads",
                "source_images": [name for name, _image in unit["views"]],
            })
            detail_line["units"].append({
                "unit_rtl": unit["unit_rtl"],
                "read_a": reads[0][1],
                "read_b": reads[1][1],
                "read_a_text": first_text,
                "read_b_text": second_text,
                "consensus": consensus,
                "similarity": similarity,
            })

        line_a.append(" ".join(units_a))
        line_b.append(" ".join(units_b))
        line_consensus.append(" ".join(units_consensus))
        details["lines"].append(detail_line)

    if not successful_calls:
        raise RuntimeError("all isolated multi-view OCR calls failed")
    full_a = "\n".join(line_a)
    full_b = "\n".join(line_b)
    final_text = "\n".join(line_consensus)
    alternatives = [text for text in (full_a, full_b) if text and text != final_text]
    candidates = [
        {"source": "isolated_multiview_a", "ocr": {"full_transcription": full_a}},
        {"source": "isolated_multiview_b", "ocr": {"full_transcription": full_b}},
    ]
    final = {
        "classification": "handwritten_text",
        "final_transcription": final_text,
        "alternative_transcriptions": list(dict.fromkeys(alternatives)),
        "words": final_words,
        "notes": [
            "experimental isolated multi-view consensus; no candidate or neighboring line supplied",
        ],
        "needs_review": "[?]" in final_text or full_a != full_b,
        "decision_status": "isolated_multiview_consensus",
    }
    details.update({
        "read_a": full_a,
        "read_b": full_b,
        "final_transcription": final_text,
        "successful_calls": successful_calls,
    })
    return final, candidates, details


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
    return (
        source in _FAITHFUL_VISUAL_SOURCES
        or source.startswith("chunk_")
        or source.startswith("core4_multi")
        or source.startswith("core2_faithful_retry")
    )


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


def _numeric_sequence(text: str) -> str | None:
    """Return visible digits when text contains no letters, or None for non-numeric text."""
    cleaned = (text or "").replace("[?]", "")
    if re.search(r"[A-Za-z\u0590-\u05ff]", cleaned):
        return None
    digits = "".join(re.findall(r"\d", cleaned))
    return digits if len(digits) >= 2 else None


def preserve_numeric_candidate_order(final: dict, usable: list[dict]) -> dict:
    """Prevent RTL adjudication from permuting an otherwise agreed numeric sequence."""
    final_digits = _numeric_sequence(final.get("final_transcription", ""))
    if not final_digits:
        return final
    candidates = []
    for result in usable:
        text = (result.get("ocr") or {}).get("full_transcription", "")
        digits = _numeric_sequence(text)
        if digits and len(digits) == len(final_digits) and sorted(digits) == sorted(final_digits):
            candidates.append((result.get("source", ""), text, digits))
    if not candidates:
        return final
    counts = collections.Counter(digits for _source, _text, digits in candidates)

    def rank(candidate: tuple[str, str, str]) -> tuple[int, int]:
        source, _text, digits = candidate
        source_priority = {
            "01_original_crop.png": 3,
            "04_denoised.png": 2,
            "02_upscaled_contrast.png": 1,
        }.get(source.split("#", 1)[0], 0)
        return counts[digits], source_priority

    source, text, digits = max(candidates, key=rank)
    if digits == final_digits or counts[digits] < 2:
        return final
    ordered = dict(final)
    ordered["final_transcription"] = text.strip()
    ordered.setdefault("alternative_transcriptions", []).insert(
        0, final.get("final_transcription", "")
    )
    ordered.setdefault("notes", []).append(
        "numeric-order guard: preserved the candidate sequence supported by faithful images"
    )
    ordered["decision_status"] = "numeric_order_guard"
    ordered["numeric_order_gate"] = {
        "rejected_transcription": final.get("final_transcription", ""),
        "selected_source": source,
        "selected_support": counts[digits],
    }
    return ordered


def _information_text(text: str) -> str:
    return re.sub(r"[^\w\u0590-\u05ff]", "", (text or "").replace("[?]", ""))


def prefer_eligible_ivrit_candidate(final: dict, evidence: dict) -> dict:
    """Use HASH only after the existing vision-correlation gate admitted it."""
    gate = evidence.get("advisory_gate") or {}
    ivrit_text = str(evidence.get("text") or "").strip()
    current_text = str(final.get("final_transcription") or "").strip()
    if (
        evidence.get("status") != "complete"
        or not gate.get("eligible")
        or not ivrit_text
        or _information_text(ivrit_text) == _information_text(current_text)
    ):
        return final
    if final.get("classification") not in (None, "handwritten_text"):
        return final

    selected = dict(final)
    selected["final_transcription"] = ivrit_text
    selected["classification"] = "handwritten_text"
    alternatives = selected.setdefault("alternative_transcriptions", [])
    if current_text and current_text != ivrit_text and current_text not in alternatives:
        alternatives.insert(0, current_text)
    selected.setdefault("notes", []).append(
        "eligible HASH tie-break: selected independently gated compact Hebrew evidence"
    )
    selected["needs_review"] = bool(final.get("needs_review"))
    selected["decision_status"] = "ivrit_eligible_tiebreak"
    selected["ivrit_tiebreak"] = {
        "rejected_transcription": current_text,
        "reason": gate.get("reason"),
        "vision_source": gate.get("vision_source"),
        "character_agreement": gate.get("character_agreement"),
        "selected_variant": evidence.get("selected_variant", "original"),
    }
    return selected


def separate_numeric_field_marks(final: dict) -> dict:
    """Keep standalone X marks out of otherwise numeric transcription."""
    if str(final.get("field_type") or "").lower() != "numeric":
        return final
    text = str(final.get("final_transcription") or "")
    if not re.search(r"(?<!\w)[xX×](?!\w)", text):
        return final
    numeric_text = " ".join(
        re.sub(r"(?<!\w)[xX×](?!\w)", " ", text).split()
    )
    if not _numeric_sequence(numeric_text):
        return final

    separated = dict(final)
    separated["final_transcription"] = numeric_text
    marks = separated.setdefault("marks", [])
    if "x" not in marks:
        marks.append("x")
    alternatives = separated.setdefault("alternative_transcriptions", [])
    if text not in alternatives:
        alternatives.insert(0, text)
    separated.setdefault("notes", []).append(
        "structured mark separation: moved standalone X out of numeric text"
    )
    separated["decision_status"] = "structured_mark_separation"
    return separated


def normalize_numeric_s_homoglyph(final: dict) -> dict:
    """Interpret S as 5 only when the entire OCR result is numeric-like."""
    text = str(final.get("final_transcription") or "")
    cleaned = text.replace("[?]", "")
    letters = re.findall(r"[A-Za-z]", cleaned)
    if (
        not letters
        or any(letter not in {"S", "s"} for letter in letters)
        or len(re.findall(r"\d", cleaned)) < 4
        or re.search(r"[\u0590-\u05ff]", cleaned)
    ):
        return final
    remainder = re.sub(r"[Ss\d\s,./:_-]", "", cleaned)
    if remainder:
        return final

    normalized_text = re.sub(r"[Ss]", "5", text)
    normalized = dict(final)
    normalized["final_transcription"] = normalized_text
    alternatives = normalized.setdefault("alternative_transcriptions", [])
    if text not in alternatives:
        alternatives.insert(0, text)
    normalized.setdefault("notes", []).append(
        "numeric homoglyph normalization: interpreted S-shaped glyphs as 5"
    )
    normalized["decision_status"] = "numeric_homoglyph_normalization"
    return normalized


def _structured_result_confidence(final: dict) -> float:
    if final.get("confidence") is not None:
        return float(final.get("confidence") or 0.0)
    confidences = []
    for word in final.get("words") or []:
        if isinstance(word, dict) and word.get("confidence") is not None:
            confidences.append(float(word.get("confidence") or 0.0))
    return sum(confidences) / len(confidences) if confidences else 0.0


def reconcile_repeated_structured_fields(finals: list[dict], regions: list[dict]) -> list[dict]:
    """Use a stronger duplicate identifier to repair one-digit OCR drift."""
    position_words = re.compile(
        r"\b(?:top|bottom|upper|lower|middle|left|right|form|handwritten|entry|"
        r"next|below|above|line|table|on|in)\b",
        re.I,
    )
    identifier_words = re.compile(r"\b(?:account|customer|reference|identifier|number|id)\b", re.I)
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for index, region in enumerate(regions):
        label = str(region.get("label") or "")
        normalized_label = " ".join(position_words.sub(" ", label).lower().split())
        if identifier_words.search(normalized_label):
            groups[normalized_label].append(index)

    reconciled = [dict(final) for final in finals]
    for indexes in groups.values():
        for left_offset, left_index in enumerate(indexes):
            for right_index in indexes[left_offset + 1:]:
                left = reconciled[left_index]
                right = reconciled[right_index]
                left_digits = "".join(re.findall(r"\d", left.get("final_transcription", "")))
                right_digits = "".join(re.findall(r"\d", right.get("final_transcription", "")))
                if len(left_digits) < 4 or len(left_digits) != len(right_digits):
                    continue
                if sum(a != b for a, b in zip(left_digits, right_digits)) != 1:
                    continue
                left_confidence = _structured_result_confidence(left)
                right_confidence = _structured_result_confidence(right)
                winner_index, loser_index = (
                    (left_index, right_index)
                    if left_confidence >= right_confidence
                    else (right_index, left_index)
                )
                winner_confidence = max(left_confidence, right_confidence)
                loser_confidence = min(left_confidence, right_confidence)
                if winner_confidence < 0.90 or winner_confidence - loser_confidence < 0.10:
                    continue
                winner = reconciled[winner_index]
                loser = dict(reconciled[loser_index])
                rejected = loser.get("final_transcription", "")
                selected = winner.get("final_transcription", "")
                loser["final_transcription"] = selected
                alternatives = loser.setdefault("alternative_transcriptions", [])
                if rejected and rejected != selected and rejected not in alternatives:
                    alternatives.insert(0, rejected)
                loser.setdefault("notes", []).append(
                    "cross-region consensus: matched the stronger repeated identifier"
                )
                loser["decision_status"] = "cross_region_structured_consensus"
                loser["cross_region_consensus"] = {
                    "matched_region": winner_index + 1,
                    "rejected_transcription": rejected,
                    "winner_confidence": winner_confidence,
                    "loser_confidence": loser_confidence,
                }
                reconciled[loser_index] = loser
    return reconciled


def _date_values(value: object) -> list[str]:
    pattern = re.compile(
        r"(?<!\d)(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})(?!\d)"
    )
    found: list[str] = []

    def collect(item: object) -> None:
        if isinstance(item, str):
            found.extend("/".join(match) for match in pattern.findall(item))
        elif isinstance(item, dict):
            for nested in item.values():
                collect(nested)
        elif isinstance(item, list):
            for nested in item:
                collect(nested)

    collect(value)
    return found


def reconcile_repeated_dates(finals: list[dict], regions: list[dict]) -> list[dict]:
    """Repair one-digit date drift when context and another region agree."""
    reconciled = [dict(final) for final in finals]
    final_dates = [
        _date_values(final.get("final_transcription", ""))
        for final in reconciled
    ]
    for index, final in enumerate(reconciled):
        current_dates = final_dates[index]
        current_text = str(final.get("final_transcription") or "").strip()
        if (
            len(current_dates) != 1
            or _information_text(current_text)
            != _information_text(current_dates[0])
        ):
            continue
        contextual_dates = _date_values(final.get("contextual_evidence") or {})
        other_dates = [
            date
            for other_index, dates in enumerate(final_dates)
            if other_index != index
            for date in dates
        ]
        current_digits = current_dates[0].replace("/", "")
        candidates = []
        for candidate in contextual_dates:
            candidate_digits = candidate.replace("/", "")
            if (
                candidate == current_dates[0]
                or candidate not in other_dates
                or len(candidate_digits) != len(current_digits)
                or sum(
                    left != right
                    for left, right in zip(candidate_digits, current_digits)
                ) != 1
            ):
                continue
            candidates.append(candidate)
        if not candidates:
            continue
        selected_date = max(
            set(candidates),
            key=lambda candidate: (
                other_dates.count(candidate),
                contextual_dates.count(candidate),
            ),
        )
        matched_region = next(
            other_index + 1
            for other_index, dates in enumerate(final_dates)
            if other_index != index and selected_date in dates
        )
        selected = dict(final)
        selected["final_transcription"] = selected_date
        alternatives = selected.setdefault("alternative_transcriptions", [])
        if current_text not in alternatives:
            alternatives.insert(0, current_text)
        selected.setdefault("notes", []).append(
            "cross-region date consensus: contextual alternative matched another field"
        )
        selected["decision_status"] = "cross_region_date_consensus"
        selected["cross_region_date_consensus"] = {
            "matched_region": matched_region,
            "rejected_transcription": current_text,
            "contextual_support": contextual_dates.count(selected_date),
            "page_support": other_dates.count(selected_date),
        }
        reconciled[index] = selected
    return reconciled


def _text_edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1]
                + (left_character != right_character),
            ))
        previous = current
    return previous[-1]


def reconcile_with_cu_numeric_tokens(
    finals: list[dict], regions: list[dict], cu_markdown: str,
) -> list[dict]:
    """Use CU only for a unique one-edit correction of a typed numeric field."""
    cu_tokens = []
    for token in re.findall(
        r"(?<!\w)\d[\d,./-]*\d(?!\w)|(?<!\w)\d(?!\w)",
        cu_markdown or "",
    ):
        digits = "".join(re.findall(r"\d", token))
        if len(digits) >= 4:
            cu_tokens.append((digits, token))
    all_cu_digits = [digits for digits, _token in cu_tokens]
    reconciled = [dict(final) for final in finals]
    for index, final in enumerate(reconciled):
        if str(final.get("field_type") or "").lower() != "numeric":
            continue
        if "date" in str(regions[index].get("label") or "").lower():
            continue
        current_text = str(final.get("final_transcription") or "").strip()
        current_digits = "".join(re.findall(r"\d", current_text))
        if (
            len(current_digits) < 4
            or _information_text(current_text) != current_digits
            or current_digits in all_cu_digits
        ):
            continue
        nearby = {
            digits: token
            for digits, token in cu_tokens
            if _text_edit_distance(current_digits, digits) == 1
        }
        if len(nearby) != 1:
            continue
        selected_digits, selected_token = next(iter(nearby.items()))
        selected = dict(final)
        selected["final_transcription"] = selected_token
        alternatives = selected.setdefault("alternative_transcriptions", [])
        if current_text not in alternatives:
            alternatives.insert(0, current_text)
        selected.setdefault("notes", []).append(
            "CU numeric consensus: unique one-edit token matched typed numeric field"
        )
        selected["decision_status"] = "cu_numeric_consensus"
        selected["cu_numeric_consensus"] = {
            "rejected_transcription": current_text,
            "selected_digits": selected_digits,
            "edit_distance": 1,
        }
        reconciled[index] = selected
    return reconciled


def reconcile_uncertain_repeated_numbers(
    finals: list[dict], regions: list[dict], cu_markdown: str,
) -> list[dict]:
    """Repair uncertain numeric text when another region and CU exactly agree."""
    cu_lines = [
        "".join(re.findall(r"\d", line))
        for line in (cu_markdown or "").splitlines()
        if len(re.findall(r"\d", line)) >= 4
    ]
    reconciled = [dict(final) for final in finals]
    sequences = []
    for final in reconciled:
        text = str(final.get("final_transcription") or "")
        info = _information_text(text)
        digits = "".join(re.findall(r"\d", text))
        sequences.append(
            (digits, text)
            if digits and not re.search(r"[A-Za-z\u0590-\u05ff]", info)
            else ("", "")
        )
    for index, final in enumerate(reconciled):
        if str(final.get("field_type") or "").lower() != "numeric":
            continue
        current_text = str(final.get("final_transcription") or "")
        current_digits = "".join(re.findall(r"\d", current_text))
        info = _information_text(current_text)
        if (
            "[?]" not in current_text
            or len(current_digits) < 4
            or "/" in current_text
            or re.search(r"[A-Za-z\u0590-\u05ff]", info)
            or any(current_digits in line for line in cu_lines)
        ):
            continue
        candidates = {
            digits: text
            for other_index, (digits, text) in enumerate(sequences)
            if other_index != index
            and len(digits) >= 6
            and _text_edit_distance(current_digits, digits) <= 3
            and any(digits in line for line in cu_lines)
        }
        if len(candidates) != 1:
            continue
        selected_digits, selected_text = next(iter(candidates.items()))
        selected = dict(final)
        selected["final_transcription"] = selected_text
        alternatives = selected.setdefault("alternative_transcriptions", [])
        if current_text not in alternatives:
            alternatives.insert(0, current_text)
        selected.setdefault("notes", []).append(
            "repeated numeric consensus: another region and CU agree"
        )
        selected["decision_status"] = "repeated_numeric_cu_consensus"
        selected["repeated_numeric_cu_consensus"] = {
            "rejected_transcription": current_text,
            "selected_digits": selected_digits,
            "edit_distance": _text_edit_distance(
                current_digits, selected_digits,
            ),
        }
        reconciled[index] = selected
    return reconciled


def protect_faithful_candidate(final: dict, usable: list[dict]) -> dict:
    """Do not replace faithful OCR with a reordered or lower-information adjudication."""
    faithful = [
        result for result in usable
        if _is_faithful_source(result.get("source", ""))
        and _information_text((result.get("ocr") or {}).get("full_transcription", ""))
    ]
    if not faithful:
        return final
    candidate = max(
        faithful,
        key=lambda result: len(_information_text(
            (result.get("ocr") or {}).get("full_transcription", "")
        )),
    )
    candidate_ocr = candidate.get("ocr") or {}
    candidate_text = candidate_ocr.get("full_transcription", "")
    candidate_info = _information_text(candidate_text)
    final_text = final.get("final_transcription", "")
    final_info = _information_text(final_text)
    if not final_info:
        return final
    same_characters = (
        len(candidate_info) >= 2
        and candidate_info != final_info
        and collections.Counter(candidate_info) == collections.Counter(final_info)
    )
    lower_information = (
        len(candidate_info) >= 4
        and len(final_info) < 0.72 * len(candidate_info)
    )
    if not same_characters and not lower_information:
        return final
    protected = dict(final)
    protected["final_transcription"] = candidate_text
    protected["words"] = candidate_ocr.get("words", [])
    if not protected.get("marks"):
        protected["classification"] = "handwritten_text"
    protected.setdefault("alternative_transcriptions", []).insert(0, final_text)
    protected.setdefault("notes", []).append(
        "candidate-protection guard: preserved faithful spatial order/information"
    )
    protected["decision_status"] = "faithful_candidate_protected"
    protected["candidate_protection"] = {
        "rejected_transcription": final_text,
        "selected_source": candidate.get("source"),
        "reason": "reordered" if same_characters else "lower_information",
    }
    return protected


def should_contextual_recheck(final: dict, usable: list[dict]) -> bool:
    """Escalate uncertain or language-bearing fields; leave clean numeric fields fast."""
    if final.get("classification") in {"signature_non_text", "no_handwriting"}:
        return False
    texts = [
        (result.get("ocr") or {}).get("full_transcription", "")
        for result in usable
        if (result.get("ocr") or {}).get("full_transcription", "").strip()
    ]
    candidate = max(texts, key=lambda text: len(_information_text(text)), default="")
    final_text = final.get("final_transcription", "")
    combined = f"{candidate} {final_text}"
    if not _information_text(final_text):
        return True
    if "[?]" in combined:
        return True
    compact_final = re.sub(r"\s+", "", final_text)
    if re.search(r"\d[/.-]\d", compact_final) and not re.fullmatch(
        r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}", compact_final
    ):
        return True
    if final.get("alternative_transcriptions"):
        return True
    if re.search(r"[\u0590-\u05ff]", combined):
        return True
    confidences = [
        float(word.get("confidence", 0.0) or 0.0)
        for result in usable
        for word in ((result.get("ocr") or {}).get("words") or [])
    ]
    if confidences and min(confidences) < 0.65:
        return True
    if candidate and final_text:
        similarity = difflib.SequenceMatcher(
            None, _norm_read(candidate), _norm_read(final_text)
        ).ratio()
        if similarity < 0.85:
            return True
    return False


def build_contextual_region_images(
    page_img: Image.Image, box: list[int],
) -> tuple[Image.Image, Image.Image]:
    """Return a larger local crop and full page, both with the target outlined."""
    page_width, page_height = page_img.size
    x, y, width, height = box
    horizontal_pad = max(round(width * 1.1), round(page_width * 0.08))
    vertical_pad = max(round(height * 1.8), round(page_height * 0.025))
    left = max(0, x - horizontal_pad)
    top = max(0, y - vertical_pad)
    right = min(page_width, x + width + horizontal_pad)
    bottom = min(page_height, y + height + vertical_pad)
    context = page_img.crop((left, top, right, bottom)).convert("RGB")
    context_draw = ImageDraw.Draw(context)
    context_draw.rectangle(
        (x - left, y - top, x + width - left, y + height - top),
        outline=(220, 20, 20), width=max(4, round(min(context.size) * 0.008)),
    )
    highlighted_page = page_img.convert("RGB").copy()
    page_draw = ImageDraw.Draw(highlighted_page)
    page_draw.rectangle(
        (x, y, x + width, y + height),
        outline=(220, 20, 20), width=max(8, round(min(page_img.size) * 0.003)),
    )
    return context, highlighted_page


def contextual_region_recheck(
    page_img: Image.Image, region: dict, candidate: dict, printed_context: str = "",
    reasoning: str | None = None,
) -> tuple[dict, tuple[Image.Image, Image.Image]]:
    """Re-read a difficult field with local form labels and page-level position."""
    context_image, highlighted_page = build_contextual_region_images(page_img, region["box"])
    candidate_text = candidate.get("final_transcription", "")
    candidate_has_hebrew = bool(re.search(r"[\u0590-\u05ff]", candidate_text))
    candidate_hint = "(withheld for an independent Hebrew reread)" if candidate_has_hebrew \
        else candidate_text or "(none)"
    preservation_rule = "" if candidate_has_hebrew else """
- Preserve all visible words/digits and their grouping. Do not shorten or reorder the candidate
  unless the highlighted visual evidence clearly proves the correction."""
    context_block = build_context_block(printed_context)
    numeric_cell_rules = ""
    numeric_cell_schema = ""
    if SPATIAL_NUMERIC_CELLS:
        numeric_cell_rules = """
- For numeric fields split across visible cells, return each cell separately in numeric_cells.
  Set x to the cell center measured from the LEFT edge of the red rectangle (0.0 to 1.0)."""
        numeric_cell_schema = """
  "numeric_cells": [
    {"value": "visible digits", "x": 0.0, "confidence": 0.0}
  ],"""
    prompt = f"""Re-read the handwriting inside the RED rectangle on this Hebrew form.

You receive a larger local crop and the full page so nearby printed labels, table cells,
and field position are visible. Transcribe ONLY handwriting inside the red rectangle.

Current OCR candidate: {candidate_hint}

Rules:
- The visible strokes are authoritative; the candidate is only a hint.
- Use nearby printed labels only to identify the field type, never copy them as handwriting.
- Read Hebrew words right-to-left, but digits and numeric cells in physical left-to-right order.
{numeric_cell_rules}
{preservation_rule}
- Return uncertainty as [?]. Classify signatures and marks explicitly.
{context_block}
Return JSON only:
{{
  "classification": "handwritten_text | mark_or_checkbox | signature_non_text | no_handwriting",
  "field_type": "date | numeric | name | address | free_text | mark | signature | unknown",
  "final_transcription": "...",
{numeric_cell_schema}
  "confidence": 0.0,
  "alternative_transcriptions": [],
  "words": [],
  "notes": []
}}
"""
    result = _responses_call([
        {"type": "input_text", "text": prompt},
        {"type": "input_text", "text": "Larger context crop with target rectangle:"},
        {"type": "input_image", "image_url": _to_data_uri(context_image)},
        {"type": "input_text", "text": "Full page with the same target rectangle:"},
        {"type": "input_image", "image_url": _to_data_uri(highlighted_page)},
    ], retries=0, max_tokens=10000, reasoning=reasoning)
    return result, (context_image, highlighted_page)


def script_conflict_region_recheck(
    page_img: Image.Image, region: dict, printed_context: str = "",
) -> tuple[dict, dict[str, Image.Image]]:
    """Resolve Hebrew-versus-numeric disagreement without exposing OCR candidates."""
    context_image, highlighted_page = build_contextual_region_images(
        page_img, region["box"]
    )
    original_crop = crop_region(page_img, *region["box"])
    blue_only = prepare_ivrit_blue_only_image(original_crop)
    context_block = build_context_block(printed_context)
    prompt = f"""Candidate-blind script audit for handwriting inside the RED rectangle.

Independent OCR engines disagree whether these pen strokes are Hebrew letters or digits.
You receive NO prior OCR strings. Determine the script directly from the visible strokes.

Rules:
- First classify script as hebrew, numeric, mixed, mark, signature, or none.
- Use surrounding printed grammar only to identify the field type, never to invent its value.
- If script is hebrew, transcribe the handwriting right-to-left.
- If script is numeric, preserve physical left-to-right digit order.
- Ignore black printed text and the horizontal form line crossing the blue handwriting.
- Use [?] for unsupported characters. Do not substitute a plausible name for unclear strokes.
{context_block}
Return JSON only:
{{
  "script": "hebrew | numeric | mixed | mark | signature | none",
  "classification": "handwritten_text | mark_or_checkbox | signature_non_text | no_handwriting",
  "field_type": "date | numeric | name | address | free_text | mark | signature | unknown",
  "final_transcription": "...",
  "confidence": 0.0,
  "alternative_transcriptions": [],
  "words": [],
  "notes": []
}}
"""
    content: list[dict] = [
        {"type": "input_text", "text": prompt},
        {"type": "input_text", "text": "Authoritative tight original crop:"},
        {"type": "input_image", "image_url": _to_data_uri(original_crop)},
    ]
    images = {
        "original": original_crop,
        "context": context_image,
        "page": highlighted_page,
    }
    if blue_only is not None:
        images["blue_only"] = blue_only
        content.extend([
            {"type": "input_text", "text": "Blue-pen-only support view:"},
            {"type": "input_image", "image_url": _to_data_uri(blue_only)},
        ])
    content.extend([
        {"type": "input_text", "text": "Larger context with target rectangle:"},
        {"type": "input_image", "image_url": _to_data_uri(context_image)},
        {"type": "input_text", "text": "Full page with the same target rectangle:"},
        {"type": "input_image", "image_url": _to_data_uri(highlighted_page)},
    ])
    result = _responses_call(
        content, retries=0, max_tokens=8000, reasoning="high"
    )
    return result, images


def accept_hebrew_numeric_script_recheck(current: dict, recheck: dict) -> dict:
    """Replace numeric text only with a confident candidate-blind Hebrew field read."""
    text = str(recheck.get("final_transcription") or "").strip()
    if (
        str(recheck.get("script") or "").lower() != "hebrew"
        or recheck.get("classification") != "handwritten_text"
        or str(recheck.get("field_type") or "").lower()
        not in {"name", "address", "free_text"}
        or len(_hebrew_letter_sequence(text)) < 3
        or re.search(r"\d", text)
        or float(recheck.get("confidence", 0.0) or 0.0) < 0.55
    ):
        return current
    accepted = dict(recheck)
    alternatives = list(accepted.get("alternative_transcriptions") or [])
    previous = str(current.get("final_transcription") or "")
    if previous and previous != text and previous not in alternatives:
        alternatives.append(previous)
    accepted["alternative_transcriptions"] = alternatives
    accepted.setdefault("notes", []).append(
        "candidate-blind script audit replaced a numeric read with Hebrew text"
    )
    accepted["needs_review"] = True
    accepted["decision_status"] = "hebrew_numeric_script_conflict"
    accepted["pre_script_conflict_transcription"] = previous
    accepted["script_conflict_evidence"] = recheck
    return accepted


def preserve_uncertain_hebrew_script_conflict(
    current: dict, recheck: dict,
) -> dict:
    """Do not retain numeric text after a credible audit identifies Hebrew script."""
    text = str(recheck.get("final_transcription") or "").strip()
    if (
        str(recheck.get("script") or "").lower() != "hebrew"
        or recheck.get("classification") != "handwritten_text"
        or len(_hebrew_letter_sequence(text)) < 3
        or re.search(r"\d", text)
        or float(recheck.get("confidence", 0.0) or 0.0) < 0.35
    ):
        return current
    unresolved = dict(recheck)
    previous = str(current.get("final_transcription") or "")
    alternatives = list(unresolved.get("alternative_transcriptions") or [])
    if previous and previous != text and previous not in alternatives:
        alternatives.append(previous)
    unresolved["alternative_transcriptions"] = alternatives
    unresolved.setdefault("notes", []).append(
        "candidate-blind script audit identified Hebrew but exact letters require review"
    )
    unresolved["needs_review"] = True
    unresolved["decision_status"] = "script_conflict_unresolved"
    unresolved["pre_script_conflict_transcription"] = previous
    unresolved["script_conflict_evidence"] = recheck
    return unresolved


def should_high_uncertainty_rescue(
    final: dict, signature_flourish: bool,
) -> bool:
    """Retry only an explicit [?]-only text field after signature rejection."""
    text = str(final.get("final_transcription") or "").strip()
    return (
        not signature_flourish
        and final.get("classification") == "handwritten_text"
        and "[?]" in text
        and not _information_text(text)
    )


def accept_high_uncertainty_rescue(current: dict, recheck: dict) -> dict:
    """Accept a high-reasoning rescue only as confident Hebrew handwriting."""
    text = _information_text(recheck.get("final_transcription", ""))
    if (
        recheck.get("classification") != "handwritten_text"
        or not re.search(r"[\u0590-\u05ff]", text)
        or float(recheck.get("confidence", 0.0) or 0.0) < 0.70
    ):
        return current
    return accept_contextual_recheck(current, recheck)


def should_emit_region(final: dict, per_image: list[dict]) -> bool:
    """Hide a detector false positive only when all vision evidence is empty."""
    return not (
        final.get("classification") == "no_handwriting"
        and not _information_text(final.get("final_transcription", ""))
        and not any(
            _information_text(result.get("full_transcription", ""))
            for result in per_image
        )
    )


def assemble_spatial_numeric_cells(recheck: dict) -> dict:
    """Assemble numeric table cells in physical left-to-right order."""
    if recheck.get("field_type") != "numeric":
        return recheck
    raw_cells = recheck.get("numeric_cells")
    if not isinstance(raw_cells, list) or len(raw_cells) < 2:
        return recheck
    cells = []
    for cell in raw_cells:
        if not isinstance(cell, dict):
            return recheck
        value = str(cell.get("value", "")).strip()
        try:
            x_position = float(cell.get("x"))
        except (TypeError, ValueError):
            return recheck
        cleaned_value = value.replace("[?]", "")
        if (
            not 0.0 <= x_position <= 1.0
            or re.search(r"[A-Za-z\u0590-\u05ff]", cleaned_value)
            or not re.search(r"\d", cleaned_value)
        ):
            return recheck
        cells.append((x_position, value, float(cell.get("confidence", 0.0) or 0.0)))
    cells.sort(key=lambda cell: cell[0])
    if any(right[0] - left[0] < 0.01 for left, right in zip(cells, cells[1:])):
        return recheck
    groups: list[list[tuple[float, str, float]]] = []
    for cell in cells:
        if (
            groups
            and len(groups[-1][-1][1].replace("[?]", "")) == 1
            and len(cell[1].replace("[?]", "")) == 1
            and cell[0] - groups[-1][-1][0] <= 0.09
        ):
            groups[-1].append(cell)
        else:
            groups.append([cell])
    assembled_text = " ".join("".join(cell[1] for cell in group) for group in groups)
    current_info = _information_text(recheck.get("final_transcription", ""))
    assembled_info = _information_text(assembled_text)
    if current_info and len(assembled_info) < 0.72 * len(current_info):
        return recheck
    assembled = dict(recheck)
    assembled["final_transcription"] = assembled_text
    assembled["numeric_cells"] = [
        {"value": value, "x": x_position, "confidence": confidence}
        for x_position, value, confidence in cells
    ]
    assembled.setdefault("notes", []).append(
        "numeric cells assembled deterministically by physical x-coordinate"
    )
    assembled["spatial_numeric_assembly"] = True
    return assembled


def accept_contextual_recheck(current: dict, recheck: dict) -> dict:
    """Accept a contextual read only when it is non-empty and not less informative."""
    if SPATIAL_NUMERIC_CELLS:
        recheck = assemble_spatial_numeric_cells(recheck)
    if recheck.get("classification") in {"signature_non_text", "no_handwriting"}:
        return current
    if recheck.get("classification") == "mark_or_checkbox" and not recheck.get("marks"):
        return current
    current_info = _information_text(current.get("final_transcription", ""))
    recheck_info = _information_text(recheck.get("final_transcription", ""))
    if not recheck_info:
        return current
    if current_info and len(recheck_info) < 0.72 * len(current_info):
        return current
    current_has_hebrew = bool(re.search(r"[\u0590-\u05ff]", current_info))
    recheck_has_hebrew = bool(re.search(r"[\u0590-\u05ff]", recheck_info))
    if current_has_hebrew and not recheck_has_hebrew:
        return current
    structured_pattern = re.compile(r"(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|[O0]\+)", re.I)
    current_structured = {
        token.upper().replace(".", "/").replace("-", "/")
        for token in structured_pattern.findall(current.get("final_transcription", ""))
    }
    recheck_structured = {
        token.upper().replace(".", "/").replace("-", "/")
        for token in structured_pattern.findall(recheck.get("final_transcription", ""))
    }
    if not current_structured.issubset(recheck_structured):
        return current
    confidence = float(recheck.get("confidence", 0.0) or 0.0)
    if confidence < CONTEXTUAL_RECHECK_MIN_CONFIDENCE and recheck_info != current_info:
        return current
    accepted = dict(recheck)
    accepted.setdefault("alternative_transcriptions", []).append(
        current.get("final_transcription", "")
    )
    accepted.setdefault("notes", []).append("adaptive highlighted-context recheck")
    accepted["decision_status"] = "contextual_recheck"
    accepted["contextual_recheck"] = True
    return accepted


def protect_contextual_recheck_with_cross_engine_agreement(
    current: dict,
    recheck: dict,
    accepted: dict,
    vision_results: list[dict],
    ivrit_evidence: dict,
) -> dict:
    """Do not let one context guess override exact visual/Core4/HASH agreement."""
    current_text = str(current.get("final_transcription") or "")
    recheck_text = str(recheck.get("final_transcription") or "")
    accepted_text = str(accepted.get("final_transcription") or "")
    current_info = _information_text(current_text)
    if (
        len(current_info) < 3
        or _information_text(recheck_text) == current_info
        or _information_text(accepted_text) == current_info
        or ivrit_evidence.get("status") != "complete"
        or _information_text(str(ivrit_evidence.get("text") or "")) != current_info
    ):
        return accepted
    faithful_supporters = [
        result.get("source", "")
        for result in vision_results
        if _is_faithful_source(result.get("source", ""))
        and _information_text(
            str((result.get("ocr") or {}).get("full_transcription") or "")
        ) == current_info
    ]
    advisory_gate = ivrit_evidence.get("advisory_gate") or {}
    eligible_hash_support = bool(advisory_gate.get("eligible"))
    if not faithful_supporters and not eligible_hash_support:
        return accepted

    protected = dict(current)
    if not protected.get("field_type") and accepted.get("field_type"):
        protected["field_type"] = accepted.get("field_type")
    alternatives = list(protected.get("alternative_transcriptions") or [])
    for text in [recheck_text, *(recheck.get("alternative_transcriptions") or [])]:
        if text and text != current_text and text not in alternatives:
            alternatives.append(text)
    protected["alternative_transcriptions"] = alternatives
    protected.setdefault("notes", []).append(
        "contextual override rejected: visual and independently gated HASH agree"
    )
    protected["needs_review"] = True
    protected["decision_status"] = "cross_engine_agreement_protected"
    protected["cross_engine_agreement"] = {
        "selected_transcription": current_text,
        "rejected_contextual_transcription": recheck_text,
        "vision_supporting_sources": faithful_supporters,
        "vision_correlated_source": advisory_gate.get("vision_source"),
        "ivrit_advisory_gate": advisory_gate,
        "ivrit_source": ivrit_evidence.get("source", IVRIT_OCR_SOURCE),
        "ivrit_selected_variant": ivrit_evidence.get("selected_variant", "original"),
    }
    return protected


def is_signature_flourish(image: Image.Image) -> bool:
    """High-specificity blue-ink signature gate based on one dominant crossing stroke."""
    rgb = np.asarray(image.convert("RGB"))
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    mask = (
        (blue - red >= 24)
        & (blue - green >= 17)
        & (blue <= 245)
    ).astype(np.uint8)
    ink_area = int(mask.sum())
    if ink_area < 100:
        return False
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return False
    largest = max(stats[1:], key=lambda stat: int(stat[cv2.CC_STAT_AREA]))
    dominance = int(largest[cv2.CC_STAT_AREA]) / ink_area
    width_fraction = int(largest[cv2.CC_STAT_WIDTH]) / max(1, image.width)
    return dominance >= 0.78 and width_fraction >= 0.78


def is_black_signature_flourish(image: Image.Image) -> bool:
    """High-specificity black-ink signature gate for monochrome forms."""
    gray = np.array(image.convert("L"))
    _, ink = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(20, gray.shape[1] // 8), 1),
        ),
    )
    vertical = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(20, gray.shape[0] // 3)),
        ),
    )
    cleaned = cv2.subtract(cv2.subtract(ink, horizontal), vertical)
    ink_area = int((cleaned > 0).sum())
    if ink_area < 100:
        return False
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cleaned, 8,
    )
    components = [
        stats[index]
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= 40
        and stats[index, cv2.CC_STAT_HEIGHT] >= 12
    ]
    if not components:
        return False
    largest = max(components, key=lambda stat: int(stat[cv2.CC_STAT_AREA]))
    dominance = int(largest[cv2.CC_STAT_AREA]) / ink_area
    width_fraction = int(largest[cv2.CC_STAT_WIDTH]) / max(1, image.width)
    height_fraction = int(largest[cv2.CC_STAT_HEIGHT]) / max(1, image.height)
    dominant_flourish = dominance >= 0.78 and width_fraction >= 0.65
    complex_flourish = (
        len(components) >= 15
        and width_fraction >= 0.40
        and height_fraction <= 0.60
    )
    return dominant_flourish or complex_flourish


def finalize_region_semantics(
    final: dict, usable: list[dict], signature_flourish: bool = False,
) -> dict:
    """Apply explicit non-text semantics and deterministic numeric ordering."""
    resolved = protect_faithful_candidate(
        preserve_numeric_candidate_order(final, usable), usable
    )
    classification = resolved.get("classification")
    if classification not in {
        "handwritten_text", "mark_or_checkbox", "signature_non_text", "no_handwriting"
    }:
        candidate_classes = [
            (result.get("ocr") or {}).get("classification")
            for result in usable
            if (result.get("ocr") or {}).get("classification") in {
                "handwritten_text", "mark_or_checkbox", "signature_non_text", "no_handwriting"
            }
        ]
        notes = " ".join(str(note) for note in resolved.get("notes", [])).lower()
        if candidate_classes and collections.Counter(candidate_classes).most_common(1)[0][1] >= 2:
            classification = collections.Counter(candidate_classes).most_common(1)[0][0]
        elif "signature" in notes or "חתימה" in notes:
            classification = "signature_non_text"
        elif resolved.get("marks"):
            classification = "mark_or_checkbox"
        else:
            classification = "handwritten_text"
    notes = " ".join(str(note) for note in resolved.get("notes", [])).lower()
    if (
        signature_flourish
        and classification == "handwritten_text"
        and not _norm_read(resolved.get("final_transcription", ""))
        and re.search(r"signature|scribble|חתימה|שרבוט", notes, re.I)
    ):
        classification = "signature_non_text"
        resolved["classification"] = classification
        resolved.setdefault("notes", []).append(
            "signature geometry: empty text and explicit signature evidence agree"
        )
    resolved["classification"] = classification
    if (
        not signature_flourish
        and not _norm_read(resolved.get("final_transcription", ""))
        and classification not in {"signature_non_text", "no_handwriting"}
        and not resolved.get("marks")
    ):
        faithful_nonempty = [
            result for result in usable
            if _is_faithful_source(result.get("source", ""))
            and _norm_read((result.get("ocr") or {}).get("full_transcription", ""))
        ]
        if faithful_nonempty:
            selected = max(
                faithful_nonempty,
                key=lambda result: len(_norm_read(
                    (result.get("ocr") or {}).get("full_transcription", "")
                )),
            )
            selected_ocr = selected.get("ocr") or {}
            resolved["final_transcription"] = selected_ocr.get("full_transcription", "")
            resolved["words"] = selected_ocr.get("words", [])
            resolved["classification"] = "handwritten_text"
            resolved["decision_status"] = "nonempty_faithful_fallback"
            resolved.setdefault("notes", []).append(
                "empty-output guard: preserved non-empty faithful OCR on a non-signature crop"
            )
    if classification == "signature_non_text" and not signature_flourish:
        classification = "handwritten_text"
        resolved["classification"] = classification
        if not _norm_read(resolved.get("final_transcription", "")):
            faithful_nonempty = [
                result for result in usable
                if _is_faithful_source(result.get("source", ""))
                and _norm_read((result.get("ocr") or {}).get("full_transcription", ""))
            ]
            if faithful_nonempty:
                selected = max(
                    faithful_nonempty,
                    key=lambda result: len(_norm_read(
                        (result.get("ocr") or {}).get("full_transcription", "")
                    )),
                )
                selected_ocr = selected.get("ocr") or {}
                resolved["final_transcription"] = selected_ocr.get("full_transcription", "")
                resolved["words"] = selected_ocr.get("words", [])
                resolved["decision_status"] = "false_signature_faithful_fallback"
                resolved.setdefault("notes", []).append(
                    "signature gate rejected: crop contains multiple letter-like stroke components"
                )
    if classification in {"signature_non_text", "no_handwriting"}:
        resolved["final_transcription"] = ""
        resolved["words"] = []
        resolved["decision_status"] = classification
    elif classification == "mark_or_checkbox":
        marks = [str(mark).lower() for mark in resolved.get("marks", [])]
        digits = [str(digit) for digit in resolved.get("digits", []) if str(digit).isdigit()]
        if any(mark in {"x", "×"} for mark in marks) and digits:
            resolved["final_transcription"] = f"סימון x ו המספר {' '.join(digits)}"
            resolved["decision_status"] = "structured_mark"
    return resolved


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
    usable = [
        r for r in results
        if (r.get("ocr") or {}).get("full_transcription")
        or (r.get("ocr") or {}).get("classification")
    ]

    candidates = "\n\n".join(
        f"[{r['source']}]\n{json.dumps(r.get('ocr') or {}, ensure_ascii=False)}" for r in usable
    ) or "(none)"
    readings = [
        (r.get("ocr") or {}).get("full_transcription", "")
        for r in usable
        if r.get("source") != IVRIT_OCR_SOURCE
        if (r.get("ocr") or {}).get("full_transcription", "").strip()
    ]
    agreement = aggregate_votes(readings) if readings else {}
    agreement_text = agreement.get("final_transcription", "") or "(none)"
    context_block = build_context_block(printed_context)
    ivrit_rule = ""
    if any(result.get("source") == IVRIT_OCR_SOURCE for result in usable):
        ivrit_rule = """- The candidate source ivrit_hash_v1 is an independent HASH v1
  CNN-CTC OCR reading. Its confidence is uncalibrated and its measured natural-form
  word error rate is high. Use it only as spelling evidence when the visible strokes
  support it. Never select it solely because it supplies a complete or familiar Hebrew word.
"""
    instruction = f"""You are doing final visual adjudication of Hebrew handwriting OCR.

You will receive:
1. Several enhanced image versions of the SAME handwritten region.
2. Candidate OCR readings from previous passes.
3. Optional nearby printed context.

Rules:
- The images are the source of truth.
- Candidate readings are hints only.
{ivrit_rule}- Original crop, contrast, denoised, and original-image chunks are faithful evidence.
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
- Read every numeric sequence left-to-right and preserve the physical left-to-right order
    of numeric cells/groups. Never reverse numeric groups because the document is Hebrew.
- Ignore printed text unless it helps understand the form field.
- Do not copy printed context as handwriting.
- Do not invent unsupported words.
- Mark uncertain words with [?].
- Classify signatures and checkbox/mark regions explicitly; never transcribe a signature as text.
- Return JSON only.
{context_block}
Candidate OCR readings (hints only):
{candidates}

Deterministic agreement summary across candidates (hint only, not ground truth):
{agreement_text}

Return ONLY a single JSON object:
{{
    "classification": "handwritten_text | mark_or_checkbox | signature_non_text | no_handwriting",
  "final_transcription": "best Hebrew transcription, right-to-left",
    "marks": ["x"],
    "digits": ["1"],
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
    compact_terms = (
        "date", "name", "number", "phone", "email", "account",
        "reference", "identifier", "identification", "id field",
    )
    if any(term in hint for term in compact_terms):
        vpad_frac = min(vpad_frac, 0.12)
    has_color_hint = any(term in hint for term in ("blue", "green", "phone", "email"))
    vpad = max(int(h * vpad_frac), 90 if has_color_hint else 0)
    y0, y1 = max(0, y - vpad), min(H, y + h + vpad)
    color_refined = refine_box_to_colored_ink(page_img, box, y0, y1, color_hint)
    if color_refined is not None:
        return color_refined
    rgb = np.array(page_img.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    colored_page = (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] <= 245)
    if int(colored_page.sum()) >= W * H * 0.00035:
        return box
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
        and component[1] < y + height and component[3] > y
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
    # refine_box_to_ink already pads the detected stroke bounds. Page-relative
    # padding made compact fields absorb adjacent rows on high-resolution pages.
    xpad = max(8, min(int(W * 0.012), int(w * 0.08)))
    top = y
    bottom = min(H, y + h)
    left = max(0, x - xpad)
    right = min(W, x + w + xpad)
    return [left, top, right - left, bottom - top]


def filter_colorless_regions_on_colored_page(
    page_img: Image.Image, regions: list[dict],
) -> list[dict]:
    """Reject printed-text false positives when the page is filled with colored pen."""
    rgb = np.array(page_img.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    colored = (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] <= 245)
    if int(colored.sum()) < page_img.width * page_img.height * 0.00035:
        return regions

    kept = []
    for region in regions:
        x, y, width, height = region["box"]
        local = colored[y:y + height, x:x + width]
        minimum_pixels = max(18, round(min(width, height) * 0.12))
        if int(local.sum()) >= minimum_pixels:
            kept.append(region)
    return kept


def tighten_regions_to_colored_ink(
    page_img: Image.Image, regions: list[dict],
) -> list[dict]:
    """Replace model geometry with local pen-pixel bands on colored-ink pages."""
    rgb = np.array(page_img.convert("RGB")).astype(np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    pen = (
        (blue - red >= 24)
        & (blue - green >= 17)
        & (blue <= 245)
    )
    if int(pen.sum()) < page_img.width * page_img.height * 0.0002:
        return regions

    tightened = []
    for region in regions:
        x, y, width, height = region["box"]
        local = pen[y:y + height, x:x + width]
        if local.size == 0:
            continue
        row_counts = local.sum(axis=1)
        active_rows = row_counts >= max(3, round(width * 0.002))
        active_rows = cv2.dilate(
            active_rows.astype(np.uint8).reshape(-1, 1),
            np.ones((25, 1), np.uint8),
        ).ravel().astype(bool)
        starts = np.flatnonzero(active_rows & ~np.r_[False, active_rows[:-1]])
        ends = np.flatnonzero(active_rows & ~np.r_[active_rows[1:], False]) + 1
        for band_index, (start, end) in enumerate(zip(starts, ends), 1):
            band = local[start:end]
            ys, xs = np.where(band)
            if len(xs) < max(25, round(width * 0.08)):
                continue
            left = max(0, x + int(xs.min()) - 12)
            right = min(page_img.width, x + int(xs.max()) + 13)
            top = max(0, y + start + int(ys.min()) - 12)
            bottom = min(page_img.height, y + start + int(ys.max()) + 13)
            label = region.get("label") or "handwriting"
            if len(starts) > 1:
                label = f"{label} band {band_index}"
            tightened.append({
                "label": label,
                "box": [
                    int(left), int(top), int(right - left), int(bottom - top),
                ],
            })
    return tightened


def tighten_regions_to_black_ink_rows(
    page_img: Image.Image, regions: list[dict],
) -> list[dict]:
    """Snap oversized black-ink form crops to the strongest ruled row."""
    page_width, page_height = page_img.size
    gray_page = np.array(page_img.convert("L"))
    tightened = []
    selected_rows: list[tuple[int, int, float]] = []
    minimum_region_height = max(120, round(page_height * 0.055))

    for region in regions:
        candidate = dict(region, box=list(region["box"]))
        x, y, width, height = candidate["box"]
        label = str(candidate.get("label") or "").lower()
        if "paragraph block" in label or height < minimum_region_height:
            tightened.append(candidate)
            continue

        margin = max(
            40,
            min(round(height * 0.35), round(page_height * 0.04)),
        )
        search_top = max(0, y - margin)
        search_bottom = min(page_height, y + height + margin)
        gray = gray_page[search_top:search_bottom, x:x + width]
        if gray.size == 0:
            tightened.append(candidate)
            continue

        _, ink = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        joined = cv2.morphologyEx(
            ink,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)),
        )
        row_coverage = (joined > 0).sum(axis=1)
        longest_runs = np.zeros(joined.shape[0], dtype=np.int32)
        for row_index, row in enumerate(joined > 0):
            padded = np.r_[False, row, False]
            edges = np.flatnonzero(padded[1:] != padded[:-1])
            if len(edges):
                longest_runs[row_index] = int(
                    (edges[1::2] - edges[::2]).max()
                )

        separator_rows = (
            (longest_runs >= max(60, round(width * 0.08)))
            & (row_coverage >= max(80, round(width * 0.10)))
        )
        separator_rows = cv2.dilate(
            separator_rows.astype(np.uint8).reshape(-1, 1),
            np.ones((3, 1), np.uint8),
        ).ravel().astype(bool)
        starts = np.flatnonzero(
            separator_rows & ~np.r_[False, separator_rows[:-1]]
        )
        ends = np.flatnonzero(
            separator_rows & ~np.r_[separator_rows[1:], False]
        ) + 1
        separators = [int((start + end) // 2) for start, end in zip(starts, ends)]
        if len(separators) < 2:
            tightened.append(candidate)
            continue

        horizontal = cv2.morphologyEx(
            ink,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_RECT, (max(40, width // 12), 1),
            ),
        )
        vertical = cv2.morphologyEx(
            ink,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (1, max(30, (search_bottom - search_top) // 3)),
            ),
        )
        cleaned = cv2.subtract(cv2.subtract(ink, horizontal), vertical)
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8),
        )
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            cleaned, 8,
        )
        minimum_area = max(8, round(page_height * 0.004))
        components = []
        for component_index in range(1, count):
            left = int(stats[component_index, cv2.CC_STAT_LEFT])
            top = int(stats[component_index, cv2.CC_STAT_TOP])
            component_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
            component_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
            area = int(stats[component_index, cv2.CC_STAT_AREA])
            if (
                area >= minimum_area
                and component_width >= 3
                and component_height >= 4
                and component_width < width * 0.32
            ):
                components.append(
                    (left, top, component_width, component_height, area)
                )
        if not components:
            tightened.append(candidate)
            continue

        component_heights = np.array(
            [component[3] for component in components], dtype=np.int32,
        )
        minimum_tall = max(18, round(page_height * 0.0065))
        maximum_tall_cutoff = max(minimum_tall + 8, round(page_height * 0.013))
        tall_cutoff = max(
            minimum_tall,
            min(
                maximum_tall_cutoff,
                int(np.percentile(component_heights, 75) * 1.6),
            ),
        )
        tall_components = [
            component for component in components
            if component[3] >= tall_cutoff
        ]
        if not tall_components:
            tightened.append(candidate)
            continue

        region_center = y + height / 2
        minimum_row_height = max(24, round(page_height * 0.01))
        maximum_row_height = max(
            140,
            min(round(height * 0.75), round(page_height * 0.07)),
        )
        row_candidates = []
        for row_start, row_end in zip(separators, separators[1:]):
            row_height = row_end - row_start
            if not minimum_row_height <= row_height <= maximum_row_height:
                continue
            members = [
                component for component in tall_components
                if row_start <= component[1] + component[3] / 2 <= row_end
            ]
            if len(members) < 2:
                continue
            evidence = sum(
                component[4] * (component[3] / tall_cutoff) ** 1.5
                for component in members
            )
            absolute_top = search_top + row_start
            absolute_bottom = search_top + row_end
            if absolute_top <= region_center <= absolute_bottom:
                distance = 0.0
            else:
                distance = min(
                    abs(region_center - absolute_top),
                    abs(region_center - absolute_bottom),
                )
            score = evidence / (1 + distance / max(1, height * 0.30))
            row_candidates.append(
                (score, absolute_top, absolute_bottom, members)
            )
        if not row_candidates:
            tightened.append(candidate)
            continue

        row_candidates.sort(key=lambda item: item[0], reverse=True)
        best_score = row_candidates[0][0]
        selected = row_candidates[0]
        for row_candidate in row_candidates:
            score, candidate_top, candidate_bottom, _members = row_candidate
            if score < best_score * 0.20:
                break
            collision = False
            for used_top, used_bottom, used_center_x in selected_rows:
                overlap = max(
                    0,
                    min(candidate_bottom, used_bottom)
                    - max(candidate_top, used_top),
                )
                overlap_fraction = overlap / max(
                    1,
                    min(candidate_bottom - candidate_top, used_bottom - used_top),
                )
                if (
                    overlap_fraction >= 0.70
                    and abs((x + width / 2) - used_center_x) <= width * 0.65
                ):
                    collision = True
                    break
            if not collision:
                selected = row_candidate
                break
        _score, row_top, row_bottom, members = selected
        vertical_padding = max(6, round(page_height * 0.0017))
        crop_top = max(0, row_top - vertical_padding)
        crop_bottom = min(page_height, row_bottom + vertical_padding)
        if crop_bottom - crop_top >= height * 0.85:
            tightened.append(candidate)
            continue
            selected_rows.append((row_top, row_bottom, x + width / 2))

        member_left = min(component[0] for component in members)
        member_right = max(component[0] + component[2] for component in members)
        horizontal_span = member_right - member_left
        if horizontal_span >= width * 0.18:
            horizontal_padding = max(12, round((row_bottom - row_top) * 0.18))
            crop_left = max(0, x + member_left - horizontal_padding)
            crop_right = min(
                page_width, x + member_right + horizontal_padding,
            )
        else:
            crop_left, crop_right = x, x + width
        candidate["box"] = [
            int(crop_left),
            int(crop_top),
            int(crop_right - crop_left),
            int(crop_bottom - crop_top),
        ]
        tightened.append(candidate)
    return tightened


def detect_black_ink_form_rows(page_img: Image.Image) -> list[dict]:
    """Find strong handwritten rows bounded by long rules on monochrome forms."""
    page_width, page_height = page_img.size
    gray = np.array(page_img.convert("L"))
    _, ink = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    joined = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1)),
    )
    row_coverage = (joined > 0).sum(axis=1)
    longest_runs = np.zeros(page_height, dtype=np.int32)
    for row_index, row in enumerate(joined > 0):
        padded = np.r_[False, row, False]
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        if len(edges):
            longest_runs[row_index] = int(
                (edges[1::2] - edges[::2]).max()
            )
    separator_rows = (
        (longest_runs >= round(page_width * 0.20))
        & (row_coverage >= round(page_width * 0.22))
    )
    separator_rows = cv2.dilate(
        separator_rows.astype(np.uint8).reshape(-1, 1),
        np.ones((3, 1), np.uint8),
    ).ravel().astype(bool)
    starts = np.flatnonzero(
        separator_rows & ~np.r_[False, separator_rows[:-1]]
    )
    ends = np.flatnonzero(
        separator_rows & ~np.r_[separator_rows[1:], False]
    ) + 1
    separators = [int((start + end) // 2) for start, end in zip(starts, ends)]
    if len(separators) < 4:
        return []

    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(80, page_width // 12), 1),
        ),
    )
    vertical = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(60, page_height // 14)),
        ),
    )
    cleaned = cv2.subtract(cv2.subtract(ink, horizontal), vertical)
    cleaned = cv2.morphologyEx(
        cleaned, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8),
    )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cleaned, 8,
    )
    components = []
    for component_index in range(1, count):
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if (
            area >= max(8, round(page_height * 0.004))
            and width >= 3
            and height >= 4
            and width < page_width * 0.25
        ):
            components.append((x, y, width, height, area))
    if not components:
        return []

    component_heights = np.array(
        [component[3] for component in components], dtype=np.int32,
    )
    tall_cutoff = max(
        18,
        min(
            max(30, round(page_height * 0.013)),
            int(np.percentile(component_heights, 85) * 1.35),
        ),
    )
    tall_components = [
        component for component in components
        if component[3] >= tall_cutoff
    ]
    minimum_row_height = max(24, round(page_height * 0.01))
    maximum_row_height = max(140, round(page_height * 0.04))
    minimum_tall_area = max(300, round(page_height * 0.90))
    minimum_horizontal_span = max(160, round(page_width * 0.18))
    rows = []
    for row_top, row_bottom in zip(separators, separators[1:]):
        row_height = row_bottom - row_top
        if not minimum_row_height <= row_height <= maximum_row_height:
            continue
        members = [
            component for component in tall_components
            if row_top <= component[1] + component[3] / 2 <= row_bottom
        ]
        if len(members) < 6:
            continue
        total_area = sum(component[4] for component in members)
        core_width = max(12, round(page_width * 0.0145))
        core_area = max(80, round(page_height * 0.085))
        core_members = [
            component for component in members
            if component[2] >= core_width and component[4] >= core_area
        ]
        if len(core_members) < 2:
            continue
        core_left = min(component[0] for component in core_members)
        core_right = max(
            component[0] + component[2] for component in core_members
        )
        support_gap = max(30, round(row_height * 2.20))
        support_width = max(8, round(core_width * 0.50))
        supported_members = [
            component for component in members
            if component[2] >= support_width
            and component[0] + component[2] >= core_left - support_gap
            and component[0] <= core_right + support_gap
        ]
        left = min(component[0] for component in supported_members)
        right = max(
            component[0] + component[2] for component in supported_members
        )
        if total_area < minimum_tall_area or right - left < minimum_horizontal_span:
            continue
        horizontal_padding = max(12, round(row_height * 0.18))
        vertical_padding = max(6, round(page_height * 0.0017))
        crop_left = max(0, left - horizontal_padding)
        crop_right = min(page_width, right + horizontal_padding)
        crop_top = max(0, row_top - vertical_padding)
        crop_bottom = min(page_height, row_bottom + vertical_padding)
        rows.append({
            "label": f"black handwritten form row {len(rows) + 1}",
            "box": [
                int(crop_left), int(crop_top),
                int(crop_right - crop_left), int(crop_bottom - crop_top),
            ],
        })
    return rows if len(rows) >= 3 else []


def merge_colored_line_fragments(regions: list[dict]) -> list[dict]:
    """Join adjacent pen fragments only when they occupy the same text line."""
    merged = [dict(region, box=list(region["box"])) for region in regions]
    changed = True
    while changed:
        changed = False
        for first_index, first in enumerate(merged):
            fx, fy, fw, fh = first["box"]
            for second_index in range(first_index + 1, len(merged)):
                second = merged[second_index]
                sx, sy, sw, sh = second["box"]
                vertical_overlap = max(0, min(fy + fh, sy + sh) - max(fy, sy))
                if vertical_overlap < 0.60 * min(fh, sh):
                    continue
                horizontal_gap = max(0, max(fx, sx) - min(fx + fw, sx + sw))
                if horizontal_gap > max(90, round(max(fh, sh) * 0.75)):
                    continue
                left, top = min(fx, sx), min(fy, sy)
                right = max(fx + fw, sx + sw)
                bottom = max(fy + fh, sy + sh)
                first["box"] = [left, top, right - left, bottom - top]
                merged[first_index] = first
                del merged[second_index]
                changed = True
                break
            if changed:
                break
    return merged


def detect_colored_pen_regions(
    page_img: Image.Image, model_regions: list[dict],
) -> list[dict]:
    """Detect all blue pen components; model regions provide labels, not boxes."""
    rgb = np.array(page_img.convert("RGB")).astype(np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    pen = (
        (blue - red >= 24)
        & (blue - green >= 17)
        & (blue <= 245)
    ).astype(np.uint8) * 255
    page_height, page_width = pen.shape
    if int((pen > 0).sum()) < page_width * page_height * 0.0002:
        return []
    pen = cv2.morphologyEx(
        pen, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (35, 7)),
    )
    pen = cv2.dilate(
        pen, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
    )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(pen, 8)
    components = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if (
            area < 180 or width < 25 or height < 12
            or width >= page_width * 0.55 or height >= page_height * 0.08
        ):
            continue
        left, top = max(0, x - 12), max(0, y - 12)
        right = min(page_width, x + width + 12)
        bottom = min(page_height, y + height + 12)
        components.append({
            "label": "colored handwriting",
            "box": [left, top, right - left, bottom - top],
        })
    components = merge_colored_line_fragments(components)
    for component in components:
        x, y, width, height = component["box"]
        best_score, best_label = 0.0, None
        for model_region in model_regions:
            mx, my, mw, mh = model_region["box"]
            intersection = max(0, min(x + width, mx + mw) - max(x, mx)) \
                * max(0, min(y + height, my + mh) - max(y, my))
            score = intersection / max(1, width * height)
            if score > best_score:
                best_score = score
                best_label = model_region.get("label")
        if best_label and best_score >= 0.20:
            component["label"] = best_label
    return components


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
            same_descriptive_label = (
                bool(first.get("label", "").strip())
                and first.get("label", "").strip().lower()
                == second.get("label", "").strip().lower()
                and first.get("label", "").strip().lower()
                not in {"handwriting", "colored handwriting", "blue handwriting"}
            )
            compact_fragment = (
                same_descriptive_label
                and containment >= 0.45
                and area_ratio >= 1.50
                and center_distance <= 0.65 * max(fw, fh, sw, sh)
                and max(first_area, second_area)
                <= 0.05 * page_width * page_height
            )
            if compact_fragment:
                remove.add(smaller_index)
                continue
            duplicate = (
                iou >= 0.70
                or (
                    containment >= 0.70
                    and area_ratio >= 1.20
                    and center_distance <= 0.35 * max(fw, fh, sw, sh)
                )
            )
            if duplicate:
                # Keep the tighter crop. Unioning duplicate boxes lets a chain of
                # nearby fields grow into one page-sized region.
                remove.add(larger_index)
    return [region for index, region in enumerate(kept) if index not in remove]


def widen_midpage_handwriting_lines(
    regions: list[dict], page_width: int, page_height: int,
) -> list[dict]:
    """Preserve local detector geometry for compatibility with older callers."""
    return [dict(region, box=list(region["box"])) for region in regions]


def needs_lower_detection_pass(regions: list[dict], page_height: int) -> bool:
    """Request a detailed lower-page scan when full-page detection only saw the header."""
    if not regions:
        return True
    lowest_center = max(
        region["box"][1] + region["box"][3] / 2
        for region in regions
    )
    return lowest_center < page_height * 0.30


def is_compact_handwriting_label(label: str) -> bool:
    """Return whether detector geometry should stay local to one form field."""
    compact_terms = (
        "date", "name", "number", "phone", "email", "account",
        "reference", "identifier", "identification", "id field", "details",
    )
    lowered = label.lower()
    return any(term in lowered for term in compact_terms)


def compact_handwriting_kind(label: str) -> str | None:
    """Normalize compact labels so overlapping tile detections can be matched."""
    lowered = label.lower()
    if "date" in lowered:
        return "date"
    if "name" in lowered:
        return "name"
    if any(term in lowered for term in ("id number", "identification", "identifier")):
        return "id"
    for kind in ("phone", "email", "account", "reference", "number", "details"):
        if kind in lowered:
            return kind
    return None


def clean_compact_black_pen_regions(
    page_img: Image.Image, regions: list[dict],
) -> list[dict]:
    """Drop empty compact boxes, remove clipped duplicates, and shift weak boxes."""
    if not regions:
        return []
    raw_ink = np.array(page_img.convert("L")) < 180
    thick_ink = np.array(enhance_pen_stroke_isolation(page_img).convert("L")) < 180
    page_height, page_width = raw_ink.shape

    def metrics(box: list[int]) -> tuple[float, float, float, int]:
        x, y, width, height = box
        raw = raw_ink[y:y + height, x:x + width]
        thick = thick_ink[y:y + height, x:x + width]
        if raw.size == 0:
            return 0.0, 0.0, 1.0, 0
        thick_total = int(thick.sum())
        raw_total = int(raw.sum())
        edge = max(3, round(min(width, height) * 0.08))
        edge_total = int(
            thick[:edge].sum() + thick[-edge:].sum()
            + thick[:, :edge].sum() + thick[:, -edge:].sum()
        )
        return (
            thick_total / raw.size,
            thick_total / max(1, raw_total),
            edge_total / max(1, thick_total),
            thick_total,
        )

    supported = []
    for region in regions:
        candidate = dict(region, box=list(region["box"]))
        if is_compact_handwriting_label(str(candidate.get("label") or "")):
            density, _retention, _edge_fraction, _total = metrics(candidate["box"])
            if density < 0.006:
                continue
        supported.append(candidate)

    remove: set[int] = set()
    for first_index, first in enumerate(supported):
        if first_index in remove:
            continue
        first_kind = compact_handwriting_kind(str(first.get("label") or ""))
        if first_kind not in {
            "date", "name", "id", "phone", "email", "account", "reference",
        }:
            continue
        fx, fy, fw, fh = first["box"]
        for second_index in range(first_index + 1, len(supported)):
            if second_index in remove:
                continue
            second = supported[second_index]
            if compact_handwriting_kind(str(second.get("label") or "")) != first_kind:
                continue
            sx, sy, sw, sh = second["box"]
            horizontal_overlap = max(0, min(fx + fw, sx + sw) - max(fx, sx))
            if horizontal_overlap < 0.70 * min(fw, sw):
                continue
            center_distance = abs((fy + fh / 2) - (sy + sh / 2))
            if center_distance > 0.75 * max(fh, sh):
                continue
            first_edge = metrics(first["box"])[2]
            second_edge = metrics(second["box"])[2]
            if abs(first_edge - second_edge) >= 0.015:
                remove.add(first_index if first_edge > second_edge else second_index)
            else:
                first_area, second_area = fw * fh, sw * sh
                remove.add(first_index if first_area > second_area else second_index)
            if first_index in remove:
                break
    supported = [
        region for index, region in enumerate(supported) if index not in remove
    ]

    relocated = []
    for region in supported:
        candidate = dict(region, box=list(region["box"]))
        if compact_handwriting_kind(str(candidate.get("label") or "")) is None:
            relocated.append(candidate)
            continue
        density, retention, _edge_fraction, _total = metrics(candidate["box"])
        if density < 0.006 or retention >= 0.23:
            relocated.append(candidate)
            continue

        x, y, width, height = candidate["box"]
        center_y = y + height / 2
        search_left = max(0, x - round(width * 0.35))
        search_right = min(page_width, x + width + round(width * 0.35))
        search_top = max(0, y - 3 * height)
        search_bottom = min(page_height, y + 4 * height)
        local_thick = thick_ink[
            search_top:search_bottom, search_left:search_right
        ]
        local_raw = raw_ink[
            search_top:search_bottom, search_left:search_right
        ]
        row_counts = local_thick.sum(axis=1)
        active = row_counts >= max(4, round(local_thick.shape[1] * 0.006))
        dilation_height = max(9, min(25, round(height * 0.13)))
        active = cv2.dilate(
            active.astype(np.uint8).reshape(-1, 1),
            np.ones((dilation_height, 1), np.uint8),
        ).ravel().astype(bool)
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        ends = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
        bands = []
        for start, end in zip(starts, ends):
            band_mask = local_thick[start:end]
            thick_total = int(band_mask.sum())
            raw_total = int(local_raw[start:end].sum())
            band_retention = thick_total / max(1, raw_total)
            absolute_center = search_top + (start + end) / 2
            component_count, _labels, stats, _centroids = (
                cv2.connectedComponentsWithStats(
                    band_mask.astype(np.uint8), 8,
                )
            )
            glyph_components = [
                component_index
                for component_index in range(1, component_count)
                if stats[component_index, cv2.CC_STAT_AREA] >= 8
                and stats[component_index, cv2.CC_STAT_HEIGHT] >= 12
                and stats[component_index, cv2.CC_STAT_WIDTH] >= 3
                and stats[component_index, cv2.CC_STAT_WIDTH]
                < local_thick.shape[1] * 0.45
            ]
            if (
                thick_total >= 80
                and len(glyph_components) >= 2
                and band_retention >= max(0.30, retention + 0.10)
                and abs(absolute_center - center_y) <= 3 * height
            ):
                bands.append((abs(absolute_center - center_y), start, end))
        if not bands:
            relocated.append(candidate)
            continue
        _distance, start, end = min(bands)
        rows, columns = np.where(local_thick[start:end])
        if not len(columns):
            relocated.append(candidate)
            continue
        horizontal_padding = max(8, round(width * 0.06))
        vertical_padding = max(8, round(height * 0.08))
        left = max(0, search_left + int(columns.min()) - horizontal_padding)
        right = min(
            page_width,
            search_left + int(columns.max()) + horizontal_padding + 1,
        )
        top = max(
            0, search_top + int(start) + int(rows.min()) - vertical_padding,
        )
        bottom = min(
            page_height,
            search_top + int(start) + int(rows.max()) + vertical_padding + 1,
        )
        candidate["box"] = [left, top, right - left, bottom - top]
        relocated.append(candidate)
    return relocated


def _large_black_handwriting_components(
    page_img: Image.Image,
) -> list[tuple[int, int, int, int]]:
    """Return large thick-stroke components on a monochrome form."""
    rgb = np.array(page_img.convert("RGB")).astype(np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    colored_pen = (
        (blue - red >= 24)
        & (blue - green >= 17)
        & (blue <= 245)
    )
    if int(colored_pen.sum()) >= page_img.width * page_img.height * 0.0002:
        return []

    page_width, page_height = page_img.size
    gray = np.array(page_img.convert("L"))
    _, ink = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(80, page_width // 15), 1),
        ),
    )
    vertical = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(60, page_height // 35)),
        ),
    )
    cleaned = cv2.subtract(cv2.subtract(ink, horizontal), vertical)
    distance = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cleaned, 8,
    )
    minimum_height = max(38, round(page_height * 0.008))
    components = []
    for component_index in range(1, count):
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if (
            height >= minimum_height
            and area >= 120
            and width >= 6
            and width < page_width * 0.18
            and float(distance[labels == component_index].max()) >= 2.4
        ):
            components.append((x, y, x + width, y + height))
    return components


def supplement_black_handwriting_peer_regions(
    page_img: Image.Image, regions: list[dict],
) -> list[dict]:
    """Recover omitted name/ID peers on the same monochrome declaration row."""
    if not regions:
        return []
    components = _large_black_handwriting_components(page_img)
    if not components:
        return [dict(region, box=list(region["box"])) for region in regions]

    page_width, page_height = page_img.size

    peer_terms = ("name", "id number", "hebrew name", "declarant", "entry")
    maximum_anchor_height = max(120, round(page_height * 0.05))
    anchors = [
        index for index, region in enumerate(regions)
        if region["box"][3] <= maximum_anchor_height
        and any(
            term in str(region.get("label") or "").lower()
            for term in peer_terms
        )
    ]
    removed: set[int] = set()
    additions: list[dict] = []
    processed: set[int] = set()
    component_gap = max(80, round(page_height * 0.025))
    minimum_group_span = max(100, round(page_height * 0.025))
    for anchor_index in sorted(
        anchors,
        key=lambda index: (
            regions[index]["box"][1] + regions[index]["box"][3] / 2,
            regions[index]["box"][0],
        ),
    ):
        if anchor_index in processed:
            continue
        _x, y, _width, height = regions[anchor_index]["box"]
        center_y = y + height / 2
        row_tolerance = max(height * 0.75, page_height * 0.018)
        row_anchors = [
            index for index in anchors
            if abs(
                regions[index]["box"][1]
                + regions[index]["box"][3] / 2
                - center_y
            ) <= row_tolerance
        ]
        row_components = sorted(
            component for component in components
            if abs((component[1] + component[3]) / 2 - center_y)
            <= row_tolerance
        )
        groups: list[list[tuple[int, int, int, int]]] = []
        for component in row_components:
            if (
                not groups
                or component[0] - groups[-1][-1][2] > component_gap
            ):
                groups.append([component])
            else:
                groups[-1].append(component)
        peer_groups = []
        for group in groups:
            if len(group) < 3:
                continue
            left = min(component[0] for component in group)
            top = min(component[1] for component in group)
            right = max(component[2] for component in group)
            bottom = max(component[3] for component in group)
            if right - left < minimum_group_span:
                continue
            peer_groups.append((left, top, right, bottom))
        if len(peer_groups) < 2:
            processed.update(row_anchors)
            continue
        horizontal_padding = max(12, round(page_height * 0.004))
        vertical_padding = max(8, round(page_height * 0.002))
        for left, top, right, bottom in peer_groups:
            crop_left = max(0, left - horizontal_padding)
            crop_top = max(0, top - vertical_padding)
            crop_right = min(page_width, right + horizontal_padding)
            crop_bottom = min(page_height, bottom + vertical_padding)
            additions.append({
                "label": "handwritten declaration peer",
                "box": [
                    crop_left,
                    crop_top,
                    crop_right - crop_left,
                    crop_bottom - crop_top,
                ],
            })
        removed.update(row_anchors)
        processed.update(row_anchors)

    supplemented = [
        dict(region, box=list(region["box"]))
        for index, region in enumerate(regions)
        if index not in removed
    ]
    supplemented.extend(additions)
    return supplemented


def supplement_uncovered_black_handwriting_groups(
    page_img: Image.Image, regions: list[dict],
) -> list[dict]:
    """Add large uncovered text groups after declaration peers establish context."""
    supplemented = [
        dict(region, box=list(region["box"])) for region in regions
    ]
    if sum(
        str(region.get("label") or "") == "handwritten declaration peer"
        for region in regions
    ) < 2:
        return supplemented
    components = _large_black_handwriting_components(page_img)
    if not components:
        return supplemented

    page_width, page_height = page_img.size
    row_tolerance = max(55, round(page_height * 0.015))
    component_gap = max(80, round(page_height * 0.025))
    minimum_group_span = max(100, round(page_height * 0.025))
    rows: list[dict] = []
    for component in sorted(
        components,
        key=lambda item: ((item[1] + item[3]) / 2, item[0]),
    ):
        center_y = (component[1] + component[3]) / 2
        matching_row = next(
            (
                row for row in rows
                if abs(center_y - row["center_y"]) <= row_tolerance
            ),
            None,
        )
        if matching_row is None:
            rows.append({"center_y": center_y, "components": [component]})
        else:
            matching_row["components"].append(component)
            matching_row["center_y"] = sum(
                (item[1] + item[3]) / 2
                for item in matching_row["components"]
            ) / len(matching_row["components"])

    horizontal_padding = max(12, round(page_height * 0.004))
    vertical_padding = max(8, round(page_height * 0.002))
    for row in rows:
        if row["center_y"] < page_height * 0.12:
            continue
        groups: list[list[tuple[int, int, int, int]]] = []
        for component in sorted(row["components"]):
            if (
                not groups
                or component[0] - groups[-1][-1][2] > component_gap
            ):
                groups.append([component])
            else:
                groups[-1].append(component)
        for group in groups:
            if len(group) < 3:
                continue
            left = min(component[0] for component in group)
            top = min(component[1] for component in group)
            right = max(component[2] for component in group)
            bottom = max(component[3] for component in group)
            if right - left < minimum_group_span:
                continue
            crop_left = max(0, left - horizontal_padding)
            crop_top = max(0, top - vertical_padding)
            crop_right = min(page_width, right + horizontal_padding)
            crop_bottom = min(page_height, bottom + vertical_padding)
            candidate = [
                crop_left,
                crop_top,
                crop_right - crop_left,
                crop_bottom - crop_top,
            ]
            candidate_area = candidate[2] * candidate[3]
            covered = False
            for region in supplemented:
                rx, ry, rw, rh = region["box"]
                intersection = (
                    max(0, min(crop_right, rx + rw) - max(crop_left, rx))
                    * max(0, min(crop_bottom, ry + rh) - max(crop_top, ry))
                )
                if intersection / max(1, candidate_area) >= 0.35:
                    covered = True
                    break
            if not covered:
                supplemented.append({
                    "label": "handwritten stroke group",
                    "box": candidate,
                })
    return supplemented


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

You receive two aligned views with identical dimensions. The first is the
authoritative original. The second suppresses thin print and rules while retaining
the thick cores of pen strokes. Use the second only as supporting evidence: a box
must contain irregular pen evidence there and visible handwriting in the original.

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
            elif is_compact_handwriting_label(label):
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
        {"type": "input_text", "text": "View 1: original document"},
        {"type": "input_image", "image_url": _to_data_uri(det_img)},
        {"type": "input_text", "text": "View 2: pen-stroke-isolated support"},
        {
            "type": "input_image",
            "image_url": _to_data_uri(enhance_pen_stroke_isolation(det_img)),
        },
    ]
    def detection_call(evidence: list[dict]) -> dict:
        return _responses_call(
            evidence,
            retries=0,
            reasoning=DETECTION_REASONING,
            max_tokens=DETECTION_MAX_TOKENS,
            endpoint_limit=1,
        )

    try:
        initial_response = detection_call(content)
    except RuntimeError as exc:
        print(
            f"[detect] full-page detection unavailable; trying tiled coverage: {exc}"
        )
        regions = []
    else:
        regions = parse_response(initial_response)
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
        try:
            retry_regions = parse_response(detection_call(retry_content))
        except RuntimeError as exc:
            print(f"[detect] geometry repair unavailable; keeping initial regions: {exc}")
        else:
            repaired = repair_invalid_detection_regions(regions, retry_regions, W, H)
            repaired_penalty, _ = detection_geometry_penalty(repaired, W, H)
            if repaired_penalty < penalty:
                print(f"[detect] geometry repair improved penalty {penalty:.1f} -> {repaired_penalty:.1f}")
                regions = repaired
    missing_lower_coverage = needs_lower_detection_pass(regions, H)
    if missing_lower_coverage:
        coverage_bands = (
            (0.0, 0.30), (0.24, 0.48), (0.42, 0.66),
            (0.60, 0.82), (0.76, 1.0),
        )
        coverage_regions: list[dict] = []
        successful_segments = 0
        for band_index, (start_fraction, end_fraction) in enumerate(
            coverage_bands, 1,
        ):
            lower_top = round(H * start_fraction)
            lower_bottom = round(H * end_fraction)
            lower_page = page_img.crop((0, lower_top, W, lower_bottom))
            lower_scale = min(1.0, det_max_side / max(lower_page.size))
            lower_width = int(lower_page.width * lower_scale)
            lower_height = int(lower_page.height * lower_scale)
            lower_image = lower_page.resize(
                (lower_width, lower_height), Image.LANCZOS
            ) if lower_scale < 1.0 else lower_page
            lower_prompt = f"""This is vertical segment {band_index} of 2 from a scanned Hebrew document,
covering {start_fraction:.0%}-{end_fraction:.0%} of the full page and measuring
{lower_width}x{lower_height} pixels. Scan this entire segment from top to bottom.
Find every true handwritten pen entry that a whole-page scan may have missed.
Include long handwriting crossing printed form lines. Exclude all machine-printed
text, italic printed text, logos, stamps, signatures, and black redaction boxes.
Do not infer handwriting from a printed field label; include a box only when
irregular handwritten pen strokes are visibly present inside it. You receive the
aligned original followed by a pen-stroke-isolated support view; require evidence
in both and use the original for coordinates.

Return ONLY one JSON object with coordinates in THIS image:
{{"regions": [{{"label": "short english description", "box": [x, y, width, height]}}]}}
Each box must tightly cover one handwritten field or line with a small margin."""
            lower_content = [
                {"type": "input_text", "text": lower_prompt},
                {"type": "input_text", "text": "View 1: original document segment"},
                {"type": "input_image", "image_url": _to_data_uri(lower_image)},
                {"type": "input_text", "text": "View 2: pen-stroke-isolated support"},
                {
                    "type": "input_image",
                    "image_url": _to_data_uri(
                        enhance_pen_stroke_isolation(lower_image)
                    ),
                },
            ]
            try:
                lower_response = detection_call(lower_content)
            except RuntimeError as exc:
                print(
                    f"[detect] coverage segment {band_index} unavailable; "
                    f"keeping other regions: {exc}"
                )
                continue
            successful_segments += 1
            lower_regions = parse_response(
                lower_response,
                coordinate_inv=1.0 / lower_scale,
                offset_y=lower_top,
            )
            if lower_regions:
                print(
                    f"[detect] coverage segment {band_index} added "
                    f"{len(lower_regions)} candidate region(s)"
                )
                coverage_regions.extend(lower_regions)
        if successful_segments == len(coverage_bands):
            print(
                "[detect] complete tiled coverage replaced the incomplete "
                "full-page scan"
            )
            regions = coverage_regions
        else:
            regions.extend(coverage_regions)
    colored_regions = detect_colored_pen_regions(page_img, regions)
    if colored_regions:
        regions = colored_regions
    else:
        black_form_rows = detect_black_ink_form_rows(page_img)
        regions = black_form_rows or tighten_regions_to_black_ink_rows(
            page_img, regions,
        )
        regions = merge_colored_line_fragments(
            tighten_regions_to_colored_ink(page_img, regions)
        )
        regions = supplement_black_handwriting_peer_regions(page_img, regions)
        regions = clean_compact_black_pen_regions(page_img, regions)
        regions = deduplicate_detection_regions(regions, W, H)
        regions = supplement_uncovered_black_handwriting_groups(
            page_img, regions,
        )
    return deduplicate_detection_regions(regions, W, H)


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
    (ocr_tasks, all_images): ocr_tasks contains the four human-benchmarked core
    views; all_images retains every transform for visual inspection/reconciliation."""
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
        ocr_tasks += [(name, versions[name]) for name in CORE_OCR_VIEWS]
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
    print(
        "[GPT] endpoint pool: "
        + ", ".join(
            f"{target['name']} ({target['deployment']})" for target in GPT_TARGETS
        )
    )

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

    # HASH v1 is optional advisory evidence. Start it before GPT so CPU inference
    # overlaps network calls; one health failure disables it for the whole page.
    ivrit_available, ivrit_detail = ivrit_ocr_service_status()
    ivrit_status = "pending" if ivrit_available else (
        "disabled" if not IVRIT_OCR_ENABLED else "unavailable"
    )
    ivrit_results: list[dict] = [
        {
            "status": ivrit_status,
            "source": IVRIT_OCR_SOURCE,
            "model": "HASH v1",
            "text": "",
            "confidence": 0.0,
            "lines": [],
            "detail": ivrit_detail,
        }
        for _region in regions
    ]
    ivrit_executor = None
    ivrit_futures = {}
    if ivrit_available:
        print(
            f"[ivrit] HASH v1 ready; scheduling {len(regions)} region crop(s) "
            f"with {min(IVRIT_OCR_WORKERS, len(regions))} worker(s) ..."
        )
        ivrit_executor = ThreadPoolExecutor(
            max_workers=min(IVRIT_OCR_WORKERS, len(regions)),
        )
        ivrit_futures = {
            ivrit_executor.submit(
                ocr_image_with_ivrit,
                dict(region_images[region_index])["01_original_crop.png"],
            ): region_index
            for region_index in range(len(regions))
        }
    else:
        print(f"[ivrit] HASH v1 advisory OCR unavailable: {ivrit_detail}")

    # ---- run ALL vision calls in parallel across ALL regions -------------- #
    # Self-consistency: draw each image version VOTE_SEEDS times (distinct seeds)
    # so the deterministic aggregation has multiple genuine samples to vote over.
    vote_seeds = list(range(1, VOTE_SEEDS + 1)) if VOTE_ENABLED else [None]
    segmented_plans: dict[int, dict] = {}
    standard_regions = set(range(len(regions)))
    if OCR_STRATEGY == ISOLATED_MULTIVIEW_STRATEGY:
        for ri, images in enumerate(region_images):
            try:
                segmented_plans[ri] = prepare_isolated_multiview_plan(
                    dict(images)["01_original_crop.png"], region_dirs[ri],
                )
                standard_regions.discard(ri)
                print(
                    f"[OCR] region {ri + 1}: isolated multi-view plan with "
                    f"{segmented_plans[ri]['draws']} blind draw(s)"
                )
            except Exception as exc:
                print(
                    f"[OCR] region {ri + 1}: isolated multi-view unavailable "
                    f"({exc}); using standard OCR"
                )

    def _standard_tasks(indices: set[int]) -> list[tuple]:
        if MULTI_IMAGE_OCR:
            return [
                (ri, f"core4_multi#s{s}" if s is not None else "core4_multi",
                 region_tasks[ri], s)
                for ri in sorted(indices)
                for s in vote_seeds
            ]
        return [
            (ri, f"{name}#s{s}" if s is not None else name, image, s)
            for ri in sorted(indices)
            for name, image in region_tasks[ri]
            for s in vote_seeds
        ]

    flat = _standard_tasks(standard_regions)
    region_results: list[list[dict]] = [[] for _ in regions]
    segmented_finals: list[dict | None] = [None for _ in regions]
    draw_progress = {
        "completed": 0,
        "failed": 0,
        "total": len(flat) + sum(plan["draws"] for plan in segmented_plans.values()),
    }

    def _publish_ocr_progress() -> None:
        _write_json_atomic(out / "ocr_progress.json", {
            **draw_progress,
            "page": page,
            "usage": _get_usage(),
            "updated_at": time.time(),
            "regions": [
                {
                    "region": index + 1,
                    "label": regions[index].get("label"),
                    "box": regions[index]["box"],
                    "completed_draws": len(results),
                    "ivrit_ocr": ivrit_results[index],
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
    if segmented_plans:
        print(
            f"[OCR] strategy={OCR_STRATEGY}: {draw_progress['total']} total draw(s); "
            f"up to {ISOLATED_MULTIVIEW_WORKERS} isolated calls run concurrently"
        )
    else:
        print(f"[OCR] {len(flat)} vision draws across {len(regions)} region(s) "
              f"({len(vote_seeds)} seed(s)/version), {MAX_WORKERS} in parallel ...")

    def _ocr_batch(tasks):
        """Run a batch of (region_idx, name, image evidence, seed) OCR draws in parallel.
        Appends successes to region_results; returns the tasks that failed."""
        failed = []
        workers = max(1, min(MAX_WORKERS, len(tasks)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_meta = {
                ex.submit(
                    ocr_image_set if isinstance(evidence, list) else ocr_image,
                    evidence,
                    prompt,
                    seed,
                ): (ri, name, evidence, seed)
                for ri, name, evidence, seed in tasks
            }
            for fut in as_completed(fut_meta):
                ri, name, evidence, seed = fut_meta[fut]
                try:
                    region_results[ri].append({"source": name, "ocr": fut.result()})
                    draw_progress["completed"] += 1
                except Exception as exc:
                    failed.append((ri, name, evidence, seed))
                    draw_progress["failed"] += 1
                    print(f"   region {ri + 1} {name} failed: {exc}", file=sys.stderr)
                _publish_ocr_progress()
        return failed

    def _retryable(task) -> bool:
        source = task[1].split("#", 1)[0]
        return source.startswith("chunk_") or source in RETRYABLE_OCR_VIEWS

    # Retry only faithful evidence. Optional derived transforms are one-shot so a
    # slow enhancement cannot dominate tail latency.
    failures = _ocr_batch(flat) if flat else []
    for ri, plan in segmented_plans.items():
        def _segmented_progress(success: bool) -> None:
            draw_progress["completed" if success else "failed"] += 1
            _publish_ocr_progress()

        try:
            final, candidates, details = run_isolated_multiview_plan(
                plan, on_draw=_segmented_progress,
            )
            segmented_finals[ri] = final
            region_results[ri].extend(candidates)
            _write_json_atomic(
                region_dirs[ri] / "segmented_multiview_result.json", details,
            )
            _publish_ocr_progress()
        except Exception as exc:
            print(
                f"[OCR] region {ri + 1}: isolated multi-view failed ({exc}); "
                "running standard fallback",
                file=sys.stderr,
            )
            fallback = _standard_tasks({ri})
            draw_progress["total"] += len(fallback)
            _publish_ocr_progress()
            failures.extend(_ocr_batch(fallback))

    for rnd in range(1, 3):
        failures = [task for task in failures if _retryable(task)]
        if not failures:
            break
        print(f"[OCR] retry round {rnd}: re-running {len(failures)} failed call(s) ...")
        time.sleep(3 * rnd)
        failures = _ocr_batch(failures)

    # Safety net: any region that still has NO successful reading -> retry its images once more.
    stranded = [ri for ri in range(len(regions))
                if not any(
                    (r.get("ocr") or {}).get("full_transcription")
                    or (r.get("ocr") or {}).get("classification")
                    for r in region_results[ri]
                )]
    if stranded:
        if MULTI_IMAGE_OCR:
            retry_tasks = [
                (
                    ri,
                    "core2_faithful_retry",
                    [(name, image) for name, image in region_tasks[ri]
                     if name in RETRYABLE_OCR_VIEWS],
                    900,
                )
                for ri in stranded
            ]
        else:
            retry_tasks = [(ri, f"{name}#r{k}", img, 900 + k)
                           for ri in stranded for k, (name, img) in enumerate(region_tasks[ri])
                           if name in RETRYABLE_OCR_VIEWS or name.startswith("chunk_")]
        print(f"[OCR] {len(stranded)} region(s) still empty -> final retry of {len(retry_tasks)} call(s) ...")
        time.sleep(5)
        rescue_failures = _ocr_batch(retry_tasks)
        if MULTI_IMAGE_OCR and rescue_failures and OCR_FINAL_ORIGINAL_RESCUE:
            still_stranded = [
                ri for ri, _name, _evidence, _seed in rescue_failures
                if not any(
                    (result.get("ocr") or {}).get("full_transcription")
                    or (result.get("ocr") or {}).get("classification")
                    for result in region_results[ri]
                )
            ]
            original_retries = [
                (
                    ri,
                    "01_original_crop.png#final_rescue",
                    dict(region_tasks[ri])["01_original_crop.png"],
                    901,
                )
                for ri in sorted(set(still_stranded))
            ]
            if original_retries:
                print(
                    f"[OCR] final original-image rescue for "
                    f"{len(set(still_stranded))} region(s) ..."
                )
                _ocr_batch(original_retries)
    if failures:
        print(f"[OCR] WARNING: {len(failures)} call(s) still failing after all retries", file=sys.stderr)

    if ivrit_futures:
        for future in as_completed(ivrit_futures):
            region_index = ivrit_futures[future]
            try:
                ivrit_results[region_index] = future.result()
                print(
                    f"[ivrit] region {region_index + 1}: "
                    f"{ivrit_results[region_index].get('text', '')!r}"
                )
            except Exception as exc:
                ivrit_results[region_index] = {
                    "status": "failed",
                    "source": IVRIT_OCR_SOURCE,
                    "model": "HASH v1",
                    "text": "",
                    "confidence": 0.0,
                    "lines": [],
                    "detail": str(exc),
                }
                print(
                    f"[ivrit] region {region_index + 1} failed: {exc}",
                    file=sys.stderr,
                )
            _write_json_atomic(
                region_dirs[region_index] / "ivrit_ocr.json",
                ivrit_results[region_index],
            )
            _publish_ocr_progress()
        ivrit_executor.shutdown(wait=True)
    else:
        for region_index, evidence in enumerate(ivrit_results):
            _write_json_atomic(region_dirs[region_index] / "ivrit_ocr.json", evidence)

    blue_only_inputs = {}
    for region_index, evidence in enumerate(ivrit_results):
        if not should_try_ivrit_blue_only(
            evidence, region_results[region_index], regions[region_index]
        ):
            continue
        blue_image = prepare_ivrit_blue_only_image(
            dict(region_images[region_index])["01_original_crop.png"]
        )
        if blue_image is not None:
            blue_only_inputs[region_index] = blue_image
    if blue_only_inputs:
        print(
            f"[ivrit] retrying {len(blue_only_inputs)} compact region(s) "
            "with blue-only isolation ..."
        )
        with ThreadPoolExecutor(
            max_workers=min(IVRIT_OCR_WORKERS, len(blue_only_inputs))
        ) as executor:
            futures = {
                executor.submit(ocr_image_with_ivrit, image): region_index
                for region_index, image in blue_only_inputs.items()
            }
            for future in as_completed(futures):
                region_index = futures[future]
                try:
                    blue_evidence = future.result()
                except Exception as exc:
                    blue_evidence = {
                        "status": "failed",
                        "source": IVRIT_OCR_SOURCE,
                        "model": "HASH v1",
                        "text": "",
                        "confidence": 0.0,
                        "lines": [],
                        "detail": str(exc),
                    }
                ivrit_results[region_index] = select_ivrit_preprocessing_variant(
                    ivrit_results[region_index],
                    blue_evidence,
                    region_results[region_index],
                    regions[region_index],
                )
                print(
                    f"[ivrit] region {region_index + 1}: selected "
                    f"{ivrit_results[region_index].get('selected_variant')} "
                    f"{ivrit_results[region_index].get('text', '')!r}"
                )

    for region_index, evidence in enumerate(ivrit_results):
        evidence["advisory_gate"] = ivrit_ocr_advisory_gate(
            evidence, region_results[region_index], regions[region_index]
        )
        _write_json_atomic(region_dirs[region_index] / "ivrit_ocr.json", evidence)
        if evidence["advisory_gate"]["eligible"]:
            print(
                f"[ivrit] region {region_index + 1}: advisory candidate eligible "
                f"({evidence['advisory_gate']['character_agreement']:.0%} "
                "character agreement)"
            )
    _publish_ocr_progress()

    cu_context_markdown = ""
    if NEARBY_CU_CONTEXT and cu_future is not None:
        try:
            cu_context_markdown = cu_future.result()
        except Exception as exc:
            print(f"[CU] context unavailable: {exc}", file=sys.stderr)

    # ---- final aggregation per region (deterministic vote -> visual -> text) - #
    print("[vote] aggregating self-consistency draws per region ..." if VOTE_ENABLED
          else "[reconcile] final visual adjudication per region ...")

    def _results_with_ivrit(region_index: int) -> list[dict]:
        results = list(region_results[region_index])
        evidence = ivrit_results[region_index]
        candidate = (
            ivrit_ocr_candidate(evidence)
            if (evidence.get("advisory_gate") or {}).get("eligible")
            else None
        )
        if candidate is not None:
            results.append(candidate)
        return results

    def _reconcile_region(ri: int) -> dict:
        if segmented_finals[ri] is not None:
            return segmented_finals[ri] or {}
        res = _results_with_ivrit(ri)
        usable = [
            result for result in res
            if (result.get("ocr") or {}).get("full_transcription", "").strip()
            or (result.get("ocr") or {}).get("classification")
        ]
        semantic_usable = [
            result for result in usable
            if result.get("source") != IVRIT_OCR_SOURCE
        ]
        if VOTE_ENABLED:
            readings = [
                (r.get("ocr") or {}).get("full_transcription", "")
                for r in semantic_usable
            ]
            if readings:
                vote = aggregate_votes(readings)
                if "[?]" not in vote.get("final_transcription", ""):
                    return finalize_region_semantics(vote, semantic_usable)  # no model call
        imgs = region_images[ri]
        original_crop = dict(imgs)["01_original_crop.png"]
        signature_flourish = (
            is_signature_flourish(original_crop)
            or is_black_signature_flourish(original_crop)
        )
        local_cu_context = nearby_printed_context(
            cu_context_markdown, regions[ri]["box"], page_img.height
        )
        region_context = "\n".join(
            context for context in (printed_context, local_cu_context) if context.strip()
        )

        def _contextualize(final: dict) -> dict:
            final = finalize_region_semantics(
                final, semantic_usable, signature_flourish,
            )
            if should_recheck_hebrew_numeric_script_conflict(
                final, ivrit_results[ri], regions[ri],
            ):
                try:
                    script_recheck, script_images = script_conflict_region_recheck(
                        page_img, regions[ri], region_context,
                    )
                    _write_json_atomic(
                        region_dirs[ri] / "script_conflict_recheck.json",
                        script_recheck,
                    )
                    if "blue_only" in script_images:
                        script_images["blue_only"].save(
                            region_dirs[ri] / "08_script_blue_only.png"
                        )
                    script_images["context"].save(
                        region_dirs[ri] / "09_script_context.png"
                    )
                    script_images["page"].save(
                        region_dirs[ri] / "10_script_page.png"
                    )
                    script_resolved = dict(accept_hebrew_numeric_script_recheck(
                        final, script_recheck,
                    ))
                    if script_resolved.get("final_transcription") != final.get(
                        "final_transcription"
                    ):
                        return script_resolved
                    unresolved = dict(preserve_uncertain_hebrew_script_conflict(
                        final, script_recheck,
                    ))
                    if unresolved.get("final_transcription") != final.get(
                        "final_transcription"
                    ):
                        return unresolved
                except Exception as script_exc:
                    final.setdefault("notes", []).append(
                        f"script conflict recheck failed: {script_exc}"
                    )
            if not should_contextual_recheck(final, usable):
                return final
            try:
                pre_context_text = final.get("final_transcription", "")
                recheck, context_images = contextual_region_recheck(
                    page_img, regions[ri], final, region_context
                )
                context_images[0].save(region_dirs[ri] / "08_context_highlight.png")
                context_images[1].save(region_dirs[ri] / "09_page_highlight.png")
                _write_json_atomic(region_dirs[ri] / "contextual_recheck.json", recheck)
                accepted = dict(accept_contextual_recheck(final, recheck))
                accepted = protect_contextual_recheck_with_cross_engine_agreement(
                    final,
                    recheck,
                    accepted,
                    semantic_usable,
                    ivrit_results[ri],
                )
                accepted["pre_context_transcription"] = pre_context_text
                accepted["contextual_evidence"] = recheck
                pre_info = _information_text(pre_context_text)
                recheck_info = _information_text(recheck.get("final_transcription", ""))
                if (
                    re.search(r"[\u0590-\u05ff]", pre_info)
                    and re.search(r"[\u0590-\u05ff]", recheck_info)
                    and pre_info != recheck_info
                ):
                    accepted["needs_review"] = True
                    alternatives = accepted.setdefault("alternative_transcriptions", [])
                    for text in (pre_context_text, recheck.get("final_transcription", "")):
                        if text and text != accepted.get("final_transcription") and text not in alternatives:
                            alternatives.append(text)
                    accepted.setdefault("notes", []).append(
                        "independent Hebrew context reread disagreed; human review required"
                    )
                if should_high_uncertainty_rescue(
                    accepted, signature_flourish,
                ):
                    high_recheck, _high_images = contextual_region_recheck(
                        page_img,
                        regions[ri],
                        accepted,
                        region_context,
                        reasoning="high",
                    )
                    _write_json_atomic(
                        region_dirs[ri] / "high_uncertainty_recheck.json",
                        high_recheck,
                    )
                    accepted["high_uncertainty_evidence"] = high_recheck
                    rescued = dict(accept_high_uncertainty_rescue(
                        accepted, high_recheck,
                    ))
                    if rescued.get("final_transcription") != accepted.get(
                        "final_transcription"
                    ):
                        rescued["pre_high_rescue_transcription"] = accepted.get(
                            "final_transcription", ""
                        )
                        rescued["high_uncertainty_evidence"] = high_recheck
                        rescued["decision_status"] = "high_uncertainty_rescue"
                        rescued.setdefault("notes", []).append(
                            "high-reasoning rescue resolved explicit uncertainty-only text"
                        )
                        return rescued
                return accepted
            except Exception as recheck_exc:
                final.setdefault("notes", []).append(
                    f"contextual recheck failed: {recheck_exc}"
                )
                return final

        try:  # 1) visual adjudication with images (primary)
            final = visual_reconcile_ocr_results(imgs, res, region_context)
            final.setdefault("notes", []).append("visual reconciliation")
            return _contextualize(final)
        except Exception as vexc:
            try:  # 2) text-only reconciliation (fallback)
                final = reconcile_ocr_results(res, region_context)
                final.setdefault("notes", []).append(f"text reconciliation (visual failed: {vexc})")
                return _contextualize(final)
            except Exception as texc:  # 3) best non-empty per-image reading
                vision_usable = [
                    result for result in semantic_usable
                    if (result.get("ocr") or {}).get("full_transcription")
                ]
                if vision_usable:
                    best = max(
                        vision_usable,
                        key=lambda r: len(r["ocr"].get("full_transcription", "")),
                    )
                    return _contextualize({
                        "final_transcription": best["ocr"].get("full_transcription", ""),
                        "words": best["ocr"].get("words", []),
                        "notes": [f"per-image fallback (text failed: {texc})"],
                    })
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

    cu_evidence_markdown = cu_context_markdown
    if not cu_evidence_markdown and cu_future is not None:
        try:
            cu_evidence_markdown = cu_future.result()
        except Exception as exc:
            print(f"[CU] numeric evidence unavailable: {exc}", file=sys.stderr)

    finals = [separate_numeric_field_marks(final) for final in finals]
    finals = [normalize_numeric_s_homoglyph(final) for final in finals]
    finals = [
        prefer_eligible_ivrit_candidate(final, ivrit_results[index])
        for index, final in enumerate(finals)
    ]
    finals = reconcile_repeated_dates(finals, regions)
    finals = reconcile_with_cu_numeric_tokens(
        finals, regions, cu_evidence_markdown,
    )
    finals = reconcile_repeated_structured_fields(finals, regions)
    finals = reconcile_uncertain_repeated_numbers(
        finals, regions, cu_evidence_markdown,
    )

    # ---- write per-region debug + combined -------------------------------- #
    all_results = []
    rejected_results = []
    for i, region in enumerate(regions, 1):
        rdir = region_dirs[i - 1]
        res = region_results[i - 1]
        reconciliation_results = _results_with_ivrit(i - 1)
        final = finals[i - 1] or {}
        local_cu_context = nearby_printed_context(
            cu_context_markdown, region["box"], page_img.height
        )
        region_context = "\n".join(
            context for context in (printed_context, local_cu_context) if context.strip()
        )

        per_image = [
            {"source": r["source"],
             "full_transcription": (r.get("ocr") or {}).get("full_transcription", ""),
             "classification": (r.get("ocr") or {}).get("classification"),
             "marks": (r.get("ocr") or {}).get("marks", []),
             "digits": (r.get("ocr") or {}).get("digits", []),
             "words": (r.get("ocr") or {}).get("words", [])}
            for r in res
        ]
        (rdir / "per_image_ocr.json").write_text(
            json.dumps(per_image, ensure_ascii=False, indent=2), encoding="utf-8")

        sel = _select_visual_images(region_images[i - 1], res)
        vinput = {
            "printed_context_used": region_context,
            "selected_images": [n for n, _ in sel],
            "candidate_readings": [
                {"source": r["source"], "text": (r.get("ocr") or {}).get("full_transcription", "")}
                for r in reconciliation_results
            ],
        }
        (rdir / "visual_reconcile_input.json").write_text(
            json.dumps(vinput, ensure_ascii=False, indent=2), encoding="utf-8")

        result_debug = {
            "label": region.get("label"),
            "box": region["box"],
            "printed_context_used": region_context,
            "per_image": per_image,
            "ivrit_ocr": ivrit_results[i - 1],
            "field_type": final.get("field_type"),
            "final_transcription": final.get("final_transcription", ""),
            "numeric_cells": final.get("numeric_cells", []),
            "spatial_numeric_assembly": final.get("spatial_numeric_assembly", False),
            "pre_context_transcription": final.get("pre_context_transcription"),
            "contextual_evidence": final.get("contextual_evidence"),
            "pre_high_rescue_transcription": final.get(
                "pre_high_rescue_transcription"
            ),
            "high_uncertainty_evidence": final.get("high_uncertainty_evidence"),
            "pre_script_conflict_transcription": final.get(
                "pre_script_conflict_transcription"
            ),
            "script_conflict_evidence": final.get("script_conflict_evidence"),
            "alternative_transcriptions": final.get("alternative_transcriptions", []),
            "words": final.get("words", []),
            "notes": final.get("notes", []),
            "needs_review": final.get("needs_review", False),
            "classification": final.get("classification", "handwritten_text"),
            "decision_status": final.get("decision_status", "visual_reconciliation"),
            "agreement_gate": final.get("agreement_gate"),
            "cross_engine_agreement": final.get("cross_engine_agreement"),
            "cross_region_consensus": final.get("cross_region_consensus"),
            "vote": {
                "candidate_count": len(per_image),
                "selected_text": final.get("final_transcription", ""),
                "alternatives": final.get("alternative_transcriptions", []),
                "method": (final.get("notes") or ["unknown"])[0],
            },
        }
        (rdir / "result.json").write_text(
            json.dumps(result_debug, ensure_ascii=False, indent=2), encoding="utf-8")

        exported_result = {
            "region": i, "folder": rdir.name,
            "label": region.get("label"), "box": region["box"],
            "field_type": final.get("field_type"),
            "final_transcription": final.get("final_transcription"),
            "pre_context_transcription": final.get("pre_context_transcription"),
            "contextual_evidence": final.get("contextual_evidence"),
            "pre_high_rescue_transcription": final.get(
                "pre_high_rescue_transcription"
            ),
            "high_uncertainty_evidence": final.get("high_uncertainty_evidence"),
            "pre_script_conflict_transcription": final.get(
                "pre_script_conflict_transcription"
            ),
            "script_conflict_evidence": final.get("script_conflict_evidence"),
            "ivrit_ocr": ivrit_results[i - 1],
            "numeric_cells": final.get("numeric_cells", []),
            "spatial_numeric_assembly": final.get("spatial_numeric_assembly", False),
            "words": final.get("words"), "notes": final.get("notes"),
            "classification": final.get("classification", "handwritten_text"),
            "needs_review": final.get("needs_review", False),
            "decision_status": final.get("decision_status", "visual_reconciliation"),
            "agreement_gate": final.get("agreement_gate"),
            "cross_engine_agreement": final.get("cross_engine_agreement"),
            "cross_region_consensus": final.get("cross_region_consensus"),
        }
        if should_emit_region(final, per_image):
            all_results.append(exported_result)
        else:
            exported_result["exclusion_reason"] = "empty_no_handwriting"
            rejected_results.append(exported_result)
        print(f"  [{i}] {region.get('label')}: {final.get('final_transcription', '')!r}")

    combined = {
        "source": input_path, "page": page,
        "region_map": overlay_name, "regions": all_results,
        "rejected_regions": rejected_results,
        "ocr_strategy": OCR_STRATEGY,
    }
    # handwriting_result.json is written AFTER the merge step below so the token
    # totals include the merge call and the elapsed time covers the whole run.

    # ---- merged doc: handwriting (red) over printed page ------------------- #
    cu_markdown = cu_evidence_markdown if merge_md else ""
    if merge_md and cu_future is not None and not cu_markdown:
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
    if usage.get("targets"):
        print(
            "GPT targets:   "
            + ", ".join(
                f"{name}={calls}" for name, calls in sorted(usage["targets"].items())
            )
        )
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
