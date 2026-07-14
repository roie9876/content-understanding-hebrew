#!/usr/bin/env python3
"""
Extract printed + handwritten Hebrew text from a scanned PDF using BOTH:

  1. Azure AI Content Understanding  -> OCR of printed Hebrew + layout +
     structured field extraction (LLM reads handwriting into fields, with
     confidence scores).
  2. Azure OpenAI GPT vision         -> independent full transcription that
     separates PRINTED vs HANDWRITTEN text and extracts the same fields.

Both results are written side-by-side to out/result.json so you can compare
how each engine read the Hebrew handwriting.

Auth: Microsoft Entra ID (Azure CLI token) for BOTH services. No API keys are
stored anywhere. Just run `az login` first.

Usage:
    python extract_hebrew.py "sample/אישור מסירה -דירה 8.pdf"
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration (override with environment variables)
# --------------------------------------------------------------------------- #
CU_ENDPOINT = os.environ.get(
    "CU_ENDPOINT",
    "https://example.invalid/",
).rstrip("/") + "/"
CU_API_VERSION = os.environ.get("CU_API_VERSION", "2025-11-01")
ANALYZER_ID = os.environ.get("CU_ANALYZER_ID", "hebrewHandoverForm")

GPT_ENDPOINT = os.environ.get(
    "GPT_ENDPOINT",
    "https://example.invalid/openai/v1/",
).rstrip("/") + "/"
GPT_DEPLOYMENT = os.environ.get("GPT_DEPLOYMENT", "gpt-5.4")
# How many times to read each page with GPT and majority-vote the fields
# (self-consistency). Higher = more stable handwriting, more cost.
GPT_SAMPLES = int(os.environ.get("GPT_SAMPLES", "3"))
# Reasoning effort for GPT-5.x: low | medium | high
GPT_REASONING = os.environ.get("GPT_REASONING", "high")

SCOPE = "https://cognitiveservices.azure.com/.default"
RENDER_DPI = int(os.environ.get("RENDER_DPI", "300"))
# Image preprocessing to help handwriting OCR (grayscale + contrast + sharpen)
PREPROCESS = os.environ.get("PREPROCESS", "1") == "1"
OUT_DIR = Path(os.environ.get("OUT_DIR", "out"))

_credential = AzureCliCredential()
_token_cache: dict[str, tuple[float, str]] = {}


def bearer() -> str:
    """Return a cached AAD access token for Cognitive Services."""
    now = time.time()
    exp, tok = _token_cache.get(SCOPE, (0.0, ""))
    if now < exp - 60:
        return tok
    at = _credential.get_token(SCOPE)
    _token_cache[SCOPE] = (at.expires_on, at.token)
    return at.token


def auth_header() -> dict[str, str]:
    return {"Authorization": f"Bearer {bearer()}"}


# --------------------------------------------------------------------------- #
# The Hebrew field schema Content Understanding will extract.
# method="extract" -> copy value as printed; method="generate" -> let the model
# read/interpret (used for handwriting).
# --------------------------------------------------------------------------- #
FIELD_SCHEMA = {
    "name": "HebrewApartmentHandover",
    "description": "אישור מסירת/קבלת חזקה בדירה - טופס עברי מודפס עם מילוי בכתב יד",
    "fields": {
        "DocumentDate": {
            "type": "date",
            "method": "extract",
            "description": "התאריך הרשום בראש המסמך (שדה 'תאריך'), לרוב בכתב יד",
        },
        "BuildingNumber": {
            "type": "string",
            "method": "extract",
            "description": "מספר הבניין ('בבניין מס')",
        },
        "ApartmentNumber": {
            "type": "string",
            "method": "extract",
            "description": "מספר הדירה ('דירה מס')",
        },
        "Location": {
            "type": "string",
            "method": "extract",
            "description": "הכתובת / המיקום ('בקומה', 'ברחוב', 'בעיר')",
        },
        "RecipientName": {
            "type": "string",
            "method": "generate",
            "description": "שם מקבל/מוסר החזקה ('אני הח\"מ', 'נושא'), לרוב בכתב יד",
        },
        "RecipientId": {
            "type": "string",
            "method": "extract",
            "description": "מספר תעודת הזהות של מקבל/מוסר ('ת.ז מס')",
        },
        "HandwrittenNotes": {
            "type": "string",
            "method": "generate",
            "description": "כל ההערות שנכתבו בכתב יד בגוף הטופס (בסעיפים הממוספרים)",
        },
        "IsSigned": {
            "type": "boolean",
            "method": "generate",
            "description": "האם קיימת חתימה על המסמך",
        },
        "HasStamp": {
            "type": "boolean",
            "method": "generate",
            "description": "האם מופיעה חותמת על המסמך",
        },
    },
}


# --------------------------------------------------------------------------- #
# Content Understanding
# --------------------------------------------------------------------------- #
def cu_ensure_analyzer() -> None:
    """Create (or replace) the custom analyzer and wait until it is ready."""
    base = f"{CU_ENDPOINT}contentunderstanding/analyzers/{ANALYZER_ID}"
    body = {
        "description": "Hebrew apartment handover form (printed + handwritten)",
        "baseAnalyzerId": "prebuilt-document",
        "config": {
            "enableOcr": True,
            "enableLayout": True,
            "returnDetails": True,
            "estimateFieldSourceAndConfidence": True,
        },
        # Bind the completion/embedding models explicitly (field extraction needs
        # a completion model). These logical names resolve to the resource's
        # configured deployment defaults.
        "models": {"completion": "gpt-4.1", "embedding": "text-embedding-3-large"},
        "fieldSchema": FIELD_SCHEMA,
    }
    print(f"[CU] creating/replacing analyzer '{ANALYZER_ID}' ...")
    r = requests.put(
        f"{base}?api-version={CU_API_VERSION}&allowReplace=true",
        headers={**auth_header(), "Content-Type": "application/json"},
        data=json.dumps(body),
        timeout=60,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"[CU] analyzer create failed {r.status_code}: {r.text}")

    op_url = r.headers.get("Operation-Location")
    for _ in range(60):
        if op_url:
            g = requests.get(op_url, headers=auth_header(), timeout=30).json()
        else:
            g = requests.get(f"{base}?api-version={CU_API_VERSION}", headers=auth_header(), timeout=30).json()
        status = (g.get("status") or "").lower()
        if status in ("ready", "succeeded"):
            print("[CU] analyzer ready.")
            return
        if status == "failed":
            raise RuntimeError(f"[CU] analyzer creation failed: {json.dumps(g, ensure_ascii=False)}")
        time.sleep(2)
    raise TimeoutError("[CU] analyzer did not become ready in time")


def cu_analyze(pdf_bytes: bytes) -> dict:
    """Run the analyzer on the PDF and return the first content object."""
    data_b64 = base64.b64encode(pdf_bytes).decode()
    url = f"{CU_ENDPOINT}contentunderstanding/analyzers/{ANALYZER_ID}:analyze?api-version={CU_API_VERSION}"
    print("[CU] submitting document for analysis ...")
    r = requests.post(
        url,
        headers={**auth_header(), "Content-Type": "application/json"},
        data=json.dumps({"inputs": [{"data": data_b64, "mimeType": "application/pdf"}]}),
        timeout=120,
    )
    if r.status_code != 202:
        raise RuntimeError(f"[CU] analyze submit failed {r.status_code}: {r.text}")
    op_url = r.headers["Operation-Location"]

    for _ in range(90):
        g = requests.get(op_url, headers=auth_header(), timeout=30).json()
        status = (g.get("status") or "").lower()
        if status == "succeeded":
            print("[CU] analysis complete.")
            contents = g.get("result", {}).get("contents", [])
            return contents[0] if contents else {}
        if status == "failed":
            raise RuntimeError(f"[CU] analysis failed: {json.dumps(g)[:1000]}")
        time.sleep(2)
    raise TimeoutError("[CU] analysis timed out")


def _field_value(f: dict):
    """Pull the typed value out of a Content Understanding field object."""
    for k in (
        "valueString", "valueDate", "valueTime", "valueNumber",
        "valueInteger", "valueBoolean", "valueArray", "valueObject", "valueJson",
    ):
        if k in f:
            return f[k]
    return None


def cu_simplify_fields(content: dict) -> dict:
    out = {}
    for name, f in (content.get("fields") or {}).items():
        out[name] = {"value": _field_value(f), "confidence": f.get("confidence")}
    return out


# --------------------------------------------------------------------------- #
# GPT vision
# --------------------------------------------------------------------------- #
GPT_SYSTEM = (
    "You are an expert transcriber of Hebrew documents. You read both printed "
    "Hebrew text and Hebrew handwriting (כתב יד) with high accuracy, including "
    "dates and numbers. Read numbers and dates digit by digit. Use the printed "
    "form structure to understand what each handwritten blank should contain. "
    "Always answer with a single valid JSON object and nothing else."
)

GPT_USER = """התמונה היא סריקה של טופס עברי (אישור מסירת/קבלת חזקה בדירה).
הטופס מכיל טקסט מודפס וגם טקסט שמולא בכתב יד, וכן חתימות/חותמות ואזורים מושחרים (מוסתרים).

החזר אך ורק אובייקט JSON יחיד במבנה הבא:
{
  "printed_text": "כל הטקסט המודפס בטופס",
  "handwritten_text": "כל הטקסט שנכתב בכתב יד בלבד",
  "fields": {
    "document_date": "",
    "building_number": "",
    "apartment_number": "",
    "location": "",
    "recipient_name": "",
    "recipient_id": "",
    "handwritten_notes": "",
    "is_signed": false,
    "has_stamp": false
  },
  "uncertain": "מילים שלא זוהו בוודאות, אם יש"
}

הנחיות לזיהוי כתב היד:
- התבונן בכל שדה בנפרד. ליד כל תווית מודפסת (כמו 'תאריך', 'דירה מס', 'בבניין', 'ת.ז מס')
  נמצא הערך שנכתב בכתב יד — קרא אותו בהתאם למה שהתווית מצפה לו.
- מספרים ותאריכים: קרא ספרה-ספרה. תאריך בפורמט יום.חודש.שנה.
- אם מילה/ספרה אינה ברורה — נסה את הפירוש הסביר ביותר לפי ההקשר, וציין אותה גם ב-"uncertain".
- אזור מושחר/מוסתר סמן כ-"[מוסתר]".
- אל תמציא ערכים. שדה שלא קיים בטופס - השאר ריק.
"""


def gpt_vision(image_bytes: bytes, context_markdown: str = "") -> dict:
    b64 = base64.b64encode(image_bytes).decode()
    user_content = [{"type": "input_text", "text": GPT_USER}]
    if context_markdown.strip():
        user_content.append({
            "type": "input_text",
            "text": (
                "לעזר בלבד — כך זוהה הטקסט המודפס של הטופס על ידי מנוע OCR "
                "(השתמש בו כדי להבין את מבנה הטופס ומה אמור להופיע בכל שדה, "
                "אך קרא את כתב היד מהתמונה עצמה):\n\n" + context_markdown[:4000]
            ),
        })
    user_content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}"})

    payload = {
        "model": GPT_DEPLOYMENT,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": GPT_SYSTEM}]},
            {"role": "user", "content": user_content},
        ],
        "max_output_tokens": 6000,
        "reasoning": {"effort": GPT_REASONING},
    }
    r = requests.post(
        f"{GPT_ENDPOINT}responses",
        headers={**auth_header(), "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=240,
    )
    if r.status_code == 400 and "reasoning" in r.text.lower():
        # Model/deployment doesn't accept the reasoning param -> retry without it.
        payload.pop("reasoning", None)
        r = requests.post(
            f"{GPT_ENDPOINT}responses",
            headers={**auth_header(), "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=240,
        )
    if r.status_code != 200:
        raise RuntimeError(f"[GPT] request failed {r.status_code}: {r.text}")
    return _loads_lenient(_gpt_output_text(r.json()))


def _norm(v) -> str:
    return str(v).strip() if v is not None else ""


def _mode(values: list) -> object:
    """Most common non-empty value; None if none."""
    from collections import Counter
    norm = [_norm(v) for v in values if _norm(v) not in ("", "[מוסתר]")]
    if not norm:
        return None
    return Counter(norm).most_common(1)[0][0]


def gpt_vision_consistent(image_bytes: bytes, context_markdown: str = "") -> dict:
    """Read a page GPT_SAMPLES times and majority-vote each field.

    Returns the voted fields plus a per-field agreement score (0..1) and the
    raw samples for transparency.
    """
    samples = []
    for k in range(max(1, GPT_SAMPLES)):
        print(f"[GPT] reading page with {GPT_DEPLOYMENT} (sample {k + 1}/{GPT_SAMPLES}) ...")
        try:
            samples.append(gpt_vision(image_bytes, context_markdown))
        except Exception as exc:
            print(f"[GPT] sample {k + 1} failed: {exc}", file=sys.stderr)
    if not samples:
        return {"_error": "all GPT samples failed"}

    keys = ["document_date", "building_number", "apartment_number", "location",
            "recipient_name", "recipient_id", "handwritten_notes", "is_signed", "has_stamp"]
    voted, agreement = {}, {}
    for key in keys:
        vals = [s.get("fields", {}).get(key) for s in samples]
        if key in ("is_signed", "has_stamp"):
            trues = sum(1 for v in vals if v is True)
            voted[key] = trues * 2 >= len(vals)
            agreement[key] = max(trues, len(vals) - trues) / len(vals)
        else:
            top = _mode(vals)
            voted[key] = top
            if top is not None:
                agreement[key] = sum(1 for v in vals if _norm(v) == _norm(top)) / len(vals)
            else:
                agreement[key] = 0.0

    # Pick the handwritten_text / uncertain from the sample closest to the vote.
    def score(s):
        return sum(1 for key in keys if _norm(s.get("fields", {}).get(key)) == _norm(voted[key]))
    best = max(samples, key=score)

    return {
        "fields": voted,
        "field_agreement": agreement,
        "handwritten_text": best.get("handwritten_text", ""),
        "printed_text": best.get("printed_text", ""),
        "uncertain": best.get("uncertain", ""),
        "num_samples": len(samples),
        "samples": [s.get("fields", {}) for s in samples],
    }


def _gpt_output_text(resp: dict) -> str:
    # Prefer the aggregated helper if present.
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
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"_parse_error": str(e), "_raw": text}


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
# Maps Content Understanding field names <-> GPT vision field keys.
_FIELD_MAP = [
    ("Date",            "DocumentDate",     "document_date"),
    ("Building",        "BuildingNumber",   "building_number"),
    ("Apartment",       "ApartmentNumber",  "apartment_number"),
    ("Location",        "Location",         "location"),
    ("Recipient name",  "RecipientName",    "recipient_name"),
    ("Recipient ID",    "RecipientId",      "recipient_id"),
    ("Handwritten notes", "HandwrittenNotes", "handwritten_notes"),
    ("Signed",          "IsSigned",         "is_signed"),
    ("Stamp",           "HasStamp",         "has_stamp"),
]


def _cell(v) -> str:
    if v is None or v == "":
        return "—"
    return str(v).replace("\n", " ").replace("|", "\\|")


def build_markdown(result: dict) -> str:
    cu = result.get("content_understanding", {})
    cu_fields = cu.get("fields", {})
    gpt_pages = result.get("gpt_vision", [])
    gpt0 = gpt_pages[0].get("fields", {}) if gpt_pages else {}
    gpt0_agree = gpt_pages[0].get("field_agreement", {}) if gpt_pages else {}

    lines: list[str] = []
    lines.append(f"# Hebrew extraction — {Path(result.get('source_file', '')).name}\n")
    lines.append(
        "Combined output of **Azure AI Content Understanding** (printed text + fields) "
        "and **GPT vision** (handwriting).\n"
    )

    # --- Field comparison table --- #
    lines.append("## Extracted fields\n")
    lines.append("| Field | Content Understanding | Conf. | GPT vision (handwriting) | Agree |")
    lines.append("| --- | --- | --- | --- | --- |")
    for label, cu_key, gpt_key in _FIELD_MAP:
        cuf = cu_fields.get(cu_key, {})
        conf = cuf.get("confidence")
        conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
        agr = gpt0_agree.get(gpt_key)
        agr_s = f"{agr:.0%}" if isinstance(agr, (int, float)) else "—"
        lines.append(
            f"| {label} | {_cell(cuf.get('value'))} | {conf_s} "
            f"| {_cell(gpt0.get(gpt_key))} | {agr_s} |"
        )
    lines.append("")

    # --- Printed text (CU) --- #
    lines.append("## Printed text — Content Understanding OCR\n")
    lines.append(cu.get("markdown", "").strip() or "_(no printed text extracted)_")
    lines.append("")

    # --- Handwritten + full GPT transcription --- #
    lines.append("## Handwritten text — GPT vision\n")
    for p in gpt_pages:
        if len(gpt_pages) > 1:
            lines.append(f"### Page {p.get('page')}\n")
        hw = (p.get("handwritten_text") or "").strip()
        lines.append("**Handwritten:**\n")
        lines.append(hw or "_(none detected)_")
        lines.append("")
        if p.get("uncertain"):
            lines.append(f"**Uncertain:** {p['uncertain']}\n")
        printed = (p.get("printed_text") or "").strip()
        if printed:
            lines.append("<details><summary>GPT reading of the printed text</summary>\n")
            lines.append(printed)
            lines.append("\n</details>\n")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# PDF rendering
# --------------------------------------------------------------------------- #
def _preprocess(png_bytes: bytes) -> bytes:
    """Grayscale + autocontrast + sharpen to make handwriting clearer.

    Falls back to the original bytes if Pillow is unavailable.
    """
    if not PREPROCESS:
        return png_bytes
    try:
        import io
        from PIL import Image, ImageOps, ImageFilter

        img = Image.open(io.BytesIO(png_bytes)).convert("L")   # grayscale
        img = ImageOps.autocontrast(img, cutoff=1)             # stretch contrast
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:  # Pillow not installed / decode error
        print(f"[render] preprocess skipped: {exc}", file=sys.stderr)
        return png_bytes


def render_pages(pdf_path: Path) -> list[bytes]:
    doc = fitz.open(pdf_path)
    images = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=RENDER_DPI)
        png = _preprocess(pix.tobytes("png"))
        (OUT_DIR / f"page_{i + 1}.png").write_bytes(png)
        images.append(png)
    doc.close()
    return images


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample/אישור מסירה -דירה 8.pdf")
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}", file=sys.stderr)
        return 1

    pdf_bytes = pdf_path.read_bytes()
    page_images = render_pages(pdf_path)
    print(f"Loaded '{pdf_path.name}' ({len(page_images)} page(s)) @ {RENDER_DPI} dpi\n")

    # ---- Content Understanding -------------------------------------------- #
    cu_content = {}
    try:
        cu_ensure_analyzer()
        cu_content = cu_analyze(pdf_bytes)
    except Exception as exc:  # keep going so we still get GPT output
        print(f"[CU] ERROR: {exc}\n", file=sys.stderr)

    # ---- GPT vision (per page) -------------------------------------------- #
    cu_markdown = cu_content.get("markdown", "")
    gpt_pages = []
    if os.environ.get("SKIP_GPT") == "1":
        print("[GPT] skipped (SKIP_GPT=1)\n")
    else:
        for idx, img in enumerate(page_images, 1):
            try:
                gpt_pages.append({"page": idx, **gpt_vision_consistent(img, cu_markdown)})
            except Exception as exc:
                print(f"[GPT] ERROR on page {idx}: {exc}\n", file=sys.stderr)
                gpt_pages.append({"page": idx, "_error": str(exc)})

    # ---- Merge & save ----------------------------------------------------- #
    result = {
        "source_file": str(pdf_path),
        "content_understanding": {
            "markdown": cu_content.get("markdown", ""),
            "fields": cu_simplify_fields(cu_content),
        },
        "gpt_vision": gpt_pages,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "result.json"
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    md_file = OUT_DIR / "result.md"
    md_file.write_text(build_markdown(result), encoding="utf-8")

    # ---- Console summary -------------------------------------------------- #
    print("\n" + "=" * 70)
    print("CONTENT UNDERSTANDING - extracted fields")
    print("=" * 70)
    for name, v in result["content_understanding"]["fields"].items():
        conf = f"  (conf {v['confidence']:.2f})" if v.get("confidence") is not None else ""
        print(f"  {name:16}: {v['value']}{conf}")

    print("\n" + "=" * 70)
    print(f"GPT VISION - handwriting read ({GPT_DEPLOYMENT}, {GPT_SAMPLES}x vote)")
    print("=" * 70)
    for p in gpt_pages:
        f = p.get("fields", {})
        agr = p.get("field_agreement", {})

        def _a(key):
            a = agr.get(key)
            return f" [{a:.0%}]" if isinstance(a, (int, float)) else ""

        print(f"  page {p.get('page')}: "
              f"date={f.get('document_date')!r}{_a('document_date')} "
              f"apt={f.get('apartment_number')!r}{_a('apartment_number')} "
              f"bldg={f.get('building_number')!r}{_a('building_number')}")
        if f.get("recipient_name"):
            print(f"    recipient: {f['recipient_name']}{_a('recipient_name')}")
        if f.get("handwritten_notes"):
            print(f"    notes: {f['handwritten_notes']}")
        if p.get("uncertain"):
            print(f"    uncertain: {p['uncertain']}")

    print(f"\nFull structured output -> {out_file}")
    print(f"Readable Markdown       -> {md_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
