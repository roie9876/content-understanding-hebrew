#!/usr/bin/env python3
"""Benchmark Azure Document Intelligence (prebuilt-read) on the sample handwriting.

Runs the deterministic Azure OCR engine on each golden region crop (twice, to
confirm it is deterministic), and scores similarity to the golden. Prints text so
we can compare accuracy vs the GPT self-consistency vote.
"""
import base64
import difflib
import io
import json
import time
from pathlib import Path

import requests

import handwriting_pipeline as hp

DI_ENDPOINT = "https://iai-docintelegent.cognitiveservices.azure.com/"
API_VERSIONS = ["2024-11-30", "2023-07-31"]

SAMPLES = [
    ("אישור מסירה -דירה 8", "sample/אישור מסירה -דירה 8.pdf"),
    ("חוזר מכר - דף ראשון", "sample/חוזר מכר - דף ראשון.png"),
    ("פרוטוקול משולב ידני ומודפס  דוגמה", "sample/פרוטוקול משולב ידני ומודפס  דוגמה.pdf"),
]


def _png(img) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def di_read(png_bytes: bytes):
    """Return (content_text, api_version) from Document Intelligence prebuilt-read."""
    b64 = base64.b64encode(png_bytes).decode()
    for ver in API_VERSIONS:
        seg = "documentintelligence" if ver >= "2024" else "formrecognizer"
        url = f"{DI_ENDPOINT}{seg}/documentModels/prebuilt-read:analyze?api-version={ver}"
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {hp._bearer()}",
                                            "Content-Type": "application/json"},
                              json={"base64Source": b64}, timeout=60)
        except Exception as exc:
            print(f"    [di] {ver} error: {exc}")
            continue
        if r.status_code != 202:
            print(f"    [di] {ver} -> HTTP {r.status_code}: {r.text[:120]}")
            continue
        op = r.headers["Operation-Location"]
        for _ in range(40):
            time.sleep(2)
            g = requests.get(op, headers={"Authorization": f"Bearer {hp._bearer()}"},
                             timeout=30).json()
            st = (g.get("status") or "").lower()
            if st == "succeeded":
                return (g.get("analyzeResult", {}).get("content", "") or "").strip(), ver
            if st == "failed":
                print(f"    [di] {ver} analysis failed")
                break
    return None, None


def _sim(a: str, b: str) -> float:
    n = lambda t: " ".join((t or "").replace("[?]", " ").split())
    return difflib.SequenceMatcher(None, n(a), n(b)).ratio()


for stem, inp in SAMPLES:
    print(f"\n{'=' * 78}\n{stem}\n{'=' * 78}")
    ext = Path(inp).suffix.lower()
    page = hp.render_pdf_page(inp, 1, 400) if ext == ".pdf" else hp.load_image(inp)
    golden = json.load(open(f"golden/{stem}/handwriting_result.json", encoding="utf-8"))
    for reg in golden["regions"]:
        x, y, w, h = reg["box"]
        crop = hp.crop_region(page, x, y, w, h)
        png = _png(crop)
        t1, ver = di_read(png)
        t2, _ = di_read(png)
        det = "DET" if (t1 or "") == (t2 or "") else "NON-DET"
        gold = reg["final_transcription"]
        print(f"\n  [{reg['region']}] {reg.get('label','')}")
        print(f"     golden: {gold!r}")
        print(f"     azure : {t1!r}  ({ver}, {det}, sim={_sim(t1 or '', gold):.2f})")
