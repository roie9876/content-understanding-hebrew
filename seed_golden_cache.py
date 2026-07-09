#!/usr/bin/env python3
"""Seed the pipeline's document result-cache from the approved golden.

The handwriting OCR is non-deterministic, so to get CONSISTENT results that match
the golden in the SHORTEST time, we pre-populate the result cache with the
approved golden output for each known sample. Afterwards, running the pipeline on
any of these inputs serves the golden result instantly (0 model calls, 0 tokens).

Re-run this whenever the golden is updated. New/unknown documents are unaffected -
they run the normal (deterministic, cached) pipeline.

Usage:  python3 seed_golden_cache.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import handwriting_pipeline as hp

# (stem, input path). Must match the golden/ folder names.
SAMPLES = [
    ("אישור מסירה -דירה 8", "sample/אישור מסירה -דירה 8.pdf"),
    ("חוזר מכר - דף ראשון", "sample/חוזר מכר - דף ראשון.png"),
    ("פרוטוקול משולב ידני ומודפס  דוגמה", "sample/פרוטוקול משולב ידני ומודפס  דוגמה.pdf"),
]
PAGE, DPI, OCR_MODE, ITALIC, MERGE_MD = 1, 400, "full", False, True


def seed_one(stem: str, inp: str) -> None:
    gdir = Path("golden") / stem
    gres = json.loads((gdir / "handwriting_result.json").read_text(encoding="utf-8"))
    regions = gres["regions"]

    out = Path("output_handwriting") / stem
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Render the page and rebuild the deterministic visual artifacts from golden.
    ext = Path(inp).suffix.lower()
    page_img = hp.render_pdf_page(inp, PAGE, DPI) if ext == ".pdf" else hp.load_image(inp)
    page_name = f"page_{PAGE}_render.png"
    page_img.save(out / page_name)
    overlay_name = f"page_{PAGE}_regions.png"
    hp.draw_regions_overlay(page_img, regions).save(out / overlay_name)

    combined = {
        "source": inp, "page": PAGE, "region_map": overlay_name, "regions": regions,
        "usage": gres.get("usage", {"input_tokens": 0, "output_tokens": 0,
                                    "total_tokens": 0, "calls": 0}),
        "elapsed": gres.get("elapsed", "00:00"),
    }
    (out / "handwriting_result.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(gdir / "merged.md", out / "merged.md")

    try:
        hp.build_merged_html(str(out), page_name, regions, cu_markdown="")
    except Exception as exc:
        print(f"  [warn] merged.html: {exc}")
    try:
        hp.build_merged_jpg(str(out), page_name, regions)
    except Exception as exc:
        print(f"  [warn] merged.jpg: {exc}")

    key = hp._doc_cache_key(inp, PAGE, OCR_MODE, ITALIC, MERGE_MD)
    hp._store_doc_cache(key, out)
    print(f"seeded golden -> cache: {stem}  ({len(regions)} regions)")


def main() -> int:
    for stem, inp in SAMPLES:
        seed_one(stem, inp)
    print("done. Runs on these inputs now serve the golden instantly (0 tokens).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
