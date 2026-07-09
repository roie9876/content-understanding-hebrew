#!/usr/bin/env python3
"""Stability probe on אישור (free-text heavy): single-draw vs vote, all regions.

For every golden region, run M independent sessions of K fresh OCR draws (cache
OFF), aggregate each session with the deterministic voter, and report how stable
the VOTE is across sessions vs the single-draw variance.
"""
import itertools
import json
import difflib
from concurrent.futures import ThreadPoolExecutor

import handwriting_pipeline as hp

hp.CACHE_ENABLED = False  # genuine calls, no replay

INP = "sample/אישור מסירה -דירה 8.pdf"
K, M = 4, 3  # draws per session, independent sessions

page = hp.render_pdf_page(INP, 1, 400)
golden = json.load(open("golden/אישור מסירה -דירה 8/handwriting_result.json", encoding="utf-8"))
regions = golden["regions"]
prompt = hp.build_ocr_prompt("", italic=False)

crops = {}
for reg in regions:
    x, y, w, h = reg["box"]
    crops[reg["region"]] = hp.enhance_upscaled_contrast(hp.crop_region(page, x, y, w, h), scale=4)

tasks = [(reg["region"], s, k, 1000 * (s + 1) + k)
         for reg in regions for s in range(M) for k in range(K)]


def _do(t):
    rid, s, k, seed = t
    return t, (hp.ocr_image(crops[rid], prompt, seed=seed) or {}).get("full_transcription", "")


reads_by = {}
with ThreadPoolExecutor(max_workers=12) as ex:
    for t, txt in ex.map(_do, tasks):
        reads_by[(t[0], t[1], t[2])] = txt

print(f"\n{'region':38} {'single':>7} {'vote':>6} {'sim':>5}   example vote")
print("-" * 100)
for reg in regions:
    rid = reg["region"]
    session_aggs, singles = [], []
    for s in range(M):
        reads = [reads_by[(rid, s, k)] for k in range(K)]
        singles += reads
        session_aggs.append(hp.aggregate_votes(reads)["final_transcription"])
    sims = [difflib.SequenceMatcher(None, a, b).ratio()
            for a, b in itertools.combinations(session_aggs, 2)]
    avg = sum(sims) / len(sims) if sims else 1.0
    lbl = (reg.get("label", "") or "")[:36]
    print(f"{lbl:38} {len(set(singles))}/{len(singles):<5} {len(set(session_aggs))}/{M:<3} "
          f"{avg:5.2f}   {session_aggs[0]!r}")
