#!/usr/bin/env python3
"""Stability probe: single-draw vs self-consistency vote (cache OFF).

For an EASY field (date) and a HARD field (cursive note), run M independent
"sessions". Each session draws K fresh OCR samples (distinct seeds) and
aggregates them with the deterministic voter. We then measure how stable the
VOTE result is across sessions vs the single-draw chaos measured before.
"""
import itertools
import json
import difflib
from concurrent.futures import ThreadPoolExecutor

import handwriting_pipeline as hp

hp.CACHE_ENABLED = False  # genuine calls, no replay

INP = "sample/אישור מסירה -דירה 8.pdf"
K, M = 5, 3  # draws per session, independent sessions

page = hp.render_pdf_page(INP, 1, 400)
golden = json.load(open("golden/אישור מסירה -דירה 8/handwriting_result.json", encoding="utf-8"))
regions = golden["regions"]
prompt = hp.build_ocr_prompt("", italic=False)

fields = [("EASY (date)", regions[0]), ("HARD (cursive note)", regions[7])]


def _crop(reg):
    x, y, w, h = reg["box"]
    return hp.enhance_upscaled_contrast(hp.crop_region(page, x, y, w, h), scale=4)


crops = {label: _crop(reg) for label, reg in fields}
tasks = [(label, s, k, 1000 * (s + 1) + k)
         for label, _ in fields for s in range(M) for k in range(K)]


def _do(t):
    label, s, k, seed = t
    return t, (hp.ocr_image(crops[label], prompt, seed=seed) or {}).get("full_transcription", "")


reads_by = {}
with ThreadPoolExecutor(max_workers=12) as ex:
    for t, txt in ex.map(_do, tasks):
        reads_by[(t[0], t[1], t[2])] = txt

for label, reg in fields:
    print(f"\n=== {label}  golden={reg['final_transcription']!r} ===")
    session_aggs, all_singles = [], []
    for s in range(M):
        reads = [reads_by[(label, s, k)] for k in range(K)]
        all_singles += reads
        agg = hp.aggregate_votes(reads)["final_transcription"]
        session_aggs.append(agg)
        print(f"  session {s + 1} vote -> {agg!r}")
    sims = [difflib.SequenceMatcher(None, a, b).ratio()
            for a, b in itertools.combinations(session_aggs, 2)]
    print(f"  single draws: {len(set(all_singles))} distinct of {len(all_singles)}")
    if sims:
        print(f"  VOTE cross-session: {len(set(session_aggs))} distinct of {M}, "
              f"avg sim {sum(sims) / len(sims):.2f}")
