#!/usr/bin/env python3
"""Regression check for handwriting_pipeline.py against a GOLDEN baseline.

The handwriting OCR uses a vision LLM and is NON-DETERMINISTIC: the same code
produces slightly different words on each run. So this tool does NOT demand an
exact match. It:

  - matches regions by spatial position (box center), because labels and box
    coordinates drift between runs;
  - scores per-region text similarity (ignoring [?] confidence markers);
  - FAILS only on structural regressions: a region that went empty, a golden
    region with no match (dropped), or a large text divergence;
  - treats small word drift as expected noise (reported, not failed).

Usage:
  python3 compare_golden.py                 # compare output_handwriting/* vs golden/*
  python3 compare_golden.py --run           # re-run the 3 samples first, then compare
  python3 compare_golden.py --update-golden # snapshot current output as the new golden
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

# The three sample documents that make up the golden dataset.
SAMPLES = [
    {"stem": "אישור מסירה -דירה 8", "input": "sample/אישור מסירה -דירה 8.pdf", "args": []},
    {"stem": "חוזר מכר - דף ראשון", "input": "sample/חוזר מכר - דף ראשון.png", "args": []},
    {"stem": "פרוטוקול משולב ידני ומודפס  דוגמה",
     "input": "sample/פרוטוקול משולב ידני ומודפס  דוגמה.pdf", "args": []},
]

OUTPUT_DIR = Path("output_handwriting")
GOLDEN_DIR = Path("golden")

# Similarity thresholds (on [?]-stripped text).
OK_RATIO = 0.85       # >= this: effectively the same reading
DIVERGE_RATIO = 0.60  # < this: flagged as a real divergence (structural fail)


# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    """Normalize for comparison: drop [?] markers, bracket/slash alternatives,
    and collapse whitespace so run-to-run formatting noise is ignored."""
    text = (text or "").replace("[?]", " ")
    text = re.sub(r"[\[\]/]", " ", text)   # alternative markers like [a/b]
    return " ".join(text.split())


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _center(box: list[int]) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def _load_regions(result_json: Path) -> list[dict]:
    data = json.loads(result_json.read_text(encoding="utf-8"))
    return data.get("regions", [])


def _match(golden: list[dict], current: list[dict]):
    """Greedy nearest-center matching. Returns (pairs, extras) where pairs is a
    list of (golden_region, current_region_or_None) and extras are unmatched
    current regions."""
    used: set[int] = set()
    pairs: list[tuple[dict, dict | None]] = []
    for g in golden:
        gc = _center(g["box"])
        best_d, best_i = None, None
        for ci, c in enumerate(current):
            if ci in used:
                continue
            cc = _center(c["box"])
            d = (gc[0] - cc[0]) ** 2 + (gc[1] - cc[1]) ** 2
            if best_d is None or d < best_d:
                best_d, best_i = d, ci
        if best_i is not None:
            used.add(best_i)
            pairs.append((g, current[best_i]))
        else:
            pairs.append((g, None))
    extras = [c for ci, c in enumerate(current) if ci not in used]
    return pairs, extras


def _status(golden_text: str, current_text: str | None) -> tuple[str, float, bool]:
    """Return (label, ratio, is_fail)."""
    g = (golden_text or "").strip()
    if current_text is None:
        return ("MISSING", 0.0, True)
    c = (current_text or "").strip()
    if g and not _norm(c):
        return ("EMPTY", 0.0, True)
    r = _ratio(g, c)
    if r >= OK_RATIO:
        return ("OK", r, False)
    if r >= DIVERGE_RATIO:
        return ("DRIFT", r, False)          # within model variance -> warn only
    return ("DIVERGED", r, True)            # real divergence -> fail


def compare_sample(stem: str) -> dict:
    gpath = GOLDEN_DIR / stem / "handwriting_result.json"
    cpath = OUTPUT_DIR / stem / "handwriting_result.json"
    if not gpath.exists():
        return {"stem": stem, "error": f"no golden ({gpath})"}
    if not cpath.exists():
        return {"stem": stem, "error": f"no current output ({cpath})"}

    golden = _load_regions(gpath)
    current = _load_regions(cpath)
    pairs, extras = _match(golden, current)

    rows, ratios, fails = [], [], 0
    for g, c in pairs:
        gt = g.get("final_transcription") or ""
        ct = None if c is None else (c.get("final_transcription") or "")
        label, r, is_fail = _status(gt, ct)
        fails += int(is_fail)
        if label not in ("MISSING",):
            ratios.append(r)
        rows.append({"golden_label": g.get("label", ""), "golden": gt,
                     "current": ct, "status": label, "ratio": r})

    count_fail = len(current) != len(golden)
    avg = sum(ratios) / len(ratios) if ratios else 0.0
    # merged.md similarity (secondary signal, never a hard fail)
    gmd = (GOLDEN_DIR / stem / "merged.md")
    cmd = (OUTPUT_DIR / stem / "merged.md")
    md_ratio = (_ratio(gmd.read_text(encoding="utf-8"), cmd.read_text(encoding="utf-8"))
                if gmd.exists() and cmd.exists() else None)

    verdict = "PASS" if fails == 0 else "FAIL"
    return {
        "stem": stem, "rows": rows, "extras": extras,
        "golden_count": len(golden), "current_count": len(current),
        "count_fail": count_fail, "avg_ratio": avg, "md_ratio": md_ratio,
        "fails": fails, "verdict": verdict,
    }


def _print_report(res: dict) -> None:
    stem = res["stem"]
    print(f"\n{'=' * 78}\n{stem}\n{'=' * 78}")
    if res.get("error"):
        print(f"  ERROR: {res['error']}")
        return
    print(f"regions: golden={res['golden_count']} current={res['current_count']}"
          + ("   [COUNT CHANGED]" if res["count_fail"] else ""))
    for i, row in enumerate(res["rows"], 1):
        mark = {"OK": "ok  ", "DRIFT": "drift", "DIVERGED": "FAIL", "EMPTY": "FAIL", "MISSING": "FAIL"}
        print(f"  [{i}] {mark.get(row['status'], '?')} {row['status']:9} "
              f"sim={row['ratio']:.2f}  ({row['golden_label']})")
        print(f"        golden : {row['golden'][:90]!r}")
        print(f"        current: {(row['current'] or '')[:90]!r}")
    for ex in res["extras"]:
        print(f"  [+] EXTRA region not in golden: {ex.get('label','')!r} -> "
              f"{(ex.get('final_transcription') or '')[:60]!r}")
    md = f"{res['md_ratio']:.2f}" if res["md_ratio"] is not None else "n/a"
    print(f"  avg region similarity: {res['avg_ratio']:.2f}   merged.md similarity: {md}")
    print(f"  VERDICT: {res['verdict']}"
          + (f"  ({res['fails']} structural regression(s))" if res["fails"] else ""))


def _run_samples() -> None:
    for s in SAMPLES:
        out = OUTPUT_DIR / s["stem"]
        cmd = [sys.executable, "-u", "handwriting_pipeline.py",
               "--input", s["input"], "--auto-detect", "--merge-md",
               "--ocr-mode", "full", "--workers", "10", "--output", str(out), *s["args"]]
        print(f"[run] {s['stem']} ...")
        subprocess.run(cmd, check=False)


def _update_golden() -> None:
    for s in SAMPLES:
        stem = s["stem"]
        (GOLDEN_DIR / stem).mkdir(parents=True, exist_ok=True)
        for name in ("handwriting_result.json", "merged.md"):
            src = OUTPUT_DIR / stem / name
            if src.exists():
                (GOLDEN_DIR / stem / name).write_text(
                    src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"golden updated <- {stem}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Regression check vs golden baseline.")
    ap.add_argument("--run", action="store_true", help="re-run the 3 samples before comparing")
    ap.add_argument("--update-golden", action="store_true", help="snapshot current output as golden")
    args = ap.parse_args()

    if args.update_golden:
        _update_golden()
        return 0
    if args.run:
        _run_samples()

    results = [compare_sample(s["stem"]) for s in SAMPLES]
    for res in results:
        _print_report(res)

    failed = [r for r in results if r.get("verdict") == "FAIL" or r.get("error")]
    print(f"\n{'=' * 78}")
    print(f"OVERALL: {len(results) - len(failed)}/{len(results)} passed"
          + (f"   FAILED: {[r['stem'] for r in failed]}" if failed else "   (no structural regressions)"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
