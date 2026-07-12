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
import functools
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


def _is_stable(text: str) -> bool:
    """A golden reading is 'stable' (reproducible enough to ENFORCE) when it is
    short, unmarked ([?]-free), single-line, and dominated by digits/ASCII
    (dates, numbers, IDs, emails). Long/cursive Hebrew notes are 'volatile' - the
    vision model rewords them run-to-run, so they are reported but never fail.
    """
    t = (text or "").strip()
    if not t or "[?]" in t or "\n" in t:
        return False
    core = t.replace(" ", "")
    if not core or len(core) > 20:
        return False
    asciiish = sum(ch.isascii() for ch in core)
    return asciiish >= max(1, int(0.7 * len(core)))


def _load_regions(result_json: Path) -> list[dict]:
    data = json.loads(result_json.read_text(encoding="utf-8"))
    return data.get("regions", [])


def _match(golden: list[dict], current: list[dict]):
    """Globally minimize center distance so overlapping boxes cannot cross-match."""
    if not golden or not current:
        return [(region, None) for region in golden], list(current)

    costs = []
    for golden_region in golden:
        gx, gy = _center(golden_region["box"])
        costs.append([
            (gx - cx) ** 2 + (gy - cy) ** 2
            for cx, cy in (_center(current_region["box"]) for current_region in current)
        ])

    @functools.lru_cache(maxsize=None)
    def assign(golden_index: int, used_mask: int):
        if golden_index == len(golden):
            return 0.0, ()
        choices = []
        for current_index in range(len(current)):
            if used_mask & (1 << current_index):
                continue
            tail_cost, tail = assign(golden_index + 1, used_mask | (1 << current_index))
            choices.append((costs[golden_index][current_index] + tail_cost,
                            (current_index,) + tail))
        if not choices:
            return float("inf"), ()
        return min(choices, key=lambda choice: choice[0])

    matched_count = min(len(golden), len(current))
    if len(golden) <= len(current) and len(current) <= 15:
        _cost, assignment = assign(0, 0)
    else:
        assignment = tuple(range(matched_count))
    used = set(assignment)
    pairs = [
        (region, current[assignment[index]] if index < len(assignment) else None)
        for index, region in enumerate(golden)
    ]
    extras = [region for index, region in enumerate(current) if index not in used]
    return pairs, extras


def _status(golden_text: str, current_text: str | None, stable: bool) -> tuple[str, float, bool]:
    """Return (label, ratio, is_fail). Structural problems (missing/empty) always
    fail. Text divergence fails ONLY on stable fields; on volatile fields it is
    reported but not failed, because the model is non-deterministic there."""
    g = (golden_text or "").strip()
    if current_text is None:
        return ("MISSING", 0.0, True)
    c = (current_text or "").strip()
    if g and not _norm(c):
        return ("EMPTY", 0.0, True)
    r = _ratio(g, c)
    if r >= OK_RATIO:
        return ("OK", r, False)
    if not stable:
        return ("VOLATILE-DIFF", r, False)   # expected model variance -> not a fail
    if r >= DIVERGE_RATIO:
        return ("DRIFT", r, False)           # minor drift on a stable field -> warn
    return ("DIVERGED", r, True)             # stable field genuinely changed -> fail


def compare_sample(stem: str, current_result: Path | None = None) -> dict:
    gpath = GOLDEN_DIR / stem / "handwriting_result.json"
    cpath = current_result or OUTPUT_DIR / stem / "handwriting_result.json"
    if not gpath.exists():
        return {"stem": stem, "error": f"no golden ({gpath})"}
    if not cpath.exists():
        return {"stem": stem, "error": f"no current output ({cpath})"}

    golden = _load_regions(gpath)
    cur_data = json.loads(cpath.read_text(encoding="utf-8"))
    current = cur_data.get("regions", [])
    pairs, extras = _match(golden, current)

    rows, ratios, fails = [], [], 0
    for g, c in pairs:
        gt = g.get("final_transcription") or ""
        ct = None if c is None else (c.get("final_transcription") or "")
        stable = _is_stable(gt)
        label, r, is_fail = _status(gt, ct, stable)
        fails += int(is_fail)
        if label != "MISSING":
            ratios.append(r)
        rows.append({"golden_label": g.get("label", ""), "golden": gt,
                     "current": ct, "status": label, "ratio": r, "stable": stable})

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
        "usage": cur_data.get("usage"), "elapsed": cur_data.get("elapsed"),
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
        mark = {"OK": "ok  ", "DRIFT": "warn", "VOLATILE-DIFF": "info",
                "DIVERGED": "FAIL", "EMPTY": "FAIL", "MISSING": "FAIL"}
        tag = "stable" if row.get("stable") else "volatile"
        print(f"  [{i}] {mark.get(row['status'], '?'):4} {row['status']:13} "
              f"sim={row['ratio']:.2f}  [{tag}] ({row['golden_label']})")
        print(f"        golden : {row['golden'][:90]!r}")
        print(f"        current: {(row['current'] or '')[:90]!r}")
    for ex in res["extras"]:
        print(f"  [+] EXTRA region not in golden: {ex.get('label','')!r} -> "
              f"{(ex.get('final_transcription') or '')[:60]!r}")
    md = f"{res['md_ratio']:.2f}" if res["md_ratio"] is not None else "n/a"
    print(f"  avg region similarity: {res['avg_ratio']:.2f}   merged.md similarity: {md}")
    u = res.get("usage") or {}
    if u or res.get("elapsed"):
        tok = (f"{u.get('total_tokens', 0):,} tokens "
               f"({u.get('input_tokens', 0):,} in / {u.get('output_tokens', 0):,} out, "
               f"{u.get('calls', 0)} calls)") if u else "n/a"
        print(f"  time: {res.get('elapsed', 'n/a')}   tokens: {tok}")
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
    ap.add_argument("--result", type=Path, help="score one saved handwriting_result.json")
    ap.add_argument("--golden-stem", choices=[sample["stem"] for sample in SAMPLES],
                    help="golden sample used with --result")
    ap.add_argument("--json", action="store_true", help="print machine-readable report")
    args = ap.parse_args()

    if args.update_golden:
        _update_golden()
        return 0
    if args.run:
        _run_samples()

    if args.result:
        if not args.golden_stem:
            ap.error("--result requires --golden-stem")
        result = compare_sample(args.golden_stem, args.result)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_report(result)
        return 1 if result.get("verdict") == "FAIL" or result.get("error") else 0

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
