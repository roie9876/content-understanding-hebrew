#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


def _center(box: list[int]) -> tuple[float, float]:
    x, y, width, height = box
    return x + width / 2, y + height / 2


def _minimum_cost_assignment(costs: list[list[float]]) -> tuple[int, ...]:
    row_count = len(costs)
    column_count = len(costs[0]) if costs else 0
    if not row_count:
        return ()
    if row_count > column_count:
        raise ValueError("consensus runs must contain at least as many regions as the anchor")
    row_potentials = [0.0] * (row_count + 1)
    column_potentials = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        column = 0
        minimums = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[column] = True
            active_row = matched_row[column]
            delta = float("inf")
            next_column = 0
            for candidate in range(1, column_count + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[active_row - 1][candidate - 1]
                    - row_potentials[active_row]
                    - column_potentials[candidate]
                )
                if reduced < minimums[candidate]:
                    minimums[candidate] = reduced
                    predecessor[candidate] = column
                if minimums[candidate] < delta:
                    delta = minimums[candidate]
                    next_column = candidate
            for candidate in range(column_count + 1):
                if used[candidate]:
                    row_potentials[matched_row[candidate]] += delta
                    column_potentials[candidate] -= delta
                else:
                    minimums[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break
    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column]:
            assignment[matched_row[column] - 1] = column - 1
    return tuple(assignment)


def _align(anchor: list[dict], current: list[dict]) -> list[dict]:
    costs = []
    for anchor_region in anchor:
        anchor_x, anchor_y = _center(anchor_region["box"])
        costs.append([
            (anchor_x - current_x) ** 2 + (anchor_y - current_y) ** 2
            for current_x, current_y in (_center(region["box"]) for region in current)
        ])
    assignment = _minimum_cost_assignment(costs)
    return [current[index] for index in assignment]


def _vote_key(region: dict) -> tuple[str, str]:
    classification = region.get("classification", "handwritten_text")
    text = re.sub(
        r"[^\w\u0590-\u05ff]",
        "",
        (region.get("final_transcription") or "").replace("[?]", ""),
    ).lower()
    return classification, text


def _confidence(region: dict) -> float:
    try:
        return float(region.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def build_consensus(results: list[dict]) -> dict:
    if len(results) < 3:
        raise ValueError("Sol consensus requires at least three independent runs")
    anchor = results[0].get("regions", [])
    aligned_runs = [anchor] + [_align(anchor, result.get("regions", [])) for result in results[1:]]
    final_regions = []
    for index, candidates in enumerate(zip(*aligned_runs), 1):
        key_counts = Counter(_vote_key(candidate) for candidate in candidates)
        winning_key, winning_count = key_counts.most_common(1)[0]
        eligible = [candidate for candidate in candidates if _vote_key(candidate) == winning_key]
        if winning_count < 2:
            eligible = list(candidates)
        selected = max(eligible, key=_confidence)
        final_region = dict(selected)
        final_region["box"] = list(anchor[index - 1]["box"])
        final_region["consensus"] = {
            "method": "majority" if winning_count >= 2 else "confidence_fallback",
            "agreement": winning_count,
            "run_count": len(candidates),
            "candidates": [{
                "classification": candidate.get("classification", "handwritten_text"),
                "transcription": candidate.get("final_transcription", ""),
                "confidence": _confidence(candidate),
            } for candidate in candidates],
        }
        final_regions.append(final_region)
    return {
        "page": results[0].get("page", 1),
        "method": "gpt-5.6-sol-three-run-consensus",
        "golden_used_during_inference": False,
        "regions": final_regions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a golden-free consensus from Sol runs.")
    parser.add_argument("results", nargs="*", type=Path)
    parser.add_argument("--input", type=Path, help="run fresh Sol analyses for this document")
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result_paths = list(args.results)
    if args.input:
        if result_paths:
            parser.error("use either result paths or --input, not both")
        if args.runs < 3:
            parser.error("--runs must be at least 3")
        from sealed_hybrid_benchmark import _run_pipeline
        run_dir = Path("agent_benchmarks") / (
            f"{args.input.stem}__sol-consensus__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        for run_index in range(1, args.runs + 1):
            result_dir = run_dir / f"run_{run_index}"
            _run_pipeline(args.input.resolve(), args.page, result_dir, "gpt-5.6-sol", args.workers)
            result_paths.append(result_dir / "handwriting_result.json")
        output = args.output or run_dir / "handwriting_result.json"
    else:
        if len(result_paths) < 3:
            parser.error("provide at least three result paths or use --input")
        if not args.output:
            parser.error("result paths require --output")
        output = args.output
    data = [json.loads(result.read_text(encoding="utf-8")) for result in result_paths]
    consensus = build_consensus(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(consensus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())