#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "agent_benchmarks"
CANDIDATE_PROFILES = ("gpt-5.4", "gpt-5.6-sol")


def _model_environment(profile: str) -> dict[str, str]:
    environment = os.environ.copy()
    if profile == "gpt-5.4":
        environment.update(
            GPT_ENDPOINT=os.environ.get("GPT_ENDPOINT", ""),
            GPT_DEPLOYMENT=os.environ.get("GPT_DEPLOYMENT", "gpt-5.4"),
            GPT_TARGETS=os.environ.get("GPT_TARGETS", ""),
            GPT_REASONING="medium",
            OCR_REASONING="medium",
        )
    elif profile == "gpt-5.6-sol":
        environment.update(
            GPT_ENDPOINT=os.environ.get("GPT_SOL_ENDPOINT", ""),
            GPT_DEPLOYMENT=os.environ.get("GPT_SOL_DEPLOYMENT", "gpt-5.6-sol"),
            GPT_TARGETS="",
            GPT_REASONING="low",
            OCR_REASONING="low",
        )
    else:
        raise ValueError(f"Unsupported sealed candidate profile: {profile}")
    return environment


def _run_pipeline(input_path: Path, page: int, output_dir: Path, profile: str,
                  workers: int) -> dict:
    output_dir.mkdir(parents=True)
    command = [
        sys.executable, "-u", str(ROOT / "handwriting_pipeline.py"),
        "--input", str(input_path),
        "--page", str(page),
        "--workers", str(workers),
        "--output", str(output_dir),
    ]
    started = time.time()
    with (output_dir / "pipeline.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=_model_environment(profile),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    status = {"return_code": completed.returncode, "wall_seconds": round(time.time() - started, 1)}
    (output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    if completed.returncode != 0 or not (output_dir / "handwriting_result.json").exists():
        raise RuntimeError(f"{profile} pipeline failed; inspect {output_dir / 'pipeline.log'}")
    return status


def _candidate(region: dict | None) -> dict | None:
    if region is None:
        return None
    return {
        "transcription": region.get("final_transcription", ""),
        "classification": region.get("classification", "handwritten_text"),
        "confidence": region.get("confidence"),
    }


def _center(box: list[int]) -> tuple[float, float]:
    x, y, width, height = box
    return x + width / 2, y + height / 2


def _nearest_regions(anchor_regions: list[dict], other_regions: list[dict]) -> list[dict | None]:
    available = set(range(len(other_regions)))
    matches = []
    for anchor in anchor_regions:
        if not available:
            matches.append(None)
            continue
        x, y = _center(anchor["box"])
        selected = min(
            available,
            key=lambda index: sum(
                (left - right) ** 2
                for left, right in zip((x, y), _center(other_regions[index]["box"]))
            ),
        )
        available.remove(selected)
        matches.append(other_regions[selected])
    return matches


def _make_evidence(source_path: Path, regions: list[dict], evidence_dir: Path) -> None:
    evidence_dir.mkdir()
    with Image.open(source_path) as source:
        source = source.convert("RGB")
        for region in regions:
            x, y, width, height = region["box"]
            margin_x = max(24, round(width * 0.2))
            margin_y = max(24, round(height * 0.5))
            crop = source.crop((
                max(0, x - margin_x),
                max(0, y - margin_y),
                min(source.width, x + width + margin_x),
                min(source.height, y + height + margin_y),
            ))
            scale = min(4.0, 1800 / max(crop.width, 1))
            if scale > 1:
                crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS)
            crop = ImageOps.expand(crop, border=8, fill="white")
            crop.save(evidence_dir / f"region_{region['region_index']:02d}.png")


def _sealed_request(run_dir: Path, gpt54: dict, sol: dict, timeout: int,
                    reasoning: str) -> dict:
    anchor_regions = gpt54.get("regions", [])
    matches = _nearest_regions(anchor_regions, sol.get("regions", []))
    evidence_dir = run_dir / "evidence"
    source_path = run_dir / "candidates" / "gpt54" / f"page_{gpt54.get('page', 1)}_render.png"
    regions = []
    for index, (anchor, other) in enumerate(zip(anchor_regions, matches), 1):
        regions.append({
            "region_index": index,
            "box": anchor["box"],
            "neutral_label": f"handwriting region {index}",
            "primary": _candidate(anchor),
            "secondary": _candidate(other),
            "evidence_path": str(evidence_dir / f"region_{index:02d}.png"),
        })
    _make_evidence(source_path, regions, evidence_dir)
    return {
        "version": 1,
        "endpoint": os.environ.get("GPT_ENDPOINT", ""),
        "deployment": os.environ.get("GPT_DEPLOYMENT", "gpt-5.4"),
        "reasoning": reasoning,
        "timeout_seconds": timeout,
        "output_dir": str(run_dir),
        "regions": regions,
    }


def _assert_sealed(request_data: dict) -> None:
    serialized = json.dumps(request_data, ensure_ascii=False).lower()
    forbidden = ("golden", "human", "expected", "ground_truth", "ground truth")
    leaked = [term for term in forbidden if term in serialized]
    if leaked:
        raise RuntimeError(f"sealed request contains forbidden evaluation terms: {', '.join(leaked)}")


def _build_result(request_data: dict, adjudication: dict, run_dir: Path) -> Path:
    decisions = {region["region_index"]: region for region in adjudication["regions"]}
    final_regions = []
    for region in request_data["regions"]:
        decision = decisions[region["region_index"]]
        final_regions.append({
            "box": region["box"],
            "label": region["neutral_label"],
            "classification": decision["classification"],
            "final_transcription": decision["final_transcription"],
            "confidence": decision["confidence"],
            "adjudication_reason": decision["reason"],
            "candidate_evidence": {
                "primary_gpt54": region["primary"],
                "secondary_sol": region["secondary"],
            },
        })
    result_path = run_dir / "handwriting_result.json"
    result_path.write_text(json.dumps({
        "page": 1,
        "method": "sealed_hybrid_gpt54_sol",
        "golden_used_during_inference": False,
        "regions": final_regions,
        "elapsed_seconds": adjudication.get("elapsed_seconds"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result_path


def _write_evaluation_report(run_dir: Path, scores: dict[str, dict]) -> None:
    rows = []
    for name, score in scores.items():
        strict = score["strict"]
        rows.append({
            "name": name,
            "exact": strict["text_exact"],
            "total": strict["text_total"],
            "text_accuracy": strict["text_accuracy"],
            "similarity": score["avg_ratio"],
            "classification_accuracy": strict["classification_accuracy"],
        })
    ranked = sorted(
        rows,
        key=lambda row: (row["text_accuracy"], row["similarity"]),
        reverse=True,
    )
    summary = {
        "evaluation_only": True,
        "golden_used_during_inference": False,
        "ranking": [row["name"] for row in ranked],
        "results": rows,
    }
    (run_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Sealed OpenAI-only evaluation",
        "",
        "Golden data was loaded only after all candidate and adjudicator processes exited.",
        "Scores are evaluation outputs and never alter the saved inference results.",
        "",
        "| Rank | Output | Exact | Similarity | Classification |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {row['name']} | {row['exact']}/{row['total']} "
            f"({row['text_accuracy']:.1%}) | {row['similarity']:.1%} | "
            f"{row['classification_accuracy']:.1%} |"
        )
    (run_dir / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(input_path: Path, page: int, workers: int, timeout: int, reasoning: str) -> Path:
    run_dir = RUN_ROOT / f"{input_path.stem}__sealed-hybrid__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    candidate_dir = run_dir / "candidates"
    run_dir.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            "gpt54": executor.submit(
                _run_pipeline, input_path, page, candidate_dir / "gpt54",
                CANDIDATE_PROFILES[0], workers,
            ),
            "sol": executor.submit(
                _run_pipeline, input_path, page, candidate_dir / "sol",
                CANDIDATE_PROFILES[1], workers,
            ),
        }
        statuses = {name: future.result() for name, future in futures.items()}

    gpt54 = json.loads((candidate_dir / "gpt54" / "handwriting_result.json").read_text(encoding="utf-8"))
    sol = json.loads((candidate_dir / "sol" / "handwriting_result.json").read_text(encoding="utf-8"))
    request_data = _sealed_request(run_dir, gpt54, sol, timeout, reasoning)
    _assert_sealed(request_data)
    request_path = run_dir / "sealed_request.json"
    request_path.write_text(json.dumps(request_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "adjudicator.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            ["node", str(ROOT / "copilot_hybrid_adjudicator.mjs"), str(request_path)],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout + 360,
            check=False,
        )
    if completed.returncode != 0 or not (run_dir / "adjudication.json").exists():
        raise RuntimeError(f"adjudication failed; inspect {run_dir / 'adjudicator.log'}")
    adjudication = json.loads((run_dir / "adjudication.json").read_text(encoding="utf-8"))
    result_path = _build_result(request_data, adjudication, run_dir)

    # Evaluation is deliberately imported only after every inference process has exited.
    import compare_golden
    scores = {
        "gpt-5.4 staged": compare_golden.compare_sample(
            input_path.stem, candidate_dir / "gpt54" / "handwriting_result.json"
        ),
        "gpt-5.6-sol staged": compare_golden.compare_sample(
            input_path.stem, candidate_dir / "sol" / "handwriting_result.json"
        ),
        "sealed adjudication": compare_golden.compare_sample(input_path.stem, result_path),
    }
    score = scores["sealed adjudication"]
    (run_dir / "score.json").write_text(
        json.dumps(score, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, candidate_score in scores.items():
        safe_name = name.replace(" ", "_")
        (run_dir / f"score_{safe_name}.json").write_text(
            json.dumps(candidate_score, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    _write_evaluation_report(run_dir, scores)
    (run_dir / "manifest.json").write_text(json.dumps({
        "input": input_path.name,
        "page": page,
        "candidate_statuses": statuses,
        "reasoning": reasoning,
        "golden_used_during_inference": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a sealed GPT-5.4 + GPT-5.6 Sol benchmark.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--reasoning", choices=("low", "medium", "high"), default="high")
    args = parser.parse_args()
    try:
        run_dir = run(args.input.resolve(), args.page, args.workers, args.timeout, args.reasoning)
    except Exception as exc:
        print(f"sealed hybrid failed: {exc}", file=sys.stderr)
        return 1
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())