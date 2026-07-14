#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import fitz
from dotenv import load_dotenv

import compare_golden

load_dotenv()

ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = ROOT / "agent_benchmarks"
DEFAULT_PROFILES = ("sol-low", "sol-medium", "sol-high", "gpt-5.4-medium")


def profile_matrix() -> dict[str, dict[str, str]]:
    primary_endpoint = os.environ.get("GPT_ENDPOINT", "")
    primary_deployment = os.environ.get("GPT_DEPLOYMENT", "gpt-5.4")
    sol_endpoint = os.environ.get("GPT_SOL_ENDPOINT", "")
    sol_deployment = os.environ.get("GPT_SOL_DEPLOYMENT", "gpt-5.6-sol")
    profiles = {}
    for reasoning in ("low", "medium", "high"):
        profiles[f"sol-{reasoning}"] = {
            "endpoint": sol_endpoint,
            "deployment": sol_deployment,
            "reasoning": reasoning,
        }
        profiles[f"gpt-5.4-{reasoning}"] = {
            "endpoint": primary_endpoint,
            "deployment": primary_deployment,
            "reasoning": reasoning,
        }
    return profiles


def _render_page(input_path: Path, page_number: int, output_path: Path, dpi: int) -> tuple[int, int]:
    if input_path.suffix.lower() == ".pdf":
        with fitz.open(input_path) as document:
            if page_number < 1 or page_number > len(document):
                raise ValueError(f"Page {page_number} is outside this {len(document)}-page document")
            pixmap = document[page_number - 1].get_pixmap(dpi=dpi, alpha=False)
            pixmap.save(output_path)
            return pixmap.width, pixmap.height
    if page_number != 1:
        raise ValueError("Image inputs only have page 1")
    with fitz.open(input_path) as document:
        page = document[0]
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        pixmap.save(output_path)
        return pixmap.width, pixmap.height


def _score(stem: str, result_path: Path) -> dict:
    if not (compare_golden.GOLDEN_DIR / stem / "handwriting_result.json").exists():
        return {"available": False, "reason": f"No human golden exists for {stem}"}
    comparison = compare_golden.compare_sample(stem, result_path)
    comparison["available"] = not bool(comparison.get("error"))
    return comparison


def build_summary(run_dir: Path, manifest: dict) -> dict:
    rows = []
    for profile in manifest["profiles"]:
        profile_dir = run_dir / profile
        result_path = profile_dir / "handwriting_result.json"
        score_path = profile_dir / "score.json"
        score = json.loads(score_path.read_text(encoding="utf-8")) if score_path.exists() else {}
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        strict = score.get("strict") or {}
        rows.append({
            "profile": profile,
            "status": "complete" if result else "failed",
            "regions": len(result.get("regions", [])),
            "exact": strict.get("text_exact"),
            "total": strict.get("text_total"),
            "text_accuracy": strict.get("text_accuracy"),
            "classification_accuracy": strict.get("classification_accuracy"),
            "average_similarity": score.get("avg_ratio"),
            "elapsed_seconds": result.get("elapsed_seconds"),
        })
    ranked = sorted(
        rows,
        key=lambda row: (
            row["text_accuracy"] if row["text_accuracy"] is not None else -1,
            row["average_similarity"] if row["average_similarity"] is not None else -1,
            -(row["elapsed_seconds"] or float("inf")),
        ),
        reverse=True,
    )
    return {"run": manifest, "results": rows, "ranking": [row["profile"] for row in ranked]}


def write_report(run_dir: Path, summary: dict) -> None:
    lines = [
        f"# Copilot agent benchmark: {summary['run']['document_name']}",
        "",
        f"Page: {summary['run']['page']}  ",
        f"Created: {summary['run']['created_at']}  ",
        "Golden data was used only after inference for scoring.",
        "",
        "| Rank | Profile | Status | Regions | Exact | Similarity | Classification | Time |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    by_profile = {row["profile"]: row for row in summary["results"]}
    for rank, profile in enumerate(summary["ranking"], 1):
        row = by_profile[profile]
        exact = "n/a" if row["exact"] is None else f"{row['exact']}/{row['total']} ({row['text_accuracy']:.1%})"
        similarity = "n/a" if row["average_similarity"] is None else f"{row['average_similarity']:.1%}"
        classification = "n/a" if row["classification_accuracy"] is None else f"{row['classification_accuracy']:.1%}"
        elapsed = "n/a" if row["elapsed_seconds"] is None else f"{row['elapsed_seconds']:.1f}s"
        lines.append(
            f"| {rank} | {profile} | {row['status']} | {row['regions']} | {exact} | "
            f"{similarity} | {classification} | {elapsed} |"
        )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(input_path: Path, page: int, profiles: list[str], dpi: int,
                  timeout_seconds: int, output_dir: Path | None = None) -> Path:
    matrix = profile_matrix()
    unknown = [profile for profile in profiles if profile not in matrix]
    if unknown:
        raise ValueError(f"Unknown profiles: {', '.join(unknown)}")
    missing = [profile for profile in profiles if not matrix[profile]["endpoint"]]
    if missing:
        raise ValueError(f"Missing endpoint configuration for: {', '.join(missing)}")

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_name = f"{input_path.stem}__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = output_dir or BENCHMARK_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    source_path = run_dir / "source.png"
    width, height = _render_page(input_path, page, source_path, dpi)
    manifest = {
        "version": 1,
        "document_name": input_path.name,
        "document_stem": input_path.stem,
        "page": page,
        "dpi": dpi,
        "image_size": [width, height],
        "profiles": profiles,
        "created_at": created_at,
        "golden_used_during_inference": False,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for profile in profiles:
        profile_dir = run_dir / profile
        profile_dir.mkdir()
        config = {
            "profile": profile,
            "document_name": input_path.name,
            "page": page,
            "image_path": str(source_path),
            "image_width": width,
            "image_height": height,
            "output_dir": str(profile_dir),
            "timeout_seconds": timeout_seconds,
            **matrix[profile],
        }
        config_path = profile_dir / "request.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        started = time.time()
        with (profile_dir / "runner.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                ["node", str(ROOT / "copilot_agent_benchmark.mjs"), str(config_path)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds + 360,
                check=False,
            )
        status = {
            "return_code": completed.returncode,
            "wall_seconds": round(time.time() - started, 1),
        }
        (profile_dir / "status.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
        result_path = profile_dir / "handwriting_result.json"
        if result_path.exists():
            score = _score(input_path.stem, result_path)
            (profile_dir / "score.json").write_text(
                json.dumps(score, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    summary = build_summary(run_dir, manifest)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(run_dir, summary)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark full Copilot agents on a document page.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES),
                        help="Comma-separated agent profiles")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--timeout", type=int, default=900, help="Seconds per agent run")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list-profiles", action="store_true")
    args = parser.parse_args()
    if args.list_profiles:
        print("\n".join(profile_matrix()))
        return 0
    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    profiles = [profile.strip() for profile in args.profiles.split(",") if profile.strip()]
    try:
        run_dir = run_benchmark(args.input.resolve(), args.page, profiles, args.dpi,
                                args.timeout, args.output)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"agent benchmark failed: {exc}", file=sys.stderr)
        return 1
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())