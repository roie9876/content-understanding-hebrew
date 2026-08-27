#!/usr/bin/env python3
"""Benchmark a Kraken recognition model on existing handwriting region crops."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import subprocess
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import compare_golden


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / ".models" / "pp-ocrv6-small" / "small.safetensors"
DEFAULT_OUTPUT_ROOT = ROOT / "analysis_results"


def discover_regions(input_dir: Path, view: str) -> list[tuple[int, Path]]:
    regions = []
    for image_path in input_dir.glob(f"region_*/{view}"):
        match = re.match(r"region_(\d+)(?:_|$)", image_path.parent.name)
        if match:
            regions.append((int(match.group(1)), image_path))
    return sorted(regions)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"[^\w\u0590-\u05ff]", "", text).lower()


def similarity(reference: str, prediction: str) -> float:
    return difflib.SequenceMatcher(
        None, normalize_text(reference), normalize_text(prediction)
    ).ratio()


def load_regions(result_path: Path) -> list[dict]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return payload.get("regions", [])


def reference_by_region(
    input_result: Path | None, golden_result: Path | None
) -> dict[int, dict]:
    if not input_result or not golden_result:
        return {}
    current = load_regions(input_result)
    golden = load_regions(golden_result)
    pairs, _ = compare_golden._match(golden, current)
    current_indexes = {id(region): index + 1 for index, region in enumerate(current)}
    return {
        current_indexes[id(current_region)]: golden_region
        for golden_region, current_region in pairs
        if current_region is not None
    }


def recognize(
    kraken: Path, model: Path, image_path: Path, output_path: Path,
    device: str,
) -> tuple[str, float, str | None]:
    started = time.monotonic()
    command = [
        str(kraken),
        "--device",
        device,
        "--raise-on-error",
        "--native",
        "-i",
        str(image_path),
        str(output_path),
        "ocr",
        "--no-segmentation",
        "--base-dir",
        "R",
        "-m",
        str(model),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return "", elapsed, detail[-2000:] or f"Kraken exited {completed.returncode}"
    return output_path.read_text(encoding="utf-8").strip(), elapsed, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--kraken", type=Path, default=Path("kraken"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--view", default="01_original_crop.png")
    parser.add_argument("--input-result", type=Path)
    parser.add_argument("--golden-result", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, help="Run only the first N regions")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    model = args.model.resolve()
    model_name = model.stem
    if not model.is_file():
        parser.error(f"model not found: {model}")
    regions = discover_regions(input_dir, args.view)
    if args.limit is not None:
        regions = regions[:max(0, args.limit)]
    if not regions:
        parser.error(f"no region_*/{args.view} files under {input_dir}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / f"{input_dir.name}__kraken-{model_name}__{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    references = reference_by_region(args.input_result, args.golden_result)

    results = []
    for region_id, image_path in regions:
        text_path = output_dir / f"region_{region_id:02d}.txt"
        text, elapsed, error = recognize(
            args.kraken, model, image_path, text_path, args.device,
        )
        reference = references.get(region_id, {})
        reference_text = reference.get("final_transcription", "")
        row = {
            "region": region_id,
            "image": str(image_path.relative_to(ROOT)),
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "prediction": text,
            "elapsed_seconds": round(elapsed, 3),
            "error": error,
        }
        if reference:
            row.update({
                "reference": reference_text,
                "similarity": round(similarity(reference_text, text), 4),
                "exact": normalize_text(reference_text) == normalize_text(text),
            })
        results.append(row)
        state = f"error: {error}" if error else repr(text)
        print(f"region {region_id:02d} ({elapsed:.2f}s): {state}", flush=True)

    scored = [row for row in results if "similarity" in row]
    summary = {
        "regions": len(results),
        "successful": sum(row["error"] is None for row in results),
        "errors": sum(row["error"] is not None for row in results),
        "scored": len(scored),
        "exact": sum(row.get("exact", False) for row in scored),
        "average_similarity": round(
            sum(row["similarity"] for row in scored) / len(scored), 4
        ) if scored else None,
        "elapsed_seconds": round(sum(row["elapsed_seconds"] for row in results), 3),
    }
    report = {
        "engine": "kraken",
        "model": model_name,
        "model_path": str(model),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "device": args.device,
        "text_direction": "horizontal-rl",
        "segmentation": "whole-line",
        "view": args.view,
        "input_dir": str(input_dir),
        "summary": summary,
        "results": results,
    }
    report_path = output_dir / "benchmark.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {report_path}")
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())