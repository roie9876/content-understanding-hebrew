#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
CORE_VIEWS = (
    "01_original_crop.png",
    "02_upscaled_contrast.png",
    "04_denoised.png",
    "06_optional_line_removed.png",
)
FORBIDDEN_PARTS = ("golden", "handwriting_result.json", "merged.md", "result.json")


def build_request(source_dir: Path, output_dir: Path, reasoning: str,
                  concurrency: int, timeout: int, limit: int | None = None) -> dict:
    detected = json.loads((source_dir / "detected_regions.json").read_text(encoding="utf-8"))
    regions = []
    for raw_region in detected["regions"][:limit]:
        index = int(raw_region["region"])
        candidates = sorted(source_dir.glob(f"region_{index:02d}_*"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one crop directory for region {index}, found {len(candidates)}")
        views = [{"name": name, "path": str(candidates[0] / name)} for name in CORE_VIEWS]
        missing = [view["path"] for view in views if not Path(view["path"]).exists()]
        if missing:
            raise ValueError(f"Missing crop views for region {index}: {missing}")
        regions.append({"region_index": index, "box": raw_region["box"], "views": views})
    request = {
        "version": 1,
        "page": int(detected.get("page", 1)),
        "endpoint": os.environ.get("GPT_SOL_ENDPOINT", ""),
        "deployment": os.environ.get("GPT_SOL_DEPLOYMENT", "gpt-5.6-sol"),
        "reasoning": reasoning,
        "concurrency": concurrency,
        "timeout_seconds": timeout,
        "output_dir": str(output_dir),
        "regions": regions,
    }
    assert_sealed(request)
    return request


def assert_sealed(request: dict) -> None:
    serialized = json.dumps(request, ensure_ascii=False).lower()
    leaked = [part for part in FORBIDDEN_PARTS if part in serialized]
    if leaked:
        raise ValueError(f"Crop-agent request contains forbidden inference inputs: {', '.join(leaked)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one sealed Copilot SDK Sol agent per crop.")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--reasoning", choices=("low", "medium", "high", "xhigh"), default="xhigh")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "agent_benchmarks" / (
        f"{args.source_run.name}__crop-agent-{args.reasoning}__"
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    output.mkdir(parents=True, exist_ok=False)
    try:
        request = build_request(
            args.source_run.resolve(), output.resolve(), args.reasoning,
            max(1, args.concurrency), args.timeout, args.limit,
        )
        request_path = output / "sealed_request.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        completed = subprocess.run(
            ["node", str(ROOT / "copilot_crop_agent_runner.mjs"), str(request_path)],
            cwd=ROOT,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return completed.returncode
    except Exception as exc:
        print(f"crop-agent benchmark failed: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())