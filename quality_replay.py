#!/usr/bin/env python3
"""Replay current deterministic quality rules against saved OCR runs.

Golden answers are loaded only after postprocessing and are used solely for
measurement. No model calls are made.

Usage:
  python3 quality_replay.py \
    --run 'חיוב חשבון1=analysis_results/חיוב חשבון1.pdf__2026-07-15_09-58-19' \
    --run 'דוח_יוסי=analysis_results/דוח_יוסי.pdf__2026-07-15_11-27-12'
"""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

from PIL import Image

import compare_golden
import handwriting_pipeline as pipeline


def _quality_ratio(expected: str, actual: str) -> float:
    """Match the strict punctuation- and layout-free quality experiment metric."""
    return difflib.SequenceMatcher(
        None,
        compare_golden._exact_norm(expected),
        compare_golden._exact_norm(actual),
    ).ratio()


def apply_current_rules(regions: list[dict], cu_markdown: str) -> list[dict]:
    """Apply the production deterministic postprocessing order to saved regions."""
    finals = []
    for region in regions:
        final = pipeline.separate_numeric_field_marks(dict(region))
        final = pipeline.normalize_numeric_s_homoglyph(final)
        finals.append(final)
    finals = pipeline.reconcile_repeated_dates(finals, regions)
    finals = pipeline.reconcile_with_cu_numeric_tokens(finals, regions, cu_markdown)
    finals = pipeline.reconcile_repeated_structured_fields(finals, regions)
    return pipeline.reconcile_uncertain_repeated_numbers(
        finals, regions, cu_markdown
    )


def finalize_saved_classification(final: dict, region: dict, run_dir: Path) -> dict:
    """Reapply production signature geometry when the saved crop is available."""
    candidate = dict(final)
    crop_path = run_dir / str(region.get("folder") or "") / "01_original_crop.png"
    if not crop_path.exists():
        return candidate
    with Image.open(crop_path) as crop:
        signature_flourish = (
            pipeline.is_signature_flourish(crop)
            or pipeline.is_black_signature_flourish(crop)
        )
    return pipeline.finalize_region_semantics(
        candidate, [], signature_flourish=signature_flourish
    )


def emitted_saved_regions(
    original: list[dict], processed: list[dict], run_dir: Path,
) -> list[dict]:
    """Replay output filtering only when the saved blind evidence is available."""
    emitted = []
    for region, final in zip(original, processed):
        result_path = run_dir / str(region.get("folder") or "") / "result.json"
        if not result_path.exists():
            emitted.append(final)
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if "per_image" not in result:
            emitted.append(final)
            continue
        if pipeline.should_emit_region(final, result.get("per_image") or []):
            emitted.append(final)
    return emitted


def score_saved_run(stem: str, run_dir: Path) -> dict:
    result_path = run_dir / "handwriting_result.json"
    golden_path = compare_golden.GOLDEN_DIR / stem / "handwriting_result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"saved result not found: {result_path}")
    if not golden_path.exists():
        raise FileNotFoundError(f"golden not found: {golden_path}")

    original = json.loads(result_path.read_text(encoding="utf-8")).get("regions", [])
    cu_path = run_dir / "content_understanding.md"
    cu_markdown = cu_path.read_text(encoding="utf-8") if cu_path.exists() else ""
    replay_regions = original
    page_path = run_dir / "page_1_render.png"
    if page_path.exists():
        with Image.open(page_path) as page_image:
            replay_regions = pipeline.deduplicate_detection_regions(
                original, page_image.width, page_image.height
            )
    processed = emitted_saved_regions(
        replay_regions,
        apply_current_rules(replay_regions, cu_markdown),
        run_dir,
    )
    golden = json.loads(golden_path.read_text(encoding="utf-8")).get("regions", [])
    before_pairs, before_extras = compare_golden._match(golden, original)
    after_pairs, after_extras = compare_golden._match(golden, processed)

    metrics = {
        "text_total": 0,
        "before_exact": 0,
        "after_exact": 0,
        "before_similarity_sum": 0.0,
        "after_similarity_sum": 0.0,
        "classification_total": 0,
        "before_classification_correct": 0,
        "after_classification_correct": 0,
    }
    for (golden_region, before), (_, after) in zip(before_pairs, after_pairs):
        expected_classification = golden_region.get("human_classification")
        if expected_classification:
            metrics["classification_total"] += 1
            metrics["before_classification_correct"] += int(
                (before or {}).get("classification") == expected_classification
            )
            finalized = finalize_saved_classification(
                after or {}, before or {}, run_dir
            )
            metrics["after_classification_correct"] += int(
                finalized.get("classification") == expected_classification
            )
            continue

        golden_text = golden_region.get("final_transcription") or ""
        before_text = (before or {}).get("final_transcription") or ""
        after_text = (after or {}).get("final_transcription") or ""
        metrics["text_total"] += 1
        metrics["before_exact"] += int(
            compare_golden._exact_norm(golden_text)
            == compare_golden._exact_norm(before_text)
        )
        metrics["after_exact"] += int(
            compare_golden._exact_norm(golden_text)
            == compare_golden._exact_norm(after_text)
        )
        metrics["before_similarity_sum"] += _quality_ratio(
            golden_text, before_text
        )
        metrics["after_similarity_sum"] += _quality_ratio(
            golden_text, after_text
        )

    metrics.update({
        "stem": stem,
        "run_dir": str(run_dir),
        "golden_regions": len(golden),
        "saved_regions": len(original),
        "processed_regions": len(processed),
        "before_extras": len(before_extras),
        "after_extras": len(after_extras),
    })
    return metrics


def aggregate(results: list[dict]) -> dict:
    totals = {
        key: sum(result[key] for result in results)
        for key in (
            "text_total",
            "before_exact",
            "after_exact",
            "before_similarity_sum",
            "after_similarity_sum",
            "classification_total",
            "before_classification_correct",
            "after_classification_correct",
            "golden_regions",
            "saved_regions",
            "processed_regions",
            "before_extras",
            "after_extras",
        )
    }
    text_total = totals["text_total"]
    totals["before_similarity"] = (
        totals["before_similarity_sum"] / text_total if text_total else 0.0
    )
    totals["after_similarity"] = (
        totals["after_similarity_sum"] / text_total if text_total else 0.0
    )
    totals["documents"] = len(results)
    return totals


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected STEM=RUN_DIR")
    stem, raw_path = value.split("=", 1)
    if not stem.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected non-empty STEM=RUN_DIR")
    return stem.strip(), Path(raw_path).expanduser()


def _summary(metrics: dict) -> dict:
    text_total = metrics["text_total"]
    classification_total = metrics["classification_total"]
    return {
        "exact": (
            f"{metrics['before_exact']}->{metrics['after_exact']}/{text_total}"
        ),
        "similarity": (
            f"{metrics['before_similarity_sum'] / text_total if text_total else 0.0:.3f}"
            f"->{metrics['after_similarity_sum'] / text_total if text_total else 0.0:.3f}"
        ),
        "classification": (
            f"{metrics['before_classification_correct']}"
            f"->{metrics['after_classification_correct']}/{classification_total}"
        ),
        "regions": (
            f"{metrics['saved_regions']}->{metrics['processed_regions']}"
            f"/{metrics['golden_regions']}"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay deterministic OCR quality rules on saved runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run,
        required=True,
        metavar="STEM=RUN_DIR",
        help="golden stem and saved run directory; repeat for multiple documents",
    )
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()

    results = [score_saved_run(stem, run_dir) for stem, run_dir in args.run]
    totals = aggregate(results)
    report = {
        "documents": [
            {"stem": result["stem"], **_summary(result)} for result in results
        ],
        "total": _summary(totals),
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for document in report["documents"]:
            print(
                f"{document['stem']}: exact {document['exact']}; "
                f"similarity {document['similarity']}; "
                f"classification {document['classification']}; "
                f"regions {document['regions']}"
            )
        total = report["total"]
        print(
            f"TOTAL: exact {total['exact']}; similarity {total['similarity']}; "
            f"classification {total['classification']}; regions {total['regions']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())