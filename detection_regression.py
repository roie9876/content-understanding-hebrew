#!/usr/bin/env python3
"""Detection-only regression benchmark against local human golden geometry."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from PIL import Image

import compare_golden
import handwriting_pipeline as pipeline


def _text_targets(golden: dict) -> list[dict]:
    return [
        region for region in golden.get("regions", [])
        if not region.get("human_classification")
        and (
            region.get("detection_target") is True
            or str(region.get("final_transcription") or "").strip()
        )
    ]


def _center_distance(
    first: list[int], second: list[int], page_size: tuple[int, int],
) -> float:
    first_center = compare_golden._center(first)
    second_center = compare_golden._center(second)
    diagonal = math.hypot(*page_size)
    return math.dist(first_center, second_center) / max(1.0, diagonal)


def _overlap_metrics(
    expected: list[int], actual: list[int],
) -> tuple[float, float, float]:
    ex, ey, ew, eh = expected
    ax, ay, aw, ah = actual
    intersection_width = max(0, min(ex + ew, ax + aw) - max(ex, ax))
    intersection_height = max(0, min(ey + eh, ay + ah) - max(ey, ay))
    intersection = intersection_width * intersection_height
    union = ew * eh + aw * ah - intersection
    return (
        intersection / max(1, union),
        intersection / max(1, ew * eh),
        intersection / max(1, aw * ah),
    )


def _spatial_match(
    expected: list[int], actual: list[int], page_size: tuple[int, int],
    maximum_center_distance: float,
) -> tuple[bool, float, float, float, float]:
    center_distance = _center_distance(expected, actual, page_size)
    iou, target_coverage, detection_coverage = _overlap_metrics(expected, actual)
    matched = (
        (
            center_distance <= maximum_center_distance
            and (
                target_coverage >= 0.08
                or detection_coverage >= 0.35
                or iou >= 0.08
            )
        )
        or (
            detection_coverage >= 0.80
            and target_coverage >= 0.05
        )
    )
    return matched, center_distance, iou, target_coverage, detection_coverage


def score_detection(
    golden: dict, detected: list[dict], page_size: tuple[int, int],
    maximum_center_distance: float = 0.12,
) -> dict:
    targets = _text_targets(golden)
    rows = []
    matched = 0
    supported_detections: set[int] = set()
    for expected in targets:
        candidates = []
        for detection_index, actual in enumerate(detected):
            metrics = _spatial_match(
                expected["box"], actual["box"], page_size,
                maximum_center_distance,
            )
            accepted, center_distance, iou, target_coverage, detection_coverage = metrics
            if accepted:
                supported_detections.add(detection_index)
            support = max(iou, target_coverage, detection_coverage)
            candidates.append((
                not accepted, -support, center_distance,
                detection_index, actual, metrics,
            ))
        best = min(candidates) if candidates else None
        actual = None if best is None else best[4]
        center_distance = None
        iou = 0.0
        target_coverage = 0.0
        detection_coverage = 0.0
        accepted = False
        if best is not None:
            accepted, center_distance, iou, target_coverage, detection_coverage = best[5]
            matched += int(accepted)
        rows.append({
            "expected_label": expected.get("label", ""),
            "expected_box": expected["box"],
            "actual_label": None if actual is None else actual.get("label", ""),
            "actual_box": None if actual is None else actual["box"],
            "center_distance": center_distance,
            "alignment_distance": (
                0.0
                if accepted
                and center_distance is not None
                and center_distance > maximum_center_distance
                and detection_coverage >= 0.80
                and target_coverage >= 0.05
                else center_distance
            ),
            "iou": iou,
            "target_coverage": target_coverage,
            "detection_coverage": detection_coverage,
            "matched": accepted,
        })
    target_count = len(targets)
    extras = [
        region for index, region in enumerate(detected)
        if index not in supported_detections
    ]
    allowed_extras = sum(
        1 for region in golden.get("regions", []) if region not in targets
    ) + 1
    recall = matched / target_count if target_count else 1.0
    accepted_rows = [row for row in rows if row["matched"]]
    average_center_distance = (
        sum(row["alignment_distance"] for row in accepted_rows)
        / len(accepted_rows)
        if accepted_rows else 1.0
    )
    passed = (
        matched == target_count
        and len(extras) <= allowed_extras
        and average_center_distance <= 0.07
    )
    return {
        "target_count": target_count,
        "detected_count": len(detected),
        "matched_count": matched,
        "recall": recall,
        "extra_count": len(extras),
        "allowed_extras": allowed_extras,
        "average_center_distance": average_center_distance,
        "pass": passed,
        "rows": rows,
        "extras": extras,
    }


def _load_page(sample: dict, golden: dict) -> Image.Image:
    source = Path(sample["input"])
    page_number = int(golden.get("page", 1))
    if source.suffix.lower() == ".pdf":
        return pipeline.render_pdf_page(str(source), page_number, 400).convert("RGB")
    return pipeline.load_image(str(source))


def _selected_samples(stems: list[str] | None) -> list[dict]:
    selected = []
    known_stems: set[str] = set()
    for sample in compare_golden.SAMPLES:
        if stems and sample["stem"] not in stems:
            continue
        golden_folder = compare_golden.GOLDEN_DIR / sample["stem"]
        if (
            (golden_folder / "detection_regions.json").exists()
            or (golden_folder / "handwriting_result.json").exists()
        ) and Path(sample["input"]).exists():
            selected.append(sample)
            known_stems.add(sample["stem"])
    for golden_folder in compare_golden.GOLDEN_DIR.iterdir():
        stem = golden_folder.name
        if stem in known_stems or (stems and stem not in stems):
            continue
        if not (golden_folder / "detection_regions.json").exists():
            continue
        source = Path("sample") / f"{stem}.pdf"
        if not source.exists():
            source = Path("sample") / f"{stem}.png"
        if source.exists():
            selected.append({"stem": stem, "input": str(source), "args": []})
    return selected


def _golden_path(stem: str) -> Path:
    folder = compare_golden.GOLDEN_DIR / stem
    detection_path = folder / "detection_regions.json"
    return detection_path if detection_path.exists() else folder / "handwriting_result.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rerun handwriting detection only and compare with human golden boxes.",
    )
    parser.add_argument("--stem", action="append", dest="stems")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or Path("analysis_results") / (
        "detection_regression__" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for sample in _selected_samples(args.stems):
        stem = sample["stem"]
        golden_path = _golden_path(stem)
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        page = _load_page(sample, golden)
        try:
            detected = pipeline.detect_handwriting_regions(page)
        except RuntimeError as exc:
            report = {"stem": stem, "pass": False, "error": str(exc)}
        else:
            report = {
                "stem": stem,
                **score_detection(golden, detected, page.size),
            }
            page_dir = output / stem
            page_dir.mkdir(parents=True, exist_ok=True)
            pipeline.draw_regions_overlay(page, detected).save(
                page_dir / "detected_regions.png",
            )
            (page_dir / "detected_regions.json").write_text(
                json.dumps({"regions": detected}, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
        reports.append(report)
        if report.get("error"):
            print(f"FAIL {stem}: {report['error']}")
        else:
            print(
                f"{'PASS' if report['pass'] else 'FAIL'} {stem}: "
                f"recall={report['matched_count']}/{report['target_count']} "
                f"extras={report['extra_count']}/{report['allowed_extras']} "
                f"center={report['average_center_distance']:.3f}"
            )
    suite = {"pass": bool(reports) and all(report["pass"] for report in reports),
             "reports": reports}
    (output / "report.json").write_text(
        json.dumps(suite, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"report: {output / 'report.json'}")
    return 0 if suite["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())