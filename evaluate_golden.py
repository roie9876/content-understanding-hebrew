#!/usr/bin/env python3
"""Evaluate fresh OCR outputs against unchanged region and document goldens.

References are read only by this evaluator, never passed to the OCR subprocess.
An exit code of zero means evaluation completed, not that every answer was correct.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image

import compare_golden
import detection_regression


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Sample:
    stem: str
    source: Path
    reference: Path
    page: int


class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"br", "div", "p", "tr", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"div", "p", "tr", "li"}:
            self.parts.append("\n")


def plain_text(markdown: str) -> str:
    parser = _PlainText()
    parser.feed(markdown)
    parser.close()
    return "".join(parser.parts)


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.replace("[?]", ""))
    return compare_golden._exact_norm(text)


def edit_distance(expected, actual) -> int:
    previous = list(range(len(actual) + 1))
    for row, expected_item in enumerate(expected, 1):
        current = [row]
        for column, actual_item in enumerate(actual, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (expected_item != actual_item),
            ))
        previous = current
    return previous[-1]


def text_metrics(expected: str, actual: str) -> dict:
    expected_chars, actual_chars = normalized(expected), normalized(actual)
    expected_words = re.findall(r"\w+", unicodedata.normalize("NFKC", expected.replace("[?]", "")))
    actual_words = re.findall(r"\w+", unicodedata.normalize("NFKC", actual.replace("[?]", "")))
    char_errors = edit_distance(expected_chars, actual_chars)
    word_errors = edit_distance(expected_words, actual_words)
    return {
        "exact": expected_chars == actual_chars,
        "character_errors": char_errors,
        "reference_characters": len(expected_chars),
        "character_error_rate": char_errors / max(1, len(expected_chars)),
        "word_errors": word_errors,
        "reference_words": len(expected_words),
        "word_error_rate": word_errors / max(1, len(expected_words)),
        "reference_uncertain": "?" in expected,
        "uncertainty_markers": actual.count("[?]"),
    }


def _numeric_groups(text: str) -> list[str] | None:
    text = text.replace("[?]", "").strip()
    if not re.fullmatch(r"[\d,\s|]+", text):
        return None
    return [group.replace(",", "") for group in re.findall(r"\d[\d,]*", text)]


def discover_samples(golden_dir: Path = ROOT / "golden") -> list[Sample]:
    samples = []
    for folder in sorted(golden_dir.iterdir()):
        if not folder.is_dir():
            continue
        reference = folder / "golden.json"
        if not reference.exists():
            reference = folder / "handwriting_result.json"
        if not reference.exists():
            raise ValueError(f"No supported reference in {folder}")
        golden = json.loads(reference.read_text(encoding="utf-8"))
        source = (golden.get("document") or {}).get("file") or golden.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError(f"Missing source document in {reference}")
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Golden source document does not exist: {source_path}")
        samples.append(Sample(folder.name, source_path, reference, int(golden.get("page", 1))))
    return samples


def expected_value(expected: dict, path: str):
    value = expected
    for part in path.split("."):
        match = re.fullmatch(r"(\w+)\[(\d+)\]", part)
        value = value[match[1]][int(match[2])] if match else value[part]
    return value


def value_present(value, text: str) -> bool:
    if value is None or not str(value).strip():
        raise ValueError("Cannot evaluate presence of an empty critical value")
    value = str(value)
    if value == "ILS":
        return bool(re.search(r"₪|ש[\"״']?ח|שקל", text))
    if value.isdigit():
        pattern = rf"(?<!\d){re.escape(value)}(?!\d)"
        joined = re.sub(r"(?<=\d)[ ,]+(?=\d)", "", text)
        return bool(re.search(pattern, text) or re.search(pattern, joined))
    return normalized(value) in normalized(text)


def date_value_present(value: str, text: str) -> bool | None:
    pattern = re.compile(r"(?<!\d)(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{2,4})(?!\d)")
    expected = pattern.fullmatch(value.strip())
    if expected is None:
        return None
    expected_key = (int(expected[1]), int(expected[2]), expected[3])
    return any(
        (int(match[1]), int(match[2]), match[3]) == expected_key
        for match in pattern.finditer(text)
    )


def score_document(golden: dict, markdown: str) -> dict:
    text = plain_text(markdown)
    relationships = [
        {"field": path, "expected": expected_value(golden["expected"], path)}
        for path in golden.get("critical_fields", [])
        if path.endswith("_index")
    ]
    values = [{
        "field": path,
        "expected": expected_value(golden["expected"], path),
        "present": value_present(expected_value(golden["expected"], path), text),
    } for path in golden.get("critical_fields", []) if not path.endswith("_index")]
    report = {
        "kind": "document",
        "text": text_metrics(golden["literal_transcription"], text),
        "critical_values_present": sum(value["present"] for value in values),
        "critical_values_total": len(values),
        "critical_values": values,
        "relationship_expectations_not_scored": relationships,
        "limitation": "Value presence does not verify source/destination association.",
    }
    document_date = (golden.get("expected") or {}).get("document_date") or {}
    if isinstance(document_date.get("raw_value"), str):
        expected_date = document_date["raw_value"]
        report["document_date"] = {
            "expected": expected_date,
            "present": date_value_present(expected_date, text),
            "rule": "Day/month leading zeros are ignored; the written year is not expanded.",
        }
    return report


def score_regions(golden: dict, current: dict, page_size: tuple[int, int]) -> dict:
    pairs, extras = compare_golden._match(golden["regions"], current.get("regions", []))
    rows = []
    for expected, actual in pairs:
        if actual is not None and not detection_regression._spatial_match(
            expected["box"], actual["box"], page_size, 0.12,
        )[0]:
            actual = None
        expected_class = expected.get("human_classification")
        text = "" if actual is None else str(actual.get("final_transcription") or "")
        row = {
            "label": expected.get("label", ""),
            "expected_box": expected["box"],
            "actual_box": None if actual is None else actual["box"],
            "expected": expected.get("final_transcription") or "",
            "actual": text,
            "missing": actual is None,
            "expected_classification": expected_class,
        }
        if expected_class:
            row["classification_correct"] = actual is not None and actual.get(
                "classification"
            ) == expected_class
            row["non_text_safely_excluded"] = not text.strip()
        else:
            row.update(text_metrics(row["expected"], text))
            expected_groups = _numeric_groups(row["expected"])
            if expected_groups and len(expected_groups) > 1:
                actual_groups = _numeric_groups(text)
                row["numeric_groups"] = {
                    "expected": expected_groups,
                    "actual": actual_groups,
                    "same_values": actual_groups is not None and Counter(expected_groups) == Counter(actual_groups),
                    "same_order": expected_groups == actual_groups,
                }
            if actual is not None and actual.get("pre_context_transcription") is not None:
                row["before_context"] = actual["pre_context_transcription"]
                row["before_context_metrics"] = text_metrics(
                    row["expected"], actual["pre_context_transcription"],
                )
        rows.append(row)
    texts = [row for row in rows if not row["expected_classification"]]
    classes = [row for row in rows if row["expected_classification"]]
    grouped_numbers = [row["numeric_groups"] for row in texts if "numeric_groups" in row]
    char_errors = sum(row["character_errors"] for row in texts)
    char_count = sum(row["reference_characters"] for row in texts)
    word_errors = sum(row["word_errors"] for row in texts)
    word_count = sum(row["reference_words"] for row in texts)
    reference_tokens = Counter(
        normalized(word)
        for region in golden["regions"] if not region.get("human_classification")
        for word in re.findall(r"\w+", (region.get("final_transcription") or "").replace("[?]", ""))
    )
    prediction_tokens = Counter(
        normalized(word)
        for region in current.get("regions", [])
        for word in re.findall(r"\w+", (region.get("final_transcription") or "").replace("[?]", ""))
    )
    matching_tokens = sum((reference_tokens & prediction_tokens).values())
    precision = matching_tokens / max(1, sum(prediction_tokens.values()))
    recall = matching_tokens / max(1, sum(reference_tokens.values()))
    return {
        "kind": "regions",
        "exact_regions": sum(row["exact"] and not row["missing"] for row in texts),
        "text_regions": len(texts),
        "missing_text_regions": sum(row["missing"] for row in texts),
        "character_errors": char_errors,
        "reference_characters": char_count,
        "character_error_rate": char_errors / max(1, char_count),
        "word_errors": word_errors,
        "reference_words": word_count,
        "word_error_rate": word_errors / max(1, word_count),
        "classification_correct": sum(row["classification_correct"] for row in classes),
        "classification_total": len(classes),
        "non_text_safely_excluded": sum(row["non_text_safely_excluded"] for row in classes),
        "unmatched_predictions": len(extras),
        "numeric_group_regions": len(grouped_numbers),
        "numeric_groups_same_values": sum(row["same_values"] for row in grouped_numbers),
        "numeric_groups_same_order": sum(row["same_order"] for row in grouped_numbers),
        "word_inventory": {
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
            "matching_tokens": matching_tokens,
            "reference_tokens": sum(reference_tokens.values()),
            "prediction_tokens": sum(prediction_tokens.values()),
        },
        "rows": rows,
        "limitation": (
            "Exact region text uses spatial one-to-one matching; changed segmentation can affect it. "
            "Word inventory is segmentation/order independent but does not verify field association."
        ),
    }


def score_sample(sample: Sample, output: Path) -> dict:
    golden = json.loads(sample.reference.read_text(encoding="utf-8"))
    result = json.loads((output / "handwriting_result.json").read_text(encoding="utf-8"))
    recovery = result.get("recovery") or {}
    recovered_merge = (
        recovery.get("stage") == "final_markdown_merge"
        and recovery.get("merge_succeeded") is True
        and recovery.get("recognition_reused") is True
        and isinstance(recovery.get("fresh_successful_model_calls"), int)
        and recovery["fresh_successful_model_calls"] > 0
        and (output / "merged.md").is_file()
        and (output / "merged.md").stat().st_size > 0
    )
    if (result.get("usage") or {}).get("calls") == 0:
        raise ValueError("No successful model calls; this is not a valid OCR quality sample")
    log_path = output / "pipeline.log"
    if log_path.exists():
        log = log_path.read_text(encoding="utf-8").casefold()
        if not recovered_merge and any(marker in log for marker in (
            "tenant provided in token does not match resource token",
            "modelaccesserror:",
            "azureclicredential.get_token failed:",
        )):
            raise ValueError("Azure authentication failed during this run; exclude it and rerun after sign-in")
    if "literal_transcription" in golden:
        metrics = score_document(golden, (output / "merged.md").read_text(encoding="utf-8"))
    else:
        with Image.open(output / f"page_{sample.page}_render.png") as page:
            metrics = score_regions(golden, result, page.size)
            detection_reference = sample.reference.parent / "detection_regions.json"
            if detection_reference.exists():
                detection_golden = json.loads(detection_reference.read_text(encoding="utf-8"))
            else:
                detection_golden = golden
            detected = json.loads((output / "detected_regions.json").read_text(encoding="utf-8"))
            metrics["detection"] = detection_regression.score_detection(
                detection_golden, detected["regions"], page.size,
            )
    return {
        "stem": sample.stem, "output": str(output), **metrics,
        "elapsed_seconds": result.get("elapsed_seconds"),
        "calls": (
            (result.get("usage") or {}).get("calls")
            if (result.get("usage") or {}).get("complete", True) else None
        ),
        "usage_complete": (result.get("usage") or {}).get("complete", True),
        "successful_calls_lower_bound": (
            (result.get("usage") or {}).get("calls")
            if (result.get("usage") or {}).get("complete") is False else None
        ),
        "recovery": recovery or None,
        "review_required_regions": sum(
            bool(region.get("needs_review")) for region in result.get("regions", [])
        ),
    }


def validate_output_location(results: Path, golden_dir: Path = ROOT / "golden") -> None:
    results, golden_dir = results.resolve(), golden_dir.resolve()
    if results.is_relative_to(golden_dir) or golden_dir.is_relative_to(results):
        raise ValueError("Output must be separate from the golden reference tree and its ancestors")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="Run fresh GPT-6.1 analysis before scoring")
    mode.add_argument("--results", type=Path, help="Score an existing STEM/output directory tree")
    parser.add_argument("--output", type=Path, help="Fresh-run directory (with --run)")
    parser.add_argument("--stem", action="append", help="Limit to a golden folder; repeat to combine selectors")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pipeline", type=Path, default=ROOT / "handwriting_pipeline.py")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output and not args.run:
        parser.error("--output requires --run")
    samples = discover_samples()
    if args.stem:
        unknown = set(args.stem) - {sample.stem for sample in samples}
        if unknown:
            parser.error(f"Unknown golden folders: {', '.join(sorted(unknown))}")
        samples = [sample for sample in samples if sample.stem in args.stem]
    results = args.results or args.output or ROOT / "analysis_results" / (
        "quality_evaluation__" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    )
    try:
        validate_output_location(results)
    except ValueError as exc:
        parser.error(str(exc))
    results.mkdir(parents=True, exist_ok=True)
    reports = []
    if args.run:
        import app
        environment = app._page_environment("gpt-6.1-sol", "standard")
    for sample in samples:
        output = results / sample.stem
        try:
            if args.run:
                output.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable, "-u", str(args.pipeline),
                    "--input", str(sample.source), "--page", str(sample.page),
                    "--dpi", "400", "--workers", str(args.workers), "--auto-detect",
                    "--output", str(output),
                ]
                with (output / "pipeline.log").open("w", encoding="utf-8") as log:
                    subprocess.run(
                        command, cwd=ROOT, env=environment,
                        stdout=log, stderr=subprocess.STDOUT, check=True,
                    )
            report = score_sample(sample, output)
            summary = {key: report[key] for key in (
                'exact_regions', 'text_regions', 'critical_values_present',
                'critical_values_total', 'character_error_rate',
            ) if key in report}
            print(f"SCORED {sample.stem}: {json.dumps(summary, ensure_ascii=False)}", flush=True)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            report = {"stem": sample.stem, "error": str(exc)}
            print(f"ERROR {sample.stem}: {exc}", file=sys.stderr, flush=True)
        reports.append(report)
        (results / "quality-report.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    return int(any("error" in report for report in reports))


if __name__ == "__main__":
    raise SystemExit(main())
