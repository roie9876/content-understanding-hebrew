#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

import handwriting_pipeline as pipeline
import segmented_ocr as shared_segmented


PROMPT_A = """BLIND GLYPH-LEVEL HEBREW HANDWRITING READ.

The attached image contains exactly one isolated handwritten line. You have no document
context, neighboring lines, prior OCR candidate, expected entity, or reference answer.

Read from visible blue pen strokes only. Process glyph positions from RIGHT to LEFT.
Do not complete a familiar word, name, place, institution, address, or phrase. Do not repair
spelling. When a glyph is not visually supported, write [?]. A plausible word is not evidence.

Return JSON only:
{
  "transcription": "raw visible glyph sequence",
  "glyphs": [
    {
      "position_rtl": 1,
      "character": "one visible character or [?]",
      "confidence": "high | medium | low",
      "visual_evidence": "brief stroke description"
    }
  ],
  "notes": []
}
"""


PROMPT_B = """FORENSIC, CANDIDATE-BLIND HEBREW GLYPH AUDIT.

Inspect this single handwritten line without guessing its meaning. No previous reading,
document type, field label, neighboring line, known phrase, person, location, or institution
is available to you. Treat each glyph as an unknown pen shape before forming any text.

Move RIGHT to LEFT. Preserve spaces and punctuation visible in the image. Output [?] for every
position whose strokes do not identify one character. Never choose characters merely because
they create a valid Hebrew word or familiar phrase.

Return JSON only:
{
  "transcription": "strict stroke-grounded sequence",
  "glyphs": [
    {
      "position_rtl": 1,
      "character": "one visible character or [?]",
      "confidence": "high | medium | low",
      "visual_evidence": "brief stroke description"
    }
  ],
  "notes": []
}
"""


def blue_pen_mask(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB")).astype(np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return (
        (blue - red >= 24)
        & (blue - green >= 17)
        & (blue <= 245)
    ).astype(np.uint8)


def _group_rows(rows: np.ndarray) -> list[int]:
    if not len(rows):
        return []
    groups = [[int(rows[0])]]
    for row in rows[1:]:
        if int(row) <= groups[-1][-1] + 1:
            groups[-1].append(int(row))
        else:
            groups.append([int(row)])
    return [round(sum(group) / len(group)) for group in groups]


def detect_form_baselines(image: Image.Image) -> list[int]:
    gray = np.array(image.convert("L"))
    dark = np.where(gray <= 120, 255, 0).astype(np.uint8)
    width = image.width
    horizontal = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, width // 3), 1)),
    )
    strengths = (horizontal > 0).sum(axis=1)
    return _group_rows(np.flatnonzero(strengths >= width * 0.35))


def segment_handwritten_lines(
    image: Image.Image, expected_lines: int | None = None,
) -> tuple[list[Image.Image], list[dict], np.ndarray]:
    mask = blue_pen_mask(image)
    baselines = detect_form_baselines(image)
    if expected_lines is not None and len(baselines) != expected_lines:
        raise ValueError(
            f"Expected {expected_lines} form baselines, detected {len(baselines)}: {baselines}"
        )
    if not baselines:
        raise ValueError("No horizontal form baselines were detected.")

    grouped = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 11)),
    )
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(grouped, 8)
    line_masks = [np.zeros_like(mask) for _baseline in baselines]
    for component_index in range(1, component_count):
        if int(stats[component_index, cv2.CC_STAT_AREA]) < 12:
            continue
        component = labels == component_index
        original_pixels = component & (mask > 0)
        y_positions = np.flatnonzero(original_pixels.any(axis=1))
        if not len(y_positions):
            continue
        anchor_y = float(np.percentile(y_positions, 80))
        line_index = min(
            range(len(baselines)),
            key=lambda index: abs(baselines[index] - anchor_y),
        )
        line_masks[line_index][original_pixels] = 1

    source = np.array(image.convert("RGB"))
    source_int = source.astype(np.int16)
    visible_blue = (
        (source_int[:, :, 2] > source_int[:, :, 0])
        & (source_int[:, :, 2] > source_int[:, :, 1])
    )
    lines: list[Image.Image] = []
    metadata: list[dict] = []
    for line_index, (baseline, line_mask) in enumerate(zip(baselines, line_masks), start=1):
        y_positions, x_positions = np.nonzero(line_mask)
        if not len(x_positions):
            raise ValueError(f"No blue ink assigned to line {line_index}.")
        left = max(0, int(x_positions.min()) - 14)
        right = min(image.width, int(x_positions.max()) + 15)
        top = max(0, int(y_positions.min()) - 14)
        bottom = min(image.height, int(y_positions.max()) + 15)

        selected_neighborhood = cv2.dilate(
            line_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        ) > 0
        isolated = np.full_like(source, 255)
        copy_pixels = selected_neighborhood & visible_blue
        isolated[copy_pixels] = source[copy_pixels]
        crop = Image.fromarray(isolated[top:bottom, left:right]).convert("RGB")
        lines.append(crop)
        metadata.append({
            "line": line_index,
            "baseline_y": baseline,
            "box": [left, top, right - left, bottom - top],
            "blue_pixels": int(line_mask.sum()),
        })
    return lines, metadata, mask


def _reading_text(result: dict) -> str:
    return " ".join(str(result.get("transcription", "")).strip().split())


def configured_model_name(targets: list[dict]) -> str:
    models = {str(target.get("deployment", "")).strip() for target in targets}
    models.discard("")
    if len(models) != 1:
        raise RuntimeError(f"Expected exactly one configured model, got {sorted(models)}")
    return next(iter(models))


def align_readings(first: str, second: str) -> tuple[str, float]:
    matcher = difflib.SequenceMatcher(None, first, second)
    consensus: list[str] = []
    for operation, first_start, first_end, _second_start, _second_end in matcher.get_opcodes():
        if operation == "equal":
            consensus.append(first[first_start:first_end])
        elif not consensus or consensus[-1] != "[?]":
            consensus.append("[?]")
    return "".join(consensus).strip(), round(matcher.ratio(), 4)


def segment_word_groups(
    line: Image.Image, join_width: int | None = None,
) -> tuple[list[Image.Image], list[dict]]:
    mask = blue_pen_mask(line)
    source = np.array(line.convert("RGB"))
    source_int = source.astype(np.int16)
    visible_blue = (
        (source_int[:, :, 2] > source_int[:, :, 0])
        & (source_int[:, :, 2] > source_int[:, :, 1])
    )
    join_width = join_width or max(9, round(line.height * 0.12))
    if join_width < 1:
        raise ValueError("Word join width must be positive.")
    joined = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (join_width, 3)),
    )
    active_columns = np.flatnonzero(joined.any(axis=0))
    if not len(active_columns):
        raise ValueError("No blue ink found for word segmentation.")

    spans = []
    start = previous = int(active_columns[0])
    for column in active_columns[1:]:
        column = int(column)
        if column > previous + 1:
            spans.append((start, previous))
            start = column
        previous = column
    spans.append((start, previous))

    words: list[Image.Image] = []
    metadata: list[dict] = []
    for word_rtl, (span_left, span_right) in enumerate(reversed(spans), start=1):
        local_mask = mask[:, span_left:span_right + 1]
        y_positions, x_positions = np.nonzero(local_mask)
        if len(x_positions) < 12:
            continue
        left = max(0, span_left + int(x_positions.min()) - 10)
        right = min(line.width, span_left + int(x_positions.max()) + 11)
        top = max(0, int(y_positions.min()) - 10)
        bottom = min(line.height, int(y_positions.max()) + 11)
        selected = np.zeros_like(mask)
        selected[:, span_left:span_right + 1] = mask[:, span_left:span_right + 1]
        selected_neighborhood = cv2.dilate(
            selected,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        ) > 0
        isolated = np.full_like(source, 255)
        copy_pixels = selected_neighborhood & visible_blue
        isolated[copy_pixels] = source[copy_pixels]
        words.append(Image.fromarray(isolated[top:bottom, left:right]).convert("RGB"))
        metadata.append({
            "word_rtl": word_rtl,
            "box": [left, top, right - left, bottom - top],
            "blue_pixels": int(local_mask.sum()),
        })
    return words, metadata


def build_word_view_sets(
    source_image: Image.Image, line_metadata: dict,
    words: list[Image.Image], word_metadata: list[dict],
) -> list[list[tuple[str, Image.Image]]]:
    source = np.array(source_image.convert("RGB"))
    line_left, line_top, _line_width, _line_height = line_metadata["box"]
    view_sets: list[list[tuple[str, Image.Image]]] = []
    for word, metadata in zip(words, word_metadata):
        left, top, width, height = metadata["box"]
        source_crop = source[
            line_top + top:line_top + top + height,
            line_left + left:line_left + left + width,
        ].copy()
        selected_mask = blue_pen_mask(word) > 0
        selected_neighborhood = cv2.dilate(
            selected_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        ) > 0
        source_int = source_crop.astype(np.int16)
        source_blue = (
            (source_int[:, :, 2] > source_int[:, :, 0])
            & (source_int[:, :, 2] > source_int[:, :, 1])
        )
        source_crop[source_blue & ~selected_neighborhood] = 255
        original = Image.fromarray(source_crop).convert("RGB")

        enlarged = word.resize((word.width * 3, word.height * 3), Image.LANCZOS)
        enlarged = ImageEnhance.Contrast(enlarged).enhance(1.25)
        enlarged = enlarged.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2))
        view_sets.append([
            ("original_with_form_line", original),
            ("blue_only", word),
            ("blue_only_enlarged", enlarged),
        ])
    return view_sets


# Keep historical benchmark helpers import-compatible while using the same implementation
# as the app strategy for all new runs.
PROMPT_A = shared_segmented.PROMPT_A
PROMPT_B = shared_segmented.PROMPT_B
blue_pen_mask = shared_segmented.blue_pen_mask
detect_form_baselines = shared_segmented.detect_form_baselines
segment_handwritten_lines = shared_segmented.segment_handwritten_lines
segment_word_groups = shared_segmented.segment_word_groups
build_word_view_sets = shared_segmented.build_word_view_sets


def save_segmentation(
    image: Image.Image, lines: list[Image.Image], metadata: list[dict],
    mask: np.ndarray, output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    image.save(output_dir / "source.png")
    Image.fromarray(mask * 255).save(output_dir / "blue_mask.png")
    overview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overview)
    for item in metadata:
        left, top, width, height = item["box"]
        draw.rectangle((left, top, left + width, top + height), outline="red", width=2)
        draw.text((left + 4, top + 4), str(item["line"]), fill="red")
    overview.save(output_dir / "line_boxes.png")
    for index, line in enumerate(lines, start=1):
        line.save(output_dir / f"line_{index:02d}.png")
    (output_dir / "segmentation.json").write_text(
        json.dumps({"lines": metadata}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_word_segmentation(
    line: Image.Image, words: list[Image.Image], metadata: list[dict], output_dir: Path,
    view_sets: list[list[tuple[str, Image.Image]]] | None = None,
) -> None:
    overview = line.convert("RGB").copy()
    draw = ImageDraw.Draw(overview)
    for item in metadata:
        left, top, width, height = item["box"]
        draw.rectangle((left, top, left + width, top + height), outline="red", width=2)
        draw.text((left + 4, top + 4), str(item["word_rtl"]), fill="red")
    overview.save(output_dir / "word_boxes.png")
    for index, word in enumerate(words, start=1):
        word.save(output_dir / f"word_{index:02d}_rtl.png")
    for index, views in enumerate(view_sets or [], start=1):
        for view_name, image in views:
            image.save(output_dir / f"word_{index:02d}_{view_name}.png")
    (output_dir / "word_segmentation.json").write_text(
        json.dumps({"words": metadata}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _unit_content(
    prompt: str, unit: Image.Image | list[tuple[str, Image.Image]],
) -> list[dict]:
    if isinstance(unit, Image.Image):
        return [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": pipeline._to_data_uri(unit)},
        ]
    content = [{
        "type": "input_text",
        "text": prompt + """

All attached views are faithful renderings of the SAME isolated handwriting unit.
They contain no neighboring word. Compare only their shared strokes; a form line or
image enhancement is not an additional character and does not provide semantic context.
""",
    }]
    for view_name, image in unit:
        content.append({"type": "input_text", "text": f"View: {view_name}"})
        content.append({"type": "input_image", "image_url": pipeline._to_data_uri(image)})
    return content


def run_model_reads(
    images: list[Image.Image | list[tuple[str, Image.Image]]], reasoning: str, workers: int,
    max_output_tokens: int = 4000, unit_name: str = "line",
) -> tuple[list[dict], dict]:
    pipeline._reset_usage()
    tasks = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for unit_index, image in enumerate(images, start=1):
            for pass_name, prompt, seed in (
                ("a", PROMPT_A, 101),
                ("b", PROMPT_B, 202),
            ):
                future = executor.submit(
                    pipeline._responses_call,
                    _unit_content(prompt, image),
                    0,
                    reasoning,
                    max_output_tokens,
                    seed,
                )
                tasks.append((future, unit_index, pass_name))

        responses: dict[tuple[int, str], dict] = {}
        for future, unit_index, pass_name in tasks:
            responses[(unit_index, pass_name)] = future.result()

    results = []
    for unit_index in range(1, len(images) + 1):
        first = responses[(unit_index, "a")]
        second = responses[(unit_index, "b")]
        first_text, second_text = _reading_text(first), _reading_text(second)
        consensus, similarity = align_readings(first_text, second_text)
        results.append({
            unit_name: unit_index,
            "read_a": first,
            "read_b": second,
            "read_a_text": first_text,
            "read_b_text": second_text,
            "consensus": consensus,
            "similarity": similarity,
            "agreed": first_text == second_text and bool(first_text),
        })
    return results, pipeline._get_usage()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Candidate-blind, segmented-line vision handwriting experiment."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-lines", type=int)
    parser.add_argument("--reasoning", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--expected-model")
    parser.add_argument("--max-output-tokens", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--word-line", type=int)
    parser.add_argument("--word-join-width", type=int)
    parser.add_argument("--expected-words", type=int)
    parser.add_argument("--multi-view-words", action="store_true")
    parser.add_argument("--segment-only", action="store_true")
    args = parser.parse_args()

    configured_model = configured_model_name(pipeline.GPT_TARGETS)
    if args.expected_model and configured_model != args.expected_model:
        raise RuntimeError(
            f"Expected model {args.expected_model!r}, configured model is {configured_model!r}"
        )

    image = Image.open(args.input).convert("RGB")
    lines, metadata, mask = segment_handwritten_lines(image, args.expected_lines)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_slug = re.sub(r"[^a-zA-Z0-9]+", "-", configured_model).strip("-")
    output_dir = args.output or Path("analysis_results") / (
        f"{args.input.name}__{model_slug}-segmented-lines__{timestamp}"
    )
    save_segmentation(image, lines, metadata, mask, output_dir)
    word_images = None
    word_metadata = None
    word_view_sets = None
    if args.word_line is not None:
        if args.word_line < 1 or args.word_line > len(lines):
            raise ValueError(f"--word-line must be between 1 and {len(lines)}")
        word_images, word_metadata = segment_word_groups(
            lines[args.word_line - 1], join_width=args.word_join_width,
        )
        if args.expected_words is not None and len(word_images) != args.expected_words:
            raise ValueError(
                f"Expected {args.expected_words} word groups, detected {len(word_images)}"
            )
        if args.multi_view_words:
            word_view_sets = build_word_view_sets(
                image, metadata[args.word_line - 1], word_images, word_metadata,
            )
        save_word_segmentation(
            lines[args.word_line - 1], word_images, word_metadata, output_dir,
            view_sets=word_view_sets,
        )
    if args.segment_only:
        print(output_dir)
        return 0

    units = word_view_sets or word_images or lines
    unit_name = "word_rtl" if word_images is not None else "line"
    results, usage = run_model_reads(
        units, args.reasoning, max(1, args.workers),
        max_output_tokens=max(512, args.max_output_tokens), unit_name=unit_name,
    )
    method = (
        "isolated_word_groups_two_candidate_blind_reads"
        if word_images is not None
        else "segmented_lines_two_candidate_blind_reads"
    )
    consensus_separator = " " if word_images is not None else "\n"
    payload = {
        "version": 1,
        "method": method,
        "model": configured_model,
        "reasoning": args.reasoning,
        "max_output_tokens": max(512, args.max_output_tokens),
        "source_image_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "reference_text_loaded": False,
        "prompts": {
            "a_sha256": hashlib.sha256(PROMPT_A.encode()).hexdigest(),
            "b_sha256": hashlib.sha256(PROMPT_B.encode()).hexdigest(),
        },
        "segmentation": metadata,
        "word_line": args.word_line,
        "word_join_width": args.word_join_width,
        "multi_view_words": args.multi_view_words,
        "word_segmentation": word_metadata,
        "units": results,
        "consensus_transcription": consensus_separator.join(
            result["consensus"] for result in results
        ),
        "usage": usage,
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "prompt_a.txt").write_text(PROMPT_A, encoding="utf-8")
    (output_dir / "prompt_b.txt").write_text(PROMPT_B, encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "consensus_transcription": payload["consensus_transcription"],
        "usage": usage,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())