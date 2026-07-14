from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


PROMPT_A = """BLIND GLYPH-LEVEL HEBREW HANDWRITING READ.

The attached images contain exactly one isolated handwritten unit. You have no document
context, neighboring words or lines, prior OCR candidate, expected entity, or reference answer.

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

Inspect this single isolated handwriting unit without guessing its meaning. No previous reading,
document type, field label, neighboring word or line, known phrase, person, location, or
institution is available to you. Treat each glyph as an unknown pen shape before forming text.

Move RIGHT to LEFT. Preserve punctuation visible in the image. Output [?] for every position
whose strokes do not identify one character. Never choose characters merely because they create
a valid Hebrew word or familiar phrase.

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
        lines.append(Image.fromarray(isolated[top:bottom, left:right]).convert("RGB"))
        metadata.append({
            "line": line_index,
            "baseline_y": baseline,
            "box": [left, top, right - left, bottom - top],
            "blue_pixels": int(line_mask.sum()),
        })
    return lines, metadata, mask


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
        enlarged = enlarged.filter(
            ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2)
        )
        view_sets.append([
            ("original_with_form_line", original),
            ("blue_only", word),
            ("blue_only_enlarged", enlarged),
        ])
    return view_sets


def build_line_view_set(
    source_image: Image.Image, line: Image.Image, line_metadata: dict,
) -> list[tuple[str, Image.Image]]:
    source = np.array(source_image.convert("RGB"))
    left, top, width, height = line_metadata["box"]
    source_crop = source[top:top + height, left:left + width].copy()
    selected_mask = blue_pen_mask(line) > 0
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

    enlarged = line.resize((line.width * 3, line.height * 3), Image.LANCZOS)
    enlarged = ImageEnhance.Contrast(enlarged).enhance(1.25)
    enlarged = enlarged.filter(
        ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2)
    )
    return [
        ("original_with_form_line", original),
        ("blue_only", line),
        ("blue_only_enlarged", enlarged),
    ]