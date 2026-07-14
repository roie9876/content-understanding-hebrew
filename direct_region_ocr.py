#!/usr/bin/env python3
"""Run independent image-only OCR requests over numbered region crops."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "analysis_results"
PROMPT_VERSION = "blind-blue-ink-v1"
BLIND_PROMPT = """BLIND IMAGE-ONLY HANDWRITING TRANSCRIPTION.
Your sole evidence is the attached crop. You have no reference answer or document context.
Transcribe ONLY the blue handwritten ink. Ignore every black printed glyph.
Preserve the exact visible character order, line breaks, separators, and punctuation.
Do not infer or normalize likely dates, names, banks, branches, addresses, or field values.
If any character is uncertain, use [?] and list at most two visually supported alternatives.
If the blue ink is only a signature or selection mark, classify it instead of inventing text.
Return exactly these sections:
CLASSIFICATION
handwritten_text | mark_or_checkbox | signature_non_text | no_handwriting | uncertain
TRANSCRIPTION
<raw visible handwriting, or empty for non-text>
CONFIDENCE
high | medium | low
ALTERNATIVES
<only visually supported alternatives, or none>"""

STROKE_GROUNDED_PROMPT_VERSION = "stroke-grounded-v2"
STROKE_GROUNDED_PROMPT = """STRICT STROKE-GROUNDED HANDWRITING DECODING.
Your sole evidence is the attached crop. You have no reference answer, hidden metadata,
document context outside this crop, or database of expected entities.

Read ONLY blue handwritten ink. Black printed glyphs may identify nearby field structure,
but must never be transcribed as handwriting. Do not use world knowledge about institutions,
branches, cities, streets, people, account formats, or likely form values.

Work in two passes:
1. VISUAL PASS: decode each line and token only from visible pen strokes. Preserve line breaks,
   slashes, spaces, and character order. Write [?] for every unresolved character or token.
2. CORRECTION PASS: contextual reasoning may correct a raw token only when the corrected token
   is a minimal spelling repair supported by the same visible glyph sequence. A correction is
   allowed only when every changed character was explicitly uncertain in the visual pass and
   the correction changes at most one character per token. Never introduce a new proper noun,
   place, branch, address, or number that was not already supported character-by-character.
   Context may resolve a near-word; it may not replace an unreadable token with a plausible fact.

For every corrected token, show RAW -> CORRECTED, the one changed character, and its visible
stroke evidence. If that evidence cannot be stated, keep the raw token with [?]. Do not choose
an answer merely because it forms a familiar phrase. Prefer incomplete text over invention.

Return exactly these sections:
CLASSIFICATION
handwritten_text | mark_or_checkbox | signature_non_text | no_handwriting | uncertain
VISUAL_TRANSCRIPTION
<line-by-line stroke-only reading>
GLYPH_EVIDENCE
<one line per token: token, visible character pattern, and uncertain positions>
CORRECTIONS
<RAW -> CORRECTED with one-character evidence, or none>
FINAL_TRANSCRIPTION
<visual transcription plus only permitted corrections>
CONFIDENCE
<high | medium | low for each line>"""

PROMPT_PROFILES = {
    PROMPT_VERSION: BLIND_PROMPT,
    STROKE_GROUNDED_PROMPT_VERSION: STROKE_GROUNDED_PROMPT,
}


def discover_region_images(input_dir: Path) -> list[tuple[int, Path]]:
    """Discover pipeline-style crops without reading any metadata or JSON files."""
    found: dict[int, Path] = {}
    for crop_path in sorted(input_dir.glob("region_*/01_original_crop.png")):
        match = re.match(r"region_(\d+)(?:_|$)", crop_path.parent.name)
        if not match:
            continue
        region_id = int(match.group(1))
        if region_id in found:
            raise ValueError(f"Duplicate crop for region {region_id}")
        found[region_id] = crop_path
    return sorted(found.items())


def image_sha256(image_path: Path) -> str:
    return hashlib.sha256(image_path.read_bytes()).hexdigest()


def build_payload(image_bytes: bytes, deployment: str, reasoning: str,
                  max_output_tokens: int, prompt: str = BLIND_PROMPT) -> dict[str, Any]:
    image_data = base64.b64encode(image_bytes).decode("ascii")
    return {
        "model": deployment,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_data}",
                },
            ],
        }],
        "reasoning": {"effort": reasoning},
        "max_output_tokens": max_output_tokens,
    }


def response_output_text(response: dict[str, Any]) -> str:
    texts = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                texts.append(str(content["text"]))
    return "\n".join(texts).strip()


def parse_sections(text: str) -> dict[str, Any]:
    headings = re.compile(
        r"(?mi)^(CLASSIFICATION|TRANSCRIPTION|CONFIDENCE|ALTERNATIVES)\s*:?\s*$"
    )
    matches = list(headings.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1).upper()] = text[match.end():end].strip()

    alternatives_text = sections.get("ALTERNATIVES", "")
    alternatives = []
    if alternatives_text and alternatives_text.lower() not in {"none", "n/a", "[]"}:
        alternatives = [
            line.strip().lstrip("-* ").strip()
            for line in alternatives_text.splitlines()
            if line.strip()
        ]
    return {
        "classification": sections.get("CLASSIFICATION", "uncertain").splitlines()[0].strip(),
        "transcription": sections.get("TRANSCRIPTION", "").strip(),
        "confidence": sections.get("CONFIDENCE", "low").splitlines()[0].strip(),
        "alternatives": alternatives,
    }


def parse_stroke_grounded_sections(text: str) -> dict[str, Any]:
    headings = re.compile(
        r"(?mi)^(CLASSIFICATION|VISUAL_TRANSCRIPTION|GLYPH_EVIDENCE|CORRECTIONS|"
        r"FINAL_TRANSCRIPTION|CONFIDENCE)\s*:?\s*$"
    )
    matches = list(headings.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1).upper()] = text[match.end():end].strip()
    return {
        "classification": sections.get("CLASSIFICATION", "uncertain").splitlines()[0].strip(),
        "visual_transcription": sections.get("VISUAL_TRANSCRIPTION", "").strip(),
        "glyph_evidence": sections.get("GLYPH_EVIDENCE", "").strip(),
        "corrections": sections.get("CORRECTIONS", "").strip(),
        "transcription": sections.get("FINAL_TRANSCRIPTION", "").strip(),
        "confidence": sections.get("CONFIDENCE", "low").strip(),
        "alternatives": [],
    }


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def call_region(image_path: Path, endpoint: str, deployment: str, reasoning: str,
                max_output_tokens: int, timeout: int, token: str,
                retries: int, prompt_version: str = PROMPT_VERSION) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = PROMPT_PROFILES[prompt_version]
    payload = build_payload(
        image_path.read_bytes(), deployment, reasoning, max_output_tokens, prompt=prompt
    )
    last_error = ""
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                f"{endpoint.rstrip('/')}/responses",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if attempt < retries:
                    retry_after = float(response.headers.get("Retry-After", attempt + 1))
                    time.sleep(max(1.0, retry_after))
                    continue
            response.raise_for_status()
            api_response = response.json()
            raw_output = response_output_text(api_response)
            if not raw_output:
                last_error = "Model returned no visible output text"
                if attempt < retries:
                    continue
                raise RuntimeError(last_error)
            if prompt_version == STROKE_GROUNDED_PROMPT_VERSION:
                parsed = parse_stroke_grounded_sections(raw_output)
            else:
                parsed = parse_sections(raw_output)
            return ({
                **parsed,
                "raw_output": raw_output,
                "model": api_response.get("model", deployment),
                "response_id": api_response.get("id"),
                "response_status": api_response.get("status"),
                "usage": api_response.get("usage", {}),
            }, api_response)
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(attempt + 1)
                continue
    raise RuntimeError(last_error or "Direct region request failed")


def _summary(run: dict[str, Any], results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    ordered = [results[region_id] for region_id in sorted(results)]
    usage = {
        "input_tokens": sum(item.get("usage", {}).get("input_tokens", 0) for item in ordered),
        "output_tokens": sum(item.get("usage", {}).get("output_tokens", 0) for item in ordered),
        "reasoning_tokens": sum(
            item.get("usage", {}).get("output_tokens_details", {}).get("reasoning_tokens", 0)
            for item in ordered
        ),
    }
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return {
        **run,
        "completed_regions": len(ordered),
        "failed_regions": sum(item.get("status") == "failed" for item in ordered),
        "usage": usage,
        "regions": ordered,
    }


def run_batch(input_dir: Path, output_dir: Path, endpoint: str, deployment: str,
              reasoning: str, max_output_tokens: int, timeout: int,
              workers: int, retries: int, expected_count: Optional[int],
              prompt_version: str = PROMPT_VERSION) -> Path:
    discovered = discover_region_images(input_dir)
    if not discovered:
        raise ValueError(f"No region_*/01_original_crop.png files found under {input_dir}")
    if expected_count is not None and len(discovered) != expected_count:
        raise ValueError(f"Expected {expected_count} regions, found {len(discovered)}")

    ids = [region_id for region_id, _path in discovered]
    if expected_count is not None and ids != list(range(1, expected_count + 1)):
        raise ValueError(f"Expected contiguous IDs 1..{expected_count}, found {ids}")

    output_dir.mkdir(parents=True, exist_ok=False)
    inputs_dir = output_dir / "inputs"
    responses_dir = output_dir / "responses"
    inputs_dir.mkdir()
    responses_dir.mkdir()

    staged = []
    for region_id, source_path in discovered:
        staged_path = inputs_dir / f"region_{region_id:02d}.png"
        shutil.copyfile(source_path, staged_path)
        staged.append((region_id, staged_path, image_sha256(staged_path)))

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run = {
        "version": 1,
        "method": "direct_region_image_only",
        "prompt_version": prompt_version,
        "model": deployment,
        "reasoning_effort": reasoning,
        "created_at": created_at,
        "expected_regions": expected_count,
        "discovered_region_ids": ids,
        "golden_text_loaded_during_inference": False,
        "input_metadata_files_read": [],
    }
    _atomic_write_json(output_dir / "manifest.json", run)

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    token = credential.get_token("https://ai.azure.com/.default").token
    started = time.time()
    results: dict[int, dict[str, Any]] = {}

    def process(item: tuple[int, Path, str]) -> tuple[int, dict[str, Any], dict[str, Any]]:
        region_id, image_path, sha256 = item
        result, api_response = call_region(
            image_path, endpoint, deployment, reasoning, max_output_tokens,
            timeout, token, retries, prompt_version,
        )
        result.update({
            "region": region_id,
            "image_id": f"region_{region_id:02d}",
            "image_sha256": sha256,
            "status": "complete",
        })
        return region_id, result, api_response

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process, item): item[0] for item in staged}
        for future in as_completed(futures):
            region_id = futures[future]
            try:
                completed_id, result, api_response = future.result()
                _atomic_write_json(
                    responses_dir / f"region_{completed_id:02d}.json", api_response
                )
                results[completed_id] = result
                print(
                    f"[{len(results)}/{len(staged)}] region {completed_id:02d}: "
                    f"{result['transcription'] or result['classification']}",
                    flush=True,
                )
            except Exception as exc:
                results[region_id] = {
                    "region": region_id,
                    "image_id": f"region_{region_id:02d}",
                    "image_sha256": next(item[2] for item in staged if item[0] == region_id),
                    "status": "failed",
                    "error": str(exc),
                    "usage": {},
                }
                print(f"[{len(results)}/{len(staged)}] region {region_id:02d}: FAILED {exc}",
                      file=sys.stderr, flush=True)
            progress = _summary(run, results)
            progress["elapsed_seconds"] = round(time.time() - started, 1)
            _atomic_write_json(output_dir / "direct_region_ocr_result.json", progress)

    summary = _summary(run, results)
    summary["elapsed_seconds"] = round(time.time() - started, 1)
    _atomic_write_json(output_dir / "direct_region_ocr_result.json", summary)
    return output_dir


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run one independent image-only model request per numbered region crop."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reasoning", choices=("none", "low", "medium", "high"),
                        default="high")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=32000)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--prompt-profile", choices=tuple(PROMPT_PROFILES),
                        default=PROMPT_VERSION)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    try:
        discovered = discover_region_images(input_dir)
        if args.dry_run:
            print(json.dumps({
                "count": len(discovered),
                "region_ids": [region_id for region_id, _path in discovered],
                "metadata_files_read": [],
            }, indent=2))
            return 0

        endpoint = os.environ.get("GPT_SOL_ENDPOINT", "")
        deployment = os.environ.get("GPT_SOL_DEPLOYMENT", "gpt-5.6-sol")
        if not endpoint:
            raise ValueError("GPT_SOL_ENDPOINT is not configured")
        output = args.output
        if output is None:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output = DEFAULT_OUTPUT_ROOT / f"direct-sol-{args.reasoning}__{stamp}"
        result_dir = run_batch(
            input_dir=input_dir,
            output_dir=output.resolve(),
            endpoint=endpoint,
            deployment=deployment,
            reasoning=args.reasoning,
            max_output_tokens=args.max_output_tokens,
            timeout=args.timeout,
            workers=args.workers,
            retries=args.retries,
            expected_count=args.expected_count,
            prompt_version=args.prompt_profile,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"direct region OCR failed: {exc}", file=sys.stderr)
        return 1
    print(result_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())