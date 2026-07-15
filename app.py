#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import fitz
import markdown
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from dotenv import load_dotenv
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

import compare_golden

load_dotenv()
register_heif_opener()


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / ".ui_data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
RESULTS_DIR = ROOT / "analysis_results"
IMAGE_EXTENSIONS = {
    ".bmp", ".gif", ".heic", ".heif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp",
}
ALLOWED_EXTENSIONS = {".pdf", *IMAGE_EXTENSIONS}
SUPPORTED_UPLOAD_TYPES = "PDF, PNG, JPEG, WebP, TIFF, BMP, GIF, HEIC, or HEIF"
PAGE_CONCURRENCY = max(1, int(os.environ.get("PAGE_CONCURRENCY", "2")))
TOTAL_OCR_WORKERS = max(1, int(os.environ.get("MAX_WORKERS", "8")))
_PRIMARY_ENDPOINT = os.environ.get("GPT_ENDPOINT", "https://example.invalid/openai/v1/")
_PRIMARY_DEPLOYMENT = os.environ.get("GPT_DEPLOYMENT", "gpt-5.4")
_PRIMARY_TARGETS = json.loads(os.environ.get("GPT_TARGETS", "[]"))

MODEL_PROFILES = {
    "gpt-5.4": {
        "label": "GPT-5.4",
        "endpoint": _PRIMARY_ENDPOINT,
        "deployment": _PRIMARY_DEPLOYMENT,
        "targets": _PRIMARY_TARGETS,
    },
    "gpt-5.4-pro": {
        "label": "GPT-5.4 Pro · medium reasoning (experimental, slower)",
        "endpoint": os.environ.get("GPT_PRO_ENDPOINT", _PRIMARY_ENDPOINT),
        "deployment": os.environ.get("GPT_PRO_DEPLOYMENT", "gpt-5.4-pro"),
        "environment": {
            "GPT_REASONING": "medium",
            "OCR_REASONING": "medium",
            "GPT_READ_TIMEOUT": "180",
        },
    },
    "gpt-5.6-luna": {
        "label": "GPT-5.6 Luna (experimental)",
        "endpoint": os.environ.get("GPT_LUNA_ENDPOINT", "https://example.invalid/openai/v1/"),
        "deployment": os.environ.get("GPT_LUNA_DEPLOYMENT", "gpt-5.6-luna"),
    },
    "gpt-5.6-sol-none": {
        "label": "GPT-5.6 Sol · Fast (no reasoning)",
        "endpoint": os.environ.get("GPT_SOL_ENDPOINT", "https://example.invalid/openai/v1/"),
        "deployment": os.environ.get("GPT_SOL_DEPLOYMENT", "gpt-5.6-sol"),
        "environment": {
            "GPT_REASONING": "none",
            "OCR_REASONING": "none",
        },
    },
    "gpt-5.6-sol": {
        "label": "GPT-5.6 Sol · low reasoning (experimental)",
        "endpoint": os.environ.get("GPT_SOL_ENDPOINT", "https://example.invalid/openai/v1/"),
        "deployment": os.environ.get("GPT_SOL_DEPLOYMENT", "gpt-5.6-sol"),
        "environment": {
            "GPT_REASONING": "low",
            "OCR_REASONING": "low",
        },
    },
    "gpt-5.6-sol-medium": {
        "label": "GPT-5.6 Sol · medium reasoning (experimental, slower)",
        "endpoint": os.environ.get("GPT_SOL_ENDPOINT", "https://example.invalid/openai/v1/"),
        "deployment": os.environ.get("GPT_SOL_DEPLOYMENT", "gpt-5.6-sol"),
        "environment": {
            "GPT_REASONING": "medium",
            "OCR_REASONING": "medium",
            "GPT_READ_TIMEOUT": "180",
        },
    },
    "copilot-claude-opus-4.8": {
        "label": "Claude Opus 4.8 · GitHub Copilot SDK (high cost, experimental)",
        "endpoint": "copilot-sdk://local",
        "deployment": "claude-opus-4.8",
        "environment": {
            "MODEL_BACKEND": "github-copilot-sdk",
            "GPT_REASONING": "high",
            "OCR_REASONING": "high",
            "GPT_READ_TIMEOUT": "180",
        },
    },
    "copilot-gemini-3.5-flash": {
        "label": "Gemini 3.5 Flash · GitHub Copilot SDK (experimental)",
        "endpoint": "copilot-sdk://local",
        "deployment": "gemini-3.5-flash",
        "environment": {
            "MODEL_BACKEND": "github-copilot-sdk",
            "GPT_REASONING": "low",
            "OCR_REASONING": "low",
            "GPT_READ_TIMEOUT": "120",
        },
    },
}
DEFAULT_MODEL_PROFILE = "gpt-5.4"
DEFAULT_OCR_STRATEGY = "standard"
OCR_STRATEGIES = {
    "standard": {"label": "Standard core4 + visual reconciliation"},
    "isolated-multiview": {
        "label": "Experimental isolated multi-view (lined handwriting)",
        "model_profiles": {"gpt-5.4", "gpt-5.6-sol"},
    },
}

app = Flask(__name__, static_folder="ui", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

_jobs: dict[str, dict] = {}
_uploads: dict[str, dict] = {}
_processes: dict[str, dict[int, subprocess.Popen]] = {}
_lock = threading.Lock()


@app.after_request
def disable_browser_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def _now() -> float:
    return time.time()


def _run_name(file_name: str, created_at: float) -> str:
    safe_name = re.sub(r"[^\w.() -]+", "_", Path(file_name).name, flags=re.UNICODE).strip(" ._")
    safe_name = safe_name or "document"
    timestamp = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d_%H-%M-%S")
    base = f"{safe_name}__{timestamp}"
    candidate = base
    suffix = 2
    while (RESULTS_DIR / candidate).exists():
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _manifest_data(job_id: str, job: dict) -> dict:
    return {
        "version": 1,
        "id": job_id,
        "run_name": job.get("run_name", Path(job["output_dir"]).name),
        "file_name": job["file_name"],
        "model_profile": job.get("model_profile", DEFAULT_MODEL_PROFILE),
        "ocr_strategy": job.get("ocr_strategy", DEFAULT_OCR_STRATEGY),
        "page_count": job["page_count"],
        "selected_pages": _selected_pages(job),
        "status": job.get("status", "complete"),
        "stage": job.get("stage", "complete"),
        "message": job.get("message", "Document analysis complete"),
        "created_at": job.get("created_at", _now()),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "elapsed_seconds": job.get("elapsed_seconds"),
        "error": job.get("error"),
    }


def _persist_job(job_id: str) -> None:
    with _lock:
        job = dict(_jobs[job_id])
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "run.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(_manifest_data(job_id, job), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(manifest)


def _job_from_manifest(manifest: dict, output_dir: Path) -> dict:
    selected_pages = [int(page) for page in manifest.get("selected_pages", [])]
    return {
        "file_name": manifest["file_name"],
        "model_profile": manifest.get("model_profile", DEFAULT_MODEL_PROFILE),
        "ocr_strategy": manifest.get("ocr_strategy", DEFAULT_OCR_STRATEGY),
        "input_path": "",
        "output_dir": str(output_dir),
        "run_name": manifest.get("run_name", output_dir.name),
        "page_count": int(manifest["page_count"]),
        "selected_pages": selected_pages,
        "page_states": {
            page: {"stage": "complete", "message": "Page complete"}
            for page in selected_pages
        },
        "current_page": selected_pages[0] if selected_pages else 1,
        "status": "complete",
        "stage": "complete",
        "message": "Document analysis complete",
        "created_at": manifest.get("created_at", output_dir.stat().st_mtime),
        "started_at": manifest.get("started_at"),
        "completed_at": manifest.get("completed_at"),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "error": manifest.get("error"),
        "log": [],
        "historical": True,
    }


def _load_saved_jobs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    restored = {}
    for manifest_path in RESULTS_DIR.glob("*/run.json"):
        manifest = _read_json(manifest_path)
        if not manifest or not manifest.get("id") or not manifest.get("file_name"):
            continue
        job = _job_from_manifest(manifest, manifest_path.parent)
        if all((_page_dir(job, page) / "handwriting_result.json").exists() for page in _selected_pages(job)):
            restored[manifest["id"]] = job
    with _lock:
        _jobs.update(restored)


def _page_count(path: Path) -> int:
    if path.suffix.lower() != ".pdf":
        return 1
    with fitz.open(path) as document:
        return len(document)


def _flatten_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        flattened = Image.new("RGB", rgba.size, "white")
        flattened.paste(rgba, mask=rgba.getchannel("A"))
        return flattened
    return image.convert("RGB")


def _normalize_image_upload(source_path: Path, upload_folder: Path) -> Path:
    frames: list[Image.Image] = []
    with Image.open(source_path) as source:
        for frame_number in range(getattr(source, "n_frames", 1)):
            source.seek(frame_number)
            frames.append(_flatten_image(source.copy()))
    if not frames or any(frame.width < 1 or frame.height < 1 for frame in frames):
        raise ValueError("The uploaded image has no readable frames.")
    if len(frames) == 1:
        normalized_path = upload_folder / "source.png"
        frames[0].save(normalized_path, format="PNG")
        return normalized_path

    normalized_path = upload_folder / "source.pdf"
    document = fitz.open()
    try:
        for frame in frames:
            page = document.new_page(width=frame.width * 72 / 200, height=frame.height * 72 / 200)
            encoded = io.BytesIO()
            frame.save(encoded, format="PNG")
            page.insert_image(page.rect, stream=encoded.getvalue())
        document.save(normalized_path)
    finally:
        document.close()
    return normalized_path


def _parse_page_spec(spec: str, total: int) -> list[int]:
    if not spec or not spec.strip():
        raise ValueError("Select at least one page.")
    pages: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            bounds = token.split("-", 1)
            if len(bounds) != 2 or not all(value.strip().isdigit() for value in bounds):
                raise ValueError(f"Invalid page range: {token}")
            start, end = (int(value.strip()) for value in bounds)
            if start > end:
                raise ValueError(f"Page range must be ascending: {token}")
            pages.update(range(start, end + 1))
        elif token.isdigit():
            pages.add(int(token))
        else:
            raise ValueError(f"Invalid page number: {token}")
    invalid = sorted(page for page in pages if page < 1 or page > total)
    if invalid:
        raise ValueError(f"Page {invalid[0]} is outside this {total}-page document.")
    if not pages:
        raise ValueError("Select at least one page.")
    return sorted(pages)


def _selected_pages(job: dict) -> list[int]:
    return list(job.get("selected_pages") or range(1, job["page_count"] + 1))


def _save_upload(document) -> tuple[str, dict]:
    if not document or not document.filename:
        raise ValueError(f"Choose a {SUPPORTED_UPLOAD_TYPES} document.")
    suffix = Path(document.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type. Use {SUPPORTED_UPLOAD_TYPES}.")
    upload_id = uuid.uuid4().hex
    upload_folder = UPLOAD_DIR / upload_id
    upload_folder.mkdir(parents=True, exist_ok=True)
    original_name = Path(document.filename).name or f"document{suffix}"
    uploaded_path = upload_folder / f"uploaded{suffix}"
    document.save(uploaded_path)
    input_path = None
    try:
        input_path = (
            uploaded_path.rename(upload_folder / "source.pdf")
            if suffix == ".pdf"
            else _normalize_image_upload(uploaded_path, upload_folder)
        )
        page_count = _page_count(input_path)
    except Exception as exc:
        uploaded_path.unlink(missing_ok=True)
        if input_path is not None:
            input_path.unlink(missing_ok=True)
        raise ValueError("The uploaded document could not be opened.") from exc
    uploaded_path.unlink(missing_ok=True)
    upload = {
        "id": upload_id,
        "file_name": original_name,
        "input_path": str(input_path),
        "page_count": page_count,
        "created_at": _now(),
    }
    with _lock:
        _uploads[upload_id] = upload
    return upload_id, upload


def _render_upload_preview(upload: dict, page_number: int) -> Path:
    preview_dir = Path(upload["input_path"]).parent / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"page_{page_number:04d}.jpg"
    if preview_path.exists():
        return preview_path
    with fitz.open(upload["input_path"]) as document:
        page = document.load_page(page_number - 1)
        longest_side = max(page.rect.width, page.rect.height)
        scale = min(1.5, 1000 / longest_side) if longest_side else 1.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        pixmap.save(preview_path)
    return preview_path


def _model_environment(profile_name: str) -> dict[str, str]:
    profile = MODEL_PROFILES[profile_name]
    environment = os.environ.copy()
    environment.update(
        GPT_ENDPOINT=profile["endpoint"],
        GPT_DEPLOYMENT=profile["deployment"],
    )
    if profile.get("targets"):
        environment["GPT_TARGETS"] = json.dumps(profile["targets"])
    else:
        environment["GPT_TARGETS"] = ""
    environment.update(profile.get("environment", {}))
    if environment.get("MODEL_BACKEND") == "github-copilot-sdk":
        github_user = os.environ.get("COPILOT_GITHUB_USER", "").strip()
        token_command = ["gh", "auth", "token", "--hostname", "github.com"]
        if github_user:
            token_command.extend(["--user", github_user])
        try:
            token_result = subprocess.run(
                token_command,
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                "GitHub Copilot SDK authentication failed; verify `gh auth status` and COPILOT_GITHUB_USER."
            ) from exc
        token = token_result.stdout.strip()
        if not token:
            raise RuntimeError("GitHub CLI returned an empty authentication token.")
        environment["COPILOT_GITHUB_TOKEN"] = token
        environment["COPILOT_DISABLE_KEYTAR"] = "1"
    return environment


def _page_environment(model_profile: str, ocr_strategy: str) -> dict[str, str]:
    environment = _model_environment(model_profile)
    environment["OCR_STRATEGY"] = ocr_strategy
    if model_profile == "gpt-5.4" and ocr_strategy == "standard":
        environment["GPT_REASONING"] = "low"
        environment["OCR_REASONING"] = "low"
        environment["OCR_MAX_TOKENS"] = "4000"
        environment["OCR_FINAL_ORIGINAL_RESCUE"] = "0"
        environment["DETECTION_REASONING"] = "low"
        environment["DETECTION_MAX_TOKENS"] = "2500"
        environment["GPT_READ_TIMEOUT"] = "60"
    if ocr_strategy == "isolated-multiview":
        environment["GPT_READ_TIMEOUT"] = "120" if model_profile == "gpt-5.4" else "240"
        environment["ISOLATED_MULTIVIEW_WORKERS"] = "2"
        environment["ISOLATED_MULTIVIEW_MAX_TOKENS"] = (
            "6000" if model_profile == "gpt-5.4" else "1800"
        )
    return environment


def _page_dir(job: dict, page_number: int) -> Path:
    output = Path(job["output_dir"])
    return output / f"page_{page_number:04d}" if job["page_count"] > 1 else output


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def _artifact_url(job_id: str, page_dir: Path, file_name: str) -> str | None:
    file_path = page_dir / file_name
    if not file_path.exists():
        return None
    with _lock:
        output_dir = Path(_jobs[job_id]["output_dir"])
    relative = file_path.relative_to(output_dir)
    return f"/api/jobs/{job_id}/files/{relative.as_posix()}"


def _page_snapshot(job_id: str, job: dict, page_number: int) -> dict:
    page_dir = _page_dir(job, page_number)
    result_path = page_dir / "handwriting_result.json"
    result = _read_json(result_path)
    progress = _read_json(page_dir / "ocr_progress.json") or {}

    page_state = (job.get("page_states") or {}).get(page_number, {})
    stage = page_state.get("stage", "waiting")
    if result:
        stage = "complete"

    usage = (result or {}).get("usage") or progress.get("usage") or {}
    with _lock:
        process = _processes.get(job_id, {}).get(page_number)
    process_alive = process is not None and process.poll() is None
    last_activity_at = page_state.get("last_activity_at") or page_state.get("started_at")
    return {
        "page": page_number,
        "stage": stage,
        "regions": len((result or {}).get("regions", [])),
        "tokens": usage.get("total_tokens", 0),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "calls": usage.get("calls", 0),
        "draws_completed": progress.get("completed", 0),
        "draws_failed": progress.get("failed", 0),
        "draws_total": progress.get("total", 0),
        "process_alive": process_alive,
        "last_activity_seconds": round(_now() - last_activity_at, 1) if last_activity_at else None,
        "elapsed_seconds": (result or {}).get("elapsed_seconds", page_state.get("elapsed_seconds")),
        "original_url": _artifact_url(job_id, page_dir, f"page_{page_number}_render.png"),
        "regions_url": _artifact_url(job_id, page_dir, f"page_{page_number}_regions.png"),
        "content_url": _artifact_url(job_id, page_dir, "content_understanding.md"),
        "handwriting_url": _artifact_url(job_id, page_dir, "merged.html"),
        "merged_document_url": f"/api/jobs/{job_id}/pages/{page_number}/merged-document",
        "merged_png_url": f"/api/jobs/{job_id}/pages/{page_number}/merged.png",
        "golden_comparison_url": f"/api/jobs/{job_id}/pages/{page_number}/golden-comparison",
        "diagnostics_url": f"/api/jobs/{job_id}/pages/{page_number}/diagnostics",
    }


def _reconcile_job_state(job_id: str) -> None:
    with _lock:
        job = _jobs[job_id]
        if job["status"] not in {"running", "cancelling"}:
            return
        states = job.get("page_states", {})
        processes = dict(_processes.get(job_id, {}))
        for page_number, state in states.items():
            result_exists = (_page_dir(job, page_number) / "handwriting_result.json").exists()
            process = processes.get(page_number)
            if result_exists:
                state.update(stage="complete", message="Page complete")
            elif state.get("stage") not in {"waiting", "complete", "failed", "cancelled"} \
                    and process is not None and process.poll() is not None:
                state.update(stage="failed", message=f"Page process exited ({process.returncode})")
        stages = [state.get("stage") for state in states.values()]
        if stages and all(stage == "complete" for stage in stages):
            job.update(status="complete", stage="complete", message="Document analysis complete")
            if job.get("elapsed_seconds") is None and job.get("started_at"):
                job["elapsed_seconds"] = round(_now() - job["started_at"], 1)
        elif any(stage == "failed" for stage in stages) and not processes:
            job.update(status="failed", stage="failed", message="Document analysis failed",
                       error="A page process exited before producing its result.")


def _job_snapshot(job_id: str) -> dict:
    _reconcile_job_state(job_id)
    with _lock:
        source = dict(_jobs[job_id])
    selected_pages = _selected_pages(source)
    pages = [_page_snapshot(job_id, source, page) for page in selected_pages]
    completed = sum(page["stage"] == "complete" for page in pages)
    finished_elapsed = sum(page["elapsed_seconds"] or 0 for page in pages)
    elapsed = source.get("elapsed_seconds")
    if elapsed is None and source["status"] in {"running", "cancelling"}:
        elapsed = round(_now() - source["started_at"], 1)
    return {
        "id": job_id,
        "run_name": source.get("run_name", Path(source["output_dir"]).name),
        "created_at": source.get("created_at"),
        "completed_at": source.get("completed_at"),
        "historical": source.get("historical", False),
        "file_name": source["file_name"],
        "model_profile": source.get("model_profile", DEFAULT_MODEL_PROFILE),
        "model_label": MODEL_PROFILES[
            source.get("model_profile", DEFAULT_MODEL_PROFILE)
        ]["label"],
        "ocr_strategy": source.get("ocr_strategy", DEFAULT_OCR_STRATEGY),
        "ocr_strategy_label": OCR_STRATEGIES[
            source.get("ocr_strategy", DEFAULT_OCR_STRATEGY)
        ]["label"],
        "status": source["status"],
        "stage": source.get("stage", "queued"),
        "message": source.get("message", "Waiting to start"),
        "error": source.get("error"),
        "page_count": source["page_count"],
        "selected_page_count": len(selected_pages),
        "selected_pages": selected_pages,
        "processed_pages": completed,
        "current_page": source.get("current_page", 1),
        "active_pages": sorted(
            page for page, state in (source.get("page_states") or {}).items()
            if state.get("stage") not in {"waiting", "complete", "failed", "cancelled"}
        ),
        "progress": round((completed / len(selected_pages)) * 100) if selected_pages else 0,
        "tokens": sum(page["tokens"] for page in pages),
        "calls": sum(page["calls"] for page in pages),
        "elapsed_seconds": elapsed if elapsed is not None else finished_elapsed,
        "pages": pages,
        "log_tail": source.get("log", [])[-30:],
    }


def _set_job(job_id: str, **updates) -> None:
    with _lock:
        _jobs[job_id].update(updates)


def _handle_output(job_id: str, page_number: int, line: str) -> None:
    tagged_line = f"[page {page_number}] {line}"
    with _lock:
        job = _jobs[job_id]
        job.setdefault("log", []).append(tagged_line)
        job["log"] = job["log"][-100:]
        page_state = job["page_states"][page_number]
        stage = page_state.get("stage", "preparing")
        message = page_state.get("message", f"Preparing page {page_number}")

        if "Rendered page" in line:
            stage = "rendered"
            message = "Page rendered at high resolution"
        elif "[CU] starting" in line:
            stage = "content_understanding"
            message = "Reading printed content and document structure"
        elif "[detect] locating" in line:
            stage = "detecting_handwriting"
            message = "Locating handwritten regions"
        elif "[detect] found" in line:
            count = re.search(r"found (\d+)", line)
            stage = "handwriting_ocr"
            message = f"Reading {count.group(1) if count else ''} handwritten regions".strip()
        elif line.startswith("[OCR]"):
            stage = "handwriting_ocr"
            message = "Running multi-pass handwriting recognition"
        elif line.startswith("[vote]") or line.startswith("[reconcile]"):
            stage = "reconciling"
            message = "Reconciling OCR candidates and confidence"
        elif line.startswith("[merge]"):
            stage = "merging"
            message = "Combining printed and handwritten content"
        page_state.update(stage=stage, message=message)
        page_state["last_activity_at"] = _now()
        active = sorted(
            page for page, state in job["page_states"].items()
            if state.get("stage") not in {"waiting", "complete", "failed", "cancelled"}
        )
        job["current_page"] = active[0] if active else page_number
        job["stage"] = stage
        job["message"] = (
            f"Processing pages {', '.join(map(str, active))}: {message}"
            if len(active) > 1 else f"Page {page_number}: {message}"
        )
        log_path = Path(job["output_dir"]) / "pipeline.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(tagged_line + "\n")


def _run_page(job_id: str, page_number: int, workers: int) -> int:
    with _lock:
        job = _jobs[job_id]
        if job["status"] == "cancelling":
            job["page_states"][page_number].update(stage="cancelled")
            return -15
        input_path = job["input_path"]
        model_profile = job.get("model_profile", DEFAULT_MODEL_PROFILE)
        ocr_strategy = job.get("ocr_strategy", DEFAULT_OCR_STRATEGY)
        page_output = _page_dir(job, page_number)
        job["page_states"][page_number].update(
            stage="preparing", message=f"Preparing page {page_number}",
            started_at=_now(), last_activity_at=_now()
        )
    page_output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", str(ROOT / "handwriting_pipeline.py"),
        "--input", input_path,
        "--page", str(page_number),
        "--workers", str(workers),
        "--output", str(page_output),
    ]
    environment = _page_environment(model_profile, ocr_strategy)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    with _lock:
        _processes.setdefault(job_id, {})[page_number] = process
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if line:
                _handle_output(job_id, page_number, line)
        return_code = process.wait()
        with _lock:
            state = _jobs[job_id]["page_states"][page_number]
            cancelled = _jobs[job_id]["status"] == "cancelling"
            state["stage"] = "cancelled" if cancelled else ("complete" if return_code == 0 else "failed")
            state["message"] = "Cancelled" if cancelled else ("Page complete" if return_code == 0 else "Page failed")
            state["elapsed_seconds"] = round(_now() - state["started_at"], 1)
        return return_code
    finally:
        with _lock:
            _processes.get(job_id, {}).pop(page_number, None)


def _build_document_markdown(job: dict) -> None:
    page_docs = []
    for page_number in _selected_pages(job):
        markdown = _page_dir(job, page_number) / "merged.md"
        if markdown.exists():
            page_docs.append((page_number, markdown.read_text(encoding="utf-8").strip()))
    if page_docs:
        combined = "\n\n".join(
            f"<!-- ===================== Page {page} ===================== -->\n\n{text}"
            for page, text in page_docs
        )
        (Path(job["output_dir"]) / "document.md").write_text(combined + "\n", encoding="utf-8")


def _run_job(job_id: str) -> None:
    with _lock:
        job = _jobs[job_id]
        selected_pages = _selected_pages(job)
    started_at = _now()
    concurrency = min(PAGE_CONCURRENCY, len(selected_pages))
    workers_per_page = max(1, TOTAL_OCR_WORKERS // concurrency)
    _set_job(
        job_id,
        status="running",
        stage="preparing",
        message=f"Starting {concurrency} page(s) in parallel",
        started_at=started_at,
    )
    try:
        return_codes = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_run_page, job_id, page, workers_per_page): page
                for page in selected_pages
            }
            for future in as_completed(futures):
                return_codes.append(future.result())
        with _lock:
            cancelled = _jobs[job_id]["status"] == "cancelling"
        if cancelled:
            _set_job(job_id, status="cancelled", stage="cancelled", message="Processing cancelled")
        elif all(return_code == 0 for return_code in return_codes):
            with _lock:
                completed_job = dict(_jobs[job_id])
            _build_document_markdown(completed_job)
            _set_job(
                job_id, status="complete", stage="complete",
                message="Document analysis complete", completed_at=_now(),
            )
        else:
            _set_job(
                job_id,
                status="failed",
                stage="failed",
                message="Document analysis failed",
                error="One or more pages failed. See pipeline.log for details.",
            )
    except Exception as exc:
        _set_job(job_id, status="failed", stage="failed", message="Document analysis failed", error=str(exc))
    finally:
        with _lock:
            _processes.pop(job_id, None)
            job = _jobs[job_id]
            job["elapsed_seconds"] = round(_now() - started_at, 1)
        _persist_job(job_id)


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.post("/api/uploads")
def create_upload():
    try:
        upload_id, upload = _save_upload(request.files.get("document"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({
        "id": upload_id,
        "file_name": upload["file_name"],
        "page_count": upload["page_count"],
        "pages": [
            {
                "page": page,
                "preview_url": f"/api/uploads/{upload_id}/pages/{page}/preview",
            }
            for page in range(1, upload["page_count"] + 1)
        ],
    }), 201


@app.get("/api/uploads/<upload_id>/pages/<int:page_number>/preview")
def upload_page_preview(upload_id: str, page_number: int):
    with _lock:
        upload = _uploads.get(upload_id)
    if not upload or page_number < 1 or page_number > upload["page_count"]:
        abort(404)
    preview = _render_upload_preview(upload, page_number)
    return send_from_directory(preview.parent, preview.name)


@app.post("/api/jobs")
def create_job():
    model_profile = request.form.get("model_profile", DEFAULT_MODEL_PROFILE)
    if model_profile not in MODEL_PROFILES:
        return jsonify({"error": "Unsupported model profile."}), 400
    ocr_strategy = request.form.get("ocr_strategy", DEFAULT_OCR_STRATEGY)
    strategy = OCR_STRATEGIES.get(ocr_strategy)
    if not strategy:
        return jsonify({"error": "Unsupported OCR strategy."}), 400
    allowed_profiles = strategy.get("model_profiles")
    if allowed_profiles and model_profile not in allowed_profiles:
        return jsonify({
            "error": "Experimental isolated multi-view requires GPT-5.4 or GPT-5.6 Sol low."
        }), 400
    upload_id = request.form.get("upload_id")
    if upload_id:
        with _lock:
            upload = _uploads.get(upload_id)
        if not upload:
            return jsonify({"error": "The uploaded document is no longer available."}), 404
    else:
        try:
            upload_id, upload = _save_upload(request.files.get("document"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    try:
        selected_pages = _parse_page_spec(
            request.form.get("pages", f"1-{upload['page_count']}"),
            upload["page_count"],
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    job_id = uuid.uuid4().hex
    created_at = _now()
    run_name = _run_name(upload["file_name"], created_at)
    output_folder = RESULTS_DIR / run_name
    output_folder.mkdir(parents=True, exist_ok=True)
    page_count = upload["page_count"]

    with _lock:
        _jobs[job_id] = {
            "file_name": upload["file_name"],
            "model_profile": model_profile,
            "ocr_strategy": ocr_strategy,
            "upload_id": upload_id,
            "input_path": upload["input_path"],
            "output_dir": str(output_folder),
            "run_name": run_name,
            "page_count": page_count,
            "selected_pages": selected_pages,
            "page_states": {
                page: {"stage": "waiting", "message": "Waiting"}
                for page in selected_pages
            },
            "current_page": selected_pages[0],
            "status": "queued",
            "stage": "queued",
            "message": "Waiting to start",
            "created_at": created_at,
            "elapsed_seconds": None,
            "error": None,
            "log": [],
        }
    _persist_job(job_id)
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return jsonify(_job_snapshot(job_id)), 202


@app.get("/api/runs")
def list_runs():
    with _lock:
        run_ids = [
            job_id for job_id, job in _jobs.items()
            if job.get("status") == "complete"
            and Path(job["output_dir"]).is_relative_to(RESULTS_DIR)
        ]
    runs = [_job_snapshot(job_id) for job_id in run_ids]
    return jsonify([
        {
            "id": run["id"],
            "run_name": run["run_name"],
            "file_name": run["file_name"],
            "model_label": run["model_label"],
            "ocr_strategy": run["ocr_strategy"],
            "ocr_strategy_label": run["ocr_strategy_label"],
            "selected_pages": run["selected_pages"],
            "selected_page_count": run["selected_page_count"],
            "created_at": run["created_at"],
            "completed_at": run["completed_at"],
            "elapsed_seconds": run["elapsed_seconds"],
        }
        for run in sorted(runs, key=lambda item: item.get("created_at") or 0, reverse=True)
    ])


@app.get("/api/jobs/<job_id>")
def get_job(job_id: str):
    with _lock:
        if job_id not in _jobs:
            abort(404)
    return jsonify(_job_snapshot(job_id))


@app.get("/api/jobs/<job_id>/pages/<int:page_number>/diagnostics")
def page_diagnostics(job_id: str, page_number: int):
    with _lock:
        if job_id not in _jobs:
            abort(404)
        job = dict(_jobs[job_id])
    if page_number not in _selected_pages(job):
        abort(404)
    page_dir = _page_dir(job, page_number)
    detected = _read_json(page_dir / "detected_regions.json") or {}
    progress = _read_json(page_dir / "ocr_progress.json") or {}
    final_regions = []
    for region_dir in sorted(page_dir.glob("region_*")):
        result = _read_json(region_dir / "result.json")
        if result:
            if not result.get("contextual_evidence"):
                contextual_evidence = _read_json(region_dir / "contextual_recheck.json")
                if contextual_evidence:
                    result["contextual_evidence"] = contextual_evidence
            final_regions.append({"folder": region_dir.name, **result})
    page_log_prefix = f"[page {page_number}]"
    with _lock:
        page_log = [
            line for line in _jobs[job_id].get("log", [])
            if line.startswith(page_log_prefix)
        ][-100:]
    return jsonify({
        "page": page_number,
        "detected": detected,
        "progress": progress,
        "final_regions": final_regions,
        "log": page_log,
    })


@app.get("/api/jobs/<job_id>/pages/<int:page_number>/merged-document")
def merged_document(job_id: str, page_number: int):
    with _lock:
        job = dict(_jobs[job_id]) if job_id in _jobs else None
    if not job or page_number not in _selected_pages(job):
        abort(404)
    merged_path = _page_dir(job, page_number) / "merged.md"
    if not merged_path.exists():
        return jsonify({"error": "Merged document is not available yet."}), 404
    html = _merged_document_html(merged_path)
    return app.response_class(html, mimetype="text/html")


def _merged_document_body(merged_path: Path) -> str:
    return markdown.markdown(
        merged_path.read_text(encoding="utf-8"),
        extensions=["extra", "sane_lists", "nl2br"],
    )


def _merged_document_html(merged_path: Path) -> str:
    body = _merged_document_body(merged_path)
    return f"""<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--paper:#fff;--ink:#202020;--line:#dedede}} *{{box-sizing:border-box}}
body{{margin:0;background:#f5f5f5;color:var(--ink);font-family:"Noto Sans Hebrew","Arial Hebrew",Arial,sans-serif;line-height:1.75}}
main{{width:min(900px,calc(100% - 32px));min-height:100vh;margin:0 auto;padding:40px 52px;background:var(--paper);box-shadow:0 0 0 1px var(--line)}}
h1,h2,h3{{margin:1.4em 0 .55em;line-height:1.3}} h1{{font-size:1.65rem}} h2{{font-size:1.35rem}} h3{{font-size:1.15rem}}
p{{margin:.65em 0}} ul,ol{{padding-right:1.7em}} table{{width:100%;border-collapse:collapse}} th,td{{padding:7px 10px;border:1px solid var(--line);text-align:right}}
span[style*="color:#d00"],span[style*="color: #d00"]{{color:#c5163a!important;font-weight:700}} a{{color:inherit}}
@media(max-width:640px){{main{{width:100%;padding:24px 20px;box-shadow:none}}}}
</style></head><body><main>{body}</main></body></html>"""


def _merged_document_png(merged_path: Path) -> io.BytesIO:
    body = _merged_document_body(merged_path)
    story = fitz.Story(
        html=f'<main dir="rtl">{body}</main>',
        user_css="""
            body { color: #202020; font-family: sans-serif; line-height: 1.75; }
            main { direction: rtl; }
            h1 { font-size: 26pt; } h2 { font-size: 21pt; } h3 { font-size: 17pt; }
            p { margin: 10pt 0; } ul, ol { padding-right: 24pt; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #dedede; padding: 7pt 10pt; text-align: right; }
            span { color: #c5163a; font-weight: bold; }
        """,
    )
    pdf_output = io.BytesIO()
    writer = fitz.DocumentWriter(pdf_output)
    more = True
    while more:
        device = writer.begin_page(fitz.Rect(0, 0, 900, 1273))
        more, _ = story.place(fitz.Rect(52, 40, 848, 1233))
        story.draw(device)
        writer.end_page()
    writer.close()

    document = fitz.open(stream=pdf_output.getvalue(), filetype="pdf")
    rendered_pages = []
    for page in document:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        rendered_pages.append(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))
    document.close()

    canvas = Image.new(
        "RGB",
        (max(image.width for image in rendered_pages), sum(image.height for image in rendered_pages)),
        "white",
    )
    top = 0
    for image in rendered_pages:
        canvas.paste(image, (0, top))
        top += image.height

    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    output.seek(0)
    return output


@app.get("/api/jobs/<job_id>/pages/<int:page_number>/merged.png")
def download_merged_png(job_id: str, page_number: int):
    with _lock:
        job = dict(_jobs[job_id]) if job_id in _jobs else None
    if not job or page_number not in _selected_pages(job):
        abort(404)
    merged_path = _page_dir(job, page_number) / "merged.md"
    if not merged_path.exists():
        return jsonify({"error": "Merged document is not available yet."}), 404
    output = _merged_document_png(merged_path)
    download_name = f"{Path(job['file_name']).stem}-page-{page_number}-merged.png"
    return send_file(output, mimetype="image/png", as_attachment=True, download_name=download_name)


@app.get("/api/jobs/<job_id>/pages/<int:page_number>/golden-comparison")
def golden_comparison(job_id: str, page_number: int):
    with _lock:
        job = dict(_jobs[job_id]) if job_id in _jobs else None
    if not job or page_number not in _selected_pages(job):
        abort(404)
    result_path = _page_dir(job, page_number) / "handwriting_result.json"
    if not result_path.exists():
        return jsonify({"available": False, "reason": "Analysis is not complete yet."})
    stem = Path(job["file_name"]).stem
    golden_path = compare_golden.GOLDEN_DIR / stem / "handwriting_result.json"
    golden_data = _read_json(golden_path)
    if not golden_data or int(golden_data.get("page", 1)) != page_number:
        return jsonify({"available": False, "reason": "No human golden is available for this page."})
    comparison = compare_golden.compare_sample(stem, result_path)
    if comparison.get("error"):
        return jsonify({"available": False, "reason": comparison["error"]})
    return jsonify({
        "available": True,
        "page": page_number,
        "success_percent": round(comparison["avg_ratio"] * 100, 1),
        "exact_percent": round(comparison["strict"]["text_accuracy"] * 100, 1),
        "classification_percent": round(
            comparison["strict"]["classification_accuracy"] * 100, 1
        ),
        "matched_regions": comparison["golden_count"],
        "current_regions": comparison["current_count"],
        "rows": comparison["rows"],
    })


@app.post("/api/jobs/<job_id>/cancel")
def cancel_job(job_id: str):
    with _lock:
        if job_id not in _jobs:
            abort(404)
        processes = list(_processes.get(job_id, {}).values())
        _jobs[job_id].update(status="cancelling", stage="cancelling", message="Stopping after active calls finish")
    for process in processes:
        if process.poll() is None:
            process.terminate()
    return jsonify(_job_snapshot(job_id))


@app.get("/api/jobs/<job_id>/files/<path:file_path>")
def job_file(job_id: str, file_path: str):
    with _lock:
        if job_id not in _jobs:
            abort(404)
        output_dir = _jobs[job_id]["output_dir"]
    return send_from_directory(output_dir, file_path)


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _load_saved_jobs()
    app.run(host="127.0.0.1", port=5173, debug=False, threaded=True)