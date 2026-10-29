from __future__ import annotations

import json
import math
import os
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


class DemoError(RuntimeError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class PublicDemo:
    def __init__(self, data_dir: Path):
        self.private_dir = data_dir / ".private"
        self.private_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.Lock()
        self.max_upload_bytes = self._positive("DEMO_MAX_UPLOAD_BYTES", 20 * 1024 * 1024)
        self.max_pages = self._positive("DEMO_MAX_PAGES", 2)
        self.max_regions = self._positive("DEMO_MAX_REGIONS", 40)
        self.hourly_limit = self._positive("DEMO_MAX_JOBS_PER_HOUR", 10)
        self.total_limit = self._positive("DEMO_MAX_JOBS_TOTAL", 25)
        key_path = self.private_dir / "session.key"
        if (self.private_dir / "usage.json").exists() and not key_path.is_file():
            raise RuntimeError("The persisted demo signing key is missing; refusing to reset browser ownership.")
        try:
            descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(secrets.token_bytes(32))
                stream.flush()
                os.fsync(stream.fileno())
        self.secret_key = key_path.read_bytes()
        if len(self.secret_key) != 32:
            raise RuntimeError("Invalid persisted demo signing key; refusing to reset browser ownership.")
        self.state_path = self.private_dir / "usage.json"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._validate_state()
        else:
            self.state = {"created_at": time.time(), "starts": [], "uploads": []}
            self._persist()
        expiration = os.environ.get("DEMO_EXPIRES_AT")
        if expiration:
            parsed = datetime.fromisoformat(expiration.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("DEMO_EXPIRES_AT must include a timezone.")
            self.expires_at = parsed.timestamp()
        else:
            self.expires_at = None

    @staticmethod
    def _positive(name: str, default: int) -> int:
        value = int(os.environ.get(name, str(default)))
        if value < 1:
            raise ValueError(f"{name} must be positive.")
        return value

    def _validate_state(self) -> None:
        def valid_timestamp(value):
            return type(value) in (int, float) and math.isfinite(value) and value >= 0

        if not isinstance(self.state, dict) or not valid_timestamp(self.state.get("created_at")):
            raise RuntimeError("Invalid persisted demo usage state; refusing to reset usage limits.")
        for field in ("starts", "uploads"):
            values = self.state.get(field)
            if not isinstance(values, list) or any(
                not valid_timestamp(value) for value in values
            ):
                raise RuntimeError("Invalid persisted demo usage state; refusing to reset usage limits.")

    def _persist(self) -> None:
        write_json_atomic(self.state_path, self.state)

    def admit(self, kind: str) -> None:
        with self.lock:
            now = time.time()
            if self.expires_at is not None and now >= self.expires_at:
                raise DemoError("The public demo processing window has expired. Contact the owner.", 410)
            if kind not in {"starts", "uploads"}:
                raise ValueError("Unknown demo admission type.")
            multiplier = 1 if kind == "starts" else 2
            entries = self.state[kind]
            if len(entries) >= self.total_limit * multiplier:
                raise DemoError("The public demo usage allowance is exhausted. Contact the owner.", 429)
            if sum(stamp > now - 3600 for stamp in entries) >= self.hourly_limit * multiplier:
                raise DemoError("The public demo hourly limit is reached. Please try again later.", 429)
            entries.append(now)
            self._persist()

    def public_config(self) -> dict:
        return {
            "public_demo": True,
            "max_upload_bytes": self.max_upload_bytes,
            "max_pages": self.max_pages,
            "max_regions": self.max_regions,
            "max_jobs_per_hour": self.hourly_limit,
            "max_jobs_total": self.total_limit,
            "expires_at": (
                datetime.fromtimestamp(self.expires_at, timezone.utc).isoformat()
                if self.expires_at is not None else None
            ),
        }
