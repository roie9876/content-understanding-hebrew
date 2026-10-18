#!/usr/bin/env python3
"""Audit a code-only context, then every layer of a locally available image."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

from dotenv import dotenv_values

if __package__:
    from .build_context import FILES
else:
    from build_context import FILES


PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\r?\n[A-Za-z0-9+/=]{16,}"),
    re.compile(rb"\b(?:ghp|gho|ghs|ghr)_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(rb"\bDefaultEndpointsProtocol=[^\r\n]{0,1000}AccountKey=[A-Za-z0-9+/=]{20,}"),
    re.compile(rb"\bAccountEndpoint=https://[^\r\n]{0,1000}AccountKey=[A-Za-z0-9+/=]{20,}"),
    re.compile(rb"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|rediss?)://[^ \r\n\"']+:[^ \r\n\"']+@"),
    re.compile(rb"\b(?:Server|Data Source)=[^\r\n]{0,1000}(?:Password|Pwd)=[^;\"'\r\n]{6,}", re.I),
)
SECRET_NAME = re.compile(r"SECRET|PASSWORD|TOKEN|CONNECTION.?STRING|API.?KEY|ACCESS.?KEY|IDENTITY_HEADER", re.I)


def digest(stream) -> str:
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(chunk)
    return result.hexdigest()


def local_credentials(root: Path) -> list[bytes]:
    values = dotenv_values(root / ".env", interpolate=False)
    return [
        value.encode() for key, value in values.items()
        if value and len(value) >= 8 and SECRET_NAME.search(key) and "<" not in value
    ]


def suspicious(data: bytes, credentials: list[bytes]) -> bool:
    return any(pattern.search(data) for pattern in PATTERNS) or any(
        value in data for value in credentials
    )


def audit_context(context: Path, credentials: list[bytes]) -> dict:
    actual = {p.relative_to(context).as_posix() for p in context.rglob("*") if p.is_file()}
    if actual != set(FILES) or any(p.is_symlink() for p in context.rglob("*")):
        raise RuntimeError("Build context failed the exact code-file allowlist.")
    matches = sum(suspicious((context / name).read_bytes(), credentials) for name in FILES)
    if matches:
        raise RuntimeError("Potential credentials found in build inputs; values withheld.")
    dockerfile = (context / "Dockerfile").read_text()
    for line in dockerfile.splitlines():
        if re.match(r"^\s*(?:ADD|ARG)\s", line, re.I):
            raise RuntimeError("Unreviewed ADD/build arguments are not permitted in the image.")
        if re.match(r"^\s*COPY\s", line, re.I):
            for name in line.split()[1:-1]:
                if name not in FILES:
                    raise RuntimeError("Dockerfile COPY is broader than the reviewed code allowlist.")
    return {
        "passed": True,
        "file_count": len(actual),
        "allowlisted_files": sorted(actual),
        "potential_credentials_found": 0,
        "local_credential_values_printed_or_exported": False,
        "data_files_in_context": 0,
    }


def private_fingerprints(root: Path) -> set[str]:
    root = root.resolve(strict=True)
    candidates = [p for p in root.glob(".env*") if p.name != ".env.example"]
    for pattern in ("*.pdf", "*.pem", "*.key", "*.pfx"):
        candidates.extend(root.glob(pattern))
    for directory in ("sample", "golden", "analysis_results", ".ui_data", "architecture"):
        folder = root / directory
        if folder.is_dir():
            candidates.extend(p for p in folder.rglob("*") if p.is_file())
    fingerprints = set()
    for path in candidates:
        if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root) \
                and path.stat().st_size >= 64:
            with path.open("rb") as stream:
                fingerprints.add(digest(stream))
    return fingerprints


def docker_json(*arguments: str):
    result = subprocess.run(["docker", *arguments], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Docker image audit command failed; build/pull the intended image first.")
    return json.loads(result.stdout)


def audit_image(
    image: str, root: Path, credentials: list[bytes], expected_hashes=None,
    source_archive=None, reviewed_public_material=None,
) -> dict:
    details = docker_json("image", "inspect", image)[0]
    configuration = details["Config"]
    if details["Architecture"] != "amd64" or configuration.get("User") not in {"10001", "10001:10001"}:
        raise RuntimeError("The image does not match the approved architecture/non-root runtime.")
    if suspicious(json.dumps(configuration).encode(), credentials):
        raise RuntimeError("Potential credentials in image configuration; values withheld.")
    for item in configuration.get("Env") or []:
        key, _, value = item.partition("=")
        if value and (SECRET_NAME.search(key) or key == "AZURE_CONFIG_DIR"):
            raise RuntimeError("A credential-related build-time environment variable was found; value withheld.")
    history = subprocess.run(
        ["docker", "history", "--no-trunc", "--format", "{{json .CreatedBy}}", image],
        capture_output=True, check=True,
    ).stdout
    if suspicious(history, credentials):
        raise RuntimeError("Potential credentials in image history; values withheld.")
    fingerprints = private_fingerprints(root)
    if expected_hashes is not None:
        fingerprints.difference_update(expected_hashes.values())
    runtime_files = set(FILES) - {"Dockerfile", ".dockerignore"}
    reviewed = reviewed_public_material or {}
    if reviewed and not image.endswith("@" + reviewed.get("image_digest", "")):
        raise RuntimeError("Reviewed public-library material belongs to a different image digest.")
    public_files = {entry["path"]: entry for entry in reviewed.get("files", [])}
    public_material_found = []
    seen = set()
    checked_files = 0
    layers = 0
    with tempfile.TemporaryDirectory(prefix="inksight-image-audit-") as directory:
        archive_path = Path(source_archive) if source_archive else Path(directory) / "image.tar"
        if source_archive is None:
            subprocess.run(["docker", "image", "save", "--output", str(archive_path), image], check=True)
        with tarfile.open(archive_path) as archive:
            manifest_stream = archive.extractfile("manifest.json")
            if manifest_stream is None:
                raise RuntimeError("Image archive has no Docker manifest.")
            with manifest_stream:
                manifest = json.load(manifest_stream)
            for layer_name in dict.fromkeys(
                name for entry in manifest for name in entry["Layers"]
            ):
                layer_stream = archive.extractfile(layer_name)
                if layer_stream is None:
                    raise RuntimeError("Image archive is missing a referenced layer.")
                layers += 1
                with layer_stream, tarfile.open(fileobj=layer_stream, mode="r|*") as layer:
                    for member in layer:
                        name = member.name.removeprefix("./")
                        if member.issym() or member.islnk():
                            if name.startswith(("app/", "data/")):
                                raise RuntimeError("Unexpected application/data symlink in an image layer.")
                        if not member.isfile():
                            continue
                        if name.startswith("data/"):
                            raise RuntimeError("Runtime data was found baked into an image layer.")
                        if name.startswith("app/"):
                            relative = name[4:]
                            if relative not in runtime_files:
                                raise RuntimeError("Unexpected application file in an image layer; name withheld.")
                            seen.add(relative)
                        content = layer.extractfile(member)
                        if content is None:
                            raise RuntimeError("Cannot inspect an image-layer file.")
                        with content:
                            file_digest = hashlib.sha256()
                            tail = b""
                            observed_patterns = set()
                            overlap = max(4096, max(map(len, credentials), default=0))
                            for chunk in iter(lambda: content.read(1024 * 1024), b""):
                                file_digest.update(chunk)
                                data = tail + chunk
                                patterns = {i for i, pattern in enumerate(PATTERNS) if pattern.search(data)}
                                known_value = any(value in data for value in credentials)
                                entry = public_files.get(name)
                                if known_value or (
                                    patterns and (
                                        entry is None or not patterns.issubset(set(entry["pattern_indexes"]))
                                    )
                                ):
                                    raise RuntimeError("Potential credentials in an image layer; values withheld.")
                                observed_patterns.update(patterns)
                                tail = (tail + chunk)[-overlap:]
                            file_hash = file_digest.hexdigest()
                        if observed_patterns:
                            if file_hash != public_files[name]["sha256"]:
                                raise RuntimeError("Flagged library bytes differ from the reviewed public baseline.")
                            public_material_found.append(public_files[name])
                        if file_hash in fingerprints:
                            raise RuntimeError("A local private file was found in an image layer; details withheld.")
                        if name.startswith("app/") and expected_hashes is not None \
                                and file_hash != expected_hashes[name[4:]]:
                            raise RuntimeError("Image application bytes differ from the audited build context.")
                        checked_files += 1
    if seen != runtime_files:
        raise RuntimeError("The built image is missing reviewed application files.")
    return {
        "passed": True, "image_id": details["Id"], "architecture": details["Architecture"],
        "runtime_user": configuration.get("User"), "layers_checked": layers,
        "files_hashed_across_all_layers": checked_files,
        "local_private_file_matches": 0, "credential_environment_or_history_matches": 0,
        "runtime_data_baked_into_image": False,
        "private_fingerprints_stay_local": True,
        "application_bytes_match_audited_context": expected_hashes is not None,
        "reviewed_public_dependency_material": public_material_found,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--image")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--reviewed-public-material", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    args.report.write_text(json.dumps({
        "context": {"passed": False, "status": "audit_in_progress"},
        "image": {"passed": False, "status": "audit_in_progress"},
    }, indent=2) + "\n", encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    credentials = local_credentials(root)
    context_report = audit_context(args.context, credentials)
    expected = {
        name: hashlib.sha256((args.context / name).read_bytes()).hexdigest()
        for name in FILES if name not in {"Dockerfile", ".dockerignore"}
    }
    reviewed = json.loads(args.reviewed_public_material.read_text()) if args.reviewed_public_material else None
    report = {
        "context": context_report,
        "image": audit_image(args.image, root, credentials, expected, args.archive, reviewed) if args.image else {
            "passed": False, "status": "not_built_or_not_yet_audited",
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
