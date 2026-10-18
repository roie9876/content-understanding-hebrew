#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


FILES = (
    "Dockerfile", ".dockerignore", "requirements.txt", "requirements-container.txt",
    "app.py", "handwriting_pipeline.py", "segmented_ocr.py", "public_demo.py",
    "container_wsgi.py", "ui/index.html", "ui/app.js", "ui/styles.css",
)


def build_context(root: Path, destination: Path) -> dict[str, str]:
    root = root.resolve(strict=True)
    destination = destination.resolve()
    if destination == root or root.is_relative_to(destination):
        raise ValueError("The build context must not replace the repository or an ancestor.")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Build context destination must be empty; refuse stale or unexpected files.")
    sources = {}
    for name in FILES:
        path = root / name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Build input is missing, outside the repository, or a symlink: {name}")
        sources[name] = path
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, path in sources.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    actual = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    if actual != set(FILES):
        raise RuntimeError("Build context does not match the explicit application allowlist.")
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage application code only, never the whole repository.")
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    hashes = build_context(Path(__file__).resolve().parents[1], args.destination)
    print(json.dumps({"context": str(args.destination.resolve()), "files": hashes}, indent=2))


if __name__ == "__main__":
    main()
