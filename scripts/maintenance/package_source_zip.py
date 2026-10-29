#!/usr/bin/env python3
"""Build a deterministic, lightweight source ZIP from this repository.

Run from any directory:
    python3 scripts/maintenance/package_source_zip.py --output /path/to/bioacoustic-ssl-codebase-context-final.zip

The allowlist deliberately selects source/config/docs by type and directory;
results data, datasets, checkpoints, caches, environments, secrets, and unrelated
parent-directory files are never traversed into the archive.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import stat
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = {
    "bioacoustic_data": {".py"},
    "configs": {".yaml", ".yml"},
    "evaluation": {".py"},
    "experiments": {".py"},
    "models": {".py"},
    "results": {".py"},
    "scripts": {".py", ".sh", ".slurm"},
    "tests": {".py"},
    "utils": {".py", ".sh", ".slurm"},
}
ROOT_FILES = {"README.md", "requirements.txt", ".gitignore", "check_env.py", "SOURCE_ARCHIVE.md"}
DOC_DIRS = {"docs", "documentation"}
EXCLUDED_PARTS = {
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "venv", ".venv", "env", "node_modules", "data", "checkpoints",
    "diagnostic_archive", "logs", "tmp", "temp",
}
EXCLUDED_SUFFIXES = {
    ".pyc", ".pyo", ".bak", ".tmp", ".swp", ".swo", ".pt", ".pth", ".ckpt",
    ".npz", ".npy", ".wav", ".ogg", ".flac", ".mp3", ".m4a", ".csv", ".jsonl",
}
SECRET_NAMES = {".env", ".env.local", "credentials", "credentials.json", "secrets.json"}
FIXED_ZIP_TIME = (2020, 1, 1, 0, 0, 0)


def selected_files() -> list[Path]:
    chosen: set[Path] = set()
    for name in ROOT_FILES:
        path = ROOT / name
        if path.is_file():
            chosen.add(path)
    for dirname, extensions in SOURCE_DIRS.items():
        directory = ROOT / dirname
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if any(part in EXCLUDED_PARTS for part in rel.parts):
                continue
            if path.suffix.lower() not in extensions or path.name.lower() in SECRET_NAMES:
                continue
            chosen.add(path)
    for dirname in DOC_DIRS:
        directory = ROOT / dirname
        if directory.is_dir():
            for path in directory.rglob("*.md"):
                if path.is_file() and not any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
                    chosen.add(path)
    for path in chosen:
        rel = path.relative_to(ROOT)
        if path.name.lower() in SECRET_NAMES or path.suffix.lower() in EXCLUDED_SUFFIXES:
            raise ValueError(f"Refusing to package excluded/private file: {rel}")
        if any(part in EXCLUDED_PARTS for part in rel.parts):
            raise ValueError(f"Refusing to package excluded directory content: {rel}")
    return sorted(chosen, key=lambda p: p.relative_to(ROOT).as_posix())


def build(output: Path) -> tuple[int, int]:
    output = output.expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Output ZIP must be outside the source repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = selected_files()
    if not files:
        raise RuntimeError("No source files selected")
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            arcname = path.relative_to(ROOT).as_posix()
            info = ZipInfo(arcname, date_time=FIXED_ZIP_TIME)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, path.read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=9)
    return len(files), output.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="ZIP path outside this repository")
    args = parser.parse_args()
    count, size = build(args.output)
    print(f"Created {args.output.expanduser().resolve()} ({count} files, {size:,} bytes)")


if __name__ == "__main__":
    main()
