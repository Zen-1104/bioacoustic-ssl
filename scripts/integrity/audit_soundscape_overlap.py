"""
Audit exact file and filename overlaps between field recordings and training audio.
This audit catches identical files and obvious filename reuse only. 
It does not prove source independence for segments, edited excerpts, or re-encoded copies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".m4a"}


def inventory_audio(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Audio directory does not exist: '{root}'.")
        
    files = []
    for current, dirs, names in os.walk(root, followlinks = False):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (Path(current) / d).is_symlink())
        
        for name in names:
            path = Path(current) / name
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in AUDIO_SUFFIXES:
                files.append(path)
                
    return sorted(files, key = lambda p: (p.as_posix().casefold(), p.as_posix()))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            
            if not chunk:
                break
                
            digest.update(chunk)
            
    return digest.hexdigest()


def audit_overlap(soundscape_dir: str | Path, training_dirs: list[str | Path]) -> dict:
    sound_root = Path(soundscape_dir).expanduser().resolve()
    sound_files = inventory_audio(sound_root)
    if not sound_files:
        raise ValueError(f"No supported audio files found under soundscape directory '{sound_root}'.")

    train_files = []
    for raw_root in training_dirs:
        root = Path(raw_root).expanduser().resolve()
        train_files.extend(inventory_audio(root))
        
    if not train_files:
        raise ValueError("Training audio inventory is empty; refusing to report a zero overlap result.")

    sound_by_name = defaultdict(list)
    sound_by_hash = defaultdict(list)
    train_by_name = defaultdict(list)
    train_by_hash = defaultdict(list)
    
    print(f"Hashing {len(sound_files)} soundscape files and {len(train_files)} training files; this may take several minutes.", flush = True)
    
    for i, path in enumerate(sound_files, 1):
        sound_by_name[path.name.casefold()].append(path)
        sound_by_hash[sha256_file(path)].append(path)
        
        if i % 100 == 0 or i == len(sound_files):
            print(f"Soundscape files hashed: {i}/{len(sound_files)}", flush = True)
            
    for i, path in enumerate(train_files, 1):
        train_by_name[path.name.casefold()].append(path)
        train_by_hash[sha256_file(path)].append(path)
        
        if i % 500 == 0 or i == len(train_files):
            print(f"Training files hashed: {i}/{len(train_files)}", flush = True)

    basename_matches = []
    for name in sorted(set(sound_by_name) & set(train_by_name)):
        basename_matches.append({
            "basename": name,
            "soundscape_paths": [str(p) for p in sound_by_name[name]],
            "training_paths": [str(p) for p in train_by_name[name]],
        })
        
    hash_matches = []
    for digest in sorted(set(sound_by_hash) & set(train_by_hash)):
        hash_matches.append({
            "sha256": digest,
            "soundscape_paths": [str(p) for p in sound_by_hash[digest]],
            "training_paths": [str(p) for p in train_by_hash[digest]],
        })
    return {
        "schema_version": 1,
        "soundscape_dir": str(sound_root),
        "training_dirs": [str(Path(p).expanduser().resolve()) for p in training_dirs],
        "n_soundscape_files": len(sound_files),
        "n_training_files": len(train_files),
        "basename_match_group_count": len(basename_matches),
        "exact_sha256_match_group_count": len(hash_matches),
        "basename_matches": basename_matches,
        "exact_sha256_matches": hash_matches,
        "overlap_detected": bool(basename_matches or hash_matches),
        "limitations": [
            "Exact SHA-256 detects identical encoded files, not re-encoded copies.",
            "Basename matches are review flags and may be coincidental; inspect each pair.",
            "This does not detect a soundscape excerpt contained inside a longer training file.",
            "It does not establish independence from the historical BirdCLEF 2024 benchmark, which used parts of the Western Ghats collection.",
        ],
    }


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--soundscape-dir", default = "data/raw/western_ghats_soundscapes/soundscape_data")
    parser.add_argument("--training-dirs", nargs="+", default = ["data/raw/xeno_canto", "data/raw/birdclef/train_audio"])
    parser.add_argument("--output", default = "results/soundscape_overlap_audit.json")
    args = parser.parse_args(argv)
    
    result = audit_overlap(args.soundscape_dir, args.training_dirs)
    output = Path(args.output)
    output.parent.mkdir(parents = True, exist_ok = True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent = 2), encoding = "utf-8")
    os.replace(tmp, output)
    
    print(json.dumps({
        "output": str(output), 
        "soundscape_files": result["n_soundscape_files"],
        "training_files": result["n_training_files"], 
        "basename_match_groups": result["basename_match_group_count"],
        "exact_hash_match_groups": result["exact_sha256_match_group_count"], 
        "overlap_detected": result["overlap_detected"],
    }, indent = 2))


if __name__ == "__main__":
    main()
