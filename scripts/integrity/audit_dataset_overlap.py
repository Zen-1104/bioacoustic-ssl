"""Audit BirdCLEF/Xeno-Canto overlap using catalogue IDs plus exact file hashes.

BirdCLEF audio may re-encode/resample Xeno-Canto recordings, so byte hashes alone
are insufficient. Filenames with XC<digits> are compared by catalogue ID first;
SHA-256 remains a secondary exact-duplicate check for files without IDs.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.dataset_manifest import fingerprint_relative_paths

EXTS = {".wav", ".mp3", ".flac", ".ogg"}


def audio_files(root: Path):
    for current_root, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))

        for name in sorted(files):
            path = Path(current_root) / name
            if path.suffix.lower() in EXTS:
                yield path


def xeno_canto_id(path: Path) -> str | None:
    match = re.match(r"(?i)^XC[_ -]?(\d+)(?:\D.*)?$", path.stem.strip())
    return match.group(1) if match else None


def sha256_file(path: Path, chunk_bytes = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
            
    return digest.hexdigest()


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--xeno-dir", default = "data/raw/xeno_canto")
    parser.add_argument("--birdclef-dir", default = "data/raw/birdclef")
    parser.add_argument("--output", default = "data/processed/xeno_canto_excluded.json")
    args = parser.parse_args(argv)
    
    xeno_root, bird_root = Path(args.xeno_dir).resolve(), Path(args.birdclef_dir).resolve()
    if not xeno_root.is_dir() or not bird_root.is_dir():
        raise FileNotFoundError(f"Both dataset roots must exist. Xeno-Canto = {xeno_root.exists()}, BirdCLEF = {bird_root.exists()}.")

    bird_audio_root = bird_root / "train_audio" if (bird_root / "train_audio").is_dir() else bird_root
    class_dirs = sorted(
        (p for p in bird_audio_root.iterdir() if p.is_dir() and not p.name.startswith(".")),
        key = lambda p: (p.name.casefold(), p.name),
    )
    
    # Match BirdCLEFDataset's class discovery semantics so its audit fingerprint
    # and the training dataset fingerprint refer to exactly the same audio paths.
    if class_dirs:
        bird_paths = [p for class_dir in class_dirs for p in audio_files(class_dir)]
    else:
        bird_paths = [p for p in bird_audio_root.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
        
    bird_paths = sorted(bird_paths, key = lambda p: (p.relative_to(bird_root).as_posix().casefold(), p.relative_to(bird_root).as_posix()))
    xeno_paths = sorted(audio_files(xeno_root), key = lambda p: (p.relative_to(xeno_root).as_posix().casefold(), p.relative_to(xeno_root).as_posix()))
    bird_relative_paths = [p.relative_to(bird_root).as_posix() for p in bird_paths]
    xeno_relative_paths = [p.relative_to(xeno_root).as_posix() for p in xeno_paths]
    bird_hashes: dict[str, list[str]] = {}
    bird_ids: dict[str, list[str]] = {}
    
    for p in bird_paths:
        bird_rel = p.relative_to(bird_root).as_posix()
        bird_hashes.setdefault(sha256_file(p), []).append(bird_rel)
        recording_id = xeno_canto_id(p)
        
        if recording_id is not None:
            bird_ids.setdefault(recording_id, []).append(bird_rel)

    excluded: list[str] = []
    duplicates: list[dict[str, object]] = []
    id_matches = 0
    hash_matches = 0
    xeno_ids_found = 0
    xeno_count = 0
    
    for p in xeno_paths:
        xeno_count += 1
        rel = p.relative_to(xeno_root).as_posix()
        recording_id = xeno_canto_id(p)
        
        if recording_id is not None:
            xeno_ids_found += 1
            
        id_match_paths = bird_ids.get(recording_id, []) if recording_id is not None else []
        digest = None
        hash_match_paths = []
        
        if not id_match_paths:
            digest = sha256_file(p)
            hash_match_paths = bird_hashes.get(digest, [])
            
        if id_match_paths or hash_match_paths:
            excluded.append(rel)
            
            if id_match_paths:
                id_matches += 1
                duplicates.append({
                    "xeno_canto": rel, 
                    "xeno_canto_id": recording_id,
                    "birdclef_matches": id_match_paths, 
                    "match_method": "xeno_canto_catalogue_id",
                })
            else:
                hash_matches += 1
                duplicates.append({
                    "xeno_canto": rel, 
                    "birdclef_matches": hash_match_paths,
                    "sha256": digest, 
                    "match_method": "byte_identical_sha256",
                })

    output = Path(args.output)
    output.parent.mkdir(parents = True, exist_ok = True)
    
    payload = {
        "method": "xeno_canto_catalogue_id_then_byte_identical_sha256",
        "limitation": "Catalogue-ID matching requires XC<digits> filenames. Re-encoded/trimmed recordings without catalogue-ID filenames may still evade detection; inspect filenames/metadata before reporting no overlap.",
        "xeno_canto_file_count": xeno_count,
        "xeno_canto_paths_sha256": fingerprint_relative_paths(xeno_relative_paths),
        "birdclef_file_count": len(bird_relative_paths),
        "birdclef_paths_sha256": fingerprint_relative_paths(bird_relative_paths),
        "birdclef_unique_hash_count": len(bird_hashes),
        "birdclef_xeno_canto_id_count": len(bird_ids),
        "xeno_canto_filenames_with_ids": xeno_ids_found,
        "catalogue_id_duplicate_count": id_matches,
        "byte_identical_duplicate_count": hash_matches,
        "exact_duplicate_count": hash_matches,
        "excluded_overlap_count": len(excluded),
        "excluded_relative_paths": sorted(excluded),
        "matches": duplicates,
    }
    
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent = 2), encoding = "utf-8")
    os.replace(tmp, output)
    
    print(f"Scanned {xeno_count} Xeno-Canto audio files.")
    print(f"Excluded {id_matches} catalogue-ID overlaps and {hash_matches} byte identical overlaps ({len(excluded)} unique Xeno-Canto files).")
    print(f"Xeno-Canto filenames with parseable XC IDs: {xeno_ids_found}/{xeno_count}.")
    print(f"Wrote {output}.")
    
    if xeno_count and xeno_ids_found == 0:
        print("WARNING: no XC catalogue IDs parsed from Xeno-Canto filenames; inspect metadata/file naming before claiming overlap was audited.")
    if len(excluded) == 0:
        print("No ID/hash matches found; re-encoded/trimmed overlap remains possible.")


if __name__ == "__main__":
    main()
