"""Validated ESC-50 habitat-noise pool loading shared by experiment runners."""
from __future__ import annotations

import json
from pathlib import Path

from utils.audio_io import load_and_preprocess_audio
from utils.habitat_split import read_verified_habitat_csv


def load_seen_habitat_noise_pool(split_csv_path, esc50_audio_dir, sample_rate, duration_sec):
    """Load only verified seen ESC-50 categories for training augmentation."""
    split_path = Path(split_csv_path)
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Seen-habitat split CSV missing: '{split_path}'. Run scripts/create_habitat_split.py first."
        )
        
    frame = read_verified_habitat_csv(split_path, esc50_audio_dir, expected_partition = "seen")
    
    manifest_path = split_path.parent / "esc50_habitat_split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Habitat split manifest is missing: '{manifest_path}'.")
        
    manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    unseen_categories = set(map(str, manifest.get("unseen_categories", [])))
    pool_categories = set(frame["category"].astype(str)) if "category" in frame.columns else set()
    
    leaked = sorted(unseen_categories & pool_categories)
    if leaked:
        raise RuntimeError(f"Seen habitat noise pool contains unseen categories: {leaked}.")
        
    if frame.empty or "filename" not in frame:
        raise ValueError(f"Seen habitat CSV '{split_path}' contains no usable filenames.")

    root = Path(esc50_audio_dir)
    pool = []
    
    for filename in sorted(frame["filename"].astype(str).tolist()):
        candidates = (root / filename, root / "audio" / filename)
        audio_path = next((path for path in candidates if path.is_file()), None)
        
        if audio_path is None:
            raise FileNotFoundError(f"ESC-50 noise file '{filename}' not found under '{root}'.")
            
        pool.append(load_and_preprocess_audio(
            audio_path, target_sr = int(sample_rate), 
            duration_sec = float(duration_sec),
            crop_mode = "center", 
            short_audio_policy = "repeat",
        ))
        
    if not pool:
        raise RuntimeError(f"No noise files loaded from '{split_path}'.")
        
    return pool
