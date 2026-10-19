"""Load and validate the saved ESC-50 seen/unseen habitat partition."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def read_verified_habitat_csv(
    csv_path: str | Path,
    esc50_root: str | Path,
    *,
    expected_partition: str,
) -> pd.DataFrame:
    if expected_partition not in {"seen", "unseen"}:
        raise ValueError(
            f"expected_partition must be 'seen' or 'unseen'; got {expected_partition!r}."
        )
        
    partition = expected_partition
    path = Path(csv_path)
    root = Path(esc50_root)
    
    if not path.is_file():
        raise FileNotFoundError(f"Habitat split CSV missing: '{path}'. Run scripts/create_habitat_split.py first.")
        
    manifest_path = path.parent / "esc50_habitat_split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Habitat split manifest missing: '{manifest_path}'. Recreate the split before proceeding.")
        
    import json
    manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    recorded_source = Path(manifest.get("source_metadata_path", ""))
    
    if not recorded_source.is_file():
        recorded_source = root / "meta" / "esc50.csv"
        
    source_metadata = recorded_source
    if not source_metadata.is_file():
        raise FileNotFoundError(f"ESC-50 source metadata missing: '{source_metadata}'.")

    current_hash = hashlib.sha256(source_metadata.read_bytes()).hexdigest()
    if manifest.get("source_metadata_sha256") != current_hash:
        raise RuntimeError(
            f"Habitat manifest '{manifest_path}' was built from different ESC-50 metadata. "
            "Review the metadata change and deliberately regenerate the split."
        )

    expected_files = manifest.get(f"{partition}_filenames")
    expected_categories = manifest.get(f"{partition}_categories")
    other_files = manifest.get("seen_filenames" if partition == "unseen" else "unseen_filenames")
    other_categories = manifest.get("seen_categories" if partition == "unseen" else "unseen_categories")
    
    if not isinstance(expected_files, list) or not expected_files:
        raise ValueError(f"Manifest has no non empty {partition}_filenames list.")
        
    if not isinstance(expected_categories, list) or not expected_categories:
        raise ValueError(f"Manifest has no non-empty {partition}_categories list.")
        
    if not isinstance(other_files, list):
        raise ValueError("Manifest is missing the opposite partition's filename list.")
        
    if not isinstance(other_categories, list) or not other_categories:
        raise ValueError("Manifest is missing the opposite partition's category list.")
        
    if set(expected_files) & set(other_files):
        raise ValueError("Seen and unseen habitat filename lists overlap in the manifest.")
    if set(expected_categories) & set(other_categories):
        raise ValueError("Seen and unseen habitat category lists overlap in the manifest.")

    frame = pd.read_csv(path)
    if not {"filename", "category"}.issubset(frame.columns) or frame.empty:
        raise ValueError(f"Habitat CSV '{path}' must contain non-empty filename/category columns.")
        
    filenames = frame["filename"].astype(str).tolist()
    categories = frame["category"].astype(str).tolist()
    
    if len(filenames) != len(set(filenames)):
        raise ValueError(f"Habitat CSV '{path}' contains duplicate filenames.")
        
    if set(filenames) != set(expected_files) or len(filenames) != len(expected_files):
        raise RuntimeError(
            f"Habitat CSV '{path}' does not match the saved {partition} filename manifest. "
            "Recreate the split deliberately; do not use a partial or edited CSV."
        )
        
    if set(categories) != set(expected_categories):
        raise RuntimeError(
            f"Habitat CSV '{path}' has category labels inconsistent with the saved {partition} partition."
        )

    # Resolve the category labels against the source metadata too, to catch accidental CSV edits.
    source = pd.read_csv(source_metadata, usecols=["filename", "category"])
    source = source[source["filename"].astype(str).isin(expected_files)]
    
    if len(source) != len(expected_files):
        raise RuntimeError(f"Saved habitat filenames no longer match source metadata for '{path}'.")
        
    source_map = dict(zip(source["filename"].astype(str), source["category"].astype(str)))
    if any(source_map.get(filename) != category for filename, category in zip(filenames, categories)):
        raise RuntimeError(f"Habitat CSV '{path}' labels differ from the source metadata.")
        
    return frame
    
