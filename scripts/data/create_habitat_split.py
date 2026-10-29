"""Create a deterministic ESC-50 seen/unseen outdoor noise category split."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Explicit whitelist avoids accidentally treating animals, indoor human actions, or unrelated household sounds as habitat noise. 
# Traffic/transport and outdoor sound events are included deliberately as field noise conditions.
HABITAT_CATEGORIES = {
    "rain", "sea_waves", "crackling_fire", "water_drops", "wind",
    "pouring_water", "thunderstorm", "helicopter", "chainsaw", "siren",
    "car_horn", "engine", "train", "airplane", "fireworks", "hand_saw",
}

# A single random draw can accidentally place almost all weather or almost all mechanical sound types into one partition. 
# Split within these broad source families to make the held out test less dominated by one kind of noise.
CATEGORY_FAMILIES = {
    "environmental_weather": {"rain", "sea_waves", "crackling_fire", "water_drops", "wind", "pouring_water", "thunderstorm"},
    "mechanical_manmade": {"helicopter", "chainsaw", "siren", "car_horn", "engine", "train", "airplane", "fireworks", "hand_saw"},
}
SPLIT_POLICY = "family_stratified_60_40_v2"


def create_habitat_split(esc50_csv_path = "data/raw/esc50/meta/esc50.csv", output_dir = "data/processed", seed = 42, overwrite = False):
    csv_path = Path(esc50_csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"ESC-50 metadata CSV not found: '{csv_path}'. Extract ESC-50 under data/raw/esc50 first.")
        
    source_metadata_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    df = pd.read_csv(csv_path)
    
    required = {"filename", "category"}
    if not required.issubset(df.columns):
        raise ValueError(f"ESC-50 metadata must include columns {sorted(required)}; got {list(df.columns)}.")
        
    found = set(df["category"].dropna().astype(str).unique())
    
    selected = sorted(found & HABITAT_CATEGORIES)
    if len(selected) < 4:
        raise ValueError(f"Only {len(selected)} approved habitat categories found: {selected}. Check the ESC-50 metadata version.")
        
    rejected = sorted(found - HABITAT_CATEGORIES)
    filtered = df[df["category"].isin(selected)].copy().sort_values(["category", "filename"])

    output = Path(output_dir)
    seen_path = output / "esc50_seen_habitat.csv"
    unseen_path = output / "esc50_unseen_habitat.csv"
    manifest_path = output / "esc50_habitat_split_manifest.json"
    
    if any(p.exists() for p in (seen_path, unseen_path, manifest_path)) and not overwrite:
        if all(p.exists() for p in (seen_path, unseen_path, manifest_path)):
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            
            if int(existing.get("seed", -1)) != int(seed):
                raise RuntimeError(
                    f"Existing habitat split uses seed={existing.get('seed')}, requested seed={seed}. "
                    "Use --overwrite only if deliberately changing the benchmark split."
                )
                
            if existing.get("source_metadata_sha256") != source_metadata_sha256:
                raise RuntimeError(
                    "Existing habitat split was built from a different ESC-50 metadata file. "
                    "Inspect the changes and use --overwrite only if deliberately rebuilding the benchmark split."
                )
                
            if sorted(existing.get("approved_categories_present", [])) != selected:
                raise RuntimeError(
                    "Existing habitat split was built from a different set of approved categories. "
                    "Inspect the metadata and use --overwrite only if deliberately rebuilding the split."
                )
                
            if existing.get("split_policy") != SPLIT_POLICY:
                raise RuntimeError(
                    f"Existing habitat split uses split_policy={existing.get('split_policy')!r}, expected {SPLIT_POLICY!r}. "
                    "Back up prior results and use --overwrite only when deliberately rebuilding the partition."
                )
                
            print(f"Preserving family stratified habitat split (seed = {seed}; manifest = {manifest_path}). Use --overwrite to redraw deliberately.")
            
            return str(seen_path), str(unseen_path)
            
        raise FileExistsError("Partial existing habitat split found. Inspect data/processed and use --overwrite only if intentionally recreating it.")

    rng = np.random.default_rng(seed)
    seen_categories: list[str] = []
    unseen_categories: list[str] = []
    
    family_by_category = {}
    for family, members in CATEGORY_FAMILIES.items():
        present = sorted(found & members & HABITAT_CATEGORIES)
        
        if not present:
            continue
            
        family_by_category.update({category: family for category in present})
        shuffled = np.asarray(present, dtype=object)
        
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            raise ValueError(f"Family '{family}' has only one approved category and cannot be split seen/unseen.")
            
        split_idx = int(round(len(shuffled) * 0.6))
        split_idx = min(max(split_idx, 1), len(shuffled) - 1)
        
        seen_categories.extend(str(c) for c in shuffled[:split_idx])
        unseen_categories.extend(str(c) for c in shuffled[split_idx:])
        
    seen_categories = sorted(seen_categories)
    unseen_categories = sorted(unseen_categories)
    
    if set(seen_categories) | set(unseen_categories) != set(selected):
        raise AssertionError("Family-stratified split failed to assign every approved category.")
        
    if set(seen_categories) & set(unseen_categories):
        raise AssertionError("Seen and unseen habitat categories overlap.")

    output.mkdir(parents = True, exist_ok = True)
    
    filtered[filtered["category"].isin(seen_categories)].to_csv(seen_path, index = False)
    filtered[filtered["category"].isin(unseen_categories)].to_csv(unseen_path, index = False)
    
    manifest = {
        "seed": int(seed),
        "source_metadata_path": str(csv_path.resolve()),
        "source_metadata_sha256": source_metadata_sha256,
        "policy": "explicit_outdoor_noise_whitelist_v1",
        "split_policy": SPLIT_POLICY,
        "category_families": family_by_category,
        "approved_categories_present": selected,
        "excluded_categories_present": rejected,
        "seen_categories": seen_categories,
        "unseen_categories": unseen_categories,
        "seen_filenames": filtered[filtered["category"].isin(seen_categories)]["filename"].tolist(),
        "unseen_filenames": filtered[filtered["category"].isin(unseen_categories)]["filename"].tolist(),
    }
    
    manifest_path.write_text(json.dumps(manifest, indent = 2), encoding = "utf-8")
    
    print(f"Approved habitat categories ({len(selected)}): {selected}")
    print(f"Seen categories ({len(seen_categories)}): {seen_categories}")
    print(f"Held-out categories ({len(unseen_categories)}): {unseen_categories}")
    print(f"Excluded from habitat pool: {rejected}")
    
    return str(seen_path), str(unseen_path)


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--metadata", default = "data/raw/esc50/meta/esc50.csv")
    parser.add_argument("--output-dir", default = "data/processed")
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--overwrite", action = "store_true")
    args = parser.parse_args(argv)
    create_habitat_split(args.metadata, args.output_dir, args.seed, args.overwrite)

if __name__ == "__main__":
    main()