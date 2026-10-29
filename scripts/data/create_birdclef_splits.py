"""Create stable, reusable BirdCLEF train/validation/test and nested label fraction splits."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Keep the documented direct invocation (`python scripts/data/create_birdclef_splits.py`) working.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from bioacoustic_data import BirdCLEFDataset
from bioacoustic_data.noise_mixer import validate_minimum_mixable_rms
from utils.audio_io import DEFAULT_EVALUATION_MINIMUM_RMS
from utils.dataset_manifest import fingerprint_paths_labels
from scripts.preflight.preflight_birdclef_audio import (
    DEFAULT_MAX_EXCLUDED_FRACTION,
    DEFAULT_MAX_READ_ERROR_FRACTION,
    run_or_reuse_preflight,
)

LOG = logging.getLogger("create_birdclef_splits")
FRACTIONS = (0.01, 0.05, 0.10, 1.0)


def canonical_fraction(value: float) -> str:
    return f"{value:.2f}" if value < 1 else "1.00"


def fingerprint_dataset(relative_paths: list[str], labels: list[int]) -> str:
    return fingerprint_paths_labels(relative_paths, labels)


def _check_classes(indices: list[int], labels: list[int], expected: set[int], name: str) -> None:
    present = {labels[i] for i in indices}
    missing = expected - present
    if missing:
        raise ValueError(f"{name} split is missing {len(missing)} species labels (indices {sorted(missing)[:20]}). "
                         "Use more recording groups or inspect the metadata/group split.")


def split_indices(
    labels: list[int], groups: list[str], seed: int, grouping_source: str,
    *, min_recordings: int = 5, min_groups: int = 5, rare_class_policy: str = "exclude",
    fraction_seed: int | None = None,
    eligible_indices: list[int] | None = None,
):
    """Create train/validation/test splits with explicit rare-class policy.

    rare_class_policy = 'exclude' - omits under supported classes from all split
    lists and records those exclusions in the caller's manifest. 
    It never silently falls back from group level to file level splitting.
    """
    if not labels:
        raise ValueError("Cannot split an empty BirdCLEF dataset.")
    if len(labels) != len(groups):
        raise ValueError(f"labels/groups length mismatch: {len(labels)} labels vs {len(groups)} groups.")
        
    if min_recordings < 3 or min_groups < 3:
        raise ValueError("min_recordings and min_groups must be at least 3 to allocate train/validation/test samples.")
    if rare_class_policy not in {"error", "exclude"}:
        raise ValueError("rare_class_policy must be 'error' or 'exclude'.")

    labels_arr = np.asarray(labels)
    groups_arr = np.asarray(groups)
    if eligible_indices is None:
        eligible_indices = list(range(len(labels)))
        
    if not eligible_indices or len(set(eligible_indices)) != len(eligible_indices):
        raise ValueError("eligible_indices must contain unique indices and at least one eligible recording.")
    if any(index < 0 or index >= len(labels) for index in eligible_indices):
        raise ValueError("eligible_indices contains an index outside the dataset.")
        
    eligible_set = set(eligible_indices)
    all_labels = sorted({labels[i] for i in eligible_indices})
    class_counts = {label: sum(labels[i] == label for i in eligible_indices) for label in all_labels}
    
    # Missing metadata is represented by `missing:<relative_path>` singleton groups to prevent accidental train/test leakage. 
    # Those synthetic IDs are NOT evidence of independent recordists/locations for the rare class gate.
    class_group_counts = {
        label: len({
            groups[i] for i, item_label in enumerate(labels)
            if i in eligible_set and item_label == label and not str(groups[i]).startswith("missing:")
        })
        for label in all_labels
    }
    
    use_groups = grouping_source != "file_level_no_metadata"
    reasons: dict[int, list[str]] = {}
    for label in all_labels:
        if class_counts[label] < min_recordings:
            reasons.setdefault(label, []).append(f"recordings<{min_recordings}")
            
        if use_groups and class_group_counts[label] < min_groups:
            reasons.setdefault(label, []).append(f"distinct_groups<{min_groups}")

    if reasons and rare_class_policy == "error":
        preview = "; ".join(
            f"label {label}: {class_counts[label]} recordings, {class_group_counts[label]} groups ({','.join(why)})"
            for label, why in list(sorted(reasons.items()))[:25]
        )
        
        raise ValueError(
            f"Every active species needs at least {min_recordings} recordings"
            + (f" and {min_groups} distinct groups" if use_groups else "")
            + ". Under-supported species cannot be split into train/validation/test subsets: " + preview + ".  "
            "Use --rare-class-policy exclude to create a documented class scope, or acquire more independent recordings/groups."
        )

    excluded_labels = set(reasons) if rare_class_policy == "exclude" else set()
    active_labels = set(all_labels) - excluded_labels
    if not active_labels:
        raise ValueError("Rare class policy excluded every species; lower the thresholds or acquire more data.")
        
    active_indices = np.asarray([i for i in eligible_indices if labels[i] in active_labels], dtype = int)
    expected = set(active_labels)
    rng_labels = labels_arr[active_indices]

    if use_groups:
        unique_group_count = len(set(groups[i] for i in active_indices))
        
        if unique_group_count < 5:
            raise ValueError(
                f"Grouping metadata '{grouping_source}' leaves only {unique_group_count} unique groups after class scope filtering; "
                "at least 5 are required. Do not silently fall back to file level splitting."
            )
            
        try:
            first = StratifiedGroupKFold(
                n_splits=5, shuffle=True, random_state=seed
            )
            outer_splits = list(
                first.split(active_indices, rng_labels, groups_arr[active_indices])
            )
        except ValueError as exc:
            raise ValueError(
                f"Stratified group splitting by '{grouping_source}' failed for seed {seed}: {exc}. "
                "Inspect group coverage, thresholds, and the explicit class scope."
            ) from exc

        selected = None
        best_missing_count = None
        best_detail = None
        checked_combinations = 0

        for test_fold, (remaining_pos, test_pos) in enumerate(outer_splits, start=1):
            remaining = active_indices[remaining_pos]
            test_candidate = active_indices[test_pos]

            second = StratifiedGroupKFold(
                n_splits = 4, 
                shuffle = True, 
                random_state = seed + 1
            )
            try:
                inner_splits = list(
                    second.split(remaining, labels_arr[remaining], groups_arr[remaining])
                )
            except ValueError:
                continue

            for validation_fold, (train_rel, val_rel) in enumerate(inner_splits, start=1):
                checked_combinations += 1
                train_candidate = remaining[train_rel]
                val_candidate = remaining[val_rel]

                missing_by_split = {
                    "train": expected - set(labels_arr[train_candidate].tolist()),
                    "validation": expected - set(labels_arr[val_candidate].tolist()),
                    "test": expected - set(labels_arr[test_candidate].tolist()),
                }
                missing_count = sum(len(values) for values in missing_by_split.values())

                if best_missing_count is None or missing_count < best_missing_count:
                    best_missing_count = missing_count
                    best_detail = (
                        test_fold,
                        validation_fold,
                        missing_count,
                        {
                            name: sorted(int(label) for label in values)
                            for name, values in missing_by_split.items()
                            if values
                        },
                    )

                if missing_count != 0:
                    continue

                train_group_ids = set(groups_arr[train_candidate].tolist())
                val_group_ids = set(groups_arr[val_candidate].tolist())
                test_group_ids = set(groups_arr[test_candidate].tolist())

                if (
                    train_group_ids & val_group_ids
                    or train_group_ids & test_group_ids
                    or val_group_ids & test_group_ids
                ):
                    continue

                selected = (
                    train_candidate, 
                    val_candidate, 
                    test_candidate,
                    test_fold, 
                    validation_fold,
                )
                break

            if selected is not None:
                break

        if selected is None:
            if best_detail is None:
                detail = "No nested fold combinations could be evaluated."
            else:
                detail = (
                    f"Best candidate: test fold {best_detail[0]}, "
                    f"validation fold {best_detail[1]}, "
                    f"{best_detail[2]} missing class-partition entries; "
                    f"missing labels: {best_detail[3]}."
                )
            raise ValueError(
                f"No valid grouped split for '{grouping_source}' after checking "
                f"{checked_combinations} fold combinations for seed {seed}. "
                f"{detail} Author separation and species coverage remain enforced."
            )

        train, val, test, test_fold, validation_fold = selected
        train, val, test = train.tolist(), val.tolist(), test.tolist()

        LOG.info(
            "Selected grouped folds: test %d/5, validation %d/4; "
            "%d active species covered without author-group leakage.",
            test_fold, validation_fold, len(expected),
        )

        method = f"stratified_group_{grouping_source}"
        train_groups = {groups[i] for i in train}
        val_groups = {groups[i] for i in val}
        test_groups = {groups[i] for i in test}
        
        if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
            raise AssertionError("Group leakage detected across train/validation/test splits.")
            
    else:
        LOG.warning("No usable recordist/location metadata detected. Using stratified file level splits; do not describe these as recordist independent.")
        
        train_val, test = train_test_split(
            active_indices, test_size = 0.20, random_state = seed, stratify = rng_labels
        )
        train, val = train_test_split(
            train_val, test_size = 0.25, random_state = seed + 1, stratify = labels_arr[train_val]
        )
        
        train, val, test = train.tolist(), val.tolist(), test.tolist()
        method = "stratified_file_level_no_group_metadata"

    for split_name, idx in (("train", train), ("validation", val), ("test", test)):
        _check_classes(idx, labels, expected, split_name)
        
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise AssertionError("Train/validation/test splits overlap.")

    rng = np.random.default_rng(seed if fraction_seed is None else fraction_seed)
    train_by_label: dict[int, list[int]] = {}
    
    for label in sorted(expected):
        class_idx = np.asarray([i for i in train if labels[i] == label], dtype = int)
        rng.shuffle(class_idx)
        train_by_label[label] = class_idx.tolist()
        
        if not train_by_label[label]:
            raise ValueError(f"No training items for active class {label} after splitting.")

    fraction_map: dict[str, list[int]] = {}
    previous_set: set[int] = set()
    
    for frac in FRACTIONS:
        selected: list[int] = []
        if frac == 1.0:
            selected = list(train)
        else:
            for label in sorted(expected):
                class_idx = train_by_label[label]
                count = max(1, int(np.ceil(len(class_idx) * frac)))
                selected.extend(class_idx[:count])
                
        selected = sorted(set(selected))
        selected_set = set(selected)
        
        if not previous_set.issubset(selected_set):
            raise AssertionError("Nested label fraction subsets are not nested.")
            
        previous_set = selected_set
        fraction_map[canonical_fraction(frac)] = selected
        
    return train, val, test, fraction_map, method


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--audio-dir", default = "data/raw/birdclef")
    parser.add_argument("--output-dir", default = "checkpoints")
    parser.add_argument("--sample-rate", type=int, default = 16000)
    parser.add_argument("--clip-sec", type=float, default = 10.0)
    parser.add_argument("--seeds", type=int, nargs="+", default = [42, 123, 999], help = "Model/pretraining seeds to create manifests for.")
    parser.add_argument("--split-seed", type=int, default = 42, help = "Fixed seed for one shared data split across all model seeds.")
    parser.add_argument("--min-recordings", type=int, default = 5, help = "Minimum recordings per species for a three way split.")
    parser.add_argument("--min-groups", type=int, default = 5, help = "Minimum distinct author/location groups per species for grouped splits.")
    parser.add_argument("--rare-class-policy", choices = ("error", "exclude"), default = "exclude", help = "Policy for under supported species; default exclude records the resulting active class scope in each manifest.")
    parser.add_argument("--silence-manifest", default = "data/processed/birdclef_audio_preflight.json", help = "Cached center crop silence preflight; generated once and reused when source inventory/settings match.")
    parser.add_argument("--minimum-rms", type=float, default = DEFAULT_EVALUATION_MINIMUM_RMS, help = "Minimum RMS for the shared fixed length evaluation crop policy; must be compatible with NoiseMixer.")
    parser.add_argument("--max-read-error-fraction", type=float, default = DEFAULT_MAX_READ_ERROR_FRACTION,
                        help = "Fail closed if read/decode errors exceed this share of BirdCLEF files (default: 0.01).")
    parser.add_argument("--max-excluded-fraction", type=float, default = DEFAULT_MAX_EXCLUDED_FRACTION,
                        help = "Fail closed if all silent/unusable/read-error exclusions exceed this share (default: 0.05).")
    parser.add_argument("--overwrite-silence-manifest", action = "store_true", help = "Rescan BirdCLEF crop usability; required after changing relevant audio files/settings.")
    parser.add_argument("--overwrite", action = "store_true", help = "Replace existing manifests (changes the fixed benchmark split).")
    args = parser.parse_args(argv)
    
    try:
        args.minimum_rms = validate_minimum_mixable_rms(args.minimum_rms)
    except ValueError as exc:
        parser.error(str(exc))
        
    logging.basicConfig(level = logging.INFO, format = "[%(levelname)s] %(message)s")

    dataset = BirdCLEFDataset(args.audio_dir, sample_rate = args.sample_rate, duration_sec = args.clip_sec, crop_mode = "center")
    if not dataset.file_list:
        raise RuntimeError("BirdCLEF dataset has no usable records.")
        
    silence_preflight = run_or_reuse_preflight(
        dataset, 
        sample_rate = args.sample_rate, 
        clip_sec = args.clip_sec,
        minimum_rms = args.minimum_rms, 
        output_path = args.silence_manifest,
        overwrite = args.overwrite_silence_manifest,
        max_read_error_fraction = args.max_read_error_fraction,
        max_excluded_fraction = args.max_excluded_fraction,
    )
    
    if not silence_preflight.get("thresholds_passed", False):
        raise RuntimeError("BirdCLEF preflight manifest has not passed the configured exclusion thresholds.")
        
    silent_paths = set(silence_preflight["silent_or_unusable_filenames"])
    preflight_read_error_paths = set(silence_preflight.get("read_error_filenames", []))
    preflight_excluded_paths = silent_paths | preflight_read_error_paths
    path_to_index = {name: i for i, name in enumerate(dataset.relative_paths)}
    
    unknown_preflight = preflight_excluded_paths - set(path_to_index)
    if unknown_preflight:
        raise RuntimeError(f"Audio preflight refers to unknown paths: {sorted(unknown_preflight)[:5]}")
        
    eligible_indices = [i for i, name in enumerate(dataset.relative_paths) if name not in preflight_excluded_paths]
    if not eligible_indices:
        raise RuntimeError("BirdCLEF silence preflight excluded every recording.")
        
    ds_fingerprint = fingerprint_dataset(dataset.relative_paths, dataset.labels)
    os.makedirs(args.output_dir, exist_ok = True)
    index_paths = dataset.relative_paths

    for seed in args.seeds:
        manifest_path = Path(args.output_dir) / f"data_split_s{seed}.json"
        compatibility_path = Path(args.output_dir) / f"test_split_s{seed}.json"
        
        if manifest_path.exists() and not args.overwrite:
            with manifest_path.open(encoding = "utf-8") as f:
                old = json.load(f)
                
            if int(old.get("schema_version", -1)) != 5:
                raise RuntimeError(f"Existing split '{manifest_path}' uses an old schema/crop/RMS compatibility policy. Back it up and rebuild with v1.6.4 using --overwrite deliberately.")
            if old.get("dataset_fingerprint") != ds_fingerprint:
                raise RuntimeError(f"Existing split '{manifest_path}' was built for a different dataset ordering/content. Review it and rerun with --overwrite deliberately.")
            if old.get("split_seed") != args.split_seed:
                raise RuntimeError(f"Existing split '{manifest_path}' uses split_seed = {old.get('split_seed')}, requested {args.split_seed}. Use --overwrite deliberately to recreate it.")
            if old.get("fraction_seed") != seed:
                raise RuntimeError(f"Existing split '{manifest_path}' uses fraction_seed = {old.get('fraction_seed')}, requested model seed {seed}. Use --overwrite deliberately to recreate it.")
                
            policy = {
                "rare_class_policy": args.rare_class_policy,
                "min_recordings": args.min_recordings,
                "min_groups": args.min_groups,
                "silence_preflight_fingerprint": silence_preflight["source_fingerprint"],
                "silence_minimum_rms": float(args.minimum_rms),
                "preflight_max_read_error_fraction": float(silence_preflight["max_read_error_fraction"]),
                "preflight_max_excluded_fraction": float(silence_preflight["max_excluded_fraction"]),
                "preflight_reason_histogram": silence_preflight.get("reason_histogram", {}),
            }
            
            if any(old.get(key) != value for key, value in policy.items()):
                raise RuntimeError(f"Existing split '{manifest_path}' lacks or differs from class-scope policy {policy}. Use --overwrite deliberately to regenerate it.")
                
            LOG.info("Preserving existing split: %s", manifest_path)
            
            compat_expected = {
                "schema_version": 5, 
                "seed": seed,
                "dataset_fingerprint": ds_fingerprint,
                "silence_preflight_fingerprint": silence_preflight["source_fingerprint"],
                "test_filenames": old.get("test_filenames", []),
                "label_map": dataset.label_map,
            }
            
            if compatibility_path.is_file():
                compat_existing = json.loads(compatibility_path.read_text(encoding = "utf-8"))
                
                if compat_existing != compat_expected:
                    raise RuntimeError(
                        f"Compatibility test split '{compatibility_path}' disagrees with the authoritative manifest. "
                        "Back up old outputs and rerun with --overwrite only if deliberately rebuilding the split."
                    )
            else:
                tmp_compat = compatibility_path.with_suffix(".json.tmp")
                tmp_compat.write_text(json.dumps(compat_expected, indent = 2), encoding = "utf-8")
                os.replace(tmp_compat, compatibility_path)
                
            continue

        train, val, test, train_fractions, method = split_indices(
            dataset.labels, 
            dataset.group_ids, 
            args.split_seed, 
            dataset.grouping_source,
            min_recordings = args.min_recordings, 
            min_groups = args.min_groups,
            rare_class_policy = args.rare_class_policy, 
            fraction_seed = seed,
            eligible_indices = eligible_indices,
        )
        
        to_names = lambda ids: [index_paths[i] for i in ids]
        active_label_ids = sorted({dataset.labels[i] for i in train})
    
        # Include labels with zero eligible recordings. Otherwise a species whose
        # every file fails the silence preflight disappears from the exclusion
        # report, even though it remains in the source label map.
        all_label_ids = set(dataset.labels)
        active_label_set = set(active_label_ids)
        excluded_label_ids = sorted(all_label_ids - active_label_set)
        eligible_set = set(eligible_indices)
        
        count_by_label = {
            label: sum(dataset.labels[i] == label for i in eligible_indices)
            for label in all_label_ids
        }
        
        groups_by_label = {
            label: len({
                dataset.group_ids[i] for i, value in enumerate(dataset.labels)
                if i in eligible_set and value == label and not str(dataset.group_ids[i]).startswith("missing:")
            })
            
            for label in all_label_ids
        }
        
        excluded_classes = []
        for label in excluded_label_ids:
            name = next(name for name, idx in dataset.label_map.items() if idx == label)
            why = []
            
            if count_by_label[label] == 0:
                why.append("no_eligible_recordings_after_audio_preflight")
            if count_by_label[label] < args.min_recordings:
                why.append(f"recordings<{args.min_recordings}")
            if dataset.grouping_source != "file_level_no_metadata" and groups_by_label[label] < args.min_groups:
                why.append(f"distinct_groups<{args.min_groups}")
                
            excluded_classes.append({"label_id": label, "species": name, "recordings": count_by_label[label], "distinct_groups": groups_by_label[label], "reasons": why})
            
        manifest = {
            "schema_version": 5,
            "seed": seed,
            "split_seed": args.split_seed,
            "fraction_seed": seed,
            "dataset_root": str(Path(args.audio_dir).resolve()),
            "dataset_fingerprint": ds_fingerprint,
            "silence_preflight_manifest": str(Path(args.silence_manifest)),
            "silence_preflight_fingerprint": silence_preflight["source_fingerprint"],
            "silence_minimum_rms": float(args.minimum_rms),
            "silence_preflight_policy": "exclude files whose deterministic center first crop recovery cannot find any fixed length crop above minimum RMS",
            "n_silent_files_excluded": len(silent_paths),
            "silent_excluded_filenames": sorted(silent_paths),
            "n_preflight_read_error_files_excluded": len(preflight_read_error_paths),
            "preflight_read_error_filenames": sorted(preflight_read_error_paths),
            "n_preflight_excluded_files": len(preflight_excluded_paths),
            "preflight_excluded_filenames": sorted(preflight_excluded_paths),
            "preflight_reason_histogram": silence_preflight.get("reason_histogram", {}),
            "preflight_read_error_fraction": float(silence_preflight.get("read_error_fraction", 0.0)),
            "preflight_excluded_fraction": float(silence_preflight.get("excluded_fraction", 0.0)),
            "preflight_max_read_error_fraction": float(silence_preflight.get("max_read_error_fraction", args.max_read_error_fraction)),
            "preflight_max_excluded_fraction": float(silence_preflight.get("max_excluded_fraction", args.max_excluded_fraction)),
            "num_files": len(dataset),
            "num_eligible_before_class_scope": len(eligible_indices),
            "num_active_files": len(train) + len(val) + len(test),
            "num_active_classes": len(active_label_ids),
            "num_classes_total": len(dataset.label_map),
            "label_map": dataset.label_map,
            "active_label_map": {name: idx for name, idx in dataset.label_map.items() if idx in set(active_label_ids)},
            "rare_class_policy": args.rare_class_policy,
            "min_recordings": args.min_recordings,
            "min_groups": args.min_groups,
            "excluded_classes": excluded_classes,
            "excluded_filenames": sorted(set(preflight_excluded_paths) | {index_paths[i] for i, label in enumerate(dataset.labels) if i in eligible_set and label in set(excluded_label_ids)}),
            "class_scope_note": "Metrics cover active_label_map only; excluded species are intentionally absent from all train/validation/test lists.",
            "n_train": len(train), "n_validation": len(val), "n_test": len(test),
            "fraction_counts": {frac: len(ids) for frac, ids in train_fractions.items()},
            "grouping_source": dataset.grouping_source,
            "split_method": method,
            "train_filenames": to_names(train),
            "validation_filenames": to_names(val),
            "test_filenames": to_names(test),
            "train_fraction_filenames": {frac: to_names(ids) for frac, ids in train_fractions.items()},
        }
        
        tmp_path = manifest_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(manifest, indent = 2), encoding = "utf-8")
        os.replace(tmp_path, manifest_path)
        
        compat = {
            "schema_version": 5,
            "seed": seed,
            "dataset_fingerprint": ds_fingerprint,
            "test_filenames": manifest["test_filenames"],
            "label_map": dataset.label_map,
            "silence_preflight_fingerprint": silence_preflight["source_fingerprint"],
        }
        
        compatibility_path.write_text(json.dumps(compat, indent = 2), encoding = "utf-8")
        LOG.info("Saved model-seed %d manifest using shared split_seed=%d (%s): train=%d val=%d test=%d; fingerprint=%s", seed, args.split_seed, method, len(train), len(val), len(test), ds_fingerprint[:12])


if __name__ == "__main__":
    main()
