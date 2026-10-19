"""Paired recording level statistics for Exp3 stress test outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import binomtest

from utils.preflight_metadata import validate_preflight_audit

SEEDS_DEFAULT = (42, 123, 999)
MODELS = {
    "habitat": "moco_habitat",
    "standard": "moco_standard",
    "white_noise": "moco_white_noise",
    "supervised": "supervised_baseline",
    "supervised_noise": "supervised_noise_baseline",
}


def t_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError("At least two seeds are required for a seed level t interval.")
        
    if not np.isfinite(values).all():
        raise ValueError("Non finite seed level effect values.")
        
    half = float(stats.sem(values) * stats.t.ppf((1 + confidence) / 2, len(values) - 1))
    return float(values.mean()), half


PRIMARY_ENDPOINT = {
    "pool": "unseen_habitat",
    "snr": "0dB",
    "comparison": "moco_habitat-moco_standard",
    "analysis": "paired_accuracy_delta",
}


def holm_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    
    p = np.asarray(p_values, dtype = float)
    if p.ndim != 1 or not len(p):
        return np.asarray([], dtype = float)
        
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Holm adjustment requires finite p-values in [0, 1].")
        
    order = np.argsort(p, kind = "stable")
    ranked = p[order]
    m = len(ranked)
    
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * ranked)
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    
    return adjusted


def _result_path(root: Path, seed: int, fraction: float) -> Path:
    if abs(fraction - 1.0) < 1e-8:
        return root / f"unseen_habitat_stress_test_matrix_s{seed}.json"
        
    return root / f"unseen_habitat_stress_test_matrix_s{seed}_frac{float(fraction)}.json"


def _index_predictions(metrics: dict, context: str) -> dict[str, dict]:
    rows = metrics.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{context}: missing per recording predictions.")
        
    mapping = {}
    for row in rows:
        name = str(row.get("file", ""))
        if not name or name in mapping or "correct" not in row or "y_true" not in row:
            raise ValueError(f"{context}: malformed/duplicate per recording prediction.")
            
        mapping[name] = row
        
    return mapping


def _validate_pair(a: dict, b: dict, context: str):
    if tuple(a) != tuple(b):
        raise ValueError(f"{context}: model predictions do not share identical ordered test filenames.")
        
    for name in a:
        if int(a[name]["y_true"]) != int(b[name]["y_true"]):
            raise ValueError(f"{context}: ground-truth mismatch for {name}.")
        if "group_id" in a[name] and "group_id" in b[name] and str(a[name]["group_id"]) != str(b[name]["group_id"]):
            raise ValueError(f"{context}: group ID mismatch for {name}.")


def _paired_delta(a: dict, b: dict):
    names = list(a)
    ca = np.asarray([bool(a[n]["correct"]) for n in names], dtype = float)
    cb = np.asarray([bool(b[n]["correct"]) for n in names], dtype = float)
    diffs = ca - cb
    a_wins = int(np.sum((ca == 1) & (cb == 0)))
    b_wins = int(np.sum((ca == 0) & (cb == 1)))
    discordant = a_wins + b_wins
    pvalue = float(binomtest(a_wins, discordant, 0.5, alternative = "two-sided").pvalue) if discordant else 1.0
    
    return diffs, a_wins, b_wins, pvalue


def _bootstrap_ci(
    differences: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    cluster_ids: list[str] | None = None,
):
    """Paired percentile CI, resampling independent groups when metadata allows."""
    differences = np.asarray(differences, dtype = float)
    n = len(differences)
    if n < 1:
        raise ValueError("Cannot bootstrap zero recordings.")
        
    usable_clusters = None
    if cluster_ids is not None and len(cluster_ids) == n and all(str(g) for g in cluster_ids):
        grouped: dict[str, list[int]] = {}
        
        for idx, group in enumerate(cluster_ids):
            grouped.setdefault(str(group), []).append(idx)
        if 1 < len(grouped) < n:
            usable_clusters = [np.asarray(indices, dtype=int) for _, indices in sorted(grouped.items())]

    means = np.empty(n_boot, dtype = float)
    # Chunking avoids allocating large bootstrap index matrices.
    for start in range(0, n_boot, 256):
        size = min(256, n_boot - start)
        
        if usable_clusters is None:
            indices = rng.integers(0, n, size = (size, n))
            means[start:start + size] = differences[indices].mean(axis = 1)
        else:
            # Cluster resampling preserves within recordist/location dependence.
            n_groups = len(usable_clusters)
            samples = rng.integers(0, n_groups, size = (size, n_groups))
            
            for row_idx, chosen_groups in enumerate(samples, start=start):
                chosen = np.concatenate([usable_clusters[group_idx] for group_idx in chosen_groups])
                means[row_idx] = differences[chosen].mean()
                
    unit = "group" if usable_clusters is not None else "recording"
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), unit


def validate_no_skipped_noise_mixes(payload: dict, snrs: list[str]) -> None:
    """Fail closed if a noisy SNR metric contains clean fallback examples."""
    for pool in ("seen_habitat", "unseen_habitat"):
        for model_name in MODELS.values():
            for snr in snrs:
                try:
                    raw = payload[pool][model_name][snr]
                except KeyError as exc:
                    raise ValueError(f"Missing result while validating skipped mixes: {pool}/{model_name}/{snr}.") from exc
                    
                if "n_silent_mix_skipped" not in raw or "silent_mix_skipped_files" not in raw:
                    raise ValueError(
                        f"Missing v1.6.3 skipped mix metadata at {pool}/{model_name}/{snr}; rerun Exp3 before paired inference."
                    )
                    
                skipped = int(raw["n_silent_mix_skipped"])
                files = raw["silent_mix_skipped_files"]
                if skipped < 0 or not isinstance(files, list) or skipped != len(set(map(str, files))):
                    raise ValueError(f"Invalid skipped-mix count/list at {pool}/{model_name}/{snr}.")
                if skipped:
                    raise ValueError(
                        f"Cannot run paired inference: {pool}/{model_name}/{snr} has {skipped} skipped mixes scored as clean fallback. "
                        "Rebuild the BirdCLEF silence preflight/splits and rerun Exp3; do not use this SNR result."
                    )


def analyze_seed(payload: dict, seed: int, snrs: list[str], n_boot: int) -> list[dict]:
    validate_no_skipped_noise_mixes(payload, snrs)
    output = []
    metadata = payload.get("metadata", {})
    grouping_source = str(metadata.get("grouping_source", "file_level_no_metadata"))
    use_clusters = "file_level" not in grouping_source
    excluded_classes = metadata.get("excluded_classes", [])
    active_label_map = metadata.get("active_label_map", {})
    preflight_audit = validate_preflight_audit(metadata, context = f"Exp3 seed {seed}")
    
    scope_fields = {
        "n_preflight_excluded_files": preflight_audit["n_preflight_excluded_files"],
        "preflight_excluded_filenames_json": json.dumps(preflight_audit["preflight_excluded_filenames"], sort_keys = True),
        "n_preflight_read_error_files_excluded": preflight_audit["n_preflight_read_error_files_excluded"],
        "preflight_read_error_filenames_json": json.dumps(preflight_audit["preflight_read_error_filenames"], sort_keys = True),
        "preflight_reason_histogram_json": json.dumps(preflight_audit["preflight_reason_histogram"], sort_keys = True),
        "preflight_read_error_fraction": preflight_audit["preflight_read_error_fraction"],
        "preflight_excluded_fraction": preflight_audit["preflight_excluded_fraction"],
        "n_active_classes": int(metadata.get("n_active_classes", len(active_label_map))),
        "n_total_classes": int(metadata.get("n_total_classes", len(metadata.get("full_label_map", {})) or len(active_label_map) + len(excluded_classes))),
        "n_excluded_classes": int(metadata.get("n_excluded_classes", len(excluded_classes))),
        "n_silent_files_excluded": int(metadata.get("n_silent_files_excluded", 0)),
        "silent_excluded_filenames_json": json.dumps(metadata.get("silent_excluded_filenames", []), sort_keys = True),
        "rare_class_policy": str(metadata.get("rare_class_policy", "unknown")),
        "excluded_classes_json": json.dumps(excluded_classes, sort_keys = True),
        "class_scope_note": str(metadata.get("class_scope_note", "Class scope not recorded in this result; rerun with v1.6.3.")),
    }
    
    for pool in ("seen_habitat", "unseen_habitat"):
        for snr in snrs:
            all_preds = {}
            
            for model_name in MODELS.values():
                try:
                    all_preds[model_name] = _index_predictions(payload[pool][model_name][snr], f"{pool}/{model_name}/{snr}")
                except KeyError as exc:
                    raise ValueError(f"Missing prediction results for {pool}/{model_name}/{snr}.") from exc
                    
            reference_names = tuple(all_preds["moco_habitat"])
            for model_name, pred in all_preds.items():
                _validate_pair(all_preds["moco_habitat"], pred, f"{pool}/{model_name}/{snr}")
                
                if tuple(pred) != reference_names:
                    raise ValueError("Paired statistics require identical test ordering across models.")
                    
            for comparator in ("moco_standard", "moco_white_noise", "supervised_baseline", "supervised_noise_baseline"):
                a_name = "moco_habitat"
                a, b = all_preds[a_name], all_preds[comparator]
                diffs, a_wins, b_wins, pvalue = _paired_delta(a, b)
                
                # Stable numeric seed derived from experiment coordinates; avoid Python's salted hash().
                coordinate = f"{seed}|{pool}|{snr}|{a_name}|{comparator}"
                bootstrap_seed = seed + sum(
                    (i + 1) * ord(ch)
                    for i, ch in enumerate(coordinate)
                )

                names = list(a)

                clusters = [
                    str(a[n].get("group_id", ""))
                    for n in names
                ] if use_clusters else None
                
                ci_low, ci_high, bootstrap_unit = _bootstrap_ci(
                    diffs,
                    np.random.default_rng(bootstrap_seed),
                    n_boot,
                    clusters,
                )
                
                output.append({
                    "seed": seed, 
                    "pool": pool, 
                    "snr": snr, 
                    "comparison": f"{a_name}-{comparator}",
                    "analysis": "paired_accuracy_delta", 
                    "n_recordings": len(diffs),
                    "delta_accuracy": float(diffs.mean()), 
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high, 
                    "bootstrap_unit": bootstrap_unit,
                    "mcnemar_a_wins": a_wins,
                    "mcnemar_b_wins": b_wins, 
                    "mcnemar_discordant": a_wins + b_wins,
                    "mcnemar_exact_pvalue": pvalue,
                    **scope_fields,
                })

    # Difference-in-differences: (baseline seen-unseen gap) - (habitat-SSL gap).
    for snr in snrs:
        for comparator in ("moco_standard", "moco_white_noise", "supervised_baseline", "supervised_noise_baseline"):
            seen_h = _index_predictions(payload["seen_habitat"]["moco_habitat"][snr], f"seen/habitat/{snr}")
            unseen_h = _index_predictions(payload["unseen_habitat"]["moco_habitat"][snr], f"unseen/habitat/{snr}")
            seen_b = _index_predictions(payload["seen_habitat"][comparator][snr], f"seen/{comparator}/{snr}")
            unseen_b = _index_predictions(payload["unseen_habitat"][comparator][snr], f"unseen/{comparator}/{snr}")
            
            _validate_pair(seen_h, unseen_h, f"habitat model seen/unseen {snr}")
            _validate_pair(seen_b, unseen_b, f"{comparator} seen/unseen {snr}")
            _validate_pair(seen_h, seen_b, f"seen model pair {snr}")
            _validate_pair(unseen_h, unseen_b, f"unseen model pair {snr}")
            
            names = list(seen_h)
            gh = np.asarray([bool(seen_h[n]["correct"]) - bool(unseen_h[n]["correct"]) for n in names], dtype = float)
            gb = np.asarray([bool(seen_b[n]["correct"]) - bool(unseen_b[n]["correct"]) for n in names], dtype = float)
            
            diffs = gb - gh
            coordinate = f"{seed}|difference_in_differences|{snr}|{comparator}"
            bootstrap_seed = seed + sum((i + 1) * ord(ch) for i, ch in enumerate(coordinate))
            clusters = [str(seen_h[n].get("group_id", "")) for n in names] if use_clusters else None
            ci_low, ci_high, bootstrap_unit = _bootstrap_ci(
                diffs, np.random.default_rng(bootstrap_seed), n_boot, clusters
            )
            
            output.append({
                "seed": seed, 
                "pool": "seen_vs_unseen", 
                "snr": snr,
                "comparison": f"{comparator}_gap_minus_habitat_gap", 
                "analysis": "difference_in_differences",
                "n_recordings": len(diffs), 
                "delta_accuracy": float(diffs.mean()),
                "bootstrap_ci95_low": ci_low, 
                "bootstrap_ci95_high": ci_high,
                "bootstrap_unit": bootstrap_unit,
                "mcnemar_a_wins": "", "mcnemar_b_wins": "", "mcnemar_discordant": "",
                "mcnemar_exact_pvalue": "",
                **scope_fields,
            })
    return output


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--results-dir", default = "results")
    parser.add_argument("--expected-seeds", nargs="+", type=int, default = list(SEEDS_DEFAULT))
    parser.add_argument("--fraction", type=float, default = 1.0)
    parser.add_argument("--n-bootstrap", type=int, default = 5000)
    parser.add_argument("--snrs", default = None, help = "Optional comma separated SNR output labels. If omitted, infer from first result.")
    
    args = parser.parse_args(argv)
    if not 0 < args.fraction <= 1 or args.n_bootstrap < 100:
        parser.error("fraction must be in (0,1] and n-bootstrap must be >= 100")
    if len(args.expected_seeds) < 2 or len(set(args.expected_seeds)) != len(args.expected_seeds):
        parser.error("expected seeds must contain at least two distinct training seeds")
    if args.snrs is not None:
        requested_snr_labels = [value.strip() for value in args.snrs.split(",") if value.strip()]
        
        if len(set(requested_snr_labels)) != len(requested_snr_labels):
            parser.error("snrs must not contain duplicate labels")
            
    root = Path(args.results_dir)
    seed_rows = []
    expected_snrs = [v for v in args.snrs.split(",") if v] if args.snrs else None
    reference_crop_policy = None
    reference_preflight_audit = None
    
    for seed in args.expected_seeds:
        path = _result_path(root, seed, args.fraction)
        if not path.is_file():
            raise FileNotFoundError(f"Missing Exp3 output for seed={seed}, fraction={args.fraction}: {path}")
            
        payload = json.loads(path.read_text(encoding = "utf-8"))
        
        meta = payload.get("metadata", {})
        if meta.get("schema_version") != 5:
            raise ValueError(f"Result '{path}' is not Exp3 schema; rerun Exp3 before paired inference.")
        if int(meta.get("seed", -1)) != seed or abs(float(meta.get("fraction", -1)) - args.fraction) > 1e-8:
            raise ValueError(f"Metadata mismatch in '{path}'.")
            
        preflight_audit = validate_preflight_audit(meta, context=f"Exp3 result '{path}'")
        if reference_preflight_audit is None:
            reference_preflight_audit = preflight_audit
        elif preflight_audit != reference_preflight_audit:
            raise ValueError("Paired inference requires identical BirdCLEF preflight exclusions/reason counts across seeds.")
        try:
            crop_policy = {
                "crop_policy": str(meta["crop_policy"]),
                "crop_rms_threshold": float(meta["crop_rms_threshold"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Result '{path}' lacks valid v1.6.3 crop-policy metadata.") from exc
            
        if (crop_policy["crop_policy"] != "center-first-deterministic-quarter-window-rms-recovery-v2"
                or not np.isfinite(crop_policy["crop_rms_threshold"])
                or crop_policy["crop_rms_threshold"] <= 0):
            raise ValueError(f"Result '{path}' contains unsupported or invalid crop policy metadata.")
            
        if reference_crop_policy is None:
            reference_crop_policy = crop_policy
            
        elif crop_policy != reference_crop_policy:
            raise ValueError("Paired inference requires all seeds to use the same crop policy and RMS threshold.")
            
        snrs = expected_snrs or list(meta.get("snr_labels", []))
        if not snrs:
            raw_levels = meta.get("snr_levels", [])
            snrs = ["Clean" if value is None else f"{float(value):g}dB" for value in raw_levels]
            
        if not snrs:
            raise ValueError(f"No SNR labels stored in '{path}'.")
            
        if expected_snrs is None:
            expected_snrs = snrs
        elif snrs != expected_snrs:
            raise ValueError(f"SNR condition list differs in '{path}'.")
            
        new_rows = analyze_seed(payload, seed, snrs, args.n_bootstrap)
        for row in new_rows:
            row["is_primary_endpoint"] = bool(
                row["pool"] == PRIMARY_ENDPOINT["pool"]
                and row["snr"] == PRIMARY_ENDPOINT["snr"]
                and row["comparison"] == PRIMARY_ENDPOINT["comparison"]
                and row["analysis"] == PRIMARY_ENDPOINT["analysis"]
            )
            
            row["mcnemar_holm_adjusted_pvalue"] = ""
            row["mcnemar_holm_reject_0_05"] = ""
            row["mcnemar_adjustment_family"] = "primary_seed_level_t_test" if row["is_primary_endpoint"] else "secondary_all_endpoints"
            
        seed_rows.extend(new_rows)

    # Fail closed if this invocation did not actually evaluate the prespecified
    # primary endpoint for each seed. This prevents silently publishing only
    # exploratory secondary tests when the config omitted the primary 0 dB cell.
    primary_rows = [row for row in seed_rows if row["is_primary_endpoint"]]
    if len(primary_rows) != len(args.expected_seeds) or {int(row["seed"]) for row in primary_rows} != set(args.expected_seeds):
        raise ValueError(
            f"The requested SNR/result matrix did not produce exactly one primary endpoint row for each of "
            f"{args.expected_seeds}. Required endpoint: {PRIMARY_ENDPOINT}. Ensure unseen_habitat, 0dB, "
            "moco_habitat vs moco_standard are present in every seed result."
        )

    # The planned primary contrast is evaluated across independent training seeds (3 values by default). 
    # Recording level McNemar tests are kept as diagnostics.
    secondary_rows = [
        row for row in seed_rows
        if row["analysis"] == "paired_accuracy_delta" and not row["is_primary_endpoint"]
    ]
    
    secondary_p = [float(row["mcnemar_exact_pvalue"]) for row in secondary_rows]
    secondary_adjusted = holm_adjust(secondary_p) if secondary_p else np.asarray([], dtype=float)
    for row, adjusted in zip(secondary_rows, secondary_adjusted):
        row["mcnemar_holm_adjusted_pvalue"] = float(adjusted)
        row["mcnemar_holm_reject_0_05"] = bool(adjusted <= 0.05)

    analysis_plan = {
        "version": "v1.6.3",
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_inference": "two sided one sample t-test of per-training-seed paired accuracy deltas against zero; reports seed level t CI; requires at least two seeds and is low powered with three seeds",
        "secondary_family": "all per seed recording level exact McNemar tests for all other pool/SNR/model comparisons at this fraction",
        "secondary_multiplicity_method": "Holm-Bonferroni family wise error rate control at alpha=0.05 within this fraction; correcting claims pooled over fractions requires a larger combined family",
        "primary_mcnemar_values": "unadjusted, descriptive diagnostics only; primary claim uses the across seed test",
        "dependence_caveat": "Seeds share test recordings; recording level McNemar tests are secondary diagnostics, not independent replications across model seeds. File level bootstrap intervals respect optional grouping metadata when available.",
        "snr_scope": expected_snrs,
        "fraction": float(args.fraction),
        "expected_seeds": [int(seed) for seed in args.expected_seeds],
        "alpha": 0.05,
    }
    
    plan_name = ("paired_tests_analysis_plan.json" if abs(args.fraction - 1.0) < 1e-8
                 else f"paired_tests_analysis_plan_frac{args.fraction}.json")
    plan_path = root / plan_name
    plan_path.write_text(json.dumps(analysis_plan, indent=2) + "\n", encoding="utf-8")

    out = root / ("paired_tests_by_seed.csv" if abs(args.fraction - 1) < 1e-8 else f"paired_tests_by_seed_frac{args.fraction}.csv")
    out.parent.mkdir(parents = True, exist_ok = True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=seed_rows[0].keys())
        writer.writeheader(); writer.writerows(seed_rows)

    group_keys = sorted({(r["pool"], r["snr"], r["comparison"], r["analysis"]) for r in seed_rows})
    summary = []
    for pool, snr, comparison, analysis in group_keys:
        rows = [r for r in seed_rows if (r["pool"], r["snr"], r["comparison"], r["analysis"]) == (pool, snr, comparison, analysis)]
        values = [float(r["delta_accuracy"]) for r in rows]
        
        if len(values) >= 2:
            mean, half = t_interval(values)
        else:
            mean, half = values[0], float("nan")
        is_primary = (pool == PRIMARY_ENDPOINT["pool"] and snr == PRIMARY_ENDPOINT["snr"]
                      and comparison == PRIMARY_ENDPOINT["comparison"] and analysis == PRIMARY_ENDPOINT["analysis"])
                      
        seed_t_p = float(stats.ttest_1samp(np.asarray(values, dtype = float), popmean = 0.0).pvalue) if is_primary and len(values) >= 2 else ""
        summary.append({
            "pool": pool, 
            "snr": snr, 
            "comparison": comparison, 
            "analysis": analysis,
            "is_primary_endpoint": is_primary,
            "n_seeds": len(values), 
            "seed_effects": json.dumps(values), 
            "mean_effect": mean,
            "seed_t_ci95_half_width": half,
            "primary_seed_level_t_pvalue": seed_t_p,
            "multiple_comparison_policy": "prespecified primary seed level t-test; all secondary per seed McNemar p-values Holm-adjusted",
        })
        
    summary_path = root / ("paired_statistics_summary.csv" if abs(args.fraction - 1) < 1e-8 else f"paired_statistics_summary_frac{args.fraction}.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames = summary[0].keys())
        writer.writeheader(); writer.writerows(summary)
        
    print(f"Wrote per seed paired tests: {out}")
    print(f"Wrote seed level paired effect summary: {summary_path}")
    print(f"Wrote prespecified analysis plan: {plan_path}")


if __name__ == "__main__":
    main()
