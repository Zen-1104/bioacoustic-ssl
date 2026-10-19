"""Aggregate all Exp2 runs into a validated LaTeX table, CSV, and plot."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from utils.preflight_metadata import validate_preflight_audit

DEFAULT_SEEDS = (42, 123, 999)
DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 1.00)
DEFAULT_VARIANTS = ("habitat", "standard", "white_noise", "supervised", "supervised_noise")


def calculate_t_interval_ci(values, confidence = 0.95):

    arr = np.asarray(list(values), dtype = float)
    if len(arr) < 2:
        raise ValueError(f"At least 2 independent seed level observations are required for a t-interval; got {len(arr)}.")
    if not np.isfinite(arr).all():
        raise ValueError("Non finite metrics found while aggregating probe results.")
        
    mean = float(arr.mean())
    half_width = float(stats.sem(arr) * stats.t.ppf((1 + confidence) / 2.0, len(arr) - 1))
    
    return mean, half_width

def _frac_equal(a, b):
    return abs(float(a) - float(b)) < 1e-8

def load_probe_records(results_dir = "results", expected_seeds = DEFAULT_SEEDS):
    paths = sorted(glob.glob(os.path.join(results_dir, "probe_*.json")))
    if not paths:
        raise FileNotFoundError(f"No probe result files found in '{results_dir}'. Run all Exp2 tasks first.")
        
    records = {}
    fingerprints = set()
    config_hashes: dict[tuple[str, float], set[str]] = {}
    reference_class_scope = None
    reference_evaluation_policy = None
    
    for path in paths:
        data = json.loads(Path(path).read_text(encoding = "utf-8"))
        if int(data.get("schema_version", -1)) != 4:
            raise ValueError(f"Probe result '{path}' uses an older evaluation schema; rerun Exp2 with v1.6.3 before aggregation.")
            
        required = {"variant", "seed", "fraction", "test_accuracy", "mode"}
        if not required.issubset(data):
            raise ValueError(f"Probe result '{path}' is missing fields {sorted(required - set(data))}.")
            
        variant, seed, fraction = str(data["variant"]), int(data["seed"]), float(data["fraction"])
        if variant not in DEFAULT_VARIANTS:
            raise ValueError(f"Unexpected variant '{variant}' in '{path}'.")
        if seed not in expected_seeds:
            continue
        if not 0 < fraction <= 1:
            raise ValueError(f"Invalid fraction {fraction} in '{path}'.")
            
        expected_mode = variant if variant in {"supervised", "supervised_noise"} else "probe"
        if data["mode"] != expected_mode:
            raise ValueError(f"Mode/variant mismatch in '{path}'.")
            
        key = (variant, seed, next((f for f in DEFAULT_FRACTIONS if _frac_equal(fraction, f)), fraction))
        if key in records:
            raise ValueError(f"Duplicate result for variant = {variant}, seed = {seed}, fraction = {fraction}.")
            
        accuracy = float(data["test_accuracy"])
        if not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(f"Invalid test_accuracy={accuracy} in '{path}'.")
            
        if not data.get("dataset_fingerprint"):
             raise ValueError(f"Missing dataset_fingerprint in '{path}'.")
        if not data.get("config_sha256"):
            raise ValueError(f"Missing config_sha256 in '{path}'.")
            
        required_scope = {
            "n_active_classes", 
            "n_total_classes", 
            "n_excluded_classes", 
            "excluded_classes", 
            "n_silent_files_excluded", 
            "rare_class_policy", 
            "class_scope_note", 
            "evaluation_crop_policy", 
            "evaluation_minimum_rms"
        }

        if not required_scope.issubset(data):
            raise ValueError(f"Probe result '{path}' lacks explicit class-scope/evaluation-crop metadata {sorted(required_scope - set(data))}.")
            
        evaluation_policy = {
            "evaluation_crop_policy": str(data["evaluation_crop_policy"]),
            "evaluation_minimum_rms": float(data["evaluation_minimum_rms"]),
        }
        
        if (evaluation_policy["evaluation_crop_policy"] != "center-first-deterministic-quarter-window-rms-recovery-v2"
                or not np.isfinite(evaluation_policy["evaluation_minimum_rms"])
                or evaluation_policy["evaluation_minimum_rms"] <= 0):
            raise ValueError(f"Probe result '{path}' records an unsupported or invalid evaluation crop policy.")
            
        if reference_evaluation_policy is None:
            reference_evaluation_policy = evaluation_policy
        elif evaluation_policy != reference_evaluation_policy:
            raise ValueError("Probe outputs use different evaluation crop policies/RMS thresholds; do not aggregate them together.")
            
        preflight_audit = validate_preflight_audit(data, context = f"Probe result '{path}'")
        
        scope = {
            **preflight_audit,
            "n_active_classes": int(data["n_active_classes"]),
            "n_total_classes": int(data["n_total_classes"]),
            "n_excluded_classes": int(data["n_excluded_classes"]),
            "excluded_classes": data["excluded_classes"],
            "n_silent_files_excluded": int(data["n_silent_files_excluded"]),
            "rare_class_policy": str(data["rare_class_policy"]),
            "class_scope_note": str(data["class_scope_note"]),
        }
        
        if scope["n_active_classes"] != int(data.get("num_classes", scope["n_active_classes"])) or scope["n_total_classes"] < scope["n_active_classes"] or scope["n_excluded_classes"] != len(scope["excluded_classes"]):
            raise ValueError(f"Inconsistent class-scope counts in '{path}'.")
            
        if reference_class_scope is None:
            reference_class_scope = scope
        elif scope != reference_class_scope:
            raise ValueError("Probe outputs use different active/excluded class scopes; do not aggregate together.")
            
        records[key] = {**data, "source_file": path}
        fingerprints.add(data["dataset_fingerprint"])
        config_hashes.setdefault((variant, key[2]), set()).add(data["config_sha256"])
        
    if len(fingerprints) > 1:
        raise ValueError("Probe outputs use different dataset fingerprints. Do not aggregate across different BirdCLEF file sets.")
        
    inconsistent_configs = [key for key, hashes in config_hashes.items() if len(hashes) > 1]
    if inconsistent_configs:
        raise ValueError(f"Probe outputs use different configs for variant/fraction pairs: {inconsistent_configs}. Do not aggregate mixed settings.")
        
    missing = []
    for variant in DEFAULT_VARIANTS:
        for fraction in DEFAULT_FRACTIONS:
            for seed in expected_seeds:
                key = (variant, seed, fraction)
                if key not in records:
                    missing.append(key)
                    
    if missing:
        examples = ", ".join(f"{v}/seed={s}/fraction = {f:.2f}" for v, s, f in missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} expected Exp2 results. Examples: {examples}. Do not publish an incomplete aggregate.")
        
    return records


def generate_probe_outputs(records, results_dir = "results", expected_seeds = DEFAULT_SEEDS):
    out = Path(results_dir)
    out.mkdir(parents = True, exist_ok = True)
    summary_rows = []
    
    for variant in DEFAULT_VARIANTS:
        for fraction in DEFAULT_FRACTIONS:
            values = [float(records[(variant, seed, fraction)]["test_accuracy"]) for seed in expected_seeds]
            mean, half = calculate_t_interval_ci(values)
            
            summary_rows.append({
                "variant": variant, 
                "fraction": fraction, 
                "n_seeds": len(values),
                "seed_accuracies": json.dumps(values), 
                "mean_accuracy": mean, 
                "ci95_half_width": half,
                "n_active_classes": int(records[(variant, expected_seeds[0], fraction)]["n_active_classes"]),
                "n_total_classes": int(records[(variant, expected_seeds[0], fraction)]["n_total_classes"]),
                "n_excluded_classes": int(records[(variant, expected_seeds[0], fraction)]["n_excluded_classes"]),
                "n_silent_files_excluded": int(records[(variant, expected_seeds[0], fraction)]["n_silent_files_excluded"]),
                "rare_class_policy": str(records[(variant, expected_seeds[0], fraction)]["rare_class_policy"]),
                "excluded_classes_json": json.dumps(records[(variant, expected_seeds[0], fraction)]["excluded_classes"], sort_keys = True),
                "class_scope_note": str(records[(variant, expected_seeds[0], fraction)]["class_scope_note"]),
                "n_preflight_excluded_files": int(records[(variant, expected_seeds[0], fraction)]["n_preflight_excluded_files"]),
                "preflight_excluded_filenames_json": json.dumps(records[(variant, expected_seeds[0], fraction)]["preflight_excluded_filenames"], sort_keys = True),
                "n_preflight_read_error_files_excluded": int(records[(variant, expected_seeds[0], fraction)]["n_preflight_read_error_files_excluded"]),
                "preflight_read_error_filenames_json": json.dumps(records[(variant, expected_seeds[0], fraction)]["preflight_read_error_filenames"], sort_keys = True),
                "preflight_reason_histogram_json": json.dumps(records[(variant, expected_seeds[0], fraction)]["preflight_reason_histogram"], sort_keys = True),
                "preflight_read_error_fraction": float(records[(variant, expected_seeds[0], fraction)]["preflight_read_error_fraction"]),
                "preflight_excluded_fraction": float(records[(variant, expected_seeds[0], fraction)]["preflight_excluded_fraction"]),
            })

    csv_path = out / "label_efficiency_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader(); writer.writerows(summary_rows)

    latex = [
        r"\begin{table}[t]", 
        r"\centering",
        r"\caption{Test accuracy by labeled-data fraction (mean $\pm$ 95\% Student's $t$ confidence interval over seeds).}",
        r"\begin{tabular}{lcccc}", 
        r"\toprule",
        r"Model & 1\% & 5\% & 10\% & 100\% " + r"\\", 
        r"\midrule",
    ]
    
    display_names = {
        "habitat": "Habitat SSL", 
        "standard": "Standard SSL", 
        "white_noise": "White noise SSL", 
        "supervised": "Supervised clean", 
        "supervised_noise": "Supervised + habitat noise"}

    for variant in DEFAULT_VARIANTS:
        row = [display_names[variant]]
        
        for fraction in DEFAULT_FRACTIONS:
            item = next(x for x in summary_rows if x["variant"] == variant and _frac_equal(x["fraction"], fraction))
            row.append(f"{item['mean_accuracy'] * 100:.1f} $\\pm$ {item['ci95_half_width'] * 100:.1f}")
            
        latex.append(" & ".join(row) + " " + chr(92) * 2)
        
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (out / "table_label_efficiency.tex").write_text("\n".join(latex), encoding = "utf-8")

    fig, ax = plt.subplots(figsize = (7.2, 4.6))
    
    x = np.arange(len(DEFAULT_FRACTIONS))
    for variant in DEFAULT_VARIANTS:
        items = [next(r for r in summary_rows if r["variant"] == variant and _frac_equal(r["fraction"], frac)) for frac in DEFAULT_FRACTIONS]
        ax.errorbar(x, [r["mean_accuracy"] for r in items], yerr = [r["ci95_half_width"] for r in items], marker = "o", capsize = 3, label = display_names[variant])
        
    ax.set_xticks(x, ["1%", "5%", "10%", "100%"])
    ax.set_xlabel("Labeled training data")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha = 0.25)
    ax.legend(frameon = False)
    fig.tight_layout()
    fig.savefig(out / "figure_label_efficiency.png", dpi = 180)
    plt.close(fig)
    
    print(f"Wrote {csv_path}")
    print(f"Wrote {out / 'table_label_efficiency.tex'}")
    print(f"Wrote {out / 'figure_label_efficiency.png'}")


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--results-dir", default = "results")
    parser.add_argument("--expected-seeds", nargs = "+", type=int, default = list(DEFAULT_SEEDS))
    args = parser.parse_args(argv)
    records = load_probe_records(args.results_dir, tuple(args.expected_seeds))
    generate_probe_outputs(records, args.results_dir, tuple(args.expected_seeds))

if __name__ == "__main__":
    main()