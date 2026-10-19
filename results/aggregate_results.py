"""Aggregate paired Exp3 results, including seen/unseen habitat gaps."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.stress_config import load_stress_snr_config
from utils.preflight_metadata import validate_preflight_audit

DEFAULT_SEEDS = (42, 123, 999)
DEFAULT_MODELS = ("moco_habitat", "moco_standard", "moco_white_noise", "supervised_baseline", "supervised_noise_baseline")
DEFAULT_POOLS = ("seen_habitat", "unseen_habitat")
DEFAULT_SNRS = ("Clean", "20dB", "15dB", "10dB", "5dB", "0dB", "-5dB")


def t_interval(values, confidence = 0.95):
    arr = np.asarray(list(values), dtype = float)
    if len(arr) < 2:
        raise ValueError(f"At least two independent seed-level values are required; got {len(arr)}.")
    if not np.isfinite(arr).all():
        raise ValueError("Non finite values encountered in stress-test aggregation.")
        
    mean = float(arr.mean())
    half = float(stats.sem(arr) * stats.t.ppf((1 + confidence) / 2.0, len(arr) - 1))
    
    return mean, half

def _metric_value(raw, path, context):
    if not isinstance(raw, dict) or "accuracy" not in raw or "predictions" not in raw:
        raise ValueError(f"Result {path} {context} lacks per recording predictions. Rerun Exp3 with the current code.")
        
    acc = float(raw["accuracy"])
    n = int(raw["n"])
    pred = raw["predictions"]
    if "n_silent_mix_skipped" not in raw or "silent_mix_skipped_files" not in raw:
        raise ValueError(f"Result {path} ({context}) lacks v1.6.3 skipped-mix metadata; rerun Exp3 before aggregation.")
        
    skipped_count = int(raw["n_silent_mix_skipped"])
    skipped_files = raw["silent_mix_skipped_files"]
    if skipped_count < 0 or not isinstance(skipped_files, list):
        raise ValueError(f"Invalid skipped mix metadata in {path} ({context}).")
    if skipped_count != len(set(map(str, skipped_files))):
        raise ValueError(f"Skipped-mix count/list disagree in {path} ({context}); rerun Exp3.")
    if skipped_count:
        raise ValueError(
            f"Result {path} ({context}) contains {skipped_count} skipped noise mixes that were evaluated as clean fallback. "
            "These are not valid samples of the requested noisy SNR condition; rebuild BirdCLEF preflight/splits and rerun Exp3 before aggregation."
        )
        
    if not np.isfinite(acc) or not 0 <= acc <= 1 or n != len(pred) or n < 1:
        raise ValueError(f"Invalid accuracy/sample count in {path} ({context}).")
        
    seen_paths = [x.get("file") for x in pred]
    if len(set(seen_paths)) != len(seen_paths):
        raise ValueError(f"Duplicate per recording filename in {path} ({context}).")
        
    measured = sum(bool(x.get("correct")) for x in pred) / n
    if abs(measured - acc) > 1e-9:
        raise ValueError(f"Stored accuracy does not match per recording predictions in {path} ({context}).")
        
    return acc, {str(x["file"]): x for x in pred}


def load_all_seed_results(results_dir = "results", expected_seeds = DEFAULT_SEEDS, expected_snrs = DEFAULT_SNRS, expected_fraction = 1.0):
    pattern = os.path.join(results_dir, "unseen_habitat_stress_test_matrix_s*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No stress test result files matching '{pattern}'. Run all Exp3 seed tasks first.")
        
    payloads = {}
    fingerprints = set()
    config_hashes = set()
    expected_test_files = None
    reference_class_scope = None
    reference_crop_policy = None
    reference_preflight_audit = None
    
    for path in paths:
        data = json.loads(Path(path).read_text(encoding = "utf-8"))
        
        meta = data.get("metadata", {})
        if meta.get("schema_version") != 5 or "seed" not in meta or "fraction" not in meta:
            raise ValueError(f"Legacy/incomplete Exp3 schema in '{path}'. Back up old results and rerun Exp3.")
            
        result_fraction = float(meta["fraction"])
        if not 0.0 < result_fraction <= 1.0:
            raise ValueError(f"Invalid fraction = {result_fraction} in '{path}'.")
        if abs(result_fraction - float(expected_fraction)) >= 1e-8:
            continue
            
        seed = int(meta["seed"])
        if seed not in expected_seeds:
            continue
        if seed in payloads:
            raise ValueError(f"Duplicate Exp3 result file for seed {seed}.")
        if not meta.get("dataset_fingerprint"):
            raise ValueError(f"Missing dataset_fingerprint in '{path}'.")
            
        fingerprints.add(meta["dataset_fingerprint"])

        required_scope = {
            "n_active_classes", 
            "n_total_classes", 
            "n_excluded_classes", 
            "rare_class_policy", 
            "excluded_classes", 
            "n_silent_files_excluded", 
            "class_scope_note", 
            "crop_policy", 
            "crop_rms_threshold", 
            "n_preflight_excluded_files", 
            "n_preflight_read_error_files_excluded", 
            "preflight_reason_histogram"
        }

        if not required_scope.issubset(meta):
            raise ValueError(f"Exp3 result '{path}' lacks class scope metadata {sorted(required_scope - set(meta))}; rerun Exp3 with v1.6.3 before aggregation.")
            
        preflight_audit = validate_preflight_audit(meta, context=f"Exp3 result '{path}'")

        if reference_preflight_audit is None:
            reference_preflight_audit = preflight_audit
        elif preflight_audit != reference_preflight_audit:
            raise ValueError("Exp3 results use different BirdCLEF preflight exclusions/reason counts; do not aggregate them together.")
            
        crop_policy = {
            "crop_policy": str(meta["crop_policy"]),
            "crop_rms_threshold": float(meta["crop_rms_threshold"]),
        }
        if (crop_policy["crop_policy"] != "center-first-deterministic-quarter-window-rms-recovery-v2"
                or not np.isfinite(crop_policy["crop_rms_threshold"])
                or crop_policy["crop_rms_threshold"] <= 0):
            raise ValueError(f"Exp3 result '{path}' records an unsupported/invalid evaluation crop policy.")
            
        if reference_crop_policy is None:
            reference_crop_policy = crop_policy
        elif crop_policy != reference_crop_policy:
            raise ValueError("Exp3 results use different crop-recovery policies/RMS thresholds; do not aggregate them together.")
            
        class_scope = {
            "n_active_classes": int(meta["n_active_classes"]),
            "n_total_classes": int(meta["n_total_classes"]),
            "n_excluded_classes": int(meta["n_excluded_classes"]),
            "rare_class_policy": str(meta["rare_class_policy"]),
            "excluded_classes": meta["excluded_classes"],
            "n_silent_files_excluded": int(meta["n_silent_files_excluded"]),
            "class_scope_note": str(meta["class_scope_note"]),
        }
        if class_scope["n_active_classes"] <= 0 or class_scope["n_total_classes"] < class_scope["n_active_classes"] or class_scope["n_excluded_classes"] != len(class_scope["excluded_classes"]):
            raise ValueError(f"Invalid class-scope counts in '{path}'.")
            
        if reference_class_scope is None:
            reference_class_scope = class_scope
        elif class_scope != reference_class_scope:
            raise ValueError("Exp3 results use different active/excluded class scopes; do not aggregate them together.")
            
        if not meta.get("config_sha256"):
            raise ValueError(f"Missing config_sha256 in '{path}'.")
            
        config_hashes.add(meta["config_sha256"])
        metadata_test_files = meta.get("test_filenames")
        if not isinstance(metadata_test_files, list) or not metadata_test_files or len(set(metadata_test_files)) != len(metadata_test_files):
            raise ValueError(f"Invalid or duplicate metadata.test_filenames in '{path}'.")
            
        if expected_test_files is None:
            expected_test_files = tuple(metadata_test_files)
        elif tuple(metadata_test_files) != expected_test_files:
            raise ValueError(f"Seed {seed} uses a different ordered test split from other seeds.")
            
        if not all(pool in data for pool in DEFAULT_POOLS):
            raise KeyError(f"'{path}' is missing seen_habitat or unseen_habitat results.")
            
        for pool in DEFAULT_POOLS:
            for model in DEFAULT_MODELS:
                if model not in data[pool]:
                    raise KeyError(f"'{path}' is missing {pool}/{model}.")
                    
                if set(data[pool][model]) != set(expected_snrs):
                    raise ValueError(f"SNR keys mismatch in '{path}' for {pool}/{model}: {list(data[pool][model])} vs expected {list(expected_snrs)}.")
                    
                for snr in expected_snrs:
                    _, condition_predictions = _metric_value(data[pool][model][snr], path, f"{pool}/{model}/{snr}")
                    if tuple(condition_predictions) != expected_test_files:
                        raise ValueError(
                            f"Per recording prediction filenames/order mismatch in '{path}' for {pool}/{model}/{snr}."
                        )
                        
        payloads[seed] = {"path": path, "data": data}
        
    missing_seeds = sorted(set(expected_seeds) - set(payloads))
    if missing_seeds:
        raise FileNotFoundError(
            f"Missing Exp3 results for fraction = {expected_fraction:g}, seeds {missing_seeds}; "
            "refusing to aggregate a partial seed set."
        )
    if len(fingerprints) > 1:
        raise ValueError("Exp3 runs used different dataset fingerprints. Do not aggregate them together.")
    if len(config_hashes) > 1:
        raise ValueError("Exp3 runs used different stress test config files. Do not aggregate mixed SNR/model configurations.")
        
    return payloads


def summarize(payloads, expected_seeds = DEFAULT_SEEDS, snrs = DEFAULT_SNRS):
    performance = []
    gaps = []
    for model in DEFAULT_MODELS:
        for snr in snrs:
            unseen_values, seen_values, gap_values = [], [], []
            
            for seed in expected_seeds:
                data = payloads[seed]["data"]
                unseen, unseen_preds = _metric_value(data["unseen_habitat"][model][snr], payloads[seed]["path"], f"unseen/{model}/{snr}")
                seen, seen_preds = _metric_value(data["seen_habitat"][model][snr], payloads[seed]["path"], f"seen/{model}/{snr}")
                
                if set(seen_preds) != set(unseen_preds):
                    raise ValueError(f"Seen/unseen evaluations used different test files for seed={seed}, model={model}, SNR={snr}.")
                    
                for filename in seen_preds:
                    if int(seen_preds[filename]["y_true"]) != int(unseen_preds[filename]["y_true"]):
                        raise ValueError(
                            f"Ground-truth label mismatch for {filename!r}, seed={seed}, model={model}, SNR={snr}."
                        )
                    
                unseen_values.append(unseen)
                seen_values.append(seen)
                gap_values.append(seen - unseen)
                
            u_mean, u_ci = t_interval(unseen_values)
            s_mean, s_ci = t_interval(seen_values)
            g_mean, g_ci = t_interval(gap_values)
            scope = payloads[int(expected_seeds[0])]["data"]["metadata"]
            preflight_audit = validate_preflight_audit(scope, context = "Exp3 aggregate reference metadata")
            
            scope_fields = {
                "n_preflight_excluded_files": preflight_audit["n_preflight_excluded_files"],
                "preflight_excluded_filenames_json": json.dumps(preflight_audit["preflight_excluded_filenames"], sort_keys = True),
                "n_preflight_read_error_files_excluded": preflight_audit["n_preflight_read_error_files_excluded"],
                "preflight_read_error_filenames_json": json.dumps(preflight_audit["preflight_read_error_filenames"], sort_keys = True),
                "preflight_reason_histogram_json": json.dumps(preflight_audit["preflight_reason_histogram"], sort_keys = True),
                "preflight_read_error_fraction": preflight_audit["preflight_read_error_fraction"],
                "preflight_excluded_fraction": preflight_audit["preflight_excluded_fraction"],
                "n_active_classes": int(scope["n_active_classes"]),
                "n_total_classes": int(scope["n_total_classes"]),
                "n_excluded_classes": int(scope["n_excluded_classes"]),
                "n_silent_files_excluded": int(scope["n_silent_files_excluded"]),
                "silent_excluded_filenames_json": json.dumps(scope.get("silent_excluded_filenames", []), sort_keys = True),
                "rare_class_policy": str(scope["rare_class_policy"]),
                "excluded_classes_json": json.dumps(scope["excluded_classes"], sort_keys = True),
                "class_scope_note": str(scope["class_scope_note"]),
            }
            
            performance.append({
                "model": model, 
                "snr": snr, 
                "n_seeds": len(expected_seeds), 
                "unseen_mean": u_mean, 
                "unseen_ci95": u_ci, 
                "seen_mean": s_mean, 
                "seen_ci95": s_ci, 
                **scope_fields
            })

            gaps.append({
                "model": model, 
                "snr": snr, 
                "n_seeds": len(expected_seeds), 
                "gap_mean": g_mean, 
                "gap_ci95": g_ci, 
                "seed_gaps": gap_values, 
                **scope_fields
            })
            
    return performance, gaps


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames = rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def _snr_tick_label(snr: str) -> str:
    if snr == "Clean":
        return "Clean"
        
    if snr.endswith("dB"):
        value = snr[:-2]
        try:
            return f"+{value}" if float(value) > 0 else value
        except ValueError:
            return value
    return snr


def write_outputs(performance, gaps, results_dir = "results", snrs = DEFAULT_SNRS, fraction = 1.0):
    snrs = tuple(snrs)
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
        
    suffix = "" if abs(float(fraction) - 1.0) < 1e-8 else f"_frac{float(fraction)}"
    if not snrs:
        raise ValueError("Cannot write aggregate output for an empty SNR list.")
        
    out = Path(results_dir); out.mkdir(parents = True, exist_ok = True)
    _write_csv(out / f"stress_test_summary{suffix}.csv", [{**row, "fraction": float(fraction)} for row in performance])
    _write_csv(out / f"generalization_gap_summary{suffix}.csv", [{**row, "fraction": float(fraction)} for row in gaps])

    latex = [
        r"\begin{table}[t]", 
        r"\centering", 
        f"\\caption{{Unseen-habitat test accuracy at {float(fraction) * 100:g}\\% labeled data (mean $\\pm$ 95\\% Student's $t$ CI across seeds).}}", 
        f"\\begin{{tabular}}{{l{'c' * len(snrs)}}}", 
        r"\toprule", "Model & " + " & ".join("Clean" if s == "Clean" else f"{_snr_tick_label(s)} dB" for s in snrs) + " " + chr(92) * 2, 
        r"\midrule"
    ]
    
    for model in DEFAULT_MODELS:
        row = [model.replace("_", " ").title()]
        for snr in snrs:
            item = next(r for r in performance if r["model"] == model and r["snr"] == snr)
            row.append(f"{item['unseen_mean'] * 100:.1f} $\\pm$ {item['unseen_ci95'] * 100:.1f}")
            
        latex.append(" & ".join(row) + " " + chr(92) * 2)
        
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (out / f"table_stress_test{suffix}.tex").write_text("\n".join(latex), encoding = "utf-8")

    latex_gap = [
        r"\begin{table}[t]", 
        r"\centering", 
        f"\\caption{{Seen minus unseen habitat accuracy gap at {float(fraction) * 100:g}\\% labeled data (percentage points; mean $\\pm$ 95\\% Student's $t$ CI across paired seed level gaps).}}", 
        f"\\begin{{tabular}}{{l{'c' * len(snrs)}}}", r"\toprule", "Model & " + " & ".join("Clean" if s == "Clean" else f"{_snr_tick_label(s)} dB" for s in snrs) + " " + chr(92) * 2, 
        r"\midrule"
    ]

    for model in DEFAULT_MODELS:
        row = [model.replace("_", " ").title()]
        for snr in snrs:
            item = next(r for r in gaps if r["model"] == model and r["snr"] == snr)
            row.append(f"{item['gap_mean'] * 100:.1f} $\\pm$ {item['gap_ci95'] * 100:.1f}")
            
        latex_gap.append(" & ".join(row) + " " + chr(92) * 2)
        
    latex_gap.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (out / f"table_generalization_gap{suffix}.tex").write_text("\n".join(latex_gap), encoding = "utf-8")

    labels = {
        "moco_habitat": "Habitat SSL", 
        "moco_standard": "Standard SSL", 
        "moco_white_noise": "White noise SSL", 
        "supervised_baseline": "Supervised clean", 
        "supervised_noise_baseline": "Supervised + habitat noise"
    }

    fig, ax = plt.subplots(figsize = (7.3, 4.6))
    x = np.arange(len(snrs))
    
    for model in DEFAULT_MODELS:
        rows = [next(r for r in performance if r["model"] == model and r["snr"] == snr) for snr in snrs]
        ax.errorbar(x, [r["unseen_mean"] for r in rows], yerr = [r["unseen_ci95"] for r in rows], marker = "o", capsize = 3, label = labels[model])
        
    ax.set_xticks(x, [_snr_tick_label(s) for s in snrs])
    ax.set_xlabel("SNR (dB); Clean is uncorrupted")
    ax.set_ylabel("Accuracy on unseen habitat noise")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha = 0.25); ax.legend(frameon = False); fig.tight_layout()
    fig.savefig(out / f"figure_unseen_habitat_stress_test{suffix}.png", dpi = 180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    for model in DEFAULT_MODELS:
        rows = [next(r for r in gaps if r["model"] == model and r["snr"] == snr) for snr in snrs]
        ax.errorbar(x, [r["gap_mean"] for r in rows], yerr = [r["gap_ci95"] for r in rows], marker = "o", capsize = 3, label = labels[model])
        
    ax.axhline(0, linewidth = 1)
    ax.set_xticks(x, [_snr_tick_label(s) for s in snrs])
    ax.set_xlabel("SNR (dB); Clean is uncorrupted")
    ax.set_ylabel("Seen minus unseen accuracy")
    ax.grid(True, alpha = 0.25); ax.legend(frameon = False); fig.tight_layout()
    fig.savefig(out / f"figure_generalization_gap{suffix}.png", dpi = 180); plt.close(fig)
    
    print(f"Wrote {out / f'table_stress_test{suffix}.tex'} and {out / f'table_generalization_gap{suffix}.tex'}")
    print(f"Wrote {out / f'figure_unseen_habitat_stress_test{suffix}.png'} and {out / f'figure_generalization_gap{suffix}.png'}")


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--results-dir", default = "results")
    parser.add_argument("--expected-seeds", nargs = "+", type=int, default = list(DEFAULT_SEEDS))
    parser.add_argument(
        "--config", default = "configs/stress_test.yaml",
        help = "Stress test YAML. Its evaluation.snr_levels are used unless --expected-snrs-csv is supplied.",
    )
    parser.add_argument(
        "--expected-snrs-csv", default = None,
        help = "Optional explicit comma separated SNR keys as stored in results, e.g. Clean,20dB,10dB,0dB,-5dB.",
    )
    parser.add_argument("--fraction", type=float, default = 1.0, help = "Label fraction used to train the evaluated heads/backbone (default: 1.0).")
    
    args = parser.parse_args(argv)
    if not 0.0 < args.fraction <= 1.0:
        parser.error("--fraction must be in (0, 1].")
        
    if args.expected_snrs_csv is None:
        _, _, configured_labels = load_stress_snr_config(args.config)
        expected_snrs = tuple(configured_labels)
    else:
        expected_snrs = tuple(item.strip() for item in args.expected_snrs_csv.split(",") if item.strip())
        
    if not expected_snrs:
        raise ValueError("No expected SNR conditions were configured.")
    if len(set(expected_snrs)) != len(expected_snrs):
        raise ValueError(f"Duplicate expected SNR labels: {expected_snrs}.")
        
    payloads = load_all_seed_results(args.results_dir, tuple(args.expected_seeds), expected_snrs, args.fraction)
    performance, gaps = summarize(payloads, tuple(args.expected_seeds), expected_snrs)
    write_outputs(performance, gaps, args.results_dir, expected_snrs, args.fraction)


if __name__ == "__main__":
    main()
