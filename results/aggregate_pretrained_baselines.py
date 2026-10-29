"""Validate and aggregate optional PANNs/BirdNET frozen feature probe results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from scipy import stats

from utils.preflight_metadata import validate_preflight_audit

DEFAULT_SEEDS = (42, 123, 999)
DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 1.00)
DEFAULT_MODELS = ("panns", "birdnet")
METRICS = ("test_accuracy", "test_macro_f1", "test_balanced_accuracy")


def mean_t_interval(values, confidence: float = 0.95) -> tuple[float, float]:
    array = np.asarray(list(values), dtype = float)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("A finite metric from at least two independent seeds is required for an interval.")

    mean = float(array.mean())
    half = float(stats.sem(array) * stats.t.ppf((1 + confidence) / 2, len(array) - 1))

    return mean, half

def _tag(value: float) -> str:
    return str(float(value))

def load_external_records(results_dir: str | Path, *, seeds = DEFAULT_SEEDS, fractions = DEFAULT_FRACTIONS,
                          models = DEFAULT_MODELS, allow_three_crop_analysis: bool = False) -> dict[tuple[str, int, float], dict]:
    root = Path(results_dir)
    records: dict[tuple[str, int, float], dict] = {}
    fingerprints: set[str] = set()
    per_seed_maps: dict[int, dict] = {}
    reference_active_map = None
    per_model_names: dict[str, set[str]] = {model: set() for model in models}
    per_model_hashes: dict[str, set[str]] = {model: set() for model in models}
    per_model_encoder_identities: dict[str, set[str]] = {model: set() for model in models}
    reference_class_scope = None
    reference_crop_policy = None

    for model in models:
        for fraction in fractions:
            for seed in seeds:
                path = root / f"pretrained_probe_{model}_s{int(seed)}_frac{_tag(fraction)}.json"

                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing external-baseline result '{path}'. Run all requested seeds and fractions before aggregation."
                    )

                data = json.loads(path.read_text(encoding = "utf-8"))
                if int(data.get("schema_version", -1)) != 3:
                    raise ValueError(f"External-baseline result '{path}' uses an older extraction schema.")

                required = {"schema_version", "mode", "variant", "seed", "fraction", "dataset_fingerprint",
                            "active_label_map", "model_name", "external_checkpoint", "external_checkpoint_sha256",
                            "encoder_identity", "n_active_classes", "n_total_classes", "n_excluded_classes",
                            "excluded_classes", "n_silent_files_excluded", "silent_excluded_filenames",
                            "rare_class_policy", "class_scope_note", "num_fixed_crops_requested",
                            "fixed_crop_aggregation", "crop_policy", "feature_extraction_spec_version", "minimum_rms",
                            "fixed_crop_count_min", "fixed_crop_count_mean", "fixed_crop_count_max",
                            "fixed_crop_shortfall_filenames", *METRICS}

                absent = required - set(data)
                if absent:
                    raise ValueError(f"Result '{path}' is missing fields {sorted(absent)}.")

                if data["mode"] != "external_pretrained_linear_probe" or data["variant"] != model:
                    raise ValueError(f"Unexpected model/mode metadata in '{path}'.")

                if int(data["seed"]) != int(seed) or abs(float(data["fraction"]) - float(fraction)) > 1e-8:
                    raise ValueError(f"Seed/fraction does not match expected result path '{path}'.")

                if not data["dataset_fingerprint"]:
                    raise ValueError(f"Missing dataset fingerprint in '{path}'.")

                for metric in METRICS:
                    value = float(data[metric])

                    if not np.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f"Invalid {metric} = {value} in '{path}'.")

                if not data["active_label_map"]:
                    raise ValueError(f"Empty active class map in '{path}'.")

                if int(data.get("num_classes", -1)) != len(data["active_label_map"]):
                    raise ValueError(f"Class count and class map disagree in '{path}'.")

                preflight_audit = validate_preflight_audit(data, context = f"External baseline result '{path}'")
                class_scope = {
                    **preflight_audit,
                    "n_active_classes": int(data["n_active_classes"]),
                    "n_total_classes": int(data["n_total_classes"]),
                    "n_excluded_classes": int(data["n_excluded_classes"]),
                    "excluded_classes": data["excluded_classes"],
                    "n_silent_files_excluded": int(data["n_silent_files_excluded"]),
                    "silent_excluded_filenames": sorted(data["silent_excluded_filenames"]),
                    "rare_class_policy": str(data["rare_class_policy"]),
                    "class_scope_note": str(data["class_scope_note"]),
                }

                if class_scope["n_active_classes"] != len(data["active_label_map"]) or class_scope["n_total_classes"] < class_scope["n_active_classes"] or class_scope["n_excluded_classes"] != len(class_scope["excluded_classes"]):
                    raise ValueError(f"Inconsistent class scope counts in '{path}'.")

                if reference_class_scope is None:
                    reference_class_scope = class_scope
                elif class_scope != reference_class_scope:
                    raise ValueError("External baseline outputs use different active/excluded class scopes.")

                crop_policy = {
                    "num_fixed_crops_requested": int(data["num_fixed_crops_requested"]),
                    "fixed_crop_aggregation": str(data["fixed_crop_aggregation"]),
                    "crop_policy": str(data["crop_policy"]),
                    "feature_extraction_spec_version": str(data["feature_extraction_spec_version"]),
                    "minimum_rms": float(data["minimum_rms"]),
                    "fixed_crop_shortfall_filenames": sorted(data["fixed_crop_shortfall_filenames"]),
                }

                if crop_policy["num_fixed_crops_requested"] not in (1, 3):
                    raise ValueError(f"Unsupported fixed crop count in '{path}'; choose 1 or 3.")

                if crop_policy["num_fixed_crops_requested"] == 3 and not allow_three_crop_analysis:
                    raise ValueError(
                        f"Result '{path}' uses three crop averaging, which is exploratory rather than the primary comparison. "
                        "Rerun external encoders with --num-fixed-crops 1, or pass --allow-three-crop-analysis to produce a separately labelled exploratory table."
                    )
                if not np.isfinite(crop_policy["minimum_rms"]) or crop_policy["minimum_rms"] <= 0:
                    raise ValueError(f"Invalid minimum_rms in '{path}'.")

                if reference_crop_policy is None:
                    reference_crop_policy = crop_policy

                elif crop_policy != reference_crop_policy:
                    raise ValueError("External baseline outputs use different fixed crop policies/spec versions; rerun them with matched settings.")

                key = (model, int(seed), float(fraction))
                if key in records:
                    raise ValueError(f"Duplicate model/seed/fraction record {key}.")

                records[key] = {**data, "source_file": str(path)}
                fingerprints.add(str(data["dataset_fingerprint"]))
                per_seed_maps.setdefault(int(seed), data["active_label_map"])

                if reference_active_map is None:
                    reference_active_map = data["active_label_map"]
                elif data["active_label_map"] != reference_active_map:
                    raise ValueError("Active class maps differ across seeds/models/fractions; these results are not directly comparable.")

                encoder_identity = data.get("encoder_identity")
                if not isinstance(encoder_identity, dict) or not encoder_identity:
                    raise ValueError(f"Missing encoder identity metadata in '{path}'.")

                checkpoint_sha256 = str(data.get("external_checkpoint_sha256", ""))
                if len(checkpoint_sha256) != 64 or any(char not in "0123456789abcdef" for char in checkpoint_sha256.lower()):
                    raise ValueError(f"Missing or malformed external checkpoint SHA-256 in '{path}'.")

                if str(encoder_identity.get("model_file_sha256", "")) != checkpoint_sha256:
                    raise ValueError(f"Checkpoint SHA-256 disagrees with encoder identity in '{path}'.")

                if not encoder_identity.get("package_version") or not encoder_identity.get("model_version"):
                    raise ValueError(f"Encoder identity in '{path}' does not record package/model versions.")

                per_model_names[model].add(str(data["model_name"]))
                per_model_hashes[model].add(checkpoint_sha256)
                per_model_encoder_identities[model].add(json.dumps(encoder_identity, sort_keys = True, separators = (",", ":")))

                if data["active_label_map"] != per_seed_maps[int(seed)]:
                    raise ValueError(
                        f"Active class map differs across models/fractions for seed {seed}; results are not directly comparable."
                    )

    if len(fingerprints) != 1:
        raise ValueError("External baseline runs use different BirdCLEF inventory fingerprints.")

    for model in models:
        if (len(per_model_names[model]) != 1 or len(per_model_hashes[model]) != 1
                or len(per_model_encoder_identities[model]) != 1):
            raise ValueError(f"Model name/hash/package/model identity changes across runs for '{model}'.")

    return records


def summarize_external_records(records: dict, *, seeds = DEFAULT_SEEDS, fractions = DEFAULT_FRACTIONS,
                               models = DEFAULT_MODELS) -> list[dict]:
    rows = []
    for model in models:
        for fraction in fractions:
            row = {"model": model, "fraction": float(fraction), "n_seeds": len(seeds)}
            reference = records[(model, int(seeds[0]), float(fraction))]

            row.update({
                "n_active_classes": int(reference["n_active_classes"]),
                "n_total_classes": int(reference["n_total_classes"]),
                "n_excluded_classes": int(reference["n_excluded_classes"]),
                "excluded_classes_json": json.dumps(reference["excluded_classes"], sort_keys = True),
                "n_silent_files_excluded": int(reference["n_silent_files_excluded"]),
                "silent_excluded_filenames_json": json.dumps(reference.get("silent_excluded_filenames", []), sort_keys = True),
                "rare_class_policy": str(reference["rare_class_policy"]),
                "class_scope_note": str(reference["class_scope_note"]),
                "n_preflight_excluded_files": int(reference["n_preflight_excluded_files"]),
                "preflight_excluded_filenames_json": json.dumps(reference["preflight_excluded_filenames"], sort_keys = True),
                "n_preflight_read_error_files_excluded": int(reference["n_preflight_read_error_files_excluded"]),
                "preflight_read_error_filenames_json": json.dumps(reference["preflight_read_error_filenames"], sort_keys = True),
                "preflight_reason_histogram_json": json.dumps(reference["preflight_reason_histogram"], sort_keys = True),
                "preflight_read_error_fraction": float(reference["preflight_read_error_fraction"]),
                "preflight_excluded_fraction": float(reference["preflight_excluded_fraction"]),
                "num_fixed_crops_requested": int(reference["num_fixed_crops_requested"]),
                "crop_analysis_scope": "primary_single_center_crop" if int(reference["num_fixed_crops_requested"]) == 1 else "exploratory_three_crop",
                "fixed_crop_aggregation": str(reference["fixed_crop_aggregation"]),
                "fixed_crop_count_min": int(reference["fixed_crop_count_min"]),
                "fixed_crop_count_mean": float(reference["fixed_crop_count_mean"]),
                "fixed_crop_count_max": int(reference["fixed_crop_count_max"]),
                "fixed_crop_shortfall_filenames_json": json.dumps(reference["fixed_crop_shortfall_filenames"], sort_keys = True),
            })

            for metric in METRICS:
                values = [float(records[(model, int(seed), float(fraction))][metric]) for seed in seeds]
                mean, half = mean_t_interval(values)
                row[f"{metric}_seed_values"] = json.dumps(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_ci95_half_width"] = half
            rows.append(row)

    return rows


def write_outputs(rows: list[dict], output_dir: str | Path) -> tuple[Path, Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents = True, exist_ok = True)

    csv_path = output / "pretrained_baseline_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames = list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def write_latex(metric: str, path: Path) -> None:
        display = {"panns": "PANNs Cnn14", "birdnet": "BirdNET acoustic v3.0"}
        models = list(dict.fromkeys(row["model"] for row in rows))
        fractions = list(dict.fromkeys(float(row["fraction"]) for row in rows))
        column_spec = "l" + "c" * len(fractions)
        fraction_headers = " & ".join(f"{fraction:.0%}" for fraction in fractions)

        tex = [
            r"\begin{table}[t]", r"\centering",
            rf"\caption{{External pretrained linear probe {metric.replace('_', ' ')} (mean $\pm$ 95\% Student's $t$ CI across seeds). Evaluation used {rows[0]['num_fixed_crops_requested']} requested fixed crops per recording, mean pooled to one feature row; active classes {rows[0]['n_active_classes']}/{rows[0]['n_total_classes']} ({rows[0]['n_excluded_classes']} excluded; RARE_CLASS_POLICY = {rows[0]['rare_class_policy']}).}}",
            rf"\begin{{tabular}}{{{column_spec}}}", r"\toprule",
            "Model & " + fraction_headers + " " + r"\\", r"\midrule",
        ]

        for model in models:
            values = [display.get(model, model)]
            for fraction in fractions:
                item = next(row for row in rows if row["model"] == model and abs(float(row["fraction"]) - fraction) < 1e-8)
                values.append(f"{item[f'{metric}_mean'] * 100:.1f} $\\pm$ {item[f'{metric}_ci95_half_width'] * 100:.1f}")

            tex.append(" & ".join(values) + " " + chr(92) * 2)

        tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
        path.write_text("\n".join(tex), encoding = "utf-8")

    accuracy_tex = output / "table_pretrained_accuracy.tex"
    macro_f1_tex = output / "table_pretrained_macro_f1.tex"
    write_latex("test_accuracy", accuracy_tex)
    write_latex("test_macro_f1", macro_f1_tex)
    return csv_path, accuracy_tex, macro_f1_tex


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--results-dir", default = "results")
    parser.add_argument("--output-dir", default = None)
    parser.add_argument("--seeds", nargs="+", type=int, default = list(DEFAULT_SEEDS))
    parser.add_argument("--fractions", nargs="+", type=float, default = list(DEFAULT_FRACTIONS))
    parser.add_argument("--models", nargs="+", choices = list(DEFAULT_MODELS), default = list(DEFAULT_MODELS))
    parser.add_argument("--allow-three-crop-analysis", action = "store_true", help = "Permit a separate exploratory 3-crop table; this must not replace the 1-center-crop primary comparison.")

    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        raise ValueError("At least two distinct seeds are required for across-seed confidence intervals.")
    if len(set(args.fractions)) != len(args.fractions) or any(not 0 < value <= 1 for value in args.fractions):
        raise ValueError("Fractions must be unique and in (0,1].")
    if len(set(args.models)) != len(args.models):
        raise ValueError("Models must not be duplicated.")

    records = load_external_records(
        args.results_dir,
        seeds = args.seeds,
        fractions = args.fractions,
        models = args.models,
        allow_three_crop_analysis = args.allow_three_crop_analysis
    )

    rows = summarize_external_records(
        records,
        seeds = args.seeds,
        fractions = args.fractions,
        models = args.models
    )

    outputs = write_outputs(rows, args.output_dir or args.results_dir)

    print(json.dumps({"n_records": len(records), "outputs": [str(path) for path in outputs]}, indent = 2))

if __name__ == "__main__":
    main()
