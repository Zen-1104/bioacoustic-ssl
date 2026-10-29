"""Aggregate completed, uncapped real soundscape metrics across model seeds."""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from utils.preflight_metadata import validate_preflight_audit
from scipy import stats
from sklearn.metrics import average_precision_score

DEFAULT_SEEDS = (42, 123, 999)
DEFAULT_MODELS = ("habitat", "standard", "white_noise", "supervised", "supervised_noise")
EXTERNAL_MODELS = ("panns", "birdnet")


def mean_t_interval(values, confidence: float = 0.95) -> tuple[float, float]:
    array = np.asarray(list(values), dtype = float)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("A finite metric from at least two seeds is required for a confidence interval.")

    return float(array.mean()), float(stats.sem(array) * stats.t.ppf((1 + confidence) / 2, len(array) - 1))


def load_soundscape_records(results_dir: str | Path, fraction: float, seeds = DEFAULT_SEEDS,
                            models = DEFAULT_MODELS) -> dict[int, dict]:
    root = Path(results_dir)
    records = {}
    reference_identity = None
    reference_class_map = None
    reference_class_scope = None
    reference_preflight_audit = None

    for seed in seeds:
        pattern = str(root / f"real_soundscape_s{int(seed)}_frac{str(float(fraction))}_*.json")

        matches = sorted(Path(path) for path in glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No soundscape summary found for seed = {seed}, fraction = {fraction} using {pattern}."
            )

        candidates = []
        incomplete_candidates = []
        for candidate in matches:
            candidate_data = json.loads(candidate.read_text(encoding = "utf-8"))

            if int(candidate_data.get("seed", -1)) != int(seed) or abs(float(candidate_data.get("fraction", -1)) - float(fraction)) > 1e-8:
                continue
            if candidate_data.get("capped_smoke_run") or candidate_data.get("identity", {}).get("max_recordings") is not None:
                continue
            if candidate_data.get("incomplete_audio_inventory") or int(candidate_data.get("n_annotation_recordings_missing_audio", 0)) > 0:
                incomplete_candidates.append(candidate)
                continue

            if set(models).issubset(candidate_data.get("models", {})):
                candidates.append((candidate, candidate_data))

        if not candidates and incomplete_candidates:
            raise ValueError(f"Refusing incomplete soundscape audio inventory in '{incomplete_candidates[0]}'.")

        if not candidates:
            raise FileNotFoundError(
                f"No uncapped soundscape summary for seed = {seed}, fraction = {fraction} contains all requested models {list(models)}."
            )

        most_complete = max(len(data.get("models", {})) for _path, data in candidates)
        candidates = [(path, data) for path, data in candidates if len(data.get("models", {})) == most_complete]

        if len(candidates) != 1:
            raise ValueError(
                f"Multiple equally complete soundscape summaries exist for seed = {seed}, fraction = {fraction}; move stale outputs aside."
            )

        path, data = candidates[0]
        if int(data.get("schema_version", -1)) != 2:
            raise ValueError(f"Soundscape result '{path}' uses an older schema.")

        preflight_audit = validate_preflight_audit(data, context = f"Soundscape result '{path}'")
        if reference_preflight_audit is None:
            reference_preflight_audit = preflight_audit
        elif preflight_audit != reference_preflight_audit:
            raise ValueError("Soundscape results use different BirdCLEF preflight exclusions/reason counts; do not aggregate them together.")

        identity = data.get("identity", {})
        class_map = data.get("active_label_map")

        required_scope = {
            "full_label_map",
            "n_active_classes",
            "n_total_classes",
            "n_excluded_classes",
            "n_silent_files_excluded",
            "silent_excluded_filenames",
            "rare_class_policy",
            "excluded_classes",
            "class_scope_note",
        }

        missing_scope = required_scope - set(data)
        if missing_scope or not isinstance(class_map, dict) or not class_map:
            raise ValueError(
                f"Soundscape result '{path}' lacks a complete active/excluded class scope "
                f"({sorted(missing_scope)}); rerun evaluation with v1.6.3 before aggregation."
            )

        if not isinstance(data["full_label_map"], dict) or not isinstance(data["excluded_classes"], list) or not isinstance(data["silent_excluded_filenames"], list):
            raise ValueError(f"Malformed class scope metadata in '{path}'.")

        if int(data["n_active_classes"]) != len(class_map) or int(data["n_total_classes"]) != len(data["full_label_map"]):
            raise ValueError(f"Class-scope counts disagree with label maps in '{path}'.")
        if int(data["n_excluded_classes"]) != len(data["excluded_classes"]) or int(data["n_silent_files_excluded"]) != len(set(map(str, data["silent_excluded_filenames"]))):
            raise ValueError(f"Class/file-exclusion counts disagree with exclusion manifests in '{path}'.")
        if int(data["n_active_classes"]) < 1 or int(data["n_total_classes"]) < int(data["n_active_classes"]):
            raise ValueError(f"Invalid active/total class counts in '{path}'.")

        if not identity.get("annotation_csv_sha256") or not identity.get("recording_names_sha256"):
            raise ValueError(f"Missing annotation/recording identity metadata in '{path}'.")
        if identity.get("background_seed") is None:
            raise ValueError(f"Soundscape result '{path}' has no independent background_seed; rerun with v1.6.3 to ensure identical background samples across model seeds.")

        comparable_identity = {
            key: identity.get(key) for key in (
                "annotation_csv_sha256",
                "recording_names_sha256",
                "sample_rate",
                "clip_sec",
                "max_recordings",
                "background_segments_per_recording",
                "background_seed",
                "min_annotation_coverage",
            )
        }

        class_scope = {
            **preflight_audit,
            "n_active_classes": int(data["n_active_classes"]),
            "n_total_classes": int(data["n_total_classes"]),
            "n_excluded_classes": int(data["n_excluded_classes"]),
            "n_silent_files_excluded": int(data["n_silent_files_excluded"]),
            "silent_excluded_filenames": sorted(map(str, data["silent_excluded_filenames"])),
            "rare_class_policy": str(data["rare_class_policy"]),
            "excluded_classes": data["excluded_classes"],
            "class_scope_note": str(data["class_scope_note"]),
        }

        if comparable_identity.get("max_recordings") is not None:
            raise ValueError(f"Result '{path}' is capped and cannot be aggregated as a paper result.")

        if reference_identity is None:
            reference_identity = comparable_identity
            reference_class_map = class_map
            reference_class_scope = class_scope
        elif comparable_identity != reference_identity or class_map != reference_class_map or class_scope != reference_class_scope:
            raise ValueError("Soundscape runs do not use identical annotations, recording inventory, fixed background sampling, sample rate, clip duration and class scope.")

        if "annotation_code_coverage_fraction" not in data or "minimum_annotation_coverage_required" not in data:
            raise ValueError(f"Soundscape result '{path}' lacks the annotation coverage gate metadata; rerun with the updated evaluator.")

        coverage = float(data["annotation_code_coverage_fraction"])
        required_coverage = float(data["minimum_annotation_coverage_required"])
        if not np.isfinite(coverage) or not 0 <= coverage <= 1 or coverage < required_coverage:
            raise ValueError(f"Soundscape result '{path}' failed its recorded annotation coverage gate.")

        if "n_background_segments" not in data or "background_segments_per_recording" not in data:
            raise ValueError(f"Soundscape result '{path}' lacks background negative protocol metadata; rerun with the updated evaluator.")
        if int(data["background_segments_per_recording"]) > 0 and int(data["n_background_segments"]) < 1:
            raise ValueError(f"Soundscape result '{path}' requested background negatives but evaluated none; false alarm proxy is unavailable.")

        model_rows = data.get("models", {})
        missing_models = sorted(set(models) - set(model_rows))
        if missing_models:
            raise ValueError(f"Soundscape result '{path}' is missing required models: {missing_models}.")

        for model in models:
            metrics = model_rows[model]

            for metric in ("macro_average_precision", "micro_average_precision"):
                value = float(metrics.get(metric, float("nan")))

                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid {metric} for {model} in '{path}'.")

        records[int(seed)] = {
            **data, "source_file": str(path),
            "comparable_identity": comparable_identity,
            "comparable_class_scope": class_scope
        }

    return records


def _ap_pair(targets: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Macro/micro AP on classes with both positive and negative targets."""

    valid = [i for i in range(targets.shape[1]) if 0 < int(targets[:, i].sum()) < len(targets)]
    if not valid:
        raise ValueError("No class with both positive and negative targets in this bootstrap draw.")

    aps = [average_precision_score(targets[:, i], scores[:, i]) for i in valid]
    macro = float(np.mean(aps))
    micro = float(average_precision_score(targets[:, valid].reshape(-1), scores[:, valid].reshape(-1)))

    return macro, micro


def recording_cluster_bootstrap(prediction_path: str | Path, model: str, *, replicates: int = 200,
                                 seed: int = 42) -> dict:
    """Bootstrap recordings, not segments, to respect within recording correlation."""
    path = Path(prediction_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction archive required for recording cluster bootstrap is missing: '{path}'.")

    with np.load(path, allow_pickle = False) as archive:
        targets = archive["targets"].astype(np.uint8, copy = True)

        scores_key = f"scores_{model}"
        if scores_key not in archive:
            raise ValueError(f"Prediction archive '{path}' lacks {scores_key}.")

        scores = archive[scores_key].astype(np.float64, copy = True)
        file_names = archive["file_names"].astype(str, copy = True)
        class_names = archive["class_names"].astype(str, copy = True)

    if (targets.ndim != 2 or targets.shape != scores.shape or len(file_names) != len(targets)
            or len(class_names) != targets.shape[1] or not np.isfinite(scores).all()):
        raise ValueError(f"Malformed prediction archive '{path}': targets/scores/files/classes do not align or scores are non finite.")

    unique_recordings = sorted(set(file_names.tolist()))
    if len(unique_recordings) < 2:
        raise ValueError("Recording cluster bootstrap requires at least two recordings.")

    group_rows = {name: np.flatnonzero(file_names == name) for name in unique_recordings}
    rng = np.random.default_rng(int(seed))
    macro_values, micro_values = [], []

    for _ in range(int(replicates)):
        draw = rng.integers(0, len(unique_recordings), size = len(unique_recordings))
        chosen_names = [unique_recordings[int(i)] for i in draw]
        indices = np.concatenate([group_rows[name] for name in chosen_names])

        try:
            macro, micro = _ap_pair(targets[indices], scores[indices])
        except ValueError:
            continue

        if np.isfinite(macro) and np.isfinite(micro):
            macro_values.append(macro)
            micro_values.append(micro)

    min_valid = max(50, int(np.ceil(0.8 * int(replicates))))
    if len(macro_values) < min_valid:
        raise ValueError(
            f"Only {len(macro_values)}/{replicates} recording cluster bootstrap draws had defined AP metrics; "
            f"need at least {min_valid}. More recordings or broader class coverage may be required."
        )
    return {
        "model": model,
        "n_recording_clusters": len(unique_recordings),
        "n_segments": int(len(targets)),
        "bootstrap_replicates_requested": int(replicates),
        "bootstrap_replicates_valid": len(macro_values),
        "macro_average_precision_point": float(_ap_pair(targets, scores)[0]),
        "macro_average_precision_ci95_low": float(np.percentile(macro_values, 2.5)),
        "macro_average_precision_ci95_high": float(np.percentile(macro_values, 97.5)),
        "micro_average_precision_point": float(_ap_pair(targets, scores)[1]),
        "micro_average_precision_ci95_low": float(np.percentile(micro_values, 2.5)),
        "micro_average_precision_ci95_high": float(np.percentile(micro_values, 97.5)),
        "prediction_archive": str(path),
    }


def _prediction_archive_path(record: dict, results_dir: str | Path) -> Path:
    raw_value = str(record.get("prediction_archive", "")).strip()
    if not raw_value:
        raise FileNotFoundError(f"Soundscape summary '{record.get('source_file')}' has no prediction_archive path.")

    raw = Path(raw_value).expanduser()
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, Path(results_dir) / raw, Path(results_dir) / raw.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Prediction archive from '{record.get('source_file')}' cannot be resolved; tried "
        f"{[str(candidate) for candidate in candidates]}."
    )


def collect_recording_cluster_bootstraps(records: dict[int, dict], results_dir: str | Path, *,
                                          seeds = DEFAULT_SEEDS, models = DEFAULT_MODELS,
                                          replicates: int = 200) -> list[dict]:
    rows = []
    for model in models:
        for seed in seeds:
            record = records[int(seed)]
            prediction = _prediction_archive_path(record, results_dir)
            result = recording_cluster_bootstrap(
                prediction,
                model,
                replicates = replicates,
                seed = 91000 + int(seed),
            )

            result["seed"] = int(seed)
            result["fraction"] = float(record["fraction"])
            result["source_summary"] = record["source_file"]
            rows.append(result)

    return rows

def summarize_soundscape_records(records: dict[int, dict], *, seeds = DEFAULT_SEEDS, models = DEFAULT_MODELS) -> list[dict]:
    rows = []
    for model in models:
        row = {
            "model": model,
            "n_seeds": len(seeds),
            "n_recordings_per_seed": json.dumps([int(records[int(seed)]["n_recordings_evaluated"]) for seed in seeds]),
            "n_annotated_segments_per_seed": json.dumps([int(records[int(seed)]["n_annotated_segments"]) for seed in seeds]),
            "n_background_segments_per_seed": json.dumps([int(records[int(seed)].get("n_background_segments", 0)) for seed in seeds]),
            "seed_values_source_files": json.dumps([records[int(seed)]["source_file"] for seed in seeds]),
            "n_active_classes": records[int(seeds[0])]["comparable_class_scope"]["n_active_classes"],
            "n_total_classes": records[int(seeds[0])]["comparable_class_scope"]["n_total_classes"],
            "n_excluded_classes": records[int(seeds[0])]["comparable_class_scope"]["n_excluded_classes"],
            "n_silent_files_excluded": records[int(seeds[0])]["comparable_class_scope"]["n_silent_files_excluded"],
            "silent_excluded_filenames_json": json.dumps(records[int(seeds[0])]["comparable_class_scope"]["silent_excluded_filenames"], sort_keys = True),
            "rare_class_policy": records[int(seeds[0])]["comparable_class_scope"]["rare_class_policy"],
            "excluded_classes_json": json.dumps(records[int(seeds[0])]["comparable_class_scope"]["excluded_classes"], sort_keys = True),
            "class_scope_note": records[int(seeds[0])]["comparable_class_scope"]["class_scope_note"],
            "n_preflight_excluded_files": records[int(seeds[0])]["comparable_class_scope"]["n_preflight_excluded_files"],
            "preflight_excluded_filenames_json": json.dumps(records[int(seeds[0])]["comparable_class_scope"]["preflight_excluded_filenames"], sort_keys = True),
            "n_preflight_read_error_files_excluded": records[int(seeds[0])]["comparable_class_scope"]["n_preflight_read_error_files_excluded"],
            "preflight_read_error_filenames_json": json.dumps(records[int(seeds[0])]["comparable_class_scope"]["preflight_read_error_filenames"], sort_keys = True),
            "preflight_reason_histogram_json": json.dumps(records[int(seeds[0])]["comparable_class_scope"]["preflight_reason_histogram"], sort_keys = True),
            "preflight_read_error_fraction": records[int(seeds[0])]["comparable_class_scope"]["preflight_read_error_fraction"],
            "preflight_excluded_fraction": records[int(seeds[0])]["comparable_class_scope"]["preflight_excluded_fraction"],
            "background_seed": records[int(seeds[0])]["comparable_identity"]["background_seed"],
        }

        for metric in ("macro_average_precision", "micro_average_precision"):
            values = [float(records[int(seed)]["models"][model][metric]) for seed in seeds]
            mean, half = mean_t_interval(values)

            row[f"{metric}_seed_values"] = json.dumps(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_half_width"] = half

        for metric in ("background_top1_confidence_mean", "background_top1_confidence_ge_0_5_rate"):
            values = [records[int(seed)]["models"][model].get(metric) for seed in seeds]
            finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]

            row[f"{metric}_seed_values"] = json.dumps(values)
            row[f"{metric}_mean"] = float(np.mean(finite)) if finite else None
            row[f"{metric}_n_valid_seeds"] = len(finite)

        rows.append(row)
    return rows


def write_soundscape_outputs(rows: list[dict], output_dir: str | Path, cluster_rows: list[dict] | None = None) -> tuple[Path, Path, Path | None]:
    out = Path(output_dir)
    out.mkdir(parents = True, exist_ok = True)

    csv_path = out / "real_soundscape_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames = list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tex_path = out / "table_real_soundscape_macro_ap.tex"
    display = {
        "habitat": "Habitat-noise SSL",
        "standard": "Standard SSL",
        "white_noise": "White-noise SSL",
        "supervised": "Supervised clean",
        "supervised_noise": "Supervised + habitat noise",
        "panns": "PANNs Cnn14",
        "birdnet": "BirdNET acoustic v3.0",
    }

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Western Ghats soundscape macro average precision (mean $\pm$ 95\% Student's $t$ CI across model seeds).}",
        r"\begin{tabular}{lc}",
        r"\toprule", "Model & Macro AP (95\\% CI) " + chr(92) * 2,
        r"\midrule",
    ]

    for row in rows:
        mean = float(row["macro_average_precision_mean"])
        half = float(row["macro_average_precision_ci95_half_width"])
        tex.append(f"{display.get(row['model'], row['model'])} & {mean:.3f} $\\pm$ {half:.3f} " + chr(92) * 2)

    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    tex_path.write_text("\n".join(tex), encoding = "utf-8")

    cluster_path = None
    if cluster_rows:
        cluster_path = out / "real_soundscape_recording_cluster_bootstrap.csv"

        with cluster_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames = list(cluster_rows[0]))
            writer.writeheader()
            writer.writerows(cluster_rows)

    return csv_path, tex_path, cluster_path


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--results-dir", default = "results")
    parser.add_argument("--output-dir", default = None)
    parser.add_argument("--fraction", type=float, default = 0.10)
    parser.add_argument("--seeds", nargs="+", type=int, default = list(DEFAULT_SEEDS))
    parser.add_argument("--models", nargs="+", choices = list(DEFAULT_MODELS + EXTERNAL_MODELS), default=list(DEFAULT_MODELS))
    parser.add_argument("--bootstrap-replicates", type=int, default = 200,
                        help = "Recording-cluster bootstrap draws per model and seed; at least 50 valid draws are required.")

    args = parser.parse_args(argv)
    if not 0 < args.fraction <= 1 or len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        raise ValueError("fraction must be in (0,1] and at least two distinct seeds are required.")
    if len(set(args.models)) != len(args.models) or not args.models:
        raise ValueError("At least one non duplicated model is required.")
    if args.bootstrap_replicates < 50:
        raise ValueError("--bootstrap-replicates must be at least 50.")

    records = load_soundscape_records(args.results_dir, args.fraction, seeds = args.seeds, models = args.models)
    rows = summarize_soundscape_records(records, seeds = args.seeds, models = args.models)

    cluster_rows = collect_recording_cluster_bootstraps(
        records,
        args.results_dir,
        seeds = args.seeds,
        models = args.models,
        replicates = args.bootstrap_replicates,
    )

    outputs = write_soundscape_outputs(rows, args.output_dir or args.results_dir, cluster_rows)
    print(json.dumps({"n_seed_summaries": len(records), "models": list(args.models), "outputs": [str(x) for x in outputs if x is not None]}, indent = 2))


if __name__ == "__main__":
    main()
