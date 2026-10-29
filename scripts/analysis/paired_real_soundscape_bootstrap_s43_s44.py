from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


ARCHIVES = {
    43: Path("results/real_soundscape_pair_s43_20261023_094200.npz"),
    44: Path("results/real_soundscape_pair_s44_20261025.npz"),
}

DEFAULT_OUT = Path(
    "results/real_soundscape_aggregate_s43_s44_fixnames/"
    "paired_delta_bootstrap_20261025"
)


def load_archive(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction archive not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "targets", "file_names", "segment_start_seconds",
            "segment_types", "class_names",
            "scores_habitat", "scores_standard",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        return {key: archive[key].copy() for key in required}


def validate_archives(data: dict[int, dict[str, np.ndarray]]) -> None:
    reference = data[43]
    for seed in (43, 44):
        item = data[seed]
        for key in (
            "targets", "file_names", "segment_start_seconds",
            "segment_types", "class_names",
        ):
            if not np.array_equal(reference[key], item[key]):
                raise ValueError(
                    f"Archives differ in {key!r}; paired resampling is unsafe."
                )

        targets = item["targets"]
        for score_key in ("scores_habitat", "scores_standard"):
            scores = item[score_key]
            if scores.shape != targets.shape:
                raise ValueError(
                    f"Seed {seed}: {score_key} shape {scores.shape} "
                    f"does not match targets {targets.shape}."
                )
            if not np.isfinite(scores).all():
                raise ValueError(f"Seed {seed}: non-finite values in {score_key}.")


def ap_pair(
    targets: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    """Match the evaluator's class-filtered macro and micro AP definitions."""
    valid = [
        i for i in range(targets.shape[1])
        if np.unique(targets[:, i]).size == 2
    ]
    if not valid:
        raise ValueError("No class has both positive and negative targets.")

    macro = float(np.mean([
        average_precision_score(targets[:, i], scores[:, i])
        for i in valid
    ]))
    micro = float(average_precision_score(
        targets[:, valid].reshape(-1),
        scores[:, valid].reshape(-1),
    ))
    return macro, micro


def make_recording_groups(file_names: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    unique_names, inverse = np.unique(file_names, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    groups = [
        order[offsets[i]:offsets[i + 1]]
        for i in range(len(unique_names))
    ]
    return unique_names, groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired recording-cluster bootstrap for real soundscapes."
    )
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20261025)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.replicates < 50:
        raise ValueError("--replicates must be at least 50.")
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}. "
            "Choose a new --output-dir to avoid overwriting."
        )

    data = {seed: load_archive(path) for seed, path in ARCHIVES.items()}
    validate_archives(data)

    targets = data[43]["targets"]
    file_names = data[43]["file_names"]
    unique_names, groups = make_recording_groups(file_names)
    n_recordings = len(groups)

    if n_recordings < 2:
        raise ValueError("At least two recording clusters are required.")

    metrics = ("macro_average_precision", "micro_average_precision")
    seed_points: dict[int, dict[str, dict[str, float]]] = {}
    point_deltas: dict[int, dict[str, float]] = {}

    print(f"Segments: {len(targets)}", flush=True)
    print(f"Classes: {targets.shape[1]}", flush=True)
    print(f"Recording clusters: {n_recordings}", flush=True)
    print(f"Bootstrap replicates: {args.replicates}", flush=True)
    print("Computing point estimates...", flush=True)

    for seed in (43, 44):
        item = data[seed]
        h_macro, h_micro = ap_pair(targets, item["scores_habitat"])
        s_macro, s_micro = ap_pair(targets, item["scores_standard"])

        seed_points[seed] = {
            "habitat": {
                "macro_average_precision": h_macro,
                "micro_average_precision": h_micro,
            },
            "standard": {
                "macro_average_precision": s_macro,
                "micro_average_precision": s_micro,
            },
        }
        point_deltas[seed] = {
            "macro_average_precision": h_macro - s_macro,
            "micro_average_precision": h_micro - s_micro,
        }

        print(
            f"Seed {seed}: macro AP habitat={h_macro:.8f}, "
            f"standard={s_macro:.8f}, delta={h_macro - s_macro:+.8f}",
            flush=True,
        )
        print(
            f"Seed {seed}: micro AP habitat={h_micro:.8f}, "
            f"standard={s_micro:.8f}, delta={h_micro - s_micro:+.8f}",
            flush=True,
        )

    rng = np.random.default_rng(args.random_seed)
    boot = {
        metric: {43: [], 44: [], "mean_across_seeds": []}
        for metric in metrics
    }
    accepted = 0
    attempts = 0
    max_attempts = args.replicates * 5
    start = time.monotonic()

    print(
        "Starting paired bootstrap. Progress prints every 50 valid replicates.",
        flush=True,
    )

    while accepted < args.replicates:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not obtain {args.replicates} valid replicates; "
                f"only {accepted} were valid."
            )

        # One shared recording sample is used for both models and both seeds.
        sampled_groups = rng.integers(0, n_recordings, size=n_recordings)
        sampled_indices = np.concatenate([groups[i] for i in sampled_groups])
        sampled_targets = targets[sampled_indices]

        replicate_deltas: dict[int, dict[str, float]] = {}
        try:
            for seed in (43, 44):
                item = data[seed]
                h_macro, h_micro = ap_pair(
                    sampled_targets, item["scores_habitat"][sampled_indices]
                )
                s_macro, s_micro = ap_pair(
                    sampled_targets, item["scores_standard"][sampled_indices]
                )
                replicate_deltas[seed] = {
                    "macro_average_precision": h_macro - s_macro,
                    "micro_average_precision": h_micro - s_micro,
                }
        except ValueError:
            # An exceptionally degenerate sample can contain no usable class.
            continue

        accepted += 1
        for metric in metrics:
            d43 = replicate_deltas[43][metric]
            d44 = replicate_deltas[44][metric]
            boot[metric][43].append(d43)
            boot[metric][44].append(d44)
            boot[metric]["mean_across_seeds"].append((d43 + d44) / 2.0)

        if accepted % 50 == 0 or accepted == args.replicates:
            elapsed = time.monotonic() - start
            print(
                f"Completed {accepted}/{args.replicates} valid replicates "
                f"({elapsed:.1f}s elapsed; {attempts} attempts).",
                flush=True,
            )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    summary_rows = []
    for metric in metrics:
        for analysis in (43, 44, "mean_across_seeds"):
            values = np.asarray(boot[metric][analysis], dtype=float)
            if analysis == "mean_across_seeds":
                point = (
                    point_deltas[43][metric] + point_deltas[44][metric]
                ) / 2.0
                standard_point = (
                    seed_points[43]["standard"][metric]
                    + seed_points[44]["standard"][metric]
                ) / 2.0
                analysis_name = "mean_across_seeds"
            else:
                point = point_deltas[analysis][metric]
                standard_point = seed_points[analysis]["standard"][metric]
                analysis_name = f"seed_{analysis}"

            summary_rows.append({
                "analysis": analysis_name,
                "metric": metric,
                "n_recording_clusters": n_recordings,
                "n_seeds_averaged": 2 if analysis == "mean_across_seeds" else 1,
                "habitat_minus_standard_point_delta": point,
                "relative_improvement_percent_vs_standard": (
                    100.0 * point / standard_point
                    if standard_point != 0 else None
                ),
                "paired_bootstrap_ci95_low": float(np.percentile(values, 2.5)),
                "paired_bootstrap_ci95_high": float(np.percentile(values, 97.5)),
                "bootstrap_replicates_requested": args.replicates,
                "bootstrap_replicates_valid": len(values),
                "random_seed": args.random_seed,
                "uncertainty_scope": (
                    "Recording-cluster sampling; does not estimate "
                    "between-training-seed variability."
                ),
            })

    summary_csv = output_dir / "paired_recording_cluster_bootstrap.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    replicate_csv = output_dir / "paired_recording_cluster_bootstrap_replicates.csv"
    with replicate_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["replicate"]
        for metric in metrics:
            short = "macro" if metric.startswith("macro") else "micro"
            fields += [
                f"seed43_{short}_delta",
                f"seed44_{short}_delta",
                f"mean_{short}_delta_across_seeds",
            ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(args.replicates):
            row = {"replicate": i + 1}
            for metric in metrics:
                short = "macro" if metric.startswith("macro") else "micro"
                row[f"seed43_{short}_delta"] = boot[metric][43][i]
                row[f"seed44_{short}_delta"] = boot[metric][44][i]
                row[f"mean_{short}_delta_across_seeds"] = (
                    boot[metric]["mean_across_seeds"][i]
                )
            writer.writerow(row)

    report = {
        "method": (
            "Paired percentile bootstrap: sample recording clusters with "
            "replacement; use the same recording draw for both models and both "
            "seeds; calculate habitat-minus-standard AP differences."
        ),
        "n_recording_clusters": n_recordings,
        "n_segments": int(len(targets)),
        "n_classes": int(targets.shape[1]),
        "bootstrap_replicates_requested": args.replicates,
        "bootstrap_replicates_valid": accepted,
        "attempts": attempts,
        "random_seed": args.random_seed,
        "point_metrics_by_seed": seed_points,
        "point_deltas_habitat_minus_standard_by_seed": point_deltas,
        "bootstrap_summary_rows": summary_rows,
        "uncertainty_caveat": (
            "Intervals describe recording-cluster sampling uncertainty "
            "conditional on these two trained seeds; they do not quantify "
            "between-training-seed variability."
        ),
        "source_archives": {str(k): str(v) for k, v in ARCHIVES.items()},
    }
    json_path = output_dir / "paired_recording_cluster_bootstrap.json"
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nPAIRED BOOTSTRAP RESULTS", flush=True)
    for row in summary_rows:
        print(
            f"{row['analysis']} | {row['metric']} | "
            f"delta={row['habitat_minus_standard_point_delta']:+.8f} | "
            f"95% CI [{row['paired_bootstrap_ci95_low']:+.8f}, "
            f"{row['paired_bootstrap_ci95_high']:+.8f}] | "
            f"relative improvement="
            f"{row['relative_improvement_percent_vs_standard']:.2f}%",
            flush=True,
        )

    print("\nSaved outputs:", flush=True)
    print(summary_csv, flush=True)
    print(replicate_csv, flush=True)
    print(json_path, flush=True)


if __name__ == "__main__":
    main()
