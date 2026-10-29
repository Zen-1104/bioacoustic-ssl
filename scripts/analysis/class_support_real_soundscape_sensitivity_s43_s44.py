from __future__ import annotations

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
THRESHOLDS = (20, 50, 100)
MODELS = ("habitat", "standard")
METRICS = ("macro_average_precision", "micro_average_precision")


def load_archive(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing archive: {path}")

    with np.load(path, allow_pickle = False) as data:
        required = (
            "targets",
            "file_names",
            "segment_start_seconds",
            "segment_types",
            "class_names",
            "scores_habitat",
            "scores_standard",
        )

        missing = set(required).difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")

        return {key: data[key].copy() for key in required}


def make_groups(file_names: np.ndarray):
    unique_names, inverse = np.unique(file_names, return_inverse = True)

    order = np.argsort(inverse, kind = "stable")
    counts = np.bincount(inverse)
    offsets = np.concatenate(([0], np.cumsum(counts)))

    groups = [
        order[offsets[i]:offsets[i + 1]]
        for i in range(len(unique_names))
    ]

    return unique_names, groups


def ap_scores(y: np.ndarray, scores: np.ndarray, class_ids: np.ndarray):
    yt = y[:, class_ids]
    ys = scores[:, class_ids]

    macro = float(average_precision_score(yt, ys, average="macro"))
    micro = float(average_precision_score(yt.reshape(-1), ys.reshape(-1)))

    return {
        "macro_average_precision": macro,
        "micro_average_precision": micro,
    }


def main():
    out = Path(
        "results/real_soundscape_aggregate_s43_s44_fixnames/"
        "class_support_sensitivity_20261025"
    )
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")

    data = {seed: load_archive(path) for seed, path in ARCHIVES.items()}
    ref = data[43]

    for seed in (43, 44):
        item = data[seed]
        for key in (
            "targets",
            "file_names",
            "segment_start_seconds",
            "segment_types",
            "class_names",
        ):
            if not np.array_equal(ref[key], item[key]):
                raise ValueError(
                    f"Seeds 43 and 44 differ in {key}; paired comparison unsafe."
                )

        for model in MODELS:
            scores = item[f"scores_{model}"]
            if scores.shape != item["targets"].shape:
                raise ValueError(f"Seed {seed}, {model}: score shape mismatch.")
            if not np.isfinite(scores).all():
                raise ValueError(f"Seed {seed}, {model}: non-finite scores.")

    y = ref["targets"]
    class_names = ref["class_names"]
    unique_names, groups = make_groups(ref["file_names"])
    n_recordings = len(groups)
    n_segments, n_classes = y.shape

    # Support = number of distinct recordings with >=1 positive segment
    # for the given species. This avoids counting repeated segments as
    # independent species observations.
    group_positive = np.zeros((n_recordings, n_classes), dtype = bool)
    for i, indices in enumerate(groups):
        group_positive[i] = np.any(y[indices] > 0, axis = 0)

    positive_recording_support = group_positive.sum(axis = 0)

    has_positive = np.any(y > 0, axis = 0)
    has_negative = np.any(y == 0, axis = 0)

    class_ids_by_threshold = {}
    for threshold in THRESHOLDS:
        ids = np.flatnonzero(
            (positive_recording_support >= threshold)
            & has_positive
            & has_negative
        )
        class_ids_by_threshold[threshold] = ids
        print(
            f"Support >= {threshold} positive recordings: "
            f"{len(ids)} species",
            flush=True,
        )
        if len(ids) < 2:
            raise ValueError(
                f"Only {len(ids)} eligible species at threshold {threshold}; "
                "not enough for a useful macro-AP sensitivity analysis."
            )

    # Each threshold uses a fixed class set selected from the full label
    # inventory, not a replicate dependent set of classes.
    base_ids = class_ids_by_threshold[THRESHOLDS[0]]
    base_position = {int(cid): i for i, cid in enumerate(base_ids)}

    positions_by_threshold = {
        t: np.array([base_position[int(cid)]
                     for cid in class_ids_by_threshold[t]], dtype = int)
        for t in THRESHOLDS
    }

    out.mkdir(parents=True, exist_ok=False)

    support_path = out / "class_support_inventory.csv"
    with support_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "class_index", "class_name", "positive_recording_support",
            "included_support_ge_20", "included_support_ge_50",
            "included_support_ge_100",
        ]

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, name in enumerate(class_names):
            writer.writerow({
                "class_index": i,
                "class_name": str(name),
                "positive_recording_support": int(positive_recording_support[i]),
                "included_support_ge_20": bool(
                    i in set(class_ids_by_threshold[20].tolist())
                ),
                "included_support_ge_50": bool(
                    i in set(class_ids_by_threshold[50].tolist())
                ),
                "included_support_ge_100": bool(
                    i in set(class_ids_by_threshold[100].tolist())
                ),
            })

    point = {t: {} for t in THRESHOLDS}
    point_delta = {t: {} for t in THRESHOLDS}

    print(f"Segments: {n_segments}; classes: {n_classes}", flush=True)
    print(f"Recording clusters: {n_recordings}", flush=True)
    print("Calculating fixed-class point estimates...", flush=True)

    for t in THRESHOLDS:
        ids = class_ids_by_threshold[t]

        for seed in (43, 44):
            point[t][seed] = {}

            for model in MODELS:
                point[t][seed][model] = ap_scores(
                    y, data[seed][f"scores_{model}"], ids
                )

            point_delta[t][seed] = {
                metric: (
                    point[t][seed]["habitat"][metric]
                    - point[t][seed]["standard"][metric]
                )

                for metric in METRICS
            }

            print(
                f"Threshold {t}, seed {seed}: "
                f"macro delta={point_delta[t][seed]['macro_average_precision']:+.8f}; "
                f"micro delta={point_delta[t][seed]['micro_average_precision']:+.8f}",
                flush = True,
            )

    rng = np.random.default_rng(20261025)
    bootstrap = {
        t: {
            metric: {43: [], 44: [], "mean_across_seeds": []}
            for metric in METRICS
        }
        for t in THRESHOLDS
    }

    accepted = 0
    attempts = 0
    requested = 1000
    max_attempts = requested * 5
    start = time.monotonic()

    print(
        "Starting paired recording-cluster bootstrap: 1,000 valid replicates.",
        flush=True,
    )

    while accepted < requested:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Only {accepted} valid replicates after {attempts} attempts."
            )

        # Identical recording draw for both models, both seeds and all support thresholds.
        selected_groups = rng.integers(0, n_recordings, size = n_recordings)
        sampled_indices = np.concatenate(
            [groups[i] for i in selected_groups]
        )
        yb = y[sampled_indices]

        # Enforce valid targets for every species in the broadest fixed class set.
        positive_counts = yb[:, base_ids].sum(axis = 0)
        if np.any(positive_counts == 0) or np.any(
            positive_counts == yb.shape[0]
        ):
            continue

        # AP for each species in the broadest class set is computed once
        # per seed/model/replicate; narrower macro sets are fixed subsets.
        replicate_delta = {t: {} for t in THRESHOLDS}

        for seed in (43, 44):
            seed_scores = {}

            for model in MODELS:
                scores_b = data[seed][f"scores_{model}"][sampled_indices]
                class_ap = average_precision_score(
                    yb[:, base_ids], scores_b[:, base_ids], average=None
                )

                seed_scores[model] = {}
                for t in THRESHOLDS:
                    positions = positions_by_threshold[t]
                    ids = class_ids_by_threshold[t]
                    macro = float(np.mean(class_ap[positions]))
                    micro = float(average_precision_score(
                        yb[:, ids].reshape(-1),
                        scores_b[:, ids].reshape(-1),
                    ))
                    seed_scores[model][t] = {
                        "macro_average_precision": macro,
                        "micro_average_precision": micro,
                    }

            for t in THRESHOLDS:
                replicate_delta[t][seed] = {
                    metric: (
                        seed_scores["habitat"][t][metric]
                        - seed_scores["standard"][t][metric]
                    )
                    for metric in METRICS
                }

        accepted += 1
        for t in THRESHOLDS:
            for metric in METRICS:
                d43 = replicate_delta[t][43][metric]
                d44 = replicate_delta[t][44][metric]
                bootstrap[t][metric][43].append(d43)
                bootstrap[t][metric][44].append(d44)
                bootstrap[t][metric]["mean_across_seeds"].append(
                    (d43 + d44) / 2.0
                )

        if accepted % 50 == 0 or accepted == requested:
            print(
                f"Completed {accepted}/{requested} valid replicates "
                f"({time.monotonic() - start:.1f}s elapsed; "
                f"{attempts} attempts).",
                flush=True,
            )

    summary_rows = []
    for t in THRESHOLDS:
        for metric in METRICS:
            for analysis in (43, 44, "mean_across_seeds"):
                values = np.asarray(
                    bootstrap[t][metric][analysis], dtype=float
                )
                if analysis == "mean_across_seeds":
                    delta = (
                        point_delta[t][43][metric]
                        + point_delta[t][44][metric]
                    ) / 2
                    standard_point = (
                        point[t][43]["standard"][metric]
                        + point[t][44]["standard"][metric]
                    ) / 2
                    analysis_name = "mean_across_seeds"
                else:
                    delta = point_delta[t][analysis][metric]
                    standard_point = point[t][analysis]["standard"][metric]
                    analysis_name = f"seed_{analysis}"

                summary_rows.append({
                    "minimum_positive_recording_support": t,
                    "n_species": len(class_ids_by_threshold[t]),
                    "analysis": analysis_name,
                    "metric": metric,
                    "habitat_minus_standard_delta": float(delta),
                    "relative_improvement_percent": float(
                        100 * delta / standard_point
                    ),
                    "paired_bootstrap_ci95_low": float(
                        np.percentile(values, 2.5)
                    ),
                    "paired_bootstrap_ci95_high": float(
                        np.percentile(values, 97.5)
                    ),
                    "valid_replicates": len(values),
                    "random_seed": 20261025,
                    "uncertainty_scope": (
                        "Recording-cluster sampling, conditional on the two "
                        "trained seeds; not training-seed uncertainty."
                    ),
                })

    summary_path = out / "class_support_sensitivity_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    replicate_path = out / "class_support_sensitivity_replicates.csv"
    with replicate_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["replicate"]
        for t in THRESHOLDS:
            for metric in METRICS:
                short = "macro" if metric.startswith("macro") else "micro"
                fields.extend([
                    f"support{t}_seed43_{short}_delta",
                    f"support{t}_seed44_{short}_delta",
                    f"support{t}_mean_seeds_{short}_delta",
                ])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(requested):
            row = {"replicate": i + 1}
            for t in THRESHOLDS:
                for metric in METRICS:
                    short = "macro" if metric.startswith("macro") else "micro"
                    row[f"support{t}_seed43_{short}_delta"] = (
                        bootstrap[t][metric][43][i]
                    )
                    row[f"support{t}_seed44_{short}_delta"] = (
                        bootstrap[t][metric][44][i]
                    )
                    row[f"support{t}_mean_seeds_{short}_delta"] = (
                        bootstrap[t][metric]["mean_across_seeds"][i]
                    )
            writer.writerow(row)

    report = {
        "method": (
            "Paired recording-cluster percentile bootstrap. Fixed nested "
            "species sets are selected by positive annotation in >=20, >=50, "
            "or >=100 distinct recordings. A shared recording sample is used "
            "for both models and both seeds."
        ),
        "n_recordings": n_recordings,
        "n_segments": int(n_segments),
        "total_classes": int(n_classes),
        "support_threshold_species_counts": {
            str(t): int(len(class_ids_by_threshold[t])) for t in THRESHOLDS
        },
        "bootstrap_replicates_requested": requested,
        "bootstrap_replicates_valid": accepted,
        "attempts": attempts,
        "random_seed": 20261025,
        "point_metrics_by_threshold_seed_model": {
            str(t): {
                str(seed): point[t][seed] for seed in (43, 44)
            }
            for t in THRESHOLDS
        },
        "point_deltas_by_threshold_seed": {
            str(t): {
                str(seed): point_delta[t][seed] for seed in (43, 44)
            }
            for t in THRESHOLDS
        },
        "summary": summary_rows,
        "interpretation_caveat": (
            "Support-filtered analyses are sensitivity analyses selected using "
            "annotation support in these soundscapes. Intervals describe "
            "recording-sampling uncertainty conditional on two trained seeds, "
            "not variation across training runs."
        ),
        "source_archives": {str(k): str(v) for k, v in ARCHIVES.items()},
    }
    json_path = out / "class_support_sensitivity_summary.json"
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nCLASS-SUPPORT SENSITIVITY RESULTS", flush=True)
    for row in summary_rows:
        print(
            f"support>={row['minimum_positive_recording_support']} "
            f"(species={row['n_species']}) | {row['analysis']} | "
            f"{row['metric']} | delta={row['habitat_minus_standard_delta']:+.8f} "
            f"({row['relative_improvement_percent']:+.2f}%) | "
            f"95% CI [{row['paired_bootstrap_ci95_low']:+.8f}, "
            f"{row['paired_bootstrap_ci95_high']:+.8f}]",
            flush=True,
        )

    print("\nSaved files:", flush=True)
    print(summary_path, flush=True)
    print(replicate_path, flush=True)
    print(support_path, flush=True)
    print(json_path, flush=True)


if __name__ == "__main__":
    main()
