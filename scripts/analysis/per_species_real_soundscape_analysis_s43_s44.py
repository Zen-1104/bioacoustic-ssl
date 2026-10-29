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
OUT = Path(
    "results/real_soundscape_aggregate_s43_s44_fixnames/"
    "per_species_analysis_20261025"
)
THRESHOLD = 20
N_BOOTSTRAP = 1000
RANDOM_SEED = 20261025
MODELS = ("habitat", "standard")


def load_archive(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction archive missing: {path}")

    with np.load(path, allow_pickle=False) as archive:
        required = (
            "targets", "file_names", "segment_start_seconds",
            "segment_types", "class_names",
            "scores_habitat", "scores_standard",
        )
        missing = set(required).difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        return {key: archive[key].copy() for key in required}


def make_groups(file_names: np.ndarray):
    unique_names, inverse = np.unique(file_names, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    groups = [
        order[offsets[i]:offsets[i + 1]]
        for i in range(len(unique_names))
    ]
    return unique_names, groups


def per_class_ap(
    targets: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
) -> np.ndarray:
    result = average_precision_score(
        targets[:, class_ids],
        scores[:, class_ids],
        average=None,
    )
    return np.asarray(result, dtype=float)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results: {OUT}"
        )

    data = {seed: load_archive(path) for seed, path in ARCHIVES.items()}
    reference = data[43]

    for seed in (43, 44):
        item = data[seed]

        for key in (
            "targets", "file_names", "segment_start_seconds",
            "segment_types", "class_names",
        ):
            if not np.array_equal(reference[key], item[key]):
                raise ValueError(
                    f"Seed archives differ in {key!r}; paired comparison "
                    "cannot safely continue."
                )

        for model in MODELS:
            scores = item[f"scores_{model}"]
            if scores.shape != item["targets"].shape:
                raise ValueError(
                    f"Seed {seed}, model {model}: score/target shape mismatch."
                )
            if not np.isfinite(scores).all():
                raise ValueError(
                    f"Seed {seed}, model {model}: non-finite predictions."
                )

    targets = reference["targets"]
    class_names = reference["class_names"]
    file_names = reference["file_names"]
    _, groups = make_groups(file_names)
    n_recordings = len(groups)

    recording_positive = np.zeros(
        (n_recordings, targets.shape[1]), dtype=bool
    )
    for i, indices in enumerate(groups):
        recording_positive[i] = np.any(targets[indices] > 0, axis=0)

    support = recording_positive.sum(axis=0)
    ids = np.flatnonzero(
        (support >= THRESHOLD)
        & np.any(targets > 0, axis=0)
        & np.any(targets == 0, axis=0)
    )

    if len(ids) != 27:
        print(
            f"NOTICE: Found {len(ids)} eligible species, not the previously "
            "observed 27. The analysis will use the verified current inventory.",
            flush=True,
        )

    if len(ids) < 2:
        raise ValueError("Too few eligible species for this analysis.")

    species_names = [str(class_names[i]) for i in ids]
    n_species = len(ids)

    print(f"Recordings: {n_recordings}", flush=True)
    print(f"Segments: {len(targets)}", flush=True)
    print(f"Classes included (support >= {THRESHOLD} recordings): {n_species}",
          flush=True)
    print("Calculating full-dataset per-species AP...", flush=True)

    point_ap: dict[int, dict[str, np.ndarray]] = {}
    point_delta: dict[int, np.ndarray] = {}

    for seed in (43, 44):
        point_ap[seed] = {}
        for model in MODELS:
            point_ap[seed][model] = per_class_ap(
                targets,
                data[seed][f"scores_{model}"],
                ids,
            )
        point_delta[seed] = (
            point_ap[seed]["habitat"] - point_ap[seed]["standard"]
        )

        print(
            f"Seed {seed}: improved species "
            f"{int(np.sum(point_delta[seed] > 0))}/{n_species}; "
            f"mean per-species delta={np.mean(point_delta[seed]):+.8f}",
            flush=True,
        )

    mean_habitat_ap = (
        point_ap[43]["habitat"] + point_ap[44]["habitat"]
    ) / 2.0
    mean_standard_ap = (
        point_ap[43]["standard"] + point_ap[44]["standard"]
    ) / 2.0
    mean_delta = (point_delta[43] + point_delta[44]) / 2.0

    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_delta = {
        43: np.empty((N_BOOTSTRAP, n_species), dtype=float),
        44: np.empty((N_BOOTSTRAP, n_species), dtype=float),
    }
    bootstrap_mean = np.empty((N_BOOTSTRAP, n_species), dtype=float)

    accepted = 0
    attempts = 0
    max_attempts = N_BOOTSTRAP * 5
    start_time = time.monotonic()

    print(
        "Starting paired recording bootstrap. Each draw is shared by both "
        "models and both seeds.",
        flush=True,
    )

    while accepted < N_BOOTSTRAP:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Only {accepted} valid draws after {attempts} attempts."
            )

        selected_groups = rng.integers(
            0, n_recordings, size=n_recordings
        )
        sampled_indices = np.concatenate(
            [groups[i] for i in selected_groups]
        )
        sampled_targets = targets[sampled_indices]
        selected_targets = sampled_targets[:, ids]

        # Keep the species set fixed in every bootstrap replicate.
        # Reject draws where any selected species has no positive or
        # no negative segment.
        if (
            not np.all(np.any(selected_targets > 0, axis=0))
            or not np.all(np.any(selected_targets == 0, axis=0))
        ):
            continue

        replicate_deltas = {}
        for seed in (43, 44):
            habitat = per_class_ap(
                sampled_targets,
                data[seed]["scores_habitat"][sampled_indices],
                ids,
            )
            standard = per_class_ap(
                sampled_targets,
                data[seed]["scores_standard"][sampled_indices],
                ids,
            )
            replicate_deltas[seed] = habitat - standard

        bootstrap_delta[43][accepted] = replicate_deltas[43]
        bootstrap_delta[44][accepted] = replicate_deltas[44]
        bootstrap_mean[accepted] = (
            replicate_deltas[43] + replicate_deltas[44]
        ) / 2.0

        accepted += 1
        if accepted % 50 == 0 or accepted == N_BOOTSTRAP:
            elapsed = time.monotonic() - start_time
            print(
                f"Completed {accepted}/{N_BOOTSTRAP} valid replicates "
                f"({elapsed:.1f}s elapsed; {attempts} attempts).",
                flush=True,
            )

    rows = []
    for j, class_id in enumerate(ids):
        delta43 = float(point_delta[43][j])
        delta44 = float(point_delta[44][j])
        delta_avg = float(mean_delta[j])

        b43 = bootstrap_delta[43][:, j]
        b44 = bootstrap_delta[44][:, j]
        bmean = bootstrap_mean[:, j]

        rows.append({
            "species": species_names[j],
            "class_index": int(class_id),
            "positive_recording_support": int(support[class_id]),
            "habitat_ap_seed43": float(point_ap[43]["habitat"][j]),
            "standard_ap_seed43": float(point_ap[43]["standard"][j]),
            "delta_ap_seed43": delta43,
            "seed43_ci95_low": float(np.percentile(b43, 2.5)),
            "seed43_ci95_high": float(np.percentile(b43, 97.5)),
            "habitat_ap_seed44": float(point_ap[44]["habitat"][j]),
            "standard_ap_seed44": float(point_ap[44]["standard"][j]),
            "delta_ap_seed44": delta44,
            "seed44_ci95_low": float(np.percentile(b44, 2.5)),
            "seed44_ci95_high": float(np.percentile(b44, 97.5)),
            "mean_habitat_ap_across_seeds": float(mean_habitat_ap[j]),
            "mean_standard_ap_across_seeds": float(mean_standard_ap[j]),
            "mean_delta_ap_across_seeds": delta_avg,
            "mean_seed_ci95_low": float(np.percentile(bmean, 2.5)),
            "mean_seed_ci95_high": float(np.percentile(bmean, 97.5)),
            "bootstrap_fraction_mean_delta_gt0": float(np.mean(bmean > 0)),
            "improved_seed43": bool(delta43 > 0),
            "improved_seed44": bool(delta44 > 0),
            "improved_both_seeds": bool(delta43 > 0 and delta44 > 0),
        })

    rows.sort(
        key=lambda row: row["mean_delta_ap_across_seeds"],
        reverse=True,
    )

    out = OUT
    out.mkdir(parents=True, exist_ok=False)

    result_path = out / "per_species_ap_comparison.csv"
    with result_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    replicate_path = out / "per_species_bootstrap_replicates.csv"
    with replicate_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "replicate", "species", "positive_recording_support",
            "delta_seed43", "delta_seed44", "mean_delta_across_seeds",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rep in range(N_BOOTSTRAP):
            for j, class_id in enumerate(ids):
                writer.writerow({
                    "replicate": rep + 1,
                    "species": species_names[j],
                    "positive_recording_support": int(support[class_id]),
                    "delta_seed43": float(bootstrap_delta[43][rep, j]),
                    "delta_seed44": float(bootstrap_delta[44][rep, j]),
                    "mean_delta_across_seeds": float(bootstrap_mean[rep, j]),
                })

    positive_deltas = np.maximum(mean_delta, 0.0)
    positive_sum = float(positive_deltas.sum())
    sorted_positive = np.sort(positive_deltas)[::-1]
    top3_share = (
        float(sorted_positive[:3].sum() / positive_sum)
        if positive_sum > 0 else None
    )
    top5_share = (
        float(sorted_positive[:5].sum() / positive_sum)
        if positive_sum > 0 else None
    )

    summary = {
        "method": (
            "Per-species AP differences for species with positive targets in "
            "at least 20 distinct recordings. Recording-cluster bootstrap "
            "resamples the same recording groups for both models and both "
            "seeds. Species set is fixed for every replicate."
        ),
        "n_recordings": n_recordings,
        "n_segments": int(len(targets)),
        "n_species": n_species,
        "bootstrap_replicates_requested": N_BOOTSTRAP,
        "bootstrap_replicates_valid": accepted,
        "bootstrap_attempts": attempts,
        "random_seed": RANDOM_SEED,
        "mean_species_delta": float(np.mean(mean_delta)),
        "median_species_delta": float(np.median(mean_delta)),
        "min_species_delta": float(np.min(mean_delta)),
        "max_species_delta": float(np.max(mean_delta)),
        "n_improved_mean_across_seeds": int(np.sum(mean_delta > 0)),
        "n_improved_seed43": int(np.sum(point_delta[43] > 0)),
        "n_improved_seed44": int(np.sum(point_delta[44] > 0)),
        "n_improved_in_both_seeds": int(np.sum(
            (point_delta[43] > 0) & (point_delta[44] > 0)
        )),
        "n_worse_in_both_seeds": int(np.sum(
            (point_delta[43] < 0) & (point_delta[44] < 0)
        )),
        "top3_share_of_sum_of_positive_species_deltas": top3_share,
        "top5_share_of_sum_of_positive_species_deltas": top5_share,
        "top_species_by_mean_delta": rows[:5],
        "bottom_species_by_mean_delta": rows[-5:],
        "per_species_rows": rows,
        "uncertainty_caveat": (
            "Per-species intervals describe recording-cluster resampling "
            "uncertainty conditional on these two trained seeds. They do not "
            "quantify variation across independently trained models. Species "
            "were selected using annotation support in the same evaluation "
            "dataset, so this is a sensitivity analysis rather than an "
            "independent confirmation."
        ),
        "source_archives": {str(k): str(v) for k, v in ARCHIVES.items()},
    }

    summary_path = out / "per_species_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nPER-SPECIES RESULTS SUMMARY", flush=True)
    print(
        f"Positive species deltas: {summary['n_improved_mean_across_seeds']}/"
        f"{n_species} on seed-averaged estimates",
        flush=True,
    )
    print(
        f"Improved in both seeds: {summary['n_improved_in_both_seeds']}/"
        f"{n_species}",
        flush=True,
    )
    print(
        f"Mean delta: {summary['mean_species_delta']:+.8f}; "
        f"median delta: {summary['median_species_delta']:+.8f}",
        flush=True,
    )
    print(
        f"Top-3 share of positive species deltas: "
        f"{100 * top3_share:.1f}%" if top3_share is not None
        else "Top-3 share: undefined",
        flush=True,
    )
    print(
        f"Top-5 share of positive species deltas: "
        f"{100 * top5_share:.1f}%" if top5_share is not None
        else "Top-5 share: undefined",
        flush=True,
    )

    print(
        "\nRank | Species | Support | Habitat AP | Standard AP | "
        "Delta | Paired 95% CI | Both seeds",
        flush=True,
    )
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:>2} | {row['species']:<8} | "
            f"{row['positive_recording_support']:>3} | "
            f"{row['mean_habitat_ap_across_seeds']:.5f} | "
            f"{row['mean_standard_ap_across_seeds']:.5f} | "
            f"{row['mean_delta_ap_across_seeds']:+.5f} | "
            f"[{row['mean_seed_ci95_low']:+.5f}, "
            f"{row['mean_seed_ci95_high']:+.5f}] | "
            f"{'yes' if row['improved_both_seeds'] else 'no'}",
            flush=True,
        )

    print("\nSaved files:", flush=True)
    print(result_path, flush=True)
    print(replicate_path, flush=True)
    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
