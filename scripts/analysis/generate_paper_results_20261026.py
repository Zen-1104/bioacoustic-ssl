"""Build publication figures and tables from the verified 2026-10-26 artifacts.

Run from the repository root. Outputs are written to a new directory and the
script refuses to overwrite it. The real-soundscape bootstrap fixes AP class
eligibility from the complete target matrix and resamples whole recordings.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/publication_results_2026-10-26_release"
STRESS = {
    43: ROOT / "results/unseen_habitat_stress_pair_s43_20261023_084036.json",
    44: ROOT / "results/unseen_habitat_pair_stress_s44_bn-independent_key_views-concatenated.json",
}
SOUND = {
    43: ROOT / "results/real_soundscape_pair_s43_20261023_094200.npz",
    44: ROOT / "results/real_soundscape_pair_s44_20261025.npz",
}
AGG = ROOT / "results/real_soundscape_aggregate_s43_s44_fixnames"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding = "utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline = "", encoding = "utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = fields or list(rows[0])
    with path.open("w", newline = "", encoding = "utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames = fields)
        writer.writeheader()
        writer.writerows(rows)


def load_archives() -> dict[int, dict[str, np.ndarray]]:
    result = {}

    required = {"targets", "file_names", "segment_start_seconds", "segment_types", "class_names", "scores_habitat", "scores_standard"}
    for seed, path in SOUND.items():
        with np.load(path, allow_pickle = False) as archive:
            missing = required.difference(archive.files)

            if missing:
                raise ValueError(f"{path} lacks {sorted(missing)}")
            
            result[seed] = {key: archive[key].copy() for key in required}

    ref = result[43]
    for seed in (43, 44):
        cur = result[seed]

        for key in ("targets", "file_names", "segment_start_seconds", "segment_types", "class_names"):
            if not np.array_equal(ref[key], cur[key]):
                raise ValueError(f"Seed archives do not share {key}; paired comparison is invalid")
            
        for model in ("habitat", "standard"):
            score = cur[f"scores_{model}"]
            if score.shape != cur["targets"].shape or not np.isfinite(score).all():
                raise ValueError(f"Invalid score matrix for seed {seed}, {model}")
            
    return result


def make_recording_groups(file_names: np.ndarray) -> list[np.ndarray]:
    _, inverse = np.unique(file_names, return_inverse = True)

    order = np.argsort(inverse, kind = "stable")
    counts = np.bincount(inverse)
    offsets = np.r_[0, np.cumsum(counts)]

    return [order[offsets[i]:offsets[i + 1]] for i in range(len(counts))]


def fixed_class_bootstrap(data: dict[int, dict[str, np.ndarray]], replicates: int = 1000, seed: int = 20261026, support_threshold: int = 20) -> list[dict]:
    """Paired recording-cluster intervals for a fixed, support qualified AP set."""
    y = data[43]["targets"]
    groups = make_recording_groups(data[43]["file_names"])
    group_positive = np.zeros((len(groups), y.shape[1]), dtype = bool)

    for i, indices in enumerate(groups):
        group_positive[i] = np.any(y[indices] > 0, axis = 0)

    positive_support = group_positive.sum(axis = 0)

    eligible = np.flatnonzero((positive_support >= support_threshold) & np.any(y > 0, axis = 0) & np.any(y == 0, axis = 0))
    if len(eligible) == 0:
        raise ValueError("No classes have both positive and negative labels in the full data")
    
    points = {}
    for run_seed in (43, 44):
        points[run_seed] = {}

        for model in ("habitat", "standard"):
            scores = data[run_seed][f"scores_{model}"]
            points[run_seed][model] = {
                "macro": float(average_precision_score(y[:, eligible], scores[:, eligible], average="macro")),
                "micro": float(average_precision_score(y[:, eligible].ravel(), scores[:, eligible].ravel())),
            }

    rng = np.random.default_rng(seed)
    deltas = {m: {s: [] for s in (43, 44)} for m in ("macro", "micro")}

    accepted = attempts = 0
    while accepted < replicates:
        attempts += 1

        if attempts > replicates * 100:
            raise RuntimeError(f"Only {accepted} fixed-class bootstrap draws valid after {attempts} attempts")
        
        selected = rng.integers(0, len(groups), size = len(groups))
        indices = np.concatenate([groups[i] for i in selected])
        yb = y[indices][:, eligible]

        # Every replicate uses the exact full-data class set; reject undefined AP draws.
        if not (np.all(np.any(yb > 0, axis = 0)) and np.all(np.any(yb == 0, axis = 0))):
            continue

        for run_seed in (43, 44):
            h = data[run_seed]["scores_habitat"][indices][:, eligible]
            s = data[run_seed]["scores_standard"][indices][:, eligible]

            deltas["macro"][run_seed].append(float(
                average_precision_score(yb, h, average = "macro")
                - average_precision_score(yb, s, average = "macro")
            ))

            deltas["micro"][run_seed].append(float(
                average_precision_score(yb.ravel(), h.ravel())
                - average_precision_score(yb.ravel(), s.ravel())
            ))
        accepted += 1

    rows = []
    for metric in ("macro", "micro"):
        for run_seed in (43, 44, "mean_across_seeds"):
            values = np.asarray(deltas[metric][run_seed] if run_seed != "mean_across_seeds" else (np.asarray(deltas[metric][43]) + np.asarray(deltas[metric][44])) / 2)
            point = points[run_seed]["habitat"][metric] - points[run_seed]["standard"][metric] if run_seed != "mean_across_seeds" else np.mean([points[s]["habitat"][metric] - points[s]["standard"][metric] for s in (43, 44)])

            rows.append({"analysis": f"seed_{run_seed}" if run_seed in (43, 44) else run_seed, 
                         "metric": f"{metric}_average_precision", 
                         "minimum_positive_recording_support": support_threshold, 
                         "fixed_eligible_classes": len(eligible), 
                         "recording_clusters": len(groups), 
                         "valid_replicates": replicates, 
                         "attempts": attempts, 
                         "point_delta_habitat_minus_standard": point, 
                         "ci95_low": float(np.percentile(values, 2.5)), 
                         "ci95_high": float(np.percentile(values, 97.5)), 
                         "bootstrap_seed": seed, 
                         "uncertainty_scope": "Paired recording-cluster sampling conditional on these trained seeds; fixed support qualified class set; not training seed uncertainty."}
                         )
    return rows


def generate() -> Path:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    
    data = load_archives()
    OUT.mkdir(parents = True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "figure.dpi": 150, "savefig.dpi": 300})

    # Clean linear probe table and plot.
    probe_rows = []
    for model in ("habitat", "standard"):
        for seed in (43, 44):
            d = read_json(ROOT / f"results/probe_{model}_s{seed}_frac1.0.json")

            probe_rows.append({"model": model, 
                               "seed": seed, 
                               "accuracy": d["test_accuracy"], 
                               "correct_of_test": round(d["test_accuracy"] * d["n_test"]), 
                               "n_test": d["n_test"], 
                               "classes_in_probe": d["num_classes"], 
                               "split": d["split_method"], 
                               "source": f"results/probe_{model}_s{seed}_frac1.0.json"}
                               )

    write_csv(OUT / "clean_linear_probe_data.csv", probe_rows)
    fig, ax = plt.subplots(figsize=(6.2, 4.1))

    for x, model in enumerate(("standard", "habitat")):
        vals = [next(r["accuracy"] for r in probe_rows if r["model"] == model and r["seed"] == s) for s in (43, 44)]
        ax.scatter([x - .08, x + .08], vals, s = 55, zorder = 3, label = None)

        for dx, val, seed in zip((-.08, .08), vals, (43, 44)):
            ax.annotate(f"s{seed}", (x + dx, val), xytext = (3, 4), textcoords = "offset points", fontsize = 8)

        ax.plot([x - .08, x + .08], vals, color = "#777777", alpha = .65, linewidth = 1)

    ax.set_xticks([0, 1], ["Standard pretraining", "Habitat pretraining"])
    ax.set_ylabel("Clean linear-probe top-1 accuracy")
    ax.set_ylim(0, .55); ax.grid(axis = "y", alpha = .25)
    ax.set_title("Clean BirdCLEF probe (two matched training seeds)")
    fig.tight_layout(); fig.savefig(OUT / "clean_linear_probe.png"); plt.close(fig)

    # Synthetic-noise robustness, point estimates from per-recording predictions.
    noise_rows = []
    labels = None
    for seed, path in STRESS.items():
        d = read_json(path); labels = d["metadata"]["snr_labels"]

        for pool in ("seen_habitat", "unseen_habitat"):
            for key, model in (("moco_standard", "standard"), ("moco_habitat", "habitat")):
                for snr in labels:
                    v = d[pool][key][snr]

                    noise_rows.append({"seed": seed, 
                                       "noise_pool": pool, 
                                       "model": model, 
                                       "snr": snr, 
                                       "accuracy": v["accuracy"], "n": v["n"], 
                                       "n_silent_mix_skipped": v["n_silent_mix_skipped"], 
                                       "source": str(path.relative_to(ROOT))}
                                       )

    write_csv(OUT / "synthetic_noise_accuracy_data.csv", noise_rows)
    snr_order = ["Clean", "20dB", "15dB", "10dB", "5dB", "0dB", "-5dB"]
    xvals = np.array([25, 20, 15, 10, 5, 0, -5])

    fig, axes = plt.subplots(1, 2, figsize = (10, 4.2), sharey = True)
    for ax, pool in zip(axes, ("seen_habitat", "unseen_habitat")):
        for model, color in (("standard", "#4C78A8"), ("habitat", "#E45756")):
            for seed, marker in ((43, "o"), (44, "s")):
                ys = [next(r["accuracy"] for r in noise_rows if r["seed"] == seed and r["noise_pool"] == pool and r["model"] == model and r["snr"] == snr) for snr in snr_order]
                ax.plot(xvals, ys, marker = marker, color = color, linewidth = 1.6, markersize = 4, label = f"{model}, seed {seed}")

        ax.set_title("Seen habitat noise" if pool == "seen_habitat" else "Held-out habitat noise")
        ax.set_xlabel("SNR (dB; Clean shown at +25 dB position)")
        ax.set_xticks(xvals, ["Clean", "20", "15", "10", "5", "0", "−5"])
        ax.grid(alpha = .25)

    axes[0].set_ylabel("Top-1 accuracy (n = 4,847 per condition)")
    axes[1].legend(frameon = False, fontsize = 8, loc = "lower right")

    fig.suptitle("Synthetic-noise robustness; points show each training seed")
    fig.tight_layout(); fig.savefig(OUT / "synthetic_noise_robustness.png"); plt.close(fig)

    # Soundscape point estimates with a corrected fixed eligibility recording bootstrap.
    y = data[43]["targets"]
    eligible = np.flatnonzero(np.any(y > 0, axis=0) & np.any(y == 0, axis = 0))
    ap_rows = []
    for seed in (43, 44):
        for model in ("standard", "habitat"):
            score = data[seed][f"scores_{model}"]

            ap_rows.append({"seed": seed, 
                            "model": model, 
                            "metric": "macro_average_precision", 
                            "value": float(average_precision_score(y[:, eligible], score[:, eligible], average = "macro")), 
                            "classes_in_metric": len(eligible), 
                            "n_segments": len(y), 
                            "n_recordings": len(np.unique(data[seed]["file_names"])), 
                            "source": str(SOUND[seed].relative_to(ROOT))}
                            )
            
            ap_rows.append({"seed": seed, 
                            "model": model, 
                            "metric": "micro_average_precision", 
                            "value": float(average_precision_score(y[:, eligible].ravel(), score[:, eligible].ravel())), 
                            "classes_in_metric": len(eligible), 
                            "n_segments": len(y), "n_recordings": len(np.unique(data[seed]["file_names"])), 
                            "source": str(SOUND[seed].relative_to(ROOT))}
                            )

    write_csv(OUT / "real_soundscape_ap_data.csv", ap_rows)
    support_path = AGG / "class_support_sensitivity_20261025/class_support_sensitivity_summary.csv"
    support_rows = read_csv(support_path)
    boot_rows = []

    for row in support_rows:
        if row["minimum_positive_recording_support"] != "20":
            continue

        boot_rows.append({
            "analysis": row["analysis"],
            "metric": row["metric"],
            "minimum_positive_recording_support": int(row["minimum_positive_recording_support"]),
            "fixed_eligible_classes": int(row["n_species"]),
            "recording_clusters": 1113,
            "valid_replicates": int(row["valid_replicates"]),
            "point_delta_habitat_minus_standard": float(row["habitat_minus_standard_delta"]),
            "ci95_low": float(row["paired_bootstrap_ci95_low"]),
            "ci95_high": float(row["paired_bootstrap_ci95_high"]),
            "bootstrap_seed": int(row["random_seed"]),
            "uncertainty_scope": row["uncertainty_scope"],
            "source": str(support_path.relative_to(ROOT)),
        })

    if len(boot_rows) != 6:
        raise ValueError(f"Expected six support-20 bootstrap rows, found {len(boot_rows)}")
    
    write_csv(OUT / "real_soundscape_fixed_eligibility_bootstrap.csv", boot_rows)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.2))

    for col, metric in enumerate(("macro_average_precision", "micro_average_precision")):
        boot_metric = "macro_average_precision" if metric.startswith("macro") else "micro_average_precision"

        for seed, marker, offset in ((43, "o", -.08), (44, "s", .08)):
            for model, x in (("standard", 0), ("habitat", 1)):
                row = next(r for r in ap_rows if r["seed"] == seed and r["model"] == model and r["metric"] == metric)
                axes[0, col].scatter(x + offset, row["value"], marker = marker, s = 48, label = f"seed {seed}" if model == "standard" else None, zorder = 3)

            standard = next(r["value"] for r in ap_rows if r["seed"] == seed and r["model"] == "standard" and r["metric"] == metric)
            habitat = next(r["value"] for r in ap_rows if r["seed"] == seed and r["model"] == "habitat" and r["metric"] == metric)
            axes[0, col].plot([offset, 1 + offset], [standard, habitat], color = "#777777", alpha = .5, linewidth = 1)

        for seed, offset in ((43, -.08), (44, .08)):
            row = next(r for r in boot_rows if r["analysis"] == f"seed_{seed}" and r["metric"] == boot_metric)
            delta = row["point_delta_habitat_minus_standard"]
            low = row["ci95_low"]; high = row["ci95_high"]

            axes[1, col].errorbar(offset, 
                                  delta, 
                                  yerr = [[delta-low], [high-delta]], 
                                  fmt = "o" if seed == 43 else "s", 
                                  color = "#4C78A8" if seed == 43 else "#F58518", 
                                  capsize = 3, 
                                  markersize = 5, 
                                  label = f"seed {seed}"
                                  )
            
        axes[0, col].set_xticks([0, 1], ["Standard", "Habitat"]); axes[0, col].set_title("Macro AP" if metric.startswith("macro") else "Micro AP"); axes[0, col].set_ylabel("Average precision"); axes[0, col].grid(axis = "y", alpha = .25)

        axes[1, col].axhline(0, color = "black", 
                             linewidth = .8); axes[1, col].set_xticks([-.08, .08], ["Seed 43", "Seed 44"]); axes[1, col].set_title("Habitat − standard, paired 95% CI"); axes[1, col].set_ylabel("AP difference"); axes[1, col].grid(axis = "y", alpha = .25
                                                                                                                                                                                                                                     )

    axes[0, 1].legend(frameon = False, fontsize = 8)
    fig.suptitle("Western Ghats soundscapes; full-scope per-seed AP and fixed support paired intervals")
    fig.tight_layout(); fig.savefig(OUT / "real_soundscape_ap.png"); plt.close(fig)

    # Existing fixed support exploratory results, preserving original draws and intervals.
    for row in support_rows:
        row["analysis_status"] = "Exploratory; recording-cluster interval conditional on trained seeds; not seed variability or confirmatory inference"

    write_csv(OUT / "fixed_class_support_sensitivity_data.csv", support_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex = True)
    for ax, metric in zip(axes, ("macro_average_precision", "micro_average_precision")):
        for analysis, marker, color in (("seed_43", "o", "#4C78A8"), ("seed_44", "s", "#F58518"), ("mean_across_seeds", "D", "#54A24B")):
            rows = [r for r in support_rows if r["metric"] == metric and r["analysis"] == analysis]
            rows.sort(key=lambda r: int(r["minimum_positive_recording_support"]))
            xx = [int(r["minimum_positive_recording_support"]) for r in rows]
            yy = [float(r["habitat_minus_standard_delta"]) for r in rows]
            lo = [v-float(r["paired_bootstrap_ci95_low"]) for v,r in zip(yy,rows)]
            hi = [float(r["paired_bootstrap_ci95_high"])-v for v,r in zip(yy,rows)]

            ax.errorbar(xx, yy, yerr = [lo,hi], marker = marker, color = color, capsize = 3, linewidth = 1.4, label = analysis.replace("_", " "))

            ax.axhline(0, color = "black", 
                       linewidth = .8); ax.set_xlim(15, 105); ax.set_xticks([20,50,100],["20","50","100"]); ax.set_xlabel("Minimum positive recording groups per class"); ax.set_title("Macro AP" if metric.startswith("macro") else "Micro AP"); ax.grid(alpha = .2)

    axes[0].set_ylabel("Habitat − standard AP difference (paired 95% interval)"); axes[1].legend(frameon = False, fontsize = 8)
    fig.suptitle("Exploratory fixed-class-support sensitivity; class counts: 27, 12, 3")
    fig.tight_layout(); fig.savefig(OUT / "fixed_class_support_sensitivity.png"); plt.close(fig)

    # Existing per-species paired recording bootstrap; individual species are exploratory.
    species = read_csv(AGG / "per_species_analysis_20261025/per_species_ap_comparison.csv")
    species.sort(key = lambda r: float(r["mean_delta_ap_across_seeds"]))
    write_csv(OUT / "per_species_ap_difference_data.csv", species)

    fig, ax = plt.subplots(figsize=(7.2, 8.0))
    yy = np.arange(len(species)); values = np.array([float(r["mean_delta_ap_across_seeds"]) for r in species]); low = np.array([float(r["mean_seed_ci95_low"]) for r in species]); high = np.array([float(r["mean_seed_ci95_high"]) for r in species])

    ax.errorbar(values, yy, xerr = [values-low, high-values], fmt = "o", color = "#4C78A8", ecolor = "#9BB4CC", capsize = 2, markersize = 3)
    ax.axvline(0, color = "black", linewidth = .8); ax.set_yticks(yy, [f"{r['species']} (n+={r['positive_recording_support']})" for r in species], fontsize = 7); ax.set_xlabel("Habitat − standard AP (mean across seeds; paired 95% recording bootstrap interval)"); ax.set_title("Per-species AP differences (support ≥20 recording groups)\nExploratory; no multiple comparison correction")
    ax.grid(axis = "x", alpha = .2); fig.tight_layout(); fig.savefig(OUT / "per_species_ap_difference.png"); plt.close(fig)

    (OUT / "CAPTIONS.txt").write_text(
        "Figure 1. Clean BirdCLEF linear-probe top-1 accuracy for standard and habitat pretraining, with each available training seed shown separately (4,847 test clips per seed; 179 probe classes).\n"
        "Figure 2. Top-1 accuracy under synthetic mixtures at the configured SNRs for seen and held-out habitat noise pools. Each line represents one model and seed; n=4,847 per point; zero skipped noise mixes. Clean is placed at +25 dB for ordered display.\n"
        "Figure 3. Top row: Western Ghats macro and micro AP by model and seed on the evaluator's 108 AP-eligible classes. Bottom row: paired habitat-minus-standard intervals on a fixed support ≥20 class set (27 species), from 1,000 recording-cluster bootstrap replicates. Full-scope intervals are omitted because many eligible species have very low recording support and would disappear from many resamples.\n"
        "Figure 4. Exploratory fixed-class-support sensitivity for 20, 50, and 100 positive recording groups per species. Points show habitat-minus-standard deltas; intervals are paired recording-cluster bootstrap intervals conditional on the trained seeds.\n"
        "Figure 5. Exploratory species AP differences for classes with at least 20 positive recording groups. Intervals quantify recording-cluster resampling conditional on both seeds; they are not multiplicity-adjusted confirmatory tests.\n",
        encoding = "utf-8")
    return OUT


if __name__ == "__main__":
    print(generate())
