from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ARCHIVES = {
    43: Path("results/real_soundscape_pair_s43_20261023_094200.npz"),
    44: Path("results/real_soundscape_pair_s44_20261025.npz"),
}
SPECIES_TO_AUDIT = [
    "gyhcaf1", "rewbul", "barfly1",
    "kerlau2", "crbsun2", "yebbul3",
]
OUT = Path(
    "results/real_soundscape_aggregate_s43_s44_fixnames/"
    "species_context_audit_20261025"
)
AUDIO_ROOT = Path("data/raw/western_ghats_soundscapes")
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
TOP_N = 5


def load_archive(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction archive: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = [
            "targets", "file_names", "segment_start_seconds",
            "segment_types", "class_names",
            "scores_habitat", "scores_standard",
        ]
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        return {key: archive[key].copy() for key in required}


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_audio_index():
    index: dict[str, list[Path]] = {}
    if not AUDIO_ROOT.is_dir():
        print(
            f"NOTICE: Audio root not found: {AUDIO_ROOT}. "
            "The report will still include recording names and timestamps.",
            flush=True,
        )
        return index

    for path in AUDIO_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            index.setdefault(path.name.lower(), []).append(path)
            index.setdefault(path.stem.lower(), []).append(path)

    return index


def find_audio(name: str, audio_index: dict[str, list[Path]]) -> str:
    p = Path(name)
    candidates = [p.name.lower(), p.stem.lower()]
    matches = []
    for key in candidates:
        matches.extend(audio_index.get(key, []))
    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return str(unique[0].resolve())
    return ""


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")

    data = {seed: load_archive(path) for seed, path in ARCHIVES.items()}
    reference = data[43]

    # Verify row alignment before comparing models or seeds.
    shared_fields = [
        "targets", "file_names", "segment_start_seconds",
        "segment_types", "class_names",
    ]
    for seed in (43, 44):
        item = data[seed]
        for field in shared_fields:
            if not np.array_equal(reference[field], item[field]):
                raise ValueError(
                    f"Archives differ in {field!r}; comparison is unsafe."
                )
        for model in ("habitat", "standard"):
            scores = item[f"scores_{model}"]
            if scores.shape != item["targets"].shape:
                raise ValueError(f"Score shape mismatch: seed {seed}, {model}")
            if not np.isfinite(scores).all():
                raise ValueError(f"Non-finite score: seed {seed}, {model}")

    targets = reference["targets"]
    class_names = [str(x) for x in reference["class_names"]]
    file_names = reference["file_names"].astype(str)
    starts = reference["segment_start_seconds"]
    segment_types = reference["segment_types"].astype(str)

    name_to_id = {name: i for i, name in enumerate(class_names)}
    missing_species = [s for s in SPECIES_TO_AUDIT if s not in name_to_id]
    if missing_species:
        raise ValueError(
            f"Species absent from class_names: {missing_species}. "
            f"Available classes: {class_names}"
        )

    ids = {species: name_to_id[species] for species in SPECIES_TO_AUDIT}

    # Unique recording support is more meaningful than segment count alone.
    unique_files = np.unique(file_names)
    recording_positive = {
        species: sum(
            np.any(targets[file_names == filename, cid] > 0)
            for filename in unique_files
        )
        for species, cid in ids.items()
    }

    unique_types, type_counts = np.unique(segment_types, return_counts=True)
    type_inventory = [
        {"segment_type": str(t), "segment_count": int(n)}
        for t, n in zip(unique_types, type_counts)
    ]

    bg_mask = np.array([
        ("background" in t.lower()) or ("unannotated" in t.lower())
        for t in segment_types
    ])

    print(f"Segments: {len(targets)}; recordings: {len(unique_files)}", flush=True)
    print("Segment types:", type_inventory, flush=True)

    summary_rows = []
    distribution_rows = []
    candidate_rows = []
    audio_index = build_audio_index()

    for species, cid in ids.items():
        y = targets[:, cid].astype(np.uint8)
        positive = y == 1
        negative = y == 0

        print(f"\n===== {species} =====", flush=True)
        print(
            f"Positive segments: {int(positive.sum())}; "
            f"positive recordings: {int(recording_positive[species])}",
            flush=True,
        )

        for seed in (43, 44):
            habitat = data[seed]["scores_habitat"][:, cid]
            standard = data[seed]["scores_standard"][:, cid]
            delta = habitat - standard

            ap_habitat = float(average_precision_score(y, habitat))
            ap_standard = float(average_precision_score(y, standard))
            ap_delta = ap_habitat - ap_standard

            summary_rows.append({
                "species": species,
                "positive_recording_support": int(recording_positive[species]),
                "positive_segment_count": int(positive.sum()),
                "negative_segment_count": int(negative.sum()),
                "seed": seed,
                "habitat_ap": ap_habitat,
                "standard_ap": ap_standard,
                "ap_delta_habitat_minus_standard": ap_delta,
                "mean_habitat_score_positive_segments": float(habitat[positive].mean()),
                "mean_standard_score_positive_segments": float(standard[positive].mean()),
                "mean_score_delta_positive_segments": float(delta[positive].mean()),
                "mean_habitat_score_negative_segments": float(habitat[negative].mean()),
                "mean_standard_score_negative_segments": float(standard[negative].mean()),
                "mean_score_delta_negative_segments": float(delta[negative].mean()),
                "background_segment_count": int(bg_mask.sum()),
                "positive_background_segment_count": int((positive & bg_mask).sum()),
                "negative_background_segment_count": int((negative & bg_mask).sum()),
            })

            print(
                f"Seed {seed}: AP habitat={ap_habitat:.5f}, "
                f"standard={ap_standard:.5f}, delta={ap_delta:+.5f}; "
                f"mean score delta on positive segments={delta[positive].mean():+.5f}; "
                f"on target-negative segments={delta[negative].mean():+.5f}",
                flush=True,
            )

            # Score distributions split by label and segment type.
            for seg_type in unique_types:
                type_mask = segment_types == seg_type
                for label_value, label_mask in (
                    ("positive", positive),
                    ("target_negative", negative),
                ):
                    mask = type_mask & label_mask
                    if not np.any(mask):
                        continue
                    h = habitat[mask]
                    s = standard[mask]
                    d = delta[mask]
                    distribution_rows.append({
                        "species": species,
                        "seed": seed,
                        "segment_type": str(seg_type),
                        "target_status": label_value,
                        "n_segments": int(mask.sum()),
                        "habitat_score_mean": float(h.mean()),
                        "habitat_score_median": float(np.median(h)),
                        "habitat_score_p90": float(np.percentile(h, 90)),
                        "habitat_score_p95": float(np.percentile(h, 95)),
                        "standard_score_mean": float(s.mean()),
                        "standard_score_median": float(np.median(s)),
                        "standard_score_p90": float(np.percentile(s, 90)),
                        "standard_score_p95": float(np.percentile(s, 95)),
                        "mean_habitat_minus_standard_score": float(d.mean()),
                    })

            # Candidate segments for manual listening. Score deltas are
            # diagnostic priorities, not proof that an event was detected.
            positive_idx = np.flatnonzero(positive)
            if len(positive_idx):
                gain_order = positive_idx[np.argsort(delta[positive_idx])[::-1]]
                loss_order = positive_idx[np.argsort(delta[positive_idx])]
                for kind, chosen in (
                    ("positive_target_habitat_score_gain", gain_order[:TOP_N]),
                    ("positive_target_standard_score_gain", loss_order[:TOP_N]),
                ):
                    for row_idx in chosen:
                        candidate_rows.append(make_candidate(
                            species, seed, kind, int(row_idx), 1,
                            habitat, standard, delta, file_names, starts,
                            segment_types, audio_index,
                        ))

            # Target-negative background segments are weak negatives:
            # unannotated audio can still contain real bird activity.
            bg_neg_idx = np.flatnonzero(negative & bg_mask)
            if len(bg_neg_idx) == 0:
                bg_neg_idx = np.flatnonzero(negative)
                fallback_kind = "target_negative_top_habitat_score_no_named_background_type"
            else:
                fallback_kind = "background_target_negative_top_habitat_score"

            for row_idx in bg_neg_idx[
                np.argsort(habitat[bg_neg_idx])[::-1][:TOP_N]
            ]:
                candidate_rows.append(make_candidate(
                    species, seed, fallback_kind, int(row_idx), 0,
                    habitat, standard, delta, file_names, starts,
                    segment_types, audio_index,
                ))

            # Also capture the largest habitat-vs-standard increases in the
            # background negatives, even if their absolute scores are modest.
            for row_idx in bg_neg_idx[
                np.argsort(delta[bg_neg_idx])[::-1][:TOP_N]
            ]:
                candidate_rows.append(make_candidate(
                    species, seed, "background_target_negative_largest_score_delta",
                    int(row_idx), 0, habitat, standard, delta, file_names,
                    starts, segment_types, audio_index,
                ))

    OUT.mkdir(parents=True, exist_ok=False)
    write_csv(OUT / "species_context_summary.csv", summary_rows)
    write_csv(OUT / "score_distributions_by_species_and_segment_type.csv",
              distribution_rows)
    write_csv(OUT / "candidate_segments_for_listening.csv", candidate_rows)
    write_csv(OUT / "segment_type_inventory.csv", type_inventory)

    report = {
        "species_audited": SPECIES_TO_AUDIT,
        "n_recordings": int(len(unique_files)),
        "n_segments": int(len(targets)),
        "segment_type_inventory": type_inventory,
        "candidate_clip_duration_seconds": 10,
        "candidate_count": len(candidate_rows),
        "audio_root": str(AUDIO_ROOT),
        "interpretation": (
            "This is a targeted diagnostic, not a causal test of noise "
            "robustness. Segment types and prediction scores are available in "
            "the archives, but they do not encode semantic noise/habitat labels. "
            "Target-negative background segments are weak negatives because "
            "unannotated clips may contain real bird activity. Listen to the "
            "listed audio timestamps before attributing score differences to "
            "noise or habitat."
        ),
        "output_files": [
            "species_context_summary.csv",
            "score_distributions_by_species_and_segment_type.csv",
            "candidate_segments_for_listening.csv",
            "segment_type_inventory.csv",
        ],
    }
    (OUT / "species_context_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print("\nAUDIT COMPLETE", flush=True)
    print("\nSummary by species and seed:", flush=True)
    for row in summary_rows:
        print(
            f"{row['species']} | seed {row['seed']} | "
            f"AP delta={row['ap_delta_habitat_minus_standard']:+.5f} | "
            f"positive score delta={row['mean_score_delta_positive_segments']:+.5f} | "
            f"target-negative score delta={row['mean_score_delta_negative_segments']:+.5f}",
            flush=True,
        )

    print("\nCandidate segment examples (first 30 rows):", flush=True)
    for row in candidate_rows[:30]:
        print(
            f"{row['species']} | seed {row['seed']} | {row['candidate_type']} | "
            f"{row['file_name']} @ {row['segment_start_seconds']:.1f}s | "
            f"type={row['segment_type']} | "
            f"habitat={row['habitat_score']:.4f} | "
            f"standard={row['standard_score']:.4f} | "
            f"delta={row['score_delta']:+.4f} | "
            f"audio={row['audio_path'] or 'not resolved'}",
            flush=True,
        )

    print("\nSaved under:", OUT, flush=True)
    for name in report["output_files"] + ["species_context_audit.json"]:
        print(OUT / name, flush=True)


def make_candidate(
    species, seed, kind, idx, target, habitat, standard, delta,
    file_names, starts, segment_types, audio_index,
):
    name = str(file_names[idx])
    return {
        "species": species,
        "seed": seed,
        "candidate_type": kind,
        "target_for_species": target,
        "file_name": name,
        "segment_start_seconds": float(starts[idx]),
        "segment_duration_seconds": 10,
        "segment_type": str(segment_types[idx]),
        "habitat_score": float(habitat[idx]),
        "standard_score": float(standard[idx]),
        "score_delta": float(delta[idx]),
        "audio_path": find_audio(name, audio_index),
    }


if __name__ == "__main__":
    main()
