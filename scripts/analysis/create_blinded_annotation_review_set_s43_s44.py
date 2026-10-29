from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import zipfile

import numpy as np
import soundfile as sf

ARCHIVES = {
    43: Path("results/real_soundscape_pair_s43_20261023_094200.npz"),
    44: Path("results/real_soundscape_pair_s44_20261025.npz"),
}
AUDIO_ROOT = Path(
    "data/raw/western_ghats_soundscapes/soundscape_data"
)
OUT = Path(
    "results/real_soundscape_aggregate_s43_s44_fixnames/"
    "blinded_annotation_review_20261025"
)
RNG_SEED = 20261025
PER_GROUP = 20
CLIP_SECONDS = 10.0

# These five recordings were already heard by the first reviewer; exclude the
# entire recordings so none of those previously reviewed examples is repeated.
PREVIOUSLY_REVIEWED = {
    "wgh_0241_001_20181203_121000.flac",
    "wgh_0117_001_20181201_091000.flac",
    "wgh_0713_026_20200313_094500.flac",
    "wgh_0054_001_20181129_064000.flac",
    "wgh_0747_027_20201216_163000.flac",
}


def load_archive(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing archive: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = (
            "targets", "file_names", "segment_start_seconds",
            "segment_types", "class_names", "scores_habitat",
            "scores_standard",
        )
        missing = set(required) - set(data.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        return {key: data[key].copy() for key in required}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")
    if not AUDIO_ROOT.is_dir():
        raise FileNotFoundError(f"Soundscape audio folder not found: {AUDIO_ROOT}")

    data = {seed: load_archive(path) for seed, path in ARCHIVES.items()}
    a43, a44 = data[43], data[44]
    for field in ("targets", "file_names", "segment_start_seconds",
                  "segment_types", "class_names"):
        if not np.array_equal(a43[field], a44[field]):
            raise ValueError(f"Seed archives differ in {field}; aborting.")

    targets = a43["targets"].astype(np.uint8)
    filenames = a43["file_names"].astype(str)
    starts = a43["segment_start_seconds"].astype(float)
    types = np.char.lower(a43["segment_types"].astype(str))
    classes = a43["class_names"].astype(str)

    # Rank by the mean of the two seeds' maximum per-class raw score for each
    # model. These rankings select examples only; scores are hidden from the reviewer.
    mean_habitat = (
        a43["scores_habitat"].astype(np.float64)
        + a44["scores_habitat"].astype(np.float64)
    ) / 2.0
    mean_standard = (
        a43["scores_standard"].astype(np.float64)
        + a44["scores_standard"].astype(np.float64)
    ) / 2.0
    h_top_class = np.argmax(mean_habitat, axis=1)
    s_top_class = np.argmax(mean_standard, axis=1)
    h_top_score = mean_habitat[np.arange(len(targets)), h_top_class]
    s_top_score = mean_standard[np.arange(len(targets)), s_top_class]

    bg_groups: dict[str, list[int]] = defaultdict(list)
    pos_groups: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(filenames):
        key = name.lower()
        if key in PREVIOUSLY_REVIEWED:
            continue
        if types[i] == "background":
            bg_groups[key].append(i)
        elif types[i] == "annotated" and np.any(targets[i] > 0):
            pos_groups[key].append(i)

    rng = np.random.default_rng(RNG_SEED)
    used_files = set(PREVIOUSLY_REVIEWED)
    selections: list[dict] = []

    # One highest-scoring background segment per recording for each model's
    # ranking; choose 10 distinct recordings per model with no cross-group reuse.
    for model_name, score_vector in (
        ("habitat", h_top_score), ("standard", s_top_score)
    ):
        best_by_file = {}
        for name, indices in bg_groups.items():
            idx = max(indices, key=lambda i: float(score_vector[i]))
            best_by_file[name] = {
                "idx": int(idx), "source_filename": str(filenames[idx]),
                "ranking_score": float(score_vector[idx]),
            }
        ranked = sorted(
            best_by_file.values(),
            key=lambda item: item["ranking_score"], reverse=True,
        )
        got = 0
        for candidate in ranked:
            key = candidate["source_filename"].lower()
            if key in used_files:
                continue
            used_files.add(key)
            selections.append({
                **candidate,
                "hidden_stratum": "high_scoring_background",
                "hidden_selection_basis": f"top_{model_name}_mean_score",
            })
            got += 1
            if got == PER_GROUP // 2:
                break
        if got != PER_GROUP // 2:
            raise RuntimeError(
                f"Only selected {got} high-scoring backgrounds for {model_name}."
            )

    # 20 random background recordings, sampled independently of score rank.
    eligible_bg = [name for name in bg_groups if name not in used_files]
    if len(eligible_bg) < PER_GROUP:
        raise RuntimeError(f"Only {len(eligible_bg)} eligible background recordings remain.")
    random_names = rng.permutation(np.array(eligible_bg, dtype=object))[:PER_GROUP]
    for name_obj in random_names:
        name = str(name_obj)
        idx = int(rng.choice(bg_groups[name]))
        used_files.add(name)
        selections.append({
            "idx": idx,
            "source_filename": str(filenames[idx]),
            "hidden_stratum": "random_background",
            "hidden_selection_basis": "uniform_random_recording_and_segment",
        })

    # 20 positive controls from annotated segments, using 20 further distinct recordings.
    eligible_pos = [name for name in pos_groups if name not in used_files]
    if len(eligible_pos) < PER_GROUP:
        raise RuntimeError(f"Only {len(eligible_pos)} eligible positive recordings remain.")
    positive_names = rng.permutation(np.array(eligible_pos, dtype=object))[:PER_GROUP]
    for name_obj in positive_names:
        name = str(name_obj)
        idx = int(rng.choice(pos_groups[name]))
        used_files.add(name)
        selections.append({
            "idx": idx,
            "source_filename": str(filenames[idx]),
            "hidden_stratum": "annotated_positive_control",
            "hidden_selection_basis": "uniform_random_annotated_positive_recording_and_segment",
        })

    if len(selections) != 3 * PER_GROUP:
        raise RuntimeError(f"Expected 60 selections, found {len(selections)}")

    # Randomize the full set before assigning anonymous IDs, preventing ID or
    # ZIP order from revealing category. Resolve source file paths by basename.
    rng.shuffle(selections)
    audio_index: dict[str, list[Path]] = defaultdict(list)
    for path in AUDIO_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".flac", ".wav", ".mp3", ".ogg", ".m4a"}:
            audio_index[path.name.lower()].append(path)

    for item in selections:
        paths = audio_index.get(item["source_filename"].lower(), [])
        if len(paths) != 1:
            raise FileNotFoundError(
                f"Expected one audio path for {item['source_filename']}, found: {paths}"
            )
        item["audio_path"] = paths[0]

    # Build everything only after selection and audio-path validation succeeded.
    OUT.mkdir(parents=True, exist_ok=False)
    clip_dir = OUT / "audio"
    clip_dir.mkdir()

    key_rows = []
    template_rows = []
    for number, item in enumerate(selections, start=1):
        clip_id = f"clip_{number:03d}"
        idx = int(item["idx"])
        source_path: Path = item["audio_path"]
        start_sec = float(starts[idx])
        with sf.SoundFile(str(source_path)) as audio:
            sr = int(audio.samplerate)
            start_frame = int(round(start_sec * sr))
            frames_required = int(round(CLIP_SECONDS * sr))
            if start_frame < 0 or start_frame + frames_required > len(audio):
                raise ValueError(
                    f"Cannot extract full {CLIP_SECONDS}s clip at {start_sec}s "
                    f"from {source_path.name} (sample rate {sr}, frames {len(audio)})."
                )
            audio.seek(start_frame)
            clip = audio.read(
                frames=frames_required, dtype="float32", always_2d=True
            )
        clip_path = clip_dir / f"{clip_id}.flac"
        sf.write(str(clip_path), clip, sr, format="FLAC", subtype="PCM_24")

        positive_classes = [
            str(classes[j]) for j in np.flatnonzero(targets[idx] > 0)
        ]
        key_rows.append({
            "clip_id": clip_id,
            "hidden_stratum": item["hidden_stratum"],
            "hidden_selection_basis": item["hidden_selection_basis"],
            "source_filename": str(filenames[idx]),
            "source_start_seconds": start_sec,
            "source_end_seconds": start_sec + CLIP_SECONDS,
            "source_segment_type": str(types[idx]),
            "positive_target_species_codes": ";".join(positive_classes),
            "positive_target_count": len(positive_classes),
            "mean_seed_habitat_top_species": str(classes[h_top_class[idx]]),
            "mean_seed_habitat_top_score": float(h_top_score[idx]),
            "mean_seed_standard_top_species": str(classes[s_top_class[idx]]),
            "mean_seed_standard_top_score": float(s_top_score[idx]),
            "source_audio_path": str(source_path.resolve()),
        })
        template_rows.append({
            "clip_id": clip_id,
            "bird_vocalization_heard_yes_no_uncertain": "",
            "species_identified_if_confident": "",
            "confidence_high_medium_low": "",
            "notes_other_sounds_or_uncertainty": "",
        })

    template_path = OUT / "review_template.csv"
    write_csv(template_path, template_rows)
    key_path = OUT / "answer_key_internal_DO_NOT_SHARE_WITH_REVIEWER.csv"
    write_csv(key_path, key_rows)

    readme = (
        "BLINDED SOUNDscape AUDIO REVIEW\n\n"
        "Listen to each 10-second clip without consulting model results or other labels.\n"
        "For each clip, fill in review_template.csv:\n"
        "- whether any bird vocalization is audible (yes/no/uncertain);\n"
        "- species identity only if genuinely confident (otherwise leave uncertain);\n"
        "- confidence (high/medium/low);\n"
        "- notes about masking, overlapping sounds, or uncertainty.\n\n"
        "Do not infer species from clip ID or try to guess how clips were selected.\n"
        "Some sounds may be ambiguous; uncertainty is a valid response.\n"
        "The audio has been extracted at its original level and is not per-clip normalized.\n"
    )
    readme_path = OUT / "README_FOR_REVIEWER.txt"
    readme_path.write_text(readme, encoding="utf-8")

    summary = {
        "n_clips": len(selections),
        "n_unique_source_recordings": len({r["source_filename"] for r in key_rows}),
        "clips_per_hidden_stratum": {
            "random_background": PER_GROUP,
            "high_scoring_background": PER_GROUP,
            "annotated_positive_control": PER_GROUP,
        },
        "high_score_background_selection": (
            "10 unique recordings ranked by the two-seed mean habitat-model maximum class score; "
            "10 further unique recordings ranked by the two-seed mean standard-model maximum class score."
        ),
        "random_seed": RNG_SEED,
        "previously_reviewed_recordings_excluded": sorted(PREVIOUSLY_REVIEWED),
        "answer_key_is_in_zip": False,
        "important": (
            "Share only reviewer_package_blinded.zip with the second reviewer. "
            "Keep answer_key_internal_DO_NOT_SHARE_WITH_REVIEWER.csv private. "
            "The selection groups are deliberately hidden from the reviewer."
        ),
    }
    (OUT / "sampling_summary_internal.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    zip_path = OUT / "reviewer_package_blinded.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(readme_path, arcname="README_FOR_REVIEWER.txt")
        archive.write(template_path, arcname="review_template.csv")
        for clip_path in sorted(clip_dir.glob("clip_*.flac")):
            archive.write(clip_path, arcname=f"audio/{clip_path.name}")

    counts = defaultdict(int)
    for row in key_rows:
        counts[row["hidden_stratum"]] += 1

    print("BLINDED REVIEW SET CREATED", flush=True)
    print(f"Clips: {len(key_rows)}", flush=True)
    print(f"Unique source recordings: {len({r['source_filename'] for r in key_rows})}", flush=True)
    print("Hidden category counts (do not share this output with reviewer):", dict(counts), flush=True)
    print(f"Reviewer ZIP: {zip_path.resolve()}", flush=True)
    print(f"Internal answer key (DO NOT SHARE): {key_path.resolve()}", flush=True)
    print(f"Blank template: {template_path.resolve()}", flush=True)
    print(f"Internal summary: {(OUT / 'sampling_summary_internal.json').resolve()}", flush=True)


if __name__ == "__main__":
    main()
