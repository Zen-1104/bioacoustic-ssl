from pathlib import Path
import csv
import hashlib
import zipfile

import soundfile as sf

ROOT = Path(".")
SOURCE = ROOT / (
    "results/real_soundscape_aggregate_s43_s44_fixnames/"
    "species_context_audit_20261025/candidate_segments_for_listening.csv"
)
OUT = ROOT / (
    "results/real_soundscape_aggregate_s43_s44_fixnames/"
    "species_context_listening_clips_20261025"
)

SPECIES = [
    "gyhcaf1", "rewbul", "barfly1",
    "kerlau2", "crbsun2", "yebbul3",
]

CANDIDATE_TYPES = [
    "positive_target_habitat_score_gain",
    "positive_target_standard_score_gain",
    "background_target_negative_top_habitat_score",
    "background_target_negative_largest_score_delta",
]

def rank_key(row, kind):
    delta = float(row["score_delta"])
    habitat = float(row["habitat_score"])
    if kind == "positive_target_standard_score_gain":
        return delta
    if kind == "background_target_negative_top_habitat_score":
        return -habitat
    return -delta

def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Candidate manifest not found: {SOURCE}")
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")

    with SOURCE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    chosen = []
    missing = []

    # Choose one representative for each category, species and model seed.
    for species in SPECIES:
        for seed in ("43", "44"):
            for kind in CANDIDATE_TYPES:
                options = [
                    r for r in rows
                    if r["species"] == species
                    and r["seed"] == seed
                    and r["candidate_type"] == kind
                    and r.get("audio_path", "").strip()
                ]
                options.sort(key=lambda r: rank_key(r, kind))
                if options:
                    chosen.append(options[0])
                else:
                    missing.append((species, seed, kind))

    if not chosen:
        raise RuntimeError("No candidates with resolved audio paths were found.")

    OUT.mkdir(parents=True)
    clips_dir = OUT / "clips"
    clips_dir.mkdir()

    clip_cache = {}
    manifest_rows = []

    for n, row in enumerate(chosen, start=1):
        source_path = Path(row["audio_path"])
        if not source_path.is_file():
            print(f"SKIP: missing audio file {source_path}")
            continue

        start_sec = float(row["segment_start_seconds"])
        duration = float(row.get("segment_duration_seconds", 10))
        cache_key = (str(source_path.resolve()), start_sec, duration)

        if cache_key not in clip_cache:
            token = hashlib.sha256(
                f"{cache_key[0]}|{start_sec}|{duration}".encode()
            ).hexdigest()[:12]
            clip_path = clips_dir / f"clip_{token}.flac"

            with sf.SoundFile(str(source_path)) as audio_file:
                sr = audio_file.samplerate
                start_frame = int(round(start_sec * sr))
                if start_frame < 0 or start_frame >= len(audio_file):
                    print(
                        f"SKIP: timestamp {start_sec}s outside {source_path.name}"
                    )
                    continue

                frames = min(
                    int(round(duration * sr)),
                    len(audio_file) - start_frame,
                )
                audio_file.seek(start_frame)
                audio = audio_file.read(
                    frames=frames, dtype="float32", always_2d=True
                )

            sf.write(
                str(clip_path), audio, sr,
                format="FLAC", subtype="PCM_24",
            )
            clip_cache[cache_key] = (
                clip_path.name, sr, audio.shape[1], len(audio) / sr
            )

        clip_name, sr, channels, actual_duration = clip_cache[cache_key]

        manifest_rows.append({
            "clip_file": f"clips/{clip_name}",
            "species": row["species"],
            "seed": row["seed"],
            "candidate_type": row["candidate_type"],
            "file_name": row["file_name"],
            "start_seconds": start_sec,
            "duration_seconds": round(actual_duration, 4),
            "segment_type": row["segment_type"],
            "target_for_species": row["target_for_species"],
            "habitat_score": row["habitat_score"],
            "standard_score": row["standard_score"],
            "score_delta": row["score_delta"],
            "original_audio_path": str(source_path),
            "sample_rate_hz": sr,
            "channels": channels,
        })

    manifest = OUT / "listening_manifest.csv"
    if manifest_rows:
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(manifest_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

    zip_path = OUT / "species_context_listening_clips.zip"
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.write(manifest, arcname="listening_manifest.csv")
        for clip_path in sorted(clips_dir.glob("*.flac")):
            archive.write(
                clip_path, arcname=f"clips/{clip_path.name}"
            )

    print("\nCLIP EXPORT COMPLETE")
    print(f"Selected candidate rows: {len(chosen)}")
    print(f"Manifest rows exported: {len(manifest_rows)}")
    print(f"Unique audio clips: {len(list(clips_dir.glob('*.flac')))}")
    print(f"ZIP: {zip_path.resolve()}")
    print(f"Manifest: {manifest.resolve()}")
    if missing:
        print(f"Categories without a resolved candidate: {len(missing)}")
        for item in missing:
            print("MISSING:", " | ".join(item))

if __name__ == "__main__":
    main()
