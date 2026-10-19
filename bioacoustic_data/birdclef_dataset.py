"""Deterministic label BirdCLEF dataset with stable relative filenames."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.audio_io import DEFAULT_EVALUATION_MINIMUM_RMS, load_and_preprocess_audio, log_mel_spectrogram

_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg")
_LOG = logging.getLogger(__name__)


def _audio_files_under(root: Path) -> list[Path]:
    found: list[Path] = []

    for current_root, dirs, filenames in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))

        for filename in filenames:
            if filename.lower().endswith(_AUDIO_EXTS):
                found.append(Path(current_root) / filename)

    return sorted(found, key=lambda p: (p.as_posix().casefold(), p.as_posix()))


class BirdCLEFDataset(Dataset):
    def __init__(
        self,
        audio_dir: str,
        noise_pool: list | None = None,
        noise_mixer: Any = None,
        snr_db: float | None = None,
        sample_rate: int = 16000,
        duration_sec: float = 10,
        *,
        crop_mode: str = "random",
        short_audio_policy: str = "repeat",
        include_path: bool = False,
        noise_snr_db_range: tuple[float, float] | None = None,
        noise_probability: float = 1.0,
        minimum_rms: float | None = None,
        include_mix_status: bool = False,
    ):
        requested_root = Path(audio_dir).expanduser().resolve()

        if not requested_root.exists() or not requested_root.is_dir():
            raise FileNotFoundError(f"BirdCLEF directory does not exist: '{requested_root}'.")

        if (requested_root / "train_audio").is_dir():
            audio_root = requested_root / "train_audio"
        else:
            audio_root = requested_root

        self.requested_root = requested_root
        self.audio_root = audio_root
        self.sample_rate = int(sample_rate)
        self.duration_sec = float(duration_sec)
        self.crop_mode = crop_mode
        self.short_audio_policy = short_audio_policy
        self.include_path = include_path
        self.noise_pool = noise_pool
        self.noise_mixer = noise_mixer
        self.snr_db = snr_db
        self.noise_snr_db_range = noise_snr_db_range
        self.noise_probability = float(noise_probability)
        self.minimum_rms = None if minimum_rms is None else float(minimum_rms)
        self.include_mix_status = bool(include_mix_status)

        if self.minimum_rms is not None and (not torch.isfinite(torch.tensor(self.minimum_rms)) or self.minimum_rms <= 0):
            raise ValueError("minimum_rms must be positive and finite when supplied.")
        
        self.silent_mix_skipped_count = 0

        if crop_mode not in ("random", "center"):
            raise ValueError("crop_mode must be 'random' or 'center'.")
        
        if self.snr_db is not None and self.noise_snr_db_range is not None:
            raise ValueError("Use either fixed snr_db or noise_snr_db_range, not both.")
        
        if self.noise_snr_db_range is not None:
            if len(self.noise_snr_db_range) != 2:
                raise ValueError("noise_snr_db_range must be a (min_db, max_db) pair.")
            
            low, high = map(float, self.noise_snr_db_range)

            if not (float("-inf") < low <= high < float("inf")):
                raise ValueError("noise_snr_db_range must contain finite min_db <= max_db.")
            
            if low != int(low) or high != int(high):
                raise ValueError("noise_snr_db_range endpoints must be integers to match the MoCo SNR sampling.")
            
            self.noise_snr_db_range = (int(low), int(high))

        if not 0.0 <= self.noise_probability <= 1.0:
            raise ValueError("noise_probability must be between 0 and 1.")
        
        if (self.snr_db is not None or (self.noise_snr_db_range is not None and self.noise_probability > 0)) and (not self.noise_pool or self.noise_mixer is None):
            raise ValueError("Noise augmentation requires a non-empty noise_pool and a NoiseMixer.")

        metadata_df = self._read_metadata(requested_root, audio_root)
        metadata_by_key = self._index_metadata(metadata_df)

        class_dirs = sorted(
            (p for p in audio_root.iterdir() if p.is_dir() and not p.name.startswith(".")),
            key = lambda p: (p.name.casefold(), p.name),
        )
        discovered: list[tuple[Path, str | None]] = []

        for class_dir in class_dirs:
            for path in _audio_files_under(class_dir):
                discovered.append((path, class_dir.name))

        # Metadata only fallback for archives that store all audio files flat
        if not discovered:
            direct_files = [p for p in audio_root.iterdir() if p.is_file() and p.suffix.lower() in _AUDIO_EXTS]

            if metadata_df is not None and "primary_label" in metadata_df.columns:
                for path in sorted(direct_files, key=lambda p: (p.name.casefold(), p.name)):
                    row = self._metadata_for_path(path, metadata_by_key, requested_root, audio_root)

                    if row is None or not str(row.get("primary_label", "")).strip():
                        raise ValueError(f"No primary_label metadata found for '{path}'.")
                    
                    discovered.append((path, str(row["primary_label"]).strip()))

        if not discovered:
            raise FileNotFoundError(
                f"No labelled audio found under '{audio_root}'. Expected train_audio/<species>/*.ogg "
                "or one directory per species. Do not point at the unfiltered archive root."
            )

        # Use metadata primary_label only if metadata covers every discovered path
        metadata_labels: list[str] = []
        metadata_rows: list[dict[str, Any] | None] = []
        metadata_is_complete = metadata_df is not None and "primary_label" in metadata_df.columns

        for path, folder_label in discovered:
            row = self._metadata_for_path(path, metadata_by_key, requested_root, audio_root)
            metadata_rows.append(row)

            if row is None or not str(row.get("primary_label", "")).strip():
                metadata_is_complete = False
                metadata_labels.append(folder_label or "")

            else:
                metadata_labels.append(str(row["primary_label"]).strip())

        if not metadata_is_complete:
            metadata_labels = [folder_label or "" for _, folder_label in discovered]

        if any(not label for label in metadata_labels):
            raise ValueError("Some audio files have no usable class label. Provide species subdirectories or complete metadata.")

        classes = sorted(set(metadata_labels), key=str.lower)

        self.label_map = {name: idx for idx, name in enumerate(classes)}
        self.file_list = [str(path) for path, _ in discovered]
        self.relative_paths = [Path(os.path.relpath(path, requested_root)).as_posix() for path, _ in discovered]
        self.labels = [self.label_map[label] for label in metadata_labels]
        self.group_ids, self.grouping_source = self._make_group_ids(metadata_rows, self.relative_paths)
        self.label_names = metadata_labels

        # Keep the stable ID as the source of truth; path order is deterministic.
        ordered = sorted(zip(self.file_list, self.relative_paths, self.labels, self.label_names, self.group_ids), key = lambda x: (x[1].casefold(), x[1]))

        self.file_list = [x[0] for x in ordered]
        self.relative_paths = [x[1] for x in ordered]
        self.labels = [x[2] for x in ordered]
        self.label_names = [x[3] for x in ordered]
        self.group_ids = [x[4] for x in ordered]

    @staticmethod
    def _read_metadata(requested_root: Path, audio_root: Path) -> pd.DataFrame | None:
        candidates = [
            requested_root / "train_metadata.csv",
            requested_root / "train.csv",
            requested_root / "metadata.csv",
            audio_root / "train_metadata.csv",
            requested_root.parent / "train_metadata.csv",
            requested_root.parent / "train.csv",
        ]

        for candidate in candidates:
            if candidate.is_file():
                df = pd.read_csv(candidate)

                if "filename" in df.columns:
                    return df
                _LOG.warning("Ignoring metadata file without a 'filename' column: %s", candidate)

        return None

    @staticmethod
    def _index_metadata(df: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
        if df is None:
            return {}
        
        index: dict[str, dict[str, Any]] = {}

        for row in df.to_dict(orient="records"):
            filename = str(row.get("filename", "")).replace("\\", "/").strip()

            if not filename:
                continue

            keys = {filename, Path(filename).name}

            for key in keys:
                # Duplicate basenames are marked ambiguous instead of overwritten
                if key in index and index[key] != row:
                    index[key] = {"__ambiguous__": True}
                else:
                    index[key] = row
        return index

    @staticmethod
    def _metadata_for_path(
        path: Path,
        indexed: dict[str, dict[str, Any]],
        requested_root: Path | None = None,
        audio_root: Path | None = None,
    ) -> dict[str, Any] | None:
        """Match metadata by absolute, archive relative, species relative, or unique basename."""
        keys = [path.as_posix(), path.name]

        for root in (requested_root, audio_root):
            if root is not None:
                try:
                    keys.append(path.relative_to(root).as_posix())
                except ValueError:
                    pass
                
        for key in dict.fromkeys(keys):
            row = indexed.get(key)
            if row is not None and not row.get("__ambiguous__"):
                return row
        return None

    @staticmethod
    def _make_group_ids(rows: list[dict[str, Any] | None], relative_paths: list[str]) -> tuple[list[str], str]:
        def valid(v: Any) -> bool:
            if v is None or pd.isna(v):
                return False
            
            return str(v).strip().lower() not in {"", "nan", "none", "unknown"}

        for col in ("author", "recordist", "recordist_id"):
            if any(row and valid(row.get(col)) for row in rows):
                groups = []

                for row, relpath in zip(rows, relative_paths):
                    value = row.get(col) if row else None
                    groups.append(f"{col}:{str(value).strip()}" if valid(value) else f"missing:{relpath}")

                missing_groups = any(not valid(row.get(col) if row else None) for row in rows)
                source = f"{col}_with_missing_singletons" if missing_groups else col

                return groups, source

        if any(row and valid(row.get("latitude")) and valid(row.get("longitude")) for row in rows):
            groups = []

            for row, relpath in zip(rows, relative_paths):
                if row and valid(row.get("latitude")) and valid(row.get("longitude")):
                    try:
                        lat = round(float(row["latitude"]), 2)
                        lon = round(float(row["longitude"]), 2)
                        groups.append(f"location:{lat:.2f},{lon:.2f}")
                    except (TypeError, ValueError):
                        groups.append(f"missing:{relpath}")
                else:
                    groups.append(f"missing:{relpath}")

            missing_groups = any(not (row and valid(row.get("latitude")) and valid(row.get("longitude"))) for row in rows)
            source = "rounded_latitude_longitude_with_missing_singletons" if missing_groups else "rounded_latitude_longitude"

            return groups, source
        
        return [f"file:{p}" for p in relative_paths], "file_level_no_metadata"

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        path = self.file_list[idx]

        needs_noise_mix = (
            self.snr_db is not None
            or (self.noise_snr_db_range is not None and self.noise_probability > 0.0)
        )
        skip_noise_mix = False

        try:
            waveform = load_and_preprocess_audio(
                path,
                target_sr = self.sample_rate,
                duration_sec = self.duration_sec,
                crop_mode = self.crop_mode,
                short_audio_policy = self.short_audio_policy,

                minimum_rms = (
                    self.minimum_rms if self.minimum_rms is not None
                    else DEFAULT_EVALUATION_MINIMUM_RMS if needs_noise_mix
                    else None
                ),
            )

        except ValueError as exc:
            if not needs_noise_mix or "SILENT_CROP_REJECTED" not in str(exc):
                raise
            
            waveform = load_and_preprocess_audio(
                path, 
                target_sr = self.sample_rate, 
                duration_sec = self.duration_sec,
                crop_mode = self.crop_mode, 
                short_audio_policy = self.short_audio_policy,
            )
            skip_noise_mix = True

            self.silent_mix_skipped_count += 1
            _LOG.warning(
                "SILENT_CROP_MIX_SKIPPED path=%s count=%d policy=keep_clean_not_noise_only reason=%s",
                path, self.silent_mix_skipped_count, str(exc).replace("\n", " "),
            )

        if self.snr_db is not None and not skip_noise_mix:
            noise_wave = self.noise_pool[idx % len(self.noise_pool)]
            
            # Evaluation uses a deterministic center offset; training may randomize.
            noise_offset = max((noise_wave.shape[-1] - waveform.shape[-1]) // 2, 0) if self.crop_mode == "center" else None

            waveform = self.noise_mixer.mix(
                waveform, noise_wave, snr_db = float(self.snr_db), noise_offset = noise_offset
            )
        elif self.noise_snr_db_range is not None and self.noise_probability > 0 and not skip_noise_mix:
            if float(torch.rand(()).item()) < self.noise_probability:
                noise_idx = int(torch.randint(0, len(self.noise_pool), (1,)).item())
                noise_wave = self.noise_pool[noise_idx]
                low_db, high_db = self.noise_snr_db_range
                snr_db = int(torch.randint(low_db, high_db + 1, (1,)).item())
                waveform = self.noise_mixer.mix(waveform, noise_wave, snr_db=float(snr_db))

        mel_spec = log_mel_spectrogram(waveform, sample_rate = self.sample_rate)
        item = (mel_spec.squeeze(0), int(self.labels[idx]))

        if self.include_path:
            if self.include_mix_status:
                return (*item, self.relative_paths[idx], bool(skip_noise_mix))
            
            return (*item, self.relative_paths[idx])
        
        if self.include_mix_status:
            return (*item, bool(skip_noise_mix))
        
        return item