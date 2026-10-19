"""Unlabelled Xeno-Canto dataset yielding matched, augmented contrastive views."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from utils.audio_io import load_and_preprocess_audio, log_mel_spectrogram
from utils.dataset_manifest import fingerprint_relative_paths, fingerprint_file_inventory
from bioacoustic_data.augmentations import SpecAugment

_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg")
_LOG = logging.getLogger(__name__)


class XenoCantoDataset(Dataset):
    def __init__(
        self,
        audio_dir,
        noise_pool = None,
        noise_mixer = None,
        variant = "habitat",
        sample_rate = 16000,
        duration_sec = 10,
        exclude_manifest_path = None,
        require_overlap_audit = False,
        preflight_manifest_path = None,
        cache_dir = None,
        require_preflight = False,
        max_crop_retries = 5,
        min_crop_rms = 1e-5,
    ):
        root = Path(audio_dir).expanduser().resolve()

        if not root.is_dir():
            raise FileNotFoundError(f"Xeno-Canto directory does not exist: '{root}'.")
        
        if max_crop_retries < 1 or min_crop_rms <= 0:
            raise ValueError("max_crop_retries and min_crop_rms must be positive.")
        
        exclusions: set[str] = set()
        audit_data = None

        if exclude_manifest_path:
            manifest = Path(exclude_manifest_path)

            if not manifest.is_file():
                if require_overlap_audit:
                    raise FileNotFoundError(
                        f"Overlap audit manifest '{manifest}' is missing. Run scripts/audit_dataset_overlap.py before pretraining."
                    )
            else:
                audit_data = json.loads(manifest.read_text(encoding = "utf-8"))
                exclusions = set(audit_data.get("excluded_relative_paths", []))

                if require_overlap_audit and not audit_data.get("xeno_canto_paths_sha256"):
                    raise RuntimeError(f"Overlap audit '{manifest}' has an unsupported schema; rerun the audit.")

        all_files: list[tuple[str, str]] = []

        for current_root, dirs, filenames in os.walk(root):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))

            for filename in sorted(filenames, key = lambda x: (x.casefold(), x)):
                if filename.lower().endswith(_AUDIO_EXTS):
                    path = Path(current_root) / filename
                    all_files.append((str(path), path.relative_to(root).as_posix()))

        all_files.sort(key = lambda pair: (pair[1].casefold(), pair[1]))
        all_relative_paths = [rel for _, rel in all_files]
        current_path_hash = fingerprint_relative_paths(all_relative_paths)
        current_inventory_hash = fingerprint_file_inventory(root, all_relative_paths)

        if audit_data is not None and audit_data.get("xeno_canto_paths_sha256") != current_path_hash:
            raise RuntimeError(
                f"Current Xeno-Canto file index differs from overlap audit '{exclude_manifest_path}'. "
                "Rerun the audit before training."
            )

        preflight_data = None
        cached_records: dict[str, str] = {}

        if preflight_manifest_path:
            pf_path = Path(preflight_manifest_path)

            if not pf_path.is_file():
                if require_preflight:
                    raise FileNotFoundError(
                        f"Xeno-Canto preflight manifest '{pf_path}' is missing. Run preflight_xeno_audio.py first."
                    )
            else:
                preflight_data = json.loads(pf_path.read_text(encoding = "utf-8"))

                if preflight_data.get("schema_version") != 3:
                    raise RuntimeError(
                        "Xeno-Canto preflight manifest schema is outdated; "
                        "rebuild it with scripts/preflight_xeno_audio.py --overwrite."
                    )

                if preflight_data.get("audio_inventory_sha256") != current_inventory_hash:
                    raise RuntimeError("Xeno-Canto source inventory (relative paths, byte sizes, and modification times) "
                                       "differs from the preflight manifest; rerun preflight."
                                    )
                
                if int(preflight_data.get("sample_rate", -1)) != int(sample_rate):
                    raise RuntimeError("Xeno-Canto cache sample rate differs from the requested sample rate.")
                
                if abs(float(preflight_data.get("duration_sec", -1)) - float(duration_sec)) > 1e-8:
                    raise RuntimeError("Xeno-Canto cache clip duration differs from the requested duration.")
                
                cached_records = {str(k): str(v["cache_path"]) for k, v in preflight_data.get("records", {}).items()}

                if not cached_records:
                    raise RuntimeError("Xeno-Canto preflight manifest has no valid cached recordings.")
                
                if cache_dir is None:
                    raise ValueError("cache_dir must be supplied when using a preflight manifest.")

        self.audio_dir = str(root)
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self.cache_records = cached_records
        self.file_list = []
        self.relative_paths = []

        for path, rel in all_files:
            if rel in exclusions:
                continue
            if preflight_data is not None and rel not in cached_records:
                continue

            self.file_list.append(path)
            self.relative_paths.append(rel)

        if not self.file_list:
            raise FileNotFoundError(f"No eligible Xeno-Canto audio files found under '{root}'.")

        self.excluded_count = len(exclusions)
        self.preflight_excluded_count = len(preflight_data.get("invalid", {})) if preflight_data else 0
        self.overlap_audit = audit_data
        self.preflight_manifest = preflight_data
        self.noise_pool = noise_pool
        self.noise_mixer = noise_mixer
        self.variant = variant
        self.sample_rate = int(sample_rate)
        self.duration_sec = float(duration_sec)
        self.max_crop_retries = int(max_crop_retries)
        self.min_crop_rms = float(min_crop_rms)
        self.spec_augment = SpecAugment()
        self._crop_retries = 0

        if variant not in {"habitat", "standard", "white_noise"}:
            raise ValueError(f"Unknown augmentation variant: {variant!r}.")
        
        if variant == "habitat" and (not noise_pool or noise_mixer is None):
            raise ValueError("Variant 'habitat' requires a non empty seen habitat noise pool and NoiseMixer.")
        
        if variant == "white_noise" and noise_mixer is None:
            raise ValueError("Variant 'white_noise' requires a NoiseMixer.")
        
        if preflight_data is not None and self.cache_dir is None:
            raise ValueError("A cache directory is required for preflight-cached audio.")

    def __len__(self):
        return len(self.file_list)

    @staticmethod
    def _apply_time_shift(waveform, max_fraction = 0.05):
        """Shift a short distance and zero fill the boundary (same policy for all arms)."""

        if not 0.0 <= max_fraction < 0.5:
            raise ValueError("max_fraction must be in [0, 0.5).")
        
        length = waveform.shape[-1]
        max_shift = int(length * max_fraction)

        if max_shift < 1:
            return waveform.clone()
        
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())

        if shift == 0:
            return waveform.clone()
        
        result = torch.zeros_like(waveform)

        if shift > 0:
            result[..., shift:] = waveform[..., :length - shift]
        else:
            result[..., :length + shift] = waveform[..., -shift:]

        return result

    def _load_clean_crop_pair(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load two crops from the same source recording for symmetric contrastive views."""
        rel = self.relative_paths[idx]

        if self.cache_records:
            cached_path = self.cache_dir / self.cache_records[rel]

            if not cached_path.is_file():
                raise FileNotFoundError(f"Preflight audio cache file missing: '{cached_path}'. Rerun preflight.")
            crops = torch.load(cached_path, map_location = "cpu", weights_only = True)

            if not isinstance(crops, torch.Tensor) or crops.ndim != 3 or crops.shape[1] != 1 or crops.shape[0] < 1:
                raise ValueError(f"Invalid cached crops in '{cached_path}': expected [N,1,T].")
            
            n_crops = int(crops.shape[0])

            if n_crops >= 2:
                first_index = int(torch.randint(0, n_crops, (1,)).item())
                second_index = int(torch.randint(0, n_crops - 1, (1,)).item())

                if second_index >= first_index:
                    second_index += 1
            else:
                first_index = second_index = 0

            pair = (crops[first_index].to(torch.float32), crops[second_index].to(torch.float32))

            for waveform in pair:
                if not torch.isfinite(waveform).all() or float(waveform.square().mean().sqrt()) < self.min_crop_rms:
                    raise ValueError(f"Cached crop for '{rel}' is silent or invalid; rerun the audio preflight.")
                
            return pair

        pair = []
        for attempt_idx in range(2):
            waveform = None

            for attempt in range(self.max_crop_retries):
                candidate = load_and_preprocess_audio(
                    self.file_list[idx], 
                    self.sample_rate, 
                    self.duration_sec,
                    crop_mode = "random", 
                    short_audio_policy = "repeat",
                )

                rms = float(candidate.square().mean().sqrt())

                if torch.isfinite(candidate).all() and rms >= self.min_crop_rms:
                    waveform = candidate

                    if attempt:
                        self._crop_retries += attempt
                    break

            if waveform is None:
                raise ValueError(
                    f"Could not obtain a non silent {self.duration_sec:g}s crop from '{rel}' after "
                    f"{self.max_crop_retries} attempts. Run preflight_xeno_audio.py and inspect the excluded recordings report."
                )
            pair.append(waveform)

        return pair[0], pair[1]

    def __getitem__(self, idx):
        clean_view_1, clean_view_2 = self._load_clean_crop_pair(idx)
        view1_wave = self._apply_time_shift(clean_view_1)
        view2_wave = self._apply_time_shift(clean_view_2)

        if self.variant == "habitat":
            noise_idx = int(torch.randint(0, len(self.noise_pool), (1,)).item())

            view2_wave = self.noise_mixer.mix(
                view2_wave, self.noise_pool[noise_idx],
                snr_db=float(torch.randint(-5, 21, (1,)).item()),
            )
        elif self.variant == "white_noise":
            white_noise = torch.randn_like(view2_wave)

            view2_wave = self.noise_mixer.mix(
                view2_wave, white_noise, snr_db = float(torch.randint(-5, 21, (1,)).item())
            )

        view1_mel = self.spec_augment(log_mel_spectrogram(view1_wave, sample_rate = self.sample_rate))
        view2_mel = self.spec_augment(log_mel_spectrogram(view2_wave, sample_rate = self.sample_rate))

        return view1_mel.squeeze(0), view2_mel.squeeze(0) 