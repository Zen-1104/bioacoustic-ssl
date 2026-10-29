"""Audio loading and log mel frontend shared by training and evaluation."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import torchaudio

CropMode = Literal["random", "center"]
ShortAudioPolicy = Literal["zero_pad", "repeat"]

_RESAMPLER_CACHE: dict[tuple[int, int], torch.nn.Module] = {}
_MEL_TRANSFORM_CACHE: dict[tuple[int, int, int, int, str], tuple[torch.nn.Module, torch.nn.Module]] = {}

# Keep this threshold and policy identifier shared by preflight, probe, stress, external baseline and smoke test code.
# Changing the crop rule requires a new identifier so cached features cannot silently cross protocols.
DEFAULT_EVALUATION_MINIMUM_RMS = 1e-5
EVALUATION_CROP_POLICY = "center-first-deterministic-quarter-window-rms-recovery-v2"


def get_resampler(orig_sr: int, target_sr: int) -> torch.nn.Module:
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Sample rates must be positive; got orig_sr = {orig_sr}, target_sr = {target_sr}.")

    key = (int(orig_sr), int(target_sr))
    if key not in _RESAMPLER_CACHE:
        _RESAMPLER_CACHE[key] = torchaudio.transforms.Resample(orig_freq = orig_sr, new_freq = target_sr)

    return _RESAMPLER_CACHE[key]


def load_and_preprocess_audio(
    filepath: str | Path,
    target_sr: int = 16000,
    duration_sec: float = 10,
    *,
    crop_mode: CropMode = "random",
    short_audio_policy: ShortAudioPolicy = "zero_pad",
    minimum_rms: float | None = None,
) -> torch.Tensor:
    """Load mono audio and return a fixed length [1, T] tensor.

    ``crop_mode='center'`` is intended for deterministic test runs.
    Short clips are zero-padded by default to avoid periodic repetition artifacts;
    noise pool callers may explicitly choose ``short_audio_policy='repeat'``.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: '{path}'.")

    if target_sr <= 0 or duration_sec <= 0:
        raise ValueError("target_sr and duration_sec must be positive.")

    if crop_mode not in ("random", "center"):
        raise ValueError(f"Unknown crop_mode = {crop_mode!r}; expected 'random' or 'center'.")

    if short_audio_policy not in ("zero_pad", "repeat"):
        raise ValueError(f"Unknown short_audio_policy={short_audio_policy!r}.")

    if minimum_rms is not None and (not math.isfinite(float(minimum_rms)) or float(minimum_rms) <= 0):
        raise ValueError("minimum_rms must be a positive finite value when supplied.")

    waveform, sr = torchaudio.load(str(path))
    if waveform.ndim != 2 or waveform.shape[0] < 1 or waveform.shape[1] < 1:
        raise ValueError(f"Audio '{path}' has an invalid/empty shape {tuple(waveform.shape)}.")

    waveform = waveform.to(dtype = torch.float32)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim = 0, keepdim = True)

    if sr != target_sr:
        waveform = get_resampler(sr, target_sr)(waveform)

    target_samples = int(round(target_sr * duration_sec))
    if target_samples < 1:
        raise ValueError(f"Requested duration produces invalid sample count {target_samples}.")

    num_samples = waveform.shape[-1]

    if num_samples < target_samples:
        if short_audio_policy == "repeat":
            repeats = (target_samples + num_samples - 1) // num_samples
            waveform = waveform.repeat(1, repeats)[..., :target_samples]
        else:
            waveform = F.pad(waveform, (0, target_samples - num_samples))

        if minimum_rms is not None and float(waveform.square().mean().sqrt().item()) <= float(minimum_rms):
            raise ValueError(
                f"SILENT_CROP_REJECTED path = '{path}': the complete short recording is silent / near-silent "
                f"(RMS <= {float(minimum_rms):.3e}); there is no alternate crop to select."
            )
    elif num_samples > target_samples:
        max_start = num_samples - target_samples
        initial_start = max_start // 2 if crop_mode == "center" else int(torch.randint(0, max_start + 1, (1,)).item())

        if minimum_rms is None:
            waveform = waveform[..., initial_start:initial_start + target_samples]
        else:
            threshold = float(minimum_rms)
            seen_starts = {initial_start}
            selected_start = None
            retries = 0

            def crop_at(start: int) -> torch.Tensor:
                return waveform[..., start:start + target_samples]

            if float(crop_at(initial_start).square().mean().sqrt().item()) > threshold:
                selected_start = initial_start
            else:
                # Retry randomized crops for training; center crop evaluation falls
                # back to deterministic positions so successful recovery stays stable.
                if crop_mode == "random":
                    for _ in range(8):
                        start_idx = int(torch.randint(0, max_start + 1, (1,)).item())

                        if start_idx in seen_starts:
                            continue

                        seen_starts.add(start_idx)
                        retries += 1

                        if float(crop_at(start_idx).square().mean().sqrt().item()) > threshold:
                            selected_start = start_idx
                            break

                # A quarter window stride provides overlapping fallback candidates;
                # this recovery path is reached only when the first crop is silent.
                stride = max(1, target_samples // 4)
                fallback_starts = list(range(0, max_start + 1, stride))

                if fallback_starts[-1] != max_start:
                    fallback_starts.append(max_start)

                fallback_starts.sort(key = lambda value: (abs(value - initial_start), value))

                if selected_start is None:
                    for start_idx in fallback_starts:
                        if start_idx in seen_starts:
                            continue

                        seen_starts.add(start_idx)
                        retries += 1

                        if float(crop_at(start_idx).square().mean().sqrt().item()) > threshold:
                            selected_start = start_idx
                            break

                if selected_start is None:
                    logging.getLogger(__name__).error(
                        "SILENT_CROP_REJECTED path=%s attempts=%d minimum_rms=%.3e",
                        path, retries + 1, threshold,
                    )
                    raise ValueError(
                        f"SILENT_CROP_REJECTED path='{path}': no {duration_sec:g}-second crop exceeded "
                        f"minimum RMS {threshold:.3e} after {retries + 1} candidate crops."
                    )

                logging.getLogger(__name__).warning(
                    "SILENT_CROP_RECOVERED path=%s retries=%d initial_start=%d selected_start=%d minimum_rms=%.3e",
                    path, retries, initial_start, selected_start, threshold,
                )

            waveform = crop_at(selected_start)

    if minimum_rms is not None and float(waveform.square().mean().sqrt().item()) <= float(minimum_rms):
        raise ValueError(
            f"SILENT_CROP_REJECTED path = '{path}': selected crop is silent / near-silent "
            f"(RMS <= {float(minimum_rms):.3e})."
        )
    if not torch.isfinite(waveform).all():
        raise ValueError(f"Audio '{path}' contains NaN or infinite samples after preprocessing.")
    return waveform.contiguous()


def load_evaluation_center_crop(
    filepath: str | Path,
    target_sr: int = 16000,
    duration_sec: float = 10.0,
    *,
    minimum_rms: float = DEFAULT_EVALUATION_MINIMUM_RMS,
) -> torch.Tensor:
    """Load the shared deterministic evaluation crop. """

    return load_and_preprocess_audio(
        filepath,
        target_sr = target_sr,
        duration_sec = duration_sec,
        crop_mode = "center",
        short_audio_policy = "repeat",
        minimum_rms = minimum_rms,
    )


def load_fixed_crops(
    filepath: str | Path,
    target_sr: int = 16000,
    duration_sec: float = 10.0,
    *,
    num_crops: int = 3,
    minimum_rms: float = 1e-5,
) -> torch.Tensor:
    """Load deterministic, temporally spread, non silent crops as ``[N,1,T]``. """

    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: '{path}'.")

    if target_sr < 1 or duration_sec <= 0 or num_crops < 1 or minimum_rms <= 0:
        raise ValueError("target_sr, duration_sec, num_crops and minimum_rms must be positive.")

    wave, original_sr = torchaudio.load(str(path))
    if wave.ndim != 2 or wave.shape[0] < 1 or wave.shape[-1] < 1:
        raise ValueError(f"Audio '{path}' has an invalid/empty shape {tuple(wave.shape)}.")

    wave = wave.to(torch.float32)
    if wave.shape[0] > 1:
        wave = wave.mean(dim = 0, keepdim = True)

    if int(original_sr) != int(target_sr):
        wave = get_resampler(int(original_sr), int(target_sr))(wave)

    if not torch.isfinite(wave).all():
        raise ValueError(f"Audio '{path}' contains NaN or infinite samples after preprocessing.")

    target_samples = int(round(float(target_sr) * float(duration_sec)))
    if target_samples < 1:
        raise ValueError("Requested duration produces an invalid target sample count.")

    if wave.shape[-1] < target_samples:
        repeats = (target_samples + wave.shape[-1] - 1) // wave.shape[-1]
        crop = wave.repeat(1, repeats)[..., :target_samples].contiguous()

        if float(crop.square().mean().sqrt()) <= minimum_rms:
            raise ValueError(f"SILENT_CROP_REJECTED path = '{path}': short recording is below minimum RMS {minimum_rms:.3e}.")

        return crop.unsqueeze(0)

    max_start = int(wave.shape[-1] - target_samples)
    if num_crops == 1:
        targets = [max_start // 2]
    else:
        targets = sorted(set(int(round(x)) for x in torch.linspace(0, max_start, steps = num_crops).tolist()))

    stride = max(1, target_samples // max(4, num_crops * 2))
    candidates = sorted(set(targets + list(range(0, max_start + 1, stride)) + [max_start]))
    valid: dict[int, torch.Tensor] = {}

    for start in candidates:
        crop = wave[..., start:start + target_samples]
        if float(crop.square().mean().sqrt()) > minimum_rms:
            valid[start] = crop.contiguous()

    if not valid:
        raise ValueError(f"SILENT_CROP_REJECTED path='{path}': no fixed length crop exceeds minimum RMS {minimum_rms:.3e}.")

    selected: list[torch.Tensor] = []
    used: set[int] = set()

    for target in targets:
        choice = next((start for start in sorted(valid, key=lambda item: (abs(item - target), item)) if start not in used), None)
        if choice is None:
            break

        selected.append(valid[choice])
        used.add(choice)

    if not selected:
        raise RuntimeError(f"Fixed crop selection returned no valid crop for '{path}'.")
    return torch.stack(selected, dim = 0).contiguous()


def get_mel_transform(
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    device: torch.device | str,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    device = torch.device(device)
    key = (sample_rate, n_mels, n_fft, hop_length, str(device))

    if key not in _MEL_TRANSFORM_CACHE:
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate = sample_rate,
            n_fft = n_fft,
            hop_length = hop_length,
            n_mels = n_mels,
        ).to(device)

        amp = torchaudio.transforms.AmplitudeToDB(stype = "power", top_db = 80).to(device)
        _MEL_TRANSFORM_CACHE[key] = (mel, amp)

    return _MEL_TRANSFORM_CACHE[key]


def log_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> torch.Tensor:
    """Convert [1,T] or [B,1,T] waveform to normalized [B,1,F,T] log mel."""
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(1)
    elif waveform.ndim != 3:
        raise ValueError(f"Expected waveform shape [C,T] or [B,C,T], got {tuple(waveform.shape)}.")

    if waveform.shape[1] != 1:
        raise ValueError(f"Expected mono waveform (channel dimension 1), got {tuple(waveform.shape)}.")

    if waveform.shape[-1] < 1:
        raise ValueError("Cannot compute a spectrogram for an empty waveform.")

    mel_transform, amp_to_db = get_mel_transform(sample_rate, n_mels, n_fft, hop_length, waveform.device)
    log_mel = amp_to_db(mel_transform(waveform))
    mean = log_mel.mean(dim=(-2, -1), keepdim = True)
    std = log_mel.std(dim=(-2, -1), keepdim = True, unbiased = False).clamp_min(1e-6)
    result = (log_mel - mean) / std

    if not torch.isfinite(result).all():
        raise FloatingPointError("Log mel frontend produced NaN or infinite values.")

    return result


if __name__ == "__main__":
    dummy_wave = torch.randn(2, 1, 160000)
    spec = log_mel_spectrogram(dummy_wave)
    print(f"Input waveform: {tuple(dummy_wave.shape)}")
    print(f"Log-mel spectrogram: {tuple(spec.shape)}")
