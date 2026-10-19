"""SNR controlled additive noise mixing with loudness renormalization."""
import math
import torch


DEFAULT_POWER_EPSILON = 1e-10
MINIMUM_MIXABLE_RMS = math.sqrt(DEFAULT_POWER_EPSILON)


def validate_minimum_mixable_rms(value: float, *, option: str = "--minimum-rms") -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{option} must be a finite number >= {MINIMUM_MIXABLE_RMS:.1e}.") from exc

    if not math.isfinite(threshold) or threshold < MINIMUM_MIXABLE_RMS:
        raise ValueError(
            f"{option} must be finite and >= {MINIMUM_MIXABLE_RMS:.1e}; "
            "lower values can admit crops that NoiseMixer rejects as near-silent."
        )
    return threshold


class NoiseMixer:
    def __init__(self, sample_rate: int = 16000, power_epsilon: float = DEFAULT_POWER_EPSILON):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        
        power_epsilon = float(power_epsilon)

        if not math.isfinite(power_epsilon) or power_epsilon <= 0:
            raise ValueError("power_epsilon must be positive and finite.")
        
        self.sample_rate = int(sample_rate)
        self.power_epsilon = power_epsilon

    def mix(
        self, clean_wave: torch.Tensor, noise_wave: torch.Tensor, snr_db: float, *,
        noise_offset: int | None = None, generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Mix noise at a target SNR and renormalize mixture RMS to clean RMS."""

        if clean_wave.ndim != 2 or noise_wave.ndim != 2:
            raise ValueError(f"Expected [channels,time] tensors; got clean = {tuple(clean_wave.shape)}, noise={tuple(noise_wave.shape)}.")
        
        if clean_wave.shape[0] != noise_wave.shape[0]:
            if noise_wave.shape[0] == 1:
                noise_wave = noise_wave.expand(clean_wave.shape[0], -1)
            elif clean_wave.shape[0] == 1:
                clean_wave = clean_wave.expand(noise_wave.shape[0], -1)
            else:
                raise ValueError("Clean/noise channel counts are incompatible.")
            
        if clean_wave.shape[-1] < 1 or noise_wave.shape[-1] < 1:
            raise ValueError("Clean/noise waveforms must not be empty.")
        
        if not math.isfinite(float(snr_db)):
            raise ValueError(f"snr_db must be finite; got {snr_db!r}.")
        
        if not torch.isfinite(clean_wave).all() or not torch.isfinite(noise_wave).all():
            raise ValueError("Clean/noise waveform contains NaN or infinite values.")

        target_len = clean_wave.shape[-1]
        noise_len = noise_wave.shape[-1]

        if noise_len < target_len:
            repeats = (target_len + noise_len - 1) // noise_len
            noise_wave = noise_wave.repeat(1, repeats)
            noise_len = noise_wave.shape[-1]

        if noise_len > target_len:
            max_offset = noise_len - target_len

            if noise_offset is None:
                # A caller can pass a freshly seeded generator per sample to make
                # random offset selection reproducible across models/processes.
                if generator is None:
                    offset_tensor = torch.randint(0, max_offset + 1, (1,))
                else:
                    offset_device = getattr(generator, "device", torch.device("cpu"))
                    offset_tensor = torch.randint(
                        0, max_offset + 1, (1,), generator = generator, device = offset_device
                    )
                offset = int(offset_tensor.item())
                
            else:
                offset = int(noise_offset)

                if not 0 <= offset <= max_offset:
                    raise ValueError(f"noise_offset = {offset} outside [0, {max_offset}].")
                
            noise_wave = noise_wave[..., offset:offset + target_len]

        p_clean = clean_wave.square().mean(dim = -1, keepdim = True)
        p_noise = noise_wave.square().mean(dim = -1, keepdim = True)

        if torch.any(p_clean <= self.power_epsilon):
            raise ValueError("Clean waveform is silent or near-silent; refusing to compute an SNR mixture.")
        
        if torch.any(p_noise <= self.power_epsilon):
            raise ValueError("Noise waveform is silent or near-silent; refusing to substitute clean audio.")

        snr_linear = 10.0 ** (float(snr_db) / 10.0)
        alpha = torch.sqrt(p_clean / (snr_linear * p_noise))
        mixture = clean_wave + alpha * noise_wave
        p_mix = mixture.square().mean(dim = -1, keepdim = True)

        if torch.any(p_mix <= self.power_epsilon) or not torch.isfinite(p_mix).all():
            raise FloatingPointError("Noise mixing produced invalid mixture power.")
        
        mixture = mixture * torch.sqrt(p_clean / p_mix)

        if not torch.isfinite(mixture).all():
            raise FloatingPointError("Noise mixing produced NaN or infinite samples.")
        
        return mixture