"""Spectrogram augmentations used by the standard-SSL control."""
import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param: int = 8, time_mask_param: int = 16):
        super().__init__()

        if freq_mask_param < 0 or time_mask_param < 0:
            raise ValueError("Mask parameters must be non negative.")
        
        self.freq_mask_param = int(freq_mask_param)
        self.time_mask_param = int(time_mask_param)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.ndim < 2:
            raise ValueError(f"SpecAugment expects at least 2 dimensions [F,T], got {tuple(spec.shape)}.")
        
        masked = spec.clone()
        num_mels, num_steps = masked.shape[-2:]

        if num_mels < 1 or num_steps < 1:
            raise ValueError(f"Invalid spectrogram dimensions {tuple(masked.shape)}.")

        max_f = min(self.freq_mask_param, num_mels)
        f = int(torch.randint(0, max_f + 1, (1,)).item())
        f0 = int(torch.randint(0, max(1, num_mels - f + 1), (1,)).item())

        if f:
            masked[..., f0:f0 + f, :] = 0.0

        max_t = min(self.time_mask_param, num_steps)
        t = int(torch.randint(0, max_t + 1, (1,)).item())
        t0 = int(torch.randint(0, max(1, num_steps - t + 1), (1,)).item())

        if t:
            masked[..., :, t0:t0 + t] = 0.0
        return masked