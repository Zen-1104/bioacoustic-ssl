"""Shared validation for stress test SNR configuration."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


def validate_snr_levels(raw_levels: Any) -> tuple[list[float | None], list[str]]:
    """Validate YAML SNR values and return normalized values plus output labels."""
    if not isinstance(raw_levels, list) or not raw_levels:
        raise ValueError("evaluation.snr_levels must be a non-empty YAML list, e.g. [null, 20, 10, 0, -5].")

    values: list[float | None] = []
    labels: list[str] = []
    for index, raw in enumerate(raw_levels):
        if raw is None:
            value = None
            label = "Clean"
        else:
            if isinstance(raw, bool):
                raise ValueError(f"evaluation.snr_levels[{index}] must be numeric or null, not boolean.")
                
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"evaluation.snr_levels[{index}] must be numeric or null; got {raw!r}.") from exc
                
            if not math.isfinite(value):
                raise ValueError(f"evaluation.snr_levels[{index}] must be finite; got {raw!r}.")
                
            label = f"{value:g}dB"
            
        values.append(value)
        labels.append(label)

    if len(set(labels)) != len(labels):
        raise ValueError(f"evaluation.snr_levels contains duplicate output conditions: {labels}.")
        
    return values, labels


def load_stress_snr_config(config_path: str | Path) -> tuple[dict[str, Any], list[float | None], list[str]]:
    """Read a stress test YAML and return config, normalized SNR values, and labels."""
    
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Stress-test config does not exist: '{path}'.")
        
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or not isinstance(cfg.get("evaluation"), dict):
        raise ValueError(f"Stress test config '{path}' must contain an evaluation mapping.")
        
    values, labels = validate_snr_levels(cfg["evaluation"].get("snr_levels"))
    return cfg, values, labels


def snr_labels_csv(config_path: str | Path) -> str:
    """Return the comma separated result labels used by the configured SNR sweep."""
    _, _, labels = load_stress_snr_config(config_path)
    return ",".join(labels)
