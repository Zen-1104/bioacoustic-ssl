"""Validation helpers for versioned BirdCLEF audio preflight metadata."""
from __future__ import annotations

import math
from collections.abc import Mapping


PREFLIGHT_AUDIT_FIELDS = frozenset({
    "n_preflight_excluded_files", 
    "preflight_excluded_filenames",
    "n_preflight_read_error_files_excluded", 
    "preflight_read_error_filenames",
    "preflight_reason_histogram", 
    "preflight_read_error_fraction", 
    "preflight_excluded_fraction",
    "preflight_max_read_error_fraction", 
    "preflight_max_excluded_fraction",
})


def validate_preflight_audit(record: Mapping, *, context: str) -> dict:
    """Validate and normalize the preflight audit fields saved with a result. """
    
    missing = PREFLIGHT_AUDIT_FIELDS - set(record)
    if missing:
        raise ValueError(f"{context} is missing BirdCLEF preflight audit fields {sorted(missing)}.")

    def string_list(key: str) -> list[str]:
        value = record[key]
        
        if not isinstance(value, list):
            raise ValueError(f"{context} has malformed {key}; expected a list.")
            
        normalized = [str(item) for item in value]
        if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError(f"{context} has empty or duplicate values in {key}.")
            
        return sorted(normalized)

    excluded = string_list("preflight_excluded_filenames")
    read_errors = string_list("preflight_read_error_filenames")
    n_excluded = int(record["n_preflight_excluded_files"])
    n_read_errors = int(record["n_preflight_read_error_files_excluded"])
    
    if n_excluded != len(excluded) or n_read_errors != len(read_errors):
        raise ValueError(f"{context} preflight exclusion counts do not match the saved filename lists.")
        
    if n_read_errors > n_excluded or not set(read_errors).issubset(excluded):
        raise ValueError(f"{context} has read error exclusions not contained in the full exclusion list.")

    histogram_raw = record["preflight_reason_histogram"]
    if not isinstance(histogram_raw, dict):
        raise ValueError(f"{context} has malformed preflight_reason_histogram; expected an object.")
        
    histogram: dict[str, int] = {}
    for reason, count in histogram_raw.items():
        try:
            count_int = int(count)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context} has a non integer count in preflight_reason_histogram.") from exc
            
        if not str(reason) or count_int < 0 or count_int != count:
            raise ValueError(f"{context} has an invalid count in preflight_reason_histogram.")
            
        histogram[str(reason)] = count_int
        
    if sum(histogram.values()) != n_excluded:
        raise ValueError(f"{context} preflight reason histogram does not sum to total excluded files.")

    values = {}
    for key in ("preflight_read_error_fraction", "preflight_excluded_fraction",
                "preflight_max_read_error_fraction", "preflight_max_excluded_fraction"):
        try:
            value = float(record[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context} has an invalid numeric {key}.") from exc
            
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{context} has {key} outside [0, 1].")
            
        values[key] = value
        
    if values["preflight_read_error_fraction"] > values["preflight_max_read_error_fraction"] + 1e-12:
        raise ValueError(f"{context} read error fraction exceeds the saved accepted threshold.")
        
    if values["preflight_excluded_fraction"] > values["preflight_max_excluded_fraction"] + 1e-12:
        raise ValueError(f"{context} total exclusion fraction exceeds the saved accepted threshold.")
        
    return {
        "n_preflight_excluded_files": n_excluded,
        "preflight_excluded_filenames": excluded,
        "n_preflight_read_error_files_excluded": n_read_errors,
        "preflight_read_error_filenames": read_errors,
        "preflight_reason_histogram": dict(sorted(histogram.items())),
        **values,
    }
