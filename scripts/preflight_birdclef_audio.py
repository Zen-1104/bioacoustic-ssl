"""Scan BirdCLEF center/recovery crops once and manifest unusable silent recordings."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bioacoustic_data import BirdCLEFDataset
from bioacoustic_data.noise_mixer import validate_minimum_mixable_rms
from utils.audio_io import (
    DEFAULT_EVALUATION_MINIMUM_RMS,
    EVALUATION_CROP_POLICY,
    load_evaluation_center_crop,
)
from utils.dataset_manifest import fingerprint_paths_labels

PREFLIGHT_SCHEMA_VERSION = 5
CROP_POLICY = EVALUATION_CROP_POLICY

DEFAULT_MAX_READ_ERROR_FRACTION = 0.01
DEFAULT_MAX_EXCLUDED_FRACTION = 0.05


def reason_histogram(silent_count: int, read_errors: list[dict[str, str]]) -> dict[str, int]:
    """Count exclusions by actionable root cause category."""
    
    counts: Counter[str] = Counter()
    if silent_count:
        counts["silent_or_unusable"] = int(silent_count)
        
    seen_read_paths: set[str] = set()
    for row in read_errors:
        path = str(row.get("relative_path", ""))
        
        if not path or path in seen_read_paths:
            continue
            
        seen_read_paths.add(path)
        counts[f"read_error:{row.get('error_type', 'UnknownError')}"] += 1
        
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def print_reason_histogram(histogram: dict[str, int]) -> None:
    print("BirdCLEF preflight exclusion reason histogram:", flush = True)
    
    if not histogram:
        print("  none: 0", flush=True)
        return
        
    for reason, count in histogram.items():
        print(f"  {reason}: {count}", flush = True)


def threshold_failures(
    *, n_files: int, n_excluded: int, n_read_errors: int,
    max_read_error_fraction: float, max_excluded_fraction: float,
) -> list[str]:
    read_error_fraction = n_read_errors / max(n_files, 1)
    excluded_fraction = n_excluded / max(n_files, 1)
    failures = []
    
    if read_error_fraction > max_read_error_fraction:
        failures.append(
            f"read/decode-error fraction {read_error_fraction:.2%} exceeds "
            f"--max-read-error-fraction={max_read_error_fraction:.2%}"
        )
        
    if excluded_fraction > max_excluded_fraction:
        failures.append(
            f"total exclusion fraction {excluded_fraction:.2%} exceeds "
            f"--max-excluded-fraction={max_excluded_fraction:.2%}"
        )
        
    return failures


def validate_threshold_settings(max_read_error_fraction: float, max_excluded_fraction: float) -> None:
    for name, value in (
        ("max_read_error_fraction", max_read_error_fraction),
        ("max_excluded_fraction", max_excluded_fraction),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, inclusive.")


def apply_manifest_thresholds(manifest: dict, max_read_error_fraction: float, max_excluded_fraction: float) -> list[str]:
    """Update threshold metadata and return failures for fresh or reused scans."""
    n_files = int(manifest.get("num_files_scanned", 0))
    
    failures = threshold_failures(
        n_files = n_files,
        n_excluded = int(manifest.get("n_excluded_filenames", 0)),
        n_read_errors = int(manifest.get("n_read_error_files", 0)),
        max_read_error_fraction = max_read_error_fraction,
        max_excluded_fraction = max_excluded_fraction,
    )
    
    manifest["max_read_error_fraction"] = float(max_read_error_fraction)
    manifest["max_excluded_fraction"] = float(max_excluded_fraction)
    manifest["threshold_failures"] = failures
    manifest["thresholds_passed"] = not failures
    manifest["status"] = (
        "threshold_exceeded" if failures else
        "completed_with_exclusions" if int(manifest.get("n_excluded_filenames", 0)) else "passed"
    )
    
    return failures


def validate_preflight_manifest(manifest: dict, *, context: str) -> None:
    """Validate internal counts, filename inventories, histogram and fractions."""
    
    required = {
        "schema_version", 
        "source_fingerprint", 
        "num_files_scanned",
        "n_silent_or_unusable", 
        "silent_or_unusable_filenames",
        "n_read_error_files", 
        "read_error_filenames", 
        "read_errors",
        "n_excluded_filenames", 
        "excluded_filenames", 
        "read_error_fraction",
        "excluded_fraction", 
        "reason_histogram", 
        "max_read_error_fraction",
        "max_excluded_fraction",
    }
    
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"BirdCLEF preflight manifest '{context}' is missing fields {sorted(missing)}; rebuild it with --overwrite.")
        
    total = int(manifest["num_files_scanned"])
    if total <= 0 or not str(manifest["source_fingerprint"]):
        raise ValueError(f"BirdCLEF preflight manifest '{context}' has an empty inventory or source fingerprint.")

    def unique_paths(field: str) -> list[str]:
        value = manifest[field]
        
        if not isinstance(value, list):
            raise ValueError(f"BirdCLEF preflight manifest '{context}' field {field} must be a list.")
            
        paths = [str(item) for item in value]
        if any(not item for item in paths) or len(paths) != len(set(paths)):
            raise ValueError(f"BirdCLEF preflight manifest '{context}' has empty/duplicate paths in {field}.")
            
        return paths

    silent_paths = unique_paths("silent_or_unusable_filenames")
    read_error_paths = unique_paths("read_error_filenames")
    excluded_paths = unique_paths("excluded_filenames")
    
    errors = manifest["read_errors"]
    if not isinstance(errors, list) or any(not isinstance(row, dict) for row in errors):
        raise ValueError(f"BirdCLEF preflight manifest '{context}' has malformed read_errors.")
        
    error_paths = [str(row.get("relative_path", "")) for row in errors]
    if any(not value for value in error_paths) or len(error_paths) != len(set(error_paths)):
        raise ValueError(f"BirdCLEF preflight manifest '{context}' has empty/duplicate read-error paths.")
    if set(error_paths) != set(read_error_paths):
        raise ValueError(f"BirdCLEF preflight manifest '{context}' read-error list does not match read_errors.")
    if int(manifest["n_silent_or_unusable"]) != len(silent_paths):
        raise ValueError(f"BirdCLEF preflight manifest '{context}' silent-file count does not match its path list.")
    if int(manifest["n_read_error_files"]) != len(read_error_paths):
        raise ValueError(f"BirdCLEF preflight manifest '{context}' read-error count does not match its path list.")
        
    expected_excluded = sorted(set(silent_paths) | set(read_error_paths))
    if excluded_paths != expected_excluded or int(manifest["n_excluded_filenames"]) != len(expected_excluded):
        raise ValueError(f"BirdCLEF preflight manifest '{context}' total exclusion list/count is inconsistent.")
        
    expected_histogram = reason_histogram(len(silent_paths), errors)
    if manifest.get("reason_histogram") != expected_histogram or sum(expected_histogram.values()) != len(expected_excluded):
        raise ValueError(f"BirdCLEF preflight manifest '{context}' reason histogram is inconsistent with its exclusion lists.")
        
    expected_read_fraction = len(read_error_paths) / total
    expected_excluded_fraction = len(expected_excluded) / total
    
    for key, expected in (("read_error_fraction", expected_read_fraction), ("excluded_fraction", expected_excluded_fraction)):
        value = float(manifest[key])
        
        if not (0.0 <= value <= 1.0) or abs(value - expected) > 1e-12:
            raise ValueError(f"BirdCLEF preflight manifest '{context}' has inconsistent {key}.")
            
    validate_threshold_settings(float(manifest["max_read_error_fraction"]), float(manifest["max_excluded_fraction"]))


def write_manifest_atomic(output: Path, manifest: dict) -> None:
    output.parent.mkdir(parents = True, exist_ok = True)
    tmp = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(manifest, indent = 2), encoding = "utf-8")
    os.replace(tmp, output)


def raise_threshold_failure(failures: list[str], output: Path) -> None:
    if failures:
        raise RuntimeError(
            "BirdCLEF preflight failed exclusion thresholds: " + "; ".join(failures) + ". "
            f"The diagnostic manifest is retained at '{output}'. Inspect its reason histogram and decoder support. "
            "Override thresholds explicitly only after reviewing the exclusions."
        )


def dataset_source_fingerprint(dataset: BirdCLEFDataset, sample_rate: int, clip_sec: float, minimum_rms: float) -> str:
    records = []
    for path, relative, label in zip(dataset.file_list, dataset.relative_paths, dataset.labels):
        st = Path(path).stat()
        
        records.append({
            "relative_path": str(relative).replace("\\", "/"), "label": int(label),
            "size_bytes": int(st.st_size),
        })
        
    payload = {
        "dataset_path_label_fingerprint": fingerprint_paths_labels(dataset.relative_paths, dataset.labels),
        "records_stat": records, 
        "sample_rate": int(sample_rate), 
        "clip_sec": float(clip_sec),
        "minimum_rms": float(minimum_rms), 
        "crop_policy": CROP_POLICY,
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
    }
    
    return hashlib.sha256(json.dumps(payload, sort_keys = True, separators = (",", ":")).encode()).hexdigest()


def run_or_reuse_preflight(
    dataset: BirdCLEFDataset,
    *, sample_rate: int, clip_sec: float, minimum_rms: float,
    output_path: str | Path, overwrite: bool = False, list_read_errors: bool = False,
    max_read_error_fraction: float = DEFAULT_MAX_READ_ERROR_FRACTION,
    max_excluded_fraction: float = DEFAULT_MAX_EXCLUDED_FRACTION,
) -> dict:
    """Reuse a matching manifest or perform one deterministic audio scan. """
    
    minimum_rms = validate_minimum_mixable_rms(minimum_rms)
    validate_threshold_settings(max_read_error_fraction, max_excluded_fraction)
    
    output = Path(output_path)
    if not dataset.file_list:
        raise RuntimeError("BirdCLEF audio preflight cannot scan an empty file inventory.")
        
    fingerprint = dataset_source_fingerprint(dataset, sample_rate, clip_sec, minimum_rms)
    if output.is_file() and not overwrite:
        saved = json.loads(output.read_text(encoding = "utf-8"))
        
        if saved.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported BirdCLEF preflight schema in '{output}'. Rebuild it once with --overwrite.")
            
        if saved.get("source_fingerprint") != fingerprint:
            raise RuntimeError(
                f"BirdCLEF preflight manifest '{output}' does not match current inventory/file stats/settings. "
                "Rerun scripts/preflight_birdclef_audio.py --overwrite after reviewing data changes."
            )
            
        saved["reason_histogram"] = saved.get("reason_histogram") or reason_histogram(
            len(saved.get("silent_or_unusable_filenames", [])), saved.get("read_errors", [])
        )
        
        validate_preflight_manifest(saved, context=str(output))
        print(f"Reusing BirdCLEF audio preflight: {output} ({len(saved.get('excluded_filenames', saved.get('silent_or_unusable_filenames', [])))} preflight exclusions)")
        print_reason_histogram(saved["reason_histogram"])
        
        if list_read_errors:
            for row in saved.get("read_errors", []):
                print(f"READ_ERROR {row.get('relative_path')}: {row.get('error', 'unknown error')}", flush = True)
                
        failures = apply_manifest_thresholds(saved, max_read_error_fraction, max_excluded_fraction)
        validate_preflight_manifest(saved, context = str(output))
        
        # If an operator explicitly changes the accepted threshold after
        # reviewing the diagnostics, persist the active policy as well.
        write_manifest_atomic(output, saved)
        raise_threshold_failure(failures, output)
        
        return saved

    silent: list[str] = []
    read_errors: list[dict[str, str]] = []
    n = len(dataset.file_list)
    
    for i, path in enumerate(dataset.file_list):
        try:
            load_evaluation_center_crop(
                path, 
                target_sr = sample_rate, 
                duration_sec = clip_sec,
                minimum_rms = minimum_rms,
            )
            
        except ValueError as exc:
            if "SILENT_CROP_REJECTED" in str(exc):
                silent.append(dataset.relative_paths[i])
            else:
                read_errors.append({
                    "relative_path": dataset.relative_paths[i], 
                    "error_type": type(exc).__name__, 
                    "error": str(exc)[:500]
                })
                
        except Exception as exc:  # decoder backends may raise non ValueError exceptions
            read_errors.append({
                "relative_path": dataset.relative_paths[i], 
                "error_type": type(exc).__name__, 
                "error": f"{type(exc).__name__}: {str(exc)[:450]}"
            })
            
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"BirdCLEF audio preflight: {i+1}/{n}; silent = {len(silent)}; read_errors = {len(read_errors)}", flush = True)

    silent_unique = sorted(set(silent))
    read_error_unique = sorted({str(row["relative_path"]) for row in read_errors})
    excluded_unique = sorted(set(silent_unique) | set(read_error_unique))
    histogram = reason_histogram(len(silent_unique), read_errors)
    
    failures = threshold_failures(
        n_files = n, 
        n_excluded = len(excluded_unique), 
        n_read_errors = len(read_error_unique),
        max_read_error_fraction = max_read_error_fraction, 
        max_excluded_fraction = max_excluded_fraction,
    )
    
    result = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "threshold_exceeded" if failures else ("completed_with_exclusions" if excluded_unique else "passed"),
        "source_fingerprint": fingerprint,
        "dataset_path_label_fingerprint": fingerprint_paths_labels(dataset.relative_paths, dataset.labels),
        "dataset_root": str(dataset.requested_root),
        "num_files_scanned": n,
        "minimum_rms": float(minimum_rms), "sample_rate": int(sample_rate), "clip_sec": float(clip_sec),
        "crop_policy": CROP_POLICY,
        "n_silent_or_unusable": len(silent_unique),
        "silent_or_unusable_filenames": silent_unique,
        "n_read_error_files": len(read_error_unique),
        "read_error_filenames": read_error_unique,
        "read_errors": sorted(read_errors, key = lambda row: str(row["relative_path"])),
        "n_excluded_filenames": len(excluded_unique),
        "excluded_filenames": excluded_unique,
        "read_error_fraction": len(read_error_unique) / n,
        "excluded_fraction": len(excluded_unique) / n,
        "max_read_error_fraction": float(max_read_error_fraction),
        "max_excluded_fraction": float(max_excluded_fraction),
        "reason_histogram": histogram,
        "threshold_failures": failures,
        "thresholds_passed": not failures,
        "policy": "Exclude recordings without a usable deterministic fixed-length crop or with audio read/decode errors from all split partitions; retain explicit paths/reasons in this manifest.",
    }
    
    validate_preflight_manifest(result, context = str(output))
    print_reason_histogram(histogram)
    
    if list_read_errors:
        for row in result["read_errors"]:
            print(f"READ_ERROR {row['relative_path']}: {row['error']}", flush = True)
            
    output.parent.mkdir(parents = True, exist_ok = True)
    tmp = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result, indent = 2), encoding = "utf-8")
    os.replace(tmp, output)
    
    print(f"Wrote BirdCLEF preflight manifest: {output} ({len(silent_unique)} silent/unusable; {len(read_error_unique)} read/decode errors; {len(excluded_unique)}/{n} total excluded)")
    raise_threshold_failure(failures, output)
    
    return result


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--audio-dir", default = "data/raw/birdclef")
    parser.add_argument("--output", default = "data/processed/birdclef_audio_preflight.json")
    parser.add_argument("--sample-rate", type=int, default = 16000)
    parser.add_argument("--clip-sec", type=float, default = 10.0)
    parser.add_argument("--minimum-rms", type=float, default = DEFAULT_EVALUATION_MINIMUM_RMS)
    parser.add_argument("--max-read-error-fraction", type=float, default = DEFAULT_MAX_READ_ERROR_FRACTION,
                        help = "Fail closed if audio read/decode errors exceed this share of the inventory (default: 0.01).")
    parser.add_argument("--max-excluded-fraction", type = float, default = DEFAULT_MAX_EXCLUDED_FRACTION,
                        help = "Fail closed if all preflight exclusions exceed this share of the inventory (default: 0.05).")
    parser.add_argument("--overwrite", action = "store_true")
    parser.add_argument("--list-read-errors", action = "store_true", help = "Print every unreadable audio path/error; unreadable files are recorded and excluded rather than aborting the scan.")
    args = parser.parse_args(argv)
    
    if args.sample_rate < 1 or args.clip_sec <= 0:
        parser.error("sample-rate and clip-sec must be positive")
        
    try:
        args.minimum_rms = validate_minimum_mixable_rms(args.minimum_rms)
    except ValueError as exc:
        parser.error(str(exc))
        
    try:
        validate_threshold_settings(args.max_read_error_fraction, args.max_excluded_fraction)
    except ValueError as exc:
        parser.error(str(exc))
        
    dataset = BirdCLEFDataset(
        args.audio_dir, 
        sample_rate = args.sample_rate, 
        duration_sec = args.clip_sec, 
        crop_mode = "center"
    )
    
    result = run_or_reuse_preflight(
        dataset, 
        sample_rate = args.sample_rate, 
        clip_sec = args.clip_sec,
        minimum_rms = args.minimum_rms, 
        output_path = args.output, 
        overwrite = args.overwrite,
        list_read_errors = args.list_read_errors,
        max_read_error_fraction = args.max_read_error_fraction,
        max_excluded_fraction = args.max_excluded_fraction,
    )
    
    print(json.dumps({"manifest": str(args.output), 
                      "scanned": result["num_files_scanned"],
                      "excluded_silent_or_unusable": result["n_silent_or_unusable"],
                      "excluded_read_errors": result.get("n_read_error_files", 0),
                      "reason_histogram": result.get("reason_histogram", {}),
                      "thresholds_passed": result.get("thresholds_passed", False),
                      "total_preflight_exclusions": result.get("n_excluded_filenames", result["n_silent_or_unusable"])}, indent = 2))


if __name__ == "__main__":
    main()