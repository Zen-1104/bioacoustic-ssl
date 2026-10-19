"""Helpers for stable data index fingerprints used by all experiment stages."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence


def fingerprint_paths_labels(relative_paths: Sequence[str], labels: Sequence[int]) -> str:
    """Fingerprint the ordered relative path / class label index, not audio bytes.

    This detects changes to the evaluated file set, its stable order, or class mapping.
    It intentionally does not claim to detect a replacement of audio bytes at an
    unchanged path; the overlap audit is a separate (byte hash) check.
    """
    if len(relative_paths) != len(labels):
        raise ValueError(f"Path/label length mismatch: {len(relative_paths)} paths vs {len(labels)} labels.")
        
    payload = "\n".join(f"{path}\t{int(label)}" for path, label in zip(relative_paths, labels)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fingerprint_relative_paths(relative_paths: Sequence[str]) -> str:
    """Fingerprint a sorted path index, independent of filesystem enumeration order."""
    
    normalized = [str(path).replace("\\", "/") for path in relative_paths]
    
    if any(not path for path in normalized):
        raise ValueError("Relative path index contains an empty path.")
        
    if len(normalized) != len(set(normalized)):
        raise ValueError("Relative path index contains duplicate paths.")
        
    payload = "\n".join(sorted(normalized, key = lambda value: (value.casefold(), value))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_manifest_file_index(manifest: dict, relative_paths: Sequence[str], labels: Sequence[int], *, context: str) -> None:
    """Fail if a split manifest was built for a different index or has split leakage."""
    
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError(f"{context}: dataset index contains duplicate relative filenames.")

    split_names = {}
    for field in ("train_filenames", "validation_filenames", "test_filenames"):
        names = manifest.get(field)
        
        if not isinstance(names, list) or not names:
            raise ValueError(f"{context}: manifest field '{field}' must be a non empty filename list.")
            
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"{context}: manifest field '{field}' contains an invalid filename.")
            
        if len(names) != len(set(names)):
            raise ValueError(f"{context}: manifest field '{field}' contains duplicate filenames.")
            
        split_names[field] = set(names)

    train = split_names["train_filenames"]
    validation = split_names["validation_filenames"]
    test = split_names["test_filenames"]
    
    if train & validation or train & test or validation & test:
        raise ValueError(f"{context}: train/validation/test filenames overlap.")

    expected = set(relative_paths)
    excluded_names = manifest.get("excluded_filenames", [])
    
    if not isinstance(excluded_names, list) or any(not isinstance(name, str) or not name for name in excluded_names):
        raise ValueError(f"{context}: excluded_filenames must be a list of relative filenames.")
        
    excluded = set(excluded_names)
    if len(excluded_names) != len(excluded) or excluded & (train | validation | test):
        raise ValueError(f"{context}: excluded_filenames contain duplicates or overlap active splits.")
        
    declared = train | validation | test | excluded
    if expected != declared:
        missing = sorted(expected - declared)[:5]
        stale = sorted(declared - expected)[:5]
        raise ValueError(f"{context}: file set differs from split manifest. Missing from manifest = {missing}; absent from dataset = {stale}.")

    observed = fingerprint_paths_labels(relative_paths, labels)
    declared_fingerprint = manifest.get("dataset_fingerprint")
    if not declared_fingerprint or observed != declared_fingerprint:
        raise ValueError(f"{context}: current path/label fingerprint does not match split manifest. Rebuild the splits deliberately.")

    all_names = set(expected)
    for fraction_key, fraction_names in manifest.get("train_fraction_filenames", {}).items():
    
        if not isinstance(fraction_names, list) or not fraction_names:
            raise ValueError(f"{context}: fraction '{fraction_key}' must have a non empty filename list.")
            
        if len(fraction_names) != len(set(fraction_names)):
            raise ValueError(f"{context}: fraction '{fraction_key}' contains duplicate filenames.")
            
        if not set(fraction_names).issubset(train) or not set(fraction_names).issubset(all_names):
            raise ValueError(f"{context}: fraction '{fraction_key}' contains filenames outside the training split.")
