"""Validate Xeno-Canto audio and cache several fixed length 16 kHz crops per recording."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import torch
import torchaudio

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.dataset_manifest import fingerprint_file_inventory

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


def indexed_audio(root: Path) -> list[tuple[Path, str]]:
    items = []
    for current_root, dirs, filenames in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))

        for name in sorted(filenames, key = lambda x: (x.casefold(), x)):
            path = Path(current_root) / name

            if path.suffix.lower() in AUDIO_EXTS:
                items.append((path, path.relative_to(root).as_posix()))

    return sorted(items, key = lambda x: (x[1].casefold(), x[1]))


def seeded_processing_order(indexed: list[tuple[Path, str]], seed: int) -> list[tuple[Path, str]]:
    """Return a reproducible shuffled processing order without changing inventory order."""
    ordered = list(indexed)
    random.Random(int(seed)).shuffle(ordered)

    return ordered


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False

    return True


_SLURM_ACTIVE_STATES = {
    "CONFIGURING", "COMPLETING", "PENDING", "RUNNING", "RESIZING",
    "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "SIGNALING", "STAGE_OUT",
    "STOPPED", "SUSPENDED", "UPDATE_DB",
}
_SLURM_TERMINAL_STATES = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
    "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "SPECIAL_EXIT",
    "TIMEOUT",
}


def _classify_slurm_states(output: str) -> bool | None:
    """Classify newline separated SLURM states; preserve if any state is active."""
    states = [line.strip().split()[0].upper().rstrip("+")
              for line in output.splitlines() if line.strip()]

    if not states:
        return None

    if any(state in _SLURM_ACTIVE_STATES for state in states):
        return True
    if all(state in _SLURM_TERMINAL_STATES for state in states):
        return False

    return None


def _slurm_job_is_active(job_id: str) -> bool | None:
    """Return whether a SLURM job is active, consulting accounting if needed. """
    if not job_id:
        return None

    squeue = shutil.which("squeue")
    if squeue is not None:
        try:
            result = subprocess.run(
                [squeue, "--noheader", "--jobs", str(job_id), "--format=%T"],
                capture_output = True, text = True, timeout = 5, check = False,
            )
        except (OSError, subprocess.TimeoutExpired):
            result = None

        if result is not None and result.returncode == 0:
            if not result.stdout.strip():
                return False

            state = _classify_slurm_states(result.stdout)
            if state is not None:
                return state

    sacct = shutil.which("sacct")
    if sacct is not None:
        try:
            result = subprocess.run(
                [sacct, "--noheader", "--parsable2", "--allocations", "--jobs", str(job_id), "--format=State"],
                capture_output = True, text = True, timeout = 5, check = False,
            )
        except (OSError, subprocess.TimeoutExpired):
            result = None

        if result is not None and result.returncode == 0:
            state = _classify_slurm_states(result.stdout.replace("|", "\n"))

            if state is not None:
                return state
    return None


def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """Return the lower bound of a two sided 95% Wilson interval by default."""
    if trials <= 0:
        return 0.0

    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")

    if not math.isfinite(float(z)) or z <= 0:
        raise ValueError("z must be finite and positive")

    n = float(trials)
    p_hat = float(successes) / n
    z2 = float(z) ** 2
    denominator = 1.0 + z2 / n
    center = p_hat + z2 / (2.0 * n)
    margin = float(z) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))

    return max(0.0, min(1.0, (center - margin) / denominator))


def cleanup_stale_staging_dirs(
    cache_root: Path, stale_after_hours: float = 72.0, *, now: float | None = None,
) -> list[Path]:
    """Safely remove stale preflight staging directories and report what was removed. """

    if not math.isfinite(float(stale_after_hours)) or stale_after_hours <= 0:
        raise ValueError("stale_after_hours must be finite and positive")

    current_time = time.time() if now is None else float(now)
    host = socket.gethostname()
    parent = cache_root.parent
    prefix = f".{cache_root.name}.staging-"

    removed: list[Path] = []
    if not parent.exists():
        return removed

    for candidate in sorted(parent.glob(prefix + "*"), key = lambda p: p.name):
        if candidate.is_symlink() or not candidate.is_dir():
            continue

        marker = candidate / ".preflight-owner.json"
        owner = None

        try:
            if marker.is_file():
                owner = json.loads(marker.read_text(encoding = "utf-8"))
        except (OSError, json.JSONDecodeError):
            owner = None

        try:
            fallback_mtime = candidate.stat().st_mtime
            raw_started = owner.get("started_unix", fallback_mtime) if isinstance(owner, dict) else fallback_mtime

            try:
                started_at = float(raw_started)
            except (TypeError, ValueError):
                started_at = fallback_mtime

            if not math.isfinite(started_at):
                started_at = fallback_mtime

            age_seconds = max(0.0, current_time - started_at)

        except OSError:
            continue

        age_hours = age_seconds / 3600.0
        stale = False
        reason = "age threshold exceeded"

        if isinstance(owner, dict):
            job_id = str(owner.get("slurm_job_id") or "")
            job_active = _slurm_job_is_active(job_id) if job_id else None

            if job_active is True:
                continue
            if job_active is False:
                stale, reason = True, f"SLURM job {job_id} is no longer queued/running"

            elif owner.get("hostname") == host:
                try:
                    pid = int(owner.get("pid", -1))
                except (TypeError, ValueError):
                    pid = -1
                if _pid_is_alive(pid) and age_hours < stale_after_hours:
                    continue

                stale = (not _pid_is_alive(pid)) or age_hours >= stale_after_hours
                reason = "owner PID is gone" if not _pid_is_alive(pid) else "staging age threshold exceeded"
            else:
                stale = age_hours >= stale_after_hours

        else:
            stale = age_hours >= stale_after_hours

        if stale:
            try:
                shutil.rmtree(candidate)
                removed.append(candidate)
                print(f"Removed stale preflight staging directory {candidate} ({reason}; age = {age_hours:.1f}h).", flush = True)
            except OSError as exc:
                print(f"WARNING: could not remove stale staging directory {candidate}: {exc}", file = sys.stderr, flush = True)
        else:
            print(f"Preserving non stale preflight staging directory {candidate} (age = {age_hours:.1f}h).", flush = True)

    return removed


def write_stage_owner_marker(stage_root: Path, order_seed: int) -> Path:
    """Write ownership metadata used to distinguish active from abandoned staging."""

    marker = stage_root / ".preflight-owner.json"

    payload = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_unix": time.time(),
        "order_seed": int(order_seed),
    }

    marker.write_text(json.dumps(payload, indent = 2), encoding = "utf-8")
    return marker


def make_crops(wave: torch.Tensor, target_samples: int, num_crops: int, min_rms: float) -> torch.Tensor:
    """Select non silent fixed length crops spread across the whole recording."""

    if wave.ndim != 2 or wave.shape[0] != 1 or wave.shape[-1] < 1:
        raise ValueError(f"Invalid mono waveform shape: {tuple(wave.shape)}")

    if not torch.isfinite(wave).all():
        raise ValueError("Audio contains NaN or infinite samples")

    length = wave.shape[-1]
    if length < target_samples:
        repeats = (target_samples + length - 1) // length
        padded = wave.repeat(1, repeats)[..., :target_samples]

        if float(padded.square().mean().sqrt()) < min_rms:
            raise ValueError("repeated short audio crop is below the silence threshold")

        return padded.unsqueeze(0).contiguous()

    max_start = length - target_samples

    # Candidate starts are spread over the complete source, not biased toward the beginning.
    # For each target position, search nearby alternatives if a silent interval would otherwise produce a near silent crop.
    candidate_count = max(num_crops * 16, num_crops)
    candidates = sorted(set(int(round(v)) for v in torch.linspace(0, max_start, steps = candidate_count).tolist()))

    target_positions = [
        int(round(v)) for v in torch.linspace(0, max_start, steps = num_crops).tolist()
    ]

    valid_by_start: dict[int, torch.Tensor] = {}
    for start in candidates:
        crop = wave[..., start:start + target_samples]
        rms = float(crop.square().mean().sqrt())

        if torch.isfinite(crop).all() and rms >= min_rms:
            valid_by_start[start] = crop.contiguous()

    if not valid_by_start:
        raise ValueError("no non silent fixed length crop found")

    selected: list[torch.Tensor] = []
    used_starts: set[int] = set()

    for target in target_positions:
        choices = sorted(valid_by_start, key = lambda start: (abs(start - target), start))

        chosen = next((start for start in choices if start not in used_starts), None)
        if chosen is None:
            # Preserve the true number of independent usable windows.
            break

        used_starts.add(chosen)
        selected.append(valid_by_start[chosen])

    if not selected:
        raise ValueError("no non silent fixed length crop found")

    return torch.stack(selected, dim = 0).contiguous()


def write_exclusion_audit(invalid: dict[str, str], manifest_path: Path) -> Path:
    """Write the current rejected file audit and summarize reason counts."""

    invalid_path = manifest_path.with_name("xeno_canto_preflight_excluded.csv")
    if not invalid:
        return invalid_path

    reason_counts = Counter(reason.split(":", 1)[0] for reason in invalid.values())
    print("Preflight exclusion reason histogram:", flush = True)

    for reason, count in sorted(reason_counts.items(), key = lambda item: (-item[1], item[0])):
        print(f"  {reason}: {count}", flush = True)

    import csv
    with invalid_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_path", "reason"])
        writer.writerows(sorted(invalid.items()))

    print(f"Excluded-file audit: {invalid_path}", flush = True)
    return invalid_path


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--audio-dir", default = "data/raw/xeno_canto")
    parser.add_argument("--cache-dir", default = "data/processed/xeno_canto_cache")
    parser.add_argument("--manifest", default = "data/processed/xeno_canto_preflight.json")
    parser.add_argument("--sample-rate", type=int, default = 16000)
    parser.add_argument("--clip-sec", type=float, default = 10.0)
    parser.add_argument("--num-crops", type=int, default = 3)
    parser.add_argument("--min-rms", type=float, default = 1e-5)
    parser.add_argument(
        "--max-excluded-fraction", type=float, default = 0.05,
        help = "Abort if this fraction of source audio files cannot be cached (default: 0.05).",
    )
    parser.add_argument(
        "--early-abort-min-files", type=int, default = 1000,
        help = "Minimum files before statistical early abort checks (default: 1000).",
    )
    parser.add_argument(
        "--early-abort-check-interval", type=int, default = 100,
        help = "Check early abort evidence every N files (default: 100).",
    )
    parser.add_argument(
        "--order-seed", type=int, default = 42,
        help = "Seed for shuffled processing order before the early abort check (default: 42).",
    )
    parser.add_argument(
        "--staging-stale-hours", type=float, default = 72.0,
        help = "Remove abandoned staging directories older than this many hours (default: 72).",
    )
    parser.add_argument("--overwrite", action = "store_true")
    args = parser.parse_args(argv)

    root = Path(args.audio_dir).expanduser().resolve()
    cache_root = Path(args.cache_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Xeno-Canto audio directory not found: '{root}'.")

    if args.sample_rate <= 0 or args.clip_sec <= 0 or args.num_crops <= 0 or args.min_rms <= 0:
        parser.error("sample_rate, clip_sec, num_crops and min_rms must be positive")

    if not 0.0 <= args.max_excluded_fraction <= 1.0:
        parser.error("max-excluded-fraction must be between 0 and 1, inclusive")

    if args.early_abort_min_files < 1:
        parser.error("early-abort-min-files must be >= 1")
    if args.early_abort_check_interval < 1:
        parser.error("early-abort-check-interval must be >= 1")

    if not math.isfinite(args.staging_stale_hours) or args.staging_stale_hours <= 0:
        parser.error("staging-stale-hours must be finite and > 0")

    cache_root.parent.mkdir(parents = True, exist_ok = True)

    removed_staging = cleanup_stale_staging_dirs(cache_root, args.staging_stale_hours)
    if removed_staging:
        print(f"Cleaned {len(removed_staging)} stale staging director{'y' if len(removed_staging) == 1 else 'ies'}.", flush = True)

    indexed = indexed_audio(root)
    if not indexed:
        raise RuntimeError(f"No audio files found under '{root}'.")

    relative_paths = [rel for _, rel in indexed]
    inventory_hash = fingerprint_file_inventory(root, relative_paths)

    if manifest_path.exists() and not args.overwrite:
        existing = json.loads(manifest_path.read_text(encoding = "utf-8"))

        compatible = (
            existing.get("schema_version") == 3
            and existing.get("audio_inventory_sha256") == inventory_hash
            and int(existing.get("sample_rate", -1)) == args.sample_rate
            and abs(float(existing.get("duration_sec", -1)) - args.clip_sec) < 1e-8
            and int(existing.get("num_crops", -1)) == args.num_crops
            and abs(float(existing.get("min_rms", -1)) - args.min_rms) < 1e-12
        )

        if not compatible:
            raise RuntimeError(f"Existing preflight manifest '{manifest_path}' has different inputs; pass --overwrite deliberately.")

        missing_cache = [rel for rel, record in existing.get("records", {}).items() if not (cache_root / record["cache_path"]).is_file()]
        existing_total = int(existing.get("audio_file_count", len(indexed)))
        existing_excluded = int(existing.get("excluded_file_count", len(existing.get("invalid", {}))))
        existing_excluded_fraction = existing_excluded / max(existing_total, 1)

        if existing_excluded_fraction > args.max_excluded_fraction:
            raise RuntimeError(
                f"Existing preflight manifest excludes {existing_excluded}/{existing_total} files "
                f"({existing_excluded_fraction:.2%}), above --max-excluded-fraction={args.max_excluded_fraction:.2%}. "
                "Inspect the exclusion CSV and use --overwrite to rebuild after resolving the cause."
            )

        if not missing_cache:
            print(
                f"Preflight/cache already complete: total = {existing_total}, "
                f"valid = {len(existing['records'])}, excluded = {existing_excluded} "
                f"({existing_excluded_fraction:.2%}); preserving {manifest_path}."
            )
            return

        raise RuntimeError(f"Preflight manifest exists but {len(missing_cache)} cache files are missing. Back up outputs, then use --overwrite to rebuild.")

    # Keep the published manifest paired with the current cache while the replacement is built.
    # The new manifest is promoted below only after the staged cache has passed validation.
    # Do not unlink manifest_path here.
    manifest_path.parent.mkdir(parents = True, exist_ok = True)

    # Build into a sibling staging directory.
    # A failed or early aborted run discards the staging output without destroying a previously usable cache.
    stage_cache_root = Path(tempfile.mkdtemp(
        prefix = f".{cache_root.name}.staging-", dir = cache_root.parent
    ))

    stage_owner_marker = write_stage_owner_marker(stage_cache_root, args.order_seed)

    # Keep the inventory hash based on sorted paths, while processing in a seeded
    # shuffled order so the early abort threshold is not biased by directory order.
    processing_order = seeded_processing_order(indexed, args.order_seed)

    print(
        f"Preflight indexed {len(indexed)} files; processing with seed = {args.order_seed} "
        f"and early-abort threshold = {args.max_excluded_fraction:.2%}; statistical checks "
        f"start at {args.early_abort_min_files} files and repeat every "
        f"{args.early_abort_check_interval} files (95% Wilson lower bound).",
        flush = True,
    )

    target_samples = int(round(args.sample_rate * args.clip_sec))
    records: dict[str, dict[str, object]] = {}
    invalid: dict[str, str] = {}
    total_crops = 0

    for index, (path, relative) in enumerate(processing_order, 1):
        try:
            wave, source_rate = torchaudio.load(str(path))
        except Exception as exc:
            invalid[relative] = f"decode_error: {type(exc).__name__}: {exc}"

        else:
            try:
                if wave.ndim != 2 or wave.shape[-1] < 1 or wave.shape[0] < 1:
                    raise ValueError("empty/invalid waveform")

                wave = wave.to(torch.float32)
                if wave.shape[0] > 1:
                    wave = wave.mean(dim=0, keepdim=True)

                if int(source_rate) != args.sample_rate:
                    resampler = torchaudio.transforms.Resample(int(source_rate), args.sample_rate)
                    wave = resampler(wave)

                crops = make_crops(wave, target_samples, args.num_crops, args.min_rms)
                cache_rel = relative + ".pt"
                cache_path = stage_cache_root / cache_rel
                cache_path.parent.mkdir(parents = True, exist_ok = True)
                temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
                torch.save(crops.to(dtype = torch.float32), temp_path)
                os.replace(temp_path, cache_path)

                records[relative] = {
                    "cache_path": cache_rel,
                    "num_crops": int(crops.shape[0]),
                    "source_sample_rate": int(source_rate),
                    "source_samples": int(wave.shape[-1]),
                }

                total_crops += int(crops.shape[0])

            except Exception as exc:
                message = str(exc).lower()

                category = "crop_validation_error" if any(
                    token in message for token in ("silent", "crop", "rms", "waveform")
                ) else "processing_error"

                invalid[relative] = f"{category}: {type(exc).__name__}: {exc}"

        if index % 100 == 0 or index == len(indexed):
            print(f"[{index}/{len(indexed)}] valid = {len(records)} excluded = {len(invalid)} cached_crops = {total_crops}", flush = True)

        should_check_early = (
            index >= args.early_abort_min_files
            and (index - args.early_abort_min_files) % args.early_abort_check_interval == 0
            and index < len(indexed)
        )

        if should_check_early:
            running_fraction = len(invalid) / index
            lower_bound = wilson_lower_bound(len(invalid), index)

            if lower_bound > args.max_excluded_fraction:
                invalid_path = write_exclusion_audit(invalid, manifest_path)
                shutil.rmtree(stage_cache_root, ignore_errors = True)

                raise RuntimeError(
                    f"Early preflight abort after {index}/{len(indexed)} files: excluded "
                    f"{len(invalid)}/{index} ({running_fraction:.2%}); 95% Wilson lower bound "
                    f"is {lower_bound:.2%}, above --max-excluded-fraction="
                    f"{args.max_excluded_fraction:.2%} (processing order seed = {args.order_seed}). "
                    f"Inspect {invalid_path}. No successful manifest was written; "
                    "the previous cache directory, if present, was preserved."
                )

    try:
        final_indexed = indexed_audio(root)
        final_relative_paths = [rel for _, rel in final_indexed]
        final_inventory_hash = fingerprint_file_inventory(root, final_relative_paths)
    except:
        shutil.rmtree(stage_cache_root, ignore_errors = True)
        raise RuntimeError(
            "Could not re-verify the Xeno_Canto source inventory after preflight; "
            "the staged cache was discarded. Rerun preflight."
        ) from exc

    if final_inventory_hash != inventory_hash:
        shutil.rmtree(stage_cache_root, ignore_errors = True)
        raise RuntimeError(
            "Xeno-Canto source inventory changed during preflight; "
            " the staged cache was discarded. Rerun preflight."
        )

    excluded_fraction = len(invalid) / len(indexed)
    invalid_path = write_exclusion_audit(invalid, manifest_path)

    print(
        f"Preflight summary: total = {len(indexed)}, valid = {len(records)}, "
        f"excluded = {len(invalid)} ({excluded_fraction:.2%}), cached_crops = {total_crops}.",
        flush = True,
    )
    if excluded_fraction > args.max_excluded_fraction:
        shutil.rmtree(stage_cache_root, ignore_errors = True)

        raise RuntimeError(
            f"Preflight excluded {len(invalid)}/{len(indexed)} files ({excluded_fraction:.2%}), "
            f"exceeding --max-excluded-fraction = {args.max_excluded_fraction:.2%}. "
            f"Inspect {invalid_path} and confirm torchaudio codec support on this compute node. "
            "No successful preflight manifest was written; the previous cache directory, if present, was preserved."
        )

    if not records:
        shutil.rmtree(stage_cache_root, ignore_errors = True)
        raise RuntimeError("Preflight rejected every Xeno-Canto file; inspect the exclusion audit and RMS threshold.")

    if not invalid and invalid_path.exists():
        invalid_path.unlink()

    payload = {
        "schema_version": 3,
        "audio_root": str(root),
        "cache_root": str(cache_root),
        "audio_inventory_sha256": inventory_hash,
        "audio_file_count": len(indexed),
        "valid_file_count": len(records),
        "excluded_file_count": len(invalid),
        "sample_rate": int(args.sample_rate),
        "duration_sec": float(args.clip_sec),
        "target_samples": target_samples,
        "num_crops": int(args.num_crops),
        "min_rms": float(args.min_rms),
        "processing_order_seed": int(args.order_seed),
        "processing_order": "seeded shuffle of sorted relative path inventory",
        "staging_stale_hours": float(args.staging_stale_hours),
        "max_excluded_fraction": float(args.max_excluded_fraction),
        "early_abort_min_files": int(args.early_abort_min_files),
        "early_abort_check_interval": int(args.early_abort_check_interval),
        "early_abort_method": "95% Wilson lower confidence bound on running exclusion fraction",
        "excluded_fraction": float(excluded_fraction),
        "padding_policy": "repeat short recordings to target duration",
        "crop_policy": "deterministic evenly spaced candidates; non silent crops only",
        "records": records,
        "invalid": invalid,
    }
    temp_manifest = manifest_path.with_suffix(manifest_path.suffix + f".{os.getpid()}.tmp")
    temp_manifest.write_text(json.dumps(payload, indent = 2), encoding = "utf-8")
    backup_cache_root = cache_root.with_name(f".{cache_root.name}.backup-{os.getpid()}")
    moved_old_cache = False
    promoted_new_cache = False

    try:
        # Ownership metadata must not become part of the published dataset cache.
        stage_owner_marker.unlink(missing_ok = True)
        if backup_cache_root.exists():
            raise FileExistsError(
                f"Refusing to overwrite leftover cache backup '{backup_cache_root}'. Inspect it before rebuilding."
            )

        if cache_root.exists():
            os.replace(cache_root, backup_cache_root)
            moved_old_cache = True

        os.replace(stage_cache_root, cache_root)
        promoted_new_cache = True
        os.replace(temp_manifest, manifest_path)

    except Exception:
        if promoted_new_cache:
            shutil.rmtree(cache_root, ignore_errors = True)

        if moved_old_cache and backup_cache_root.exists() and not cache_root.exists():
            os.replace(backup_cache_root, cache_root)

        if temp_manifest.exists():
            temp_manifest.unlink()

        shutil.rmtree(stage_cache_root, ignore_errors = True)
        raise

    else:
        if moved_old_cache:
            shutil.rmtree(backup_cache_root, ignore_errors = True)

    print(f"Preflight passed: total = {len(indexed)}, valid = {len(records)}, excluded = {len(invalid)} ({excluded_fraction:.2%}), cached_crops = {total_crops}.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
