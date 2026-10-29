"""Evaluate trained BirdCLEF classifiers on time annotated Western Ghats soundscapes."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.metrics import average_precision_score

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.exp3_unseen_habitat_pair_stress_test import load_model_pair
from utils.audio_io import get_resampler, log_mel_spectrogram
from utils.seed import set_seed

_AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}


def _normalise_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def resolve_annotation_columns(columns) -> dict[str, str]:
    """Resolve common annotation CSV column-name variants to required fields."""
    normalised = {_normalise_column(col): str(col) for col in columns}
    aliases = {
        "filename": ("filename", "file_name", "audio_filename", "recording_filename", "audio_file", "file"),
        "start": ("start_time", "start_time_s", "start_seconds", "start_s", "start", "begin_time", "begin_s"),
        "end": ("end_time", "end_time_s", "end_seconds", "end_s", "end", "stop_time", "stop_s"),
        "species": ("species_code", "species_ebird_code", "ebird_code", "ebird_species_code", "e_bird_code", "species", "primary_label", "code"),
    }
    resolved = {}

    for field, candidates in aliases.items():
        match = next((normalised[name] for name in candidates if name in normalised), None)
        if match is None:
            raise ValueError(
                f"Annotation CSV is missing a recognizable '{field}' column. "
                f"Available columns: {list(columns)}; accepted names include {list(candidates)}."
            )

        resolved[field] = match

    return resolved


def load_annotation_intervals(annotation_csv: str | Path):
    frame = pd.read_csv(annotation_csv)
    columns = resolve_annotation_columns(frame.columns)

    frame = frame[[columns["filename"], columns["start"], columns["end"], columns["species"]]].copy()
    frame.columns = ["filename", "start", "end", "species"]
    frame["filename"] = frame["filename"].astype(str).str.replace("\\", "/", regex = False).str.strip()
    frame["species"] = frame["species"].astype(str).str.strip().str.lower()
    frame["start"] = pd.to_numeric(frame["start"], errors = "coerce")
    frame["end"] = pd.to_numeric(frame["end"], errors = "coerce")
    frame = frame.dropna(subset=["start", "end"])
    frame = frame[(frame["filename"] != "") & (frame["species"] != "") & (frame["end"] >= frame["start"])]

    if frame.empty:
        raise ValueError(f"No valid annotation intervals in '{annotation_csv}'.")
    return frame


def compute_multilabel_ap(targets: np.ndarray, scores: np.ndarray, class_names: list[str]) -> dict:
    targets = np.asarray(targets, dtype = np.uint8)
    scores = np.asarray(scores, dtype = np.float64)

    if targets.ndim != 2 or targets.shape != scores.shape:
        raise ValueError(f"Targets/scores shape mismatch: {targets.shape} vs {scores.shape}.")

    if targets.shape[0] < 1 or not np.isfinite(scores).all():
        raise ValueError("Metric inputs are empty or contain non finite scores.")

    class_rows = []
    for i, name in enumerate(class_names):
        y = targets[:, i]
        positives = int(y.sum())
        negatives = int(len(y) - positives)

        if positives == 0 or negatives == 0:
            continue

        class_rows.append({"species_code": name,
                           "average_precision": float(average_precision_score(y, scores[:, i])),
                           "positive_segments": positives,
                           "negative_segments": negatives})
    if not class_rows:
        raise ValueError("No class has both positive and negative annotated segments; AP is undefined.")

    valid_columns = [i for i in range(targets.shape[1]) if 0 < int(targets[:, i].sum()) < targets.shape[0]]
    if not valid_columns:
        raise ValueError("No class has both positive and negative annotated segments; micro AP is undefined.")

    return {
        "macro_average_precision": float(np.mean([row["average_precision"] for row in class_rows])),
        "micro_average_precision": float(average_precision_score(targets[:, valid_columns].reshape(-1), scores[:, valid_columns].reshape(-1))),
        "n_classes_scored": len(class_rows),
        "per_species_average_precision": class_rows,
        "metric_definition": "Average precision across mapped annotated positive segments plus deterministically sampled segments with no annotation interval of any species (all zero weak background targets). Macro and micro AP include only classes with at least one positive and one non positive target. Unannotated species in otherwise annotated segments and sampled background segments remain weak negatives, not definitive absence labels.",
    }


def resolve_audio_files(audio_root: str | Path, annotation_names: set[str]) -> tuple[dict[str, Path], list[str]]:
    root = Path(audio_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Audio directory does not exist: '{root}'. Extract soundscape_data.zip first.")

    by_basename: dict[str, list[Path]] = defaultdict(list)
    by_relative = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _AUDIO_SUFFIXES:
            by_basename[path.name].append(path)
            by_relative[path.relative_to(root).as_posix()] = path

    resolved: dict[str, Path] = {}
    missing = []
    for raw in sorted(annotation_names):
        normalised = raw.replace("\\", "/").lstrip("./")
        direct = root / normalised

        if direct.is_file():
            resolved[raw] = direct
        elif normalised in by_relative:
            resolved[raw] = by_relative[normalised]
        elif Path(normalised).name in by_basename:
            choices = by_basename[Path(normalised).name]

            if len(choices) != 1:
                raise ValueError(f"Ambiguous audio basename '{Path(normalised).name}' under '{root}'.")

            resolved[raw] = choices[0]
        else:
            missing.append(raw)

    return resolved, missing


def _load_full_mono(path: Path, target_sr: int) -> torch.Tensor:
    waveform, original_sr = torchaudio.load(str(path))
    if waveform.ndim != 2 or waveform.shape[0] < 1 or waveform.shape[-1] < 1:
        raise ValueError(f"Invalid/empty soundscape audio: '{path}'.")

    waveform = waveform.to(torch.float32)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim = 0, keepdim = True)
    if original_sr != target_sr:
        waveform = get_resampler(original_sr, target_sr)(waveform)

    if not torch.isfinite(waveform).all():
        raise ValueError(f"Non-finite waveform in '{path}'.")

    return waveform.squeeze(0).contiguous()


def build_all_annotation_segment_ids(intervals: pd.DataFrame, filename: str, clip_sec: float,
                                     recording_duration: float) -> set[int]:
    """Return segment IDs touched by any annotation, including unmapped species."""
    ids: set[int] = set()

    rows = intervals[intervals["filename"] == filename]
    for row in rows.itertuples(index=False):
        start, end = max(0.0, float(row.start)), max(0.0, float(row.end))

        if end < start:
            continue

        if end == start:
            candidates = [int(start // clip_sec)]
        else:
            first = int(start // clip_sec)
            last = max(first, int(np.nextafter(end, -np.inf) // clip_sec))
            candidates = range(first, last + 1)

        for segment_id in candidates:
            if segment_id * clip_sec < recording_duration:
                ids.add(int(segment_id))

    return ids


def calculate_annotation_coverage(all_annotation_codes: set[str], active_class_codes: set[str]) -> tuple[float, set[str], list[str]]:
    """Return the mapped code fraction and exact mapped/unmapped eBird codes."""

    all_codes = {str(code).strip().lower() for code in all_annotation_codes if str(code).strip()}
    active_codes = {str(code).strip().lower() for code in active_class_codes if str(code).strip()}

    mapped = all_codes & active_codes
    unmapped = sorted(all_codes - active_codes)

    return len(mapped) / max(len(all_codes), 1), mapped, unmapped


def sample_background_segment_ids(all_annotated_ids: set[int], n_total_segments: int,
                                  requested: int, *, seed: int, recording_name: str) -> list[int]:
    """Choose a deterministic sample of segments with no annotation of any species."""

    if n_total_segments < 0 or requested < 0:
        raise ValueError("n_total_segments and requested background segment count must be non negative.")

    if any(int(segment_id) < 0 or int(segment_id) >= n_total_segments for segment_id in all_annotated_ids):
        raise ValueError("Annotated segment IDs fall outside the recording segment grid.")

    candidates = [segment_id for segment_id in range(n_total_segments) if segment_id not in all_annotated_ids]

    take = min(int(requested), len(candidates))
    if take == 0:
        return []

    digest = hashlib.sha256(f"{int(seed)}:{recording_name}".encode()).digest()
    local_rng = np.random.default_rng(int.from_bytes(digest[:8], "little", signed = False))

    return sorted(int(value) for value in local_rng.choice(candidates, size = take, replace = False))


def repeat_pad_segment(clip: torch.Tensor, target_samples: int) -> torch.Tensor:
    """Repeat pad an incomplete final segment so all models see the same policy."""

    if target_samples < 1:
        raise ValueError("target_samples must be positive.")

    if clip.numel() < 1:
        raise ValueError("Cannot repeat pad an empty segment; audio duration/grid alignment is inconsistent.")

    if clip.numel() < target_samples:
        repeats = (target_samples + clip.numel() - 1) // clip.numel()
        clip = clip.repeat(repeats)[:target_samples]
    else:
        clip = clip[:target_samples]

    return clip.contiguous()

def build_segment_targets(intervals: pd.DataFrame, filename: str, duration_sec: float, num_classes: int,
                          code_to_index: dict[str, int], recording_duration: float) -> dict[int, np.ndarray]:
    targets: dict[int, np.ndarray] = {}

    rows = intervals[intervals["filename"] == filename]
    for row in rows.itertuples(index = False):
        code = str(row.species).strip().lower()
        cls = code_to_index.get(code)

        if cls is None:
            continue

        start, end = float(row.start), float(row.end)
        if end == start:
            segment_ids = [int(max(0, start) // duration_sec)]
        else:
            first = max(0, int(start // duration_sec))
            last = max(first, int(np.nextafter(end, -np.inf) // duration_sec))
            segment_ids = range(first, last + 1)

        for segment_id in segment_ids:
            seg_start = segment_id * duration_sec
            if seg_start >= recording_duration or seg_start + duration_sec <= start or end < start:
                continue

            target = targets.setdefault(int(segment_id), np.zeros(num_classes, dtype = np.uint8))
            target[cls] = 1

    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--annotation-csv", default = "data/raw/western_ghats_soundscapes/annotation.csv")
    parser.add_argument("--audio-dir", default = "data/raw/western_ghats_soundscapes/soundscape_data")
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--fraction", type=float, default = 0.1)
    parser.add_argument("--sample-rate", type=int, default = 16000)
    parser.add_argument("--clip-sec", type=float, default = 10.0)
    parser.add_argument("--batch-size", type=int, default = 16)
    parser.add_argument("--background-segments-per-recording", type=int, default = 10,
                        help = "Deterministically sample this many unannotated segments per recording as weak background negatives; 0 disables them.")
    parser.add_argument("--background-seed", type=int, default = 4242,
                        help="Independent fixed seed for background segment sampling; leave unchanged across model seeds to keep evaluated segments paired.")
    parser.add_argument("--min-annotation-coverage", type=float, default = 0.50,
                        help = "Minimum fraction of distinct annotation eBird codes that must map to active BirdCLEF classes.")
    parser.add_argument("--max-recordings", type=int, default = None, help = "Smoke-test only; capped runs are not paper results.")
    parser.add_argument("--allow-missing-recordings", action = "store_true", help = "Allow an incomplete audio archive for exploratory diagnostics only; result aggregation will reject it.")
    parser.add_argument("--include-external", action = "store_true", help = "Also score PANNs/BirdNET heads trained by exp4_pretrained_feature_baselines.py.")
    parser.add_argument("--external-models", nargs="+", choices=["panns", "birdnet"], default=["panns", "birdnet"])
    parser.add_argument("--panns-checkpoint", default = None, help = "Required local Cnn14 checkpoint when evaluating a PANNs head; no automatic download.")
    parser.add_argument("--birdnet-device", choices = ["CPU", "GPU:0"], default = "CPU", help = "BirdNET ONNX device; GPU requires a compatible GPU enabled ONNX Runtime.")
    parser.add_argument("--output-prefix", default = None)
    parser.add_argument("--variants", nargs="+", choices=["habitat", "standard"],
                        default=["habitat", "standard"])
    parser.add_argument("--bn-policy",
                        choices=["copy_query", "ema_query", "frozen_key", "independent_key"],
                        default="independent_key")
    parser.add_argument("--view-batch-policy",
                        choices=["separate", "concatenated"],
                        default="concatenated")

    args = parser.parse_args(argv)
    if not 0 < args.fraction <= 1 or args.sample_rate < 1 or args.clip_sec <= 0 or args.batch_size < 1:
        raise ValueError("fraction, sample rate, clip duration or batch size is invalid.")

    if args.background_segments_per_recording < 0:
        raise ValueError("--background-segments-per-recording must be non negative.")

    if not 0.0 <= args.min_annotation_coverage <= 1.0:
        raise ValueError("--min-annotation-coverage must be between 0 and 1.")

    if args.include_external and "panns" in args.external_models:
        if not args.panns_checkpoint:
            raise ValueError("--panns-checkpoint is required when PANNs is included; use a pre downloaded local file.")

        if not Path(args.panns_checkpoint).is_file():
            raise FileNotFoundError(f"PANNs checkpoint does not exist: '{args.panns_checkpoint}'.")

    if args.max_recordings is not None and args.max_recordings < 1:
        raise ValueError("--max-recordings must be positive when supplied.")

    set_seed(args.seed)
    annotation_path = Path(args.annotation_csv)
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Annotation CSV missing: '{annotation_path}'. Download annotation.csv from "
            "https://zenodo.org/records/16881041 and extract soundscape_data.zip."
        )
    intervals = load_annotation_intervals(annotation_path)

    split_path = Path("checkpoints") / f"data_split_s{args.seed}.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"BirdCLEF split manifest '{split_path}' is missing. Run split creation first.")

    split = json.loads(split_path.read_text(encoding = "utf-8"))
    if int(split.get("schema_version", -1)) != 5:
        raise ValueError(f"BirdCLEF split '{split_path}' uses an older class-scope/preflight schema.")

    active_map = split.get("active_label_map", split.get("label_map", {}))
    full_label_map = split.get("label_map", active_map)
    excluded_classes = split.get("excluded_classes", [])
    ordered_classes = [name for name, _ in sorted(active_map.items(), key = lambda item: int(item[1]))]

    code_to_index = {str(code).strip().lower(): i for i, code in enumerate(ordered_classes)}
    if not code_to_index:
        raise ValueError("Split manifest has no active classes.")

    audio_map, missing_recordings = resolve_audio_files(args.audio_dir, set(intervals["filename"].unique()))
    if missing_recordings and not args.allow_missing_recordings:
        raise FileNotFoundError(
            f"The soundscape archive is incomplete: {len(missing_recordings)} annotated recordings are missing under '{args.audio_dir}'. "
            f"Examples: {missing_recordings[:10]}. Re-extract the complete archive or use --allow-missing-recordings for exploratory diagnostics only."
        )

    all_annotation_codes = set(intervals["species"].astype(str).str.lower())
    annotation_coverage, mapped_codes, unmapped_codes = calculate_annotation_coverage(
        all_annotation_codes, set(code_to_index)
    )

    if annotation_coverage < args.min_annotation_coverage:
        raise ValueError(
            f"Annotation-code coverage {annotation_coverage:.1%} ({len(mapped_codes)}/{len(all_annotation_codes)}) "
            f"is below --min-annotation-coverage = {args.min_annotation_coverage:.1%}. Inspect the class scope/eBird mapping; "
            "lower the gate only with a documented reason."
        )

    candidate_names = sorted(set(audio_map))
    if args.max_recordings is not None:
        candidate_names = candidate_names[:args.max_recordings]

    if not candidate_names:
        raise ValueError("No annotated audio recordings resolved under --audio-dir.")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    variants = args.variants
    expected_output_map = {name: i for i, name in enumerate(ordered_classes)}
    models = {
        v: load_model_pair(
            v, args.seed, args.fraction, len(ordered_classes), device,
            args.bn_policy, args.view_batch_policy
        )
        for v in variants
    }

    for variant, (backbone, head) in models.items():
        head_path = Path("checkpoints") / f"linear_head_{variant}_s{args.seed}_frac{float(args.fraction)}.pt"
        head_payload = torch.load(head_path, map_location="cpu", weights_only = True)

        if head_payload.get("label_map") != expected_output_map:
            raise ValueError(f"Core head '{head_path}' uses a different active class ordering from the split manifest.")

        backbone.eval(); head.eval()

    external_models = {}
    if args.include_external:
        from experiments.exp4_pretrained_feature_baselines import build_encoder

        expected_output_map = {name: i for i, name in enumerate(ordered_classes)}
        for external_key in args.external_models:
            head_path = Path("checkpoints") / f"pretrained_head_{external_key}_s{args.seed}_frac{float(args.fraction)}.pt"

            if not head_path.is_file():
                raise FileNotFoundError(f"External head missing: '{head_path}'. Run exp4_pretrained_feature_baselines.py --model {external_key} --seed {args.seed} --fraction {args.fraction} first.")

            saved = torch.load(head_path, map_location = "cpu", weights_only = True)
            if saved.get("label_map") != expected_output_map:
                raise ValueError(f"External head '{head_path}' uses a different active class mapping from the real soundscape evaluation.")

            encoder = build_encoder(
                external_key, device,
                args.panns_checkpoint if external_key == "panns" else None,
                birdnet_device = args.birdnet_device,
            )

            if saved.get("external_checkpoint_sha256") != encoder.checkpoint_sha256:
                raise ValueError(
                    f"External head '{head_path}' was trained with a different/missing model file SHA-256. "
                    "Re-train the head with the currently pinned external model artifact."
                )

            if saved.get("encoder_identity") != encoder.identity:
                raise ValueError(
                    f"External head '{head_path}' encoder identity does not match the currently loaded model/library."
                )

            ext_head = torch.nn.Linear(int(saved["feature_dim"]), len(ordered_classes)).to(device)
            ext_head.load_state_dict(saved["model_state"], strict = True)
            ext_head.eval()
            external_models[external_key] = (encoder, ext_head)

    target_batches: list[np.ndarray] = []
    file_rows: list[str] = []
    start_rows: list[float] = []
    segment_type_rows: list[str] = []
    annotated_segment_count = 0
    background_segment_count = 0
    score_rows = {variant: [] for variant in variants}
    score_rows.update({variant: [] for variant in external_models})
    target_samples = int(round(args.sample_rate * args.clip_sec))

    with torch.inference_mode():
        for n_recording, annotation_name in enumerate(candidate_names, start = 1):
            audio_path = audio_map[annotation_name]
            waveform = _load_full_mono(audio_path, args.sample_rate)
            n_samples = waveform.numel()
            duration = n_samples / float(args.sample_rate)

            external_rates = sorted({
                int(getattr(encoder, "input_sample_rate", args.sample_rate))
                for encoder, _head in external_models.values()
            })

            external_waveforms = {
                rate: waveform if rate == args.sample_rate else _load_full_mono(audio_path, rate)
                for rate in external_rates
            }

            segment_targets = build_segment_targets(intervals, annotation_name, args.clip_sec, len(ordered_classes), code_to_index, duration)
            mapped_annotated_ids = {int(k) for k, y in segment_targets.items() if y.sum() > 0}
            all_annotated_ids = build_all_annotation_segment_ids(intervals, annotation_name, args.clip_sec, duration)
            n_total_segments = int(np.ceil(duration / args.clip_sec))

            sampled_background_ids = sample_background_segment_ids(
                all_annotated_ids,
                n_total_segments,
                args.background_segments_per_recording,
                seed = args.background_seed,
                recording_name = annotation_name,
            )

            selected_ids = sorted(mapped_annotated_ids | set(sampled_background_ids))
            if not selected_ids:
                continue

            for start in range(0, len(selected_ids), args.batch_size):
                segment_ids = selected_ids[start:start + args.batch_size]
                waves = []
                ys = []

                current_segment_types = []
                for segment_id in segment_ids:
                    lo = segment_id * target_samples
                    clip = waveform[lo:lo + target_samples]
                    clip = repeat_pad_segment(clip, target_samples)
                    waves.append(clip.unsqueeze(0))
                    ys.append(segment_targets.get(segment_id, np.zeros(len(ordered_classes), dtype = np.uint8)))
                    current_segment_types.append("annotated" if segment_id in mapped_annotated_ids else "background")

                batch_wave = torch.stack(waves, dim = 0)

                mel = log_mel_spectrogram(batch_wave, sample_rate = args.sample_rate).to(device)
                for variant, (backbone, head) in models.items():
                    logits = head(backbone(mel))

                    if not torch.isfinite(logits).all():
                        raise FloatingPointError(f"Non finite logits from '{variant}' on {annotation_name}.")

                    score_rows[variant].append(logits.detach().cpu().numpy().astype(np.float32))

                external_batches_by_rate = {}

                for rate in external_rates:
                    rate_waveform = external_waveforms[rate]
                    rate_samples = int(round(rate * args.clip_sec))
                    ext_waves = []

                    for segment_id in segment_ids:
                        lo = segment_id * rate_samples
                        clip = rate_waveform[lo:lo + rate_samples]
                        clip = repeat_pad_segment(clip, rate_samples)
                        ext_waves.append(clip.unsqueeze(0))

                    external_batches_by_rate[rate] = torch.stack(ext_waves, dim = 0)

                for variant, (encoder, head) in external_models.items():
                    encoder_sr = int(getattr(encoder, "input_sample_rate", args.sample_rate))
                    features = encoder.encode_batch(external_batches_by_rate[encoder_sr], encoder_sr)
                    feature_tensor = torch.from_numpy(features).to(device)
                    logits = head(feature_tensor)

                    if not torch.isfinite(logits).all():
                        raise FloatingPointError(f"Non finite logits from external model '{variant}' on {annotation_name}.")

                    score_rows[variant].append(logits.detach().cpu().numpy().astype(np.float32))

                target_batches.extend(ys)
                file_rows.extend([annotation_name] * len(segment_ids))
                start_rows.extend([float(s * args.clip_sec) for s in segment_ids])
                segment_type_rows.extend(current_segment_types)
                annotated_segment_count += sum(kind == "annotated" for kind in current_segment_types)
                background_segment_count += sum(kind == "background" for kind in current_segment_types)

            if n_recording % 25 == 0 or n_recording == len(candidate_names):
                print(
                    f"Soundscapes processed: {n_recording}/{len(candidate_names)}; "
                    f"mapped annotated segments = {annotated_segment_count}; sampled background segments = {background_segment_count}",
                    flush = True,
                )

    if not target_batches:
        raise ValueError(
            "No segments with annotations matching the active BirdCLEF class scope were found. "
            "Check species code compatibility and the active label manifest."
        )

    targets = np.stack(target_batches).astype(np.uint8)
    all_scores = {name: np.concatenate(parts, axis = 0) for name, parts in score_rows.items()}
    background_mask = np.asarray(segment_type_rows, dtype = str) == "background"
    metrics = {}

    for name, scores in all_scores.items():
        model_metrics = compute_multilabel_ap(targets, scores, ordered_classes)

        if background_mask.any():
            shifted = scores[background_mask].astype(np.float64)
            shifted -= shifted.max(axis = 1, keepdims = True)

            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis = 1, keepdims = True).clip(min = 1e-12)

            top1_confidence = probabilities.max(axis = 1)
            background_metrics = {
                "background_segment_count": int(background_mask.sum()),
                "background_top1_confidence_mean": float(top1_confidence.mean()),
                "background_top1_confidence_ge_0_5_rate": float((top1_confidence >= 0.5).mean()),
                "background_metric_note": "Weak-negative proxy: fraction of sampled unannotated segments whose multiclass softmax top-1 confidence is >=0.5; unannotated bird activity may still be present.",
            }

        else:
            background_metrics = {
                "background_segment_count": 0,
                "background_top1_confidence_mean": None,
                "background_top1_confidence_ge_0_5_rate": None,
                "background_metric_note": "No sampled background segments; pass --background-segments-per-recording > 0.",
            }

        metrics[name] = {
            **model_metrics,
            **background_metrics,
            "n_segments": int(len(targets)),
            "n_recordings": int(len(set(file_rows))),
        }

    config_identity = {
        "annotation_csv_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        "recording_names_sha256": hashlib.sha256("\n".join(sorted(candidate_names)).encode()).hexdigest(),
        "seed": args.seed,
        "fraction": args.fraction,
        "sample_rate": args.sample_rate,
        "core_variants": variants,
        "bn_policy": args.bn_policy,
        "view_batch_policy": args.view_batch_policy,
        "clip_sec": args.clip_sec,
        "max_recordings": args.max_recordings,
        "background_segments_per_recording": args.background_segments_per_recording,
        "background_seed": args.background_seed,
        "min_annotation_coverage": args.min_annotation_coverage,
        "include_external": args.include_external,
        "external_models": sorted(external_models),
        "external_encoder_identities": {name: encoder.identity for name, (encoder, _head) in external_models.items()},
        "dataset": "Western Ghats fully annotated soundscapes, Zenodo 10.5281/zenodo.16881041",
        "active_class_map": active_map,
    }

    identity = hashlib.sha256(json.dumps(config_identity, sort_keys = True).encode()).hexdigest()[:12]

    output_prefix = args.output_prefix or f"results/real_soundscape_s{args.seed}_frac{float(args.fraction)}_{identity}"
    output_path = Path(output_prefix)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    prediction_path = output_path.with_suffix(".npz")
    temp_prediction_path = prediction_path.with_suffix(prediction_path.suffix + ".tmp")

    payload = {
        "targets": targets,
        "file_names": np.asarray(file_rows, dtype = str),
        "segment_start_seconds": np.asarray(start_rows, dtype = np.float32),
        "segment_types": np.asarray(segment_type_rows, dtype = str),
        "class_names": np.asarray(ordered_classes, dtype = str),
    }

    payload.update({f"scores_{name}": score for name, score in all_scores.items()})

    with temp_prediction_path.open("wb") as stream:
        np.savez_compressed(stream, **payload)

    temp_prediction_path.replace(prediction_path)

    summary = {
        "schema_version": 2,
        "dataset": config_identity["dataset"],
        "dataset_license": "CC BY 4.0; cite the Zenodo dataset record and original paper/documentation.",
        "seed": args.seed,
        "fraction": args.fraction,
        "n_candidate_recordings": len(candidate_names),
        "n_recordings_evaluated": len(set(file_rows)),
        "n_segments_scored": int(len(targets)),
        "n_annotated_segments": int(annotated_segment_count),
        "n_background_segments": int(background_segment_count),
        "background_segments_per_recording": args.background_segments_per_recording,
        "n_unique_annotated_species_codes": len(all_annotation_codes),
        "n_species_codes_matched_to_active_classes": len(mapped_codes),
        "annotation_code_coverage_fraction": float(annotation_coverage),
        "minimum_annotation_coverage_required": float(args.min_annotation_coverage),
        "matched_annotation_codes": sorted(mapped_codes),
        "unmapped_species_codes": unmapped_codes,
        "n_annotation_recordings_missing_audio": len(missing_recordings),
        "missing_recording_names_preview": missing_recordings[:20],
        "incomplete_audio_inventory": bool(missing_recordings),
        "capped_smoke_run": args.max_recordings is not None,
        "active_label_map": active_map,
        "full_label_map": full_label_map,
        "n_active_classes": len(active_map),
        "n_total_classes": len(full_label_map),
        "n_excluded_classes": len(excluded_classes),
        "excluded_classes": excluded_classes,
        "n_silent_files_excluded": int(split.get("n_silent_files_excluded", len(split.get("silent_excluded_filenames", [])))),
        "silent_excluded_filenames": split.get("silent_excluded_filenames", []),
        "n_preflight_excluded_files": int(split.get("n_preflight_excluded_files", 0)),
        "preflight_excluded_filenames": split.get("preflight_excluded_filenames", []),
        "n_preflight_read_error_files_excluded": int(split.get("n_preflight_read_error_files_excluded", 0)),
        "preflight_read_error_filenames": split.get("preflight_read_error_filenames", []),
        "preflight_reason_histogram": split.get("preflight_reason_histogram", {}),
        "preflight_read_error_fraction": float(split.get("preflight_read_error_fraction", 0.0)),
        "preflight_excluded_fraction": float(split.get("preflight_excluded_fraction", 0.0)),
        "preflight_max_read_error_fraction": float(split.get("preflight_max_read_error_fraction", 0.01)),
        "preflight_max_excluded_fraction": float(split.get("preflight_max_excluded_fraction", 0.05)),
        "rare_class_policy": split.get("rare_class_policy", "unknown"),
        "class_scope_note": split.get("class_scope_note", "Metrics cover active_label_map only; excluded species are absent from train/validation/test."),
        "annotation_columns": {key: val for key, val in resolve_annotation_columns(pd.read_csv(annotation_path, nrows=0).columns).items()},

        "target_definition": f"Mapped species are positive in each fixed non-overlapping {args.clip_sec:g}-s segment overlapping an expert annotation. In addition, up to {args.background_segments_per_recording} deterministically sampled segments with no annotation interval of any species are scored as all zero weak-background targets per recording.",
        "background_segment_sampling": "Deterministic without-replacement sampling from segments with no annotation interval of any species; samples are keyed by the independent fixed background seed and recording name, not the model seed.",
        "padding_policy": "Repeat pad the final partial segment to the configured clip length for both core and external models; do not zero pad.",
        "metric_caveat": "Annotations are high confidence detections and may not be exhaustive. Sampled unannotated segments are weak negatives and may still contain unannotated bird activity; the background top-1 confidence rate is therefore a false alarm proxy, not a definitive absence based false alarm rate. Portions of this collection were used as the hidden BirdCLEF 2024 test set, so do not describe it as independent of the historical BirdCLEF benchmark.",

        "models": metrics, "prediction_archive": str(prediction_path), "identity": config_identity,
    }

    result_path = output_path.with_suffix(".json")
    temp_result_path = result_path.with_suffix(result_path.suffix + ".tmp")
    temp_result_path.write_text(json.dumps(summary, indent = 2), encoding = "utf-8")
    temp_result_path.replace(result_path)

    print(json.dumps({"summary": str(result_path),
                      "predictions": str(prediction_path),
                      "models": {k: {"macro_AP": v["macro_average_precision"],
                                     "micro_AP": v["micro_average_precision"]} for k, v in metrics.items()}}, indent = 2))


if __name__ == "__main__":
    main()