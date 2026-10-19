"""Run a small real audio embedding smoke test without fitting a probe or writing paper results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bioacoustic_data import BirdCLEFDataset
from experiments.exp4_pretrained_feature_baselines import build_encoder, FEATURE_EXTRACTION_SPEC_VERSION
from utils.audio_io import DEFAULT_EVALUATION_MINIMUM_RMS, EVALUATION_CROP_POLICY, load_evaluation_center_crop


def count_distinct_feature_rows(matrix: np.ndarray, *, rtol: float = 1e-6, atol: float = 1e-8) -> int:
    """Count approximate unique feature rows, rejecting malformed matrices."""

    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"Expected a non-empty [files,features] matrix; got {values.shape}.")
    
    if not np.isfinite(values).all():
        raise FloatingPointError("Feature diversity check received non-finite embeddings.")
    
    distinct_rows = []
    for row in values:
        if not any(np.allclose(row, prior, rtol=rtol, atol=atol) for prior in distinct_rows):
            distinct_rows.append(row)

    return len(distinct_rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--model", choices = ["panns", "birdnet"], required = True)
    parser.add_argument("--config", default = "configs/linear_probe.yaml")
    parser.add_argument("--checkpoint", default = None, help = "Required local PANNs .pth checkpoint; not used by BirdNET.")
    parser.add_argument("--birdnet-device", choices = ["CPU", "GPU:0"], default = "CPU")
    parser.add_argument("--num-files", type=int, default = 20)
    parser.add_argument("--batch-size", type=int, default = 4)
    parser.add_argument("--minimum-rms", type=float, default = DEFAULT_EVALUATION_MINIMUM_RMS,
                        help = "Use the same center-crop recovery threshold as probes/stress evaluation.")
    parser.add_argument("--output", default = None, help = "Optional JSON output path; filename list and model identity will be recorded.")

    args = parser.parse_args(argv)
    if args.num_files < 1 or args.batch_size < 1 or not np.isfinite(args.minimum_rms) or args.minimum_rms <= 0:
        raise ValueError("--num-files and --batch-size must be positive and --minimum-rms finite/positive.")
    if args.model == "panns" and not args.checkpoint:
        raise ValueError("--checkpoint is required for PANNs; download the checkpoint before running this smoke test.")
    if args.model == "birdnet" and args.checkpoint:
        raise ValueError("--checkpoint is PANNs-only; BirdNET uses the version pinned official model cache.")

    config = yaml.safe_load(Path(args.config).read_text(encoding = "utf-8"))
    data_cfg = config["dataset"]

    dataset = BirdCLEFDataset(
        data_cfg["birdclef_dir"], 
        sample_rate = int(data_cfg["sample_rate"]),
        duration_sec = float(data_cfg["clip_sec"]), 
        crop_mode = "center",
    )

    if len(dataset) < args.num_files:
        raise ValueError(f"Requested {args.num_files} files, but the dataset contains only {len(dataset)}.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_encoder(args.model, device, args.checkpoint, birdnet_device = args.birdnet_device)
    paths = []
    waves_to_encode = []
    skipped_silent = 0
    attempted_files = 0

    for index, path in enumerate(dataset.file_list):
        if len(paths) >= args.num_files:
            break
        attempted_files += 1

        try:
            wave = load_evaluation_center_crop(
                path, 
                target_sr = encoder.input_sample_rate,
                duration_sec = dataset.duration_sec, 
                minimum_rms = args.minimum_rms,
            )
        except ValueError as exc:
            if "SILENT_CROP_REJECTED" not in str(exc):
                raise      
            skipped_silent += 1
            print(f"[{args.model}] SILENT_CROP_SKIPPED path = {dataset.relative_paths[index]}", flush = True)
            continue

        waves_to_encode.append(wave)
        paths.append(dataset.relative_paths[index])

    if len(paths) < args.num_files:
        raise ValueError(
            f"Only {len(paths)} non silent files found after inspecting {attempted_files}/{len(dataset)} recordings; "
            f"the smoke test requested {args.num_files}."
        )
    
    embeddings = []
    for start in range(0, args.num_files, args.batch_size):
        stop = min(start + args.batch_size, args.num_files)
        waves = waves_to_encode[start:stop]
        features = encoder.encode_batch(torch.stack(waves, dim = 0), encoder.input_sample_rate)

        if features.ndim != 2 or features.shape[0] != len(waves) or not np.isfinite(features).all():
            raise RuntimeError(f"Invalid smoke embeddings from {args.model}: shape = {features.shape}.")
        
        embeddings.append(features)
        print(f"[{args.model}] encoded {stop}/{args.num_files}; embedding_shape = {features.shape}", flush = True)

    matrix = np.concatenate(embeddings, axis = 0)

    n_distinct_rows = count_distinct_feature_rows(matrix)
    if n_distinct_rows < 2:
        raise RuntimeError(
            f"{args.model} smoke test returned identical features for all {len(matrix)} files; "
            "inspect masking, crop selection, weights, and cache invalidation."
        )
    
    result = {
        "smoke_only": True, 
        "paper_result": False, 
        "model": args.model,
        "num_files_requested": args.num_files, 
        "num_files_encoded": len(paths),
        "num_files_inspected": attempted_files, 
        "silent_files_skipped": skipped_silent,
        "files": paths, 
        "embedding_shape": list(matrix.shape),
        "all_embeddings_finite": bool(np.isfinite(matrix).all()),
        "distinct_feature_rows": n_distinct_rows,
        "feature_diversity_check_passed": True,
        "feature_extraction_spec_version": FEATURE_EXTRACTION_SPEC_VERSION,
        "crop_policy": EVALUATION_CROP_POLICY,
        "minimum_rms": float(args.minimum_rms),
        "encoder_identity": encoder.identity,
    }

    text = json.dumps(result, indent = 2)
    print(text)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents = True, exist_ok = True)

        temp = output.with_suffix(output.suffix + ".tmp")
        temp.write_text(text + "\n", encoding = "utf-8")
        temp.replace(output)

if __name__ == "__main__":
    main()