"""Evaluate saved models on the same fixed test recordings under seen/unseen habitat noise."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bioacoustic_data import BirdCLEFDataset, NoiseMixer
from models.backbone_cnn10 import CNN_10_Backbone
from models.linear_head import Linear_Head
from utils.audio_io import DEFAULT_EVALUATION_MINIMUM_RMS, EVALUATION_CROP_POLICY, load_and_preprocess_audio
from utils.dataset_manifest import validate_manifest_file_index
from utils.habitat_split import read_verified_habitat_csv
from utils.logger import get_logger
from utils.seed import set_seed
from utils.stress_config import load_stress_snr_config
from utils.label_scope import ActiveClassDataset


def parse_args(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--config", type=str, default = "configs/stress_test.yaml")
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--fraction", type=float, default = 1.0)
    parser.add_argument("--variants", nargs="+", choices=["habitat", "standard"],
                        default=["habitat", "standard"])
    parser.add_argument("--bn-policy", choices=["copy_query", "ema_query", "frozen_key", "independent_key"],
                        default="independent_key")
    parser.add_argument("--view-batch-policy", choices=["separate", "concatenated"],
                        default="concatenated")
    parser.add_argument("--output", type=str, required=True,
                        help="New output JSON path; existing files are never overwritten.")
    return parser.parse_args(argv)


def load_noise_pool_from_csv(split_csv_path, esc50_audio_dir, sample_rate, duration_sec, *, expected_partition: str):
    split_path = Path(split_csv_path)
    
    if not split_path.is_file():
        raise FileNotFoundError(f"Habitat split CSV missing: '{split_path}'. Run scripts/data/create_habitat_split.py first.")

    df = read_verified_habitat_csv(
        split_path, esc50_audio_dir, expected_partition=expected_partition
    )

    if df.empty:
        raise ValueError(f"Habitat split CSV '{split_path}' has no filenames.")
    root = Path(esc50_audio_dir)
    
    pool = []

    for filename in sorted(df["filename"].astype(str).tolist()):
        candidates = (root / filename, root / "audio" / filename)
        path = next((p for p in candidates if p.is_file()), None)

        if path is None:
            raise FileNotFoundError(f"Noise sample '{filename}' not found under '{root}'.")

        pool.append(load_and_preprocess_audio(
            path, target_sr = sample_rate, duration_sec = duration_sec,
            crop_mode = "center", short_audio_policy = "repeat",
        ))

    if not pool:
        raise RuntimeError(f"No noise data loaded from '{split_path}'.")
    
    return pool

def fraction_tag(fraction: float) -> str:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1].")
    
    return str(float(fraction))


def load_model_pair(variant, seed, fraction, num_classes, device, bn_policy, view_batch_policy):
    backbone = CNN_10_Backbone().to(device)
    head = Linear_Head(in_features = 512, num_classes = num_classes).to(device)
    tag = fraction_tag(fraction)
    head_path = f"checkpoints/linear_head_{variant}_s{seed}_frac{tag}.pt"

    if not os.path.isfile(head_path):
        raise FileNotFoundError(f"Classifier checkpoint missing: '{head_path}'. Run Exp2 for this variant, seed and fraction.")
    
    head_payload = torch.load(head_path, map_location = "cpu", weights_only = True)

    if head_payload.get("label_map") is not None and len(head_payload["label_map"]) != num_classes:
        raise ValueError(f"Class count in '{head_path}' differs from the current dataset.")
    
    head.load_state_dict(head_payload.get("model_state", head_payload), strict = True)

    if variant in {"supervised", "supervised_noise"}:
        backbone_name = "supervised_noise_baseline" if variant == "supervised_noise" else "supervised_baseline"
        bb_path = f"checkpoints/{backbone_name}_s{seed}_frac{tag}.pt"

        if not os.path.isfile(bb_path):
            raise FileNotFoundError(f"Supervised backbone checkpoint missing: '{bb_path}'.")

        bb_payload = torch.load(bb_path, map_location = "cpu", weights_only = True)
        backbone.load_state_dict(bb_payload.get("model_state", bb_payload), strict = True)

    else:
        moco_path = f"checkpoints/moco_{variant}_s{seed}_bn-{bn_policy}_views-{view_batch_policy}.pt"

        if not os.path.isfile(moco_path):
            raise FileNotFoundError(f"MoCo checkpoint missing: '{moco_path}'. Run Exp1 first.")
        
        moco_payload = torch.load(moco_path, map_location = "cpu", weights_only = True)
        state = moco_payload.get("model_state", {})
        prefix = "encoder_q.0."
        extracted = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}

        if not extracted:
            raise KeyError(f"No backbone state with prefix '{prefix}' in '{moco_path}'.")
        
        backbone.load_state_dict(extracted, strict = True)

    backbone.eval()
    head.eval()
    return backbone, head

def evaluate_all_models_on_loader(models, dataloader, device, group_id_by_file=None):
    """Evaluate models on shared batches and preserve stable group IDs for cluster bootstrap."""

    state = {
        variant: {"loss_sum": 0.0, 
                  "correct": 0, 
                  "n": 0, 
                  "predictions": [],
                  "silent_mix_skipped_files": []}
        for variant in models
    }

    criterion = torch.nn.CrossEntropyLoss(reduction = "sum")
    
    with torch.inference_mode():
        for batch in dataloader:
            if len(batch) == 4:
                x, y, filenames, mix_skipped = batch
                skipped_files = [str(name) for name, skipped in zip(filenames, mix_skipped) if bool(skipped)]
            elif len(batch) == 3:
                x, y, filenames = batch
                skipped_files = []
            else:
                raise ValueError(f"Unexpected Exp3 evaluation batch with {len(batch)} fields.")
            
            x = x.to(device, non_blocking = device.type == "cuda")
            y_device = y.to(device, non_blocking = device.type == "cuda")

            for variant, (backbone, head) in models.items():
                logits = head(backbone(x))

                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f"Model '{variant}' produced non finite logits.")
                
                preds = logits.argmax(dim = 1)
                item = state[variant]
                item["silent_mix_skipped_files"].extend(skipped_files)
                item["loss_sum"] += float(criterion(logits, y_device).item())
                item["correct"] += int((preds == y_device).sum().item())
                item["n"] += int(y_device.size(0))

                for filename, actual, guessed in zip(filenames, y.tolist(), preds.cpu().tolist()):
                    row = {
                        "file": str(filename), 
                        "y_true": int(actual), 
                        "y_pred": int(guessed),
                        "correct": bool(int(actual) == int(guessed)),
                    }

                    if group_id_by_file is not None:
                        group_id = group_id_by_file.get(str(filename))
                        if group_id is not None:
                            row["group_id"] = str(group_id)

                    item["predictions"].append(row)
    output = {}
    for variant, item in state.items():
        if item["n"] == 0:
            raise RuntimeError(f"No test examples evaluated for model '{variant}'.")
        
        output[variant] = {
            "accuracy": item["correct"] / item["n"],
            "loss": item["loss_sum"] / item["n"],
            "n": item["n"],
            "predictions": item["predictions"],
            "n_silent_mix_skipped": len(set(item["silent_mix_skipped_files"])),
            "silent_mix_skipped_files": sorted(set(item["silent_mix_skipped_files"])),
        }

    return output

def main(argv = None):
    args = parse_args(argv)

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file missing: '{args.config}'.")

    cfg, configured_snr_levels, configured_snr_labels = load_stress_snr_config(args.config)
    set_seed(args.seed)

    logger = get_logger(f"exp3_stress_{args.seed}")
    dataset_cfg = cfg["dataset"]
    eval_cfg = cfg["evaluation"]
    sample_rate = int(dataset_cfg["sample_rate"])
    clip_sec = float(dataset_cfg["clip_sec"])

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("seed=%d fraction=%s device=%s", args.seed, args.fraction, device)

    base_dataset = BirdCLEFDataset(
        dataset_cfg["birdclef_dir"], 
        sample_rate = sample_rate, 
        duration_sec = clip_sec,
        crop_mode="center", 
        include_path = True,
    )

    manifest_path = Path("checkpoints") / f"data_split_s{args.seed}.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest '{manifest_path}' is missing. Run scripts/data/create_birdclef_splits.py before Exp3.")
    
    split = json.loads(manifest_path.read_text(encoding = "utf-8"))
    if int(split.get("schema_version", -1)) != 5:
        raise ValueError(f"Split manifest '{manifest_path}' uses an older crop/preflight schema.")
    
    if split.get("label_map") != base_dataset.label_map:
        raise ValueError("BirdCLEF label map differs from Exp2's saved split manifest.")
    
    crop_rms_threshold = float(split.get("silence_minimum_rms", DEFAULT_EVALUATION_MINIMUM_RMS))
    if not torch.isfinite(torch.tensor(crop_rms_threshold)) or crop_rms_threshold <= 0:
        raise ValueError(f"Split manifest '{manifest_path}' contains an invalid silence_minimum_rms.")
    
    validate_manifest_file_index(split, base_dataset.relative_paths, base_dataset.labels, context = "Exp3 split")

    active_label_map = split.get("active_label_map", base_dataset.label_map)
    if not active_label_map:
        raise ValueError("Saved split has an empty active_label_map.")
    
    output_label_map = ActiveClassDataset(base_dataset, active_label_map).active_label_map
    path_to_index = {name: i for i, name in enumerate(base_dataset.relative_paths)}

    test_names = split.get("test_filenames", [])
    if not test_names:
        raise ValueError("Saved test split contains no filenames.")
    
    unknown = sorted(set(test_names) - set(path_to_index))
    if unknown:
        raise ValueError(f"Saved test filenames no longer exist in BirdCLEF: {unknown[:5]}.")
    
    test_indices = [path_to_index[name] for name in test_names]
    test_set = set(test_names)
    if len(test_set) != len(test_names):
        raise ValueError("Saved test split contains duplicate filenames.")

    esc50_dir = dataset_cfg.get("esc50_dir", "data/raw/esc50")
    variants = args.variants
    snr_levels = configured_snr_levels
    snr_labels = configured_snr_labels

    noise_pools = {
        "unseen_habitat": load_noise_pool_from_csv(
            "data/processed/esc50_unseen_habitat.csv", esc50_dir, sample_rate, clip_sec,
            expected_partition = "unseen",
        ),
        "seen_habitat": load_noise_pool_from_csv(
            "data/processed/esc50_seen_habitat.csv", esc50_dir, sample_rate, clip_sec,
            expected_partition = "seen",
        ),
    }

    noise_mixer = NoiseMixer(sample_rate=sample_rate)
    logger.info("Loading selected model checkpoints once; each evaluation batch will be shared across the selected variants.")

    models = {variant: load_model_pair(variant, args.seed, args.fraction, len(active_label_map), device, args.bn_policy, args.view_batch_policy) for variant in variants}
    model_keys = {
        variant: ("supervised_noise_baseline" if variant == "supervised_noise" else
                  "supervised_baseline" if variant == "supervised" else f"moco_{variant}")
        for variant in variants
    }

    sweep = {
        "metadata": {
            "schema_version": 5, "seed": args.seed, 
            "fraction": float(args.fraction),
            "dataset_fingerprint": split.get("dataset_fingerprint"),
            "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            "split_method": split.get("split_method"), "n_test": len(test_indices),
            "active_label_map": output_label_map,
            "original_active_label_map": active_label_map,
            "full_label_map": split.get("label_map", base_dataset.label_map),
            "excluded_classes": split.get("excluded_classes", []),
            "n_active_classes": len(output_label_map),
            "n_total_classes": len(split.get("label_map", base_dataset.label_map)),
            "n_excluded_classes": len(split.get("excluded_classes", [])),
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
            "test_filenames": test_names, "snr_levels": snr_levels, "snr_labels": snr_labels,
            "snr_definition": "clip-global waveform power ratio before mixture RMS renormalization",
            "grouping_source": base_dataset.grouping_source,
            "paired_evaluation": True,
            "crop_rms_threshold": crop_rms_threshold,
            "crop_policy": EVALUATION_CROP_POLICY,
        },
    }
    group_id_by_file = dict(zip(base_dataset.relative_paths, base_dataset.group_ids))

    for pool_name in noise_pools:
        sweep[pool_name] = {model_keys[variant]: {} for variant in variants}

    for snr, snr_label in zip(snr_levels, snr_labels):
        if snr is None:

            # Clean input is identical for the seen and unseen comparisons, so evaluate it once.
            eval_ds = ActiveClassDataset(BirdCLEFDataset(
                dataset_cfg["birdclef_dir"], 
                sample_rate = sample_rate, 
                duration_sec = clip_sec,
                crop_mode = "center", 
                short_audio_policy = "repeat", 
                include_path = True,
                minimum_rms = crop_rms_threshold, 
                include_mix_status = True,
            ), active_label_map)

            if eval_ds.relative_paths != base_dataset.relative_paths or eval_ds.label_map != base_dataset.label_map:
                raise RuntimeError("Clean evaluation dataset order/labels changed; paired evaluation is invalid.")
            
            loader = DataLoader(
                Subset(eval_ds, test_indices), 
                batch_size = int(eval_cfg["batch_size"]), 
                shuffle = False,
                num_workers = int(eval_cfg.get("num_workers", 0)), 
                pin_memory = device.type == "cuda",
            )

            metrics_by_variant = evaluate_all_models_on_loader(models, loader, device, group_id_by_file)
            for pool_name in noise_pools:
                for variant in variants:
                    sweep[pool_name][model_keys[variant]][snr_label] = copy.deepcopy(metrics_by_variant[variant])
            logger.info("Evaluated clean condition once for both seen/unseen references.")
            continue

        for pool_name, noise_pool in noise_pools.items():

            eval_ds = ActiveClassDataset(BirdCLEFDataset(
                dataset_cfg["birdclef_dir"], 
                noise_pool = noise_pool, 
                noise_mixer = noise_mixer,
                snr_db = float(snr), 
                sample_rate = sample_rate, 
                duration_sec = clip_sec,
                crop_mode = "center", 
                short_audio_policy = "repeat", 
                include_path = True,
                minimum_rms = crop_rms_threshold, 
                include_mix_status = True,
            ), active_label_map)

            if eval_ds.relative_paths != base_dataset.relative_paths or eval_ds.label_map != base_dataset.label_map:
                raise RuntimeError("Evaluation dataset order/labels changed; paired evaluation is invalid.")
            
            loader = DataLoader(
                Subset(eval_ds, test_indices), 
                batch_size = int(eval_cfg["batch_size"]), 
                shuffle = False,
                num_workers = int(eval_cfg.get("num_workers", 0)), 
                pin_memory = device.type == "cuda",
            )

            metrics_by_variant = evaluate_all_models_on_loader(models, loader, device, group_id_by_file)
            for variant in variants:
                metric = metrics_by_variant[variant]
                sweep[pool_name][model_keys[variant]][snr_label] = metric
                logger.info("pool=%s model=%s SNR=%s acc=%.4f n=%d", 
                            pool_name, variant, snr_label, metric["accuracy"], metric["n"])

    os.makedirs("results", exist_ok = True)
    output_path = args.output
    if Path(output_path).exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    tmp_path = output_path + ".tmp"

    Path(tmp_path).write_text(json.dumps(sweep, indent = 2), encoding = "utf-8")
    os.replace(tmp_path, output_path)
    logger.info("Saved paired per-recording results to '%s'.", output_path)


if __name__ == "__main__":
    main()