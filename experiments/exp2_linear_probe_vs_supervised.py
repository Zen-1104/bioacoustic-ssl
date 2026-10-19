"""Run a frozen linear probe or a fully supervised CNN-10 baseline."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bioacoustic_data import BirdCLEFDataset, NoiseMixer
from models.backbone_cnn10 import CNN_10_Backbone
from models.linear_head import Linear_Head
from utils.dataset_manifest import validate_manifest_file_index
from utils.logger import get_logger
from utils.seed import set_seed
from utils.label_scope import ActiveClassDataset
from utils.habitat_noise import load_seen_habitat_noise_pool
from utils.audio_io import DEFAULT_EVALUATION_MINIMUM_RMS, EVALUATION_CROP_POLICY

SUPERVISED_MODES = {"supervised", "supervised_noise"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default = "configs/linear_probe.yaml")
    parser.add_argument("--mode", type=str, default = "probe", choices = ["probe", "supervised", "supervised_noise"])
    parser.add_argument("--checkpoint", type=str, default = None)
    parser.add_argument("--variant", type=str, default = "habitat", choices = ["habitat", "standard", "white_noise", "supervised", "supervised_noise"])
    parser.add_argument("--fraction", type=float, default = 1.0)
    parser.add_argument("--seed", type=int, default = 42)
    return parser.parse_args(argv)


def fraction_key(fraction: float) -> str:
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"fraction must be in (0,1], got {fraction}.")    
    return f"{fraction:.2f}"


def extract_backbone_state(checkpoint_path):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"MoCo checkpoint file not found: {checkpoint_path!r}. Run Exp1 first.")
    
    checkpoint = torch.load(checkpoint_path, map_location = "cpu", weights_only = True)
    state_dict = checkpoint.get("model_state", checkpoint)

    prefix = "encoder_q.0."
    result = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}

    if not result:
        raise KeyError(f"No backbone parameters with prefix '{prefix}' found in '{checkpoint_path}'.")
    
    return result

def load_split_manifest(path: Path, dataset: BirdCLEFDataset, seed: int):
    if not path.is_file():
        raise FileNotFoundError(
            f"Split manifest '{path}' is missing. Run scripts/create_birdclef_splits.py --seeds 42 123 999 before experiments."
        )
    data = json.loads(path.read_text(encoding = "utf-8"))

    if int(data.get("schema_version", -1)) != 5:
        raise ValueError(f"Split manifest '{path}' is from an older preflight/crop protocol.")
    
    if int(data.get("seed", -1)) != seed:
        raise ValueError(f"Split manifest seed mismatch in '{path}'.")
    
    if data.get("label_map") != dataset.label_map:
        raise ValueError("BirdCLEF class map differs from the split manifest. Recreate splits deliberately after checking the data.")
    
    validate_manifest_file_index(data, 
                                 dataset.relative_paths, 
                                 dataset.labels, 
                                 context = f"Exp2 split {path}")
    return data

def precompute_embeddings(backbone, dataset, indices, device, batch_size, num_workers):
    """Cache frozen backbone features once for a linear probe's fixed inputs."""

    loader = DataLoader(
        Subset(dataset, indices), 
        batch_size = batch_size, 
        shuffle = False,
        num_workers = num_workers, 
        pin_memory = device.type == "cuda",
    )
    feature_batches, label_batches = [], []
    backbone.eval()
    
    with torch.no_grad():
        for x, y in loader:
            embeddings = backbone(x.to(device, non_blocking = device.type == "cuda")).cpu()
            feature_batches.append(embeddings)
            label_batches.append(y.cpu())

    if not feature_batches:
        raise RuntimeError("Embedding cache creation processed zero batches.")
    
    features = torch.cat(feature_batches)
    labels = torch.cat(label_batches).long()

    if not torch.isfinite(features).all():
        raise FloatingPointError("Frozen backbone produced non finite cached embeddings.")
    
    return features, labels

def main(argv = None):
    args = parse_args(argv)

    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be greater than 0 and at most 1.")
    
    if args.mode == "probe" and not args.checkpoint:
        raise ValueError("--checkpoint is required for --mode probe.")
    
    if args.mode in SUPERVISED_MODES and args.variant != args.mode:
        raise ValueError(f"Use --variant {args.mode} when --mode {args.mode}.")
    
    if args.mode == "probe" and args.variant in SUPERVISED_MODES:
        raise ValueError("Supervised variants require their corresponding supervised mode.")
    
    set_seed(args.seed)
    os.makedirs("results", exist_ok = True)

    log_path = Path("results") / f"exp2_{args.variant}_s{args.seed}_frac{float(args.fraction)}.jsonl"
    log_path.write_text("", encoding = "utf-8")
    logger = get_logger(f"exp2_{args.mode}_{args.variant}_{args.seed}")

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Configuration file missing: '{args.config}'.")
    
    with open(args.config, encoding = "utf-8") as stream:
        cfg = yaml.safe_load(stream)

    config_sha256 = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr = subprocess.DEVNULL, text = True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config '{args.config}'.")

    dataset_cfg = cfg["dataset"]
    exp_cfg = cfg["experiment"]
    data_root = dataset_cfg["birdclef_dir"]
    sample_rate = int(dataset_cfg["sample_rate"])
    clip_sec = float(dataset_cfg["clip_sec"])

    # Build one stable label index, then use separate crop policies for train and evaluation.
    index_dataset = BirdCLEFDataset(data_root, sample_rate = sample_rate, duration_sec = clip_sec, crop_mode = "center")
    manifest_path = Path("checkpoints") / f"data_split_s{args.seed}.json"
    manifest = load_split_manifest(manifest_path, index_dataset, args.seed)
    evaluation_minimum_rms = float(manifest.get("silence_minimum_rms", DEFAULT_EVALUATION_MINIMUM_RMS))

    if not math.isfinite(evaluation_minimum_rms) or evaluation_minimum_rms <= 0:
        raise ValueError(f"Split manifest '{manifest_path}' contains an invalid silence_minimum_rms.")
    
    path_to_index = {path: i for i, path in enumerate(index_dataset.relative_paths)}
    fraction = fraction_key(args.fraction)
    fraction_names = manifest["train_fraction_filenames"].get(fraction)
    if fraction_names is None:
        valid = list(manifest["train_fraction_filenames"])
        raise ValueError(f"Fraction {fraction} is not available in the split manifest. Supported fractions: {valid}.")

    train_names = set(manifest["train_filenames"])
    val_names = manifest["validation_filenames"]
    test_names = manifest["test_filenames"]
    if not set(fraction_names).issubset(train_names):
        raise ValueError(f"Fraction {fraction} contains paths outside the training split.")
    
    train_indices = [path_to_index[name] for name in fraction_names]
    val_indices = [path_to_index[name] for name in val_names]
    test_indices = [path_to_index[name] for name in test_names]
    if not train_indices or not val_indices or not test_indices:
        raise ValueError(f"Empty split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}.")

    active_label_map = manifest.get("active_label_map", index_dataset.label_map)
    if not active_label_map:
        raise ValueError("Saved split has an empty active_label_map.")
    
    train_noise_pool = None
    train_noise_mixer = None
    train_noise_kwargs = {}
    if args.mode == "supervised_noise":
        noise_cfg = cfg.get("noise_augmentation", {})
        split_csv = noise_cfg.get("seen_habitat_csv", "data/processed/esc50_seen_habitat.csv")
        esc50_dir = dataset_cfg.get("esc50_dir", "data/raw/esc50")
        snr_range = noise_cfg.get("snr_db_range", [-5, 20])

        if not isinstance(snr_range, (list, tuple)) or len(snr_range) != 2:
            raise ValueError("noise_augmentation.snr_db_range must contain [min_db, max_db].")
        
        train_noise_pool = load_seen_habitat_noise_pool(split_csv, esc50_dir, sample_rate, clip_sec)
        train_noise_mixer = NoiseMixer(sample_rate = sample_rate)

        train_noise_kwargs = {
            "noise_pool": train_noise_pool,
            "noise_mixer": train_noise_mixer,
            "noise_snr_db_range": (snr_range[0], snr_range[1]),
            "noise_probability": float(noise_cfg.get("probability", 1.0)),
        }

        logger.info(
            "Matched noise supervised training: %d seen-habitat noise clips; SNR range = %s dB; probability = %.3f; eval remains clean",
            len(train_noise_pool), snr_range, train_noise_kwargs["noise_probability"],
        )

    train_dataset = ActiveClassDataset(
        BirdCLEFDataset(data_root, 
                        sample_rate = sample_rate, 
                        duration_sec = clip_sec, 
                        crop_mode = "random",
                        minimum_rms = evaluation_minimum_rms, 
                        **train_noise_kwargs),
        active_label_map,
    )

    eval_dataset = ActiveClassDataset(
        BirdCLEFDataset(data_root, 
                        sample_rate = sample_rate, 
                        duration_sec = clip_sec, 
                        crop_mode = "center",
                        minimum_rms = evaluation_minimum_rms),
        active_label_map,
    )

    output_label_map = train_dataset.active_label_map
    if output_label_map != eval_dataset.active_label_map:
        raise RuntimeError("Train/evaluation active label maps disagree.")
    
    if train_dataset.relative_paths != index_dataset.relative_paths or eval_dataset.label_map != index_dataset.label_map:
        raise RuntimeError("BirdCLEF file order or label map changed between dataset instances.")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    logger.info("Mode=%s variant=%s fraction=%s seed=%s device=%s", 
                args.mode, args.variant, fraction, args.seed, device)
    
    logger.info("Split=%s | train=%d validation=%d test=%d active_classes=%d full_inventory_classes=%d", 
                manifest["split_method"], 
                len(train_indices), len(val_indices), len(test_indices), len(active_label_map), len(index_dataset.label_map))

    if manifest.get("grouping_source") == "file_level_no_metadata":
        logger.warning("Split is file level because no usable location metadata was detected; it is not a recordist independent evaluation.")

    loader_kwargs = {
        "batch_size": int(exp_cfg["batch_size"]),
        "num_workers": int(exp_cfg.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }

    if loader_kwargs["num_workers"] < 0:
        raise ValueError("num_workers must be non-negative.")
    
    train_loader = DataLoader(Subset(train_dataset, train_indices), shuffle = True, **loader_kwargs)
    val_loader = DataLoader(Subset(eval_dataset, val_indices), shuffle = False, **loader_kwargs)
    test_loader = DataLoader(Subset(eval_dataset, test_indices), shuffle = False, **loader_kwargs)

    if len(train_loader) == 0 or len(val_loader) == 0 or len(test_loader) == 0:
        raise RuntimeError("A DataLoader is empty. Reduce batch size or inspect the split manifest.")

    num_classes = len(active_label_map)
    backbone = CNN_10_Backbone().to(device)
    head = Linear_Head(in_features = 512, num_classes = num_classes).to(device)

    if args.mode == "probe":
        backbone.load_state_dict(extract_backbone_state(args.checkpoint), strict=True)
        backbone.eval()

        for parameter in backbone.parameters():
            parameter.requires_grad_(False)

        optimizer = torch.optim.Adam(head.parameters(), lr = float(exp_cfg["learning_rate"]))

    else:
        optimizer = torch.optim.Adam(
            list(backbone.parameters()) + list(head.parameters()),
            lr = float(exp_cfg.get("supervised_lr", exp_cfg["learning_rate"])),
        )

    criterion = nn.CrossEntropyLoss()

    if args.mode == "probe":
        train_x, train_y = precompute_embeddings(backbone, eval_dataset, train_indices, device,
                                                 loader_kwargs["batch_size"], loader_kwargs["num_workers"])
        
        val_x, val_y = precompute_embeddings(backbone, eval_dataset, val_indices, device,
                                             loader_kwargs["batch_size"], loader_kwargs["num_workers"])
        
        test_x, test_y = precompute_embeddings(backbone, eval_dataset, test_indices, device,
                                               loader_kwargs["batch_size"], loader_kwargs["num_workers"])
        
        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size = loader_kwargs["batch_size"], shuffle = True)
        val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size = loader_kwargs["batch_size"], shuffle = False)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size = loader_kwargs["batch_size"], shuffle = False)

        logger.info("Cached frozen embeddings: train=%s val=%s test=%s", tuple(train_x.shape), tuple(val_x.shape), tuple(test_x.shape))

    best_val_loss = float("inf")
    best_head_state = None
    best_backbone_state = None

    for epoch in range(1, int(exp_cfg["epochs"]) + 1):
        head.train()

        if args.mode in SUPERVISED_MODES:
            backbone.train()
        else:
            backbone.eval()

        train_loss_sum = 0.0
        seen_train = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none = True)
            features = x if args.mode == "probe" else backbone(x)
            logits = head(features)
            loss = criterion(logits, y)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non finite training loss at epoch {epoch}.")
            
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * x.size(0)
            seen_train += x.size(0)

        if seen_train == 0:
            raise RuntimeError("Training loop processed zero examples.")

        backbone.eval()
        head.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                features = x if args.mode == "probe" else backbone(x)
                logits = head(features)
                loss = criterion(logits, y)
                val_loss_sum += float(loss.item()) * x.size(0)
                val_correct += int((logits.argmax(1) == y).sum().item())
                val_total += y.size(0)

        if val_total == 0:
            raise RuntimeError("Validation loop processed zero examples.")
        
        avg_val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        train_loss = train_loss_sum / seen_train

        logger.info("Epoch %d/%d train_loss=%.5f val_loss=%.5f val_acc=%.4f", 
                    epoch, exp_cfg["epochs"], train_loss, avg_val_loss, val_acc)

        epoch_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(), 
            "mode": args.mode,
            "variant": args.variant, 
            "seed": args.seed, 
            "fraction": float(args.fraction),
            "epoch": epoch, 
            "train_loss": train_loss, 
            "validation_loss": avg_val_loss,
            "validation_accuracy": val_acc, 
            "config_path": args.config,
            "config_sha256": config_sha256, 
            "git_commit": git_commit,
            "dataset_fingerprint": manifest["dataset_fingerprint"],
        }

        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(epoch_record) + "\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_head_state = copy.deepcopy(head.state_dict())
            best_backbone_state = copy.deepcopy(backbone.state_dict())

    if best_head_state is None or best_backbone_state is None:
        raise RuntimeError("No best checkpoint was selected; validation did not produce a usable metric.")
    
    head.load_state_dict(best_head_state, strict = True)

    if args.mode in SUPERVISED_MODES:
        backbone.load_state_dict(best_backbone_state, strict = True)

    backbone.eval()
    head.eval()

    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            features = x if args.mode == "probe" else backbone(x)
            logits = head(features)
            loss = criterion(logits, y)
            test_loss_sum += float(loss.item()) * x.size(0)
            test_correct += int((logits.argmax(1) == y).sum().item())
            test_total += y.size(0)

    if test_total == 0:
        raise RuntimeError("Test loop processed zero examples.")
    
    test_acc = test_correct / test_total
    avg_test_loss = test_loss_sum / test_total

    logger.info("FINAL test_loss=%.5f test_acc=%.4f (%d examples)", 
                avg_test_loss, test_acc, test_total)

    os.makedirs("checkpoints", exist_ok = True)
    os.makedirs("results", exist_ok = True)

    fraction_tag = str(float(args.fraction))
    head_path = f"checkpoints/linear_head_{args.variant}_s{args.seed}_frac{fraction_tag}.pt"
    tmp_head_path = head_path + ".tmp"

    torch.save({"model_state": best_head_state, 
                "label_map": output_label_map, 
                "original_active_label_map": active_label_map, 
                "full_label_map": index_dataset.label_map}, tmp_head_path)
    os.replace(tmp_head_path, head_path)

    if args.mode in SUPERVISED_MODES:
        backbone_baseline_name = "supervised_noise_baseline" if args.mode == "supervised_noise" else "supervised_baseline"
        bb_path = f"checkpoints/{backbone_baseline_name}_s{args.seed}_frac{fraction_tag}.pt"
        tmp_bb_path = bb_path + ".tmp"

        torch.save({"model_state": best_backbone_state, 
                    "label_map": output_label_map, 
                    "original_active_label_map": active_label_map, 
                    "full_label_map": index_dataset.label_map}, tmp_bb_path)
        os.replace(tmp_bb_path, bb_path)

    result_path = f"results/probe_{args.variant}_s{args.seed}_frac{fraction_tag}.json"

    result = {
        "schema_version": 4,
        "mode": args.mode,
        "variant": args.variant,
        "seed": args.seed,
        "fraction": float(args.fraction),
        "test_loss": avg_test_loss,
        "test_accuracy": test_acc,
        "n_train": len(train_indices),
        "n_validation": len(val_indices),
        "n_test": test_total,
        "num_classes": num_classes,
        "label_map": output_label_map,
        "original_active_label_map": active_label_map,
        "full_label_map": index_dataset.label_map,
        "n_active_classes": len(output_label_map),
        "n_total_classes": len(manifest.get("label_map", index_dataset.label_map)),
        "n_excluded_classes": len(manifest.get("excluded_classes", [])),
        "excluded_classes": manifest.get("excluded_classes", []),
        "n_silent_files_excluded": int(manifest.get("n_silent_files_excluded", len(manifest.get("silent_excluded_filenames", [])))),
        "silent_excluded_filenames": manifest.get("silent_excluded_filenames", []),
        "n_preflight_excluded_files": int(manifest.get("n_preflight_excluded_files", 0)),
        "preflight_excluded_filenames": manifest.get("preflight_excluded_filenames", []),
        "n_preflight_read_error_files_excluded": int(manifest.get("n_preflight_read_error_files_excluded", 0)),
        "preflight_read_error_filenames": manifest.get("preflight_read_error_filenames", []),
        "preflight_reason_histogram": manifest.get("preflight_reason_histogram", {}),
        "preflight_read_error_fraction": float(manifest.get("preflight_read_error_fraction", 0.0)),
        "preflight_excluded_fraction": float(manifest.get("preflight_excluded_fraction", 0.0)),
        "preflight_max_read_error_fraction": float(manifest.get("preflight_max_read_error_fraction", 0.01)),
        "preflight_max_excluded_fraction": float(manifest.get("preflight_max_excluded_fraction", 0.05)),
        "evaluation_crop_policy": EVALUATION_CROP_POLICY,
        "evaluation_minimum_rms": evaluation_minimum_rms,
        "rare_class_policy": manifest.get("rare_class_policy", "unknown"),
        "class_scope_note": manifest.get("class_scope_note", "Metrics cover active classes only."),
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "split_method": manifest["split_method"],
        "best_validation_loss": best_val_loss,
        "config_sha256": config_sha256,
        "git_commit": git_commit,
        "training_log": str(log_path),
    }

    tmp_result = result_path + ".tmp"

    with open(tmp_result, "w", encoding = "utf-8") as stream:
        json.dump(result, stream, indent = 2)

    os.replace(tmp_result, result_path)
    logger.info("Saved result '%s' and head '%s'.", result_path, head_path)


if __name__ == "__main__":
    main()