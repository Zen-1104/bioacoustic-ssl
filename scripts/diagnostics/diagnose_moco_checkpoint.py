#!/usr/bin/env python3
"""Inspect a saved MoCo checkpoint's BirdCLEF frozen-backbone representations.

This is evaluation-only: it does not train or modify a checkpoint.
Run from any directory; paths default to this repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bioacoustic_data import BirdCLEFDataset
from evaluation.knn_probe_eval import balanced_subset_indices, _encode
from experiments.exp1_pretrain_moco import load_split_for_monitor
from models.backbone_cnn10 import CNN_10_Backbone
from models.moco_wrapper import MoCo_Wrapper
from models.projection_head import Projection_Head


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def neighbor_accuracy(train_x, train_y, val_x, val_y, k: int) -> float:
    similarities = val_x @ train_x.T
    nearest = similarities.topk(k=min(k, train_x.shape[0]), dim=1).indices
    neighbor_labels = train_y[nearest]
    max_label = int(max(train_y.max().item(), val_y.max().item()))
    preds = []
    for row in neighbor_labels:
        counts = torch.bincount(row, minlength=max_label + 1)
        preds.append(int(counts.argmax().item()))
    pred = torch.tensor(preds, dtype=torch.long)
    return float((pred == val_y).float().mean().item())


def pairwise_cosine_stats(x: torch.Tensor, labels: torch.Tensor, seed: int, n_pairs: int = 30000):
    n = int(x.shape[0])
    if n < 2:
        return {"random_pair_cosine_mean": None, "same_label_cosine_mean": None,
                "different_label_cosine_mean": None, "same_label_pair_count": 0,
                "different_label_pair_count": 0}
    gen = torch.Generator(device="cpu").manual_seed(seed + 2027)
    i = torch.randint(0, n, (n_pairs,), generator=gen)
    j = torch.randint(0, n, (n_pairs,), generator=gen)
    mask = i != j
    i, j = i[mask], j[mask]
    sims = (x[i] * x[j]).sum(dim=1)
    same = labels[i] == labels[j]
    same_vals = sims[same]
    diff_vals = sims[~same]
    return {
        "random_pair_cosine_mean": float(sims.mean().item()),
        "random_pair_cosine_std": float(sims.std(unbiased=False).item()),
        "same_label_cosine_mean": float(same_vals.mean().item()) if same_vals.numel() else None,
        "different_label_cosine_mean": float(diff_vals.mean().item()) if diff_vals.numel() else None,
        "same_label_pair_count": int(same_vals.numel()),
        "different_label_pair_count": int(diff_vals.numel()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Path to moco_habitat_s42.pt")
    ap.add_argument("--config", default=str(ROOT / "configs/pretrain_moco_habitat.yaml"))
    ap.add_argument("--split", default=None, help="Defaults to checkpoints/data_split_s<seed>.json")
    ap.add_argument("--split-seed", type=int, default=None, help="Evaluation split seed; defaults to checkpoint seed.")
    ap.add_argument("--max-train", type=int, default=2000)
    ap.add_argument("--max-validation", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if min(args.max_train, args.max_validation, args.batch_size, args.k) < 1:
        raise ValueError("Sample limits, batch size, and k must be positive.")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg, model_cfg = cfg["dataset"], cfg["model"]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    seed = int(payload.get("seed", cfg.get("training", {}).get("seed", 42)))
    split_seed = seed if args.split_seed is None else args.split_seed
    device = choose_device(args.device)

    backbone_q, backbone_k = CNN_10_Backbone(), CNN_10_Backbone()
    head_q = Projection_Head(512, 512, int(model_cfg["feature_dim"]))
    head_k = Projection_Head(512, 512, int(model_cfg["feature_dim"]))
    model = MoCo_Wrapper(
        backbone_q, backbone_k, head_q, head_k,
        dim=int(model_cfg["feature_dim"]), K=int(model_cfg["queue_size"]),
        m=float(model_cfg["momentum"]),
        bn_policy=payload.get("bn_policy", model_cfg.get("bn_policy", "copy_query")),
        view_batch_policy=payload.get("view_batch_policy", model_cfg.get("view_batch_policy", "separate")),
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    backbone = model.encoder_q[0]

    dataset = BirdCLEFDataset(
        data_cfg.get("birdclef_dir", "data/raw/birdclef"),
        sample_rate=int(data_cfg["sample_rate"]),
        duration_sec=float(data_cfg["clip_sec"]),
        crop_mode="center",
    )
    split_path = Path(args.split).expanduser().resolve() if args.split else ROOT / f"checkpoints/data_split_s{split_seed}.json"
    manifest = load_split_for_monitor(split_path, dataset, split_seed)
    train_files = manifest["train_filenames"]
    val_files = manifest["validation_filenames"]
    train_idx = balanced_subset_indices(dataset, train_files, args.max_train, split_seed)
    val_idx = balanced_subset_indices(dataset, val_files, args.max_validation, split_seed + 1)

    print(f"Device: {device}")
    print(f"Checkpoint epoch: {payload.get('epoch', 'unknown')} | seed: {seed}")
    print(f"Evaluation split seed: {split_seed}")
    print(f"Extracting {len(train_idx)} balanced support features and {len(val_idx)} balanced validation features...", flush=True)
    train_x, train_y = _encode(backbone, dataset, train_idx, device, args.batch_size, args.num_workers)
    val_x, val_y = _encode(backbone, dataset, val_idx, device, args.batch_size, args.num_workers)

    train_std = train_x.std(dim=0, unbiased=False)
    val_std = val_x.std(dim=0, unbiased=False)
    accuracy = neighbor_accuracy(train_x, train_y, val_x, val_y, args.k)
    active_classes = int(torch.unique(train_y).numel())
    stats = pairwise_cosine_stats(val_x, val_y, split_seed)
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "device": str(device),
        "support_examples": int(train_x.shape[0]),
        "validation_examples": int(val_x.shape[0]),
        "support_classes": active_classes,
        "balanced_chance_reference": 1.0 / active_classes if active_classes else None,
        "knn_k": args.k,
        "knn_accuracy": accuracy,
        "support_feature_std_mean_over_dimensions": float(train_std.mean().item()),
        "support_feature_std_median_over_dimensions": float(train_std.median().item()),
        "validation_feature_std_mean_over_dimensions": float(val_std.mean().item()),
        "validation_feature_std_median_over_dimensions": float(val_std.median().item()),
        **stats,
    }
    print("\nDIAGNOSTIC RESULTS")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
