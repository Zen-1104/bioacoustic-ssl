"""Small frozen backbone KNN monitor for self supervised pretraining."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from bioacoustic_data import BirdCLEFDataset


def balanced_subset_indices(dataset: BirdCLEFDataset, filenames: list[str], maximum: int, seed: int) -> list[int]:
    if maximum < 1:
        raise ValueError("knn monitor sample limits must be positive.")
    
    index_by_path = {name: idx for idx, name in enumerate(dataset.relative_paths)}
    unknown = sorted(set(filenames) - set(index_by_path))

    if unknown:
        raise ValueError(f"knn monitor split contains unknown file paths: {unknown[:5]}.")
    
    buckets: dict[int, list[int]] = defaultdict(list)

    for name in filenames:
        idx = index_by_path[name]
        buckets[dataset.labels[idx]].append(idx)

    rng = np.random.default_rng(seed)

    for label in sorted(buckets):
        rng.shuffle(buckets[label])

    selected: list[int] = []
    label_order = sorted(buckets)
    cursor = 0

    while len(selected) < maximum:
        added = False

        for label in label_order:
            if cursor < len(buckets[label]) and len(selected) < maximum:
                selected.append(buckets[label][cursor])
                added = True
        if not added:
            break

        cursor += 1

    if len(selected) == 0:
        raise ValueError("No items available for knn monitoring.")
    
    return selected


def _encode(backbone, dataset, indices, device, batch_size, num_workers):
    loader = DataLoader(
        Subset(dataset, indices), 
        batch_size = batch_size, 
        shuffle = False,
        num_workers = num_workers, 
        pin_memory = (device.type == "cuda"),
    )
    embeddings, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feats = F.normalize(backbone(x), dim = 1)
            embeddings.append(feats.cpu())
            labels.append(y.cpu())

    if not embeddings:
        raise RuntimeError("knn monitor processed no batches.")
    
    return torch.cat(embeddings), torch.cat(labels)

def knn_probe_accuracy(
    backbone,
    dataset: BirdCLEFDataset,
    train_filenames: list[str],
    validation_filenames: list[str],
    device: torch.device,
    *,
    train_max: int = 1000,
    validation_max: int = 500,
    k: int = 5,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
) -> float:
    if k < 1:
        raise ValueError("k must be positive.")
    
    train_idx = balanced_subset_indices(dataset, train_filenames, train_max, seed)
    val_idx = balanced_subset_indices(dataset, validation_filenames, validation_max, seed + 1)
    was_training = backbone.training
    backbone.eval()

    train_x, train_y = _encode(backbone, dataset, train_idx, device, batch_size, num_workers)
    val_x, val_y = _encode(backbone, dataset, val_idx, device, batch_size, num_workers)

    if train_x.shape[0] < k:
        raise ValueError(f"k={k} exceeds available knn support samples ({train_x.shape[0]}).")
    
    similarities = val_x @ train_x.T
    nearest = similarities.topk(k = min(k, train_x.shape[0]), dim = 1).indices
    neighbor_labels = train_y[nearest]
    predictions = []

    for row in neighbor_labels:
        counts = torch.bincount(row, minlength = int(max(train_y.max(), val_y.max()).item()) + 1)
        predictions.append(int(counts.argmax().item()))

    predictions = torch.tensor(predictions, dtype = torch.long)
    accuracy = float((predictions == val_y).float().mean().item())

    if was_training:
        backbone.train()
        
    return accuracy