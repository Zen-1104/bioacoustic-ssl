"""Objective functions used by the self supervised training pipeline."""
import torch
import torch.nn.functional as F


def info_nce_loss(query: torch.Tensor, positive_key: torch.Tensor, queue: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Compute MoCo style InfoNCE loss for [B,D], [B,D], and [D,K] inputs."""
    
    if query.ndim != 2 or positive_key.ndim != 2 or queue.ndim != 2:
        raise ValueError(f"Expected query/key [B,D] and queue [D,K]; got {tuple(query.shape)}, {tuple(positive_key.shape)}, {tuple(queue.shape)}.")
    
    if query.shape != positive_key.shape:
        raise ValueError(f"Query/key shapes must match, got {tuple(query.shape)} vs {tuple(positive_key.shape)}.")
    
    if queue.shape[0] != query.shape[1]:
        raise ValueError(f"Queue feature dimension {queue.shape[0]} does not match query dimension {query.shape[1]}.")
    
    if query.shape[0] == 0 or queue.shape[1] == 0:
        raise ValueError("InfoNCE requires a non empty batch and negative queue.")
    
    if not isinstance(tau, (int, float)) or tau <= 0:
        raise ValueError(f"Temperature tau must be positive, got {tau!r}.")
    
    if not torch.isfinite(query).all() or not torch.isfinite(positive_key).all() or not torch.isfinite(queue).all():
        raise FloatingPointError("InfoNCE inputs contain NaN or infinite values.")

    query = F.normalize(query, dim = 1)
    positive_key = F.normalize(positive_key.detach(), dim = 1)
    queue = F.normalize(queue.detach(), dim = 0)

    l_pos = torch.einsum("bd,bd->b", query, positive_key).unsqueeze(1)
    l_neg = query @ queue

    logits = torch.cat((l_pos, l_neg), dim = 1) / float(tau)
    labels = torch.zeros(logits.shape[0], dtype = torch.long, device = query.device)
    loss = F.cross_entropy(logits, labels)

    if not torch.isfinite(loss):
        raise FloatingPointError("InfoNCE loss is not finite")
    
    return loss