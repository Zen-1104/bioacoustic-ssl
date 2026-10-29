"""MoCo-style wrapper with symmetric two-view support and explicit BN policy."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCo_Wrapper(nn.Module):
    BN_POLICIES = ("copy_query", "ema_query", "frozen_key", "independent_key")
    VIEW_BATCH_POLICIES = ("separate", "concatenated")

    def __init__(self, backbone_q, backbone_k, head_q, head_k,
                 dim = 128, K = 8192, m = 0.999,
                 bn_policy = "copy_query", view_batch_policy = "separate"):

        super().__init__()

        if bn_policy not in self.BN_POLICIES:
            raise ValueError(f"bn_policy must be one of {self.BN_POLICIES}; got {bn_policy!r}.")

        if view_batch_policy not in self.VIEW_BATCH_POLICIES:
            raise ValueError(f"view_batch_policy must be one of {self.VIEW_BATCH_POLICIES}; got {view_batch_policy!r}.")

        self.bn_policy = bn_policy
        self.view_batch_policy = view_batch_policy

        if dim <= 0 or K <= 0:
            raise ValueError("Projection dimension and queue size must be positive.")
        if not 0.0 <= m < 1.0:
            raise ValueError("Momentum m must be in [0, 1).")

        self.K = int(K)
        self.m = float(m)
        self.encoder_q = nn.Sequential(backbone_q, head_q)
        self.encoder_k = nn.Sequential(backbone_k, head_k)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim = 0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype = torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        query_parameters = dict(self.encoder_q.named_parameters())
        key_parameters = dict(self.encoder_k.named_parameters())

        if query_parameters.keys() != key_parameters.keys():
            raise RuntimeError("Query/key encoder parameter structures differ.")

        for name, param_k in key_parameters.items():
            param_k.mul_(self.m).add_(query_parameters[name], alpha = 1.0 - self.m)

    @torch.no_grad()
    def _sync_key_buffers_from_query(self):
        if self.bn_policy in ("frozen_key", "independent_key"):
            return

        query_buffers = dict(self.encoder_q.named_buffers())

        key_buffers = dict(self.encoder_k.named_buffers())
        if query_buffers.keys() != key_buffers.keys():
            raise RuntimeError("Query/key encoder buffer structures differ.")

        for name, buffer_k in key_buffers.items():
            buffer_q = query_buffers[name]

            if buffer_k.shape != buffer_q.shape:
                raise RuntimeError(f"Query/key buffer shape mismatch for {name!r}.")

            if self.bn_policy == "copy_query" or not buffer_k.is_floating_point():
                buffer_k.copy_(buffer_q)
            else:
                buffer_k.mul_(self.m).add_(buffer_q, alpha = 1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        if keys.ndim != 2 or keys.shape[1] != self.queue.shape[0]:
            raise ValueError(f"Expected key embeddings [B,{self.queue.shape[0]}], got {tuple(keys.shape)}.")

        batch_size = int(keys.shape[0])
        if batch_size < 1 or batch_size > self.K:
            raise ValueError(f"Enqueue batch size must be between 1 and queue size {self.K}; got {batch_size}.")

        if self.K % batch_size != 0:
            raise ValueError(f"Queue size {self.K} must be divisible by enqueue batch size {batch_size}; use drop_last = True and a compatible queue size.")

        if not torch.isfinite(keys).all():
            raise FloatingPointError("Cannot enqueue non finite key embeddings.")

        keys = F.normalize(keys.detach(), dim=1)
        ptr = int(self.queue_ptr.item())
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    def forward(self, im_q, im_k, *, symmetric: bool = False):
        with torch.no_grad():
            self._momentum_update_key_encoder()
            if self.bn_policy == "independent_key":
                self.encoder_k.train(self.training)
            else:
                self.encoder_k.eval()

        if symmetric and self.view_batch_policy == "concatenated":
            query_outputs = self.encoder_q(torch.cat((im_q, im_k), dim = 0))
            q1, q2 = query_outputs.split((im_q.shape[0], im_k.shape[0]), dim = 0)
            q1, q2 = F.normalize(q1, dim = 1), F.normalize(q2, dim = 1)
        else:
            q1 = F.normalize(self.encoder_q(im_q), dim = 1)
            q2 = F.normalize(self.encoder_q(im_k), dim = 1) if symmetric else None

        with torch.no_grad():
            self._sync_key_buffers_from_query()

            if symmetric and self.view_batch_policy == "concatenated":
                key_outputs = self.encoder_k(torch.cat((im_k, im_q), dim = 0))
                k1, k2 = key_outputs.split((im_k.shape[0], im_q.shape[0]), dim = 0)
                k1, k2 = F.normalize(k1, dim = 1), F.normalize(k2, dim = 1)
            else:
                k1 = F.normalize(self.encoder_k(im_k), dim = 1)
                k2 = F.normalize(self.encoder_k(im_q), dim = 1) if symmetric else None

        if symmetric:
            return q1, k1, q2, k2, self.queue
        return q1, k1, self.queue

if __name__ == "__main__":
    bq, bk = nn.Linear(512, 512), nn.Linear(512, 512)
    hq, hk = nn.Linear(512, 128), nn.Linear(512, 128)

    moco = MoCo_Wrapper(bq, bk, hq, hk, dim = 128, K = 128, m = 0.999)
    x_q, x_k = torch.randn(16, 512), torch.randn(16, 512)
    q1, k1, q2, k2, queue = moco(x_q, x_k, symmetric = True)

    moco._dequeue_and_enqueue(torch.cat([k1, k2]))
    print(f"Queries: {q1.shape}, {q2.shape}; keys: {k1.shape}, {k2.shape}; queue: {queue.shape}")