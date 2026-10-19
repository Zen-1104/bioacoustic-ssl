"""Random seeding and serializable RNG state checkpoint helpers."""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42, *, deterministic: bool | None = None, deterministic_warn_only: bool | None = None) -> None:
    if deterministic is None:
        deterministic = os.environ.get("BIOACOUSTIC_DETERMINISTIC", "0").strip().lower() in {"1", "true", "yes", "on"}
        
    if deterministic_warn_only is None:
        deterministic_warn_only = os.environ.get("BIOACOUSTIC_DETERMINISTIC_WARN_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
        
    if deterministic:
        # cuBLAS reads this when its CUDA context is initialized. 
        # Callers must invoke set_seed before creating CUDA models or running CUDA ops.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = False if deterministic else torch.backends.cudnn.benchmark
        
    torch.use_deterministic_algorithms(bool(deterministic), warn_only = bool(deterministic and deterministic_warn_only))
    
    mps_backend = getattr(torch, "mps", None)
    mps_available = getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()
    if mps_available and mps_backend is not None and hasattr(mps_backend, "manual_seed"):
        mps_backend.manual_seed(seed)


def capture_rng_state() -> dict[str, Any]:
    np_name, np_keys, np_pos, np_has_gauss, np_cached_gaussian = np.random.get_state()
    
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": str(np_name),
            "keys": np_keys.tolist(),
            "position": int(np_pos),
            "has_gauss": int(np_has_gauss),
            "cached_gaussian": float(np_cached_gaussian),
        },
        "torch_cpu": torch.get_rng_state().cpu(),
    }
    
    if torch.cuda.is_available():
        state["torch_cuda"] = [rng_state.cpu() for rng_state in torch.cuda.get_rng_state_all()]

    mps_backend = getattr(torch, "mps", None)
    mps_available = getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()
    if mps_available and mps_backend is not None and hasattr(mps_backend, "get_rng_state"):
        try:
            state["torch_mps"] = mps_backend.get_rng_state().cpu()
        except (AttributeError, RuntimeError, TypeError):
            pass
            
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        raise TypeError("RNG checkpoint state must be a dictionary.")
        
    required = {"python", "numpy", "torch_cpu"}
    
    missing = required.difference(state)
    if missing:
        raise ValueError(f"RNG checkpoint state is missing keys: {sorted(missing)}")

    random.setstate(state["python"])
    np_state = state["numpy"]
    
    np.random.set_state((
        str(np_state["algorithm"]),
        np.asarray(np_state["keys"], dtype = np.uint32),
        int(np_state["position"]),
        int(np_state["has_gauss"]),
        float(np_state["cached_gaussian"]),
    ))
    
    torch.set_rng_state(state["torch_cpu"].detach().cpu().to(dtype = torch.uint8))

    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([rng.detach().cpu().to(dtype = torch.uint8) for rng in state["torch_cuda"]])

    mps_backend = getattr(torch, "mps", None)
    mps_available = getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()
    if mps_available and "torch_mps" in state and mps_backend is not None and hasattr(mps_backend, "set_rng_state"):
        try:
            mps_backend.set_rng_state(state["torch_mps"].detach().cpu().to(dtype = torch.uint8))
        except (AttributeError, RuntimeError, TypeError):
            pass

if __name__ == "__main__":
    set_seed(42)
    a = torch.randn(2, 2)
    
    set_seed(42)
    b = torch.randn(2, 2)
    
    print(f"Seeds matched: {torch.equal(a, b)}")
