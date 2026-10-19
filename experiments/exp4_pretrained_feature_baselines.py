"""Frozen PANNs/BirdNET embeddings with a linear probe on the existing BirdCLEF splits."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bioacoustic_data import BirdCLEFDataset
from utils.audio_io import (
    DEFAULT_EVALUATION_MINIMUM_RMS,
    EVALUATION_CROP_POLICY,
    get_resampler,
    load_evaluation_center_crop,
    load_fixed_crops,
)
from utils.dataset_manifest import validate_manifest_file_index
from utils.seed import set_seed



EXPECTED_PANNS_CNN14_MD5 = "541141fa2ee191a88f24a3219fff024e"  # Zenodo record 3987831
# SHA-256 of the same 327,428,481-byte artifact, cross checked from its
# content addressed Hugging Face mirror. Both hashes are checked at runtime.
EXPECTED_PANNS_CNN14_SHA256 = "0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31"
FEATURE_EXTRACTION_SPEC_VERSION = "v1.6.3-external-feature-pooling-unified-center-recovery-v4"


def crop_policy_description(num_fixed_crops: int) -> str:
    """Return an auditable label for the exact temporal-crop policy."""
    if num_fixed_crops == 1:
        return EVALUATION_CROP_POLICY
    if num_fixed_crops == 3:
        return "three-temporal-crops-rms-recovery-mean-pooling-v2"
    raise ValueError("num_fixed_crops must be 1 (primary recovered center crop) or 3 (exploratory multi-crop).")


def sha256_file(path: str | Path) -> str:
    """Compute a streaming SHA-256 digest for a model artifact."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def md5_file(path: str | Path) -> str:
    """Compute MD5 for matching the checksum published by the PANNs Zenodo record."""
    digest = hashlib.md5()

    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _existing_model_files(value: Any, *, depth: int = 0, seen: set[int] | None = None) -> list[Path]:
    """Find ONNX model files exposed through a loaded BirdNET model object."""
    if seen is None:
        seen = set()
    if depth > 4 or value is None:
        return []
    
    if isinstance(value, (str, Path)):
        candidate = Path(value).expanduser()
        if candidate.is_file() and candidate.suffix.lower() in {".onnx", ".pt", ".pth", ".tflite"}:
            return [candidate.resolve()]
        if candidate.is_dir() and depth < 2:
            return sorted(
                p.resolve() for p in candidate.rglob("*.onnx") if p.is_file()
            )
        return []
    
    if id(value) in seen:
        return []
    
    seen.add(id(value))
    found: list[Path] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if any(token in str(key).lower() for token in ("path", "model", "backend", "session")):
                found.extend(_existing_model_files(child, depth = depth + 1, seen = seen))

        return list(dict.fromkeys(found))
    
    if isinstance(value, (list, tuple)):
        for child in value:
            found.extend(_existing_model_files(child, depth = depth + 1, seen = seen))

        return list(dict.fromkeys(found))
    
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        for key, child in attrs.items():
            if any(token in key.lower() for token in ("path", "model", "backend", "session")):
                found.extend(_existing_model_files(child, depth = depth + 1, seen = seen))

    return list(dict.fromkeys(found))


def _birdnet_model_artifact(model: Any) -> tuple[Path, str]:
    candidates = _existing_model_files(model)
    onnx = [path for path in candidates if path.suffix.lower() == ".onnx"]

    if not onnx:
        roots = []
        if os.environ.get("BIRDNET_APP_DATA"):
            roots.append(Path(os.environ["BIRDNET_APP_DATA"]).expanduser())

        roots.extend((Path.home() / ".local" / "share" / "birdnet", Path.home() / ".cache" / "birdnet"))
        for root in roots:
            if root.is_dir():
                matches = sorted(p.resolve() for p in root.rglob("*.onnx") if p.is_file())
                if len(matches) == 1:
                    onnx = matches
                    break

    if len(onnx) != 1:
        raise RuntimeError(
            "Could not identify exactly one loaded BirdNET acoustic v3.0 ONNX model file for hashing. "
            "Set BIRDNET_APP_DATA to the persistent BirdNET model cache and inspect the model path exposed by the installed library."
        )
    
    return onnx[0], sha256_file(onnx[0])

def fraction_tag(fraction: float) -> str:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1].")
    return str(float(fraction))


def pool_segment_embeddings(raw: Any) -> np.ndarray:
    """Mean-pool valid BirdNET segments into one clip level feature vector."""
    embeddings = getattr(raw, "embeddings", raw)
    mask = getattr(raw, "embeddings_masked", None)

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    array = np.asarray(embeddings, dtype=np.float32)
    if array.size == 0:
        raise ValueError("BirdNET returned an empty embedding array.")
    if array.ndim < 1 or array.shape[-1] < 1:
        raise ValueError(f"Invalid BirdNET embedding shape {array.shape}; expected a non empty feature dimension.")

    if mask is not None:
        if array.ndim < 2:
            raise ValueError(f"Invalid masked BirdNET embedding shape {array.shape}; expected segment and feature dimensions.")
        
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        mask_array = np.asarray(mask)
        if mask_array.dtype.kind != "b":
            raise ValueError(
                f"BirdNET embeddings_masked must be a Boolean mask; received dtype {mask_array.dtype}."
            )
        
        if mask_array.shape != array.shape:
            raise ValueError(
                f"BirdNET embedding mask shape {mask_array.shape} does not match embeddings {array.shape}."
            )

        feature_dim = array.shape[-1]
        vectors = array.reshape(-1, feature_dim)
        mask_vectors = mask_array.reshape(-1, feature_dim)
        valid_rows = np.all(~mask_vectors, axis = 1)
        invalid_rows = np.all(mask_vectors, axis = 1)

        if not np.all(valid_rows | invalid_rows):
            raise ValueError("BirdNET validity mask partially masks a segment's feature vector.")
        
        vectors = vectors[valid_rows]
        if vectors.shape[0] == 0:
            raise ValueError("BirdNET returned no valid embedding segments after masking.")
        
        if not np.isfinite(vectors).all():
            raise FloatingPointError("BirdNET returned non-finite values in valid embeddings.")
        pooled = vectors.mean(axis = 0)

    else:
        array = np.squeeze(array)
        if array.ndim == 0:
            raise ValueError(f"Invalid BirdNET embedding shape {array.shape}.")
        if array.ndim == 1:
            pooled = array
        else:
            pooled = array.reshape(-1, array.shape[-1]).mean(axis = 0)

    if pooled.ndim != 1 or not np.isfinite(pooled).all():
        raise ValueError("BirdNET embedding pooling produced invalid features.")
    
    return pooled.astype(np.float32, copy = False)


class PannsCnn14Encoder:
    """AudioSet pretrained PANNs Cnn14 2048-D embedding extractor."""
    key = "panns"
    input_sample_rate = 32000
    feature_dim = 2048

    def __init__(self, device: str, checkpoint: str | None = None):
        if not checkpoint:
            raise ValueError(
                "PANNs requires --checkpoint pointing to a pre-downloaded local Cnn14 .pth file. "
                "Automatic/default checkpoint downloads are disabled so offline compute nodes do not fail unexpectedly."
            )
        
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"PANNs checkpoint does not exist: '{checkpoint_path}'.")
        try:
            from panns_inference import AudioTagging
            self.package_version = importlib.metadata.version("panns-inference")
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            raise RuntimeError(
                "PANNs baseline requires the optional 'panns-inference==0.1.1' package. "
                "Install it in the experiment environment using requirements.txt."
            ) from exc
        
        if self.package_version != "0.1.1":
            raise RuntimeError(
                f"Expected panns-inference==0.1.1 for reproducibility, found {self.package_version}."
            )
        self.device = device

        # The checkpoint is validated and hashed before constructing the model;
        # this avoids a default download on offline compute nodes.
        self.checkpoint = str(checkpoint_path)
        self.checkpoint_md5 = md5_file(checkpoint_path)
        if self.checkpoint_md5.lower() != EXPECTED_PANNS_CNN14_MD5:
            raise ValueError(
                f"PANNs checkpoint MD5 mismatch for '{checkpoint_path}': expected the official Cnn14_mAP=0.431.pth "
                f"digest {EXPECTED_PANNS_CNN14_MD5}, got {self.checkpoint_md5}. Re-download the official artifact."
            )
        
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        if self.checkpoint_sha256.lower() != EXPECTED_PANNS_CNN14_SHA256:
            raise ValueError(
                f"PANNs checkpoint SHA-256 mismatch for '{checkpoint_path}': expected {EXPECTED_PANNS_CNN14_SHA256}, "
                f"got {self.checkpoint_sha256}. Re-download the official Cnn14_mAP=0.431.pth artifact."
            )
        
        self.model = AudioTagging(checkpoint_path = str(checkpoint_path), device=device)
        self.model_name = "PANNs Cnn14 AudioSet pretrained"
        self.identity = {
            "model_key": self.key, 
            "model_name": self.model_name,
            "model_version": "Cnn14_mAP=0.431", 
            "backend": "panns-inference",
            "package_version": self.package_version, 
            "input_sample_rate": self.input_sample_rate,
            "feature_dim": self.feature_dim, 
            "model_file_name": checkpoint_path.name,
            "model_file_sha256": self.checkpoint_sha256, 
            "model_file_md5": self.checkpoint_md5,
            "official_model_file_md5": EXPECTED_PANNS_CNN14_MD5,
            "official_model_file_sha256": EXPECTED_PANNS_CNN14_SHA256, 
            "runtime_device": self.device,
            "model_source_url": "https://zenodo.org/records/3987831",
            "license_status": "review exact license/terms accompanying the selected Cnn14 checkpoint before distribution or commercial use",
            "license_note": "Review the exact checkpoint terms and cite the upstream PANNs repository and model record.",
        }

    def encode_batch(self, waves: torch.Tensor, sample_rate: int) -> np.ndarray:
        waves = waves.detach().cpu().to(torch.float32)

        if sample_rate != self.input_sample_rate:
            waves = get_resampler(sample_rate, self.input_sample_rate)(waves)

        audio = waves.squeeze(1).numpy()
        _, embeddings = self.model.inference(audio)
        features = np.asarray(embeddings, dtype = np.float32)

        if features.ndim != 2 or features.shape[0] != waves.shape[0]:
            raise ValueError(f"Unexpected PANNs embedding shape {features.shape} for batch {waves.shape[0]}.")
        
        if not np.isfinite(features).all():
            raise FloatingPointError("PANNs returned non finite embeddings.")
        
        return features

class BirdNetV3Encoder:
    """BirdNET acoustic v3.0 embeddings using the pinned official ONNX API."""
    key = "birdnet"
    input_sample_rate = 32000
    feature_dim = 1280
    expected_package_version = "1.1.1"

    def __init__(self, device: str = "CPU", checkpoint: str | None = None):
        if checkpoint is not None:
            raise ValueError("BirdNET v3 weights are managed by the official library; --checkpoint is PANNs-only.")
        
        try:
            import birdnet
        except ImportError as exc:
            raise RuntimeError(
                "BirdNET baseline requires birdnet[onnx]==1.1.1. Use the isolated Python 3.11+ environment in "
                "requirements.txt."
            ) from exc
        
        self.package_version = importlib.metadata.version("birdnet")
        if self.package_version != self.expected_package_version:
            raise RuntimeError(
                f"BirdNET integration is pinned to birdnet == {self.expected_package_version}; found {self.package_version}. "
                "Install the exact optional requirements file before extraction."
            )
        self.device = device
        self.model = birdnet.load("acoustic", "3.0", "onnx")
        if not callable(getattr(self.model, "encode_arrays", None)):
            raise RuntimeError(
                "The loaded BirdNET model does not expose encode_arrays; refusing to use an unverified embedding API. "
                "Check that birdnet==1.1.1 is installed."
            )
        
        get_dim = getattr(self.model, "get_embeddings_dim", None)
        if not callable(get_dim):
            raise RuntimeError("The loaded BirdNET model does not expose documented get_embeddings_dim().")
        
        self.feature_dim = int(get_dim())
        if self.feature_dim < 1:
            raise RuntimeError(f"BirdNET reported an invalid embedding dimension: {self.feature_dim}.")
        
        get_segment = getattr(self.model, "get_segment_size_s", None)
        self.segment_size_sec = float(get_segment()) if callable(get_segment) else 3.0
        if abs(self.segment_size_sec - 3.0) > 1e-6:
            raise RuntimeError(f"Expected BirdNET v3.0 3-second segments; model reports {self.segment_size_sec}.")
        
        model_path, model_sha256 = _birdnet_model_artifact(self.model)
        self.model_name = "BirdNET acoustic v3.0 embeddings"
        self.checkpoint = str(model_path)
        self.checkpoint_sha256 = model_sha256

        self.identity = {
            "model_key": self.key, 
            "model_name": self.model_name, 
            "model_version": "3.0",
            "backend": "onnx", 
            "package_version": self.package_version,
            "input_sample_rate": self.input_sample_rate, 
            "segment_size_sec": self.segment_size_sec,
            "feature_dim": self.feature_dim, 
            "model_file_name": model_path.name,
            "model_file_sha256": model_sha256, 
            "runtime_device": self.device,
            "model_license": "CC BY-NC-SA 4.0",
            "model_license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "model_license_source": "BirdNET Python library model terms documented for the pinned 1.1.1 release",
            "model_license_status": "library-documented-terms; verify accompanying license for the exact downloaded ONNX artifact",
            "license_note": "V3.0 is a preview model. Verify exact downloaded model terms and intended-use compatibility before use; cite BirdNET and its official model documentation.",
        }

    def encode_batch(self, waves: torch.Tensor, sample_rate: int) -> np.ndarray:
        if sample_rate != self.input_sample_rate:
            waves = get_resampler(sample_rate, self.input_sample_rate)(waves)
            sample_rate = self.input_sample_rate

        pooled = []
        for wave in waves:
            audio = wave.detach().cpu().to(torch.float32).squeeze(0).numpy()
            result = self.model.encode_arrays(
                [(audio, int(sample_rate))], batch_size = 1, device = self.device,
            )
            pooled.append(pool_segment_embeddings(result))

        features = np.stack(pooled).astype(np.float32, copy = False)
        if features.ndim != 2 or features.shape[0] != waves.shape[0]:
            raise ValueError(f"Unexpected BirdNET pooled embedding shape {features.shape}.")
        
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"BirdNET reported {self.feature_dim} embedding dimensions but returned {features.shape[1]}."
            )
        
        if not np.isfinite(features).all():
            raise FloatingPointError("BirdNET returned non finite embeddings.")
        
        return features

def build_encoder(model_key: str, device: torch.device, checkpoint: str | None = None, *, birdnet_device: str = "CPU"):
    if model_key == "panns":
        return PannsCnn14Encoder("cuda" if device.type == "cuda" else "cpu", checkpoint)
    
    if model_key == "birdnet":
        # Use CPU by default: not every birdnet[onnx] environment includes a
        # CUDA enabled ONNX Runtime provider. Users can explicitly select GPU:0.
        return BirdNetV3Encoder(birdnet_device, checkpoint)
    raise ValueError(f"Unsupported external model '{model_key}'.")


def load_external_baseline_crops(dataset: BirdCLEFDataset, filepath: str | Path, target_sr: int, num_fixed_crops: int) -> torch.Tensor:
    """Load external baseline crops under a documented matched evaluation policy."""
    minimum_rms = getattr(dataset, "minimum_rms", None)

    if minimum_rms is None:
        minimum_rms = DEFAULT_EVALUATION_MINIMUM_RMS

    if num_fixed_crops == 1:
        center = load_evaluation_center_crop(
            filepath, 
            target_sr = target_sr, 
            duration_sec = dataset.duration_sec,
            minimum_rms = float(minimum_rms),
        )
        return center.unsqueeze(0).contiguous()
    
    if num_fixed_crops == 3:
        return load_fixed_crops(
            filepath, 
            target_sr = target_sr, 
            duration_sec = dataset.duration_sec,
            num_crops = 3, 
            minimum_rms = float(minimum_rms),
        )
    raise ValueError("num_fixed_crops must be 1 (primary recovered center crop) or 3 (exploratory multi crop).")

def encode_features(encoder, dataset: BirdCLEFDataset, selected_names: list[str], *, batch_size: int, cache_path: Path, fingerprint: str, num_fixed_crops: int = 1):
    """Extract or reuse a fingerprinted cache; flock prevents concurrent SLURM races."""

    cache_path.parent.mkdir(parents = True, exist_ok = True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")

    def load_cache_if_valid():
        if not cache_path.is_file():
            return None
        
        with np.load(cache_path, allow_pickle = False) as cached:
            cached_paths = cached["relative_paths"].astype(str).tolist()
            cached_fingerprint = str(cached["fingerprint"].item())
            cached_features = cached["features"].astype(np.float32, copy = True)
            cached_crop_counts = cached["crop_counts"].astype(np.int32, copy = True) if "crop_counts" in cached else None

        if cached_paths != selected_names or cached_fingerprint != fingerprint:
            raise ValueError(
                f"Feature cache '{cache_path}' does not match current file inventory/fingerprint; "
                "move it aside before regenerating."
            )
        
        if cached_features.ndim != 2 or len(cached_features) != len(selected_names) or not np.isfinite(cached_features).all():
            raise FloatingPointError(f"Feature cache '{cache_path}' has an invalid matrix {cached_features.shape}.")
        
        if cached_crop_counts is None or cached_crop_counts.shape != (len(selected_names),) or np.any(cached_crop_counts < 1):
            raise ValueError(f"Feature cache '{cache_path}' lacks valid fixed-crop counts; move it aside and regenerate.")
        
        return cached_features, cached_crop_counts
    
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        cached = load_cache_if_valid()
        if cached is not None:
            return cached

        path_to_index = dict(zip(dataset.relative_paths, range(len(dataset.relative_paths))))
        features: list[np.ndarray] = []
        crop_counts: list[int] = []
        encoder_sr = int(getattr(encoder, "input_sample_rate", dataset.sample_rate))

        for start in range(0, len(selected_names), batch_size):
            batch_names = selected_names[start:start + batch_size]
            crop_waves: list[torch.Tensor] = []
            owners: list[int] = []
            local_counts = []

            for owner, name in enumerate(batch_names):
                idx = path_to_index[name]
                crops = load_external_baseline_crops(
                    dataset, dataset.file_list[idx], encoder_sr, num_fixed_crops,
                )
                local_counts.append(int(crops.shape[0]))

                for crop in crops:
                    crop_waves.append(crop)
                    owners.append(owner)

            extracted_parts = []
            max_model_batch = max(batch_size, 1)

            for crop_start in range(0, len(crop_waves), max_model_batch):
                wave_batch = torch.stack(crop_waves[crop_start:crop_start + max_model_batch], dim = 0)
                extracted_parts.append(encoder.encode_batch(wave_batch, encoder_sr))

            extracted_crops = np.concatenate(extracted_parts, axis = 0)
            if extracted_crops.shape[0] != len(crop_waves):
                raise ValueError(f"Encoder emitted {extracted_crops.shape[0]} rows for {len(crop_waves)} fixed crops.")
            
            owners_arr = np.asarray(owners, dtype = np.int64)
            for owner, count in enumerate(local_counts):
                features.append(extracted_crops[owners_arr == owner].mean(axis = 0, dtype = np.float64).astype(np.float32))
                crop_counts.append(count)

            print(f"[{encoder.key}] cached fixed crop features {min(start + len(batch_names), len(selected_names))}/{len(selected_names)}", flush = True)

        feature_array = np.asarray(features, dtype = np.float32)
        crop_count_array = np.asarray(crop_counts, dtype = np.int32)

        if feature_array.ndim != 2 or feature_array.shape[0] != len(selected_names) or not np.isfinite(feature_array).all():
            raise FloatingPointError(f"Invalid extracted feature matrix: {feature_array.shape}.")
        
        tmp_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
        with tmp_path.open("wb") as stream:
            np.savez_compressed(stream, features = feature_array, 
                                crop_counts = crop_count_array,
                                relative_paths = np.asarray(selected_names, dtype = str), 
                                fingerprint = np.asarray(fingerprint))
            
        os.replace(tmp_path, cache_path)
        return feature_array, crop_count_array


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--model", choices = ["panns", "birdnet"], required = True)
    parser.add_argument("--config", default = "configs/linear_probe.yaml")
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--fraction", type=float, default = 0.1)
    parser.add_argument("--checkpoint", default=None, help = "Required pre downloaded local PANNs Cnn14 .pth checkpoint; not used by BirdNET.")
    parser.add_argument("--birdnet-device", choices = ["CPU", "GPU:0"], default = "CPU", help = "BirdNET ONNX device; GPU requires a compatible GPU-enabled ONNX Runtime.")
    parser.add_argument("--batch-size", type=int, default = 16)
    parser.add_argument("--num-fixed-crops", type=int, choices = [1, 3], default = 1,
                        help = "Primary comparison uses one center crop to match in-repo probes; 3-crop averaging is exploratory and must be reported separately.")
    parser.add_argument("--epochs", type=int, default = None)
    parser.add_argument("--learning-rate", type=float, default = None)
    parser.add_argument("--cache-dir", default = "data/processed/external_features")

    args = parser.parse_args(argv)
    if not (0 < args.fraction <= 1):
        raise ValueError("--fraction must be in (0,1].")
    if args.model == "birdnet" and args.checkpoint is not None:
        raise ValueError("--checkpoint is supported only for PANNs; BirdNET v3 uses the library-managed official model.")
    if args.model == "panns" and args.checkpoint is None:
        raise ValueError(
            "--checkpoint is required for PANNs. Download Cnn14_mAP=0.431.pth before using an offline compute node."
        )
    
    if args.checkpoint is not None and not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"External model checkpoint does not exist: '{args.checkpoint}'.")
    
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    
    set_seed(args.seed)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding = "utf-8"))
    data_cfg, exp_cfg = cfg["dataset"], cfg["experiment"]

    dataset = BirdCLEFDataset(
        data_cfg["birdclef_dir"], 
        sample_rate = int(data_cfg["sample_rate"]),
        duration_sec = float(data_cfg["clip_sec"]), 
        crop_mode = "center",
    )

    split_path = Path("checkpoints") / f"data_split_s{args.seed}.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split manifest '{split_path}' missing. Create and validate the regular BirdCLEF splits first.")
    
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if int(split.get("schema_version", -1)) != 5:
        raise ValueError(f"Split manifest '{split_path}' uses an older crop/preflight schema.")
    
    minimum_rms = float(split.get("silence_minimum_rms", DEFAULT_EVALUATION_MINIMUM_RMS))
    if not torch.isfinite(torch.tensor(minimum_rms)) or minimum_rms <= 0:
        raise ValueError(f"Split manifest '{split_path}' contains an invalid silence_minimum_rms.")
    
    dataset.minimum_rms = minimum_rms
    if int(split.get("seed", -1)) != args.seed or split.get("label_map") != dataset.label_map:
        raise ValueError("Current BirdCLEF inventory/labels do not match the saved split manifest.")
    
    validate_manifest_file_index(split, dataset.relative_paths, dataset.labels, context = f"external {args.model} probe")
    active_label_map = split.get("active_label_map", dataset.label_map)
    old_to_new = {int(old_id): new_id for new_id, (_, old_id) in enumerate(sorted(active_label_map.items(), key = lambda x: int(x[1])))}

    fraction = f"{args.fraction:.2f}"
    if fraction not in split.get("train_fraction_filenames", {}):
        raise ValueError(f"Fraction {fraction} not in manifest; choices: {list(split.get('train_fraction_filenames', {}))}.")
    
    train_names = set(split["train_fraction_filenames"][fraction])
    val_names = list(split["validation_filenames"])
    test_names = list(split["test_filenames"])
    selected_allowed = train_names | set(split["train_filenames"]) | set(val_names) | set(test_names)
    
    # Exclude rare-policy-excluded classes from feature extraction to avoid wasting time.
    selected_names = [name for name, label in zip(dataset.relative_paths, dataset.labels) if name in selected_allowed and label in old_to_new]
    if not selected_names:
        raise ValueError("No active labelled files overlap the split manifest.")
    
    split_names = set(split["train_filenames"]) | set(val_names) | set(test_names)

    missing_split_files = split_names - set(dataset.relative_paths)
    if missing_split_files:
        raise ValueError(f"Split references missing audio files: {sorted(missing_split_files)[:5]}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_encoder(args.model, device, args.checkpoint, birdnet_device=args.birdnet_device)
    encoder_identity = dict(encoder.identity)

    import hashlib as _hashlib
    fingerprint_raw = json.dumps({
        "dataset_fingerprint": split.get("dataset_fingerprint"),
        "selected_names": selected_names, 
        "model": args.model,
        "encoder_identity": encoder_identity,
        "feature_extraction_spec_version": FEATURE_EXTRACTION_SPEC_VERSION,
        "num_fixed_crops": args.num_fixed_crops,
        "crop_policy": crop_policy_description(args.num_fixed_crops),
        "minimum_rms": minimum_rms,
        "sample_rate": int(data_cfg["sample_rate"]), 
        "duration_sec": float(data_cfg["clip_sec"]),
    }, sort_keys = True, separators = (",", ":")).encode()

    feature_fingerprint = _hashlib.sha256(fingerprint_raw).hexdigest()
    cache_path = Path(args.cache_dir) / f"{args.model}_{feature_fingerprint[:20]}.npz"

    features, cached_crop_counts = encode_features(
        encoder, 
        dataset, 
        selected_names, 
        batch_size = args.batch_size, 
        cache_path = cache_path,
        fingerprint = feature_fingerprint, 
        num_fixed_crops = args.num_fixed_crops,
    )
    
    if features.shape[1] != int(encoder.feature_dim):
        print(f"Warning: encoder reported feature_dim = {encoder.feature_dim}; cache uses {features.shape[1]}.", flush = True)

    path_to_feature = {name: i for i, name in enumerate(selected_names)}
    path_to_label = {name: old_to_new[int(dataset.labels[idx])] for idx, name in enumerate(dataset.relative_paths) if int(dataset.labels[idx]) in old_to_new}

    def indices_for(names, *, skip_missing = False):
        out = []
        missing = []

        for name in names:
            if name not in path_to_feature or name not in path_to_label:
                missing.append(name)
            else:
                out.append(path_to_feature[name])

        if missing and not skip_missing:
            raise ValueError(f"Selected feature set lacks split file names: {missing[:5]}.")
        
        return out

    train_indices = indices_for(list(split["train_fraction_filenames"][fraction]))
    val_indices = indices_for(val_names)
    test_indices = indices_for(test_names)

    if min(len(train_indices), len(val_indices), len(test_indices)) < 1:
        raise ValueError("External model probe has an empty train/validation/test split.")
    
    labels_by_feature = np.asarray([path_to_label[name] for name in selected_names], dtype = np.int64)

    x = torch.from_numpy(features).float()
    y = torch.from_numpy(labels_by_feature).long()
    train_x, train_y = x[train_indices], y[train_indices]
    val_x, val_y = x[val_indices], y[val_indices]
    test_x, test_y = x[test_indices], y[test_indices]

    output_label_map = {
        name: i for i, (name, _old_id) in enumerate(
            sorted(active_label_map.items(), key = lambda item: int(item[1]))
        )
    }

    num_classes = len(output_label_map)
    head = nn.Linear(features.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr = float(args.learning_rate or exp_cfg["learning_rate"]))
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size = int(args.batch_size), shuffle = True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size = int(args.batch_size), shuffle = False)
    best_state = None
    best_val_loss = float("inf")

    epochs = int(args.epochs or exp_cfg["epochs"])
    for epoch in range(1, epochs + 1):
        head.train()
        loss_sum = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none = True)
            logits = head(xb)

            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non finite external baseline loss at epoch {epoch}.")
            
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(yb)
            n_train += len(yb)

        head.eval()
        val_logits, val_losses = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                logits = head(xb.to(device))
                val_losses.append(float(criterion(logits, yb.to(device)).item()) * len(yb))
                val_logits.extend(logits.cpu().numpy())

        val_loss = sum(val_losses) / len(val_y)
        val_acc = float(accuracy_score(val_y.numpy(), np.asarray(val_logits).argmax(axis=1)))

        print(f"[{args.model}] epoch {epoch}/{epochs} train_loss = {loss_sum / max(n_train, 1):.5f} val_loss = {val_loss:.5f} val_acc = {val_acc:.4f}", flush = True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best external baseline head selected.")
    
    head.load_state_dict(best_state)
    head.eval()

    with torch.no_grad():
        test_logits = head(test_x.to(device)).cpu().numpy()

    test_pred = test_logits.argmax(axis=1)
    test_true = test_y.numpy()

    metrics = {
        "test_accuracy": float(accuracy_score(test_true, test_pred)),
        "test_macro_f1": float(f1_score(test_true, test_pred, average = "macro", zero_division = 0)),
        "test_balanced_accuracy": float(balanced_accuracy_score(test_true, test_pred)),
        "test_loss": float(criterion(torch.from_numpy(test_logits), test_y).item()),
    }

    os.makedirs("checkpoints", exist_ok = True)
    os.makedirs("results", exist_ok = True)

    tag = fraction_tag(args.fraction)
    head_path = Path("checkpoints") / f"pretrained_head_{args.model}_s{args.seed}_frac{tag}.pt"
    tmp_head = head_path.with_suffix(head_path.suffix + ".tmp")

    torch.save({
        "model_state": best_state, 
        "label_map": output_label_map,
        "n_active_classes": len(output_label_map),
        "n_total_classes": len(split.get("label_map", dataset.label_map)),
        "n_excluded_classes": len(split.get("excluded_classes", [])),
        "excluded_classes": split.get("excluded_classes", []),
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
        "class_scope_note": split.get("class_scope_note", "Metrics cover active classes only."),
        "model_key": args.model, 
        "model_name": encoder.model_name, 
        "feature_dim": int(features.shape[1]),
        "seed": args.seed, 
        "fraction": args.fraction, 
        "dataset_fingerprint": split.get("dataset_fingerprint"),
        "feature_fingerprint": feature_fingerprint, 
        "feature_cache": str(cache_path), 
        "external_checkpoint": encoder.checkpoint,
        "num_fixed_crops_requested": args.num_fixed_crops,
        "fixed_crop_count_min": int(cached_crop_counts.min()),
        "fixed_crop_count_mean": float(cached_crop_counts.mean()),
        "fixed_crop_count_max": int(cached_crop_counts.max()),
        "fixed_crop_shortfall_filenames": [name for name, count in zip(selected_names, cached_crop_counts) if int(count) < args.num_fixed_crops],
        "fixed_crop_aggregation": "mean of per crop embeddings to one recording level row",
        "external_checkpoint_sha256": encoder.checkpoint_sha256, 
        "encoder_identity": encoder_identity,
    }, tmp_head)

    os.replace(tmp_head, head_path)
    result_path = Path("results") / f"pretrained_probe_{args.model}_s{args.seed}_frac{tag}.json"

    result = {
        "schema_version": 3, 
        "mode": "external_pretrained_linear_probe", 
        "variant": args.model,
        "seed": args.seed, 
        "fraction": args.fraction, **metrics,
        "n_train": len(train_indices), 
        "n_validation": len(val_indices), 
        "n_test": len(test_indices),
        "num_classes": num_classes, 
        "active_label_map": output_label_map,
        "n_active_classes": len(output_label_map),
        "n_total_classes": len(split.get("label_map", dataset.label_map)),
        "n_excluded_classes": len(split.get("excluded_classes", [])),
        "excluded_classes": split.get("excluded_classes", []),
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
        "class_scope_note": split.get("class_scope_note", "Metrics cover active classes only."),
        "split_method": split.get("split_method"), 
        "grouping_source": dataset.grouping_source,
        "dataset_fingerprint": split.get("dataset_fingerprint"), 
        "feature_fingerprint": feature_fingerprint,
        "num_fixed_crops_requested": args.num_fixed_crops,
        "fixed_crop_aggregation": "mean of per crop embeddings to one recording level row",
        "crop_policy": crop_policy_description(args.num_fixed_crops),
        "feature_extraction_spec_version": FEATURE_EXTRACTION_SPEC_VERSION,
        "minimum_rms": minimum_rms,
        "fixed_crop_count_min": int(cached_crop_counts.min()),
        "fixed_crop_count_mean": float(cached_crop_counts.mean()),
        "fixed_crop_count_max": int(cached_crop_counts.max()),
        "fixed_crop_shortfall_filenames": [name for name, count in zip(selected_names, cached_crop_counts) if int(count) < args.num_fixed_crops],
        "model_name": encoder.model_name, 
        "external_checkpoint": encoder.checkpoint,
        "external_checkpoint_sha256": encoder.checkpoint_sha256, 
        "encoder_identity": encoder_identity,
        "feature_cache": str(cache_path), 
        "linear_head_checkpoint": str(head_path),
        "metrics_note": "Frozen pretrained embeddings; a trained linear classifier. Compare only with matched split, class scope, crop policy and label fraction.",
    }

    tmp_result = result_path.with_suffix(result_path.suffix + ".tmp")
    tmp_result.write_text(json.dumps(result, indent = 2), encoding = "utf-8")

    os.replace(tmp_result, result_path)
    print(json.dumps({"metrics": metrics, "head": str(head_path), "result": str(result_path)}, indent = 2))


if __name__ == "__main__":
    main()