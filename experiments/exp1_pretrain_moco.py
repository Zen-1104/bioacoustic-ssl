"""MoCo style self supervised pretraining for bioacoustic representations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bioacoustic_data import NoiseMixer, XenoCantoDataset, BirdCLEFDataset
from evaluation.knn_probe_eval import knn_probe_accuracy
from evaluation.metrics import info_nce_loss
from models.backbone_cnn10 import CNN_10_Backbone
from models.moco_wrapper import MoCo_Wrapper
from models.projection_head import Projection_Head
from utils.audio_io import DEFAULT_EVALUATION_MINIMUM_RMS, load_and_preprocess_audio
from utils.dataset_manifest import fingerprint_relative_paths, validate_manifest_file_index
from utils.habitat_split import read_verified_habitat_csv
from utils.logger import get_logger
from utils.seed import set_seed, capture_rng_state, restore_rng_state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--variant", type=str, default = "habitat", choices = ["habitat", "standard", "white_noise"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help = "Resume from the variant latest checkpoint and append reconciled JSONL logs.")
    parser.add_argument("--bn-policy", choices = MoCo_Wrapper.BN_POLICIES, default = None,
                        help = "Override model.bn_policy. Explicitly supplied policies use separate *_bn-POLICY checkpoints for controlled BN ablations.")
    parser.add_argument("--view-batch-policy", choices = MoCo_Wrapper.VIEW_BATCH_POLICIES, default=None,
                        help="Use separate query-view forwards or one concatenated query-view forward to directly test the BatchNorm train/eval-statistics confound.")
    return parser.parse_args(argv)


def load_noise_pool(split_csv_path, esc50_audio_dir, sample_rate, duration_sec):
    split_path = Path(split_csv_path)

    if not split_path.is_file():
        raise FileNotFoundError(f"Habitat split CSV missing: '{split_path}'. Run create_habitat_split.py before training.")
    
    df = read_verified_habitat_csv(
        split_path,
        esc50_audio_dir,
        expected_partition = "seen",
    )

    habitat_manifest_path = split_path.parent / "esc50_habitat_split_manifest.json"
    habitat_manifest = json.loads(habitat_manifest_path.read_text(encoding = "utf-8"))
    unseen_categories = set(habitat_manifest.get("unseen_categories", []))
    pool_categories = set(df["category"].astype(str)) if "category" in df.columns else set()
    leaked_categories = pool_categories & unseen_categories

    if leaked_categories:
        raise RuntimeError(
            "Seen habitat pretraining pool contains held out unseen categories: "
            f"{sorted(leaked_categories)}. Recreate and verify the habitat split before training."
        )

    if "filename" not in df.columns or df.empty:
        raise ValueError(
            f"Habitat noise CSV '{split_path}' has no filename rows."
        )
    
    root = Path(esc50_audio_dir)
    pool = []

    for filename in sorted(df["filename"].astype(str).tolist()):
        candidates = (root / filename, root / "audio" / filename)
        filepath = next((p for p in candidates if p.is_file()), None)

        if filepath is None:
            raise FileNotFoundError(f"Noise audio '{filename}' not found under '{root}'.")
        
        wave = load_and_preprocess_audio(
            filepath, 
            target_sr = sample_rate, 
            duration_sec = duration_sec,
            crop_mode = "center", 
            short_audio_policy = "repeat",
        )
        pool.append(wave)

    if not pool:
        raise ValueError(f"No noise waveforms loaded from '{split_path}'.")
    return pool


def load_split_for_monitor(path: Path, dataset: BirdCLEFDataset, seed: int):

    if not path.is_file():
        raise FileNotFoundError(f"Split manifest '{path}' is required for the knn monitor. Run scripts/data/create_birdclef_splits.py first.")
    
    manifest = json.loads(path.read_text(encoding = "utf-8"))

    if int(manifest.get("schema_version", -1)) != 5:
        raise ValueError(f"knn monitor split '{path}' uses an older crop schema.")   
    if int(manifest.get("seed", -1)) != seed:
        raise ValueError(f"knn monitor split seed mismatch in '{path}'.")   
    if manifest.get("label_map") != dataset.label_map:
        raise ValueError("knn monitor BirdCLEF label map differs from its saved split manifest.")
    
    minimum_rms = float(manifest.get("silence_minimum_rms", DEFAULT_EVALUATION_MINIMUM_RMS))

    if not torch.isfinite(torch.tensor(minimum_rms)) or minimum_rms <= 0:
        raise ValueError(f"knn monitor split '{path}' contains an invalid silence_minimum_rms.")
    
    dataset.minimum_rms = minimum_rms

    validate_manifest_file_index(manifest, dataset.relative_paths, dataset.labels, context = "knn monitor")
    return manifest


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr = subprocess.DEVNULL, text = True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def atomic_torch_save(payload, path: str | Path):
    path = str(path)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main(argv = None):
    args = parse_args(argv)

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Configuration file missing: '{args.config}'.")
    
    cfg = yaml.safe_load(Path(args.config).read_text(encoding = "utf-8"))
    data_cfg, model_cfg, train_cfg = cfg["dataset"], cfg["model"], cfg["training"]
    deterministic = bool(train_cfg.get("deterministic", False))
    deterministic_warn_only = bool(train_cfg.get("deterministic_warn_only", False))

    set_seed(args.seed, 
             deterministic = deterministic, 
             deterministic_warn_only = deterministic_warn_only)

    logger = get_logger(f"pretrain_{args.variant}_{args.seed}")
    logger.info("deterministic_algorithms=%s deterministic_warn_only=%s (warn-only runs are not strict determinism)", 
                deterministic, 
                deterministic_warn_only)
    bn_policy = args.bn_policy or model_cfg.get("bn_policy", "copy_query")

    if bn_policy not in MoCo_Wrapper.BN_POLICIES:
        raise ValueError(f"Invalid model.bn_policy {bn_policy!r}; expected one of {MoCo_Wrapper.BN_POLICIES}.")
    
    view_batch_policy = args.view_batch_policy or model_cfg.get("view_batch_policy", "separate")

    if view_batch_policy not in MoCo_Wrapper.VIEW_BATCH_POLICIES:
        raise ValueError(f"Invalid model.view_batch_policy {view_batch_policy!r}; expected one of {MoCo_Wrapper.VIEW_BATCH_POLICIES}.")
    
    bn_suffix = f"_bn-{bn_policy}" if args.bn_policy is not None else ""
    view_suffix = f"_views-{view_batch_policy}" if args.view_batch_policy is not None else ""
    run_suffix = bn_suffix + view_suffix
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("variant=%s seed=%d device=%s", args.variant, args.seed, device)

    noise_pool = None
    noise_mixer = None
    
    if args.variant == "habitat":
        noise_pool = load_noise_pool(
            split_csv_path = data_cfg.get("seen_habitat_csv", "data/processed/esc50_seen_habitat.csv"),
            esc50_audio_dir = data_cfg.get("esc50_dir", "data/raw/esc50"),
            sample_rate = int(data_cfg["sample_rate"]), duration_sec = float(data_cfg["clip_sec"]),
        )
        noise_mixer = NoiseMixer(sample_rate = int(data_cfg["sample_rate"]))

    elif args.variant == "white_noise":
        noise_mixer = NoiseMixer(sample_rate = int(data_cfg["sample_rate"]))

    overlap_manifest = data_cfg.get("overlap_exclusions", "data/processed/xeno_canto_excluded.json")

    dataset = XenoCantoDataset(
        audio_dir = data_cfg["xeno_canto_dir"], 
        noise_pool = noise_pool,
        noise_mixer = noise_mixer, 
        variant = args.variant,
        sample_rate = int(data_cfg["sample_rate"]), 
        duration_sec = float(data_cfg["clip_sec"]),
        exclude_manifest_path = overlap_manifest,
        require_overlap_audit = bool(data_cfg.get("require_overlap_audit", True)),
        preflight_manifest_path=data_cfg.get("preflight_manifest"),
        cache_dir = data_cfg.get("cache_dir"),
        require_preflight = bool(data_cfg.get("require_audio_preflight", True)),
        max_crop_retries = int(data_cfg.get("max_crop_retries", 5)),
        min_crop_rms = float(data_cfg.get("min_crop_rms", 1e-5)),
        crop_pair_policy = str(data_cfg.get("crop_pair_policy", "independent_crops")),
    )
    batch_size = int(train_cfg["batch_size"])

    if len(dataset) < batch_size:
        raise RuntimeError(f"Only {len(dataset)} eligible XenoCanto files for batch_size = {batch_size}; reduce batch size or add data.")
    
    queue_size = int(model_cfg["queue_size"])
    enqueue_batch_size = 2 * batch_size

    if queue_size % enqueue_batch_size != 0:
        raise ValueError(f"queue_size ({queue_size}) must be divisible by 2*batch_size ({enqueue_batch_size}) for symmetric enqueue operations.")
    
    num_workers = int(os.environ.get("BIO_SSL_NUM_WORKERS", train_cfg.get("num_workers", 0)))

    if num_workers < 0:
        raise ValueError("num_workers must be non negative.")
    
    # Restart worker processes at each epoch boundary. 
    loader = DataLoader(dataset, 
                        batch_size = batch_size, 
                        shuffle = True, 
                        drop_last = True, 
                        num_workers = num_workers,
                        pin_memory = device.type == "cuda", 
                        persistent_workers = False)
    
    if len(loader) == 0:
        raise RuntimeError("Pretraining DataLoader has zero batches; refusing to save an empty checkpoint.")

    backbone_q, backbone_k = CNN_10_Backbone(), CNN_10_Backbone()
    head_q = Projection_Head(512, 512, int(model_cfg["feature_dim"]))
    head_k = Projection_Head(512, 512, int(model_cfg["feature_dim"]))

    model = MoCo_Wrapper(backbone_q, backbone_k, head_q, head_k,
                         dim = int(model_cfg["feature_dim"]), K = queue_size,
                         m = float(model_cfg["momentum"]), bn_policy = bn_policy, view_batch_policy = view_batch_policy).to(device)
    
    optimizer = torch.optim.Adam(model.encoder_q.parameters(), lr = float(train_cfg["learning_rate"]))
    epochs = int(train_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = max(1, epochs))
    tau = float(train_cfg.get("tau", model_cfg.get("temperature", 0.1)))

    if tau <= 0:
        raise ValueError(f"InfoNCE temperature must be positive, got {tau}.")

    monitor_cfg = train_cfg.get("knn_monitor", {})
    monitor_every = int(monitor_cfg.get("every_epochs", 10))
    monitor = None

    monitor_dataset = BirdCLEFDataset(
        data_cfg.get("birdclef_dir", "data/raw/birdclef"),
        sample_rate = int(data_cfg["sample_rate"]), 
        duration_sec = float(data_cfg["clip_sec"]), 
        crop_mode = "center",
    )
    audit_data = dataset.overlap_audit

    if audit_data is None or not audit_data.get("birdclef_paths_sha256"):
        raise RuntimeError("Overlap audit does not include a BirdCLEF file index. Rerun audit_dataset_overlap.py.")
    
    current_birdclef_path_hash = fingerprint_relative_paths(monitor_dataset.relative_paths)

    if current_birdclef_path_hash != audit_data["birdclef_paths_sha256"]:
        raise RuntimeError(
            "Current BirdCLEF file index differs from the overlap audit. "
            "Rerun audit_dataset_overlap.py and review the split manifests before pretraining."
        )
    
    if monitor_every > 0:
        monitor = load_split_for_monitor(Path(f"checkpoints/data_split_s{args.seed}.json"), monitor_dataset, args.seed)

    os.makedirs("checkpoints", exist_ok = True)
    os.makedirs("results", exist_ok = True)

    log_path = Path(f"results/pretrain_{args.variant}_s{args.seed}{run_suffix}.jsonl")
    latest_path = Path(f"checkpoints/moco_{args.variant}_s{args.seed}{run_suffix}_latest.pt")
    final_path = Path(f"checkpoints/moco_{args.variant}_s{args.seed}{run_suffix}.pt")

    commit = git_commit()

    config_sha256 = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    data_fingerprint = fingerprint_relative_paths(dataset.relative_paths)
    preflight_manifest_path = data_cfg.get("preflight_manifest")
    preflight_manifest_sha256 = None
    
    if preflight_manifest_path:
        manifest_file = Path(preflight_manifest_path)
        if manifest_file.is_file():
            preflight_manifest_sha256 = hashlib.sha256(manifest_file.read_bytes()).hexdigest()

    start_epoch = 1

    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"Cannot resume: latest checkpoint not found at '{latest_path}'.")
        
        resume_payload = torch.load(latest_path, map_location="cpu", weights_only=True)
        expected = {
            "variant": args.variant,
            "seed": args.seed,
            "bn_policy": bn_policy,
            "view_batch_policy": view_batch_policy,
            "config_sha256": config_sha256,
            "dataset_fingerprint": data_fingerprint,
            "preflight_manifest_sha256": preflight_manifest_sha256,
        }

        for key, value in expected.items():
            if resume_payload.get(key) != value:
                raise RuntimeError(
                    f"Resume checkpoint metadata mismatch for {key}: saved={resume_payload.get(key)!r}, current={value!r}. "
                    "Use the original config/data or deliberately start a new run."
                )
            
        model.load_state_dict(resume_payload["model_state"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        scheduler.load_state_dict(resume_payload["scheduler_state"])
        start_epoch = int(resume_payload["epoch"]) + 1
        saved_rng_state = resume_payload.get("rng_state")

        if saved_rng_state is not None:
            restore_rng_state(saved_rng_state)
        else:
            set_seed(args.seed, deterministic = deterministic, deterministic_warn_only = deterministic_warn_only)
            logger.warning(
                "Checkpoint has no saved RNG state; resumed with seed %d. "
                "This legacy resume is reproducible from this point but not bitwise-continuous.",
                args.seed,
            )
        retained = []

        if log_path.is_file():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)

                if int(record.get("epoch", 0)) <= int(resume_payload["epoch"]):
                    retained.append(json.dumps(record))

        retained_epochs = {int(json.loads(line).get("epoch", 0)) for line in retained}
        last_record = resume_payload.get("log_record")

        if isinstance(last_record, dict) and int(last_record.get("epoch", -1)) == int(resume_payload["epoch"]) and int(resume_payload["epoch"]) not in retained_epochs:
            retained.append(json.dumps(last_record))

        retained.sort(key = lambda line: int(json.loads(line).get("epoch", 0)))
        log_path.write_text("\n".join(retained) + ("\n" if retained else ""), encoding = "utf-8")
        logger.info("Resuming from epoch %d/%d (checkpoint=%s).", start_epoch, epochs, latest_path)

        if start_epoch > epochs:
            atomic_torch_save(resume_payload, final_path)
            logger.info("Checkpoint already reached configured epochs; copied it to %s.", final_path)
            return
    else:
        if latest_path.exists() or final_path.exists():
            raise FileExistsError(
                f"Checkpoint already exists for variant = {args.variant}, seed = {args.seed}. "
                "Pass --resume to continue or move the old checkpoints aside to start a fresh run."
            )
        log_path.write_text("", encoding = "utf-8")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sum, n_seen, n_batches = 0.0, 0, 0

        for clean_views, degraded_views in loader:
            clean_views = clean_views.to(device, non_blocking = device.type == "cuda")
            degraded_views = degraded_views.to(device, non_blocking = device.type == "cuda")

            optimizer.zero_grad(set_to_none = True)

            query_1, key_1, query_2, key_2, queue = model(clean_views, degraded_views, symmetric = True)
            loss_forward = info_nce_loss(query_1, key_1, queue, tau=tau)
            loss_reverse = info_nce_loss(query_2, key_2, queue, tau=tau)
            loss = 0.5 * (loss_forward + loss_reverse)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non finite symmetric InfoNCE loss at epoch {epoch}, batch {n_batches}.")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder_q.parameters(), max_norm = float(train_cfg.get("grad_clip_norm", 5.0)))
            optimizer.step()
            model._dequeue_and_enqueue(torch.cat([key_1.detach(), key_2.detach()], dim = 0))
            loss_sum += float(loss.item())
            n_seen += clean_views.shape[0]
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError(f"Epoch {epoch} contained zero training batches.")
        
        avg_loss = loss_sum / n_batches
        scheduler.step()

        knn_acc = None
        if monitor is not None and (epoch % monitor_every == 0 or epoch == 1 or epoch == epochs):
            knn_acc = knn_probe_accuracy(
                model.encoder_q[0], monitor_dataset,
                monitor["train_fraction_filenames"].get("0.10", monitor["train_filenames"]),
                monitor["validation_filenames"], device,

                train_max = int(monitor_cfg.get("train_max", 1000)),
                validation_max = int(monitor_cfg.get("validation_max", 500)),

                k = int(monitor_cfg.get("k", 5)), batch_size = int(monitor_cfg.get("batch_size", 64)),
                num_workers = int(os.environ.get("BIO_SSL_MONITOR_NUM_WORKERS", monitor_cfg.get("num_workers", 0))), seed = args.seed,
            )
            logger.info("epoch=%d InfoNCE=%.5f kNN_probe=%.4f", epoch, avg_loss, knn_acc)

        else:
            logger.info("epoch=%d/%d InfoNCE=%.5f lr=%.7g", epoch, epochs, avg_loss, scheduler.get_last_lr()[0])

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "variant": args.variant, 
            "seed": args.seed, 
            "bn_policy": bn_policy, 
            "view_batch_policy": view_batch_policy, 
            "epoch": epoch,
            "train_infonce_loss": avg_loss, 
            "knn_probe_accuracy": knn_acc,
            "learning_rate": scheduler.get_last_lr()[0], 
            "batches": n_batches, 
            "examples": n_seen,
            "config_path": args.config, 
            "config_sha256": config_sha256, 
            "git_commit": commit,
        }

        payload = {
            "model_state": model.state_dict(), 
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), 
            "epoch": epoch,
            "variant": args.variant, 
            "seed": args.seed, 
            "bn_policy": bn_policy, 
            "view_batch_policy": view_batch_policy, 
            "config_path": args.config,
            "git_commit": commit, 
            "config_sha256": config_sha256,
            "dataset_fingerprint": data_fingerprint,
            "preflight_manifest": data_cfg.get("preflight_manifest"),
            "preflight_manifest_sha256": preflight_manifest_sha256,
            "train_infonce_loss": avg_loss, 
            "knn_probe_accuracy": knn_acc,
            "label_map": monitor_dataset.label_map if monitor is not None else None,
            "rng_state": capture_rng_state(),
            "log_record": record,
        }

        atomic_torch_save(payload, str(latest_path))
        with log_path.open("a", encoding = "utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    atomic_torch_save(payload, str(final_path))
    logger.info("Pretraining complete: %s | JSONL log: %s", final_path, log_path)


if __name__ == "__main__":
    main()