"""Summarize the optional habitat pretraining BatchNorm policy ablation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

POLICIES = ("copy_query", "ema_query", "frozen_key")
SEEDS = (42, 123, 999)


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--results-dir", default = "results")
    parser.add_argument("--output", default = "results/bn_ablation_summary.csv")
    parser.add_argument("--expected-epochs", type=int, default = 100)
    
    args = parser.parse_args(argv)
    if args.expected_epochs < 1:
        parser.error("--expected-epochs must be positive")
        
    root = Path(args.results_dir)
    rows = []
    missing = []
    
    for policy in POLICIES:
        for seed in SEEDS:
            path = root / f"pretrain_habitat_s{seed}_bn-{policy}.jsonl"
            
            if not path.is_file():
                missing.append(str(path))
                continue
                
            records = [json.loads(line) for line in path.read_text(encoding = "utf-8").splitlines() if line.strip()]
            records = [row for row in records if row.get("variant") == "habitat" and row.get("bn_policy") == policy and int(row.get("seed", -1)) == seed]
            if not records:
                raise ValueError(f"No matching records found in {path}.")
                
            records.sort(key = lambda row: int(row["epoch"]))
            epochs = [int(row["epoch"]) for row in records]
            if len(epochs) != len(set(epochs)):
                raise ValueError(f"Duplicate epoch records in {path}.")
                
            final = records[-1]
            if epochs[-1] < args.expected_epochs:
                raise ValueError(f"Run {path} stopped at epoch {epochs[-1]}/{args.expected_epochs}; refusing to call it complete.")
                
            finite_losses = [float(row["train_infonce_loss"]) for row in records if math.isfinite(float(row.get("train_infonce_loss", float("nan"))))]
            knn_records = [row for row in records if row.get("knn_probe_accuracy") is not None and math.isfinite(float(row["knn_probe_accuracy"]))]
            
            rows.append({
                "bn_policy": policy, 
                "seed": seed, 
                "final_epoch": epochs[-1],
                "final_infonce_loss": float(final["train_infonce_loss"]),
                "last_finite_infonce_loss": finite_losses[-1] if finite_losses else "",
                "last_knn_monitor_epoch": int(knn_records[-1]["epoch"]) if knn_records else "",
                "last_knn_probe_accuracy": float(knn_records[-1]["knn_probe_accuracy"]) if knn_records else "",
                "log_path": str(path),
                "interpretation_note": "Representation proxy ablation only; downstream validation and test probes are not included in this CSV.",
            })
            
    if missing:
        raise FileNotFoundError("Missing BN ablation runs; no partial summary written:\n" + "\n".join(missing))
        
    output = Path(args.output)
    output.parent.mkdir(parents = True, exist_ok = True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames = rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        
    tmp.replace(output)
    print(f"Wrote {len(rows)} complete BN-ablation records to {output}")


if __name__ == "__main__":
    main()
