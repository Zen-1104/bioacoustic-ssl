# Noise-Robust Bioacoustic SSL — v1.6.4

This repository implements a MoCo-style self-supervised learning pipeline for bioacoustic classification, with an explicit comparison between habitat-noise contrastive pretraining, a time-shift/SpecAugment control, a Gaussian-noise control, a clean supervised baseline, and a supervised baseline trained with the same habitat-noise distribution. A plain global gain change is not used as a primary control because per-sample log-mel standardization largely cancels global amplitude changes.

## 1. What this pipeline produces

- Three pretraining variants across seeds `42`, `123`, and `999`.
- Linear-probe, clean supervised, and matched-habitat-noise supervised results at nested `1%`, `5%`, `10%`, and `100%` label fractions.
- Stress-test predictions for the same held-out BirdCLEF filenames on clean, seen-habitat-noisy, and unseen-habitat-noisy conditions, including the matched-noise supervised baseline.
- Optional frozen-feature linear probes using PANNs Cnn14 and BirdNET acoustic v3.0 pretrained embeddings.
- A separate multilabel ranking evaluation on time-annotated Western Ghats soundscape recordings.
- Per-recording predictions, JSONL epoch logs, summary CSV files, LaTeX tables, and figures.

The scripts reject missing files, empty data loaders, silent noise clips, non-finite losses, unknown splits, and incomplete result aggregations rather than silently substituting values. Xeno-Canto source files are preflighted and cached as fixed-length, 16 kHz crops before training so the multi-epoch loop does not repeatedly decode/resample the original files. Preflight uses a statistical early-abort guard: checks start after 1,000 files and repeat every 100 files by default (`--early-abort-min-files`, `--early-abort-check-interval`, or the matching `PREFLIGHT_EARLY_ABORT_*` variables); it aborts only when the lower bound of the 95% Wilson interval for the running exclusion rate exceeds the configured threshold. The exact final exclusion fraction remains the final gate. Processing order is deterministically shuffled (`--order-seed` / `PREFLIGHT_ORDER_SEED`, default 42) so checks are not tied to alphabetic directory order. It builds in a staging directory so failed rebuilds preserve the previous cache, records staging ownership, and removes stale staging directories at startup (default age threshold 72 hours via `--staging-stale-hours` / `PREFLIGHT_STAGING_STALE_HOURS`). SLURM terminal states are considered inactive; when `squeue` errors or cannot classify a state, cleanup tries `sacct` before falling back to age-based cleanup. The two contrastive views sample crops from the same source recording; their loss is symmetric, with both views passing through the query encoder. By default, query BatchNorm sees the clean/noisy views in separate forwards and its running buffers are copied to the eval-mode key encoder. `--view-batch-policy concatenated` instead processes both query views in one combined forward, directly testing whether separate per-view BatchNorm statistics contribute to the train/eval mismatch. Both are single-device policies, not distributed shuffle-BN; document the selected policy and evaluate any selected alternative downstream.

## 2. Expected data layout

```text
data/raw/
  xeno_canto/                  # unlabeled audio; nested folders are allowed
  birdclef/
    train_audio/
      <species_label>/*.ogg
    train_metadata.csv         # optional; author/recordist/latitude/longitude fields improve grouping
  esc50/
    audio/*.wav
    meta/esc50.csv
  western_ghats_soundscapes/
    annotation.csv
    soundscape_data/*.flac  # extracted 9.5 GB archive
```

The BirdCLEF loader also accepts a directory that already directly contains one folder per species. It automatically resolves `birdclef/train_audio` when given the archive root. The configured class map is sorted and checked against the saved split manifest.

**Grouped split note:** when BirdCLEF metadata contains author/recordist or valid coordinates, the split builder uses stratified group folds. If grouping metadata is absent, it logs and stores an explicit file-level split method. If some grouping metadata is present but there are too few distinct groups to construct all folds, the builder fails rather than silently degrading to file-level splitting. Do not describe a file-level split as recordist/location-independent.

## 3. Environment and dataset acquisition

Use Python 3.10 or 3.11 in the environment you intend to train with.

**There is no `scripts/download_datasets.sh` in this repository.** Download/acquire the datasets through their official channels (Xeno-Canto API/account as required, BirdCLEF/Kaggle, and ESC-50), then extract them into the expected `data/raw/` layout below. The pipeline deliberately does not automate account-gated downloads or invent credentials. Empty directories created by `mkdir -p` are not evidence that data have been downloaded.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_env.py
```

The requirements constrain NumPy below 2.x for compatibility with the audio stack. On a compute cluster, make sure the installed `torch` and `torchaudio` builds match the available CUDA runtime; the SLURM module names in the scripts are templates and should be adjusted to the cluster.

## 4. Prepare data, cache, and fixed splits

Acquire and extract all three datasets before this phase. The recommended full pipeline runs the expensive audio decode/cache and overlap scans as a batch preflight job, not on the SLURM login node. For a manual/local run from the repository root:

```bash
mkdir -p logs checkpoints results data/processed data/raw
python scripts/create_habitat_split.py --seed 42
python scripts/preflight_xeno_audio.py \
  --early-abort-min-files 1000 --early-abort-check-interval 100 \
  --order-seed 42 --staging-stale-hours 72 \
  --audio-dir data/raw/xeno_canto \
  --cache-dir data/processed/xeno_canto_cache \
  --manifest data/processed/xeno_canto_preflight.json \
  --sample-rate 16000 --clip-sec 10 --num-crops 3 --min-rms 0.00001
python scripts/audit_dataset_overlap.py \
  --xeno-dir data/raw/xeno_canto \
  --birdclef-dir data/raw/birdclef \
  --output data/processed/xeno_canto_excluded.json
python scripts/preflight_birdclef_audio.py \
  --audio-dir data/raw/birdclef \
  --output data/processed/birdclef_audio_preflight.json \
  --sample-rate 16000 --clip-sec 10 --minimum-rms 0.00001 \
  --max-read-error-fraction 0.01 --max-excluded-fraction 0.05 --list-read-errors
python scripts/create_birdclef_splits.py \
  --audio-dir data/raw/birdclef --output-dir checkpoints \
  --seeds 42 123 999 --split-seed 42 \
  --min-recordings 5 --min-groups 5 --rare-class-policy exclude \
  --max-read-error-fraction 0.01 --max-excluded-fraction 0.05
```

The preflight stores up to three non-silent, fixed-length crops per Xeno-Canto recording and emits a manifest; when decode/silence failures occur, it also writes an exclusion CSV. The cache takes additional space (about 12 GB for 6,433 recordings × 3 × 10 seconds of float32 mono audio, before filesystem overhead). The manifest fingerprints the source file inventory and key settings; missing or stale cache shards cause a hard failure. Interrupted builds can leave staging directories behind; startup cleanup removes a stage when its owner is known to be gone or its age exceeds the configured threshold. For SLURM-owned stages, terminal scheduler/accounting states trigger cleanup before the age limit; active jobs are preserved. If both `squeue` and `sacct` fail to establish state, age-based cleanup is used conservatively. The 72-hour default exceeds the provided 48-hour preflight wall-time; adjust it if your cluster uses a longer job limit. Staging and final cache occupy space simultaneously during rebuilds. Use `--overwrite` only when deliberately rebuilding the cache after backing up/understanding prior outputs.

The split builder first scans each BirdCLEF recording once with a deterministic center-first crop-recovery policy. Files for which no fixed-length crop exceeds the RMS threshold are excluded from every partition and listed as `silent_excluded_filenames`. The preflight cache fingerprints normalized relative paths, labels, file sizes, crop settings, and schema version (not modification times), so copying the same inventory to scratch does not force a rescan solely because mtimes changed. Use `--list-read-errors` to print decoder failures; read-error paths are recorded and excluded explicitly rather than aborting the scan. The BirdCLEF preflight fails closed by default when audio read/decode errors exceed 1% of the indexed inventory or when all preflight exclusions exceed 5%. Both limits are configurable but must be reviewed explicitly; every run prints and saves an exclusion-reason histogram. The manifest is written before a threshold failure is raised so diagnostics remain available. If a threshold is intentionally changed, split regeneration records the selected thresholds and requires `--overwrite` when the existing split policy differs. v1.6.4 bumps the preflight and split schema to 5, and rejects --minimum-rms values below the NoiseMixer-compatible 1e-5 floor; back up and rebuild old manifests once. The split builder does not silently fall back from grouped splitting. Classes with fewer than the configured minimum recordings or independent author/location groups are excluded from every split by default (`--rare-class-policy exclude`); choose `--rare-class-policy error` to stop instead; excluded species (including those with zero usable recordings) and active-class scope are recorded in each manifest and carried into result JSON/CSV outputs. `missing:<filename>` singleton group IDs prevent leakage but do not count as evidence of distinct authors/locations. Report `n_train` and per-fraction sample counts rather than assuming that “1%” is exactly 1%: the per-class minimum-one rule can make the actual subset larger.

The same train/validation/test files are shared across model seeds (`--split-seed 42`), while low-fraction subsets use seed-specific draws for sensitivity. Test recordings remain fixed, so variation across the three seeds mainly reflects model/training randomness; it is not a complete estimate of split uncertainty. Each model seed gets a named manifest and `test_split_s{seed}.json`. Existing manifests are preserved and checked. Use `--overwrite` only when deliberately creating a new benchmark split, and keep the resulting small manifests with the paper artifacts.

The SNR used by `NoiseMixer` is a **clip-global waveform power ratio**, calculated over the whole fixed-length clip and then mixture RMS-renormalized. This can be affected by silence or padding and does not guarantee framewise SNR around a sparse bird call. The default sweep is Clean, +20, +15, +10, +5, 0, and −5 dB. The predeclared primary endpoint is habitat-SSL versus standard-SSL on unseen-habitat noise at 0 dB; the more severe −5 dB condition is secondary. See `docs/REPORTING_AND_ANALYSIS_PLAN_v1.6.4.md`. The manuscript abstract is not part of this repository and must separately be updated to accurately describe the configured sweep and which condition is primary.

## 5. Run one component manually

### One short pretraining smoke test

For a quick local test, temporarily reduce `training.epochs` and `training.batch_size` in a copy of the YAML and ensure `model.queue_size` is divisible by the chosen batch size. Keep the train/validation/test manifests fixed. The configured k-NN probe uses the 10% training filenames and validation filenames from the saved split.

```bash
python -m experiments.exp1_pretrain_moco \
  --config configs/pretrain_moco_habitat.yaml --variant habitat --seed 42
```

Then test one probe and one supervised run:

```bash
python -m experiments.exp2_linear_probe_vs_supervised \
  --config configs/linear_probe.yaml --mode probe \
  --checkpoint checkpoints/moco_habitat_s42.pt \
  --variant habitat --fraction 0.01 --seed 42

python -m experiments.exp2_linear_probe_vs_supervised \
  --config configs/supervised_baseline.yaml --mode supervised \
  --variant supervised --fraction 0.01 --seed 42
```

Exp3 requires all five model variants' 100% checkpoints/heads for the selected seed:

```bash
python -m experiments.exp3_unseen_habitat_stress_test \
  --config configs/stress_test.yaml --seed 42 --fraction 1.0
```

## 6. Submit the full SLURM pipeline

On a SLURM login node, after adapting the module names, GPU resources, memory/time limits, and Python/CUDA modules to your cluster:

```bash
bash scripts/submit_pipeline.sh
```

Partition names are not hard-coded in the batch scripts. If your cluster requires partition selection, set `GPU_PARTITION` and `PREFLIGHT_PARTITION`, for example `GPU_PARTITION=gpu PREFLIGHT_PARTITION=cpu bash scripts/submit_pipeline.sh` (replace these with your cluster's actual names). The launcher creates `logs/` before `sbatch` is called because SLURM opens log paths before a job starts.

To run the stress test using a lower-label-fraction classifier, set the fraction before submitting. The full pipeline runs Exp2 first, so the matching heads/checkpoints will be created before Exp3 starts. For a standalone Exp3 run, those matching artifacts must already exist. Low-fraction outputs use separate filenames so they do not overwrite the 100% results:

```bash
STRESS_FRACTION=0.05 bash scripts/submit_pipeline.sh
```

The SNR sweep is read from `configs/stress_test.yaml` by both Exp3 and the aggregator. To use a different sweep, copy and edit the config, then pass that same path to the pipeline:

```bash
cp configs/stress_test.yaml configs/stress_test_custom.yaml
# Edit evaluation.snr_levels in configs/stress_test_custom.yaml.
STRESS_CONFIG=configs/stress_test_custom.yaml STRESS_FRACTION=0.05 bash scripts/submit_pipeline.sh
```

Before submitting, make sure the datasets have actually been acquired and extracted under `data/raw/`; there is no automatic download script. The launcher creates `logs/`, `checkpoints/`, `results/`, `data/processed/`, and `data/raw/` **on the login node before calling `sbatch`**. It then submits a CPU preflight job to build/validate the habitat split, cache fixed-length Xeno-Canto crops, run the catalogue-ID/hash overlap audit, and create the BirdCLEF split manifests. GPU stages are submitted with `afterok` dependencies on that preflight job:

1. One pretraining task per `variant × seed`.
2. One Exp2 task per `model variant × label fraction × seed`.
3. One Exp3 task per seed at `STRESS_FRACTION` (defaults to `1.0`).
4. One aggregation job for that same fraction after all stress tests complete successfully.

Shared tables are never written by concurrent array tasks. `afterok` dependencies prevent downstream stages from starting if the preceding array fails. `RESUME_PRETRAIN=1` resumes pretraining tasks that have a `_latest.pt` checkpoint and starts tasks with no checkpoint fresh; resumption is allowed only when the config, dataset inventory, and preflight manifest match the saved checkpoint. New v1.6.4 checkpoints save Python, NumPy, CPU PyTorch, and available CUDA/MPS RNG states, and pretraining requests deterministic algorithms with `deterministic_warn_only: true` by default. Unsupported operations warn rather than abort; warn-only is not a strict determinism guarantee. Set it false to require strict deterministic kernels, which may fail on the target device; bitwise reproducibility is scoped to a matching hardware/software stack. The global spatial pool uses a mean reduction instead of `AdaptiveAvgPool2d`. Pretraining worker processes restart at epoch boundaries so restored RNG state can recreate worker seeds; older checkpoints lack RNG state and resume with a documented seed-reset warning. On first use of v1.6.4, rebuild the BirdCLEF preflight/split manifests once because the preflight schema now fingerprints relative path and file size, and includes explicit read/decode-error exclusions. Back up old manifests before this deliberate migration. In the pipeline launcher, set `REBUILD_BIRDCLEF_PREFLIGHT=1` and `REBUILD_SPLITS=1` for that deliberate initial migration, then reset them to `0`. Use `REBUILD_PREFLIGHT=1` only when deliberately rebuilding the Xeno-Canto cache. `REBUILD_SPLITS=1` and `REBUILD_HABITAT=1` replace benchmark partitions; back up old outputs first because the benchmark may change.

## 7. Outputs

```text
checkpoints/
  data_split_s{seed}.json
  test_split_s{seed}.json
  moco_{variant}_s{seed}.pt
  linear_head_{variant}_s{seed}_frac{fraction}.pt
  supervised_baseline_s{seed}_frac{fraction}.pt
  supervised_noise_baseline_s{seed}_frac{fraction}.pt
  pretrained_head_{panns|birdnet}_s{seed}_frac{fraction}.pt  # optional external baselines
results/
  pretrain_{variant}_s{seed}.jsonl
  exp2_{variant}_s{seed}_frac{fraction}.jsonl
  probe_{variant}_s{seed}_frac{fraction}.json
  unseen_habitat_stress_test_matrix_s{seed}.json  # 100% labels
  unseen_habitat_stress_test_matrix_s{seed}_frac{fraction}.json  # non-100% labels
  label_efficiency_summary.csv
  table_label_efficiency.tex
  figure_label_efficiency.png
  stress_test_summary.csv                         # 100% labels
  table_stress_test.tex                            # 100% labels
  generalization_gap_summary.csv                  # 100% labels
  table_generalization_gap.tex                     # 100% labels
  figure_unseen_habitat_stress_test.png            # 100% labels
  figure_generalization_gap.png                    # 100% labels
  pretrained_probe_{panns|birdnet}_s{seed}_frac{fraction}.json  # optional external baselines
  real_soundscape_s{seed}_frac{fraction}_{id}.json/.npz         # field evaluation
  stress_test_summary_frac{fraction}.csv           # lower label fractions
  table_stress_test_frac{fraction}.tex
  generalization_gap_summary_frac{fraction}.csv
  table_generalization_gap_frac{fraction}.tex
  figure_unseen_habitat_stress_test_frac{fraction}.png
  figure_generalization_gap_frac{fraction}.png
```

Stress-test JSON files use Exp3 schema v5 and contain per-recording labels, predictions, correctness values, and skipped-mix counters for paired analysis. The aggregator verifies the expected seeds, model variants, SNR conditions, class scope, and seen/unseen test-file pairing; it rejects older schema outputs or any skipped mixes in a requested noisy condition. It uses a Student's t interval across seeds and refuses to publish a partial result set.

## 8. Added controlled baseline and real-soundscape experiments (introduced in v1.5.10)

### Matched-noise supervised baseline

The main probe job matrix now includes both `supervised` (clean training) and `supervised_noise` (noise-augmented training). The noisy baseline uses the verified **seen** ESC-50 habitat pool, the same inclusive integer SNR distribution from −5 through +20 dB as habitat MoCo, and the same waveform mixer/RMS renormalization. It uses `probability: 0.5` to match the fraction of noisy views in the two-view habitat-MoCo stream (one of two views is mixed); validation/test data remain clean. Configuration: `configs/supervised_noise_baseline.yaml`. It requires `data/processed/esc50_seen_habitat.csv` and its split manifest. The launcher runs five model/variant groups per seed/fraction.

### Optional pretrained external baselines

`experiments/exp4_pretrained_feature_baselines.py` supports two frozen-embedding linear probes:

- **PANNs Cnn14**: general audio-event representation pretrained on AudioSet. The wrapper uses `panns-inference==0.1.1` and requires an explicit local checkpoint. It never relies on an implicit/default checkpoint download on the compute node. See the [official repository](https://github.com/qiuqiangkong/panns_inference).
- **BirdNET acoustic v3.0**: bioacoustic-specific representations, mean-pooled across the official API's reported segment embeddings. The integration pins `birdnet[onnx]==1.1.1`, checks the documented embedding API/segment duration, and records the detected ONNX model file SHA-256 and library version. The official library documents model terms as CC BY-NC-SA 4.0; this may restrict commercial use. Review the exact terms shipped with the model artifact and obtain appropriate permission before any use that may fall outside the license. BirdNET V3.0 is a preview model. The library requires Python 3.11+ on supported Linux architectures, so use a separate optional environment if the core cluster environment is Python 3.10.

Examples (run for all three seeds and all four fractions before reporting aggregated baseline results):

```bash
# Optional general audio baseline: download checkpoint on an internet-enabled node
pip install -r requirements.txt
scripts/download_panns_checkpoint.sh data/pretrained/panns/Cnn14_mAP=0.431.pth
python -m experiments.smoke_external_baselines --model panns --checkpoint data/pretrained/panns/Cnn14_mAP=0.431.pth --num-files 20
python -m experiments.exp4_pretrained_feature_baselines --model panns --checkpoint data/pretrained/panns/Cnn14_mAP=0.431.pth --seed 42 --fraction 0.10

# Optional bioacoustic baseline in a separate Python 3.11+ environment
python3.11 -m venv venv-birdnet
source venv-birdnet/bin/activate
pip install -r requirements.txt
python -m experiments.exp4_pretrained_feature_baselines --model birdnet --seed 42 --fraction 0.10
```

The optional script reuses the saved BirdCLEF split, active class scope, center crop, and train fraction. PANNs and BirdNET v3 receive their native **32 kHz** audio input; the core CNN models continue to use 16 kHz log-mel features. These are external pretrained reference levels, not models pretrained on the exact same training corpus: BirdNET may have exposure to Xeno-Canto-style sources, and AudioSet includes bird sounds. Interpret them primarily as frozen-feature label-efficiency references and describe pretraining-data differences, not as a controlled same-data accuracy contest. Frozen features are cached with an inventory/model fingerprint, including encoder/library/model identity and checkpoint SHA-256, and a filesystem lock to prevent concurrent SLURM jobs racing to build the same cache. Caches may consume hundreds of MB. Download PANNs weights explicitly before compute jobs. For BirdNET on offline compute nodes, load the pinned model once on a network-enabled node into a persistent cache and set `BIRDNET_APP_DATA` to that cache location; verify that the expected ONNX artifact can be hashed before running the probe.

Before a full extraction, run the 20-file smoke test against real recordings and actual weights. The test records the package/model identity and SHA-256; it is marked `smoke_only` and is never a paper result. It skips visibly silent files rather than failing on the first silent recording.

Run every seed/fraction before creating paper tables. PANNs runs in the core environment:

```bash
pip install -r requirements.txt
for seed in 42 123 999; do
  for fraction in 0.01 0.05 0.10 1.00; do
    python -m experiments.exp4_pretrained_feature_baselines --model panns --checkpoint data/pretrained/panns/Cnn14_mAP=0.431.pth --seed "$seed" --fraction "$fraction"
  done
done
python -m results.aggregate_pretrained_baselines --models panns
```

BirdNET v3 currently requires the official 1.x library API and an isolated Python 3.11+ environment on supported Linux architectures:

```bash
python3.11 -m venv venv-birdnet
source venv-birdnet/bin/activate
pip install -r requirements.txt
# On a network-enabled node, populate the persistent BirdNET model cache and verify 20 real files:
export BIRDNET_APP_DATA=/shared/models/birdnet-v3
python -m experiments.smoke_external_baselines --model birdnet --num-files 20 --output results/birdnet_smoke.json
for seed in 42 123 999; do
  for fraction in 0.01 0.05 0.10 1.00; do
    python -m experiments.exp4_pretrained_feature_baselines --model birdnet --seed "$seed" --fraction "$fraction"
  done
done
python -m results.aggregate_pretrained_baselines --models birdnet
```

For a combined external table after both models finish, run `python -m results.aggregate_pretrained_baselines --models panns birdnet`. Aggregation refuses missing seed/fraction records, inconsistent class maps, changed dataset inventories, or inconsistent checkpoint file hashes, package versions and model identities. Absolute cache paths may differ across machines; identity is checked by artifact hash and declared model/runtime metadata. The pinned BirdNET library terms currently document CC BY-NC-SA 4.0 for models; see the [official library documentation](https://github.com/birdnet-team/birdnet) and [license](https://creativecommons.org/licenses/by-nc-sa/4.0/). The V3.0 model is a preview; check the actual downloaded artifact's accompanying terms rather than treating repository and library model licenses as automatically equivalent. Inspect the selected PANNs checkpoint terms as well.

### Real field-soundscape evaluation (with historical benchmark-overlap caveat)

`experiments/real_soundscape_eval.py` evaluates the five core models on 10-second multilabel segments from the [Western Ghats fully annotated soundscape collection on Zenodo](https://zenodo.org/records/16881041), DOI `10.5281/zenodo.16881041`. Download the small `annotation.csv` and extract the 9.5 GB `soundscape_data.zip` into `data/raw/western_ghats_soundscapes/`. Do not download the 40.3 GB unlabeled archive for this evaluation. The data are licensed CC BY 4.0; cite the [Zenodo dataset record](https://zenodo.org/records/16881041) and follow its attribution requirements. Its metadata documents 1,115 recordings, more than 79 hours, and 30,324 high-confidence annotations spanning 116 bird species.

```bash
python experiments/real_soundscape_eval.py --seed 42 --fraction 0.10
# Once the matching PANNs/BirdNET linear heads have been trained:
python experiments/real_soundscape_eval.py --seed 42 --fraction 0.10 --include-external --external-models panns birdnet --panns-checkpoint data/pretrained/panns/Cnn14_mAP=0.431.pth
```

The evaluator maps annotated eBird codes to the saved BirdCLEF active class map, reports mapped/unmapped codes, and enforces `--min-annotation-coverage` (default 0.50). It saves macro/micro average precision plus compressed per-segment predictions. In addition to mapped-positive segments, it deterministically samples up to 10 per-recording segments with no annotation interval for any species and gives them all-zero weak-background targets; this improves false-alarm assessment but does not prove species absence. Background top-1 confidence and the fraction above 0.5 are reported per model and summarized across seeds as weak false-alarm proxies; these are not calibrated operating-point false-alarm rates. Core models run on 16 kHz audio and optional external models on their native 32 kHz input; all methods use the same 10-second time segments and repeat-pad the final partial segment. `--max-recordings` exists only for smoke tests; capped runs have unique identities and are excluded by the aggregator. Missing annotated audio aborts by default; `--allow-missing-recordings` is diagnostic-only and its output is rejected by the aggregator. The dataset record states that portions of this collection were used as the hidden BirdCLEF 2024 test set, so do not call it independent of the historical BirdCLEF benchmark. Before evaluation, run the included filename/exact-hash audit against your actual training audio:

```bash
python -m scripts.audit_soundscape_overlap \
  --soundscape-dir data/raw/western_ghats_soundscapes/soundscape_data \
  --training-dirs data/raw/xeno_canto data/raw/birdclef/train_audio
```

Review every match in `results/soundscape_overlap_audit.json`. The audit catches exact-file and obvious filename overlap, not re-encoded copies or soundscape excerpts embedded in longer training recordings. The collection was used in part as the hidden BirdCLEF 2024 test set, so it must not be described as independent of the historical BirdCLEF benchmark; cite this limitation. It also does not establish geographic independence by itself.

Run the complete soundscape evaluation for all three seeds, then aggregate the same fraction (default 10%) into CSV and LaTeX tables:

```bash
for seed in 42 123 999; do
  python -m experiments.real_soundscape_eval --seed "$seed" --fraction 0.10
done
python -m results.aggregate_real_soundscape --fraction 0.10 --seeds 42 123 999 --bootstrap-replicates 200
```

If including the pretrained baselines, ensure both external heads exist for every selected seed/fraction and pass `--include-external --external-models panns birdnet` on each evaluation run. Then aggregate with `python -m results.aggregate_real_soundscape --fraction 0.10 --seeds 42 123 999 --models habitat standard white_noise supervised supervised_noise panns birdnet`. The aggregator requires matching annotations, recording inventory, sample rate, clip duration, active class map, background sampling policy and coverage gate; it refuses capped smoke-test runs, runs with no background negatives despite requesting them, and old outputs without these protocol fields. It reports the background-confidence proxies and writes a separate `real_soundscape_recording_cluster_bootstrap.csv` with 95% percentile intervals obtained by resampling recordings rather than correlated segments. The main table also reports across-seed t intervals; describe both uncertainty sources accurately.

### Still not implemented or not experimentally demonstrated

The code includes the three core MoCo variants, clean/noisy supervised baselines, optional pretrained linear probes, and a real-soundscape evaluator. It also includes a separate, optional BatchNorm-policy representation-proxy ablation, but does **not** implement a full queue-size/temperature grid, a second SSL objective, or external Perch/BEATs wrappers. The optional external-baseline integrations have not yet been run here with real data or pretrained weights. Do not describe any added experiment as a measured result until its full run succeeds; the real-soundscape metric and unmatched-species coverage should be included in the analysis.

The paired analysis saves per-recording correctness, runs exact McNemar comparisons, and performs a paired bootstrap. When non-file-level recordist/location groups exist, the bootstrap resamples groups to better preserve within-group dependence; with file-level splits/no metadata it falls back to recording-level resampling. Three-seed t-intervals should be reported alongside the raw seed values and interpreted cautiously.

## 9. Reliability guards introduced earlier and tightened in v1.6.4

- BirdCLEF evaluation paths share the versioned RMS-aware center-first crop-recovery policy across Exp2, Exp3 clean/noisy conditions, Exp4 primary external baselines, and preflight. Noisy BirdCLEF samples use RMS-aware crop recovery (`SILENT_CROP_RECOVERED`). If every candidate crop is near-silent, the item is retained as clean rather than converted into a noise-only clip with a false bird label; the skip is logged as `SILENT_CROP_MIX_SKIPPED` and recorded in Exp3 JSON. v1.6.4 result aggregation and paired inference fail closed if any skip occurred because clean-fallback rows do not represent the requested noisy SNR. Rebuild the silence preflight/splits and rerun Exp3 rather than interpreting those scores.
- PANNs requires an explicit `--checkpoint` path. `scripts/download_panns_checkpoint.sh` fetches the pinned Cnn14 artifact on an internet-enabled node and verifies both its official Zenodo MD5 and a separately pinned SHA-256 before promoting the download. Carry the verified file to the compute node and supply the same path to the smoke test and probe commands.
- BirdNET is pinned to `birdnet[onnx]==1.1.1`. The wrapper refuses unknown API shapes and records library version, model metadata, ONNX path and SHA-256. Store the model cache persistently and verify model terms for the actual artifact.
- Soundscape analysis adds deterministically sampled weak-background segments, an annotation-code coverage gate, repeat-padding for the last partial segment, and recording-clustered bootstrap intervals. Background segments are weak negatives, not verified silent/absence samples.

## 10. Validation

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
python experiments/exp4_pretrained_feature_baselines.py --help
python -m experiments.smoke_external_baselines --help
python experiments/real_soundscape_eval.py --help
python results/aggregate_probes.py --help
python results/aggregate_results.py --help
# For a lower-label stress result, select the same fraction used by Exp3:
python results/aggregate_results.py --config configs/stress_test.yaml --fraction 0.05 --expected-seeds 42 123 999
```

Passing unit tests and syntax checks are not equivalent to validating a full experiment. Before committing long runs, run the 20-file PANNs/BirdNET extraction smoke test with actual weights, verify the BirdNET version/hash/license identity and PANNs checkpoint hash, verify silent-crop recovery/skip markers, then verify a small pretraining run, finite InfoNCE loss, non-collapsed k-NN probe behavior, one probe checkpoint, clean evaluation, and at least one noisy seen/unseen condition. For soundscapes, verify the coverage gate, sampled background segments, recording-cluster bootstrap, complete audio inventory and historical benchmark-overlap caveat. Do not write paper claims until all expected seeds and output files are complete.


## Optional correctness smoke tests and BatchNorm ablation

Before full training or publishing optional-baseline results, run the real 20-file smoke test for each external encoder. Every run checks finite/dimensionally valid features **and requires at least two distinct feature rows**; it records filenames, silent-file skips, package/model identity, and feature-extraction specification. The external feature cache uses deterministic fixed temporal crops. The primary external-baseline comparison defaults to one center crop (`--num-fixed-crops 1`) to match in-repo probe evaluation; three-crop averaging is an exploratory analysis and requires `--allow-three-crop-analysis` when aggregating. The versioned fingerprint prevents reuse of prior-version BirdNET cache entries.

```bash
python experiments/smoke_external_baselines.py --model birdnet --num-files 20 --output results/smoke_birdnet.json
python scripts/download_panns_checkpoint.sh data/pretrained/panns/Cnn14_mAP=0.431.pth
python experiments/smoke_external_baselines.py --model panns --checkpoint data/pretrained/panns/Cnn14_mAP=0.431.pth --num-files 20 --output results/smoke_panns.json
```

The downloader and PANNs encoder verify both the official Zenodo MD5 and pinned SHA-256. The reported SHA-256 is a checkpoint identity, not a license grant; check the artifact's terms before redistribution/use.

BatchNorm policy is deliberately configurable (`copy_query`, `ema_query`, `frozen_key`). The `copy_query` policy is the default. The buffer-policy ablation is optional and compute-intensive; it runs habitat pretraining for three policies across seeds 42, 123, and 999. It reports only representation proxies (InfoNCE and k-NN monitor accuracy), not full downstream performance. The separate view-batch ablation tests separate versus concatenated query-view forwards to probe the running-statistics confound. These remain screening ablations; evaluate the selected policy in downstream comparisons:

```bash
sbatch utils/slurm/submit_bn_ablation.slurm
python scripts/summarize_bn_ablation.py --expected-epochs 100

# Separate vs. concatenated query-view forward ablation (3 seeds x 2 policies)
sbatch utils/slurm/submit_view_batch_ablation.slurm
python scripts/summarize_view_batch_ablation.py --expected-epochs 100
```

The paired statistics pipeline writes `results/paired_tests_analysis_plan.json`, marks the primary endpoint, estimates its effect over independent training seeds, and applies Holm-Bonferroni correction to the family of secondary per-seed McNemar tests. The default pipeline uses three seeds; if compute allows five for the primary comparison, consistently extend the pretraining, probe, stress, and aggregation job arrays to seeds `42 123 999 2026 2027`. See the analysis-plan document for interpretation and dependence limitations.

