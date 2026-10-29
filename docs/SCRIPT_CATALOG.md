# Script catalog

This catalog describes the maintained Python and shell files under `scripts/`. The source is grouped by job; archived backups are inventoried separately in [`archive/backups/README.md`](../archive/backups/README.md). The main training entry points remain in `experiments/`, and the shell submitter remains at `scripts/submit_pipeline.sh` for its established command path.

## How the utilities fit together

The dataset preparation and integrity workflow is ordered: `preflight_xeno_audio.py` validates Xeno-Canto files and builds fixed crop caches; `preflight_birdclef_audio.py` checks evaluation crops; `audit_dataset_overlap.py` finds likely BirdCLEF/Xeno-Canto duplicates; `create_habitat_split.py` writes ESC-50 seen/unseen noise splits; and `create_birdclef_splits.py` creates stable grouped BirdCLEF train/validation/test and label-fraction manifests. Training consumes these artifacts. The three audits are complementary: `audit_dataset_overlap.py` compares the two training datasets by catalogue ID and file hash, `audit_soundscape_overlap.py` checks field soundscapes against training audio by exact file and filename, and `audit_selected_soundscape_annotations.py` checks selected soundscape labels against source annotation intervals.

`diagnose_moco_checkpoint.py` is an evaluation-only checkpoint inspection utility. `generate_paper_results_20261026.py` reads existing probe, stress-test, real-soundscape, and analysis files to create release figures and captions; it does not train models or produce new model predictions. The separate research analyses are: `paired_real_soundscape_bootstrap_s43_s44.py` for paired recording-level uncertainty, `per_species_real_soundscape_analysis_s43_s44.py` for per-species comparisons, and `class_support_real_soundscape_sensitivity_s43_s44.py` for a fixed-support sensitivity analysis. Their outputs are downstream analysis artifacts used by paper-result generation; their presence does not by itself establish that outputs exist or support a particular claim.

The listening workflow is `species_context_audit_s43_s44.py` → `create_blinded_annotation_review_set_s43_s44.py` → `export_species_context_listening_clips_s43_s44.py` → human review → `analyze_blinded_soundscape_review.py`. The first three inspect candidate segments, select a blinded sample, and export audio clips and a reviewer manifest/template. The final script reads a completed returned CSV plus `results/real_soundscape_aggregate_s43_s44_fixnames/blinded_annotation_review_20261025/answer_key_internal_DO_NOT_SHARE_WITH_REVIEWER.csv`, validates the responses, and writes aggregate-only agreement counts. Current project reporting says completed listening-review responses are unavailable; no valid human-review result is claimed here.

`summarize_bn_ablation.py` and `summarize_view_batch_ablation.py` are optional post-run log summarizers; they require complete logs and do not run experiments. `download_panns_checkpoint.sh` fetches and hash-checks a pinned external checkpoint. `package_source_zip.py` creates a filtered source ZIP. `submit_pipeline.sh` schedules the established preflight → pretraining → probes → stress evaluation → aggregation chain on SLURM.

## Dataset preparation and preflight

### Create BirdCLEF splits

Creates reusable grouped training, validation, test, and nested label-fraction manifests for BirdCLEF. It consumes BirdCLEF audio and the BirdCLEF silence preflight manifest; the output is written under the chosen output directory (default `checkpoints/`). The pipeline and tests use it before training. It uses the `bioacoustic_data` BirdCLEF dataset, audio utilities, split-manifest helpers, and the `scripts.preflight.preflight_birdclef_audio` crop validation helper. It imports the BirdCLEF preflight helper.

Run: `python scripts/data/create_birdclef_splits.py --audio-dir data/raw/birdclef --seeds 42 123 999`

### Create ESC-50 habitat split

Splits ESC-50 outdoor/noise categories into seen and unseen groups with a recorded seed. It reads `data/raw/esc50/meta/esc50.csv` and writes split CSVs and a JSON manifest under `data/processed/`. Training data helpers expect this split before habitat-noise training.

Run: `python scripts/data/create_habitat_split.py --seed 42`

### Preflight BirdCLEF audio

Scans BirdCLEF evaluation center crops, records unreadable or unusable files, and writes a reusable JSON preflight manifest. The split builder consumes this manifest so training/evaluation use the same crop policy.

Run: `python scripts/preflight/preflight_birdclef_audio.py --audio-dir data/raw/birdclef`

### Preflight Xeno-Canto audio

Validates Xeno-Canto recordings, caches fixed-length 16 kHz crops, and writes a preflight manifest plus an invalid-recording report. The training dataset uses the manifest/cache; the pipeline runs this before overlap auditing and training.

Run: `python scripts/preflight/preflight_xeno_audio.py --audio-dir data/raw/xeno_canto --cache-dir data/processed/xeno_canto_cache`

## Training and experiment launchers

### Submit the SLURM pipeline

`scripts/submit_pipeline.sh` checks the environment, datasets, stress-test configuration, and relevant settings, then submits the dependent preflight, training, probe, stress-test, and aggregation jobs. It is a supported launcher; the actual training and evaluation programs live under `experiments/`.

Run: `bash scripts/submit_pipeline.sh`

## Evaluation and metrics

### Inspect a MoCo checkpoint

Loads a supplied MoCo checkpoint and YAML configuration (default `configs/pretrain_moco_habitat.yaml`), BirdCLEF audio, and a split manifest (default seed inferred from the checkpoint), then evaluates frozen-backbone nearest-neighbor representations and reports diagnostic metrics. It uses PyTorch, the `bioacoustic_data`, `evaluation`, `experiments`, and `models` packages. It does not train or modify the checkpoint. It does not train or modify the checkpoint. Inputs include the checkpoint, configuration, split manifest, and BirdCLEF data.

Run: `python scripts/diagnostics/diagnose_moco_checkpoint.py --checkpoint checkpoints/moco_habitat_s42.pt`

## Data-integrity and provenance audits

### Audit BirdCLEF/Xeno-Canto overlap

Compares catalogue IDs in filenames and exact SHA-256 hashes between the two dataset trees, then writes an exclusion JSON used by training. Hash-only comparison cannot detect re-encoded excerpts without catalogue IDs.

Run: `python scripts/integrity/audit_dataset_overlap.py --xeno-dir data/raw/xeno_canto --birdclef-dir data/raw/birdclef --output data/processed/xeno_canto_excluded.json`

### Audit selected soundscape annotations

Checks five explicitly selected field-recording windows against source annotation intervals, including nearby context. It writes a window summary, interval detail CSV, and JSON report under `results/real_soundscape_aggregate_s43_s44_fixnames/annotation_record_audit_20261025/`. Inputs are the Western Ghats annotation CSV and the encoded cases in the script. It is a targeted research audit, not a general annotation converter.

Run: `python scripts/integrity/audit_selected_soundscape_annotations.py`

### Audit soundscape overlap

Compares the Western Ghats soundscape directory against training audio for identical file hashes and obvious filename reuse; writes a JSON overlap report. It does not prove independence for edited or re-encoded excerpts.

Run: `python -m scripts.integrity.audit_soundscape_overlap --soundscape-dir data/raw/western_ghats_soundscapes/soundscape_data --training-dirs data/raw/xeno_canto data/raw/birdclef/train_audio --output results/soundscape_overlap_audit.json`

## Research-result generation and statistical analysis

### Generate paper figures and tables

Reads `results/probe_{habitat,standard}_s{43,44}_frac1.0.json`, two fixed seed stress JSON files, `results/real_soundscape_pair_s43_20261023_094200.npz`, `results/real_soundscape_pair_s44_20261025.npz`, `class_support_sensitivity_20261025/class_support_sensitivity_summary.csv`, and `per_species_analysis_20261025/per_species_ap_comparison.csv` under `results/real_soundscape_aggregate_s43_s44_fixnames/`. It writes plots, plotted data, and captions to a new `results/publication_results_2026-10-26_release/` folder and refuses to overwrite an existing release directory. It does not run training or infer from listening-review responses.

Run: `python scripts/analysis/generate_paper_results_20261026.py`

### Paired real-soundscape bootstrap

Resamples whole recordings in paired seed-43/seed-44 real-soundscape predictions and computes paired aggregate metric intervals. It reads `results/real_soundscape_pair_s43_20261023_094200.npz` and `results/real_soundscape_pair_s44_20261025.npz` (arrays: targets, recording names, segment metadata, class names, and habitat/standard scores). By default it writes summary CSV, replicate CSV, and JSON under `results/real_soundscape_aggregate_s43_s44_fixnames/paired_delta_bootstrap_20261025/`; `--replicates`, `--random-seed`, and `--output-dir` are configurable. These estimates are distinct from per-species and fixed-class support sensitivity analyses.

Run: `python scripts/analysis/paired_real_soundscape_bootstrap_s43_s44.py`

### Per-species real-soundscape analysis

Computes per-species Average Precision and score-distribution comparisons for habitat and standard models from paired real-soundscape prediction archives and target/recording metadata. It reads the same seed-43/44 prediction archives listed above and writes per-species comparison CSV, bootstrap replicate CSV, and summary JSON under `results/real_soundscape_aggregate_s43_s44_fixnames/per_species_analysis_20261025/`. The implementation uses NumPy and scikit-learn; support threshold, bootstrap count, and random seed are encoded in the script.

Run: `python scripts/analysis/per_species_real_soundscape_analysis_s43_s44.py`

### Fixed class-support sensitivity analysis

Restricts the analysis to classes meeting the script's full-data support rule, then resamples paired recordings to examine sensitivity of real-soundscape comparisons. It reads `results/real_soundscape_pair_s43_20261023_094200.npz` and `results/real_soundscape_pair_s44_20261025.npz`, including targets, file names, segment metadata, class names, and both models’ scores. It writes fixed-support eligibility, replicate, summary, and JSON files under `results/real_soundscape_aggregate_s43_s44_fixnames/class_support_sensitivity_20261025/`. This answers a subset-specific question and should not be conflated with the full-class metric.

Run: `python scripts/analysis/class_support_real_soundscape_sensitivity_s43_s44.py`

### Audit soundscape species context

Inspects positive and negative real-soundscape segments for selected species and compares score distributions by segment type. It reads the same seed-43/44 prediction NPZ archives, indexes source audio below `data/raw/western_ghats_soundscapes/`, and writes species summaries, score distributions, a candidate-segment CSV, and JSON below `results/real_soundscape_aggregate_s43_s44_fixnames/species_context_audit_20261025/`. It uses NumPy and scikit-learn.

Run: `python scripts/analysis/species_context_audit_s43_s44.py`

### Summarize BatchNorm ablation

Reads the complete habitat pretraining JSONL logs for three seeds and three BatchNorm policies. It refuses to write a partial summary if runs are missing or incomplete; outputs a CSV of final-epoch losses and the last available kNN proxy. The output is a representation proxy, not a downstream effect estimate.

Run: `python scripts/analysis/summarize_bn_ablation.py --expected-epochs 100`

### Summarize view-batch ablation

Reads complete separate-view and concatenated-view habitat pretraining JSONL logs for the requested seeds, then writes a CSV summary. It requires at least two distinct seeds and complete runs. The loss values are not presented as direct effect estimates.

Run: `python scripts/analysis/summarize_view_batch_ablation.py --expected-epochs 100`

## Listening-review utilities

### Create blinded annotation review set

Samples positive and background candidates from the paired soundscape artifacts and writes a reviewer package ZIP, clip IDs, response template, and internal sampling metadata under `results/real_soundscape_aggregate_s43_s44_fixnames/blinded_annotation_review_20261025/`. It requires the Western Ghats source audio and NumPy/SoundFile. Human responses must be collected separately. The private answer key is not placed in the reviewer package.

Run: `python scripts/analysis/create_blinded_annotation_review_set_s43_s44.py`

### Export species-context listening clips

Reads candidate-segment data from the species-context audit, extracts audio windows from source soundscapes, and writes clip audio, a manifest, and a response template under `results/real_soundscape_aggregate_s43_s44_fixnames/species_context_listening_clips_20261025/`. The input candidate CSV is `species_context_audit_20261025/candidate_segments_for_listening.csv`; the source audio is under `data/raw/western_ghats_soundscapes/soundscape_data/`. It is a review-preparation utility, not an analysis of human responses.

Run: `python scripts/analysis/export_species_context_listening_clips_s43_s44.py`

### Analyze a completed blinded soundscape review

Validates exact response columns and the expected clip IDs in a completed reviewer CSV against the private answer key at `results/real_soundscape_aggregate_s43_s44_fixnames/blinded_annotation_review_20261025/answer_key_internal_DO_NOT_SHARE_WITH_REVIEWER.csv`. It writes aggregate counts and binary agreement to a new JSON file; it suppresses clip-level hidden labels and refuses to overwrite the requested output. It is a future analysis tool and is not part of training/evaluation. Existing tests exercise the validator; current reporting says no completed responses are available, so no review result is claimed.

Run: `python scripts/analysis/analyze_blinded_soundscape_review.py path/to/completed_review.csv --output-json path/to/new_aggregate_report.json`

## Checkpoint and environment diagnostics

There are no additional standalone environment diagnostic scripts under `scripts/`. The MoCo checkpoint diagnostic is cataloged under Evaluation and metrics because it computes representation diagnostics from model outputs.

## Packaging and maintenance

### Download the pinned PANNs checkpoint

Downloads the published Cnn14 checkpoint to the requested path (default `data/pretrained/panns/Cnn14_mAP=0.431.pth`), verifies the pinned MD5 and SHA-256, then promotes the temporary file. Requires `curl` and a SHA-256 utility on a network-enabled host.

Run: `bash scripts/maintenance/download_panns_checkpoint.sh data/pretrained/panns/Cnn14_mAP=0.431.pth`

### Package a source ZIP

Builds a deterministic source-only ZIP from an explicit allowlist of code, configuration, and documentation. It excludes datasets, results payloads, checkpoints, caches, environments, and secrets. The output path is required and should be outside the repository.

Run: `python3 scripts/maintenance/package_source_zip.py --output /path/to/source.zip`

## Historical or obsolete utilities

No active script was classified as obsolete. The ten historical `.bak` files were moved to `archive/backups/`; they remain outside runtime imports and are documented with hashes in the backup inventory. No maintained script, test, or documented workflow depends on them.
