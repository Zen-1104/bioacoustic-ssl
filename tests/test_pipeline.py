import json
import os
import random
import socket
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

import torch
import torch.nn.functional as F
import torchaudio

from bioacoustic_data.augmentations import SpecAugment
from bioacoustic_data.birdclef_dataset import BirdCLEFDataset
from bioacoustic_data.xeno_canto_dataset import XenoCantoDataset
from bioacoustic_data.noise_mixer import (
    NoiseMixer, MINIMUM_MIXABLE_RMS, validate_minimum_mixable_rms,
)
from experiments.exp4_pretrained_feature_baselines import (
    pool_segment_embeddings, BirdNetV3Encoder, PannsCnn14Encoder, load_external_baseline_crops, crop_policy_description,
)
from results.aggregate_pretrained_baselines import load_external_records, summarize_external_records, write_outputs
from results.aggregate_real_soundscape import (
    load_soundscape_records, summarize_soundscape_records, write_soundscape_outputs,
    recording_cluster_bootstrap,
)
from experiments.real_soundscape_eval import (
    load_annotation_intervals, build_segment_targets, compute_multilabel_ap,
    calculate_annotation_coverage, sample_background_segment_ids, repeat_pad_segment,
)
from evaluation.metrics import info_nce_loss
from experiments.exp1_pretrain_moco import atomic_torch_save
from experiments.smoke_external_baselines import count_distinct_feature_rows
from evaluation.linear_probe_eval import evaluate_accuracy
from evaluation.stress_test_eval import evaluate_at_snr
from results.aggregate_results import load_all_seed_results, _metric_value
from results.paired_tests import MODELS, _bootstrap_ci, analyze_seed, holm_adjust, validate_no_skipped_noise_mixes
from models.backbone_cnn10 import CNN_10_Backbone
from models.moco_wrapper import MoCo_Wrapper
from models.projection_head import Projection_Head
from scripts.create_birdclef_splits import split_indices, main as create_splits_main
from scripts.audit_dataset_overlap import main as audit_overlap_main
from scripts.audit_soundscape_overlap import audit_overlap as audit_soundscape_overlap
from scripts.preflight_xeno_audio import (
    main as preflight_xeno_main, make_crops, cleanup_stale_staging_dirs, seeded_processing_order,
    wilson_lower_bound, _slurm_job_is_active,
)
from scripts.create_habitat_split import create_habitat_split
from scripts.preflight_birdclef_audio import (
    dataset_source_fingerprint, run_or_reuse_preflight, main as preflight_birdclef_main,
    DEFAULT_MAX_READ_ERROR_FRACTION, DEFAULT_MAX_EXCLUDED_FRACTION, PREFLIGHT_SCHEMA_VERSION,
)
from utils.audio_io import (DEFAULT_EVALUATION_MINIMUM_RMS, EVALUATION_CROP_POLICY, load_and_preprocess_audio, load_evaluation_center_crop, load_fixed_crops, log_mel_spectrogram)
from utils.dataset_manifest import fingerprint_paths_labels, fingerprint_relative_paths, validate_manifest_file_index
from utils.habitat_split import read_verified_habitat_csv
from utils.stress_config import validate_snr_levels
from utils.label_scope import ActiveClassDataset
from utils.preflight_metadata import validate_preflight_audit
from utils.seed import capture_rng_state, restore_rng_state, set_seed


class TestBioacousticPipeline(unittest.TestCase):

    def test_minimum_rms_matches_noise_mixer_power_floor(self):
        self.assertAlmostEqual(MINIMUM_MIXABLE_RMS, 1e-5, places = 12)
        self.assertEqual(validate_minimum_mixable_rms(1e-5), 1e-5)
        self.assertEqual(validate_minimum_mixable_rms(2e-5), 2e-5)
        
        for value in (0.0, 9e-6, -1.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(minimum_rms = value):
                with self.assertRaisesRegex(ValueError, "minimum-rms"):
                    validate_minimum_mixable_rms(value)
                    
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(power_epsilon = value):
                with self.assertRaisesRegex(ValueError, "power_epsilon"):
                    NoiseMixer(power_epsilon = value)

    def test_birdclef_cli_rejects_incompatible_minimum_rms_before_data_access(self):
        with self.assertRaises(SystemExit) as split_error:
            create_splits_main(["--minimum-rms", "9e-6"])
        self.assertEqual(split_error.exception.code, 2)
        
        with self.assertRaises(SystemExit) as preflight_error:
            preflight_birdclef_main(["--minimum-rms", "9e-6"])
            
        self.assertEqual(preflight_error.exception.code, 2)

    def test_preflight_api_rejects_incompatible_minimum_rms_before_scan(self):
        dataset = SimpleNamespace(file_list = ["not-read.wav"], relative_paths = ["not-read.wav"], labels = [0])
        
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "minimum-rms"):
            run_or_reuse_preflight(
                dataset, 
                sample_rate = 16000, 
                clip_sec = 10.0, 
                minimum_rms = 9e-6,
                output_path = Path(tmp) / "preflight.json", 
                overwrite = True,
            )

    def test_birdclef_preflight_schema_bumped_for_rms_floor(self):
        self.assertEqual(PREFLIGHT_SCHEMA_VERSION, 5)

    def test_stress_aggregator_rejects_results_with_skipped_noise_mixes(self):
        raw = {
            "accuracy": 0.5, "n": 2,
            "predictions": [
                {"file": "a.wav", "correct": True, "y_true": 0, "y_pred": 0},
                {"file": "b.wav", "correct": False, "y_true": 1, "y_pred": 0},
            ],
            "n_silent_mix_skipped": 1, "silent_mix_skipped_files": ["b.wav"],
        }
        
        with self.assertRaisesRegex(ValueError, "not valid samples of the requested noisy-SNR"):
            _metric_value(raw, "stress.json", "unseen_habitat/moco_habitat/0dB")

    def test_paired_tests_reject_clean_fallback_rows_in_noisy_conditions(self):
        payload = {pool: {model: {"0dB": {"n_silent_mix_skipped": 0, "silent_mix_skipped_files": []}}
                          for model in MODELS.values()}
                   for pool in ("seen_habitat", "unseen_habitat")}
                   
        payload["unseen_habitat"]["moco_habitat"]["0dB"] = {
            "n_silent_mix_skipped": 1, "silent_mix_skipped_files": ["silent.wav"]
        }
        
        with self.assertRaisesRegex(ValueError, "scored as clean fallback"):
            validate_no_skipped_noise_mixes(payload, ["0dB"])


    def test_soundscape_overlap_audit_detects_identical_file_and_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sound = root / "soundscapes"
            train = root / "training"
            sound.mkdir(); train.mkdir()
            
            (sound / "shared.flac").write_bytes(b"identical full recording bytes")
            (train / "shared.flac").write_bytes(b"identical full recording bytes")
            (train / "another.wav").write_bytes(b"different audio content")
            
            report = audit_soundscape_overlap(sound, [train])
            
            self.assertEqual(report["n_soundscape_files"], 1)
            self.assertEqual(report["n_training_files"], 2)
            self.assertEqual(report["basename_match_group_count"], 1)
            self.assertEqual(report["exact_sha256_match_group_count"], 1)
            self.assertTrue(report["overlap_detected"])
            self.assertTrue(any("re-encoded" in item for item in report["limitations"]))

    def test_real_soundscape_aggregator_refuses_capped_runs_and_writes_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            
            identity = {
                "annotation_csv_sha256": "ann-sha", 
                "recording_names_sha256": "rec-sha",
                "sample_rate": 16000, 
                "clip_sec": 10.0, 
                "max_recordings": None,
                "background_segments_per_recording": 10, 
                "background_seed": 4242, 
                "min_annotation_coverage": 0.5,
            }
            
            for seed, ap in ((42, 0.45), (123, 0.55)):
                data = {
                    "schema_version": 2, 
                    "seed": seed, 
                    "fraction": 0.1, 
                    "capped_smoke_run": False,
                    "identity": identity, 
                    "active_label_map": {"bird_a": 0, "bird_b": 1},
                    "full_label_map": {"bird_a": 0, "bird_b": 1},
                    "n_active_classes": 2, 
                    "n_total_classes": 2, 
                    "n_excluded_classes": 0,
                    "n_silent_files_excluded": 0, 
                    "silent_excluded_filenames": [],
                    "n_preflight_excluded_files": 0, 
                    "preflight_excluded_filenames": [],
                    "n_preflight_read_error_files_excluded": 0, 
                    "preflight_read_error_filenames": [],
                    "preflight_reason_histogram": {}, 
                    "preflight_read_error_fraction": 0.0,
                    "preflight_excluded_fraction": 0.0, 
                    "preflight_max_read_error_fraction": 0.01,
                    "preflight_max_excluded_fraction": 0.05,
                    "rare_class_policy": "exclude", 
                    "excluded_classes": [], 
                    "class_scope_note": "fixture scope",
                    "n_recordings_evaluated": 10, 
                    "n_annotated_segments": 30, 
                    "n_background_segments": 12,
                    "annotation_code_coverage_fraction": 0.75, 
                    "minimum_annotation_coverage_required": 0.5,
                    "background_segments_per_recording": 10,
                    "models": {"habitat": {"macro_average_precision": ap, "micro_average_precision": ap + 0.05,
                                               "background_segment_count": 12, "background_top1_confidence_ge_0_5_rate": 0.1}},
                }
                (root / f"real_soundscape_s{seed}_frac0.1_hash.json").write_text(json.dumps(data), encoding = "utf-8")
                
            # An unrelated capped smoke result must not be selected over the full run.
            smoke = {
                "seed": 42, 
                "fraction": 0.1, 
                "capped_smoke_run": True,
                "identity": {**identity, 
                "max_recordings": 2}, 
                "active_label_map": {"bird_a": 0, "bird_b": 1},
                "models": {"habitat": {"macro_average_precision": 0.99, "micro_average_precision": 0.99}},
            }
            
            (root / "real_soundscape_s42_frac0.1_smoke.json").write_text(json.dumps(smoke), encoding = "utf-8")
            records = load_soundscape_records(root, 0.1, seeds = (42, 123), models = ("habitat",))
            rows = summarize_soundscape_records(records, seeds = (42, 123), models = ("habitat",))
            
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["macro_average_precision_mean"], 0.5)
            self.assertAlmostEqual(rows[0]["background_top1_confidence_ge_0_5_rate_mean"], 0.1)
            
            csv_path, tex_path, cluster_path = write_soundscape_outputs(rows, root)
            
            self.assertTrue(csv_path.is_file())
            self.assertTrue(tex_path.is_file())
            self.assertIsNone(cluster_path)


    def test_birdnet_v3_encoder_calls_supported_encode_arrays_api(self):
        class FakeBirdNET:
            def __init__(self):
                self.calls = []
            def encode_arrays(self, inputs, *, batch_size, device):
                self.calls.append((inputs, batch_size, device))
                
                return SimpleNamespace(embeddings = np.asarray([[1.0, 3.0], [3.0, 5.0]], dtype = np.float32))

        encoder = BirdNetV3Encoder.__new__(BirdNetV3Encoder)
        encoder.device = "CPU"
        encoder.model = FakeBirdNET()
        encoder.model_name = "fake"
        encoder.feature_dim = 2
        features = encoder.encode_batch(torch.zeros(2, 1, 32000), 32000)
        
        self.assertEqual(features.shape, (2, 2))
        np.testing.assert_allclose(features, np.asarray([[2.0, 4.0], [2.0, 4.0]], dtype = np.float32))
        self.assertEqual(len(encoder.model.calls), 2)
        self.assertEqual(encoder.model.calls[0][1:], (1, "CPU"))

    def test_real_soundscape_aggregator_rejects_different_background_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            
            for seed, background_seed in ((42, 4242), (123, 4243)):
                data = {
                    "schema_version": 2, 
                    "seed": seed, 
                    "fraction": 0.1, 
                    "capped_smoke_run": False,
                    
                    "identity": {"annotation_csv_sha256": "ann", "recording_names_sha256": "rec",
                                 "sample_rate": 16000, "clip_sec": 10.0, "max_recordings": None,
                                 "background_segments_per_recording": 10, "background_seed": background_seed,
                                 "min_annotation_coverage": 0.5},
                                 
                    "active_label_map": {"bird_a": 0}, 
                    "full_label_map": {"bird_a": 0},
                    "n_active_classes": 1, 
                    "n_total_classes": 1, "n_excluded_classes": 0,
                    "n_silent_files_excluded": 0, 
                    "silent_excluded_filenames": [],
                    "n_preflight_excluded_files": 0, 
                    "preflight_excluded_filenames": [],
                    "n_preflight_read_error_files_excluded": 0, 
                    "preflight_read_error_filenames": [],
                    "preflight_reason_histogram": {}, 
                    "preflight_read_error_fraction": 0.0,
                    "preflight_excluded_fraction": 0.0, 
                    "preflight_max_read_error_fraction": 0.01,
                    "preflight_max_excluded_fraction": 0.05,
                    "rare_class_policy": "exclude", 
                    "excluded_classes": [], 
                    "class_scope_note": "fixture scope",
                    "n_recordings_evaluated": 10, 
                    "n_annotated_segments": 30, 
                    "n_background_segments": 10,
                    "annotation_code_coverage_fraction": 0.8, 
                    "minimum_annotation_coverage_required": 0.5,
                    "background_segments_per_recording": 10,
                    
                    "models": {"habitat": {"macro_average_precision": 0.5, "micro_average_precision": 0.5}},
                }
                
                (root / f"real_soundscape_s{seed}_frac0.1_hash.json").write_text(json.dumps(data), encoding = "utf-8")
                
            with self.assertRaisesRegex(ValueError, "fixed background sampling"):
                load_soundscape_records(root, 0.1, seeds = (42, 123), models = ("habitat",))

    def test_real_soundscape_aggregator_rejects_incomplete_audio_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            
            for seed in (42, 123):
                data = {
                    "seed": seed, 
                    "fraction": 0.1, 
                    "capped_smoke_run": False,
                    
                    "identity": {"annotation_csv_sha256": "ann", "recording_names_sha256": "rec",
                                 "sample_rate": 16000, "clip_sec": 10.0, "max_recordings": None},
                                 
                    "active_label_map": {"bird_a": 0},
                    "n_recordings_evaluated": 10, 
                    "n_annotated_segments": 30,
                    "n_annotation_recordings_missing_audio": 1, 
                    "incomplete_audio_inventory": True,
                    
                    "models": {"habitat": {"macro_average_precision": 0.5, "micro_average_precision": 0.5}},
                }
                
                (root / f"real_soundscape_s{seed}_frac0.1_hash.json").write_text(json.dumps(data), encoding = "utf-8")
                
            with self.assertRaisesRegex(ValueError, "incomplete soundscape audio inventory"):
                load_soundscape_records(root, 0.1, seeds = (42, 123), models = ("habitat",))


    def test_external_crop_policy_labels_do_not_claim_rms_recovery_for_center_crop(self):
        self.assertEqual(crop_policy_description(1), EVALUATION_CROP_POLICY)
        self.assertEqual(crop_policy_description(3), "three-temporal-crops-rms-recovery-mean-pooling-v2")
        
        with self.assertRaises(ValueError):
            crop_policy_description(2)

    def test_external_baseline_aggregator_validates_complete_seed_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            
            for seed, acc in ((42, 0.6), (123, 0.7)):
                record = {
                    "schema_version": 3, 
                    "mode": "external_pretrained_linear_probe", 
                    "variant": "panns",
                    "seed": seed, 
                    "fraction": 0.1, 
                    "dataset_fingerprint": "same-dataset",
                    "active_label_map": {"bird_a": 0, "bird_b": 1}, 
                    "num_classes": 2,
                    "n_active_classes": 2, 
                    "n_total_classes": 2, 
                    "n_excluded_classes": 0,
                    "excluded_classes": [], 
                    "n_silent_files_excluded": 0, 
                    "silent_excluded_filenames": [],
                    "n_preflight_excluded_files": 0, 
                    "preflight_excluded_filenames": [],
                    "n_preflight_read_error_files_excluded": 0, 
                    "preflight_read_error_filenames": [],
                    "preflight_reason_histogram": {}, 
                    "preflight_read_error_fraction": 0.0,
                    "preflight_excluded_fraction": 0.0, 
                    "preflight_max_read_error_fraction": 0.01,
                    "preflight_max_excluded_fraction": 0.05,
                    "rare_class_policy": "exclude", 
                    "class_scope_note": "fixture scope",
                    "num_fixed_crops_requested": 3, 
                    "fixed_crop_aggregation": "mean of per-crop embeddings to one recording-level row",
                    "crop_policy": "three-temporal-crops-rms-recovery-mean-pooling-v2", 
                    "minimum_rms": 1e-5,
                    "feature_extraction_spec_version": "v1.6.3-external-feature-pooling-unified-center-recovery-v4",
                    "fixed_crop_count_min": 3, 
                    "fixed_crop_count_mean": 3.0, 
                    "fixed_crop_count_max": 3,
                    "fixed_crop_shortfall_filenames": [],
                    "model_name": "PANNs fake", 
                    "external_checkpoint": "official fake weights",
                    "external_checkpoint_sha256": "a" * 64,
                    
                    "encoder_identity": {"model_file_sha256": "a" * 64, "package_version": "0.1.1", "model_version": "test"},
                    "test_accuracy": acc, "test_macro_f1": acc - 0.1, "test_balanced_accuracy": acc - 0.05,
                }
                
                (root / f"pretrained_probe_panns_s{seed}_frac0.1.json").write_text(json.dumps(record), encoding = "utf-8")
                
            with self.assertRaisesRegex(ValueError, "exploratory rather than the primary comparison"):
                load_external_records(root, seeds = (42, 123), fractions = (0.1,), models = ("panns",))
                
            records = load_external_records(root, seeds = (42, 123), fractions = (0.1,), models = ("panns",), allow_three_crop_analysis = True)
            rows = summarize_external_records(records, seeds = (42, 123), fractions = (0.1,), models = ("panns",))
            
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["test_accuracy_mean"], 0.65)
            self.assertEqual(rows[0]["n_active_classes"], 2)
            self.assertEqual(rows[0]["n_total_classes"], 2)
            self.assertEqual(rows[0]["rare_class_policy"], "exclude")
            self.assertIn("excluded_classes_json", rows[0])
            
            csv_path, accuracy_tex, macro_f1_tex = write_outputs(rows, root)
            self.assertTrue(csv_path.is_file())
            self.assertTrue(accuracy_tex.is_file())
            self.assertTrue(macro_f1_tex.is_file())


    def test_external_baseline_aggregator_rejects_mismatched_fixed_crop_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            
            for model in ("panns", "birdnet"):
                crop_count = 3 if model == "panns" else 1
                crop_policy = crop_policy_description(crop_count)
                
                for seed in (42, 123):
                    digest = ("a" if model == "panns" else "b") * 64
                    
                    record = {
                        "schema_version": 3, 
                        "mode": "external_pretrained_linear_probe", 
                        "variant": model,
                        "seed": seed, 
                        "fraction": 0.1, 
                        "dataset_fingerprint": "same-dataset",
                        "active_label_map": {"bird_a": 0}, "num_classes": 1,
                        "n_active_classes": 1, 
                        "n_total_classes": 1, 
                        "n_excluded_classes": 0,
                        "excluded_classes": [], 
                        "n_silent_files_excluded": 0, 
                        "silent_excluded_filenames": [],
                        "n_preflight_excluded_files": 0, 
                        "preflight_excluded_filenames": [],
                        "n_preflight_read_error_files_excluded": 0, 
                        "preflight_read_error_filenames": [],
                        "preflight_reason_histogram": {}, 
                        "preflight_read_error_fraction": 0.0,
                        "preflight_excluded_fraction": 0.0, 
                        "preflight_max_read_error_fraction": 0.01,
                        "preflight_max_excluded_fraction": 0.05,
                        "rare_class_policy": "exclude", 
                        "class_scope_note": "fixture scope",
                        "num_fixed_crops_requested": crop_count,
                        "fixed_crop_aggregation": "mean of per-crop embeddings to one recording-level row",
                        "crop_policy": crop_policy, 
                        "minimum_rms": 1e-5,
                        "feature_extraction_spec_version": "v1.6.3-external-feature-pooling-unified-center-recovery-v4",
                        "fixed_crop_count_min": crop_count, 
                        "fixed_crop_count_mean": float(crop_count),
                        "fixed_crop_count_max": crop_count, 
                        "fixed_crop_shortfall_filenames": [],
                        "model_name": model, 
                        "external_checkpoint": f"{model}.weights",
                        "external_checkpoint_sha256": digest,
                        
                        "encoder_identity": {"model_file_sha256": digest, "package_version": "1.1.1", "model_version": "test"},
                        "test_accuracy": 0.6, "test_macro_f1": 0.5, "test_balanced_accuracy": 0.5,
                    }
                    
                    (root / f"pretrained_probe_{model}_s{seed}_frac0.1.json").write_text(json.dumps(record), encoding = "utf-8")
                    
            with self.assertRaisesRegex(ValueError, "different fixed-crop policies"):
                load_external_records(root, seeds = (42, 123), fractions = (0.1,), models = ("panns", "birdnet"), allow_three_crop_analysis = True)

    def test_active_class_dataset_remaps_excluded_class_ids_contiguously(self):
        class BaseDataset(torch.utils.data.Dataset):
            relative_paths = ["a.wav", "b.wav", "c.wav"]
            label_map = {"species_a": 0, "species_b": 1, "species_c": 2}
            labels = [0, 1, 2]

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return torch.tensor([float(idx)]), self.labels[idx]

        dataset = ActiveClassDataset(BaseDataset(), {"species_a": 0, "species_c": 2})
        self.assertEqual(dataset.active_label_map, {"species_a": 0, "species_c": 1})
        self.assertEqual(dataset[0][1], 0)
        self.assertEqual(dataset[2][1], 1)
        
        with self.assertRaisesRegex(ValueError, "excluded class id"):
            dataset[1]

    def test_atomic_torch_save_accepts_path_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            atomic_torch_save({"epoch": 1, "value": torch.tensor([2.0])}, path)
            
            loaded = torch.load(path, map_location = "cpu", weights_only = True)
            self.assertEqual(loaded["epoch"], 1)
            self.assertTrue(torch.equal(loaded["value"], torch.tensor([2.0])))
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_tensor_shapes(self):
        backbone = CNN_10_Backbone()
        head = Projection_Head(input_dim = 512, hidden_dim = 512, out_dim = 128)
        dummy_spec = torch.randn(2, 1, 64, 313)
        
        with torch.no_grad():
            features = backbone(dummy_spec)
            projections = head(features)
            
        self.assertEqual(features.shape, (2, 512))
        self.assertEqual(projections.shape, (2, 128))

    def test_specaugment_supports_four_dimensional_frontend_output(self):
        aug = SpecAugment(freq_mask_param = 8, time_mask_param = 16)
        spec = torch.randn(1, 1, 64, 313)
        result = aug(spec)
        
        self.assertEqual(result.shape, spec.shape)
        self.assertTrue(torch.isfinite(result).all())

    def test_log_mel_frontend_output_shape(self):
        waveform = torch.randn(1, 16000)
        spec = log_mel_spectrogram(waveform)
        
        self.assertEqual(spec.shape[:3], (1, 1, 64))
        self.assertGreater(spec.shape[-1], 0)
        self.assertTrue(torch.isfinite(spec).all())

    def test_fixed_crop_cache_uses_three_spread_crops_and_center_for_single_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long.wav"
            
            # Each section has a different signal so the temporal crop positions
            # and averaging protocol can be exercised without external data.
            t = torch.arange(3000, dtype=torch.float32) / 1000.0
            wave = torch.sin(2 * torch.pi * (30.0 + 5.0 * torch.floor(t)) * t).unsqueeze(0) * 0.2
            torchaudio.save(str(path), wave, 1000)
            
            crops = load_fixed_crops(path, target_sr = 1000, duration_sec = 1.0, num_crops = 3, minimum_rms = 1e-5)
            self.assertEqual(tuple(crops.shape), (3, 1, 1000))
            self.assertTrue(torch.isfinite(crops).all())
            
            center = load_fixed_crops(path, target_sr = 1000, duration_sec = 1.0, num_crops = 1, minimum_rms = 1e-5)
            self.assertEqual(tuple(center.shape), (1, 1, 1000))
            
            torch.testing.assert_close(center[0], wave[:, 1000:2000], rtol = 0, atol = 2e-5)

    def test_primary_external_baseline_uses_shared_center_crop_recovery(self):
        dataset = SimpleNamespace(duration_sec = 10.0, minimum_rms = DEFAULT_EVALUATION_MINIMUM_RMS)
        expected = torch.arange(16000, dtype = torch.float32).unsqueeze(0)
        
        with patch("experiments.exp4_pretrained_feature_baselines.load_evaluation_center_crop", return_value = expected) as loader, \
             patch("experiments.exp4_pretrained_feature_baselines.load_fixed_crops") as fixed_loader:
            crops = load_external_baseline_crops(dataset, "recording.wav", 16000, 1)
            
        self.assertEqual(tuple(crops.shape), (1, 1, 16000))
        
        loader.assert_called_once_with(
            "recording.wav", 
            target_sr = 16000, 
            duration_sec = 10.0,
            minimum_rms = DEFAULT_EVALUATION_MINIMUM_RMS,
        )
        
        fixed_loader.assert_not_called()

    def test_smoke_feature_diversity_rejects_constant_rows(self):
        self.assertEqual(count_distinct_feature_rows(np.asarray([[1, 2], [1, 2], [1, 2]], dtype = np.float32)), 1)
        self.assertEqual(count_distinct_feature_rows(np.asarray([[1, 2], [1, 2], [1, 3]], dtype = np.float32)), 2)
        
        with self.assertRaises(FloatingPointError):
            count_distinct_feature_rows(np.asarray([[float("nan"), 0]], dtype = np.float32))

    def test_holm_bonferroni_adjustment_is_monotone_and_preserves_order(self):
        adjusted = holm_adjust([0.04, 0.001, 0.03])
        np.testing.assert_allclose(adjusted, [0.06, 0.003, 0.06])
        
        self.assertTrue(np.all(adjusted >= np.asarray([0.04, 0.001, 0.03])))
        self.assertEqual(holm_adjust([]).size, 0)
        
        with self.assertRaisesRegex(ValueError, "finite p-values"):
            holm_adjust([0.01, float("nan")])

    def test_preflight_fingerprint_ignores_mtime_but_tracks_inventory_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "bird.wav"
            audio.write_bytes(b"123456")
            
            dataset = SimpleNamespace(file_list = [str(audio)], relative_paths = ["species/bird.wav"],
                                     labels = [0], requested_root = root)
                                     
            first = dataset_source_fingerprint(dataset, 16000, 10.0, 1e-5)
            stat = audio.stat()
            os.utime(audio, ns = (stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
            second = dataset_source_fingerprint(dataset, 16000, 10.0, 1e-5)
            self.assertEqual(first, second)
            audio.write_bytes(b"123456789")
            third = dataset_source_fingerprint(dataset, 16000, 10.0, 1e-5)
            self.assertNotEqual(second, third)

    def test_preflight_records_read_errors_and_continues_scanning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good.wav"
            bad = root / "bad.wav"
            good.write_bytes(b"good-audio")
            bad.write_bytes(b"broken-audio")
            
            dataset = SimpleNamespace(
                file_list = [str(good), str(bad)],
                relative_paths = ["species/good.wav", "species/bad.wav"],
                labels = [0, 0], requested_root = root,
            )
            
            with patch("scripts.preflight_birdclef_audio.load_evaluation_center_crop",
                       side_effect = [torch.ones(1, 160000), RuntimeError("decoder failure")]):
                       
                manifest = run_or_reuse_preflight(
                    dataset, 
                    sample_rate = 16000, 
                    clip_sec = 10, 
                    minimum_rms = 1e-5,
                    output_path = root / "preflight.json", 
                    list_read_errors = True,
                    max_read_error_fraction = 0.5, 
                    max_excluded_fraction = 0.75,
                )
                
            self.assertEqual(manifest["num_files_scanned"], 2)
            self.assertEqual(manifest["n_read_error_files"], 1)
            self.assertEqual(manifest["read_error_filenames"], ["species/bad.wav"])
            self.assertEqual(manifest["excluded_filenames"], ["species/bad.wav"])
            self.assertEqual(manifest["status"], "completed_with_exclusions")
            self.assertEqual(manifest["reason_histogram"], {"read_error:RuntimeError": 1})

    def test_preflight_metadata_validator_rejects_inconsistent_histogram(self):
        record = {
            "n_preflight_excluded_files": 1, 
            "preflight_excluded_filenames": ["species/a.wav"],
            "n_preflight_read_error_files_excluded": 1, 
            "preflight_read_error_filenames": ["species/a.wav"],
            "preflight_reason_histogram": {"read_error:RuntimeError": 1},
            "preflight_read_error_fraction": 0.01, 
            "preflight_excluded_fraction": 0.01,
            "preflight_max_read_error_fraction": 0.01, 
            "preflight_max_excluded_fraction": 0.05,
        }
        
        validated = validate_preflight_audit(record, context = "fixture")
        self.assertEqual(validated["n_preflight_excluded_files"], 1)
        record["preflight_reason_histogram"] = {"read_error:RuntimeError": 2}
        
        with self.assertRaisesRegex(ValueError, "histogram does not sum"):
            validate_preflight_audit(record, context = "fixture")

    def test_preflight_fails_closed_above_read_error_cap_and_reports_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relative = []
            file_paths = []
            labels = []
            
            for index in range(100):
                name = f"species/bird_{index:03d}.wav"
                path = root / name
                path.parent.mkdir(parents = True, exist_ok = True)
                path.write_bytes(b"fixture")
                relative.append(name)
                file_paths.append(str(path))
                labels.append(0)
                
            dataset = SimpleNamespace(
                file_list = file_paths, 
                relative_paths = relative, 
                labels = labels, 
                requested_root = root
            )
            
            def fake_crop(path, **kwargs):
                if Path(path).name in {"bird_000.wav", "bird_001.wav"}:
                    raise RuntimeError("decoder failed")
                return torch.ones(1, 16000)
                
            manifest_path = root / "preflight.json"
            with patch("scripts.preflight_birdclef_audio.load_evaluation_center_crop", side_effect = fake_crop):
                with self.assertRaisesRegex(RuntimeError, "max-read-error-fraction=1.00%"):
                    run_or_reuse_preflight(
                        dataset, 
                        sample_rate = 16000, 
                        clip_sec = 1.0, 
                        minimum_rms = 1e-5,
                        output_path = manifest_path,
                    )
                    
            saved = json.loads(manifest_path.read_text(encoding = "utf-8"))
            self.assertFalse(saved["thresholds_passed"])
            self.assertEqual(saved["status"], "threshold_exceeded")
            self.assertEqual(saved["reason_histogram"], {"read_error:RuntimeError": 2})
            self.assertAlmostEqual(saved["read_error_fraction"], 0.02)
            self.assertEqual(DEFAULT_MAX_READ_ERROR_FRACTION, 0.01)
            self.assertEqual(DEFAULT_MAX_EXCLUDED_FRACTION, 0.05)

    def test_preflight_fails_closed_above_total_exclusion_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relative, file_paths = [], []
            
            for index in range(100):
                rel = f"species/bird_{index:03d}.wav"
                path = root / rel
                path.parent.mkdir(parents = True, exist_ok = True)
                path.write_bytes(b"fixture")
                relative.append(rel)
                file_paths.append(str(path))
                
            dataset = SimpleNamespace(file_list = file_paths, relative_paths = relative,
                                     labels = [0] * 100, requested_root = root)
                                     
            def fake_crop(path, **kwargs):
                if Path(path).name in {f"bird_{i:03d}.wav" for i in range(6)}:
                    raise ValueError("SILENT_CROP_REJECTED: fixture silence")
                    
                return torch.ones(1, 16000)
                
            manifest_path = root / "preflight.json"
            with patch("scripts.preflight_birdclef_audio.load_evaluation_center_crop", side_effect = fake_crop):
                with self.assertRaisesRegex(RuntimeError, "max-excluded-fraction=5.00%"):
                    run_or_reuse_preflight(
                        dataset, 
                        sample_rate = 16000, 
                        clip_sec = 1.0, 
                        minimum_rms = 1e-5,
                        output_path = manifest_path,
                    )
                    
            saved = json.loads(manifest_path.read_text(encoding = "utf-8"))
            self.assertFalse(saved["thresholds_passed"])
            self.assertEqual(saved["reason_histogram"], {"silent_or_unusable": 6})
            self.assertAlmostEqual(saved["excluded_fraction"], 0.06)

    def test_evaluation_crop_recovery_matches_birdclef_dataset_and_external_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "birdclef"
            species = root / "species_a"
            species.mkdir(parents = True)
            wav_path = species / "near_silent_center.wav"
            torch.manual_seed(17)
            waveform = torch.zeros(1, 64000, dtype = torch.float32)
            waveform[:, :24000] = torch.randn(1, 24000) * 0.1
            torchaudio.save(str(wav_path), waveform, 16000)

            recovered = load_evaluation_center_crop(
                wav_path, 
                target_sr = 16000, 
                duration_sec = 1.0, 
                minimum_rms = DEFAULT_EVALUATION_MINIMUM_RMS,
            )
            
            self.assertGreater(float(recovered.square().mean().sqrt()), DEFAULT_EVALUATION_MINIMUM_RMS)
            
            external_dataset = SimpleNamespace(
                duration_sec = 1.0, 
                minimum_rms = DEFAULT_EVALUATION_MINIMUM_RMS,
            )
            
            external = load_external_baseline_crops(external_dataset, wav_path, 16000, 1)
            torch.testing.assert_close(external[0], recovered, rtol = 0, atol = 0)

            bird_dataset = BirdCLEFDataset(
                str(root), 
                sample_rate = 16000, 
                duration_sec = 1.0, 
                crop_mode = "center",
                minimum_rms = DEFAULT_EVALUATION_MINIMUM_RMS,
            )
            
            mel, _label = bird_dataset[0]
            expected_mel = log_mel_spectrogram(recovered, sample_rate = 16000)
            torch.testing.assert_close(mel, expected_mel.squeeze(0), rtol = 0, atol = 0)

    def test_deterministic_warn_only_is_forwarded_to_pytorch(self):
        with patch("utils.seed.torch.use_deterministic_algorithms") as deterministic_algorithms:
            set_seed(431, deterministic = True, deterministic_warn_only = True)
        deterministic_algorithms.assert_called_once_with(True, warn_only = True)

    def test_view_batch_policy_combines_query_views_in_one_forward(self):
        class TrackingBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(3)
                self.batch_sizes = []
                
            def forward(self, x):
                self.batch_sizes.append(int(x.shape[0]))
                return self.bn(x)

        def make_model(policy):
            query, key = TrackingBackbone(), TrackingBackbone()
            
            model = MoCo_Wrapper(
                query, 
                key, 
                torch.nn.Linear(3, 2), 
                torch.nn.Linear(3, 2),
                dim = 2, K = 4, m = 0.5, 
                view_batch_policy = policy
            )
                
            model.train()
            return model

        x_clean = torch.randn(2, 3) * 2 + 3
        x_noisy = torch.randn(2, 3) - 4
        separate = make_model("separate")
        concatenated = make_model("concatenated")
        outs_separate = separate(x_clean, x_noisy, symmetric = True)
        outs_concat = concatenated(x_clean, x_noisy, symmetric = True)
        
        self.assertEqual(separate.encoder_q[0].batch_sizes, [2, 2])
        self.assertEqual(concatenated.encoder_q[0].batch_sizes, [4])
        
        for output in (*outs_separate[:4], *outs_concat[:4]):
            self.assertTrue(torch.isfinite(output).all())

    def test_bn_policy_ablation_policies_apply_expected_buffer_rules(self):
        class TinyBackbone(torch.nn.Sequential):
            def __init__(self):
                super().__init__(torch.nn.BatchNorm1d(3))

        def make_model(policy):
            return MoCo_Wrapper(
                TinyBackbone(), 
                TinyBackbone(), 
                torch.nn.Linear(3, 2),
                torch.nn.Linear(3, 2), 
                dim = 2, K = 4, m = 0.5, 
                bn_policy = policy
            )

        copy_model = make_model("copy_query")
        ema_model = make_model("ema_query")
        frozen_model = make_model("frozen_key")
        
        for model in (copy_model, ema_model, frozen_model):
            qbn = model.encoder_q[0][0]
            kbn = model.encoder_k[0][0]
            qbn.running_mean.fill_(2.0)
            qbn.running_var.fill_(4.0)
            qbn.num_batches_tracked.fill_(7)
            kbn.running_mean.zero_()
            kbn.running_var.fill_(1.0)
            kbn.num_batches_tracked.fill_(1)
            
        copy_model._sync_key_buffers_from_query()
        ema_model._sync_key_buffers_from_query()
        frozen_model._sync_key_buffers_from_query()
        
        torch.testing.assert_close(copy_model.encoder_k[0][0].running_mean, torch.full((3,), 2.0))
        torch.testing.assert_close(ema_model.encoder_k[0][0].running_mean, torch.full((3,), 1.0))
        torch.testing.assert_close(ema_model.encoder_k[0][0].running_var, torch.full((3,), 2.5))
        self.assertEqual(int(ema_model.encoder_k[0][0].num_batches_tracked), 7)
        
        torch.testing.assert_close(frozen_model.encoder_k[0][0].running_mean, torch.zeros(3))
        torch.testing.assert_close(frozen_model.encoder_k[0][0].running_var, torch.ones(3))
        self.assertEqual(int(frozen_model.encoder_k[0][0].num_batches_tracked), 1)

    def test_explicit_deterministic_mode_can_be_enabled_and_disabled(self):
        was_enabled = torch.are_deterministic_algorithms_enabled()
        
        try:
            set_seed(93, deterministic=True)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            a = torch.randn(4)
            set_seed(93, deterministic=True)
            torch.testing.assert_close(a, torch.randn(4), rtol = 0, atol = 0)
            
        finally:
            set_seed(93, deterministic = was_enabled)

    def test_noise_mixer_preserves_clean_rms_and_rejects_silence(self):
        mixer = NoiseMixer(sample_rate = 16000)
        clean = torch.randn(1, 16000)
        noise = torch.randn(1, 16000)
        mixed = mixer.mix(clean, noise, snr_db = 0.0)
        
        self.assertAlmostEqual(float(clean.square().mean().sqrt()), float(mixed.square().mean().sqrt()), places = 5)
        self.assertTrue(torch.isfinite(mixed).all())
        
        with self.assertRaisesRegex(ValueError, "silent or near-silent"):
            mixer.mix(clean, torch.zeros_like(noise), snr_db = 0.0)
            
        with self.assertRaisesRegex(ValueError, "Clean waveform"):
            mixer.mix(torch.zeros_like(clean), noise, snr_db = 0.0)

    def test_stress_snr_config_labels_match_exp3_result_keys(self):
        values, labels = validate_snr_levels([None, 20, 10.0, 0, -5])
        self.assertEqual(values, [None, 20.0, 10.0, 0.0, -5.0])
        self.assertEqual(labels, ["Clean", "20dB", "10dB", "0dB", "-5dB"])
        
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_snr_levels([None, 20, 20.0])
            
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_snr_levels([float("nan")])

    def test_noise_mixer_seeded_offset_is_reproducible(self):
        mixer = NoiseMixer(sample_rate = 16000)
        clean = torch.randn(1, 256)
        noise = torch.randn(1, 1024)
        
        g1 = torch.Generator(device = "cpu").manual_seed(20261010)
        g2 = torch.Generator(device = "cpu").manual_seed(20261010)
        
        first = mixer.mix(clean, noise, snr_db = 5.0, generator = g1)
        second = mixer.mix(clean, noise, snr_db = 5.0, generator = g2)
        self.assertTrue(torch.equal(first, second))

    def test_stress_evaluation_returns_per_recording_predictions(self):
        class Dataset(torch.utils.data.Dataset):
            def __len__(self):
                return 3
                
            def __getitem__(self, idx):
                value = float(idx - 1)
                label = 1 if value > 0 else 0
                
                return torch.tensor([value]), label, f"recording_{idx}.wav"

        class Backbone(torch.nn.Module):
            def forward(self, x):
                return x

        class Head(torch.nn.Module):
            def forward(self, x):
                return torch.cat([-x, x], dim = 1)

        result = evaluate_at_snr(Backbone(), Head(), torch.utils.data.DataLoader(Dataset(), batch_size = 2), "cpu")
        self.assertEqual(result["n"], 3)
        self.assertEqual(len(result["predictions"]), 3)
        self.assertEqual([row["file"] for row in result["predictions"]], ["recording_0.wav", "recording_1.wav", "recording_2.wav"])
        self.assertTrue(all("correct" in row and "y_true" in row and "y_pred" in row for row in result["predictions"]))

    def test_missing_metadata_singletons_do_not_count_as_independent_groups(self):
        labels = [0] * 10 + [1] * 15
        groups = ["author:only_known_author"] + [f"missing:file{i}" for i in range(9)]
        groups += [f"author:known_author_{i}" for i in range(15)]
        
        with self.assertRaisesRegex(ValueError, "distinct_groups<5"):
            split_indices(labels, groups, 42, "author_with_missing_singletons", min_recordings = 5, min_groups = 5, rare_class_policy = "error")
            
        train, val, test, _, _ = split_indices(
            labels, groups, 42, "author_with_missing_singletons", min_recordings = 5, min_groups = 5,
            rare_class_policy = "exclude",
        )
        
        self.assertTrue(all(labels[i] == 1 for i in train + val + test))

    def test_preflight_crops_are_spread_across_recording(self):
        torch.manual_seed(0)
        wave = torch.randn(1, 16000 * 60)
        crops = make_crops(wave, target_samples = 16000 * 10, num_crops = 3, min_rms = 1e-5)
        
        self.assertEqual(tuple(crops.shape), (3, 1, 160000))
        # Three output crops should represent the beginning, middle and end of a long recording.
        self.assertFalse(torch.equal(crops[0], crops[1]))
        self.assertFalse(torch.equal(crops[1], crops[2]))

    def test_cluster_bootstrap_is_deterministic_and_reports_unit_by_cardinality(self):
        differences = __import__("numpy").array([1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
        groups = ["a", "a", "b", "b", "c", "c"]
        
        import numpy as np
        low1, high1, unit1 = _bootstrap_ci(differences, np.random.default_rng(10), 300, groups)
        low2, high2, unit2 = _bootstrap_ci(differences, np.random.default_rng(10), 300, groups)
        self.assertEqual((low1, high1, unit1), (low2, high2, unit2))
        self.assertEqual(unit1, "group")
        _, _, recording_unit = _bootstrap_ci(differences, np.random.default_rng(10), 300, None)
        self.assertEqual(recording_unit, "recording")

    def test_split_rejects_small_class_counts_for_grouped_and_file_splits(self):
        labels = [0] * 4 + [1] * 10
        groups = [f"group_{i}" for i in range(len(labels))]
        
        with self.assertRaisesRegex(ValueError, "at least 5 recordings"):
            split_indices(labels, groups, 42, "file_level_no_metadata", rare_class_policy = "error")
            
        with self.assertRaisesRegex(ValueError, "at least 5 recordings"):
            split_indices(labels, groups, 42, "author", rare_class_policy = "error")

    def test_stress_aggregator_selects_fraction_and_rejects_missing_seed(self):
        def result_payload(seed, fraction):
            predictions = [
                {"file": "a.wav", "y_true": 0, "y_pred": 0, "correct": True},
                {"file": "b.wav", "y_true": 1, "y_pred": 1, "correct": True},
            ]
            
            metrics = {"accuracy": 1.0, "loss": 0.1, "n": 2, "predictions": predictions,
                       "n_silent_mix_skipped": 0, "silent_mix_skipped_files": []}
            return {
                "metadata": {
                    "schema_version": 5, 
                    "seed": seed, 
                    "fraction": fraction,
                    "dataset_fingerprint": "fixture_dataset", 
                    "config_sha256": "fixture_config",
                    "test_filenames": ["a.wav", "b.wav"],
                    "n_active_classes": 2, 
                    "n_total_classes": 2, 
                    "n_excluded_classes": 0,
                    "n_silent_files_excluded": 0, 
                    "rare_class_policy": "exclude",
                    "excluded_classes": [], 
                    "class_scope_note": "fixture scope",
                    "crop_policy": EVALUATION_CROP_POLICY, 
                    "crop_rms_threshold": 1e-5,
                    "n_preflight_excluded_files": 0, 
                    "preflight_excluded_filenames": [],
                    "n_preflight_read_error_files_excluded": 0, 
                    "preflight_read_error_filenames": [],
                    "preflight_reason_histogram": {}, 
                    "preflight_read_error_fraction": 0.0,
                    "preflight_excluded_fraction": 0.0, 
                    "preflight_max_read_error_fraction": 0.01,
                    "preflight_max_excluded_fraction": 0.05,
                },
                
                "seen_habitat": {m: {"Clean": metrics} for m in ("moco_habitat", "moco_standard", "moco_white_noise", "supervised_baseline", "supervised_noise_baseline")},
                "unseen_habitat": {m: {"Clean": metrics} for m in ("moco_habitat", "moco_standard", "moco_white_noise", "supervised_baseline", "supervised_noise_baseline")},
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "unseen_habitat_stress_test_matrix_s42.json").write_text(json.dumps(result_payload(42, 1.0)))
            (root / "unseen_habitat_stress_test_matrix_s42_frac0.05.json").write_text(json.dumps(result_payload(42, 0.05)))
            
            loaded = load_all_seed_results(root, expected_seeds = (42,), expected_snrs = ("Clean",), expected_fraction = 0.05)
            self.assertEqual(loaded[42]["data"]["metadata"]["fraction"], 0.05)
            
            stale = result_payload(123, 0.05)
            stale["metadata"]["schema_version"] = 4
            (root / "unseen_habitat_stress_test_matrix_s123_frac0.05.json").write_text(json.dumps(stale))
            
            with self.assertRaisesRegex(ValueError, "schema v5"):
                load_all_seed_results(root, expected_seeds = (42, 123), expected_snrs = ("Clean",), expected_fraction = 0.05)
                
            (root / "unseen_habitat_stress_test_matrix_s123_frac0.05.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "fraction=0.1"):
                load_all_seed_results(root, expected_seeds = (42, 123), expected_snrs = ("Clean",), expected_fraction = 0.1)

    def test_birdclef_noise_augmentation_range_and_probability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "birdclef"
            species = root / "species_a"
            species.mkdir(parents = True)
            torch.manual_seed(42)
            audio = torch.randn(1, 1600) * 0.1
            noise = torch.randn(1, 1600) * 0.1
            torchaudio.save(str(species / "bird.wav"), audio, 16000)
            clean = BirdCLEFDataset(str(root), sample_rate = 16000, duration_sec = 0.1, crop_mode = "center")
            
            no_noise_probability = BirdCLEFDataset(
                str(root), 
                sample_rate = 16000, 
                duration_sec = 0.1, 
                crop_mode = "center",
                noise_snr_db_range = (-5, 20), 
                noise_probability = 0.0,
            )
            
            matched_noise = BirdCLEFDataset(
                str(root), 
                sample_rate = 16000, 
                duration_sec = 0.1, 
                crop_mode = "center",
                noise_pool = [noise], 
                noise_mixer = NoiseMixer(sample_rate = 16000),
                noise_snr_db_range = (0, 0), 
                noise_probability = 1.0,
            )
            
            self.assertTrue(torch.equal(clean[0][0], no_noise_probability[0][0]))
            self.assertFalse(torch.allclose(clean[0][0], matched_noise[0][0]))
            self.assertEqual(matched_noise.noise_snr_db_range, (0, 0))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "birdclef"
            (root / "species_a").mkdir(parents=True)
            torchaudio.save(str(root / "species_a" / "bird.wav"), torch.randn(1, 1600), 16000)
            
            with self.assertRaisesRegex(ValueError, "endpoints must be integers"):
                BirdCLEFDataset(str(root), noise_pool = [torch.randn(1, 1600)], noise_mixer = NoiseMixer(), noise_snr_db_range = (-4.5, 20.0))

    def test_soundscape_annotation_columns_and_segment_targets(self):
        import pandas as pd
        
        with tempfile.TemporaryDirectory() as tmp:
            annotation_path = Path(tmp) / "annotation.csv"
            
            pd.DataFrame([
                {"Audio filename": "site_01.flac", "Start Time (s)": 8.0, "End Time (s)": 12.0, "eBird Species Code": "sp_a"},
                {"Audio filename": "site_01.flac", "Start Time (s)": 11.0, "End Time (s)": 13.0, "eBird Species Code": "sp_b"},
            ]).to_csv(annotation_path, index = False)
            
            frame = load_annotation_intervals(annotation_path)
            targets = build_segment_targets(frame, "site_01.flac", 10.0, 2, {"sp_a": 0, "sp_b": 1}, 30.0)
            
            self.assertEqual(sorted(targets), [0, 1])
            self.assertEqual(targets[0].tolist(), [1, 0])
            self.assertEqual(targets[1].tolist(), [1, 1])

    def test_soundscape_multilabel_average_precision_and_birdnet_embedding_pool(self):
        targets = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype = np.uint8)
        scores = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]], dtype = np.float32)
        
        metrics = compute_multilabel_ap(targets, scores, ["sp_a", "sp_b"])
        self.assertAlmostEqual(metrics["macro_average_precision"], 1.0, places = 6)
        self.assertEqual(metrics["n_classes_scored"], 2)
        
        embedding = pool_segment_embeddings(SimpleNamespace(embeddings = np.asarray([[1.0, 0.0], [3.0, 2.0]], dtype = np.float32)))
        self.assertTrue(np.allclose(embedding, np.asarray([2.0, 1.0], dtype = np.float32)))
        one_vector = pool_segment_embeddings(np.asarray([7.0, 8.0], dtype = np.float32))
        np.testing.assert_allclose(one_vector, np.asarray([7.0, 8.0], dtype = np.float32))

    def test_birdnet_embedding_pool_matches_real_library_mask_polarity(self):
        embedding_values = np.asarray([[[1.0, 3.0], [3.0, 5.0], [999.0, 999.0]]], dtype = np.float32)
        mask = np.ones_like(embedding_values, dtype = bool)
        mask[:, :2, :] = False
        result = SimpleNamespace(embeddings = embedding_values, embeddings_masked = mask)
        pooled = pool_segment_embeddings(result)
        np.testing.assert_allclose(pooled, np.asarray([2.0, 4.0], dtype = np.float32))

    def test_birdnet_pool_rejects_all_valid_segments_marked_as_masked(self):
        result = SimpleNamespace(
            embeddings = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype = np.float32),
            embeddings_masked = np.ones((1, 2, 2), dtype = bool),
        )
        
        with self.assertRaisesRegex(ValueError, "no valid embedding segments"):
            pool_segment_embeddings(result)

    def test_birdnet_embedding_pool_rejects_invalid_masks(self):
        embeddings = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype = np.float32)
        
        with self.assertRaisesRegex(ValueError, "does not match embeddings"):
            pool_segment_embeddings(SimpleNamespace(
                embeddings = embeddings, 
                embeddings_masked = np.asarray([[True, False]], 
                dtype = bool),
            ))
            
        with self.assertRaisesRegex(ValueError, "no valid embedding segments"):
            pool_segment_embeddings(SimpleNamespace(
                embeddings = embeddings, 
                embeddings_masked = np.ones_like(embeddings, dtype = bool),
            ))
            
        partial_mask = np.asarray([[[True, False], [True, True]]], dtype = bool)
        
        with self.assertRaisesRegex(ValueError, "partially masks"):
            pool_segment_embeddings(SimpleNamespace(embeddings = embeddings, embeddings_masked = partial_mask))

    def test_rng_state_roundtrip_is_safe_loadable_and_restores_cpu_generators(self):
        original = capture_rng_state()
        try:
            set_seed(618)
            random.random()
            np.random.random()
            torch.rand(3)
            saved = capture_rng_state()
            expected = (random.random(), np.random.random(), torch.rand(5))
            
            with tempfile.TemporaryDirectory() as tmp:
                checkpoint = Path(tmp) / "rng.pt"
                torch.save({"rng_state": saved}, checkpoint)
                safe_loaded = torch.load(checkpoint, map_location = "cpu", weights_only = True)
                
            restore_rng_state(safe_loaded["rng_state"])
            actual = (random.random(), np.random.random(), torch.rand(5))
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])
            
            torch.testing.assert_close(actual[2], expected[2], rtol = 0, atol = 0)
            
        finally:
            restore_rng_state(original)

    def test_noisy_center_crop_evaluation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "birdclef"
            species = root / "species_a"
            species.mkdir(parents=True)
            audio = torch.randn(1, 4800)
            torchaudio.save(str(species / "bird.wav"), audio, 16000)
            noise_pool = [torch.randn(1, 6400)]
            
            dataset = BirdCLEFDataset(
                str(root), 
                noise_pool = noise_pool, 
                noise_mixer = NoiseMixer(), 
                snr_db = 0.0,
                sample_rate = 16000, 
                duration_sec = 0.1, 
                crop_mode = "center", 
                include_path = True,
            )
            
            first = dataset[0]
            second = dataset[0]
            self.assertEqual(first[2], second[2])
            self.assertTrue(torch.equal(first[0], second[0]))

    def test_birdclef_short_clips_repeat_pad_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "birdclef"
            species = root / "species_a"
            species.mkdir(parents = True)
            short = torch.tensor([[0.1, -0.2, 0.3, -0.4]], dtype = torch.float32)
            torchaudio.save(str(species / "short.wav"), short, 16000)
            
            ds = BirdCLEFDataset(str(root), sample_rate = 16000, duration_sec = 0.001, crop_mode = "center")
            
            waveform = load_and_preprocess_audio(
                ds.file_list[0], 
                target_sr = 16000, 
                duration_sec = 0.001,
                crop_mode = "center", 
                short_audio_policy = ds.short_audio_policy,
            )
            
            self.assertEqual(ds.short_audio_policy, "repeat")
            self.assertGreater(float(waveform.abs().sum()), 0.0)
            self.assertAlmostEqual(float(waveform[0, 0]), float(waveform[0, 4]), places = 6)

    def test_center_crop_is_deterministic_and_short_clips_pad(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.wav"
            source = torch.arange(2000, dtype = torch.float32).reshape(1, -1) / 2000
            torchaudio.save(str(path), source, 16000)
            
            first = load_and_preprocess_audio(path, 16000, 0.05, crop_mode = "center")
            second = load_and_preprocess_audio(path, 16000, 0.05, crop_mode = "center")
            
            self.assertTrue(torch.equal(first, second))
            self.assertEqual(first.shape, (1, 800))

            short = torch.ones(1, 100)
            short_path = Path(tmp) / "short.wav"
            torchaudio.save(str(short_path), short, 16000)
            padded = load_and_preprocess_audio(short_path, 16000, 0.02, crop_mode = "center", short_audio_policy = "zero_pad")
            
            self.assertEqual(padded.shape, (1, 320))
            self.assertTrue(torch.allclose(padded[:, :100], short, atol = 1e-4))
            self.assertTrue(torch.equal(padded[:, 100:], torch.zeros(1, 220)))

    def test_xeno_canto_dataset_rejects_stale_overlap_audit_file_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xeno = root / "xeno"
            xeno.mkdir()
            torchaudio.save(str(xeno / "one.wav"), torch.randn(1, 16000), 16000)
            
            manifest = root / "overlap.json"
            manifest.write_text(json.dumps({
                "excluded_relative_paths": [],
                "xeno_canto_paths_sha256": fingerprint_relative_paths(["one.wav"]),
            }), encoding = "utf-8")
            
            dataset = XenoCantoDataset(
                xeno, 
                variant = "standard", 
                exclude_manifest_path = manifest, 
                require_overlap_audit = True,
                sample_rate = 16000, 
                duration_sec = 1,
            )
            
            self.assertEqual(len(dataset), 1)
            torchaudio.save(str(xeno / "two.wav"), torch.randn(1, 16000), 16000)
            
            with self.assertRaisesRegex(RuntimeError, "file index differs from overlap audit"):
                XenoCantoDataset(
                    xeno, 
                    variant = "standard", 
                    exclude_manifest_path = manifest, 
                    require_overlap_audit = True,
                    sample_rate = 16000, 
                    duration_sec = 1,
                )

    def test_birdclef_root_resolves_train_audio_and_stable_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "birdclef"
            (root / "train_audio" / "species_b").mkdir(parents = True)
            (root / "train_audio" / "species_a").mkdir(parents = True)
            (root / "train_soundscapes").mkdir(parents = True)
            
            torchaudio.save(str(root / "train_audio" / "species_b" / "b.wav"), torch.randn(1, 16000), 16000)
            torchaudio.save(str(root / "train_audio" / "species_a" / "a.wav"), torch.randn(1, 16000), 16000)
            torchaudio.save(str(root / "train_soundscapes" / "not_species.wav"), torch.randn(1, 16000), 16000)
            
            dataset = BirdCLEFDataset(str(root), duration_sec = 1, crop_mode = "center", include_path = True)
            
            self.assertEqual(dataset.label_map, {"species_a": 0, "species_b": 1})
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0][2], "train_audio/species_a/a.wav")


    def test_birdclef_metadata_relative_filename_matches_grouping_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "birdclef"
            audio_dir = root / "train_audio" / "species_a"
            audio_dir.mkdir(parents = True)
            (root / "train_audio" / "species_b").mkdir(parents = True)
            torchaudio.save(str(audio_dir / "a.wav"), torch.randn(1, 16000), 16000)
            torchaudio.save(str(root / "train_audio" / "species_b" / "b.wav"), torch.randn(1, 16000), 16000)
            
            import pandas as pd
            pd.DataFrame([
                {"filename": "train_audio/species_a/a.wav", "primary_label": "species_a", "author": "recordist_1"},
                {"filename": "train_audio/species_b/b.wav", "primary_label": "species_b", "author": "recordist_2"},
            ]).to_csv(root / "train_metadata.csv", index = False)
            
            dataset = BirdCLEFDataset(str(root), duration_sec = 1, crop_mode = "center")
            self.assertEqual(dataset.grouping_source, "author")
            self.assertEqual(dataset.group_ids, ["author:recordist_1", "author:recordist_2"])

    def test_split_fractions_are_nested_and_disjoint(self):
        labels = [label for label in range(3) for _ in range(15)]
        groups = [f"file:{i}" for i in range(len(labels))]
        train, val, test, fractions, method = split_indices(labels, groups, 42, "file_level_no_metadata")
        
        self.assertEqual(method, "stratified_file_level_no_group_metadata")
        self.assertFalse(set(train) & set(val))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(val) & set(test))
        
        fraction_sets = [set(fractions[key]) for key in ("0.01", "0.05", "0.10", "1.00")]
        self.assertTrue(fraction_sets[0].issubset(fraction_sets[1]))
        self.assertTrue(fraction_sets[1].issubset(fraction_sets[2]))
        self.assertTrue(fraction_sets[2].issubset(fraction_sets[3]))
        self.assertEqual(fraction_sets[-1], set(train))

    def test_habitat_split_refuses_seed_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "esc50.csv"
            categories = ["rain", "sea_waves", "water_drops", "wind", "engine", "car_horn"]
            
            import pandas as pd
            pd.DataFrame([{"filename": f"{c}.wav", "category": c} for c in categories]).to_csv(csv_path, index = False)
            output = root / "processed"
            create_habitat_split(csv_path, output, seed = 42)
            
            with self.assertRaisesRegex(RuntimeError, "uses seed=42"):
                create_habitat_split(csv_path, output, seed = 123)

    def test_grouped_split_refuses_insufficient_groups_instead_of_falling_back(self):
        labels = [0] * 10 + [1] * 10
        groups = [f"author:{i % 4}" for i in range(len(labels))]
        
        with self.assertRaisesRegex(ValueError, "4 groups"):
            split_indices(labels, groups, 42, "author", rare_class_policy = "error")

    def test_grouped_split_keeps_groups_disjoint(self):
        labels = [label for label in range(3) for _ in range(15)]
        groups = [f"author:{i}" for i in range(len(labels))]
        train, val, test, _, method = split_indices(labels, groups, 42, "author")
        
        self.assertEqual(method, "stratified_group_author")
        self.assertFalse({groups[i] for i in train} & {groups[i] for i in val})
        self.assertFalse({groups[i] for i in train} & {groups[i] for i in test})
        self.assertFalse({groups[i] for i in val} & {groups[i] for i in test})

    def test_habitat_split_detects_stale_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "esc50.csv"
            categories = ["rain", "sea_waves", "water_drops", "wind", "engine", "car_horn"]
            
            import pandas as pd
            pd.DataFrame([{"filename": f"{c}.wav", "category": c} for c in categories]).to_csv(csv_path, index = False)
            output = root / "processed"
            
            create_habitat_split(csv_path, output, seed = 42)
            pd.DataFrame([{"filename": f"{c}.wav", "category": c} for c in categories] +
                         [{"filename": "rain_extra.wav", "category": "rain"}]).to_csv(csv_path, index = False)
                         
            with self.assertRaisesRegex(RuntimeError, "different ESC-50 metadata file"):
                create_habitat_split(csv_path, output, seed=42)

    def test_habitat_csv_is_verified_against_saved_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            esc50_root = root / "esc50"
            (esc50_root / "meta").mkdir(parents=True)
            categories = ["rain", "sea_waves", "water_drops", "wind", "engine", "car_horn"]
            
            import pandas as pd
            metadata_path = esc50_root / "meta" / "esc50.csv"
            pd.DataFrame([{"filename": f"{c}.wav", "category": c} for c in categories]).to_csv(metadata_path, index = False)
            output = root / "processed"
            
            create_habitat_split(metadata_path, output, seed = 42)
            seen_path = output / "esc50_seen_habitat.csv"
            unseen_path = output / "esc50_unseen_habitat.csv"
            frame = read_verified_habitat_csv(seen_path, esc50_root, expected_partition = "seen")
            
            self.assertGreater(len(frame), 0)
            with self.assertRaisesRegex(RuntimeError, "does not match the saved seen filename manifest"):
                read_verified_habitat_csv(unseen_path, esc50_root, expected_partition = "seen")
                
            unseen_frame = read_verified_habitat_csv(
                unseen_path, 
                esc50_root, 
                expected_partition = "unseen"
            )
            
            self.assertGreater(len(unseen_frame), 0)
            self.assertFalse(set(frame["category"]) & set(unseen_frame["category"]))
            
            frame.iloc[[0]].to_csv(seen_path, index = False)
            with self.assertRaisesRegex(RuntimeError, "does not match the saved seen filename manifest"):
                read_verified_habitat_csv(seen_path, esc50_root, expected_partition = "seen")

    def test_linear_probe_evaluation_raises_on_empty_loader(self):
        with self.assertRaisesRegex(RuntimeError, "zero examples"):
            evaluate_accuracy(torch.nn.Identity(), torch.nn.Identity(), [], device = "cpu")

    def test_manifest_validator_rejects_split_overlap(self):
        paths = ["a.wav", "b.wav", "c.wav", "d.wav"]
        labels = [0, 0, 1, 1]
        
        manifest = {
            "train_filenames": ["a.wav", "b.wav"],
            "validation_filenames": ["b.wav"],
            "test_filenames": ["c.wav", "d.wav"],
            "train_fraction_filenames": {"1.00": ["a.wav", "b.wav"]},
            "dataset_fingerprint": fingerprint_paths_labels(paths, labels),
        }
        
        with self.assertRaisesRegex(ValueError, "filenames overlap"):
            validate_manifest_file_index(manifest, paths, labels, context = "test manifest")

    def test_manifest_validator_accepts_valid_filename_manifest(self):
        paths = ["a.wav", "b.wav", "c.wav", "d.wav"]
        labels = [0, 0, 1, 1]
        
        manifest = {
            "train_filenames": ["a.wav", "b.wav"],
            "validation_filenames": ["c.wav"],
            "test_filenames": ["d.wav"],
            "train_fraction_filenames": {"1.00": ["a.wav", "b.wav"]},
            "dataset_fingerprint": fingerprint_paths_labels(paths, labels),
        }
        
        validate_manifest_file_index(manifest, paths, labels, context = "test manifest")

    def test_infonce_loss_is_finite_positive_and_validates_inputs(self):
        q = F.normalize(torch.randn(16, 128), dim = 1)
        k = F.normalize(torch.randn(16, 128), dim = 1)
        queue = F.normalize(torch.randn(128, 1024), dim = 0)
        loss = info_nce_loss(q, k, queue, tau = 0.1)
        
        self.assertGreater(float(loss.item()), 0.0)
        self.assertTrue(torch.isfinite(loss))
        
        with self.assertRaises(ValueError):
            info_nce_loss(q, k, queue, tau = 0)

    def test_queue_cycling_and_overwrite(self):
        bq, bk = CNN_10_Backbone(), CNN_10_Backbone()
        hq, hk = Projection_Head(512, 512, 128), Projection_Head(512, 512, 128)
        moco = MoCo_Wrapper(bq, bk, hq, hk, dim = 128, K = 16, m = 0.999)
        initial_ptr = int(moco.queue_ptr.item())
        dummy_k = F.normalize(torch.randn(4, 128), dim = 1)
        moco._dequeue_and_enqueue(dummy_k)
        
        self.assertEqual((initial_ptr + 4) % 16, int(moco.queue_ptr.item()))
        self.assertTrue(torch.allclose(moco.queue[:, :4], dummy_k.T))

    def test_queue_rejects_incompatible_batch(self):
        model = MoCo_Wrapper(
            CNN_10_Backbone(), 
            CNN_10_Backbone(), 
            Projection_Head(), 
            Projection_Head(), 
            dim = 128, K = 10
        )
        
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            model._dequeue_and_enqueue(F.normalize(torch.randn(4, 128), dim = 1))

    def test_no_grad_key_encoder(self):
        moco = MoCo_Wrapper(
            CNN_10_Backbone(), 
            CNN_10_Backbone(), 
            Projection_Head(), 
            Projection_Head(), 
            dim = 128, K = 16
        )

        x_q = torch.randn(2, 1, 64, 313)
        x_k = torch.randn(2, 1, 64, 313)
        q, k, queue = moco(x_q, x_k)
        loss = info_nce_loss(q, k, queue, tau = 0.1)
        loss.backward()
        
        for param in moco.encoder_k.parameters():
            self.assertIsNone(param.grad)

    def test_key_encoder_batchnorm_uses_synced_query_buffers(self):
        moco = MoCo_Wrapper(
            CNN_10_Backbone(), 
            CNN_10_Backbone(), 
            Projection_Head(), 
            Projection_Head(), 
            dim = 128, K = 16
        )
        moco.train()
        
        query_bn = moco.encoder_q[0].block1.bn1
        key_bn = moco.encoder_k[0].block1.bn1
        x_q = torch.randn(2, 1, 64, 64) * 2 + 4
        x_k = torch.randn(2, 1, 64, 64) - 3
        _, _, _ = moco(x_q, x_k)
        
        self.assertFalse(moco.encoder_k.training)
        self.assertTrue(torch.equal(key_bn.running_mean, query_bn.running_mean))
        self.assertTrue(torch.equal(key_bn.running_var, query_bn.running_var))

    def test_momentum_update(self):
        moco = MoCo_Wrapper(
            CNN_10_Backbone(), 
            CNN_10_Backbone(), 
            Projection_Head(), 
            Projection_Head(), 
            dim = 128, K = 16, m = 0.0
        )
        
        for param in moco.encoder_q.parameters():
            param.data.add_(1.0)
            
        moco._momentum_update_key_encoder()
        for q, k in zip(moco.encoder_q.parameters(), moco.encoder_k.parameters()):
            self.assertTrue(torch.allclose(q.data, k.data))


    def test_infonce_matches_reference_cross_entropy(self):
        torch.manual_seed(17)
        q = torch.randn(4, 8)
        k = torch.randn(4, 8)
        queue = torch.randn(8, 23)
        tau = 0.17
        qn = F.normalize(q, dim = 1)
        kn = F.normalize(k.detach(), dim = 1)
        qnq = F.normalize(queue.detach(), dim = 0)
        positive = (qn * kn).sum(dim = 1, keepdim = True)
        negative = qn @ qnq
        logits = torch.cat([positive, negative], dim = 1) / tau
        expected = F.cross_entropy(logits, torch.zeros(4, dtype = torch.long))
        actual = info_nce_loss(q, k, queue, tau=tau)
        self.assertTrue(torch.allclose(actual, expected, atol = 1e-7, rtol = 1e-6))

    def test_infonce_can_overfit_one_fixed_batch(self):
        torch.manual_seed(29)
        encoder = torch.nn.Linear(8, 8)
        x = torch.randn(8, 8)
        keys = F.normalize(torch.randn(8, 8), dim = 1)
        queue = F.normalize(torch.randn(8, 128), dim = 0)
        optimizer = torch.optim.Adam(encoder.parameters(), lr = 0.04)
        
        with torch.no_grad():
            initial = float(info_nce_loss(encoder(x), keys, queue, tau = 0.1))
            
        for _ in range(40):
            optimizer.zero_grad(set_to_none = True)
            loss = info_nce_loss(encoder(x), keys, queue, tau = 0.1)
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            final = float(info_nce_loss(encoder(x), keys, queue, tau = 0.1))
            
        self.assertLess(final, initial * 0.7)

    def test_probe_evaluation_does_not_mutate_batchnorm_buffers(self):
        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(4)
                
            def forward(self, x):
                return self.bn(x)
                
        backbone = TinyBackbone()
        head = torch.nn.Linear(4, 2)
        data = torch.utils.data.TensorDataset(torch.randn(12, 4), torch.randint(0, 2, (12,)))
        loader = torch.utils.data.DataLoader(data, batch_size = 3)
        before_mean = backbone.bn.running_mean.clone()
        before_var = backbone.bn.running_var.clone()
        before_batches = backbone.bn.num_batches_tracked.clone()
        evaluate_accuracy(head, backbone, loader, device = "cpu")
        
        self.assertTrue(torch.equal(backbone.bn.running_mean, before_mean))
        self.assertTrue(torch.equal(backbone.bn.running_var, before_var))
        self.assertTrue(torch.equal(backbone.bn.num_batches_tracked, before_batches))

    def test_key_batchnorm_buffers_sync_after_query_views(self):
        moco = MoCo_Wrapper(
            CNN_10_Backbone(), 
            CNN_10_Backbone(), 
            Projection_Head(), 
            Projection_Head(), 
            dim = 128, K = 32
        )
        
        moco.train()
        q_bn = moco.encoder_q[0].block1.bn1
        k_bn = moco.encoder_k[0].block1.bn1
        
        q1 = torch.randn(4, 1, 64, 64) * 3.0 + 5.0
        q2 = torch.randn(4, 1, 64, 64) * 1.5 - 2.0
        _, _, _, _, _ = moco(q1, q2, symmetric = True)
        
        self.assertTrue(torch.equal(k_bn.running_mean, q_bn.running_mean))
        self.assertTrue(torch.equal(k_bn.running_var, q_bn.running_var))
        self.assertFalse(moco.encoder_k.training)

    def test_rare_class_policy_defaults_to_exclude(self):
        labels = [0] * 3 + [1] * 15
        groups = [f"author:{i}" for i in range(len(labels))]
        
        train, val, test, _fractions, _method = split_indices(
            labels, groups, 42, "author", min_recordings = 5, min_groups = 5,
        )
        
        active = set(train) | set(val) | set(test)
        self.assertTrue(active)
        self.assertTrue(all(labels[index] == 1 for index in active))

    def test_rare_class_exclusion_is_explicit_and_keeps_other_classes(self):
        labels = [0] * 3 + [1] * 15 + [2] * 15
        groups = [f"author:{i}" for i in range(len(labels))]
        
        train, val, test, fractions, method = split_indices(
            labels, groups, 42, "author", min_recordings = 5, min_groups = 5,
            rare_class_policy = "exclude", fraction_seed = 123,
        )
        
        active_indices = set(train) | set(val) | set(test)
        self.assertTrue(all(labels[i] in {1, 2} for i in active_indices))
        self.assertFalse(any(labels[i] == 0 for i in train + val + test))
        self.assertEqual({labels[i] for i in test}, {1, 2})
        
        fraction_sizes = [len(fractions[k]) for k in ("0.01", "0.05", "0.10", "1.00")]
        self.assertEqual(fraction_sizes, sorted(fraction_sizes))

    def test_split_eligibility_excludes_silent_indices_from_every_partition(self):
        labels = [label for label in range(3) for _ in range(15)]
        groups = [f"author:{i}" for i in range(len(labels))]
        eligible = [i for i in range(len(labels)) if i not in {0, 16, 32}]
        
        train, val, test, fractions, _method = split_indices(
            labels, groups, 42, "author", min_recordings = 5, min_groups = 5,
            rare_class_policy = "exclude", eligible_indices = eligible,
        )
        
        all_partition_indices = set(train) | set(val) | set(test)
        self.assertTrue(all(index in set(eligible) for index in all_partition_indices))
        self.assertTrue(all_partition_indices.isdisjoint({0, 16, 32}))
        self.assertTrue(set(fractions["1.00"]).issubset(set(train)))

    def test_fraction_seed_changes_low_fraction_draw_not_test_split(self):
        labels = [label for label in range(3) for _ in range(20)]
        groups = [f"author:{i}" for i in range(len(labels))]
        
        first = split_indices(labels, groups, 42, "author", fraction_seed = 1)
        second = split_indices(labels, groups, 42, "author", fraction_seed = 2)
        self.assertEqual(first[0:3], second[0:3])
        self.assertNotEqual(first[3]["0.10"], second[3]["0.10"])

    def test_overlap_audit_matches_xc_id_across_reencoded_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xeno = root / "xeno"
            bird = root / "birdclef" / "train_audio" / "species_a"
            (xeno / "species_a").mkdir(parents = True)
            bird.mkdir(parents = True)
            (xeno / "species_a" / "XC123456.mp3").write_bytes(b"different-source-encoding")
            (bird / "XC123456.ogg").write_bytes(b"another-encoding")
            out = root / "excluded.json"
            
            audit_overlap_main([
                "--xeno-dir", str(xeno), "--birdclef-dir", str(root / "birdclef"), "--output", str(out)
            ])
            
            result = json.loads(out.read_text())
            self.assertEqual(result["catalogue_id_duplicate_count"], 1)
            self.assertIn("species_a/XC123456.mp3", result["excluded_relative_paths"])
            self.assertEqual(result["matches"][0]["match_method"], "xeno_canto_catalogue_id")

    def test_paired_analysis_supports_author_grouped_bootstrap(self):
        filenames = ["rec_1.wav", "rec_2.wav", "rec_3.wav", "rec_4.wav"]
        group_ids = ["author_a", "author_a", "author_b", "author_c"]
        
        payload = {"metadata": {
            "grouping_source": "author_metadata",
            "n_preflight_excluded_files": 0, 
            "preflight_excluded_filenames": [],
            "n_preflight_read_error_files_excluded": 0, 
            "preflight_read_error_filenames": [],
            "preflight_reason_histogram": {}, 
            "preflight_read_error_fraction": 0.0,
            "preflight_excluded_fraction": 0.0, 
            "preflight_max_read_error_fraction": 0.01,
            "preflight_max_excluded_fraction": 0.05,
        }}
        
        for pool in ("seen_habitat", "unseen_habitat"):
            payload[pool] = {}
            
            for model_index, model_name in enumerate(MODELS.values()):
                correct_values = [
                    bool((index + model_index + (pool == "unseen_habitat")) % 2)
                    for index in range(len(filenames))
                ]
                
                rows = []
                for index, filename in enumerate(filenames):
                    y_true = index % 2
                    correct = correct_values[index]
                    rows.append({
                        "file": filename,
                        "y_true": y_true,
                        "y_pred": y_true if correct else 1 - y_true,
                        "correct": correct,
                        "group_id": group_ids[index],
                    })
                    
                payload[pool][model_name] = {"Clean": {
                    "predictions": rows, "n_silent_mix_skipped": 0, "silent_mix_skipped_files": []
                }}

        result = analyze_seed(payload, seed = 42, snrs = ["Clean"], n_boot = 100)
        self.assertEqual(len(result), 12)
        
        paired_rows = [row for row in result if row["analysis"] == "paired_accuracy_delta"]
        self.assertEqual(len(paired_rows), 8)
        self.assertTrue(all(row["bootstrap_unit"] == "group" for row in paired_rows))

    def test_seeded_preflight_processing_order_is_reproducible_and_shuffled(self):
        inventory = [(Path(f"species_{i}.wav"), f"species_{i}.wav") for i in range(20)]
        first = seeded_processing_order(inventory, 42)
        second = seeded_processing_order(inventory, 42)
        different_seed = seeded_processing_order(inventory, 7)
        
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(inventory))
        self.assertNotEqual(first, inventory)
        self.assertNotEqual(first, different_seed)

    def test_wilson_lower_bound_is_conservative_near_the_exclusion_limit(self):
        self.assertLess(wilson_lower_bound(6, 200), 0.05)
        self.assertLess(wilson_lower_bound(40, 1000), 0.05)
        self.assertGreater(wilson_lower_bound(80, 1000), 0.05)
        self.assertEqual(wilson_lower_bound(0, 0), 0.0)

    def test_slurm_terminal_states_are_inactive_and_running_is_active(self):
        def run_for_state(command, **kwargs):
            state_by_job = {"101": "RUNNING", "102": "COMPLETED", "103": "TIMEOUT"}
            
            return SimpleNamespace(returncode = 0, stdout = state_by_job[command[3]] + "\n")

        with patch("scripts.preflight_xeno_audio.shutil.which", side_effect = lambda name: "/fake/" + name if name == "squeue" else None), \
             patch("scripts.preflight_xeno_audio.subprocess.run", side_effect = run_for_state):
            self.assertIs(_slurm_job_is_active("101"), True)
            self.assertIs(_slurm_job_is_active("102"), False)
            self.assertIs(_slurm_job_is_active("103"), False)

    def test_slurm_accounting_fallback_recognizes_terminal_job_after_squeue_error(self):
        def run_commands(command, **kwargs):
            if command[0].endswith("squeue"):
                return SimpleNamespace(returncode = 1, stdout = "", stderr = "job not found")
                
            return SimpleNamespace(returncode = 0, stdout = "TIMEOUT\n", stderr = "")

        with patch("scripts.preflight_xeno_audio.shutil.which", side_effect = lambda name: "/fake/" + name if name in {"squeue", "sacct"} else None), \
             patch("scripts.preflight_xeno_audio.subprocess.run", side_effect = run_commands):
            self.assertIs(_slurm_job_is_active("404"), False)

    def test_stale_foreign_staging_uses_terminal_slurm_state_before_age_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            terminal = root / ".cache.staging-terminal"
            active = root / ".cache.staging-active-job"
            now = time.time()
            
            for path, job_id in ((terminal, "501"), (active, "502")):
                path.mkdir()
                
                (path / ".preflight-owner.json").write_text(json.dumps({
                    "hostname": "another-host", 
                    "pid": 999999,
                    "slurm_job_id": job_id, 
                    "started_unix": now - 10 * 3600,
                }), encoding = "utf-8")

            def run_for_state(command, **kwargs):
                return SimpleNamespace(returncode = 0, stdout = {"501": "TIMEOUT", "502": "RUNNING"}[command[3]] + "\n")

            with patch("scripts.preflight_xeno_audio.shutil.which", side_effect = lambda name: "/fake/squeue" if name == "squeue" else None), \
                 patch("scripts.preflight_xeno_audio.subprocess.run", side_effect = run_for_state):
                removed = cleanup_stale_staging_dirs(cache_root, stale_after_hours = 72, now=now)
                
            self.assertIn(terminal, removed)
            self.assertFalse(terminal.exists())
            self.assertTrue(active.exists())

    def test_preflight_stale_staging_cleanup_removes_orphans_but_preserves_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            stale_legacy = root / ".cache.staging-legacy"
            stale_legacy.mkdir()
            (stale_legacy / "partial.bin").write_bytes(b"partial")
            old = time.time() - 100 * 3600
            os.utime(stale_legacy, (old, old))

            stale_owned = root / ".cache.staging-dead-owner"
            stale_owned.mkdir()
            (stale_owned / ".preflight-owner.json").write_text(json.dumps({
                "hostname": socket.gethostname(), "pid": 2**31 - 1,
                "slurm_job_id": None, "started_unix": time.time(),
            }),  encoding = "utf-8")

            active = root / ".cache.staging-active"
            active.mkdir()
            (active / ".preflight-owner.json").write_text(json.dumps({
                "hostname": socket.gethostname(), "pid": os.getpid(),
                "slurm_job_id": None, "started_unix": time.time(),
            }), encoding = "utf-8")

            removed = cleanup_stale_staging_dirs(cache_root, stale_after_hours = 72)
            self.assertIn(stale_legacy, removed)
            self.assertIn(stale_owned, removed)
            self.assertFalse(stale_legacy.exists())
            self.assertFalse(stale_owned.exists())
            self.assertTrue(active.is_dir())

    def test_preflight_staging_symlinks_are_not_followed_or_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            target = root / "target"
            target.mkdir()
            (target / "keep.bin").write_bytes(b"keep")
            link = root / ".cache.staging-link"
            
            try:
                link.symlink_to(target, target_is_directory = True)
            except (OSError, NotImplementedError):
                self.skipTest("directory symlinks are unavailable")
                
            cleanup_stale_staging_dirs(cache_root, stale_after_hours = 0.001, now = time.time() + 100000)
            self.assertTrue(link.is_symlink())
            self.assertTrue((target / "keep.bin").is_file())

    def test_xeno_preflight_caches_nonsilent_crops_and_excludes_silent_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "xeno"
            (audio / "species_a").mkdir(parents=True)
            torchaudio.save(str(audio / "species_a" / "good.wav"), torch.randn(1, 32000) * 0.1, 16000)
            torchaudio.save(str(audio / "species_a" / "silent.wav"), torch.zeros(1, 32000), 16000)
            cache = root / "cache"
            cache.mkdir()
            
            stale_marker = cache / "stale-cache-marker.bin"
            stale_marker.write_bytes(b"replace only after a successful cache build")
            manifest = root / "preflight.json"
            
            preflight_xeno_main([
                "--audio-dir", str(audio), "--cache-dir", str(cache), "--manifest", str(manifest),
                "--sample-rate", "16000", "--clip-sec", "1", "--num-crops", "2", "--min-rms", "0.00001",
                # This fixture intentionally excludes one of two files (50%).
                "--max-excluded-fraction", "0.51",
            ])
            
            data = json.loads(manifest.read_text())
            self.assertEqual(data["processing_order_seed"], 42)
            self.assertEqual(data["processing_order"], "seeded shuffle of sorted relative path inventory")
            self.assertEqual(data["early_abort_min_files"], 1000)
            self.assertEqual(data["early_abort_check_interval"], 100)
            self.assertIn("95% Wilson", data["early_abort_method"])
            self.assertFalse((cache / ".preflight-owner.json").exists())
            self.assertFalse(stale_marker.exists())
            self.assertIn("species_a/good.wav", data["records"])
            self.assertIn("species_a/silent.wav", data["invalid"])
            
            ds = XenoCantoDataset(
                audio, 
                variant = "standard", 
                sample_rate = 16000, 
                duration_sec = 1,
                preflight_manifest_path = manifest, 
                cache_dir = cache, 
                require_preflight = True,
            )
            
            self.assertEqual(len(ds), 1)
            first, second = ds[0]
            self.assertEqual(first.shape, second.shape)
            self.assertTrue(torch.isfinite(first).all() and torch.isfinite(second).all())


    def test_xeno_preflight_aborts_early_and_preserves_old_cache_on_overwrite_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "xeno"
            (audio / "species_a").mkdir(parents=True)
            
            torchaudio.save(
                str(audio / "species_a" / "good.wav"),
                torch.randn(1, 32000) * 0.1,
                16000,
            )
            
            torchaudio.save(
                str(audio / "species_a" / "silent.wav"),
                torch.zeros(1, 32000),
                16000,
            )
            
            torchaudio.save(
                str(audio / "species_a" / "silent_extra.wav"),
                torch.zeros(1, 32000),
                16000,
            )
            
            cache = root / "cache"
            cache.mkdir()
            preserved_file = cache / "old-cache-marker.bin"
            preserved_file.write_bytes(b"keep until replacement passes")
            manifest = root / "preflight.json"
            manifest.write_text('{"old": true}', encoding = "utf-8")

            with self.assertRaisesRegex(RuntimeError, "Early preflight abort"):
                preflight_xeno_main([
                    "--audio-dir", str(audio), 
                    "--cache-dir", str(cache), 
                    "--manifest", str(manifest),
                    "--sample-rate", "16000", 
                    "--clip-sec", "1", 
                    "--num-crops", "2",
                    "--min-rms", "0.00001", 
                    "--max-excluded-fraction", "0.05",
                    "--early-abort-min-files", "2", 
                    "--early-abort-check-interval", "1", 
                    "--overwrite",
                ])

            self.assertTrue(preserved_file.is_file())
            self.assertFalse(manifest.exists())
            self.assertTrue((root / "xeno_canto_preflight_excluded.csv").is_file())
            self.assertFalse(list(root.glob(".cache.staging-*")))


    def test_birdclef_noise_augmentation_skips_wholly_silent_item_without_failing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "birdclef"
            species = root / "species_a"
            species.mkdir(parents=True)
            silent_path = species / "silent.wav"
            torchaudio.save(str(silent_path), torch.zeros(1, 1600), 16000)
            noise = torch.randn(1, 1600) * 0.1
            
            dataset = BirdCLEFDataset(
                str(root), 
                sample_rate = 16000, 
                duration_sec = 0.1, 
                crop_mode = "center",
                noise_pool = [noise], 
                noise_mixer = NoiseMixer(sample_rate = 16000),
                noise_snr_db_range = (-5, 20), 
                noise_probability = 1.0, 
                include_path = True,
            )
            
            mel, label, relative_path = dataset[0]
            self.assertEqual(label, 0)
            self.assertEqual(relative_path, "species_a/silent.wav")
            self.assertEqual(dataset.silent_mix_skipped_count, 1)
            self.assertTrue(torch.isfinite(mel).all())

    def test_birdclef_silent_selected_crop_is_recropped_before_noise_mixing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "birdclef"
            species = root / "species_a"
            species.mkdir(parents = True)
            
            audio = torch.zeros(1, 6400)
            generator = torch.Generator().manual_seed(7)
            audio[:, :2000] = torch.randn(1, 2000, generator=generator) * 0.1
            torchaudio.save(str(species / "recover.wav"), audio, 16000)
            
            dataset = BirdCLEFDataset(
                str(root), 
                sample_rate = 16000, 
                duration_sec = 0.1, 
                crop_mode = "center",
                noise_pool = [torch.randn(1, 1600) * 0.1], 
                noise_mixer = NoiseMixer(sample_rate = 16000),
                noise_snr_db_range = (-5, 20), 
                noise_probability = 1.0, 
                include_path = True,
            )
            
            with self.assertLogs("utils.audio_io", level = "WARNING") as captured:
                mel, label, relative_path = dataset[0]
                
            self.assertEqual(label, 0)
            self.assertEqual(relative_path, "species_a/recover.wav")
            self.assertEqual(dataset.silent_mix_skipped_count, 0)
            self.assertTrue(any("SILENT_CROP_RECOVERED" in line for line in captured.output))
            self.assertTrue(torch.isfinite(mel).all())

    def test_background_segment_sampler_is_deterministic_and_excludes_all_annotated_ids(self):
        annotated = {0, 2, 4, 7}
        first = sample_background_segment_ids(annotated, 10, 3, seed = 123, recording_name = "site_a.wav")
        second = sample_background_segment_ids(annotated, 10, 3, seed = 123, recording_name = "site_a.wav")
        other_seed = sample_background_segment_ids(annotated, 10, 3, seed = 124, recording_name = "site_a.wav")
        
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(set(first).isdisjoint(annotated))
        self.assertTrue(all(0 <= value < 10 for value in first))
        self.assertEqual(len(other_seed), 3)
        self.assertTrue(set(other_seed).isdisjoint(annotated))
        self.assertEqual(sample_background_segment_ids(set(range(5)), 5, 10, seed = 1, recording_name = "x"), [])

    def test_annotation_coverage_helper_returns_mapped_and_unmapped_codes(self):
        coverage, mapped, unmapped = calculate_annotation_coverage(
            {"sp_a", "sp_b", "sp_c"}, {"sp_a", "sp_c", "sp_x"},
        )
        
        self.assertAlmostEqual(coverage, 2 / 3)
        self.assertEqual(mapped, {"sp_a", "sp_c"})
        self.assertEqual(unmapped, ["sp_b"])
        self.assertEqual(calculate_annotation_coverage(set(), {"sp_a"})[0], 0.0)

    def test_repeat_pad_segment_repeats_final_partial_segment_and_rejects_empty(self):
        clip = torch.tensor([1.0, -2.0, 3.0])
        padded = repeat_pad_segment(clip, 8)
        self.assertEqual(padded.tolist(), [1.0, -2.0, 3.0, 1.0, -2.0, 3.0, 1.0, -2.0])
        
        with self.assertRaisesRegex(ValueError, "empty segment"):
            repeat_pad_segment(torch.empty(0), 4)

    def test_cluster_bootstrap_resamples_recordings_and_writes_defined_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.npz"
            patterns = np.asarray([[1, 0], [0, 1], [0, 0], [1, 1]], dtype = np.uint8)
            targets = np.concatenate([patterns for _ in range(4)], axis = 0)
            scores = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.3, 0.4], [0.7, 0.6]], dtype = np.float32)
            scores = np.concatenate([scores for _ in range(4)], axis = 0)
            file_names = np.asarray([f"recording_{i}.wav" for i in range(4) for _ in range(4)])
            
            np.savez_compressed(
                path, 
                targets = targets, 
                scores_habitat = scores, 
                file_names = file_names,
                class_names = np.asarray(["sp_a", "sp_b"])
            )
            
            result = recording_cluster_bootstrap(path, "habitat", replicates = 50, seed = 123)
            self.assertEqual(result["n_recording_clusters"], 4)
            self.assertGreaterEqual(result["bootstrap_replicates_valid"], 50)
            self.assertLessEqual(result["macro_average_precision_ci95_low"], result["macro_average_precision_ci95_high"])

    def test_external_baseline_aggregator_rejects_checkpoint_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            
            for seed in (42, 123):
                record = {
                    "schema_version": 3, 
                    "mode": "external_pretrained_linear_probe", 
                    "variant": "panns",
                    "seed": seed, 
                    "fraction": 0.1, 
                    "dataset_fingerprint": "same-dataset",
                    "active_label_map": {"bird_a": 0}, "num_classes": 1,
                    "n_active_classes": 1, 
                    "n_total_classes": 1, 
                    "n_excluded_classes": 0,
                    "excluded_classes": [], 
                    "n_silent_files_excluded": 0, 
                    "silent_excluded_filenames": [],
                    "n_preflight_excluded_files": 0, 
                    "preflight_excluded_filenames": [],
                    "n_preflight_read_error_files_excluded": 0, 
                    "preflight_read_error_filenames": [],
                    "preflight_reason_histogram": {}, 
                    "preflight_read_error_fraction": 0.0,
                    "preflight_excluded_fraction": 0.0, 
                    "preflight_max_read_error_fraction": 0.01,
                    "preflight_max_excluded_fraction": 0.05,
                    "rare_class_policy": "exclude", 
                    "class_scope_note": "fixture scope",
                    "num_fixed_crops_requested": 3, 
                    "fixed_crop_aggregation": "mean of per-crop embeddings to one recording-level row",
                    "crop_policy": "three-temporal-crops-rms-recovery-mean-pooling-v2", 
                    "minimum_rms": 1e-5,
                    "feature_extraction_spec_version": "v1.6.3-external-feature-pooling-unified-center-recovery-v4",
                    "fixed_crop_count_min": 3, 
                    "fixed_crop_count_mean": 3.0, 
                    "fixed_crop_count_max": 3,
                    "fixed_crop_shortfall_filenames": [],
                    "model_name": "PANNs fake", 
                    "external_checkpoint": "fake.pth",
                    "external_checkpoint_sha256": "b" * 64,
                    "encoder_identity": {"model_file_sha256": "a" * 64, 
                    "package_version": "0.1.1", 
                    "model_version": "test"},
                    "test_accuracy": 0.6, 
                    "test_macro_f1": 0.5, 
                    "test_balanced_accuracy": 0.5,
                }
                
                (root / f"pretrained_probe_panns_s{seed}_frac0.1.json").write_text(json.dumps(record), encoding = "utf-8")
                
            with self.assertRaisesRegex(ValueError, "disagrees with encoder identity"):
                load_external_records(
                    root, 
                    seeds = (42, 123), 
                    fractions = (0.1,), 
                    models = ("panns",), 
                    allow_three_crop_analysis = True
                )

    def test_panns_encoder_requires_explicit_local_checkpoint_before_import(self):
        with self.assertRaisesRegex(ValueError, "pre-downloaded local Cnn14"):
            PannsCnn14Encoder("cpu", None)

if __name__ == "__main__":
    unittest.main(verbosity = 2)
