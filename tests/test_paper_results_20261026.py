import numpy as np
import unittest
import csv
import tempfile
from pathlib import Path

from scripts.analysis.generate_paper_results_20261026 import fixed_class_bootstrap, make_recording_groups
from scripts.analysis.analyze_blinded_soundscape_review import EXPECTED_FIELDS, validate_review


class PaperResultsTests(unittest.TestCase):
    def test_recording_groups_keep_segments_together(self):
        names = np.array(["b", "a", "b", "c", "a"])
        groups = make_recording_groups(names)
        observed = [set(names[group]) for group in groups]

        self.assertEqual(observed, [{"a"}, {"b"}, {"c"}])
        self.assertEqual(sorted(np.concatenate(groups).tolist()), list(range(len(names))))

    def test_fixed_bootstrap_uses_full_data_class_set_and_is_paired(self):
        targets = np.array([
            [1, 0], [0, 1], [1, 0], [0, 1],
            [0, 0], [0, 0], [0, 0], [0, 0],
        ], dtype = np.int8)

        names = np.array(["a", "b", "c", "d", "e", "f", "g", "h"])
        base = {"targets": targets, "file_names": names}
        data = {43: dict(base), 44: dict(base)}

        for run_seed in (43, 44):
            data[run_seed]["scores_habitat"] = np.array([
                [.9, .1], [.1, .9], [.8, .2], [.2, .8],
                [.4, .4], [.3, .3], [.2, .2], [.1, .1],
            ])

            data[run_seed]["scores_standard"] = np.array([
                [.8, .2], [.2, .8], [.7, .3], [.3, .7],
                [.4, .4], [.3, .3], [.2, .2], [.1, .1],
            ])

        rows = fixed_class_bootstrap(data, replicates=50, seed=7, support_threshold=1)
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["fixed_eligible_classes"] for row in rows}, {2})
        self.assertEqual({row["valid_replicates"] for row in rows}, {50})
        self.assertLessEqual(rows[0]["ci95_low"], rows[0]["point_delta_habitat_minus_standard"])
        self.assertGreaterEqual(rows[0]["ci95_high"], rows[0]["point_delta_habitat_minus_standard"])
        mean_rows = {row["metric"]: row for row in rows if row["analysis"] == "mean_across_seeds"}

        for metric, row in mean_rows.items():
            seed_rows = [r for r in rows if r["analysis"] in {"seed_43", "seed_44"} and r["metric"] == metric]
            self.assertEqual(
                row["point_delta_habitat_minus_standard"],
                np.mean([r["point_delta_habitat_minus_standard"] for r in seed_rows])
                )

    def test_blinded_review_analysis_returns_aggregate_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            review_path = Path(temp_dir) / "review.csv"
            key_path = Path(temp_dir) / "key.csv"

            with review_path.open("w", newline = "", encoding = "utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=EXPECTED_FIELDS)
                writer.writeheader()
                writer.writerow(dict(zip(EXPECTED_FIELDS, ["clip_001", "yes", "species", "high", "note"])))
                writer.writerow(dict(zip(EXPECTED_FIELDS, ["clip_002", "uncertain", "", "low", ""])))

            with key_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["clip_id", "positive_target_count"])
                writer.writeheader()
                writer.writerows([{"clip_id": "clip_001", "positive_target_count": "1"},
                                  {"clip_id": "clip_002", "positive_target_count": "0"}])

            result = validate_review(review_path, key_path)
            self.assertTrue(result["responses_complete"])
            self.assertEqual(result["binary_evaluable_count"], 1)
            self.assertEqual(result["binary_agreement_count"], 1)
            self.assertNotIn("clip_001", str(result))
            self.assertNotIn("positive_target_count", str(result))


if __name__ == "__main__":
    unittest.main()
