"""Validate and privately summarize a completed blinded clip review CSV.

The reviewer package does not contain the answer key. This script is for the
analysis copy kept with the research data, not for distribution to reviewers.
It never emits per-clip hidden labels or scores and refuses to overwrite output.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REVIEW_DIR = Path("results/real_soundscape_aggregate_s43_s44_fixnames/blinded_annotation_review_20261025")
KEY = REVIEW_DIR / "answer_key_internal_DO_NOT_SHARE_WITH_REVIEWER.csv"
EXPECTED_FIELDS = [
    "clip_id",
    "bird_vocalization_heard_yes_no_uncertain",
    "species_identified_if_confident",
    "confidence_high_medium_low",
    "notes_other_sounds_or_uncertainty",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)

        if reader.fieldnames != EXPECTED_FIELDS:
            raise ValueError(f"Review CSV columns must match the provided template: {EXPECTED_FIELDS}")

        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def validate_review(path: Path, key_path: Path = KEY) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)

    if not key_path.is_file():
        raise FileNotFoundError(f"Private answer key missing: {key_path}")

    rows = read_rows(path)
    with key_path.open(newline = "", encoding = "utf-8-sig") as stream:
        key_reader = csv.DictReader(stream)
        key_rows = list(key_reader)

    expected_ids = {row["clip_id"] for row in key_rows}
    ids = [row["clip_id"] for row in rows]

    if len(ids) != len(set(ids)):
        raise ValueError("Review CSV contains duplicate clip_id values")

    if set(ids) != expected_ids:
        raise ValueError(f"Review CSV must contain all {len(expected_ids)} expected clip IDs exactly once")

    allowed_heard = {"", "yes", "no", "uncertain"}
    allowed_confidence = {"", "high", "medium", "low"}

    for row in rows:
        if row["bird_vocalization_heard_yes_no_uncertain"].lower() not in allowed_heard:
            raise ValueError(f"Invalid bird-vocalization response for {row['clip_id']}")

        if row["confidence_high_medium_low"].lower() not in allowed_confidence:
            raise ValueError(f"Invalid confidence response for {row['clip_id']}")

    reviewed = [r for r in rows if r["bird_vocalization_heard_yes_no_uncertain"]]
    complete = len(reviewed) == len(expected_ids)

    # Suppress clip-level labels and responses from the output; only aggregate counts are returned.
    true_bird = {r["clip_id"] for r in key_rows if int(r["positive_target_count"]) > 0}
    binary_evaluable = [r for r in reviewed if r["bird_vocalization_heard_yes_no_uncertain"].lower() != "uncertain"]

    binary_correct = sum(
        (r["bird_vocalization_heard_yes_no_uncertain"].lower() == "yes")
        == (r["clip_id"] in true_bird)
        for r in binary_evaluable
    )

    return {
        "review_csv": str(path),
        "expected_clip_count": len(expected_ids),
        "returned_clip_count": len(rows),
        "responses_complete": complete,
        "heard_yes_count": sum(r["bird_vocalization_heard_yes_no_uncertain"].lower() == "yes" for r in rows),
        "heard_no_count": sum(r["bird_vocalization_heard_yes_no_uncertain"].lower() == "no" for r in rows),
        "uncertain_count": sum(r["bird_vocalization_heard_yes_no_uncertain"].lower() == "uncertain" for r in rows),
        "missing_heard_response_count": len(expected_ids) - len(reviewed),
        "binary_evaluable_count": len(binary_evaluable),
        "binary_agreement_count": binary_correct,
        "binary_agreement_rate": binary_correct / len(binary_evaluable) if binary_evaluable else None,
        "species_identification_response_count": sum(bool(r["species_identified_if_confident"]) for r in rows),
        "confidence_response_count": sum(bool(r["confidence_high_medium_low"]) for r in rows),
        "analysis_scope": "Descriptive agreement on this stratified 60-clip review set only; not a population estimate. Uncertain responses are excluded from binary agreement.",
        "human_review_status": "Completed CSV validated; this tool does not perform or substitute for listening.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_csv", type = Path, help = "Completed copy of review_template.csv")
    parser.add_argument("--output-json", type = Path, required = True, help = "New aggregate-only JSON output path")
    args = parser.parse_args()

    if args.output_json.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_json}")

    result = validate_review(args.review_csv)
    args.output_json.parent.mkdir(parents = True, exist_ok = True)
    args.output_json.write_text(json.dumps(result, indent = 2) + "\n", encoding = "utf-8")

    print(f"Validated {result['returned_clip_count']} blinded review rows; aggregate report: {args.output_json}")


if __name__ == "__main__":
    main()
