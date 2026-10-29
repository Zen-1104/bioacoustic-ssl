from pathlib import Path
import json
import re
import pandas as pd

ANN = Path("data/raw/western_ghats_soundscapes/annotation.csv")
OUT = Path(
    "results/real_soundscape_aggregate_s43_s44_fixnames/"
    "annotation_record_audit_20261025"
)

CASES = [
    ("WGH_0241_001_20181203_121000.flac", 150, "barfly1", 0, "background"),
    ("WGH_0117_001_20181201_091000.flac", 30, "gyhcaf1", 0, "background"),
    ("WGH_0713_026_20200313_094500.flac", 20, "crbsun2", 0, "background"),
    ("WGH_0054_001_20181129_064000.flac", 80, "gyhcaf1", 1, "annotated"),
    ("WGH_0747_027_20201216_163000.flac", 180, "yebbul3", 1, "annotated"),
]

def norm(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

def choose(columns, aliases, required=True):
    lookup = {norm(c): c for c in columns}
    for alias in aliases:
        if norm(alias) in lookup:
            return lookup[norm(alias)]
    if required:
        raise ValueError(
            f"Cannot identify a required field among {aliases}. "
            f"CSV columns: {list(columns)}"
        )
    return None

def overlaps(frame, start, end):
    normal = (frame["_start"] < end) & (frame["_end"] > start)
    # Also handle any zero-duration annotation rows.
    points = (
        (frame["_start"] == frame["_end"])
        & (frame["_start"] >= start)
        & (frame["_start"] < end)
    )
    return normal | points

def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")

    df = pd.read_csv(ANN)
    filename_col = choose(
        df.columns,
        ["filename", "file_name", "audio_filename", "recording_filename", "file"],
    )
    start_col = choose(
        df.columns,
        ["start_time", "start_time_s", "start_seconds", "start_s", "start"],
    )
    end_col = choose(
        df.columns,
        ["end_time", "end_time_s", "end_seconds", "end_s", "end"],
    )
    species_col = choose(
        df.columns,
        ["Species eBird Code", "species_code", "ebird_code", "ebird_species_code",
         "species", "primary_label", "code"],
    )

    df["_row"] = range(2, len(df) + 2)
    df["_filename"] = (
        df[filename_col].astype(str)
        .str.replace("\\", "/", regex=False)
        .str.strip()
        .map(lambda x: Path(x).name.lower())
    )
    df["_species"] = df[species_col].astype(str).str.strip().str.lower()
    df["_start"] = pd.to_numeric(df[start_col], errors="coerce")
    df["_end"] = pd.to_numeric(df[end_col], errors="coerce")
    df = df.dropna(subset=["_start", "_end"]).copy()

    detail_rows = []
    summary_rows = []
    original_cols = [
        c for c in df.columns
        if not c.startswith("_")
    ]

    for filename, start, focal_species, expected, segment_type in CASES:
        end = start + 10.0
        recording = df[df["_filename"] == filename.lower()].copy()

        exact = recording[overlaps(recording, start, end)].copy()
        context_start = max(0.0, start - 10.0)
        context_end = end + 10.0
        context = recording[overlaps(
            recording, context_start, context_end
        )].copy()
        context_only = context[
            ~context["_row"].isin(exact["_row"])
        ]

        exact_species = sorted(exact["_species"].unique().tolist())
        context_species = sorted(context["_species"].unique().tolist())
        focal_present = focal_species.lower() in exact_species

        if expected == 1:
            label_check = focal_present
            interpretation = (
                "Focal species appears in the source annotations for this window."
                if focal_present else
                "CHECK: focal species target is positive, but no matching focal-species "
                "interval overlaps this window."
            )
        else:
            label_check = len(exact) == 0
            interpretation = (
                "No source annotation interval overlaps this background window."
                if len(exact) == 0 else
                "CHECK: source annotations overlap a segment marked background."
            )

        summary_rows.append({
            "filename": filename,
            "window_start_s": start,
            "window_end_s": end,
            "segment_type_in_evaluation": segment_type,
            "focal_species": focal_species,
            "expected_focal_target": expected,
            "annotation_rows_overlapping_window": len(exact),
            "species_codes_overlapping_window": ";".join(exact_species),
            "focal_species_annotation_present": focal_present,
            "context_start_s": context_start,
            "context_end_s": context_end,
            "context_annotation_row_count": len(context),
            "context_only_annotation_row_count": len(context_only),
            "context_species_codes": ";".join(context_species),
            "label_consistency_check_passed": label_check,
            "interpretation": interpretation,
        })

        for relation, rows in (
            ("within_target_window", exact),
            ("nearby_context_only", context_only),
        ):
            for _, row in rows.iterrows():
                start_overlap = max(start, float(row["_start"]))
                end_overlap = min(end, float(row["_end"]))
                detail_rows.append({
                    "filename": filename,
                    "focal_species_under_review": focal_species,
                    "target_window_start_s": start,
                    "target_window_end_s": end,
                    "relation_to_target_window": relation,
                    "source_csv_row_number": int(row["_row"]),
                    "annotation_start_s": float(row["_start"]),
                    "annotation_end_s": float(row["_end"]),
                    "overlap_with_target_window_s": max(
                        0.0, end_overlap - start_overlap
                    ) if relation == "within_target_window" else 0.0,
                    "annotation_species_code": row["_species"],
                    "original_annotation_fields_json": json.dumps(
                        {c: (None if pd.isna(row[c]) else str(row[c]))
                         for c in original_cols},
                        ensure_ascii=False,
                    ),
                })

        print(f"\n{filename} | {start:.1f}-{end:.1f}s")
        print(
            f"Focal species: {focal_species}; expected target: {expected}; "
            f"annotation rows overlapping window: {len(exact)}"
        )
        print(interpretation)

        if exact.empty:
            print("  No annotation intervals overlap the target window.")
        else:
            for _, row in exact.sort_values("_start").iterrows():
                print(
                    f"  {row['_start']:.2f}-{row['_end']:.2f}s "
                    f"| {row['_species']} "
                    f"| source CSV row {int(row['_row'])}"
                )

        if not context_only.empty:
            print("  Nearby annotations in the 10-second context on either side:")
            for _, row in context_only.sort_values("_start").iterrows():
                print(
                    f"    {row['_start']:.2f}-{row['_end']:.2f}s "
                    f"| {row['_species']} "
                    f"| source CSV row {int(row['_row'])}"
                )

    OUT.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(summary_rows).to_csv(
        OUT / "annotation_window_summary.csv", index=False
    )
    pd.DataFrame(detail_rows).to_csv(
        OUT / "annotation_intervals_detail.csv", index=False
    )

    report = {
        "annotation_source": str(ANN),
        "annotation_rows_read": int(len(df)),
        "annotation_columns": list(original_cols),
        "cases_checked": len(CASES),
        "n_label_consistency_checks_passed": sum(
            r["label_consistency_check_passed"] for r in summary_rows
        ),
        "summary": summary_rows,
        "caveat": (
            "Absence of a source annotation means no interval was recorded "
            "for the window; it does not prove that no bird was audible. "
            "The dataset excludes some faint, ambiguous, or overlapping calls."
        ),
    }
    (OUT / "annotation_window_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nAUDIT FILES")
    print(OUT / "annotation_window_summary.csv")
    print(OUT / "annotation_intervals_detail.csv")
    print(OUT / "annotation_window_audit.json")

if __name__ == "__main__":
    main()
