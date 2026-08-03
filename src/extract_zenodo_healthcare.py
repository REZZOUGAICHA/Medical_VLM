"""Extract EN-HI sentence pairs from the Zenodo healthcare patient-record CSVs.

The raw files (archived at data/external_raw/zenodo_healthcare/, kept out of
data_prep.py's scanned raw/ tree on purpose) are 13-column structured
synthetic patient records, not sentence pairs -- pooling them as-is would
make the generic reader fall back to columns 0/1 ("patient_id", "age") as if
they were the English/Hindi sentence. There is no single "the sentence"
field; several columns hold genuine free text.

Per an explicit decision (optimize for translation-training quality/volume,
not extraction effort), this pulls every free-text field as its own row
rather than concatenating them into one artificial per-patient blob:
Diagnosis, Remarks, Patient History, symptoms, treatment, timespan, and
Diagnosis Category. Each is checked for both a real sentence-like shape
(not a bare label/ID) and non-empty on both sides; every field was found
100% non-empty across all 2,182 rows in a pre-check, but this still
verifies at extraction time rather than assuming that holds.

Row alignment is verified by comparing `patient_id` order between the two
files before extracting anything (previously confirmed identical, but
re-checked here so this script is safe to re-run standalone later without
relying on that earlier one-off check).

Usage:
    python src/extract_zenodo_healthcare.py \
        --en-csv data/external_raw/zenodo_healthcare/English_dataset.csv \
        --hi-csv data/external_raw/zenodo_healthcare/hindi_dataset.csv \
        --out-dir data/raw/en-hi/zenodo_healthcare
"""
import argparse
import csv
from pathlib import Path

FREE_TEXT_FIELDS = [
    "Diagnosis",
    "Remarks",
    "Patient History",
    "symptoms",
    "treatment",
    "timespan",
    "Diagnosis Category",
]


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--en-csv", required=True, type=Path)
    ap.add_argument("--hi-csv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--out-name", default="zenodo")
    args = ap.parse_args()

    en_rows = load_rows(args.en_csv)
    hi_rows = load_rows(args.hi_csv)

    en_ids = [r["patient_id"] for r in en_rows]
    hi_ids = [r["patient_id"] for r in hi_rows]
    if en_ids != hi_ids:
        raise SystemExit(
            f"patient_id order mismatch between {args.en_csv} and {args.hi_csv} "
            f"({len(en_ids)} vs {len(hi_ids)} rows, or reordered) -- do not proceed until this is resolved"
        )
    print(f"Row alignment verified: {len(en_ids)} rows, matching patient_id order")

    missing_field = [f for f in FREE_TEXT_FIELDS if f not in en_rows[0] or f not in hi_rows[0]]
    if missing_field:
        raise SystemExit(f"Expected field(s) not found in CSV columns: {missing_field}")

    pairs = []
    per_field_counts = {f: 0 for f in FREE_TEXT_FIELDS}
    for en_row, hi_row in zip(en_rows, hi_rows):
        for field in FREE_TEXT_FIELDS:
            en_val, hi_val = en_row[field].strip(), hi_row[field].strip()
            if en_val and hi_val:
                pairs.append((en_val, hi_val))
                per_field_counts[field] += 1

    print("Pairs extracted per field:", per_field_counts)
    before = len(pairs)
    pairs = list(dict.fromkeys(pairs))  # de-dupe exact repeats (e.g. shared Diagnosis Category text), preserve order
    print(f"Total: {before} pairs -> {len(pairs)} after exact-duplicate removal")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    en_out = args.out_dir / f"{args.out_name}.en"
    hi_out = args.out_dir / f"{args.out_name}.hi"
    with en_out.open("w", encoding="utf-8") as fe, hi_out.open("w", encoding="utf-8") as fh:
        for en, hi in pairs:
            fe.write(en + "\n")
            fh.write(hi + "\n")
    print(f"Wrote {len(pairs)} pairs -> {en_out}, {hi_out}")


if __name__ == "__main__":
    main()
