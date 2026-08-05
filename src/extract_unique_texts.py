"""Extract the unique strings from text_to_translate for a much smaller translation job.

`radlex_phrase` is templated (built from a finite set of RadLex term
combinations), so despite 156,461 rows needing translation, only ~3,379
distinct strings actually appear. Translating each unique string once and
mapping the result back to every matching row is ~46x less work than
translating every row independently -- turns this from a bulk-GPU-job into
something that finishes in a couple of minutes.

Usage:
    python src/extract_unique_texts.py --csv data/to_translate/prepped_for_translation.csv --out data/to_translate/unique_texts.txt
"""
import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with args.csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        texts = [row["text_to_translate"] for row in reader if row["text_to_translate"].strip()]

    unique_texts = list(dict.fromkeys(texts))  # de-dupe, preserve first-seen order (reproducible)
    print(f"{len(texts)} total non-empty rows -> {len(unique_texts)} unique strings ({len(unique_texts)/len(texts)*100:.1f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(unique_texts) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
