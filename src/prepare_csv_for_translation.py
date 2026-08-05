"""Select the English text to translate for each row of the merged VLM dataset.

Priority per row (confirmed with the user):
  1. `report`, if it's real text (not empty, not a placeholder like "not mentioned")
  2. `radlex_phrase`, if present (a sentence already generated from the RadLex
     label(s) -- e.g. "Presence of a central venous catheter." -- this is the
     "phrase" the supervisor's note referred to, not the bare radlex_libelle
     label, which would be poor MT input on its own: no sentence context,
     multi-term rows aren't even grammatical as a semicolon list.)
  3. otherwise, left blank -- no fallback to `classification` (same reasoning
     as radlex_libelle: an isolated 2-4 word category label is exactly the
     kind of input that triggers nonsense output, e.g. the "atelectasis"
     phonetic-transliteration failure found during baseline eval).

Doesn't touch the model at all -- this only prepares the source text column,
so the (expensive, GPU-bound) translation step can run against something
already verified correct.

Usage:
    python src/prepare_csv_for_translation.py \
        --csv "data/to_translate/ALB_JEPA_final_dataset - ALB_JEPA_final_par_image_merged_clean_v4.csv" \
        --out data/to_translate/prepped_for_translation.csv
"""
import argparse
import csv
from pathlib import Path

PLACEHOLDER_VALUES = {"", "not mentioned", "n/a", "na", "none", "nan"}


def pick_text(row: dict) -> tuple[str, str]:
    """Returns (text, source_field) where source_field is 'report', 'radlex_phrase', or ''."""
    report = row.get("report", "").strip()
    if report and report.lower() not in PLACEHOLDER_VALUES:
        return report, "report"
    phrase = row.get("radlex_phrase", "").strip()
    if phrase:
        return phrase, "radlex_phrase"
    return "", ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with args.csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    print(f"Loaded {len(rows)} rows, columns: {fieldnames}")

    from collections import Counter
    source_counts = Counter()
    for row in rows:
        text, source = pick_text(row)
        row["text_to_translate"] = text
        row["text_source"] = source
        source_counts[source or "<none>"] += 1

    print("Source of text_to_translate:", dict(source_counts))
    n_translatable = len(rows) - source_counts["<none>"]
    print(f"Rows with something to translate: {n_translatable} / {len(rows)} ({n_translatable/len(rows)*100:.1f}%)")

    out_fieldnames = fieldnames + ["text_to_translate", "text_source"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
