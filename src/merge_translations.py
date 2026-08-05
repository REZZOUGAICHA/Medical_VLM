"""Map unique-string translations back onto every row of the merged VLM dataset.

Pure lookup, no model involved -- `unique_texts.txt`, `unique_texts_hi.txt`,
and `unique_texts_zh.txt` (produced by extract_unique_texts.py locally, then
colab_translate_csv.ipynb) are line-aligned, so this builds an
english-text -> (hindi, mandarin) dict and applies it to every row of
prepped_for_translation.csv via its text_to_translate column. Rows with no
text_to_translate (the 1,053 rows with neither a real report nor a
radlex_phrase) get blank translations, not a fallback guess.

Usage:
    python src/merge_translations.py \
        --prepped-csv data/to_translate/prepped_for_translation.csv \
        --unique-en data/to_translate/unique_texts.txt \
        --unique-hi data/to_translate/unique_texts_hi.txt \
        --unique-zh data/to_translate/unique_texts_zh.txt \
        --out data/to_translate/final_translated.csv
"""
import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepped-csv", required=True, type=Path)
    ap.add_argument("--unique-en", required=True, type=Path)
    ap.add_argument("--unique-hi", required=True, type=Path)
    ap.add_argument("--unique-zh", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    en_lines = args.unique_en.read_text(encoding="utf-8").splitlines()
    hi_lines = args.unique_hi.read_text(encoding="utf-8").splitlines()
    zh_lines = args.unique_zh.read_text(encoding="utf-8").splitlines()
    assert len(en_lines) == len(hi_lines) == len(zh_lines), (
        f"line count mismatch: en={len(en_lines)} hi={len(hi_lines)} zh={len(zh_lines)}"
    )
    lookup = {en: (hi, zh) for en, hi, zh in zip(en_lines, hi_lines, zh_lines)}
    print(f"Loaded {len(lookup)} unique translated strings")

    with args.prepped_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    print(f"Loaded {len(rows)} rows")

    missing = 0
    for row in rows:
        text = row["text_to_translate"]
        if not text.strip():
            row["hindi"] = ""
            row["mandarin"] = ""
            continue
        if text not in lookup:
            missing += 1
            row["hindi"] = ""
            row["mandarin"] = ""
            continue
        hi, zh = lookup[text]
        row["hindi"] = hi
        row["mandarin"] = zh

    if missing:
        print(f"WARNING: {missing} rows had text_to_translate not found in the unique-string lookup -- left blank")

    n_translated = sum(1 for row in rows if row["hindi"])
    print(f"Rows with translations: {n_translated} / {len(rows)} ({n_translated/len(rows)*100:.1f}%)")

    out_fieldnames = fieldnames + ["hindi", "mandarin"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
