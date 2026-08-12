"""Rebuild Hindi translations for radlex_phrase-sourced rows from per-term
translations, replacing the whole-sentence model output that fails 34% of
the time on this data (see docs/process_guide.md).

For each unique string in unique_texts.txt:
  - if it came from `radlex_phrase` (a "Presence of X, Y, and Z." sentence
    generated from radlex_libelle): rebuild the Hindi as the concatenation
    of each term's own independently-translated short sentence, using the
    real radlex_libelle -> radlex_phrase mapping already in the CSV to
    recover which terms produced this exact phrase. No Hindi grammar is
    invented here -- each piece is the model's own real translation of a
    short sentence, just reassembled instead of asking the model to
    translate one long enumeration (its proven failure mode).
  - if it came from `report` (real free text, no fixed vocabulary to
    decompose): keep the original whole-sentence translation unchanged.

Writes unique_texts_hi_v2.txt, line-aligned with unique_texts.txt exactly
like the original unique_texts_hi.txt -- a drop-in replacement for
merge_translations.py.

Usage:
    python src/reconstruct_radlex_hindi.py \
        --prepped-csv data/to_translate/prepped_for_translation.csv \
        --unique-en data/to_translate/unique_texts.txt \
        --unique-hi data/to_translate/unique_texts_hi.txt \
        --term-sentences-json data/to_translate/radlex_term_sentences.json \
        --term-sentences-hi data/to_translate/radlex_term_sentences_hi.txt \
        --out data/to_translate/unique_texts_hi_v2.txt
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepped-csv", required=True, type=Path)
    ap.add_argument("--unique-en", required=True, type=Path)
    ap.add_argument("--unique-hi", required=True, type=Path)
    ap.add_argument("--term-sentences-json", required=True, type=Path)
    ap.add_argument("--term-sentences-hi", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    term_order = list(json.loads(args.term_sentences_json.read_text(encoding="utf-8")).keys())
    term_hi_lines = args.term_sentences_hi.read_text(encoding="utf-8").splitlines()
    assert len(term_order) == len(term_hi_lines) == 35, (
        f"expected 35 terms aligned with 35 hi lines, got {len(term_order)} terms / {len(term_hi_lines)} hi lines"
    )
    term_to_hi = dict(zip(term_order, term_hi_lines))
    print(f"Loaded {len(term_to_hi)} term -> Hindi sentence mappings")

    # Recover radlex_phrase -> radlex_libelle from the real CSV (first-seen wins,
    # same tie-break convention as extract_unique_texts.py).
    phrase_to_libelle = {}
    with args.prepped_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("text_source") != "radlex_phrase":
                continue
            phrase = row["text_to_translate"]
            if phrase not in phrase_to_libelle:
                phrase_to_libelle[phrase] = row["radlex_libelle"]
    print(f"Recovered radlex_libelle for {len(phrase_to_libelle)} distinct radlex_phrase strings")

    en_lines = args.unique_en.read_text(encoding="utf-8").splitlines()
    hi_lines = args.unique_hi.read_text(encoding="utf-8").splitlines()
    assert len(en_lines) == len(hi_lines)

    out_lines = []
    n_reconstructed = 0
    n_kept = 0
    n_missing_terms = 0
    for en, old_hi in zip(en_lines, hi_lines):
        libelle = phrase_to_libelle.get(en)
        if libelle is None:
            out_lines.append(old_hi)  # report-sourced (or not found) -- keep as-is
            n_kept += 1
            continue
        terms = [t.strip() for t in libelle.split(";") if t.strip()]
        missing = [t for t in terms if t not in term_to_hi]
        if missing:
            # shouldn't happen -- all 35 terms were resolved -- but don't
            # silently produce a wrong sentence if it does
            out_lines.append(old_hi)
            n_missing_terms += 1
            continue
        rebuilt = " ".join(term_to_hi[t] for t in terms)
        out_lines.append(rebuilt)
        n_reconstructed += 1

    print(f"Reconstructed: {n_reconstructed}, kept original (report-sourced): {n_kept}, missing-term fallback: {n_missing_terms}")
    assert len(out_lines) == len(en_lines)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
