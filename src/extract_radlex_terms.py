"""Extract each distinct RadLex term's real single-term English sentence.

`radlex_phrase` is generated from `radlex_libelle` (semicolon-separated RadLex
codes) via a fixed template: "Presence of {term1}, {term2}, and {term3}."
Long multi-term versions of this sentence are exactly what breaks the Hindi
model (see docs/process_guide.md); short single-term versions translate
reliably. This script pulls the real single-term sentence for each of the 35
distinct terms directly from rows that already have exactly one term -- no
sentence is invented, every one is a real value that appears in the CSV.

One term (swan-ganz_catheter) never appears alone in this dataset, so its
phrase is derived by string subtraction from a real two-term row
("Presence of a central venous catheter and a Swan-Ganz catheter.") --
verified with an assertion, not guessed.

Usage:
    python src/extract_radlex_terms.py \
        --csv data/to_translate/prepped_for_translation.csv \
        --out-json data/to_translate/radlex_term_sentences.json \
        --out-txt data/to_translate/radlex_term_sentences.txt
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-txt", required=True, type=Path)
    args = ap.parse_args()

    single_term_sentence = {}  # term -> real "Presence of X." sentence
    two_term_example = None   # fallback row for the one term with no single-term example
    all_terms = set()

    with args.csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lib = row.get("radlex_libelle", "").strip()
            phrase = row.get("radlex_phrase", "").strip()
            if not lib:
                continue
            terms = [t.strip() for t in lib.split(";") if t.strip()]
            all_terms.update(terms)
            if len(terms) == 1 and phrase:
                single_term_sentence.setdefault(terms[0], phrase)
            if (set(terms) == {"central_venous_catheter", "swan-ganz_catheter"}
                    and two_term_example is None and phrase):
                two_term_example = phrase

    missing = all_terms - set(single_term_sentence.keys())
    assert missing == {"swan-ganz_catheter"}, (
        f"Expected only swan-ganz_catheter to lack a single-term example, got: {missing}"
    )
    assert two_term_example is not None, "No central_venous_catheter + swan-ganz_catheter row found"

    prefix = "Presence of a central venous catheter and "
    assert two_term_example.startswith(prefix) and two_term_example.endswith("."), (
        f"Unexpected format for swan-ganz two-term example: {two_term_example!r}"
    )
    swan_ganz_fragment = two_term_example[len(prefix):-1]  # "a Swan-Ganz catheter"
    single_term_sentence["swan-ganz_catheter"] = f"Presence of {swan_ganz_fragment}."

    print(f"{len(all_terms)} distinct terms, {len(single_term_sentence)} single-term sentences resolved")
    assert set(single_term_sentence.keys()) == all_terms

    ordered_terms = sorted(single_term_sentence.keys())
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump({t: single_term_sentence[t] for t in ordered_terms}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out_json}")

    with args.out_txt.open("w", encoding="utf-8") as f:
        f.write("\n".join(single_term_sentence[t] for t in ordered_terms) + "\n")
    print(f"Wrote {args.out_txt} (line order matches sorted term order in the JSON)")


if __name__ == "__main__":
    main()
