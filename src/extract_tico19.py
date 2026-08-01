"""Extract a single language pair out of the raw TICO-19 dev/test TSVs.

TICO-19's raw files (data/tico19-testset/tico19-testset/{dev,test}/*.tsv) hold
~30 languages and use an 8-column schema:

    sourceLang  targetLang  sourceString  targetString  stringID  url  license  translator_ID

That's not directly usable by data_prep.py's generic aligned-file / 2-column
table reader, so this pulls just sourceString/targetString for one language
pair and writes them out as a clean aligned pair -- dev split goes to a raw/
training-pool folder, test split goes to an eval/ held-out folder. The
original TICO-19 files are only read, never modified.

Usage:
    python src/extract_tico19.py --tico-dir data/tico19-testset/tico19-testset \
        --lang zh --dev-out data/raw/en-zh/tico19 --test-out data/eval/tico19_zh
"""
import argparse
import csv
from pathlib import Path


def extract(tsv_path: Path, out_dir: Path, out_name: str, src_ext: str, tgt_ext: str):
    with tsv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        src_idx = header.index("sourceString")
        tgt_idx = header.index("targetString")
        rows = [(row[src_idx].strip(), row[tgt_idx].strip()) for row in reader if row]

    out_dir.mkdir(parents=True, exist_ok=True)
    src_out = out_dir / f"{out_name}.{src_ext}"
    tgt_out = out_dir / f"{out_name}.{tgt_ext}"
    with src_out.open("w", encoding="utf-8") as fs, tgt_out.open("w", encoding="utf-8") as ft:
        for s, t in rows:
            fs.write(s + "\n")
            ft.write(t + "\n")
    print(f"{tsv_path} -> {src_out}, {tgt_out} ({len(rows)} pairs)")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tico-dir", required=True, type=Path, help="path to the extracted tico19-testset folder (contains dev/ and test/)")
    ap.add_argument("--lang", required=True, help="target language suffix as it appears in the TICO-19 filename, e.g. zh, hi, es-LA")
    ap.add_argument("--out-ext", default=None, help="file extension to use for the target language output (defaults to --lang, e.g. es-LA -> pass --out-ext es)")
    ap.add_argument("--dev-out", required=True, type=Path, help="output dir for the dev split (training pool)")
    ap.add_argument("--test-out", required=True, type=Path, help="output dir for the test split (held-out eval)")
    ap.add_argument("--out-name", default="tico19")
    args = ap.parse_args()

    out_ext = args.out_ext or args.lang

    dev_tsv = args.tico_dir / "dev" / f"dev.en-{args.lang}.tsv"
    test_tsv = args.tico_dir / "test" / f"test.en-{args.lang}.tsv"

    extract(dev_tsv, args.dev_out, args.out_name, "en", out_ext)
    extract(test_tsv, args.test_out, args.out_name, "en", out_ext)


if __name__ == "__main__":
    main()
