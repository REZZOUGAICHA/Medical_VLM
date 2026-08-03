"""Extract clean EN-HI aligned pairs from BPCC's hin_Deva.tsv.

The raw file (archived at data/external_raw/bpcc_ilci/hin_Deva.tsv, kept out
of data_prep.py's scanned raw/ tree on purpose) uses a 4-column schema:

    src_lang    tgt_lang    src    tgt
    eng_Latn    hin_Deva    <English sentence>    <Hindi sentence>

data_prep.py's generic table reader looks for a header matching the literal
language codes ("en"/"hi") to find the right columns, and falls back to
columns 0/1 positionally otherwise -- here that fallback would grab the
literal repeated "eng_Latn"/"hin_Deva" strings instead of the actual
sentences. So, same pattern as extract_tico19.py: pull the right columns
here explicitly and write a clean aligned pair that data_prep.py's normal
reader handles correctly.

Usage:
    python src/extract_bpcc_ilci.py --tsv data/external_raw/bpcc_ilci/hin_Deva.tsv --out-dir data/raw/en-hi/bpcc_ilci
"""
import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--out-name", default="hin_Deva")
    args = ap.parse_args()

    total = 0
    bad_schema = 0
    empty = 0
    pairs = []
    with args.tsv.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        expected = ["src_lang", "tgt_lang", "src", "tgt"]
        if header != expected:
            raise SystemExit(f"Unexpected header {header}, expected {expected} -- schema may have changed, check before proceeding")
        for row in reader:
            total += 1
            if len(row) != 4:
                bad_schema += 1
                continue
            _sl, _tl, s, t = row
            s, t = s.strip(), t.strip()
            if not s or not t:
                empty += 1
                continue
            pairs.append((s, t))

    print(f"Read {total} data rows: {bad_schema} malformed (wrong column count), {empty} with an empty side")
    print(f"Usable pairs: {len(pairs)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    en_out = args.out_dir / f"{args.out_name}.en"
    hi_out = args.out_dir / f"{args.out_name}.hi"
    with en_out.open("w", encoding="utf-8") as fe, hi_out.open("w", encoding="utf-8") as fh:
        for s, t in pairs:
            fe.write(s + "\n")
            fh.write(t + "\n")
    print(f"Wrote {len(pairs)} pairs -> {en_out}, {hi_out}")


if __name__ == "__main__":
    main()
