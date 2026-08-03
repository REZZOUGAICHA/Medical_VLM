"""Extract EN-HI aligned pairs from the IITB dataset's Hugging Face Arrow format.

data_prep.py's generic reader only understands aligned .en/.hi text files and
.tsv/.csv tables -- it has no support for datasets.load_from_disk()-style
Arrow shards (dataset_dict.json + train/validation/test/ each with a
data-*.arrow file), which is the format `raw/en-hi/iitb/` is stored in. So,
same pattern as the other non-conforming sources: extract it here into a
plain aligned pair that data_prep.py's normal reader handles.

All three of IITB's own splits (train/validation/test) are merged into one
output pair here -- nothing in this project's evaluation currently holds out
IITB's test split for anything, so there's no leakage concern in re-pooling
it as general-domain training input; data_prep.py does its own train/val
split downstream regardless.

Usage:
    python src/extract_iitb.py --iitb-dir data/raw/en-hi/iitb --out-dir data/raw/en-hi/iitb
"""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc


def read_split(arrow_path: Path) -> list[tuple[str, str]]:
    with pa.memory_map(str(arrow_path), "r") as source:
        table = ipc.open_stream(source).read_all()
    col = table.column("translation")
    pairs = []
    for rec in col.to_pylist():
        en, hi = rec.get("en", "").strip(), rec.get("hi", "").strip()
        if en and hi:
            pairs.append((en, hi))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iitb-dir", required=True, type=Path, help="the datasets.load_from_disk() directory (contains dataset_dict.json)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--out-name", default="iitb")
    args = ap.parse_args()

    dict_path = args.iitb_dir / "dataset_dict.json"
    if not dict_path.exists():
        raise SystemExit(f"{dict_path} not found -- expected a datasets.load_from_disk() directory")
    splits = json.loads(dict_path.read_text(encoding="utf-8"))["splits"]

    all_pairs = []
    for split in splits:
        arrow_path = args.iitb_dir / split / "data-00000-of-00001.arrow"
        pairs = read_split(arrow_path)
        print(f"{split}: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    print(f"Total: {len(all_pairs)} pairs across {splits}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    en_out = args.out_dir / f"{args.out_name}.en"
    hi_out = args.out_dir / f"{args.out_name}.hi"
    with en_out.open("w", encoding="utf-8") as fe, hi_out.open("w", encoding="utf-8") as fh:
        for en, hi in all_pairs:
            fe.write(en + "\n")
            fh.write(hi + "\n")
    print(f"Wrote {len(all_pairs)} pairs -> {en_out}, {hi_out}")


if __name__ == "__main__":
    main()
