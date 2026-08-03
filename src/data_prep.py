"""Build train/val parallel-text splits from raw per-source dumps.

Expected layout under --raw-dir (one subfolder per dataset source, e.g.
data/raw/en-zh/nejm_enzh, data/raw/en-zh/tico19):

  Each source subfolder may contain, in any combination:
    - Aligned monolingual pairs: any basename repeated with the two language
      extensions, e.g. `train.en` + `train.zh` (line N of one corresponds to
      line N of the other). This is the standard OPUS/Moses export format.
    - Tabular files (.tsv or .csv) with two columns holding the source and
      target language sentences. Column headers are matched case-insensitively
      against the language codes (e.g. "en", "zh") if present; otherwise the
      first two columns are used positionally.

Sentence pairs are pooled across all source subfolders (or just the ones
named in --sources, if given -- use this to keep, e.g., general-domain and
medical-domain sources in separate pools for a two-stage fine-tune, since
they can live as sibling subfolders under the same --raw-dir), exact-duplicate
pairs are dropped, and the result is shuffled (fixed seed) and split into
train/val. Every output row is tagged with which source subfolder it came
from (train.source / val.source, line-aligned with train.<lang> / val.<lang>)
so a later ablation can filter or exclude rows by provenance (e.g. human- vs
machine-translated sources) without re-running extraction.
"""
import argparse
import csv
import random
import sys
from pathlib import Path

Pair = tuple[str, str, str]  # (source_text, target_text, source_name)


def read_aligned_pair(src_file: Path, tgt_file: Path, source_name: str) -> list[Pair]:
    with src_file.open(encoding="utf-8") as fs, tgt_file.open(encoding="utf-8") as ft:
        src_lines = [l.rstrip("\n") for l in fs]
        tgt_lines = [l.rstrip("\n") for l in ft]
    if len(src_lines) != len(tgt_lines):
        print(
            f"  ! WARNING: {src_file.name} has {len(src_lines)} lines but "
            f"{tgt_file.name} has {len(tgt_lines)} lines -- skipping this pair",
            file=sys.stderr,
        )
        return []
    return [(s, t, source_name) for s, t in zip(src_lines, tgt_lines)]


def read_table(path: Path, src_lang: str, tgt_lang: str, source_name: str) -> list[Pair]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = [r for r in reader if r]
    if not rows:
        return []
    header = [c.strip().lower() for c in rows[0]]
    if src_lang.lower() in header and tgt_lang.lower() in header:
        src_idx = header.index(src_lang.lower())
        tgt_idx = header.index(tgt_lang.lower())
        rows = rows[1:]
    else:
        src_idx, tgt_idx = 0, 1
    pairs = []
    for row in rows:
        if len(row) <= max(src_idx, tgt_idx):
            continue
        s, t = row[src_idx].strip(), row[tgt_idx].strip()
        if s and t:
            pairs.append((s, t, source_name))
    return pairs


def collect_source_dir(source_dir: Path, src_lang: str, tgt_lang: str) -> list[Pair]:
    pairs: list[Pair] = []
    source_name = source_dir.name

    src_files = {p.stem: p for p in source_dir.glob(f"*.{src_lang}")}
    tgt_files = {p.stem: p for p in source_dir.glob(f"*.{tgt_lang}")}
    for stem in sorted(set(src_files) & set(tgt_files)):
        found = read_aligned_pair(src_files[stem], tgt_files[stem], source_name)
        print(f"  {source_name}/{stem}.{{{src_lang},{tgt_lang}}}: {len(found)} pairs")
        pairs.extend(found)

    for table in list(source_dir.glob("*.tsv")) + list(source_dir.glob("*.csv")):
        found = read_table(table, src_lang, tgt_lang, source_name)
        print(f"  {source_name}/{table.name}: {len(found)} pairs")
        pairs.extend(found)

    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang-pair", required=True, help="e.g. en-zh, zh-en, en-hi, hi-en")
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--sources", default=None, help="comma-separated list of source subfolder names to include (default: every subfolder under --raw-dir). Use this to pool a subset -- e.g. only the general-domain sources for a Stage 1 fine-tune -- without those sources having to live in a separate --raw-dir.")
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len-chars", type=int, default=2000, help="drop pairs where either side exceeds this many characters")
    args = ap.parse_args()

    src_lang, tgt_lang = args.lang_pair.split("-")
    if not args.raw_dir.is_dir():
        sys.exit(f"raw dir not found: {args.raw_dir}")

    source_dirs = sorted(p for p in args.raw_dir.iterdir() if p.is_dir())
    if not source_dirs:
        sys.exit(
            f"No source subfolders found in {args.raw_dir}. "
            f"Drop each dataset's files into its own subfolder first."
        )

    if args.sources:
        wanted = {s.strip() for s in args.sources.split(",") if s.strip()}
        available = {p.name for p in source_dirs}
        missing = wanted - available
        if missing:
            sys.exit(f"--sources named folders not found under {args.raw_dir}: {sorted(missing)} (available: {sorted(available)})")
        source_dirs = [p for p in source_dirs if p.name in wanted]
        print(f"Restricting to --sources: {sorted(wanted)}")

    all_pairs: list[Pair] = []
    for source_dir in source_dirs:
        all_pairs.extend(collect_source_dir(source_dir, src_lang, tgt_lang))

    if not all_pairs:
        sys.exit(
            f"No usable sentence pairs found under {args.raw_dir} "
            f"(sources considered: {[p.name for p in source_dirs]}). "
            f"Check the file-format expectations in this script's docstring."
        )

    before = len(all_pairs)
    all_pairs = [
        (s, t, src) for s, t, src in all_pairs
        if s and t and len(s) <= args.max_len_chars and len(t) <= args.max_len_chars
    ]
    # De-dupe on (source_text, target_text) only, keeping the first source tag seen,
    # so the same sentence pair appearing in two sources doesn't get double-counted.
    seen: dict[tuple[str, str], str] = {}
    for s, t, src in all_pairs:
        seen.setdefault((s, t), src)
    all_pairs = [(s, t, src) for (s, t), src in seen.items()]
    print(f"\nPooled {before} pairs -> {len(all_pairs)} after length filter + de-dupe")

    by_source: dict[str, int] = {}
    for _s, _t, src in all_pairs:
        by_source[src] = by_source.get(src, 0) + 1
    print("Composition by source:", by_source)

    rng = random.Random(args.seed)
    rng.shuffle(all_pairs)
    n_val = max(1, int(len(all_pairs) * args.val_ratio))
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        src_out = args.out_dir / f"{split_name}.{src_lang}"
        tgt_out = args.out_dir / f"{split_name}.{tgt_lang}"
        prov_out = args.out_dir / f"{split_name}.source"
        with src_out.open("w", encoding="utf-8") as fs, tgt_out.open("w", encoding="utf-8") as ft, prov_out.open("w", encoding="utf-8") as fp:
            for s, t, src in split_pairs:
                fs.write(s + "\n")
                ft.write(t + "\n")
                fp.write(src + "\n")
        print(f"Wrote {len(split_pairs)} pairs -> {src_out}, {tgt_out}, {prov_out}")


if __name__ == "__main__":
    main()
