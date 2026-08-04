"""Detokenize NEJM-enzh's raw train/dev files before they enter data_prep.py's pool.

Bug this fixes: `baseline_eval.py` detokenizes the NEJM *test* split before
scoring (English has Moses-style `@-@` hyphen-splitting and spaced
punctuation; Chinese is word-segmented with spaces between words, neither of
which is how the languages are actually written), but nothing ever
detokenized the *train*/*dev* split before data_prep.py pooled it. A model
fine-tuned on that raw pool learns to output space-segmented Chinese too,
matching its training data's storage format -- confirmed on the first
Mandarin fine-tune: it tanked the raw BLEU score (10.7 -> 5.0, looked like a
regression) purely from this token-granularity mismatch against the
properly-detokenized eval reference, while LaBSE and chrF (both
tokenization-insensitive) correctly showed the real improvement underneath.
`baseline_eval.py` was patched to normalize model output at eval time, which
is enough to score any checkpoint already trained on the raw data -- but
that's a workaround, not a fix. This is the fix: clean the training data
itself, so future fine-tunes learn to produce naturally-formatted Chinese
directly, and don't carry this footnote at all.

The raw files stay archived (untouched) at data/external_raw/nejm_enzh/, so
this can be re-run or reverted without re-fetching anything. `nejm.test.*`
is not handled here -- baseline_eval.py already detokenizes it on the fly at
eval time and it's never part of the training pool.

Usage:
    python src/detokenize_nejm.py --raw-dir data/external_raw/nejm_enzh --out-dir data/raw/en-zh/nejm_enzh
"""
import argparse
import re
from pathlib import Path


def detokenize_moses_en(text: str) -> str:
    text = text.replace(" @-@ ", "-")
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return text.strip()


def detokenize_segmented_zh(text: str) -> str:
    return text.replace(" ", "").strip()


def detokenize_file(src_path: Path, out_path: Path, fn) -> int:
    lines = src_path.read_text(encoding="utf-8").splitlines()
    out_lines = [fn(line) for line in lines]
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return len(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, type=Path, help="archived raw nejm_enzh dir (contains nejm.{train,dev}.{en,zh})")
    ap.add_argument("--out-dir", required=True, type=Path, help="where to write clean detokenized versions")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        n_en = detokenize_file(args.raw_dir / f"nejm.{split}.en", args.out_dir / f"nejm.{split}.en", detokenize_moses_en)
        n_zh = detokenize_file(args.raw_dir / f"nejm.{split}.zh", args.out_dir / f"nejm.{split}.zh", detokenize_segmented_zh)
        assert n_en == n_zh, f"{split}: {n_en} English lines vs {n_zh} Chinese lines -- should never happen, pairing would be broken"
        print(f"nejm.{split}: {n_en} lines detokenized -> {args.out_dir}")


if __name__ == "__main__":
    main()
