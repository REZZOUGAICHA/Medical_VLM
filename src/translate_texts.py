"""Batch-translate a plain-text file (one sentence per line) with a fine-tuned
MarianMT checkpoint, preserving line order and count 1:1 with the input.

Built for the CSV translation task: `text_to_translate` in the merged VLM
dataset is 99%+ populated by a templated `radlex_phrase` field, so only
~3,379 distinct strings exist across 156,461 rows (see
`extract_unique_texts.py`) -- this script translates just those unique
strings once; mapping the results back onto every row is a separate,
model-free step (`merge_translations.py`).

Usage:
    python src/translate_texts.py \
        --model models/opus-mt-en-hi-medical/final \
        --in data/to_translate/unique_texts.txt \
        --out data/to_translate/unique_texts_hi.txt \
        --batch-size 32
"""
import argparse
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer


def detokenize_segmented_zh(text: str) -> str:
    return text.replace(" ", "").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF hub ID or local checkpoint dir")
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--detok-zh", action="store_true",
                     help="strip the space-segmentation artifact this project's zh checkpoint was trained to emit (see docs/process_guide.md)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.model}")
    tokenizer = MarianTokenizer.from_pretrained(args.model)
    model = MarianMTModel.from_pretrained(args.model).to(device)
    model.eval()

    lines = args.in_path.read_text(encoding="utf-8").splitlines()
    print(f"Translating {len(lines)} lines, batch size {args.batch_size}")

    outputs = []
    for start in range(0, len(lines), args.batch_size):
        batch = lines[start:start + args.batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                            max_length=args.max_length, padding=True).to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_length=args.max_length)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        if args.detok_zh:
            decoded = [detokenize_segmented_zh(d) for d in decoded]
        outputs.extend(decoded)
        done = min(start + args.batch_size, len(lines))
        print(f"  {done}/{len(lines)}", end="\r")

    print()
    assert len(outputs) == len(lines), f"line count mismatch: {len(outputs)} outputs vs {len(lines)} inputs"

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text("\n".join(outputs) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
