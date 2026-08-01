"""Evaluate a fine-tuned (or base) MarianMT checkpoint with BLEU + chrF.

Intended for the held-out sets called out in the spec: WMT18 (general
held-out, matching teammate 1's usage, zh-en only) and the TICO-19 test
splits (en-zh, en-hi) -- so results are comparable across languages and
across teammates' models on a shared benchmark.

Usage:
    python src/evaluate.py \
        --model-dir models/opus-mt-zh-en-medical/final \
        --test-src data/raw/eval/wmt18.zh \
        --test-tgt data/raw/eval/wmt18.en
"""
import argparse
from pathlib import Path

import sacrebleu
import torch
from transformers import MarianMTModel, MarianTokenizer


def batched(seq, batch_size):
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="local path or HF hub id of the model to evaluate")
    ap.add_argument("--test-src", required=True, type=Path)
    ap.add_argument("--test-tgt", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--out-predictions", type=Path, default=None, help="optional path to dump raw translations, one per line")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_dir} on {device} ...")
    tokenizer = MarianTokenizer.from_pretrained(args.model_dir)
    model = MarianMTModel.from_pretrained(args.model_dir).to(device)
    model.eval()

    src_lines = args.test_src.read_text(encoding="utf-8").splitlines()
    tgt_lines = args.test_tgt.read_text(encoding="utf-8").splitlines()
    assert len(src_lines) == len(tgt_lines), (
        f"{args.test_src} has {len(src_lines)} lines, {args.test_tgt} has {len(tgt_lines)}"
    )
    print(f"Evaluating on {len(src_lines)} sentence pairs")

    predictions = []
    with torch.no_grad():
        for batch in batched(src_lines, args.batch_size):
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to(device)
            generated = model.generate(**inputs, max_length=args.max_length)
            predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    bleu = sacrebleu.corpus_bleu(predictions, [tgt_lines])
    chrf = sacrebleu.corpus_chrf(predictions, [tgt_lines])
    print(f"\nBLEU:  {bleu.score:.2f}")
    print(f"chrF:  {chrf.score:.2f}")

    if args.out_predictions:
        args.out_predictions.write_text("\n".join(predictions) + "\n", encoding="utf-8")
        print(f"Wrote predictions to {args.out_predictions}")


if __name__ == "__main__":
    main()
