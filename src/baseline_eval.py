"""Baseline evaluation of a *pretrained, not-yet-fine-tuned* MarianMT checkpoint,
scored against real held-out human-translated data already in this project.

Why this script exists and runs before any fine-tuning: fine-tuning without a
baseline means there is nothing to compare against afterward -- you'd have a
model and a feeling that it's probably better, with no number to put in a
paper. This measures the pretrained checkpoint first, so `train.py`'s output
can later be measured with the exact same script and compared directly.

*** Standing rule: never generate ground truth ***
An earlier version of this script used hand-written pilot sentences with
reference translations invented by Claude. That was wrong and has been
replaced: every source/reference pair scored here is pulled verbatim from
real, existing, human-translated data -- never written or guessed on the
spot, even as a placeholder. Inventing a reference makes the eval circular
(you'd be measuring "how similar is this to what an AI guessed," not "how
correct is this translation"), which defeats the entire purpose of a number
meant to appear in a paper. Real sources used:
  - Mandarin: `data/eval/nejm_zh/nejm.test.{en,zh}` -- NEJM-enzh's own
    test split (2,102 pairs), medical, professionally human-translated.
    (Moved out of data/raw/en-zh/nejm_enzh/ into data/eval/ so data_prep.py's
    pooling can't accidentally sweep the held-out test split into training --
    it originally could, which would have made this baseline invalid once a
    fine-tuned model was compared against it.)
  - Hindi:    `data/eval/tico19_hi/tico19.{en,hi}` -- TICO-19's test split
    (2,100 pairs), medical/COVID-domain, human-translated.
Both are held-out test splits, never touched by data_prep.py's training
pool, so they stay valid for a fine-tuned-model re-run later too.

Since the pilot label set (cardiomegaly, pleural effusion, pneumothorax,
edema, "no acute cardiopulmonary abnormality", support devices,
atelectasis, consolidation) comes from chest-X-ray reporting (CheXpert/
MIMIC-CXR labels) and these two real corpora are general internal-medicine
literature / COVID guidance -- not radiology reports -- most of those exact
terms are rare or absent in the real data. Per the standing rule, that
absence is reported explicitly (see "pilot_term_coverage" in the output),
not papered over with invented sentences. Where a term *is* found, treat it
as a rough substring match, not a guaranteed topical match -- inspect the
matched sentence text in the report; e.g. "cardiopulmonary" mostly turns up
"cardiopulmonary resuscitation" in NEJM, not the radiology sense.

Known preprocessing fix applied here (per the spec's tokenization note):
NEJM-enzh's raw files are Moses-style pre-tokenized (English has `@-@`
hyphen-splitting and spaced punctuation; Chinese is word-segmented with
spaces between words, which is not how written Chinese normally looks).
Both are de-tokenized on the fly before translation/scoring below --
otherwise the model would be penalized for an artifact of how the file is
stored, not for actual translation quality.

Usage:
    python src/baseline_eval.py --lang zh --model Helsinki-NLP/opus-mt-en-zh --label baseline --out results/baseline_en-zh.json
    python src/baseline_eval.py --lang hi --model Helsinki-NLP/opus-mt-en-hi --label baseline --out results/baseline_en-hi.json

    # after fine-tuning, re-run against the fine-tuned checkpoint for a direct delta:
    python src/baseline_eval.py --lang zh --model models/opus-mt-en-zh-medical/final --label fine-tuned --out results/finetuned_en-zh.json
"""
import argparse
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL = {
    "zh": "Helsinki-NLP/opus-mt-en-zh",
    "hi": "Helsinki-NLP/opus-mt-en-hi",
}

DATA_SOURCES = {
    "zh": {
        "src": Path("data/eval/nejm_zh/nejm.test.en"),
        "tgt": Path("data/eval/nejm_zh/nejm.test.zh"),
        "description": "NEJM-enzh test split (2,102 pairs), medical, human-translated (professional NEJM translators)",
        "detokenize_src": True,   # Moses-style: @-@ splitting, spaced punctuation
        "detokenize_tgt": True,   # word-segmented Chinese -> strip spaces
        # A model fine-tuned on nejm_enzh's raw (never-detokenized) training pool
        # learns to output space-segmented Chinese too, matching its training
        # data's storage format. That's a formatting artifact, not a translation
        # quality difference -- a native reader reads both forms identically --
        # but it wrecks word-level metrics (BLEU) since it creates a token-
        # granularity mismatch against the properly-detokenized reference above.
        # Confirmed on the first fine-tuned run: stripping spaces from the
        # hypothesis only (leaving the reference untouched) moved corpus BLEU
        # from 5.0 to 27.8, vs. 10.7 for the pretrained baseline -- the raw
        # (un-normalized) score was actively misleading, not just noisy.
        "normalize_hyp_for_scoring": True,
    },
    "hi": {
        "src": Path("data/eval/tico19_hi/tico19.en"),
        "tgt": Path("data/eval/tico19_hi/tico19.hi"),
        "description": "TICO-19 en-hi test split (2,100 pairs), medical/COVID-domain, human-translated",
        "detokenize_src": False,
        "detokenize_tgt": False,
        "normalize_hyp_for_scoring": False,  # Hindi uses spaces between words natively, nothing to strip
    },
}

# (label, substring) -- substring matching is a rough proxy, not a guaranteed
# topical match; the report includes the actual matched sentence so this can
# be sanity-checked by eye.
PILOT_TERMS = [
    ("cardiomegaly", "cardiomegaly"),
    ("pleural_effusion", "pleural effusion"),
    ("pneumothorax", "pneumothorax"),
    ("edema", "edema"),
    ("no_acute_cardiopulmonary_abnormality_EXACT_PHRASE", "no acute cardiopulmonary abnormality"),
    ("cardiopulmonary_BROAD_LOOSE_MATCH", "cardiopulmonary"),
    ("support_devices", "support device"),
    ("atelectasis", "atelectasis"),
    ("consolidation", "consolidation"),
]


def detokenize_moses_en(text: str) -> str:
    text = text.replace(" @-@ ", "-")
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return text.strip()


def detokenize_segmented_zh(text: str) -> str:
    return text.replace(" ", "").strip()


def load_pairs(src_path: Path, tgt_path: Path, detok_src: bool, detok_tgt: bool):
    src_lines = src_path.read_text(encoding="utf-8").splitlines()
    tgt_lines = tgt_path.read_text(encoding="utf-8").splitlines()
    assert len(src_lines) == len(tgt_lines), f"{src_path} has {len(src_lines)} lines, {tgt_path} has {len(tgt_lines)}"
    pairs = []
    for s, t in zip(src_lines, tgt_lines):
        if detok_src:
            s = detokenize_moses_en(s)
        if detok_tgt:
            t = detokenize_segmented_zh(t)
        pairs.append((s, t))
    return pairs


def find_term_matches(pairs, terms):
    matches = {label: [] for label, _ in terms}
    for i, (en, _ref) in enumerate(pairs):
        low = en.lower()
        for label, substring in terms:
            if substring in low:
                matches[label].append(i)
    return matches


def cosine_sim(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=sorted(DATA_SOURCES), help="target language")
    ap.add_argument("--model", default=None, help=f"checkpoint to evaluate (HF hub id or local dir). Defaults per --lang: {DEFAULT_MODEL}")
    ap.add_argument("--labse-model", default="sentence-transformers/LaBSE")
    ap.add_argument("--label", default="baseline", help="free-text tag for this run (e.g. 'baseline', 'fine-tuned'), stored in the report")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--sample-size", type=int, default=200, help="random sample size drawn from the non-pilot-matched remainder of the test split (0 = use the entire remainder, i.e. the full test split)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model_id = args.model or DEFAULT_MODEL[args.lang]
    source = DATA_SOURCES[args.lang]
    for key in ("src", "tgt"):
        if not source[key].exists():
            raise SystemExit(f"Expected real held-out data at {source[key]} but it doesn't exist.")

    print(f"Data source: {source['description']}")
    pairs = load_pairs(source["src"], source["tgt"], source["detokenize_src"], source["detokenize_tgt"])
    print(f"Loaded {len(pairs)} real held-out pairs from {source['src'].name} / {source['tgt'].name}")

    term_matches = find_term_matches(pairs, PILOT_TERMS)
    pilot_term_coverage = {label: len(idxs) for label, idxs in term_matches.items()}
    print("\nPilot term coverage in real held-out data (0 = absent, reported honestly, not filled in):")
    for label, count in pilot_term_coverage.items():
        print(f"  {label}: {count}")

    matched_indices = sorted({i for idxs in term_matches.values() for i in idxs})
    matched_set = set(matched_indices)
    remaining_indices = [i for i in range(len(pairs)) if i not in matched_set]

    import random
    rng = random.Random(args.seed)
    if args.sample_size and args.sample_size > 0:
        sample_indices = rng.sample(remaining_indices, min(args.sample_size, len(remaining_indices)))
    else:
        sample_indices = remaining_indices
    selected_indices = sorted(matched_set | set(sample_indices))
    print(f"\nEvaluating {len(selected_indices)} sentences total: {len(matched_indices)} pilot-term matches + {len(sample_indices)} randomly sampled (seed={args.seed}) from the remainder")

    index_to_terms = {}
    for label, idxs in term_matches.items():
        for i in idxs:
            index_to_terms.setdefault(i, []).append(label)

    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import MarianMTModel, MarianTokenizer
    import sacrebleu

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Loading MT checkpoint: {model_id}")
    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id).to(device)
    model.eval()

    print(f"Loading LaBSE: {args.labse_model}")
    labse = SentenceTransformer(args.labse_model, device=device)

    per_sentence = []
    hypotheses, references = [], []

    for n, i in enumerate(selected_indices, 1):
        en, ref = pairs[i]
        terms_here = index_to_terms.get(i, [])

        inputs = tokenizer([en], return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        start = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(**inputs, max_length=args.max_length)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        hyp = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        # Score against a normalized form when the checkpoint is known to emit
        # a formatting artifact (see DATA_SOURCES comment) -- the raw `hyp` is
        # still what's stored below, for transparency about what the model
        # actually produced.
        hyp_scored = detokenize_segmented_zh(hyp) if source.get("normalize_hyp_for_scoring") else hyp

        emb = labse.encode([hyp_scored, ref], normalize_embeddings=False)
        labse_score = cosine_sim(emb[0], emb[1])
        sent_bleu = sacrebleu.sentence_bleu(hyp_scored, [ref]).score

        per_sentence.append({
            "test_split_index": i,
            "pilot_terms_matched": terms_here,
            "source_en": en,
            "reference": ref,
            "hypothesis": hyp,
            "hypothesis_scored": hyp_scored if hyp_scored != hyp else None,
            "labse_cosine": labse_score,
            "sentence_bleu": sent_bleu,
            "inference_seconds": elapsed,
        })
        hypotheses.append(hyp_scored)
        references.append(ref)

        if n % 25 == 0 or n == len(selected_indices):
            print(f"  [{n}/{len(selected_indices)}] {elapsed*1000:.0f}ms  LaBSE={labse_score:.3f}  BLEU={sent_bleu:.1f}" + (f"  terms={terms_here}" if terms_here else ""))

    def summarize(rows):
        if not rows:
            return None
        labse_scores = [r["labse_cosine"] for r in rows]
        timings = [r["inference_seconds"] for r in rows]
        hyps = [r["hypothesis_scored"] or r["hypothesis"] for r in rows]
        refs = [r["reference"] for r in rows]
        return {
            "n_sentences": len(rows),
            "mean_labse_cosine": statistics.mean(labse_scores),
            "min_labse_cosine": min(labse_scores),
            "corpus_bleu": sacrebleu.corpus_bleu(hyps, [refs]).score,
            "corpus_chrf": sacrebleu.corpus_chrf(hyps, [refs]).score,
            "mean_inference_seconds": statistics.mean(timings),
            "median_inference_seconds": statistics.median(timings),
            "max_inference_seconds": max(timings),
        }

    pilot_rows = [r for r in per_sentence if r["pilot_terms_matched"]]
    summary_overall = summarize(per_sentence)
    summary_pilot_subset = summarize(pilot_rows)

    report = {
        "label": args.label,
        "model": model_id,
        "lang_pair": f"en-{args.lang}",
        "device": device,
        "labse_model": args.labse_model,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": {
            "src_file": str(source["src"]),
            "tgt_file": str(source["tgt"]),
            "description": source["description"],
        },
        "reference_translation_provenance": "Real human-translated held-out test data, never generated or invented.",
        "pilot_term_coverage": pilot_term_coverage,
        "sample_size_requested": args.sample_size,
        "seed": args.seed,
        "per_sentence": per_sentence,
        "summary_overall": summary_overall,
        "summary_pilot_term_subset": summary_pilot_subset,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Summary (overall, all evaluated sentences) ===")
    print(f"model:              {model_id}")
    print(f"n sentences:        {summary_overall['n_sentences']}")
    print(f"mean LaBSE cosine:  {summary_overall['mean_labse_cosine']:.3f}")
    print(f"corpus BLEU:        {summary_overall['corpus_bleu']:.1f}")
    print(f"corpus chrF:        {summary_overall['corpus_chrf']:.1f}")
    print(f"mean inference:     {summary_overall['mean_inference_seconds']*1000:.0f} ms/sentence")
    print(f"median inference:   {summary_overall['median_inference_seconds']*1000:.0f} ms/sentence")
    if summary_pilot_subset:
        print(f"\n=== Summary (pilot-term-matched subset only, n={summary_pilot_subset['n_sentences']}) ===")
        print(f"mean LaBSE cosine:  {summary_pilot_subset['mean_labse_cosine']:.3f}")
        print(f"corpus BLEU:        {summary_pilot_subset['corpus_bleu']:.1f}")
    else:
        print("\n(no pilot-term matches found in this real test split -- see pilot_term_coverage above)")
    print(f"\nReport saved to {args.out}")


if __name__ == "__main__":
    main()
