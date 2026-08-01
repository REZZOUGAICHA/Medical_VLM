"""Baseline evaluation of a *pretrained, not-yet-fine-tuned* MarianMT checkpoint
on a pilot set of radiology-report-style medical sentences.

Why this script exists and runs before any fine-tuning: fine-tuning without a
baseline means there is nothing to compare against afterward -- you'd have a
model and a feeling that it's probably better, with no number to put in a
paper. This measures the pretrained checkpoint first, so `train.py`'s output
can later be measured with the exact same script and compared directly.

The pilot sentences use the standard CheXpert/MIMIC-CXR chest-X-ray finding
labels (cardiomegaly, pleural effusion, pneumothorax, edema, atelectasis,
consolidation, support devices, "no acute cardiopulmonary abnormality"),
each in three phrasings: stated directly, negated ("no evidence of X"), and
uncertain ("X cannot be excluded"). This matters because BLEU (which scores
by counting overlapping words) is easily fooled by these three phrasings --
they share most of their vocabulary but mean very different things
clinically. LaBSE embeds meaning instead of counting words, so it is used as
the primary metric here (matching the team's standardized metric); sacreBLEU
is kept as a secondary, more familiar number for comparability.

*** IMPORTANT CAVEAT ***
The Chinese and Hindi reference translations in PILOT_SET below were drafted
by Claude as scaffolding, NOT sourced from a professional clinical
translator or verified by a native speaker. They are almost certainly
reasonable, but "almost certainly reasonable" is not the same as "verified,"
and this matters more than usual because clinical translation errors
(especially around negation/hedging) can be clinically meaningful. Treat the
LaBSE/BLEU numbers this script produces as good enough for a *relative*
before/after fine-tuning comparison -- do not present them as validated
ground truth in the paper until a native-speaking clinician has reviewed
PILOT_SET's "ref" fields.

Usage:
    python src/baseline_eval.py --lang zh --model Helsinki-NLP/opus-mt-en-zh --label baseline --out results/baseline_en-zh.json
    python src/baseline_eval.py --lang hi --model Helsinki-NLP/opus-mt-en-hi --label baseline --out results/baseline_en-hi.json

    # after fine-tuning, re-run against the fine-tuned checkpoint for a direct delta:
    python src/baseline_eval.py --lang zh --model models/opus-mt-en-zh-medical/final --label fine-tuned --out results/finetuned_en-zh.json
"""
import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL = {
    "zh": "Helsinki-NLP/opus-mt-en-zh",
    "hi": "Helsinki-NLP/opus-mt-en-hi",
}

# Each entry: (term, polarity, English source, reference translation).
# See the caveat above about reference translation provenance.
PILOT_SET = {
    "zh": [
        ("cardiomegaly", "positive", "The chest X-ray shows cardiomegaly.", "胸片显示心脏肥大。"),
        ("cardiomegaly", "negated", "There is no evidence of cardiomegaly.", "未见心脏肥大征象。"),
        ("cardiomegaly", "uncertain", "Cardiomegaly cannot be excluded.", "不能排除心脏肥大。"),

        ("pleural_effusion", "positive", "There is a small pleural effusion on the right side.", "右侧可见少量胸腔积液。"),
        ("pleural_effusion", "negated", "No pleural effusion is identified.", "未见胸腔积液。"),
        ("pleural_effusion", "uncertain", "A small pleural effusion cannot be ruled out.", "不能排除少量胸腔积液。"),

        ("pneumothorax", "positive", "A pneumothorax is present on the left side.", "左侧可见气胸。"),
        ("pneumothorax", "negated", "No pneumothorax is seen.", "未见气胸。"),
        ("pneumothorax", "uncertain", "A small pneumothorax cannot be excluded.", "不能排除少量气胸。"),

        ("edema", "positive", "There is evidence of pulmonary edema.", "可见肺水肿征象。"),
        ("edema", "negated", "No pulmonary edema is present.", "未见肺水肿。"),
        ("edema", "uncertain", "Mild pulmonary edema cannot be ruled out.", "不能排除轻度肺水肿。"),

        ("no_acute_cardiopulmonary_abnormality", "positive", "No acute cardiopulmonary abnormality is identified.", "未见急性心肺异常。"),
        ("no_acute_cardiopulmonary_abnormality", "negated", "There is no acute cardiopulmonary process.", "未见急性心肺病变。"),
        ("no_acute_cardiopulmonary_abnormality", "uncertain", "No definite acute cardiopulmonary abnormality; a subtle process cannot be excluded.", "未见明确急性心肺异常，但不能排除轻微病变。"),

        ("support_devices", "positive", "Support devices are noted, including an endotracheal tube in standard position.", "可见支持性装置，包括位置正常的气管插管。"),
        ("support_devices", "negated", "No support devices are present.", "未见支持性装置。"),
        ("support_devices", "uncertain", "Possible malposition of a support device cannot be excluded.", "不能排除支持性装置位置不当。"),

        ("atelectasis", "positive", "There is mild atelectasis at the left lung base.", "左肺底可见轻度肺不张。"),
        ("atelectasis", "negated", "No atelectasis is seen.", "未见肺不张。"),
        ("atelectasis", "uncertain", "Subsegmental atelectasis cannot be excluded.", "不能排除亚段性肺不张。"),

        ("consolidation", "positive", "There is a focal consolidation in the right lower lobe.", "右肺下叶可见局灶性实变。"),
        ("consolidation", "negated", "No consolidation is identified.", "未见实变。"),
        ("consolidation", "uncertain", "Early consolidation cannot be ruled out.", "不能排除早期实变。"),
    ],
    "hi": [
        ("cardiomegaly", "positive", "The chest X-ray shows cardiomegaly.", "छाती के एक्स-रे में हृदय का बढ़ना (कार्डियोमेगाली) दिखाई देता है।"),
        ("cardiomegaly", "negated", "There is no evidence of cardiomegaly.", "कार्डियोमेगाली का कोई प्रमाण नहीं है।"),
        ("cardiomegaly", "uncertain", "Cardiomegaly cannot be excluded.", "कार्डियोमेगाली से इनकार नहीं किया जा सकता।"),

        ("pleural_effusion", "positive", "There is a small pleural effusion on the right side.", "दाईं ओर थोड़ी मात्रा में फुफ्फुस बहाव (प्लूरल इफ्यूजन) मौजूद है।"),
        ("pleural_effusion", "negated", "No pleural effusion is identified.", "कोई फुफ्फुस बहाव नहीं पाया गया।"),
        ("pleural_effusion", "uncertain", "A small pleural effusion cannot be ruled out.", "थोड़े फुफ्फुस बहाव से इनकार नहीं किया जा सकता।"),

        ("pneumothorax", "positive", "A pneumothorax is present on the left side.", "बाईं ओर न्यूमोथोरैक्स मौजूद है।"),
        ("pneumothorax", "negated", "No pneumothorax is seen.", "कोई न्यूमोथोरैक्स नहीं दिखाई देता।"),
        ("pneumothorax", "uncertain", "A small pneumothorax cannot be excluded.", "थोड़े न्यूमोथोरैक्स से इनकार नहीं किया जा सकता।"),

        ("edema", "positive", "There is evidence of pulmonary edema.", "फुफ्फुसीय शोथ (पल्मोनरी एडिमा) के प्रमाण मौजूद हैं।"),
        ("edema", "negated", "No pulmonary edema is present.", "कोई फुफ्फुसीय शोथ मौजूद नहीं है।"),
        ("edema", "uncertain", "Mild pulmonary edema cannot be ruled out.", "हल्के फुफ्फुसीय शोथ से इनकार नहीं किया जा सकता।"),

        ("no_acute_cardiopulmonary_abnormality", "positive", "No acute cardiopulmonary abnormality is identified.", "कोई तीव्र हृदय-फुफ्फुसीय असामान्यता नहीं पाई गई।"),
        ("no_acute_cardiopulmonary_abnormality", "negated", "There is no acute cardiopulmonary process.", "कोई तीव्र हृदय-फुफ्फुसीय प्रक्रिया मौजूद नहीं है।"),
        ("no_acute_cardiopulmonary_abnormality", "uncertain", "No definite acute cardiopulmonary abnormality; a subtle process cannot be excluded.", "कोई स्पष्ट तीव्र हृदय-फुफ्फुसीय असामान्यता नहीं है, लेकिन एक सूक्ष्म प्रक्रिया से इनकार नहीं किया जा सकता।"),

        ("support_devices", "positive", "Support devices are noted, including an endotracheal tube in standard position.", "सहायक उपकरण देखे गए हैं, जिसमें मानक स्थिति में एंडोट्रेकियल ट्यूब शामिल है।"),
        ("support_devices", "negated", "No support devices are present.", "कोई सहायक उपकरण मौजूद नहीं है।"),
        ("support_devices", "uncertain", "Possible malposition of a support device cannot be excluded.", "सहायक उपकरण की संभावित गलत स्थिति से इनकार नहीं किया जा सकता।"),

        ("atelectasis", "positive", "There is mild atelectasis at the left lung base.", "बाएं फेफड़े के आधार पर हल्का एटेलेक्टेसिस मौजूद है।"),
        ("atelectasis", "negated", "No atelectasis is seen.", "कोई एटेलेक्टेसिस नहीं दिखाई देता।"),
        ("atelectasis", "uncertain", "Subsegmental atelectasis cannot be excluded.", "सबसेगमेंटल एटेलेक्टेसिस से इनकार नहीं किया जा सकता।"),

        ("consolidation", "positive", "There is a focal consolidation in the right lower lobe.", "दाएं फेफड़े के निचले लोब में फोकल कंसॉलिडेशन मौजूद है।"),
        ("consolidation", "negated", "No consolidation is identified.", "कोई कंसॉलिडेशन नहीं पाया गया।"),
        ("consolidation", "uncertain", "Early consolidation cannot be ruled out.", "प्रारंभिक कंसॉलिडेशन से इनकार नहीं किया जा सकता।"),
    ],
}


def cosine_sim(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=sorted(PILOT_SET), help="target language of the pilot set")
    ap.add_argument("--model", default=None, help=f"checkpoint to evaluate (HF hub id or local dir). Defaults per --lang: {DEFAULT_MODEL}")
    ap.add_argument("--labse-model", default="sentence-transformers/LaBSE")
    ap.add_argument("--label", default="baseline", help="free-text tag for this run (e.g. 'baseline', 'fine-tuned'), stored in the report")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()

    model_id = args.model or DEFAULT_MODEL[args.lang]

    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import MarianMTModel, MarianTokenizer
    import sacrebleu

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading MT checkpoint: {model_id}")
    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id).to(device)
    model.eval()

    print(f"Loading LaBSE: {args.labse_model}")
    labse = SentenceTransformer(args.labse_model, device=device)

    pilot = PILOT_SET[args.lang]
    per_sentence = []
    hypotheses, references = [], []

    for term, polarity, en, ref in pilot:
        inputs = tokenizer([en], return_tensors="pt", truncation=True, max_length=args.max_length).to(device)

        start = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(**inputs, max_length=args.max_length)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        hyp = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        emb = labse.encode([hyp, ref], normalize_embeddings=False)
        labse_score = cosine_sim(emb[0], emb[1])
        sent_bleu = sacrebleu.sentence_bleu(hyp, [ref]).score

        per_sentence.append({
            "term": term,
            "polarity": polarity,
            "source_en": en,
            "reference": ref,
            "hypothesis": hyp,
            "labse_cosine": labse_score,
            "sentence_bleu": sent_bleu,
            "inference_seconds": elapsed,
        })
        hypotheses.append(hyp)
        references.append(ref)
        print(f"[{term}/{polarity}] {elapsed*1000:.0f}ms  LaBSE={labse_score:.3f}  BLEU={sent_bleu:.1f}")
        print(f"  en:  {en}")
        print(f"  ref: {ref}")
        print(f"  hyp: {hyp}")

    corpus_bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    corpus_chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    labse_scores = [row["labse_cosine"] for row in per_sentence]
    timings = [row["inference_seconds"] for row in per_sentence]

    summary = {
        "n_sentences": len(pilot),
        "mean_labse_cosine": statistics.mean(labse_scores),
        "min_labse_cosine": min(labse_scores),
        "corpus_bleu": corpus_bleu.score,
        "corpus_chrf": corpus_chrf.score,
        "mean_inference_seconds": statistics.mean(timings),
        "median_inference_seconds": statistics.median(timings),
        "max_inference_seconds": max(timings),
    }

    report = {
        "label": args.label,
        "model": model_id,
        "lang_pair": f"en-{args.lang}",
        "device": device,
        "labse_model": args.labse_model,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference_translation_provenance": (
            "Drafted by Claude as scaffolding, not verified by a native-speaking "
            "clinical translator -- see caveat in baseline_eval.py docstring."
        ),
        "per_sentence": per_sentence,
        "summary": summary,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(f"model:              {model_id}")
    print(f"mean LaBSE cosine:  {summary['mean_labse_cosine']:.3f}")
    print(f"corpus BLEU:        {summary['corpus_bleu']:.1f}")
    print(f"corpus chrF:        {summary['corpus_chrf']:.1f}")
    print(f"mean inference:     {summary['mean_inference_seconds']*1000:.0f} ms/sentence")
    print(f"median inference:   {summary['median_inference_seconds']*1000:.0f} ms/sentence")
    print(f"Report saved to {args.out}")


if __name__ == "__main__":
    main()
