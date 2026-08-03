# Local Tiny MT: Hindi & Mandarin (medical domain)

Implements the Hindi/Mandarin portion of [`docs/spec.md`](docs/spec.md):
fine-tuning small pretrained MarianMT checkpoints per direction on medical
parallel text, following the pattern teammate 2 already used for
ELRC-Medical (EN-DE) and PEACH (EN-AR). See [`docs/process_guide.md`](docs/process_guide.md)
for the full task breakdown and reasoning behind each step.

**Spanish has been dropped from this project** — Nadjiba confirmed she's
covering it (along with French), so all Spanish-specific data, configs, and
docs have been removed to avoid two people training the same pair.

**Deadline**: 7 days to have models ready, 10 days after that for training,
before paper writing starts.

**Status**: baseline testing is done for both languages (see
`results/baseline_en-{zh,hi}.json`). Mandarin fine-tuning is running now.
Hindi's data pipeline is fixed and both stage pools are built; Hindi
training configs/kickoff are next.

## The mental model: two different jobs a dataset can do

- **General-domain data** teaches a model basic grammar, vocabulary, and
  sentence structure in the target language. Large, but not medical.
- **Domain-specific (medical) data** teaches medical vocabulary and
  phrasing, but there's much less of it.

**Mandarin** has enough medical-domain data alone (NEJM-enzh, 66k pairs) to
skip straight to a **single-stage** fine-tune directly on medical text.

**Hindi** does not — its medical sources total only a few thousand pairs,
dwarfed by ~1.58M general-domain pairs (IITB + BPCC). Hindi needs a
**two-stage** fine-tune: adapt to Hindi generally first, then specialize on
pooled medical data. Same technique the NEJM-enzh/ParaMed paper and
IndicTrans2 both use for low-resource domain-specific MT.

## Directions covered

| Direction | Base checkpoint | Config | Status |
|---|---|---|---|
| EN → ZH | `Helsinki-NLP/opus-mt-en-zh` | [configs/en-zh.yaml](configs/en-zh.yaml) | **fine-tuning now** (single-stage) |
| ZH → EN | `Helsinki-NLP/opus-mt-zh-en` | [configs/zh-en.yaml](configs/zh-en.yaml) | ready to fine-tune (single-stage), not started |
| EN → HI | `Helsinki-NLP/opus-mt-en-hi` | not written yet | data pipeline ready (both stage pools built); configs next |
| HI → EN | `Helsinki-NLP/opus-mt-hi-en` | not written yet | data pipeline ready (both stage pools built); configs next |

## Setup

```
pip install -r requirements.txt
```

CPU works for inference (baseline eval ran fine at ~1-1.4s/sentence). For
training, CPU throughput needs to actually be measured before trusting an
ETA — see the fine-tuning section below for why.

## Data inventory (current, real state — not a plan)

```
data/raw/en-zh/nejm_enzh/         64,163 pairs (62,127 train / 2,036 dev), medical, human -- test split moved out, see below
data/raw/en-zh/tico19/            971 pairs, medical (COVID), human -- dev split, training pool
data/raw/zh-en/nejm_enzh/         empty -- mirror en-zh/nejm_enzh/ here if/when zh-en is trained

data/raw/en-hi/iitb/              1,656,008 pairs, general, extracted from HF Arrow (all 3 of IITB's own splits merged)
data/raw/en-hi/bpcc_ilci/         165,649 pairs, general, extracted from hin_Deva.tsv (raw tsv archived at data/external_raw/bpcc_ilci/)
data/raw/en-hi/lokmat_healthcare/ 2,202 pairs, healthcare news, human
data/raw/en-hi/tico19/            970 pairs, medical (COVID), human -- dev split, training pool
data/raw/en-hi/zenodo_healthcare/ 1,022 pairs, medical, extracted from 7 free-text fields per patient record, see below

data/eval/tico19_zh/              2,100 pairs, held out (test split, never trained on)
data/eval/tico19_hi/              2,100 pairs, held out (test split, never trained on)
data/eval/nejm_zh/                2,102 pairs, held out (NEJM's own test split, moved here -- see "test-split leakage" below)
data/raw/eval/                    WMT18 zh-en -- not stored, fetched on demand by sacrebleu (see "Evaluate")

data/processed/en-zh/             pooled train/val, ready, used by the running fine-tune
data/processed/en-hi-general/     Stage 1 pool: iitb + bpcc_ilci (1,549,415 train / 31,620 val)
data/processed/en-hi-medical/     Stage 2 pool: tico19 + lokmat_healthcare + zenodo_healthcare (3,985 train / 209 val)
```

**HiMed-West is excluded, not pending** — verified against the actual JSON
(116,859 records total, matching the number originally reported) and the
published paper's data table: every record is already-Hindi text
(`prompt`/`ground_truth`, machine-translated, English terms kept only as
parenthetical glosses) plus a `source` field naming the *original English
benchmark* (MedMCQA, MedReason, GPQA-med, ...), not the English text
itself. There's no real English/Hindi sentence pair to extract here, and no
join key to reliably reconstruct one. Same issue confirmed in the
Bench/Exam/Trad files in the same repo. This was a structural finding, not
the pending approval question it started as — the supervisor separately
approved MT-derived data in principle, but that's moot here since there's
no parallel data to extract in the first place.

**IndicMedDialog** (github.com/ShubhamKumarNigam/IndicMedDialog) is a
genuinely parallel EN-HI resource found afterward — noted as an optional,
lower-priority future addition, not integrated yet. Real caveats before
using it: synthetic (LLM-generated) dialogues, not real patient
conversations; **CC BY-NC-ND 4.0** license (No-Derivatives clause needs a
check before this project reformats/pools it); dialogue-turn format needs
its own extraction into utterance pairs; small once extracted (~1,500-1,700
pairs).

**Not yet available**: ILCI Health (~25k) and EILMT Health (~15k), both
pending approval on India's TDIL portal. Not a blocker.

### Extractors for non-conforming source formats

`data_prep.py`'s generic reader handles two shapes: aligned `<name>.en` +
`<name>.hi` files, or a 2-column `.tsv`/`.csv`. Sources that don't fit that
shape get a small dedicated extractor (same pattern each time — read the
real format, write a clean aligned pair, leave the original untouched):

| Source | Problem | Extractor |
|---|---|---|
| TICO-19 | 8-column schema (`sourceLang, targetLang, sourceString, targetString, ...`) | `src/extract_tico19.py` |
| BPCC/ILCI (`hin_Deva.tsv`) | 4-column schema (`src_lang, tgt_lang, src, tgt`) — generic reader would grab the language-code columns instead of the sentences | `src/extract_bpcc_ilci.py` |
| IITB | Hugging Face `datasets.load_from_disk()` Arrow directory, not text/tsv at all | `src/extract_iitb.py` |
| Zenodo healthcare | 13-column structured patient records, no single "the sentence" field | `src/extract_zenodo_healthcare.py` |

```
python src/extract_tico19.py --tico-dir data/tico19-testset/tico19-testset --lang zh --dev-out data/raw/en-zh/tico19 --test-out data/eval/tico19_zh
python src/extract_tico19.py --tico-dir data/tico19-testset/tico19-testset --lang hi --dev-out data/raw/en-hi/tico19 --test-out data/eval/tico19_hi
python src/extract_bpcc_ilci.py --tsv data/external_raw/bpcc_ilci/hin_Deva.tsv --out-dir data/raw/en-hi/bpcc_ilci
python src/extract_iitb.py --iitb-dir data/raw/en-hi/iitb --out-dir data/raw/en-hi/iitb
python src/extract_zenodo_healthcare.py --en-csv data/external_raw/zenodo_healthcare/English_dataset.csv --hi-csv data/external_raw/zenodo_healthcare/hindi_dataset.csv --out-dir data/raw/en-hi/zenodo_healthcare
```

The other ~30 languages in `data/tico19-testset/` are left untouched — not
copied anywhere, candidates for deletion later if disk space matters.

### `zenodo_healthcare` extraction

`{hindi,English}_dataset.csv` are structured synthetic patient records (13
columns), not sentence pairs — pooling them as-is would make the generic
reader fall back to columns 0/1 (`patient_id`, `age`) as if they were the
sentence. Decision made: rather than concatenate fields into one artificial
per-patient blob, `src/extract_zenodo_healthcare.py` emits each of 7
free-text fields (`Diagnosis`, `Remarks`, `Patient History`, `symptoms`,
`treatment`, `timespan`, `Diagnosis Category`) as its own row when
non-empty on both sides (all were, 2,182/2,182) — keeps each row a natural
clinical phrase instead of a long unnatural composite. Row alignment is
verified by comparing `patient_id` order between the two files before
extracting anything. Yields 15,274 raw pairs -> **1,022 after exact-dedup**
(most of the raw volume is templated/repeated categorical text across
synthetic patients, e.g. the same `Diagnosis Category` string recurring).
Raw CSVs archived (untouched) at `data/external_raw/zenodo_healthcare/`.

```
python src/extract_zenodo_healthcare.py --en-csv data/external_raw/zenodo_healthcare/English_dataset.csv --hi-csv data/external_raw/zenodo_healthcare/hindi_dataset.csv --out-dir data/raw/en-hi/zenodo_healthcare
```

(Both files are correctly row-aligned by `patient_id` — verified with a
real CSV parser; a naive `wc -l` shows 2184 vs 2183 lines, which is a false
alarm from embedded newlines inside quoted fields, not a real mismatch.)

## 1. Pool + split into train/val

```
python src/data_prep.py --lang-pair en-zh --raw-dir data/raw/en-zh --out-dir data/processed/en-zh
python src/data_prep.py --lang-pair zh-en --raw-dir data/raw/zh-en --out-dir data/processed/zh-en

# Hindi: --sources restricts pooling to a named subset, so general and medical
# stay separate even though they're sibling subfolders under the same raw-dir.
python src/data_prep.py --lang-pair en-hi --raw-dir data/raw/en-hi --sources iitb,bpcc_ilci --out-dir data/processed/en-hi-general
python src/data_prep.py --lang-pair en-hi --raw-dir data/raw/en-hi --sources tico19,lokmat_healthcare --out-dir data/processed/en-hi-medical --val-ratio 0.05
```

Every pooling run also writes `{train,val}.source` (line-aligned with
`{train,val}.<lang>`) recording which source subfolder each row came from —
this is what makes a later ablation (e.g. "does adding X help or hurt")
possible without re-extracting anything, and is required now for any future
machine-translated source, not just HiMed.

**Test-split leakage, caught and fixed**: `nejm_enzh/` originally held
train/dev/**test** together in one pooled folder. Pooling it as-is would
have swept the exact 2,102 held-out sentences `baseline_eval.py` scores
against into the training set, silently invalidating any before/after
fine-tuning comparison. Fixed by moving `nejm.test.{en,zh}` to
`data/eval/nejm_zh/` (mirroring `eval/tico19_zh`) and updating
`baseline_eval.py`'s path to match. Worth checking any future source the
same way before pooling it.

## 2. Fine-tune

```
python src/train.py --config configs/en-zh.yaml   # running now
python src/train.py --config configs/zh-en.yaml   # not started
```

Each run loads the pretrained checkpoint, fine-tunes with `Seq2SeqTrainer`
(early stopping on validation BLEU, fp16 automatically if a GPU is
present), and saves the final model to `models/<name>-medical/final/`
(checkpoints also saved every epoch under `--output_dir`).

**CPU throughput, confirmed on a clean machine**: ~10-13s/step, steady
across 16+ real training steps (an earlier attempt showed ~30-60s/step, but
that coincided with a large IITB extraction job running at the same time
and was a contention artifact, not the model's real speed -- stopped
before any checkpoint was written and restarted cleanly). At this rate the
full Mandarin run (11,970 steps) is roughly **36-40 hours**.

Hindi's two-stage fine-tune (Stage 1 on `data/processed/en-hi-general/`,
Stage 2 starting from Stage 1's output checkpoint on
`data/processed/en-hi-medical/`) needs its own configs, not written yet --
and Stage 1 specifically needs a scale decision first, see "Known gaps"
below: at the same per-step rate, Stage 1's 1,549,415 pairs would take
**12+ days for a single epoch**, on its own longer than the likely training
budget.

## 3. Baseline testing (done for both languages)

`src/baseline_eval.py` loads each pretrained base checkpoint with no
fine-tuning, and scores it against **real held-out human-translated data
already in this project** — `eval/nejm_zh/` for Mandarin, `eval/tico19_hi/`
for Hindi — never invented sentences or reference translations (an earlier
draft of this script did that and it was corrected before running; see
[docs/process_guide.md](docs/process_guide.md)'s "Standing rule: never
generate ground truth"). It separately flags which of the pilot CheXpert
labels (cardiomegaly, pleural effusion, pneumothorax, edema, "no acute
cardiopulmonary abnormality", support devices, atelectasis, consolidation)
actually turn up in that real data via substring search, and reports 0
honestly for the ones that don't. It also de-tokenizes NEJM-enzh's raw
Moses-style formatting (`@-@` hyphen-splitting, spaced punctuation,
word-segmented Chinese) before scoring, since otherwise the model would be
penalized for a storage artifact, not real translation quality.

```
python src/baseline_eval.py --lang zh --model Helsinki-NLP/opus-mt-en-zh --label baseline --out results/baseline_en-zh.json
python src/baseline_eval.py --lang hi --model Helsinki-NLP/opus-mt-en-hi --label baseline --out results/baseline_en-hi.json
```

**Results** (CPU, real held-out data):

| | n | mean LaBSE | corpus BLEU | corpus chrF | median ms/sentence |
|---|---|---|---|---|---|
| EN-ZH | 212 | 0.872 | 10.7 | 28.3 | 1177 |
| EN-HI | 209 | 0.807 | 13.5 | 34.4 | 1030 |

Concrete failures worth citing: the Mandarin model transliterated
"atelectasis" phonetically (阿亚特西) instead of translating it (肺不张 is
correct); the Hindi model's LaBSE score dropped to 0.645 specifically on
the pilot-term-matched (dense pathology) sentences, e.g. turning
"Macroscopy: pleurisy, pericarditis, lung consolidation and pulmonary
oedema" into a near-nonsensical Hindi hallucination. Both are real evidence
for why medical-domain fine-tuning should help, not guesses.

After fine-tuning, re-run the same command against the fine-tuned
checkpoint (`--model models/opus-mt-en-zh-medical/final`) for a directly
comparable before/after delta.

## 4. Evaluate a fine-tuned model against held-out test sets

WMT18's news task includes `zh-en`/`en-zh` (not `hi`/`es`), and `sacrebleu`
fetches it with no manual download:

```
sacrebleu -t wmt18 -l zh-en --echo src > data/raw/eval/wmt18.zh
sacrebleu -t wmt18 -l zh-en --echo ref > data/raw/eval/wmt18.en

python src/evaluate.py \
  --model-dir models/opus-mt-zh-en-medical/final \
  --test-src data/raw/eval/wmt18.zh \
  --test-tgt data/raw/eval/wmt18.en
```

For Hindi, use the TICO-19 held-out test split instead:

```
python src/evaluate.py \
  --model-dir models/opus-mt-en-hi-medical/final \
  --test-src data/eval/tico19_hi/tico19.en \
  --test-tgt data/eval/tico19_hi/tico19.hi
```

Reports corpus BLEU and chrF (via `sacrebleu`). Pass `--out-predictions` to
also dump raw translations for manual inspection.

## Known gaps to flag back to the team

- **Hindi Stage 1 is ~24x bigger than the entire Mandarin pool (1,549,415
  vs 63,831 train pairs), and at Mandarin's confirmed clean-machine rate
  (~11s/step) that scales to roughly 12+ days for a single epoch of Stage
  1 alone** on this CPU-only machine — that's longer than the whole
  training budget by itself, before Stage 2 or a second direction. This
  needs a decision before Hindi training is kicked off: subsample Stage 1
  to something CPU-feasible, cap it with `max_steps` instead of a full
  epoch, or find GPU compute. Flagging this now, not discovering it
  mid-run.
- ILCI Health / EILMT Health are pending TDIL portal approval — not
  blocking, just not in the pool yet.
- Hindi training configs (Stage 1 + Stage 2, both directions) aren't
  written yet — blocked on the scale decision above.
- IndicMedDialog's CC BY-NC-ND license (specifically the No-Derivatives
  clause) needs a quick check before it's ever integrated.
