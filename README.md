# Local Tiny MT: Hindi & Mandarin (medical domain)

Implements the Hindi/Mandarin portion of [`docs/spec.md`](docs/spec.md):
fine-tuning small pretrained MarianMT checkpoints per direction on medical
parallel text, following the pattern teammate 2 already used for
ELRC-Medical (EN-DE) and PEACH (EN-AR). See [`docs/process_guide.md`](docs/process_guide.md)
for the full task breakdown and reasoning behind each step.

**Spanish has been dropped from this project** — Nadjiba confirmed she's
covering it (along with French), so all Spanish-specific data, configs, and
docs have been removed to avoid two people training the same pair. See
[`docs/spec_v1_superseded_spanish_mandarin_hindi.md`](docs/spec_v1_superseded_spanish_mandarin_hindi.md)
for the old three-language version, kept for history only — it is not
current.

**Deadline**: 7 days to have models ready, 10 days after that for training,
before paper writing starts.

**Current priority is baseline testing, not more data work or fine-tuning**
— both other teammates have already reported real before/after numbers
(LaBSE similarity, inference timing), and Hindi/Mandarin haven't had this
done yet. Dataset collection for both languages is already sufficient. See
[src/baseline_eval.py](src/baseline_eval.py) and the "Baseline testing"
section below.

## The mental model: two different jobs a dataset can do

- **General-domain data** teaches a model basic grammar, vocabulary, and
  sentence structure in the target language. Large, but not medical.
- **Domain-specific (medical) data** teaches medical vocabulary and
  phrasing, but there's much less of it.

**Mandarin** has enough medical-domain data alone (NEJM-enzh, 66k pairs) to
skip straight to a **single-stage** fine-tune directly on medical text.

**Hindi** does not — its medical sources (TICO-19, Lokmat, Zenodo) total
only ~5k pairs, dwarfed by the ~1.66M general-domain pairs available
(IITB + BPCC). Hindi needs a **two-stage** fine-tune instead: first adapt
the base model to Hindi generally on the general-domain pool, then take
that checkpoint and fine-tune it again on just the pooled medical data to
specialize it. This is the same technique the NEJM-enzh/ParaMed paper and
IndicTrans2 both use for low-resource domain-specific MT — not something
invented for this project.

**Known follow-up before Hindi fine-tuning (Task 4, not done yet)**:
`data/raw/en-hi/` currently holds both roles side by side (`iitb/`,
`bpcc_ilci/` = general; `tico19/`, `lokmat_healthcare/`,
`zenodo_healthcare/` = medical). `data_prep.py --raw-dir data/raw/en-hi`
pools *everything* in that directory together, which would incorrectly mix
the two stages. Before Stage 1 training starts, `data_prep.py` needs a
`--sources` filter (or similar) so each stage can be pooled independently —
flagging this now so it isn't a surprise later, not fixing it yet since
baseline testing doesn't need it.

## Directions covered

| Direction | Base checkpoint | Config | Status |
|---|---|---|---|
| EN → ZH | `Helsinki-NLP/opus-mt-en-zh` | [configs/en-zh.yaml](configs/en-zh.yaml) | ready to fine-tune (single-stage) |
| ZH → EN | `Helsinki-NLP/opus-mt-zh-en` | [configs/zh-en.yaml](configs/zh-en.yaml) | ready to fine-tune (single-stage) |
| EN → HI | `Helsinki-NLP/opus-mt-en-hi` | not written yet | data ready; needs two-stage configs (Task 4) |
| HI → EN | `Helsinki-NLP/opus-mt-hi-en` | not written yet | data ready; needs two-stage configs (Task 4) |

## Setup

```
pip install -r requirements.txt
```

Needs a GPU with a few GB of VRAM to fine-tune (or run baseline eval) at a
reasonable speed; scripts will run on CPU too (much slower) if none is
available.

## Data inventory (current, real state — not a plan)

```
data/raw/en-zh/nejm_enzh/         66,265 pairs (62,127 train / 2,036 dev / 2,102 test), medical, human
data/raw/en-zh/tico19/            971 pairs, medical (COVID), human -- dev split, training pool
data/raw/zh-en/nejm_enzh/         empty -- mirror en-zh/nejm_enzh/ here if/when zh-en is trained

data/raw/en-hi/iitb/              1,662,110 pairs (1,659,083 train / 520 val / 2,507 test), general, HF Arrow format
data/raw/en-hi/bpcc_ilci/         hin_Deva.tsv, ~165k rows, general (ILCI+NLLB-Seed mixed, not separable) -- needs a parser fix, see below
data/raw/en-hi/lokmat_healthcare/ 2,202 pairs, healthcare news, human
data/raw/en-hi/tico19/            971 pairs, medical (COVID), human -- dev split, training pool
data/raw/en-hi/zenodo_healthcare/ 2,182 rows, medical (diagnoses/symptoms/treatment), translation method undisclosed

data/eval/tico19_zh/              2,100 pairs, held out (test split, never trained on)
data/eval/tico19_hi/              2,100 pairs, held out (test split, never trained on)
data/raw/eval/                    WMT18 zh-en -- not stored, fetched on demand by sacrebleu (see "Evaluate")
```

**Held pending supervisor approval, not in the pool above**: HiMed-West
(~116,859 pairs) — its Hindi side is machine-translated (NLLB + medical
lexicon), not human, and training a translation model on another model's
machine translations risks inheriting that model's specific errors rather
than learning from ground truth. The team is waiting on an explicit answer
on whether MT-generated training data is acceptable before this is added.

**Not yet available**: ILCI Health (~25k) and EILMT Health (~15k), both
pending approval on India's TDIL portal. Not a blocker — what's already in
place is enough to start.

### TICO-19 extraction (en-zh, en-hi)

`data/tico19-testset/tico19-testset/{dev,test}/*.tsv` contains ~30
languages in an 8-column schema (`sourceLang, targetLang, sourceString,
targetString, stringID, url, license, translator_ID`) that isn't directly
usable by `data_prep.py`'s generic reader. `src/extract_tico19.py` pulls
just the `sourceString`/`targetString` columns for one language pair,
writing the dev split to the training-pool folder and the test split to
the matching `data/eval/` folder (held out, never used in training):

```
python src/extract_tico19.py --tico-dir data/tico19-testset/tico19-testset --lang zh --dev-out data/raw/en-zh/tico19 --test-out data/eval/tico19_zh
python src/extract_tico19.py --tico-dir data/tico19-testset/tico19-testset --lang hi --dev-out data/raw/en-hi/tico19 --test-out data/eval/tico19_hi
```

The other ~30 languages in `data/tico19-testset/` are left untouched (not
copied anywhere) — candidates for deletion later if disk space matters, but
nothing currently depends on removing them.

### Two known preprocessing issues on the newest Hindi sources

- **`bpcc_ilci/hin_Deva.tsv`**: real parallel data (`src_lang, tgt_lang,
  src, tgt` columns, ~165k rows), but the header names don't match
  `data_prep.py`'s "en"/"hi" column-matching, so it would currently fall
  back to the wrong columns (the literal `eng_Latn`/`hin_Deva` language-code
  strings, not the sentences). Needs a small dedicated parser, the same
  pattern as `extract_tico19.py` — not written yet, not needed for baseline
  testing.
- **`zenodo_healthcare/{hindi,English}_dataset.csv`**: structured synthetic
  patient records (13 columns), not sentence pairs — using them for MT
  training means deciding which fields to concatenate into a "sentence."
  Naive `wc -l` reports different line counts (2184 vs 2183) because some
  fields contain embedded newlines inside quoted CSV text; parsed properly,
  both files have 2,182 rows with identical `patient_id` order, so they
  **are** correctly aligned — just don't use `wc -l` to check this again.

`src/data_prep.py` accepts either format per source folder for datasets
that don't need a dedicated parser:

- **Aligned monolingual files**: `<name>.en` + `<name>.hi` (or `.zh`), line
  N of one corresponds to line N of the other (standard OPUS/Moses export
  format).
- **Tabular**: a `.tsv`/`.csv` with two columns for source/target text. If
  there's a header matching the language codes it's used to locate the
  columns; otherwise the first two columns are used positionally.

## 1. Pool + split into train/val

```
python src/data_prep.py --lang-pair en-zh --raw-dir data/raw/en-zh --out-dir data/processed/en-zh
python src/data_prep.py --lang-pair zh-en --raw-dir data/raw/zh-en --out-dir data/processed/zh-en
```

This pools every source subfolder for that direction, drops
exact-duplicate and over-length pairs, shuffles with a fixed seed, and
writes `{train,val}.<lang>` files under `--out-dir` (2% val split by
default, see `--val-ratio`).

Not run for `en-hi`/`hi-en` yet — see the two-stage note above; pooling
`data/raw/en-hi/` as-is today would mix general and medical data
incorrectly.

## 2. Fine-tune (Mandarin ready now; Hindi is Task 4, not started)

```
python src/train.py --config configs/en-zh.yaml
python src/train.py --config configs/zh-en.yaml
```

Each run loads the pretrained checkpoint, fine-tunes with
`Seq2SeqTrainer` (early stopping on validation BLEU, fp16 automatically if
a GPU is present), and saves the final model to
`models/<name>-medical/final/`.

## 3. Baseline testing (do this before fine-tuning — current priority)

`src/baseline_eval.py` loads each pretrained base checkpoint
(`opus-mt-en-zh`, `opus-mt-en-hi`) with no fine-tuning, runs it on a pilot
set of medical sentences (including negated/uncertain phrasings), and
scores the output two ways:

- **LaBSE** (primary, matches the team standard) — embeds the model's
  output and a reference translation with `sentence-transformers/LaBSE`
  and reports cosine similarity between the two meaning-vectors. This
  catches cases where a translation is correct but phrased differently
  than the reference — exactly what happens often with negation/hedging in
  clinical text, and exactly what BLEU (which just counts matching words)
  misses.
- **sacreBLEU** (secondary, for comparability with anything already
  reported using it).

It also times inference **per sentence**, not batched, since that reflects
what a real single request feels like rather than batch throughput.

```
python src/baseline_eval.py --lang zh --model Helsinki-NLP/opus-mt-en-zh --out results/baseline_en-zh.json
python src/baseline_eval.py --lang hi --model Helsinki-NLP/opus-mt-en-hi --out results/baseline_en-hi.json
```

After fine-tuning, re-run the same command against the fine-tuned
checkpoint (`--model models/opus-mt-en-zh-medical/final`) to get a directly
comparable before/after delta, in the same format teammates already
reported for Arabic/German.

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

- HiMed-West's machine-translated-data question is still open with the
  supervisor — affects final Hindi medical training volume either way.
- ILCI Health / EILMT Health are pending TDIL portal approval — not
  blocking, just not in the pool yet.
- `bpcc_ilci/hin_Deva.tsv` needs a dedicated parser before Stage 1 Hindi
  training (see above) — not written yet.
- `data_prep.py` needs a way to pool a subset of source folders (not the
  whole raw-dir) before Hindi's two-stage split can actually be run —
  not written yet.
