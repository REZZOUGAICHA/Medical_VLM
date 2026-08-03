# Local Tiny MT Models for Hindi and Mandarin — Datasets & Plan

## Context and change of direction

The supervisor clarified the project must use **local, tiny, fine-tuned
models**, not API calls to hosted LLMs. Two teammates are already fine-tuning
MarianMT/ByT5 checkpoints per language: one is covering French + Spanish,
another Arabic + German. **Mandarin and Hindi are this document's scope.**
(Spanish was originally picked up too but has since been dropped — Nadjiba
confirmed she's covering it, so all Spanish content and files have been
removed from this document and from the project structure.)

**Deadline** (per team discussion): 7 days to have models ready, 10 days
after that for training, before paper writing starts. Both other teammates
have already completed baseline testing and reported real numbers (LaBSE
similarity scores, inference timing, before/after fine-tuning comparisons).
**Baseline testing for Hindi and Mandarin has not been done yet and is now
the most urgent open item** — not further dataset collection, which is
already more than sufficient for both languages.

## Base models

- `Helsinki-NLP/opus-mt-en-zh` / `opus-mt-zh-en` — Mandarin
- `Helsinki-NLP/opus-mt-en-hi` / `opus-mt-hi-en` — Hindi

Same OPUS-MT/MarianMT family teammates are already using for German/Arabic,
so the fine-tuning pipeline is directly reusable.
Sources: https://huggingface.co/Helsinki-NLP/opus-mt-en-zh ,
https://huggingface.co/Helsinki-NLP/opus-mt-hi-en

**Alternative worth testing for Hindi**: `ai4bharat/indictrans2`,
purpose-built for Indian languages, MIT-licensed, reported to outperform
general-purpose baselines on Indic translation.
Source: https://www.emergentmind.com/topics/indictrans2

**Metric note**: teammates have standardized on **LaBSE** (semantic
similarity) alongside sacreBLEU for evaluation, since it better captures
cases where two translations are semantically equivalent but lexically very
different (e.g. a French/English example Nadjiba shared: "Le patient a une
anomalie dans le côté gauche de la poitrine" vs "The patient has maybe a
left chest infection which isn't normal" — clinically equivalent, almost no
shared tokens). Use the same metric for Hindi/Mandarin so results are
comparable across the whole team for the eventual paper.

**LaBSE vs BLEU, briefly**: BLEU scores translation quality by counting
matching words/short phrases (n-grams) against a reference — it's fast and
standard, but penalizes correct translations that just use different
wording than the reference. LaBSE (Language-agnostic BERT Sentence
Embedding) instead converts both the model's output and the reference into
meaning-vectors in a shared cross-lingual space, then scores similarity
between those vectors — so two sentences with almost no shared words but
the same meaning score well. This matters specifically for clinical text,
where negation, hedging, and synonyms frequently produce very different
wording for the same clinical meaning, which is exactly the case BLEU
handles poorly and LaBSE is built for.

## Mandarin (EN-ZH) — dataset status

| Dataset | Size | Domain | Quality | Status | Link |
|---|---|---|---|---|---|
| NEJM-enzh / ParaMed | 66,265 pairs (62,127 train / 2,036 dev / 2,102 test) | Medical | Human (professional NEJM translations) | ✅ Downloaded, verified | https://raw.githubusercontent.com/boxiangliu/med_translation/master/data/nejm-open-access.tar.gz |
| WMT18 (zh-en) | — | General | Human | ✅ Auto-fetched via `sacrebleu`, no manual download | n/a |
| TICO-19 (en-zh) | 3,071 pairs total (971 dev / 2,100 test after extraction) | Medical (COVID) | Human-translated | ✅ Downloaded, extracted | https://github.com/tico-19/tico-19.github.io |

**Assessment**: Mandarin is fully covered. NEJM-enzh alone is large enough
(66k) to be both base and domain training data — no two-stage approach
needed, unlike Hindi. WMT18 and TICO-19 provide two independent,
non-overlapping evaluation sets. **Nothing further to source for Mandarin —
proceed straight to baseline testing and fine-tuning.**

Known content note: text is pre-tokenized (English has `@-@` hyphen
splitting, Chinese is word-segmented with spaces) — account for this in
preprocessing, either de-tokenize or keep consistent with how the base
model expects input.

## Hindi (EN-HI) — dataset status

| Dataset | Size | Domain | Quality | Status | Link |
|---|---|---|---|---|---|
| IIT Bombay | ~1.66M pairs | General (verified: no health/medical component in its 17 source corpora) | Human/compiled | ✅ Downloaded | https://huggingface.co/datasets/cfilt/iitb-english-hindi |
| TICO-19 (en-hi) | 3,071 pairs total (971 dev / 2,100 test) | Medical (COVID) | Human-translated | ✅ Downloaded, extracted | https://github.com/tico-19/tico-19.github.io |
| Anuvaad / Lokmat Healthcare | 2,202 pairs | Healthcare news | Human (professional news outlet) | ✅ Downloaded | https://anuvaad-parallel-corpus.s3-us-west-2.amazonaws.com/lokmat-healthcare_20210501_en_hi.zip |
| BPCC (`ilci/hin_Deva.tsv`) | 62.6 MB | General — **NOT domain-separable**, ILCI merged with NLLB-Seed data with no distinguishing column | Mixed | ✅ Downloaded (needs preprocessing fix — see below) | https://huggingface.co/datasets/ai4bharat/BPCC/resolve/main/ilci/hin_Deva.tsv |
| Zenodo Hindi Healthcare | 2,182 records -> 1,022 usable sentence pairs after extraction | Medical (diagnoses/symptoms/treatments) | **Translation method not disclosed on source page** | ✅ Downloaded, extracted | https://zenodo.org/records/14599295 |
| HiMed-West | ~116,859 records | N/A — **excluded, not a parallel corpus** | N/A | ❌ Dropped, see below | https://github.com/FreedomIntelligence/HiMed |
| IndicMedDialog | ~298 dialogues (~1,500–1,700 utterance pairs once extracted) | Medical dialogue (synthetic, LLM-generated then translated) | Human/native-speaker-verified translation (9.75/10 reported quality) | 💡 Optional lead, not integrated — see below | https://github.com/ShubhamKumarNigam/IndicMedDialog |
| ILCI Health (TDIL) | ~25,000 pairs | Health-specific, clean domain label | Human-translated | ⏳ Pending TDIL portal approval | https://tdil-dc.in (search "Hindi-English Health Text Corpus-ILCI") |
| EILMT Health (TDIL) | ~14,984 pairs | Health-specific, clean domain label | Human-translated | ⏳ Pending TDIL portal approval | https://tdil-dc.in/index.php?option=com_download&task=showresourceDetails&toolid=1786&lang=en |

**Assessment**: Hindi now has substantial real data across both general and
medical domains — this is no longer the blocker it was earlier in the
project. **Enough exists right now to start baseline testing and a first
fine-tuning pass without waiting for TDIL.** ILCI/EILMT, if approved, add a
meaningful volume of clean human-translated health data on top of what's
already in place, but training should not be blocked on their approval
given the 7-day deadline.

**HiMed-West is excluded, not "pending" — this was resolved and closed.**
The supervisor approved using machine-translated data if it helps, so this
was never a quality/approval question in the end. It's a structural one:
verified against the actual JSON (all 5 shards, 116,859 records total,
matching the number in this doc) and cross-checked against the published
paper's own data table — every record has only `prompt` / `Complex_CoT` /
`ground_truth` / `source`, and `prompt`/`ground_truth` are *already Hindi*
(machine-translated, with English medical terms kept in parentheses, e.g.
"वृद्धि हार्मोन (Growth hormone)"). `source` names the original English
benchmark the question came from (MedMCQA, MedReason, GPQA-med, ...) as a
plain string label, not the English text itself, and the West split has no
ID field to join back to those upstream English datasets even if fetched
separately. English only ever survives as inline parenthetical glosses
inside Hindi text, never as full parallel sentences — so there is no real
English/Hindi sentence pair to extract here, per the standing "never invent
ground truth" rule. This applies to the Bench/Exam/Trad files in the same
repo too (checked, same issue).

**IndicMedDialog is a genuinely parallel EN-HI resource, found afterward,
kept as optional/lower-priority** — not something to build around or wait
for given the deadline. Real caveats to weigh before integrating it later:
synthetic dialogues (LLM-generated, then translated — not real patient
conversations, and the authors flag this themselves); licensed
**CC BY-NC-ND 4.0** (non-commercial is fine here, but the No-Derivatives
clause needs a quick check before this project reformats/redistributes it
in any pooled/processed form); dialogue-turn formatted, not sentence pairs,
so it needs its own extraction step; and small once extracted (~1,500–1,700
utterance pairs) — not a major volume boost even if used.

**Preprocessing issues on the Hindi sources — status**:
1. `bpcc_ilci/hin_Deva.tsv` had metadata columns before the actual text —
   **fixed**: `src/extract_bpcc_ilci.py` pulls the real 165,649 usable pairs
   into a clean aligned pair; the raw tsv is archived (untouched) at
   `data/external_raw/bpcc_ilci/hin_Deva.tsv`.
2. `iitb/` is stored as a Hugging Face `datasets.load_from_disk()` Arrow
   directory, which `data_prep.py`'s generic reader never supported —
   **fixed**: `src/extract_iitb.py` merges all three of IITB's own splits
   (train/validation/test — nothing in this project holds IITB's test split
   out for anything, so re-pooling it as general-domain input is fine) into
   one clean aligned pair.
3. The Zenodo healthcare CSVs are structured patient records (13 columns),
   not sentence pairs — **fixed**: `src/extract_zenodo_healthcare.py` pulls
   7 free-text fields (`Diagnosis`, `Remarks`, `Patient History`,
   `symptoms`, `treatment`, `timespan`, `Diagnosis Category`) as their own
   rows rather than concatenating them, per an explicit decision to
   optimize for training-data quality over extraction effort. 15,274 raw
   pairs -> 1,022 after exact-dedup (most of the raw volume is
   templated/repeated text across synthetic patients). They also initially
   looked row-mismatched via a plain `wc -l` line count (2184 vs 2183) —
   false alarm from embedded newlines inside quoted CSV fields; both files
   are actually 2,182 rows with matching `patient_id` order, verified with
   a real CSV parser (not `wc -l`) both before and inside the extractor.
4. `data_prep.py` originally pooled every subfolder under a `--raw-dir`
   indiscriminately, which would have mixed Hindi's general-domain sources
   (iitb, bpcc_ilci) with its medical sources (tico19, lokmat_healthcare,
   zenodo_healthcare) if run directly against `data/raw/en-hi/` —
   **fixed**: added `--sources` to pool only a named subset, plus per-row
   provenance tracking (`train.source`/`val.source`, line-aligned with
   `train.<lang>`/`val.<lang>`) so composition and, later, an ablation by
   source, are both possible without re-extracting anything.
5. **Separately caught while wiring up Mandarin training** (not a Hindi
   issue, but the same class of bug): `nejm_enzh/` originally held
   train/dev/**test** together in one pooled folder, so a naive
   `data_prep.py` run would have swept the held-out test split — the exact
   data `baseline_eval.py` scores against — into the training set, silently
   invalidating any before/after fine-tuning comparison. Fixed by moving
   `nejm.test.{en,zh}` to `data/eval/nejm_zh/` (mirroring the
   `eval/tico19_zh` convention) and updating `baseline_eval.py`'s path to
   match. Worth double-checking any other pooled-with-its-own-test-split
   source the same way before pooling it.

## Baseline testing — the current priority

Both other teammates have already done this and reported real comparative
numbers. This has not yet been done for Hindi or Mandarin and is now the
most time-sensitive item given the 7-day deadline — ahead of any further
data work.

What "done" looks like, matching the team's established format:
1. Load each pretrained base model (`opus-mt-en-zh`, `opus-mt-en-hi`).
2. Run it on real medical sentences (not generic text) — use the pilot
   label set (cardiomegaly, pleural effusion, pneumothorax, edema, "no
   acute cardiopulmonary abnormality", support devices, atelectasis,
   consolidation), including negation and uncertainty phrasings.
3. Score with **LaBSE** (primary, matches team standard) and sacreBLEU
   (secondary), against reference translations.
4. Measure per-sentence inference time individually (not batched), to
   report realistic single-request response time.
5. Fine-tune, re-run the same eval, and report the delta — this is the
   exact comparison teammates already shared (e.g. "+5% improvement and
   15% faster" for Arabic; "barely helped, made it a bit slower" for
   German) — Hindi/Mandarin results need to be in this same comparable
   format for the paper.

No baseline script currently exists in this project — it needs to be
written from scratch (by Claude Code, per the accompanying `SKILL.md`
instructions), covering: loading each pretrained base model, evaluating on
real sentence pairs pulled from existing held-out test data (`nejm_enzh`'s
test split for Mandarin, `eval/tico19_hi/` for Hindi — never invented
reference translations, since that would make the evaluation circular),
scoring with LaBSE (primary) and sacreBLEU (secondary), and timing
inference per-sentence rather than batched.

**Correction, logged for the record**: an earlier draft of this baseline
script used hand-written pilot sentences with reference translations
written by the AI assistant itself, rather than pulled from real data.
This was caught before running the script and corrected — flagged here so
the reasoning isn't lost if the question comes up again later.

## Suggested next steps, in order

1. ✅ **Baseline test both languages** — done, see `results/baseline_en-{zh,hi}.json`.
2. ✅ **Resolved**: supervisor approved MT-derived data in principle; moot
   for HiMed-West specifically since it turned out not to be a parallel
   corpus at all (see Hindi table above). Nothing further to decide here.
3. 🔄 **Fine-tune Mandarin** directly on NEJM-enzh + TICO-19 — running now,
   confirmed ~10-13s/step on a clean CPU-only run, full run ~36-40 hours.
4. **Fine-tune Hindi** using a two-stage approach — data pipeline is ready
   (Stage 1 general pool: 1,549,415 pairs from IITB+BPCC; Stage 2 medical
   pool: 3,985 pairs from TICO-19+Lokmat+Zenodo), but **Stage 1's scale
   needs a decision before kicking it off**: at the confirmed per-step
   rate, one epoch over 1,549,415 pairs is ~12+ days on this CPU-only
   machine — longer than the likely training budget by itself. Options:
   subsample Stage 1, cap training with `max_steps` rather than full
   epochs, or move to GPU compute. Not decided yet.
5. **Report results back to the team** in the same LaBSE/BLEU +
   before/after fine-tuning format teammates have already used, so results
   are directly comparable across all languages for the paper.
