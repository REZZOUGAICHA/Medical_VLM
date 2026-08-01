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
| Zenodo Hindi Healthcare | 2,182 rows | Medical (diagnoses/symptoms/treatments) | **Translation method not disclosed on source page** | ✅ Downloaded | https://zenodo.org/records/14599295 |
| HiMed-West | ~116,859 pairs (Q&A/reasoning format) | Medical | **Machine-translated (NLLB + medical lexicon), NOT human** | ⏸️ Held — pending supervisor's answer on whether MT-generated training data is acceptable | https://github.com/FreedomIntelligence/HiMed |
| ILCI Health (TDIL) | ~25,000 pairs | Health-specific, clean domain label | Human-translated | ⏳ Pending TDIL portal approval | https://tdil-dc.in (search "Hindi-English Health Text Corpus-ILCI") |
| EILMT Health (TDIL) | ~14,984 pairs | Health-specific, clean domain label | Human-translated | ⏳ Pending TDIL portal approval | https://tdil-dc.in/index.php?option=com_download&task=showresourceDetails&toolid=1786&lang=en |

**Assessment**: Hindi now has substantial real data across both general and
medical domains — this is no longer the blocker it was earlier in the
project. **Enough exists right now to start baseline testing and a first
fine-tuning pass without waiting for TDIL.** ILCI/EILMT, if approved, add a
meaningful volume of clean human-translated health data on top of what's
already in place, but training should not be blocked on their approval
given the 7-day deadline.

**Open decision for the supervisor**: whether HiMed-West's machine-translated
data is acceptable to include. If yes, it roughly doubles available medical
training volume; if no, proceed with the smaller human-sourced set
(TICO-19 + Lokmat + Zenodo + ILCI/EILMT once approved).

**Two preprocessing issues to fix before training on the newest sources**:
1. `bpcc_ilci/hin_Deva.tsv` has metadata columns before the actual text —
   a generic `src`/`tgt` column reader will grab the wrong columns. Needs a
   small dedicated parser, similar to the one already written for TICO-19.
2. The Zenodo healthcare CSVs initially looked row-mismatched via a plain
   `wc -l` line count (2184 vs 2183) — this was a false alarm caused by
   embedded newlines inside quoted CSV fields. Both files are actually 2,182
   rows with matching `patient_id` order and are correctly aligned. Don't
   re-flag this without checking with a real CSV parser (not `wc -l`).

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
instructions), covering: loading each pretrained base model, running it on
the pilot medical sentence set (including negation/uncertainty
phrasings), scoring with LaBSE (primary) and sacreBLEU (secondary), and
timing inference per-sentence rather than batched.

## Suggested next steps, in order

1. **Baseline test both languages now** — highest priority given the
   deadline and that teammates have already delivered this.
2. **Get the supervisor's answer on HiMed-West** (machine-translated data
   question) — doesn't block starting, but affects how much Hindi medical
   data ends up in the final training pool.
3. **Fine-tune Mandarin** directly on NEJM-enzh (no two-stage approach
   needed, dataset is large enough and already medical-domain).
4. **Fine-tune Hindi** using a two-stage approach: Stage 1 on general data
   (IIT Bombay + BPCC), Stage 2 fine-tune on pooled medical data (TICO-19 +
   Lokmat + Zenodo, plus HiMed-West and ILCI/EILMT if/when approved).
5. **Report results back to the team** in the same LaBSE/BLEU +
   before/after fine-tuning format teammates have already used, so results
   are directly comparable across all languages for the paper.
