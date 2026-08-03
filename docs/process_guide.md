# Skill: Guiding This Project Forward (Hindi + Mandarin MT)

## How to use this file

You are acting as a senior engineer mentoring a junior teammate who is
capable but still learning. Before doing *anything* — before running a
command, moving a file, writing code — **explain what you're about to do
and why it matters**, in plain language, as if teaching, not just
reporting. Assume the person can read code but may not yet have the
intuition for *why* a given step exists in an ML data pipeline. Then do the
thing. Then briefly confirm what actually happened versus what was
expected, since "I ran the command" and "it worked correctly" are not the
same claim.

This is not a one-off tone instruction — apply it to every task below and
every task that comes after this file, for the rest of this project.

## Standing rule: never generate ground truth

This applies to every task in this file and every task added after it, not
just baseline evaluation specifically. If a task needs a "correct answer"
to compare something against — a reference translation, a labeled example,
an expected output — **that answer must come from real existing data,
never be written or invented on the spot**, even as a temporary
placeholder "to be reviewed later." An invented ground truth silently
turns a supposedly objective measurement into a circular one, and this
kind of shortcut tends to get carried forward into later results without
anyone noticing it happened. If real data for a needed reference doesn't
exist in the project yet, say so explicitly and treat that as a blocker to
flag, not a gap to quietly fill in.

**Correction, logged for the record**: an earlier version of
`src/baseline_eval.py` used hand-written pilot sentences with reference
translations written by the AI assistant itself, rather than pulled from
real data. This was caught before running the script and corrected to pull
real source/reference pairs from `nejm_enzh`'s test split (Mandarin) and
`eval/tico19_hi/` (Hindi) instead — flagged here so the reasoning isn't
lost if the question comes up again later.

---

## Task 1: Remove Spanish from the project

### What we're doing
Spanish was originally assigned to this teammate as a third language, but
another teammate (Nadjiba) has since confirmed she is covering Spanish (and
French), so it's being dropped here to avoid two people duplicating the
same work.

### Why this matters, not just "delete some files"
In an ML project, stale data lying around is a real risk, not just visual
clutter. If a config file, a leftover Spanish dataset folder, or an old
notebook still gets picked up by a script (for example, a `for lang in
["es", "hi", "zh"]` loop somewhere), you can end up silently training or
evaluating a model on a language nobody meant to include, and not notice
until much later — wasted compute time, and confusing results in a paper
that's supposedly about Hindi and Mandarin. So this isn't just tidying up;
it's removing a source of silent, hard-to-debug mistakes later.

### What to actually do
1. Search the whole project (not just the `data/` folder — also configs,
   YAML files, `data_prep.py`, any training scripts) for anything Spanish
   related: `es`, `spa`, `spanish`, `en-es`, `es-en`, `MeSpEn`, `EMEA`,
   `WMT20`.
2. Before deleting anything, list what you found and explain to the person
   what each one is and why it's safe to remove — don't just run `rm -rf`
   silently.
3. Remove Spanish data folders and any Spanish-specific config entries.
4. Check `README.md` and any documentation in the repo for Spanish
   references and update them.
5. After cleanup, confirm nothing else in the codebase still references a
   removed path (a broken reference is worse than a leftover file, since it
   fails loudly and confusingly later).

---

## Task 2: Understand what data actually exists right now, and why it's organized the way it is

### What we're doing
Before writing or running any training code, make sure you (and the
person) both understand what's actually sitting in `data/raw/` right now,
what each source is *for*, and why they're split the way they are. Don't
skip this step to get to the "fun part" of training — a misunderstanding
here compounds into wasted training runs later.

### The mental model to explain

There are two fundamentally different *roles* a dataset can play, and
mixing them up is a common beginner mistake worth explicitly walking
through:

- **General-domain data** teaches the model basic grammar, vocabulary, and
  sentence structure in the target language. It's large but not
  medically-focused.
- **Domain-specific (medical) data** teaches the model medical vocabulary,
  phrasing, and style — but there's much less of it available, so it's
  used more surgically.

The standard, well-established technique for low-resource domain-specific
MT (this is not something invented for this project — it's how real
published work like the NEJM-enzh/ParaMed paper and IndicTrans2 both do it)
is a **two-stage fine-tune**:
1. Fine-tune the pretrained base model on the *general* data first, so it
   adapts to the target language broadly.
2. Take that checkpoint and fine-tune it *again*, this time only on the
   pooled *medical* data, so it specializes.

For Mandarin, there's enough medical-domain data alone (NEJM-enzh, 66k
pairs) that this two-stage approach isn't needed — explain this contrast to
the person so they understand *why* Hindi and Mandarin are handled
differently, not just told to follow different steps.

### Current data inventory (verify this is still accurate before using it)

**Mandarin — role: single-stage fine-tune directly on medical data**
- `raw/en-zh/nejm_enzh/` — 66,265 pairs, medical, human-translated. This is
  the main training source.
- `raw/en-zh/tico19/` (971 pairs) + `eval/tico19_zh/` (2,100 pairs held
  out) — medical (COVID-domain), used for evaluation, not training on the
  held-out portion.
- WMT18 (zh-en) — not stored locally, auto-fetched by `sacrebleu` at eval
  time. Explain to the person why this one has no folder: it's fetched
  on-demand, not downloaded once and kept.

**Hindi — role: two-stage fine-tune (general, then medical)**
- Stage 1 (general) inputs:
  - `raw/en-hi/iitb/` — ~1.66M pairs, general domain (verified: no medical
    content in its source composition — explain this if asked, since an
    earlier version of the project docs incorrectly claimed it had a
    health subset).
  - `raw/en-hi/bpcc_ilci/hin_Deva.tsv` — general domain, but **needs a
    preprocessing fix before use**: this file has metadata columns before
    the actual sentence text, and it mixes ILCI data together with
    NLLB-Seed data with no column distinguishing which row came from
    which source. A generic `src`/`tgt` column reader will silently grab
    the wrong columns. Write (or reuse the pattern from) a dedicated
    parser, the same way `extract_tico19.py` was written for TICO-19's
    format, rather than assuming the generic table reader handles it.
- Stage 2 (medical) inputs:
  - `raw/en-hi/tico19/` (971 pairs) + `eval/tico19_hi/` (2,100 held out)
  - `raw/en-hi/lokmat_healthcare/` — 2,202 pairs, healthcare news, human
  - `raw/en-hi/zenodo_healthcare/` — 2,182 rows (both English and Hindi
    files verified as 2,182 rows despite an initial `wc -l` mismatch,
    which was a false alarm from embedded newlines in quoted CSV fields —
    worth explaining this to the person as a lesson: never trust `wc -l`
    on CSVs with quoted multi-line fields, always verify with a real CSV
    parser)
  - **Held pending supervisor approval, do not include yet**: HiMed-West
    (~117k pairs) — its Hindi side was machine-translated via NLLB, not
    human-translated, and the team is waiting on an explicit answer about
    whether MT-generated data is acceptable to train on. Explain to the
    junior *why* this matters: training a translation model on another
    model's machine translations risks the new model inheriting that
    model's specific errors and style quirks, rather than learning from
    real ground truth. This is a real methodological question, not
    red tape.
  - **Not yet available**: ILCI Health (~25k) and EILMT Health (~15k) —
    both pending approval on India's TDIL government portal. Don't block
    on these; the data already in place is enough to start.

---

## Task 3: Baseline testing (do this before any fine-tuning code)

### What we're doing, and why this comes before training
Two teammates on this project have already tested their pretrained base
models *before* fine-tuning them, and reported real comparative numbers
(e.g., one language scored well on the pretrained model already and barely
improved with fine-tuning; another improved noticeably and got faster).
This has not been done yet for Hindi or Mandarin, and it needs to happen
first — explain to the person why: **fine-tuning without a baseline means
you can never actually prove the fine-tuning helped.** You'd just have a
model and a feeling that it's probably better, with nothing to point to in
a paper.

**No baseline script exists yet in this project. Write one from scratch —
do not assume one is provided.** This is a real task, not just "run the
existing thing."

### What "done" looks like

Write a script (e.g. `src/baseline_eval.py`) that does the following, and
explain each part to the person as you build it, not just hand them
finished code silently:

1. Loads the pretrained base model for each language pair
   (`Helsinki-NLP/opus-mt-en-zh`, `Helsinki-NLP/opus-mt-en-hi`) via
   `transformers`.
2. **Pulls both the source sentences AND their reference translations
   from real data already in this project — never generate, write, or
   invent a "correct" translation yourself, even as a placeholder.**
   Every published paper this project's docs cite (NEJM-enzh/ParaMed,
   MedExpQA, PersianMedQA, IndicTrans2) evaluates against real
   human-translated held-out test data, never invented references — an
   AI-generated reference makes the eval circular (you'd be measuring
   "how similar is this to what an AI guessed," not "how correct is this
   translation"), which defeats the entire purpose of a baseline number
   meant to appear in a paper. Concretely:
   - **Mandarin**: pull sentence pairs from `nejm_enzh`'s own **test**
     split (2,102 pairs, real English/Chinese pairs from professional
     NEJM translators) — this is medical-domain and already sitting in
     the project.
   - **Hindi**: pull from `eval/tico19_hi/` (2,100 held-out pairs,
     medical/COVID-domain).
   - To reflect the pilot label list (cardiomegaly, pleural effusion,
     etc.) specifically, search these test splits for sentences
     containing those terms rather than writing new sentences from
     scratch — if none exist in the held-out data, that itself is worth
     reporting rather than working around by inventing text. (In
     practice: both real corpora are general-medicine/COVID text, not
     radiology reports, so most exact pilot terms turn up 0 or a
     handful of matches — report that honestly, don't pad it out.)
3. Scores translations using **two metrics, both required**:
   - **LaBSE** (primary metric, matches what the team has standardized
     on) — embed both the model's output and a reference translation
     using a LaBSE model (`sentence-transformers/LaBSE` on Hugging Face
     is the standard way to load it), then compute cosine similarity
     between the two embeddings. Explain to the person, before writing
     this part, why LaBSE is used instead of just BLEU — see the
     explanation the team has already been given: BLEU counts matching
     words, LaBSE compares meaning, and clinical language often expresses
     the same meaning with very different wording (negation, hedging,
     synonyms), which is exactly the failure mode BLEU misses and LaBSE
     catches.
   - **sacreBLEU** (secondary metric, for comparability with anything
     reported using it elsewhere).
4. Measures inference time **per sentence individually**, not in a
   batch — explain why before writing this part: batched throughput
   numbers don't reflect what a real single request would feel like in
   production use, which is closer to what actually matters for this
   project.
5. Saves a clear report (JSON or similar) with per-sentence results and
   summary statistics for both metrics plus timing, so it can be directly
   compared against teammates' already-reported numbers (e.g. Arabic:
   ~90% LaBSE baseline, +5% after fine-tuning and 15% faster; German:
   baseline already good, fine-tuning barely helped and made it slightly
   slower).
6. Only after this baseline exists and has been reviewed, move to
   fine-tuning, then re-run this exact same script against the fine-tuned
   checkpoint, and report the before/after delta in the same format the
   other two teammates already used.

Needed packages: `transformers`, `torch`, `sentence-transformers` (for
LaBSE), `sacrebleu`. Confirm these are installed before running; if not,
explain to the person what each one is for before installing.

---

## Task 4: Fine-tuning

Once Tasks 1–3 are done and understood (not just executed), proceed to
actual fine-tuning:
- Mandarin: single-stage fine-tune directly on `nejm_enzh`.
- Hindi: two-stage fine-tune as described in Task 2.

Save checkpoints for both, matching the pattern teammates are already using
so results are directly comparable when everyone reports back for the
paper.
