# Local Tiny MT Models for Spanish, Mandarin, and Hindi — Datasets & Plan

## Context and change of direction

The supervisor has clarified that the project must use **local, tiny,
fine-tuned models**, not API calls to large hosted LLMs (Claude, GPT-4,
Gemini). This supersedes the earlier API-based pipeline design. The
pattern is now consistent with what two teammates have already started:

- **Teammate 1**: fine-tuning `google/byt5-small` (small, multilingual,
  character-level) on EMEA-V3 (starting with EN→FR, 50k pairs), planning to
  extend across all available languages from a combined dataset pool
  (Translation Initiative for COVID-19, UFAL Medical Corpus, EMEA-V3, COPPA
  corpus; WMT18 reserved for evaluation only).
- **Teammate 2**: fine-tuning separate **MarianMT** checkpoints per
  direction — ELRC-Medical-V2 for EN↔DE, PEACH for EN↔AR.

**Coverage gap**: between them, French, German, and Arabic are claimed.
**Spanish, Mandarin (Chinese), and Hindi have no owner yet.** This document
covers those three, following the same per-language MarianMT fine-tuning
pattern teammate 2 used, to avoid duplicating teammate 1's broader
multi-language ByT5 sweep in case it later reaches these languages too —
this should be confirmed with the team before starting, so two people don't
train the same language pair independently.

## Recommended base models (verified on Hugging Face)

Consistent with teammate 2's approach (pretrained MarianMT, fine-tuned per
direction), the following pretrained checkpoints exist and are small enough
for local fine-tuning and inference:

- `Helsinki-NLP/opus-mt-en-es` and `Helsinki-NLP/opus-mt-es-en`
- `Helsinki-NLP/opus-mt-en-zh` and `Helsinki-NLP/opus-mt-zh-en`
- `Helsinki-NLP/opus-mt-en-hi` and `Helsinki-NLP/opus-mt-hi-en`

All are part of the OPUS-MT project (Tiedemann & Thottingal, 2020),
originally trained with MarianNMT and converted to PyTorch/Hugging Face
`transformers` — the same family of models teammate 2 is already using for
German and Arabic, so the fine-tuning code/pipeline should be directly
reusable across languages with only the checkpoint name and dataset
changed.
Sources: https://huggingface.co/Helsinki-NLP/opus-mt-en-es ,
https://huggingface.co/Helsinki-NLP/opus-mt-en-zh ,
https://huggingface.co/Helsinki-NLP/opus-mt-hi-en

**Alternative worth testing for Hindi specifically**: `ai4bharat/indictrans2`
models, purpose-built for Indian languages including Hindi, MIT-licensed,
and reported to outperform general-purpose baselines on Indic translation
— see the "Hindi is the hardest of the three" note below for why this may
be worth the extra setup effort.
Source: https://www.emergentmind.com/topics/indictrans2

## Datasets, verified per language

### Spanish (EN↔ES)

1. **MeSpEn** — a dedicated English-Spanish medical resource combining
   parallel text collected from IBECS (Spanish Bibliographical Index in
   Health Sciences), SciELO (Scientific Electronic Library Online), PubMed,
   and MedlinePlus. This is the closest Spanish equivalent to what
   teammate 2 is using for German (ELRC-Medical) — purpose-built for the
   medical domain, not a general-domain corpus repurposed.
   Source: https://www.academia.edu/64821989/The_MeSpEn_Resource_for_English_Spanish_Medical_Machine_Translation_and_Terminologies_Census_of_Parallel_Corpora_Glossaries_and_Term_Translations
   (also referenced in the ParaMed paper below)

2. **WMT20 Biomedical Translation Shared Task** — provides training
   sentence pairs from Medline abstracts for English↔Spanish (also German,
   Portuguese, French, Italian, Russian). A standard, citable benchmark
   source, good for supplementing MeSpEn or for evaluation.
   Source: https://arxiv.org/pdf/2005.09133 (cited within the ParaMed
   paper, which also documents this)

3. **EMEA (via OPUS)** — the same European Medicines Agency corpus
   teammate 1 is already using for French is also available for Spanish
   through OPUS, since EMEA covers many EU languages. Worth checking
   whether teammate 1's centralized dataset pool already includes an
   EN-ES EMEA split before duplicating this download.
   Source: https://www.sketchengine.eu/opus-parallel-corpora/ (confirms
   OPUS's broad language coverage, Spanish included)

**Assessment**: Spanish is the best-resourced of the three languages here —
multiple dedicated medical corpora exist, comparable in quality/purpose to
what teammate 2 has for German.

### Mandarin Chinese (EN↔ZH)

1. **NEJM-enzh (also released as "ParaMed")** — a dedicated English-Chinese
   biomedical parallel corpus built from the New England Journal of
   Medicine, ~100,000 sentence pairs, ~3,000,000 tokens per side. This is
   explicitly built because, at the time of its creation, no other
   English-Chinese biomedical parallel corpus existed publicly — general
   OPUS-style EN-ZH data (news, subtitles) is not medical in nature and
   would need heavy filtering to be useful here. NEJM-enzh is the clear
   first choice for Mandarin.
   Sources: https://arxiv.org/pdf/2005.09133 (paper),
   https://github.com/boxiangliu/ParaMed (code and data)
   Also indexed at: https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-021-01621-8

2. **WMT20 Biomedical Translation Shared Task** — provides **test-only**
   sentence pairs for English-Chinese (no full training set), so this is
   useful for evaluation but cannot replace NEJM-enzh as a training source.
   Source: same as above (arxiv 2005.09133, background section)

**Assessment**: Mandarin has exactly one dedicated, purpose-built medical
parallel corpus (NEJM-enzh/ParaMed) at a similar scale to what teammate 1
used for the French pilot (50k pairs vs. ~100k here) — workable, but there
is no second independent source to combine with it the way Spanish has
MeSpEn + WMT20 + EMEA together. This is worth flagging to the team: Mandarin
translation quality was also separately found, in earlier research for this
project's language-selection phase, to degrade more than other languages as
text complexity increases — so this smaller, single-source data situation
compounds a risk that was already flagged.

### Hindi (EN↔HI)

This is the weakest-resourced of the three, and worth being upfront about
with the team rather than discovering it partway through fine-tuning.

1. **IIT Bombay English-Hindi Parallel Corpus** — the largest freely
   available EN-HI corpus (1.49 million sentence pairs), free for
   non-commercial research. However, it is a **general-domain** corpus
   (news, government documents, and other public sources), not
   medical-specific. It is a reasonable base/pretraining corpus but will
   likely need to be combined with a domain-adaptation step to be useful
   for clinical translation on its own.
   Source: https://arxiv.org/abs/1710.02855

2. **FutureBeeAI English-Hindi Medical Parallel Corpus** — a
   purpose-built medical-domain corpus (50,000+ sentence pairs,
   professionally translated). This is the closest Hindi equivalent to
   MeSpEn (Spanish) or NEJM-enzh (Mandarin) in terms of being
   domain-specific rather than general text. **Important caveat: this is a
   commercial dataset**, not freely downloadable research data — the
   provider requires contact for samples/licensing, with "custom licensing
   packages... for enterprise, research, or regulatory applications."
   This needs to be confirmed as actually accessible (and at what cost, if
   any, for academic use) before relying on it, unlike every other dataset
   in this document, which is free.
   Source: https://www.futurebeeai.com/dataset/parallel-corpora/hindi-english-translated-parallel-corpus-for-medical-domain

3. **AI4Bharat BPCC / Samanantar** — very large (BPCC: ~230 million
   sentence pairs across 22 Indic languages; Samanantar: ~49.7 million
   pairs across 11 Indic languages including Hindi), freely available,
   CC0-licensed. Not medical-specific, but large enough that a
   medical-domain subset could potentially be mined from it (e.g. by
   filtering for medical keywords/terminology), or it could serve as a
   strong general pretraining base before a smaller medical fine-tuning
   pass using option 1 or 2 above.
   Sources: https://github.com/AI4Bharat/indicnlp_catalog ,
   https://ai4bharat.iitm.ac.in/areas/nmt

**Assessment**: unlike Spanish and Mandarin, there is currently **no large,
free, purpose-built medical-domain EN-Hindi corpus** located. The
realistic path is a two-stage approach: fine-tune first on general-domain
data (IIT Bombay or a filtered BPCC/Samanantar subset) to adapt the model
to Hindi generally, then do a second, smaller fine-tuning pass on whatever
medical-domain Hindi data can be obtained (FutureBeeAI, if accessible, or
a manually curated smaller set built from the same UMLS/RadLex-style
grounding approach discussed for the earlier API-based plan). This should
be flagged to the supervisor directly: Hindi is likely to need more setup
work and a different strategy than the other two languages, not a
copy-paste of the Spanish/Mandarin approach.

## Summary table

| Language | Best dataset(s) | Domain-specific? | Free? | Base model |
|---|---|---|---|---|
| Spanish | MeSpEn, WMT20 Biomedical, EMEA (via OPUS) | Yes | Yes | `Helsinki-NLP/opus-mt-en-es` |
| Mandarin | NEJM-enzh / ParaMed | Yes | Yes | `Helsinki-NLP/opus-mt-en-zh` |
| Hindi | IIT Bombay (general) + FutureBeeAI (medical, commercial) or BPCC/Samanantar (general, mineable) | Partial / mixed | Mostly, one option isn't | `Helsinki-NLP/opus-mt-en-hi` or `ai4bharat/indictrans2` |

## Suggested next steps

1. **Confirm with the team** (especially teammate 1, who plans to "fine-tune
   again using the available languages") whether Spanish, Mandarin, or
   Hindi are already inside their planned dataset pool, to avoid two people
   training the same pair.
2. **Start with Spanish and Mandarin first** — both have a clear, free,
   dedicated medical corpus and a directly available MarianMT checkpoint,
   so the setup should closely mirror what teammate 2 already did for
   German and Arabic.
3. **Treat Hindi as a separate, harder sub-task**: get the FutureBeeAI
   licensing question answered early (is it actually free for this
   academic project or not), and in parallel test whether a
   medical-keyword-filtered subset of BPCC/Samanantar is usable, rather
   than waiting until the Spanish/Mandarin work is done to discover Hindi
   needs a different approach.
4. **Evaluation**: use WMT18 the way teammate 1 is already using it (held
   out, evaluation only), and, where available, the WMT20 Biomedical test
   sets for Spanish and Chinese specifically, so results are comparable
   across languages and across teammates' models using a shared benchmark
   rather than each person inventing their own test set.
