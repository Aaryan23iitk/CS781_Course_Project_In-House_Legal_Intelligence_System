# In-House Legal Intelligence System (CS781 Capstone)

**Author:** Aaryan Tiwari (Roll No. 230023), IIT Kanpur
**Course:** CS781, Indian Institute of Technology Kanpur
**Codabench username:** `a_230023`

A confidential, self-hosted legal NLP pipeline for Indian court judgments, built across **seven competition phases**. A compact language model family is pretrained from scratch on the supplied legal corpora (no external pretrained LMs, no external legal data), and the resulting checkpoints are adapted to six downstream legal tasks.

> **Note on scores.** *Local* scores come from held-out splits I built from the public train/dev data. *Official* scores are the Codabench leaderboard values. They are not interchangeable, and this README keeps them separate.

---

## Table of contents

1. [Project at a glance](#1-project-at-a-glance)
2. [Official results](#2-official-results)
3. [Repository structure](#3-repository-structure)
4. [Phase 1: Foundation language model](#4-phase-1-foundation-language-model)
5. [Transfer protocol (Phases 2-7)](#5-transfer-protocol-phases-27)
6. [Phase-by-phase details](#6-phase-by-phase-details)
7. [Local results across all experiments](#7-local-results-across-all-experiments)
8. [Compute and training setup](#8-compute-and-training-setup)
9. [Key observations and honest caveats](#9-key-observations-and-honest-caveats)
10. [Limitations and future work](#10-limitations-and-future-work)
11. [How to run](#11-how-to-run)
12. [References](#12-references)

---

## 1. Project at a glance

| Phase | Task | Input -> Output | Primary metric |
|---|---|---|---|
| 1 | Legal LM pretraining | Legal text prompt -> ~128-word continuation | BERTScore F1 |
| 2 | Legal NER | Judgment text -> typed character spans (12 entity types) | Strict span-level Macro-F1 |
| 3 | Rhetorical role prediction | Sentences of a judgment -> one of 13 roles per sentence | Macro-F1 |
| 4 | Court judgment prediction | Long judgment -> binary outcome (0/1) | Macro-F1 |
| 5 | Legal statute identification | Judgment -> set of statutes (100 labels, multi-label) | Macro-F1 |
| 6 | Abstractive summarization | Long judgment -> summary | BERTScore F1 |
| 7 | Prior case retrieval (PCR) | Query case + candidate pool -> 7 ranked case IDs | Micro-F1@7 |

**Constraints that shaped the design**

- Phase-1 models must have **50M-300M parameters** and be trained **from scratch**.
- **No externally pretrained LMs** for the foundation and **no external legal corpora**.
- Colab/Kaggle-class compute (mostly Kaggle Tesla T4 GPUs), with sessions that can be killed.
- Strict span matching for NER, multi-label output for statutes, BERTScore for generation tasks.

**Scope.** Everything from tokenizer training and pretraining to downstream preprocessing, fine-tuning, inference and evaluation for all seven tasks. It does **not** include a production legal application, and it is not a substitute for expert legal judgment.

---

## 2. Official results

Ranks were recorded on **09/11/2026**. Total valid Codabench submissions across all phases: **41** (the minimum of 4 per phase was met for every phase).

| Phase | Task | Primary metric (official) | Rank | Valid submissions |
|---|---|---|---|---|
| 1 | Legal LM pretraining | 0.8327 | 56 | 4 |
| 2 | Legal NER | 0.5469 | 48 | 5 |
| 3 | Rhetorical role prediction | 0.4924 | 70 | 7 |
| 4 | Court judgment prediction | 0.6934 | 82 | 6 |
| 5 | Legal statute identification | 0.3613 | 85 | 6 |
| 6 | Abstractive summarization | 0.8369 | 72 | 7 |
| 7 | Prior case retrieval | 0.3473 | 86 | 6 |

---

## 3. Repository structure


```
.
|-- README.md
|-- report/                  # Final report (PDF + LaTeX source)
|-- phase1_pretraining/      # Tokenizer, model definitions, pretraining, generation
|-- phase2_ner/              # Legal NER notebooks
|-- phase3_rhetorical_roles/ # Rhetorical-role notebooks
|-- phase4_judgment/         # Judgment prediction notebooks
|-- phase5_statutes/         # Statute identification notebooks
|-- phase6_summarization/    # Summarization notebooks
|-- phase7_pcr/              # Prior case retrieval notebooks
|-- docs/                    # Per-phase technical documentation (.md)
`-- results/                 # Local metrics, submission files
```

Per-phase technical documentation (the experiment logs this README summarises) lives in the `docs/` folder as `CS781_Phase{2..7}_*_Documentation.md` and `CS781_Master_Technical_Documentation.md` (Phase 1).

---

## 4. Phase 1: Foundation language model

### 4.1 Data

The foundation corpus has four sources:

| Source | Raw size | Est. documents (sampled) | Median length (words) | Token budget (hybrid run) |
|---|---|---|---|---|
| Supreme Court | 1.32 GB | 35,486 | 3,869 | 180M |
| High Courts | 10.67 GB | 3,913,049 | 206 | 180M |
| Tribunals | 6.48 GB | 226,580 | 3,432 | 90M |
| Acts | 0.73 GB | 15,929 | 2,827 | 50M |

Document counts and lengths are **sampled estimates**, not an exact census. The 500M-token budget above belongs to the detailed hybrid (Model 1) run. High Court documents are very numerous but short, while Supreme Court and tribunal documents are long, so the mix caps each source instead of sampling in proportion to its raw size.

### 4.2 Tokenizer: `LegalBPE`

- Custom **byte-level BPE**, vocabulary **24,576**, with 4 special tokens (`PAD`, `UNK`, `BOS`, `EOS`).
- Trained on a sample from all four legal sources (~36.8M characters in the hybrid run).
- A round-trip test on legal terminology, section references, the rupee symbol and an em dash confirmed lossless encode/decode.

### 4.3 Four architectures investigated

| Model | Family | Notes |
|---|---|---|
| Model 1 | Encoder / prefix-LM hybrid | 118.11M params; 14 blocks, hidden 768, 12 heads, FFN 3,072; sequence length 1,536 (positional capacity 2,048); alternates prefix-LM and masked-LM objectives; 500M tokens over 20,332 optimizer steps; final val loss 2.387566 (perplexity 10.89) |
| Model 2 | Causal decoder-only | Standard decoder baseline |
| Model 3 | Encoder-decoder | Seq2seq layout |
| **Model 4** | **Efficient decoder-only** | **~99.31M params; width 640, 17 blocks, 10 query heads, 2 key/value heads (GQA), FFN 2,048; RoPE, RMSNorm, SwiGLU** |

All four satisfy the 50M-300M bound. Model 4 gave the best local Phase-1 BERTScore F1 (**0.837933**) and is the backbone behind the best result in several downstream phases.

### 4.4 Generation

Target length is 128 words. The documented Model-1 run decoded with temperature 0.8, top-p 0.95, repetition penalty 1.03 and KV-cache generation, with `PAD`/`UNK`/`BOS` blocked and word counts checked. Check the Model-4 notebook for its exact decoding settings before quoting them.

---

## 5. Transfer protocol (Phases 2-7)

Downstream phases **initialise from Phase-1 checkpoints**. Each phase is trained and evaluated **independently**: no phase's predictions are used as input to another, so the project has one shared foundation and separate task-specific descendants, not a pipeline of chained models.

Adaptation methods tried across phases:

- **Full and partial fine-tuning** (top blocks plus a task head, layer-wise LR decay)
- **LoRA**, **rsLoRA**, **AdaLoRA**, **LoRMA / LoRMA+**
- **IA3**, **BitFit**, **prefix tuning**
- **Houlsby / Pfeiffer-style adapters**
- **QLoRA (NF4)**
- **Knowledge distillation** (teacher -> smaller student)

```
Four legal sources -> LegalBPE (24,576) -> Phase-1 checkpoints (Models 1-4)
        |-> NER  |-> Rhetorical roles  |-> Judgment  |-> Statutes
        |-> Summarization  |-> Prior-case retrieval
```

---

## 6. Phase-by-phase details

### Phase 2: Legal NER
- **Formulation:** token classification with BILOU tags, converted back to character spans with tokenizer offset mappings. Strict matching needs exact boundaries and entity type.
- **Experiments (5):** knowledge distillation; Model 1 + LoRA; Model 2 + Q/V-only LoRMA+ with a BiGRU head; Model 4 full fine-tuning; Model 4 partial fine-tuning (top 6 blocks).
- **Long inputs:** document windowing (768-token windows) and document-level decoding.
- **Best local result:** knowledge distillation, strict Macro-F1 **0.503338**, from a 70-document labelled held-out fold. Official score: 0.5469.

### Phase 3: Rhetorical role prediction
- **Formulation:** sentence-level classification into 13 roles, using Corporate Law (CL) and Information Technology (IT) data, evaluated on CL-dev + IT-dev combined.
- **Experiments (7):** rsLoRA, KD, IA3, LoRMA, AdaLoRA, prefix tuning and LoRA on Models 1-4, including a BiGRU head in one Model-2 configuration.
- **Decoding:** raw predictions vs **transition-aware decoding**.
- **Best local result:** Model 4 + LoRA (rank 16, attention + MLP projections): Macro-F1 **0.430887** raw -> **0.448189** with transition-aware decoding. Official score: 0.4924.

### Phase 4: Court judgment prediction
- **Formulation:** binary classification of long judgments using **four selected chunks** and a hierarchical classifier.
- **Data:** 32,305 deduplicated training cases; 994 clean development cases (497 per class, after removing Test overlaps); 1,517 test cases. Train, Dev and Test differ in formatting (uppercase ratio, newlines), so the pipeline lowercases text and collapses whitespace.
- **Experiments (6):** Model 4 + LoRMA; Model-4 -> Model-2 KD; Model 1 + BitFit; Model 2 + AdaLoRA; Model 3 + Houlsby adapters; Model 4 partial fine-tuning.
- **Best local result:** KD student, Macro-F1 **0.707568** (precision 0.730512, recall 0.659960). Official score: 0.6934.

### Phase 5: Legal statute identification
- **Formulation:** 100-label multi-label classification with independent sigmoid outputs. Labels are long-tailed (272 to 11,730 occurrences, median 679); square-root inverse-frequency weights are clipped at 10.
- **Data:** 42,750 Train / 10,181 Dev (10,063 after removing normalised-text overlap with Train) / 13,019 Test.
- **Thresholding:** global, per-label and **frequency-shrunk per-label** thresholds, searched from 0.01 to 0.85 in steps of 0.01 with repeated half-split cross-fitting. If no label passes its threshold, the label with the highest probability-to-threshold ratio is selected.
- **Experiments (6):** KD, prefix tuning, AdaLoRA, Houlsby adapters, LoRMA+, partial fine-tuning.
- **Best local result:** Model 4 partial fine-tuning (embeddings + lower 12 blocks frozen; top 5 blocks, final RMSNorm and classifier trained), clean-Dev Macro-F1 **0.361166** (0.360677 on official Dev; 0.352110 cross-fitted). Official score: 0.3613.

### Phase 6: Abstractive summarization
- **Formulation:** decoder-only generation conditioned on the source document; loss is applied only to summary tokens.
- **Data:** 7,028 usable document-summary pairs (6,677 Train / 351 validation in one duplicate-grouped split), 100 test documents. Sources are heavily truncated: in the Model-4 LoRA run, 98.1% of Train and 97.7% of validation documents exceed the 1,500-token source budget.
- **Long inputs:** learned sentence selection or deterministic head-tail truncation.
- **Decoding:** no-repeat n-gram constraints (12-gram for Model 4 + LoRA; 8-gram with repetition penalty 1.05 for Model 2 + QLoRA) and minimum-length constraints.
- **Experiments (7):** Houlsby, Pfeiffer, QLoRA, LoRA, LoRMA+ and KD across Models 1-4.
- **Best local result:** Model 4 + Houlsby adapters, BERTScore F1 **0.837293** (ROUGE-1 0.368580, ROUGE-L 0.208071). Official score: 0.8369.

### Phase 7: Prior case retrieval
- **Formulation:** BM25 sparse candidate retrieval followed by a **neural cross-encoder reranker** over query-candidate pairs, with BM25-neural score fusion. The output is seven ranked case IDs per query.
- **Data (one documented run):** 827 / 118 / 237 Train / Dev / Test queries and 4,320 / 1,023 / 1,727 candidates. Ten Train queries without a usable positive were excluded in the KD workflow.
- **Experiments (6):** logit-level KD, full and partial fine-tuning, NF4 QLoRA, LoRMA+, and a run labelled "prefix tuning".
- **Best local result:** Model 1 partial fine-tuning with BM25-cross-encoder fusion, Micro-F1@7 **0.345748** (Precision@7 0.376513, Recall@7 0.319630). Official score: 0.3473.
- **Known discrepancy:** one notebook is *titled* "Prefix Tuning" but the executed code performs partial fine-tuning. That reranker was also worse than BM25 on Dev, so BM25 was used for Test inference in that run.

---

## 7. Local results across all experiments

All numbers are **local validation** scores, with each phase's primary metric. Protocols differ between phases and sometimes between experiments inside a phase (for example, Phase 6 used 16-, 32- and 100-document evaluation slices), so compare within a phase with care.

### Phases 4 and 5 (Macro-F1)

| Phase | Model / method | Score |
|---|---|---|
| 4 | Model 4 + LoRMA | 0.699496 |
| 4 | Model 4 -> Model 2 KD | **0.707568** |
| 4 | Model 1 + BitFit | 0.603042 |
| 4 | Model 2 + AdaLoRA | 0.652899 |
| 4 | Model 3 + Houlsby | 0.553946 |
| 4 | Model 4 partial FT | 0.676710 |
| 5 | Model 4 -> Model 2 KD | 0.338461 |
| 5 | Model 1 + prefix tuning | 0.223700 |
| 5 | Model 2 + AdaLoRA | 0.304206 |
| 5 | Model 3 + Houlsby-style | 0.140435 |
| 5 | Model 4 + LoRMA+ | 0.350580 |
| 5 | Model 4 partial FT | **0.361166** |

### Phases 6 and 7

| Phase | Model / method | Score |
|---|---|---|
| 6 | Model 4 + Houlsby | **0.837293** (BERTScore) |
| 6 | Model 1 + Houlsby | 0.786144 |
| 6 | Model 2 + QLoRA (NF4) | 0.822136 |
| 6 | Model 3 + Pfeiffer | 0.777894 |
| 6 | Model 4 -> Model 2 KD | 0.752478 |
| 6 | Model 4 + LoRA | 0.826415 |
| 6 | Model 4 + LoRMA+ | 0.785586 |
| 7 | Model 1 -> Model 3 KD | 0.334630 (Micro-F1@7) |
| 7 | Model 1 full FT | 0.332407 |
| 7 | Model 1 partial FT | **0.345748** |
| 7 | Model 2 + NF4 QLoRA | 0.219011 |
| 7 | BM25 baseline | 0.330183 |
| 7 | Model 3 partial FT reranker | 0.131184 |
| 7 | Model 4 + LoRMA+ | 0.123402 |

### Best local result per phase

| Phase | Best local method | Score |
|---|---|---|
| 1 | Model-4 efficient decoder | 0.837933 (BERTScore F1) |
| 2 | Knowledge distillation | 0.503338 (strict Macro-F1) |
| 3 | Model 4 + LoRA, transition-aware decoding | 0.448189 (Macro-F1) |
| 4 | Model 4 -> Model 2 KD | 0.707568 (Macro-F1) |
| 5 | Model 4 partial FT | 0.361166 (clean-Dev Macro-F1) |
| 6 | Model 4 + Houlsby | 0.837293 (BERTScore F1) |
| 7 | Model 1 partial FT | 0.345748 (Micro-F1@7) |

---

## 8. Compute and training setup

- **Hardware:** mainly Kaggle GPU sessions with NVIDIA Tesla T4 GPUs (CUDA).
- **Memory management:** micro-batching with gradient accumulation, mixed precision, and parameter-efficient methods.
- **Optimisation:** AdamW; linear warmup with cosine decay in several downstream runs; gradient clipping generally at 1.0; weight decay, learning rate and dropout vary per method (see the report's hyperparameter table and per-phase documentation).
- **Checkpointing:** checkpoints saved during training so runs can resume after killed sessions (optimizer, scheduler and training state saved where implemented).
- **Model selection:** by task-specific dev metrics: validation loss (LM), strict span Macro-F1 (NER), Macro-F1 (classification), BERTScore F1 (summarisation) and Micro-F1@7 (retrieval). Early stopping was used in some experiments only.
- **Wall-clock time:** recorded for selected runs, not consistently for all experiments.

---

## 9. Key observations and honest caveats

- **No single best adaptation method.** KD won Phase 4, partial fine-tuning won Phases 5 and 7, Houlsby adapters won Phase 6, and LoRA with transition-aware decoding won Phase 3.
- **Model 4 is the most frequent top backbone** (Phases 1, 3, 4, 5, 6), and it is also the best Phase-1 model by local BERTScore. Model 3 and prefix tuning are among the weakest results wherever they appear. These are patterns in the logs, not controlled proofs: several comparisons change backbone, method and data handling at once.
- **BM25 beat a neural reranker in one Phase-7 run** (0.330183 vs 0.131184 on Dev).
- **Transition-aware decoding helped Phase 3** (0.430887 -> 0.448189).
- **A fixed-length output is not a quality signal.** One Phase-6 KD run produced exactly 400-word summaries.
- **Local vs official scores differ** in several phases (for example Phase 2: 0.503 local vs 0.547 official; Phase 3: 0.448 local vs 0.492 official). Local splits are small or built differently from the official test set.
- **Not every comparison is a controlled ablation.** The cleanest ones are the Phase 5 threshold strategies, the Phase 6 decoding constraints, and BM25 vs reranking in Phase 7.

---

## 10. Limitations and future work

- Constrained notebook compute and context lengths; long documents are truncated in several runs, especially summarisation.
- Validation sizes and protocols vary between experiments, so cross-experiment comparisons are weak.
- No human legal-expert evaluation, no repeated-seed confidence intervals, and no end-to-end evaluation connecting all seven tasks.
- ROUGE and BERTScore measure similarity to a reference, not legal correctness; strict NER F1 rewards exact offsets; Micro-F1@7 only evaluates within the supplied candidate pool.

Natural next experiments: controlled ablations that change one factor at a time (tokenizer, backbone, adaptation method), repeated seeds with confidence intervals, longer-context or hierarchical encoders for summarisation, and a per-label and per-class error analysis for Phases 2-5.

---

## 11. How to run

> **TODO (author):** fill in the real commands and notebook names. The steps below are placeholders in the order the phases build on each other.

1. **Environment**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt   # TODO: add this file
   ```
2. **Phase 1:** train the tokenizer, then pretrain a model.
   ```bash
   # TODO: add the real script or notebook name
   ```
3. **Phases 2-7:** open the matching notebook for each phase, point it at the Phase-1 checkpoint and the official data, and run it top to bottom.
4. **Submissions:** each notebook writes the Codabench submission file for its phase to `results/`.

Datasets are the **official competition data** and are not redistributed in this repository. Model checkpoints are **not** included unless you add a download link here.

---

## 12. References

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022.
- Zhang et al., "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning," ICLR 2023.
- Liu et al., "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning" (IA3), NeurIPS 2022.
- Houlsby et al., "Parameter-Efficient Transfer Learning for NLP," ICML 2019.
- Pfeiffer et al., "AdapterHub: A Framework for Adapting Transformers," EMNLP 2020 (System Demonstrations).
- Li and Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation," ACL 2021.
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS 2023.
- Hinton, Vinyals and Dean, "Distilling the Knowledge in a Neural Network," 2015.
- Zhang et al., "BERTScore: Evaluating Text Generation with BERT," ICLR 2020.
- Lin, "ROUGE: A Package for Automatic Evaluation of Summaries," ACL 2004 Workshop.
- Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," 2009.

---

## Acknowledgements and LLM usage

Course: CS781, IIT Kanpur. LLM use for this project is described in the report's LLM Usage Declaration. Keep this section consistent with that declaration.
