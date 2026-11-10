# In-House Legal Intelligence System

**CS781 Capstone Project, Indian Institute of Technology Kanpur (IIT Kanpur)**

| | |
|---|---|
| **Author** | Aaryan Tiwari |
| **Roll number** | 230023 |
| **Email** | aaryant23@iitk.ac.in |
| **Codabench username** | `a_230023` |
| **Ranks recorded on** | 09/11/2026 |

A confidential, self-hosted legal NLP pipeline for Indian court judgments, built across **seven competition phases**. Compact language models are pretrained **from scratch** on the supplied legal corpora, with no externally pretrained language models and no external legal data, and the resulting checkpoints are then adapted to six downstream legal tasks.

> **Local vs. official scores.** *Local* scores come from held-out splits built from the public train/dev data. *Official* scores are the Codabench leaderboard values. They use different data and protocols, so this README always labels which is which and never mixes them.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Official competition results](#2-official-competition-results)
3. [What this repository contains](#3-what-this-repository-contains)
4. [Data and preprocessing](#4-data-and-preprocessing)
5. [Phase 1: Foundation language model](#5-phase-1-foundation-language-model)
6. [Transfer protocol (Phases 2-7)](#6-transfer-protocol-phases-27)
7. [Phase-by-phase approach](#7-phase-by-phase-approach)
8. [Local results](#8-local-results)
9. [Training and compute setup](#9-training-and-compute-setup)
10. [Findings and caveats](#10-findings-and-caveats)
11. [Limitations and future work](#11-limitations-and-future-work)
12. [Reproducing the work](#12-reproducing-the-work)
13. [References](#13-references)
14. [LLM usage](#14-llm-usage)

---

## 1. Project overview

Indian court judgments are long and structurally complex. They contain case facts, procedural history, arguments, statutory references, judicial reasoning and final decisions, and their language varies across courts and document sources. This project builds an in-house system that handles seven legal understanding tasks using compact models trained only on the supplied corpora.

| Phase | Task | Input -> output | Primary metric |
|---|---|---|---|
| 1 | Legal language-model pretraining | Legal text prompt -> ~128-word continuation | BERTScore F1 |
| 2 | Legal named entity recognition | Judgment text -> typed character spans (12 entity types) | Strict span-level Macro-F1 |
| 3 | Rhetorical role prediction | Sentences of a judgment -> one of 13 roles per sentence | Macro-F1 |
| 4 | Court judgment prediction | Long judgment -> binary outcome (0 / 1) | Macro-F1 |
| 5 | Legal statute identification | Judgment -> set of statutes (100 labels, multi-label) | Macro-F1 |
| 6 | Abstractive summarization | Long judgment -> summary | BERTScore F1 |
| 7 | Prior case retrieval (PCR) | Query case + candidate pool -> 7 ranked case IDs | Micro-F1@7 |

**Constraints that shaped the design**

- Phase-1 models must have **50M-300M parameters** and be trained **from scratch**.
- **No externally pretrained language models** and **no external legal corpora**.
- Limited Colab/Kaggle-class compute; most work ran on Kaggle Tesla T4 GPUs.
- Exact span matching for NER, multi-label prediction for statutes, long-document handling, and BERTScore-based evaluation for generation tasks.

**Scope.** The project covers the full pipeline: tokenizer development, foundation-model pretraining, downstream preprocessing, training, inference and evaluation for all seven tasks. It does **not** cover a production-scale legal application, and it is not a replacement for expert legal judgment.

---

## 2. Official competition results

Total valid Codabench submissions across all phases: **41**. The minimum requirement of 4 distinct valid submissions per phase was met for all seven phases.

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

## 3. What this repository contains

The repository holds the full body of work for the seven phases:

- **Phase 1 code:** the `LegalBPE` tokenizer, the four foundation-model architectures, pretraining and generation.
- **Phase 2-7 notebooks:** one set of experiment notebooks per phase, each initialising from a Phase-1 checkpoint and producing a Codabench submission for that phase.
- **Technical documentation:** a master document for Phase 1 and one experiment document per downstream phase, recording the setup, results and caveats of every run summarised in this README.
- **Final report:** the CS781 report that accompanies this project.

Competition datasets are **not** redistributed here; they belong to the course and must be obtained through the official competition.

---

## 4. Data and preprocessing

### 4.1 Phase-1 pretraining corpus

| Source | Raw size | Est. documents | Median length (words) | 10th-90th percentile (words) | Token budget (hybrid run) |
|---|---|---|---|---|---|
| Supreme Court | 1.32 GB | 35,486 | 3,869 | 1,580-11,757 | 180M |
| High Courts | 10.67 GB | 3,913,049 | 206 | 68-960 | 180M |
| Tribunals | 6.48 GB | 226,580 | 3,432 | 238-9,831 | 90M |
| Acts | 0.73 GB | 15,929 | 2,827 | 407-14,848 | 50M |
| **Total** | | | | | **500M** |

Document counts and lengths are **sampled estimates**, not exact counts. High Court documents are very numerous but short, while Supreme Court, tribunal and Acts documents are long, so the mix gives each source a fixed token budget instead of sampling in proportion to raw size. The 500M-token budget belongs to the detailed hybrid (Model 1) run.

### 4.2 Downstream datasets

| Phase | Data facts |
|---|---|
| 2 | 12 entity types; labelled folds; 70 labelled documents in the knowledge-distillation held-out workflow |
| 3 | Sentence-level labels from Corporate Law (CL) and Information Technology (IT); 13 rhetorical roles; combined CL-dev / IT-dev evaluation |
| 4 | 32,305 deduplicated Train cases; 994 clean Dev cases (497 per class); 1,517 Test cases |
| 5 | 42,750 Train / 10,181 Dev / 13,019 Test; 100 statute labels; 10,063 clean Dev records after removing normalised-text overlap with Train |
| 6 | 7,028 usable document-summary pairs (6,677 Train / 351 validation in one duplicate-grouped split); 100 Test documents |
| 7 | 827 / 118 / 237 Train / Dev / Test queries; 4,320 / 1,023 / 1,727 candidate documents (one documented run); 10 Train query groups without a usable positive candidate excluded in the KD workflow, leaving 817 |

### 4.3 Statistics that informed preprocessing

- **Phase 1 prompts:** across 1,000 test prompts, mean length is about 1,082 BPE tokens, the 95th percentile is 1,139 and the maximum is 1,367.
- **Phase 4:** sampled Train text has a median uppercase-letter ratio of 0.0351, versus 0.0000 in clean Dev and Test. Newlines appear in 0% of sampled Train, 100% of Dev and 98.9% of Test documents. To remove this formatting mismatch, the pipeline lowercases text and collapses whitespace. The deduplicated Train set also has more class-0 than class-1 examples, while clean Dev is balanced.
- **Phase 5:** label frequencies range from 272 to 11,730 (median 679). Square-root inverse-frequency weights are clipped at 10 to handle the long tail.
- **Phase 6:** in the Model-4 LoRA run, sources are limited to 1,500 tokens and targets to 800 tokens, and 98.1% of Train and 97.7% of validation documents are truncated at the source side.

### 4.4 Representation choices

- **Tokenizer:** custom byte-level BPE (`LegalBPE`), vocabulary 24,576, with `PAD`, `UNK`, `BOS` and `EOS` special tokens. It was trained on a sample from all four sources (about 36.8M characters in the hybrid run). A round-trip test on legal terminology, section references, the rupee symbol and an em dash confirmed lossless encoding and decoding.
- **NER:** token-level BILOU predictions are converted into typed character spans through tokenizer offset mappings, so predicted boundaries and types can be matched exactly.
- **Rhetorical roles:** sentence-level classification, evaluated both on raw predictions and after transition-aware decoding.
- **Judgment prediction:** four selected document chunks fed to a hierarchical classifier.
- **Statutes:** multi-hot target vectors over the 100 statute IDs.
- **Summarization:** loss applied only to summary tokens; sources reduced by learned sentence selection or head-tail truncation.
- **Retrieval:** BM25 builds split-specific candidate sets; a cross-encoder scores query-candidate pairs.

---

## 5. Phase 1: Foundation language model

Four compact architectures were trained from scratch, all inside the 50M-300M parameter range.

| Model | Family | Details |
|---|---|---|
| Model 1 | Encoder / prefix-LM hybrid | 118.11M parameters; 14 blocks, hidden size 768, 12 heads, FFN 3,072; sequence length 1,536 (positional capacity 2,048); alternates prefix-LM and masked-LM objectives; 500M tokens over 20,332 optimizer steps; final validation loss 2.387566 (perplexity 10.89) |
| Model 2 | Causal decoder-only | Standard decoder-only baseline |
| Model 3 | Encoder-decoder | Sequence-to-sequence layout |
| **Model 4** | **Efficient decoder-only** | **About 99.31M parameters; width 640, 17 decoder blocks, 10 query heads and 2 key/value heads (grouped-query attention), FFN 2,048; rotary positional embeddings (RoPE), RMSNorm, SwiGLU** |

**Local Phase-1 BERTScore F1** (155 prompt-reference pairs):

| Rank | Model | BERTScore F1 |
|---|---|---|
| 1 | Model 4: efficient decoder | **0.837933** |
| 2 | Model 2: decoder-only | 0.826497 |
| 3 | Model 1: encoder + prefix-LM hybrid | 0.819721 |

The four models differ in size, objective and token budget, so this is a comparison of complete systems and not a controlled ablation of one design choice.

**Generation.** The target length is 128 words. The documented Model-1 run decoded with temperature 0.8, top-p 0.95 and repetition penalty 1.03 with KV-cache generation, blocking `PAD`, `UNK` and `BOS` and checking generated word counts. These settings belong to the Model-1 run and are not assumed to be identical for Model 4.

---

## 6. Transfer protocol (Phases 2-7)

Downstream phases **initialise from Phase-1 checkpoints**. Each phase is trained and evaluated **independently**: no task's predictions feed another task, so the project has one shared foundation with separate task-specific descendants and not a chain of dependent models.

```
Four legal sources (SC / HC / Tribunals / Acts)
            |
     LegalBPE tokenizer (24,576)
            |
 Phase-1 checkpoints (Models 1-4)
   |        |         |          |            |             |
  NER   Rhetorical  Judgment / Summarization  Prior-case
 (span   roles      statutes    (generation)  retrieval
tagging) (sentence  (document                 (BM25 +
          classif.)  classif.)                 reranking)
```

Adaptation techniques used across phases:

- Full and partial fine-tuning (top blocks plus a task head, layer-wise learning-rate decay)
- LoRA, rsLoRA, AdaLoRA, LoRMA / LoRMA+
- IA3, BitFit, prefix tuning
- Houlsby and Pfeiffer-style adapters
- QLoRA (NF4 quantisation)
- Knowledge distillation from a teacher to a smaller student

---

## 7. Phase-by-phase approach

### Phase 2: Legal NER
Formulated as token classification with BILOU tags, reconstructed to character spans with tokenizer offsets. Five experiments: knowledge distillation; Model 1 + LoRA; Model 2 + Q/V-only LoRMA+ with a BiGRU head; Model 4 full fine-tuning; Model 4 partial fine-tuning of the top six blocks. Model-4 runs use 768-token document windows and document-level decoding for long inputs. The distillation run uses all 70 labelled documents for its final refit.
**Best local:** knowledge distillation, strict Macro-F1 **0.503338** (70-document held-out fold). **Official:** 0.5469.

### Phase 3: Rhetorical role prediction
Sentence-level classification into 13 roles over CL and IT data, with task-specific heads (a BiGRU head in one Model-2 configuration). Seven experiments covering rsLoRA, KD, IA3, LoRMA, AdaLoRA, prefix tuning and standard LoRA on Models 1-4. Raw and transition-aware decoded predictions are both evaluated.
**Best local:** Model 4 + LoRA (rank 16, attention and MLP projections), Macro-F1 0.430887 raw and **0.448189** after transition-aware decoding. **Official:** 0.4924.

### Phase 4: Court judgment prediction
Binary classification with four selected chunks per judgment and a hierarchical classifier. Six experiments: Model 4 + LoRMA; Model-4-to-Model-2 knowledge distillation; Model 1 + BitFit; Model 2 + AdaLoRA; Model 3 + Houlsby adapters; Model 4 partial fine-tuning. The checkpoint was chosen on the local clean-Dev evaluator, and refit or ensemble results are kept separate from that held-out evaluation.
**Best local:** KD student, Macro-F1 **0.707568** (precision 0.730512, recall 0.659960). **Official:** 0.6934.

### Phase 5: Legal statute identification
100-label multi-label classification with one probability per label. Six experiments: KD (Model 4 to Model 2), prefix tuning, AdaLoRA, Houlsby adapters, LoRMA+ and partial fine-tuning. In the partial fine-tuning run, embeddings and the lower 12 blocks are frozen, and the upper five blocks, the final RMSNorm and the hierarchical classifier are trained (4 chunks of 384 tokens per document).
Thresholding compares global, per-label and frequency-shrunk per-label thresholds, searching 0.01-0.85 in steps of 0.01 with repeated half-split cross-fitting. If no label passes its threshold, the label with the highest probability-to-threshold ratio is selected.
**Best local:** Model 4 partial fine-tuning, clean-Dev Macro-F1 **0.361166** (0.360677 on official Dev; 0.352110 cross-fitted). **Official:** 0.3613.

### Phase 6: Abstractive summarization
Decoder-only generation conditioned on the source document, with loss masked on source tokens. Seven experiments: Houlsby and Pfeiffer adapters, QLoRA, standard LoRA, LoRMA+ and KD across Models 1-4, with learned sentence selection or head-tail truncation for long inputs. Decoding uses no-repeat n-gram constraints (12-gram for Model 4 + LoRA; 8-gram with repetition penalty 1.05 for Model 2 + QLoRA) and minimum-length constraints.
**Best local:** Model 4 + Houlsby adapters, BERTScore F1 **0.837293** (ROUGE-1 0.368580, ROUGE-L 0.208071). **Official:** 0.8369.

### Phase 7: Prior case retrieval
BM25 produces sparse candidates, a neural cross-encoder reranks them, and BM25-neural score fusion produces the final seven ranked IDs. The knowledge-distillation workflow adds negative mining, cached teacher relevance scores, and student training with distillation, hard-label and pairwise-ranking losses. Six experiments across Models 1-4: logit-level KD, full and partial fine-tuning, NF4 QLoRA and LoRMA+.
**Best local:** Model 1 partial fine-tuning with BM25-cross-encoder fusion, Micro-F1@7 **0.345748** (Precision@7 0.376513, Recall@7 0.319630). **Official:** 0.3473.
**Known discrepancy:** one notebook is titled "Prefix Tuning" but its executed code performs partial fine-tuning. In that run the reranker was worse than BM25 on Dev, so BM25 was used for Test inference.

---

## 8. Local results

All values are **local validation** scores on each phase's primary metric. Protocols differ between phases and sometimes between experiments inside a phase (for example, Phase 6 used 16-, 32- and 100-document evaluation slices), so compare scores within a phase with care.

### Best local result per phase

| Phase | Best local method | Score |
|---|---|---|
| 1 | Model-4 efficient decoder | 0.837933 (BERTScore F1) |
| 2 | Knowledge distillation | 0.503338 (strict Macro-F1) |
| 3 | Model 4 + LoRA, transition-aware decoding | 0.448189 (Macro-F1) |
| 4 | Model 4 -> Model 2 KD | 0.707568 (Macro-F1) |
| 5 | Model 4 partial fine-tuning | 0.361166 (clean-Dev Macro-F1) |
| 6 | Model 4 + Houlsby adapters | 0.837293 (BERTScore F1) |
| 7 | Model 1 partial fine-tuning | 0.345748 (Micro-F1@7) |

### Phase 3 top results (decoded Macro-F1)

| Method | Score |
|---|---|
| Model 4 + LoRA | 0.448189 |
| Model 4 + AdaLoRA | 0.444992 |
| Model 4 rsLoRA teacher -> Model 2 student (KD) | 0.415334 |

### Phases 4 and 5: all experiments (Macro-F1)

| Phase | Model / method | Score |
|---|---|---|
| 4 | Model 4 -> Model 2 KD | **0.707568** |
| 4 | Model 4 + LoRMA | 0.699496 |
| 4 | Model 4 partial fine-tuning | 0.676710 |
| 4 | Model 2 + AdaLoRA | 0.652899 |
| 4 | Model 1 + BitFit | 0.603042 |
| 4 | Model 3 + Houlsby adapters | 0.553946 |
| 5 | Model 4 partial fine-tuning | **0.361166** |
| 5 | Model 4 + LoRMA+ | 0.350580 |
| 5 | Model 4 -> Model 2 KD | 0.338461 |
| 5 | Model 2 + AdaLoRA | 0.304206 |
| 5 | Model 1 + prefix tuning | 0.223700 |
| 5 | Model 3 + Houlsby-style adapters | 0.140435 |

### Phases 6 and 7: all experiments

| Phase | Model / method | Score |
|---|---|---|
| 6 | Model 4 + Houlsby adapters | **0.837293** (BERTScore F1) |
| 6 | Model 4 + standard LoRA | 0.826415 |
| 6 | Model 2 + QLoRA (NF4) | 0.822136 |
| 6 | Model 1 + Houlsby adapters | 0.786144 |
| 6 | Model 4 + LoRMA+ | 0.785586 |
| 6 | Model 3 + Pfeiffer adapters | 0.777894 |
| 6 | Model 4 -> Model 2 KD | 0.752478 |
| 7 | Model 1 partial fine-tuning | **0.345748** (Micro-F1@7) |
| 7 | Model 1 -> Model 3 logit-level KD | 0.334630 |
| 7 | Model 1 full fine-tuning | 0.332407 |
| 7 | BM25 baseline | 0.330183 |
| 7 | Model 2 + NF4 QLoRA | 0.219011 |
| 7 | Model 3 partial fine-tuning reranker | 0.131184 |
| 7 | Model 4 + LoRMA+ | 0.123402 |

---

## 9. Training and compute setup

- **Hardware:** mainly Kaggle GPU sessions with NVIDIA Tesla T4 GPUs and CUDA.
- **Memory management:** micro-batching with gradient accumulation, mixed-precision execution and parameter-efficient methods.
- **Optimisation:** AdamW; linear warmup followed by cosine decay in several downstream runs; gradient clipping generally at 1.0; weight decay, learning rate and dropout depend on the model and adaptation method.
- **Checkpointing:** checkpoints are saved during training so runs can resume after interrupted sessions, with optimizer, scheduler and training state saved where implemented.
- **Model selection:** by task-specific development metrics: validation loss for language modelling, strict span Macro-F1 for NER, Macro-F1 for classification, BERTScore F1 for summarization and Micro-F1@7 for retrieval. Early stopping was used in some experiments and not in others.
- **Wall-clock time:** recorded for selected runs, not for every experiment.

### Configuration of the selected model per phase

| | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | Phase 6 | Phase 7 |
|---|---|---|---|---|---|---|---|
| Selected model (for these settings) | Model 4 | Model 4 partial FT | Model 4 LoRA | Model 4 -> Model 2 KD | Model 4 partial FT | Model 4 Houlsby | Model 1 partial FT |
| Parameters | 99.31M | 99.34M (backbone) | 103.31M (total) | Model 2 student | 100.19M (total) | 102.12M (total) | 118.11M (backbone) |
| Input | Decoder-only sequence | 768-token windows | Sentence-level context | Four-chunk hierarchical | 4 chunks x 384 tokens | 1,500-token source budget | Query-candidate pairs |
| Learning rate | Model-specific | 1e-4 (top blocks) | 2e-4 LoRA; 1e-3 head | Distillation student training | 4e-5 backbone; 4e-4 head | 2e-4 | 3e-5 (top blocks) |
| Batch x accumulation | 4 x 4 (hybrid run) | 2 x 2 windows | 4 x 4 | Teacher-student training | 2 x 8 documents | 1 x 8 | 2 query groups per update |
| Duration | 20,332 steps (hybrid run) | up to 10 epochs, 6 executed | up to 6 epochs, 5 executed | 2-epoch student setup | up to 3 epochs | 3 epochs, 2,505 steps | 3 epochs, 1,227 planned updates |
| Regularisation | Weight decay 0.10 (hybrid run) | Head dropout 0.10 | LoRA dropout 0.05; weight decay 0.01 | Knowledge distillation | Weight decay 0.01; EMA 0.999 | Adapter dropout 0.05; weight decay 0.01 | Layer-wise LR decay 0.8; weight decay 0.02 |
| Output strategy | 128-word generation | BILOU span decoding | Transition-aware decoding | Binary classification | Frequency-shrunk per-label thresholds | No-repeat 12-gram decoding | BM25 retrieval + neural ranking |

These settings summarise the selected local-validation runs and are not necessarily identical to the final submitted checkpoints. Phase-1 step counts and optimiser settings come from the Model-1 hybrid run, not from the separate Model-4 training run.

---

## 10. Findings and caveats

- **No single best adaptation method.** Knowledge distillation won Phase 4, partial fine-tuning won Phases 5 and 7, Houlsby adapters won Phase 6, and LoRA with transition-aware decoding won Phase 3.
- **Model 4 is the most frequent top backbone** (Phases 1, 3, 4, 5 and 6), and it is also the best Phase-1 model by local BERTScore. Model 3 and prefix tuning are among the weakest results wherever they appear. These are patterns in the logs, not controlled proofs, because many comparisons change backbone, adaptation method and data handling at once.
- **BM25 beat a neural reranker in one Phase-7 run** (Dev Micro-F1@7 0.330183 versus 0.131184), so BM25 was used for Test inference there.
- **Transition-aware decoding helped Phase 3**, raising Macro-F1 from 0.430887 to 0.448189.
- **Fixed-length output is not a quality signal.** One Phase-6 KD run produced exactly 400-word summaries, which reflects an output-length property and not better summaries.
- **Local and official scores differ** in several phases (for example Phase 2: 0.503 local versus 0.547 official; Phase 3: 0.448 local versus 0.492 official). Local splits are small or built differently from the official test set.
- **Not every comparison is a controlled ablation.** The cleanest controlled comparisons are the Phase 5 threshold strategies, the Phase 6 decoding constraints and BM25 versus reranking in Phase 7.
- **What the metrics measure.** BERTScore and ROUGE reward similarity to a reference summary, not legal correctness; strict NER F1 rewards exact offsets; Micro-F1@7 only evaluates within the supplied candidate pool.

---

## 11. Limitations and future work

**Limitations**

- Constrained notebook compute and context lengths; long documents are truncated in several runs, especially in summarization.
- Validation sizes and protocols vary across experiments, which weakens comparisons between them.
- No human legal-expert evaluation, no repeated-seed confidence intervals for every phase, and no end-to-end evaluation connecting all seven tasks.
- Checkpoint and tokenizer provenance concerns were noted in one knowledge-distillation workflow, and one notebook title does not match its executed method (Phase 7).

**Next steps**

- Controlled ablations that change one factor at a time: tokenizer, backbone and adaptation method.
- Repeated seeds with confidence intervals.
- Longer-context or hierarchical encoders for summarization.
- Per-label and per-class error analysis for Phases 2-5.
- Human evaluation of legal accuracy for generated summaries.

---

## 12. Reproducing the work

1. **Data.** Obtain the official competition datasets for all seven phases. They are not included in this repository.
2. **Environment.** Experiments ran on Kaggle with an NVIDIA Tesla T4 GPU, PyTorch and CUDA. Training relies on micro-batching, gradient accumulation and mixed precision to fit T4 memory, so smaller GPUs may need smaller micro-batches.
3. **Phase 1.** Train the `LegalBPE` tokenizer (vocabulary 24,576) on a sample drawn from all four legal sources, then pretrain a foundation model on the capped token budget described in Section 4.1. Sessions can be interrupted, so keep checkpoints and resume from them.
4. **Phases 2-7.** For each downstream phase, load the chosen Phase-1 checkpoint, apply the adaptation method listed for that phase in Sections 7 and 9, train with development-set model selection, and generate the Codabench submission file.
5. **Evaluation.** Local metrics follow the competition definitions: BERTScore F1 (Phases 1 and 6), strict span-level Macro-F1 (Phase 2), Macro-F1 (Phases 3-5) and Micro-F1@7 (Phase 7).

---

## 13. References

1. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022.
2. Zhang et al., "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning," ICLR 2023.
3. Liu et al., "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning" (IA3), NeurIPS 2022.
4. Houlsby et al., "Parameter-Efficient Transfer Learning for NLP," ICML 2019.
5. Li and Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation," ACL 2021.
6. Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," Foundations and Trends in Information Retrieval, vol. 3, no. 4, 2009.
7. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS 2023.
8. Zhang et al., "BERTScore: Evaluating Text Generation with BERT," ICLR 2020.
9. Lin, "ROUGE: A Package for Automatic Evaluation of Summaries," Text Summarization Branches Out, ACL-04 Workshop, 2004.
10. Pfeiffer et al., "AdapterHub: A Framework for Adapting Transformers," EMNLP 2020 System Demonstrations.
11. Hinton, Vinyals and Dean, "Distilling the Knowledge in a Neural Network," 2015.

---

## 14. LLM usage

The use of large language models in this project is described in the LLM Usage Declaration of the accompanying report, which is the authoritative record.
