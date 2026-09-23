# Leakage-Aware Passage-Disjoint Evaluation of Parameter-Efficient Models for Vietnamese Hallucination Detection

This repository contains the reproducibility artifacts for the paper **Leakage-Aware Passage-Disjoint Evaluation of Parameter-Efficient Models for Vietnamese Hallucination Detection**, prepared for EAI FISAT 2026.

The study investigates Vietnamese hallucination detection under a passage-disjoint evaluation protocol designed to prevent exact normalized source contexts from crossing training, development, and held-out test partitions.

Row-level random partitioning can permit source-passage overlap between training and held-out examples, compromising the interpretation of held-out generalization. We construct and audit a passage-disjoint evaluation protocol from the released 7,000 labeled ViHallu examples, grouping all examples that share the same normalized source passage and searching 20,000 deterministic assignments for approximate class balance.

This is **not an official ViHallu test split** and results should not be interpreted as leaderboard performance.

## Research Motivation

```
row-level split
    |
same source passage may appear across partitions
    |
held-out evaluation may not represent passage-independent generalization
```

Our solution:

```
normalize source context
    |
group examples by passage
    |
assign whole passage groups
    |
train / development / test
```

## Overview

The protocol normalizes source contexts, groups examples that share the same passage, then assigns whole passage groups to train, development, and test partitions. The selected assignment is frozen before any model training. After the split was finalized, held-out test labels were not used for model optimization, checkpoint selection, hyperparameter tuning, or ablation design.

## Contributions

1. **Passage-disjoint evaluation protocol** constructed from the released 7,000 labeled ViHallu examples.
2. **Leakage audits** covering normalized context, IDs, complete triplets, and near-duplicate context analysis.
3. **Reproducible evaluation** of two PEFT generative configurations and two encoder baselines.
4. **Multi-seed and statistical analysis** for the PEFT configurations.
5. **Context-sensitivity and qualitative error analyses** with saved predictions and reproducible artifacts.

## Dataset and Evaluation Protocol

### Released ViHallu data

The DSC2025 ViHallu Challenge provides 7,000 labeled examples in Vietnamese. A historical 14,000-row evaluation file was found to contain 7,000 unique IDs, every ID appearing twice, with rows exactly reproducing the labeled triplets. This legacy file is quarantined and not used for evaluation.

### Passage grouping

All examples sharing the same NFKC-normalized, lowercased, whitespace-collapsed context form one passage group (3,865 unique normalized contexts overall).

### Passage-disjoint split

| Split | Rows | Unique contexts | No hallucination | Intrinsic | Extrinsic |
|---|---:|---:|---:|---:|---:|
| Train | 4,900 | 2,693 | 1,573 | 1,714 | 1,613 |
| Development | 1,050 | 589 | 337 | 367 | 346 |
| Test | 1,050 | 583 | 335 | 367 | 348 |

**Frozen properties**: base seed = 42; candidate deterministic group assignments searched = 20,000; selected effective seed = 11,073; proportions = 70% / 15% / 15%.

Exact normalized-context overlap across train, development, and test is zero according to the frozen audit artifacts.

### Leakage audits

- Exact overlap for normalized context, row ID, and complete triplet: **zero** across all partition pairs.
- Near-duplicate context analysis performed with char_wb TF-IDF 4-5-gram and nearest-neighbor matching.
- Quarantined legacy 14,000-row artifact diagnosed and documented.

### Important dataset note

The purpose of repartitioning was not to define a new official ViHallu test set but to construct train, development, and held-out partitions in which examples derived from the same normalized source passage cannot cross partition boundaries.

## Evaluated Models

| Configuration | Base model | Role | Adaptation | Model card |
|---|---|---|---|---|
| Qwen PEFT | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Generative detector | 4-bit QLoRA | [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Gemma PEFT | [Gemma 4 E2B IT](https://huggingface.co/google/gemma-4-E2B-it) | Generative detector | 4-bit QLoRA | [Hugging Face](https://huggingface.co/google/gemma-4-E2B-it) |
| PhoBERT | [PhoBERT-base-v2](https://huggingface.co/vinai/phobert-base-v2) | Vietnamese encoder baseline | Full fine-tuning | [Hugging Face](https://huggingface.co/vinai/phobert-base-v2) |
| XLM-R | [XLM-R-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | Multilingual encoder baseline | Full fine-tuning | [Hugging Face](https://huggingface.co/FacebookAI/xlm-roberta-base) |

The historical Vistral-7B-Chat model is not part of the final four-model experiment matrix.

## Main Results

**Seed-42 results on the ViHallu-derived passage-disjoint test partition**

| Configuration | Accuracy | Macro-F1 |
|---|---:|---:|
| Qwen3.5-4B QLoRA | 0.8238 | **0.8251** |
| Gemma 4 E2B QLoRA | 0.8219 | **0.8222** |
| PhoBERT-base-v2 | 0.7438 | **0.7452** |
| XLM-R-base | 0.7295 | **0.7295** |

**Multi-seed PEFT results** (seeds 42, 43, 44):

- Qwen3.5-4B QLoRA: Macro-F1 **0.8260 ± 0.0037**
- Gemma 4 E2B QLoRA: Macro-F1 **0.8202 ± 0.0045**

> Encoder baselines were evaluated only with the primary seed, so these results should be interpreted as configuration-level comparisons rather than a fully symmetric multi-seed model-family comparison.

Under the evaluated configurations and the same passage-disjoint protocol, the Qwen3.5-4B and Gemma 4 E2B PEFT systems obtained higher Macro-F1 than the selected PhoBERT and XLM-R encoder baselines. This does not support the general claim that LLMs outperform all encoders.

## Repository Structure

```
configs/           Fixed data and model configurations
data/              Data directory (raw, processed, legacy)
docs/              Consistency audit, reviewer-to-artifact map, reproducibility checklist
reports/           Dataset audit and split metadata
results/           Main results, multiseed, context ablation, statistics, error analysis
scripts/           P0-P5 pipeline commands
src/vihallu_repro/ Reusable data, audit, prompt, metric, and statistics code
tests/             Unit tests for data pipeline and statistics
```

## Reproducibility

### Environment

Python 3.11+ with dependencies from `requirements.txt` (core) and `requirements-gpu.txt` (training). The data-only audit requires only core dependencies.

### Dataset audit and split construction

```bash
bash scripts/run_p0_data.sh
```

### Main seed-42 model runs (requires GPU)

```bash
bash scripts/run_p1_main_seed42.sh
```

### Analysis (context ablation, statistics, error analysis)

```bash
bash scripts/run_p2_p4.sh
```

### PEFT multi-seed runs (requires GPU)

```bash
bash scripts/run_p5_multiseed.sh
```

### Frozen model revisions

| Model | HF ID | Revision | Adaptation |
|---|---|---|---|
| PhoBERT | vinai/phobert-base-v2 | e2375d266bdf39c6e8e9a87af16a5da3190b0cc8 | Full fine-tuning |
| XLM-R | FacebookAI/xlm-roberta-base | e73636d4f797dec63c3081bb6ed5c7b0bb3f2089 | Full fine-tuning |
| Qwen3.5-4B | Qwen/Qwen3.5-4B | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a | 4-bit QLoRA (r=64, alpha=128) |
| Gemma 4 E2B | google/gemma-4-E2B-it | 179516f0c449474fdc46f08f30ead5b11e178497 | 4-bit QLoRA (r=64, alpha=128) |

## Artifact Integrity

See `BUNDLE_MANIFEST_SHA256.txt` for SHA-256 checksums of all preserved artifacts.

## Important Interpretation Notes

### Passage-disjoint split

This is a derived evaluation split constructed from released labeled ViHallu data. It is **not** the official ViHallu test set.

### Response-only ablation

Response-only evaluation changes the input structure seen during fine-tuning, so its degradation reflects both context removal and distribution shift.

### Shuffled-context ablation

Shuffled context is treated as a context-sensitivity intervention because replacing the source passage can invalidate the semantic relation represented by the original gold label.

### Model comparison

Qwen, Gemma, PhoBERT, and XLM-R differ in architecture, scale, pretraining, tokenizer, and adaptation method. The repository therefore reports empirical results for the evaluated configurations rather than general model-family superiority.

### Ablation consistency note

The main seed-42 Qwen result (Macro-F1 0.8251) and the full-input context-ablation result (0.8280) come from different inference pipeline runs. See docs/CAMERA_READY_CONSISTENCY_AUDIT.md for details. No rerun was performed.

### No random-row counterfactual

The current study audits passage overlap structurally but does not include a same-model random-row-split counterfactual. Row-level partitioning permits passage overlap and therefore can compromise held-out generalization interpretation, but numerical inflation was not directly measured.

## References

| Reference | Type | Link |
|---|---|---|
| ViHallu / DSC2025 | Dataset | [DSC2025 ViHallu Challenge](https://github.com/DSC-UIT-2025/ViHallu) |
| PhoBERT | Paper | [Nguyen and Nguyen, Findings of EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.92/) |
| PhoBERT model | Model | [vinai/phobert-base-v2](https://huggingface.co/vinai/phobert-base-v2) |
| XLM-R | Paper | [Conneau et al., ACL 2020](https://aclanthology.org/2020.acl-main.747/) |
| XLM-R model | Model | [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) |
| LoRA | Method | [Hu et al.](https://arxiv.org/abs/2106.09685) |
| QLoRA | Method | [Dettmers et al.](https://arxiv.org/abs/2305.14314) |
| Qwen3.5 | Model | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Gemma 4 | Model | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) |
| VnCoreNLP | Toolkit | [Vu et al., NAACL-HLT 2018 Demonstrations](https://aclanthology.org/N18-5012/) |

## Citation

**Provisional** (no final proceedings metadata yet):

```bibtex
@inproceedings{nguyen2026leakage,
  title     = {Leakage-Aware Passage-Disjoint Evaluation of Parameter-Efficient Models for Vietnamese Hallucination Detection},
  author    = {Vinh Dinh Nguyen and Nhan Huu Tran},
  booktitle = {EAI FISAT 2026},
  year      = {2026},
  note      = {Provisional citation. No DOI assigned yet.}
}
```

## Data and Model Licensing

No repository-wide software license has been specified at this time. Third-party datasets, pretrained models, and software components remain subject to their respective licenses and terms. Raw ViHallu data is not redistributed in this repository. Please download from the DSC2025 challenge page.

## Acknowledgements

We thank the DSC2025 ViHallu Challenge organizers for the released labeled data. FPT University Can Tho provided the computing environment for model training and evaluation.
