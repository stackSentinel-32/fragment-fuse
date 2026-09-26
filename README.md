# fragment-fuse ⚡

> **High-Performance Multi-Source Business Entity Resolution System**  
> *Amazon ML Challenge 2026 — End-to-End Solution*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-LightGBM%20Pairwise-brightgreen)](https://lightgbm.readthedocs.io/)
[![Evaluation](https://img.shields.io/badge/Metric-Macro%20F0.5%20Optimized-orange)](#-evaluation-metric-macro-f05)
[![Fair Play](https://img.shields.io/badge/Rules-Zero%20External%20APIs-blue)](#-fair-play--integrity-compliance)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📌 Executive Summary

In large-scale commercial platforms, business identity data arrives from multiple independent, noisy data sources without common identifiers. **fragment-fuse** is an end-to-end Machine Learning pipeline that identifies and links records corresponding to the same real-world business entity across three distinct sources:

- **Source 1 ($S_1$)**: The deduplicated reference source.
- **Source 2 ($S_2$) & Source 3 ($S_3$)**: Independent sources containing partial, noisy fragments (with zero, one, or multiple matches per $S_1$ entity).

The system handles real-world data noise: legal suffix variations (`Corp` vs `Corporation`, `Pvt Ltd`), punctuation differences, token permutations, address typos, landmark-based descriptors (`Near SBI ATM`), and open-set international records (**France** in test set vs **US** and **India** in training).

---

## ⚡ Key Architectural Innovations

1. **Sublinear Inverted-Index Blocking Engine**  
   Naive pairwise comparison over millions of records requires $\mathcal{O}(N \times M)$ evaluations (billions of pairs). Our inverted-index blocking builds weighted token signatures, character n-grams, and normalized postal/locality keys to retrieve high-recall candidate sets ($\le 50$ per $S_1$ entity) in sublinear time.

2. **Multi-Signal String & Address Feature Engineering**  
   High-throughput feature generation using C-accelerated `RapidFuzz` to compute token sort ratios, token set ratios, Levenshtein distances, word-level Jaccard similarities, length deltas, and exact match flags across both business names and physical addresses.

3. **Singleton-Stratified LightGBM Classifier**  
   A gradient-boosted pairwise decision tree model trained on stratified positive and negative candidate pairs. Singleton ratios are explicitly preserved during validation splits to prevent distribution shift.

4. **Precision-Heavy Macro $F_{0.5}$ Threshold Tuning**  
   In commercial entity resolution, false merges (merging two distinct businesses) are far more damaging than missed links. The decision threshold is dynamically tuned on validation holdout to maximize macro $F_{0.5}$ (weighting precision 2× over recall).

5. **Open-Set Domain Generalization**  
   Text normalizers and feature extractors operate string-agnostically without hardcoding country vocabularies, ensuring seamless transfer to unseen test domains like France.

---

## 🏗️ Pipeline Architecture

```
   ┌────────────────────────────────────────────────────────────────┐
   │                       Input TSV Sources                        │
   │  • S1 (Reference)       • S2 (Noisy)       • S3 (Noisy)        │
   └───────────────────────────────┬────────────────────────────────┘
                                   │
                                   ▼
   ┌────────────────────────────────────────────────────────────────┐
   │            Stage 1: Preprocessing & Normalization              │
   │  • Legal suffix cleansing     • Accent & punctuation stripping │
   │  • Address standardization    • Token canonicalization         │
   └───────────────────────────────┬────────────────────────────────┘
                                   │
                                   ▼
   ┌────────────────────────────────────────────────────────────────┐
   │            Stage 2: Multi-Key Inverted-Index Blocking          │
   │  • Token & 3-gram signatures  • Weight-ranked candidate sets   │
   │  • Cap at <= 50 candidates    ──► candidate_pairs.tsv          │
   └───────────────────────────────┬────────────────────────────────┘
                                   │
                                   ▼
   ┌────────────────────────────────────────────────────────────────┐
   │            Stage 3: Pairwise Feature Engineering               │
   │  • RapidFuzz token set/sort   • Levenshtein & Jaccard metrics  │
   │  • Cross-field length ratios  • Country consistency flags      │
   └───────────────────────────────┬────────────────────────────────┘
                                   │
                                   ▼
   ┌────────────────────────────────────────────────────────────────┐
   │            Stage 4: LightGBM Scoring & Classification          │
   │  • Pairwise probability inference                              │
   │  • Stratified cross-validation holdout                         │
   └───────────────────────────────┬────────────────────────────────┘
                                   │
                                   ▼
   ┌────────────────────────────────────────────────────────────────┐
   │            Stage 5: Macro F0.5 Thresholding & Output           │
   │  • Dynamic threshold tuning   • Singleton identification       │
   │  • Formatted TSV output       ──► matching_results.tsv         │
   └────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Layout

```
amazon_inside_out/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── preprocess.py        # String cleaning & token normalization
│       │   ├── blocking.py          # Inverted-index candidate generation
│       │   ├── features.py          # Pairwise similarity feature extraction
│       │   ├── evaluate.py          # Macro F0.5 evaluation & threshold search
│       │   ├── train.py             # LightGBM pairwise model training
│       │   ├── predict.py           # Test set inference & candidate scoring
│       │   └── pipeline.py          # Master CLI orchestrator (train/predict/full)
│       ├── requirements.txt         # Pinned runtime dependencies
│       └── README.md                # Detailed pipeline reproduction instructions
├── output/                          # Generated outputs & submission artifacts
│   ├── matching_results.tsv         # Final predicted entity matches
│   ├── candidate_pairs.tsv          # Candidate sets from blocking stage
│   └── .gitkeep                     # Directory tracking
├── utils/
│   └── validate_submission.py       # Official submission format validator
├── Documentation_template.md        # Comprehensive technical methodology report
├── PROBLEM_STATEMENT.md             # Full Amazon ML Challenge problem specification
├── .gitignore                       # Dataset & binary artifact exclusion rules
└── README.md                        # Project landing page & documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
Clone the repository and install dependencies in Python 3.10+:

```bash
git clone https://github.com/stackSentinel-32/fragment-fuse.git
cd fragment-fuse

pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Dataset Placement
Download and extract the competition dataset into `dataset/`:
```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### 3. Running the Pipeline

The master orchestrator (`pipeline.py`) supports three execution modes:

```bash
# Mode 1: Full end-to-end execution (train + threshold tuning + predict)
python code/business_entity_resolution/src/pipeline.py --mode full

# Mode 2: Train only (builds blocking index, trains model, tunes best threshold)
python code/business_entity_resolution/src/pipeline.py --mode train

# Mode 3: Inference only (uses pre-trained model on test dataset)
python code/business_entity_resolution/src/pipeline.py --mode predict
```

#### Key Command-Line Options:
| Flag | Default | Description |
|---|---|---|
| `--mode` | `full` | Execution mode (`train`, `predict`, `full`) |
| `--data-dir` | `dataset/` | Path to dataset directory |
| `--output-dir` | `output/` | Destination for generated TSV files |
| `--max-candidates` | `50` | Maximum candidate matches per $S_1$ entity |
| `--threshold` | `auto` | Decision threshold override (defaults to tuned value) |

---

## 📊 Evaluation Metric: Macro $F_{0.5}$

Submissions are evaluated on macro-averaged $F_{0.5}$ across all $S_1$ entities:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Why Macro $F_{0.5}$ Matters:
- **Precision Weighting**: Precision is weighted $2\times$ higher than recall. Merging two different businesses (false positive) causes significant damage in enterprise databases.
- **Singleton Handling**: Entities with no true matches (singletons) score:
  - **1.0** if correctly predicted as empty (`""`).
  - **0.0** if any incorrect match is assigned.
- **Strict Macro Average**: Each $S_1$ entity contributes equally to the final score, preventing dense clusters from dominating the metric.

---

## 🔍 Submission Verification

Before submitting predictions to the evaluation portal, verify formatting with the validator:

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

### Validation Checks Enforced:
- [x] Exact tab-separated format (`.tsv`) without commas as column delimiters.
- [x] Every $S_1$ test entity appears exactly once.
- [x] Matched entities are a strict subset of candidate pairs.
- [x] No duplicate entity IDs in lists.
- [x] ID lists reference only valid $S_2$ and $S_3$ IDs.
- [x] Singletons correctly represented by empty values.

---

## ⚖️ Fair Play & Integrity Compliance

- **Zero External APIs**: Strictly self-contained — no external APIs, search engines, geocoding lookups, or third-party web scraping.
- **Permissive Open-Source Model**: Trained purely with LightGBM (MIT License, << 8 Billion parameters).
- **Fully Reproducible**: Deterministic random seeds across data splitting, candidate sampling, and LightGBM training.

---

## 📖 Additional Documentation

- For the complete official challenge guidelines and background, see [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md).
- For in-depth methodology, feature definitions, and ablation studies, refer to [Documentation_template.md](Documentation_template.md).
- For module-level execution and hyperparameter tuning, see [code/business_entity_resolution/README.md](code/business_entity_resolution/README.md).
