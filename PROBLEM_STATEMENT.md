# 📋 Amazon ML Challenge 2026: Official Problem Statement

> **Track**: Business Entity Resolution at Scale  
> **Target**: Cross-source record linkage with zero shared identifiers under precision-heavy macro $F_{0.5}$ evaluation.

---

> [!IMPORTANT]
> ### 📢 Crucial Competition Update
> **`candidate_pairs.tsv` is part of your final submission package.**  
> Amazon resolves business entities across billions of records, so pairwise exhaustive comparison is intractable ($\mathcal{O}(N \times M)$). Your blocking step must prune the search space to a small, high-recall candidate set per $S_1$ entity.  
> **Candidate set compactness counts toward final ranking:** Among pipelines with comparable $F_{0.5}$ scores on `matching_results.tsv`, approaches producing smaller candidate sets per entity will be ranked higher.

---

## 🎯 1. Challenge Overview

In large-scale commercial platforms, business identity data arrives asynchronously from multiple independent, noisy channels. Each source contributes partial, inconsistent fragments of information about real-world commercial entities. Crucially, **these fragments share no common identifiers (keys/foreign keys).**

The task of determining which records refer to the same real-world business entity is known as **Entity Resolution (ER)**.

```
       Source 1 (Reference)               Source 2 (Noisy Stream)            Source 3 (Noisy Stream)
  ┌──────────────────────────────┐     ┌──────────────────────────────┐   ┌──────────────────────────────┐
  │ ID: S1-00102                 │     │ ID: S2-04910                 │   │ ID: S3-91823                 │
  │ Name: FedEx Ground Hub       │ ◄───┤ Name: Federal Express Corp   │   │ Name: FedEx Office           │
  │ Addr: 100 Express Rd, Dallas │     │ Addr: 100 Express Road       │   │ Addr: Dallas Fort Worth Area │
  │ Country: US                  │     │ Country: US                  │   │ Country: US                  │
  └──────────────────────────────┘     └──────────────────────────────┘   └──────────────────────────────┘
                 │                                    │                                  │
                 └────────────────────────┬───────────┴──────────────────────────────────┘
                                          ▼
                               [ Entity Resolution ML ]
                                          ▼
                       Match: S1-00102 ──► [S2-04910]
```

### The Three Data Sources

| Source | Role | Nature | Match Cardinality |
|---|---|---|---|
| **Source 1 ($S_1$)** | **Anchor Reference** | Deduplicated reference catalog | Exactly 1 row per entity in output files |
| **Source 2 ($S_2$)** | **Noisy Partner A** | Fragmented, noisy external records | $0$, $1$, or multiple records per $S_1$ entity |
| **Source 3 ($S_3$)** | **Noisy Partner B** | Fragmented, noisy external records | $0$, $1$, or multiple records per $S_1$ entity |

- A single $S_1$ entity may link to **zero** records (singleton), **one** record, or **multiple** records from $S_2$ and/or $S_3$.
- Ground truth is provided for training entities; test entities require model prediction.

---

## 📂 2. Dataset Schema & Structure

All files are strictly **tab-separated (`.tsv`)**. Tabs are mandatory because business names, addresses, and ID lists frequently contain commas.

```python
import pandas as pd

# ALWAYS load TSV files with explicit tab delimiter
df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
```

### 2.1 File Inventory

```
dataset/
├── train/
│   ├── train_source1.tsv          # Deduplicated reference records (~210 MB)
│   ├── train_source2.tsv          # Noisy Source 2 records (~489 MB)
│   ├── train_source3.tsv          # Noisy Source 3 records (~504 MB)
│   └── train_ground_truth.tsv     # True S1 -> [S2, S3] linkage mapping (~127 MB)
└── test/
    ├── test_source1.tsv           # S1 test reference entities (Generate matches for all!)
    ├── test_source2.tsv           # S2 test candidate pool
    └── test_source3.tsv           # S3 test candidate pool
```

### 2.2 Record Columns (`*_source[1|2|3].tsv`)

| Column Name | Data Type | Description & Real-World Characteristics |
|---|---|---|
| **`entity_id`** | String | Unique identifier. Prefix specifies the source: `S1-`, `S2-`, or `S3-`. |
| **`business_name`** | String | Business title. Contains typos, transliterations, legal abbreviations (`Pvt Ltd`, `LLC`, `Corp`), DBA names. |
| **`business_address`** | String | Street address, landmark references (`Near SBI ATM`), missing pincodes/states, formatting variations. |
| **`country`** | String | Country string label. **Train: `US`, `India`**. **Test: `US`, `India`, and `France`** *(unseen)*. |

> [!WARNING]
> **Open-Set Country Distribution Alert**:  
> The test set introduces **France**, which does **not** appear in training data. Do not hardcode, filter, or one-hot encode country to only `['US', 'India']`. Your pipeline must handle unseen countries gracefully.

### 2.3 Ground Truth Columns (`train_ground_truth.tsv`)

| Column | Description | Format |
|---|---|---|
| `source1_entity_id` | $S_1$ entity identifier | e.g. `S1-00001` |
| `matched_entity_ids` | Comma-separated list of true $S_2$ and $S_3$ matches | e.g. `S2-00047,S3-00812` (empty string for singletons) |

---

## 🔍 3. Common Noise & Variation Patterns

1. **Name Inconsistencies**:
   - **Legal Suffixes**: `Corp` vs `Corporation`, `Inc.` vs `Incorporated`, `Pvt` vs `Private`, `Ltd` vs `Limited`.
   - **Trade Names (DBA)**: Registered corporate entity vs operational storefront name.
   - **Punctuation & Symbols**: `&` vs `and`, slashes, hyphens, extra whitespace.
   - **Word Reordering & Typos**: `Federal Express` vs `Express Federal`, phonetic misspellings.

2. **Address Inconsistencies**:
   - **Street Abbreviations**: `Rd` vs `Road`, `St` vs `Street`, `Ave` vs `Avenue`, `Blvd` vs `Boulevard`.
   - **Landmark Descriptors**: Descriptive locations (`Opposite Metro Station`, `Behind Post Office`).
   - **Missing Subcomponents**: Omission of postal codes, states, or suite/apartment numbers.
   - **Transliteration Variants**: Regional language transcriptions into English Latin characters.

---

## 📤 4. Required Output Deliverables

Your inference pipeline must generate **two tab-separated files** in the `output/` directory:

### Deliverable 1: `output/matching_results.tsv` *(Leaderboard Scored)*
Contains your model's final, high-confidence predicted matches.

| Column | Description |
|---|---|
| `source1_entity_id` | Every $S_1$ entity from `test_source1.tsv` (exactly 1 row per entity) |
| `matched_entity_ids` | Comma-separated list of matching $S_2$ and $S_3$ IDs (empty for singletons) |

**Example:**
```tsv
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

### Deliverable 2: `output/candidate_pairs.tsv` *(Blocking Audit)*
The candidate pool produced by your blocking stage *immediately before* pairwise scoring.

| Column | Description |
|---|---|
| `source1_entity_id` | Every $S_1$ entity from `test_source1.tsv` (exactly 1 row per entity) |
| `candidate_entity_ids` | Comma-separated list of all plausible $S_2$/$S_3$ candidates passed to the model |

**Example:**
```tsv
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004,S2-00088
S1-00003	
```

> [!CAUTION]
> **Strict Output Constraints**:
> 1. `matched_entity_ids` must be a strict **subset** of `candidate_entity_ids`.
> 2. Every single $S_1$ entity in `test_source1.tsv` must exist as a row.
> 3. No self-matches ($S_1$ matching $S_1$).
> 4. No duplicate IDs inside any row's list.
> 5. Singletons must be represented by an **empty string** (not `None` or `"nan"`).

---

## 📐 5. Evaluation Metric: Macro $F_{0.5}$

Submissions are evaluated on **Macro-Averaged $F_{\beta}$ Score with $\beta = 0.5$**:

$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Key Metric Properties:
- **Precision Weighted 2× Over Recall**: In production databases, incorrectly merging two separate businesses (false positive) corrupts business integrity far more severely than failing to link them.
- **Macro-Averaged Across Entities**: The metric is calculated independently per $S_1$ record and then averaged across all records:
  $$\text{Macro } F_{0.5} = \frac{1}{N_{S_1}} \sum_{i=1}^{N_{S_1}} F_{0.5}^{(i)}$$
- **Singleton Scoring**:
  - If an $S_1$ entity has no true matches and you predict empty (`""`): **Score = 1.0**.
  - If an $S_1$ entity has no true matches and you predict any match: **Score = 0.0**.

---

## 🛠️ 6. Pre-Submission Local Validation

A standalone validator script [`utils/validate_submission.py`](utils/validate_submission.py) is provided to verify submission files before upload:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Result**:
- Prints `PASS` (exit 0) if compliant.
- Prints numbered violations (exit 1) if any schema or constraint fails.

---

## 📦 7. Final Package Submission Structure

At the conclusion of the challenge, teams submit a reproducible zip archive:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv          # Final leaderboard predictions
│   └── candidate_pairs.tsv           # Blocking candidate pool
├── code/
│   └── business_entity_resolution/
│       ├── src/                      # Complete source code modules
│       ├── README.md                 # End-to-end execution guide
│       └── requirements.txt          # Version-pinned dependencies
└── Documentation_template.md         # Completed methodology report
```

---

## ⚖️ 8. Integrity, Fair Play & Constraints

| Rule | Requirement |
|---|---|
| 🚫 **External Data Lookup** | **STRICTLY FORBIDDEN**. No external APIs, web scraping, government registry lookups, or geocoding services. |
| 🤖 **Model Constraints** | Open-source model license (**MIT or Apache 2.0**) with $\le 8$ Billion parameters. |
| 📊 **Leaderboards** | **Public Leaderboard**: Scored on a subset of the test data during the hackathon. <br>**Private Leaderboard**: Scored on remaining test data for final rankings. |
| 🏆 **Candidate Generation Bonus** | Final evaluation audits `candidate_pairs.tsv`. Pipelines with smaller, cleaner candidate sets rank higher. |