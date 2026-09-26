# Business Entity Resolution — ML Challenge 2026

Multi-signal entity resolution pipeline that matches business records across
three independent data sources (S1, S2, S3) using inverted-index blocking
followed by a LightGBM pairwise classifier.

---

## Setup

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
```

**Tested with:** Python 3.10–3.12, Windows / Linux

---

## How to Run

All commands are run from the **project root** (`c:\Projects\amazon_inside_out`).

### Full pipeline (train + predict)
```bash
python code/business_entity_resolution/src/pipeline.py --mode full
```

### Train only (build model from train data)
```bash
python code/business_entity_resolution/src/pipeline.py --mode train
```

### Predict only (requires trained model)
```bash
python code/business_entity_resolution/src/pipeline.py --mode predict
```

### Common options
```
--max-candidates INT    Max candidates per S1 from blocking (default: 50)
--threshold FLOAT       Override decision threshold (default: read from best_threshold.txt)
--data-dir PATH         Path to dataset/ folder (default: dataset/)
--output-dir PATH       Path to output/ folder (default: output/)
```

### Run individual steps
```bash
# Blocking only (train)
python code/business_entity_resolution/src/blocking.py --limit-s1 10000 --limit-s2-s3 500000

# Train model (requires train_candidate_pairs.tsv)
python code/business_entity_resolution/src/train.py

# Inference (requires lgbm_model.pkl)
python code/business_entity_resolution/src/predict.py

# Validate output
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv
```

---

## Pipeline Architecture

```
train_source1.tsv ──┐
train_source2.tsv ──┼──► [1. Blocking]  ──► train_candidate_pairs.tsv
train_source3.tsv ──┘         │
                              │
train_ground_truth.tsv ───────┼──► [2. Features] ──► feature matrix (parquet)
                              │
                              └──► [3. LightGBM]  ──► lgbm_model.pkl
                                        │
                                   [4. Threshold]  ──► best_threshold.txt
                                        │
test_source1.tsv ───┐                   │
test_source2.tsv ───┼──► [5. Blocking + Score] ──► matching_results.tsv
test_source3.tsv ───┘                              candidate_pairs.tsv
```

### Module summary

| File | Purpose |
|---|---|
| `preprocess.py` | Text normalization, abbreviation expansion, feature extraction |
| `blocking.py` | Weighted inverted-index blocking (4-tier, ~99.5% key overlap) |
| `features.py` | 15 pairwise similarity features (name + address + meta) |
| `evaluate.py` | Blocking recall + macro F0.5 scoring |
| `train.py` | LightGBM training, stratified split, threshold tuning |
| `predict.py` | End-to-end inference → submission TSVs |
| `pipeline.py` | Master orchestrator (this file) |

---

## Expected Runtime (your machine — 16 GB RAM)

| Step | Time |
|---|---|
| Preprocess S2 + S3 train (10M rows) | ~4 min |
| Build blocking index (10M entities) | ~40–60 min |
| Retrieve candidates (2.2M S1) | ~30–45 min |
| Build feature matrix (110M pairs) | ~20 min |
| Train LightGBM | ~15–30 min |
| **Total train** | **~2–3 hours** |
| Test blocking + inference | ~1.5–2 hours |
| **Total end-to-end** | **~3.5–5 hours** |

> **Memory note:** Full 10M S2+S3 index needs ~8–12 GB RAM peak.
> If you have < 2 GB free, close Chrome / other apps first, or use
> `--limit-s2-s3 2000000` for a faster (lower-recall) smoke-test run.

### Quick smoke test (~45 min, produces a real submission)
```bash
python code/business_entity_resolution/src/pipeline.py --mode full --limit-s2-s3 2000000
```
*(Add `--limit-s2-s3` by editing pipeline.py `run_blocking` calls or use blocking.py directly)*

---

## Output Files

| File | Description | Required for submission |
|---|---|---|
| `output/matching_results.tsv` | Final predicted matches — **the scored file** | Yes |
| `output/candidate_pairs.tsv` | Blocking candidates (S1 → set of S2/S3 IDs) | Yes |
| `output/lgbm_model.pkl` | Trained LightGBM model + feature column list | No |
| `output/best_threshold.txt` | Optimal classification threshold | No |
| `output/train_candidate_pairs.tsv` | Train blocking output (for reuse) | No |
| `output/train_features_chunk*.parquet` | Chunked feature matrix | No |

### File format
Both TSV files are **tab-separated, UTF-8**, with a header row:

```
# matching_results.tsv
source1_entity_id   matched_entity_ids
S1-000001           S2-012345,S3-067890
S1-000002
...

# candidate_pairs.tsv
source1_entity_id   candidate_entity_ids
S1-000001           S2-012345,S2-099999,S3-067890
S1-000002           S2-111111,S3-222222
...
```

- Every S1 entity in `test_source1.tsv` must have **exactly one row**.
- Empty `matched_entity_ids` = predicted singleton (no match).
- Only `S2-` and `S3-` prefixed IDs are allowed in the match/candidate columns.

---

## Blocking Design

The blocking step uses a **4-tier weighted inverted index**:

| Tier | Key type | Weight | Example |
|---|---|---|---|
| 1 (Exact) | Postcode, street+addr, name-bigram | 10 | `US__pc__10001` |
| 2 (Name tok) | Individual non-generic name tokens | 3 | `US__nm__bakery` |
| 3 (Addr tok) | Individual non-generic address tokens | 2 | `US__ad__westchester` |
| 4 (Bigram) | Char bigrams of norm_name (fallback) | 1 | `US__bg__ba` |

Candidates are ranked by weighted score sum — true matches (which typically
hit multiple key types simultaneously) reliably outrank noise.

**Measured key-overlap rate: 99.5%** on 200 true S1–S2 pairs.

---

## Feature Set (15 features)

**Name features (7):** `name_token_jaccard`, `name_token_sort_ratio`,
`name_partial_ratio`, `name_jaro_winkler`, `name_common_token_count`,
`name_len_diff`, `name_bigram_jaccard`

**Address features (6):** `addr_token_jaccard`, `addr_token_sort_ratio`,
`addr_partial_ratio`, `postcode_match`, `street_num_match`, `addr_len_diff`

**Meta features (2):** `source` (S2=0, S3=1), `country_same`

---

## Leaderboard Metric

**Macro-averaged F0.5** (precision-weighted) across all S1 entities.
Singletons correctly predicted as no-match score 1.0; incorrect singleton
predictions score 0.0.

```
F0.5 = (1 + 0.5²) × P × R / (0.5² × P + R)
```
