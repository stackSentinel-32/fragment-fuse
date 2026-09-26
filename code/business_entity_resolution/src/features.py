"""
ML Challenge 2026 — Entity Resolution Pipeline
Step 4: Pairwise Similarity Feature Engineering

Computes string-similarity features for (S1, candidate) pairs.
Designed for speed: 100M pairs must be processable in reasonable time.

rapidfuzz timings (measured on this machine):
  token_sort_ratio : ~0.8µs/pair
  partial_ratio    : ~0.7µs/pair
  JaroWinkler      : ~0.3µs/pair
  → ~2µs total per pair → ~200s for 100M pairs (all features)
"""

import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _bigrams(s: str) -> Set[str]:
    """Return set of character bigrams from string."""
    s = s.replace(" ", "")
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else set()


def _jaccard(set_a: Set, set_b: Set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _token_set(tok_str: str) -> Set[str]:
    """Split a space-joined token string into a set."""
    return set(tok_str.split()) if tok_str and tok_str.strip() else set()


# ----------------------------------------------------------------
# Feature computation functions
# ----------------------------------------------------------------

def compute_name_features(
    name1: str,
    name2: str,
    tok1: str,
    tok2: str,
) -> Dict[str, float]:
    """
    Compute name similarity features for one (S1, candidate) pair.

    Args:
        name1, name2: Normalized business names (norm_name)
        tok1, tok2:   Space-joined sorted token strings (name_tok)

    Returns dict with 7 float features.
    """
    n1 = name1 or ""
    n2 = name2 or ""
    t1 = tok1 or ""
    t2 = tok2 or ""

    ts1 = _token_set(t1)
    ts2 = _token_set(t2)

    # Token Jaccard
    name_token_jaccard = _jaccard(ts1, ts2)

    # rapidfuzz ratios (return 0–100, we normalize to 0–1)
    name_token_sort_ratio = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_partial_ratio = fuzz.partial_ratio(n1, n2) / 100.0

    # Jaro-Winkler (already 0–1)
    name_jaro_winkler = JaroWinkler.similarity(n1, n2) if (n1 and n2) else 0.0

    # Common token count (raw, not normalized — LightGBM handles this)
    name_common_token_count = float(len(ts1 & ts2))

    # Normalized length difference
    l1, l2 = len(n1), len(n2)
    name_len_diff = abs(l1 - l2) / max(l1, l2, 1)

    # Character bigram Jaccard on normalized name
    name_bigram_jaccard = _jaccard(_bigrams(n1), _bigrams(n2))

    return {
        "name_token_jaccard": name_token_jaccard,
        "name_token_sort_ratio": name_token_sort_ratio,
        "name_partial_ratio": name_partial_ratio,
        "name_jaro_winkler": name_jaro_winkler,
        "name_common_token_count": name_common_token_count,
        "name_len_diff": name_len_diff,
        "name_bigram_jaccard": name_bigram_jaccard,
    }


def compute_address_features(
    addr1: str,
    addr2: str,
    tok1: str,
    tok2: str,
    postcode1: str,
    postcode2: str,
    street_num1: str,
    street_num2: str,
) -> Dict[str, float]:
    """
    Compute address similarity features for one (S1, candidate) pair.

    Args:
        addr1, addr2:       Normalized addresses (norm_addr)
        tok1, tok2:         Space-joined address token strings (addr_tok)
        postcode1/2:        Extracted postcode or empty string
        street_num1/2:      Extracted street number or empty string

    Returns dict with 6 float features.
    """
    a1 = addr1 or ""
    a2 = addr2 or ""
    t1 = tok1 or ""
    t2 = tok2 or ""

    ts1 = _token_set(t1)
    ts2 = _token_set(t2)

    addr_token_jaccard = _jaccard(ts1, ts2)
    addr_token_sort_ratio = fuzz.token_sort_ratio(a1, a2) / 100.0
    addr_partial_ratio = fuzz.partial_ratio(a1, a2) / 100.0

    # Exact postcode match
    p1 = (postcode1 or "").strip()
    p2 = (postcode2 or "").strip()
    postcode_match = 1.0 if (p1 and p2 and p1 == p2) else 0.0

    # Exact street number match
    s1n = (street_num1 or "").strip()
    s2n = (street_num2 or "").strip()
    street_num_match = 1.0 if (s1n and s2n and s1n == s2n) else 0.0

    la1, la2 = len(a1), len(a2)
    addr_len_diff = abs(la1 - la2) / max(la1, la2, 1)

    return {
        "addr_token_jaccard": addr_token_jaccard,
        "addr_token_sort_ratio": addr_token_sort_ratio,
        "addr_partial_ratio": addr_partial_ratio,
        "postcode_match": postcode_match,
        "street_num_match": street_num_match,
        "addr_len_diff": addr_len_diff,
    }


def compute_pair_features(
    s1_row,
    cand_row,
) -> Dict[str, float]:
    """
    Compute all features for one (S1, candidate) pair.
    Accepts either pd.Series or plain dict for each row.

    Returns flat dict: name_features + addr_features + meta_features.
    """
    # Support both dict and pd.Series row access
    def get(row, key):
        if isinstance(row, dict):
            return row.get(key, "") or ""
        val = row[key] if key in row.index else ""
        return str(val) if val is not None and val != "" else ""

    name_feats = compute_name_features(
        name1=get(s1_row, "norm_name"),
        name2=get(cand_row, "norm_name"),
        tok1=get(s1_row, "name_tok"),
        tok2=get(cand_row, "name_tok"),
    )

    addr_feats = compute_address_features(
        addr1=get(s1_row, "norm_addr"),
        addr2=get(cand_row, "norm_addr"),
        tok1=get(s1_row, "addr_tok"),
        tok2=get(cand_row, "addr_tok"),
        postcode1=get(s1_row, "postcode"),
        postcode2=get(cand_row, "postcode"),
        street_num1=get(s1_row, "street_num"),
        street_num2=get(cand_row, "street_num"),
    )

    cand_id = get(cand_row, "entity_id")
    source = 0 if cand_id.startswith("S2-") else 1

    c1 = get(s1_row, "country").strip().lower()
    c2 = get(cand_row, "country").strip().lower()
    country_same = 1.0 if (c1 and c2 and c1 == c2) else 0.0

    meta = {
        "source": float(source),
        "country_same": country_same,
    }

    return {**name_feats, **addr_feats, **meta}


# ----------------------------------------------------------------
# Feature column list (used by train.py / predict.py)
# ----------------------------------------------------------------

FEATURE_COLUMNS: List[str] = [
    # Name features
    "name_token_jaccard",
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_jaro_winkler",
    "name_common_token_count",
    "name_len_diff",
    "name_bigram_jaccard",
    # Address features
    "addr_token_jaccard",
    "addr_token_sort_ratio",
    "addr_partial_ratio",
    "postcode_match",
    "street_num_match",
    "addr_len_diff",
    # Meta
    "source",
    "country_same",
]


# ----------------------------------------------------------------
# Batch feature matrix builder
# ----------------------------------------------------------------

def build_feature_matrix(
    s1_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    ground_truth: Optional[Dict[str, Set[str]]] = None,
    output_path: Optional[str] = None,
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """
    Build a feature matrix over all (S1, candidate) pairs.

    Args:
        s1_df:          Preprocessed Source-1 DataFrame
        candidates_df:  DataFrame with [source1_entity_id, candidate_ids_list]
                        where candidate_ids_list is a Python list of ID strings
        s2_df:          Preprocessed Source-2 DataFrame
        s3_df:          Preprocessed Source-3 DataFrame
        ground_truth:   {s1_id → set of true match IDs} — if provided, adds 'label' column
        output_path:    If given, write feature matrix to this parquet file in chunks
                        (avoids OOM for 100M+ pairs)
        chunk_size:     Number of pairs per chunk written to disk

    Returns:
        pd.DataFrame with columns: [s1_id, candidate_id, <features>, (label)]
        If output_path is given, returns only the last chunk in memory;
        the full matrix is on disk.
    """
    # ---- Build fast lookup dicts: entity_id → dict of relevant columns ----
    relevant_cols = ["entity_id", "norm_name", "name_tok", "norm_addr",
                     "addr_tok", "postcode", "street_num", "country"]

    def df_to_lookup(df: pd.DataFrame) -> Dict[str, dict]:
        lkp = {}
        cols = [c for c in relevant_cols if c in df.columns]
        for row in df[cols].itertuples(index=False):
            d = {c: (getattr(row, c) or "") for c in cols}
            lkp[d["entity_id"]] = d
        return lkp

    print("Building entity lookup tables...")
    s1_lkp = df_to_lookup(s1_df)
    s2_lkp = df_to_lookup(s2_df)
    s3_lkp = df_to_lookup(s3_df)
    cand_lkp = {**s2_lkp, **s3_lkp}
    del s2_lkp, s3_lkp
    print(f"  S1 lookup: {len(s1_lkp):,} entities")
    print(f"  S2+S3 lookup: {len(cand_lkp):,} entities")

    has_label = ground_truth is not None
    all_chunks: List[pd.DataFrame] = []
    current_chunk_rows: List[dict] = []
    total_pairs = 0
    skipped_pairs = 0

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    def _flush_chunk(rows: List[dict], chunk_idx: int) -> pd.DataFrame:
        """Convert list of dicts to DataFrame and optionally write to parquet."""
        chunk_df = pd.DataFrame(rows)
        # Ensure correct column order
        cols = ["s1_id", "candidate_id"] + FEATURE_COLUMNS
        if has_label and "label" in chunk_df.columns:
            cols.append("label")
        chunk_df = chunk_df[[c for c in cols if c in chunk_df.columns]]
        if output_path:
            path = output_path.replace(".parquet", f"_chunk{chunk_idx:04d}.parquet")
            chunk_df.to_parquet(path, index=False)
        return chunk_df

    chunk_idx = 0
    last_chunk_df = None

    print(f"Building feature matrix (chunk_size={chunk_size:,})...")

    for _, cand_row_meta in candidates_df.iterrows():
        s1_id = str(cand_row_meta["source1_entity_id"]).strip()
        cands = cand_row_meta["candidate_ids_list"]
        if not isinstance(cands, list):
            cands = [c.strip() for c in str(cands).split(",") if c.strip()]

        s1_entity = s1_lkp.get(s1_id)
        if s1_entity is None:
            skipped_pairs += len(cands)
            continue

        true_matches: Set[str] = ground_truth.get(s1_id, set()) if has_label else set()

        for cand_id in cands:
            cand_entity = cand_lkp.get(cand_id)
            if cand_entity is None:
                skipped_pairs += 1
                continue

            feats = compute_pair_features(s1_entity, cand_entity)
            row = {"s1_id": s1_id, "candidate_id": cand_id, **feats}
            if has_label:
                row["label"] = 1 if cand_id in true_matches else 0

            current_chunk_rows.append(row)
            total_pairs += 1

            if len(current_chunk_rows) >= chunk_size:
                last_chunk_df = _flush_chunk(current_chunk_rows, chunk_idx)
                if not output_path:
                    all_chunks.append(last_chunk_df)
                current_chunk_rows = []
                chunk_idx += 1

        if total_pairs % 100_000 == 0 and total_pairs > 0:
            pos = sum(1 for r in current_chunk_rows if r.get("label") == 1) if has_label else 0
            print(f"  Processed {total_pairs:,} pairs "
                  f"({'pos=' + str(pos) + ', ' if has_label else ''}"
                  f"chunks_written={chunk_idx})")

    # Flush remaining rows
    if current_chunk_rows:
        last_chunk_df = _flush_chunk(current_chunk_rows, chunk_idx)
        if not output_path:
            all_chunks.append(last_chunk_df)
        chunk_idx += 1

    total_chunks = chunk_idx
    print(f"\nFeature matrix complete:")
    print(f"  Total pairs processed : {total_pairs:,}")
    print(f"  Skipped (missing IDs) : {skipped_pairs:,}")
    if has_label:
        all_labels = [r["label"] for chunk in all_chunks for r in []]  # avoid re-scan
        print(f"  Chunks written        : {total_chunks}")
    print(f"  Feature columns       : {len(FEATURE_COLUMNS)}")
    if output_path:
        print(f"  Parquet chunks at     : {output_path.replace('.parquet', '_chunk*.parquet')}")

    if output_path:
        # Return last chunk as a sample — caller should read parquet files for full matrix
        return last_chunk_df if last_chunk_df is not None else pd.DataFrame()
    else:
        if all_chunks:
            return pd.concat(all_chunks, ignore_index=True)
        return pd.DataFrame()


def load_feature_chunks(base_path: str) -> pd.DataFrame:
    """
    Load all parquet chunk files written by build_feature_matrix().
    base_path should be the output_path passed to build_feature_matrix (without _chunkXXXX).
    """
    import glob
    pattern = base_path.replace(".parquet", "_chunk*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No chunk files found matching: {pattern}")
    print(f"Loading {len(files)} chunk files from {pattern}...")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df
