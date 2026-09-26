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
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from tqdm import tqdm

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


def precompute_embeddings(
    df: pd.DataFrame,
    model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    batch_size: int = 512,
    output_path: str = None
) -> np.ndarray:
    """
    Encode each entity as: "{business_name} [SEP] {business_address}"
    using sentence-transformers.
    
    - Process in batches of batch_size
    - Show tqdm progress bar
    - If output_path given, save embeddings as numpy .npy file
    - Returns numpy array of shape (n_entities, embedding_dim)
    """
    from sentence_transformers import SentenceTransformer

    name_col = "business_name" if "business_name" in df.columns else ("norm_name" if "norm_name" in df.columns else None)
    addr_col = "business_address" if "business_address" in df.columns else ("norm_addr" if "norm_addr" in df.columns else None)

    names = df[name_col].fillna("").astype(str).tolist() if name_col else [""] * len(df)
    addrs = df[addr_col].fillna("").astype(str).tolist() if addr_col else [""] * len(df)

    texts = [f"{n} [SEP] {a}" for n, a in zip(names, addrs)]

    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(texts):,} entities (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.save(output_path, embeddings)
        print(f"Saved embeddings ({embeddings.shape}) to: {output_path}")

    return embeddings


def add_embedding_feature(
    s1_idx: int, cand_idx: int,
    s1_embeddings: np.ndarray, cand_embeddings: np.ndarray
) -> float:
    """
    Compute cosine similarity between s1_embeddings[s1_idx] and cand_embeddings[cand_idx].
    Returns float in [-1, 1].
    """
    if s1_idx < 0 or s1_idx >= len(s1_embeddings) or cand_idx < 0 or cand_idx >= len(cand_embeddings):
        return 0.0
    u = s1_embeddings[s1_idx]
    v = cand_embeddings[cand_idx]
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    sim = float(np.dot(u, v) / (norm_u * norm_v))
    return float(np.clip(sim, -1.0, 1.0))


def compute_pair_features(
    s1_row,
    cand_row,
    s1_idx: Optional[int] = None,
    cand_idx: Optional[int] = None,
    s1_embeddings: Optional[np.ndarray] = None,
    cand_embeddings: Optional[np.ndarray] = None,
    embed_cosine: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute all features for one (S1, candidate) pair.
    Accepts either pd.Series or plain dict for each row.
    Optionally computes or attaches multilingual sentence embedding cosine similarity ('embed_cosine').

    Returns flat dict: name_features + addr_features + (embed_cosine) + meta_features.
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

    feats = {**name_feats, **addr_feats, **meta}

    # Optional embedding similarity feature
    if embed_cosine is None and s1_embeddings is not None and cand_embeddings is not None:
        if s1_idx is not None and cand_idx is not None:
            embed_cosine = add_embedding_feature(s1_idx, cand_idx, s1_embeddings, cand_embeddings)

    if embed_cosine is not None:
        feats["embed_cosine"] = float(embed_cosine)

    return feats


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
    # Embedding feature
    "embed_cosine",
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
    s1_embeddings: Optional[Union[np.ndarray, str]] = None,
    cand_embeddings: Optional[Union[np.ndarray, str]] = None,
    s2_embeddings: Optional[Union[np.ndarray, str]] = None,
    s3_embeddings: Optional[Union[np.ndarray, str]] = None,
    use_embeddings: bool = False,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    embedding_batch_size: int = 512,
) -> pd.DataFrame:
    """
    Build a feature matrix over all (S1, candidate) pairs.
    Optionally loads or precomputes multilingual sentence embeddings and attaches 'embed_cosine'.

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
        s1_embeddings:  Array or path to S1 embeddings .npy file
        cand_embeddings: Array or path to combined candidate embeddings .npy file
        s2_embeddings:  Array or path to S2 embeddings .npy file (if separate)
        s3_embeddings:  Array or path to S3 embeddings .npy file (if separate)
        use_embeddings: If True and embeddings not provided, computes them on the fly
        embedding_model: HuggingFace model name for sentence-transformers
        embedding_batch_size: Batch size for embedding encoding

    Returns:
        pd.DataFrame with columns: [s1_id, candidate_id, <features>, (label)]
        If output_path is given, returns only the last chunk in memory;
        the full matrix is on disk.
    """
    # ---- Load or precompute embeddings if requested ----
    if isinstance(s1_embeddings, str) and os.path.isfile(s1_embeddings):
        print(f"Loading S1 embeddings from: {s1_embeddings}")
        s1_embeddings = np.load(s1_embeddings)
    elif use_embeddings and s1_embeddings is None:
        print("Precomputing S1 embeddings...")
        s1_embeddings = precompute_embeddings(s1_df, model_name=embedding_model, batch_size=embedding_batch_size)

    if isinstance(cand_embeddings, str) and os.path.isfile(cand_embeddings):
        print(f"Loading candidate embeddings from: {cand_embeddings}")
        cand_embeddings = np.load(cand_embeddings)
    elif cand_embeddings is None:
        if isinstance(s2_embeddings, str) and os.path.isfile(s2_embeddings):
            s2_embeddings = np.load(s2_embeddings)
        if isinstance(s3_embeddings, str) and os.path.isfile(s3_embeddings):
            s3_embeddings = np.load(s3_embeddings)

        if s2_embeddings is not None and s3_embeddings is not None:
            cand_embeddings = np.vstack([s2_embeddings, s3_embeddings])
        elif use_embeddings:
            print("Precomputing S2 & S3 embeddings...")
            s2_emb = precompute_embeddings(s2_df, model_name=embedding_model, batch_size=embedding_batch_size)
            s3_emb = precompute_embeddings(s3_df, model_name=embedding_model, batch_size=embedding_batch_size)
            cand_embeddings = np.vstack([s2_emb, s3_emb])

    has_embeddings = (s1_embeddings is not None and cand_embeddings is not None)
    s1_id_to_idx = {}
    cand_id_to_idx = {}
    if has_embeddings:
        print(f"  Embedding feature active: S1 {s1_embeddings.shape}, Candidates {cand_embeddings.shape}")
        s1_id_to_idx = {eid: idx for idx, eid in enumerate(s1_df["entity_id"])}
        s2_id_to_idx = {eid: idx for idx, eid in enumerate(s2_df["entity_id"])}
        s3_offset = len(s2_df)
        s3_id_to_idx = {eid: idx + s3_offset for idx, eid in enumerate(s3_df["entity_id"])}
        cand_id_to_idx = {**s2_id_to_idx, **s3_id_to_idx}

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
        cols = ["s1_id", "candidate_id"] + [c for c in FEATURE_COLUMNS if c in chunk_df.columns]
        if has_label and "label" in chunk_df.columns:
            cols.append("label")
        chunk_df = chunk_df[cols]
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

        s1_idx = s1_id_to_idx.get(s1_id) if has_embeddings else None
        true_matches: Set[str] = ground_truth.get(s1_id, set()) if has_label else set()

        for cand_id in cands:
            cand_entity = cand_lkp.get(cand_id)
            if cand_entity is None:
                skipped_pairs += 1
                continue

            cand_idx = cand_id_to_idx.get(cand_id) if has_embeddings else None
            emb_cosine = None
            if has_embeddings and s1_idx is not None and cand_idx is not None:
                emb_cosine = add_embedding_feature(s1_idx, cand_idx, s1_embeddings, cand_embeddings)

            feats = compute_pair_features(
                s1_entity,
                cand_entity,
                embed_cosine=emb_cosine,
            )
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
