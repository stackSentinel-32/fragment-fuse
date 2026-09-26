"""
ML Challenge 2026 — Entity Resolution Pipeline
Step 6: Inference — Blocking → Feature Scoring → Thresholding → Output Files

Generates two submission files:
  output/candidate_pairs.tsv      — source1_entity_id TAB candidate_entity_ids
  output/matching_results.tsv     — source1_entity_id TAB matched_entity_ids
"""

import os
import sys
import csv
import pickle
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from preprocess import load_and_preprocess
from blocking import BlockingIndex, run_blocking
from features import compute_pair_features, FEATURE_COLUMNS
from evaluate import load_candidate_pairs


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _load_model(model_path: str):
    """Load pickled LightGBM model + feature column list."""
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    model = obj["model"]
    feature_cols = obj["feature_columns"]
    print(f"  Loaded model from: {model_path}")
    print(f"  Features: {len(feature_cols)}  |  Best iteration: {getattr(model, 'best_iteration_', 'N/A')}")
    return model, feature_cols


def _build_entity_lookup(df: pd.DataFrame) -> Dict[str, dict]:
    """Build {entity_id → dict of row fields} for fast pair lookup."""
    cols = ["entity_id", "norm_name", "name_tok", "norm_addr",
            "addr_tok", "postcode", "street_num", "country"]
    cols = [c for c in cols if c in df.columns]
    lkp = {}
    for row in df[cols].itertuples(index=False):
        d = {c: (getattr(row, c) or "") for c in cols}
        lkp[d["entity_id"]] = d
    return lkp


# ----------------------------------------------------------------
# Main inference function
# ----------------------------------------------------------------

def predict(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    model_path: str = "output/lgbm_model.pkl",
    threshold: float = 0.5,
    max_candidates: int = 50,
    output_dir: str = "output",
    nrows_s2_s3: Optional[int] = None,
    candidates_path: Optional[str] = None,   # pre-built candidate file (skip blocking)
) -> Dict[str, Dict]:
    """
    Full inference pipeline: Blocking → Feature scoring → Thresholding → Write outputs.

    Args:
        s1_path:          Path to test_source1.tsv
        s2_path:          Path to test_source2.tsv
        s3_path:          Path to test_source3.tsv
        model_path:       Path to trained lgbm_model.pkl
        threshold:        Decision threshold (score >= threshold → match)
        max_candidates:   Max candidates per S1 from blocking
        output_dir:       Directory to write output TSV files
        nrows_s2_s3:      Limit S2/S3 rows (for testing; None = full)
        candidates_path:  If given, skip blocking and load this candidate file

    Returns:
        dict with output file paths and summary stats
    """
    os.makedirs(output_dir, exist_ok=True)
    t_total = time.time()

    # ----------------------------------------------------------------
    # STEP A — Blocking
    # ----------------------------------------------------------------
    cand_output_path = os.path.join(output_dir, "candidate_pairs.tsv")

    if candidates_path and os.path.isfile(candidates_path):
        print("=== Step A: Loading Pre-built Candidates ===")
        print(f"  Skipping blocking — loading: {candidates_path}")
        candidates_df = load_candidate_pairs(candidates_path)
        print(f"  Loaded {len(candidates_df):,} S1 candidate rows")

        # Also need S1 for lookup later — just load IDs
        print("  Loading S1 for entity lookup...")
        s1_df = load_and_preprocess(s1_path)
    else:
        print("=== Step A: Blocking ===")
        candidates_df = run_blocking(
            s1_path=s1_path,
            s2_path=s2_path,
            s3_path=s3_path,
            output_path=cand_output_path,
            max_candidates=max_candidates,
            nrows_s2_s3=nrows_s2_s3,
        )
        print(f"  Blocking complete: {len(candidates_df):,} S1 entities")
        s1_df = load_and_preprocess(s1_path)

    all_s1_ids = list(s1_df["entity_id"])

    # ----------------------------------------------------------------
    # STEP B — Load model + build lookups
    # ----------------------------------------------------------------
    print("\n=== Step B: Loading Model & Building Lookups ===")
    model, feature_cols = _load_model(model_path)

    s1_lkp = _build_entity_lookup(s1_df)
    del s1_df

    print("  Loading S2 + S3 for feature lookup...")
    s2_df = load_and_preprocess(s2_path, nrows=nrows_s2_s3)
    s3_df = load_and_preprocess(s3_path, nrows=nrows_s2_s3)
    cand_lkp = {**_build_entity_lookup(s2_df), **_build_entity_lookup(s3_df)}
    del s2_df, s3_df
    print(f"  S2+S3 lookup: {len(cand_lkp):,} entities")

    # ----------------------------------------------------------------
    # STEP B+C — Score pairs & threshold per S1 entity
    # ----------------------------------------------------------------
    print(f"\n=== Step B+C: Feature Scoring (threshold={threshold}) ===")

    # Build candidate lookup: s1_id → list of candidate_ids
    cand_map: Dict[str, List[str]] = {}
    for _, row in candidates_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        cands = row["candidate_ids_list"]
        if not isinstance(cands, list):
            cands = [c.strip() for c in str(cands).split(",") if c.strip()]
        cand_map[s1_id] = cands

    predictions: Dict[str, List[str]] = {}  # s1_id → sorted list of matched IDs
    total_pairs = 0
    skipped = 0
    n_with_match = 0

    BATCH = 500_000   # Score in batches to avoid RAM spikes
    batch_feats: List[dict] = []
    batch_meta: List[Tuple[str, str]] = []   # (s1_id, candidate_id)

    def _flush_batch():
        nonlocal total_pairs, n_with_match
        if not batch_feats:
            return
        X = pd.DataFrame(batch_feats)[feature_cols].astype(np.float32)
        scores = model.predict_proba(X)[:, 1]
        # Group by s1_id
        score_map: Dict[str, List[Tuple[str, float]]] = {}
        for (s1_id, cid), score in zip(batch_meta, scores):
            score_map.setdefault(s1_id, []).append((cid, float(score)))
        # Apply threshold and store results
        for s1_id, scored_cands in score_map.items():
            above = [(cid, sc) for cid, sc in scored_cands if sc >= threshold]
            above_sorted = [cid for cid, _ in sorted(above, key=lambda x: -x[1])]
            if s1_id not in predictions:
                predictions[s1_id] = above_sorted
            else:
                # Merge (from previous batch for same S1)
                seen = set(predictions[s1_id])
                predictions[s1_id].extend([c for c in above_sorted if c not in seen])
        total_pairs += len(batch_feats)
        batch_feats.clear()
        batch_meta.clear()

    t_score = time.time()
    processed_s1 = 0

    for _, cand_row in candidates_df.iterrows():
        s1_id = str(cand_row["source1_entity_id"]).strip()
        cands = cand_row["candidate_ids_list"]
        if not isinstance(cands, list):
            cands = [c.strip() for c in str(cands).split(",") if c.strip()]

        s1_entity = s1_lkp.get(s1_id)
        if s1_entity is None:
            skipped += len(cands)
            predictions.setdefault(s1_id, [])
            continue

        for cid in cands:
            cand_entity = cand_lkp.get(cid)
            if cand_entity is None:
                skipped += 1
                continue
            feats = compute_pair_features(s1_entity, cand_entity)
            batch_feats.append(feats)
            batch_meta.append((s1_id, cid))

            if len(batch_feats) >= BATCH:
                _flush_batch()

        processed_s1 += 1
        if processed_s1 % 100_000 == 0:
            print(f"  Scored {processed_s1:,} / {len(candidates_df):,} S1 entities "
                  f"({total_pairs:,} pairs)...")

    _flush_batch()  # Flush remaining

    score_elapsed = time.time() - t_score
    print(f"\n  Scoring complete: {total_pairs:,} pairs in {score_elapsed:.1f}s "
          f"({total_pairs/max(score_elapsed,1):,.0f} pairs/sec)")

    # ----------------------------------------------------------------
    # STEP D — Write output files
    # ----------------------------------------------------------------
    print("\n=== Step D: Writing Output Files ===")

    matching_path = os.path.join(output_dir, "matching_results.tsv")
    # candidate_pairs.tsv already written by blocking; but if we loaded pre-built,
    # we write it fresh here to ensure it includes ALL S1 entities
    cand_out_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # Ensure every S1 entity has a row (even those with no candidates)
    s1_id_set = set(all_s1_ids)

    n_with_match = 0
    total_matches = 0

    with (
        open(matching_path, "w", encoding="utf-8", newline="") as match_f,
        open(cand_out_path, "w", encoding="utf-8", newline="") as cand_f,
    ):
        match_writer = csv.writer(match_f, delimiter="\t")
        cand_writer = csv.writer(cand_f, delimiter="\t")

        match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        cand_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for s1_id in all_s1_ids:
            matched = predictions.get(s1_id, [])
            raw_cands = cand_map.get(s1_id, [])

            matched_str = ",".join(matched)
            cands_str = ",".join(raw_cands)

            match_writer.writerow([s1_id, matched_str])
            cand_writer.writerow([s1_id, cands_str])

            if matched:
                n_with_match += 1
                total_matches += len(matched)

    n_singletons = len(all_s1_ids) - n_with_match
    avg_matches = total_matches / max(n_with_match, 1)
    total_elapsed = time.time() - t_total

    print(f"\n=== Inference Summary ===")
    print(f"  Total S1 entities          : {len(all_s1_ids):,}")
    print(f"  Entities with >=1 match    : {n_with_match:,}  ({n_with_match/len(all_s1_ids)*100:.1f}%)")
    print(f"  Singletons predicted       : {n_singletons:,}  ({n_singletons/len(all_s1_ids)*100:.1f}%)")
    print(f"  Avg matched per non-single : {avg_matches:.2f}")
    print(f"  Total elapsed              : {total_elapsed:.1f}s")
    print(f"\n  matching_results.tsv  -> {matching_path}")
    print(f"  candidate_pairs.tsv   -> {cand_out_path}")

    return {
        "matching_results_path": matching_path,
        "candidate_pairs_path": cand_out_path,
        "n_s1_entities": len(all_s1_ids),
        "n_with_match": n_with_match,
        "n_singletons": n_singletons,
        "avg_matches": avg_matches,
        "total_pairs_scored": total_pairs,
    }


# ----------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference and generate submission files")
    parser.add_argument("--s1", default=os.path.join("dataset", "test", "test_source1.tsv"))
    parser.add_argument("--s2", default=os.path.join("dataset", "test", "test_source2.tsv"))
    parser.add_argument("--s3", default=os.path.join("dataset", "test", "test_source3.tsv"))
    parser.add_argument("--model", default=os.path.join("output", "lgbm_model.pkl"))
    parser.add_argument("--threshold-file", default=os.path.join("output", "best_threshold.txt"))
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override threshold (default: read from best_threshold.txt)")
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--candidates", default=None,
                        help="Pre-built candidate_pairs.tsv (skips blocking step)")
    parser.add_argument("--limit-s2-s3", type=int, default=None,
                        help="Limit S2/S3 rows for fast testing")
    args = parser.parse_args()

    # Read threshold
    if args.threshold is not None:
        threshold = args.threshold
        print(f"Using threshold from CLI: {threshold}")
    elif os.path.isfile(args.threshold_file):
        with open(args.threshold_file) as f:
            threshold = float(f.read().strip())
        print(f"Loaded threshold from {args.threshold_file}: {threshold}")
    else:
        threshold = 0.5
        print(f"Threshold file not found — using default: {threshold}")

    stats = predict(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        model_path=args.model,
        threshold=threshold,
        max_candidates=args.max_candidates,
        output_dir=args.output_dir,
        nrows_s2_s3=args.limit_s2_s3,
        candidates_path=args.candidates,
    )

    print("\n=== DONE ===")
    print(f"  matching_results.tsv : {stats['matching_results_path']}")
    print(f"  candidate_pairs.tsv  : {stats['candidate_pairs_path']}")
