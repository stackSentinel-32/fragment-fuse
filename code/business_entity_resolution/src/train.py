"""
ML Challenge 2026 — Entity Resolution Pipeline
Step 5: LightGBM Model Training, Threshold Tuning

Trains a binary classifier on (S1, candidate) feature pairs:
  label = 1  →  true match
  label = 0  →  not a match (blocking false positive)

Uses LightGBM for speed (trains on 100M+ pairs in ~30 min on CPU).
"""

import os
import sys
import pickle
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from preprocess import load_and_preprocess
from evaluate import (
    load_ground_truth,
    load_candidate_pairs,
    evaluate_predictions,
    print_f05_report,
)
from features import build_feature_matrix, load_feature_chunks, FEATURE_COLUMNS


# ----------------------------------------------------------------
# Training data preparation
# ----------------------------------------------------------------

def prepare_training_data(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    gt_path: str,
    candidates_df: pd.DataFrame,
    val_fraction: float = 0.1,
    output_dir: str = "output",
    nrows_s2_s3: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load sources, split S1 into train/val (stratified by singleton status),
    build feature matrices for both splits, save as parquet, return both DataFrames.
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    print("=== Prepare Training Data ===")
    print("Loading ground truth...")
    gt = load_ground_truth(gt_path)  # {s1_id → set of matched IDs}

    # ---- Stratified train/val split on S1 entities ----
    # Separate singletons from non-singletons to maintain ratio
    all_s1_ids = list(gt.keys())
    singleton_ids = [s for s in all_s1_ids if not gt[s]]
    match_ids = [s for s in all_s1_ids if gt[s]]

    rng = np.random.default_rng(42)
    rng.shuffle(singleton_ids)
    rng.shuffle(match_ids)

    n_val_s = max(1, int(len(singleton_ids) * val_fraction))
    n_val_m = max(1, int(len(match_ids) * val_fraction))

    val_s1_ids = set(singleton_ids[:n_val_s] + match_ids[:n_val_m])
    train_s1_ids = set(singleton_ids[n_val_s:] + match_ids[n_val_m:])

    print(f"  Train S1 entities : {len(train_s1_ids):,}  "
          f"(singletons: {sum(1 for s in train_s1_ids if not gt[s]):,})")
    print(f"  Val   S1 entities : {len(val_s1_ids):,}  "
          f"(singletons: {sum(1 for s in val_s1_ids if not gt[s]):,})")

    # ---- Filter candidates_df by split ----
    train_cands = candidates_df[
        candidates_df["source1_entity_id"].isin(train_s1_ids)
    ].copy()
    val_cands = candidates_df[
        candidates_df["source1_entity_id"].isin(val_s1_ids)
    ].copy()

    print(f"\n  Train candidates  : {len(train_cands):,} S1 rows")
    print(f"  Val   candidates  : {len(val_cands):,} S1 rows")

    # ---- Load and preprocess source files ----
    print("\nLoading & preprocessing source files...")
    s1_df = load_and_preprocess(s1_path)
    print(f"  S1: {len(s1_df):,} rows")
    s2_df = load_and_preprocess(s2_path, nrows=nrows_s2_s3)
    print(f"  S2: {len(s2_df):,} rows")
    s3_df = load_and_preprocess(s3_path, nrows=nrows_s2_s3)
    print(f"  S3: {len(s3_df):,} rows")

    # ---- Build feature matrices ----
    train_out = os.path.join(output_dir, "train_features.parquet")
    val_out = os.path.join(output_dir, "val_features.parquet")

    print("\nBuilding TRAIN feature matrix...")
    train_df = build_feature_matrix(
        s1_df, train_cands, s2_df, s3_df,
        ground_truth=gt,
        output_path=train_out,
    )
    # Re-load all chunks if written to disk
    if os.path.isfile(train_out.replace(".parquet", "_chunk0000.parquet")):
        train_df = load_feature_chunks(train_out)

    print("\nBuilding VAL feature matrix...")
    val_df = build_feature_matrix(
        s1_df, val_cands, s2_df, s3_df,
        ground_truth=gt,
        output_path=val_out,
    )
    if os.path.isfile(val_out.replace(".parquet", "_chunk0000.parquet")):
        val_df = load_feature_chunks(val_out)

    # ---- Report ----
    elapsed = time.time() - t0
    n_train = len(train_df)
    n_val = len(val_df)
    pos_rate_train = train_df["label"].mean() if "label" in train_df else 0
    pos_rate_val = val_df["label"].mean() if "label" in val_df else 0

    print(f"\n=== Training Data Summary ===")
    print(f"  Train pairs : {n_train:,}  (positive rate: {pos_rate_train*100:.2f}%)")
    print(f"  Val   pairs : {n_val:,}  (positive rate: {pos_rate_val*100:.2f}%)")
    print(f"  Features    : {len(FEATURE_COLUMNS)}")
    print(f"  Elapsed     : {elapsed:.1f}s")

    return train_df, val_df


# ----------------------------------------------------------------
# Model training
# ----------------------------------------------------------------

def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_output_path: str = "output/lgbm_model.pkl",
) -> Tuple[lgb.LGBMClassifier, List[str]]:
    """
    Train a LightGBM binary classifier on (train_df) with early stopping on (val_df).
    Saves model to model_output_path. Returns (model, feature_columns).
    """
    os.makedirs(os.path.dirname(os.path.abspath(model_output_path)), exist_ok=True)

    # Identify feature columns
    exclude = {"s1_id", "candidate_id", "label"}
    feature_cols = [c for c in train_df.columns if c not in exclude]
    # Ensure consistent ordering with FEATURE_COLUMNS
    feature_cols = [c for c in FEATURE_COLUMNS if c in feature_cols]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.int32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["label"].values.astype(np.int32)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    print(f"\n=== Training LightGBM ===")
    print(f"  Train: {len(X_train):,} pairs  ({n_pos:,} pos / {n_neg:,} neg)")
    print(f"  Val:   {len(X_val):,} pairs  ({y_val.sum():,} pos)")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
    print(f"  Features: {len(feature_cols)}")

    model = lgb.LGBMClassifier(
        objective="binary",
        metric=["binary_logloss", "auc"],
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"  Best iteration: {model.best_iteration_}")

    # Val AUC
    from sklearn.metrics import roc_auc_score
    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"  Val AUC: {val_auc:.4f}")

    # Feature importance
    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print(f"\n  Top-{min(20, len(importances))} Feature Importances:")
    for name, imp in importances[:20]:
        bar = "=" * int(imp / max(importances[0][1], 1) * 30)
        print(f"    {name:<30} {imp:>6}  {bar}")

    # Save model
    with open(model_output_path, "wb") as f:
        pickle.dump({"model": model, "feature_columns": feature_cols}, f)
    print(f"\n  Model saved to: {model_output_path}")

    return model, feature_cols


# ----------------------------------------------------------------
# Threshold tuning
# ----------------------------------------------------------------

def tune_threshold(
    model: lgb.LGBMClassifier,
    feature_columns: List[str],
    val_df: pd.DataFrame,
    s1_val_ids: List[str],
    ground_truth: Dict[str, Set[str]],
) -> float:
    """
    Sweep threshold 0.20 → 0.80 in steps of 0.05.
    For each, build predictions dict and compute macro F0.5.
    Prints a formatted table. Returns the best threshold.
    """
    print("\n=== Threshold Tuning ===")

    X_val = val_df[feature_columns].astype(np.float32)
    val_probs = model.predict_proba(X_val)[:, 1]

    # Attach probabilities to val_df for easy groupby
    scored = val_df[["s1_id", "candidate_id"]].copy()
    scored["score"] = val_probs

    # Only evaluate S1 IDs present in val_df
    s1_ids_in_val = set(scored["s1_id"].unique())
    gt_val = {k: v for k, v in ground_truth.items() if k in s1_ids_in_val}
    # Add singletons that had no candidates (they appear in GT but not in val_df)
    val_id_set = set(s1_val_ids)
    for s1_id in val_id_set:
        if s1_id not in gt_val:
            gt_val[s1_id] = ground_truth.get(s1_id, set())

    # Group scores by s1_id
    groups = scored.groupby("s1_id")

    thresholds = [round(t, 2) for t in np.arange(0.20, 0.81, 0.05)]

    print(f"\n  {'Threshold':>10}  {'F0.5':>8}  {'Precision':>10}  {'Recall':>8}  {'Avg Matches':>12}")
    print("  " + "-" * 56)

    best_f05 = -1.0
    best_threshold = 0.5

    for threshold in thresholds:
        predictions: Dict[str, List[str]] = {}
        for s1_id in gt_val:
            if s1_id in groups.groups:
                grp = groups.get_group(s1_id)
                matched = grp.loc[grp["score"] >= threshold, "candidate_id"].tolist()
                predictions[s1_id] = matched
            else:
                predictions[s1_id] = []

        stats = evaluate_predictions(predictions, gt_val)
        f05 = stats["macro_f05"]
        p = stats["macro_precision"]
        r = stats["macro_recall"]
        non_empty = {k: v for k, v in predictions.items() if v}
        avg_matches = (
            sum(len(v) for v in non_empty.values()) / len(non_empty)
            if non_empty else 0
        )

        marker = " <-- best" if f05 > best_f05 else ""
        print(f"  {threshold:>10.2f}  {f05:>8.4f}  {p:>10.4f}  {r:>8.4f}  {avg_matches:>12.2f}{marker}")

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = threshold

    print(f"\n  Best threshold: {best_threshold:.2f}  (F0.5 = {best_f05:.4f})")
    return best_threshold


# ----------------------------------------------------------------
# Main: full training pipeline
# ----------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LightGBM entity matching model")
    parser.add_argument("--s1", default=os.path.join("dataset", "train", "train_source1.tsv"))
    parser.add_argument("--s2", default=os.path.join("dataset", "train", "train_source2.tsv"))
    parser.add_argument("--s3", default=os.path.join("dataset", "train", "train_source3.tsv"))
    parser.add_argument("--gt", default=os.path.join("dataset", "train", "train_ground_truth.tsv"))
    parser.add_argument("--candidates", default=os.path.join("output", "train_candidate_pairs.tsv"))
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--model-path", default=os.path.join("output", "lgbm_model.pkl"))
    parser.add_argument("--threshold-path", default=os.path.join("output", "best_threshold.txt"))
    parser.add_argument("--limit-s2-s3", type=int, default=None,
                        help="Limit S2/S3 rows loaded (for fast testing)")
    parser.add_argument("--skip-data-prep", action="store_true",
                        help="Skip data prep and load existing parquet files")
    args = parser.parse_args()

    # ---- Step 1: Load candidate pairs ----
    if not os.path.isfile(args.candidates):
        print(f"ERROR: Candidate file not found: {args.candidates}")
        print("Run blocking.py first to generate train_candidate_pairs.tsv")
        sys.exit(1)

    print(f"Loading candidate pairs from {args.candidates}...")
    candidates_df = load_candidate_pairs(args.candidates)
    print(f"Loaded {len(candidates_df):,} candidate rows")

    gt = load_ground_truth(args.gt)

    # ---- Step 2: Prepare training data ----
    train_feats_path = os.path.join(args.output_dir, "train_features.parquet")
    val_feats_path = os.path.join(args.output_dir, "val_features.parquet")

    if args.skip_data_prep and (
        os.path.isfile(train_feats_path) or
        os.path.isfile(train_feats_path.replace(".parquet", "_chunk0000.parquet"))
    ):
        print("\nSkipping data prep (--skip-data-prep). Loading existing feature files...")
        train_df = load_feature_chunks(train_feats_path)
        val_df = load_feature_chunks(val_feats_path)
    else:
        train_df, val_df = prepare_training_data(
            s1_path=args.s1,
            s2_path=args.s2,
            s3_path=args.s3,
            gt_path=args.gt,
            candidates_df=candidates_df,
            val_fraction=args.val_fraction,
            output_dir=args.output_dir,
            nrows_s2_s3=args.limit_s2_s3,
        )

    # ---- Step 3: Train model ----
    model, feature_cols = train_model(
        train_df=train_df,
        val_df=val_df,
        model_output_path=args.model_path,
    )

    # ---- Step 4: Tune threshold ----
    # Build val S1 ID list (all S1 IDs in the val split)
    val_s1_ids = list(val_df["s1_id"].unique())

    best_threshold = tune_threshold(
        model=model,
        feature_columns=feature_cols,
        val_df=val_df,
        s1_val_ids=val_s1_ids,
        ground_truth=gt,
    )

    # ---- Step 5: Save best threshold ----
    os.makedirs(os.path.dirname(os.path.abspath(args.threshold_path)), exist_ok=True)
    with open(args.threshold_path, "w") as f:
        f.write(str(best_threshold))
    print(f"\nBest threshold saved to: {args.threshold_path}")
    print(f"\n=== Training pipeline complete ===")
    print(f"  Model:     {args.model_path}")
    print(f"  Threshold: {best_threshold}")
