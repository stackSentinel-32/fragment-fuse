"""
ML Challenge 2026 — Entity Resolution Pipeline
Step 7: Master Orchestration Script

Usage:
  python src/pipeline.py --mode full           # train + predict end-to-end
  python src/pipeline.py --mode train          # build model only
  python src/pipeline.py --mode predict        # generate test predictions (needs model)

Modes:
  train  — blocking on train data, feature matrix, LightGBM, threshold tuning
  predict — blocking on test data, score pairs, write submission files
  full   — train then predict
"""

import os
import sys
import time
import subprocess
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from preprocess import load_and_preprocess
from blocking import run_blocking
from evaluate import (
    load_ground_truth,
    load_candidate_pairs,
    blocking_recall,
    print_blocking_report,
)
from train import prepare_training_data, train_model, tune_threshold
from predict import predict


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def banner(title: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def section(title: str):
    print(f"\n--- {title} ---")


def elapsed(t0: float) -> str:
    secs = time.time() - t0
    mins, secs = divmod(int(secs), 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}h {mins}m {secs}s"
    if mins:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def run_validator(
    matching_path: str,
    candidate_path: str,
    test_dir: str,
) -> bool:
    """
    Run the submission validator and print results.
    Returns True if PASS, False if FAIL.
    """
    validator = os.path.join(
        os.path.dirname(CURRENT_DIR),   # src/ -> code/business_entity_resolution/
        "..", "..",                      # -> project root
        "utils", "validate_submission.py",
    )
    validator = os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", "..", "utils", "validate_submission.py"))

    if not os.path.isfile(validator):
        print(f"  [WARNING] Validator not found at: {validator}")
        print("  Run manually: python utils/validate_submission.py --matching output/matching_results.tsv")
        return True  # don't block on missing validator

    cmd = [
        sys.executable, validator,
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir,
    ]
    print(f"  Running: {' '.join(os.path.basename(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


# ----------------------------------------------------------------
# Train mode
# ----------------------------------------------------------------

def run_train(args) -> dict:
    """
    Blocking on train data + prepare training features + LightGBM + threshold tuning.
    """
    t0 = time.time()
    train_dir = os.path.join(args.data_dir, "train")
    os.makedirs(args.output_dir, exist_ok=True)

    s1_train = os.path.join(train_dir, "train_source1.tsv")
    s2_train = os.path.join(train_dir, "train_source2.tsv")
    s3_train = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    train_cand_path = os.path.join(args.output_dir, "train_candidate_pairs.tsv")
    model_path = os.path.join(args.output_dir, "lgbm_model.pkl")
    threshold_path = os.path.join(args.output_dir, "best_threshold.txt")

    # ---- Step 1: Blocking on train data ----
    banner("Step 1 / 4 — Blocking (Train Data)")
    candidates_df = run_blocking(
        s1_path=s1_train,
        s2_path=s2_train,
        s3_path=s3_train,
        output_path=train_cand_path,
        max_candidates=args.max_candidates,
    )
    print(f"Train blocking done. [{elapsed(t0)}]")

    # ---- Step 2: Evaluate blocking recall ----
    banner("Step 2 / 4 — Blocking Recall (Train)")
    gt = load_ground_truth(gt_path)
    evaluated_ids = set(candidates_df["source1_entity_id"])
    gt_subset = {k: v for k, v in gt.items() if k in evaluated_ids}
    recall_stats = blocking_recall(candidates_df, gt_subset)
    print_blocking_report(recall_stats)

    if recall_stats["overall_recall"] < 0.80:
        print("\n[WARNING] Blocking recall < 80%. Model accuracy will be limited.")
        print("  Consider: increasing --max-candidates or improving blocking keys.")

    # ---- Step 3: Prepare training data + train model ----
    banner("Step 3 / 4 — Feature Engineering + LightGBM Training")
    train_df, val_df = prepare_training_data(
        s1_path=s1_train,
        s2_path=s2_train,
        s3_path=s3_train,
        gt_path=gt_path,
        candidates_df=candidates_df,
        val_fraction=0.1,
        output_dir=args.output_dir,
    )
    model, feature_cols = train_model(
        train_df=train_df,
        val_df=val_df,
        model_output_path=model_path,
    )
    print(f"Training done. [{elapsed(t0)}]")

    # ---- Step 4: Threshold tuning ----
    banner("Step 4 / 4 — Threshold Tuning")
    val_s1_ids = list(val_df["s1_id"].unique())
    best_threshold = tune_threshold(
        model=model,
        feature_columns=feature_cols,
        val_df=val_df,
        s1_val_ids=val_s1_ids,
        ground_truth=gt,
    )

    with open(threshold_path, "w") as f:
        f.write(str(best_threshold))
    print(f"\nBest threshold: {best_threshold}  (saved to {threshold_path})")
    print(f"\n[Train mode complete — {elapsed(t0)}]")

    return {
        "model_path": model_path,
        "threshold": best_threshold,
        "threshold_path": threshold_path,
        "blocking_recall": recall_stats["overall_recall"],
    }


# ----------------------------------------------------------------
# Predict mode
# ----------------------------------------------------------------

def run_predict(args, threshold: float = None) -> dict:
    """
    Blocking on test data + feature scoring + write submission files.
    """
    t0 = time.time()
    test_dir = os.path.join(args.data_dir, "test")
    os.makedirs(args.output_dir, exist_ok=True)

    s1_test = os.path.join(test_dir, "test_source1.tsv")
    s2_test = os.path.join(test_dir, "test_source2.tsv")
    s3_test = os.path.join(test_dir, "test_source3.tsv")
    model_path = os.path.join(args.output_dir, "lgbm_model.pkl")
    threshold_path = os.path.join(args.output_dir, "best_threshold.txt")
    matching_path = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_path = os.path.join(args.output_dir, "candidate_pairs.tsv")

    if not os.path.isfile(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("  Run --mode train first to produce the model.")
        sys.exit(1)

    # Resolve threshold
    if threshold is None:
        if args.threshold is not None:
            threshold = args.threshold
        elif os.path.isfile(threshold_path):
            with open(threshold_path) as f:
                threshold = float(f.read().strip())
            print(f"Loaded threshold: {threshold}  (from {threshold_path})")
        else:
            threshold = 0.5
            print(f"Using default threshold: {threshold}")

    # ---- Step 1: Predict (blocking + scoring + output) ----
    banner("Step 1 / 2 — Inference (Test Data)")
    stats = predict(
        s1_path=s1_test,
        s2_path=s2_test,
        s3_path=s3_test,
        model_path=model_path,
        threshold=threshold,
        max_candidates=args.max_candidates,
        output_dir=args.output_dir,
    )
    print(f"\nInference done. [{elapsed(t0)}]")

    # ---- Step 2: Validate output ----
    banner("Step 2 / 2 — Submission Validation")
    passed = run_validator(
        matching_path=matching_path,
        candidate_path=candidate_path,
        test_dir=test_dir,
    )

    print(f"\n[Predict mode complete — {elapsed(t0)}]")
    return {**stats, "validation_passed": passed}


# ----------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Pipeline — ML Challenge 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pipeline.py --mode full
  python src/pipeline.py --mode train --max-candidates 100
  python src/pipeline.py --mode predict --threshold 0.35
        """,
    )
    parser.add_argument(
        "--mode", choices=["train", "predict", "full"], default="full",
        help="Pipeline mode: train | predict | full (default: full)",
    )
    parser.add_argument(
        "--data-dir", default="dataset",
        help="Path to dataset/ folder (default: dataset/)",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Path to output/ folder (default: output/)",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=50,
        help="Max candidates per S1 entity from blocking (default: 50)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override decision threshold (default: read from best_threshold.txt or 0.5)",
    )
    args = parser.parse_args()

    t_start = time.time()

    banner("Business Entity Resolution Pipeline")
    print(f"  Mode         : {args.mode}")
    print(f"  Data dir     : {args.data_dir}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Max candidates: {args.max_candidates}")
    print(f"  Threshold    : {args.threshold or 'auto (from best_threshold.txt)'}")

    os.makedirs(args.output_dir, exist_ok=True)

    train_results = {}
    predict_results = {}

    if args.mode in ("train", "full"):
        train_results = run_train(args)

    if args.mode in ("predict", "full"):
        threshold = train_results.get("threshold", None)
        predict_results = run_predict(args, threshold=threshold)

    # ---- Final summary ----
    banner("Pipeline Complete")
    total = elapsed(t_start)
    print(f"  Total runtime: {total}")

    if train_results:
        print(f"  Blocking recall     : {train_results.get('blocking_recall', 0)*100:.1f}%")
        print(f"  Model path          : {train_results.get('model_path')}")
        print(f"  Best threshold      : {train_results.get('threshold')}")

    if predict_results:
        print(f"  S1 entities scored  : {predict_results.get('n_s1_entities', 0):,}")
        print(f"  Predicted matches   : {predict_results.get('n_with_match', 0):,}")
        print(f"  Predicted singletons: {predict_results.get('n_singletons', 0):,}")
        v = predict_results.get("validation_passed")
        status = "PASS" if v else "FAIL (check output above)"
        print(f"  Validator status    : {status}")
        print(f"\n  Submit these files:")
        print(f"    output/matching_results.tsv")
        print(f"    output/candidate_pairs.tsv")


if __name__ == "__main__":
    main()
