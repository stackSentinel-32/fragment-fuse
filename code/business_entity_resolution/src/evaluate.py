"""
ML Challenge 2026 — Entity Resolution Pipeline
Step 3: Blocking Recall & F0.5 Evaluation Module
"""

import os
import sys
import csv
from typing import Dict, List, Set, Optional

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


# ---------------------------------------------------------
# Ground Truth Loading
# ---------------------------------------------------------

def load_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    """
    Load train_ground_truth.tsv.
    Returns dict: {source1_entity_id → set of matched_entity_ids}
    Singletons (no matches) map to an empty set.
    """
    gt: Dict[str, Set[str]] = {}

    with open(gt_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"].strip()
            raw_matches = row.get("matched_entity_ids", "").strip()
            if raw_matches:
                matched_ids = {m.strip() for m in raw_matches.split(",") if m.strip()}
            else:
                matched_ids = set()
            gt[s1_id] = matched_ids

    n_total = len(gt)
    n_singletons = sum(1 for v in gt.values() if not v)
    n_pairs = sum(len(v) for v in gt.values())
    print(f"Loaded ground truth: {n_total:,} S1 entities, "
          f"{n_singletons:,} singletons, {n_pairs:,} total match pairs")
    return gt


# ---------------------------------------------------------
# Blocking Recall
# ---------------------------------------------------------

def blocking_recall(
    candidates_df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
) -> dict:
    """
    Measure how many true matches are captured by blocking candidates.
    Only evaluates S1 entities that have at least one true match (non-singletons).

    Args:
        candidates_df: DataFrame with columns [source1_entity_id, candidate_ids_list]
                       where candidate_ids_list is a Python list of ID strings.
        ground_truth:  {s1_id → set of true match IDs}

    Returns dict:
        'overall_recall':   fraction of all true-match pairs in candidates
        'entity_recall':    fraction of S1 entities where ALL matches were captured
        'partial_recall':   fraction of S1 entities where >= 1 match was captured
        'missed_pairs':     count of true-match pairs NOT captured
        'total_true_pairs': total true-match pairs
        'n_evaluated':      number of non-singleton S1 entities evaluated
    """
    # Build fast lookup: s1_id → set of candidates
    candidate_lookup: Dict[str, Set[str]] = {}
    for _, row in candidates_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        cands = row["candidate_ids_list"]
        if isinstance(cands, list):
            candidate_lookup[s1_id] = set(cands)
        elif isinstance(cands, str) and cands.strip():
            candidate_lookup[s1_id] = {c.strip() for c in cands.split(",") if c.strip()}
        else:
            candidate_lookup[s1_id] = set()

    total_true_pairs = 0
    captured_pairs = 0
    n_all_captured = 0      # entities where ALL matches are in candidates
    n_any_captured = 0      # entities where >= 1 match is in candidates
    n_evaluated = 0
    missed_pairs = 0

    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue  # Skip singletons

        n_evaluated += 1
        total_true_pairs += len(true_matches)

        cands = candidate_lookup.get(s1_id, set())
        found = true_matches & cands
        n_found = len(found)
        n_missed = len(true_matches) - n_found

        captured_pairs += n_found
        missed_pairs += n_missed

        if n_missed == 0:
            n_all_captured += 1
        if n_found > 0:
            n_any_captured += 1

    overall_recall = captured_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
    entity_recall = n_all_captured / n_evaluated if n_evaluated > 0 else 0.0
    partial_recall = n_any_captured / n_evaluated if n_evaluated > 0 else 0.0

    return {
        "overall_recall": overall_recall,
        "entity_recall": entity_recall,
        "partial_recall": partial_recall,
        "missed_pairs": missed_pairs,
        "captured_pairs": captured_pairs,
        "total_true_pairs": total_true_pairs,
        "n_evaluated": n_evaluated,
    }


# ---------------------------------------------------------
# F0.5 Scoring
# ---------------------------------------------------------

def f_beta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    """
    Compute F-beta score. Default beta=0.5 (precision-weighted).
    """
    if precision + recall == 0.0:
        return 0.0
    return (1.0 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)


def evaluate_predictions(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
) -> dict:
    """
    Compute macro-averaged F0.5 across ALL S1 entities (including singletons).

    Scoring rules:
    - true=empty, pred=empty   → entity score = 1.0  (correct singleton)
    - true=empty, pred=non-empty → entity score = 0.0  (false merge on singleton)
    - otherwise: compute per-entity precision, recall, then F0.5

    Args:
        predictions:  {s1_id → list of predicted match IDs}
        ground_truth: {s1_id → set of true match IDs}

    Returns dict:
        'macro_f05':             main leaderboard metric
        'macro_precision':       average per-entity precision
        'macro_recall':          average per-entity recall
        'n_entities':            total entities evaluated
        'n_singletons_correct':  correctly predicted as no-match
        'n_singletons_wrong':    singletons given a false match prediction
        'n_with_matches':        non-singleton entities
        'coverage':              fraction of S1 entities that have a prediction row
    """
    all_s1_ids = set(ground_truth.keys())
    pred_ids = set(predictions.keys())

    scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    n_singletons_correct = 0
    n_singletons_wrong = 0
    n_with_matches = 0

    for s1_id in all_s1_ids:
        true_set = ground_truth[s1_id]
        pred_list = predictions.get(s1_id, [])
        pred_set = set(pred_list)

        is_singleton = len(true_set) == 0

        if is_singleton:
            if len(pred_set) == 0:
                entity_f05 = 1.0
                p = 1.0
                r = 1.0
                n_singletons_correct += 1
            else:
                entity_f05 = 0.0
                p = 0.0
                r = 0.0
                n_singletons_wrong += 1
        else:
            n_with_matches += 1
            if len(pred_set) == 0:
                # Missed all matches
                p = 0.0
                r = 0.0
            else:
                tp = len(pred_set & true_set)
                p = tp / len(pred_set)
                r = tp / len(true_set)
            entity_f05 = f_beta_score(p, r, beta=0.5)

        scores.append(entity_f05)
        precisions.append(p)
        recalls.append(r)

    n_entities = len(scores)
    macro_f05 = sum(scores) / n_entities if n_entities > 0 else 0.0
    macro_precision = sum(precisions) / n_entities if n_entities > 0 else 0.0
    macro_recall = sum(recalls) / n_entities if n_entities > 0 else 0.0
    coverage = len(pred_ids & all_s1_ids) / len(all_s1_ids) if all_s1_ids else 0.0

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "n_entities": n_entities,
        "n_singletons_correct": n_singletons_correct,
        "n_singletons_wrong": n_singletons_wrong,
        "n_with_matches": n_with_matches,
        "coverage": coverage,
    }


# ---------------------------------------------------------
# Candidate TSV Loader (helper for __main__ and later stages)
# ---------------------------------------------------------

def load_candidate_pairs(candidates_path: str) -> pd.DataFrame:
    """
    Load a candidate_pairs.tsv file produced by blocking.py.
    Returns DataFrame with columns [source1_entity_id, candidate_ids_list]
    where candidate_ids_list is a Python list of strings.
    """
    rows = []
    with open(candidates_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"].strip()
            raw = row.get("candidate_entity_ids", "").strip()
            cands = [c.strip() for c in raw.split(",") if c.strip()] if raw else []
            rows.append({"source1_entity_id": s1_id, "candidate_ids_list": cands})

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Pretty Report Helpers
# ---------------------------------------------------------

def print_blocking_report(stats: dict):
    print("\n" + "=" * 55)
    print("  BLOCKING RECALL REPORT")
    print("=" * 55)
    print(f"  Non-singleton S1 entities evaluated : {stats['n_evaluated']:>12,}")
    print(f"  Total true match pairs               : {stats['total_true_pairs']:>12,}")
    print(f"  Captured pairs                       : {stats['captured_pairs']:>12,}")
    print(f"  Missed pairs                         : {stats['missed_pairs']:>12,}")
    print(f"  Overall pair recall                  : {stats['overall_recall']:>12.4f}  ({stats['overall_recall']*100:.2f}%)")
    print(f"  Entity-level 100% recall             : {stats['entity_recall']:>12.4f}  ({stats['entity_recall']*100:.2f}%)")
    print(f"  Entity-level partial recall (>=1 hit): {stats['partial_recall']:>12.4f}  ({stats['partial_recall']*100:.2f}%)")

    print()
    if stats["overall_recall"] >= 0.95:
        print("  [EXCELLENT] blocking recall >= 95%.")
    elif stats["overall_recall"] >= 0.90:
        print("  [GOOD] blocking recall >= 90%. Proceed to feature engineering.")
    elif stats["overall_recall"] >= 0.85:
        print("  [ACCEPTABLE] Consider adding more blocking keys (bigrams, city tokens).")
    else:
        print("  [LOW] blocking recall < 85%. STOP and improve blocking before training!")
    print("=" * 55)


def print_f05_report(stats: dict):
    print("\n" + "=" * 55)
    print("  F0.5 EVALUATION REPORT")
    print("=" * 55)
    print(f"  Total S1 entities evaluated          : {stats['n_entities']:>12,}")
    print(f"  Non-singleton entities               : {stats['n_with_matches']:>12,}")
    print(f"  Singletons correctly predicted       : {stats['n_singletons_correct']:>12,}")
    print(f"  Singletons incorrectly matched       : {stats['n_singletons_wrong']:>12,}")
    print(f"  Prediction coverage                  : {stats['coverage']:>12.4f}  ({stats['coverage']*100:.2f}%)")
    print(f"  Macro Precision                      : {stats['macro_precision']:>12.4f}")
    print(f"  Macro Recall                         : {stats['macro_recall']:>12.4f}")
    print(f"  Macro F0.5 Score                     : {stats['macro_f05']:>12.4f}  ← leaderboard metric")
    print("=" * 55)


# ---------------------------------------------------------
# Main: Evaluate Blocking Recall
# ---------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate blocking recall on training candidates")
    parser.add_argument(
        "--candidates",
        default=os.path.join("output", "train_candidate_pairs.tsv"),
        help="Path to candidate_pairs.tsv produced by blocking.py",
    )
    parser.add_argument(
        "--gt",
        default=os.path.join("dataset", "train", "train_ground_truth.tsv"),
        help="Path to train_ground_truth.tsv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N rows of the candidate file (for quick testing)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.candidates):
        print(f"\n❌ ERROR: Candidate file not found: {args.candidates}")
        print("   Run blocking.py first to generate the candidate pairs file.")
        sys.exit(1)

    print(f"\nLoading candidates from: {args.candidates}")
    candidates_df = load_candidate_pairs(args.candidates)
    if args.limit:
        candidates_df = candidates_df.head(args.limit)
    print(f"Loaded {len(candidates_df):,} candidate rows")

    print(f"\nLoading ground truth from: {args.gt}")
    gt = load_ground_truth(args.gt)

    # If candidates are a subset of S1, restrict GT to only those S1 IDs
    evaluated_s1_ids = set(candidates_df["source1_entity_id"])
    gt_subset = {k: v for k, v in gt.items() if k in evaluated_s1_ids}
    if len(gt_subset) < len(gt):
        print(f"Restricting GT to {len(gt_subset):,} S1 entities present in candidate file")

    # Run and report blocking recall
    stats = blocking_recall(candidates_df, gt_subset)
    print_blocking_report(stats)
