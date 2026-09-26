"""
ML Challenge 2026 — Entity Resolution Pipeline
Step 8: DeBERTa-v3-small Cross-Encoder Reranker

Fine-tunes a 'microsoft/deberta-v3-small' (140M params, Apache 2.0 license)
cross-encoder to score and rerank top candidates from earlier pipeline stages.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from evaluate import load_ground_truth


MODEL_NAME = "microsoft/deberta-v3-small"


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _extract_entity_text_lookup(df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """
    Extract {entity_id: (business_name, business_address)} lookup table.
    Gracefully handles raw and preprocessed column names.
    """
    name_col = next((c for c in ["business_name", "norm_name"] if c in df.columns), None)
    addr_col = next((c for c in ["business_address", "norm_addr"] if c in df.columns), None)
    id_col = next((c for c in ["entity_id", "source1_entity_id"] if c in df.columns), df.columns[0])

    lookup: Dict[str, Tuple[str, str]] = {}
    cols_to_use = [id_col]
    if name_col:
        cols_to_use.append(name_col)
    if addr_col:
        cols_to_use.append(addr_col)

    for row in df[cols_to_use].itertuples(index=False):
        eid = str(row[0]).strip()
        name = str(row[1]).strip() if len(cols_to_use) > 1 and pd.notna(row[1]) else ""
        addr = str(row[2]).strip() if len(cols_to_use) > 2 and pd.notna(row[2]) else ""
        lookup[eid] = (name, addr)

    return lookup


# ----------------------------------------------------------------
# 1. Prepare Cross-Encoder Data
# ----------------------------------------------------------------

def prepare_cross_encoder_data(
    candidates_df: Union[pd.DataFrame, Dict[str, List[str]]],
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    gt: Optional[Union[Dict[str, Set[str]], str]] = None,
    top_k: int = 20,
) -> Dataset:
    """
    Prepares training/evaluation data for the DeBERTa cross-encoder.
    For each S1 entity, takes top_k candidate entities and formats each pair as:
      "[CLS] {s1_name} [SEP] {s1_address} [SEP] {cand_name} [SEP] {cand_address}"

    Args:
        candidates_df: DataFrame with [source1_entity_id, candidate_ids_list] or
                       scored pairs [s1_id, candidate_id, score], or dict mapping.
        s1_df: Source 1 DataFrame.
        s2_df: Source 2 DataFrame.
        s3_df: Source 3 DataFrame.
        gt: Ground truth dict or file path (if provided, labels are assigned).
        top_k: Maximum candidate pairs per S1 entity to retain.

    Returns:
        HuggingFace Dataset ready for tokenization and Trainer fine-tuning.
    """
    print(f"\n=== Preparing Cross-Encoder Data (top_k={top_k}) ===")
    t0 = time.time()

    # Load ground truth if file path is given
    if isinstance(gt, str) and os.path.isfile(gt):
        gt = load_ground_truth(gt)

    # Build entity text lookups
    s1_lkp = _extract_entity_text_lookup(s1_df)
    s2_lkp = _extract_entity_text_lookup(s2_df)
    s3_lkp = _extract_entity_text_lookup(s3_df)
    cand_lkp = {**s2_lkp, **s3_lkp}
    print(f"  Entity Lookups: S1: {len(s1_lkp):,} | Candidates: {len(cand_lkp):,}")

    pairs_to_process: List[Tuple[str, str]] = []

    # Extract (s1_id, cand_id) pairs respecting top_k
    if isinstance(candidates_df, dict):
        for s1_id, cands in candidates_df.items():
            for cid in cands[:top_k]:
                pairs_to_process.append((str(s1_id).strip(), str(cid).strip()))
    elif isinstance(candidates_df, pd.DataFrame):
        # Case 1: Scored pairs dataframe (s1_id, candidate_id, score)
        if "s1_id" in candidates_df.columns and "candidate_id" in candidates_df.columns:
            df = candidates_df.copy()
            if "score" in df.columns:
                df = df.sort_values(["s1_id", "score"], ascending=[True, False])
            grouped = df.groupby("s1_id").head(top_k)
            for _, row in grouped.iterrows():
                pairs_to_process.append((str(row["s1_id"]).strip(), str(row["candidate_id"]).strip()))
        # Case 2: Candidate pairs list (source1_entity_id, candidate_ids_list)
        else:
            id_col = "source1_entity_id" if "source1_entity_id" in candidates_df.columns else candidates_df.columns[0]
            cand_col = "candidate_ids_list" if "candidate_ids_list" in candidates_df.columns else candidates_df.columns[1]

            for _, row in candidates_df.iterrows():
                s1_id = str(row[id_col]).strip()
                raw_cands = row[cand_col]
                if isinstance(raw_cands, str):
                    cands = [c.strip() for c in raw_cands.split(",") if c.strip()]
                elif isinstance(raw_cands, (list, tuple)):
                    cands = [str(c).strip() for c in raw_cands if str(c).strip()]
                else:
                    cands = []

                for cid in cands[:top_k]:
                    pairs_to_process.append((s1_id, cid))

    print(f"  Total candidate pairs extracted: {len(pairs_to_process):,}")

    # Build dataset records
    formatted_texts: List[str] = []
    s1_texts: List[str] = []
    cand_texts: List[str] = []
    labels: List[int] = []
    s1_ids: List[str] = []
    cand_ids: List[str] = []

    pos_count = 0
    for s1_id, cid in pairs_to_process:
        s1_name, s1_addr = s1_lkp.get(s1_id, ("", ""))
        cand_name, cand_addr = cand_lkp.get(cid, ("", ""))

        # Format: "[CLS] {s1_name} [SEP] {s1_address} [SEP] {cand_name} [SEP] {cand_address}"
        full_text = f"[CLS] {s1_name} [SEP] {s1_addr} [SEP] {cand_name} [SEP] {cand_addr}"
        text_a = f"{s1_name} [SEP] {s1_addr}"
        text_b = f"{cand_name} [SEP] {cand_addr}"

        lbl = 0
        if gt is not None:
            true_set = gt.get(s1_id, set())
            lbl = 1 if cid in true_set else 0
            if lbl == 1:
                pos_count += 1

        formatted_texts.append(full_text)
        s1_texts.append(text_a)
        cand_texts.append(text_b)
        labels.append(lbl)
        s1_ids.append(s1_id)
        cand_ids.append(cid)

    data_dict = {
        "text": formatted_texts,
        "s1_text": s1_texts,
        "cand_text": cand_texts,
        "label": labels,
        "s1_id": s1_ids,
        "cand_id": cand_ids,
    }

    dataset = Dataset.from_dict(data_dict)
    elapsed = time.time() - t0
    print(f"  Prepared Dataset: {len(dataset):,} samples ({pos_count:,} positive) in {elapsed:.1f}s")
    return dataset


# ----------------------------------------------------------------
# 2. Train Cross-Encoder
# ----------------------------------------------------------------

def train_cross_encoder(
    dataset: Dataset,
    output_dir: str = "output/deberta_reranker/",
    model_name: str = MODEL_NAME,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Fine-tunes microsoft/deberta-v3-small for binary classification.

    Trainer settings:
      - num_train_epochs: 3
      - per_device_train_batch_size: 16
      - learning_rate: 2e-5
      - evaluation_strategy / eval_strategy: epoch
      - load_best_model_at_end: True
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== Fine-Tuning Cross-Encoder ({model_name}) ===")
    print(f"  Output directory: {output_dir}")

    # Train / Val split
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]

    print(f"  Train set: {len(train_ds):,} | Val set: {len(val_ds):,}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    def tokenize_function(examples):
        # Tokenize sentence pair: (s1_text, cand_text)
        return tokenizer(
            examples["s1_text"],
            examples["cand_text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    print("Tokenizing datasets...")
    train_tokenized = train_ds.map(tokenize_function, batched=True)
    val_tokenized = val_ds.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    args_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size * 2,
        learning_rate=learning_rate,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
    )

    # Compatible with newer transformers (eval_strategy) and legacy (evaluation_strategy)
    try:
        training_args = TrainingArguments(eval_strategy="epoch", **args_kwargs)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="epoch", **args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting Trainer.train()...")
    trainer.train()

    print("Saving fine-tuned model and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to: {output_dir}")

    return model, tokenizer


# ----------------------------------------------------------------
# 3. Rerank with Cross-Encoder
# ----------------------------------------------------------------

def rerank_with_cross_encoder(
    model_dir: str,
    candidate_pairs: Union[pd.DataFrame, Dict[str, List[str]]],
    s1_df: pd.DataFrame,
    cand_df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]],
    threshold: float = 0.5,
    batch_size: int = 64,
    max_length: int = 256,
    device: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Reranks candidates using the fine-tuned DeBERTa cross-encoder.

    Args:
        model_dir: Path to directory containing saved fine-tuned model & tokenizer.
        candidate_pairs: DataFrame or dictionary of candidate pairs.
        s1_df: S1 DataFrame.
        cand_df: Candidate DataFrame (or tuple of (s2_df, s3_df)).
        threshold: Classification probability threshold.
        batch_size: Inference batch size.
        max_length: Maximum sequence length.
        device: 'cuda' or 'cpu' (auto-selected if None).

    Returns:
        Dict[str, List[str]]: {s1_id: [matched_candidate_ids_sorted_by_score]}
    """
    print(f"\n=== Reranking Candidates with Cross-Encoder (threshold={threshold}) ===")
    t0 = time.time()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Build entity lookups
    s1_lkp = _extract_entity_text_lookup(s1_df)

    if isinstance(cand_df, tuple):
        s2_lkp = _extract_entity_text_lookup(cand_df[0])
        s3_lkp = _extract_entity_text_lookup(cand_df[1])
        cand_lkp = {**s2_lkp, **s3_lkp}
    else:
        cand_lkp = _extract_entity_text_lookup(cand_df)

    # Flatten candidate pairs
    pairs: List[Tuple[str, str]] = []
    all_s1_ids: List[str] = list(s1_df["entity_id"].astype(str).unique()) if "entity_id" in s1_df.columns else []

    if isinstance(candidate_pairs, dict):
        for s1_id, cands in candidate_pairs.items():
            for cid in cands:
                pairs.append((str(s1_id).strip(), str(cid).strip()))
    elif isinstance(candidate_pairs, pd.DataFrame):
        if "s1_id" in candidate_pairs.columns and "candidate_id" in candidate_pairs.columns:
            for _, r in candidate_pairs.iterrows():
                pairs.append((str(r["s1_id"]).strip(), str(r["candidate_id"]).strip()))
        else:
            id_col = "source1_entity_id" if "source1_entity_id" in candidate_pairs.columns else candidate_pairs.columns[0]
            cand_col = "candidate_ids_list" if "candidate_ids_list" in candidate_pairs.columns else candidate_pairs.columns[1]
            for _, r in candidate_pairs.iterrows():
                s1_id = str(r[id_col]).strip()
                raw_cands = r[cand_col]
                cands = [c.strip() for c in str(raw_cands).split(",") if c.strip()] if isinstance(raw_cands, str) else list(raw_cands or [])
                for cid in cands:
                    pairs.append((s1_id, str(cid).strip()))

    print(f"  Scoring {len(pairs):,} pairs in batches of {batch_size}...")

    scored_results: Dict[str, List[Tuple[str, float]]] = {s: [] for s in all_s1_ids}

    # Batched inference
    with torch.no_grad():
        for start_idx in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start_idx:start_idx + batch_size]
            batch_s1_texts = []
            batch_cand_texts = []

            for s1_id, cid in batch_pairs:
                s1_name, s1_addr = s1_lkp.get(s1_id, ("", ""))
                cand_name, cand_addr = cand_lkp.get(cid, ("", ""))
                batch_s1_texts.append(f"{s1_name} [SEP] {s1_addr}")
                batch_cand_texts.append(f"{cand_name} [SEP] {cand_addr}")

            inputs = tokenizer(
                batch_s1_texts,
                batch_cand_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()

            for (s1_id, cid), prob in zip(batch_pairs, probs):
                scored_results.setdefault(s1_id, []).append((cid, float(prob)))

    # Apply threshold and sort
    predictions: Dict[str, List[str]] = {}
    total_matched = 0

    for s1_id, cand_scores in scored_results.items():
        above_thresh = [(cid, sc) for cid, sc in cand_scores if sc >= threshold]
        above_sorted = [cid for cid, _ in sorted(above_thresh, key=lambda x: -x[1])]
        predictions[s1_id] = above_sorted
        total_matched += len(above_sorted)

    elapsed = time.time() - t0
    non_empty = sum(1 for v in predictions.values() if v)
    print(f"  Reranking complete in {elapsed:.1f}s")
    print(f"  Total matched pairs: {total_matched:,} across {non_empty:,} S1 entities")

    return predictions


# ----------------------------------------------------------------
# Standalone CLI execution
# ----------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeBERTa-v3-small Cross-Encoder Reranker")
    parser.add_argument("--s1", default="dataset/train/train_source1.tsv")
    parser.add_argument("--s2", default="dataset/train/train_source2.tsv")
    parser.add_argument("--s3", default="dataset/train/train_source3.tsv")
    parser.add_argument("--gt", default="dataset/train/train_ground_truth.tsv")
    parser.add_argument("--candidates", default="output/train_candidate_pairs.tsv")
    parser.add_argument("--output-dir", default="output/deberta_reranker/")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    print(f"Loading sources from {args.s1}, {args.s2}, {args.s3}...")
    s1_df = pd.read_csv(args.s1, sep="\t", nrows=50000)
    s2_df = pd.read_csv(args.s2, sep="\t", nrows=100000)
    s3_df = pd.read_csv(args.s3, sep="\t", nrows=100000)
    cands_df = pd.read_csv(args.candidates, sep="\t", nrows=50000)

    ds = prepare_cross_encoder_data(
        candidates_df=cands_df,
        s1_df=s1_df,
        s2_df=s2_df,
        s3_df=s3_df,
        gt=args.gt,
        top_k=args.top_k,
    )

    model, tok = train_cross_encoder(
        dataset=ds,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )
