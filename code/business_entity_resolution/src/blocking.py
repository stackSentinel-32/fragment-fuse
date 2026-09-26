"""
ML Challenge 2026 — Entity Resolution Pipeline
Step 2: Candidate Generation / Multi-Signal Inverted Index Blocking

v2 — Improvements over v1:
- Weighted scoring: exact/high-value keys contribute more points than generic ones,
  so true matches reliably rank in the top-N even when the index has 10M+ entries.
- Address token keys: city, street name, and block-level tokens added as blocking keys,
  fixing the ~8.5% of true pairs that had zero key overlap with name-only blocking.
- Tiered posting-list cap: different max sizes per key type (strict for generic tokens,
  more permissive for specific keys like postcodes and street+addr combos).
"""

import os
import sys
from collections import defaultdict, Counter
from typing import List, Set, Optional, Dict, Tuple
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from preprocess import load_and_preprocess


# ----------------------------------------------------------------
# Token filters
# ----------------------------------------------------------------

GENERIC_LEGAL_TOKENS = {
    "limited", "private", "company", "corporation", "incorporated",
    "liability", "llc", "pvt", "ltd", "corp", "inc",
}

GENERIC_ADDR_TOKENS = {
    "road", "street", "avenue", "boulevard", "drive", "lane",
    "highway", "apartment", "suite", "building", "floor",
    "block", "sector", "nagar", "near", "opp", "opposite",
    "beside", "behind", "next", "above", "below", "plot",
    "floor", "unit", "no",
}

# Scoring weights per key tier
WEIGHT_EXACT = 10       # postcode, street_num+first_addr
WEIGHT_NAME_TOK = 3     # individual non-generic name token
WEIGHT_ADDR_TOK = 2     # individual non-generic address token
WEIGHT_BIGRAM = 1       # character bigram fallback

# Per-tier posting-list size caps (skip lists larger than this — too noisy)
CAP_EXACT = 5000
CAP_NAME_TOK = 1000
CAP_ADDR_TOK = 1000
CAP_BIGRAM = 1000


class BlockingIndex:
    """
    Multi-signal weighted inverted index.

    Index maps: blocking_key → set of entity_ids
    Retrieval scores each candidate by weighted sum of matched keys,
    returning the top-N highest-scoring candidates.
    """

    def __init__(self):
        # Separate sub-indexes per tier so we can apply per-tier caps at retrieval time
        # and weight them independently.
        self.idx_exact: Dict[str, Set[str]] = defaultdict(set)      # postcode, street+addr
        self.idx_name: Dict[str, Set[str]] = defaultdict(set)       # name tokens
        self.idx_addr: Dict[str, Set[str]] = defaultdict(set)       # address tokens
        self.idx_bigram: Dict[str, Set[str]] = defaultdict(set)     # char bigrams

    # ----------------------------------------------------------
    # Key generation
    # ----------------------------------------------------------

    def _generate_keyed_entries(
        self, row: dict
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Generate blocking keys for one entity row, separated by tier.
        Returns (exact_keys, name_keys, addr_keys, bigram_keys).
        """
        c = str(row.get("country", "")).strip()

        name_tok_str = str(row.get("name_tok", "")).strip()
        name_tokens = name_tok_str.split() if name_tok_str else []

        addr_tok_str = str(row.get("addr_tok", "")).strip()
        addr_tokens = addr_tok_str.split() if addr_tok_str else []

        postcode = str(row.get("postcode", "")).strip()
        street_num = str(row.get("street_num", "")).strip()
        norm_name = str(row.get("norm_name", "")).strip()

        exact_keys: List[str] = []
        name_keys: List[str] = []
        addr_keys: List[str] = []
        bigram_keys: List[str] = []

        # --- Tier 1: Exact high-precision keys ---

        # Postcode
        if postcode and len(postcode) >= 4:
            exact_keys.append(f"{c}__pc__{postcode}")

        # Street number + first meaningful address token
        if street_num and addr_tokens:
            # skip purely generic first tokens
            for at in addr_tokens:
                if at not in GENERIC_ADDR_TOKENS and not at.isdigit():
                    exact_keys.append(f"{c}__sn__{street_num}__{at}")
                    break

        # First 2 sorted name tokens combo (very specific)
        non_generic_name = [t for t in name_tokens if t not in GENERIC_LEGAL_TOKENS and len(t) >= 3]
        if len(non_generic_name) >= 2:
            exact_keys.append(f"{c}__nm2__{non_generic_name[0]}_{non_generic_name[1]}")

        # --- Tier 2: Individual name tokens ---
        for tok in name_tokens:
            if len(tok) >= 3 and tok not in GENERIC_LEGAL_TOKENS:
                name_keys.append(f"{c}__nm__{tok}")

        # --- Tier 3: Individual address tokens ---
        for tok in addr_tokens:
            if len(tok) >= 4 and tok not in GENERIC_ADDR_TOKENS and not tok.isdigit():
                addr_keys.append(f"{c}__ad__{tok}")

        # --- Tier 4: Character bigrams of norm_name (first 10 chars) ---
        clean_name = norm_name.replace(" ", "")[:10]
        seen_bg: Set[str] = set()
        for i in range(len(clean_name) - 1):
            bg = clean_name[i:i + 2]
            if bg not in seen_bg:
                bigram_keys.append(f"{c}__bg__{bg}")
                seen_bg.add(bg)

        # Deduplicate within each tier
        return (
            list(dict.fromkeys(exact_keys)),
            list(dict.fromkeys(name_keys)),
            list(dict.fromkeys(addr_keys)),
            list(dict.fromkeys(bigram_keys)),
        )

    # ----------------------------------------------------------
    # Index building
    # ----------------------------------------------------------

    def build(self, df: pd.DataFrame):
        """
        Build (extend) the index from a DataFrame (S2 or S3 entities).
        """
        total_rows = len(df)
        print(f"Building blocking index on {total_rows:,} entities...")

        entity_ids = df["entity_id"].values
        countries = df["country"].values
        norm_names = df["norm_name"].values
        name_toks = df["name_tok"].values
        addr_toks = df["addr_tok"].values
        postcodes = df["postcode"].values
        street_nums = df["street_num"].values

        for i in range(total_rows):
            eid = entity_ids[i]
            row_dict = {
                "country": countries[i],
                "norm_name": norm_names[i],
                "name_tok": name_toks[i],
                "addr_tok": addr_toks[i],
                "postcode": postcodes[i],
                "street_num": street_nums[i],
            }
            ex, nm, ad, bg = self._generate_keyed_entries(row_dict)
            for k in ex:
                self.idx_exact[k].add(eid)
            for k in nm:
                self.idx_name[k].add(eid)
            for k in ad:
                self.idx_addr[k].add(eid)
            for k in bg:
                self.idx_bigram[k].add(eid)

            if (i + 1) % 500_000 == 0 or (i + 1) == total_rows:
                n_keys = (len(self.idx_exact) + len(self.idx_name)
                          + len(self.idx_addr) + len(self.idx_bigram))
                print(f"  Processed {i + 1:,} / {total_rows:,} "
                      f"(total unique keys: {n_keys:,})")

    # ----------------------------------------------------------
    # Candidate retrieval
    # ----------------------------------------------------------

    def retrieve(self, row: dict, max_candidates: int = 50) -> Set[str]:
        """
        Retrieve top-N candidates for one S1 entity using weighted scoring.

        Phase 1 — Exact + name + addr keys (high precision).
        Phase 2 — Bigram fallback (only if Phase 1 yields < 20 distinct candidates).
        """
        ex, nm, ad, bg = self._generate_keyed_entries(row)

        scores: Counter = Counter()

        # Phase 1a — Exact keys (highest weight)
        for k in ex:
            posting = self.idx_exact.get(k)
            if posting and len(posting) <= CAP_EXACT:
                for eid in posting:
                    scores[eid] += WEIGHT_EXACT

        # Phase 1b — Name token keys
        for k in nm:
            posting = self.idx_name.get(k)
            if posting and len(posting) <= CAP_NAME_TOK:
                for eid in posting:
                    scores[eid] += WEIGHT_NAME_TOK

        # Phase 1c — Address token keys
        for k in ad:
            posting = self.idx_addr.get(k)
            if posting and len(posting) <= CAP_ADDR_TOK:
                for eid in posting:
                    scores[eid] += WEIGHT_ADDR_TOK

        # Phase 2 — Bigram fallback (only if very few found so far)
        if len(scores) < 20:
            for k in bg:
                posting = self.idx_bigram.get(k)
                if posting and len(posting) <= CAP_BIGRAM:
                    for eid in posting:
                        scores[eid] += WEIGHT_BIGRAM

        if not scores:
            return set()

        if len(scores) <= max_candidates:
            return set(scores.keys())

        return {cand for cand, _ in scores.most_common(max_candidates)}

    def stats(self) -> dict:
        n_exact = sum(len(v) for v in self.idx_exact.values())
        n_name = sum(len(v) for v in self.idx_name.values())
        n_addr = sum(len(v) for v in self.idx_addr.values())
        n_bg = sum(len(v) for v in self.idx_bigram.values())
        return {
            "unique_keys": (len(self.idx_exact) + len(self.idx_name)
                            + len(self.idx_addr) + len(self.idx_bigram)),
            "unique_exact_keys": len(self.idx_exact),
            "unique_name_keys": len(self.idx_name),
            "unique_addr_keys": len(self.idx_addr),
            "unique_bigram_keys": len(self.idx_bigram),
            "total_postings": n_exact + n_name + n_addr + n_bg,
        }


# ----------------------------------------------------------------
# Full pipeline function
# ----------------------------------------------------------------

def run_blocking(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    output_path: str,
    max_candidates: int = 50,
    chunk_size: int = 100_000,
    nrows_s1: Optional[int] = None,
    nrows_s2_s3: Optional[int] = None,
) -> pd.DataFrame:
    """
    Full blocking pipeline.
    1. Load and preprocess S2 + S3, build BlockingIndex.
    2. Load and preprocess S1, retrieve candidates per entity.
    3. Write candidate_pairs.tsv to output_path.
    4. Return DataFrame with [source1_entity_id, candidate_ids_list].
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print("=== Step 1: Loading & Preprocessing Target Sources (S2 & S3) ===")
    s2_df = load_and_preprocess(s2_path, nrows=nrows_s2_s3)
    print(f"Loaded S2: {len(s2_df):,} rows")
    s3_df = load_and_preprocess(s3_path, nrows=nrows_s2_s3)
    print(f"Loaded S3: {len(s3_df):,} rows")

    print("\n=== Step 2: Building Weighted Blocking Index ===")
    index = BlockingIndex()
    index.build(s2_df)
    index.build(s3_df)
    del s2_df, s3_df

    st = index.stats()
    print(f"\nIndex stats: {st['unique_keys']:,} unique keys, "
          f"{st['total_postings']:,} total postings")

    print("\n=== Step 3: Loading & Preprocessing Source 1 ===")
    s1_df = load_and_preprocess(s1_path, nrows=nrows_s1)
    n_s1 = len(s1_df)
    print(f"Loaded S1: {n_s1:,} rows")

    print(f"\n=== Step 4: Generating Candidates (chunk={chunk_size:,}) ===")
    results_ids: List[str] = []
    results_cands: List[List[str]] = []

    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")

        for start in range(0, n_s1, chunk_size):
            end = min(start + chunk_size, n_s1)
            chunk = s1_df.iloc[start:end]
            lines = []

            # Build dicts from arrays for fast access
            c_ids = chunk["entity_id"].values
            c_countries = chunk["country"].values
            c_norm_names = chunk["norm_name"].values
            c_name_toks = chunk["name_tok"].values
            c_addr_toks = chunk["addr_tok"].values
            c_postcodes = chunk["postcode"].values
            c_street_nums = chunk["street_num"].values

            for i in range(len(chunk)):
                row_dict = {
                    "country": c_countries[i],
                    "norm_name": c_norm_names[i],
                    "name_tok": c_name_toks[i],
                    "addr_tok": c_addr_toks[i],
                    "postcode": c_postcodes[i],
                    "street_num": c_street_nums[i],
                }
                s1_id = c_ids[i]
                cands = sorted(index.retrieve(row_dict, max_candidates=max_candidates))
                lines.append(f"{s1_id}\t{','.join(cands)}\n")
                results_ids.append(s1_id)
                results_cands.append(cands)

            out_f.writelines(lines)
            print(f"  Processed {end:,} / {n_s1:,} S1 entities...")

    cand_lens = [len(c) for c in results_cands]
    total = len(cand_lens)
    avg = sum(cand_lens) / total if total else 0
    zeros = sum(1 for x in cand_lens if x == 0)
    print("\n=== Blocking Summary ===")
    print(f"  Total S1 entities:       {total:,}")
    print(f"  Avg candidates / S1:     {avg:.2f}")
    print(f"  Max candidates / S1:     {max(cand_lens) if cand_lens else 0}")
    print(f"  S1 with 0 candidates:    {zeros:,} ({zeros/total*100:.2f}%)")
    print(f"  Output:                  {output_path}")

    return pd.DataFrame({
        "source1_entity_id": results_ids,
        "candidate_ids_list": results_cands,
    })


# ----------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Weighted Multi-Signal Blocking v2")
    parser.add_argument("--s1", default=os.path.join("dataset", "train", "train_source1.tsv"))
    parser.add_argument("--s2", default=os.path.join("dataset", "train", "train_source2.tsv"))
    parser.add_argument("--s3", default=os.path.join("dataset", "train", "train_source3.tsv"))
    parser.add_argument("--output", default=os.path.join("output", "train_candidate_pairs.tsv"))
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--limit-s1", type=int, default=None)
    parser.add_argument("--limit-s2-s3", type=int, default=None)
    args = parser.parse_args()

    run_blocking(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        output_path=args.output,
        max_candidates=args.max_candidates,
        nrows_s1=args.limit_s1,
        nrows_s2_s3=args.limit_s2_s3,
    )
