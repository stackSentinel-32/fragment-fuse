"""
ML Challenge 2026 — Entity Resolution Pipeline
Step 1: Text Preprocessing & Normalization Module
"""

import re
from typing import Optional, List
import pandas as pd

# ---------------------------------------------------------
# Dictionaries and Regex Patterns
# ---------------------------------------------------------

NAME_ABBREVIATIONS = {
    "corp": "corporation",
    "ltd": "limited",
    "pvt": "private",
    "inc": "incorporated",
    "llc": "limited liability company",
    "co": "company",
    "dept": "department",
}

NAME_ABBR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in NAME_ABBREVIATIONS.keys()) + r")\b",
    flags=re.IGNORECASE,
)

ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "apt": "apartment",
    "ste": "suite",
    "bldg": "building",
    "flr": "floor",
}

ADDR_ABBR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ADDRESS_ABBREVIATIONS.keys()) + r")\b",
    flags=re.IGNORECASE,
)

STOPWORDS = {"the", "a", "an", "of", "and", "for", "at", "in", "on"}

# Punctuation regexes
NAME_PUNCT_REGEX = re.compile(r"[^\w\s']|_")
ADDR_PUNCT_REGEX = re.compile(r"[^a-zA-Z0-9\s,]")
MULTI_SPACE_REGEX = re.compile(r"\s+")
COMMA_SPACE_REGEX = re.compile(r"\s*,\s*")
FIRST_NUMERIC_REGEX = re.compile(r"\b\d+\b")

# Postcode regexes
US_POSTCODE_REGEX = re.compile(r"\b(\d{5}(?:-\d{4})?)\b")
INDIA_POSTCODE_REGEX = re.compile(r"\b([1-9]\d{5})\b")
FRANCE_POSTCODE_REGEX = re.compile(r"\b(\d{5})\b")
GENERIC_POSTCODE_REGEX = re.compile(r"\b(\d{5,6})\b")


# ---------------------------------------------------------
# Core Normalization Functions
# ---------------------------------------------------------

def normalize_name(name: str) -> str:
    """
    Normalize a business name:
    - Lowercase
    - Replace '&' with ' and '
    - Remove punctuation except spaces and apostrophes
    - Expand common business abbreviations
    - Collapse extra spaces
    """
    if not isinstance(name, str) or not name.strip():
        return ""

    s = name.lower()
    s = s.replace("&", " and ")
    s = NAME_PUNCT_REGEX.sub(" ", s)
    s = NAME_ABBR_PATTERN.sub(lambda m: NAME_ABBREVIATIONS[m.group(0).lower()], s)
    s = MULTI_SPACE_REGEX.sub(" ", s).strip()
    return s


def name_tokens(name: str) -> List[str]:
    """
    Extract canonical sorted tokens from business name:
    - Normalize name
    - Split on whitespace
    - Remove common stopwords
    - Return sorted list of tokens
    """
    norm = normalize_name(name)
    if not norm:
        return []
    tokens = norm.split()
    filtered = [t for t in tokens if t not in STOPWORDS]
    return sorted(filtered)


def normalize_address(address: str) -> str:
    """
    Normalize a business address:
    - Lowercase
    - Expand common address abbreviations
    - Remove punctuation except commas
    - Standardize whitespace and commas
    """
    if not isinstance(address, str) or not address.strip():
        return ""

    s = address.lower()
    s = ADDR_ABBR_PATTERN.sub(lambda m: ADDRESS_ABBREVIATIONS[m.group(0).lower()], s)
    s = ADDR_PUNCT_REGEX.sub(" ", s)
    s = MULTI_SPACE_REGEX.sub(" ", s).strip()
    s = COMMA_SPACE_REGEX.sub(", ", s)
    # Remove leading/trailing commas if any
    s = s.strip(", ")
    return s


def address_tokens(address: str) -> List[str]:
    """
    Extract tokens from an address:
    - Normalize address
    - Split on whitespace and commas
    - Drop empty tokens and pure numbers shorter than 3 digits
    - Preserve order (address order matters)
    """
    norm = normalize_address(address)
    if not norm:
        return []

    raw_tokens = re.split(r"[\s,]+", norm)
    result = []
    for t in raw_tokens:
        if not t:
            continue
        if t.isdigit() and len(t) < 3:
            continue
        result.append(t)
    return result


def extract_postcode(address: str, country: str) -> Optional[str]:
    """
    Extract postal code from address string based on country:
    - US: 5-digit or 9-digit ZIP code
    - India: 6-digit PIN code
    - France: 5-digit postal code
    - Fallback: 5 or 6 digit code
    """
    if not isinstance(address, str) or not address.strip():
        return None

    c = str(country).strip().upper() if country else ""
    if c == "US":
        m = US_POSTCODE_REGEX.search(address)
        if m:
            return m.group(1)
    elif c in ("INDIA", "IN"):
        m = INDIA_POSTCODE_REGEX.search(address)
        if m:
            return m.group(1)
    elif c in ("FRANCE", "FR"):
        m = FRANCE_POSTCODE_REGEX.search(address)
        if m:
            return m.group(1)
    else:
        # Open country set fallback
        m = GENERIC_POSTCODE_REGEX.search(address)
        if m:
            return m.group(1)

    return None


def extract_street_number(address: str) -> Optional[str]:
    """
    Extract the first numeric token in the address as the likely street number.
    """
    if not isinstance(address, str) or not address.strip():
        return None

    m = FIRST_NUMERIC_REGEX.search(address)
    return m.group(0) if m else None


# ---------------------------------------------------------
# High-Performance DataFrame Preprocessing
# ---------------------------------------------------------

def load_and_preprocess(filepath: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Read a TSV file and apply preprocessing to generate all normalized columns:
    - norm_name
    - name_tok (space-separated sorted tokens)
    - norm_addr
    - addr_tok (space-separated tokens)
    - postcode
    - street_num
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        encoding="utf-8",
        nrows=nrows,
        dtype=str,
        keep_default_na=False,
    )

    names = df["business_name"].tolist() if "business_name" in df.columns else [""] * len(df)
    addrs = df["business_address"].tolist() if "business_address" in df.columns else [""] * len(df)
    countries = df["country"].tolist() if "country" in df.columns else [""] * len(df)

    # Process lists using list comprehensions for speed over 5M+ rows
    norm_names = [normalize_name(n) for n in names]
    name_toks = [" ".join(name_tokens(n)) for n in names]
    norm_addrs = [normalize_address(a) for a in addrs]
    addr_toks = [" ".join(address_tokens(a)) for a in addrs]
    postcodes = [extract_postcode(a, c) or "" for a, c in zip(addrs, countries)]
    street_nums = [extract_street_number(a) or "" for a in addrs]

    df["norm_name"] = norm_names
    df["name_tok"] = name_toks
    df["norm_addr"] = norm_addrs
    df["addr_tok"] = addr_toks
    df["postcode"] = postcodes
    df["street_num"] = street_nums

    return df


if __name__ == "__main__":
    import os

    train_s1_path = os.path.join("dataset", "train", "train_source1.tsv")
    print(f"Loading and preprocessing sample from {train_s1_path}...")

    # Load 5 sample rows for demonstration and verification
    sample_df = load_and_preprocess(train_s1_path, nrows=5)
    print("\n--- Preprocessing Results Sample (5 rows) ---")
    for idx, row in sample_df.iterrows():
        print(f"\n[Entity {row['entity_id']} | Country: {row['country']}]")
        print(f"  Raw Name:      {row['business_name']}")
        print(f"  Norm Name:     {row['norm_name']}")
        print(f"  Name Tokens:   {row['name_tok']}")
        print(f"  Raw Address:   {row['business_address']}")
        print(f"  Norm Address:  {row['norm_addr']}")
        print(f"  Addr Tokens:   {row['addr_tok']}")
        print(f"  Postcode:      {row['postcode']}")
        print(f"  Street Num:    {row['street_num']}")
