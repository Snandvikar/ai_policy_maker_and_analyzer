"""
pipeline/state_master.py
=========================
Central State / UT Mapping Table for India.

Persists two tables to DuckDB:
  state_master        — one row per state: state_code, canonical_state
  state_master_aliases — one row per alias: state_code, canonical_state, alias, alias_type

All other pipeline modules call apply_state_code() to normalise any
DataFrame's state column before joins. Joins always use canonical_state.
"""

from __future__ import annotations
import re
import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MASTER RECORDS
# Format: (state_code, canonical_state, aliases)
# ─────────────────────────────────────────────────────────────────────────────

_STATE_RECORDS: list[tuple] = [
    (1,  "andhra_pradesh", [
        "andhra pradesh", "ap", "andhra",
        "residual andhra pradesh",          # post-2014 bifurcation label
        "andhra pradesh  telangana",        # pre-bifurcation combined label
    ]),
    (2,  "arunachal_pradesh", [
        "arunachal pradesh", "ar", "arunachal",
    ]),
    (3,  "assam", [
        "assam", "as",
    ]),
    (4,  "bihar", [
        "bihar", "br",
    ]),
    (5,  "chhattisgarh", [
        "chhattisgarh", "cg", "chattisgarh", "chhatisgarh",
    ]),
    (6,  "goa", [
        "goa", "ga",
    ]),
    (7,  "gujarat", [
        "gujarat", "gj",
    ]),
    (8,  "haryana", [
        "haryana", "hr",
    ]),
    (9,  "himachal_pradesh", [
        "himachal pradesh", "hp", "himachal",
    ]),
    (10, "jharkhand", [
        "jharkhand", "jh",
    ]),
    (11, "karnataka", [
        "karnataka", "ka", "karnatak",
        "karnataka and goa",                # TRAI service area variant
        "karnataka goa",
    ]),
    (12, "kerala", [
        "kerala", "kl",
    ]),
    (13, "madhya_pradesh", [
        "madhya pradesh", "mp", "madhya",
        "m p",
    ]),
    (14, "maharashtra", [
        "maharashtra", "mh",
        "mumbai",                           # TRAI uses city as service area
        "maharashtra mumbai",               # TRAI combined label
    ]),
    (15, "manipur", [
        "manipur", "mn",
    ]),
    (16, "meghalaya", [
        "meghalaya", "ml",
    ]),
    (17, "mizoram", [
        "mizoram", "mz",
    ]),
    (18, "nagaland", [
        "nagaland", "nl",
    ]),
    (19, "odisha", [
        "odisha", "od", "orissa",           # old spelling, still in 2001 census
        "orrisa",
    ]),
    (20, "punjab", [
        "punjab", "pb",
        "punjab haryana",                   # TRAI combined service area
    ]),
    (21, "rajasthan", [
        "rajasthan", "rj", "rajsthan",
    ]),
    (22, "sikkim", [
        "sikkim", "sk",
    ]),
    (23, "tamil_nadu", [
        "tamil nadu", "tn", "tamilnadu",
        "tamil naidu",
    ]),
    (24, "telangana", [
        "telangana", "ts", "telegana",
    ]),
    (25, "tripura", [
        "tripura", "tr",
    ]),
    (26, "uttar_pradesh", [
        "uttar pradesh", "up",
        "u p",
    ]),
    (27, "uttarakhand", [
        "uttarakhand", "uk",
        "uttaranchal",                      # official name until 2007
        "uttrakhand",
    ]),
    (28, "west_bengal", [
        "west bengal", "wb",
        "kolkata",                          # TRAI service area
        "west bengal kolkata",
    ]),
    # J&K and Ladakh (bifurcated Nov 2019)
    (29, "jammu_and_kashmir", [
        "jammu and kashmir", "jk", "j and k",
        "jammu kashmir", "jammu  kashmir",
        # Dot/ampersand variants seen in real datasets
        "j k", "j  k", "jk",
        "jammu and kashmir ladakh",         # pre-bifurcation combined
    ]),
    (30, "ladakh", [
        "ladakh", "la",
    ]),
    # Union Territories
    (31, "andaman_and_nicobar", [
        "andaman and nicobar islands",
        "andaman and nicobar",
        "a and n islands",
        "andaman nicobar",
    ]),
    (32, "chandigarh", [
        "chandigarh", "ch",
    ]),
    (33, "dadra_and_nagar_haveli_and_daman_and_diu", [
        # Merged UT since Jan 2020 — many name variants in datasets
        "dadra and nagar haveli and daman and diu",
        "the dadra and nagar haveli and daman and diu",  # "the" prefix in some sources
        "dadra and nagar haveli",
        "the dadra and nagar haveli",
        "daman and diu",
        "dnh and dd", "dnhdd",
        "dadra nagar haveli daman diu",
        "d and nh and dd",
    ]),
    (34, "delhi", [
        "delhi", "dl",
        "nct of delhi",
        "national capital territory of delhi",
        "nct delhi",
        "new delhi",
    ]),
    (35, "lakshadweep", [
        "lakshadweep", "ld",
    ]),
    (36, "puducherry", [
        "puducherry", "py",
        "pondicherry",                      # older name still in datasets
        "pondichery",
    ]),
    # Legacy UT codes kept for older datasets
    (37, "the_dadra_and_nagar_haveli", [
        "the dadra and nagar haveli",
    ]),
]


# ─────────────────────────────────────────────────────────────────────────────
# BUILD DATAFRAMES
# ─────────────────────────────────────────────────────────────────────────────

STATE_MASTER = pd.DataFrame(
    [(code, name) for code, name, _ in _STATE_RECORDS],
    columns=["state_code", "canonical_state"],
)

# Flat alias table: one row per (state_code, alias)
_alias_rows = []
for code, canonical, aliases in _STATE_RECORDS:
    for alias in aliases:
        _alias_rows.append({
            "state_code":       code,
            "canonical_state":  canonical,
            "alias":            alias,
        })

STATE_MASTER_ALIASES = pd.DataFrame(_alias_rows)


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISER
# ─────────────────────────────────────────────────────────────────────────────

def normalise_state(name) -> Optional[str]:
    """
    Normalise a raw state name string for alias lookup.

    Pipeline:
      1. Handle None/NaN → return None
      2. Lowercase
      3. Replace dots with space       "J.K."  → "j k"
      4. Replace & with " and "        "J&K"   → "j and k"
      5. Strip all non-alpha-space     keeps only a-z and spaces
      6. Collapse multiple spaces
      7. Strip edges
    """
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    s = str(name).lower().strip()
    s = s.replace(".", " ")                    # dots → space
    s = s.replace("&", " and ")               # ampersand → and
    s = re.sub(r"[^a-z ]", " ", s)            # strip everything else
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None


# ─────────────────────────────────────────────────────────────────────────────
# LOOKUP MAPS  (built once at import)
# ─────────────────────────────────────────────────────────────────────────────

_ALIAS_TO_CODE:      dict[str, int] = {}
_ALIAS_TO_CANONICAL: dict[str, str] = {}

# Index aliases
for _, row in STATE_MASTER_ALIASES.iterrows():
    norm = normalise_state(row["alias"])
    if norm:
        _ALIAS_TO_CODE[norm]      = row["state_code"]
        _ALIAS_TO_CANONICAL[norm] = row["canonical_state"]

# Also index canonical_state itself (underscore replaced with space)
for _, row in STATE_MASTER.iterrows():
    norm = normalise_state(row["canonical_state"].replace("_", " "))
    if norm:
        _ALIAS_TO_CODE[norm]      = row["state_code"]
        _ALIAS_TO_CANONICAL[norm] = row["canonical_state"]


def lookup_state_code(name) -> Optional[int]:
    norm = normalise_state(name)
    return _ALIAS_TO_CODE.get(norm) if norm else None


def lookup_canonical(name) -> Optional[str]:
    norm = normalise_state(name)
    return _ALIAS_TO_CANONICAL.get(norm) if norm else None


# ─────────────────────────────────────────────────────────────────────────────
# DATAFRAME HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def apply_state_code(
    df: pd.DataFrame,
    col: str = "statename",
    warn_unresolved: bool = True,
) -> pd.DataFrame:
    """
    Adds canonical_state (snake_case) and state_code (int) columns to df.
    Overwrites df[col] with the normalised lowercase form so all tables
    share the same string for downstream joins.

    Call this on EVERY table before any join — including district-level tables.
    """
    df = df.copy()
    if col not in df.columns:
        logger.warning(f"apply_state_code: column '{col}' not in DataFrame")
        return df

    norm_series           = df[col].map(normalise_state)
    df["canonical_state"] = norm_series.map(_ALIAS_TO_CANONICAL)
    df["state_code"]      = norm_series.map(_ALIAS_TO_CODE)
    df[col]               = norm_series     # overwrite with normalised form

    if warn_unresolved:
        unresolved = df.loc[df["state_code"].isna(), col].dropna().unique()
        if len(unresolved):
            sample = ", ".join(f'"{v}"' for v in sorted(unresolved)[:10])
            logger.warning(
                f"apply_state_code [{col}]: {len(unresolved)} unresolved — {sample}"
                + (" ..." if len(unresolved) > 10 else "")
            )
    return df


def validate_join_coverage(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_col: str = "canonical_state",
    right_col: str = "canonical_state",
    label: str = "",
) -> dict:
    """
    Compares state coverage between two DataFrames before a join.
    Logs warnings for states that will produce NULLs.
    Returns {"left_only", "right_only", "common"}.
    """
    ls = set(left[left_col].dropna().unique())
    rs = set(right[right_col].dropna().unique())
    lo, ro, common = ls - rs, rs - ls, ls & rs
    tag = f"[{label}] " if label else ""
    if lo:
        logger.warning(f"{tag}Join: {len(lo)} state(s) in LEFT only → will produce NULLs: "
                       + ", ".join(sorted(lo)))
    if ro:
        logger.warning(f"{tag}Join: {len(ro)} state(s) in RIGHT only → rows ignored: "
                       + ", ".join(sorted(ro)))
    logger.info(f"{tag}Join coverage: {len(common)} states matched")
    return {"left_only": lo, "right_only": ro, "common": common}


# ─────────────────────────────────────────────────────────────────────────────
# DUCKDB PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def persist_to_duckdb(con: duckdb.DuckDBPyConnection) -> None:
    """
    Writes two tables to DuckDB:

    state_master
    ┌────────────┬──────────────────────────────────────────────────────┐
    │ state_code │ canonical_state                                      │
    ├────────────┼──────────────────────────────────────────────────────┤
    │ 1          │ andhra_pradesh                                       │
    │ ...        │ ...                                                  │
    └────────────┴──────────────────────────────────────────────────────┘

    state_master_aliases
    ┌────────────┬──────────────────┬──────────────────────────────────┐
    │ state_code │ canonical_state  │ alias                            │
    ├────────────┼──────────────────┼──────────────────────────────────┤
    │ 29         │ jammu_and_kashmir│ jammu and kashmir                │
    │ 29         │ jammu_and_kashmir│ jk                               │
    │ 29         │ jammu_and_kashmir│ j and k                          │
    │ 29         │ jammu_and_kashmir│ j k                              │
    │ ...        │ ...              │ ...                              │
    └────────────┴──────────────────┴──────────────────────────────────┘

    Call once during setup or any time _STATE_RECORDS is updated.
    """
    sm   = STATE_MASTER.copy()
    sma  = STATE_MASTER_ALIASES.copy()

    con.execute('DROP TABLE IF EXISTS "state_master"')
    con.execute('CREATE TABLE "state_master" AS SELECT * FROM sm')
    logger.info(f"Stored 'state_master': {len(sm)} rows")

    con.execute('DROP TABLE IF EXISTS "state_master_aliases"')
    con.execute('CREATE TABLE "state_master_aliases" AS SELECT * FROM sma')
    logger.info(f"Stored 'state_master_aliases': {len(sma)} rows "
                f"({sma['state_code'].nunique()} states, "
                f"{len(sma)} alias entries)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI — run standalone to initialise the master tables
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DB_PATH
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    con = duckdb.connect(DB_PATH)
    persist_to_duckdb(con)

    # Quick smoke test
    tests = ["J.K.", "Jammu & Kashmir", "NCT of Delhi", "Orissa",
             "Uttaranchal", "MH", "Tamil Naidu", "Pondicherry"]
    print("\nNormalisation smoke test:")
    for t in tests:
        code = lookup_state_code(t)
        canon = lookup_canonical(t)
        print(f"  {t!r:30s} → code={code}  canonical={canon}")

    con.close()