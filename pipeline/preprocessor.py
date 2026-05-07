"""
pipeline/preprocessor.py
=========================
Disaggregates state-level tables to district level BEFORE the MCI pipeline.

WHAT THIS FILE DOES:
─────────────────────────────────────────────────────────────────────────────
Every source table at state granularity is disaggregated to (state, district).
The MCI pipeline then does a single LOAD_QUERY joining all preprocessed tables
on canonical_state + districtname + year — so ALL tables must carry
canonical_state before they are written to DuckDB here.

TABLES HANDLED:
  State-level → district (this file):
    state_ut_wireless_teledensity           → district_wireless_teledensity
    vulnerable_women_socio_economic_9429    → district_vulnerable_women
    slum_census_2011_9022 (2001+2011 blend) → district_slum_share
    electronic_transaction_aggregation      → district_etransactions

  District-level (no disaggregation needed, but still normalised here):
    socio_economic_census_7086  → used as the skeleton; canonical_state added
    gender_wise_total_msme_7973 → normalised and written as msme_normalised
                                   so LOAD_QUERY joins on canonical_state

  Why msme needs normalisation here:
    LOAD_QUERY joins msme directly in SQL. SQL cannot call apply_state_code().
    So we must normalise msme's statename → canonical_state in Python first,
    then write it back to DuckDB so the SQL join works correctly.

STATE MAPPING:
  Every table goes through apply_state_code() before any merge or write.
  All DuckDB output tables carry canonical_state.
  LOAD_QUERY in mci_pipeline.py joins exclusively on canonical_state.

TYPED COLUMN DISAGGREGATION (state-level tables only):
  RATE  → broadcast uniformly to all districts in state.
           (teledensity %, sex_ratio, GVI, illiteracy %, MMR …)
  COUNT → distribute proportionally: district_value = state_value × district_weight
           where district_weight = district_hh / state_total_hh
           (crime_against_women total, female_population, women_headed_households)
  See COLUMN_TYPES dict for the full registry.

SLUM CENSUS BLEND:
  Has 2001 and 2011 rows per state.
  Blended value = 0.80 × value_2011 + 0.20 × value_2001 per state.
  Result is broadcast (RATE) to all district×year rows.

RUN ORDER:
  python -m pipeline.preprocessor            ← this file
  python -m pipeline.mci_pipeline
  python -m pipeline.store_dashboard_tables
"""

from __future__ import annotations
import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
import numpy as np

from config import DB_PATH
from pipeline.state_master import (
    apply_state_code,
    validate_join_coverage,
    persist_to_duckdb as persist_state_master,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

SLUM_WEIGHT_2011 = 0.80
SLUM_WEIGHT_2001 = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN TYPE REGISTRY
#
# Applies ONLY to state-level tables being disaggregated to district level.
# District-level tables (socio_economic_census_7086, gender_wise_total_msme_7973)
# do NOT need this — their columns are already at district granularity.
#
# "rate"  → broadcast uniformly. The value does not change across districts.
# "count" → distribute: district_value = state_value × (district_hh / state_hh)
#
# Default for any unlisted column: "rate"
# ─────────────────────────────────────────────────────────────────────────────

COLUMN_TYPES: dict[str, str] = {
    # ── vulnerable_women_socio_economic_9429 ──────────────────────────────────
    "sex_ratio":                        "rate",   # women per 1000 men
    "illiteracy_percent":               "rate",   # percentage
    "maternal_mortality_rate":          "rate",   # per 100k live births
    "dependent_women_percent":          "rate",   # percentage
    "gender_vulnerability_index":       "rate",   # composite index 0-1
    "crime_against_women":              "count",  # raw state-level case count
    "women_headed_households":          "count",  # raw state-level count
    "female_population":                "count",  # raw state-level count

    # ── state_ut_wireless_teledensity ─────────────────────────────────────────
    "wireless_rural_teledensity_pct":   "rate",
    "wireless_urban_teledensity_pct":   "rate",
    "wireless_total_teledensity_pct":   "rate",

    # ── slum_census_2011_9022 ─────────────────────────────────────────────────
    "share_of_slum_population":         "rate",   # share (%) is a rate

    # ── electronic_transaction_aggregation ───────────────────────────────────
    "e_transactions_per_1000_population": "rate", # per-1000 is already a rate
}


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load(con: duckdb.DuckDBPyConnection, table: str) -> pd.DataFrame:
    """Load table; lowercase all column names; warn and return empty if missing."""
    try:
        df = con.execute(f'SELECT * FROM "{table}"').df()
        df.columns = [c.lower().strip() for c in df.columns]
        return df
    except Exception as e:
        logger.warning(f"Could not load '{table}': {e}")
        return pd.DataFrame()


def _write(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
    """Drop-and-recreate a table in DuckDB."""
    con.execute(f'DROP TABLE IF EXISTS "{table}"')
    con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM df')
    logger.info(f"Stored '{table}': {len(df)} rows")


def _safe_group_median(df: pd.DataFrame, col: str, group_cols: list[str]) -> pd.Series:
    """
    Compute group median without triggering RuntimeWarning on all-NaN groups.
    Groups where every value is NaN return NaN (correct behaviour, no warning).
    """
    def _med(x):
        vals = x.dropna()
        return vals.median() if len(vals) > 0 else np.nan
    return df.groupby(group_cols)[col].transform(_med)


def _fill_missing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Four-pass null fill after broadcast/distribution:
      1. (canonical_state, year) group median — same state AND year
      2. canonical_state group median         — same state, any year
      3. year group median                    — same year, any state
      4. global median                        — last resort

    Uses _safe_group_median to avoid RuntimeWarning on all-NaN groups
    (which occur for states missing entirely from the source table).
    """
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # Pass 1: same state + same year
        if df[col].isnull().any() and "year" in df.columns:
            df[col] = df[col].fillna(
                _safe_group_median(df, col, ["canonical_state", "year"])
            )

        # Pass 2: same state, any year
        if df[col].isnull().any():
            df[col] = df[col].fillna(
                _safe_group_median(df, col, ["canonical_state"])
            )

        # Pass 3: same year, any state
        if df[col].isnull().any() and "year" in df.columns:
            df[col] = df[col].fillna(
                _safe_group_median(df, col, ["year"])
            )

        # Pass 4: global median
        if df[col].isnull().any():
            global_med = df[col].median()
            df[col] = df[col].fillna(global_med if pd.notna(global_med) else 0.0)

    return df


def _distribute(
    df: pd.DataFrame,
    data_cols: list[str],
    weight_col: str = "district_pop_weight",
) -> pd.DataFrame:
    """
    For COUNT columns: district_value = state_broadcast_value × district_pop_weight
    For RATE  columns: no change (already correct after broadcast)

    Called AFTER _fill_missing so null-filled values are sensible before
    being multiplied by the population weight.
    """
    if weight_col not in df.columns:
        logger.warning(f"_distribute: '{weight_col}' not found — COUNT columns unchanged")
        return df

    distributed = []
    for col in data_cols:
        if col not in df.columns:
            continue
        if COLUMN_TYPES.get(col, "rate") == "count":
            df[col] = df[col] * df[weight_col]
            distributed.append(col)

    if distributed:
        logger.info(f"  COUNT distributed by population weight: {distributed}")
    return df


def _align_years(
    src: pd.DataFrame,
    district_frame: pd.DataFrame,
    value_cols: list[str],
    group_col: str = "canonical_state",
) -> pd.DataFrame:
    """
    Aligns source data years to the years present in district_frame.

    Problem: source tables (e.g. etransactions 2019-2023) and the district
    skeleton (e.g. socio_economic_census 2011) often have non-overlapping years.
    A direct join on year produces all-NaN rows for every district.

    Strategy:
      For each (state, district_year) pair that has no direct match in the
      source, find the closest available source year for that state and use
      its values. This is a nearest-year forward/backward fill.

      If a state has no source rows at all, those districts remain NaN
      and are handled by _fill_missing using cross-state medians.

    Returns a new src-like DataFrame where the 'year' column matches
    district_frame years, enabling a clean join.
    """
    district_years = sorted(district_frame["year"].unique())
    source_years   = sorted(src["year"].unique()) if "year" in src.columns else []

    if not source_years:
        # No year column in source — nothing to align
        return src

    # Check if years already overlap sufficiently
    overlap = set(district_years) & set(source_years)
    if len(overlap) == len(district_years):
        # Perfect overlap — no alignment needed
        return src

    logger.info(
        f"  Year alignment: district years={district_years} | "
        f"source years={source_years} | overlap={sorted(overlap)}"
    )

    # For each district year, find the nearest source year
    year_map: dict[int, int] = {}
    for dy in district_years:
        closest = min(source_years, key=lambda sy: abs(sy - dy))
        year_map[dy] = closest
        if closest != dy:
            logger.info(f"    district year {dy} → using source year {closest}")

    # Build a new src with year replaced by the district year it maps from
    aligned_frames = []
    for district_year, source_year in year_map.items():
        chunk = src[src["year"] == source_year].copy()
        if chunk.empty:
            continue
        chunk["year"] = district_year   # relabel to the district year
        aligned_frames.append(chunk)

    if not aligned_frames:
        logger.warning("  Year alignment: no source rows found for any year mapping")
        return src

    aligned = pd.concat(aligned_frames, ignore_index=True)

    # If multiple source years mapped to the same district year, average value cols
    id_cols = [c for c in aligned.columns if c not in value_cols]
    aligned = (
        aligned
        .groupby([c for c in id_cols if c in aligned.columns and c != "year"] + ["year"],
                 dropna=False)[value_cols]
        .mean()
        .reset_index()
    )

    logger.info(f"  Year alignment complete: {len(aligned)} rows in aligned source")
    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# DISTRICT SKELETON
# Built from socio_economic_census_7086 — the only table with district×year
# granularity for ALL states. It becomes the authoritative row universe.
# ─────────────────────────────────────────────────────────────────────────────

def build_district_skeleton(
    con: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        district_frame  — unique (canonical_state, statename, statecode,
                           districtname, districtcode, year) — the row universe
        weights_df      — district_frame + households + state_households
                           + district_pop_weight
                           district_pop_weight = district_hh / state_total_hh
    """
    se = _load(con, "socio_economic_census_7086")
    if se.empty:
        raise RuntimeError("socio_economic_census_7086 is required but not found.")

    # ── Step 1: apply state master ─────────────────────────────────────────────
    se = apply_state_code(se, col="statename")

    # ── Step 2: compute population weights ─────────────────────────────────────
    state_total = (
        se.groupby(["canonical_state", "year"])["households"]
        .sum()
        .reset_index()
        .rename(columns={"households": "state_households"})
    )
    se = se.merge(state_total, on=["canonical_state", "year"], how="left")
    se["district_pop_weight"] = se["households"] / (se["state_households"] + 1e-9)

    id_cols = ["canonical_state", "statename", "statecode",
               "districtname", "districtcode", "year"]
    district_frame = se[id_cols].drop_duplicates().reset_index(drop=True)
    weights_df     = se[id_cols + ["households", "state_households",
                                   "district_pop_weight"]].copy()

    logger.info(
        f"Skeleton: {len(district_frame)} district×year rows | "
        f"{district_frame['districtname'].nunique()} districts | "
        f"{district_frame['canonical_state'].nunique()} states | "
        f"years: {sorted(district_frame['year'].unique())}"
    )
    return district_frame, weights_df


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISE DISTRICT-LEVEL TABLES
#
# These tables are already at district granularity so no disaggregation
# is needed. But they DO need apply_state_code() so LOAD_QUERY can join
# them on canonical_state in SQL (SQL cannot call Python functions).
# ─────────────────────────────────────────────────────────────────────────────

def normalise_msme(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    gender_wise_total_msme_7973 is district-level — no disaggregation.
    Applies apply_state_code() then writes back as msme_normalised so
    LOAD_QUERY in mci_pipeline can join on canonical_state.
    """
    src = _load(con, "gender_wise_total_msme_7973")
    if src.empty:
        logger.warning("gender_wise_total_msme_7973 not found — skipping")
        return pd.DataFrame()

    src = src.drop(columns=["country"], errors="ignore")
    src = apply_state_code(src, col="statename")

    for col in ["male", "female", "others", "unidentified", "total"]:
        if col in src.columns:
            src[col] = pd.to_numeric(src[col], errors="coerce")

    logger.info(f"msme_normalised: {len(src)} rows | "
                f"nulls state_code: {src['state_code'].isna().sum()}")
    return src


# ─────────────────────────────────────────────────────────────────────────────
# DISAGGREGATOR 1 — state_ut_wireless_teledensity
#
# Schema: serial_no | state_ut | wireless_teledensity_total_percent |
#         wireless_teledensity_rural_percent | wireless_teledensity_urban_percent
#
# Granularity: STATE only, no year column.
# All columns: RATE → broadcast uniformly to all district×year rows.
# ─────────────────────────────────────────────────────────────────────────────

def disaggregate_wireless_teledensity(
    con: duckdb.DuckDBPyConnection,
    district_frame: pd.DataFrame,
) -> pd.DataFrame:
    """No weights_df needed — all columns are RATE."""
    src = _load(con, "state_ut_wireless_teledensity")
    if src.empty:
        logger.warning("state_ut_wireless_teledensity not found — skipping")
        return pd.DataFrame()

    # Actual schema uses 'state_ut' not 'statename'
    if "state_ut" in src.columns:
        src = src.rename(columns={"state_ut": "statename"})

    src = src.drop(columns=["serial_no", "country"], errors="ignore")
    src = apply_state_code(src, col="statename")

    # Detect columns by keyword — order: rural, urban, total
    def _pick(cols, kw):
        return next((c for c in cols if kw in c), None)

    rural_col = _pick(src.columns, "rural")
    urban_col = _pick(src.columns, "urban")
    total_col = _pick(src.columns, "total")
    found = [c for c in [rural_col, urban_col, total_col] if c]

    if not found:
        logger.warning("wireless teledensity: no rural/urban/total columns found")
        return pd.DataFrame()

    src = src[["canonical_state"] + found].copy()
    rename = {}
    if rural_col: rename[rural_col] = "wireless_rural_teledensity_pct"
    if urban_col: rename[urban_col] = "wireless_urban_teledensity_pct"
    if total_col: rename[total_col] = "wireless_total_teledensity_pct"
    src = src.rename(columns=rename)

    data_cols = list(rename.values())
    for col in data_cols:
        src[col] = pd.to_numeric(src[col], errors="coerce")

    validate_join_coverage(district_frame, src, "canonical_state", "canonical_state",
                           label="wireless_teledensity")

    # No year in source → merge on canonical_state only → broadcasts to ALL years
    merged = district_frame.merge(src, on="canonical_state", how="left")

    # All RATE — _distribute() not needed
    merged = _fill_missing(merged, data_cols)

    logger.info(f"district_wireless_teledensity: {len(merged)} rows | "
                f"nulls: {merged[data_cols].isnull().sum().sum()}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# DISAGGREGATOR 2 — vulnerable_women_socio_economic_9429
#
# Schema: country | statename | statecode | year |
#         female_population(C) | sex_ratio(R) | illiteracy_percent(R) |
#         women_headed_households(C) | maternal_mortality_rate(R) |
#         crime_against_women(C) | dependent_women_percent(R) |
#         gender_vulnerability_index(R)
#
# Granularity: STATE + year.
# Mixed types: RATE columns broadcast; COUNT columns distributed by pop weight.
# Each state×year row broadcasts to all matching district rows.
# ─────────────────────────────────────────────────────────────────────────────

def disaggregate_vulnerable_women(
    con: duckdb.DuckDBPyConnection,
    district_frame: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> pd.DataFrame:
    src = _load(con, "vulnerable_women_socio_economic_9429")
    if src.empty:
        logger.warning("vulnerable_women_socio_economic_9429 not found — skipping")
        return pd.DataFrame()

    src = src.drop(columns=["country"], errors="ignore")
    src = apply_state_code(src, col="statename")

    data_cols = [c for c in src.columns if c not in
                 ("statename", "statecode", "canonical_state", "state_code", "year")]
    for col in data_cols:
        src[col] = pd.to_numeric(src[col], errors="coerce")

    validate_join_coverage(district_frame, src, "canonical_state", "canonical_state",
                           label="vulnerable_women")

    # Align source years to district_frame years before joining
    # (source may be 2019, district skeleton may be 2011 — nearest-year fill)
    src = _align_years(src, district_frame, value_cols=data_cols,
                       group_col="canonical_state")

    # Join on canonical_state + year (each state×year → all districts in that state×year)
    merged = district_frame.merge(
        src[["canonical_state", "year"] + data_cols],
        on=["canonical_state", "year"],
        how="left",
    )

    # Attach population weights for COUNT distribution
    merged = merged.merge(
        weights_df[["canonical_state", "districtname", "districtcode",
                    "year", "district_pop_weight"]],
        on=["canonical_state", "districtname", "districtcode", "year"],
        how="left",
    )

    # Fill nulls with state/year medians BEFORE distributing counts
    # (so medians, not zeros, get distributed across districts)
    merged = _fill_missing(merged, data_cols)

    # Apply weighted distribution to COUNT columns only
    merged = _distribute(merged, data_cols, weight_col="district_pop_weight")
    merged = merged.drop(columns=["district_pop_weight"], errors="ignore")

    logger.info(f"district_vulnerable_women: {len(merged)} rows | "
                f"nulls: {merged[data_cols].isnull().sum().sum()}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# DISAGGREGATOR 3 — slum_census_2011_9022
#
# Schema: country | statename | statecode | year | share_of_slum_population
# Years present: 2001 and 2011.
# Blend: (0.80 × value_2011) + (0.20 × value_2001) per state.
# Result type: RATE → broadcast to ALL district×year rows.
# ─────────────────────────────────────────────────────────────────────────────

def disaggregate_slum_census(
    con: duckdb.DuckDBPyConnection,
    district_frame: pd.DataFrame,
) -> pd.DataFrame:
    src = _load(con, "slum_census_2011_9022")
    if src.empty:
        logger.warning("slum_census_2011_9022 not found — skipping")
        return pd.DataFrame()

    src = src.drop(columns=["country"], errors="ignore")
    src = apply_state_code(src, col="statename")
    src["share_of_slum_population"] = pd.to_numeric(
        src["share_of_slum_population"], errors="coerce"
    )

    # ── Weighted blend: 80% × 2011 + 20% × 2001 per state ────────────────────
    v2011 = (
        src[src["year"] == 2011][["canonical_state", "share_of_slum_population"]]
        .rename(columns={"share_of_slum_population": "slum_2011"})
    )
    v2001 = (
        src[src["year"] == 2001][["canonical_state", "share_of_slum_population"]]
        .rename(columns={"share_of_slum_population": "slum_2001"})
    )
    blended = v2011.merge(v2001, on="canonical_state", how="outer")

    w11 = blended["slum_2011"].notna()
    w01 = blended["slum_2001"].notna()

    blended["share_of_slum_population"] = np.where(
        w11 & w01,
        # Both years present: weighted blend
        blended["slum_2011"] * SLUM_WEIGHT_2011 + blended["slum_2001"] * SLUM_WEIGHT_2001,
        np.where(w11, blended["slum_2011"],   # only 2011 present
        np.where(w01, blended["slum_2001"],   # only 2001 present
        np.nan))                              # neither year available
    )

    logger.info(
        f"Slum blend: {int(w11.sum())} states with 2011 data | "
        f"{int(w01.sum())} with 2001 data | "
        f"{int((w11 & w01).sum())} blended ({SLUM_WEIGHT_2011:.0%}/{SLUM_WEIGHT_2001:.0%})"
    )

    validate_join_coverage(district_frame, blended, "canonical_state", "canonical_state",
                           label="slum_census")

    # RATE → broadcast single blended value to ALL district×year rows
    merged = district_frame.merge(
        blended[["canonical_state", "share_of_slum_population"]],
        on="canonical_state",
        how="left",
    )
    # National median fill for states with no slum census data at all
    merged["share_of_slum_population"] = merged["share_of_slum_population"].fillna(
        merged["share_of_slum_population"].median()
    )

    logger.info(f"district_slum_share: {len(merged)} rows | "
                f"nulls: {merged['share_of_slum_population'].isnull().sum()}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# DISAGGREGATOR 4 — electronic_transaction_aggregation_per_1000_population
#
# Schema: country | statename | statecode | year | population |
#         number_of_e_transactions | e_transactions_per_1000_population
#
# Granularity: STATE + year (confirmed — no districtname column).
# Type: RATE → broadcast per state×year to all districts.
# ─────────────────────────────────────────────────────────────────────────────

def disaggregate_etransactions(
    con: duckdb.DuckDBPyConnection,
    district_frame: pd.DataFrame,
) -> pd.DataFrame:
    src = _load(con, "electronic_transaction_aggregation_per_1000_population_9131")
    if src.empty:
        logger.warning("electronic_transaction_aggregation not found — skipping")
        return pd.DataFrame()

    # Drop raw count columns — we only need the rate
    src = src.drop(columns=["country", "population",
                             "number_of_e_transactions", "statecode"],
                   errors="ignore")
    src = apply_state_code(src, col="statename")
    src["e_transactions_per_1000_population"] = pd.to_numeric(
        src["e_transactions_per_1000_population"], errors="coerce"
    )

    # Deduplicate to one row per canonical_state × year
    src = (
        src.groupby(["canonical_state", "year"])["e_transactions_per_1000_population"]
        .mean()
        .reset_index()
    )

    validate_join_coverage(district_frame, src, "canonical_state", "canonical_state",
                           label="etransactions")

    # Align source years to district_frame years
    # (etransactions may be 2019-2023, district skeleton may be 2011)
    src = _align_years(src, district_frame,
                       value_cols=["e_transactions_per_1000_population"],
                       group_col="canonical_state")

    # RATE → broadcast per state×year to all districts in that state×year
    merged = district_frame.merge(src, on=["canonical_state", "year"], how="left")
    merged = _fill_missing(merged, ["e_transactions_per_1000_population"])

    logger.info(f"district_etransactions: {len(merged)} rows | "
                f"nulls: {merged['e_transactions_per_1000_population'].isnull().sum()}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(con: duckdb.DuckDBPyConnection) -> None:
    logger.info("=" * 60)
    logger.info("PREPROCESSOR START")
    logger.info("=" * 60)

    # ── Step 0: Persist state master tables ────────────────────────────────────
    # Writes both 'state_master' and 'state_master_aliases' to DuckDB.
    logger.info("Step 0/6  Persisting state master to DuckDB...")
    persist_state_master(con)

    # ── Step 1: District skeleton ──────────────────────────────────────────────
    logger.info("Step 1/6  Building district skeleton (socio_economic_census_7086)...")
    district_frame, weights_df = build_district_skeleton(con)

    # ── Step 2: Normalise district-level MSME table ────────────────────────────
    # Applies apply_state_code so LOAD_QUERY can join on canonical_state in SQL.
    logger.info("Step 2/6  Normalising gender_wise_total_msme_7973...")
    msme_df = normalise_msme(con)
    if not msme_df.empty:
        _write(con, "msme_normalised", msme_df)

    # ── Steps 3-6: Disaggregate state-level tables ─────────────────────────────
    logger.info("Step 3/6  Disaggregating state_ut_wireless_teledensity...")
    td_df = disaggregate_wireless_teledensity(con, district_frame)
    if not td_df.empty:
        _write(con, "district_wireless_teledensity", td_df)

    logger.info("Step 4/6  Disaggregating vulnerable_women_socio_economic_9429...")
    vw_df = disaggregate_vulnerable_women(con, district_frame, weights_df)
    if not vw_df.empty:
        _write(con, "district_vulnerable_women", vw_df)

    logger.info("Step 5/6  Disaggregating slum_census_2011_9022 (80/20 blend)...")
    sl_df = disaggregate_slum_census(con, district_frame)
    if not sl_df.empty:
        _write(con, "district_slum_share", sl_df)

    logger.info("Step 6/6  Disaggregating electronic_transaction_aggregation...")
    et_df = disaggregate_etransactions(con, district_frame)
    if not et_df.empty:
        _write(con, "district_etransactions", et_df)

    # ── Persist population weights for downstream use ──────────────────────────
    _write(con, "district_population_weights", weights_df)

    logger.info("=" * 60)
    logger.info("PREPROCESSOR DONE")
    logger.info("Tables written to DuckDB:")
    for tbl in ["state_master", "state_master_aliases",
                "msme_normalised",
                "district_wireless_teledensity", "district_vulnerable_women",
                "district_slum_share", "district_etransactions",
                "district_population_weights"]:
        try:
            n = con.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
            logger.info(f"  {tbl:<40} {n:>7} rows")
        except Exception:
            logger.info(f"  {tbl:<40} NOT WRITTEN (source missing)")
    logger.info("Run mci_pipeline.py next.")
    logger.info("=" * 60)


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH)
    run(con)
    con.close()