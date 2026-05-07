"""
pipeline/mci_pipeline.py
=========================
MCI pipeline — district-level scoring.

GRANULARITY: (canonical_state, districtname, year)
Every output row represents one district in one year.

HOW STATE MASTER IS ENFORCED:
  The preprocessor (run first) has already:
    1. Applied apply_state_code() to every source table.
    2. Written all district_* tables with canonical_state column.
    3. Written msme_normalised with canonical_state column.

  In this file, load_raw() runs LOAD_QUERY in DuckDB SQL.
  All joins in LOAD_QUERY use canonical_state — never raw statename.
  This is the only safe join key after normalisation.

  After load_raw(), apply_state_code() is called once more on the
  resulting DataFrame as a safety net (in case any residual raw
  statename values survived the SQL join).

WHAT IS IDENTICAL TO PREVIOUS VERSION:
  - Reference-year anchored normalisation (BASELINE_YEAR p5/p95)
  - normalize_col() / normalize_df() / build_normaliser()
  - compute_MCI() — weighted geometric mean formula
  - classify_mci() / classify_risk()
  - run_spearman_diagnostics()
  - Output table names: mci_scores, mci_timeseries_scores

RUN ORDER:
  python -m pipeline.preprocessor
  python -m pipeline.mci_pipeline          ← this file
  python -m pipeline.store_dashboard_tables
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import (
    DB_PATH, BASELINE_YEAR, MCI_WEIGHTS,
    CLASS_BINS, CLASS_LABELS,
    WSI_HIGH_RISK_THRESHOLD, WSI_MOD_RISK_THRESHOLD,
    WEI_CRITICAL_THRESHOLD, WEI_MODERATE_THRESHOLD,
)
from pipeline.state_master import apply_state_code

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Identifier columns — excluded from normalisation and imputation
ID_COLS = {
    "canonical_state", "statename", "statecode", "state_code",
    "districtname", "districtcode", "year", "country",
}

# Columns where HIGHER = WORSE — inverted to HIGHER = BETTER during normalisation
INVERT_COLS = {
    "illiteracy_percent",
    "gender_vulnerability_index",
    "maternal_mortality_rate",
    "dependent_women_percent",
    "share_of_slum_population",
    "crime_against_women_rate",
    "landless_manual_labour_pct",
    "no_phone_pct",
    "income_lt5k_pct",
    "destitute_pct",
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD QUERY
#
# All joins use canonical_state — the normalised key written by the preprocessor.
# Raw statename is NOT used as a join key anywhere.
#
# Tables joined:
#   socio_economic_census_7086  (anchor — district×year)
#   msme_normalised             (district×year — normalised copy of gender_wise_total_msme)
#   district_etransactions      (district×year — disaggregated from state)
#   district_wireless_teledensity (district×year — disaggregated from state)
#   district_vulnerable_women   (district×year — disaggregated from state)
#   district_slum_share         (district×year — disaggregated from state)
# ─────────────────────────────────────────────────────────────────────────────

LOAD_QUERY = """
SELECT
    -- ── Identifiers ──────────────────────────────────────────────────────────
    se.canonical_state,
    se.statename,
    se.statecode,
    se.districtname,
    se.districtcode,
    se.year,

    -- ── SOCIO-ECONOMIC CENSUS ─────────────────────────────────────────────────
    se.households,
    se.landless_manual_labour,
    se.non_agri_enterprises,
    se.destitute_households,
    se.govt_salaried,
    se.public_salaried,
    se.private_salaried,
    se.income_less_5000,
    se.income_5000_10000,
    se.income_greater_10000,
    se.motorized_vehicle,
    se.two_wheeler,
    se.four_wheeler,
    se.refrigerator,
    se.mobile_only,
    se.landline,
    se.landline_and_mobile,
    se.no_phone,
    se.kisan_credit,
    se.irrigated_land,
    se.unirrigated_land,
    se.agri_equipment,
    se.irrigation_equipment,

    -- ── GENDER-WISE MSME (normalised) ────────────────────────────────────────
    msme.female       AS msme_female,
    msme.male         AS msme_male,
    msme.total        AS msme_total,

    -- ── E-TRANSACTIONS (state→district) ──────────────────────────────────────
    et.e_transactions_per_1000_population,

    -- ── WIRELESS TELEDENSITY (state→district) ────────────────────────────────
    wtd.wireless_rural_teledensity_pct,
    wtd.wireless_urban_teledensity_pct,
    wtd.wireless_total_teledensity_pct,

    -- ── VULNERABLE WOMEN (state→district, typed distribution applied) ─────────
    vw.sex_ratio,
    vw.illiteracy_percent,
    vw.crime_against_women,
    vw.gender_vulnerability_index,
    vw.maternal_mortality_rate,
    vw.dependent_women_percent,
    vw.women_headed_households,
    vw.female_population,

    -- ── SLUM SHARE (state→district, 80/20 blended) ───────────────────────────
    sl.share_of_slum_population

FROM socio_economic_census_7086 se

-- NOTE: socio_economic_census_7086 is loaded in Python first, then its
-- canonical_state column is used here. The anchor table already has
-- canonical_state added by the preprocessor's build_district_skeleton().
-- We rely on the preprocessor having written it back, or we do the join
-- in Python after apply_state_code(). See load_raw() for details.

-- MSME: district-level, join on all three keys
LEFT JOIN msme_normalised               msme
       ON  msme.canonical_state = se.canonical_state
       AND msme.districtname    = se.districtname

-- All district_* tables: canonical_state + districtname + year
LEFT JOIN district_etransactions        et
       ON  et.canonical_state   = se.canonical_state
       AND et.districtname      = se.districtname
       AND et.year              = se.year

LEFT JOIN district_wireless_teledensity wtd
       ON  wtd.canonical_state  = se.canonical_state
       AND wtd.districtname     = se.districtname
       AND wtd.year             = se.year

LEFT JOIN district_vulnerable_women     vw
       ON  vw.canonical_state   = se.canonical_state
       AND vw.districtname      = se.districtname
       AND vw.year              = se.year

LEFT JOIN district_slum_share           sl
       ON  sl.canonical_state   = se.canonical_state
       AND sl.districtname      = se.districtname
       AND sl.year              = se.year
"""


def load_raw(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Executes LOAD_QUERY and returns one wide row per (canonical_state, district, year).

    The LOAD_QUERY requires socio_economic_census_7086 to already have a
    canonical_state column. The preprocessor's build_district_skeleton() writes
    this back to DuckDB. If for any reason canonical_state is missing from the
    anchor table, this function adds it via apply_state_code() on the result.

    All joins are on canonical_state — raw statename is never a join key.
    """
    # Ensure socio_economic_census_7086 has canonical_state before running SQL.
    # The preprocessor writes district_population_weights with canonical_state,
    # but does NOT write canonical_state back onto the raw source table.
    # We handle this by loading se separately, normalising, then doing the
    # full join in Python where needed — OR by checking if canonical_state exists.
    try:
        has_canonical = con.execute(
            "SELECT canonical_state FROM socio_economic_census_7086 LIMIT 1"
        ).df().shape[0] > 0
    except Exception:
        has_canonical = False

    if not has_canonical:
        # Add canonical_state to the raw source table so the SQL join works
        logger.info("  Adding canonical_state to socio_economic_census_7086...")
        se_raw = con.execute("SELECT * FROM socio_economic_census_7086").df()
        se_raw.columns = [c.lower().strip() for c in se_raw.columns]
        se_raw = apply_state_code(se_raw, col="statename")
        con.execute("DROP TABLE IF EXISTS socio_economic_census_7086")
        con.execute("CREATE TABLE socio_economic_census_7086 AS SELECT * FROM se_raw")

    df = con.execute(LOAD_QUERY).df()

    # Safety net: ensure canonical_state is populated on result
    if "canonical_state" not in df.columns or df["canonical_state"].isna().any():
        df = apply_state_code(df, col="statename")

    logger.info(
        f"Loaded: {len(df)} rows | "
        f"districts: {df['districtname'].nunique()} | "
        f"states: {df['canonical_state'].nunique()} | "
        f"years: {sorted(df['year'].unique())}"
    )

    # Report join quality
    null_report = {
        col: df[col].isna().sum()
        for col in ["msme_female", "e_transactions_per_1000_population",
                    "wireless_rural_teledensity_pct", "sex_ratio",
                    "share_of_slum_population"]
        if col in df.columns
    }
    null_total = sum(null_report.values())
    if null_total > 0:
        logger.warning(f"  NULLs after join (will be imputed): {null_report}")
    else:
        logger.info("  Join quality: no NULLs in key columns")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def _rate(num: pd.Series, den: pd.Series) -> pd.Series:
    """Safe percentage rate: num / den * 100, guards div-by-zero."""
    return num / den.replace(0, np.nan) * 100


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives percentage-rate columns from raw counts.
    All derivations are guarded — missing source column → NaN (imputed later).
    """
    df = df.copy()
    hh = df["households"].replace(0, np.nan)

    # Phone access rates
    df["mobile_only_pct"]  = _rate(df["mobile_only"], hh)
    df["no_phone_pct"]     = _rate(df["no_phone"], hh)
    df["landline_pct"]     = _rate(
        df["landline"].fillna(0) + df.get("landline_and_mobile", pd.Series(0, index=df.index)).fillna(0),
        hh
    )
    df["any_phone_pct"]    = 100 - df["no_phone_pct"]

    # Wealth / asset proxies
    df["income_lt5k_pct"]       = _rate(df["income_less_5000"],     hh)
    df["income_gt10k_pct"]      = _rate(df["income_greater_10000"], hh)
    df["motorized_vehicle_pct"] = _rate(df["motorized_vehicle"],    hh)
    df["four_wheeler_pct"]      = _rate(df["four_wheeler"],         hh)
    df["refrigerator_pct"]      = _rate(df["refrigerator"],         hh)

    # Deprivation indicators
    df["landless_manual_labour_pct"] = _rate(df["landless_manual_labour"], hh)
    df["destitute_pct"]              = _rate(df["destitute_households"],   hh)
    df["non_agri_enterprise_pct"]    = _rate(df["non_agri_enterprises"],   hh)

    # Employment
    if "govt_salaried" in df.columns:
        df["govt_salaried_pct"] = _rate(df["govt_salaried"], hh)

    # MSME gender participation (district-level — from msme_normalised)
    if "msme_female" in df.columns and "msme_total" in df.columns:
        df["msme_female_share_pct"] = _rate(df["msme_female"],
                                            df["msme_total"].replace(0, np.nan))
    else:
        df["msme_female_share_pct"] = np.nan

    # Wireless connectivity composite (rural weighted 60% — deserts are rural)
    if "wireless_rural_teledensity_pct" in df.columns:
        df["wireless_connectivity_score"] = (
            pd.to_numeric(df["wireless_rural_teledensity_pct"], errors="coerce") * 0.60 +
            pd.to_numeric(df["wireless_urban_teledensity_pct"], errors="coerce") * 0.40
        )
    else:
        df["wireless_connectivity_score"] = np.nan

    # Crime against women rate per lakh women
    # Both crime_against_women and female_population come from vulnerable_women table.
    # After COUNT disaggregation in preprocessor, both are district-level counts.
    # Their ratio gives the correct district-level crime rate.
    if "crime_against_women" in df.columns and "female_population" in df.columns:
        fp = pd.to_numeric(df["female_population"], errors="coerce").replace(0, np.nan)
        caw = pd.to_numeric(df["crime_against_women"], errors="coerce")
        df["crime_against_women_rate"] = caw / fp * 100_000
    elif "crime_against_women" in df.columns:
        df["crime_against_women_rate"] = pd.to_numeric(
            df["crime_against_women"], errors="coerce"
        )
    else:
        df["crime_against_women_rate"] = np.nan

    return df


# ─────────────────────────────────────────────────────────────────────────────
# IMPUTATION
# Groups on canonical_state + year (never raw statename).
# ─────────────────────────────────────────────────────────────────────────────

def impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three-pass median imputation:
      1. (canonical_state, year) — same state, same year
      2. year                    — same year, all states
      3. global median           — absolute last resort
    """
    num_cols = [c for c in df.columns
                if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])]
    df = df.copy()
    for col in num_cols:
        if not df[col].isnull().any():
            continue
        df[col] = df[col].fillna(
            df.groupby(["canonical_state", "year"])[col].transform("median")
        )
        if df[col].isnull().any():
            df[col] = df[col].fillna(
                df.groupby("year")[col].transform("median")
            )
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION — reference-year anchored
# ─────────────────────────────────────────────────────────────────────────────

def build_normaliser(df: pd.DataFrame, baseline_year: int) -> dict:
    """
    Computes p5/p95 anchors from BASELINE_YEAR rows only.
    Applying these anchors to all years ensures cross-year comparability:
    an MCI of 60 in 2021 means the same thing as 60 in 2024.
    """
    base = df[df["year"] == baseline_year]
    if base.empty:
        logger.warning(f"Baseline year {baseline_year} not in data — using all years")
        base = df

    num_cols = [c for c in df.columns
                if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])]
    anchors = {}
    for col in num_cols:
        p5  = base[col].quantile(0.05)
        p95 = base[col].quantile(0.95)
        anchors[col] = (p5, p95, col in INVERT_COLS)
    return anchors


def normalize_col(series: pd.Series, p5: float, p95: float, invert: bool) -> pd.Series:
    if p95 == p5:
        return pd.Series(50.0, index=series.index)
    scaled = (series.clip(p5, p95) - p5) / (p95 - p5) * 100
    return (100 - scaled) if invert else scaled


def normalize_df(df: pd.DataFrame, anchors: dict) -> pd.DataFrame:
    df = df.copy()
    for col, (p5, p95, invert) in anchors.items():
        if col in df.columns:
            df[f"n_{col}"] = normalize_col(df[col], p5, p95, invert)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FACTOR SCORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _mean(*cols: str, df: pd.DataFrame) -> pd.Series:
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.Series(50.0, index=df.index)
    return df[available].mean(axis=1)


def compute_IFS(df: pd.DataFrame) -> pd.Series:
    """Infrastructure Factor Score."""
    coverage = _mean(
        "n_wireless_rural_teledensity_pct",
        "n_wireless_urban_teledensity_pct",
        "n_wireless_connectivity_score",
        df=df,
    )
    access = _mean(
        "n_any_phone_pct",
        "n_mobile_only_pct",
        "n_landline_pct",
        df=df,
    )
    activity = _mean("n_e_transactions_per_1000_population", df=df)

    return (0.35 * coverage + 0.35 * access + 0.30 * activity).rename("IFS")


def compute_DLS(df: pd.DataFrame) -> pd.Series:
    """Digital Literacy Factor Score."""
    d_access = _mean(
        "n_any_phone_pct",
        "n_mobile_only_pct",
        "n_e_transactions_per_1000_population",
        df=df,
    )
    literacy = _mean("n_illiteracy_percent", df=df)   # inverted
    gender_inc = _mean(
        "n_sex_ratio",
        "n_gender_vulnerability_index",    # inverted
        "n_msme_female_share_pct",
        df=df,
    )
    econ_digital = _mean(
        "n_non_agri_enterprise_pct",
        "n_four_wheeler_pct",
        "n_refrigerator_pct",
        df=df,
    )
    return (
        0.25 * d_access + 0.25 * literacy +
        0.30 * gender_inc + 0.20 * econ_digital
    ).rename("DLS")


def compute_SES(df: pd.DataFrame) -> pd.Series:
    """Socio-Economic Factor Score."""
    economic = _mean(
        "n_income_gt10k_pct",
        "n_income_lt5k_pct",        # inverted
        "n_motorized_vehicle_pct",
        "n_refrigerator_pct",
        df=df,
    )
    deprivation = _mean(
        "n_landless_manual_labour_pct",  # inverted
        "n_destitute_pct",               # inverted
        "n_share_of_slum_population",    # inverted
        "n_no_phone_pct",                # inverted
        df=df,
    )
    womens_econ = _mean(
        "n_msme_female_share_pct",
        "n_women_headed_households",
        "n_dependent_women_percent",     # inverted
        df=df,
    )
    livelihood = _mean(
        "n_non_agri_enterprise_pct",
        "n_govt_salaried_pct",
        "n_kisan_credit",
        "n_irrigated_land",
        df=df,
    )
    return (
        0.25 * economic + 0.25 * deprivation +
        0.25 * womens_econ + 0.25 * livelihood
    ).rename("SES")


def compute_WDI(df: pd.DataFrame) -> pd.Series:
    """Women Digital Inclusion Factor Score."""
    return _mean(
        "n_gender_vulnerability_index",  # inverted
        "n_msme_female_share_pct",
        "n_sex_ratio",
        "n_illiteracy_percent",          # inverted
        "n_crime_against_women_rate",    # inverted
        "n_dependent_women_percent",     # inverted
        "n_women_headed_households",
        df=df,
    ).rename("WDI")


def compute_MCI(IFS, DLS, SES, WDI, weights=MCI_WEIGHTS) -> pd.Series:
    """Weighted geometric mean. Forces all factors to be non-trivially positive."""
    mci = (
        (IFS + 1) ** weights["IFS"] *
        (DLS + 1) ** weights["DLS"] *
        (SES + 1) ** weights["SES"] *
        (WDI + 1) ** weights["WDI"]
    ) - 1
    return (mci / ((101.0 ** 1.0) - 1) * 100).clip(0, 100).rename("MCI")


def classify_mci(mci: pd.Series) -> pd.Series:
    return pd.cut(mci, bins=CLASS_BINS, labels=CLASS_LABELS).rename("MCI_class")


def compute_WSI(df: pd.DataFrame, MCI: pd.Series) -> pd.Series:
    """Women's Safety Index."""
    direct = (
        _mean("n_crime_against_women_rate",   df=df) * 0.30 +
        _mean("n_gender_vulnerability_index", df=df) * 0.20 +
        _mean("n_maternal_mortality_rate",    df=df) * 0.20 +
        _mean("n_share_of_slum_population",   df=df) * 0.15 +
        _mean("n_wireless_connectivity_score",df=df) * 0.15
    )
    return (0.40 * MCI + 0.60 * direct).clip(0, 100).rename("WSI")


def compute_WEI(df: pd.DataFrame, MCI: pd.Series) -> pd.Series:
    """Women's Employment Opportunity Index."""
    direct = (
        _mean("n_msme_female_share_pct",              df=df) * 0.30 +
        _mean("n_non_agri_enterprise_pct",            df=df) * 0.20 +
        _mean("n_e_transactions_per_1000_population", df=df) * 0.20 +
        _mean("n_kisan_credit",                       df=df) * 0.15 +
        _mean("n_dependent_women_percent",            df=df) * 0.15
    )
    return (0.35 * MCI + 0.65 * direct).clip(0, 100).rename("WEI")


def classify_risk(wsi: pd.Series, wei: pd.Series) -> pd.DataFrame:
    safety = pd.cut(
        wsi,
        bins=[-float("inf"), WSI_HIGH_RISK_THRESHOLD, WSI_MOD_RISK_THRESHOLD, float("inf")],
        labels=["High safety risk", "Moderate safety risk", "Low safety risk"],
    ).rename("Safety_risk")
    empl = pd.cut(
        wei,
        bins=[-float("inf"), WEI_CRITICAL_THRESHOLD, WEI_MODERATE_THRESHOLD, float("inf")],
        labels=["Critical employment gap", "Moderate employment gap", "Adequate opportunity"],
    ).rename("Employment_gap")
    return pd.concat([safety, empl], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# SPEARMAN DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def run_spearman_diagnostics(df: pd.DataFrame) -> dict:
    results = {"factor_vs_mci": {}, "cross_factor": {}, "factor_vs_women": {}}
    factors = ["IFS", "DLS", "SES", "WDI"]
    for f in factors:
        rho, p = spearmanr(df[f].dropna(), df["MCI"].dropna())
        results["factor_vs_mci"][f] = {
            "rho": round(rho, 4), "p": round(p, 6),
            "over_represented": abs(rho) > 0.85,
        }
    for f1 in factors:
        for f2 in factors:
            rho, _ = spearmanr(df[f1].dropna(), df[f2].dropna())
            results["cross_factor"][f"{f1}_{f2}"] = round(rho, 4)
    for f in factors:
        for outcome in ["WSI", "WEI"]:
            rho, p = spearmanr(df[f].dropna(), df[outcome].dropna())
            results["factor_vs_women"][f"{f}_{outcome}"] = {
                "rho": round(rho, 4), "p": round(p, 6),
            }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 60)
    logger.info("MCI PIPELINE START")
    logger.info("=" * 60)

    logger.info("Step 1/8  Loading and joining raw data...")
    df_raw = load_raw(con)

    logger.info("Step 2/8  Engineering features...")
    df = engineer_features(df_raw)

    logger.info("Step 3/8  Imputing missing values...")
    df = impute(df)

    logger.info("Step 4/8  Building baseline normalisers...")
    anchors = build_normaliser(df, BASELINE_YEAR)

    logger.info("Step 5/8  Normalising all columns...")
    df = normalize_df(df, anchors)

    logger.info("Step 6/8  Computing factor scores...")
    df["IFS"] = compute_IFS(df)
    df["DLS"] = compute_DLS(df)
    df["SES"] = compute_SES(df)
    df["WDI"] = compute_WDI(df)

    logger.info("Step 7/8  Computing MCI and Women Impact Indicators...")
    df["MCI"]       = compute_MCI(df["IFS"], df["DLS"], df["SES"], df["WDI"])
    df["MCI_class"] = classify_mci(df["MCI"])
    df["WSI"]       = compute_WSI(df, df["MCI"])
    df["WEI"]       = compute_WEI(df, df["MCI"])
    risk            = classify_risk(df["WSI"], df["WEI"])
    df              = pd.concat([df, risk], axis=1)

    logger.info("Step 8/8  Running Spearman diagnostics...")
    latest = df[df["year"] == df["year"].max()]
    diag   = run_spearman_diagnostics(latest)
    for f, vals in diag["factor_vs_mci"].items():
        flag = "  ⚠ OVER-REPRESENTED" if vals["over_represented"] else ""
        logger.info(f"  {f}↔MCI  rho={vals['rho']:+.4f}{flag}")

    # ── Select output columns ─────────────────────────────────────────────────
    SCORE_COLS = ["IFS", "DLS", "SES", "WDI", "MCI",
                  "WSI", "WEI", "MCI_class", "Safety_risk", "Employment_gap"]

    # Raw vars needed by policy_engine/rules.py
    RULES_VARS = [
        "mobile_only_pct", "no_phone_pct", "any_phone_pct",
        "illiteracy_percent", "gender_vulnerability_index",
        "crime_against_women_rate", "msme_female_share_pct",
        "share_of_slum_population", "dependent_women_percent",
        "wireless_rural_teledensity_pct", "e_transactions_per_1000_population",
        "maternal_mortality_rate", "destitute_pct",
        "non_agri_enterprise_pct", "income_lt5k_pct",
    ]

    keep = (
        ["canonical_state", "statename", "statecode",
         "districtname", "districtcode", "year"]
        + SCORE_COLS
        + [c for c in RULES_VARS if c in df.columns]
    )
    out_df = df[[c for c in keep if c in df.columns]].copy()

    logger.info(
        f"Output: {len(out_df)} rows | "
        f"{out_df['districtname'].nunique()} districts | "
        f"{out_df['canonical_state'].nunique()} states"
    )
    logger.info("=" * 60)
    return out_df, df


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH)
    out_df, _ = run_pipeline(con)

    latest_df = out_df[out_df["year"] == out_df["year"].max()].copy()

    con.execute("DROP TABLE IF EXISTS mci_scores")
    con.execute("CREATE TABLE mci_scores AS SELECT * FROM latest_df")

    con.execute("DROP TABLE IF EXISTS mci_timeseries_scores")
    con.execute("CREATE TABLE mci_timeseries_scores AS SELECT * FROM out_df")

    logger.info(f"mci_scores: {len(latest_df)} rows")
    logger.info(f"mci_timeseries_scores: {len(out_df)} rows")
    con.close()