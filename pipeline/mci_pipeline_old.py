"""
pipeline/mci_pipeline.py
========================
Timeseries-aware MCI pipeline.

Key change from v1:
  - Normalisation is anchored to BASELINE_YEAR (not rolling min/max).
    This ensures MCI scores are comparable across years —
    a score of 60 in 2022 means the same as 60 in 2024.
  - All four raw tables now have a `year` column.
  - women_indicators table is ingested for WSI/WEI enrichment.
  - Outputs: mci_scores (all years), mci_timeseries_scores (same, for trend charts).

Usage:
    python -m pipeline.mci_pipeline
"""

import warnings
warnings.filterwarnings("ignore")

import sys
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


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD RAW DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Join all four year-wise tables on (city, area, district, year).
    women_indicators joins on (city, year) — no area-level breakdown.
    Returns one row per (city, area, year).
    """
    query = """
    SELECT
        c.city, c.state, c.city_tier, c.area, c.district, c.year,
        c.area_type, c.population_m, c.area_km2,

        -- cell tower columns
        c.towers_per_100k_pop, c.towers_per_km2,
        c.dl_median_mbps          AS cell_dl_mbps,
        c.latency_ms              AS cell_latency_ms,
        c."4gplus5g_percent"      AS gen4_5g_pct,
        c.bts_fiberized_percent   AS cell_fiberized_pct,
        c.active_5g_bts,
        c."5g_bts_fiberized"      AS bts_5g_fiberized,

        -- fiber columns
        f.ofc_total_km,
        f.ftth_subs_lakh,
        f.percent_ge_25_mbps,
        f.percent_ge_50_mbps,
        f.dl_median_mbps          AS fiber_dl_mbps,
        f.ul_median_mbps,
        f.pm_wani_hotspots,
        f.broadband_isps_active,

        -- digital literacy columns
        d.hh_internet_percent,
        d.mobile_own_percent,
        d.smartphone_percent,
        d.digital_skill_percent,
        d.basic_skill_percent,
        d.school_digital_percent,
        d.women_internet_percent,
        d.men_internet_percent,
        d.women_mobile_percent,
        d.women_skill_percent,
        d.gender_gap_pp,
        d.richest_q_internet_percent,
        d.poorest_q_internet_percent,
        d.wealth_gap_pp,
        d.slum_internet_percent,
        d.slum_mobile_percent,
        d.slum_pop_percent,
        d.slum_mci,
        d.literacy_rate_percent   AS dl_literacy_rate,
        d.women_internet_percent_equity,

        -- socio-economic columns
        s.gdp_per_capita_rs_lakh,
        s.poverty_rate_percent,
        s.gini_coeff,
        s.hdi_score,
        s.literacy_rate_percent   AS se_literacy_rate,
        s.gender_parity_index_gpi,
        s.female_literacy_percent_15_49,
        s.girls_ger_secondary_percent,
        s.girls_ger_higher_edu_percent,
        s.girls_school_dropout_percent,
        s.female_lfpr_percent,
        s.overall_unempl_percent,
        s.women_gradplus_percent,
        s.jan_dhan_account_percent,
        s.female_jan_dhan_accts_percent,
        s.women_shg_members_k,
        s.women_micro_credit_rs_cr,
        s.scheme_coverage_percent,
        s.csc_centres,
        s.electricity_hrs_per_day,
        s.piped_water_access_percent,
        s.sanitation_coverage_percent,
        s.health_insurance_percent,
        s.female_digital_skill_percent,
        s.imd_score_0_100,
        s.higher_edu_enrol_percent,
        s.girls_ger_primary_percent,
        s.school_ner_percent,
        s.saidi_hrs_per_yr,
        s.pmay_beneficiaries_lakh,

        -- women indicators (city-level, joined on city+year)
        wi.crime_against_women_rate_per_lakh_women,
        wi.women_helpline_181_calls_received,
        wi.women_home_based_online_businesses_udyam_registered,
        wi.women_gig_per_platform_workers_registered_thousands,
        wi.women_owned_msme_enterprises_thousands,
        wi.women_shg_members_thousands

    FROM cell_towers_year_wise c
    LEFT JOIN fiber_and_ofc_year_wise     f  USING (city, area, district, year)
    LEFT JOIN digital_literacy_yw         d  USING (city, area, district, year)
    LEFT JOIN socio_economic_yw           s  USING (city, area, district, year)
    LEFT JOIN women_indicators            wi ON wi.city = c.city
                                           AND wi.year  = c.year
    """
    return con.execute(query).df()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MISSING DATA IMPUTATION
#     Impute within (area_type, year) groups so Tier-3/2024 values
#     are not pulled toward Tier-1/2021 medians.
# ─────────────────────────────────────────────────────────────────────────────

ID_COLS = {"city", "state", "city_tier", "area", "district", "year",
           "area_type", "population_m", "area_km2"}

def impute(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [c for c in df.columns if c not in ID_COLS]
    df = df.copy()

    # Derived density columns (before imputation so they can be imputed too)
    df["ofc_km_per_area_km2"]  = df["ofc_total_km"] / (df["area_km2"] + 1e-6)
    df["wani_per_100k"]        = df["pm_wani_hotspots"] / (df["population_m"] * 10 + 1e-6)
    df["csc_per_100k"]         = df["csc_centres"] / (df["population_m"] * 10 + 1e-6)
    df["shg_per_100k"]         = df["women_shg_members_k"] * 1000 / (df["population_m"] * 1e6 + 1e-6) * 1e5
    df["microcredit_per_cap"]  = df["women_micro_credit_rs_cr"] / (df["population_m"] + 1e-6)
    df["wealth_gap_pp"]        = (df["richest_q_internet_percent"] - df["poorest_q_internet_percent"]).fillna(df.get("wealth_gap_pp", 0))

    num_cols = [c for c in df.columns if c not in ID_COLS]

    for col in num_cols:
        if df[col].isnull().any():
            # 1st pass: median within area_type + year
            group_med = df.groupby(["area_type", "year"])[col].transform("median")
            df[col] = df[col].fillna(group_med)
            # 2nd pass: median within year (in case whole area_type is null)
            year_med = df.groupby("year")[col].transform("median")
            df[col] = df[col].fillna(year_med)
            # 3rd pass: global median
            df[col] = df[col].fillna(df[col].median())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  NORMALISATION
#     Reference-year anchored: p5/p95 computed from BASELINE_YEAR only,
#     then applied to all years. Scores stay comparable across time.
# ─────────────────────────────────────────────────────────────────────────────

def build_normaliser(df: pd.DataFrame, baseline_year: int):
    """
    Compute p5/p95 anchors from baseline year.
    Returns a dict: {col: (p5, p95, invert_flag)}
    """
    INVERT_COLS = {
        "cell_latency_ms", "gender_gap_pp", "poverty_rate_percent",
        "gini_coeff", "overall_unempl_percent", "girls_school_dropout_percent",
        "wealth_gap_pp", "crime_against_women_rate_per_lakh_women",
        "saidi_hrs_per_yr",
    }
    base = df[df["year"] == baseline_year]
    if base.empty:
        base = df  # fallback if baseline year not in data

    anchors = {}
    num_cols = [c for c in df.columns if c not in ID_COLS]
    for col in num_cols:
        p5  = pd.to_numeric(base[col], errors="coerce").quantile(0.05)
        p95 = pd.to_numeric(base[col], errors="coerce").quantile(0.95)
        anchors[col] = (p5, p95, col in INVERT_COLS)
    return anchors


def normalize_col(series: pd.Series, p5: float, p95: float, invert: bool) -> pd.Series:
    if pd.isna(p5) or pd.isna(p95) or p95 == p5:
        return pd.Series(50.0, index=series.index)
    scaled = (series.clip(p5, p95) - p5) / (p95 - p5) * 100
    return (100 - scaled) if invert else scaled


def normalize_df(df: pd.DataFrame, anchors: dict) -> pd.DataFrame:
    """Apply normalisation to all numeric columns in-place, returns new df."""
    df = df.copy()
    for col, (p5, p95, invert) in anchors.items():
        if col in df.columns:
            df[f"n_{col}"] = normalize_col(df[col], p5, p95, invert)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FACTOR SCORE FUNCTIONS
#     Each returns a Series. Column names prefixed n_ are normalised values.
# ─────────────────────────────────────────────────────────────────────────────

def _mean(*cols, df) -> pd.Series:
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.Series(50.0, index=df.index)
    return df[available].mean(axis=1)


def compute_IFS(df: pd.DataFrame) -> pd.Series:
    coverage   = _mean("n_towers_per_100k_pop", "n_towers_per_km2", df=df)
    quality    = _mean("n_gen4_5g_pct", "n_bts_5g_fiberized",
                       "n_cell_dl_mbps", "n_cell_latency_ms", df=df)
    fiber      = _mean("n_ofc_km_per_area_km2",
                       "n_ftth_subs_lakh", "n_percent_ge_25_mbps", df=df)
    last_mile  = _mean("n_wani_per_100k", "n_broadband_isps_active"
                       , df=df)
    redundancy = _mean("n_active_5g_bts", "n_ul_median_mbps", df=df)
    return (0.20*coverage + 0.30*quality + 0.25*fiber +
            0.15*last_mile + 0.10*redundancy).rename("IFS")


def compute_DLS(df: pd.DataFrame) -> pd.Series:
    connectivity = _mean("n_hh_internet_percent", "n_mobile_own_percent",
                         "n_smartphone_percent", df=df)
    skills       = _mean("n_digital_skill_percent", "n_basic_skill_percent",
                         "n_school_digital_percent", df=df)
    gender_inc   = _mean("n_women_internet_percent", "n_women_mobile_percent",
                         "n_women_skill_percent", "n_gender_gap_pp", df=df)
    equity       = _mean("n_wealth_gap_pp", "n_slum_internet_percent",
                         "n_slum_mobile_percent", df=df)
    literacy     = _mean("n_dl_literacy_rate", df=df)
    return (0.20*connectivity + 0.20*skills + 0.30*gender_inc +
            0.15*equity + 0.15*literacy).rename("DLS")


def compute_SES(df: pd.DataFrame) -> pd.Series:
    economic     = _mean("n_gdp_per_capita_rs_lakh", "n_poverty_rate_percent",
                         "n_gini_coeff", "n_hdi_score", df=df)
    edu_parity   = _mean("n_se_literacy_rate", "n_gender_parity_index_gpi",
                         "n_female_literacy_percent_15_49",
                         "n_girls_ger_secondary_percent",
                         "n_girls_school_dropout_percent", df=df)
    womens_econ  = _mean("n_female_lfpr_percent", "n_overall_unempl_percent",
                         "n_women_gradplus_percent", df=df)
    fin_incl     = _mean("n_jan_dhan_account_percent",
                         "n_female_jan_dhan_accts_percent",
                         "n_shg_per_100k", "n_microcredit_per_cap", df=df)
    life_infra   = _mean("n_electricity_hrs_per_day",
                         "n_piped_water_access_percent",
                         "n_sanitation_coverage_percent",
                         "n_health_insurance_percent", df=df)
    schemes      = _mean("n_scheme_coverage_percent", "n_csc_per_100k", df=df)
    return (0.20*economic + 0.25*edu_parity + 0.20*womens_econ +
            0.15*fin_incl + 0.10*life_infra + 0.10*schemes).rename("SES")


def compute_WDI(df: pd.DataFrame) -> pd.Series:
    ratio_col = "women_men_internet_ratio"
    df = df.copy()
    df[ratio_col] = df["women_internet_percent"] / (df["men_internet_percent"] + 1e-6) * 100
    ratio_p5, ratio_p95 = df[ratio_col].quantile(0.05), df[ratio_col].quantile(0.95)
    df[f"n_{ratio_col}"] = normalize_col(df[ratio_col], ratio_p5, ratio_p95, False)

    return _mean("n_women_internet_percent", "n_women_mobile_percent",
                 "n_women_skill_percent", "n_female_digital_skill_percent",
                 "n_gender_gap_pp", f"n_{ratio_col}", "n_slum_mci", df=df).rename("WDI")


def compute_MCI(IFS, DLS, SES, WDI, weights=MCI_WEIGHTS) -> pd.Series:
    mci = (
        (IFS + 1) ** weights["IFS"] *
        (DLS + 1) ** weights["DLS"] *
        (SES + 1) ** weights["SES"] *
        (WDI + 1) ** weights["WDI"]
    ) - 1
    # Rescale to 0-100
    theoretical_max = (101.0 ** 1.0) - 1
    return (mci / theoretical_max * 100).clip(0, 100).rename("MCI")


def classify_mci(mci: pd.Series) -> pd.Series:
    return pd.cut(mci, bins=CLASS_BINS, labels=CLASS_LABELS).rename("MCI_class")


def compute_WSI(df: pd.DataFrame, MCI: pd.Series) -> pd.Series:
    """
    Women's Safety Index — now incorporates women_indicators crime rate
    if available (inverted: lower crime = better score).
    """
    crime_component = pd.Series(50.0, index=df.index)
    if "n_crime_against_women_rate_per_lakh_women" in df.columns:
        crime_component = df["n_crime_against_women_rate_per_lakh_women"]

    direct = (
        _mean("n_sanitation_coverage_percent", df=df) * 0.20 +
        _mean("n_health_insurance_percent",    df=df) * 0.15 +
        _mean("n_csc_per_100k",               df=df) * 0.15 +
        _mean("n_shg_per_100k",               df=df) * 0.15 +
        _mean("n_electricity_hrs_per_day",    df=df) * 0.15 +
        crime_component                               * 0.20
    )
    return (0.40 * MCI/100 * 100 + 0.60 * direct).clip(0, 100).rename("WSI")


def compute_WEI(df: pd.DataFrame, MCI: pd.Series) -> pd.Series:
    """
    Women's Employment Index — incorporates gig workers and MSME data
    from women_indicators if available.
    """
    gig_component  = pd.Series(50.0, index=df.index)
    msme_component = pd.Series(50.0, index=df.index)
    if "n_women_gig_per_platform_workers_registered_thousands" in df.columns:
        gig_component  = df["n_women_gig_per_platform_workers_registered_thousands"]
    if "n_women_owned_msme_enterprises_thousands" in df.columns:
        msme_component = df["n_women_owned_msme_enterprises_thousands"]

    direct = (
        _mean("n_female_lfpr_percent",          df=df) * 0.25 +
        _mean("n_women_gradplus_percent",        df=df) * 0.15 +
        _mean("n_female_digital_skill_percent",  df=df) * 0.15 +
        _mean("n_microcredit_per_cap",           df=df) * 0.10 +
        _mean("n_shg_per_100k",                 df=df) * 0.10 +
        _mean("n_scheme_coverage_percent",       df=df) * 0.10 +
        gig_component                                   * 0.08 +
        msme_component                                  * 0.07
    )
    return (0.35 * MCI/100 * 100 + 0.65 * direct).clip(0, 100).rename("WEI")


def classify_risk(wsi: pd.Series, wei: pd.Series) -> pd.DataFrame:
    safety_risk = pd.cut(
        wsi, bins=[-float("inf"), WSI_HIGH_RISK_THRESHOLD,
                   WSI_MOD_RISK_THRESHOLD, float("inf")],
        labels=["High safety risk", "Moderate safety risk", "Low safety risk"]
    ).rename("Safety_risk")
    empl_gap = pd.cut(
        wei, bins=[-float("inf"), WEI_CRITICAL_THRESHOLD,
                   WEI_MODERATE_THRESHOLD, float("inf")],
        labels=["Critical employment gap", "Moderate employment gap", "Adequate opportunity"]
    ).rename("Employment_gap")
    return pd.concat([safety_risk, empl_gap], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SPEARMAN DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def run_spearman_diagnostics(df: pd.DataFrame) -> dict:
    results = {"factor_vs_mci": {}, "cross_factor": {}, "factor_vs_women": {}}
    factors = ["IFS", "DLS", "SES", "WDI"]

    for f in factors:
        rho, p = spearmanr(df[f], df["MCI"])
        results["factor_vs_mci"][f] = {"rho": round(rho, 4), "p": round(p, 6),
                                        "over_represented": abs(rho) > 0.85}

    for f1 in factors:
        for f2 in factors:
            rho, _ = spearmanr(df[f1], df[f2])
            results["cross_factor"][f"{f1}_{f2}"] = round(rho, 4)

    for f in factors:
        for outcome in ["WSI", "WEI"]:
            rho, p = spearmanr(df[f], df[outcome])
            results["factor_vs_women"][f"{f}_{outcome}"] = {
                "rho": round(rho, 4), "p": round(p, 6)
            }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    print("[pipeline] loading raw data...")
    df_raw = load_raw(con)
    print(f"  {len(df_raw)} rows × {df_raw.shape[1]} columns "
          f"| years: {sorted(df_raw['year'].unique())}")

    print("[pipeline] imputing missing values...")
    df = impute(df_raw)

    print("[pipeline] building baseline normalisers...")
    anchors = build_normaliser(df, BASELINE_YEAR)

    print("[pipeline] normalising...")
    df = normalize_df(df, anchors)

    print("[pipeline] computing factor scores...")
    df["IFS"] = compute_IFS(df)
    df["DLS"] = compute_DLS(df)
    df["SES"] = compute_SES(df)
    df["WDI"] = compute_WDI(df)

    print("[pipeline] computing MCI...")
    df["MCI"] = compute_MCI(df["IFS"], df["DLS"], df["SES"], df["WDI"])
    df["MCI_class"] = classify_mci(df["MCI"])

    print("[pipeline] computing Women Impact Indicators...")
    df["WSI"] = compute_WSI(df, df["MCI"])
    df["WEI"] = compute_WEI(df, df["MCI"])
    risk = classify_risk(df["WSI"], df["WEI"])
    df = pd.concat([df, risk], axis=1)

    # Keep only essential output columns (drop n_ normalised columns)
    keep_cols = (
        ["city", "state", "city_tier", "area", "district", "year",
         "area_type", "population_m", "area_km2",
         "IFS", "DLS", "SES", "WDI", "MCI",
         "WSI", "WEI", "MCI_class", "Safety_risk", "Employment_gap"]
        + [c for c in df_raw.columns if c not in ID_COLS]   # raw cols for sub-component
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
    out_df = df[keep_cols].copy()

    print("[pipeline] running Spearman diagnostics (latest year)...")
    latest = out_df[out_df["year"] == out_df["year"].max()]
    diag = run_spearman_diagnostics(latest)
    print(f"  IFS↔MCI rho={diag['factor_vs_mci']['IFS']['rho']}")
    print(f"  DLS↔MCI rho={diag['factor_vs_mci']['DLS']['rho']}")

    return out_df, df   # out_df = clean output; df = full df with n_ columns


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH)
    out_df, full_df = run_pipeline(con)

    # Write both score tables
    con.execute("DROP TABLE IF EXISTS mci_scores")
    latest_df = out_df[out_df["year"] == out_df["year"].max()].copy()
    con.execute("CREATE TABLE mci_scores AS SELECT * FROM latest_df")

    con.execute("DROP TABLE IF EXISTS mci_timeseries_scores")
    con.execute("CREATE TABLE mci_timeseries_scores AS SELECT * FROM out_df")

    print(f"\n[done] mci_scores: {len(latest_df)} rows")
    print(f"[done] mci_timeseries_scores: {len(out_df)} rows")
    con.close()