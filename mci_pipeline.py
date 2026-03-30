"""
Minimum Connectivity Index (MCI) Pipeline
==========================================
Computes:
  - Infrastructure Factor Score (IFS)
  - Digital Literacy Factor Score (DLS)
  - Socio-Economic Factor Score (SES)
  - Women Digital Inclusion Factor Score (WDI)
  - Minimum Connectivity Index (MCI)
  - Women Safety Index (WSI)
  - Women Employment Opportunity Index (WEI)
  - Spearman correlation diagnostics
  - Predictive models: Random Forest regressor + K-Means clustering
                       + Logistic regression for desert classification

Stack: Python · DuckDB · pandas · scikit-learn · scipy
"""

import warnings
warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, silhouette_score
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────────────────────────────────────
# 0.  DATABASE CONNECTION & RAW DATA LOAD
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = "database.duckdb"   # change to ":memory:" for an in-memory test run

con = duckdb.connect(DB_PATH)

QUERY = """
SELECT
    -- identifiers
    c.city,
    c.area,
    c.district,
    c.area_type,
    c.population_m,
    c.area_km2,

    -- ── INFRASTRUCTURE: cell tower variables ──────────────────────────────
    c.towers_per_100k_pop,
    c.towers_per_km2,
    c.dl_median_mbps            AS cell_dl_mbps,
    c.latency_ms                AS cell_latency_ms,
    c."4gplus5g_percent"        AS gen4_5g_pct,
    c.bts_fiberized_percent     AS cell_fiberized_pct,
    c.active_5g_bts,
    c."5g_bts_fiberized"        AS bts_5g_fiberized,

    -- ── INFRASTRUCTURE: fiber / OFC variables ─────────────────────────────
    f.ofc_total_km,
    f.ftth_subs_lakh,
    f.ftth_providers,
    f.percent_ge_25_mbps,
    f.dl_median_mbps            AS fiber_dl_mbps,
    f.ul_median_mbps,
    f.pm_wani_hotspots,
    f.broadband_isps_active,

    -- ── DIGITAL LITERACY variables ─────────────────────────────────────────
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
    d.slum_internet_percent,
    d.slum_mobile_percent,
    d.slum_pop_percent,
    d.slum_mci,
    d.literacy_rate_percent     AS dl_literacy_rate,

    -- ── SOCIO-ECONOMIC variables ───────────────────────────────────────────
    s.gdp_per_capita_rs_lakh,
    s.poverty_rate_percent,
    s.gini_coeff,
    s.hdi_score,
    s.literacy_rate_percent     AS se_literacy_rate,
    s.gender_parity_index_gpi,
    s.female_literacy_percent_15_49,
    s.girls_ger_secondary_percent,
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
    s.girls_ger_higher_edu_percent,
    s.school_ner_percent

FROM infrastructure_cell_towers c
LEFT JOIN infrastructure_fiber_and_ofc  f USING (city, area, district)
LEFT JOIN digital_literacy              d USING (city, area, district)
LEFT JOIN socio_economic_indicators     s USING (city, area, district)
"""

df_raw = con.execute(QUERY).df()
print(f"[load] {len(df_raw)} rows × {df_raw.shape[1]} columns")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MISSING DATA HANDLING
#     Strategy: median imputation WITHIN area_type (Tier 1 / 2 / 3)
#     so that missing rural values are not pulled toward metro medians.
#     Columns that are identifiers or dimensions are excluded.
# ─────────────────────────────────────────────────────────────────────────────

ID_COLS = {"city", "area", "district", "area_type", "population_m", "area_km2"}
NUM_COLS = [c for c in df_raw.columns if c not in ID_COLS]

def impute_by_area_type(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """Fill nulls with the median of the same area_type group."""
    df = df.copy()
    for col in num_cols:
        if df[col].isnull().any():
            group_medians = df.groupby("area_type")[col].transform("median")
            # Fallback to global median if an entire area_type group is null
            global_median = df[col].median()
            df[col] = df[col].fillna(group_medians).fillna(global_median)
    return df

df = impute_by_area_type(df_raw, NUM_COLS)

# Derived density columns (computed AFTER imputation)
df["ofc_km_per_area_km2"] = df["ofc_total_km"] / (df["area_km2"] + 1e-6)
df["wani_per_100k"]        = df["pm_wani_hotspots"] / (df["population_m"] * 10 + 1e-6)
df["csc_per_100k"]         = df["csc_centres"] / (df["population_m"] * 10 + 1e-6)
df["shg_per_100k"]         = df["women_shg_members_k"] * 1000 / (df["population_m"] * 1e6 + 1e-6) * 1e5
df["microcredit_per_cap"]  = df["women_micro_credit_rs_cr"] / (df["population_m"] + 1e-6)
df["wealth_gap_pp"]        = df["richest_q_internet_percent"] - df["poorest_q_internet_percent"]

missing_report = df_raw[NUM_COLS].isnull().sum()
missing_report = missing_report[missing_report > 0].sort_values(ascending=False)
if not missing_report.empty:
    print("\n[missing data] columns with nulls before imputation:")
    print(missing_report.to_string())
else:
    print("[missing data] no nulls found in numeric columns")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  NORMALISATION UTILITY
#     Min-max anchored to 5th–95th national percentile.
#     invert=True flips so that "bad when high" vars become "higher = better".
# ─────────────────────────────────────────────────────────────────────────────

def normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    """
    Scale a series to [0, 100] using national 5th/95th percentile anchors.
    Clips outliers before scaling to prevent one extreme value dominating.
    """
    p5  = series.quantile(0.05)
    p95 = series.quantile(0.95)
    if p95 == p5:                         # zero-variance column → neutral 50
        return pd.Series(50.0, index=series.index)
    clipped = series.clip(p5, p95)
    scaled  = (clipped - p5) / (p95 - p5) * 100
    return (100 - scaled) if invert else scaled


# ─────────────────────────────────────────────────────────────────────────────
# 3.  INFRASTRUCTURE FACTOR SCORE (IFS)
#     Five sub-components, each is the simple mean of its normalised variables.
#     Sub-component weights sum to 1.0.
# ─────────────────────────────────────────────────────────────────────────────

def compute_IFS(df: pd.DataFrame) -> pd.Series:
    # 3a. Coverage density (20%)
    try :
        coverage = pd.concat([
            normalize(df["towers_per_100k_pop"]),
            normalize(df["towers_per_km2"]),
        ], axis=1).mean(axis=1)
    except Exception as e:
        print(f"Error computing coverage density: {e}")

    # 3b. Network quality (30%)
    #     Average DL across both tables; invert latency
    avg_dl = (df["cell_dl_mbps"] + df["fiber_dl_mbps"]) / 2
    
    try:
        quality = pd.concat([
            normalize(df["gen4_5g_pct"]),
            normalize(df["bts_5g_fiberized"]),
            normalize(avg_dl),
            normalize(df["cell_latency_ms"], invert=True),
        ], axis=1).mean(axis=1)
    except Exception as e:
        print(f"Error computing network quality: {e}")

    # 3c. Fiber backbone (25%)
    try:
        fiber = pd.concat([
            normalize(df["ofc_km_per_area_km2"]),
            normalize(df["ftth_subs_lakh"]),
            normalize(df["percent_ge_25_mbps"]),
        ], axis=1).mean(axis=1)
    except Exception as e:
        print(f"Error computing fiber backbone: {e}")

    # 3d. Last-mile access (15%)
    try:
        last_mile = pd.concat([
            normalize(df["wani_per_100k"]),
            normalize(df["broadband_isps_active"]),
        ], axis=1).mean(axis=1)
    except Exception as e:
        print(f"Error computing last-mile access: {e}")

    # 3e. Redundancy / upload capacity (10%)
    try:
        redundancy = pd.concat([
            normalize(df["active_5g_bts"]),
            normalize(df["ul_median_mbps"]),
        ], axis=1).mean(axis=1)
    except Exception as e:
        print(f"Error computing redundancy/upload capacity: {e}")

    IFS = (
        0.20 * coverage   +
        0.30 * quality    +
        0.25 * fiber      +
        0.15 * last_mile  +
        0.10 * redundancy
    )
    return IFS.rename("IFS")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DIGITAL LITERACY FACTOR SCORE (DLS)
# ─────────────────────────────────────────────────────────────────────────────

def compute_DLS(df: pd.DataFrame) -> pd.Series:
    # 4a. Household connectivity (20%)
    try:
        connectivity = pd.concat([
            normalize(df["hh_internet_percent"]),
            normalize(df["mobile_own_percent"]),
            normalize(df["smartphone_percent"]),
        ], axis=1).mean(axis=1)
    except Exception as e:
        print(f"Error computing household connectivity: {e}")
        # connectivity = pd.Series(0, index=df.index)

    # 4b. Skills (20%)
    try:
        skills = pd.concat([
            normalize(df["digital_skill_percent"]),
            normalize(df["basic_skill_percent"]),
            normalize(df["school_digital_percent"]),
        ], axis=1).mean(axis=1)
    except Exception as e:
        print(f"Error computing skills: {e}")
        # skills = pd.Series(0, index=df.index)

    # 4c. Gender inclusion (30%) — highest weight, core mandate
    try:
        gender_inc = pd.concat([
            normalize(df["women_internet_percent"]),
            normalize(df["women_mobile_percent"]),
            normalize(df["women_skill_percent"]),
            normalize(df["gender_gap_pp"], invert=True),   # gap is bad
        ], axis=1).mean(axis=1)
    except Exception as e:
        print(f"Error computing gender inclusion: {e}")
        # gender_inc = pd.Series(0, index=df.index)


    # 4d. Equity across wealth & slum populations (15%)
    try:
        equity = pd.concat([
            normalize(df["wealth_gap_pp"], invert=True),   # gap is bad
            normalize(df["slum_internet_percent"]),
            normalize(df["slum_mobile_percent"]),
        ], axis=1).mean(axis=1)
    except Exception as e:
        print(f"Error computing equity: {e}")
        # equity = pd.Series(0, index=df.index)

    # 4e. Structural literacy (15%)
    try:
        literacy = normalize(df["dl_literacy_rate"])
    except Exception as e:
        print(f"Error computing structural literacy: {e}")
        # literacy = pd.Series(0, index=df.index)

    DLS = (
        0.20 * connectivity +
        0.20 * skills       +
        0.30 * gender_inc   +
        0.15 * equity       +
        0.15 * literacy
    )
    return DLS.rename("DLS")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SOCIO-ECONOMIC FACTOR SCORE (SES)
# ─────────────────────────────────────────────────────────────────────────────

def compute_SES(df: pd.DataFrame) -> pd.Series:
    # 5a. Economic baseline (20%)
    try:
        economic = pd.concat([
            normalize(df["gdp_per_capita_rs_lakh"]),
            normalize(df["poverty_rate_percent"],   invert=True),
            normalize(df["gini_coeff"],             invert=True),
            normalize(df["hdi_score"]),
        ], axis=1).mean(axis=1)
    except Exception as e:
        print(f"Error computing economic baseline: {e}")

    # 5b. Education & gender parity (25%)
    try:
        edu_parity = pd.concat([
            normalize(df["se_literacy_rate"]),
            normalize(df["gender_parity_index_gpi"]),
            normalize(df["female_literacy_percent_15_49"]),
            normalize(df["girls_ger_secondary_percent"]),
            normalize(df["girls_school_dropout_percent"], invert=True),
        ], axis=1).mean(axis=1)
    except Exception as e:
        print(f"Error computing education & gender parity: {e}")

    # 5c. Women's economic participation (20%)
    try:
        womens_econ = pd.concat([
            normalize(df["female_lfpr_percent"]),
            normalize(df["overall_unempl_percent"], invert=True),
            normalize(df["women_gradplus_percent"]),
        ], axis=1).mean(axis=1)
    except Exception as e:
        print(f"Error computing women's economic participation: {e}")

    # 5d. Financial inclusion (15%)
    try:
        fin_incl = pd.concat([
            normalize(df["jan_dhan_account_percent"]),
            normalize(df["female_jan_dhan_accts_percent"]),
            normalize(df["shg_per_100k"]),
            normalize(df["microcredit_per_cap"]),
        ], axis=1).mean(axis=1)
    except Exception as e:
        print(f"Error computing financial inclusion: {e}")

    # 5e. Infrastructure of daily life (10%)
    try:
        life_infra = pd.concat([
            normalize(df["electricity_hrs_per_day"]),
            normalize(df["piped_water_access_percent"]),
            normalize(df["sanitation_coverage_percent"]),
            normalize(df["health_insurance_percent"]),
        ], axis=1).mean(axis=1)
    except Exception as e:
        print(f"Error computing infrastructure of daily life: {e}")

    # 5f. Government scheme reach (10%)
    try:
        schemes = pd.concat([
            normalize(df["scheme_coverage_percent"]),
            normalize(df["csc_per_100k"]),
        ], axis=1).mean(axis=1)
    except Exception as e:
        print(f"Error computing government scheme reach: {e}")

    SES = (
        0.20 * economic    +
        0.25 * edu_parity  +
        0.20 * womens_econ +
        0.15 * fin_incl    +
        0.10 * life_infra  +
        0.10 * schemes
    )
    return SES.rename("SES")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  WOMEN DIGITAL INCLUSION FACTOR SCORE (WDI)
#     Satellite factor — feeds both MCI and Women Impact Indicators
# ─────────────────────────────────────────────────────────────────────────────

def compute_WDI(df: pd.DataFrame) -> pd.Series:
    women_internet_ratio = df["women_internet_percent"] / (df["men_internet_percent"] + 1e-6) * 100
    try:
        WDI = pd.concat([
            normalize(df["women_internet_percent"]),
            normalize(df["women_mobile_percent"]),
            normalize(df["women_skill_percent"]),
            normalize(df["female_digital_skill_percent"]),
            normalize(df["gender_gap_pp"],   invert=True),
            normalize(women_internet_ratio),
            normalize(df["slum_mci"]),        # slum-level MCI as proxy for most vulnerable
        ], axis=1).mean(axis=1)

    except Exception as e:
        print(f"Error computing women's digital inclusion: {e}")

    return WDI.rename("WDI")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MINIMUM CONNECTIVITY INDEX (MCI)
#     Weighted geometric mean — forces all factors to be non-trivially positive.
#     A score of 0 on any factor collapses the composite near zero.
# ─────────────────────────────────────────────────────────────────────────────

def compute_MCI(
    IFS: pd.Series, DLS: pd.Series,
    SES: pd.Series, WDI: pd.Series,
    w_i: float = 0.35,
    w_d: float = 0.30,
    w_s: float = 0.20,
    w_w: float = 0.15,
) -> pd.Series:
    assert abs(w_i + w_d + w_s + w_w - 1.0) < 1e-9, "Weights must sum to 1"
    # Shift by +1 to avoid log(0); subtract 1 at end to restore 0-origin scale
    mci = (
        (IFS + 1) ** w_i *
        (DLS + 1) ** w_d *
        (SES + 1) ** w_s *
        (WDI + 1) ** w_w
    ) - 1
    # Re-scale to 0–100 for interpretability
    mci_scaled = (mci / ((101 ** 1.0) - 1)) * 100   # theoretical max when all = 100
    return mci_scaled.clip(0, 100).rename("MCI")


def classify_mci(mci: pd.Series) -> pd.Series:
    """Map MCI score to a desert classification label."""
    bins   = [-np.inf, 25, 45, 60, 75, np.inf]
    labels = [
        "Severe desert",
        "Moderate desert",
        "Partial connectivity",
        "Near-connected",
        "Connected",
    ]
    return pd.cut(mci, bins=bins, labels=labels).rename("MCI_class")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  WOMEN IMPACT INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def compute_WSI(df: pd.DataFrame, MCI: pd.Series) -> pd.Series:
    """
    Women's Safety Index (WSI).
    Lower score = higher safety risk.
    MCI component (40%): poor connectivity → less access to emergency services.
    Direct factors (60%): ground-level safety conditions.
    """
    mci_norm = MCI / 100   # already 0–1

    direct = pd.concat([
        normalize(df["sanitation_coverage_percent"]) * 0.30,
        normalize(df["health_insurance_percent"])     * 0.20,
        normalize(df["csc_per_100k"])                 * 0.20,
        normalize(df["shg_per_100k"])                 * 0.15,
        normalize(df["electricity_hrs_per_day"])      * 0.15,
    ], axis=1).sum(axis=1)

    WSI = (0.40 * mci_norm * 100) + (0.60 * direct)
    return WSI.clip(0, 100).rename("WSI")


def compute_WEI(df: pd.DataFrame, MCI: pd.Series) -> pd.Series:
    """
    Women's Employment Opportunity Index (WEI).
    Lower score = lower employment opportunity.
    MCI component (35%): connectivity enables remote work / gig economy access.
    Direct factors (65%): structural employment enablers.
    """
    mci_norm = MCI / 100

    direct = pd.concat([
        normalize(df["female_lfpr_percent"])           * 0.30,
        normalize(df["women_gradplus_percent"])         * 0.20,
        normalize(df["female_digital_skill_percent"])   * 0.15,
        normalize(df["microcredit_per_cap"])            * 0.15,
        normalize(df["shg_per_100k"])                   * 0.10,
        normalize(df["scheme_coverage_percent"])         * 0.10,
    ], axis=1).sum(axis=1)

    WEI = (0.35 * mci_norm * 100) + (0.65 * direct)
    return WEI.clip(0, 100).rename("WEI")


def classify_risk(wsi: pd.Series, wei: pd.Series) -> pd.DataFrame:
    """Tag areas by safety and employment risk level."""
    safety_risk = pd.cut(
        wsi, bins=[-np.inf, 35, 55, np.inf],
        labels=["High safety risk", "Moderate safety risk", "Low safety risk"]
    ).rename("Safety_risk")

    empl_gap = pd.cut(
        wei, bins=[-np.inf, 40, 60, np.inf],
        labels=["Critical employment gap", "Moderate employment gap", "Adequate opportunity"]
    ).rename("Employment_gap")

    return pd.concat([safety_risk, empl_gap], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  RUN THE FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

print("\n[pipeline] computing factor scores...")
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
risk_df   = classify_risk(df["WSI"], df["WEI"])
df = pd.concat([df, risk_df], axis=1)

SCORE_COLS = ["IFS", "DLS", "SES", "WDI", "MCI", "WSI", "WEI"]
print("\n[pipeline] score summary:")


# Saving the full score dataframe to a DuckDB table for downstream use (e.g. in PowerBI)
con.execute("CREATE OR REPLACE TABLE factor_wise_scores AS SELECT * FROM df")

summary_df = df[["city", "area", "area_type", "MCI_class"] + SCORE_COLS].describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]).round(2)

print(summary_df.to_string())

con.execute("CREATE OR REPLACE TABLE score_summary AS SELECT * FROM summary_df")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  SPEARMAN CORRELATION DIAGNOSTICS
#      Goal: check if any factor is over-represented (rho > 0.85 with MCI).
#      Also shows cross-factor correlations to detect multicollinearity.
# ─────────────────────────────────────────────────────────────────────────────

FACTOR_COLS = ["IFS", "DLS", "SES", "WDI"]

print("\n" + "═"*60)
print("SPEARMAN CORRELATION DIAGNOSTICS")
print("═"*60)

# Factor → MCI
print("\n▸ Each factor vs MCI (flag if |rho| > 0.85):")
for col in FACTOR_COLS:
    rho, pval = spearmanr(df[col], df["MCI"])
    flag = " ⚠ OVER-REPRESENTED — consider reducing weight" if abs(rho) > 0.85 else ""
    print(f"  {col} ↔ MCI  rho={rho:+.3f}  p={pval:.3e}{flag}")

# Factor → WSI / WEI
print("\n▸ Each factor vs Women Impact Indicators:")
for factor in FACTOR_COLS:
    for outcome in ["WSI", "WEI"]:
        rho, pval = spearmanr(df[factor], df[outcome])
        print(f"  {factor} ↔ {outcome}  rho={rho:+.3f}  p={pval:.3e}")

# Cross-factor correlation matrix
print("\n▸ Cross-factor correlation matrix (flag if |rho| > 0.80):")
corr_matrix = pd.DataFrame(index=FACTOR_COLS, columns=FACTOR_COLS, dtype=float)
for f1 in FACTOR_COLS:
    for f2 in FACTOR_COLS:
        rho, _ = spearmanr(df[f1], df[f2])
        corr_matrix.loc[f1, f2] = round(rho, 3)
print(corr_matrix.to_string())

# MCI vs Women Indicators
print("\n▸ MCI ↔ Women indicators:")
for col in ["WSI", "WEI"]:
    rho, pval = spearmanr(df["MCI"], df[col])
    print(f"  MCI ↔ {col}  rho={rho:+.3f}  p={pval:.3e}")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  PREDICTIVE ANALYSIS
#
#  THREE COMPLEMENTARY METHODS SUITED TO THIS USE CASE:
#
#  A. Random Forest — Feature importance: which raw data variables actually
#     drive low MCI? Tells you WHERE to intervene, not just that there's a gap.
#     Also generates continuous MCI predictions to fill in missing-data areas.
#
#  B. K-Means Clustering — Groups areas into typologically similar profiles
#     (e.g. "urban infrastructure gap", "rural skill gap", "women excluded").
#     Each cluster gets a tailored policy recommendation, more actionable
#     than a single MCI score.
#
#  C. Gradient Boosting Classifier — Predicts desert classification from
#     a subset of easier-to-collect variables (for areas where full data
#     is unavailable). Useful for extending the model to uncovered areas.
# ─────────────────────────────────────────────────────────────────────────────

# Features available at the raw / pre-score stage for predictive models
PREDICTIVE_FEATURES = [
    "towers_per_100k_pop", "gen4_5g_pct", "cell_dl_mbps",
    "ofc_km_per_area_km2", "percent_ge_25_mbps",
    "hh_internet_percent", "women_internet_percent", "gender_gap_pp",
    "digital_skill_percent", "women_skill_percent",
    "hdi_score", "poverty_rate_percent", "gini_coeff",
    "female_lfpr_percent", "female_digital_skill_percent",
    "electricity_hrs_per_day", "sanitation_coverage_percent",
    "jan_dhan_account_percent",
]

# Keep only rows where all features are available
feat_df = df[PREDICTIVE_FEATURES + ["MCI", "MCI_class", "WSI", "WEI", "area_type"]].dropna(
    subset=PREDICTIVE_FEATURES
)
X = feat_df[PREDICTIVE_FEATURES].values
y_mci    = feat_df["MCI"].values
y_class  = LabelEncoder().fit_transform(feat_df["MCI_class"].astype(str))

print("\n" + "═"*60)
print("PREDICTIVE ANALYSIS")
print("═"*60)
print(f"  Rows available for prediction: {len(feat_df)}")


# ── 11A.  RANDOM FOREST — MCI SCORE PREDICTION + FEATURE IMPORTANCE ──────────

print("\n── A. Random Forest: MCI score prediction & feature importance ──")

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)
cv_scores = cross_val_score(rf, X, y_mci, cv=5, scoring="r2")
print(f"  5-fold CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Fit on full data for importance extraction
rf.fit(X, y_mci)
importances = pd.Series(rf.feature_importances_, index=PREDICTIVE_FEATURES)
importances = importances.sort_values(ascending=False)

print("\n  Feature importances (top variables driving MCI):")
for feat, imp in importances.items():
    bar = "█" * int(imp * 100)
    print(f"  {feat:<40} {imp:.4f}  {bar}")

# Identify top-3 levers per area_type
print("\n  Top intervention levers by area type:")
for atype in feat_df["area_type"].unique():
    mask = feat_df["area_type"] == atype
    if mask.sum() < 5:
        continue
    rf_local = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_local.fit(X[mask], y_mci[mask])
    local_imp = pd.Series(rf_local.feature_importances_, index=PREDICTIVE_FEATURES)
    top3 = local_imp.nlargest(3)
    print(f"\n  [{atype}]")
    for f, v in top3.items():
        print(f"    {f:<40} {v:.4f}")

# Add predicted MCI to df for areas where full data is available
df.loc[feat_df.index, "MCI_predicted"] = rf.predict(X)


# ── 11B.  K-MEANS CLUSTERING — CONNECTIVITY PROFILE TYPOLOGY ──────────────────

print("\n── B. K-Means: area profile clustering ──")

# Cluster on the four factor scores — these are already normalised to 0–100
cluster_features = ["IFS", "DLS", "SES", "WDI"]
X_cluster = df[cluster_features].dropna().values
idx_cluster = df[cluster_features].dropna().index

# Find optimal k via silhouette score (test k=3 to k=7)
print("  Silhouette scores for k=3..7:")
best_k, best_sil = 4, -1
for k in range(3, 8):
    km_test = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_test = km_test.fit_predict(X_cluster)
    sil = silhouette_score(X_cluster, labels_test)
    print(f"    k={k}  silhouette={sil:.4f}")
    if sil > best_sil:
        best_sil, best_k = sil, k

print(f"\n  → Optimal k = {best_k}  (silhouette = {best_sil:.4f})")

km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
cluster_labels = km.fit_predict(X_cluster)
df.loc[idx_cluster, "Profile_cluster"] = cluster_labels

# Describe each cluster by its centroid means
cluster_df = pd.DataFrame(
    km.cluster_centers_, columns=cluster_features,
    index=[f"Cluster {i}" for i in range(best_k)]
).round(1)
print("\n  Cluster centroids (factor score means):")
print(cluster_df.to_string())

# Auto-label clusters based on weakest factor
def label_cluster(row):
    weakest = row.idxmin()
    labels_map = {
        "IFS": "Infrastructure gap",
        "DLS": "Digital literacy gap",
        "SES": "Socio-economic barrier",
        "WDI": "Women exclusion gap",
    }
    mci_level = "severe" if row.mean() < 40 else ("moderate" if row.mean() < 60 else "mild")
    return f"{labels_map[weakest]} ({mci_level})"

cluster_df["Profile label"] = cluster_df.apply(label_cluster, axis=1)

print("\n  Cluster profile labels:")
for idx, row in cluster_df.iterrows():
    print(f"    {idx}: {row['Profile label']}")

# Map profile labels back to main df
label_map = {i: cluster_df.loc[f"Cluster {i}", "Profile label"] for i in range(best_k)}
df["Profile_label"] = df["Profile_cluster"].map(label_map)


# ── 11C.  GRADIENT BOOSTING CLASSIFIER — DESERT CLASSIFICATION PREDICTOR ──────

print("\n── C. Gradient Boosting: desert classification from proxy variables ──")
print("   (Use this to classify areas where full data is unavailable)")

# Use only variables that are easier to collect / more widely available
PROXY_FEATURES = [
    "towers_per_100k_pop", "gen4_5g_pct",
    "hh_internet_percent", "women_internet_percent",
    "hdi_score", "poverty_rate_percent",
    "electricity_hrs_per_day",
]

proxy_df = df[PROXY_FEATURES + ["MCI_class"]].dropna(subset=PROXY_FEATURES)
X_proxy  = proxy_df[PROXY_FEATURES].values
le       = LabelEncoder()
y_proxy  = le.fit_transform(proxy_df["MCI_class"].astype(str))

gbc = GradientBoostingClassifier(
    n_estimators=200, max_depth=4,
    learning_rate=0.05, subsample=0.8,
    random_state=42,
)
cv_gbc = cross_val_score(
    gbc, X_proxy, y_proxy,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1_weighted",
)
print(f"  5-fold CV F1 (weighted): {cv_gbc.mean():.3f} ± {cv_gbc.std():.3f}")

gbc.fit(X_proxy, y_proxy)
proxy_importances = pd.Series(gbc.feature_importances_, index=PROXY_FEATURES)
proxy_importances = proxy_importances.sort_values(ascending=False)
print("\n  Proxy variable importances for desert classification:")
for feat, imp in proxy_importances.items():
    bar = "█" * int(imp * 50)
    print(f"  {feat:<40} {imp:.4f}  {bar}")

df.loc[proxy_df.index, "MCI_class_predicted"] = le.inverse_transform(
    gbc.predict(X_proxy)
)

# Classification report (train = test here since this is index-scoring, not held-out eval)
print("\n  Classification report (in-sample — use only as sanity check):")
print(classification_report(
    le.inverse_transform(y_proxy),
    df.loc[proxy_df.index, "MCI_class_predicted"],
    zero_division=0,
))


# ─────────────────────────────────────────────────────────────────────────────
# 12.  WRITE RESULTS BACK TO DUCKDB
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_COLS = (
    ["city", "area", "district", "area_type", "population_m", "area_km2"]
    + SCORE_COLS
    + ["MCI_class", "Safety_risk", "Employment_gap", "Profile_label"]
    + ["MCI_predicted", "MCI_class_predicted"]
)
output_df = df[[c for c in OUTPUT_COLS if c in df.columns]].copy()

con.execute("DROP TABLE IF EXISTS mci_scores")
con.execute("CREATE OR REPLACE TABLE mci_scores AS SELECT * FROM output_df")
print("\n[output] written to DuckDB table: mci_scores")

# Also export to CSV for downstream use
output_df.to_csv("mci_scores.csv", index=False)
print("[output] written to: mci_scores.csv")

print("\n[done] ✓ MCI pipeline complete")
con.close()