"""
config.py — Central configuration for the MCI project.
All other modules import from here. Change paths/models in one place.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DB_PATH  = str(BASE_DIR / "database.duckdb")

# ── Year settings ──────────────────────────────────────────────────────────
BASELINE_YEAR   = 2021   # normalization anchor year for time-series scoring
LATEST_YEAR     = 2024   # default year shown in dashboard

# ── MCI weights ────────────────────────────────────────────────────────────
MCI_WEIGHTS = dict(IFS=0.35, DLS=0.30, SES=0.20, WDI=0.15)

# ── Classification bands ────────────────────────────────────────────────────
CLASS_BINS   = [-float("inf"), 25, 45, 60, 75, float("inf")]
CLASS_LABELS = ["Severe desert", "Moderate desert",
                "Partial connectivity", "Near-connected", "Connected"]

CLASS_ORDER  = CLASS_LABELS
CLASS_COLORS = {
    "Severe desert":        "#E24B4A",
    "Moderate desert":      "#EF9F27",
    "Partial connectivity": "#378ADD",
    "Near-connected":       "#639922",
    "Connected":            "#1D9E75",
}

FACTOR_COLORS = {
    "IFS": "#378ADD",
    "DLS": "#7F77DD",
    "SES": "#1D9E75",
    "WDI": "#D4537E",
}

FACTOR_LABELS = {
    "IFS": "Infrastructure",
    "DLS": "Digital literacy",
    "SES": "Socio-economic",
    "WDI": "Women inclusion",
}

# ── Risk thresholds ─────────────────────────────────────────────────────────
WSI_HIGH_RISK_THRESHOLD  = 35
WSI_MOD_RISK_THRESHOLD   = 55
WEI_CRITICAL_THRESHOLD   = 40
WEI_MODERATE_THRESHOLD   = 60

# ── Agent / LLM models ──────────────────────────────────────────────────────
# Data Query Agent — needs strong instruction-following + SQL generation
DATA_AGENT_MODEL    = "mistral"          # via ollama  (mistral 7B)

# Policy Agent — needs reasoning + long context for narrative generation
POLICY_AGENT_MODEL  = "llama3.1"         # via ollama  (llama3.1 8B)

# Ollama base URL (local)
OLLAMA_BASE_URL     = "http://localhost:11434"

# ── DB table names ──────────────────────────────────────────────────────────
RAW_TABLES = {
    "cell_towers":      "cell_towers_year_wise",
    "fiber":            "fiber_and_ofc_year_wise",
    "digital_literacy": "digital_literacy_yw",
    "socio_economic":   "socio_economic_yw",
    "women_indicators": "women_indicators",
}

SCORE_TABLES = {
    "mci_scores":          "mci_scores",
    "subcomponents":       "factor_subcomponent_scores",
    "rf_importance":       "rf_feature_importance",
    "cluster_profiles":    "cluster_profiles",
    "timeseries_scores":   "mci_timeseries_scores",
}

# ── Schema summary for SQL agent ────────────────────────────────────────────
# This is injected into the agent system prompt so it knows what to query.
DB_SCHEMA_SUMMARY = """
SCORE TABLES (use these for most queries):

mci_scores — one row per (city, area, year).
  Columns: city, state, city_tier, area, district, area_type, year,
           population_m, area_km2,
           IFS, DLS, SES, WDI, MCI, WSI, WEI,
           MCI_class, Safety_risk, Employment_gap,
           Profile_label, cluster_id, cluster_label.

mci_timeseries_scores — same as mci_scores but includes all years for trend analysis.
  Same columns as mci_scores.

factor_subcomponent_scores — one row per (city, area, year, subcomponent).
  Columns: city, area, district, area_type, year, factor, subcomponent,
           sub_score, weight_in_factor.

rf_feature_importance — one row per feature.
  Columns: feature, importance, rank, pct.

cluster_profiles — one row per cluster.
  Columns: cluster_id, label, weakest_factor, mean_score, severity,
           IFS_centroid, DLS_centroid, SES_centroid, WDI_centroid,
           area_count, area_list, intervention.

RAW DATA TABLES (use for detailed variable-level queries):

cell_towers_year_wise — columns: city, state, city_tier, area, district, year,
  area_type, population_m, area_km2, total_towers, 2g_towers, 3g_towers,
  4g_towers, 5g_bts, 4gplus5g_percent, towers_per_km2, towers_per_100k_pop,
  bts_fiberized_percent, active_5g_bts, dl_median_mbps, latency_ms.

fiber_and_ofc_year_wise — columns: city, state, city_tier, area, district, year,
  ofc_total_km, ftth_subs_lakh, ftth_providers, dl_median_mbps, ul_median_mbps,
  latency_ms, percent_ge_25_mbps, pm_wani_hotspots, broadband_isps_active.

digital_literacy_yw — columns: city, state, city_tier, area, district, year,
  hh_internet_percent, men_internet_percent, women_internet_percent,
  gender_gap_pp, mobile_own_percent, smartphone_percent, digital_skill_percent,
  basic_skill_percent, women_mobile_percent, women_skill_percent,
  richest_q_internet_percent, poorest_q_internet_percent, wealth_gap_pp,
  slum_internet_percent, slum_mci, literacy_rate_percent.

socio_economic_yw — columns: city, state, city_tier, area, district, year,
  gdp_per_capita_rs_lakh, poverty_rate_percent, gini_coeff, hdi_score,
  female_literacy_percent_15_49, girls_school_dropout_percent,
  female_digital_skill_percent, gender_parity_index_gpi, female_lfpr_percent,
  overall_unempl_percent, jan_dhan_account_percent, women_shg_members_k,
  women_micro_credit_rs_cr, scheme_coverage_percent, electricity_hrs_per_day,
  sanitation_coverage_percent, health_insurance_percent.

women_indicators — columns: city, state, city_tier, year,
  crime_against_women_rate_per_lakh_women, women_helpline_181_calls_received,
  women_home_based_online_businesses_udyam_registered,
  women_gig_per_platform_workers_registered_thousands, female_lfpr_percent,
  women_owned_msme_enterprises_thousands, women_shg_members_thousands.

KEY JOINS: All tables join on (city, area, year) where available.
           women_indicators joins on (city, year) — no area-level breakdown.
MCI classification bands: Severe <25, Moderate 25-45, Partial 46-60,
                          Near-connected 61-75, Connected >75.
"""