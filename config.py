"""
config.py — Central configuration for the MCI project.
All other modules import from here. Change paths/models in one place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "database.duckdb"))

# ── Year settings ──────────────────────────────────────────────────────────
BASELINE_YEAR   = 2021
LATEST_YEAR     = 2025

# ── MCI weights ────────────────────────────────────────────────────────────
MCI_WEIGHTS = dict(
    IFS=0.35,
    DLS=0.30,
    SES=0.20,
    WDI=0.15
)

# ── Classification bands ──────────────────────────────────────────────────
CLASS_BINS   = [-float("inf"), 25, 45, 60, 75, float("inf")]

CLASS_LABELS = [
    "Severe desert",
    "Moderate desert",
    "Partial connectivity",
    "Near-connected",
    "Connected"
]

CLASS_ORDER = CLASS_LABELS

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

# ── Risk thresholds ───────────────────────────────────────────────────────
WSI_HIGH_RISK_THRESHOLD = 35
WSI_MOD_RISK_THRESHOLD  = 55
WEI_CRITICAL_THRESHOLD  = 40
WEI_MODERATE_THRESHOLD  = 60

# ── Agent / LLM models ────────────────────────────────────────────────────
DATA_AGENT_MODEL   = os.getenv("DATA_AGENT_MODEL", "llama-3.1-8b-instant")
POLICY_AGENT_MODEL = os.getenv("POLICY_AGENT_MODEL", "llama-3.3-70b-versatile")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Table registry ────────────────────────────────────────────────────────
RAW_TABLES = {
    "rural_wireless_market_share":
        "rural_wireless_subscriber_market_share",

    "service_area_access_subscriber_base":
        "service_area_access_subscriber_base",

    "service_area_rural_urban_subscribers":
        "service_area_rural_urban_subscribers",

    "service_area_teledensity":
        "service_area_teledensity",

    "service_area_wireless_rural_urban_subscribers":
        "service_area_wireless_rural_urban_subscribers",

    "service_area_wireless_subscriber_base":
        "service_area_wireless_subscriber_base",

    "service_area_wireless_teledensity":
        "service_area_wireless_teledensity",

    "service_provider_rural_market_share":
        "service_provider_rural_market_share",

    "service_provider_subscriber_growth":
        "service_provider_subscriber_growth",

    "state_ut_subscriber_base":
        "state_ut_subscriber_base",

    "state_ut_total_teledensity":
        "state_ut_total_teledensity",

    "state_ut_wireless_subscriber_base":
        "state_ut_wireless_subscriber_base",

    "state_ut_wireless_teledensity":
        "state_ut_wireless_teledensity",

    "subscriber_base_teledensity_rural_urban":
        "subscriber_base_teledensity_rural_urban",

    "wireless_mobile_subscriber_base":
        "wireless_mobile_subscriber_base",

    "wireless_subscriber_growth":
        "wireless_subscriber_growth",

    "wireline_subscriber_base":
        "wireline_subscriber_base",

    "socio_economic_census":
        "socio_economic_census_7086",

    "gender_wise_total_msme":
        "gender_wise_total_msme_7973",

    "slum_census":
        "slum_census_2011_9022",

    "vulnerable_women":
        "vulnerable_women_socio_economic_9429",
}


SCORE_TABLES = {
    "mci_scores":                "mci_scores",
    "subcomponents":             "factor_subcomponent_scores",
    "rf_importance":             "rf_feature_importance",
    "cluster_profiles":          "cluster_profiles",
    "timeseries_scores":         "mci_timeseries_scores",
}

# ── Schema summary for SQL agent ────────────────────────────────────────────
# This is injected into the agent system prompt so it knows what to query.
DB_SCHEMA_SUMMARY = """
IMPORTANT:
- Use exact table and column names.
- Prefer aggregate SQL queries.
- Use LIMIT 20 unless user explicitly asks for all rows.
- Use LIKE for fuzzy state/service area matching.
- Never invent columns.
SCORE TABLES (use these for most queries):

mci_scores
  Main district-level MCI scoring table.
  One row per district-year.

  Columns:
    canonical_state,
    statename,
    statecode,
    districtname,
    districtcode,
    year,
    IFS,
    DLS,
    SES,
    WDI,
    MCI,
    WSI,
    WEI,
    MCI_class,
    Safety_risk,
    Employment_gap,
    mobile_only_pct,
    no_phone_pct,
    any_phone_pct,
    illiteracy_percent,
    gender_vulnerability_index,
    crime_against_women_rate,
    msme_female_share_pct,
    share_of_slum_population,
    dependent_women_percent,
    wireless_rural_teledensity_pct,
    e_transactions_per_1000_population,
    maternal_mortality_rate,
    destitute_pct,
    non_agri_enterprise_pct,
    income_lt5k_pct,
    cluster_id,
    cluster_label,
    MCI_class_sort

  Use for:
    - state rankings
    - district rankings
    - MCI comparisons
    - identifying digital deserts
    - women inclusion analysis
    - socio-economic comparisons
    - safety and employment gap analysis

────────────────────────────────────────────────────────────────────

mci_timeseries_scores
  Historical yearly MCI scores for trend analysis.

  Same columns as mci_scores:
    canonical_state,
    statename,
    statecode,
    districtname,
    districtcode,
    year,
    IFS,
    DLS,
    SES,
    WDI,
    MCI,
    WSI,
    WEI,
    MCI_class,
    Safety_risk,
    Employment_gap,
    mobile_only_pct,
    no_phone_pct,
    any_phone_pct,
    illiteracy_percent,
    gender_vulnerability_index,
    crime_against_women_rate,
    msme_female_share_pct,
    share_of_slum_population,
    dependent_women_percent,
    wireless_rural_teledensity_pct,
    e_transactions_per_1000_population,
    maternal_mortality_rate,
    destitute_pct,
    non_agri_enterprise_pct,
    income_lt5k_pct,
    cluster_id,
    cluster_label,
    MCI_class_sort

  Use for:
    - trend analysis
    - year-over-year comparisons
    - historical MCI changes
    - temporal policy impact analysis

────────────────────────────────────────────────────────────────────

factor_subcomponent_scores
  Factor-level scoring breakdown.

  Columns:
    canonical_state,
    statename,
    districtname,
    districtcode,
    year,
    factor,
    subcomponent,
    sub_score,
    weight_in_factor

  Use for:
    - root cause analysis
    - weakest factor identification
    - explainability
    - feature contribution analysis

────────────────────────────────────────────────────────────────────

rf_feature_importance
  Random Forest feature importance ranking.

  Columns:
    feature,
    importance,
    rank,
    pct

  Use for:
    - identifying most impactful variables
    - feature importance charts
    - policy prioritization

────────────────────────────────────────────────────────────────────

cluster_profiles
  Cluster-based district archetypes.

  Columns:
    cluster_id,
    label,
    weakest_factor,
    mean_score,
    severity,
    IFS_centroid,
    DLS_centroid,
    SES_centroid,
    WDI_centroid,
    area_count,
    area_list,
    intervention,
    silhouette_score

  Use for:
    - clustering analysis
    - district archetype comparison
    - intervention recommendations
    - identifying similar districts

RAW DATA TABLES (use for detailed variable-level queries):

rural_wireless_subscriber_market_share
  Rural wireless market share by provider.
  Columns:
    service_provider,
    total_wireless_subscribers_million,
    rural_wireless_subscribers_million,
    rural_subscriber_share_percent,
    rural_market_share_percent

service_area_access_subscriber_base
  Subscriber growth across telecom circles/service areas.
  Columns:
    service_area,
    subscribers_million_mar_2024,
    subscribers_million_mar_2025,
    yearly_net_additions_million,
    yearly_growth_rate_percent

service_area_rural_urban_subscribers
  Rural vs urban subscriber distribution.
  Columns:
    service_area,
    subscribers_total_mar_2025,
    subscribers_rural_mar_2025,
    subscribers_urban_mar_2025,
    rural_subscriber_share_percent

service_area_teledensity
  Rural/urban teledensity metrics.
  Columns:
    service_area,
    rural_teledensity_percent_mar_2024,
    urban_teledensity_percent_mar_2024,
    total_teledensity_percent_mar_2024,
    rural_teledensity_percent_mar_2025,
    urban_teledensity_percent_mar_2025,
    total_teledensity_percent_mar_2025

service_area_wireless_rural_urban_subscribers
  Wireless-only subscriber distribution.
  Columns:
    service_area,
    wireless_subscribers_total_mar_2025,
    wireless_subscribers_rural_mar_2025,
    wireless_subscribers_urban_mar_2025,
    wireless_rural_subscriber_share_percent

service_area_wireless_subscriber_base
  Wireless subscriber growth by service area.
  Columns:
    service_area,
    wireless_subscriber_base_million_mar_2024,
    wireless_subscriber_base_million_mar_2025,
    wireless_net_additions_million,
    wireless_growth_rate_percent

service_area_wireless_teledensity
  Wireless teledensity metrics.
  Columns:
    service_area,
    wireless_rural_teledensity_percent_mar_2024,
    wireless_urban_teledensity_percent_mar_2024,
    wireless_total_teledensity_percent_mar_2024,
    wireless_rural_teledensity_percent_mar_2025

service_provider_rural_market_share
  Rural subscriber share by telecom operator.
  Columns:
    service_provider,
    total_subscribers_million,
    rural_subscribers_million,
    rural_subscriber_share_percent,
    rural_market_share_percent

service_provider_subscriber_growth
  Growth trends by telecom operator.
  Columns:
    service_provider,
    subscriber_base_million_mar_2024,
    subscriber_base_million_mar_2025,
    yearly_net_addition_or_decline_million,
    growth_rate_percent,
    market_share_percent_mar_2024,
    market_share_percent_mar_2025

wireless_subscriber_growth
  Wireless growth by provider.
  Columns:
    service_provider,
    wireless_subscriber_base_million_mar_2024,
    wireless_subscriber_base_million_mar_2025,
    wireless_net_additions_million,
    growth_rate_percent,
    market_share_percent_mar_2024,
    market_share_percent_mar_2025

wireless_mobile_subscriber_base
  Circle-wise telecom operator subscriber counts.
  Columns:
    circle_circle,
    bharti_airtel_may_25,
    bharti_airtel_june_25,
    reliance_may_25,
    reliance_june_25,
    vodafone_idea_may_25,
    vodafone_idea_june_25,
    bsnl_may_25,
    bsnl_june_25,
    mtnl_may_25,
    mtnl_june_25,
    reliance_jio_may_25,
    reliance_jio_june_25,
    total_may_25,
    total_june_25,
    net_addition_net_addition

wireline_subscriber_base
  Wireline subscriber base by provider.
  Columns:
    service_area,
    total_june_25,
    annexure_i_addition,
    bsnl_may_25,
    bsnl_june_25,
    bharti_airtel_may_25,
    bharti_airtel_june_25,
    reliance_jio_may_25,
    reliance_jio_june_25,
    vodafone_idea_may_25,
    vodafone_idea_june_25

state_ut_subscriber_base
  State-wise telecom subscriptions.
  Columns:
    state_ut,
    total_telephone_subscription,
    rural_telephone_subscription,
    urban_telephone_subscription

state_ut_total_teledensity
  State-wise teledensity.
  Columns:
    state_ut,
    total_teledensity_percent,
    rural_teledensity_percent,
    urban_teledensity_percent

state_ut_wireless_subscriber_base
  State-wise wireless subscribers.
  Columns:
    state_ut,
    wireless_subscribers_total_million,
    wireless_subscribers_rural_million,
    wireless_subscribers_urban_million

state_ut_wireless_teledensity
  State-wise wireless teledensity.
  Columns:
    state_ut,
    wireless_teledensity_total_percent,
    wireless_teledensity_rural_percent,
    wireless_teledensity_urban_percent

socio_economic_census_7086
  Rural socio-economic indicators.
  Columns:
    statename,
    districtname,
    households,
    landless_manual_labour,
    non_agri_enterprises,
    income_less_5000,
    income_5000_10000,
    income_greater_10000,
    landline,
    mobile_only,
    landline_and_mobile,
    no_phone,
    irrigated_land,
    unirrigated_land,
    kisan_credit

gender_wise_total_msme_7973
  Gender-wise MSME ownership.
  Columns:
    statename,
    districtname,
    male,
    female,
    others,
    total

slum_census_2011_9022
  Slum population share by state.
  Columns:
    statename,
    share_of_slum_population

vulnerable_women_socio_economic_9429
  Women vulnerability indicators.
  Columns:
    statename,
    female_population,
    sex_ratio,
    illiteracy_percent,
    women_headed_households,
    maternal_mortality_rate,
    crime_against_women,
    dependent_women_percent,
    gender_vulnerability_index

KEY JOINS:
- Most district-level analytics tables join using (statecode, districtcode, year). Use these keys whenever available for reliable joins.
- State-level telecom and socio-economic tables can be joined using normalized state names: LOWER(TRIM(state_ut)) = LOWER(TRIM(statename)).
- MCI analytics tables typically join as:
    mci_scores ↔ factor_subcomponent_scores using (statecode, districtcode, year)
    mci_scores ↔ cluster_profiles using (cluster_id)
    mci_scores ↔ mci_timeseries_scores using (statecode, districtcode)
MCI classification bands: Severe <25, Moderate 25-45, Partial 46-60,
                          Near-connected 61-75, Connected >75.
"""