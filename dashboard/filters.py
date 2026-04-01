"""
dashboard/filters.py
=====================
Sidebar filter rendering and data loading for the Streamlit dashboard.
"""

from __future__ import annotations
import duckdb
import pandas as pd
import streamlit as st

from config import DB_PATH, CLASS_ORDER, LATEST_YEAR


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache(ttl=300)
def load_all_data(db_path: str = DB_PATH) -> dict:
    con = duckdb.connect(db_path, read_only=True)

    scores = con.execute("""
        SELECT city, state, city_tier, area, district, area_type,
               year, population_m, area_km2,
               IFS, DLS, SES, WDI, MCI, WSI, WEI,
               MCI_class, Safety_risk, Employment_gap,
              -- Profile_label, 
                         cluster_id, cluster_label,
               -- raw sub-vars for rules engine
               gender_gap_pp, female_lfpr_percent,
               women_skill_percent, girls_school_dropout_percent,
               poverty_rate_percent, electricity_hrs_per_day,
               crime_against_women_rate_per_lakh_women,
               female_digital_skill_percent, school_digital_percent,
               sanitation_coverage_percent
        FROM mci_scores
        ORDER BY MCI ASC
    """).df()

    timeseries = con.execute("""
        SELECT city, area, year,
               ROUND(MCI,1) AS MCI, ROUND(IFS,1) AS IFS,
               ROUND(DLS,1) AS DLS, ROUND(SES,1) AS SES,
               ROUND(WDI,1) AS WDI, ROUND(WSI,1) AS WSI, ROUND(WEI,1) AS WEI
        FROM mci_timeseries_scores
        ORDER BY city, area, year ASC
    """).df()

    subcomponents = con.execute("""
        SELECT city, area, year, factor, subcomponent, sub_score, weight_in_factor
        FROM factor_subcomponent_scores
    """).df()

    importance = con.execute("""
        SELECT feature, importance, rank, pct
        FROM rf_feature_importance
        ORDER BY rank ASC
    """).df()

    clusters = con.execute("""
        SELECT cluster_id, label, weakest_factor, mean_score, severity,
               IFS_centroid, DLS_centroid, SES_centroid, WDI_centroid,
               area_count, area_list, intervention
        FROM cluster_profiles
        ORDER BY mean_score ASC
    """).df()

    con.close()
    return {
        "scores":       scores,
        "timeseries":   timeseries,
        "subcomponents": subcomponents,
        "importance":   importance,
        "clusters":     clusters,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(scores_df: pd.DataFrame) -> dict:
    """
    Renders sidebar filters and returns a dict of selected values.
    """
    with st.sidebar:
        st.markdown("## Filters")

        # State
        state_options = ["All"] + sorted(scores_df["state"].dropna().unique().tolist())
        selected_state = st.selectbox("State", state_options, key="state_filter")

        # City tier
        tier_options = ["All"] + sorted(scores_df["city_tier"].dropna().unique().tolist())
        selected_tier = st.selectbox("City tier", tier_options, key="tier_filter")

        # City (filtered by state if selected)
        city_pool = scores_df.copy()
        if selected_state != "All":
            city_pool = city_pool[city_pool["state"] == selected_state]
        city_options = ["All"] + sorted(city_pool["city"].dropna().unique().tolist())
        selected_city = st.selectbox("City", city_options, key="city_filter")

        # Classification
        cls_options = ["All"] + CLASS_ORDER
        selected_cls = st.selectbox("MCI classification", cls_options, key="cls_filter")

        # MCI range
        mci_range = st.slider(
            "MCI score range", min_value=0, max_value=100,
            value=(0, 100), step=1, key="mci_range",
        )

        # Desert-only toggle
        desert_only = st.checkbox("Show digital deserts only", key="desert_only")

        st.markdown("---")
        st.markdown("## About")
        st.markdown(
            "**MCI** — Minimum Connectivity Index (0–100)\n\n"
            "- < 25: Severe desert\n"
            "- 25–45: Moderate desert\n"
            "- 46–60: Partial connectivity\n"
            "- 61–75: Near-connected\n"
            "- > 75: Connected\n\n"
            "**WSI** — Women Safety Index\n\n"
            "**WEI** — Women Employment Index"
        )
        st.markdown("---")
        if st.button("🔄 Reload data"):
            st.cache_data.clear()
            st.rerun()

    return {
        "state":       selected_state,
        "tier":        selected_tier,
        "city":        selected_city,
        "cls":         selected_cls,
        "mci_range":   mci_range,
        "desert_only": desert_only,
    }


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    f = df.copy()
    if filters["state"] != "All":
        f = f[f["state"] == filters["state"]]
    if filters["tier"] != "All":
        f = f[f["city_tier"] == filters["tier"]]
    if filters["city"] != "All":
        f = f[f["city"] == filters["city"]]
    if filters["cls"] != "All":
        f = f[f["MCI_class"] == filters["cls"]]
    f = f[(f["MCI"] >= filters["mci_range"][0]) & (f["MCI"] <= filters["mci_range"][1])]
    if filters["desert_only"]:
        f = f[f["MCI_class"].isin(["Severe desert", "Moderate desert"])]
    return f