"""
dashboard/app.py
=================
Main Streamlit entry point for the MCI Digital Desert Dashboard.

Run:
    streamlit run dashboard/app.py

Schema note:
  The granularity is (canonical_state, districtname, year).
  There is no 'city', 'area', or 'city_tier' column in mci_scores.
  All references use: statename (display), canonical_state (joins),
  districtname (the geographic unit).
"""

import streamlit as st
import pandas as pd

from config import FACTOR_LABELS
from dashboard.filters import load_all_data, render_sidebar, apply_filters
from dashboard.charts import (
    classification_bar, factor_avg_bar, mci_scatter,
    wsi_wei_quadrant, cluster_centroid_chart,
    rf_importance_bar, spearman_heatmap,
)
from dashboard.area_detail import render_area_detail
from dashboard.chatbot import render_chatbot


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MCI Dashboard — Digital Desert Index",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

data          = load_all_data()
scores_df     = data["scores"]
timeseries_df = data["timeseries"]
sub_df        = data["subcomponents"]
imp_df        = data["importance"]
cluster_df    = data["clusters"]


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR + FILTERS
# ─────────────────────────────────────────────────────────────────────────────

filters     = render_sidebar(scores_df)
filtered_df = apply_filters(scores_df, filters)
n_total     = len(filtered_df)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 📡 Digital Desert Dashboard")
st.markdown(
    "**Minimum Connectivity Index (MCI)** — identifying digital deserts "
    "and their impact on women's safety and employment across India.  \n"
    "Granularity: **State × District**"
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────

n_deserts = len(
    filtered_df[filtered_df["MCI_class"].isin(["Severe desert", "Moderate desert"])]
)
n_severe  = len(filtered_df[filtered_df["MCI_class"] == "Severe desert"])
avg_mci   = filtered_df["MCI"].mean() if n_total else 0.0
avg_wsi   = filtered_df["WSI"].mean() if n_total else 0.0
avg_wei   = filtered_df["WEI"].mean() if n_total else 0.0
high_risk = (
    len(filtered_df[filtered_df["Safety_risk"] == "High safety risk"])
    if "Safety_risk" in filtered_df.columns else 0
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Districts",        n_total)
c2.metric("Digital deserts",  n_deserts,
          delta=f"{n_severe} severe", delta_color="inverse")
c3.metric("Avg MCI",          f"{avg_mci:.1f}")
c4.metric("Avg WSI",          f"{avg_wsi:.1f}",
          help="Women Safety Index — lower = higher risk")
c5.metric("Avg WEI",          f"{avg_wei:.1f}",
          help="Women Employment Index — lower = fewer opportunities")
c6.metric("High safety risk", high_risk, delta_color="inverse")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# ROW 1 — Classification breakdown + Factor averages
# ─────────────────────────────────────────────────────────────────────────────

col_l, col_r = st.columns(2)
with col_l:
    st.markdown("#### District classification breakdown")
    if n_total:
        st.plotly_chart(classification_bar(filtered_df), use_container_width=True)

with col_r:
    st.markdown("#### Average factor scores")
    if n_total:
        st.plotly_chart(factor_avg_bar(filtered_df), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROW 2 — MCI scatter + WSI/WEI quadrant
# ─────────────────────────────────────────────────────────────────────────────

col_l2, col_r2 = st.columns(2)
with col_l2:
    st.markdown("#### MCI by district")
    if n_total:
        st.plotly_chart(mci_scatter(filtered_df), use_container_width=True)

with col_r2:
    st.markdown("#### WSI vs WEI — women impact quadrant")
    if n_total and "WSI" in filtered_df.columns:
        st.plotly_chart(wsi_wei_quadrant(filtered_df), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROW 3 — District-level scores table
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("#### District-level scores")

if n_total:
    display_cols = [
        "districtname", "statename",
        "MCI", "IFS", "DLS", "SES", "WDI", "WSI", "WEI",
        "MCI_class", "cluster_label",
    ]
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    table_df = filtered_df[display_cols].copy()

    # Rename for readability
    table_df = table_df.rename(columns={
        "districtname": "District",
        "statename":    "State",
        "cluster_label": "Cluster profile",
    })

    score_cols = [c for c in ["MCI", "IFS", "DLS", "SES", "WDI", "WSI", "WEI"]
                  if c in table_df.columns]

    def _color_score(val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if v < 25:   color = "#E24B4A"
        elif v < 45: color = "#EF9F27"
        elif v < 60: color = "#378ADD"
        elif v < 75: color = "#639922"
        else:        color = "#1D9E75"
        return f"background-color:{color}18;color:{color};font-weight:500"

    styled = table_df.style.applymap(_color_score, subset=score_cols)
    st.dataframe(styled, use_container_width=True, height=320)
else:
    st.info("No districts match the current filters.")


# ─────────────────────────────────────────────────────────────────────────────
# ROW 4 — District drill-down
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("#### District drill-down")

selected_area_scores = None  # passed to chatbot for context

if n_total:
    # Label: "DistrictName (StateName)"
    filtered_df["_select_label"] = (
        filtered_df["districtname"] + "  (" + filtered_df["statename"] + ")"
    )
    select_options = ["— select —"] + filtered_df["_select_label"].tolist()
    selected_label = st.selectbox("Select district to inspect", select_options)

    if selected_label != "— select —":
        row = filtered_df[filtered_df["_select_label"] == selected_label]
        if not row.empty:
            area_row_dict        = row.iloc[0].to_dict()
            selected_area_scores = area_row_dict

            render_area_detail(
                area_row=area_row_dict,
                sub_df=sub_df,
                timeseries_df=timeseries_df,
            )


# ─────────────────────────────────────────────────────────────────────────────
# ROW 5 — Cluster profiles
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("#### Connectivity profile clusters")

if not cluster_df.empty:
    st.plotly_chart(cluster_centroid_chart(cluster_df), use_container_width=True)

    CLUSTER_COLORS = [
        "#E24B4A", "#EF9F27", "#378ADD",
        "#7F77DD", "#1D9E75", "#639922", "#D4537E",
    ]
    n_cols = min(3, len(cluster_df))
    c_cols = st.columns(n_cols)
    for i, (_, cl) in enumerate(cluster_df.iterrows()):
        cc = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        with c_cols[i % n_cols]:
            st.markdown(
                f"<div style='border-left:3px solid {cc};padding:.6rem .8rem;"
                f"background:{cc}08;border-radius:0 8px 8px 0;margin-bottom:8px'>"
                f"<div style='font-size:13px;font-weight:600;color:{cc}'>"
                f"{cl['label']}</div>"
                f"<div style='font-size:11px;color:gray;margin:.3rem 0'>"
                f"{cl['area_count']} district{'s' if cl['area_count']!=1 else ''} · "
                f"avg score {cl['mean_score']:.0f}</div>"
                f"<div style='font-size:12px;margin-bottom:.4rem'>"
                f"{cl['intervention']}</div>"
                f"<div style='font-size:11px;color:gray'>"
                f"Districts: {cl['area_list']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# ROW 6 — RF Feature importance
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("#### Intervention levers — random forest feature importance")
st.caption(
    "Variables ranked by contribution to predicting MCI. "
    "Higher importance = higher-leverage policy intervention target."
)
if not imp_df.empty:
    st.plotly_chart(rf_importance_bar(imp_df), use_container_width=True)
    st.info(
        "**Key finding:** No-phone household rate and illiteracy are the top "
        "predictors of MCI in this dataset — physical access and structural "
        "literacy are the primary barriers, not just network infrastructure."
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROW 7 — Spearman correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("#### Factor Spearman correlation matrix")
st.caption(
    "High DLS↔WDI and DLS↔SES correlation is expected — "
    "kept separate for policy-narrative clarity."
)
if n_total >= 5:
    st.plotly_chart(spearman_heatmap(filtered_df), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHATBOT
# ─────────────────────────────────────────────────────────────────────────────

render_chatbot(selected_area_scores=selected_area_scores)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "MCI = weighted geometric mean: IFS×0.35, DLS×0.30, SES×0.20, WDI×0.15. "
    "Normalisation anchored to baseline year p5/p95 for cross-year comparability. "
    "WSI/WEI: 40% MCI + 60% domain-specific factors. "
    "Granularity: state × district. "
    "Chatbot: Mistral (data queries) + LLaMA 3.1 (policy) via Ollama."
)