"""
dashboard/app.py
=================
Main Streamlit entry point for the MCI Digital Desert Dashboard.

Run:
    streamlit run dashboard/app.py

Install:
    pip install streamlit plotly duckdb pandas scikit-learn requests --break-system-packages

For the AI chatbot, install Ollama and pull models:
    ollama pull mistral
    ollama pull llama3.1
    ollama serve
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

data = load_all_data()
scores_df      = data["scores"]
timeseries_df  = data["timeseries"]
sub_df         = data["subcomponents"]
imp_df         = data["importance"]
cluster_df     = data["clusters"]


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
    "and their impact on women's safety and employment across India."
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────

n_deserts  = len(filtered_df[filtered_df["MCI_class"].isin(["Severe desert", "Moderate desert"])])
n_severe   = len(filtered_df[filtered_df["MCI_class"] == "Severe desert"])
avg_mci    = filtered_df["MCI"].mean() if n_total else 0
avg_wsi    = filtered_df["WSI"].mean() if n_total else 0
avg_wei    = filtered_df["WEI"].mean() if n_total else 0
high_risk  = (len(filtered_df[filtered_df["Safety_risk"] == "High safety risk"])
              if "Safety_risk" in filtered_df.columns else 0)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Areas",               n_total)
c2.metric("Digital deserts",     n_deserts,
          delta=f"{n_severe} severe", delta_color="inverse")
c3.metric("Avg MCI",             f"{avg_mci:.1f}")
c4.metric("Avg WSI",             f"{avg_wsi:.1f}",
          help="Women Safety Index — lower = higher risk")
c5.metric("Avg WEI",             f"{avg_wei:.1f}",
          help="Women Employment Index — lower = fewer opportunities")
c6.metric("High safety risk",    high_risk,
          delta_color="inverse")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# ROW 1 — Classification distribution + Factor averages
# ─────────────────────────────────────────────────────────────────────────────

col_l, col_r = st.columns(2)
with col_l:
    st.markdown("#### Area classification breakdown")
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
    st.markdown("#### MCI by area")
    if n_total:
        st.plotly_chart(mci_scatter(filtered_df), use_container_width=True)
with col_r2:
    st.markdown("#### WSI vs WEI — women impact quadrant")
    if n_total and "WSI" in filtered_df.columns:
        st.plotly_chart(wsi_wei_quadrant(filtered_df), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROW 3 — Area table
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("#### Area-level scores")

if n_total:
    display_cols = ["area", "city", "state", "city_tier",
                    "MCI", "IFS", "DLS", "SES", "WDI",
                    "WSI", "WEI", "MCI_class", "cluster_label"]
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    table_df = filtered_df[display_cols].copy()
    for col in ["MCI","IFS","DLS","SES","WDI","WSI","WEI"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].round(1)

    def color_score(val):
        if val < 25:   color = "#E24B4A"
        elif val < 45: color = "#EF9F27"
        elif val < 60: color = "#378ADD"
        elif val < 75: color = "#639922"
        else:          color = "#1D9E75"
        return f"background-color:{color}18;color:{color};font-weight:500"

    score_cols = [c for c in ["MCI","IFS","DLS","SES","WDI","WSI","WEI"]
                  if c in table_df.columns]
    styled = table_df.style.applymap(color_score, subset=score_cols)
    st.dataframe(styled, height=320)
else:
    st.info("No areas match the current filters.")


# ─────────────────────────────────────────────────────────────────────────────
# ROW 4 — Area drill-down
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("#### Area drill-down")

selected_area_scores = None   # injected into chatbot for context

area_options = (
    filtered_df
    .apply(lambda r: f"{r['area']} — {r['city']}", axis=1)
    .tolist()
)

if area_options:
    selected_str  = st.selectbox("Select area to inspect", ["— select —"] + area_options)
    if selected_str != "— select —":
        area_name = selected_str.split(" — ")[0]
        city_name = selected_str.split(" — ")[1]
        row = filtered_df[
            (filtered_df["area"] == area_name) &
            (filtered_df["city"] == city_name)
        ]
        if not row.empty:
            area_row_dict = row.iloc[0].to_dict()
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

    CLUSTER_COLORS = ["#E24B4A","#EF9F27","#378ADD","#7F77DD","#1D9E75","#639922","#D4537E"]
    c_cols = st.columns(min(3, len(cluster_df)))
    for i, (_, cl) in enumerate(cluster_df.iterrows()):
        cc = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        with c_cols[i % len(c_cols)]:
            st.markdown(
                f"<div style='border-left:3px solid {cc};padding:.6rem .8rem;"
                f"background:{cc}08;border-radius:0 8px 8px 0;margin-bottom:8px'>"
                f"<div style='font-size:13px;font-weight:600;color:{cc}'>{cl['label']}</div>"
                f"<div style='font-size:11px;color:gray;margin:.3rem 0'>"
                f"{cl['area_count']} area{'s' if cl['area_count']!=1 else ''} · "
                f"avg score {cl['mean_score']:.0f}</div>"
                f"<div style='font-size:12px;margin-bottom:.4rem'>{cl['intervention']}</div>"
                f"<div style='font-size:11px;color:gray'>Areas: {cl['area_list']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# ROW 6 — RF Feature importance
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("#### Intervention levers — random forest feature importance")
st.caption(
    "Variables ranked by their contribution to predicting MCI. "
    "Higher importance = higher-leverage policy intervention target."
)
if not imp_df.empty:
    st.plotly_chart(rf_importance_bar(imp_df), use_container_width=True)
    st.info(
        "**Key finding:** Household internet access and women's digital skills "
        "are the top predictors of MCI — tower density ranks last. "
        "Demand-side interventions (skilling, access programmes) "
        "are higher-leverage than supply-side infrastructure alone."
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROW 7 — Spearman heatmap
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("#### Factor Spearman correlation matrix")
st.caption(
    "DLS↔WDI and DLS↔SES are expected to be highly correlated. "
    "They are kept separate for policy-narrative clarity despite data overlap."
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
    "Chatbot: Mistral (data queries) + LLaMA 3.1 (policy) via Ollama."
)