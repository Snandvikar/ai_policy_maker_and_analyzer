"""
dashboard/app.py
=================
Main Streamlit entry point.

Two top-level tabs:
  📊 Dashboard     — existing analysis dashboard
  🔬 Policy Simulation — new policy simulation module
"""
import streamlit as st
st.set_page_config(
    page_title="MCI — Digital Desert Index",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ── Initial Loading Screen ─────────────────────────────────────────────
loading_screen = st.empty()

loading_screen.markdown(
    """
    <style>
    .loader-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 85vh;
        font-family: sans-serif;
        color: inherit;
    }

    .loader {
        border: 6px solid rgba(120,120,120,0.2);
        border-top: 6px solid #00C2FF;
        border-radius: 50%;
        width: 70px;
        height: 70px;
        animation: spin 1s linear infinite;
        margin-bottom: 20px;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .loading-text {
        font-size: 24px;
        font-weight: 700;
        margin-top: 10px;
    }

    .sub-text {
        font-size: 14px;
        margin-top: 8px;
        opacity: 0.75;
    }
    </style>

    <div class="loader-container">
        <div class="loader"></div>
        <div class="loading-text">📡Loading Digital Desert Dashboard</div>
        <div class="sub-text">
            Initializing analytics, charts, and policy simulation...
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

import sys
import logging
import pandas as pd
import importlib.metadata

# Configure root logger to stdout so Streamlit server shows logs
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
root = logging.getLogger()
root.handlers = []  # avoid duplicate handlers on Streamlit reruns
root.addHandler(handler)
root.setLevel(logging.INFO)

# Ensure Streamlit's own logger uses the same handler
streamlit_logger = logging.getLogger("streamlit")
streamlit_logger.handlers = []
streamlit_logger.addHandler(handler)
streamlit_logger.setLevel(logging.INFO)

packages = sorted([
    (dist.metadata["Name"], dist.version)
    for dist in importlib.metadata.distributions()
])

# st.write(packages)
# st.write(sys.version)


from config import FACTOR_LABELS
from dashboard.filters import load_all_data, render_sidebar, apply_filters
from dashboard.charts import (
    classification_bar, factor_avg_bar, mci_scatter,
    wsi_wei_quadrant, cluster_centroid_chart,
    rf_importance_bar, spearman_heatmap,
)
from dashboard.area_detail import render_area_detail
from dashboard.chatbot import render_chatbot
from dashboard.simulation_page import render_simulation_page

tab_dashboard, tab_simulation = st.tabs(["📊 Dashboard", "🔬 Policy Simulation"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

with tab_dashboard:

    data          = load_all_data()
    loading_screen.empty()
    scores_df     = data["scores"]
    timeseries_df = data["timeseries"]
    sub_df        = data["subcomponents"]
    imp_df        = data["importance"]
    cluster_df    = data["clusters"]

    filters     = render_sidebar(scores_df)
    filtered_df = apply_filters(scores_df, filters)
    n_total     = len(filtered_df)

    st.markdown("# 📡 Digital Desert Dashboard")
    st.markdown(
        "**Minimum Connectivity Index (MCI)** — identifying digital deserts "
        "and their impact on women's safety and employment across India.  \n"
        "Granularity: **State × District**"
    )
    st.markdown("---")

    # ── KPI row ───────────────────────────────────────────────────────────────
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

    # ── Row 1: Classification + Factor averages ───────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### District classification breakdown")
        if n_total:
            st.plotly_chart(classification_bar(filtered_df), use_container_width=True)
    with col_r:
        st.markdown("#### Average factor scores")
        if n_total:
            st.plotly_chart(factor_avg_bar(filtered_df), use_container_width=True)

    # ── Row 2: MCI scatter + WSI/WEI quadrant ────────────────────────────────
    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.markdown("#### MCI by district")
        if n_total:
            st.plotly_chart(mci_scatter(filtered_df), use_container_width=True)
    with col_r2:
        st.markdown("#### WSI vs WEI — women impact quadrant")
        if n_total and "WSI" in filtered_df.columns:
            st.plotly_chart(wsi_wei_quadrant(filtered_df), use_container_width=True)

    # ── Row 3: District table ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### District-level scores")
    if n_total:
        display_cols = [
            "districtname", "statename",
            "MCI", "IFS", "DLS", "SES", "WDI", "WSI", "WEI",
            "MCI_class", "cluster_label",
        ]
        display_cols = [c for c in display_cols if c in filtered_df.columns]
        table_df = filtered_df[display_cols].copy().rename(columns={
            "districtname":  "District",
            "statename":     "State",
            "cluster_label": "Cluster profile",
        })
        score_cols = [c for c in ["MCI","IFS","DLS","SES","WDI","WSI","WEI"]
                      if c in table_df.columns]

        def _color_score(val):
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            if v < 25:   c = "#E24B4A"
            elif v < 45: c = "#EF9F27"
            elif v < 60: c = "#378ADD"
            elif v < 75: c = "#639922"
            else:        c = "#1D9E75"
            return f"background-color:{c}18;color:{c};font-weight:500"

        st.dataframe(
            table_df.style.applymap(_color_score, subset=score_cols),
            use_container_width=True, height=320,
        )
    else:
        st.info("No districts match the current filters.")

    # ── Row 4: Drill-down ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### District drill-down")
    selected_area_scores = None

    if n_total:
        filtered_df["_select_label"] = (
            filtered_df["districtname"] + "  (" + filtered_df["statename"] + ")"
        )
        selected_label = st.selectbox(
            "Select district to inspect",
            ["— select —"] + filtered_df["_select_label"].tolist(),
        )
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

    # ── Row 5: Cluster profiles ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Connectivity profile clusters")
    if not cluster_df.empty:
        st.plotly_chart(cluster_centroid_chart(cluster_df), use_container_width=True)
        CLUSTER_COLORS = [
            "#E24B4A","#EF9F27","#378ADD","#7F77DD","#1D9E75","#639922","#D4537E",
        ]
        n_cls = min(3, len(cluster_df))
        c_cols = st.columns(n_cls)
        for i, (_, cl) in enumerate(cluster_df.iterrows()):
            cc = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            with c_cols[i % n_cls]:
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

    # ── Row 6: RF importance ──────────────────────────────────────────────────
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

    # ── Row 7: Spearman heatmap ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Factor Spearman correlation matrix")
    st.caption(
        "High DLS↔WDI and DLS↔SES correlation is expected — "
        "kept separate for policy-narrative clarity."
    )
    if n_total >= 5:
        st.plotly_chart(spearman_heatmap(filtered_df), use_container_width=True)

    # ── Chatbot ───────────────────────────────────────────────────────────────
    render_chatbot(selected_area_scores=selected_area_scores)

    st.markdown("---")
    st.caption(
        "MCI = weighted geometric mean: IFS×0.35, DLS×0.30, SES×0.20, WDI×0.15. "
        "Normalisation anchored to baseline year p5/p95. "
        "Granularity: state × district. "
        "Chatbot: Mistral + LLaMA 3.1 via Ollama."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — POLICY SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

with tab_simulation:
    render_simulation_page()