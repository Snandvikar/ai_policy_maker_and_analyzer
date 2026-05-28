"""
dashboard/app.py
=================
Main Streamlit entry point.

Two top-level tabs:
  📊 Dashboard     — existing analysis dashboard
  🔬 Policy Simulation — new policy simulation module
"""

import streamlit as st
import pandas as pd

from config import FACTOR_LABELS
from dashboard.filters import load_all_data, render_sidebar, apply_filters
from dashboard.charts import (
    classification_bar,
    factor_avg_bar,
    mci_scatter,
    wsi_wei_quadrant,
    cluster_centroid_chart,
    rf_importance_bar,
)
from dashboard.area_detail import render_area_detail
from dashboard.chatbot import render_chatbot
from dashboard.simulation_page import render_simulation_page

st.set_page_config(
    page_title="Meaningful Connectivity Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

tab_dashboard, tab_simulation = st.tabs(
    ["📊 Dashboard", "🔬 Policy Simulation"]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

with tab_dashboard:

    data = load_all_data()

    scores_df = data["scores"]
    timeseries_df = data["timeseries"]
    sub_df = data["subcomponents"]
    imp_df = data["importance"]
    cluster_df = data["clusters"]

    filters = render_sidebar(scores_df)
    filtered_df = apply_filters(scores_df, filters)

    n_total = len(filtered_df)

    # ── Header ───────────────────────────────────────────────────────────────

    st.markdown("# 📡 Meaningful Connectivity Intelligence")

    st.markdown(
        "*AI-powered district analytics for digital inclusion, "
        "women’s safety & connectivity policy simulation across India.* \n"
        "**Coverage:** State × District"
    )

    st.markdown("---")

    # ── KPI Row ──────────────────────────────────────────────────────────────

    n_deserts = len(
        filtered_df[
            filtered_df["MCI_class"].isin(
                ["Severe desert", "Moderate desert"]
            )
        ]
    )

    n_severe = len(
        filtered_df[
            filtered_df["MCI_class"] == "Severe desert"
        ]
    )

    avg_mci = filtered_df["MCI"].mean() if n_total else 0.0
    avg_wsi = filtered_df["WSI"].mean() if n_total else 0.0
    avg_wei = filtered_df["WEI"].mean() if n_total else 0.0

    high_risk = (
        len(
            filtered_df[
                filtered_df["Safety_risk"] == "High safety risk"
            ]
        )
        if "Safety_risk" in filtered_df.columns
        else 0
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Districts", n_total)

    c2.metric(
        "Digital deserts",
        n_deserts,
        delta=f"{n_severe} severe",
        delta_color="inverse",
    )

    c3.metric("Avg MCI", f"{avg_mci:.1f}")

    c4.metric(
        "Avg WSI",
        f"{avg_wsi:.1f}",
        help="Women Safety Index — lower = higher risk",
    )

    c5.metric(
        "Avg WEI",
        f"{avg_wei:.1f}",
        help="Women Employment Index — lower = fewer opportunities",
    )

    c6.metric(
        "High safety risk",
        high_risk,
        delta_color="inverse",
    )

    st.markdown("---")

    # ── Row 1 ───────────────────────────────────────────────────────────────

    col_l, col_r = st.columns(2)

    with col_l:

        st.markdown("#### Connectivity Risk Distribution")

        if n_total:
            st.plotly_chart(
                classification_bar(filtered_df),
                use_container_width=True,
            )

    with col_r:

        st.markdown("#### Core Inclusion Indicators")

        if n_total:
            st.plotly_chart(
                factor_avg_bar(filtered_df),
                use_container_width=True,
            )

    # ── Row 2 ───────────────────────────────────────────────────────────────

    col_l2, col_r2 = st.columns(2)

    with col_l2:

        st.markdown("#### District Connectivity Rankings")

        if n_total:
            st.plotly_chart(
                mci_scatter(filtered_df),
                use_container_width=True,
            )

    with col_r2:

        st.markdown("#### Women Inclusion Opportunity Matrix")

        if n_total and "WSI" in filtered_df.columns:
            st.plotly_chart(
                wsi_wei_quadrant(filtered_df),
                use_container_width=True,
            )

    # ── Row 3: District Table ───────────────────────────────────────────────

    st.markdown("---")

    st.markdown("#### District Connectivity Intelligence")

    if n_total:

        display_cols = [
            "districtname",
            "statename",
            "MCI",
            "IFS",
            "DLS",
            "SES",
            "WDI",
            "WSI",
            "WEI",
            "MCI_class",
            "cluster_label",
        ]

        display_cols = [
            c for c in display_cols
            if c in filtered_df.columns
        ]

        table_df = (
            filtered_df[display_cols]
            .copy()
            .rename(
                columns={
                    "districtname": "District",
                    "statename": "State",
                    "cluster_label": "Cluster profile",
                }
            )
        )

        score_cols = [
            c for c in [
                "MCI",
                "IFS",
                "DLS",
                "SES",
                "WDI",
                "WSI",
                "WEI",
            ]
            if c in table_df.columns
        ]

        def _color_score(val):

            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""

            if v < 25:
                c = "#E24B4A"
            elif v < 45:
                c = "#EF9F27"
            elif v < 60:
                c = "#378ADD"
            elif v < 75:
                c = "#639922"
            else:
                c = "#1D9E75"

            return (
                f"background-color:{c}18;"
                f"color:{c};"
                f"font-weight:500"
            )

        st.dataframe(
            table_df.style.applymap(
                _color_score,
                subset=score_cols,
            ),
            use_container_width=True,
            height=320,
        )

    else:
        st.info("No districts match the current filters.")

    # ── Row 4: District Explorer ────────────────────────────────────────────

    st.markdown("---")

    st.markdown("#### District Explorer")

    selected_area_scores = None

    if n_total:

        filtered_df["_select_label"] = (
            filtered_df["districtname"]
            + "  ("
            + filtered_df["statename"]
            + ")"
        )

        selected_label = st.selectbox(
            "Select district for detailed analysis",
            ["— select —"] + filtered_df["_select_label"].tolist(),
        )

        if selected_label != "— select —":

            row = filtered_df[
                filtered_df["_select_label"] == selected_label
            ]

            if not row.empty:

                area_row_dict = row.iloc[0].to_dict()

                selected_area_scores = area_row_dict

                render_area_detail(
                    area_row=area_row_dict,
                    sub_df=sub_df,
                    timeseries_df=timeseries_df,
                )

    # ── Row 5: Cluster Profiles ─────────────────────────────────────────────

    st.markdown("---")

    st.markdown("#### Connectivity Intelligence Clusters")

    if not cluster_df.empty:

        st.plotly_chart(
            cluster_centroid_chart(cluster_df),
            use_container_width=True,
        )

        # Dropdown component to select the state filter
        unique_states = sorted(scores_df["statename"].dropna().unique().tolist())
        selected_state = st.selectbox(
            "Regional Gap Insights", 
            ["All India"] + unique_states,
            key="cluster_state_filter"
        )

        # Mapping color hexes to align with visual priorities
        CLUSTER_COLORS = [
            "#E24B4A",  # Infrastructure gap (severe)
            "#EF9F27",  # Digital literacy gap (moderate)
            "#378ADD",  # Infrastructure gap (moderate)
            "#639922",  # Infrastructure gap (mild)
            "#7F77DD",  
            "#1D9E75",  
            "#D4537E",  
        ]

        # Calculate layout configurations dynamically
        n_cls = min(3, len(cluster_df))
        c_cols = st.columns(n_cls)

        for i, (_, cl) in enumerate(cluster_df.iterrows()):
            cc = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            
            # Extract basic card context elements
            title_text = cl['label']
            
            if selected_state == "All India":
                count_val = cl['area_count']
                score_val = cl['mean_score']
            else:
                # Dynamically count districts belonging to this cluster profile within the selected state
                state_districts = scores_df[
                    (scores_df["statename"] == selected_state) & 
                    (scores_df["cluster_label"] == title_text)
                ]
                count_val = len(state_districts)
                score_val = state_districts["MCI"].mean() if count_val else 0.0

            subtitle_text = f"{count_val} district{'s' if count_val != 1 else ''} · avg score {score_val:.0f}"

            with c_cols[i % n_cls]:
                # Render only customized Title and Subtitle cards styled exactly like the original interface
                st.markdown(
                    f"""
                    <div style='
                        border-left: 4px solid {cc};
                        padding: 0.8rem 1rem;
                        background: #11151F;
                        border-radius: 0 8px 8px 0;
                        margin-bottom: 12px;
                        min-height: 75px;
                    '>
                        <div style='
                            font-size: 14px;
                            font-weight: 600;
                            color: {cc};
                            margin-bottom: 3px;
                        '>
                            {title_text}
                        </div>
                        <div style='
                            font-size: 11px;
                            color: #8A94A6;
                        '>
                            {subtitle_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Row 6: Policy Drivers ───────────────────────────────────────────────

    st.markdown("---")

    st.markdown(
        "#### High-Impact Policy Intervention Drivers"
    )

    st.caption(
        "AI-ranked indicators driving meaningful connectivity "
        "outcomes across districts."
    )

    if not imp_df.empty:

        st.plotly_chart(
            rf_importance_bar(imp_df),
            use_container_width=True,
        )

        st.info(
            "**Key finding:** No-phone household rate and "
            "illiteracy are the strongest predictors of "
            "meaningful connectivity gaps across districts."
        )

    # ── Chatbot ─────────────────────────────────────────────────────────────

    render_chatbot(
        selected_area_scores=selected_area_scores,
        mode="popup",
    )

    st.markdown("---")

    st.caption(
        "MCI = weighted geometric mean: "
        "IFS×0.35, DLS×0.30, SES×0.20, WDI×0.15. "
        "Normalisation anchored to baseline year p5/p95. "
        "Granularity: state × district. "
        "Chatbot: Mistral + LLaMA 3.1 via Ollama."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — POLICY SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

with tab_simulation:

    render_simulation_page()