"""
dashboard.py  —  MCI Digital Desert Dashboard
===============================================
Streamlit app that reads from mci.db and renders the full dashboard.

Install:
    pip install streamlit plotly duckdb pandas --break-system-packages

Run:
    streamlit run dashboard.py
"""

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = "database.duckdb"

CLASS_ORDER = ["Severe desert", "Moderate desert", "Partial connectivity",
               "Near-connected", "Connected"]

CLASS_COLORS = {
    "Severe desert":       "#E24B4A",
    "Moderate desert":     "#EF9F27",
    "Partial connectivity":"#378ADD",
    "Near-connected":      "#639922",
    "Connected":           "#1D9E75",
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

st.set_page_config(
    page_title="MCI Dashboard — Digital Desert Index",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)

    scores = con.execute("""
        SELECT city, area, district, area_type, population_m, area_km2,
               IFS, DLS, SES, WDI, MCI, WSI, WEI,
               MCI_class, Safety_risk, Employment_gap,
               Profile_label, cluster_id
        FROM mci_scores
        ORDER BY MCI ASC
    """).df()

    subcomponents = con.execute("""
        SELECT city, area, factor, subcomponent, sub_score, weight_in_factor
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
    return scores, subcomponents, importance, clusters


scores_df, sub_df, imp_df, cluster_df = load_data()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Filters")

    tier_options = ["All"] + sorted(scores_df["area_type"].dropna().unique().tolist())
    selected_tier = st.selectbox("Area tier", tier_options)

    city_options = ["All"] + sorted(scores_df["city"].dropna().unique().tolist())
    selected_city = st.selectbox("City", city_options)

    cls_options = ["All"] + CLASS_ORDER
    selected_cls = st.selectbox("MCI classification", cls_options)

    mci_range = st.slider(
        "MCI score range",
        min_value=0, max_value=100,
        value=(0, 100), step=1,
    )

    st.markdown("---")
    st.markdown("## About")
    st.markdown(
        "**MCI** — Minimum Connectivity Index\n\n"
        "Scores 0–100. Areas below **45** are classified as digital deserts.\n\n"
        "**WSI** — Women Safety Index\n\n"
        "**WEI** — Women Employment Index"
    )
    st.markdown("---")
    if st.button("Reload data"):
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def apply_filters(df):
    filtered = df.copy()
    if selected_tier != "All":
        filtered = filtered[filtered["area_type"] == selected_tier]
    if selected_city != "All":
        filtered = filtered[filtered["city"] == selected_city]
    if selected_cls != "All":
        filtered = filtered[filtered["MCI_class"] == selected_cls]
    filtered = filtered[
        (filtered["MCI"] >= mci_range[0]) &
        (filtered["MCI"] <= mci_range[1])
    ]
    return filtered

filtered_df = apply_filters(scores_df)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def score_color(val):
    if val < 25:   return "#E24B4A"
    elif val < 45: return "#EF9F27"
    elif val < 60: return "#378ADD"
    elif val < 75: return "#639922"
    return "#1D9E75"

def color_cell(val):
    color = score_color(val)
    return f"background-color: {color}18; color: {color}; font-weight: 500"


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 📡 Digital Desert Dashboard")
st.markdown(
    "**Minimum Connectivity Index (MCI)** — identifying digital deserts and "
    "their impact on women's safety and employment across India."
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────

n_total    = len(filtered_df)
n_deserts  = len(filtered_df[filtered_df["MCI_class"].isin(["Severe desert", "Moderate desert"])])
n_severe   = len(filtered_df[filtered_df["MCI_class"] == "Severe desert"])
avg_mci    = filtered_df["MCI"].mean() if n_total else 0
avg_wsi    = filtered_df["WSI"].mean() if n_total else 0
avg_wei    = filtered_df["WEI"].mean() if n_total else 0
high_risk  = len(filtered_df[filtered_df["Safety_risk"] == "High safety risk"]) if "Safety_risk" in filtered_df.columns else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Areas", n_total)
c2.metric("Digital deserts", n_deserts, delta=f"{n_severe} severe", delta_color="inverse")
c3.metric("Avg MCI", f"{avg_mci:.1f}")
c4.metric("Avg safety score (WSI)", f"{avg_wsi:.1f}", help="Lower = higher risk")
c5.metric("Avg employment score (WEI)", f"{avg_wei:.1f}", help="Lower = fewer opportunities")
c6.metric("High safety risk areas", high_risk, delta_color="inverse")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# ROW 1 — Classification distribution + Factor averages
# ─────────────────────────────────────────────────────────────────────────────

col_l, col_r = st.columns(2)

with col_l:
    st.markdown("#### Area classification breakdown")
    cls_counts = (
        filtered_df["MCI_class"]
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
        .reset_index()
    )
    cls_counts.columns = ["Classification", "Count"]

    fig_cls = px.bar(
        cls_counts, x="Count", y="Classification",
        orientation="h",
        color="Classification",
        color_discrete_map=CLASS_COLORS,
        text="Count",
    )
    fig_cls.update_traces(textposition="outside")
    fig_cls.update_layout(
        showlegend=False, height=260,
        margin=dict(l=0, r=20, t=10, b=10),
        xaxis_title="", yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(CLASS_ORDER))),
    )
    st.plotly_chart(fig_cls, use_container_width=True)

with col_r:
    st.markdown("#### Average factor scores")
    if n_total:
        avg_factors = {
            "Infrastructure (IFS)": round(filtered_df["IFS"].mean(), 1),
            "Digital literacy (DLS)": round(filtered_df["DLS"].mean(), 1),
            "Socio-economic (SES)": round(filtered_df["SES"].mean(), 1),
            "Women inclusion (WDI)": round(filtered_df["WDI"].mean(), 1),
        }
        fig_factor = go.Figure(go.Bar(
            x=list(avg_factors.values()),
            y=list(avg_factors.keys()),
            orientation="h",
            marker_color=list(FACTOR_COLORS.values()),
            text=[f"{v:.1f}" for v in avg_factors.values()],
            textposition="outside",
        ))
        fig_factor.add_vline(x=45, line_dash="dash", line_color="red",
                             annotation_text="desert threshold",
                             annotation_position="top right")
        fig_factor.update_layout(
            height=260, showlegend=False,
            xaxis=dict(range=[0, 110], title=""),
            yaxis_title="",
            margin=dict(l=0, r=20, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_factor, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROW 2 — MCI scatter + Women Impact scatter
# ─────────────────────────────────────────────────────────────────────────────

col_l2, col_r2 = st.columns(2)

with col_l2:
    st.markdown("#### MCI by area — sorted ascending")
    if n_total:
        plot_df = filtered_df.copy().sort_values("MCI")
        plot_df["label"] = plot_df["area"] + ", " + plot_df["city"]

        fig_mci = px.scatter(
            plot_df, x="label", y="MCI",
            color="MCI_class",
            color_discrete_map=CLASS_COLORS,
            hover_data={"IFS": True, "DLS": True, "SES": True, "WDI": True,
                        "MCI_class": True, "area_type": True},
            size_max=10,
        )
        fig_mci.add_hline(y=45, line_dash="dash", line_color="red",
                          annotation_text="desert threshold (45)")
        fig_mci.add_hline(y=25, line_dash="dot", line_color="#A32D2D",
                          annotation_text="severe (25)")
        fig_mci.update_layout(
            height=320, showlegend=False,
            xaxis=dict(tickangle=45, title=""),
            yaxis=dict(range=[0, 105], title="MCI score"),
            margin=dict(l=0, r=10, t=10, b=100),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_mci, use_container_width=True)

with col_r2:
    st.markdown("#### WSI vs WEI — women impact quadrant")
    if n_total and "WSI" in filtered_df.columns and "WEI" in filtered_df.columns:
        fig_quad = px.scatter(
            filtered_df,
            x="WSI", y="WEI",
            color="MCI_class",
            color_discrete_map=CLASS_COLORS,
            hover_name="area",
            hover_data={"city": True, "MCI": True, "area_type": True},
            size="MCI",
            size_max=18,
        )
        fig_quad.add_vline(x=35, line_dash="dash", line_color="red",
                           annotation_text="safety risk threshold")
        fig_quad.add_hline(y=40, line_dash="dash", line_color="#BA7517",
                           annotation_text="employment gap threshold")
        fig_quad.update_layout(
            height=320,
            xaxis=dict(range=[0, 105], title="Women Safety Index (WSI)"),
            yaxis=dict(range=[0, 105], title="Women Employment Index (WEI)"),
            margin=dict(l=0, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        # Add quadrant labels
        for txt, x, y in [
            ("High risk\nLow opportunity", 17, 20),
            ("Low risk\nHigh opportunity", 80, 80),
            ("High risk\nModerate opp.", 17, 70),
            ("Low risk\nLow opportunity", 70, 20),
        ]:
            fig_quad.add_annotation(
                x=x, y=y, text=txt, showarrow=False,
                font=dict(size=9, color="gray"), align="center",
            )
        st.plotly_chart(fig_quad, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROW 3 — Area table
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("#### Area-level MCI scores")

if n_total:
    display_cols = ["area", "city", "area_type", "MCI", "IFS", "DLS", "SES",
                    "WDI", "WSI", "WEI", "MCI_class", "Profile_label"]
    display_cols = [c for c in display_cols if c in filtered_df.columns]

    table_df = filtered_df[display_cols].copy()
    for col in ["MCI", "IFS", "DLS", "SES", "WDI", "WSI", "WEI"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].round(1)

    styled = table_df.style.applymap(
        color_cell,
        subset=[c for c in ["MCI", "IFS", "DLS", "SES", "WDI", "WSI", "WEI"]
                if c in table_df.columns]
    )
    st.dataframe(styled, use_container_width=True, height=320)
else:
    st.info("No areas match the current filters.")


# ─────────────────────────────────────────────────────────────────────────────
# ROW 4 — Area drill-down
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("#### Area drill-down")

area_options = filtered_df.apply(lambda r: f"{r['area']} — {r['city']}", axis=1).tolist()
if area_options:
    selected_area_str = st.selectbox("Select area to inspect", area_options)
    selected_area_name = selected_area_str.split(" — ")[0]
    area_row = filtered_df[filtered_df["area"] == selected_area_name].iloc[0]

    dcol1, dcol2, dcol3 = st.columns([1.2, 1.2, 1.6])

    with dcol1:
        st.markdown(f"**{area_row['area']}**, {area_row['city']}  \n"
                    f"`{area_row['area_type']}` · MCI = **{area_row['MCI']:.1f}**")

        for factor, col, label in [("IFS","#378ADD","Infrastructure"),
                                   ("DLS","#7F77DD","Digital literacy"),
                                   ("SES","#1D9E75","Socio-economic"),
                                   ("WDI","#D4537E","Women inclusion")]:
            val = area_row[factor]
            pct = int(val)
            st.markdown(
                f"<div style='margin-bottom:8px'>"
                f"<div style='font-size:12px;color:gray;margin-bottom:2px'>{label} ({factor}): <b style='color:{score_color(val)}'>{val:.1f}</b></div>"
                f"<div style='background:#f0f0f0;border-radius:4px;height:8px'>"
                f"<div style='background:{col};width:{pct}%;height:100%;border-radius:4px'></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown(f"**Classification:** {area_row['MCI_class']}")
        if "Profile_label" in area_row:
            st.markdown(f"**Profile:** {area_row['Profile_label']}")

    with dcol2:
        if "WSI" in area_row and "WEI" in area_row:
            wsi_val = area_row["WSI"]
            wei_val = area_row["WEI"]
            wsi_col = score_color(wsi_val)
            wei_col = score_color(wei_val)

            st.markdown(
                f"<div style='padding:12px;border-radius:8px;border:1px solid {wsi_col}30;"
                f"background:{wsi_col}08;margin-bottom:10px'>"
                f"<div style='font-size:11px;color:{wsi_col}'>Women Safety Index (WSI)</div>"
                f"<div style='font-size:28px;font-weight:600;color:{wsi_col}'>{wsi_val:.1f}</div>"
                f"<div style='font-size:12px;color:gray'>{area_row.get('Safety_risk','')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='padding:12px;border-radius:8px;border:1px solid {wei_col}30;"
                f"background:{wei_col}08;margin-bottom:10px'>"
                f"<div style='font-size:11px;color:{wei_col}'>Women Employment Index (WEI)</div>"
                f"<div style='font-size:28px;font-weight:600;color:{wei_col}'>{wei_val:.1f}</div>"
                f"<div style='font-size:12px;color:gray'>{area_row.get('Employment_gap','')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with dcol3:
        # Radar chart for the selected area
        area_sub = sub_df[
            (sub_df["area"] == selected_area_name) &
            (sub_df["city"] == area_row["city"])
        ]

        if not area_sub.empty:
            radar_labels = area_sub["subcomponent"].tolist()
            radar_values = area_sub["sub_score"].tolist()
            radar_values_closed = radar_values + [radar_values[0]]
            radar_labels_closed = radar_labels + [radar_labels[0]]

            fig_radar = go.Figure(go.Scatterpolar(
                r=radar_values_closed,
                theta=radar_labels_closed,
                fill="toself",
                fillcolor="rgba(55,138,221,0.15)",
                line=dict(color="#378ADD", width=1.5),
                name=selected_area_name,
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9)),
                    angularaxis=dict(tickfont=dict(size=9)),
                ),
                showlegend=False, height=300,
                margin=dict(l=40, r=40, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            # Fallback: factor-level radar
            factor_vals = [area_row["IFS"], area_row["DLS"],
                           area_row["SES"], area_row["WDI"]]
            labels = ["Infrastructure", "Digital literacy", "Socio-economic", "Women inclusion"]
            fig_radar = go.Figure(go.Scatterpolar(
                r=factor_vals + [factor_vals[0]],
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor="rgba(55,138,221,0.15)",
                line=dict(color="#378ADD", width=1.5),
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False, height=280,
                margin=dict(l=40, r=40, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROW 5 — Cluster profiles
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("#### Connectivity profile clusters")

if not cluster_df.empty:
    CLUSTER_COLORS_SEQ = ["#E24B4A", "#EF9F27", "#378ADD", "#7F77DD",
                          "#1D9E75", "#639922", "#D4537E"]

    # Cluster centroid bar chart
    centroid_long = []
    for _, row in cluster_df.iterrows():
        for factor in ["IFS", "DLS", "SES", "WDI"]:
            centroid_long.append({
                "Cluster": f"C{int(row['cluster_id'])}: {row['label']}",
                "Factor":  FACTOR_LABELS[factor],
                "Score":   row[f"{factor}_centroid"],
            })
    cent_df = pd.DataFrame(centroid_long)

    fig_cent = px.bar(
        cent_df, x="Factor", y="Score",
        color="Factor",
        color_discrete_map={v: list(FACTOR_COLORS.values())[i]
                            for i, v in enumerate(FACTOR_LABELS.values())},
        facet_col="Cluster",
        facet_col_wrap=min(4, len(cluster_df)),
        height=300,
    )
    fig_cent.add_hline(y=45, line_dash="dash", line_color="red", line_width=0.8)
    fig_cent.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 105]),
    )
    fig_cent.for_each_annotation(lambda a: a.update(
        text=a.text.split("=")[-1], font=dict(size=10)
    ))
    st.plotly_chart(fig_cent, use_container_width=True)

    # Cluster detail cards
    cols = st.columns(min(3, len(cluster_df)))
    for i, (_, cl) in enumerate(cluster_df.iterrows()):
        col_idx = i % len(cols)
        c_color = CLUSTER_COLORS_SEQ[i % len(CLUSTER_COLORS_SEQ)]
        with cols[col_idx]:
            st.markdown(
                f"<div style='border-left:3px solid {c_color};padding:.6rem .8rem;"
                f"background:{c_color}08;border-radius:0 8px 8px 0;margin-bottom:8px'>"
                f"<div style='font-size:13px;font-weight:600;color:{c_color}'>{cl['label']}</div>"
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
    "Variables ranked by their contribution to predicting MCI score. "
    "Higher = more impactful lever for policy intervention."
)

if not imp_df.empty:
    top_imp = imp_df.head(12).copy()
    top_imp["pct_label"] = top_imp["pct"].apply(lambda x: f"{x:.1f}%")

    fig_imp = px.bar(
        top_imp, x="importance", y="feature",
        orientation="h",
        text="pct_label",
        color="importance",
        color_continuous_scale=["#E6F1FB", "#185FA5"],
    )
    fig_imp.update_traces(textposition="outside")
    fig_imp.update_layout(
        height=max(300, len(top_imp) * 36),
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_title="Feature importance",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=60, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.info(
        "**Key finding:** Household internet access and women's digital skills "
        "are the top predictors of MCI — tower density ranks last. "
        "This suggests demand-side interventions (skilling, access programmes) "
        "are higher-leverage than supply-side infrastructure alone."
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROW 7 — Factor correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("#### Factor Spearman correlation matrix")
st.caption(
    "Correlations > 0.80 indicate factors share similar variance. "
    "DLS↔WDI and DLS↔SES being highly correlated is expected and documented "
    "in the methodology — they are conceptually distinct despite data overlap."
)

if n_total >= 5:
    corr_cols = [c for c in ["IFS", "DLS", "SES", "WDI", "MCI", "WSI", "WEI"]
                 if c in filtered_df.columns]
    corr_matrix = filtered_df[corr_cols].corr(method="spearman").round(3)

    fig_heatmap = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="Blues",
        zmin=-1, zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=11),
        showscale=True,
    ))
    fig_heatmap.update_layout(
        height=340,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "MCI methodology: weighted geometric mean of IFS (35%), DLS (30%), "
    "SES (20%), WDI (15%). Classification: Severe <25, Moderate 25–45, "
    "Partial 46–60, Near-connected 61–75, Connected >75. "
    "WSI/WEI: blended MCI + domain-specific factor scores."
)