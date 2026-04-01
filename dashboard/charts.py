"""
dashboard/charts.py
====================
All Plotly chart builder functions.
Each function takes a dataframe + optional params and returns a go.Figure.
"""

from __future__ import annotations
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from config import CLASS_ORDER, CLASS_COLORS, FACTOR_COLORS, FACTOR_LABELS


TRANSPARENT = "rgba(0,0,0,0)"


def classification_bar(df: pd.DataFrame) -> go.Figure:
    counts = (
        df["MCI_class"].value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
        .reset_index()
    )
    counts.columns = ["Classification", "Count"]
    fig = px.bar(
        counts, x="Count", y="Classification", orientation="h",
        color="Classification", color_discrete_map=CLASS_COLORS, text="Count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False, height=260,
        margin=dict(l=0, r=20, t=10, b=10),
        xaxis_title="", yaxis_title="",
        plot_bgcolor=TRANSPARENT, paper_bgcolor=TRANSPARENT,
        yaxis=dict(categoryorder="array",
                   categoryarray=list(reversed(CLASS_ORDER))),
    )
    return fig


def factor_avg_bar(df: pd.DataFrame) -> go.Figure:
    avgs = {FACTOR_LABELS[k]: round(df[k].mean(), 1) for k in FACTOR_LABELS if k in df.columns}
    fig = go.Figure(go.Bar(
        x=list(avgs.values()), y=list(avgs.keys()),
        orientation="h",
        marker_color=list(FACTOR_COLORS.values()),
        text=[f"{v:.1f}" for v in avgs.values()],
        textposition="outside",
    ))
    fig.add_vline(x=45, line_dash="dash", line_color="red",
                  annotation_text="desert threshold", annotation_position="top right")
    fig.update_layout(
        height=260, showlegend=False,
        xaxis=dict(range=[0, 110], title=""),
        yaxis_title="",
        margin=dict(l=0, r=20, t=10, b=10),
        plot_bgcolor=TRANSPARENT, paper_bgcolor=TRANSPARENT,
    )
    return fig


def mci_scatter(df: pd.DataFrame) -> go.Figure:
    plot_df = df.copy().sort_values("MCI")
    plot_df["label"] = plot_df["area"] + ", " + plot_df["city"]
    fig = px.scatter(
        plot_df, x="label", y="MCI",
        color="MCI_class", color_discrete_map=CLASS_COLORS,
        hover_data={"IFS": True, "DLS": True, "SES": True, "WDI": True,
                    "MCI_class": True, "area_type": True},
    )
    fig.add_hline(y=45, line_dash="dash", line_color="red",
                  annotation_text="desert threshold (45)")
    fig.add_hline(y=25, line_dash="dot", line_color="#A32D2D",
                  annotation_text="severe (25)")
    fig.update_layout(
        height=320, showlegend=False,
        xaxis=dict(tickangle=45, title=""),
        yaxis=dict(range=[0, 105], title="MCI score"),
        margin=dict(l=0, r=10, t=10, b=100),
        plot_bgcolor=TRANSPARENT, paper_bgcolor=TRANSPARENT,
    )
    return fig


def wsi_wei_quadrant(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="WSI", y="WEI",
        color="MCI_class", color_discrete_map=CLASS_COLORS,
        hover_name="area",
        hover_data={"city": True, "MCI": True, "area_type": True},
        size="MCI", size_max=18,
    )
    fig.add_vline(x=35, line_dash="dash", line_color="red",
                  annotation_text="safety risk threshold")
    fig.add_hline(y=40, line_dash="dash", line_color="#BA7517",
                  annotation_text="employment gap threshold")
    for txt, x, y in [
        ("High risk / Low opportunity",   17, 20),
        ("Low risk / High opportunity",   80, 80),
        ("High risk / Moderate opp.",     17, 70),
        ("Low risk / Low opportunity",    70, 20),
    ]:
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False,
                           font=dict(size=9, color="gray"), align="center")
    fig.update_layout(
        height=320,
        xaxis=dict(range=[0, 105], title="Women Safety Index (WSI)"),
        yaxis=dict(range=[0, 105], title="Women Employment Index (WEI)"),
        margin=dict(l=0, r=10, t=10, b=10),
        plot_bgcolor=TRANSPARENT, paper_bgcolor=TRANSPARENT,
    )
    return fig


def mci_trend_line(trend_df: pd.DataFrame, area_label: str) -> go.Figure:
    """Line chart of MCI (and optionally factor scores) over years."""
    fig = go.Figure()
    cols = [c for c in ["MCI", "IFS", "DLS", "SES", "WDI"] if c in trend_df.columns]
    colors_map = {"MCI": "#2C2C2A", **FACTOR_COLORS}
    widths_map  = {"MCI": 3, "IFS": 1.5, "DLS": 1.5, "SES": 1.5, "WDI": 1.5}
    for col in cols:
        fig.add_trace(go.Scatter(
            x=trend_df["year"], y=trend_df[col].round(1),
            mode="lines+markers", name=col,
            line=dict(color=colors_map.get(col,"gray"),
                      width=widths_map.get(col, 1.5)),
        ))
    fig.add_hline(y=45, line_dash="dash", line_color="red",
                  annotation_text="desert threshold")
    fig.update_layout(
        title=f"MCI trend — {area_label}",
        height=300,
        xaxis=dict(title="Year", tickmode="linear"),
        yaxis=dict(range=[0, 105], title="Score"),
        margin=dict(l=0, r=10, t=40, b=10),
        plot_bgcolor=TRANSPARENT, paper_bgcolor=TRANSPARENT,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def area_radar(area_row: dict, sub_df: pd.DataFrame) -> go.Figure:
    """Radar of sub-component scores for the selected area."""
    area_sub = sub_df[
        (sub_df["area"] == area_row.get("area")) &
        (sub_df["city"] == area_row.get("city"))
    ]

    if not area_sub.empty:
        labels = area_sub["subcomponent"].tolist()
        values = area_sub["sub_score"].tolist()
    else:
        # Fall back to factor scores
        labels = list(FACTOR_LABELS.values()) + ["WSI", "WEI"]
        values = [area_row.get(k, 50) for k in list(FACTOR_LABELS.keys()) + ["WSI", "WEI"]]

    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(55,138,221,0.15)",
        line=dict(color="#378ADD", width=1.5),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=9)),
        ),
        showlegend=False, height=300,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor=TRANSPARENT,
    )
    return fig


def cluster_centroid_chart(cluster_df: pd.DataFrame) -> go.Figure:
    """Faceted bar chart showing each cluster's factor centroids."""
    rows = []
    for _, row in cluster_df.iterrows():
        for factor in ["IFS", "DLS", "SES", "WDI"]:
            rows.append({
                "Cluster": f"C{int(row['cluster_id'])}: {row['label']}",
                "Factor":  FACTOR_LABELS[factor],
                "Score":   row[f"{factor}_centroid"],
            })
    cent_df = pd.DataFrame(rows)
    fig = px.bar(
        cent_df, x="Factor", y="Score",
        color="Factor",
        color_discrete_map={v: list(FACTOR_COLORS.values())[i]
                            for i, v in enumerate(FACTOR_LABELS.values())},
        facet_col="Cluster",
        facet_col_wrap=min(4, len(cluster_df)),
        height=300,
    )
    fig.add_hline(y=45, line_dash="dash", line_color="red", line_width=0.8)
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor=TRANSPARENT, paper_bgcolor=TRANSPARENT,
        yaxis=dict(range=[0, 105]),
    )
    fig.for_each_annotation(lambda a: a.update(
        text=a.text.split("=")[-1], font=dict(size=10)
    ))
    return fig


def rf_importance_bar(imp_df: pd.DataFrame) -> go.Figure:
    top = imp_df.head(12).copy()
    top["label"] = top["pct"].apply(lambda x: f"{x:.1f}%")
    fig = px.bar(
        top, x="importance", y="feature", orientation="h",
        text="label",
        color="importance",
        color_continuous_scale=["#E6F1FB", "#185FA5"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=max(300, len(top) * 36),
        showlegend=False, coloraxis_showscale=False,
        xaxis_title="Feature importance", yaxis_title="",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=60, t=10, b=10),
        plot_bgcolor=TRANSPARENT, paper_bgcolor=TRANSPARENT,
    )
    return fig


def spearman_heatmap(df: pd.DataFrame) -> go.Figure:
    cols = [c for c in ["IFS", "DLS", "SES", "WDI", "MCI", "WSI", "WEI"]
            if c in df.columns]
    corr = df[cols].corr(method="spearman").round(3)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="Blues", zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}",
        textfont=dict(size=11), showscale=True,
    ))
    fig.update_layout(
        height=340, margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor=TRANSPARENT,
    )
    return fig


def dynamic_chart(df: pd.DataFrame, chart_type: str) -> go.Figure:
    """
    Render agent-suggested chart from query results.
    chart_type: 'bar', 'line', 'scatter', 'table' (returns None for table)
    """
    if chart_type == "bar" and not df.empty:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        if num_cols and cat_cols:
            y_col = num_cols[0]
            x_col = cat_cols[0]
            fig = px.bar(df, x=x_col, y=y_col, text=y_col)
            fig.update_traces(textposition="outside",
                              marker_color="#378ADD")
            fig.update_layout(
                height=300, xaxis=dict(tickangle=45),
                plot_bgcolor=TRANSPARENT, paper_bgcolor=TRANSPARENT,
                margin=dict(l=0, r=0, t=10, b=80),
            )
            return fig

    elif chart_type == "line" and not df.empty:
        if "year" in df.columns:
            num_cols = [c for c in df.select_dtypes(include="number").columns
                        if c != "year"]
            if num_cols:
                fig = px.line(df, x="year", y=num_cols, markers=True)
                fig.update_layout(
                    height=300,
                    plot_bgcolor=TRANSPARENT, paper_bgcolor=TRANSPARENT,
                    margin=dict(l=0, r=0, t=10, b=10),
                )
                return fig

    elif chart_type == "scatter" and not df.empty:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) >= 2:
            fig = px.scatter(df, x=num_cols[0], y=num_cols[1],
                             hover_name=df.columns[0])
            fig.update_layout(
                height=300,
                plot_bgcolor=TRANSPARENT, paper_bgcolor=TRANSPARENT,
            )
            return fig

    return None