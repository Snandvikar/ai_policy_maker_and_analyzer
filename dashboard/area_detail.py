"""
dashboard/area_detail.py
=========================
District drill-down panel.

Schema alignment:
  districtname    — the geographic unit (replaces old 'area' and 'city')
  statename       — human-readable state name
  canonical_state — normalised state key (for sub_df and timeseries joins)
  No Profile_label, city, area, city_tier, or area_type in mci_scores.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd

from config import FACTOR_LABELS, FACTOR_COLORS
from policy_engine.rules import get_suggestions
from dashboard.charts import area_radar, mci_trend_line


def _score_color(val: float) -> str:
    if val < 25:   return "#E24B4A"
    elif val < 45: return "#EF9F27"
    elif val < 60: return "#378ADD"
    elif val < 75: return "#639922"
    return "#1D9E75"


def _factor_bar(label: str, val: float, color: str):
    pct = int(max(0, min(100, val)))
    st.markdown(
        f"<div style='margin-bottom:8px'>"
        f"<div style='font-size:12px;color:gray;margin-bottom:2px'>"
        f"{label}: <b style='color:{color}'>{val:.1f}</b></div>"
        f"<div style='background:#e8e8e8;border-radius:4px;height:7px'>"
        f"<div style='background:{color};width:{pct}%;height:100%;"
        f"border-radius:4px'></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def render_area_detail(
    area_row: dict,
    sub_df: pd.DataFrame,
    timeseries_df: pd.DataFrame,
) -> None:
    """
    Renders the full drill-down panel for a selected district.

    Parameters:
        area_row:      dict of score values (one row from mci_scores)
        sub_df:        factor_subcomponent_scores (canonical_state + districtname + year)
        timeseries_df: mci_timeseries_scores (all years)
    """
    district = area_row.get("districtname", "")
    state    = area_row.get("statename",    "")
    canon    = area_row.get("canonical_state", "")
    mci      = area_row.get("MCI", 0)
    cls      = area_row.get("MCI_class", "")

    st.markdown(f"### {district}")
    st.caption(f"{state}  ·  MCI = **{mci:.1f}**  ·  {cls}")

    col1, col2, col3 = st.columns([1.1, 1.1, 1.8])

    # ── Col 1: Factor score bars ──────────────────────────────────────────────
    with col1:
        st.markdown("**Factor scores**")
        for key, label in FACTOR_LABELS.items():
            val = float(area_row.get(key, 0) or 0)
            _factor_bar(label, val, _score_color(val))

        cluster = area_row.get("cluster_label", "—")
        st.markdown(f"**Cluster:** {cluster}")

    # ── Col 2: WSI / WEI impact cards ────────────────────────────────────────
    with col2:
        st.markdown("**Women impact**")
        for score_key, label, risk_key in [
            ("WSI", "Women Safety Index",     "Safety_risk"),
            ("WEI", "Women Employment Index", "Employment_gap"),
        ]:
            val  = float(area_row.get(score_key, 0) or 0)
            risk = area_row.get(risk_key, "")
            c    = _score_color(val)
            st.markdown(
                f"<div style='padding:10px 12px;border-radius:8px;"
                f"border:1px solid {c}30;background:{c}08;margin-bottom:10px'>"
                f"<div style='font-size:11px;color:{c}'>{label} ({score_key})</div>"
                f"<div style='font-size:26px;font-weight:600;color:{c}'>{val:.1f}</div>"
                f"<div style='font-size:11px;color:gray'>{risk}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Col 3: Radar chart ────────────────────────────────────────────────────
    with col3:
        st.markdown("**Sub-component radar**")
        fig_radar = area_radar(area_row, sub_df)
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Historical trend chart ────────────────────────────────────────────────
    # Join on canonical_state + districtname (not city + area)
    trend_mask = pd.Series([True] * len(timeseries_df), index=timeseries_df.index)
    if district and "districtname" in timeseries_df.columns:
        trend_mask &= timeseries_df["districtname"] == district
    if canon and "canonical_state" in timeseries_df.columns:
        trend_mask &= timeseries_df["canonical_state"] == canon

    trend = timeseries_df[trend_mask]
    if len(trend) > 1:
        st.markdown("**Historical trend**")
        fig_trend = mci_trend_line(trend, f"{district}, {state}")
        st.plotly_chart(fig_trend, use_container_width=True)

    # ── Key indicators summary ────────────────────────────────────────────────
    st.markdown("**Key indicators**")
    indicator_map = {
        "No phone households":        ("no_phone_pct",                    "%"),
        "Mobile-only households":     ("mobile_only_pct",                 "%"),
        "Illiteracy rate":            ("illiteracy_percent",              "%"),
        "Gender vulnerability index": ("gender_vulnerability_index",     ""),
        "Crime vs women (per lakh)":  ("crime_against_women_rate",       ""),
        "Women-owned MSMEs (share)":  ("msme_female_share_pct",         "%"),
        "E-transactions / 1k pop":    ("e_transactions_per_1000_population", ""),
        "Slum population share":      ("share_of_slum_population",       "%"),
        "Wireless rural teledensity": ("wireless_rural_teledensity_pct", "%"),
    }
    ind_cols = st.columns(3)
    items = [
        (label, area_row.get(col, None), unit)
        for label, (col, unit) in indicator_map.items()
        if col in area_row and area_row.get(col) is not None
    ]
    for idx, (label, val, unit) in enumerate(items):
        with ind_cols[idx % 3]:
            try:
                st.metric(label, f"{float(val):.1f}{unit}")
            except (TypeError, ValueError):
                st.metric(label, "—")

    # ── Rule-based policy suggestions ────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Recommended interventions")
    st.caption("Generated by the rule-based policy engine from factor scores")

    suggestions = get_suggestions(area_row, n=3)
    icons    = {1: "🔴", 2: "🟡", 3: "🟢"}
    cat_icons = {
        "IFS": "📡", "DLS": "💻", "SES": "🏛️",
        "WDI": "👩", "WSI": "🛡️", "WEI": "💼",
    }

    if suggestions:
        for s in suggestions:
            with st.expander(
                f"{icons.get(s.priority,'⚪')} {s.short_title} "
                f"{cat_icons.get(s.category,'')}",
                expanded=(s.priority == 1),
            ):
                st.markdown(s.action)
                st.caption(f"Rationale: {s.rationale}")
                if s.scheme_or_program:
                    st.markdown(f"📋 **Scheme / Programme:** {s.scheme_or_program}")
    else:
        st.success("No critical interventions triggered — area is performing adequately.")