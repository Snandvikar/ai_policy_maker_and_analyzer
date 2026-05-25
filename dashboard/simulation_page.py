"""
dashboard/simulation_page.py
=============================
Policy Optimization & Dual-Mode Simulation Page.

TWO WORKFLOWS:
  Manual Exploration  — free-form slider control, no optimization
  AI-Assisted Optimization — budget-aware greedy optimizer populates
                              sliders as recommended defaults;
                              user retains full manual override.

COMPONENTS:
  1.  District + Objective selector
  2.  Workflow mode toggle (Manual / Optimization)
  3.  Optimization Constraints panel (budget, strategy) — Optimization mode only
  4.  Live budget utilization bar
  5.  Intervention panel — lever cards with cost/speed/feasibility metadata
  6.  Before vs After impact view
  7.  Social impact panel
  8.  Root cause attribution
  9.  Policy recommendations — enhanced with tradeoff table + budget + why
  10. AI Policy Copilot
  11. Advanced Analyst Mode (custom weights, package comparison)

"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import duckdb

from config import DB_PATH, FACTOR_LABELS, FACTOR_COLORS, CLASS_COLORS
from simulation.engine import (
    INTERVENTIONS, OBJECTIVES, OBJECTIVE_MAP,
    SimulationInput, simulate, SOCIAL_OUTCOME_LABELS,
)
from simulation.cost_model import (
    COST_PROFILES, compute_intervention_cost_cr,
    compute_portfolio_cost_cr, cost_efficiency_label,
)
from simulation.optimizer import optimize, OptimizationResult
from simulation.root_cause import compute_attribution
from simulation.recommendations import generate_recommendations
from simulation.tradeoff import TRADEOFF_PROFILES, TIER_COLORS, get_tradeoff_table_data
from simulation.copilot import SimulationCopilot

TRANSPARENT = "rgba(0,0,0,0)"


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _load_districts(db_path: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    df  = con.execute("""
        SELECT canonical_state, statename, districtname, districtcode,
               IFS, DLS, SES, WDI, MCI, WSI, WEI,
               MCI_class, Safety_risk, Employment_gap,
               mobile_only_pct, no_phone_pct, any_phone_pct,
               illiteracy_percent, gender_vulnerability_index,
               crime_against_women_rate, msme_female_share_pct,
               share_of_slum_population, dependent_women_percent,
               wireless_rural_teledensity_pct,
               e_transactions_per_1000_population,
               maternal_mortality_rate, destitute_pct,
               non_agri_enterprise_pct, income_lt5k_pct
        FROM mci_scores ORDER BY MCI ASC
    """).df()
    con.close()
    return df


@st.cache_data(ttl=300)
def _load_support(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(db_path, read_only=True)
    try:
        sub = con.execute("SELECT * FROM factor_subcomponent_scores").df()
    except Exception:
        sub = pd.DataFrame()
    try:
        imp = con.execute(
            "SELECT * FROM rf_feature_importance ORDER BY rank ASC"
        ).df()
    except Exception:
        imp = pd.DataFrame()
    con.close()
    return sub, imp


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR / BADGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _sc(v: float) -> str:
    if v < 25:   return "#E24B4A"
    elif v < 45: return "#EF9F27"
    elif v < 60: return "#378ADD"
    elif v < 75: return "#639922"
    return "#1D9E75"


def _dc(d: float) -> str:
    return "#1D9E75" if d > 0 else ("#E24B4A" if d < 0 else "#888")


def _arrow(d: float) -> str:
    return "▲" if d > 1 else ("▼" if d < -1 else "→")


def _conf_c(c: float) -> str:
    return "#1D9E75" if c >= 0.75 else ("#EF9F27" if c >= 0.55 else "#E24B4A")


def _badge(label: str, color: str) -> str:
    return (f"<span style='background:{color}18;color:{color};"
            f"padding:2px 8px;border-radius:12px;"
            f"font-size:11px;font-weight:500'>{label}</span>")


def _tier_badge(tier: str) -> str:
    c = TIER_COLORS.get(tier, "#888")
    return _badge(tier, c)


def _speed_color(s: str) -> str:
    return {"Fast": "#1D9E75", "Medium": "#EF9F27", "Slow": "#E24B4A"}.get(s, "#888")


def _feas_color(f: str) -> str:
    return {"high": "#1D9E75", "medium": "#EF9F27", "low": "#E24B4A"}.get(f, "#888")


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 1 — DISTRICT + OBJECTIVE SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

def _render_selector(df: pd.DataFrame) -> tuple[dict | None, str]:
    st.markdown("### 📍 District & Objective")
    c1, c2, c3 = st.columns([1.2, 1.2, 1.6])

    with c1:
        states = ["All"] + sorted(df["statename"].dropna().unique().tolist())
        sel_state = st.selectbox("State", states, key="sim_state")

    with c2:
        pool = df if sel_state == "All" else df[df["statename"] == sel_state]
        opts = ["— select district —"] + sorted(
            pool["districtname"].dropna().unique().tolist()
        )
        sel_district = st.selectbox("District", opts, key="sim_district")

    with c3:
        obj_map = {o.label: o.key for o in OBJECTIVES}
        sel_obj = st.selectbox(
            "Policy objective", list(obj_map.keys()), key="sim_objective",
            help="Shapes intervention ranking and which outcomes are foregrounded.",
        )
        sel_obj_key = obj_map[sel_obj]

    if sel_district == "— select district —":
        return None, sel_obj_key

    row = pool[pool["districtname"] == sel_district]
    if row.empty:
        return None, sel_obj_key

    dr = row.iloc[0].to_dict()
    mci = float(dr.get("MCI", 0))
    cls = dr.get("MCI_class", "")
    st.markdown(
        f"<div style='padding:.5rem 1rem;background:{_sc(mci)}10;"
        f"border-left:3px solid {_sc(mci)};border-radius:0 8px 8px 0;"
        f"margin-bottom:.3rem'>"
        f"<b>{sel_district}, {sel_state}</b> &nbsp;|&nbsp; "
        f"<span style='color:{_sc(mci)};font-weight:600'>MCI {mci:.0f}</span>"
        f" &nbsp;|&nbsp; <span style='color:gray;font-size:12px'>{cls}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    return dr, sel_obj_key


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 2 — WORKFLOW MODE TOGGLE
# ─────────────────────────────────────────────────────────────────────────────

def _render_mode_toggle() -> str:
    """Returns 'manual' or 'optimization'."""
    st.markdown("### 🔀 Simulation Mode")
    mode = st.radio(
        "Choose workflow",
        options=["Manual Exploration", "AI-Assisted Optimization"],
        index=0,
        key="sim_mode",
        horizontal=True,
        help=(
            "Manual: freely adjust all sliders and explore scenarios.  \n"
            "Optimization: system recommends intervention allocations; "
            "you retain full manual override."
        ),
    )
    if mode == "Manual Exploration":
        st.caption(
            "🔓 **Manual mode** — adjust any lever freely. "
            "No budget enforcement unless you enable it below."
        )
        return "manual"
    else:
        st.caption(
            "🤖 **Optimization mode** — set constraints and click "
            "**Generate Optimal Policy** to auto-populate sliders. "
            "You can still override any value after optimization."
        )
        return "optimization"


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 3 — OPTIMIZATION CONSTRAINTS PANEL
# ─────────────────────────────────────────────────────────────────────────────

def _render_constraints_panel(objective_key: str) -> tuple[bool, float | None, str]:
    """
    Returns (budget_enabled, budget_cr, strategy_key).
    budget_cr is None when budget is disabled.
    """
    with st.container():
        st.markdown("#### ⚙️ Optimization Constraints")

        cc1, cc2, cc3, cc4 = st.columns([1, 1.4, 1.4, 1.2])

        with cc1:
            budget_on = st.toggle(
                "Enable budget constraint",
                value=False,
                key="budget_toggle",
                help="When ON, optimizer will not exceed the specified total budget.",
            )

        with cc2:
            budget_cr = None
            if budget_on:
                budget_cr = st.number_input(
                    "Total budget (₹ Crore)",
                    min_value=1.0,
                    max_value=5000.0,
                    value=100.0,
                    step=10.0,
                    key="budget_input",
                    format="%.0f",
                )
            else:
                st.markdown(
                    "<div style='padding:.5rem;background:#f0f0f0;"
                    "border-radius:6px;font-size:12px;color:gray;"
                    "margin-top:1.6rem'>Unlimited budget mode</div>",
                    unsafe_allow_html=True,
                )

        with cc3:
            strategy_labels = {
                "Cost Efficient":    "cost_efficient",
                "Balanced":          "balanced",
                "Fast Deployment":   "fast_deployment",
                "High Feasibility":  "high_feasibility",
            }
            sel_strategy = st.selectbox(
                "Optimization strategy",
                list(strategy_labels.keys()),
                index=1,
                key="strategy_select",
                help=(
                    "Cost Efficient: maximise MCI per ₹ Cr.  \n"
                    "Balanced: equal weight on impact, cost, speed.  \n"
                    "Fast Deployment: prioritise interventions deployable < 6 months.  \n"
                    "High Feasibility: minimise implementation risk."
                ),
            )
            strategy_key = strategy_labels[sel_strategy]

        with cc4:
            st.markdown("<div style='margin-top:1.6rem'></div>",
                        unsafe_allow_html=True)
            generate = st.button(
                "🚀 Generate Optimal Policy",
                key="generate_btn",
                use_container_width=True,
                type="primary",
            )

        return budget_on, budget_cr, strategy_key, generate


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 4 — LIVE BUDGET UTILIZATION BAR
# ─────────────────────────────────────────────────────────────────────────────

def _render_budget_bar(
    interventions: dict[str, float],
    budget_on: bool,
    budget_cr: float | None,
) -> None:
    total_cost = compute_portfolio_cost_cr(interventions)
    active_n   = sum(1 for v in interventions.values() if v > 0)

    if not budget_on or budget_cr is None:
        st.markdown(
            f"<div style='padding:.4rem .8rem;background:#f5f5f5;"
            f"border-radius:6px;font-size:12px;color:#555;display:flex;"
            f"align-items:center;gap:12px;margin-bottom:.5rem'>"
            f"<span>🔓 Unlimited exploratory mode</span>"
            f"<span style='color:#378ADD'>Estimated total: ₹{total_cost:.1f} Cr</span>"
            f"<span style='color:gray'>{active_n} intervention(s) active</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    pct  = min(100.0, total_cost / budget_cr * 100)
    bar_c = "#1D9E75" if pct < 70 else ("#EF9F27" if pct < 95 else "#E24B4A")
    status = "✓ Within budget" if pct < 95 else "⚠ Budget limit reached"

    st.markdown(
        f"<div style='padding:.5rem .9rem;background:#fafafa;"
        f"border:0.5px solid {bar_c}40;border-radius:8px;margin-bottom:.5rem'>"
        f"<div style='display:flex;justify-content:space-between;"
        f"align-items:center;margin-bottom:4px'>"
        f"<span style='font-size:12px;font-weight:500'>💰 Budget Utilization</span>"
        f"<span style='font-size:12px;color:{bar_c};font-weight:600'>"
        f"₹{total_cost:.1f} Cr / ₹{budget_cr:.0f} Cr &nbsp;·&nbsp; {status}</span>"
        f"</div>"
        f"<div style='background:#e0e0e0;border-radius:4px;height:8px'>"
        f"<div style='background:{bar_c};width:{pct:.0f}%;"
        f"height:100%;border-radius:4px;transition:width .3s'></div></div>"
        f"<div style='font-size:11px;color:gray;margin-top:3px'>"
        f"{pct:.0f}% used &nbsp;·&nbsp; "
        f"₹{max(0, budget_cr - total_cost):.1f} Cr remaining &nbsp;·&nbsp; "
        f"{active_n} intervention(s) active"
        f"</div></div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 5 — INTERVENTION PANEL
# Enhanced lever cards showing cost, speed, feasibility per intervention.
# Budget enforcement: slider max is capped when budget would be exceeded.
# ─────────────────────────────────────────────────────────────────────────────

def _render_intervention_panel(
    budget_on: bool,
    budget_cr: float | None,
    preset_values: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Renders enhanced lever cards with cost/speed/feasibility metadata.
    preset_values: filled by optimizer; used as slider defaults.
    Returns {intervention_key: intensity}.
    """
    st.markdown("### 🎛️ Policy Intervention Levers")
    st.caption(
        "Each lever shows estimated cost, deployment speed, and feasibility "
        "at the current intensity. Budget enforcement applies when enabled."
    )

    interventions: dict[str, float] = {}
    categories: dict[str, list] = {}
    for i in INTERVENTIONS:
        categories.setdefault(i.category, []).append(i)

    cat_icons = {
        "Infrastructure": "📡", "Affordability": "💰",
        "Digital Literacy": "💻", "Safety": "🛡️", "Social": "🤝",
    }

    # Compute cost already committed to OTHER interventions for budget cap
    # We'll compute per-slider below
    cols = st.columns(2)
    col_idx = 0

    for cat, items in categories.items():
        with cols[col_idx % 2]:
            st.markdown(
                f"<div style='font-size:13px;font-weight:600;margin-bottom:4px'>"
                f"{cat_icons.get(cat,'')} {cat}</div>",
                unsafe_allow_html=True,
            )
            for interv in items:
                cp      = COST_PROFILES.get(interv.key)
                default = int(preset_values.get(interv.key, 0)) if preset_values else 0

                # Compute max allowed intensity given remaining budget
                if budget_on and budget_cr and cp and cp.cost_per_point_cr > 0:
                    other_cost = sum(
                        compute_intervention_cost_cr(k, v)
                        for k, v in (preset_values or {}).items()
                        if k != interv.key and v > 0
                    )
                    remaining  = max(0, budget_cr - other_cost)
                    max_allow  = min(100, int(remaining / cp.cost_per_point_cr))
                    # Round down to nearest 10
                    max_allow  = (max_allow // 10) * 10
                    # max_allow=int(budget_cr)
                else:
                    max_allow = 100
                
                slider_max = max(max_allow, default, 10)

                val = st.slider(
                label=interv.label,
                min_value=0,
                max_value=slider_max,
                value=min(default, slider_max),
                step=5,
                key=f"sim_interv_{interv.key}",
                help=interv.description,
                format="%d%%",
                )
                interventions[interv.key] = float(val)

                # Lever metadata strip
                if cp:
                    cost_at_val = compute_intervention_cost_cr(interv.key, val)
                    sp_c  = _speed_color(cp.deployment_speed)
                    fe_c  = _feas_color(interv.feasibility)
                    at_limit = (
                        budget_on and budget_cr and max_allow < 100
                        and val >= max_allow
                    )
                    limit_warn = (
                        "<span style='color:#E24B4A;font-size:10px'>"
                        " ⛔ budget limit</span>"
                        if at_limit else ""
                    )
                    st.markdown(
                        f"<div style='font-size:11px;color:#666;"
                        f"margin-top:-8px;margin-bottom:8px;padding-left:2px'>"
                        f"₹{cost_at_val:.1f} Cr &nbsp;·&nbsp; "
                        f"<span style='color:{sp_c}'>{cp.deployment_speed}</span>"
                        f" deployment &nbsp;·&nbsp; "
                        f"<span style='color:{fe_c}'>"
                        f"{interv.feasibility.capitalize()}</span> feasibility"
                        f"{limit_warn}</div>",
                        unsafe_allow_html=True,
                    )
            st.markdown("")
        col_idx += 1

    return interventions


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 6 — BEFORE VS AFTER
# ─────────────────────────────────────────────────────────────────────────────

def _render_before_after(result) -> None:
    st.markdown("### 📊 Before vs After Impact")

    kpi_keys   = ["MCI","IFS","DLS","SES","WDI","WSI","WEI"]
    kpi_labels = {
        "MCI":"Connectivity","IFS":"Infrastructure","DLS":"Digital Literacy",
        "SES":"Socio-Economic","WDI":"Women Inclusion",
        "WSI":"Women Safety","WEI":"Women Employment",
    }
    cols = st.columns(len(kpi_keys))
    for col, k in zip(cols, kpi_keys):
        base  = result.baseline[k]
        sim   = result.simulated[k]
        delta = result.deltas[k]
        with col:
            st.markdown(
                f"<div style='text-align:center;padding:.4rem .2rem;"
                f"background:{_sc(sim)}08;border-radius:8px;"
                f"border:0.5px solid {_sc(sim)}30'>"
                f"<div style='font-size:10px;color:gray'>{kpi_labels[k]}</div>"
                f"<div style='font-size:20px;font-weight:700;"
                f"color:{_sc(sim)}'>{sim:.0f}</div>"
                f"<div style='font-size:11px;color:{_dc(delta)}'>"
                f"{_arrow(delta)} {delta:+.1f}</div>"
                f"<div style='font-size:10px;color:#aaa'>was {base:.0f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")
    factors = ["IFS","DLS","SES","WDI"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Baseline",
        y=[FACTOR_LABELS[f] for f in factors],
        x=[result.baseline[f] for f in factors],
        orientation="h",
        marker_color="#C8D8E8",
        text=[f"{result.baseline[f]:.0f}" for f in factors],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="Simulated",
        y=[FACTOR_LABELS[f] for f in factors],
        x=[result.simulated[f] for f in factors],
        orientation="h",
        marker_color=[FACTOR_COLORS[f] for f in factors],
        text=[f"{result.simulated[f]:.0f}" for f in factors],
        textposition="inside",
    ))
    fig.add_vline(x=45, line_dash="dash", line_color="red",
                  annotation_text="desert threshold")
    fig.add_vline(x=75, line_dash="dot", line_color="#1D9E75",
                  annotation_text="connected")
    fig.update_layout(
        barmode="group", height=210,
        xaxis=dict(range=[0,105], title=""),
        yaxis_title="",
        margin=dict(l=0,r=10,t=20,b=10),
        legend=dict(orientation="h", y=1.2, x=0),
        plot_bgcolor=TRANSPARENT, paper_bgcolor=TRANSPARENT,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 7 — SOCIAL IMPACT PANEL
# ─────────────────────────────────────────────────────────────────────────────

def _render_social_impact(result) -> None:
    st.markdown("### 👥 Social Impact Projections")
    so_base = result.social_outcomes_baseline
    so_sim  = result.social_outcomes
    icons   = {
        "women_connectivity":"📱", "women_employment_score":"💼",
        "service_access_score":"🌐", "safe_digital_access":"🛡️",
        "women_msme_access":"🏭",   "digital_access_score":"📡",
    }
    cols = st.columns(3)
    for idx, (k, label) in enumerate(SOCIAL_OUTCOME_LABELS.items()):
        base  = so_base.get(k, 0)
        sim   = so_sim.get(k,  0)
        delta = sim - base
        ic    = icons.get(k, "•")
        with cols[idx % 3]:
            st.markdown(
                f"<div style='padding:.6rem .8rem;background:#fafafa;"
                f"border-radius:8px;margin-bottom:8px;"
                f"border:0.5px solid {_sc(sim)}30'>"
                f"<div style='font-size:11px;color:gray'>{ic} {label}</div>"
                f"<div style='font-size:21px;font-weight:600;"
                f"color:{_sc(sim)}'>{sim:.0f}"
                f"<span style='font-size:12px;color:{_dc(delta)};margin-left:6px'>"
                f"{_arrow(delta)} {delta:+.1f}</span></div>"
                f"<div style='background:#e8e8e8;border-radius:3px;"
                f"height:5px;margin-top:4px'>"
                f"<div style='background:{_sc(sim)};width:{int(sim)}%;"
                f"height:100%;border-radius:3px'></div></div>"
                f"<div style='font-size:10px;color:#aaa;margin-top:2px'>"
                f"baseline {base:.0f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 8 — ROOT CAUSE ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def _render_root_cause(attribution: list) -> None:
    st.markdown("### 🔍 Root Cause Attribution")
    st.caption("Ranked factors driving digital exclusion. Contribution % = share of total MCI gap.")

    if not attribution:
        st.success("District is performing adequately — no significant root causes identified.")
        return

    for item in attribution:
        sc = _sc(item.score)
        st.markdown(
            f"<div style='padding:.5rem .9rem;margin-bottom:7px;"
            f"border-left:3px solid {sc};background:{sc}08;"
            f"border-radius:0 8px 8px 0'>"
            f"<div style='display:flex;justify-content:space-between'>"
            f"<span style='font-size:13px;font-weight:500'>#{item.rank} {item.variable}"
            f"<span style='font-size:11px;color:gray;margin-left:6px'>"
            f"[{FACTOR_LABELS.get(item.factor, item.factor)}]</span></span>"
            f"<span style='font-size:12px;font-weight:600;color:{sc}'>"
            f"Score {item.score:.0f}/100</span></div>"
            f"<div style='background:#ddd;border-radius:3px;height:5px;margin:.3rem 0'>"
            f"<div style='background:{sc};width:{int(item.contribution_pct)}%;"
            f"height:100%;border-radius:3px'></div></div>"
            f"<div style='font-size:11px;color:gray'>"
            f"{item.contribution_pct:.0f}% of MCI gap &nbsp;·&nbsp; "
            f"Lever: {item.policy_lever}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 9 — POLICY RECOMMENDATIONS (enhanced)
# ─────────────────────────────────────────────────────────────────────────────

def _render_tradeoff_table(active_keys: list[str] | None = None) -> None:
    """Renders the full tradeoff intelligence table."""
    rows = get_tradeoff_table_data(active_keys)
    if not rows:
        return

    st.markdown("**Policy tradeoff comparison:**")
    display_cols = [
        "Intervention", "Cost", "Inclusion Gain",
        "QoS Gain", "Deployment Speed", "Feasibility",
    ]
    df = pd.DataFrame(rows)[display_cols]

    def _color_tier(val):
        c = TIER_COLORS.get(str(val), "#888")
        return f"background-color:{c}15;color:{c};font-weight:500"

    tier_cols = ["Cost","Inclusion Gain","QoS Gain","Deployment Speed","Feasibility"]
    styled = df.style.applymap(_color_tier, subset=tier_cols)
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_recommendations(
    recommendations: list,
    active_interventions: dict[str, float],
    show_tradeoff: bool = True,
) -> None:
    st.markdown("### 💡 Policy Recommendations")
    st.caption(
        "Ranked by alignment with selected objective and expected impact. "
        "Budget and cost efficiency shown at 70% intensity."
    )

    # Tradeoff table at top (all interventions for context)
    if show_tradeoff:
        with st.expander("📊 Full Policy Tradeoff Intelligence Table", expanded=False):
            _render_tradeoff_table()

    eff_colors = {"HIGH": "#1D9E75", "MEDIUM": "#EF9F27", "LOW": "#E24B4A", "N/A": "#888"}

    for rec in recommendations:
        conf_c = _conf_c(rec.confidence)
        fc     = FACTOR_COLORS.get(rec.primary_factor, "#378ADD")
        is_active = (
            rec.intervention_key in active_interventions and
            active_interventions[rec.intervention_key] > 0
        )
        active_badge = (
            "<span style='background:#1D9E7520;color:#1D9E75;"
            "padding:1px 7px;border-radius:10px;font-size:10px;"
            "margin-left:6px'>✓ Active</span>" if is_active else ""
        )
        eff_c = eff_colors.get(rec.cost_efficiency, "#888")

        with st.expander(
            f"#{rec.rank}  {rec.intervention_label}"
            f"  |  +{rec.expected_mci_delta:.0f} MCI pts"
            f"  |  ₹{rec.estimated_budget_cr:.0f} Cr"
            f"  |  {rec.confidence:.0%} confidence",
            expanded=(rec.rank == 1),
        ):
            main_c, meta_c = st.columns([2, 1])

            with main_c:
                st.markdown(
                    f"{rec.rationale} {active_badge}",
                    unsafe_allow_html=True,
                )

                # Why selected bullets
                if rec.why_selected:
                    st.markdown("**Why this was recommended:**")
                    for w in rec.why_selected:
                        st.markdown(f"- {w}")

                # Risk factors
                if rec.risk_factors:
                    st.markdown("**Risk factors:**")
                    for r in rec.risk_factors:
                        st.markdown(f"- ⚠️ {r}")

                if rec.reference:
                    st.caption(f"📋 Reference: {rec.reference}")

                if rec.tradeoff_summary:
                    st.caption(f"ℹ️ {rec.tradeoff_summary}")

            with meta_c:
                st.markdown(
                    f"<div style='background:#0F172A;padding:.6rem;"
                    f"border-radius:8px;text-align:center'>"
                    f"<div style='font-size:10px;color:gray'>Expected uplift @ 70%</div>"
                    f"<div style='font-size:18px;font-weight:700;"
                    f"color:{_dc(rec.expected_mci_delta)}'>"
                    f"MCI +{rec.expected_mci_delta:.1f}</div>"
                    f"<div style='font-size:12px;color:{_dc(rec.expected_wsi_delta)}'>"
                    f"WSI {rec.expected_wsi_delta:+.1f}</div>"
                    f"<div style='font-size:12px;color:{_dc(rec.expected_wei_delta)}'>"
                    f"WEI {rec.expected_wei_delta:+.1f}</div>"
                    f"<hr style='margin:.4rem 0;border-color:#ddd'>"
                    f"<div style='font-size:11px'>Budget: "
                    f"<b>₹{rec.estimated_budget_cr:.0f} Cr</b></div>"
                    f"<div style='font-size:11px'>Impact/₹: "
                    f"<b style='color:{eff_c}'>{rec.cost_efficiency}</b></div>"
                    f"<div style='font-size:11px'>Feasibility: "
                    f"<b style='color:{_feas_color(rec.feasibility)}'>"
                    f"{rec.feasibility.upper()}</b></div>"
                    f"<div style='font-size:11px'>Timeline: "
                    f"<b>{rec.timeline_label}</b></div>"
                    f"<div style='font-size:11px'>Confidence: "
                    f"<b style='color:{conf_c}'>{rec.confidence:.0%}</b></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Tradeoff table for active interventions only
    active_keys = [k for k, v in active_interventions.items() if v > 0]
    if active_keys and show_tradeoff:
        st.markdown("**Tradeoff profile — your active interventions:**")
        _render_tradeoff_table(active_keys)


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 10 — AI POLICY COPILOT
# ─────────────────────────────────────────────────────────────────────────────

def _render_copilot(
    district_row: dict, result, attribution: list, objective_key: str
) -> None:
    st.markdown("### 🤖 AI Policy Copilot")
    st.caption(
        "Ask questions, generate policy briefs, or explore tradeoffs. "
        "Powered by LLaMA 3.1 (Ollama). Falls back to structured templates offline."
    )

    if "copilot" not in st.session_state:
        st.session_state.copilot = SimulationCopilot()
    if "copilot_history" not in st.session_state:
        st.session_state.copilot_history = []

    qa_cols = st.columns(3)
    if qa_cols[0].button("📄 Generate policy brief", key="copilot_brief"):
        with st.spinner("Generating..."):
            resp = st.session_state.copilot.generate_policy_brief(
                district_row, result, attribution, objective_key
            )
        st.session_state.copilot_history.append({"role": "assistant", "content": resp})
        st.rerun()

    if qa_cols[1].button("🔍 Explain simulation", key="copilot_explain"):
        with st.spinner("Analysing..."):
            resp = st.session_state.copilot.explain_simulation(
                district_row, result, attribution, objective_key
            )
        st.session_state.copilot_history.append({"role": "assistant", "content": resp})
        st.rerun()

    if qa_cols[2].button("🔄 Clear", key="copilot_clear"):
        st.session_state.copilot_history = []
        st.rerun()

    for msg in st.session_state.copilot_history:
        bg = "#034590" if msg["role"] == "assistant" else "#02561A"
        ic = "🤖" if msg["role"] == "assistant" else "👤"
        st.markdown(
            f"<div style='background:{bg};padding:.6rem .9rem;"
            f"border-radius:8px;margin-bottom:6px'>"
            f"{ic} {msg['content']}</div>",
            unsafe_allow_html=True,
        )

    q = st.chat_input("Ask about interventions, tradeoffs, or confidence...",
                      key="copilot_q")
    if q:
        st.session_state.copilot_history.append({"role": "user", "content": q})
        with st.spinner("Thinking..."):
            ans = st.session_state.copilot.answer_question(
                q, district_row, result, attribution
            )
        st.session_state.copilot_history.append({"role": "assistant", "content": ans})
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 11 — ADVANCED ANALYST MODE
# ─────────────────────────────────────────────────────────────────────────────

def _render_analyst_mode(
    district_row: dict,
    objective_key: str,
) -> dict | None:
    with st.expander("⚙️ Advanced Analyst Mode", expanded=False):
        st.caption(
            "Custom MCI factor weights, assumption visibility, "
            "and side-by-side package comparison."
        )

        wc = st.columns(4)
        w_ifs = wc[0].number_input("IFS", 0.05, 0.60, 0.35, 0.05, key="aw_ifs")
        w_dls = wc[1].number_input("DLS", 0.05, 0.60, 0.30, 0.05, key="aw_dls")
        w_ses = wc[2].number_input("SES", 0.05, 0.60, 0.20, 0.05, key="aw_ses")
        w_wdi = wc[3].number_input("WDI", 0.05, 0.60, 0.15, 0.05, key="aw_wdi")

        total = w_ifs + w_dls + w_ses + w_wdi
        if abs(total - 1.0) > 0.01:
            st.warning(f"Weights sum to {total:.2f} — normalising.")
            w_ifs /= total; w_dls /= total; w_ses /= total; w_wdi /= total

        analyst_weights = {"IFS": w_ifs, "DLS": w_dls, "SES": w_ses, "WDI": w_wdi}

        st.markdown("**Simulation assumptions:**")
        ac = st.columns(2)
        ac[0].markdown(
            "- Diminishing returns curve: `√(intensity/100)`\n"
            "- Max uplift values from comparable Indian programmes\n"
            "- Simultaneous interventions do not compound (conservative)\n"
        )
        ac[1].markdown(
            "- Cost: ₹ Cr/point, linear with intensity\n"
            "- Baseline: latest year in mci_scores\n"
            "- Factor formulas mirror mci_pipeline.py exactly\n"
        )

        st.markdown("**Compare two intervention packages:**")
        pc = st.columns(2)
        packages: list[dict] = [{}, {}]
        for pi in range(2):
            with pc[pi]:
                st.markdown(f"**Package {pi+1}**")
                for interv in INTERVENTIONS:
                    v = st.slider(
                        interv.label, 0, 100, 0, 10,
                        key=f"pkg_{pi}_{interv.key}", format="%d%%"
                    )
                    packages[pi][interv.key] = float(v)

        if st.button("Compare packages", key="compare_pkgs"):
            results = []
            for pi, pkg in enumerate(packages):
                if any(v > 0 for v in pkg.values()):
                    inp = SimulationInput(
                        district_row=district_row,
                        interventions=pkg,
                        objective_key=objective_key,
                        analyst_weights=analyst_weights,
                    )
                    results.append((f"Package {pi+1}", simulate(inp)))
            if len(results) == 2:
                metrics = ["MCI","IFS","DLS","SES","WDI","WSI","WEI"]
                comp    = {
                    "Metric":    metrics,
                    "Baseline":  [results[0][1].baseline[m] for m in metrics],
                    results[0][0]: [results[0][1].simulated[m] for m in metrics],
                    results[1][0]: [results[1][1].simulated[m] for m in metrics],
                }
                st.dataframe(
                    pd.DataFrame(comp).round(1),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("Set at least one intervention in each package.")

        return analyst_weights


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def render_simulation_page() -> None:
    st.markdown("# 🔬 Policy Optimization & Simulation")
    st.markdown(
        "Simulate connectivity interventions and evaluate projected impact on MCI, "
        "women's safety, employment, and digital inclusion.  \n"
        "**Manual mode** for free exploration · "
        "**Optimization mode** for budget-constrained policy planning."
    )
    st.markdown("---")

    try:
        districts_df   = _load_districts(DB_PATH)
        sub_df, imp_df = _load_support(DB_PATH)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return

    if districts_df.empty:
        st.warning("No district data. Run the MCI pipeline first.")
        return

    # ── 1. District + Objective ───────────────────────────────────────────────
    district_row, objective_key = _render_selector(districts_df)
    if district_row is None:
        st.info("👆 Select a state and district above.")
        return

    st.markdown("---")

    # ── 2. Mode toggle ────────────────────────────────────────────────────────
    mode = _render_mode_toggle()

    st.markdown("---")

    # ── 3. Constraints panel (both modes — optional in manual) ────────────────
    budget_on, budget_cr, strategy_key, generate_clicked = _render_constraints_panel(
        objective_key
    )

    # ── Analyst mode (right side) ─────────────────────────────────────────────
    analyst_weights = _render_analyst_mode(district_row, objective_key)

    # ── Optimization: generate preset values ─────────────────────────────────
    preset_values: dict[str, float] | None = None

    if mode == "optimization" and generate_clicked:
        with st.spinner("Generating optimal policy allocation..."):
            opt_result: OptimizationResult = optimize(
                district_row=district_row,
                objective_key=objective_key,
                strategy=strategy_key,
                budget_cr=budget_cr if budget_on else None,
                analyst_weights=analyst_weights,
            )
        preset_values = opt_result.allocations

        # Store in session so sliders pick it up on rerun
        st.session_state["sim_preset"] = preset_values
        st.session_state["sim_opt_result"] = opt_result
        st.rerun()

    # Load stored preset if available (persists across reruns)
    if "sim_preset" in st.session_state:
        preset_values = st.session_state["sim_preset"]

    # Show optimization rationale if available
    if mode == "optimization" and "sim_opt_result" in st.session_state:
        opt_r: OptimizationResult = st.session_state["sim_opt_result"]
        with st.expander(
            f"📋 Optimization rationale "
            f"(strategy: {opt_r.strategy_used}, "
            f"budget: {'₹'+str(budget_cr)+'Cr' if opt_r.budget_constrained else 'unlimited'})",
            expanded=False,
        ):
            for line in opt_r.allocation_rationale:
                st.markdown(f"- {line}")
            st.caption(
                "ℹ️ Optimized values are **recommended defaults**. "
                "You retain full manual override — adjust any slider above."
            )

    # ── 4. Budget utilization bar ─────────────────────────────────────────────
    # We need current slider values — but sliders haven't rendered yet.
    # Use preset as proxy for the bar; will update after render.
    # preview_interventions = preset_values or {i.key: 0.0 for i in INTERVENTIONS}
    # _render_budget_bar(preview_interventions, budget_on, budget_cr)

    # ── 5. Intervention panel ─────────────────────────────────────────────────
    interventions = _render_intervention_panel(
        budget_on=budget_on,
        budget_cr=budget_cr,
        preset_values=preset_values,
    )

    # Update budget bar with actual slider values
    _render_budget_bar(interventions, budget_on, budget_cr)

    # ── Run simulation ────────────────────────────────────────────────────────
    active = {k: v for k, v in interventions.items() if v > 0}
    attribution = compute_attribution(district_row, sub_df, imp_df)

    # Always show root cause (even without active interventions)
    st.markdown("---")
    _render_root_cause(attribution)

    recommendations = generate_recommendations(
        district_row, objective_key, attribution, top_n=4
    )

    if not active:
        st.markdown("---")
        _render_recommendations(recommendations, active)
        st.info("↑ Activate at least one intervention lever to see simulation results.")
        return

    sim_input = SimulationInput(
        district_row=district_row,
        interventions=interventions,
        objective_key=objective_key,
        analyst_weights=analyst_weights,
    )
    with st.spinner("Running simulation..."):
        result = simulate(sim_input)

    # ── 6–9. Results panels ───────────────────────────────────────────────────
    st.markdown("---")
    _render_before_after(result)

    st.markdown("---")
    _render_social_impact(result)

    st.markdown("---")
    _render_recommendations(recommendations, active, show_tradeoff=True)

    # ── 10. Copilot ───────────────────────────────────────────────────────────
    st.markdown("---")
    _render_copilot(district_row, result, attribution, objective_key)

    # ── Confidence footer ─────────────────────────────────────────────────────
    st.markdown("---")
    cc = _conf_c(result.confidence)
    notes = "  ·  ".join(result.confidence_notes)
    st.markdown(
        f"<div style='padding:.4rem .9rem;background:{cc}10;"
        f"border-radius:8px;border:0.5px solid {cc}30'>"
        f"<span style='color:{cc};font-weight:600;font-size:12px'>"
        f"Projection confidence: {result.confidence:.0%}</span>"
        f"<span style='font-size:12px;color:gray;margin-left:10px'>{notes}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )