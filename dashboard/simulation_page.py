"""
dashboard/simulation_page.py
=============================
Policy Simulation Page — the full Streamlit UI.

Eight components rendered on this page:
  1. District selector + Policy Objective selector
  2. Intervention panel (sliders/toggles by category)
  3. Before vs After impact view (KPI cards + factor bars)
  4. Social impact panel (6 human-centric outcome metrics)
  5. Root cause attribution panel (ranked factors with contribution bars)
  6. Policy recommendation engine (ranked interventions with confidence)
  7. AI Policy Copilot (explain / brief / Q&A)
  8. Advanced analyst mode (custom weights, intervention comparison)

All simulation is delegated to simulation/engine.py — no computation here.
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
from simulation.root_cause import compute_attribution
from simulation.recommendations import generate_recommendations
from simulation.copilot import SimulationCopilot

TRANSPARENT = "rgba(0,0,0,0)"

# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _load_districts(db_path: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute("""
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
        FROM mci_scores
        ORDER BY MCI ASC
    """).df()
    con.close()
    return df


@st.cache_data(ttl=300)
def _load_support_tables(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(db_path, read_only=True)
    try:
        sub_df = con.execute(
            "SELECT * FROM factor_subcomponent_scores"
        ).df()
    except Exception:
        sub_df = pd.DataFrame()
    try:
        imp_df = con.execute(
            "SELECT * FROM rf_feature_importance ORDER BY rank ASC"
        ).df()
    except Exception:
        imp_df = pd.DataFrame()
    con.close()
    return sub_df, imp_df


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _score_color(v: float) -> str:
    if v < 25:   return "#E24B4A"
    elif v < 45: return "#EF9F27"
    elif v < 60: return "#378ADD"
    elif v < 75: return "#639922"
    return "#1D9E75"


def _delta_color(d: float) -> str:
    return "#1D9E75" if d > 0 else ("#E24B4A" if d < 0 else "#999")


def _delta_arrow(d: float) -> str:
    return "▲" if d > 1 else ("▼" if d < -1 else "→")


def _confidence_color(c: float) -> str:
    if c >= 0.75: return "#1D9E75"
    elif c >= 0.55: return "#EF9F27"
    return "#E24B4A"


def _feasibility_badge(f: str) -> str:
    colors = {"high": "#1D9E75", "medium": "#EF9F27", "low": "#E24B4A"}
    c = colors.get(f, "#999")
    return (
        f"<span style='background:{c}18;color:{c};padding:2px 8px;"
        f"border-radius:12px;font-size:11px;font-weight:500'>{f.upper()}</span>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 1 — DISTRICT + OBJECTIVE SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

def _render_selector(districts_df: pd.DataFrame) -> tuple[dict | None, str]:
    st.markdown("### 📍 Select District & Policy Objective")

    c1, c2, c3 = st.columns([1.2, 1.2, 1.6])

    with c1:
        states = ["All"] + sorted(districts_df["statename"].dropna().unique().tolist())
        sel_state = st.selectbox("State", states, key="sim_state")

    with c2:
        pool = (districts_df if sel_state == "All"
                else districts_df[districts_df["statename"] == sel_state])
        district_options = ["— select district —"] + sorted(
            pool["districtname"].dropna().unique().tolist()
        )
        sel_district = st.selectbox("District", district_options, key="sim_district")

    with c3:
        obj_options = {o.label: o.key for o in OBJECTIVES}
        sel_obj_label = st.selectbox(
            "Policy objective",
            list(obj_options.keys()),
            key="sim_objective",
            help="Shapes which interventions are ranked highest and which outcomes are foregrounded.",
        )
        sel_obj_key = obj_options[sel_obj_label]

    if sel_district == "— select district —":
        return None, sel_obj_key

    row = pool[pool["districtname"] == sel_district]
    if row.empty:
        return None, sel_obj_key

    district_row = row.iloc[0].to_dict()

    # Quick profile strip
    mci = float(district_row.get("MCI", 0))
    cls = district_row.get("MCI_class", "")
    st.markdown(
        f"<div style='padding:.6rem 1rem;background:{_score_color(mci)}10;"
        f"border-left:3px solid {_score_color(mci)};border-radius:0 8px 8px 0;"
        f"margin-bottom:.5rem'>"
        f"<span style='font-size:13px;font-weight:500'>{sel_district}, {sel_state}</span> &nbsp;|&nbsp; "
        f"<span style='color:{_score_color(mci)};font-weight:600'>MCI {mci:.0f}</span> &nbsp;|&nbsp; "
        f"<span style='color:gray;font-size:12px'>{cls}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    return district_row, sel_obj_key


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 2 — INTERVENTION PANEL
# ─────────────────────────────────────────────────────────────────────────────

def _render_intervention_panel() -> dict[str, float]:
    st.markdown("### 🎛️ Policy Intervention Panel")
    st.caption(
        "Adjust intervention intensity (0 = not deployed, 100 = full saturation). "
        "Effects follow diminishing returns — 100% intensity ≠ 100% uplift."
    )

    interventions: dict[str, float] = {}
    categories = {}
    for i in INTERVENTIONS:
        categories.setdefault(i.category, []).append(i)

    cat_icons = {
        "Infrastructure": "📡",
        "Affordability":  "💰",
        "Digital Literacy": "💻",
        "Safety":         "🛡️",
        "Social":         "🤝",
    }

    cols = st.columns(2)
    col_idx = 0
    for cat, items in categories.items():
        with cols[col_idx % 2]:
            st.markdown(
                f"**{cat_icons.get(cat,'')} {cat}**",
            )
            for interv in items:
                val = st.slider(
                    label=interv.label,
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=10,
                    key=f"sim_interv_{interv.key}",
                    help=interv.description,
                    format="%d%%",
                )
                interventions[interv.key] = float(val)
        col_idx += 1

    active_count = sum(1 for v in interventions.values() if v > 0)
    if active_count == 0:
        st.info("↑ Select at least one intervention to run the simulation.")
    else:
        st.success(f"✓ {active_count} intervention(s) active — scroll down for results.")

    return interventions


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 3 — BEFORE VS AFTER IMPACT VIEW
# ─────────────────────────────────────────────────────────────────────────────

def _render_before_after(district_row: dict, result) -> None:
    st.markdown("### 📊 Before vs After Impact")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    kpi_keys = ["MCI", "IFS", "DLS", "SES", "WDI", "WSI", "WEI"]
    kpi_labels = {
        "MCI": "Connectivity Index", "IFS": "Infrastructure",
        "DLS": "Digital Literacy",   "SES": "Socio-Economic",
        "WDI": "Women Inclusion",    "WSI": "Women Safety",
        "WEI": "Women Employment",
    }
    cols = st.columns(len(kpi_keys))
    for col, key in zip(cols, kpi_keys):
        base  = result.baseline[key]
        sim   = result.simulated[key]
        delta = result.deltas[key]
        dc    = _delta_color(delta)
        arrow = _delta_arrow(delta)
        with col:
            st.markdown(
                f"<div style='text-align:center;padding:.5rem .3rem;"
                f"background:{_score_color(sim)}08;border-radius:8px;"
                f"border:0.5px solid {_score_color(sim)}30'>"
                f"<div style='font-size:11px;color:gray;margin-bottom:2px'>{kpi_labels[key]}</div>"
                f"<div style='font-size:20px;font-weight:600;color:{_score_color(sim)}'>{sim:.0f}</div>"
                f"<div style='font-size:11px;color:{dc}'>{arrow} {delta:+.1f} from {base:.0f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Horizontal grouped bar: baseline vs simulated ─────────────────────────
    factors = ["IFS", "DLS", "SES", "WDI"]
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
                  annotation_text="desert threshold",
                  annotation_position="top right")
    fig.add_vline(x=75, line_dash="dot", line_color="#1D9E75",
                  annotation_text="connected",
                  annotation_position="top right")
    fig.update_layout(
        barmode="group", height=220,
        xaxis=dict(range=[0, 105], title="Score (0–100)"),
        yaxis_title="",
        margin=dict(l=0, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.15, x=0),
        plot_bgcolor=TRANSPARENT, paper_bgcolor=TRANSPARENT,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Intervention contribution breakdown ────────────────────────────────────
    if result.intervention_contributions:
        st.markdown("**Intervention contribution to factor improvements:**")
        rows = []
        for ikey, fdict in result.intervention_contributions.items():
            label = next(
                (i.label for i in INTERVENTIONS if i.key == ikey), ikey
            )
            for factor, impact in fdict.items():
                rows.append({"Intervention": label, "Factor": factor, "Impact": impact})
        if rows:
            contrib_df = pd.DataFrame(rows)
            fig2 = px.bar(
                contrib_df, x="Intervention", y="Impact", color="Factor",
                color_discrete_map=FACTOR_COLORS,
                barmode="stack", height=180,
            )
            fig2.update_layout(
                margin=dict(l=0, r=0, t=10, b=60),
                xaxis=dict(tickangle=20),
                yaxis_title="Impact units",
                legend=dict(orientation="h", y=1.15),
                plot_bgcolor=TRANSPARENT, paper_bgcolor=TRANSPARENT,
            )
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 4 — SOCIAL IMPACT PANEL
# ─────────────────────────────────────────────────────────────────────────────

def _render_social_impact(result) -> None:
    st.markdown("### 👥 Social Impact Projections")
    st.caption("Human-centric outcome metrics derived from simulated scores.")

    so_base = result.social_outcomes_baseline
    so_sim  = result.social_outcomes

    cols = st.columns(3)
    outcome_icons = {
        "women_connectivity":     "📱",
        "women_employment_score": "💼",
        "service_access_score":   "🌐",
        "safe_digital_access":    "🛡️",
        "women_msme_access":      "🏭",
        "digital_access_score":   "📡",
    }

    for idx, (key, label) in enumerate(SOCIAL_OUTCOME_LABELS.items()):
        base  = so_base.get(key, 0)
        sim   = so_sim.get(key, 0)
        delta = sim - base
        dc    = _delta_color(delta)
        ic    = outcome_icons.get(key, "•")

        with cols[idx % 3]:
            # Score gauge bar
            pct = int(sim)
            st.markdown(
                f"<div style='padding:.7rem .9rem;background:#fafafa;"
                f"border-radius:8px;margin-bottom:8px;"
                f"border:0.5px solid {_score_color(sim)}30'>"
                f"<div style='font-size:11px;color:gray'>{ic} {label}</div>"
                f"<div style='font-size:22px;font-weight:600;"
                f"color:{_score_color(sim)}'>{sim:.0f}"
                f"<span style='font-size:12px;color:{dc};margin-left:6px'>"
                f"{_delta_arrow(delta)} {delta:+.1f}</span></div>"
                f"<div style='background:#e8e8e8;border-radius:3px;height:5px;margin-top:4px'>"
                f"<div style='background:{_score_color(sim)};width:{pct}%;"
                f"height:100%;border-radius:3px'></div></div>"
                f"<div style='font-size:10px;color:gray;margin-top:2px'>baseline {base:.0f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 5 — ROOT CAUSE ATTRIBUTION PANEL
# ─────────────────────────────────────────────────────────────────────────────

def _render_root_cause(district_row: dict, attribution: list) -> None:
    st.markdown("### 🔍 Root Cause Attribution")
    st.caption(
        "Ranked factors explaining why this district is underperforming. "
        "Contribution % = share of total MCI gap attributed to this variable."
    )

    if not attribution:
        st.success("District is performing adequately — no significant root causes identified.")
        return

    for item in attribution:
        score_c = _score_color(item.score)
        bar_pct = int(item.contribution_pct)
        factor_label = FACTOR_LABELS.get(item.factor, item.factor)
        st.markdown(
            f"<div style='padding:.6rem .9rem;margin-bottom:8px;"
            f"border-left:3px solid {score_c};"
            f"background:{score_c}08;border-radius:0 8px 8px 0'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<div>"
            f"<span style='font-size:13px;font-weight:500'>"
            f"#{item.rank} {item.variable}</span>"
            f"<span style='font-size:11px;color:gray;margin-left:8px'>"
            f"[{factor_label}]</span>"
            f"</div>"
            f"<span style='font-size:12px;font-weight:600;color:{score_c}'>"
            f"Score: {item.score:.0f}/100</span>"
            f"</div>"
            f"<div style='background:#e0e0e0;border-radius:3px;height:6px;"
            f"margin:.4rem 0'>"
            f"<div style='background:{score_c};width:{bar_pct}%;"
            f"height:100%;border-radius:3px'></div></div>"
            f"<div style='font-size:11px;color:gray'>"
            f"Contributes {item.contribution_pct:.0f}% of MCI gap &nbsp;·&nbsp; "
            f"Lever: {item.policy_lever}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 6 — POLICY RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _render_recommendations(
    recommendations: list,
    active_interventions: dict[str, float],
) -> None:
    st.markdown("### 💡 Policy Recommendations")
    st.caption(
        "Ranked by alignment with selected objective and expected impact. "
        "Confidence is evidence-based, not arbitrary."
    )

    for rec in recommendations:
        conf_c = _confidence_color(rec.confidence)
        factor_c = FACTOR_COLORS.get(rec.primary_factor, "#378ADD")
        is_active = rec.intervention_key in active_interventions and \
                    active_interventions[rec.intervention_key] > 0

        badge = (
            "<span style='background:#1D9E7520;color:#1D9E75;"
            "padding:1px 7px;border-radius:10px;font-size:10px'>✓ Active</span>"
            if is_active else ""
        )

        with st.expander(
            f"#{rec.rank}  {rec.intervention_label}  "
            f"| +{rec.expected_mci_delta:.0f} MCI pts  "
            f"| {rec.confidence:.0%} confidence",
            expanded=(rec.rank == 1),
        ):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(
                    f"{badge}\n\n{rec.rationale}",
                    unsafe_allow_html=True,
                )
                if rec.risk_factors:
                    st.markdown("**Risk factors:**")
                    for r in rec.risk_factors:
                        st.markdown(f"- ⚠️ {r}")
                if rec.reference:
                    st.caption(f"📋 Reference: {rec.reference}")

            with c2:
                # Delta metrics
                st.markdown(
                    f"<div style='text-align:center;padding:.5rem;"
                    f"border-radius:8px;background:#f5f5f5'>"
                    f"<div style='font-size:11px;color:gray'>Expected uplift @ 70%</div>"
                    f"<div style='font-size:18px;font-weight:600;"
                    f"color:{_delta_color(rec.expected_mci_delta)}'>"
                    f"MCI +{rec.expected_mci_delta:.1f}</div>"
                    f"<div style='font-size:12px;color:{_delta_color(rec.expected_wsi_delta)}'>"
                    f"WSI {rec.expected_wsi_delta:+.1f}</div>"
                    f"<div style='font-size:12px;color:{_delta_color(rec.expected_wei_delta)}'>"
                    f"WEI {rec.expected_wei_delta:+.1f}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"Feasibility: {_feasibility_badge(rec.feasibility)}  \n"
                    f"Confidence: <span style='color:{conf_c};font-weight:600'>"
                    f"{rec.confidence:.0%}</span>",
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 7 — AI POLICY COPILOT
# ─────────────────────────────────────────────────────────────────────────────

def _render_copilot(
    district_row: dict,
    result,
    attribution: list,
    objective_key: str,
) -> None:
    st.markdown("### 🤖 AI Policy Copilot")
    st.caption(
        "Ask questions about this simulation, get policy briefs, "
        "or explore intervention tradeoffs. Powered by LLaMA 3.1 (Ollama). "
        "Falls back to evidence-based templates when offline."
    )

    if "copilot" not in st.session_state:
        st.session_state.copilot = SimulationCopilot()
    if "copilot_history" not in st.session_state:
        st.session_state.copilot_history = []

    # Quick action buttons
    qa_cols = st.columns(3)
    if qa_cols[0].button("📄 Generate policy brief", key="copilot_brief"):
        with st.spinner("Generating policy brief..."):
            brief = st.session_state.copilot.generate_policy_brief(
                district_row, result, attribution, objective_key
            )
        st.session_state.copilot_history.append(
            {"role": "assistant", "content": brief}
        )
        st.rerun()

    if qa_cols[1].button("🔍 Explain simulation", key="copilot_explain"):
        with st.spinner("Analysing simulation..."):
            explanation = st.session_state.copilot.explain_simulation(
                district_row, result, attribution, objective_key
            )
        st.session_state.copilot_history.append(
            {"role": "assistant", "content": explanation}
        )
        st.rerun()

    if qa_cols[2].button("🔄 Clear chat", key="copilot_clear"):
        st.session_state.copilot_history = []
        st.rerun()

    # Chat history
    for msg in st.session_state.copilot_history:
        role  = msg["role"]
        icon  = "🤖" if role == "assistant" else "👤"
        align = "left"
        bg    = "#06488a" if role == "assistant" else "#067E2C"
        st.markdown(
            f"<div style='background:{bg};padding:.7rem 1rem;"
            f"border-radius:8px;margin-bottom:8px;text-align:{align}'>"
            f"<span style='font-size:12px;color:gray'>{icon}</span>&nbsp;"
            f"{msg['content']}</div>",
            unsafe_allow_html=True,
        )

    # Free-form question input
    question = st.chat_input(
        "Ask about this district, simulation, or intervention tradeoffs...",
        key="copilot_input",
    )
    if question:
        st.session_state.copilot_history.append(
            {"role": "user", "content": question}
        )
        with st.spinner("Thinking..."):
            answer = st.session_state.copilot.answer_question(
                question, district_row, result, attribution
            )
        st.session_state.copilot_history.append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 8 — ADVANCED ANALYST MODE
# ─────────────────────────────────────────────────────────────────────────────

def _render_analyst_mode(
    district_row: dict,
    objective_key: str,
    interventions: dict[str, float],
    attribution: list,
) -> dict | None:
    """
    Returns custom analyst weights if set, else None.
    Shown in a collapsed expander to keep it out of the primary policymaker view.
    """
    with st.expander("⚙️ Advanced Analyst Mode", expanded=False):
        st.caption(
            "Adjust MCI factor weights and compare multiple interventions. "
            "Changes here affect only this simulation session."
        )

        # ── Custom factor weights ─────────────────────────────────────────────
        st.markdown("**Custom MCI factor weights** (must sum to 1.0):")
        wc1, wc2, wc3, wc4 = st.columns(4)
        w_ifs = wc1.number_input("IFS weight", 0.05, 0.60, 0.35, 0.05, key="aw_ifs")
        w_dls = wc2.number_input("DLS weight", 0.05, 0.60, 0.30, 0.05, key="aw_dls")
        w_ses = wc3.number_input("SES weight", 0.05, 0.60, 0.20, 0.05, key="aw_ses")
        w_wdi = wc4.number_input("WDI weight", 0.05, 0.60, 0.15, 0.05, key="aw_wdi")

        total = w_ifs + w_dls + w_ses + w_wdi
        if abs(total - 1.0) > 0.01:
            st.warning(f"Weights sum to {total:.2f} — must equal 1.0. Normalising automatically.")
            norm = total
            w_ifs /= norm; w_dls /= norm; w_ses /= norm; w_wdi /= norm

        analyst_weights = {"IFS": w_ifs, "DLS": w_dls, "SES": w_ses, "WDI": w_wdi}

        # ── Assumption visibility ─────────────────────────────────────────────
        st.markdown("**Simulation assumptions:**")
        assume_cols = st.columns(2)
        assume_cols[0].markdown(
            "- Intervention effects follow **diminishing returns** (√ curve)\n"
            "- Maximum uplift values sourced from **comparable Indian programmes**\n"
            "- All deltas are on the normalised 0–100 scale\n"
        )
        assume_cols[1].markdown(
            "- Simultaneous interventions **do not compound** (conservative)\n"
            "- District baseline from **latest available year** in mci_scores\n"
            "- Factor formulas mirror **mci_pipeline.py** exactly\n"
        )

        # ── Side-by-side intervention comparison ─────────────────────────────
        st.markdown("**Compare two intervention packages:**")
        pkg_cols = st.columns(2)
        packages: list[dict[str, float]] = [{}, {}]

        for pi, pkg_col in enumerate(pkg_cols):
            with pkg_col:
                st.markdown(f"**Package {pi+1}**")
                for interv in INTERVENTIONS:
                    val = st.slider(
                        interv.label,
                        0, 100, 0, 10,
                        key=f"pkg_{pi}_{interv.key}",
                        format="%d%%",
                    )
                    packages[pi][interv.key] = float(val)

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
                metrics = ["MCI", "IFS", "DLS", "SES", "WDI", "WSI", "WEI"]
                comp_data = {
                    "Metric": metrics,
                    "Baseline": [results[0][1].baseline[m] for m in metrics],
                    results[0][0]: [results[0][1].simulated[m] for m in metrics],
                    results[1][0]: [results[1][1].simulated[m] for m in metrics],
                }
                comp_df = pd.DataFrame(comp_data).round(1)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
            else:
                st.info("Set at least one intervention in each package to compare.")

        return analyst_weights


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def render_simulation_page() -> None:
    st.markdown("# 🔬 Policy Simulation")
    st.markdown(
        "Simulate realistic connectivity interventions and observe projected effects "
        "on MCI, women's safety, employment, and digital inclusion. "
        "All projections are grounded in comparable Indian programme outcomes."
    )
    st.markdown("---")

    # Load data
    try:
        districts_df         = _load_districts(DB_PATH)
        sub_df, imp_df       = _load_support_tables(DB_PATH)
    except Exception as e:
        st.error(f"Could not load district data: {e}. Run the pipeline first.")
        return

    if districts_df.empty:
        st.warning("No district data found. Run the MCI pipeline and store dashboard tables first.")
        return

    # ── Component 1: District + Objective selector ────────────────────────────
    district_row, objective_key = _render_selector(districts_df)

    if district_row is None:
        st.info("👆 Select a state and district to begin the simulation.")
        return

    st.markdown("---")

    # ── Two-column layout: interventions left, analyst mode right ─────────────
    left_col, right_col = st.columns([3, 1])

    with right_col:
        # Component 8: Analyst mode (collapsed by default)
        analyst_weights = _render_analyst_mode(
            district_row, objective_key, {}, []
        )

    with left_col:
        # Component 2: Intervention panel
        interventions = _render_intervention_panel()

    # ── Run simulation if any interventions active ────────────────────────────
    active = {k: v for k, v in interventions.items() if v > 0}

    if not active:
        # Still show root cause even without active interventions
        st.markdown("---")
        attribution = compute_attribution(district_row, sub_df, imp_df)
        _render_root_cause(district_row, attribution)

        st.markdown("---")
        recommendations = generate_recommendations(
            district_row, objective_key, attribution, top_n=4
        )
        _render_recommendations(recommendations, active)
        return

    # Run simulation
    sim_input = SimulationInput(
        district_row=district_row,
        interventions=interventions,
        objective_key=objective_key,
        analyst_weights=analyst_weights,
    )

    with st.spinner("Running simulation..."):
        result = simulate(sim_input)

    # Compute attribution
    attribution = compute_attribution(district_row, sub_df, imp_df)

    # ── Render results ────────────────────────────────────────────────────────
    st.markdown("---")
    _render_before_after(district_row, result)

    st.markdown("---")
    _render_social_impact(result)

    st.markdown("---")
    _render_root_cause(district_row, attribution)

    st.markdown("---")
    recommendations = generate_recommendations(
        district_row, objective_key, attribution, top_n=4
    )
    _render_recommendations(recommendations, active)

    st.markdown("---")
    _render_copilot(district_row, result, attribution, objective_key)

    # ── Confidence strip ──────────────────────────────────────────────────────
    st.markdown("---")
    conf_c = _confidence_color(result.confidence)
    notes_text = "  ·  ".join(result.confidence_notes)
    st.markdown(
        f"<div style='padding:.5rem 1rem;background:{conf_c}10;"
        f"border-radius:8px;border:0.5px solid {conf_c}40'>"
        f"<span style='font-size:12px;color:{conf_c};font-weight:600'>"
        f"Projection confidence: {result.confidence:.0%}</span>"
        f"<span style='font-size:12px;color:gray;margin-left:12px'>{notes_text}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )