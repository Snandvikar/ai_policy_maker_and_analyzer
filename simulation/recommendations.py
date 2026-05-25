"""
simulation/recommendations.py
================================
Policy Recommendation Engine — extended with budget, cost efficiency,
and tradeoff intelligence.

New fields on PolicyRecommendation:
  estimated_budget_cr — total ₹ Crore needed at recommended intensity
  cost_efficiency     — HIGH / MEDIUM / LOW (MCI pts per ₹ Cr)
  why_selected        — list of plain-language reasons for this recommendation
  tradeoff_summary    — one-line tradeoff context
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from simulation.engine import (
    INTERVENTIONS, INTERVENTION_MAP, OBJECTIVE_MAP,
    SimulationInput, simulate, _clamp,
)
from simulation.root_cause import AttributionItem
from simulation.cost_model import (
    COST_PROFILES, compute_intervention_cost_cr, cost_efficiency_label
)
from simulation.tradeoff import TRADEOFF_MAP


@dataclass
class PolicyRecommendation:
    rank:                  int
    intervention_key:      str
    intervention_label:    str
    rationale:             str
    expected_mci_delta:    float
    expected_wsi_delta:    float
    expected_wei_delta:    float
    primary_factor:        str
    feasibility:           str
    confidence:            float
    risk_factors:          list[str]
    reference:             str
    objective_alignment:   float
    # ── New fields ────────────────────────────────────────────────────────────
    estimated_budget_cr:   float = 0.0         # ₹ Cr at recommended 70% intensity
    cost_efficiency:       str   = "MEDIUM"     # HIGH / MEDIUM / LOW
    why_selected:          list[str] = field(default_factory=list)
    tradeoff_summary:      str   = ""
    deployment_speed:      str   = "Medium"
    timeline_label:        str   = ""
    cost_tier:             str   = "Medium"


def generate_recommendations(
    district_row: dict,
    objective_key: str,
    attribution_items: list[AttributionItem],
    top_n: int = 4,
) -> list[PolicyRecommendation]:
    objective    = OBJECTIVE_MAP.get(objective_key, OBJECTIVE_MAP["general_connectivity"])
    mci_baseline = float(district_row.get("MCI", 50))

    # Build root-cause factor gaps from attribution
    factor_gap: dict[str, float] = {"IFS": 0, "DLS": 0, "SES": 0, "WDI": 0}
    for item in attribution_items:
        if item.factor in factor_gap:
            factor_gap[item.factor] += item.gap * item.contribution_pct / 100

    # Identify top contributing variable per factor for rationale
    top_var_per_factor: dict[str, str] = {}
    for item in attribution_items:
        if item.factor not in top_var_per_factor:
            top_var_per_factor[item.factor] = item.variable

    scored: list[dict] = []
    for interv in INTERVENTIONS:
        sim_input = SimulationInput(
            district_row=district_row,
            interventions={interv.key: 70.0},
            objective_key=objective_key,
        )
        result = simulate(sim_input)
        mci_delta = result.deltas.get("MCI", 0)
        wsi_delta = result.deltas.get("WSI", 0)
        wei_delta = result.deltas.get("WEI", 0)

        # ── Scoring ───────────────────────────────────────────────────────────
        obj_score = sum(
            objective.intervention_weights.get(f, 0.25) *
            factor_gap.get(f, 0) *
            (abs(d) / (max(factor_gap.get(f, 1), 1)))
            for f, d in result.deltas.items()
            if f in ("IFS", "DLS", "SES", "WDI")
        )
        feasibility_bonus = {"high": 0.15, "medium": 0.05, "low": -0.05}.get(
            interv.feasibility, 0
        )
        primary_gap = factor_gap.get(interv.primary_factor, 0)
        if primary_gap < 5:
            obj_score *= 0.6
        final_score = obj_score + feasibility_bonus + mci_delta * 0.01

        # ── Budget and cost efficiency ────────────────────────────────────────
        est_budget = compute_intervention_cost_cr(interv.key, 70.0)
        eff_label  = cost_efficiency_label(mci_delta, est_budget)

        # ── Why selected (rule-based, plain language) ─────────────────────────
        why = _build_why_selected(
            interv, district_row, attribution_items,
            factor_gap, top_var_per_factor, objective_key,
            mci_delta, eff_label
        )

        # ── Tradeoff summary ──────────────────────────────────────────────────
        tp = TRADEOFF_MAP.get(interv.key)
        tradeoff_str = tp.impact_summary if tp else ""

        # ── Cost / speed metadata ─────────────────────────────────────────────
        cp = COST_PROFILES.get(interv.key)
        deployment_speed = cp.deployment_speed if cp else "Medium"
        timeline_label   = cp.timeline_label   if cp else ""
        cost_tier        = cp.cost_tier        if cp else "Medium"

        scored.append({
            "interv":          interv,
            "score":           final_score,
            "mci_delta":       mci_delta,
            "wsi_delta":       wsi_delta,
            "wei_delta":       wei_delta,
            "risks":           _assess_risks(interv, district_row, mci_delta),
            "rationale":       _build_rationale(
                interv, attribution_items, mci_delta
            ),
            "confidence":      _score_confidence(interv, mci_baseline, mci_delta),
            "obj_align":       _clamp(obj_score * 2, 0, 1),
            "est_budget":      est_budget,
            "eff_label":       eff_label,
            "why":             why,
            "tradeoff_str":    tradeoff_str,
            "deployment_speed": deployment_speed,
            "timeline_label":  timeline_label,
            "cost_tier":       cost_tier,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    return [
        PolicyRecommendation(
            rank=i + 1,
            intervention_key=s["interv"].key,
            intervention_label=s["interv"].label,
            rationale=s["rationale"],
            expected_mci_delta=round(s["mci_delta"], 1),
            expected_wsi_delta=round(s["wsi_delta"], 1),
            expected_wei_delta=round(s["wei_delta"], 1),
            primary_factor=s["interv"].primary_factor,
            feasibility=s["interv"].feasibility,
            confidence=s["confidence"],
            risk_factors=s["risks"],
            reference=s["interv"].reference,
            objective_alignment=round(s["obj_align"], 2),
            estimated_budget_cr=s["est_budget"],
            cost_efficiency=s["eff_label"],
            why_selected=s["why"],
            tradeoff_summary=s["tradeoff_str"],
            deployment_speed=s["deployment_speed"],
            timeline_label=s["timeline_label"],
            cost_tier=s["cost_tier"],
        )
        for i, s in enumerate(scored[:top_n])
    ]


def _build_why_selected(
    interv, district_row: dict,
    attribution_items: list[AttributionItem],
    factor_gap: dict[str, float],
    top_var_per_factor: dict[str, str],
    objective_key: str,
    mci_delta: float,
    eff_label: str,
) -> list[str]:
    """
    Returns 2–4 plain-language bullet points explaining why this intervention
    was recommended for this specific district.

    Rule-based: each rule checks a measurable condition and generates a sentence.
    """
    reasons = []

    # Reason 1: primary factor weakness
    pf    = interv.primary_factor
    gap   = factor_gap.get(pf, 0)
    var   = top_var_per_factor.get(pf, "unknown variable")
    factor_names = {"IFS": "Infrastructure", "DLS": "Digital Literacy",
                    "SES": "Socio-Economic",  "WDI": "Women Inclusion"}
    if gap > 5:
        reasons.append(
            f"{factor_names.get(pf, pf)} is a primary weak factor "
            f"(main driver: {var}) — this intervention directly addresses it."
        )

    # Reason 2: objective alignment
    obj = OBJECTIVE_MAP.get(objective_key)
    if obj:
        obj_w = obj.intervention_weights.get(pf, 0)
        if obj_w >= 0.25:
            reasons.append(
                f"Strongly aligned with '{obj.label}' objective "
                f"(factor weight {obj_w:.0%})."
            )

    # Reason 3: cost efficiency
    if eff_label == "HIGH":
        cp = COST_PROFILES.get(interv.key)
        if cp:
            reasons.append(
                f"High impact per ₹ invested ({cp.cost_tier} cost tier, "
                f"+{mci_delta:.1f} MCI pts at 70% intensity)."
            )
    elif eff_label == "MEDIUM":
        reasons.append(f"Moderate cost efficiency; +{mci_delta:.1f} MCI pts projected.")

    # Reason 4: feasibility / speed
    cp = COST_PROFILES.get(interv.key)
    if cp and cp.deployment_speed == "Fast":
        reasons.append(
            f"Rapid deployment possible ({cp.timeline_label}) "
            "via existing government delivery channels."
        )
    elif cp and cp.deployment_speed == "Medium":
        reasons.append(f"Deployable within {cp.timeline_label}.")

    # Reason 5: district-specific condition
    mci_base = float(district_row.get("MCI", 50))
    if mci_base < 25 and interv.primary_factor == "IFS":
        reasons.append(
            "Severe desert — physical infrastructure is the prerequisite "
            "for all other digital interventions."
        )
    elif interv.key == "women_digital_skilling":
        gvi = float(district_row.get("gender_vulnerability_index", 50))
        if gvi > 40:
            reasons.append(
                f"High gender vulnerability index ({gvi:.0f}) indicates "
                "women-specific digital exclusion in this district."
            )

    return reasons[:4]


def _build_rationale(
    interv,
    attribution_items: list[AttributionItem],
    mci_delta: float,
) -> str:
    top_factor_items = [a for a in attribution_items if a.factor == interv.primary_factor]
    if top_factor_items:
        top_var = top_factor_items[0].variable
        return (
            f"Primary weak factor: {interv.primary_factor} "
            f"(driven by: {top_var}). "
            f"Projected MCI uplift: +{mci_delta:.1f} pts at 70% intensity."
        )
    return f"{interv.description} · MCI uplift: +{mci_delta:.1f} pts at 70%."


def _assess_risks(interv, district_row: dict, mci_delta: float) -> list[str]:
    risks = []
    mci   = float(district_row.get("MCI", 50))
    if mci < 25 and interv.primary_factor == "DLS":
        risks.append("Severe infrastructure deficit may limit digital literacy uptake")
    if interv.feasibility == "medium":
        risks.append("Requires inter-departmental coordination")
    if mci_delta > 15:
        risks.append("High projected uplift assumes full saturation coverage")
    if interv.category == "Infrastructure":
        risks.append("Last-mile connectivity gap may persist without demand-side interventions")
    cp = COST_PROFILES.get(interv.key)
    if cp and cp.deployment_speed == "Slow":
        risks.append("Slow deployment — 18–36 months before measurable impact")
    if not risks:
        risks.append("Low risk — well-tested programme model with existing delivery chain")
    return risks[:3]


def _score_confidence(interv, mci_baseline: float, mci_delta: float) -> float:
    base = 0.75
    if interv.feasibility == "high":
        base += 0.10
    if mci_delta > 20:
        base -= 0.15
    if mci_baseline < 25:
        base -= 0.08
    if interv.reference:
        base += 0.05
    cp = COST_PROFILES.get(interv.key)
    if cp and cp.deployment_speed == "Fast":
        base += 0.05   # fast = less time for things to go wrong
    return round(_clamp(base, 0.30, 0.95), 2)