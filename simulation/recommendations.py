"""
simulation/recommendations.py
================================
Policy Recommendation Engine.

Given a district's scores, the selected objective, and the root cause
attribution, generates ranked intervention recommendations with:
  - expected impact (which factors improve and by how much)
  - implementation feasibility
  - confidence score
  - risk factors
  - reference to comparable programmes

Outputs are policy-readable — no ML jargon exposed.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from simulation.engine import (
    INTERVENTIONS, INTERVENTION_MAP, OBJECTIVE_MAP,
    SimulationInput, simulate, _clamp,
)
from simulation.root_cause import AttributionItem, VAR_FACTOR


@dataclass
class PolicyRecommendation:
    rank:             int
    intervention_key: str
    intervention_label: str
    rationale:        str         # why this is recommended for this district
    expected_mci_delta: float
    expected_wsi_delta: float
    expected_wei_delta: float
    primary_factor:   str
    feasibility:      str         # high / medium / low
    confidence:       float       # 0-1
    risk_factors:     list[str]
    reference:        str
    objective_alignment: float    # 0-1, how well it aligns with selected objective


def generate_recommendations(
    district_row: dict,
    objective_key: str,
    attribution_items: list[AttributionItem],
    top_n: int = 4,
) -> list[PolicyRecommendation]:
    """
    Generates ranked recommendations for a district given the policy objective
    and root cause attribution.

    Scoring logic:
      1. Score each intervention by how much it addresses the top root causes.
      2. Weight scores by the selected objective's factor weights.
      3. Apply feasibility bonus for high-feasibility interventions.
      4. Simulate each intervention at 70% intensity to estimate delta.
    """
    objective = OBJECTIVE_MAP.get(objective_key, OBJECTIVE_MAP["general_connectivity"])
    mci_baseline = float(district_row.get("MCI", 50))

    # Build root-cause factor gaps from attribution
    factor_gap: dict[str, float] = {"IFS": 0, "DLS": 0, "SES": 0, "WDI": 0}
    for item in attribution_items:
        if item.factor in factor_gap:
            factor_gap[item.factor] += item.gap * item.contribution_pct / 100

    scored: list[dict] = []
    for interv in INTERVENTIONS:
        # Simulate at 70% intensity to estimate realistic delta
        sim_input = SimulationInput(
            district_row=district_row,
            interventions={interv.key: 70.0},
            objective_key=objective_key,
        )
        result = simulate(sim_input)
        mci_delta = result.deltas.get("MCI", 0)
        wsi_delta = result.deltas.get("WSI", 0)
        wei_delta = result.deltas.get("WEI", 0)

        # Score = objective-weighted factor improvement
        obj_score = sum(
            objective.intervention_weights.get(f, 0.25) *
            factor_gap.get(f, 0) *
            (abs(d) / (max(factor_gap.get(f, 1), 1)))
            for f, d in result.deltas.items()
            if f in ("IFS", "DLS", "SES", "WDI")
        )

        # Feasibility bonus
        feasibility_bonus = {"high": 0.15, "medium": 0.05, "low": -0.05}.get(
            interv.feasibility, 0
        )

        # Penalise if primary factor is not a weak factor
        primary_gap = factor_gap.get(interv.primary_factor, 0)
        if primary_gap < 5:
            obj_score *= 0.6  # low relevance penalty

        final_score = obj_score + feasibility_bonus + mci_delta * 0.01

        # Build risk factors
        risks = _assess_risks(interv, district_row, mci_delta)

        # Build rationale
        rationale = _build_rationale(interv, district_row, attribution_items, mci_delta)

        # Confidence
        conf = _score_confidence(interv, mci_baseline, mci_delta)

        scored.append({
            "interv":     interv,
            "score":      final_score,
            "mci_delta":  mci_delta,
            "wsi_delta":  wsi_delta,
            "wei_delta":  wei_delta,
            "risks":      risks,
            "rationale":  rationale,
            "confidence": conf,
            "obj_align":  _clamp(obj_score * 2, 0, 1),
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
        )
        for i, s in enumerate(scored[:top_n])
    ]


def _assess_risks(interv, district_row: dict, mci_delta: float) -> list[str]:
    risks = []
    mci = float(district_row.get("MCI", 50))
    if mci < 25 and interv.primary_factor == "DLS":
        risks.append("Severe infrastructure deficit may limit digital literacy uptake")
    if interv.feasibility == "medium":
        risks.append("Requires inter-departmental coordination")
    if mci_delta > 15:
        risks.append("High projected uplift assumes full saturation coverage")
    if interv.category == "Infrastructure":
        risks.append("Last-mile connectivity gap may persist without demand-side interventions")
    if not risks:
        risks.append("Low risk — well-tested programme model")
    return risks[:2]


def _build_rationale(
    interv, district_row: dict,
    attribution_items: list[AttributionItem],
    mci_delta: float,
) -> str:
    top_factor_items = [a for a in attribution_items if a.factor == interv.primary_factor]
    if top_factor_items:
        top_var = top_factor_items[0].variable
        return (
            f"This district's weakest dimension is {interv.primary_factor} "
            f"(primary driver: {top_var}). "
            f"{interv.description} is projected to lift MCI by +{mci_delta:.1f} pts."
        )
    return (
        f"{interv.description} Projected MCI uplift: +{mci_delta:.1f} pts at 70% intensity."
    )


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
    return round(_clamp(base, 0.30, 0.95), 2)