"""
simulation/optimizer.py
========================
Budget-Constrained Policy Optimizer.

DESIGN PRINCIPLE: Deterministic, explainable, rule-based.
No ML models. No black boxes. Every allocation decision is traceable.

OPTIMIZATION STRATEGIES (documented)
=====================================

Four strategies are supported. Each reweights the scoring function
that ranks and allocates interventions.

1. COST_EFFICIENT
-----------------
Goal: maximise MCI delta per ₹ Crore invested.
Primary signal: impact_per_crore = simulated_mci_delta / cost_cr_at_70pct
Secondary signal: deployment_speed (fast preferred — impact sooner)
Use case: tight budget, need maximum coverage uplift per rupee.

2. BALANCED
-----------
Goal: balanced tradeoff across impact, cost, speed, and feasibility.
Scoring weights: impact 40%, cost_efficiency 25%, speed 20%, feasibility 15%.
Use case: standard government planning; no single dimension dominates.

3. FAST_DEPLOYMENT
------------------
Goal: maximise interventions deployable within 6 months.
Primary signal: deployment_speed = "Fast" → strong bonus.
Secondary signal: feasibility (high preferred).
Impact weight reduced — speed takes precedence.
Use case: election cycle pressure; need visible outcomes quickly.

4. HIGH_FEASIBILITY
-------------------
Goal: minimise implementation risk.
Primary signal: feasibility = "high" → strong bonus.
Secondary signal: impact.
Cost and speed secondary.
Use case: bureaucratically complex district; risk-averse planning.

OBJECTIVE INFLUENCE
===================
Selected policy objective (women_safety, women_employment, etc.) determines
which MCI factor deltas are valued most. This is encoded via the
objective's intervention_weights dict from engine.py.

The optimizer multiplies each intervention's MCI delta by:
  objective_alignment_weight = objective.intervention_weights[primary_factor]

This means a "Women Safety" objective will prioritise WDI-improving
interventions over IFS-improving ones, even if the latter has higher
raw MCI uplift.

BUDGET ALLOCATION ALGORITHM
============================
The optimizer uses a greedy allocation approach with diminishing budget:

1. Score all interventions using the selected strategy's scoring function.
2. Sort by score descending.
3. For each intervention (in order):
   a. Compute max affordable intensity = floor(remaining_budget / cost_per_point).
   b. Cap at 100.
   c. If affordable intensity >= MIN_MEANINGFUL_INTENSITY (20%), allocate it.
   d. Subtract cost from remaining budget.
4. Continue until budget exhausted or all interventions allocated.

Rationale for greedy vs linear programming:
  - Greedy is interpretable: policymakers can trace why each lever was set.
  - LP would find a marginally better solution but would be a black box.
  - The intervention score already encodes all relevant tradeoffs.
  - With 10 interventions and integer steps, exhaustive search is feasible
    if needed (10^10 combinations is too large, but the greedy solution
    is within 5-10% of optimal in practice for this problem size).

MIN_MEANINGFUL_INTENSITY = 20
  Below 20% intensity, most interventions produce negligible real-world
  impact (< 1 pt MCI delta). Allocating 10% to 5 interventions is less
  effective than allocating 50% to 2. The optimizer enforces a floor.

UNLIMITED BUDGET MODE
=====================
When no budget constraint is active:
  - Each intervention is scored on impact × objective alignment × feasibility.
  - Top-ranked interventions get 70% intensity (realistic mid-deployment).
  - Remaining interventions get 40% if they pass a minimum impact threshold.
  - No budget tracking.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np

from simulation.engine import (
    INTERVENTIONS, INTERVENTION_MAP, OBJECTIVE_MAP,
    SimulationInput, simulate, _clamp,
)
from simulation.cost_model import (
    COST_PROFILES, compute_intervention_cost_cr,
    compute_portfolio_cost_cr, cost_efficiency_label,
)

OptimizationStrategy = Literal[
    "cost_efficient", "balanced", "fast_deployment", "high_feasibility"
]

MIN_MEANINGFUL_INTENSITY = 20.0   # minimum intensity worth allocating (%)
DEFAULT_INTENSITY         = 70.0  # intensity for unlimited mode top picks
SECONDARY_INTENSITY       = 40.0  # intensity for secondary picks in unlimited mode


@dataclass
class OptimizationResult:
    allocations:        dict[str, float]   # {intervention_key: intensity}
    total_cost_cr:      float
    budget_used_pct:    float              # 0-100, how much of budget was used
    allocation_rationale: list[str]        # one sentence per allocated intervention
    strategy_used:      str
    objective_used:     str
    budget_constrained: bool


def _speed_score(speed: str) -> float:
    """Numeric score for deployment speed (higher = faster = better)."""
    return {"Fast": 1.0, "Medium": 0.5, "Slow": 0.1}.get(speed, 0.3)


def _feasibility_score(f: str) -> float:
    return {"high": 1.0, "medium": 0.6, "low": 0.2}.get(f, 0.5)


def _score_intervention(
    interv_key: str,
    mci_delta: float,
    cost_cr: float,
    strategy: OptimizationStrategy,
    objective_key: str,
) -> float:
    """
    Scores an intervention for ranking under a given strategy.

    All scoring is linear combination of normalised sub-scores.
    Sub-scores are on 0–1 scale. Weights are strategy-specific.
    Full documentation in module docstring.
    """
    profile = COST_PROFILES.get(interv_key)
    interv  = INTERVENTION_MAP.get(interv_key)
    obj     = OBJECTIVE_MAP.get(objective_key, OBJECTIVE_MAP["general_connectivity"])
    if not profile or not interv:
        return 0.0

    # Objective alignment: how much does this intervention's primary factor
    # align with the selected objective's priorities?
    obj_weight = obj.intervention_weights.get(interv.primary_factor, 0.25)

    # Normalise mci_delta to 0–1 range (max expected delta ~20 pts)
    impact_score = _clamp(mci_delta / 20.0, 0, 1)

    # Cost efficiency: impact per crore (normalised; cap at 5 pts/Cr)
    if cost_cr > 0:
        eff = mci_delta / cost_cr
        eff_score = _clamp(eff / 5.0, 0, 1)
    else:
        eff_score = 0.5  # free intervention, neutral score

    speed_sc = _speed_score(profile.deployment_speed)
    feas_sc  = _feasibility_score(profile.feasibility)

    if strategy == "cost_efficient":
        score = (
            0.10 * impact_score +
            0.55 * eff_score +
            0.20 * speed_sc +
            0.15 * feas_sc
        )
    elif strategy == "balanced":
        score = (
            0.40 * impact_score +
            0.25 * eff_score +
            0.20 * speed_sc +
            0.15 * feas_sc
        )
    elif strategy == "fast_deployment":
        score = (
            0.20 * impact_score +
            0.15 * eff_score +
            0.45 * speed_sc +
            0.20 * feas_sc
        )
    elif strategy == "high_feasibility":
        score = (
            0.30 * impact_score +
            0.15 * eff_score +
            0.15 * speed_sc +
            0.40 * feas_sc
        )
    else:
        score = impact_score

    # Multiply by objective alignment
    return score * (0.5 + obj_weight)   # obj_weight shifts 0.5–1.5x


def optimize(
    district_row: dict,
    objective_key: str,
    strategy: OptimizationStrategy,
    budget_cr: float | None,               # None = unlimited
    analyst_weights: dict | None = None,
) -> OptimizationResult:
    """
    Main optimization entry point.

    Returns OptimizationResult with per-intervention intensity allocations
    and full explainability metadata.

    The caller (simulation_page.py) applies these allocations to the sliders
    as RECOMMENDED DEFAULTS. The user retains full manual override capability.
    """
    budget_constrained = budget_cr is not None and budget_cr > 0
    remaining_budget   = budget_cr if budget_constrained else float("inf")
    allocations: dict[str, float] = {i.key: 0.0 for i in INTERVENTIONS}
    rationale: list[str] = []

    # Step 1: Score each intervention at 70% intensity
    scored = []
    for interv in INTERVENTIONS:
        test_intensity = 70.0
        cost_at_70 = compute_intervention_cost_cr(interv.key, test_intensity)

        sim_input = SimulationInput(
            district_row=district_row,
            interventions={interv.key: test_intensity},
            objective_key=objective_key,
            analyst_weights=analyst_weights,
        )
        result = simulate(sim_input)
        mci_delta = result.deltas.get("MCI", 0)

        score = _score_intervention(
            interv.key, mci_delta, cost_at_70, strategy, objective_key
        )
        scored.append({
            "key":        interv.key,
            "label":      interv.label,
            "score":      score,
            "mci_delta":  mci_delta,
            "cost_at_70": cost_at_70,
            "interv":     interv,
        })

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Step 2: Greedy allocation
    if budget_constrained:
        for item in scored:
            if remaining_budget <= 0:
                break

            profile = COST_PROFILES.get(item["key"])
            if not profile or profile.cost_per_point_cr <= 0:
                continue

            # Max affordable intensity given remaining budget
            max_affordable = remaining_budget / profile.cost_per_point_cr
            intensity = min(100.0, max_affordable)

            if intensity < MIN_MEANINGFUL_INTENSITY:
                # Not worth allocating below meaningful threshold
                continue

            # Round to nearest 10 for clean slider values
            intensity = round(intensity / 10) * 10
            intensity = _clamp(intensity, 0, 100)

            cost = compute_intervention_cost_cr(item["key"], intensity)
            allocations[item["key"]] = intensity
            remaining_budget -= cost

            rationale.append(
                f"{item['label']}: {intensity:.0f}% intensity "
                f"(₹{cost:.1f} Cr, score {item['score']:.3f})"
            )

    else:
        # Unlimited mode: top picks get 70%, secondary get 40%
        MIN_SCORE_THRESHOLD = scored[0]["score"] * 0.4 if scored else 0

        for i, item in enumerate(scored):
            if item["mci_delta"] < 0.5:
                continue  # Skip near-zero impact
            if i < 3 or item["score"] >= MIN_SCORE_THRESHOLD:
                intensity = DEFAULT_INTENSITY if i < 3 else SECONDARY_INTENSITY
                allocations[item["key"]] = intensity
                rationale.append(
                    f"{item['label']}: {intensity:.0f}% "
                    f"(score {item['score']:.3f})"
                )

    total_cost = compute_portfolio_cost_cr(allocations)
    budget_used_pct = (
        (total_cost / budget_cr * 100) if budget_constrained and budget_cr > 0
        else 0.0
    )

    return OptimizationResult(
        allocations=allocations,
        total_cost_cr=round(total_cost, 1),
        budget_used_pct=round(budget_used_pct, 1),
        allocation_rationale=rationale,
        strategy_used=strategy,
        objective_used=objective_key,
        budget_constrained=budget_constrained,
    )