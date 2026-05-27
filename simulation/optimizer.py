"""
simulation/optimizer.py
========================
Budget-Constrained Policy Optimizer.

DESIGN PRINCIPLE: Deterministic, explainable, rule-based.
No ML models. No black boxes. Every allocation decision is traceable.

═══════════════════════════════════════════════════════════════════
INTENSITY / SLIDER / COST CONTRACT  (single source of truth)
═══════════════════════════════════════════════════════════════════

Three quantities are always in sync via two formulas:

  Imin = 0   (minimum slider position, fixed)
  Imax = 100 (maximum slider position, fixed)

  intensity  = Imin + (slider_pos / 100) × (Imax − Imin)
             = slider_pos                           [because Imin=0, Imax=100]

  cost (₹Cr) = intensity × cost_per_point_cr
             = slider_pos × cost_per_point_cr

  slider_pos = ((cost / cost_per_point_cr) − Imin) / (Imax − Imin) × 100
             = cost / cost_per_point_cr             [same simplification]

Because Imin=0 and Imax=100 are fixed, intensity and slider_pos are
numerically equal — but they are conceptually distinct:
  • slider_pos  — what the UI widget displays (0–100, integer steps of 5)
  • intensity   — the underlying deployment level passed to the engine (0–100 float)
  • cost        — derived from intensity, never stored independently

THE OPTIMIZER ALWAYS WORKS IN COST SPACE.
  - It decides how many ₹ Cr to allocate to each intervention.
  - It then back-calculates slider_pos = cost / cost_per_point_cr.
  - The panel uses slider_pos as the slider value directly.

This means OptimizationResult.allocations contains slider positions (0–100),
which the simulation engine receives as intensities (0–100). No conversion
needed at the panel layer.

═══════════════════════════════════════════════════════════════════
OPTIMIZATION STRATEGIES
═══════════════════════════════════════════════════════════════════

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

═══════════════════════════════════════════════════════════════════
OBJECTIVE INFLUENCE
═══════════════════════════════════════════════════════════════════

Selected policy objective determines which MCI factor deltas are valued
most, via the objective's intervention_weights dict from engine.py.

The optimizer multiplies each intervention's MCI delta by:
  objective_alignment_weight = objective.intervention_weights[primary_factor]

═══════════════════════════════════════════════════════════════════
BUDGET ALLOCATION ALGORITHM
═══════════════════════════════════════════════════════════════════

Greedy allocation with cost-space arithmetic:

1. Score all interventions at a reference intensity (70 pts).
2. Sort by score descending.
3. For each intervention (in order):
   a. Determine max affordable cost = min(remaining_budget,
                                          Imax × cost_per_point_cr)
   b. Round down to nearest 5-point slider step in cost space:
      cost_rounded = floor(max_cost / (5 × cpp)) × (5 × cpp)
   c. Back-calculate slider_pos = cost_rounded / cost_per_point_cr.
   d. Skip if slider_pos < MIN_MEANINGFUL_INTENSITY (20).
   e. Allocate; subtract cost from remaining budget.

Rationale for greedy vs LP:
  - Greedy is interpretable; policymakers can trace each decision.
  - LP finds marginally better solutions but is opaque.
  - The score already encodes all relevant tradeoffs.

MIN_MEANINGFUL_INTENSITY = 20
  Below 20 intensity points, most interventions produce negligible
  real-world impact (< 1 pt MCI delta).

UNLIMITED BUDGET MODE
=====================
  - Top-ranked interventions → DEFAULT_INTENSITY (70).
  - Secondary passes threshold → SECONDARY_INTENSITY (40).
  - slider_pos = intensity directly (identity relationship holds).
  - No budget tracking.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import math

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

# ── Intensity / slider constants ─────────────────────────────────────────────
IMIN = 0.0    # fixed lower bound for slider & intensity
IMAX = 10000.0  # fixed upper bound for slider & intensity

MIN_MEANINGFUL_INTENSITY = 20.0   # skip allocations below this slider position
DEFAULT_INTENSITY         = 70.0  # unlimited-mode top picks
SECONDARY_INTENSITY       = 40.0  # unlimited-mode secondary picks
SLIDER_STEP               = 50.0   # UI step size; cost rounding aligns to this


# ── Internal helpers ─────────────────────────────────────────────────────────

def intensity_to_cost(interv_key: str, intensity: float) -> float:
    """
    cost (₹Cr) = intensity × cost_per_point_cr

    Wraps compute_intervention_cost_cr for clarity; both are equivalent
    when cost_model is linear (cost = intensity × cpp).
    """
    return compute_intervention_cost_cr(interv_key, intensity)


def cost_to_slider_pos(interv_key: str, cost_cr: float) -> float:
    """
    slider_pos = cost / cost_per_point_cr
               = ((cost / cpp) − Imin) / (Imax − Imin) × 100

    Returns a float; callers round to nearest SLIDER_STEP as needed.
    Clamped to [IMIN, IMAX].
    """
    profile = COST_PROFILES.get(interv_key)
    if not profile or profile.cost_per_point_cr <= 0:
        return 0.0
    raw = (cost_cr / profile.cost_per_point_cr) / (IMAX) * 100.0
    return _clamp(raw, IMIN, IMAX)

def slider_pos_to_cost(interv_key: str, slider_pos: float) -> float:
    """
    cost (₹Cr) = slider_pos × cost_per_point_cr

    Inverse of cost_to_slider_pos. Used for rationale explanations.
    """
    profile = COST_PROFILES.get(interv_key)
    if not profile:
        return 0.0
    return round(slider_pos * profile.cost_per_point_cr * IMAX / 100.0, 2)


def slider_pos_to_intensity(slider_pos: float) -> float:
    """
    intensity = Imin + (slider_pos / 100) × (Imax − Imin)
              = slider_pos  [with Imin=0, Imax=100]

    Explicit formula kept for documentation; returns slider_pos unchanged.
    """
    return IMIN + (slider_pos / 100.0) * (IMAX - IMIN)


def _round_to_step(value: float, step: float = SLIDER_STEP) -> float:
    """Round value down to the nearest multiple of step."""
    return math.floor(value / step) * step


# ── Scoring ──────────────────────────────────────────────────────────────────

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

    All scoring is a linear combination of normalised sub-scores (0–1).
    Weights are strategy-specific (see module docstring).
    """
    profile = COST_PROFILES.get(interv_key)
    interv  = INTERVENTION_MAP.get(interv_key)
    obj     = OBJECTIVE_MAP.get(objective_key, OBJECTIVE_MAP["general_connectivity"])
    if not profile or not interv:
        return 0.0

    # Objective alignment multiplier
    obj_weight = obj.intervention_weights.get(interv.primary_factor, 0.25)

    # impact_score: normalised to 0–1 (max expected MCI delta ≈ 20 pts)
    impact_score = _clamp(mci_delta / 20.0, 0, 1)

    # cost_efficiency: MCI pts per ₹ Cr (cap normalisation at 5 pts/Cr)
    if cost_cr > 0:
        eff_score = _clamp((mci_delta / cost_cr) / float(SLIDER_STEP), 0, 1)
    else:
        eff_score = 0.5

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

    # obj_weight ∈ [0, 1] → multiplier ∈ [0.5, 1.5]
    return score * (0.5 + obj_weight)


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    allocations:          dict[str, float]  # {intervention_key: slider_pos (0–100)}
    total_cost_cr:        float
    budget_used_pct:      float             # 0–100 — how much of budget was used
    allocation_rationale: list[str]         # one line per allocated intervention
    strategy_used:        str
    objective_used:       str
    budget_constrained:   bool


# ── Main entry point ─────────────────────────────────────────────────────────

def optimize(
    district_row: dict,
    objective_key: str,
    strategy: OptimizationStrategy,
    budget_cr: float | None,           # None = unlimited
    analyst_weights: dict | None = None,
) -> OptimizationResult:
    """
    Main optimization entry point.

    CONTRACT
    --------
    OptimizationResult.allocations maps intervention_key → slider_pos (0–100).

    slider_pos is derived from cost via:
        slider_pos = cost / cost_per_point_cr          (budget-constrained mode)
        slider_pos = DEFAULT_INTENSITY or SECONDARY_INTENSITY  (unlimited mode)

    The panel reads slider_pos directly as the slider value and as the
    intensity passed to the simulation engine. No conversion is needed
    at the UI layer because intensity = slider_pos (Imin=0, Imax=100).

    The caller applies allocations as recommended slider defaults.
    The user retains full manual override capability.
    """
    budget_constrained = budget_cr is not None and budget_cr > 0
    remaining_budget   = budget_cr if budget_constrained else float("inf")
    allocations: dict[str, float] = {i.key: 0.0 for i in INTERVENTIONS}
    rationale: list[str] = []

    # ── Step 1: Score each intervention at reference intensity (70 pts) ───────
    # Using a fixed reference lets scores be comparable across interventions.
    # The reference intensity maps to a reference cost via cost = 70 × cpp.
    REF_INTENSITY = 70.0
    scored = []

    for interv in INTERVENTIONS:
        profile = COST_PROFILES.get(interv.key)
        if not profile:
            continue

        ref_cost = intensity_to_cost(interv.key, REF_INTENSITY)

        sim_input = SimulationInput(
            district_row=district_row,
            interventions={interv.key: REF_INTENSITY},
            objective_key=objective_key,
            analyst_weights=analyst_weights,
        )
        result    = simulate(sim_input)
        mci_delta = result.deltas.get("MCI", 0.0)

        score = _score_intervention(
            interv.key, mci_delta, ref_cost, strategy, objective_key
        )
        scored.append({
            "key":      interv.key,
            "label":    interv.label,
            "score":    score,
            "mci_delta": mci_delta,
            "ref_cost": ref_cost,
            "cpp":      profile.cost_per_point_cr,  # cost per intensity point
        })

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # ── Step 2: Allocate ──────────────────────────────────────────────────────

    if budget_constrained:
        # ── Budget-constrained greedy (cost-space arithmetic) ─────────────────
        #
        # For each intervention (ranked by score):
        #   1. Max affordable cost = min(remaining_budget, Imax × cpp)
        #      → caps cost at the slider ceiling (100 × cpp)
        #   2. Round DOWN to the nearest SLIDER_STEP boundary in cost space:
        #      cost_step = SLIDER_STEP × cpp  (cost equivalent of one slider step)
        #      cost_rounded = floor(max_cost / cost_step) × cost_step
        #   3. Back-calculate: slider_pos = cost_rounded / cpp
        #      (implements the inverse formula exactly)
        #   4. Skip if slider_pos < MIN_MEANINGFUL_INTENSITY
        #   5. Allocate slider_pos; subtract cost_rounded from remaining_budget

        for item in scored:
            if remaining_budget <= 0:
                break

            cpp = item["cpp"]
            if cpp <= 0:
                continue

            before_remaining = remaining_budget

            # Max cost this intervention can absorb (capped at Imax × cpp)
            max_cost = min(remaining_budget, IMAX * cpp)

            # Align to nearest slider step in cost space
            cost_step    = SLIDER_STEP * cpp          # cost of one 5-pt slider move
            cost_rounded = math.floor(max_cost / cost_step) * cost_step

            # Back-calculate slider_pos from cost (inverse formula)
            slider_pos = cost_to_slider_pos(item["key"], cost_rounded)

            if slider_pos < MIN_MEANINGFUL_INTENSITY:
                # Not worth allocating; skip to preserve budget for higher-ranked items
                continue

            actual_cost = slider_pos_to_cost(item["key"], slider_pos)
            allocations[item["key"]] = slider_pos
            remaining_budget -= actual_cost

            rationale.append(
                f"{item['label']}: slider {slider_pos:.0f}% → intensity {slider_pos_to_intensity(slider_pos):.0f} , remaining budget ₹{remaining_budget:.1f} Cr, before_remainingudget ₹{before_remaining:.1f} Cr "
                f"(₹{actual_cost:.1f} Cr, score {item['score']:.3f})"
            )

    else:
        # ── Unlimited mode ────────────────────────────────────────────────────
        # Assign DEFAULT_INTENSITY to top picks, SECONDARY_INTENSITY to others.
        # slider_pos = intensity directly (identity holds with Imin=0, Imax=100).
        MIN_SCORE_THRESHOLD = scored[0]["score"] * 0.4 if scored else 0.0

        for i, item in enumerate(scored):
            if item["mci_delta"] < 0.5:
                continue  # skip near-zero impact interventions

            if i < 3 or item["score"] >= MIN_SCORE_THRESHOLD:
                slider_pos = DEFAULT_INTENSITY if i < 3 else SECONDARY_INTENSITY
                allocations[item["key"]] = slider_pos
                actual_cost = intensity_to_cost(item["key"], slider_pos)
                rationale.append(
                    f"{item['label']}: slider {slider_pos:.0f}% → intensity {slider_pos_to_intensity(slider_pos):.0f} "
                    f"(₹{actual_cost:.1f} Cr, score {item['score']:.3f})"
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