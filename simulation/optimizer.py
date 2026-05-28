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

Two-phase budget-constrained allocation:

PHASE 1 — Greedy (cost-space arithmetic):
1. Score all interventions at a reference intensity (70 pts).
2. Sort by score descending.
3. For each intervention (in order):
   a. Determine max affordable cost = min(remaining_budget,
                                          Imax × cost_per_point_cr)
   b. Round DOWN to the nearest SLIDER_STEP boundary in cost space:
      cost_step    = SLIDER_STEP × cpp
      cost_rounded = floor(max_cost / cost_step) × cost_step
   c. Back-calculate slider_pos = cost_rounded / cost_per_point_cr.
      (implements the inverse formula exactly)
   d. Skip if slider_pos < MIN_MEANINGFUL_INTENSITY (20).
   e. Allocate slider_pos; subtract cost_rounded from remaining_budget.

PHASE 2 — Local-search refinement (budget-constrained only, refine=True):
Iteratively moves _SWAP_STEP intensity points from one intervention
(donor) to another (receiver), accepting any swap that strictly
improves the portfolio's simulated MCI delta while staying within budget.

  For each pass (up to _MAX_PASSES):
    For each (donor, receiver) pair:
      - donor must have >= _SWAP_STEP to give
      - receiver must not exceed 100 after receiving
      - if donor drops into (0, MIN_MEANINGFUL_INTENSITY), zero it entirely
      - if receiver would go from 0 to below MIN_MEANINGFUL_INTENSITY, skip
      - evaluate portfolio MCI delta with simulate()
      - accept if improvement > _GAIN_EPSILON
    Apply best swap found; repeat until no improvement or pass limit hit.
  Snap final intensities to nearest multiple of 10 (floor, conservative).
  Verify budget invariant holds after snapping.

Rationale for greedy + local-search vs LP:
  - Greedy + refinement is interpretable: every decision is logged.
  - LP finds marginally better solutions but is opaque.
  - Local search recovers most of LP's gain for this problem size.

MIN_MEANINGFUL_INTENSITY = 20
  Below 20 intensity points, most interventions produce negligible
  real-world impact (< 1 pt MCI delta).

UNLIMITED BUDGET MODE
=====================
  - Top-ranked interventions to DEFAULT_INTENSITY (70).
  - Secondary passes threshold to SECONDARY_INTENSITY (40).
  - slider_pos = intensity directly (identity relationship holds).
  - Refinement NOT applied (fixed intensities by design).
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

# ── Intensity / slider constants ──────────────────────────────────────────────
IMIN = 0.0      # fixed lower bound for slider & intensity
IMAX = 10000.0  # fixed upper bound for slider & intensity

MIN_MEANINGFUL_INTENSITY = 20.0   # skip allocations below this slider position
DEFAULT_INTENSITY         = 70.0  # unlimited-mode top picks
SECONDARY_INTENSITY       = 40.0  # unlimited-mode secondary picks
SLIDER_STEP               = 5.0  # UI step size; cost rounding aligns to this

# ── Local-search refinement constants ────────────────────────────────────────
_SWAP_STEP    = 10.0  # intensity points moved per swap
_MAX_PASSES   = 20    # safety cap on outer hill-climbing loop
_GAIN_EPSILON = 1e-4  # minimum improvement to accept a swap (avoids noise)

# Label lookup built once at module level — avoids repeated list scans
_INTERV_LABELS: dict[str, str] = {i.key: i.label for i in INTERVENTIONS}


# ── Internal helpers ──────────────────────────────────────────────────────────

def intensity_to_cost(interv_key: str, intensity: float) -> float:
    """
    cost (Cr) = intensity x cost_per_point_cr

    Wraps compute_intervention_cost_cr for clarity; both are equivalent
    when cost_model is linear (cost = intensity x cpp).
    """
    return compute_intervention_cost_cr(interv_key, intensity)


def cost_to_slider_pos(interv_key: str, cost_cr: float) -> float:
    """
    slider_pos = cost / cost_per_point_cr
               = ((cost / cpp) - Imin) / (Imax - Imin) x 100

    Returns a float; callers round to nearest SLIDER_STEP as needed.
    Clamped to [IMIN, IMAX].
    """
    profile = COST_PROFILES.get(interv_key)
    if not profile or profile.cost_per_point_cr <= 0:
        return 0.0
    raw = (cost_cr / profile.cost_per_point_cr) / IMAX * 100.0
    return _clamp(raw, IMIN, IMAX)


def slider_pos_to_cost(interv_key: str, slider_pos: float) -> float:
    """
    cost (Cr) = slider_pos x cost_per_point_cr

    Inverse of cost_to_slider_pos. Used for rationale explanations.
    """
    profile = COST_PROFILES.get(interv_key)
    if not profile:
        return 0.0
    return round(slider_pos * profile.cost_per_point_cr * IMAX / 100.0, 2)


def slider_pos_to_intensity(slider_pos: float) -> float:
    """
    intensity = Imin + (slider_pos / 100) x (Imax - Imin)

    Explicit formula kept for documentation.
    """
    return IMIN + (slider_pos / 100.0) * (IMAX - IMIN)


def _round_to_step(value: float, step: float = SLIDER_STEP) -> float:
    """Round value down to the nearest multiple of step."""
    return math.floor(value / step) * step


# ── Scoring ───────────────────────────────────────────────────────────────────

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

    All scoring is a linear combination of normalised sub-scores (0-1).
    Weights are strategy-specific (see module docstring).
    """
    profile = COST_PROFILES.get(interv_key)
    interv  = INTERVENTION_MAP.get(interv_key)
    obj     = OBJECTIVE_MAP.get(objective_key, OBJECTIVE_MAP["general_connectivity"])
    if not profile or not interv:
        return 0.0

    # Objective alignment multiplier
    obj_weight = obj.intervention_weights.get(interv.primary_factor, 0.25)

    # impact_score: normalised to 0-1 (max expected MCI delta ~20 pts)
    impact_score = _clamp(mci_delta / 20.0, 0, 1)

    # cost_efficiency: MCI pts per Cr (cap normalisation at SLIDER_STEP pts/Cr)
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

    # obj_weight in [0, 1] → multiplier in [0.5, 1.5]
    return score * (0.5 + obj_weight)


# ── Local-search refinement helpers ──────────────────────────────────────────

def _portfolio_mci_delta(
    allocations: dict[str, float],
    district_row: dict,
    objective_key: str,
    analyst_weights: dict | None,
) -> float:
    """
    Simulate the FULL portfolio and return the aggregate MCI delta.
    This is the true objective used during refinement — NOT _score_intervention().
    Uses only non-zero allocations so simulate() does not waste computation.

    allocations values are slider_pos (0-100), passed to the engine as
    intensities. With Imin=0, Imax=100 the identity slider_pos = intensity
    holds, so no conversion is needed before calling simulate().
    """
    active = {k: v for k, v in allocations.items() if v > 0}
    if not active:
        return 0.0
    result = simulate(SimulationInput(
        district_row=district_row,
        interventions=active,
        objective_key=objective_key,
        analyst_weights=analyst_weights,
    ))
    return result.deltas.get("MCI", 0.0)


def _snap_to_tens(allocations: dict[str, float]) -> dict[str, float]:
    """
    Snap every intensity to the nearest multiple of 10 using floor.
    Conservative: never rounds up, so budget is never exceeded by rounding.
    Entries that drop to 0 are zeroed (not raised to MIN_MEANINGFUL_INTENSITY
    at this stage — caller re-verifies the floor if needed).
    """
    return {k: float((int(v) // 10) * 10) for k, v in allocations.items()}


def _refine_allocations(
    allocations: dict[str, float],
    district_row: dict,
    objective_key: str,
    analyst_weights: dict | None,
    budget_cr: float | None,
) -> tuple[dict[str, float], list[str]]:
    """
    Local-search (swap-based hill climbing) refinement of a greedy allocation.

    Iteratively moves _SWAP_STEP intensity points from one intervention
    (donor) to another (receiver), accepting any swap that strictly
    improves the portfolio's simulated MCI delta while staying within
    the budget constraint.

    allocations values are slider_pos (0-100). Since Imin=0, Imax=100,
    slider_pos = intensity numerically — the receiver cap (100.0) and
    donor floor (MIN_MEANINGFUL_INTENSITY=20) apply directly to these values.

    Returns:
        refined_allocations : dict[str, float]
        refinement_log      : list[str]  — one entry per accepted swap
                                           plus a summary line
    """
    log: list[str] = []

    # ── Edge case: nothing allocated ──────────────────────────────────────────
    active_keys = [k for k, v in allocations.items() if v > 0]
    if len(active_keys) == 0:
        log.append(
            "Refinement: greedy allocation was already locally optimal"
            " — no improving swaps found."
        )
        return allocations, log

    # ── Edge case: only one lever allocated — no valid swaps ──────────────────
    if len(active_keys) == 1:
        log.append(
            "Refinement: greedy allocation was already locally optimal"
            " — no improving swaps found."
        )
        return allocations, log

    # Work on a mutable copy
    alloc = dict(allocations)

    # Baseline portfolio MCI delta (true objective)
    current_score = _portfolio_mci_delta(alloc, district_row, objective_key, analyst_weights)
    initial_score = current_score

    n_passes = 0
    n_swaps  = 0

    for pass_idx in range(_MAX_PASSES):
        n_passes += 1
        best_gain      = 0.0
        best_swap_info = None   # (donor, receiver, candidate_alloc, candidate_score, candidate_cost)

        all_keys = list(alloc.keys())

        for donor_key in all_keys:
            if alloc[donor_key] < _SWAP_STEP:
                continue  # nothing to give

            for receiver_key in all_keys:
                if donor_key == receiver_key:
                    continue
                if alloc[receiver_key] + _SWAP_STEP > 100.0:
                    continue  # receiver already at cap

                # ── Build candidate allocation ─────────────────────────────
                candidate = dict(alloc)
                candidate[donor_key]    -= _SWAP_STEP
                candidate[receiver_key] += _SWAP_STEP

                # ── Enforce MIN_MEANINGFUL_INTENSITY on donor ──────────────
                # If donor drops into (0, MIN_MEANINGFUL_INTENSITY), zero it
                # entirely rather than leave a stub below the floor.
                if 0 < candidate[donor_key] < MIN_MEANINGFUL_INTENSITY:
                    candidate[donor_key] = 0.0

                # ── Enforce MIN_MEANINGFUL_INTENSITY on receiver ───────────
                # Receiver cannot jump from 0 to below the floor.
                # (_SWAP_STEP=10, floor=20 → receiver at 0 would become 10 — block.)
                if (alloc[receiver_key] == 0.0 and
                        candidate[receiver_key] < MIN_MEANINGFUL_INTENSITY):
                    continue

                # ── Budget check ───────────────────────────────────────────
                if budget_cr is not None:
                    candidate_cost = compute_portfolio_cost_cr(candidate)
                    if candidate_cost > budget_cr + 1e-6:  # small float tolerance
                        continue
                else:
                    candidate_cost = compute_portfolio_cost_cr(candidate)

                # ── Evaluate true portfolio objective ──────────────────────
                candidate_score = _portfolio_mci_delta(
                    candidate, district_row, objective_key, analyst_weights
                )
                gain = candidate_score - current_score

                if gain > best_gain + _GAIN_EPSILON:
                    best_gain = gain
                    best_swap_info = (
                        donor_key, receiver_key,
                        candidate, candidate_score, candidate_cost,
                    )

        # ── Apply best swap if found ───────────────────────────────────────
        if best_swap_info is not None:
            donor_key, receiver_key, candidate, candidate_score, candidate_cost = best_swap_info
            alloc         = candidate
            current_score = candidate_score
            n_swaps      += 1

            d_label = _INTERV_LABELS.get(donor_key, donor_key)
            r_label = _INTERV_LABELS.get(receiver_key, receiver_key)
            log.append(
                f"Refinement: moved {_SWAP_STEP:.0f}% from '{d_label}'"
                f" to '{r_label}'"
                f" (+{best_gain:.2f} MCI delta,"
                f" total now {candidate_cost:.1f} Cr)"
            )
        else:
            # No improving swap found — local optimum reached
            break

    # ── Snap intensities to multiples of 10 (floor, conservative) ────────────
    alloc = _snap_to_tens(alloc)

    # ── Re-verify budget after snapping ───────────────────────────────────────
    # Floor never over-allocates, but defensive check keeps the invariant explicit.
    if budget_cr is not None:
        final_cost = compute_portfolio_cost_cr(alloc)
        assert final_cost <= budget_cr + 1e-4, (
            f"Refinement post-snap budget violation: "
            f"{final_cost:.2f} Cr > {budget_cr:.2f} Cr"
        )

    # ── Summary log ───────────────────────────────────────────────────────────
    if n_swaps == 0:
        log.append(
            "Refinement: greedy allocation was already locally optimal"
            " — no improving swaps found."
        )
    else:
        total_gain = current_score - initial_score
        log.append(
            f"Refinement complete: {n_swaps} swap(s) over {n_passes} pass(es),"
            f" MCI delta improved by {total_gain:.2f}."
        )

    return alloc, log


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    allocations:          dict[str, float]  # {intervention_key: slider_pos (0-100)}
    total_cost_cr:        float
    budget_used_pct:      float             # 0-100 — how much of budget was used
    allocation_rationale: list[str]         # one line per allocated intervention
    strategy_used:        str
    objective_used:       str
    budget_constrained:   bool


# ── Main entry point ──────────────────────────────────────────────────────────

def optimize(
    district_row: dict,
    objective_key: str,
    strategy: OptimizationStrategy,
    budget_cr: float | None,           # None = unlimited
    analyst_weights: dict | None = None,
    refine: bool = True,               # enable local-search refinement
) -> OptimizationResult:
    """
    Main optimization entry point.

    Parameters
    ----------
    district_row     : one row from mci_scores as a dict
    objective_key    : policy objective key (see engine.py OBJECTIVES)
    strategy         : one of 'cost_efficient', 'balanced',
                       'fast_deployment', 'high_feasibility'
    budget_cr        : total budget in Crore (None = unlimited)
    analyst_weights  : custom MCI factor weights (None = default)
    refine           : if True (default), run local-search hill-climbing
                       after the greedy step when budget_constrained is True.
                       Pass refine=False to reproduce pre-refinement behaviour
                       exactly (regression testing).

    CONTRACT
    --------
    OptimizationResult.allocations maps intervention_key to slider_pos (0-100).

    slider_pos is derived from cost via:
        slider_pos = cost / cost_per_point_cr          (budget-constrained mode)
        slider_pos = DEFAULT_INTENSITY or SECONDARY_INTENSITY  (unlimited mode)

    The panel reads slider_pos directly as the slider value and passes it
    to the engine as intensity. No conversion needed at the UI layer because
    intensity = slider_pos (Imin=0, Imax=100).

    The caller applies allocations as recommended slider defaults.
    The user retains full manual override capability.
    """
    budget_constrained = budget_cr is not None and budget_cr > 0
    remaining_budget   = budget_cr if budget_constrained else float("inf")
    allocations: dict[str, float] = {i.key: 0.0 for i in INTERVENTIONS}
    rationale: list[str] = []

    # ── Step 1: Score each intervention at reference intensity (70 pts) ───────
    # Using a fixed reference lets scores be comparable across interventions.
    # The reference intensity maps to a reference cost via cost = 70 x cpp.
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
            "key":       interv.key,
            "label":     interv.label,
            "score":     score,
            "mci_delta": mci_delta,
            "ref_cost":  ref_cost,
            "cpp":       profile.cost_per_point_cr,  # cost per intensity point
        })

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # ── Step 2: Allocate ──────────────────────────────────────────────────────

    if budget_constrained:
        # ── Phase 1: Greedy (cost-space arithmetic) ───────────────────────────
        #
        # For each intervention (ranked by score):
        #   1. Max affordable cost = min(remaining_budget, IMAX x cpp)
        #      caps cost at the slider ceiling (IMAX x cpp)
        #   2. Round DOWN to the nearest SLIDER_STEP boundary in cost space:
        #      cost_step    = SLIDER_STEP x cpp  (cost of one slider step)
        #      cost_rounded = floor(max_cost / cost_step) x cost_step
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

            # Max cost this intervention can absorb (capped at IMAX x cpp)
            max_cost = min(remaining_budget, IMAX * cpp)

            # Align to nearest slider step in cost space
            cost_step    = SLIDER_STEP * cpp          # cost of one slider step
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
                f"{item['label']}: slider {slider_pos:.0f}% -> intensity {slider_pos_to_intensity(slider_pos):.0f},"
                f" remaining budget {remaining_budget:.1f} Cr,"
                f" before {before_remaining:.1f} Cr"
                f" ({actual_cost:.1f} Cr, score {item['score']:.3f})"
            )

        # ── Phase 2: Local-search refinement ──────────────────────────────────
        # refine=False preserves identical pre-refinement behaviour (regression).
        if refine:
            allocations, refinement_log = _refine_allocations(
                allocations=allocations,
                district_row=district_row,
                objective_key=objective_key,
                analyst_weights=analyst_weights,
                budget_cr=budget_cr,
            )
            rationale.extend(refinement_log)

    else:
        # ── Unlimited mode ────────────────────────────────────────────────────
        # Assign DEFAULT_INTENSITY to top picks, SECONDARY_INTENSITY to others.
        # slider_pos = intensity directly (identity holds with Imin=0, Imax=100).
        # Refinement NOT applied — fixed intensities by design.
        MIN_SCORE_THRESHOLD = scored[0]["score"] * 0.4 if scored else 0.0

        for i, item in enumerate(scored):
            if item["mci_delta"] < 0.5:
                continue  # skip near-zero impact interventions

            if i < 3 or item["score"] >= MIN_SCORE_THRESHOLD:
                slider_pos = DEFAULT_INTENSITY if i < 3 else SECONDARY_INTENSITY
                allocations[item["key"]] = slider_pos
                actual_cost = intensity_to_cost(item["key"], slider_pos)
                rationale.append(
                    f"{item['label']}: slider {slider_pos:.0f}% -> intensity {slider_pos_to_intensity(slider_pos):.0f}"
                    f" ({actual_cost:.1f} Cr, score {item['score']:.3f})"
                )

    # Recompute total cost AFTER refinement so budget_used_pct is accurate
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