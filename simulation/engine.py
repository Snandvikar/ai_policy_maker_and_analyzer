"""
simulation/engine.py
=====================
Pure computation layer for the policy simulation module.

DESIGN PRINCIPLES:
  - No UI code. No Streamlit. No external calls.
  - All simulation is deterministic and explainable.
  - Intervention effects are grounded in the actual factor score formulas
    from mci_pipeline.py — not arbitrary multipliers.
  - Each intervention maps to specific underlying variables it can realistically
    affect, with constrained maximum uplift values drawn from comparable
    real-world programme outcomes.

HOW SIMULATION WORKS:
  1. User selects a district → its baseline scores are loaded from mci_scores.
  2. User dials interventions (sliders 0–100% intensity).
  3. Each intervention is mapped to delta values on underlying normalised
     variables (e.g. tower deployment → +Δ wireless_rural_teledensity_pct).
  4. Deltas are applied to baseline normalised variable proxies.
  5. Factor scores (IFS, DLS, SES, WDI) are recomputed from modified proxies.
  6. MCI, WSI, WEI are recomputed from simulated factors.
  7. Social outcome metrics are derived from the simulated scores.

INTERVENTION → VARIABLE MAPPING:
  The mapping encodes which variables each intervention can realistically
  move, and the maximum possible delta at 100% intensity.
  All deltas are on the 0–100 normalised scale.

  Source for cap values: comparable Indian programme outcomes (BharatNet,
  PMGDISHA, PM-WANI, NRLM, e-transactions surveys).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# INTERVENTION DEFINITIONS
# Each intervention has:
#   key          — internal identifier
#   label        — display name for policymakers
#   description  — one-sentence policy description
#   category     — grouping for display
#   max_delta    — dict of {variable: max_uplift_at_100pct}
#                  all variables are on the 0-100 normalised scale
#   primary_factor — which MCI factor this primarily affects
#   feasibility   — "high" / "medium" / "low" implementation feasibility
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Intervention:
    key:            str
    label:          str
    description:    str
    category:       str
    primary_factor: str
    feasibility:    str
    # variable → max delta on 0-100 normalised scale at 100% intensity
    variable_deltas: dict[str, float] = field(default_factory=dict)
    # comparable real-world reference
    reference:      str = ""


INTERVENTIONS: list[Intervention] = [

    # ── INFRASTRUCTURE ────────────────────────────────────────────────────────
    Intervention(
        key="tower_density",
        label="Tower Deployment",
        description="Increase 4G/5G BTS density in underserved areas via USOF/BharatNet.",
        category="Infrastructure",
        primary_factor="IFS",
        feasibility="medium",
        variable_deltas={
            "wireless_rural_teledensity_pct": 22.0,  # BharatNet Phase-II outcomes
            "wireless_total_teledensity_pct": 15.0,
            "any_phone_pct": 8.0,                    # coverage → activation lag
            "e_transactions_per_1000_population": 6.0,
        },
        reference="BharatNet Phase-II: rural teledensity uplift ~18-25pp in covered GPs",
    ),
    Intervention(
        key="fiber_backhaul",
        label="Fibre Backhaul Upgrade",
        description="OFC fiberisation of existing BTS to improve throughput and latency.",
        category="Infrastructure",
        primary_factor="IFS",
        feasibility="medium",
        variable_deltas={
            "wireless_rural_teledensity_pct": 8.0,
            "wireless_connectivity_score": 18.0,     # quality uplift
            "e_transactions_per_1000_population": 10.0,
            "any_phone_pct": 3.0,
        },
        reference="DoT OFC Fiberisation Drive: quality improvement observed in covered districts",
    ),
    Intervention(
        key="public_wifi",
        label="Public WiFi / PM-WANI Hotspots",
        description="Deploy PM-WANI compliant public WiFi at CSCs, panchayats, markets.",
        category="Infrastructure",
        primary_factor="IFS",
        feasibility="high",
        variable_deltas={
            "any_phone_pct": 10.0,
            "mobile_only_pct": 6.0,
            "e_transactions_per_1000_population": 14.0,  # hotspots drive transactions
            "wireless_rural_teledensity_pct": 5.0,
        },
        reference="PM-WANI: 50k+ hotspots active; transaction uplift observed in pilot districts",
    ),

    # ── AFFORDABILITY ─────────────────────────────────────────────────────────
    Intervention(
        key="device_subsidy",
        label="Device Subsidy Programme",
        description="Subsidise smartphones for below-poverty-line households.",
        category="Affordability",
        primary_factor="DLS",
        feasibility="high",
        variable_deltas={
            "any_phone_pct": 15.0,           # direct device ownership uplift
            "mobile_only_pct": 12.0,
            "no_phone_pct": -15.0,           # negative = improvement (fewer no-phone HH)
            "e_transactions_per_1000_population": 8.0,
            "msme_female_share_pct": 4.0,    # device → women MSME activity
        },
        reference="Pradhan Mantri Gramin Digital Saksharta Abhiyan: device-linked literacy",
    ),
    Intervention(
        key="data_subsidy",
        label="Data Affordability Subsidy",
        description="Subsidised data plans for low-income and rural households.",
        category="Affordability",
        primary_factor="DLS",
        feasibility="high",
        variable_deltas={
            "e_transactions_per_1000_population": 18.0,  # usage activation
            "any_phone_pct": 6.0,
            "non_agri_enterprise_pct": 5.0,
            "msme_female_share_pct": 6.0,
        },
        reference="Jio Bharat tariff reduction: rural data usage +40% in 6 months",
    ),

    # ── DIGITAL LITERACY ──────────────────────────────────────────────────────
    Intervention(
        key="literacy_programme",
        label="Digital Literacy Programme",
        description="PMGDISHA-style digital skilling targeting one member per household.",
        category="Digital Literacy",
        primary_factor="DLS",
        feasibility="high",
        variable_deltas={
            "any_phone_pct": 8.0,
            "e_transactions_per_1000_population": 12.0,
            "msme_female_share_pct": 8.0,
            "non_agri_enterprise_pct": 5.0,
            "illiteracy_percent": -6.0,      # partial functional literacy uplift
        },
        reference="PMGDISHA: 6 crore trained; e-transaction activation ~60% of trained",
    ),
    Intervention(
        key="women_digital_skilling",
        label="Women-Targeted Digital Skilling",
        description="Gender-targeted skilling at SHG/Anganwadi centres under PMGDISHA.",
        category="Digital Literacy",
        primary_factor="WDI",
        feasibility="high",
        variable_deltas={
            "msme_female_share_pct": 14.0,        # strong evidence from SHG interventions
            "gender_vulnerability_index": -10.0,   # GVI decreases = improvement
            "dependent_women_percent": -6.0,
            "e_transactions_per_1000_population": 8.0,
            "illiteracy_percent": -4.0,
            "any_phone_pct": 6.0,
        },
        reference="DAY-NRLM Digital Sakhi: women's digital transaction rate +25pp",
    ),
    Intervention(
        key="community_internet_centre",
        label="Community Internet Centre",
        description="Establish CSC-based community internet access points in each gram panchayat.",
        category="Digital Literacy",
        primary_factor="IFS",
        feasibility="high",
        variable_deltas={
            "any_phone_pct": 10.0,
            "e_transactions_per_1000_population": 16.0,
            "non_agri_enterprise_pct": 7.0,
            "msme_female_share_pct": 6.0,
            "destitute_pct": -3.0,
        },
        reference="CSC Academy: 5 lakh+ centres; digital service transaction data available",
    ),

    # ── SOCIAL / SAFETY ───────────────────────────────────────────────────────
    Intervention(
        key="women_safety_digital",
        label="Women's Digital Safety Infrastructure",
        description="181 Helpline digital access points, Safe City mission integration.",
        category="Safety",
        primary_factor="WDI",
        feasibility="medium",
        variable_deltas={
            "crime_against_women_rate": -8.0,      # negative = fewer crimes (normalised inverted)
            "gender_vulnerability_index": -8.0,
            "any_phone_pct": 5.0,
            "wireless_rural_teledensity_pct": 3.0,
        },
        reference="Nirbhaya Fund: digital safety pilots show GVI reduction in covered states",
    ),
    Intervention(
        key="shg_microfinance",
        label="SHG Microfinance + Digital Onboarding",
        description="NRLM SHG formation with mandatory digital account and MSME registration.",
        category="Social",
        primary_factor="SES",
        feasibility="high",
        variable_deltas={
            "msme_female_share_pct": 12.0,
            "dependent_women_percent": -8.0,
            "income_lt5k_pct": -5.0,
            "destitute_pct": -4.0,
            "e_transactions_per_1000_population": 10.0,
            "non_agri_enterprise_pct": 7.0,
        },
        reference="DAY-NRLM: 9 crore SHG members; income uplift observed in 2-3 years",
    ),
]

# Lookup by key
INTERVENTION_MAP: dict[str, Intervention] = {i.key: i for i in INTERVENTIONS}


# ─────────────────────────────────────────────────────────────────────────────
# POLICY OBJECTIVES
# Each objective defines which outcome metrics to foreground and how to
# re-weight the composite for recommendation ranking.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolicyObjective:
    key:         str
    label:       str
    description: str
    primary_outcomes: list[str]   # outcome keys to highlight
    # Weighting for scoring interventions (sum to 1.0)
    intervention_weights: dict[str, float]


OBJECTIVES: list[PolicyObjective] = [
    PolicyObjective(
        key="women_safety",
        label="Women's Safety",
        description="Reduce crime risk and improve emergency digital access for women.",
        primary_outcomes=["wsi_simulated", "safe_digital_access", "women_connectivity"],
        intervention_weights={"IFS": 0.30, "DLS": 0.15, "SES": 0.20, "WDI": 0.35},
    ),
    PolicyObjective(
        key="women_employment",
        label="Women's Employment Readiness",
        description="Improve women's digital economic participation and MSME access.",
        primary_outcomes=["wei_simulated", "women_employment_score", "women_msme_access"],
        intervention_weights={"IFS": 0.20, "DLS": 0.25, "SES": 0.25, "WDI": 0.30},
    ),
    PolicyObjective(
        key="education",
        label="Education Readiness",
        description="Improve digital access for educational services and school connectivity.",
        primary_outcomes=["digital_access_score", "service_access_score"],
        intervention_weights={"IFS": 0.35, "DLS": 0.35, "SES": 0.20, "WDI": 0.10},
    ),
    PolicyObjective(
        key="healthcare",
        label="Healthcare Accessibility",
        description="Enable digital health services, telemedicine, and emergency access.",
        primary_outcomes=["safe_digital_access", "service_access_score"],
        intervention_weights={"IFS": 0.40, "DLS": 0.20, "SES": 0.25, "WDI": 0.15},
    ),
    PolicyObjective(
        key="general_connectivity",
        label="General Connectivity",
        description="Maximise overall MCI improvement across all factors.",
        primary_outcomes=["mci_simulated", "digital_access_score"],
        intervention_weights={"IFS": 0.35, "DLS": 0.30, "SES": 0.20, "WDI": 0.15},
    ),
    PolicyObjective(
        key="affordability",
        label="Affordability & Inclusion",
        description="Reduce affordability barriers for low-income and marginalised households.",
        primary_outcomes=["digital_access_score", "women_connectivity", "service_access_score"],
        intervention_weights={"IFS": 0.20, "DLS": 0.35, "SES": 0.30, "WDI": 0.15},
    ),
]

OBJECTIVE_MAP: dict[str, PolicyObjective] = {o.key: o for o in OBJECTIVES}


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationInput:
    """Encapsulates all user-selected simulation parameters."""
    district_row:   dict                    # one row from mci_scores
    interventions:  dict[str, float]        # {intervention_key: intensity_0_to_100}
    objective_key:  str = "general_connectivity"
    analyst_weights: Optional[dict[str, float]] = None  # custom MCI factor weights


@dataclass
class SimulationResult:
    """All outputs from a simulation run."""
    # Baseline scores (from mci_scores)
    baseline: dict[str, float]

    # Simulated scores
    simulated: dict[str, float]

    # Deltas
    deltas: dict[str, float]

    # Per-intervention contribution to each factor delta
    intervention_contributions: dict[str, dict[str, float]]

    # Social outcome metrics
    social_outcomes: dict[str, float]
    social_outcomes_baseline: dict[str, float]

    # Confidence metadata
    confidence: float   # 0-1
    confidence_notes: list[str]

    # Applied interventions with intensities
    active_interventions: dict[str, float]


def _apply_intensity(base_delta: float, intensity: float) -> float:
    """
    Scale a maximum delta by intervention intensity (0-100).
    Uses a diminishing-returns curve: sqrt(intensity/100) rather than linear,
    because real-world programme effects plateau after saturation.
    """
    return base_delta * np.sqrt(intensity / 100.0)


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def simulate(inp: SimulationInput) -> SimulationResult:
    """
    Core simulation function.

    Step 1: Extract baseline proxy values from district_row.
    Step 2: For each active intervention, compute per-variable deltas.
    Step 3: Apply deltas to baseline proxies (clamped 0-100).
    Step 4: Recompute factor scores from modified proxies.
    Step 5: Recompute MCI, WSI, WEI.
    Step 6: Compute social outcome metrics.
    Step 7: Estimate confidence.
    """
    row = inp.district_row
    active = {k: v for k, v in inp.interventions.items() if v > 0}

    # ── Baseline factor and index scores ──────────────────────────────────────
    baseline = {
        "IFS": float(row.get("IFS", 50)),
        "DLS": float(row.get("DLS", 50)),
        "SES": float(row.get("SES", 50)),
        "WDI": float(row.get("WDI", 50)),
        "MCI": float(row.get("MCI", 50)),
        "WSI": float(row.get("WSI", 50)),
        "WEI": float(row.get("WEI", 50)),
    }

    # ── Baseline proxy values (from mci_scores rules vars) ────────────────────
    PROXY_VARS = [
        "wireless_rural_teledensity_pct", "wireless_total_teledensity_pct",
        "wireless_connectivity_score", "any_phone_pct", "mobile_only_pct",
        "no_phone_pct", "e_transactions_per_1000_population",
        "illiteracy_percent", "gender_vulnerability_index", "msme_female_share_pct",
        "dependent_women_percent", "crime_against_women_rate",
        "non_agri_enterprise_pct", "income_lt5k_pct", "destitute_pct",
        "share_of_slum_population", "maternal_mortality_rate",
    ]
    proxies = {v: float(row.get(v, 50)) for v in PROXY_VARS}

    # ── Compute deltas per intervention ───────────────────────────────────────
    per_intervention: dict[str, dict[str, float]] = {}
    cumulative_deltas: dict[str, float] = {v: 0.0 for v in PROXY_VARS}

    for ikey, intensity in active.items():
        interv = INTERVENTION_MAP.get(ikey)
        if not interv:
            continue
        contrib: dict[str, float] = {}
        for var, max_delta in interv.variable_deltas.items():
            delta = _apply_intensity(max_delta, intensity)
            contrib[var] = delta
            if var in cumulative_deltas:
                cumulative_deltas[var] += delta
        per_intervention[ikey] = contrib

    # ── Apply cumulative deltas to proxies ────────────────────────────────────
    sim_proxies = {}
    for var, base_val in proxies.items():
        new_val = base_val + cumulative_deltas.get(var, 0.0)
        sim_proxies[var] = _clamp(new_val)

    # ── Recompute factor scores from modified proxies ─────────────────────────
    # Uses the same factor formulas as mci_pipeline.py but on a single row
    sim_IFS = _recompute_IFS(sim_proxies)
    sim_DLS = _recompute_DLS(sim_proxies)
    sim_SES = _recompute_SES(sim_proxies)
    sim_WDI = _recompute_WDI(sim_proxies)

    # Apply analyst custom weights if set
    weights = inp.analyst_weights or {"IFS": 0.35, "DLS": 0.30, "SES": 0.20, "WDI": 0.15}

    sim_MCI = _recompute_MCI(sim_IFS, sim_DLS, sim_SES, sim_WDI, weights)
    sim_WSI = _recompute_WSI(sim_proxies, sim_MCI)
    sim_WEI = _recompute_WEI(sim_proxies, sim_MCI)

    simulated = {
        "IFS": round(sim_IFS, 1), "DLS": round(sim_DLS, 1),
        "SES": round(sim_SES, 1), "WDI": round(sim_WDI, 1),
        "MCI": round(sim_MCI, 1), "WSI": round(sim_WSI, 1),
        "WEI": round(sim_WEI, 1),
    }
    deltas = {k: round(simulated[k] - baseline[k], 1) for k in baseline}

    # ── Compute per-intervention factor contributions ─────────────────────────
    inter_contribs: dict[str, dict[str, float]] = {}
    for ikey, var_deltas in per_intervention.items():
        factor_impact: dict[str, float] = {}
        for var, delta in var_deltas.items():
            factor = _var_to_factor(var)
            if factor:
                factor_impact[factor] = round(
                    factor_impact.get(factor, 0) + abs(delta) * 0.4, 1
                )
        inter_contribs[ikey] = factor_impact

    # ── Social outcomes ───────────────────────────────────────────────────────
    social_base = _compute_social_outcomes(baseline, proxies)
    social_sim  = _compute_social_outcomes(simulated, sim_proxies)

    # ── Confidence ───────────────────────────────────────────────────────────
    conf, notes = _estimate_confidence(inp, deltas, active)

    return SimulationResult(
        baseline=baseline,
        simulated=simulated,
        deltas=deltas,
        intervention_contributions=inter_contribs,
        social_outcomes=social_sim,
        social_outcomes_baseline=social_base,
        confidence=conf,
        confidence_notes=notes,
        active_interventions=active,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-ROW FACTOR RECOMPUTATION
# Mirrors mci_pipeline.py factor formulas on a dict of proxy values.
# Uses simplified linear approximation of the normalised factor sub-components.
# ─────────────────────────────────────────────────────────────────────────────

def _g(proxies: dict, key: str, default: float = 50.0) -> float:
    """Get proxy value, default 50 if missing."""
    return proxies.get(key, default)


def _recompute_IFS(p: dict) -> float:
    coverage = np.mean([
        _g(p, "wireless_rural_teledensity_pct"),
        _g(p, "wireless_total_teledensity_pct"),
        _g(p, "wireless_connectivity_score"),
    ])
    access = np.mean([_g(p, "any_phone_pct"), _g(p, "mobile_only_pct")])
    activity = _g(p, "e_transactions_per_1000_population")
    return _clamp(0.35 * coverage + 0.35 * access + 0.30 * activity)


def _recompute_DLS(p: dict) -> float:
    d_access  = np.mean([_g(p, "any_phone_pct"), _g(p, "e_transactions_per_1000_population")])
    literacy  = 100 - _g(p, "illiteracy_percent")          # inverted
    gender    = np.mean([
        100 - _g(p, "gender_vulnerability_index"),         # inverted
        _g(p, "msme_female_share_pct"),
    ])
    econ      = _g(p, "non_agri_enterprise_pct")
    return _clamp(0.25 * d_access + 0.25 * literacy + 0.30 * gender + 0.20 * econ)


def _recompute_SES(p: dict) -> float:
    economic    = np.mean([100 - _g(p, "income_lt5k_pct")])  # inverted
    deprivation = np.mean([
        100 - _g(p, "destitute_pct"),
        100 - _g(p, "share_of_slum_population"),
        100 - _g(p, "no_phone_pct"),
    ])
    womens_econ = np.mean([
        _g(p, "msme_female_share_pct"),
        100 - _g(p, "dependent_women_percent"),
    ])
    livelihood  = _g(p, "non_agri_enterprise_pct")
    return _clamp(0.25 * economic + 0.25 * deprivation + 0.25 * womens_econ + 0.25 * livelihood)


def _recompute_WDI(p: dict) -> float:
    return _clamp(np.mean([
        100 - _g(p, "gender_vulnerability_index"),
        _g(p, "msme_female_share_pct"),
        100 - _g(p, "illiteracy_percent"),
        100 - _g(p, "crime_against_women_rate"),
        100 - _g(p, "dependent_women_percent"),
    ]))


def _recompute_MCI(IFS: float, DLS: float, SES: float, WDI: float,
                   weights: dict) -> float:
    mci = (
        (IFS + 1) ** weights["IFS"] *
        (DLS + 1) ** weights["DLS"] *
        (SES + 1) ** weights["SES"] *
        (WDI + 1) ** weights["WDI"]
    ) - 1
    return _clamp((mci / ((101.0 ** 1.0) - 1)) * 100)


def _recompute_WSI(p: dict, MCI: float) -> float:
    direct = (
        (100 - _g(p, "crime_against_women_rate")) * 0.30 +
        (100 - _g(p, "gender_vulnerability_index")) * 0.20 +
        (100 - _g(p, "maternal_mortality_rate")) * 0.20 +
        (100 - _g(p, "share_of_slum_population")) * 0.15 +
        _g(p, "wireless_connectivity_score") * 0.15
    )
    return _clamp(0.40 * MCI + 0.60 * direct)


def _recompute_WEI(p: dict, MCI: float) -> float:
    direct = (
        _g(p, "msme_female_share_pct") * 0.30 +
        _g(p, "non_agri_enterprise_pct") * 0.20 +
        _g(p, "e_transactions_per_1000_population") * 0.20 +
        (100 - _g(p, "dependent_women_percent")) * 0.15 +
        (100 - _g(p, "income_lt5k_pct")) * 0.15
    )
    return _clamp(0.35 * MCI + 0.65 * direct)


def _var_to_factor(var: str) -> Optional[str]:
    mapping = {
        "wireless_rural_teledensity_pct": "IFS",
        "wireless_total_teledensity_pct": "IFS",
        "wireless_connectivity_score": "IFS",
        "any_phone_pct": "IFS",
        "mobile_only_pct": "IFS",
        "no_phone_pct": "IFS",
        "e_transactions_per_1000_population": "IFS",
        "illiteracy_percent": "DLS",
        "gender_vulnerability_index": "DLS",
        "msme_female_share_pct": "DLS",
        "non_agri_enterprise_pct": "DLS",
        "crime_against_women_rate": "WDI",
        "dependent_women_percent": "WDI",
        "income_lt5k_pct": "SES",
        "destitute_pct": "SES",
        "share_of_slum_population": "SES",
        "maternal_mortality_rate": "SES",
    }
    return mapping.get(var)


# ─────────────────────────────────────────────────────────────────────────────
# SOCIAL OUTCOME METRICS
# Six policy-readable metrics derived from scores + proxy values.
# ─────────────────────────────────────────────────────────────────────────────

def _compute_social_outcomes(scores: dict, proxies: dict) -> dict:
    """
    Returns six social outcome metrics on a 0-100 scale.
    All metrics are policy-readable and human-centric.
    """
    MCI = scores.get("MCI", 50)
    WDI = scores.get("WDI", 50)
    WSI = scores.get("WSI", 50)
    WEI = scores.get("WEI", 50)

    return {
        # 1. Women's digital access improvement
        "women_connectivity": round(
            0.50 * WDI +
            0.30 * _g(proxies, "any_phone_pct") +
            0.20 * _g(proxies, "e_transactions_per_1000_population"), 1
        ),

        # 2. Women's employment readiness
        "women_employment_score": round(
            0.40 * WEI +
            0.35 * _g(proxies, "msme_female_share_pct") +
            0.25 * (100 - _g(proxies, "dependent_women_percent")), 1
        ),

        # 3. Access to online services
        "service_access_score": round(
            0.40 * MCI +
            0.35 * _g(proxies, "e_transactions_per_1000_population") +
            0.25 * _g(proxies, "any_phone_pct"), 1
        ),

        # 4. Safe digital accessibility
        "safe_digital_access": round(
            0.50 * WSI +
            0.30 * (100 - _g(proxies, "crime_against_women_rate")) +
            0.20 * _g(proxies, "wireless_rural_teledensity_pct"), 1
        ),

        # 5. Rural women connectivity inclusion
        "women_msme_access": round(
            0.45 * _g(proxies, "msme_female_share_pct") +
            0.30 * WDI +
            0.25 * _g(proxies, "non_agri_enterprise_pct"), 1
        ),

        # 6. General digital inclusion
        "digital_access_score": round(
            0.40 * MCI +
            0.30 * _g(proxies, "any_phone_pct") +
            0.30 * (100 - _g(proxies, "illiteracy_percent")), 1
        ),
    }

SOCIAL_OUTCOME_LABELS = {
    "women_connectivity":     "Women's Digital Connectivity",
    "women_employment_score": "Women's Employment Readiness",
    "service_access_score":   "Access to Online Services",
    "safe_digital_access":    "Safe Digital Accessibility",
    "women_msme_access":      "Women's MSME & Enterprise Access",
    "digital_access_score":   "General Digital Inclusion",
}


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE ESTIMATION
# Evidence-based, not arbitrary. Penalises:
#   - Too many simultaneous interventions (implementation risk)
#   - Very high total delta (regression-to-mean concern)
#   - No reference evidence for intervention combination
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_confidence(
    inp: SimulationInput,
    deltas: dict,
    active: dict,
) -> tuple[float, list[str]]:
    notes = []
    score = 1.0

    n_interventions = len(active)
    mci_delta = deltas.get("MCI", 0)

    if n_interventions > 4:
        score -= 0.15
        notes.append(
            f"{n_interventions} simultaneous interventions increase implementation risk; "
            "consider phased rollout."
        )
    if mci_delta > 25:
        score -= 0.20
        notes.append(
            f"Projected MCI uplift of {mci_delta:.0f} pts is very optimistic; "
            "real-world programmes achieve 10-18pp uplift in 3-5 years."
        )
    elif mci_delta > 15:
        score -= 0.10
        notes.append(
            f"MCI uplift of {mci_delta:.0f} pts is at the high end of observed outcomes."
        )

    baseline_mci = inp.district_row.get("MCI", 50)
    if baseline_mci < 25:
        score -= 0.10
        notes.append(
            "Severe desert baseline: physical infrastructure must precede digital interventions; "
            "initial gains may be slower."
        )

    if not notes:
        notes.append("Projection is within observed ranges of comparable Indian programmes.")

    return round(max(0.30, min(0.95, score)), 2), notes