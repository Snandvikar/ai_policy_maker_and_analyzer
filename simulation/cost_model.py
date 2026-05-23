"""
simulation/cost_model.py
=========================
Rule-based cost, deployment speed, and feasibility model for every intervention.

DECISION DOCUMENTATION
=======================
All values here are deterministic rule-based assignments, not ML predictions.
Every number has an explicit source or reasoning documented inline.

COST ASSIGNMENT LOGIC
---------------------
Cost is expressed in ₹ Crore per percentage point of intensity deployed
across an average district (~2 lakh households, ~5 GPs, ~200 km²).

Basis for cost-per-point values:
  Tower Deployment:
    BharatNet Phase-II capex: ₹25-30 Cr per 1000 GPs for towers + backhaul.
    An average district has ~200 GPs. ₹5-6 Cr per district at full deployment.
    Cost per intensity point (1% of 100%) = ₹0.05-0.06 Cr ≈ ₹0.06 Cr/pt.

  Fibre Backhaul:
    OFC fiberisation: ~₹3-5 lakh per km. A district needs ~500 km medium-haul.
    Total ₹15-25 Cr per district at full coverage. Per point ≈ ₹0.20 Cr/pt.

  Public WiFi (PM-WANI):
    ₹2-5 lakh per hotspot, 50 hotspots per district = ₹1-2.5 Cr.
    Per point ≈ ₹0.02 Cr/pt (cheapest infrastructure intervention).

  Device Subsidy:
    ₹2,500-3,500 per device, ~30k BPL HH per district targeted at full sat.
    Total ₹75-105 Cr at full deployment. Per point ≈ ₹0.90 Cr/pt (expensive).

  Data Subsidy:
    ₹100-200/month per HH, 12 months, 30k HH = ₹36-72 Cr/yr.
    Per point ≈ ₹0.50 Cr/pt. Recurring cost (modelled as 1-year programme).

  Digital Literacy Programme (PMGDISHA):
    ₹1,000-1,500 per trainee, ~30k trainees per district = ₹3-4.5 Cr.
    Per point ≈ ₹0.04 Cr/pt (very cost-effective).

  Women Digital Skilling:
    Same unit cost as literacy but smaller target cohort (SHG members ~20k).
    Per point ≈ ₹0.03 Cr/pt.

  Community Internet Centre (CSC):
    ₹5-10 lakh per centre, ~200 centres per district = ₹10-20 Cr.
    Per point ≈ ₹0.15 Cr/pt.

  Women's Digital Safety Infrastructure:
    181 helpline upgrades + CCTV integration: ₹0.5-2 Cr per district.
    Per point ≈ ₹0.02 Cr/pt (leverages existing infrastructure).

  SHG Microfinance + Digital Onboarding:
    NRLM revolving fund ₹15k per SHG (300 SHGs per district) = ₹4.5 Cr.
    Digital onboarding adds ₹1.5 Cr. Total ≈ ₹6 Cr. Per point ≈ ₹0.06 Cr/pt.

DEPLOYMENT SPEED LOGIC
----------------------
Speed is categorised as Fast / Medium / Slow.
Rule: based on procurement complexity + institutional readiness.

  Fast  (< 6 months to first impact):
    - Programmes with existing delivery infrastructure (CSCs, PMGDISHA, SHGs)
    - No new physical infrastructure required
    - Examples: device subsidy, data subsidy, digital literacy, women skilling,
               safety infrastructure, community internet centres

  Medium (6–18 months):
    - Requires some coordination but uses established frameworks
    - Examples: public WiFi (PM-WANI PDO setup), SHG onboarding (NRLM process)

  Slow  (18–36 months):
    - New physical infrastructure: towers, fibre
    - Requires spectrum allocation, RoW clearance, contractor tendering
    - Examples: tower deployment, fibre backhaul

FEASIBILITY LOGIC
-----------------
Feasibility is carried over from engine.py (high/medium/low) based on:
  high   = single department, proven delivery model, no new legislation
  medium = multi-department coordination required
  low    = requires new legislation, spectrum issues, or major procurement

COST EFFICIENCY LOGIC
---------------------
Cost efficiency = MCI delta per ₹ Crore at 70% intensity.
Computed dynamically in optimizer.py from simulation results.
Categorical labels: HIGH (>1 pt/Cr), MEDIUM (0.3–1), LOW (<0.3).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


DeploymentSpeed = Literal["Fast", "Medium", "Slow"]
CostTier        = Literal["Low", "Medium", "High"]


@dataclass
class InterventionCostProfile:
    key:                str
    cost_per_point_cr:  float          # ₹ Crore per intensity point (1–100 scale)
    deployment_speed:   DeploymentSpeed
    # feasibility is already in engine.py Intervention; mirrored here for convenience
    feasibility:        str            # "high" / "medium" / "low"
    # human-readable cost tier for the tradeoff table
    cost_tier:          CostTier
    # deployment timeline label for policymakers
    timeline_label:     str
    # notes for the tradeoff table
    deployment_note:    str


# ─────────────────────────────────────────────────────────────────────────────
# COST PROFILES
# Each entry corresponds 1:1 to an Intervention in engine.py (same key).
# ─────────────────────────────────────────────────────────────────────────────

COST_PROFILES: dict[str, InterventionCostProfile] = {

    "tower_density": InterventionCostProfile(
        key="tower_density",
        cost_per_point_cr=0.06,
        deployment_speed="Slow",
        feasibility="medium",
        cost_tier="Medium",
        timeline_label="18–36 months",
        deployment_note="RoW clearance, spectrum, contractor tendering required",
    ),

    "fiber_backhaul": InterventionCostProfile(
        key="fiber_backhaul",
        cost_per_point_cr=0.20,
        deployment_speed="Slow",
        feasibility="medium",
        cost_tier="High",
        timeline_label="18–30 months",
        deployment_note="OFC laying requires land acquisition and civil works",
    ),

    "public_wifi": InterventionCostProfile(
        key="public_wifi",
        cost_per_point_cr=0.02,
        deployment_speed="Medium",
        feasibility="high",
        cost_tier="Low",
        timeline_label="6–12 months",
        deployment_note="PM-WANI PDO registration + ISP tie-up required",
    ),

    "device_subsidy": InterventionCostProfile(
        key="device_subsidy",
        cost_per_point_cr=0.90,
        deployment_speed="Fast",
        feasibility="high",
        cost_tier="High",
        timeline_label="3–6 months",
        deployment_note="Jan Dhan DBT channel; assumes 70% BPL uptake",
    ),

    "data_subsidy": InterventionCostProfile(
        key="data_subsidy",
        cost_per_point_cr=0.50,
        deployment_speed="Fast",
        feasibility="high",
        cost_tier="Medium",
        timeline_label="1–3 months",
        deployment_note="Telecom operator MoU + DBT transfer; recurring annual cost",
    ),

    "literacy_programme": InterventionCostProfile(
        key="literacy_programme",
        cost_per_point_cr=0.04,
        deployment_speed="Fast",
        feasibility="high",
        cost_tier="Low",
        timeline_label="3–6 months",
        deployment_note="PMGDISHA / CSC Academy delivery; trainer capacity is main constraint",
    ),

    "women_digital_skilling": InterventionCostProfile(
        key="women_digital_skilling",
        cost_per_point_cr=0.03,
        deployment_speed="Fast",
        feasibility="high",
        cost_tier="Low",
        timeline_label="2–4 months",
        deployment_note="Delivered via existing SHG / Anganwadi infrastructure",
    ),

    "community_internet_centre": InterventionCostProfile(
        key="community_internet_centre",
        cost_per_point_cr=0.15,
        deployment_speed="Medium",
        feasibility="high",
        cost_tier="Medium",
        timeline_label="6–12 months",
        deployment_note="CSC Academy coordination; building/space identification needed",
    ),

    "women_safety_digital": InterventionCostProfile(
        key="women_safety_digital",
        cost_per_point_cr=0.02,
        deployment_speed="Fast",
        feasibility="medium",
        cost_tier="Low",
        timeline_label="2–6 months",
        deployment_note="Leverages 181 helpline; Safe City mission integration required",
    ),

    "shg_microfinance": InterventionCostProfile(
        key="shg_microfinance",
        cost_per_point_cr=0.06,
        deployment_speed="Medium",
        feasibility="high",
        cost_tier="Low",
        timeline_label="6–12 months",
        deployment_note="DAY-NRLM existing SHG network; Udyam registration camp needed",
    ),
}


def get_cost_profile(key: str) -> InterventionCostProfile | None:
    return COST_PROFILES.get(key)


def compute_intervention_cost_cr(key: str, intensity: float) -> float:
    """
    Computes estimated total cost in ₹ Crore for a given intervention
    at a given intensity (0–100).

    Cost = cost_per_point_cr × intensity
    (Linear scaling — cost increases proportionally with deployment scale.)
    """
    profile = COST_PROFILES.get(key)
    if not profile:
        return 0.0
    return round(profile.cost_per_point_cr * intensity, 2)


def compute_portfolio_cost_cr(interventions: dict[str, float]) -> float:
    """Total portfolio cost across all active interventions."""
    return round(sum(
        compute_intervention_cost_cr(k, v)
        for k, v in interventions.items() if v > 0
    ), 2)


def cost_efficiency_label(mci_delta: float, total_cost_cr: float) -> str:
    """
    Returns HIGH / MEDIUM / LOW cost efficiency label.
    Rule: MCI delta per ₹ Crore invested.
      ≥ 1.0 pt/Cr → HIGH
      ≥ 0.3 pt/Cr → MEDIUM
      < 0.3 pt/Cr → LOW
    """
    if total_cost_cr <= 0:
        return "N/A"
    ratio = mci_delta / total_cost_cr
    if ratio >= 1.0:
        return "HIGH"
    elif ratio >= 0.3:
        return "MEDIUM"
    return "LOW"