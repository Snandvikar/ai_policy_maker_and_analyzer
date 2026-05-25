"""
simulation/tradeoff.py
=======================
Tradeoff Intelligence data for each intervention.

Provides a structured comparison of every intervention across 5 dimensions:
  1. Cost           — Low / Medium / High (₹ perspective)
  2. Inclusion Gain — Low / Medium / High (connectivity equity impact)
  3. QoS Gain       — Low / Medium / High (quality-of-service improvement)
  4. Deployment     — Fast / Medium / Slow
  5. Feasibility    — Low / Medium / High

DECISION DOCUMENTATION
=======================
All labels are rule-based assignments. Rationale for each dimension:

INCLUSION GAIN (digital equity focus):
  High  = directly raises access for excluded groups (no-phone HH, women, slums)
           Examples: device subsidy (no-phone HH), women skilling (gender gap)
  Medium = improves access broadly but not specifically targeting excluded groups
           Examples: tower density, public WiFi
  Low   = quality improvement; doesn't expand who can connect
           Examples: fibre backhaul (quality, not coverage), data subsidy (active users)

QoS GAIN (quality of service):
  High  = measurable throughput / latency / reliability improvement
           Examples: fibre backhaul, tower density
  Medium = moderate quality improvement as side effect
           Examples: public WiFi (shared bandwidth), community centres
  Low   = minimal network quality impact; demand/skills intervention
           Examples: device subsidy, digital literacy, SHG microfinance

These two dimensions are intentionally orthogonal — an intervention can have
high inclusion gain but low QoS gain (device subsidy) or vice versa (fibre).
This captures the "supply vs demand" duality of digital exclusion.
"""

from __future__ import annotations
from dataclasses import dataclass

Tier = str   # "Low" | "Medium" | "High" | "Fast" | "Slow"


@dataclass
class TradeoffProfile:
    key:              str
    label:            str
    cost:             Tier
    inclusion_gain:   Tier
    qos_gain:         Tier
    deployment:       Tier   # Fast / Medium / Slow
    feasibility:      Tier
    # One-line impact summary for the tradeoff table tooltip
    impact_summary:   str


TRADEOFF_PROFILES: list[TradeoffProfile] = [
    TradeoffProfile(
        key="tower_density",
        label="Tower Deployment",
        cost="Medium",
        inclusion_gain="Medium",
        qos_gain="High",
        deployment="Slow",
        feasibility="Medium",
        impact_summary="Expands 4G/5G coverage in uncovered GPs; BharatNet-funded",
    ),
    TradeoffProfile(
        key="fiber_backhaul",
        label="Fibre Backhaul",
        cost="High",
        inclusion_gain="Low",
        qos_gain="High",
        deployment="Slow",
        feasibility="Medium",
        impact_summary="Improves speed and latency for existing users; no new coverage",
    ),
    TradeoffProfile(
        key="public_wifi",
        label="Public WiFi (PM-WANI)",
        cost="Low",
        inclusion_gain="High",
        qos_gain="Medium",
        deployment="Medium",
        feasibility="High",
        impact_summary="Shared access in public spaces; high inclusion per rupee invested",
    ),
    TradeoffProfile(
        key="device_subsidy",
        label="Device Subsidy",
        cost="High",
        inclusion_gain="High",
        qos_gain="Low",
        deployment="Fast",
        feasibility="High",
        impact_summary="Eliminates device ownership barrier; highest upfront cost",
    ),
    TradeoffProfile(
        key="data_subsidy",
        label="Data Affordability Subsidy",
        cost="Medium",
        inclusion_gain="High",
        qos_gain="Low",
        deployment="Fast",
        feasibility="High",
        impact_summary="Activates latent demand among phone owners; recurring cost",
    ),
    TradeoffProfile(
        key="literacy_programme",
        label="Digital Literacy Programme",
        cost="Low",
        inclusion_gain="Medium",
        qos_gain="Low",
        deployment="Fast",
        feasibility="High",
        impact_summary="Highest impact per rupee; addresses skills barrier",
    ),
    TradeoffProfile(
        key="women_digital_skilling",
        label="Women Digital Skilling",
        cost="Low",
        inclusion_gain="High",
        qos_gain="Low",
        deployment="Fast",
        feasibility="High",
        impact_summary="Closes gender gap; delivered via SHGs; strong WDI uplift",
    ),
    TradeoffProfile(
        key="community_internet_centre",
        label="Community Internet Centre",
        cost="Medium",
        inclusion_gain="High",
        qos_gain="Medium",
        deployment="Medium",
        feasibility="High",
        impact_summary="CSC-based shared access; drives e-transactions and enterprise activity",
    ),
    TradeoffProfile(
        key="women_safety_digital",
        label="Women's Digital Safety",
        cost="Low",
        inclusion_gain="Medium",
        qos_gain="Low",
        deployment="Fast",
        feasibility="Medium",
        impact_summary="Reduces safety barriers to digital participation; WSI uplift",
    ),
    TradeoffProfile(
        key="shg_microfinance",
        label="SHG Microfinance + Digital",
        cost="Low",
        inclusion_gain="High",
        qos_gain="Low",
        deployment="Medium",
        feasibility="High",
        impact_summary="Economic inclusion + digital onboarding; strong WEI and SES uplift",
    ),
]

TRADEOFF_MAP: dict[str, TradeoffProfile] = {p.key: p for p in TRADEOFF_PROFILES}

# Colour codes for tier labels (used in dashboard rendering)
TIER_COLORS: dict[str, str] = {
    "High":   "#1D9E75",
    "Medium": "#EF9F27",
    "Low":    "#E24B4A",
    "Fast":   "#1D9E75",
    "Slow":   "#E24B4A",
}


def get_tradeoff_table_data(keys: list[str] | None = None) -> list[dict]:
    """
    Returns tradeoff table rows for the given intervention keys.
    If keys is None, returns all interventions.
    """
    profiles = (
        [TRADEOFF_MAP[k] for k in keys if k in TRADEOFF_MAP]
        if keys else TRADEOFF_PROFILES
    )
    return [
        {
            "Intervention":     p.label,
            "Cost":             p.cost,
            "Inclusion Gain":   p.inclusion_gain,
            "QoS Gain":         p.qos_gain,
            "Deployment Speed": p.deployment,
            "Feasibility":      p.feasibility,
            "_key":             p.key,
            "_summary":         p.impact_summary,
        }
        for p in profiles
    ]