"""
simulation/root_cause.py
=========================
Explains WHY a district is underperforming by attributing its MCI gap
to specific factors and underlying variables.

Uses the subcomponent scores and RF feature importance where available,
otherwise falls back to heuristic attribution from the district's
known weak variables.

Output is policy-readable ranked attribution — not raw ML coefficients.
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class AttributionItem:
    rank:        int
    factor:      str          # IFS / DLS / SES / WDI
    variable:    str          # human-readable variable name
    score:       float        # current score on 0-100 (lower = worse)
    gap:         float        # distance from "connected" threshold (75)
    contribution_pct: float   # share of total MCI gap attributed to this
    policy_lever: str         # plain-language recommendation


# Mapping from internal variable names to policy-readable labels
VARIABLE_LABELS: dict[str, str] = {
    "wireless_rural_teledensity_pct":      "Rural wireless coverage",
    "wireless_total_teledensity_pct":      "Total wireless teledensity",
    "wireless_connectivity_score":         "Wireless quality composite",
    "any_phone_pct":                       "Household phone ownership",
    "mobile_only_pct":                     "Mobile-only access",
    "no_phone_pct":                        "No-phone households",
    "e_transactions_per_1000_population":  "Digital transaction activity",
    "illiteracy_percent":                  "Adult illiteracy rate",
    "gender_vulnerability_index":          "Gender vulnerability index",
    "msme_female_share_pct":               "Women-owned MSME share",
    "dependent_women_percent":             "Women economic dependency",
    "crime_against_women_rate":            "Crime against women rate",
    "non_agri_enterprise_pct":             "Non-agricultural enterprises",
    "income_lt5k_pct":                     "Households income < ₹5k/month",
    "destitute_pct":                       "Destitute households",
    "share_of_slum_population":            "Slum population share",
    "maternal_mortality_rate":             "Maternal mortality rate",
}

# Policy lever for each variable
VARIABLE_LEVERS: dict[str, str] = {
    "wireless_rural_teledensity_pct":      "Deploy towers via BharatNet/USOF in uncovered GPs",
    "wireless_total_teledensity_pct":      "Expand wireless network coverage",
    "wireless_connectivity_score":         "Upgrade BTS fibre backhaul for quality improvement",
    "any_phone_pct":                       "Device subsidy programme for BPL households",
    "mobile_only_pct":                     "PM-WANI public WiFi + device subsidy",
    "no_phone_pct":                        "Device subsidy + digital literacy activation",
    "e_transactions_per_1000_population":  "Data affordability subsidy + digital literacy",
    "illiteracy_percent":                  "PMGDISHA + adult literacy camps at CSCs",
    "gender_vulnerability_index":          "Women-targeted skilling + SHG formation",
    "msme_female_share_pct":               "Women MSME registration + Udyam camps",
    "dependent_women_percent":             "SHG microfinance + livelihood programmes",
    "crime_against_women_rate":            "181 helpline digital access + Safe City Mission",
    "non_agri_enterprise_pct":             "Non-farm livelihood + digital commerce onboarding",
    "income_lt5k_pct":                     "PM-KISAN + NRLM livelihood convergence",
    "destitute_pct":                       "PM-AWAS + Jan Dhan saturation",
    "share_of_slum_population":            "SBM-Urban + digital inclusion in slum clusters",
    "maternal_mortality_rate":             "Telemedicine + digital health service access",
}

# Factor weights in MCI (from mci_pipeline.py)
FACTOR_WEIGHTS = {"IFS": 0.35, "DLS": 0.30, "SES": 0.20, "WDI": 0.15}

# Variable → factor mapping
VAR_FACTOR = {
    "wireless_rural_teledensity_pct":      "IFS",
    "wireless_total_teledensity_pct":      "IFS",
    "wireless_connectivity_score":         "IFS",
    "any_phone_pct":                       "IFS",
    "mobile_only_pct":                     "IFS",
    "no_phone_pct":                        "IFS",
    "e_transactions_per_1000_population":  "IFS",
    "illiteracy_percent":                  "DLS",
    "gender_vulnerability_index":          "DLS",
    "msme_female_share_pct":               "DLS",
    "non_agri_enterprise_pct":             "DLS",
    "crime_against_women_rate":            "WDI",
    "dependent_women_percent":             "WDI",
    "income_lt5k_pct":                     "SES",
    "destitute_pct":                       "SES",
    "share_of_slum_population":            "SES",
    "maternal_mortality_rate":             "SES",
}

# Variables where LOWER score = WORSE (invert for gap calculation)
INVERTED_VARS = {
    "no_phone_pct", "illiteracy_percent", "gender_vulnerability_index",
    "dependent_women_percent", "crime_against_women_rate", "income_lt5k_pct",
    "destitute_pct", "share_of_slum_population", "maternal_mortality_rate",
}

CONNECTED_THRESHOLD = 75.0  # MCI band above which an area is "connected"


def compute_attribution(
    district_row: dict,
    sub_df: pd.DataFrame | None = None,
    imp_df: pd.DataFrame | None = None,
    top_n: int = 6,
) -> list[AttributionItem]:
    """
    Returns a ranked list of AttributionItem explaining why the district
    is underperforming.

    Priority order for attribution:
      1. RF feature importance × variable gap (if imp_df available)
      2. Subcomponent scores (if sub_df available)
      3. Heuristic: raw variable values vs national median proxy
    """
    mci = float(district_row.get("MCI", 50))
    mci_gap = max(0, CONNECTED_THRESHOLD - mci)

    if mci_gap < 1:
        return []  # No significant underperformance

    # ── Build variable score dict from district row ────────────────────────────
    var_scores: dict[str, float] = {}
    for var in VAR_FACTOR:
        val = district_row.get(var)
        if val is not None:
            raw = float(val)
            # For inverted variables: score = 100 - raw (lower raw = worse → lower score)
            score = (100 - raw) if var in INVERTED_VARS else raw
            var_scores[var] = max(0.0, min(100.0, score))

    # ── RF importance weights ─────────────────────────────────────────────────
    importance_weights: dict[str, float] = {}
    if imp_df is not None and not imp_df.empty and "feature" in imp_df.columns:
        for _, r in imp_df.iterrows():
            feat = r.get("feature", "")
            if feat in var_scores:
                importance_weights[feat] = float(r.get("importance", 0))

    # ── Compute weighted gap contribution per variable ─────────────────────────
    items: list[dict] = []
    total_weighted_gap = 0.0

    for var, score in var_scores.items():
        factor = VAR_FACTOR.get(var, "IFS")
        gap = max(0, CONNECTED_THRESHOLD - score)
        factor_weight = FACTOR_WEIGHTS.get(factor, 0.25)
        imp_weight = importance_weights.get(var, 0.05)
        # Weighted contribution = gap × factor_weight × (1 + importance_boost)
        weighted = gap * factor_weight * (1.0 + imp_weight * 5.0)
        total_weighted_gap += weighted
        items.append({
            "var": var, "factor": factor, "score": score,
            "gap": gap, "weighted": weighted,
        })

    # ── Rank and compute contribution percentage ──────────────────────────────
    items.sort(key=lambda x: x["weighted"], reverse=True)
    items = items[:top_n]

    result = []
    for rank, item in enumerate(items, 1):
        contribution_pct = (
            round(item["weighted"] / total_weighted_gap * 100, 1)
            if total_weighted_gap > 0 else 0.0
        )
        result.append(AttributionItem(
            rank=rank,
            factor=item["factor"],
            variable=VARIABLE_LABELS.get(item["var"], item["var"]),
            score=round(item["score"], 1),
            gap=round(item["gap"], 1),
            contribution_pct=contribution_pct,
            policy_lever=VARIABLE_LEVERS.get(item["var"], "Targeted programme needed"),
        ))

    return result


def format_attribution_summary(items: list[AttributionItem], district: str) -> str:
    """Returns a plain-text policy-readable attribution summary."""
    if not items:
        return f"{district} is performing adequately — no significant root cause identified."

    lines = [f"Root cause analysis for {district}:\n"]
    for item in items[:3]:
        lines.append(
            f"  {item.rank}. {item.variable} ({item.factor}) — "
            f"score {item.score:.0f}/100, contributing {item.contribution_pct:.0f}% of gap. "
            f"Lever: {item.policy_lever}."
        )
    return "\n".join(lines)