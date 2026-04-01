"""
policy_engine/rules.py
=======================
Rule-Based Policy Suggestion Engine.

Given an area's factor scores, classification, and cluster profile,
returns 2–3 prioritised, specific, actionable suggestions.

Design:
  - Each rule is a dict with: condition (lambda), priority, suggestion text.
  - Rules are evaluated in priority order; top 3 are returned.
  - Rules are grouped by factor so suggestions are always grounded
    in the weakest underlying dimension.
  - No LLM required — deterministic, auditable, fast.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class PolicySuggestion:
    priority: int
    category: str          # IFS / DLS / SES / WDI / WSI / WEI
    short_title: str
    action: str
    rationale: str
    scheme_or_program: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# RULE DEFINITIONS
# Each rule: (priority, category, short_title, action, rationale, scheme)
# Condition: lambda that takes the area_scores dict and returns True/False
# ─────────────────────────────────────────────────────────────────────────────

RULES = [

    # ── INFRASTRUCTURE (IFS) ─────────────────────────────────────────────────
    {
        "condition":   lambda s: s.get("IFS", 100) < 25,
        "priority":    1,
        "category":    "IFS",
        "short_title": "Emergency tower deployment",
        "action":      "Apply for USOF funding under PM-WANI and BharatNet Phase-III. "
                       "Mandate tele-density targets for all TSPs in this district.",
        "rationale":   "IFS is critically low (below 25). The area lacks basic physical "
                       "connectivity infrastructure — no other intervention is effective "
                       "until towers and fibre reach the area.",
        "scheme":      "BharatNet Phase-III / USOF / PM-WANI",
    },
    {
        "condition":   lambda s: 25 <= s.get("IFS", 100) < 45,
        "priority":    2,
        "category":    "IFS",
        "short_title": "OFC fiberisation push",
        "action":      "Prioritise BTS fiberisation for existing towers. "
                       "Target 70% OFC fiberisation within 18 months. "
                       "Engage Jio/Airtel for fibre-sharing agreements.",
        "rationale":   "IFS is in the moderate desert band. Towers exist but "
                       "backhaul quality is poor — fiberising existing BTS is the "
                       "fastest quality improvement lever.",
        "scheme":      "DoT OFC Fiberisation Drive / NIP",
    },
    {
        "condition":   lambda s: s.get("IFS", 100) < 50 and s.get("DLS", 100) > 55,
        "priority":    1,
        "category":    "IFS",
        "short_title": "High-readiness infrastructure gap",
        "action":      "This area has digitally ready users but insufficient infrastructure. "
                       "Fast-track 4G/5G tower rollout and FTTH last-mile connections. "
                       "Flag to TRAI for Priority Coverage Zone status.",
        "rationale":   "DLS is above 55 but IFS is below 50 — the population has skills "
                       "and demand but no connectivity. Infrastructure is the sole binding "
                       "constraint, making ROI on tower investment very high here.",
        "scheme":      "TRAI Priority Coverage Zone / Universal Service Obligation",
    },

    # ── DIGITAL LITERACY (DLS) ───────────────────────────────────────────────
    {
        "condition":   lambda s: s.get("DLS", 100) < 30,
        "priority":    1,
        "category":    "DLS",
        "short_title": "Mass digital literacy mobilisation",
        "action":      "Deploy Common Service Centres (CSCs) as digital skilling hubs. "
                       "Enrol residents in PMGDISHA batches. Target 1 trained member per household.",
        "rationale":   "DLS is severely low (below 30). Even if infrastructure improves, "
                       "residents cannot use it. Demand-side skilling must run in parallel.",
        "scheme":      "PMGDISHA / CSC Academy / DigiSaksharta",
    },
    {
        "condition":   lambda s: s.get("DLS", 100) < 50 and s.get("gender_gap_pp", 0) > 20,
        "priority":    1,
        "category":    "DLS",
        "short_title": "Close the gender digital divide",
        "action":      "Run women-only digital literacy batches at Anganwadi and SHG meeting points. "
                       "Distribute subsidised smartphones to women below the poverty line "
                       "under state scheme.",
        "rationale":   "Gender gap in internet access exceeds 20 percentage points. "
                       "General literacy programmes will not close this — targeted "
                       "women-only interventions are required.",
        "scheme":      "PMGDISHA Women Batches / Mahila E-Haat / State ICT schemes",
    },
    {
        "condition":   lambda s: s.get("DLS", 100) < 55 and s.get("school_digital_percent", 100) < 40,
        "priority":    2,
        "category":    "DLS",
        "short_title": "School digital infrastructure upgrade",
        "action":      "Prioritise PM eVIDYA rollout in schools of this area. "
                       "Install smart classrooms and broadband in all government schools. "
                       "Train teachers in digital pedagogy.",
        "rationale":   "Low school digital access (<40%) means the next generation is also "
                       "being left behind. School-level intervention has the highest "
                       "long-term multiplier.",
        "scheme":      "PM eVIDYA / Smart Classrooms Mission",
    },

    # ── SOCIO-ECONOMIC (SES) ─────────────────────────────────────────────────
    {
        "condition":   lambda s: s.get("SES", 100) < 30,
        "priority":    1,
        "category":    "SES",
        "short_title": "Foundational welfare convergence",
        "action":      "Saturate the area with Jan Dhan account opening camps, "
                       "PMAY housing enrolment drives, and PM Poshan coverage checks. "
                       "Assign a nodal officer for convergence coordination.",
        "rationale":   "SES is critically low. Digital inclusion cannot succeed if "
                       "residents lack basic food, shelter, and financial security. "
                       "Foundational welfare must be the first pillar.",
        "scheme":      "Jan Dhan Yojana / PMAY / PM Poshan",
    },
    {
        "condition":   lambda s: s.get("SES", 100) < 50 and s.get("female_lfpr_percent", 100) < 25,
        "priority":    2,
        "category":    "SES",
        "short_title": "Women's livelihood activation",
        "action":      "Form new SHGs targeting women not currently in employment. "
                       "Link SHGs to NRLM revolving funds and micro-credit. "
                       "Register informal women workers on e-Shram portal.",
        "rationale":   "Female LFPR is below 25% — significantly below national average. "
                       "SHG formation has demonstrated uplift in LFPR within 18–24 months "
                       "in comparable districts.",
        "scheme":      "DAY-NRLM / e-Shram / PM SVANidhi",
    },
    {
        "condition":   lambda s: s.get("SES", 100) < 55 and s.get("girls_school_dropout_percent", 0) > 15,
        "priority":    2,
        "category":    "SES",
        "short_title": "Girls' education retention",
        "action":      "Identify dropout-risk girls through school attendance monitoring. "
                       "Activate conditional cash transfers and Sukanya Samriddhi accounts. "
                       "Build toilet facilities if missing (linked to dropout rates).",
        "rationale":   "Girls' school dropout above 15% perpetuates the next-generation "
                       "gender gap in literacy and digital access. "
                       "Education retention is the highest-leverage long-term lever.",
        "scheme":      "Sukanya Samriddhi Yojana / KGBV / Beti Bachao Beti Padhao",
    },

    # ── WOMEN DIGITAL INCLUSION (WDI) ────────────────────────────────────────
    {
        "condition":   lambda s: s.get("WDI", 100) < 30,
        "priority":    1,
        "category":    "WDI",
        "short_title": "Women's digital access emergency",
        "action":      "Immediately enrol all women SHG members in 10-day digital literacy camps. "
                       "Subsidise SIM cards and entry-level smartphones. "
                       "Deploy women Digital Sakhis in every ward.",
        "rationale":   "WDI below 30 signals near-total digital exclusion of women. "
                       "This directly correlates with safety risk and employment gap in your data.",
        "scheme":      "Digital Sakhi Programme / PMGDISHA / Mahila E-Haat",
    },
    {
        "condition":   lambda s: s.get("WDI", 100) < 50 and s.get("women_skill_percent", 100) < 20,
        "priority":    1,
        "category":    "WDI",
        "short_title": "Women's skill acceleration",
        "action":      "Partner with NASSCOM Foundation and ISRO's BISAG for women-specific "
                       "digital vocational training. Focus on gig-economy-ready skills: "
                       "data entry, online retail, digital payments.",
        "rationale":   "Women's digital skill rate below 20% is the single strongest "
                       "predictor of low WEI in this dataset (RF importance: 10.8%). "
                       "Skills training has faster measurable impact than infrastructure.",
        "scheme":      "Skill India Digital / NASSCOM Foundation / DigiDhan Mission",
    },

    # ── WOMEN SAFETY (WSI) ───────────────────────────────────────────────────
    {
        "condition":   lambda s: s.get("WSI", 100) < 35,
        "priority":    1,
        "category":    "WSI",
        "short_title": "Safety infrastructure priority",
        "action":      "Install 181 Women Helpline awareness boards and CSC-based helpline access points. "
                       "Ensure electricity supply continuity (SAIDI < 4 hrs/yr). "
                       "Activate CCTV in public spaces under Safe City Mission.",
        "rationale":   "WSI below 35 (high safety risk) combined with low MCI means women "
                       "cannot access digital help channels in an emergency. "
                       "Physical safety infrastructure must precede digital inclusion.",
        "scheme":      "Nirbhaya Fund / Safe City Mission / 181 Helpline",
    },
    {
        "condition":   lambda s: s.get("WSI", 100) < 55 and s.get("sanitation_coverage_percent", 100) < 50,
        "priority":    2,
        "category":    "WSI",
        "short_title": "Sanitation for safety",
        "action":      "Saturate Individual Household Latrines (IHHL) construction under SBM-G. "
                       "Poor sanitation coverage (<50%) is a direct proxy for women's "
                       "outdoor safety risk — prioritise this district in WASH planning.",
        "rationale":   "Sanitation coverage below 50% directly increases outdoor exposure "
                       "for women, correlated with assault risk in NCRB data.",
        "scheme":      "Swachh Bharat Mission (Grameen) / WASH",
    },

    # ── WOMEN EMPLOYMENT (WEI) ───────────────────────────────────────────────
    {
        "condition":   lambda s: s.get("WEI", 100) < 40,
        "priority":    1,
        "category":    "WEI",
        "short_title": "Digital livelihood activation",
        "action":      "Register all SHG-member products on GeM portal and Meesho. "
                       "Set up one Udyam registration camp per month. "
                       "Train women in home-based online business models.",
        "rationale":   "WEI below 40 (critical employment gap). "
                       "Digital commerce platforms are the fastest employment pathway "
                       "for women already connected but not yet earning digitally.",
        "scheme":      "GeM Women Sellers / PM SVANidhi / Udyam Registration",
    },
    {
        "condition":   lambda s: s.get("WEI", 100) < 60 and s.get("female_digital_skill_percent", 100) < 30,
        "priority":    2,
        "category":    "WEI",
        "short_title": "Skill-to-employment pipeline",
        "action":      "Set up placement-linked digital skilling centres targeting women. "
                       "Partner with gig platforms (Urban Company, Swiggy Instamart) "
                       "for direct absorption post-training.",
        "rationale":   "Female digital skill rate below 30% while WEI is moderate "
                       "indicates skills are the missing link in the employment pipeline — "
                       "not jobs, not connectivity.",
        "scheme":      "Skill India Digital / Gig Economy Integration MoU",
    },

    # ── CROSS-CUTTING / COMPOSITE ─────────────────────────────────────────────
    {
        "condition":   lambda s: s.get("MCI", 100) < 25,
        "priority":    1,
        "category":    "IFS",
        "short_title": "Declare as Digital Desert — mobilise multi-department task force",
        "action":      "Formally notify the area as a Severe Digital Desert under state ICT policy. "
                       "Constitute a multi-department convergence team (DoT, MeitY, WCD, Education). "
                       "Set quarterly MCI improvement targets with departmental accountability.",
        "rationale":   "MCI below 25 indicates systemic failure across all dimensions. "
                       "No single-department intervention is sufficient — only coordinated "
                       "multi-pillar action with political accountability will move the needle.",
        "scheme":      "State ICT Policy / Digital India Mission Convergence",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def get_suggestions(area_scores: dict, n: int = 3) -> list[PolicySuggestion]:
    """
    area_scores: dict with keys like IFS, DLS, SES, WDI, MCI, WSI, WEI,
                 and optionally raw sub-variables like gender_gap_pp,
                 girls_school_dropout_percent, etc.

    Returns top-n PolicySuggestion objects ordered by:
      1. Priority (lower = more urgent)
      2. Severity of the underlying score (lower score = higher urgency)
    """
    triggered = []
    for rule in RULES:
        try:
            if rule["condition"](area_scores):
                triggered.append(PolicySuggestion(
                    priority=rule["priority"],
                    category=rule["category"],
                    short_title=rule["short_title"],
                    action=rule["action"],
                    rationale=rule["rationale"],
                    scheme_or_program=rule.get("scheme"),
                ))
        except Exception:
            continue

    # Sort: priority first, then by the relevant factor score ascending
    def sort_key(s: PolicySuggestion):
        score = area_scores.get(s.category, 50)
        return (s.priority, score)

    triggered.sort(key=sort_key)

    # Deduplicate by category (keep only the most urgent per category)
    seen_categories = set()
    deduped = []
    for s in triggered:
        if s.category not in seen_categories:
            deduped.append(s)
            seen_categories.add(s.category)

    return deduped[:n]


def format_suggestions_markdown(suggestions: list[PolicySuggestion]) -> str:
    """Returns a markdown string suitable for rendering in Streamlit."""
    if not suggestions:
        return "_No specific interventions triggered. Area is performing adequately._"

    lines = []
    icons = {1: "🔴", 2: "🟡", 3: "🟢"}
    cat_icons = {
        "IFS": "📡", "DLS": "💻", "SES": "🏛️",
        "WDI": "👩", "WSI": "🛡️", "WEI": "💼",
    }
    for i, s in enumerate(suggestions, 1):
        icon = icons.get(s.priority, "⚪")
        cat_icon = cat_icons.get(s.category, "")
        lines.append(
            f"**{icon} {i}. {s.short_title}** {cat_icon}\n\n"
            f"{s.action}\n\n"
            f"_Rationale: {s.rationale}_"
        )
        if s.scheme_or_program:
            lines.append(f"📋 **Scheme:** {s.scheme_or_program}")
        lines.append("---")
    return "\n\n".join(lines)