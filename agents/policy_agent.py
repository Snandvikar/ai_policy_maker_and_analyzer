"""
agents/policy_agent.py
=======================
Agent 2 — Policy Suggestion Agent (Impact + Recommendations)

Flow:
  User question → context assembly (scores + cluster + trend + RF importance)
               → LLaMA 3.1 (reasoning) → structured response

Model: llama3.1 (via Ollama) — stronger reasoning, longer context.
Fallback: rule-based engine (policy_engine/rules.py) if Ollama unavailable.

This agent NEVER runs SQL directly. It receives context assembled by the
orchestrator and reasons over it to generate impact analysis + recommendations.
"""

from __future__ import annotations
import logging
from typing import Optional
from vector_db.retriever import retrieve_policy_context

import requests

from config import (
    DB_PATH,
    POLICY_AGENT_MODEL,
    FACTOR_LABELS,
    CLASS_COLORS,
    GROQ_API_KEY,
)
from policy_engine.rules import get_suggestions, format_suggestions_markdown
from agents.data_query_agent import DataQueryAgent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

POLICY_AGENT_SYSTEM_PROMPT = """You are an expert policy advisor for India's digital inclusion and women's empowerment programmes.

You are given structured data about a geographic area's Meaningful Connectivity Index (MCI) scores and women's impact indicators. Your role is to:
1. Explain WHY the area is performing poorly (root cause analysis)
2. Identify which interventions will have the highest impact
3. Quantify the expected improvement in plain language
4. Suggest specific Indian government schemes by name

SCORE INTERPRETATION:
- 0–25: Severely deprived (critical intervention needed immediately)
- 26–45: Moderately deprived (moderate desert — targeted intervention)
- 46–60: Partial connectivity (one or two factors holding it back)
- 61–75: Near-connected (fine-tuning needed)
- 76–100: Connected (benchmark area)

FACTOR MEANINGS:
- IFS (Infrastructure): Physical towers, fibre, speed, latency
- DLS (Digital Literacy): Household internet, skills, gender gap in access
- SES (Socio-Economic): HDI, poverty, women's education and economic participation
- WDI (Women Digital Inclusion): Women-specific internet, mobile, skill rates
- WSI (Women Safety Index): Safety conditions including crime, sanitation, electricity
- WEI (Women Employment Index): Female LFPR, digital skills, micro-credit, gig work

WRITING STYLE:
- Be specific and actionable — name exact schemes (BharatNet, PMGDISHA, Jan Dhan, etc.)
- Be concise — 3–5 sentences per recommendation maximum
- Always ground recommendations in the actual scores provided
- If time-series data is present, note the trend (improving / worsening / stagnant)
- Do NOT make up data points not in the context
- End with a 1-sentence priority statement

OUTPUT FORMAT (always follow this):
**Root cause analysis:**
[2–3 sentences explaining why the area is low]

**Top recommendations:**
1. [Specific action with scheme name]
2. [Specific action with scheme name]
3. [Specific action with scheme name]

**Expected impact:**
[1–2 sentences on what improvement is realistic in 12–24 months]

**Priority:**
[1 sentence on the single most important first step]

**Priority order of information to use from context (if relevant to the question):**
1. Area Scores (PRIMARY)
2. Cluster Context (PRIMARY)
3. Trend Analysis (PRIMARY)
4. RF Feature Importance (PRIMARY)
5. Reference Policies (SECONDARY — 25%)
"""


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT ASSEMBLER
# ─────────────────────────────────────────────────────────────────────────────

def _assemble_context(
    area_scores: dict,
    cluster_info: Optional[dict],
    trend_data: Optional[list],
    top_features: Optional[list],
) -> str:
    """
    Build the context block injected into the policy agent prompt.
    """
    lines = []

    # Basic identity
    city  = area_scores.get("city", "Unknown")
    area  = area_scores.get("area", "Unknown")
    tier  = area_scores.get("city_tier", area_scores.get("area_type", "Unknown"))
    year  = area_scores.get("year", "latest")

    lines.append(f"AREA: {area}, {city} ({tier}) — data year: {year}")
    lines.append("")

    # Factor scores
    lines.append("FACTOR SCORES (0–100, higher = better):")
    for col, label in FACTOR_LABELS.items():
        val = area_scores.get(col)
        if val is not None:
            band = ("critical" if val < 25 else "moderate desert" if val < 45
                    else "partial" if val < 60 else "near-connected" if val < 75 else "connected")
            lines.append(f"  {label} ({col}): {val:.1f}  [{band}]")

    lines.append("")
    lines.append("WOMEN IMPACT INDICATORS:")
    for col, label in [("MCI","MCI"), ("WSI","Women Safety Index"),
                       ("WEI","Women Employment Index")]:
        val = area_scores.get(col)
        if val is not None:
            lines.append(f"  {label}: {val:.1f}")

    if area_scores.get("Safety_risk"):
        lines.append(f"  Safety classification: {area_scores['Safety_risk']}")
    if area_scores.get("Employment_gap"):
        lines.append(f"  Employment classification: {area_scores['Employment_gap']}")

    lines.append("")

    # Key raw variables if present
    key_vars = [
        ("gender_gap_pp",            "Gender gap in internet access (pp)"),
        ("female_lfpr_percent",      "Female LFPR (%)"),
        ("women_skill_percent",      "Women with digital skills (%)"),
        ("girls_school_dropout_percent", "Girls school dropout rate (%)"),
        ("poverty_rate_percent",     "Poverty rate (%)"),
        ("electricity_hrs_per_day",  "Electricity availability (hrs/day)"),
        ("crime_against_women_rate_per_lakh_women", "Crime against women (per lakh)"),
    ]
    raw_lines = []
    for col, label in key_vars:
        val = area_scores.get(col)
        if val is not None:
            raw_lines.append(f"  {label}: {val:.1f}")
    if raw_lines:
        lines.append("KEY UNDERLYING VARIABLES:")
        lines.extend(raw_lines)
        lines.append("")

    # Cluster profile
    if cluster_info:
        lines.append(f"CLUSTER PROFILE: {cluster_info.get('label', 'Unknown')}")
        lines.append(f"  Weakest factor in cluster: {cluster_info.get('weakest_factor', 'N/A')}")
        lines.append(f"  Recommended intervention: {cluster_info.get('intervention', 'N/A')}")
        lines.append("")

    # Historical trend
    if trend_data and len(trend_data) > 1:
        lines.append("HISTORICAL TREND (MCI over years):")
        for row in trend_data:
            lines.append(f"  {row.get('year')}: MCI={row.get('MCI', 'N/A'):.1f}")
        first = trend_data[0].get("MCI", 0)
        last  = trend_data[-1].get("MCI", 0)
        direction = "improving" if last > first else "worsening" if last < first else "stagnant"
        lines.append(f"  Trend: {direction} ({first:.1f} → {last:.1f})")
        lines.append("")

    # RF feature importance context
    if top_features:
        lines.append("TOP MCI LEVERS (RF feature importance across all areas):")
        for feat in top_features[:5]:
            lines.append(f"  {feat.get('feature')}: {feat.get('pct',.0):.1f}% importance")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA CALL
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq_policy(messages: list[dict], max_tokens: int = 1024, model: str | None = None) -> Optional[str]:
    if not GROQ_API_KEY:
        return None

    model = model or POLICY_AGENT_MODEL
    url = "https://api.groq.com/openai/v1/chat/completions"
    logger.info("Groq policy request: url=%s model=%s message_count=%d", url, model, len(messages))
    resp = None
    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": max_tokens,
            },
            timeout=600,
        )
        resp.raise_for_status()
        logger.info("Groq policy response success: url=%s model=%s status=%s", url, model, resp.status_code)
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        response_text = resp.text if resp is not None else "<no response>"
        logger.warning(
            "Groq policy API error: %s | url=%s | model=%s | status=%s | response=%s",
            e,
            url,
            model,
            getattr(resp, "status_code", "<no-status>"),
            response_text,
        )
        return None


def _call_policy_llm(context: str, user_question: str) -> Optional[str]:
    prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION: {user_question}"
    messages = [
        {"role": "system", "content": POLICY_AGENT_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    return _call_groq_policy(messages, model=POLICY_AGENT_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class PolicyAgent:
    """
    Generates impact analysis and policy recommendations.
    Uses Groq hosted chat; falls back to rule-based engine.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path   = db_path
        self.dq_agent  = DataQueryAgent(db_path)

    def _get_cluster_info(self, cluster_id) -> Optional[dict]:
        if cluster_id is None:
            return None
        try:
            import duckdb
            con = duckdb.connect(self.db_path, read_only=True)
            df  = con.execute(
                f"SELECT * FROM cluster_profiles WHERE cluster_id = {int(cluster_id)} LIMIT 1"
            ).df()
            con.close()
            return df.iloc[0].to_dict() if not df.empty else None
        except Exception:
            return None

    def _get_trend(self, city: str, area: str) -> list:
        try:
            import duckdb
            con = duckdb.connect(self.db_path, read_only=True)
            df  = con.execute(f"""
                SELECT year, ROUND(MCI,1) AS MCI, ROUND(IFS,1) AS IFS,
                       ROUND(DLS,1) AS DLS, ROUND(SES,1) AS SES,
                       ROUND(WDI,1) AS WDI, ROUND(WSI,1) AS WSI, ROUND(WEI,1) AS WEI
                FROM mci_timeseries_scores
                WHERE city = '{city}' AND area = '{area}'
                ORDER BY year ASC
            """).df()
            con.close()
            return df.to_dict("records")
        except Exception:
            return []

    def _get_top_features(self) -> list:
        try:
            import duckdb
            con = duckdb.connect(self.db_path, read_only=True)
            df  = con.execute(
                "SELECT feature, pct FROM rf_feature_importance ORDER BY rank ASC LIMIT 8"
            ).df()
            con.close()
            return df.to_dict("records")
        except Exception:
            return []

    def analyze(
        self,
        user_question: str,
        area_scores: Optional[dict] = None,
        city: Optional[str] = None,
        area_name: Optional[str] = None,
    ) -> dict:
        """
        Main entry point.

        Parameters:
            user_question: what the user asked
            area_scores: pre-loaded score dict (if called from dashboard drill-down)
            city / area_name: if area_scores not provided, fetch from DB

        Returns:
            {
                "question": str,
                "response": str (markdown),
                "suggestions": list[PolicySuggestion],
                "used_llm": bool,
                "context": str,
            }
        """
        # Resolve area_scores
        if area_scores is None and city and area_name:
            area_scores = self.dq_agent.get_area_scores(city, area_name)

        if not area_scores:
            return {
                "question":    user_question,
                "response":    "I need an area to analyse. Please select an area in the dashboard or specify a city and area name.",
                "suggestions": [],
                "used_llm":    False,
                "context":     "",
            }

        # Assemble context
        cluster_id   = area_scores.get("cluster_id")
        cluster_info = self._get_cluster_info(cluster_id)
        trend_data   = self._get_trend(
            area_scores.get("city", ""), area_scores.get("area", "")
        )
        top_features = self._get_top_features()

        policy_context = retrieve_policy_context(user_question)

        context = _assemble_context(
            area_scores, cluster_info, trend_data, top_features
        )

        context += """ REFERENCE POLICY EXAMPLES (Use as inspiration only. Do NOT rely heavily):"""
        context += policy_context

        # Rule-based suggestions (always computed — used as fallback or supplement)
        rule_suggestions = get_suggestions(area_scores)

        # Try LLM
        llm_response = _call_policy_llm(context, user_question)
        used_llm = llm_response is not None

        if not llm_response:
            # Build a structured fallback from rule suggestions
            area_label = f"{area_scores.get('area','')}, {area_scores.get('city','')}"
            mci = area_scores.get("MCI", 0)
            weakest = min(
                {k: area_scores.get(k, 50) for k in ["IFS","DLS","SES","WDI"]},
                key=lambda k: area_scores.get(k, 50)
            )
            llm_response = (
                f"**Root cause analysis:**\n"
                f"{area_label} has an MCI of {mci:.1f}, placing it in the "
                f"'{area_scores.get('MCI_class','unknown')}' band. "
                f"The weakest factor is {FACTOR_LABELS.get(weakest, weakest)} ({weakest}={area_scores.get(weakest,0):.1f}), "
                f"which is the primary driver of low connectivity and women's impact scores.\n\n"
                f"**Top recommendations:**\n"
                f"{format_suggestions_markdown(rule_suggestions)}\n\n"
                f"**Priority:**\n"
                f"Address {FACTOR_LABELS.get(weakest, weakest)} first — "
                f"it is the binding constraint for all other improvements in this area."
            )

        return {
            "question":    user_question,
            "response":    llm_response,
            "suggestions": rule_suggestions,
            "used_llm":    used_llm,
            "context":     context,
        }