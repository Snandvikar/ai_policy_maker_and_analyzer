"""
agents/orchestrator.py
=======================
Routes user messages to either:
  - DataQueryAgent  (data retrieval, metrics, comparisons)
  - PolicyAgent     (impact analysis, recommendations, what-if)

Intent classification uses lightweight keyword matching first,
then Mistral (same model as DataQueryAgent) for ambiguous cases.

Returns a unified response dict consumed by the chatbot UI.
"""

from __future__ import annotations
import re
import logging
from enum import Enum
from typing import Optional

from agents.data_query_agent import DataQueryAgent
from agents.policy_agent import PolicyAgent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# INTENT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

class Intent(Enum):
    DATA_QUERY  = "data_query"
    POLICY      = "policy"
    AMBIGUOUS   = "ambiguous"


# Keywords that strongly indicate a data retrieval intent
DATA_KEYWORDS = [
    "show", "list", "what is", "what are", "give me", "fetch", "find",
    "which areas", "how many", "count", "rank", "top", "bottom", "lowest",
    "highest", "compare", "vs", "versus", "trend", "over time", "history",
    "score", "mci for", "wsi for", "wei for", "table", "data",
]

# Keywords that strongly indicate a policy/recommendation intent
POLICY_KEYWORDS = [
    "why", "how can", "how to", "what should", "improve", "recommend",
    "suggestion", "action", "intervention", "fix", "address", "help",
    "impact", "effect", "what happens", "if we", "what if", "because",
    "reason", "cause", "explain", "analyse", "analyze", "policy",
    "scheme", "programme", "program", "uplift", "increase", "decrease",
    "reduce", "boost", "strengthen",
]


def classify_intent(question: str) -> Intent:
    q = question.lower()
    data_hits   = sum(1 for kw in DATA_KEYWORDS  if kw in q)
    policy_hits = sum(1 for kw in POLICY_KEYWORDS if kw in q)

    if data_hits > policy_hits:
        return Intent.DATA_QUERY
    elif policy_hits > data_hits:
        return Intent.POLICY
    else:
        # Tie-break: questions starting with "why/how/what should" → policy
        if re.match(r"^(why|how can|how to|what should|what if|explain)", q.strip()):
            return Intent.POLICY
        # Questions starting with "show/list/which/compare" → data
        if re.match(r"^(show|list|which|compare|give|find|what is|what are)", q.strip()):
            return Intent.DATA_QUERY
        return Intent.AMBIGUOUS


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────

def _format_data_response(result: dict) -> dict:
    """Wrap DataQueryAgent output into unified response dict."""
    df = result["data"]
    n  = len(df)

    if result["error"]:
        summary = f"Query error: {result['error']}"
    elif df.empty:
        summary = "No matching data found for your query."
    else:
        summary = f"Found {n} result{'s' if n != 1 else ''}."

    return {
        "agent":      "Data Query Agent",
        "intent":     Intent.DATA_QUERY.value,
        "summary":    summary,
        "data":       df,
        "chart_type": result.get("chart_type", "table"),
        "sql":        result.get("sql", ""),
        "markdown":   None,
        "suggestions": [],
        "error":      result.get("error", ""),
    }


def _format_policy_response(result: dict) -> dict:
    """Wrap PolicyAgent output into unified response dict."""
    return {
        "agent":      "Policy Suggestion Agent",
        "intent":     Intent.POLICY.value,
        "summary":    f"Analysis for {result.get('question','')}",
        "data":       None,
        "chart_type": "none",
        "sql":        "",
        "markdown":   result.get("response", ""),
        "suggestions": result.get("suggestions", []),
        "error":      "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Single entry point for the chatbot UI.
    Accepts a user message + optional area context, routes to correct agent,
    returns a unified response dict.
    """

    def __init__(self, db_path: str):
        self.db_path      = db_path
        self.data_agent   = DataQueryAgent(db_path)
        self.policy_agent = PolicyAgent(db_path)

    def handle(
        self,
        message: str,
        selected_area_scores: Optional[dict] = None,
    ) -> dict:
        """
        Parameters:
            message: user's natural language input
            selected_area_scores: if user has an area selected in the dashboard,
                                  pass its scores here for context injection
                                  into the policy agent.

        Returns unified response dict (see _format_* functions above).
        """
        intent = classify_intent(message)
        logger.info(f"[orchestrator] intent={intent.value} | question='{message[:60]}'")

        # ── Ambiguous: if area is selected, lean toward policy; else data ──
        if intent == Intent.AMBIGUOUS:
            intent = Intent.POLICY if selected_area_scores else Intent.DATA_QUERY

        if intent == Intent.DATA_QUERY:
            result = self.data_agent.query(message)
            return _format_data_response(result)

        else:  # POLICY
            result = self.policy_agent.analyze(
                user_question=message,
                area_scores=selected_area_scores,
            )
            return _format_policy_response(result)