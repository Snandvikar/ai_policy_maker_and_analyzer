"""
agents/data_query_agent.py
===========================
Agent 1 — Data Query Agent (Metrics Retrieval Agent)

Flow:
  User question → Mistral (SQL generation) → DuckDB → formatted result

Model: mistral (via Ollama) — strong instruction-following and SQL generation.
Fallback: keyword-based SQL builder if Ollama is unavailable.

The agent ONLY retrieves data. It never makes recommendations.
"""

from __future__ import annotations
import re
import json
import logging
from typing import Optional

import duckdb
import pandas as pd
import requests

from config import DB_PATH, OLLAMA_BASE_URL, DATA_AGENT_MODEL, DB_SCHEMA_SUMMARY

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

DATA_AGENT_SYSTEM_PROMPT = f"""You are a precise SQL generation agent for the MCI (Minimum Connectivity Index) database.

Your ONLY job is to convert user questions into valid DuckDB SQL queries.

{DB_SCHEMA_SUMMARY}

STRICT RULES:
1. Output ONLY a JSON object with two keys: "sql" and "chart_type".
2. "sql" must be a valid DuckDB SQL query string. Nothing else.
3. "chart_type" must be one of: "table", "bar", "line", "scatter", "none".
   Choose based on what would best display the result.
4. Never explain, never add prose, never add markdown. Only JSON.
5. Always LIMIT results to 50 rows maximum unless the user asks for all.
6. For trend queries, use mci_timeseries_scores and ORDER BY year ASC.
7. For comparisons between districts/states, use mci_scores (latest year).
8. Round all decimal scores to 1 decimal place using ROUND(col, 1).
9. If a question cannot be answered from the given schemas, return:
   {{"sql": "SELECT 'No matching data found' AS message", "chart_type": "none"}}

EXAMPLE OUTPUTS:
User: "Which areas have the lowest MCI?"
Output: {{"sql": "SELECT districtname, statename, ROUND(MCI,1) AS MCI, MCI_class FROM mci_scores ORDER BY MCI ASC LIMIT 10", "chart_type": "bar"}}

User: "Show MCI trend for Maharashtra"
Output:{{"sql": "SELECT year, statename, ROUND(AVG(MCI),1) AS avg_MCI FROM mci_timeseries_scores WHERE LOWER(statename) = LOWER('Maharashtra') GROUP BY year, statename ORDER BY year ASC","chart_type": "line"}}

User: "Compare women safety scores across states"
Output:{{"sql": "SELECT statename, ROUND(AVG(WSI),1) AS avg_WSI, ROUND(AVG(WEI),1) AS avg_WEI FROM mci_scores GROUP BY statename ORDER BY avg_WSI ASC LIMIT 20","chart_type": "bar"}}

"""


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, system: str, model: str) -> Optional[str]:
    """Call Ollama local API. Returns the model response text or None on error."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 512},
            },
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        logger.warning("Ollama not reachable — falling back to keyword SQL builder")
        return None
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK KEYWORD-BASED SQL BUILDER
# Used when Ollama is unavailable. Handles the most common query patterns.
# ─────────────────────────────────────────────────────────────────────────────

def _keyword_sql(question: str) -> dict:
    import re

    q = question.lower()

    # ─────────────────────────────────────────────────────────────
    # TREND / TIME SERIES
    # ─────────────────────────────────────────────────────────────
    if any(w in q for w in ["trend", "over time", "history", "change", "year"]):

        state_match = re.search(
            r"(?:for|in)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
            question
        )

        where = ""
        if state_match:
            state = state_match.group(1)
            where = f"WHERE LOWER(statename) = LOWER('{state}')"

        return {
            "sql":
                f"SELECT year, statename, "
                f"ROUND(AVG(MCI),1) AS avg_MCI, "
                f"ROUND(AVG(IFS),1) AS avg_IFS, "
                f"ROUND(AVG(DLS),1) AS avg_DLS, "
                f"ROUND(AVG(SES),1) AS avg_SES, "
                f"ROUND(AVG(WDI),1) AS avg_WDI "
                f"FROM mci_timeseries_scores "
                f"{where} "
                f"GROUP BY year, statename "
                f"ORDER BY year ASC LIMIT 50",
            "chart_type": "line",
        }

    # ─────────────────────────────────────────────────────────────
    # LOWEST / WORST
    # ─────────────────────────────────────────────────────────────
    if any(w in q for w in ["lowest", "worst", "bottom", "least", "minimum"]):

        col = "MCI"

        if "safety" in q or "wsi" in q:
            col = "WSI"

        elif "employment" in q or "wei" in q:
            col = "WEI"

        elif "infrastructure" in q or "ifs" in q:
            col = "IFS"

        elif "literacy" in q or "dls" in q:
            col = "DLS"

        elif "women" in q or "wdi" in q:
            col = "WDI"

        return {
            "sql":
                f"SELECT districtname, statename, "
                f"ROUND({col},1) AS {col}, "
                f"MCI_class, cluster_label "
                f"FROM mci_scores "
                f"ORDER BY {col} ASC LIMIT 10",
            "chart_type": "bar",
        }

    # ─────────────────────────────────────────────────────────────
    # HIGHEST / BEST
    # ─────────────────────────────────────────────────────────────
    if any(w in q for w in ["highest", "best", "top", "most", "maximum"]):

        col = "MCI"

        if "safety" in q or "wsi" in q:
            col = "WSI"

        elif "employment" in q or "wei" in q:
            col = "WEI"

        elif "women" in q or "wdi" in q:
            col = "WDI"

        elif "infrastructure" in q or "ifs" in q:
            col = "IFS"

        return {
            "sql":
                f"SELECT districtname, statename, "
                f"ROUND({col},1) AS {col}, "
                f"MCI_class, cluster_label "
                f"FROM mci_scores "
                f"ORDER BY {col} DESC LIMIT 10",
            "chart_type": "bar",
        }

    # ─────────────────────────────────────────────────────────────
    # DESERT FILTER
    # ─────────────────────────────────────────────────────────────
    if "desert" in q:

        return {
            "sql":
                "SELECT districtname, statename, "
                "ROUND(MCI,1) AS MCI, "
                "ROUND(WSI,1) AS WSI, "
                "ROUND(WEI,1) AS WEI, "
                "MCI_class "
                "FROM mci_scores "
                "WHERE MCI_class IN "
                "('Severe desert','Moderate desert') "
                "ORDER BY MCI ASC LIMIT 50",
            "chart_type": "bar",
        }

    # ─────────────────────────────────────────────────────────────
    # WOMEN / SAFETY ANALYSIS
    # ─────────────────────────────────────────────────────────────
    if any(w in q for w in [
        "women", "gender", "safety",
        "crime", "vulnerability"
    ]):

        return {
            "sql":
                "SELECT districtname, statename, "
                "ROUND(WDI,1) AS WDI, "
                "ROUND(WSI,1) AS WSI, "
                "ROUND(gender_vulnerability_index,1) AS vulnerability_index, "
                "ROUND(crime_against_women_rate,1) AS crime_rate "
                "FROM mci_scores "
                "ORDER BY WDI ASC LIMIT 20",
            "chart_type": "scatter",
        }

    # ─────────────────────────────────────────────────────────────
    # TELECOM / TELEDENSITY
    # ─────────────────────────────────────────────────────────────
    if any(w in q for w in [
        "teledensity", "wireless",
        "subscriber", "telecom"
    ]):

        return {
            "sql":
                "SELECT state_ut, "
                "ROUND(wireless_teledensity_total_percent,2) "
                "AS wireless_teledensity, "
                "ROUND(wireless_teledensity_rural_percent,2) "
                "AS rural_wireless_teledensity, "
                "ROUND(wireless_teledensity_urban_percent,2) "
                "AS urban_wireless_teledensity "
                "FROM state_ut_wireless_teledensity "
                "ORDER BY wireless_teledensity_total_percent DESC LIMIT 20",
            "chart_type": "bar",
        }

    # ─────────────────────────────────────────────────────────────
    # COMPARE STATES
    # ─────────────────────────────────────────────────────────────
    if any(w in q for w in ["compare", "vs", "versus", "difference"]):

        states = re.findall(
            r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b",
            question
        )

        if states:
            state_list = ", ".join(
                f"'{s}'" for s in states[:5]
            )

            return {
                "sql":
                    f"SELECT statename, "
                    f"ROUND(AVG(MCI),1) AS MCI, "
                    f"ROUND(AVG(IFS),1) AS IFS, "
                    f"ROUND(AVG(DLS),1) AS DLS, "
                    f"ROUND(AVG(SES),1) AS SES, "
                    f"ROUND(AVG(WDI),1) AS WDI, "
                    f"ROUND(AVG(WSI),1) AS WSI, "
                    f"ROUND(AVG(WEI),1) AS WEI "
                    f"FROM mci_scores "
                    f"WHERE statename IN ({state_list}) "
                    f"GROUP BY statename",
                "chart_type": "bar",
            }

    # ─────────────────────────────────────────────────────────────
    # CLUSTER ANALYSIS
    # ─────────────────────────────────────────────────────────────
    if "cluster" in q:

        return {
            "sql":
                "SELECT cluster_label, "
                "COUNT(*) AS district_count, "
                "ROUND(AVG(MCI),1) AS avg_MCI, "
                "ROUND(AVG(IFS),1) AS avg_IFS, "
                "ROUND(AVG(DLS),1) AS avg_DLS, "
                "ROUND(AVG(SES),1) AS avg_SES, "
                "ROUND(AVG(WDI),1) AS avg_WDI "
                "FROM mci_scores "
                "GROUP BY cluster_label "
                "ORDER BY avg_MCI ASC",
            "chart_type": "bar",
        }

    # ─────────────────────────────────────────────────────────────
    # DEFAULT SUMMARY
    # ─────────────────────────────────────────────────────────────
    return {
        "sql":
            "SELECT districtname, statename, "
            "ROUND(MCI,1) AS MCI, "
            "ROUND(WSI,1) AS WSI, "
            "ROUND(WEI,1) AS WEI, "
            "MCI_class, cluster_label "
            "FROM mci_scores "
            "ORDER BY MCI ASC LIMIT 30",
        "chart_type": "table",
    }

# ─────────────────────────────────────────────────────────────────────────────
# SQL EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_sql_and_chart(llm_output: str) -> dict:
    """Parse the JSON from LLM output. Handles partial or messy responses."""
    # Try direct JSON parse
    try:
        obj = json.loads(llm_output.strip())
        if "sql" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block from prose
    match = re.search(r"\{.*?\}", llm_output, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if "sql" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # Try extracting just SQL from code block
    sql_match = re.search(r"```sql\s*(.*?)\s*```", llm_output, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return {"sql": sql_match.group(1).strip(), "chart_type": "table"}

    return {"sql": None, "chart_type": "none"}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DataQueryAgent:
    """
    Converts natural language questions into SQL, executes them,
    and returns structured results.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def _run_sql(self, sql: str) -> tuple[pd.DataFrame, str]:
        """Execute SQL against DuckDB. Returns (dataframe, error_message)."""
        try:
            con = duckdb.connect(self.db_path, read_only=True)
            df = con.execute(sql).df()
            con.close()
            return df, ""
        except Exception as e:
            return pd.DataFrame(), str(e)

    def query(self, user_question: str) -> dict:
        """
        Main entry point.

        Returns:
            {
                "question": str,
                "sql": str,
                "chart_type": str,
                "data": pd.DataFrame,
                "error": str,
                "used_fallback": bool,
            }
        """
        # 1. Try LLM SQL generation
        print(f"DataQueryAgent received question: {user_question}")
        llm_output = _call_ollama(
            prompt=user_question,
            system=DATA_AGENT_SYSTEM_PROMPT,
            model=DATA_AGENT_MODEL,
        )
        used_fallback = False

        if llm_output:
            parsed = _extract_sql_and_chart(llm_output)
            sql = parsed.get("sql")
            chart_type = parsed.get("chart_type", "table")
        else:
            sql = None

        # 2. Fallback to keyword builder if LLM failed
        if not sql:
            fallback = _keyword_sql(user_question)
            sql = fallback["sql"]
            chart_type = fallback["chart_type"]
            used_fallback = True

        # 3. Execute SQL
        df, error = self._run_sql(sql)

        # 4. Safety: if SQL errored, try fallback
        if error and not used_fallback:
            logger.warning(f"SQL error: {error}. Trying fallback.")
            fallback = _keyword_sql(user_question)
            sql = fallback["sql"]
            chart_type = fallback["chart_type"]
            df, error = self._run_sql(sql)
            used_fallback = True

        return {
            "question":     user_question,
            "sql":          sql,
            "chart_type":   chart_type,
            "data":         df,
            "error":        error,
            "used_fallback": used_fallback,
        }

    def get_area_scores(self, city: str, area: str, year: int = None) -> dict:
        """Convenience: fetch all scores for a specific area."""
        year_filter = f"AND year = {year}" if year else ""
        sql = f"""
            SELECT * FROM mci_scores
            WHERE city = '{city}' AND area = '{area}' {year_filter}
            LIMIT 1
        """
        df, _ = self._run_sql(sql)
        if df.empty:
            return {}
        return df.iloc[0].to_dict()