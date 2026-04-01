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
7. For comparisons between cities/areas, use mci_scores (latest year).
8. Round all decimal scores to 1 decimal place using ROUND(col, 1).
9. If a question cannot be answered from the schema, return:
   {{"sql": "SELECT 'No matching data found' AS message", "chart_type": "none"}}

EXAMPLE OUTPUTS:
User: "Which areas have the lowest MCI?"
Output: {{"sql": "SELECT area, city, ROUND(MCI,1) AS MCI, MCI_class FROM mci_scores ORDER BY MCI ASC LIMIT 10", "chart_type": "bar"}}

User: "Show MCI trend for Mumbai"
Output: {{"sql": "SELECT year, city, ROUND(AVG(MCI),1) AS avg_MCI FROM mci_timeseries_scores WHERE city = 'Mumbai' GROUP BY year, city ORDER BY year ASC", "chart_type": "line"}}

User: "Compare women safety scores across Tier 2 cities"
Output: {{"sql": "SELECT city, ROUND(AVG(WSI),1) AS avg_WSI, ROUND(AVG(WEI),1) AS avg_WEI FROM mci_scores WHERE city_tier = 'Tier 2' GROUP BY city ORDER BY avg_WSI ASC", "chart_type": "bar"}}
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
            timeout=60,
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
    q = question.lower()

    # Trend / timeseries
    if any(w in q for w in ["trend", "over time", "year", "history", "change"]):
        city_match = re.search(r"(?:for|in)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", question)
        where = f"WHERE city = '{city_match.group(1)}'" if city_match else ""
        return {
            "sql": f"SELECT year, city, area, ROUND(MCI,1) AS MCI, ROUND(IFS,1) AS IFS, "
                   f"ROUND(DLS,1) AS DLS, ROUND(SES,1) AS SES, ROUND(WDI,1) AS WDI "
                   f"FROM mci_timeseries_scores {where} ORDER BY year ASC LIMIT 50",
            "chart_type": "line",
        }

    # Lowest / worst
    if any(w in q for w in ["lowest", "worst", "bottom", "least", "minimum"]):
        col = "MCI"
        if "safety" in q or "wsi" in q: col = "WSI"
        elif "employment" in q or "wei" in q: col = "WEI"
        elif "infrastructure" in q or "ifs" in q: col = "IFS"
        elif "literacy" in q or "dls" in q: col = "DLS"
        return {
            "sql": f"SELECT area, city, city_tier, ROUND({col},1) AS {col}, MCI_class "
                   f"FROM mci_scores ORDER BY {col} ASC LIMIT 10",
            "chart_type": "bar",
        }

    # Highest / best
    if any(w in q for w in ["highest", "best", "top", "most", "maximum"]):
        col = "MCI"
        if "safety" in q: col = "WSI"
        elif "employment" in q: col = "WEI"
        return {
            "sql": f"SELECT area, city, city_tier, ROUND({col},1) AS {col}, MCI_class "
                   f"FROM mci_scores ORDER BY {col} DESC LIMIT 10",
            "chart_type": "bar",
        }

    # Tier filter
    tier_match = re.search(r"tier\s*([123])", q)
    if tier_match:
        tier = f"Tier {tier_match.group(1)}"
        return {
            "sql": f"SELECT city, area, ROUND(MCI,1) AS MCI, ROUND(WSI,1) AS WSI, "
                   f"ROUND(WEI,1) AS WEI, MCI_class, cluster_label "
                   f"FROM mci_scores WHERE city_tier = '{tier}' ORDER BY MCI ASC LIMIT 50",
            "chart_type": "table",
        }

    # Desert only
    if "desert" in q:
        return {
            "sql": "SELECT city, area, city_tier, ROUND(MCI,1) AS MCI, "
                   "ROUND(WSI,1) AS WSI, ROUND(WEI,1) AS WEI, MCI_class "
                   "FROM mci_scores WHERE MCI_class IN ('Severe desert','Moderate desert') "
                   "ORDER BY MCI ASC LIMIT 50",
            "chart_type": "bar",
        }

    # Compare / vs
    if any(w in q for w in ["compare", "vs", "versus", "difference"]):
        cities = re.findall(r"\b([A-Z][a-z]{2,})\b", question)
        if cities:
            city_list = ", ".join(f"'{c}'" for c in cities[:3])
            return {
                "sql": f"SELECT city, ROUND(AVG(MCI),1) AS MCI, ROUND(AVG(IFS),1) AS IFS, "
                       f"ROUND(AVG(DLS),1) AS DLS, ROUND(AVG(SES),1) AS SES, "
                       f"ROUND(AVG(WDI),1) AS WDI, ROUND(AVG(WSI),1) AS WSI, "
                       f"ROUND(AVG(WEI),1) AS WEI "
                       f"FROM mci_scores WHERE city IN ({city_list}) GROUP BY city",
                "chart_type": "bar",
            }

    # Default: show all areas summary
    return {
        "sql": "SELECT city, area, city_tier, ROUND(MCI,1) AS MCI, "
               "ROUND(WSI,1) AS WSI, ROUND(WEI,1) AS WEI, MCI_class "
               "FROM mci_scores ORDER BY MCI ASC LIMIT 30",
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