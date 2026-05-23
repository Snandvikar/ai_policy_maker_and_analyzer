"""
simulation/copilot.py
======================
AI Policy Copilot for the simulation page.

The copilot receives structured simulation context (not raw data) and
generates policy-readable responses. It does NOT expose ML internals.

Use cases:
  - Explain why a district is underperforming
  - Summarise simulation results as a policy brief
  - Explain confidence and risk in plain language
  - Answer data-specific questions about the selected district
  - Compare intervention tradeoffs

Model: LLaMA 3.1 via Ollama (falls back to template-based responses).
"""

from __future__ import annotations
import logging
from typing import Optional

import requests

from config import OLLAMA_BASE_URL, POLICY_AGENT_MODEL
from simulation.engine import SimulationResult, SOCIAL_OUTCOME_LABELS, OBJECTIVE_MAP
from simulation.root_cause import AttributionItem

logger = logging.getLogger(__name__)

SIMULATION_COPILOT_SYSTEM = """You are a policy intelligence copilot for India's digital inclusion programme.

You are given structured data about a district's connectivity scores, a simulated policy intervention, and projected outcomes. Your job is to explain results clearly to policymakers — not data scientists.

RULES:
- Never mention machine learning, model weights, normalisation, or technical ML terms.
- Always ground explanations in real Indian policy context (BharatNet, PMGDISHA, DAY-NRLM, etc.).
- Be specific about which interventions to prioritise and why.
- Explain confidence and risk in plain language.
- Use rupees, percentages, and lakh/crore units where relevant.
- Keep responses concise — 3-5 sentences per section maximum.
- Always end with one clear priority action.

OUTPUT FORMAT (follow exactly):
**What the simulation shows:**
[1-2 sentences]

**Key drivers of underperformance:**
[2-3 bullet points, plain language]

**Recommended priority action:**
[1 clear, specific action with scheme name]

**Confidence note:**
[1 sentence explaining reliability of projection]

Additionally, 
IF user asks for 'Policy Brief', generate a 1-page policy brief with the above sections, using bullet points and plain language suitable for a district collector.
IF user asks about implementation, answer with detailed feasible phase wise policy implementation steps, mentioning specific schemes and programs where relevant.
example - Phase 0: Planning and Capacity Building, Phase 1: Infrastructure Development (0-6 months), Phase 2: Digital Literacy and Inclusion (6-18 months), Phase 3: Service Delivery and Feedback (18-36 months). Always ground recommendations in the Indian policy context.
"""


def _call_ollama(prompt: str) -> Optional[str]:
    try:
        print("***********************INSIDE AGENT CALL***********************")
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": POLICY_AGENT_MODEL,
                "messages": [
                    {"role": "system", "content": SIMULATION_COPILOT_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 600},
            },
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception as e:
        logger.warning(f"Copilot Ollama call failed: {e}")
        return None


def _build_simulation_context(
    district_row: dict,
    result: SimulationResult,
    attribution: list[AttributionItem],
    objective_key: str,
) -> str:
    district  = district_row.get("districtname", "this district")
    state     = district_row.get("statename", "")
    objective = OBJECTIVE_MAP.get(objective_key)
    obj_label = objective.label if objective else objective_key

    lines = [
        f"DISTRICT: {district}, {state}",
        f"POLICY OBJECTIVE: {obj_label}",
        "",
        "BASELINE SCORES (0-100, higher = better):",
        f"  MCI={result.baseline['MCI']:.1f}  IFS={result.baseline['IFS']:.1f}  "
        f"DLS={result.baseline['DLS']:.1f}  SES={result.baseline['SES']:.1f}  "
        f"WDI={result.baseline['WDI']:.1f}",
        f"  WSI={result.baseline['WSI']:.1f}  WEI={result.baseline['WEI']:.1f}",
        "",
        "SIMULATED SCORES (after interventions):",
        f"  MCI={result.simulated['MCI']:.1f} (+{result.deltas['MCI']:.1f})  "
        f"WSI={result.simulated['WSI']:.1f} (+{result.deltas['WSI']:.1f})  "
        f"WEI={result.simulated['WEI']:.1f} (+{result.deltas['WEI']:.1f})",
        "",
        "ACTIVE INTERVENTIONS:",
    ]
    for k, intensity in result.active_interventions.items():
        lines.append(f"  - {k.replace('_', ' ').title()} at {intensity:.0f}% intensity")

    if attribution:
        lines.append("")
        lines.append("TOP ROOT CAUSES:")
        for a in attribution[:3]:
            lines.append(f"  - {a.variable} (score {a.score:.0f}/100): {a.policy_lever}")

    lines.append("")
    lines.append(f"CONFIDENCE: {result.confidence:.0%}")
    for note in result.confidence_notes:
        lines.append(f"  Note: {note}")

    return "\n".join(lines)


def _template_response(
    district_row: dict,
    result: SimulationResult,
    attribution: list[AttributionItem],
) -> str:
    """Fallback when Ollama is unavailable."""
    district = district_row.get("districtname", "this district")
    mci_base = result.baseline["MCI"]
    mci_sim  = result.simulated["MCI"]
    delta    = result.deltas["MCI"]
    print("***********************INSIDE FALLBACK CALL***********************")


    weakest_factor = min(
        {"IFS", "DLS", "SES", "WDI"},
        key=lambda f: result.baseline.get(f, 50)
    )
    factor_names = {"IFS": "Infrastructure", "DLS": "Digital Literacy",
                    "SES": "Socio-Economic", "WDI": "Women Inclusion"}

    top_cause = attribution[0].variable if attribution else weakest_factor
    top_lever = attribution[0].policy_lever if attribution else "Targeted programme needed"

    n_interventions = len(result.active_interventions)

    lines = [
        f"**What the simulation shows:**  \n"
        f"Applying {n_interventions} intervention(s) in {district} is projected to lift "
        f"MCI from {mci_base:.0f} to {mci_sim:.0f} (+{delta:.0f} pts), "
        f"moving it {_classify_movement(mci_base, mci_sim)}.",
        "",
        "**Key drivers of underperformance:**",
    ]
    for a in attribution[:3]:
        lines.append(f"- {a.variable}: score {a.score:.0f}/100 ({a.contribution_pct:.0f}% of gap)")

    lines += [
        "",
        f"**Recommended priority action:**  \n{top_lever}.",
        "",
        f"**Confidence note:**  \n"
        f"Projection confidence is {result.confidence:.0%}. "
        + (result.confidence_notes[0] if result.confidence_notes else ""),
    ]
    return "\n".join(lines)


def _classify_movement(baseline: float, simulated: float) -> str:
    if baseline < 25 and simulated >= 25:
        return "from Severe Desert to Moderate Desert band"
    elif baseline < 45 and simulated >= 45:
        return "out of the Digital Desert band"
    elif baseline < 60 and simulated >= 60:
        return "to Partial Connectivity"
    elif baseline < 75 and simulated >= 75:
        return "to Connected status"
    elif simulated > baseline:
        return f"upward within the same band (+{simulated - baseline:.0f} pts)"
    return "with minimal change"


class SimulationCopilot:
    """Stateless copilot — call explain() or ask() per request."""

    def explain_simulation(
        self,
        district_row: dict,
        result: SimulationResult,
        attribution: list[AttributionItem],
        objective_key: str,
    ) -> str:
        context = _build_simulation_context(
            district_row, result, attribution, objective_key
        )
        prompt = (
            f"Explain the following simulation results to a policymaker "
            f"in plain, actionable language:\n\n{context}"
        )
        llm_response = _call_ollama(prompt)
        return llm_response or _template_response(district_row, result, attribution)

    def generate_policy_brief(
        self,
        district_row: dict,
        result: SimulationResult,
        attribution: list[AttributionItem],
        objective_key: str,
    ) -> str:
        context = _build_simulation_context(
            district_row, result, attribution, objective_key
        )
        prompt = (
            "Generate a 1-page policy brief (plain language, bullet points) "
            "for a district collector / policymaker based on this simulation:\n\n"
            + context
            + "\n\nInclude: situation overview, priority interventions, "
              "expected outcomes, and one clear next step."
        )
        llm_response = _call_ollama(prompt)
        return llm_response or _template_response(district_row, result, attribution)

    def answer_question(
        self,
        question: str,
        district_row: dict,
        result: Optional[SimulationResult],
        attribution: list[AttributionItem],
    ) -> str:
        context = ""
        if result:
            context = _build_simulation_context(
                district_row, result, attribution, "general_connectivity"
            )
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        llm_response = _call_ollama(prompt)
        if llm_response:
            return llm_response
        # Simple fallback
        district = district_row.get("districtname", "this district")
        mci = float(district_row.get("MCI", 50))
        return (
            f"Based on available data, {district} has an MCI of {mci:.0f}. "
            f"I cannot provide a more detailed answer without the AI model being available. "
            f"Please check the simulation results panel for detailed projections."
        )