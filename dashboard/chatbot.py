"""
dashboard/chatbot.py
=====================
Chatbot UI component for the MCI dashboard.
Renders message history, handles input, calls the orchestrator,
and displays data tables + charts from agent responses.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd

from config import DB_PATH
from agents.Orchestrator import Orchestrator
from dashboard.charts import dynamic_chart


AGENT_ICONS = {
    "Data Query Agent":      "🔍",
    "Policy Suggestion Agent": "💡",
}

EXAMPLE_QUERIES = [
    "Which areas have the lowest MCI?",
    "Show MCI trend for Mumbai",
    "Compare women safety scores across Tier 2 cities",
    "Why is this area performing poorly?",
    "What should we improve in the selected area?",
    "Which areas have high safety risk but low employment scores?",
    "Show all severe digital deserts",
]


def _init_chat():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = Orchestrator(db_path=DB_PATH)


def _render_message(msg: dict):
    """Render a single message bubble."""
    role = msg["role"]

    # ---------- USER MESSAGE ----------
    if role == "user":
        with st.container():
            st.markdown("### 🧑 User")
            st.markdown(msg["content"])
        return

    # ---------- ASSISTANT MESSAGE ----------
    result = msg.get("result", {})
    agent  = result.get("agent", "Assistant")
    icon   = AGENT_ICONS.get(agent, "🤖")

    with st.container():
        # Header (replacement for avatar)
        st.markdown(f"### {icon} {agent}")

        # Agent badge
        st.caption(
            f"**{agent}**" + 
            (" _(keyword fallback)_" if result.get("used_fallback") else "")
        )

        intent = result.get("intent", "")
        error  = result.get("error", "")

        if error:
            st.error(f"Error: {error}")
            return

        # ── Data Query response ──────────────────────────────────────────────
        if intent == "data_query":
            df         = result.get("data")
            chart_type = result.get("chart_type", "table")
            sql        = result.get("sql", "")

            st.markdown(result.get("summary", ""))

            if df is not None and not df.empty:
                # Chart (if suggested)
                if chart_type not in ("table", "none"):
                    fig = dynamic_chart(df, chart_type)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                # Data table
                st.dataframe(
                    df,
                    height=min(300, (len(df)+1)*36)
                )

            if sql:
                with st.expander("View SQL"):
                    st.code(sql, language="sql")

        # ── Policy response ──────────────────────────────────────────────────
        elif intent == "policy":
            markdown = result.get("markdown", "")
            if markdown:
                st.markdown(markdown)

            suggestions = result.get("suggestions", [])
            if suggestions:
                st.markdown("---")
                st.markdown("**Rule-based intervention checklist:**")
                for s in suggestions:
                    icons = {1: "🔴", 2: "🟡", 3: "🟢"}
                    st.markdown(
                        f"{icons.get(s.priority,'⚪')} **{s.short_title}**  \n"
                        f"{s.action}"
                    )
                    if s.scheme_or_program:
                        st.caption(f"Scheme: {s.scheme_or_program}")

        else:
            st.markdown(
                result.get("markdown", result.get("summary", ""))
            )


def render_chatbot(selected_area_scores: dict = None):
    """
    Renders the full chatbot UI.

    Parameters:
        selected_area_scores: scores dict of the currently selected area
                              in the dashboard drill-down (for context injection).
    """
    _init_chat()

    st.markdown("---")
    st.markdown("### 💬 Ask the MCI Assistant")

    if selected_area_scores:
        area_label = (f"{selected_area_scores.get('area','')}, "
                      f"{selected_area_scores.get('city','')}")
        st.info(
            f"📍 **Context:** {area_label} is selected. "
            "Policy questions will automatically use this area's scores.",
            icon="ℹ️",
        )

    # Example query chips
    st.caption("Try an example:")
    chip_cols = st.columns(4)
    for i, q in enumerate(EXAMPLE_QUERIES[:4]):
        if chip_cols[i % 4].button(q, key=f"chip_{i}"):
            st.session_state._pending_input = q

    # Render chat history
    for msg in st.session_state.chat_history:
        _render_message(msg)

    # Handle pending input from chips
    pending = st.session_state.pop("_pending_input", None)

    # Chat input
    user_input = st.text_input(
        "Ask about data, trends, or policy recommendations...",
    ) or pending

    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Call orchestrator
        with st.spinner("Thinking..."):
            result = st.session_state.orchestrator.handle(
                message=user_input,
                selected_area_scores=selected_area_scores,
            )

        # Add assistant message to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "result": result,
        })

        st.experimental_rerun()

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()