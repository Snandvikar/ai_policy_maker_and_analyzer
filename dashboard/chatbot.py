"""
dashboard/chatbot.py
=====================
Chatbot UI component.

Schema alignment:
  Area context uses districtname + canonical_state (not city + area).
  Example queries updated to district/state level language.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd

from config import DB_PATH
from agents.Orchestrator import Orchestrator
from dashboard.charts import dynamic_chart


AGENT_ICONS = {
    "Data Query Agent":       "🔍",
    "Policy Suggestion Agent": "💡",
}

EXAMPLE_QUERIES = [
    "Which districts have the lowest MCI?",
    "Show MCI trend for districts in Maharashtra",
    "Compare women safety scores across states",
    "Why is this district performing poorly?",
    "What should we improve in the selected district?",
    "Which districts have high safety risk and low employment?",
    "Show all severe digital deserts",
]


def _init_chat():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = Orchestrator(db_path=DB_PATH)


def _render_message(msg: dict):
    role = msg["role"]
    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
        return

    result = msg.get("result", {})
    agent  = result.get("agent", "Assistant")
    icon   = AGENT_ICONS.get(agent, "🤖")

    with st.chat_message("assistant", avatar=icon):
        st.caption(
            f"**{agent}**"
            + (" _(keyword fallback)_" if result.get("used_fallback") else "")
        )

        intent = result.get("intent", "")
        error  = result.get("error", "")

        if error:
            st.error(f"Error: {error}")
            return

        if intent == "data_query":
            df         = result.get("data")
            chart_type = result.get("chart_type", "table")
            sql        = result.get("sql", "")

            st.markdown(result.get("summary", ""))

            if df is not None and not df.empty:
                if chart_type not in ("table", "none"):
                    fig = dynamic_chart(df, chart_type)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    df, use_container_width=True,
                    height=min(300, (len(df) + 1) * 36),
                )

            if sql:
                with st.expander("View SQL"):
                    st.code(sql, language="sql")

        elif intent == "policy":
            markdown = result.get("markdown", "")
            if markdown:
                st.markdown(markdown)

            suggestions = result.get("suggestions", [])
            if suggestions:
                st.markdown("---")
                st.markdown("**Rule-based intervention checklist:**")
                priority_icons = {1: "🔴", 2: "🟡", 3: "🟢"}
                for s in suggestions:
                    st.markdown(
                        f"{priority_icons.get(s.priority,'⚪')} "
                        f"**{s.short_title}**  \n{s.action}"
                    )
                    if s.scheme_or_program:
                        st.caption(f"Scheme: {s.scheme_or_program}")
        else:
            st.markdown(result.get("markdown", result.get("summary", "")))


def render_chatbot(selected_area_scores: dict = None):
    """
    Renders the full chatbot UI.

    selected_area_scores: dict from mci_scores for the currently selected
    district. Uses districtname + canonical_state for context labels.
    """
    _init_chat()

    st.markdown("---")
    st.markdown("### 💬 Ask the MCI Assistant")

    if selected_area_scores:
        district = selected_area_scores.get("districtname", "")
        state    = selected_area_scores.get("statename", "")
        label    = f"{district}, {state}" if district else "(selected district)"
        st.info(
            f"📍 **Context:** {label} is selected. "
            "Policy questions will automatically use this district's scores.",
            icon="ℹ️",
        )

    # Example query chips
    st.caption("Try an example:")
    chip_cols = st.columns(4)
    for i, q in enumerate(EXAMPLE_QUERIES[:4]):
        if chip_cols[i % 4].button(q, key=f"chip_{i}", use_container_width=True):
            st.session_state._pending_input = q

    # Render message history
    for msg in st.session_state.chat_history:
        _render_message(msg)

    # Handle chip-triggered input
    pending    = st.session_state.pop("_pending_input", None)
    user_input = st.chat_input(
        "Ask about data, trends, or policy recommendations..."
    ) or pending

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            result = st.session_state.orchestrator.handle(
                message=user_input,
                selected_area_scores=selected_area_scores,
            )

        st.session_state.chat_history.append({
            "role": "assistant",
            "result": result,
        })
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()