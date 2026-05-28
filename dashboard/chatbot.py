"""
dashboard/chatbot.py
=====================

Modern Floating Popup Chatbot
- Stable, non-breaking layout canvas integration
- Seamless Obsidian dark palette formatting across elements
- Clean, padded top header typography alignment
"""

from __future__ import annotations

import time
import streamlit as st
import pandas as pd

from config import DB_PATH
from agents.Orchestrator import Orchestrator
from dashboard.charts import dynamic_chart

# =========================================================
# FLOAT SUPPORT
# =========================================================

from streamlit_float import *

float_init()

# =========================================================
# CONFIG
# =========================================================

AGENT_ICONS = {
    "Data Query Agent": "🎰",
    "Policy Suggestion Agent": "🏛️",
}

EXAMPLE_QUERIES = [
    "Lowest MCI districts",
    "Maharashtra MCI trends",
    "Women safety comparison",
    "Why is this district poor?",
]

# =========================================================
# INIT CHAT
# =========================================================

def _init_chat(mode="popup"):
    history_key = f"{mode}_chat_history"
    orchestrator_key = f"{mode}_orchestrator"

    if history_key not in st.session_state:
        st.session_state[history_key] = []

    if orchestrator_key not in st.session_state:
        st.session_state[orchestrator_key] = Orchestrator(db_path=DB_PATH)

    return history_key, orchestrator_key


# =========================================================
# MESSAGE RENDERER
# =========================================================

def _render_message(msg: dict):
    role = msg["role"]

    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
        return

    result = msg.get("result", {})
    agent = result.get("agent", "Assistant")
    icon = AGENT_ICONS.get(agent, "🤖")

    with st.chat_message("assistant", avatar=icon):
        st.caption(agent)

        intent = result.get("intent", "")
        error = result.get("error", "")

        if error:
            st.error(error)
            return

        if intent == "data_query":
            df = result.get("data")
            chart_type = result.get("chart_type", "table")
            sql = result.get("sql", "")
            summary = result.get("summary", "")

            if summary:
                st.markdown(summary)

            if df is not None and not df.empty:
                if "year" in df.columns and df["year"].nunique() <= 1:
                    chart_type = "table"

                if chart_type not in ("table", "none"):
                    fig = dynamic_chart(df, chart_type)
                    if fig:
                        fig.update_layout(
                            autosize=True,
                            height=200,
                            margin=dict(l=20, r=10, t=20, b=20),
                            paper_bgcolor="#111827",
                            plot_bgcolor="#111827",
                            font=dict(color="#e5e7eb"),
                            xaxis=dict(tickangle=-45, automargin=True, gridcolor="rgba(255,255,255,0.05)"),
                            yaxis=dict(automargin=True, gridcolor="rgba(255,255,255,0.05)"),
                        )
                        st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

                st.dataframe(
                    df.head(50),
                    use_container_width=True,
                    height=min(200, (len(df) + 1) * 35 + 10),
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
                for s in suggestions:
                    st.info(f"**{s.short_title}**\n\n{s.action}")
        else:
            st.markdown(result.get("markdown", result.get("summary", "")))


# =========================================================
# MAIN POPUP CHATBOT
# =========================================================

def render_chatbot(selected_area_scores: dict = None, mode: str = "popup"):
    history_key, orchestrator_key = _init_chat(mode)
    popup_open_key = "dashboard_popup_open"

    if popup_open_key not in st.session_state:
        st.session_state[popup_open_key] = False

    # Floating Action Trigger Button
    btn = st.container()
    with btn:
        if st.button("💬", key="floating_chat_toggle", use_container_width=True):
            st.session_state[popup_open_key] = not st.session_state[popup_open_key]

    btn.float(
        css="""
            position: fixed;
            bottom: 24px;
            right: 24px;
            width: 56px;
            height: 56px;
            z-index: 999999;
        """
    )

    if not st.session_state[popup_open_key]:
        return

    # Master UI Outer Framework Panel Container
    popup = st.container()
    with popup:
        st.markdown(
            """
            <style>
            /* 1. COMPACT WRAPPER CANVAS OVERRIDES */
            div[data-testid="stVerticalBlock"] div:has(> .popup-header) {
                background-color: #0b111e !important;
                border-radius: 16px !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                box-shadow: 0 20px 48px rgba(0, 0, 0, 0.6) !important;
                padding: 0px !important;
                margin: 0px !important;
            }

            /* Elegant Top Panel Header Layout */
            .popup-header {
                padding: 16px 20px;
                background-color: #0e1726 !important;
                border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
                font-size: 16px;
                font-weight: 600;
                color: #ffffff !important;
                border-top-left-radius: 15px;
                border-top-right-radius: 15px;
            }

            /* 2. CHAT INPUT BAR CONFIGURATIONS & RECONCILED STYLING */
            /* Target the specific chat position overlay inside the floating element context */
            div[data-testid="stChatInputFormContainer"] {
                background-color: #0e1726 !important;
                border-top: 1px solid rgba(255, 255, 255, 0.06) !important;
                padding: 12px 16px !important;
                position: fixed !important;
                bottom: 100px !important;
                right: 32px !important;
                width: min(84vw, 436px) !important;
                box-sizing: border-box !important;
                z-index: 999999 !important;
                border-bottom-left-radius: 16px !important;
                border-bottom-right-radius: 16px !important;
            }

            /* Custom text input field texture controls */
            div[data-testid="stChatInputFormContainer"] textarea {
                background-color: #151f32 !important;
                color: #ffffff !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                border-radius: 8px !important;
            }

            /* High contrast layout style for message logs */
            .stChatMessage {
                background-color: #111827 !important;
                border: 1px solid rgba(255, 255, 255, 0.04) !important;
                border-radius: 12px !important;
                padding: 12px !important;
                margin-bottom: 8px;
            }

            /* Option chips formatting buttons styling */
            .stButton button {
                background-color: #151f32 !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                border-radius: 8px !important;
                color: #e5e7eb !important;
                font-size: 13px !important;
            }
            .stButton button:hover {
                background-color: #1c2a45 !important;
                border-color: rgba(255, 255, 255, 0.2) !important;
                color: #ffffff !important;
            }
            
            /* Status Filter Info Boxes Adjustments */
            div[data-testid="stNotification"] {
                background-color: rgba(30, 58, 138, 0.4) !important;
                color: #bfdbfe !important;
                border: 1px solid rgba(59, 130, 246, 0.2) !important;
                border-radius: 8px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Render Header Content Card
        st.markdown('<div class="popup-header">🏛️ Ask the MCI Assistant</div>', unsafe_allow_html=True)

        if selected_area_scores:
            district = selected_area_scores.get("districtname", "")
            state = selected_area_scores.get("statename", "")
            if district:
                st.info(f"Context Filter: {district}, {state}", icon="🍑")

        # =========================================================
        # INTERNAL LOG SCROLL WINDOW (STABLE STRUCTURE)
        # =========================================================
        # Set explicitly to 440 to avoid breaking structural space boundaries
        with st.container(height=440, border=False):
            st.caption("Quick Questions")
            cols = st.columns(2)
            for i, q in enumerate(EXAMPLE_QUERIES):
                if cols[i % 2].button(q, key=f"{mode}_chip_{i}", use_container_width=True):
                    st.session_state[f"{mode}_pending_input"] = q

            st.write("")

            for msg in st.session_state[history_key]:
                _render_message(msg)
            
            # Bottom offset structural margin so history messages aren't hidden behind the text input bar
            st.write("<div style='margin-bottom: 70px;'></div>", unsafe_allow_html=True)

        # =========================================================
        # PROMPT ENTRY FIELD
        # =========================================================
        pending = st.session_state.pop(f"{mode}_pending_input", None)
        user_input = st.chat_input("Ask about districts, trends, policy...", key=f"{mode}_chat_input") or pending

        if user_input:
            st.session_state[history_key].append({
                "role": "user",
                "content": user_input,
                "timestamp": time.time(),
            })

            with st.spinner("Thinking..."):
                result = st.session_state[orchestrator_key].handle(
                    message=user_input,
                    selected_area_scores=selected_area_scores,
                )

            st.session_state[history_key].append({
                "role": "assistant",
                "result": result,
                "timestamp": time.time(),
            })
            st.rerun()

    # Float panel base frame container positioning configurations
    popup.float(
        css="""
            position: fixed;
            bottom: 96px;
            right: 24px;
            width: min(92vw, 460px);
            height: 550px;
            background-color: #0b111e !important;
            border-radius: 16px;
            overflow: hidden;
            z-index: 999998;
        """
    )