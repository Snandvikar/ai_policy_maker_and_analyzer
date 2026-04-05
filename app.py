import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Telecom Digital Divide Dashboard",
    layout="wide"
)

st.title("📡 Digital Connectivity & Telecom Analytics Dashboard")

# -----------------------------
# Connect to DuckDB
# -----------------------------

@st.cache(allow_output_mutation=True)
def get_connection():
    return duckdb.connect("trai_5factor.duckdb")

conn = get_connection()


# -----------------------------
# Load Data
# -----------------------------

@st.cache(allow_output_mutation=True)
def load_data():

    cell_towers = conn.execute("SELECT * FROM cell_towers").df()
    fiber = conn.execute("SELECT * FROM infrastructure_fiber_and_ofc").df()
    socio = conn.execute("SELECT * FROM socio_economic_indicators").df()
    digital = conn.execute("SELECT * FROM digital_literacy").df()

    return cell_towers, fiber, socio, digital

cell_towers, fiber, socio, digital = load_data()


# -----------------------------
# Sidebar Filters
# -----------------------------

st.sidebar.header("Filters")

city = st.sidebar.selectbox(
    "Select City",
    sorted(cell_towers["city"].dropna().unique())
)

district = st.sidebar.selectbox(
    "Select District",
    sorted(cell_towers["district"].dropna().unique())
)

# Filter Data

cell_filtered = cell_towers[
    (cell_towers["city"] == city) &
    (cell_towers["district"] == district)
]

fiber_filtered = fiber[
    (fiber["city"] == city) &
    (fiber["district"] == district)
]

socio_filtered = socio[
    (socio["city"] == city) &
    (socio["district"] == district)
]

digital_filtered = digital[
    (digital["city"] == city) &
    (digital["district"] == district)
]


# -----------------------------
# KPI Section
# -----------------------------

st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Towers",
    int(cell_filtered["total_towers"].values[0])
)

col2.metric(
    "5G BTS",
    int(cell_filtered["5g_bts"].values[0])
)

col3.metric(
    "Fiber Coverage %",
    round(cell_filtered["bts_fiberized_percent"].values[0], 2)
)

col4.metric(
    "Median Download Speed",
    round(cell_filtered["dl_median_mbps"].values[0], 2)
)

# -----------------------------
# Infrastructure Section
# -----------------------------

st.subheader("📡 Infrastructure Distribution")

fig = px.bar(
    cell_towers,
    x="district",
    y="total_towers",
    color="city",
    title="Tower Distribution by District"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 5G Analysis
# -----------------------------

st.subheader("📶 5G Rollout")

fig_5g = px.scatter(
    cell_towers,
    x="towers_per_km2",
    y="5g_bts",
    color="city",
    size="population_m",
    hover_data=["district"]
)

st.plotly_chart(fig_5g, use_container_width=True)

# -----------------------------
# Fiber Section
# -----------------------------

st.subheader("🧵 Fiber Infrastructure")

fig_fiber = px.bar(
    fiber,
    x="district",
    y="ofc_total_km",
    color="city"
)

st.plotly_chart(fig_fiber, use_container_width=True)


# -----------------------------
# Digital Literacy Section
# -----------------------------

st.subheader("👩‍💻 Digital Literacy")

fig_digital = px.scatter(
    digital,
    x="hh_internet_percent",
    y="digital_skill_percent",
    color="city",
    size="slum_pop_percent",
    hover_data=["district"]
)

st.plotly_chart(fig_digital, use_container_width=True)


# -----------------------------
# Socio Economic Section
# -----------------------------

st.subheader("📉 Socio Economic Insights")

fig_socio = px.scatter(
    socio,
    x="gdp_per_capita_rs_lakh",
    y="literacy_rate_percent",
    color="city",
    size="poverty_rate_percent",
    hover_data=["district"]
)

st.plotly_chart(fig_socio, use_container_width=True)


# -----------------------------
# Digital Divide Section
# -----------------------------

st.subheader("⚡ Digital Divide Analysis")

merged = digital.merge(
    socio,
    on=["city", "district", "area"],
    how="inner"
)

fig_divide = px.scatter(
    merged,
    x="poverty_rate_percent",
    y="hh_internet_percent",
    color="city",
    size="slum_pop_percent"
)

st.plotly_chart(fig_divide, use_container_width=True)


# -----------------------------
# AI Agent Section
# -----------------------------

st.subheader("🤖 Ask AI Agent")

query = st.text_input("Ask about telecom data")

if st.button("Ask"):

    # placeholder for your agent
    response = f"Agent response for: {query}"

    st.write(response)

    # Example dynamic chart
    fig_agent = px.bar(
        cell_towers,
        x="district",
        y="dl_median_mbps"
    )

    st.plotly_chart(fig_agent, use_container_width=True)

# -----------------------------
# Raw Data Viewer
# -----------------------------

st.subheader("📄 Raw Data")

table_select = st.selectbox(
    "Select Table",
    ["cell_towers", "fiber", "digital", "socio"]
)

if table_select == "cell_towers":
    st.dataframe(cell_towers)

elif table_select == "fiber":
    st.dataframe(fiber)

elif table_select == "digital":
    st.dataframe(digital)

else:
    st.dataframe(socio)