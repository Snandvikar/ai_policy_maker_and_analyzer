import streamlit as st
import duckdb
import pandas as pd
import os
import plotly.express as px

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="India Digital Infra Dashboard", layout="wide")

# -------------------------------
# DB PATH (PORTABLE)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "trai_5factor.duckdb")

# -------------------------------
# DB CONNECTION
# -------------------------------
@st.cache_resource
def get_connection(db_path):
    return duckdb.connect(db_path)

conn = get_connection(DB_FILE)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_table(table_name):
    return conn.execute(f"SELECT * FROM {table_name}").df()

# -------------------------------
# TABLE MAPPING
# -------------------------------
tables = {
    "Cell Towers": "infrastructure",
    "Fiber & OFC": "network_connectivity",
    "Digital Literacy": "digital_literacy",
    "Socio Economic": "socio_economic"
}

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("📊 Navigation")

selected_table_name = st.sidebar.selectbox(
    "Select Dataset",
    list(tables.keys())
)

table_key = tables[selected_table_name]

try:
    df = load_table(table_key)
except:
    st.error(f"Table '{table_key}' not found in database.")
    st.stop()

# -------------------------------
# HEADER
# -------------------------------
st.title("🇮🇳 India Digital Infrastructure Dashboard")
st.subheader(selected_table_name)

# -------------------------------
# FILTERS
# -------------------------------
st.sidebar.markdown("### 🔍 Filters")

if "city" in df.columns:
    cities = st.sidebar.multiselect(
        "Select City",
        sorted(df["city"].dropna().unique())
    )
    if cities:
        df = df[df["city"].isin(cities)]

# -------------------------------
# METRICS
# -------------------------------
st.markdown("### 📌 Key Metrics")

col1, col2, col3 = st.columns(3)

numeric_cols = df.select_dtypes(include="number").columns

if len(numeric_cols) >= 3:
    col1.metric("Avg " + numeric_cols[0], round(df[numeric_cols[0]].mean(), 2))
    col2.metric("Avg " + numeric_cols[1], round(df[numeric_cols[1]].mean(), 2))
    col3.metric("Avg " + numeric_cols[2], round(df[numeric_cols[2]].mean(), 2))

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.markdown("### 📄 Data Preview")
st.dataframe(df, use_container_width=True)

# -------------------------------
# VISUALIZATION
# -------------------------------
st.markdown("### 📊 Visualization")

col_x = st.selectbox("X-axis", df.columns)
col_y = st.selectbox("Y-axis", numeric_cols)

chart_type = st.radio("Chart Type", ["Bar", "Line", "Scatter"])

grouped = df.groupby(col_x)[col_y].mean().reset_index()

if chart_type == "Bar":
    fig = px.bar(grouped, x=col_x, y=col_y)
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Line":
    fig = px.line(grouped, x=col_x, y=col_y)
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Scatter":
    fig = px.scatter(df, x=col_x, y=col_y, color="city" if "city" in df.columns else None)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# CORRELATION
# -------------------------------
st.markdown("### 📈 Correlation Matrix")

if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    st.dataframe(corr)

# -------------------------------
# DOWNLOAD
# -------------------------------
st.markdown("### ⬇️ Download Data")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download CSV",
    csv,
    "filtered_data.csv",
    "text/csv"
)