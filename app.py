import os
import pandas as pd
import polars as pl
import streamlit as st
from dotenv import load_dotenv
import requests  # For MCP integration

from tools.tools import (
    fix_columns,
    generate_summary,
    strategy_recommender,
    domain_expert,
    suggest_best_target,
    run_prediction,
    missing_report,
)

# Load environment variables (API keys, etc.)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit page configuration
st.set_page_config(page_title="InsightAgent v2.5", layout="wide")
st.title("🤖 InsightAgent v2.5")

# Initialize session state for data and results
if "df" not in st.session_state:
    st.session_state.df = None
if "pl_df" not in st.session_state:
    st.session_state.pl_df = None
if "insights" not in st.session_state:
    st.session_state.insights = ""
if "strategy" not in st.session_state:
    st.session_state.strategy = ""
if "domain" not in st.session_state:
    st.session_state.domain = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = {}

# Tabs for different parts of the app
upload_tab, insights_tab, dashboard_tab, predict_tab = st.tabs(["📁 Upload", "📌 Insights", "📊 Dashboard", "🔮 Predict"])

# -------- File Upload & Cleaning --------
def load_data(uploaded_file):
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()
    df_raw = None
    try:
        # Handling different file formats
        if ext == ".csv":
            raw_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            df_raw = pd.read_csv(io.BytesIO(raw_bytes))
        elif ext in [".xls", ".xlsx"]:
            sheet_names = pd.ExcelFile(uploaded_file).sheet_names
            sheet = st.selectbox("Select Sheet (Excel Only)", sheet_names, key="sheet_selector")
            uploaded_file.seek(0)
            df_raw = pd.read_excel(uploaded_file, sheet_name=sheet)
        else:
            st.error("Unsupported file type")
            return None
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

    # Clean data: Drop null rows and duplicates, reset index
    df_raw = df_raw.dropna(how='all').drop_duplicates().reset_index(drop=True)
    return df_raw

# -------- Function to Call MCP Server for Data Cleaning --------
def clean_data(df):
    try:
        response = requests.post(f"http://127.0.0.1:5000/clean_data", json={"data": df.to_dict()})
        response.raise_for_status()
        return response.json().get("cleaned_data")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error cleaning data: {e}")
        return None

# -------- Function to Call MCP Server for Summary --------
def generate_summary(df):
    try:
        response = requests.post(f"http://127.0.0.1:5000/generate_summary", json={"data": df.to_dict()})
        response.raise_for_status()
        return response.json().get("summary")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error generating summary: {e}")
        return None

# -------- Function to Call MCP Server for Prediction --------
def run_prediction(df, target_col):
    try:
        response = requests.post(f"http://127.0.0.1:5000/run_prediction", json={"data": df.to_dict(), "target_column": target_col})
        response.raise_for_status()
        return response.json().get("prediction_results")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error running prediction: {e}")
        return None

# -------- UI Layout --------
st.markdown(""" 
    <style> 
    .block-container { padding: 1.5rem 2rem; } 
    .stTabs [data-baseweb="tab-list"] button { padding: 0.5rem 1rem; font-size: 1rem; } 
    .element-container:has(.plotly-chart) { max-width: 50% !important; } 
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size:2.3rem;'>📊 InsightAgent v2.5</h1>", unsafe_allow_html=True)
pg1, pg2, pg3, pg4 = st.tabs(["📁 Upload", "📌 Insights", "📊 Dashboard", "🔮 Predict"])

# -------- Upload Tab --------
with pg1:
    st.subheader("📁 Upload File")
    uploaded_file = st.file_uploader("Upload CSV/XLSX", type=["csv", "xls", "xlsx"])
    if uploaded_file:
        try:
            # Load data and clean it
            df = load_data(uploaded_file)
            if df is not None:
                cleaned_df = clean_data(df)
                if cleaned_df is not None:
                    st.session_state.current_df = cleaned_df
                    st.session_state.project_name = uploaded_file.name.split('.')[0]
                    st.markdown("### Preview")
                    st.dataframe(df.head(10), use_container_width=True)

                    # Generate insights using MCP tools
                    with st.spinner("Generating insights..."):
                        st.session_state.insights = generate_summary(cleaned_df)
                    with st.spinner("Recommending strategy..."):
                        st.session_state.strategy = strategy_recommender(st.session_state.insights, openai_api_key)
                    with st.spinner("Identifying domain..."):
                        st.session_state.domain = domain_expert(cleaned_df, openai_api_key)
                    with st.spinner("Auto-predicting..."):
                        target = suggest_best_target(cleaned_df)
                        st.session_state.prediction = run_prediction(cleaned_df, target) if target else {"error": "No valid numeric target"}
        except Exception as e:
            st.error(f"❌ Error processing the file: {e}")

# -------- Insights Tab --------
with pg2:
    st.subheader("📌 Insight View")
    if st.session_state.pl_df is not None:
        st.markdown(st.session_state.insights)
        st.markdown("### Strategic Summary")
        st.info(st.session_state.strategy)
        st.markdown("### Domain Analysis")
        st.success(st.session_state.domain)
    else:
        st.warning("Upload data first.")

# -------- Dashboard Tab --------
with pg3:
    st.subheader("📊 Auto Dashboard (Coming Soon)")
    st.info("Dashboard generation will appear here in the next update.")

# -------- Prediction Tab --------
with pg4:
    st.subheader("🔮 Predict Outcome")
    if st.session_state.prediction:
        result = st.session_state.prediction
        if "error" in result:
            st.warning(result["error"])
        else:
            st.success(f"Auto Model: RandomForest | Target: {result['target']} | R² Score: {result['score']:.2f}")
            st.markdown("Sample Predictions:")
            st.write(result["predictions"])
    else:
        st.info("Upload data to trigger prediction.")
