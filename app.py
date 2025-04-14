"""
app.py
------
Streamlit front-end for the LLM-Powered EDA Tool.
Upload any CSV → get stats + charts + AI insights + downloadable HTML report.
"""

import os
import sys
import time
import io
import base64
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))
from src.eda_engine    import analyse
from src.llm_insights  import generate_insights
from src.report_generator import render_html

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM EDA Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f1117; }
  [data-testid="stSidebar"]          { background: #1a1d2e; }
  h1, h2, h3 { color: #e2e8f0 !important; }
  .stMetric label { color: #94a3b8 !important; }
  .stMetric [data-testid="metric-container"] div { color: #00d4aa !important; }
  div[data-testid="stDownloadButton"] button {
    background: #6c63ff; color: white; border: none;
    border-radius: 8px; font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/data-configuration.png", width=64)
    st.title("LLM EDA Tool")
    st.caption("Upload a CSV → get instant stats, charts & AI insights")
    st.divider()

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Required for AI-generated insights. Get yours at console.anthropic.com"
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()
    st.markdown("**Settings**")
    max_rows = st.slider("Max rows to analyse", 1_000, 500_000, 50_000, step=1_000)
    show_raw = st.checkbox("Show raw data preview", value=True)
    st.divider()
    st.caption("Built with Python · Pandas · Matplotlib · Seaborn · Claude AI")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🔬 LLM-Powered Automated EDA")
st.markdown("Drop any CSV dataset and get a full exploratory analysis with **AI-generated insights** in seconds.")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded is None:
    # Demo mode — load a sample dataset
    st.info("👆 Upload a CSV to begin — or click below to try with the built-in sample dataset.")
    if st.button("▶  Run on sample dataset (Titanic)"):
        sample_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        try:
            df = pd.read_csv(sample_url)
            dataset_name = "Titanic"
            st.session_state["df"]   = df
            st.session_state["name"] = dataset_name
        except Exception as e:
            st.error(f"Could not load sample: {e}")
    elif "df" not in st.session_state:
        st.stop()

if uploaded:
    df = pd.read_csv(uploaded, nrows=max_rows)
    dataset_name = Path(uploaded.name).stem
    st.session_state["df"]   = df
    st.session_state["name"] = dataset_name

df   = st.session_state["df"]
name = st.session_state["name"]

st.success(f"✅ Loaded **{name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

if show_raw:
    with st.expander("📄 Raw data preview (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True)

# ── Run analysis ──────────────────────────────────────────────────────────────
if st.button("🚀 Run Full EDA", type="primary", use_container_width=True):
    with st.spinner("Analysing dataset …"):
        t0     = time.time()
        result = analyse(df)
        t_eda  = time.time() - t0

    # KPIs
    cols = st.columns(5)
    cols[0].metric("Rows",       f"{result.shape[0]:,}")
    cols[1].metric("Columns",    result.shape[1])
    cols[2].metric("Duplicates", f"{result.duplicates:,}")
    cols[3].metric("Null Cols",  sum(1 for c in result.columns if c.null_pct > 0))
    cols[4].metric("Memory",     f"{result.memory_mb:.1f} MB")

    st.divider()

    # Column stats table
    st.subheader("📋 Column Summary")
    table_data = []
    for cs in result.columns:
        row = {
            "Column":   cs.name,
            "Type":     cs.dtype,
            "Nulls %":  cs.null_pct,
            "Unique":   cs.unique,
        }
        if cs.mean is not None:
            row.update({"Mean": cs.mean, "Std": cs.std, "Skew": cs.skew, "Outliers %": cs.outlier_pct})
        else:
            top = ", ".join(str(v) for v, _ in cs.top_values[:3])
            row["Top Values"] = top
        table_data.append(row)
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # Correlations
    if result.correlations:
        st.subheader("🔗 Top Correlations")
        corr_df = pd.DataFrame(
            [{"Feature Pair": k, "Correlation": v} for k, v in result.correlations.items()]
        )
        st.dataframe(corr_df, use_container_width=True, hide_index=True)

    # Charts
    if result.charts:
        st.subheader("📈 Charts")
        chart_items = list(result.charts.items())
        for i in range(0, len(chart_items), 2):
            row_cols = st.columns(2)
            for j, (key, b64) in enumerate(chart_items[i:i+2]):
                img_bytes = base64.b64decode(b64)
                row_cols[j].image(img_bytes, use_container_width=True)

    # LLM insights
    st.subheader("🤖 AI-Generated Insights")
    with st.spinner("Generating insights with Claude …"):
        insights = generate_insights(result.raw_summary, name)
    st.markdown(insights)

    # HTML report download
    st.divider()
    st.subheader("📥 Download Full Report")
    with st.spinner("Rendering HTML report …"):
        html = render_html(result, insights, name)
    st.download_button(
        label     = "⬇  Download HTML Report",
        data      = html.encode("utf-8"),
        file_name = f"eda_{name}.html",
        mime      = "text/html",
        use_container_width = True,
    )
    st.caption(f"Analysis completed in {t_eda:.1f}s  •  Report is fully self-contained (no internet required to open)")
