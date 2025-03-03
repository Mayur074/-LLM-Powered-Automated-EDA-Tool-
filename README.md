# 🔬 LLM-Powered Automated EDA Tool

An AI-augmented Exploratory Data Analysis tool that accepts any CSV dataset and automatically generates **full statistical analysis, visualizations, and natural-language insights** using Claude AI — all in under 60 seconds.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![Claude AI](https://img.shields.io/badge/Claude-AI%20Insights-6c63ff)
![Pandas](https://img.shields.io/badge/Pandas-2.2-green)

---

## ✨ Features

- **One-click analysis** — upload any CSV and get a complete EDA in seconds
- **AI-generated insights** — Claude reads the stats and writes plain-English findings, quality warnings, and modelling recommendations
- **Auto-generated charts** — distributions, correlation heatmap, boxplots, missing value map, categorical bar charts
- **Downloadable HTML report** — fully self-contained, shareable, no internet required to open
- **Scales to 500K rows** — configurable row limit via sidebar slider
- **Works without an API key** — statistical analysis runs fully offline; AI insights unlock with your Anthropic key

---

## 🖼️ What You Get

| Section | Contents |
|---|---|
| KPI Cards | Row count, column count, duplicate rate, null columns, memory |
| Column Summary | Type, null %, unique count, mean/std/skew/outlier % (numeric) or top values (categorical) |
| Correlations | Top 10 feature pairs by absolute Pearson correlation |
| Charts | Histograms + KDE, correlation heatmap, boxplots, missing value map, category bar charts |
| AI Insights | Overview, data quality issues, key patterns, correlations, modelling recommendations |
| HTML Report | All of the above in a dark-themed, self-contained HTML file |

---

## 🚀 Quickstart

```bash
git clone https://github.com/yourgithub/llm-eda-tool.git
cd llm-eda-tool
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501, paste your Anthropic API key in the sidebar, upload a CSV, and click **Run Full EDA**.

---

## 🔑 API Key Setup

The statistical analysis and charts work **without** an API key.  
For AI-generated insights, get a free key at [console.anthropic.com](https://console.anthropic.com) and either:

- Paste it in the sidebar text field, **or**
- Set it as an environment variable:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
streamlit run app.py
```

---

## 🏗️ Architecture

```
User uploads CSV
      │
      ▼
 eda_engine.py          ← Pure Python/Pandas analysis
  - Per-column stats     (dtype, nulls, unique, mean/std/skew/outliers)
  - Correlations         (Pearson, top 10 pairs)
  - Chart generation     (Matplotlib + Seaborn → base64 PNG)
  - Plain-text summary   (structured for LLM input)
      │
      ├──► llm_insights.py     ← Claude API call
      │     System prompt shapes output into 5 sections
      │     Returns markdown-formatted insights
      │
      └──► report_generator.py ← Renders self-contained HTML
            Dark-themed, all charts embedded as base64

app.py  ← Streamlit UI wires everything together
```

---

## 📁 Project Structure

```
llm-eda-tool/
├── app.py                    # Streamlit application
├── src/
│   ├── eda_engine.py         # Core analysis: stats + charts
│   ├── llm_insights.py       # Claude API integration
│   └── report_generator.py  # HTML report renderer
├── requirements.txt
└── README.md
```

---

## 📊 Example Output (AI Insights)

> **Data Quality Issues**
> - `Age` has 19.9% missing values — consider median imputation grouped by `Pclass`
> - `Cabin` is 77% null — likely not usable; consider dropping or engineering a "has_cabin" binary feature
> - 107 duplicate rows detected (12%) — review before modelling
>
> **Key Patterns**
> - `Fare` is heavily right-skewed (skew=4.79) — apply log transform before linear models
> - `Age` shows a slight bimodal distribution suggesting child/adult subgroups worth segmenting
>
> **Recommendations**
> 1. Drop `Cabin`, impute `Age` with grouped median, encode `Sex` and `Embarked`
> 2. Log-transform `Fare` to reduce skew
> 3. Engineer `FamilySize = SibSp + Parch + 1` — strong signal in survival literature

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Analysis | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| AI Insights | Anthropic Claude (claude-sonnet) |
| Report | Vanilla HTML/CSS (self-contained) |

---

## 👤 Author

**Mayur Sangle** — M.S. Data Science, University of Maryland College Park  
[LinkedIn](https://linkedin.com/in/mayur-sangle-04m) • [GitHub](https://github.com/Mayur074) • [Email](msangle074@gmail.com)
