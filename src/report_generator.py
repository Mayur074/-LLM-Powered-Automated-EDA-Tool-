"""
src/report_generator.py
------------------------
Renders a self-contained HTML report from an EDAResult + LLM insights.
The output is a single .html file — no external dependencies.
"""

from __future__ import annotations
import base64
from datetime import datetime
from src.eda_engine import EDAResult


_CSS = """
:root {
  --bg: #0f1117;
  --card: #1a1d2e;
  --accent: #6c63ff;
  --accent2: #00d4aa;
  --text: #e2e8f0;
  --muted: #94a3b8;
  --border: #2d3154;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: var(--bg); color: var(--text);
  line-height: 1.7; padding: 32px 20px;
}
.wrap { max-width: 1100px; margin: 0 auto; }

/* Header */
header {
  background: linear-gradient(135deg, #1a1d2e 0%, #252847 100%);
  border: 1px solid var(--border);
  border-radius: 16px; padding: 36px 40px; margin-bottom: 32px;
  position: relative; overflow: hidden;
}
header::before {
  content: ''; position: absolute; top: -60px; right: -60px;
  width: 200px; height: 200px;
  background: radial-gradient(circle, #6c63ff33 0%, transparent 70%);
  border-radius: 50%;
}
header h1 { font-size: 2rem; font-weight: 700; color: #fff; }
header h1 span { color: var(--accent); }
.subtitle { color: var(--muted); margin-top: 6px; font-size: 0.95rem; }
.badge {
  display: inline-block; background: var(--accent); color: #fff;
  border-radius: 20px; padding: 3px 14px; font-size: 0.78rem;
  font-weight: 600; margin-top: 12px;
}

/* KPI cards */
.kpi-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 16px; margin-bottom: 32px;
}
.kpi {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 20px 24px;
}
.kpi-label { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-value { font-size: 1.9rem; font-weight: 700; color: var(--accent2); margin-top: 4px; }
.kpi-sub { font-size: 0.8rem; color: var(--muted); margin-top: 2px; }

/* Sections */
.section {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 14px; padding: 28px 32px; margin-bottom: 24px;
}
.section h2 {
  font-size: 1.15rem; font-weight: 700; color: var(--accent);
  border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-bottom: 18px;
}

/* LLM insights markdown styling */
.insights h2 { font-size: 1rem; color: var(--accent2); margin: 18px 0 8px; border: none; padding: 0; }
.insights p  { color: var(--text); margin-bottom: 10px; }
.insights ul { padding-left: 20px; }
.insights li { color: var(--text); margin-bottom: 6px; }
.insights strong { color: #fff; }

/* Column table */
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th {
  text-align: left; padding: 10px 12px;
  background: #252847; color: var(--muted);
  font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;
}
td { padding: 10px 12px; border-top: 1px solid var(--border); }
tr:hover td { background: #1e2240; }
.pill {
  display: inline-block; padding: 2px 10px; border-radius: 20px;
  font-size: 0.75rem; font-weight: 600;
}
.pill-num  { background: #1e3a5f; color: #60a5fa; }
.pill-cat  { background: #1e3a2f; color: #4ade80; }
.pill-warn { background: #3a2010; color: #fb923c; }
.pill-ok   { background: #1a2e1a; color: #4ade80; }

/* Charts */
.chart-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
  gap: 20px;
}
.chart-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 20px;
}
.chart-card h3 { font-size: 0.9rem; color: var(--muted); margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.06em; }
.chart-card img { width: 100%; border-radius: 8px; }

/* Correlation table */
.corr-row { display: flex; justify-content: space-between; padding: 8px 0; border-top: 1px solid var(--border); }
.corr-pair { color: var(--text); font-size: 0.88rem; }
.corr-bar-wrap { display: flex; align-items: center; gap: 10px; }
.corr-bar { height: 6px; border-radius: 3px; background: var(--accent); }
.corr-val  { color: var(--accent2); font-weight: 700; font-size: 0.88rem; min-width: 44px; text-align: right; }

footer { text-align: center; color: var(--muted); font-size: 0.8rem; margin-top: 40px; }
"""


def _chart_name_label(key: str) -> str:
    labels = {
        "missing_heatmap": "Missing Values Map",
        "distributions":   "Numeric Distributions",
        "correlation":     "Correlation Heatmap",
        "boxplots":        "Boxplots / Outlier View",
    }
    if key.startswith("cat_"):
        return f"Category — {key[4:]}"
    return labels.get(key, key.replace("_", " ").title())


def render_html(result: EDAResult, insights_md: str, dataset_name: str = "dataset") -> str:
    n_rows, n_cols = result.shape
    ts = datetime.now().strftime("%B %d, %Y  %H:%M")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    numeric_count  = sum(1 for c in result.columns if c.mean is not None)
    null_cols      = sum(1 for c in result.columns if c.null_pct > 0)

    kpi_html = f"""
    <div class="kpi-grid">
      <div class="kpi"><div class="kpi-label">Rows</div><div class="kpi-value">{n_rows:,}</div></div>
      <div class="kpi"><div class="kpi-label">Columns</div><div class="kpi-value">{n_cols}</div></div>
      <div class="kpi"><div class="kpi-label">Numeric Cols</div><div class="kpi-value">{numeric_count}</div></div>
      <div class="kpi"><div class="kpi-label">Duplicates</div><div class="kpi-value">{result.duplicates:,}</div>
        <div class="kpi-sub">{result.dup_pct}% of rows</div></div>
      <div class="kpi"><div class="kpi-label">Cols w/ Nulls</div><div class="kpi-value">{null_cols}</div></div>
      <div class="kpi"><div class="kpi-label">Memory</div><div class="kpi-value">{result.memory_mb:.1f}</div>
        <div class="kpi-sub">MB</div></div>
    </div>"""

    # ── Column stats table ────────────────────────────────────────────────────
    rows_html = ""
    for cs in result.columns:
        dtype_pill = f'<span class="pill pill-num">{cs.dtype}</span>' if cs.mean is not None \
                     else f'<span class="pill pill-cat">{cs.dtype}</span>'
        null_pill  = f'<span class="pill pill-warn">{cs.null_pct}%</span>' if cs.null_pct > 0 \
                     else f'<span class="pill pill-ok">0%</span>'
        if cs.mean is not None:
            extra = f"mean={cs.mean} | std={cs.std} | skew={cs.skew} | outliers={cs.outlier_pct}%"
        else:
            top3 = ", ".join(str(v) for v, _ in cs.top_values[:3])
            extra = f"top: {top3}"
        rows_html += f"""
        <tr>
          <td><strong>{cs.name}</strong></td>
          <td>{dtype_pill}</td>
          <td>{null_pill}</td>
          <td>{cs.unique:,}</td>
          <td style="color:#94a3b8;font-size:0.82rem">{extra}</td>
        </tr>"""

    col_table = f"""
    <div class="section">
      <h2>📋 Column Summary</h2>
      <table>
        <thead><tr><th>Column</th><th>Type</th><th>Nulls</th><th>Unique</th><th>Stats / Top Values</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""

    # ── Correlations ──────────────────────────────────────────────────────────
    corr_html = ""
    if result.correlations:
        corr_rows = ""
        for pair, val in result.correlations.items():
            bar_w = int(val * 160)
            corr_rows += f"""
            <div class="corr-row">
              <span class="corr-pair">{pair}</span>
              <div class="corr-bar-wrap">
                <div class="corr-bar" style="width:{bar_w}px"></div>
                <span class="corr-val">{val}</span>
              </div>
            </div>"""
        corr_html = f"""
        <div class="section">
          <h2>🔗 Top Correlations</h2>
          {corr_rows}
        </div>"""

    # ── Charts ────────────────────────────────────────────────────────────────
    chart_cards = ""
    for key, b64 in result.charts.items():
        label = _chart_name_label(key)
        chart_cards += f"""
        <div class="chart-card">
          <h3>{label}</h3>
          <img src="data:image/png;base64,{b64}" alt="{label}">
        </div>"""

    charts_section = f"""
    <div class="section">
      <h2>📈 Charts</h2>
      <div class="chart-grid">{chart_cards}</div>
    </div>""" if chart_cards else ""

    # ── LLM insights (simple markdown → html) ────────────────────────────────
    import re
    md = insights_md
    md = re.sub(r"^## (.+)$", r"<h2>\1</h2>", md, flags=re.MULTILINE)
    md = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", md)
    md = re.sub(r"^- (.+)$", r"<li>\1</li>", md, flags=re.MULTILINE)
    md = re.sub(r"(<li>.*</li>\n?)+", lambda m: f"<ul>{m.group()}</ul>", md, flags=re.DOTALL)
    md = re.sub(r"\n\n", r"<br>", md)

    insights_section = f"""
    <div class="section insights">
      <h2>🤖 AI-Generated Insights</h2>
      {md}
    </div>"""

    # ── Assemble ──────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EDA Report — {dataset_name}</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>EDA Report — <span>{dataset_name}</span></h1>
    <div class="subtitle">Generated {ts} • LLM-Powered Exploratory Data Analysis</div>
    <span class="badge">Powered by Claude AI</span>
  </header>

  {kpi_html}
  {col_table}
  {corr_html}
  {charts_section}
  {insights_section}

  <footer>Generated by llm-eda-tool • github.com/yourgithub/llm-eda-tool</footer>
</div>
</body>
</html>"""
    return html
