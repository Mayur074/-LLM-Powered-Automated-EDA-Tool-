"""
src/eda_engine.py
-----------------
Core analysis module. Accepts a pandas DataFrame and returns
a structured EDAResult containing all stats, chart data,
and raw findings ready for LLM summarisation.
"""

from __future__ import annotations
import io, base64, warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ColumnStat:
    name:        str
    dtype:       str
    nulls:       int
    null_pct:    float
    unique:      int
    top_values:  list[tuple[Any, int]]   # [(value, count), ...]
    # numeric only
    mean:        float | None = None
    median:      float | None = None
    std:         float | None = None
    skew:        float | None = None
    kurt:        float | None = None
    q1:          float | None = None
    q3:          float | None = None
    min:         float | None = None
    max:         float | None = None
    outlier_pct: float | None = None


@dataclass
class EDAResult:
    shape:          tuple[int, int]
    memory_mb:      float
    duplicates:     int
    dup_pct:        float
    columns:        list[ColumnStat]
    correlations:   dict[str, float]       # top correlated pairs
    charts:         dict[str, str]         # name → base64 PNG
    raw_summary:    str                    # plain-text for LLM prompt


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _iqr_outlier_pct(series: pd.Series) -> float:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
    return round(mask.sum() / len(series) * 100, 2)


# ── Main analyser ─────────────────────────────────────────────────────────────

def analyse(df: pd.DataFrame) -> EDAResult:
    n_rows, n_cols = df.shape
    memory_mb  = df.memory_usage(deep=True).sum() / 1e6
    duplicates = int(df.duplicated().sum())
    dup_pct    = round(duplicates / n_rows * 100, 2)

    numeric_cols  = df.select_dtypes(include="number").columns.tolist()
    categoric_cols = df.select_dtypes(exclude="number").columns.tolist()

    # ── Per-column stats ──────────────────────────────────────────────────────
    col_stats: list[ColumnStat] = []
    for col in df.columns:
        s     = df[col]
        nulls = int(s.isna().sum())
        stat  = ColumnStat(
            name      = col,
            dtype     = str(s.dtype),
            nulls     = nulls,
            null_pct  = round(nulls / n_rows * 100, 2),
            unique    = int(s.nunique()),
            top_values= list(s.value_counts().head(5).items()),
        )
        if col in numeric_cols:
            clean = s.dropna()
            stat.mean        = round(float(clean.mean()), 4)
            stat.median      = round(float(clean.median()), 4)
            stat.std         = round(float(clean.std()), 4)
            stat.skew        = round(float(clean.skew()), 4)
            stat.kurt        = round(float(clean.kurt()), 4)
            stat.q1          = round(float(clean.quantile(0.25)), 4)
            stat.q3          = round(float(clean.quantile(0.75)), 4)
            stat.min         = round(float(clean.min()), 4)
            stat.max         = round(float(clean.max()), 4)
            stat.outlier_pct = _iqr_outlier_pct(clean)
        col_stats.append(stat)

    # ── Correlations ──────────────────────────────────────────────────────────
    top_corr: dict[str, float] = {}
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        pairs = (
            corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            .stack()
            .sort_values(ascending=False)
        )
        for (c1, c2), val in pairs.head(10).items():
            top_corr[f"{c1} × {c2}"] = round(float(val), 4)

    # ── Charts ────────────────────────────────────────────────────────────────
    charts: dict[str, str] = {}

    # 1. Missing values heatmap
    if df.isnull().sum().sum() > 0:
        fig, ax = plt.subplots(figsize=(max(8, n_cols * 0.6), 4))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax, cmap="YlOrRd")
        ax.set_title("Missing Values Map", fontsize=13, fontweight="bold")
        charts["missing_heatmap"] = _fig_to_b64(fig)

    # 2. Distribution plots for numeric columns (up to 12)
    plot_cols = numeric_cols[:12]
    if plot_cols:
        cols_per_row = 3
        nrows = (len(plot_cols) + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(nrows, cols_per_row,
                                 figsize=(cols_per_row * 4, nrows * 3))
        axes = np.array(axes).flatten()
        for i, col in enumerate(plot_cols):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#4C72B0")
            axes[i].set_title(col, fontsize=10)
            axes[i].set_xlabel("")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Numeric Distributions", fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        charts["distributions"] = _fig_to_b64(fig)

    # 3. Correlation heatmap
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols) * 0.8),
                                        max(5, len(numeric_cols) * 0.7)))
        mask = np.triu(np.ones_like(df[numeric_cols].corr(), dtype=bool))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", mask=mask, ax=ax,
                    annot_kws={"size": 8}, linewidths=0.5)
        ax.set_title("Correlation Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        charts["correlation"] = _fig_to_b64(fig)

    # 4. Boxplots (outlier visualisation)
    if plot_cols:
        fig, ax = plt.subplots(figsize=(max(8, len(plot_cols)), 5))
        df[plot_cols].boxplot(ax=ax, vert=True, patch_artist=True,
                              medianprops=dict(color="red", linewidth=2))
        ax.set_title("Boxplots — Outlier Overview", fontsize=13, fontweight="bold")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        charts["boxplots"] = _fig_to_b64(fig)

    # 5. Top categorical value counts (up to 3 cols)
    for col in categoric_cols[:3]:
        vc = df[col].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        vc.plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_title(f"Top Values — {col}", fontsize=12, fontweight="bold")
        ax.invert_yaxis()
        plt.tight_layout()
        charts[f"cat_{col}"] = _fig_to_b64(fig)

    # ── Plain-text summary for LLM ────────────────────────────────────────────
    lines = [
        f"Dataset shape: {n_rows} rows × {n_cols} columns",
        f"Memory: {memory_mb:.2f} MB",
        f"Duplicate rows: {duplicates} ({dup_pct}%)",
        "",
        "=== COLUMNS ===",
    ]
    for cs in col_stats:
        lines.append(f"\n[{cs.name}] type={cs.dtype}, nulls={cs.null_pct}%, unique={cs.unique}")
        if cs.mean is not None:
            lines.append(
                f"  mean={cs.mean}, median={cs.median}, std={cs.std}, "
                f"skew={cs.skew}, outliers={cs.outlier_pct}%"
            )
        else:
            top = ", ".join(f"{v}({c})" for v, c in cs.top_values[:3])
            lines.append(f"  top values: {top}")

    if top_corr:
        lines += ["", "=== TOP CORRELATIONS ==="]
        for pair, val in list(top_corr.items())[:5]:
            lines.append(f"  {pair}: {val}")

    raw_summary = "\n".join(lines)

    return EDAResult(
        shape        = (n_rows, n_cols),
        memory_mb    = memory_mb,
        duplicates   = duplicates,
        dup_pct      = dup_pct,
        columns      = col_stats,
        correlations = top_corr,
        charts       = charts,
        raw_summary  = raw_summary,
    )
