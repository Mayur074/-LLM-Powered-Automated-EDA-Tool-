"""
src/llm_insights.py
--------------------
Sends the plain-text EDA summary to Claude and returns
structured natural-language insights.
"""

from __future__ import annotations
import os
import anthropic

CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

SYSTEM_PROMPT = """You are a senior data scientist performing exploratory data analysis.
Given a statistical summary of a dataset, produce clear, actionable insights in plain English.
Structure your response with these exact sections:

## 🔍 Overview
2-3 sentences about what this dataset looks like at a high level.

## ⚠️ Data Quality Issues
Bullet points on nulls, duplicates, suspicious distributions, or encoding issues. Be specific with column names and numbers.

## 📊 Key Patterns & Distributions
Bullet points on notable findings: skewed columns, heavy outliers, dominant categories. Use the stats provided.

## 🔗 Relationships & Correlations
Bullet points on the strongest correlations and what they might mean. Flag multicollinearity risks if relevant.

## 💡 Recommendations
3-5 concrete next steps a data scientist should take before modelling (e.g. imputation strategy, log-transform, drop column, encode categorical).

Keep the tone professional but conversational. Avoid generic filler. Be specific."""


def generate_insights(raw_summary: str, dataset_name: str = "dataset") -> str:
    """
    Call the Claude API and return markdown-formatted insights.
    Falls back to a placeholder message if the API key is missing.
    """
    if not CLIENT.api_key:
        return (
            "**⚠️ LLM insights unavailable** — set `ANTHROPIC_API_KEY` environment variable.\n\n"
            "The statistical analysis above is complete. Add your API key to unlock AI-generated insights."
        )

    prompt = f"Here is the EDA summary for the dataset '{dataset_name}':\n\n{raw_summary}"

    message = CLIENT.messages.create(
        model      = "claude-sonnet-4-20250514",
        max_tokens = 1500,
        system     = SYSTEM_PROMPT,
        messages   = [{"role": "user", "content": prompt}],
    )
    return message.content[0].text
