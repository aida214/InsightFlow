"""
report_generator.py
-------------------
Builds the files the user takes away with them:
a plain text summary and a full HTML report they can open in any browser.

Charts are embedded straight into the HTML, so the report works offline
apart from the Plotly script it loads from a CDN.
"""

from datetime import datetime

import pandas as pd

from modules.profiler import pretty_name


PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


# =========================================================
# TEXT SUMMARY
# =========================================================


def build_text_summary(df, profile, report, insights, model_result=None):
    """A short text file summarising the whole analysis."""

    lines = []

    lines.append("INSIGHTFLOW ANALYSIS SUMMARY")
    lines.append(f"Created: {datetime.now().strftime('%d %B %Y, %H:%M')}")
    lines.append("=" * 60)
    lines.append("")

    lines.append("DATASET")
    lines.append(f"  Records: {profile['rows']:,}")
    lines.append(f"  Columns: {profile['columns']:,}")
    lines.append(f"  Empty cells: {profile['missing_values']:,}")
    lines.append(f"  Repeated rows: {profile['duplicate_rows']:,}")
    lines.append("")

    lines.append("CLEANING")
    for action in report["actions"]:
        lines.append(f"  - {action}")
    lines.append("")
    lines.append(
        f"  Rows: {report['rows_before']:,} before, {report['rows_after']:,} after"
    )
    lines.append(
        f"  Empty cells: {report['missing_before']:,} before, "
        f"{report['missing_after']:,} after"
    )
    lines.append("")

    lines.append("KEY FINDINGS")
    for insight in insights:
        lines.append(f"  {insight['title']}: {insight['text']}")
    lines.append("")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if numeric_columns:
        lines.append("STATISTICS FOR NUMBER COLUMNS")
        lines.append(df[numeric_columns].describe().round(2).to_string())
        lines.append("")

    if model_result and model_result.get("ok"):
        lines.append("PREDICTION MODEL")
        lines.append(f"  Target: {pretty_name(model_result['target'])}")
        lines.append(f"  Best model: {model_result['best_model_name']}")
        for name, value in model_result["metrics"].items():
            lines.append(f"  {name}: {value}")
        lines.append(f"  {model_result['explanation']}")
        lines.append(f"  {model_result['quality']}")
        lines.append("")
        lines.append("  Important: predictions are estimates from this dataset only.")
        lines.append("")

    lines.append("=" * 60)
    lines.append(
        "Filled values and forecasts are estimates. A connection between two "
        "columns does not prove that one causes the other."
    )

    return "\n".join(lines)


# =========================================================
# HTML REPORT
# =========================================================


STYLE = """
body { font-family: 'Segoe UI', Arial, sans-serif; color: #22333F;
       max-width: 1000px; margin: 0 auto; padding: 32px; line-height: 1.6; }
h1 { color: #12232E; margin-bottom: 4px; }
h2 { color: #12232E; margin-top: 40px; border-bottom: 2px solid #0F8B8D;
     padding-bottom: 6px; }
h3 { color: #17494D; margin-top: 26px; }
.sub { color: #5A6B7B; margin-top: 0; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin: 18px 0; }
.card { border: 1px solid #E4E9EE; border-left: 4px solid #0F8B8D;
        border-radius: 10px; padding: 12px 16px; min-width: 150px; }
.card .value { font-size: 1.6rem; font-weight: 600; }
.card .label { color: #5A6B7B; font-size: .85rem; }
.insight { border: 1px solid #E4E9EE; border-radius: 10px; padding: 12px 16px;
           margin-bottom: 10px; }
.insight h4 { margin: 0 0 4px 0; }
.insight p { margin: 0; color: #44525E; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { border: 1px solid #E4E9EE; padding: 7px 10px; text-align: left; }
th { background: #F4F7F9; }
.note { background: #FCF5EC; border-left: 4px solid #E4903A; padding: 12px 16px;
        border-radius: 8px; }
.footer { color: #8A99A6; font-size: .85rem; margin-top: 40px;
          border-top: 1px solid #E4E9EE; padding-top: 14px; }
"""


def chart_to_html(figure):
    """Turn a Plotly figure into an HTML block, or return nothing if it fails."""

    try:
        return figure.to_html(full_html=False, include_plotlyjs=False)
    except Exception:
        return ""


def build_html_report(
    df,
    profile,
    report,
    insights,
    charts=None,
    model_result=None,
    quality=None,
    file_name="your dataset",
):
    """Build a complete, self contained HTML report."""

    charts = charts or []

    parts = []

    parts.append(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>InsightFlow report</title>"
        f"<script src='{PLOTLY_CDN}'></script>"
        f"<style>{STYLE}</style></head><body>"
    )

    parts.append("<h1>InsightFlow report</h1>")
    parts.append(
        f"<p class='sub'>{file_name} &middot; "
        f"{datetime.now().strftime('%d %B %Y, %H:%M')}</p>"
    )

    # --------------------------------
    # Overview
    # --------------------------------

    parts.append("<h2>1. Dataset overview</h2>")

    cards = [
        (f"{profile['rows']:,}", "Records"),
        (f"{profile['columns']:,}", "Columns"),
        (f"{profile['missing_values']:,}", "Empty cells"),
        (f"{profile['duplicate_rows']:,}", "Repeated rows"),
    ]

    if quality:
        cards.append((f"{quality['score']}/100", f"Quality: {quality['label']}"))

    parts.append("<div class='cards'>")
    for value, label in cards:
        parts.append(
            f"<div class='card'><div class='value'>{value}</div>"
            f"<div class='label'>{label}</div></div>"
        )
    parts.append("</div>")

    # --------------------------------
    # Cleaning
    # --------------------------------

    parts.append("<h2>2. What was cleaned</h2>")
    parts.append("<ul>")
    for action in report["actions"]:
        parts.append(f"<li>{action}</li>")
    parts.append("</ul>")

    parts.append(
        "<table><tr><th></th><th>Before</th><th>After</th></tr>"
        f"<tr><td>Rows</td><td>{report['rows_before']:,}</td>"
        f"<td>{report['rows_after']:,}</td></tr>"
        f"<tr><td>Empty cells</td><td>{report['missing_before']:,}</td>"
        f"<td>{report['missing_after']:,}</td></tr>"
        f"<tr><td>Repeated rows</td><td>{report['duplicates_before']:,}</td>"
        f"<td>{report.get('duplicates_after', 0):,}</td></tr></table>"
    )

    # --------------------------------
    # Findings
    # --------------------------------

    parts.append("<h2>3. Key findings</h2>")

    for insight in insights:
        parts.append(
            f"<div class='insight'><h4>{insight['icon']} {insight['title']}</h4>"
            f"<p>{insight['text']}</p></div>"
        )

    # --------------------------------
    # Statistics
    # --------------------------------

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if numeric_columns:
        parts.append("<h2>4. Statistics for number columns</h2>")
        stats = df[numeric_columns].describe().round(2)
        stats.index = [
            "Records",
            "Average",
            "Spread",
            "Lowest",
            "25%",
            "Middle",
            "75%",
            "Highest",
        ][: len(stats.index)]
        parts.append(stats.to_html(border=0))

    # --------------------------------
    # Charts
    # --------------------------------

    if charts:
        parts.append("<h2>5. Charts</h2>")

        for chart in charts:
            parts.append(f"<h3>{chart['title']}</h3>")
            parts.append(f"<p>{chart['description']}</p>")

            html = chart_to_html(chart.get("figure"))

            if html:
                parts.append(html)

            parts.append(f"<div class='note'>{chart['summary']}</div>")

    # --------------------------------
    # Model
    # --------------------------------

    if model_result and model_result.get("ok"):

        parts.append("<h2>6. Prediction model</h2>")
        parts.append(
            f"<p>Target column: <b>{pretty_name(model_result['target'])}</b><br>"
            f"Best model: <b>{model_result['best_model_name']}</b></p>"
        )

        comparison = model_result.get("comparison")

        if isinstance(comparison, pd.DataFrame) and not comparison.empty:
            parts.append(comparison.to_html(index=False, border=0))

        parts.append(
            f"<p>{model_result['explanation']} {model_result['quality']}</p>"
        )

        importance = model_result.get("importance")

        if isinstance(importance, pd.DataFrame) and not importance.empty:
            parts.append("<h3>Most influential columns</h3>")
            parts.append(
                importance[["Column", "Importance"]].head(10).to_html(
                    index=False, border=0
                )
            )

        parts.append("<div class='note'><b>Important</b><ul>")
        for line in model_result.get("limitations", []):
            parts.append(f"<li>{line}</li>")
        parts.append("</ul></div>")

    # --------------------------------
    # Footer
    # --------------------------------

    parts.append(
        "<div class='footer'>Created with InsightFlow. Filled values and "
        "forecasts are estimates, and a connection between two columns does not "
        "prove that one causes the other.</div>"
    )

    parts.append("</body></html>")

    return "\n".join(parts)
