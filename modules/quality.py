"""
quality.py
----------
Checks a dataset for the problems that make analysis go wrong, and
explains each one together with a recommended action.

Nothing here changes the data. It only reports what it finds, so the
user stays in control of what gets fixed.
"""

import pandas as pd

from modules.profiler import looks_like_date, pretty_name


# =========================================================
# MISSING VALUES
# =========================================================


def missing_report(df):
    """One row per column that has empty cells, with a suggested fix."""

    rows = []
    total = len(df)

    for column in df.columns:

        missing = int(df[column].isnull().sum())

        if missing == 0:
            continue

        percent = missing / total * 100 if total > 0 else 0

        if percent > 60:
            action = "More than half is empty. Consider dropping this column."
        elif pd.api.types.is_numeric_dtype(df[column]):
            action = "Fill with the median (the middle value of this column)."
        else:
            action = "Fill with the most common value."

        rows.append(
            {
                "Column": pretty_name(column),
                "Empty cells": missing,
                "Empty %": round(percent, 1),
                "Recommended action": action,
            }
        )

    return pd.DataFrame(rows).sort_values("Empty %", ascending=False).reset_index(
        drop=True
    ) if rows else pd.DataFrame()


# =========================================================
# TYPE PROBLEMS
# =========================================================


def clean_number_text(series):
    """
    Read numbers that are stored as text: '$ 1,200.50', '45%', '(300)'.

    Anything that is not a plain number once the currency symbol, the commas
    and the percent sign are removed is left as missing, so text such as
    'cust-0' or 'user4@mail.com' is never mistaken for a number.
    """

    text = series.astype(str).str.strip()

    text = text.str.replace(r"[,\s]", "", regex=True)          # 1,200 -> 1200
    text = text.str.replace(r"^[$£€₹¥]", "", regex=True)        # $12 -> 12
    text = text.str.replace(r"%$", "", regex=True)              # 45% -> 45
    text = text.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # (300) -> -300

    is_number = text.str.fullmatch(r"[-+]?\d*\.?\d+").fillna(False)

    return pd.to_numeric(text.where(is_number, other=None), errors="coerce")


def is_number_text(series):
    """True when a text column is really holding numbers."""

    values = series.dropna()

    if values.empty or pd.api.types.is_numeric_dtype(values):
        return False

    if looks_like_date(values):
        return False

    converted = clean_number_text(values.head(200))

    return converted.notna().mean() > 0.9


def type_report(df):
    """Find columns whose stored type does not match what they contain."""

    findings = []
    total = len(df)

    for column in df.columns:

        series = df[column]
        values = series.dropna()
        label = pretty_name(column)

        if values.empty:
            findings.append(
                {
                    "Column": label,
                    "Problem": "The column is completely empty.",
                    "Example": "-",
                    "Recommended action": "Remove this column.",
                }
            )
            continue

        example = str(values.iloc[0])[:28]

        if is_number_text(series):
            findings.append(
                {
                    "Column": label,
                    "Problem": "Numbers are stored as text, so no maths can be done.",
                    "Example": example,
                    "Recommended action": "Convert this column to numbers.",
                }
            )

        elif not pd.api.types.is_datetime64_any_dtype(series) and looks_like_date(
            series
        ):
            findings.append(
                {
                    "Column": label,
                    "Problem": "Dates are stored as text, so no trend can be drawn.",
                    "Example": example,
                    "Recommended action": "Convert this column to real dates.",
                }
            )

        if series.nunique(dropna=True) <= 1:
            findings.append(
                {
                    "Column": label,
                    "Problem": "Every row holds the same value.",
                    "Example": example,
                    "Recommended action": "Remove it. It cannot explain anything.",
                }
            )

        elif (
            not pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_datetime64_any_dtype(series)
            and total > 0
            and series.nunique(dropna=True) / total > 0.9
        ):
            findings.append(
                {
                    "Column": label,
                    "Problem": "Almost every row is different, so this is a name "
                    "or an ID rather than a group.",
                    "Example": example,
                    "Recommended action": "Keep it for reference, but do not "
                    "use it for charts or predictions.",
                }
            )

    return pd.DataFrame(findings)


# =========================================================
# OUTLIERS
# =========================================================


def outlier_report(df):
    """Find unusually high or low numbers using the IQR rule."""

    rows = []

    for column in df.select_dtypes(include="number").columns:

        values = df[column].dropna()

        if len(values) < 10:
            continue

        first_quarter = values.quantile(0.25)
        third_quarter = values.quantile(0.75)
        gap = third_quarter - first_quarter

        if gap == 0:
            continue

        low_limit = first_quarter - 1.5 * gap
        high_limit = third_quarter + 1.5 * gap

        unusual = values[(values < low_limit) | (values > high_limit)]

        if len(unusual) == 0:
            continue

        rows.append(
            {
                "Column": pretty_name(column),
                "Unusual values": int(len(unusual)),
                "Share %": round(len(unusual) / len(values) * 100, 1),
                "Usual range": f"{low_limit:,.2f} to {high_limit:,.2f}",
                "Most extreme": round(
                    float(unusual.iloc[(unusual - values.median()).abs().argmax()]), 2
                ),
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# OVERALL SCORE
# =========================================================


def quality_score(df):
    """A single 0 to 100 score so the user knows where they stand."""

    total_cells = df.shape[0] * df.shape[1]

    if total_cells == 0:
        return {"score": 0, "label": "Empty", "reasons": ["The file has no data."]}

    reasons = []
    score = 100

    missing_share = df.isnull().sum().sum() / total_cells * 100

    if missing_share > 0:
        penalty = min(35, missing_share * 1.5)
        score -= penalty
        reasons.append(f"{missing_share:,.1f}% of all cells are empty.")

    duplicate_share = df.duplicated().sum() / len(df) * 100 if len(df) > 0 else 0

    if duplicate_share > 0:
        penalty = min(25, duplicate_share * 2)
        score -= penalty
        reasons.append(f"{duplicate_share:,.1f}% of rows are exact copies.")

    problems = type_report(df)

    if not problems.empty:
        penalty = min(25, len(problems) * 5)
        score -= penalty
        reasons.append(f"{len(problems)} column(s) have a type or content problem.")

    score = int(max(0, round(score)))

    if score >= 85:
        label = "Ready to use"
    elif score >= 65:
        label = "Usable, needs a tidy"
    elif score >= 40:
        label = "Needs cleaning"
    else:
        label = "Needs serious cleaning"

    if not reasons:
        reasons.append("No problems were found in this dataset.")

    return {"score": score, "label": label, "reasons": reasons}
