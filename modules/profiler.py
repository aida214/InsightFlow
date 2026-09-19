"""
profiler.py
-----------
Looks at a dataset and describes it in simple terms:
how big it is, what each column contains, and what type it is.
"""

import warnings

import pandas as pd


# =========================================================
# HELPERS
# =========================================================


def pretty_name(column):
    """Turn 'unit_price' into 'Unit Price' for display."""

    return str(column).replace("_", " ").strip().title()


def describe_type(series):
    """Say what a column holds, in words a normal person understands."""

    if pd.api.types.is_bool_dtype(series):
        return "Yes / No"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date"

    if pd.api.types.is_numeric_dtype(series):
        return "Number"

    return "Text / Category"


def looks_like_date(series):
    """Check if a text column is really storing dates."""

    # Two columns sharing a name hand back a table instead of one column
    if isinstance(series, pd.DataFrame):
        return False

    # Numbers, dates and true/false columns are already handled elsewhere
    if pd.api.types.is_datetime64_any_dtype(series):
        return False

    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False

    sample = series.dropna().head(50)

    if sample.empty:
        return False

    # Real dates almost always contain a separator such as - / : or a space.
    # This stops number-like codes (for example '20419') being read as dates.
    text_values = sample.astype(str)

    if text_values.str.contains(r"[-/: ]").mean() < 0.8:
        return False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converted = pd.to_datetime(text_values, errors="coerce")

    except Exception:
        return False

    return converted.notna().mean() > 0.8


# =========================================================
# DATASET PROFILE
# =========================================================


def get_dataset_profile(df):
    """Generate basic information about a dataset."""

    total_cells = df.shape[0] * df.shape[1]
    missing_values = int(df.isnull().sum().sum())

    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": missing_values,
        "missing_percent": round(missing_values / total_cells * 100, 1)
        if total_cells > 0
        else 0.0,
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist(),
        "datetime_columns": [],
        "constant_columns": [],
        "id_like_columns": [],
    }

    for column in df.columns:

        series = df[column]

        # Date columns (already dates, or text that reads like dates)
        if pd.api.types.is_datetime64_any_dtype(series):
            profile["datetime_columns"].append(column)

        elif looks_like_date(series):
            profile["datetime_columns"].append(column)

        # Columns with only one value are useless for analysis
        if series.nunique(dropna=True) <= 1:
            profile["constant_columns"].append(column)

        # Columns where every row is different are usually IDs
        if len(df) > 0 and series.nunique(dropna=True) == len(df):
            if not pd.api.types.is_float_dtype(series):
                profile["id_like_columns"].append(column)

    return profile


# =========================================================
# COLUMN TABLE
# =========================================================


def get_column_table(df):
    """Build a friendly table that explains every column."""

    rows = []

    for column in df.columns:

        series = df[column]
        missing = int(series.isnull().sum())

        values = series.dropna()

        if len(values) > 0:
            example = str(values.iloc[0])
        else:
            example = "-"

        if len(example) > 30:
            example = example[:30] + "..."

        rows.append(
            {
                "Column": pretty_name(column),
                "What it holds": describe_type(series),
                "Missing": missing,
                "Missing %": round(missing / len(df) * 100, 1) if len(df) > 0 else 0.0,
                "Different values": int(series.nunique(dropna=True)),
                "Example": example,
            }
        )

    return pd.DataFrame(rows)
