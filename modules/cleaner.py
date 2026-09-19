"""
cleaner.py
----------
Fixes the most common problems in a dataset and explains,
in plain English, exactly what was changed.

Every step can be switched on or off from the app sidebar, and the
report only ever claims an action that really happened.
"""

import warnings

import numpy as np
import pandas as pd

from modules.profiler import looks_like_date
from modules.quality import clean_number_text, is_number_text


# =========================================================
# HELPER: TIDY COLUMN NAMES
# =========================================================


def tidy_names(columns):
    """Make column names lowercase, trimmed and underscore separated."""

    new_names = []

    for name in columns:

        clean = str(name).strip().lower()
        clean = clean.replace(" ", "_").replace("-", "_")

        # Keep only letters, numbers and underscores
        clean = "".join(
            character for character in clean if character.isalnum() or character == "_"
        )

        clean = clean.strip("_")

        if clean == "":
            clean = "column"

        # Avoid two columns ending up with the same name
        final = clean
        counter = 2

        while final in new_names:
            final = f"{clean}_{counter}"
            counter += 1

        new_names.append(final)

    return new_names


def make_names_unique(columns):
    """Two columns with the same name break pandas, so number the repeats."""

    seen = {}
    result = []

    for name in columns:

        text = str(name)

        if text in seen:
            seen[text] += 1
            result.append(f"{text}_{seen[text]}")
        else:
            seen[text] = 1
            result.append(text)

    return result


# =========================================================
# MAIN CLEANING FUNCTION
# =========================================================


def clean_dataset(
    df,
    remove_duplicates=True,
    tidy_column_names=True,
    trim_text=True,
    convert_numbers=True,
    convert_dates=True,
    fill_missing=True,
):
    """
    Clean common dataset problems and return:
    cleaned dataframe + cleaning report.
    """

    cleaned_df = df.copy()

    report = {
        "rows_before": int(df.shape[0]),
        "columns_before": int(df.shape[1]),
        "missing_before": int(cleaned_df.isnull().sum().sum()),
        "duplicates_before": int(cleaned_df.duplicated().sum()),
        "duplicates_removed": 0,
        "duplicates_kept": 0,
        "missing_filled": 0,
        "columns_renamed": 0,
        "text_columns_trimmed": 0,
        "number_columns_converted": 0,
        "date_columns_converted": 0,
        "infinite_values_fixed": 0,
        "name_changes": [],
        "actions": [],
    }

    # --------------------------------
    # 0. Repeated column names always get fixed
    # --------------------------------

    original_columns = cleaned_df.columns.tolist()
    unique_columns = make_names_unique(original_columns)

    if unique_columns != original_columns:
        cleaned_df.columns = unique_columns
        report["actions"].append(
            "Renamed columns that shared the same name, because pandas cannot "
            "tell them apart otherwise."
        )

    # --------------------------------
    # 1. Remove duplicate rows
    # --------------------------------

    if remove_duplicates:

        duplicates = int(cleaned_df.duplicated().sum())

        if duplicates > 0:

            rows = len(cleaned_df)
            share = duplicates / rows if rows > 0 else 0

            # In a table of pure categories (city, plan, device) identical rows
            # are normal records, not accidental copies. Deleting them would
            # throw away most of the data, so they are kept and reported.
            widest_column = max(
                cleaned_df[column].nunique(dropna=True) / rows
                for column in cleaned_df.columns
            ) if rows > 0 else 1

            if share > 0.3 and widest_column < 0.5:
                report["duplicates_kept"] = duplicates
                report["actions"].append(
                    f"Found {duplicates:,} rows that repeat an earlier row, but "
                    "every column in this dataset holds only a few different "
                    "values, so identical rows are most likely genuine separate "
                    "records. They were kept."
                )
            else:
                cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
                report["duplicates_removed"] = duplicates
                report["actions"].append(
                    f"Removed {duplicates:,} rows that were exact copies of "
                    "another row."
                )

    # --------------------------------
    # 2. Clean column names
    # --------------------------------

    if tidy_column_names:

        before_names = cleaned_df.columns.tolist()
        after_names = tidy_names(before_names)

        changes = [
            (before, after)
            for before, after in zip(before_names, after_names)
            if before != after
        ]

        cleaned_df.columns = after_names

        if changes:
            report["columns_renamed"] = len(changes)
            report["name_changes"] = changes
            report["actions"].append(
                f"Tidied {len(changes)} column name(s) so they are safe to use in "
                "charts and models. The original names are listed below."
            )

    # --------------------------------
    # 3. Trim extra spaces in text
    # --------------------------------

    if trim_text:

        trimmed = 0

        for column in cleaned_df.columns:

            if pd.api.types.is_numeric_dtype(
                cleaned_df[column]
            ) or pd.api.types.is_datetime64_any_dtype(cleaned_df[column]):
                continue

            before = cleaned_df[column].copy()

            cleaned_df[column] = cleaned_df[column].apply(
                lambda value: value.strip() if isinstance(value, str) else value
            )

            if not before.equals(cleaned_df[column]):
                trimmed += 1

        if trimmed > 0:
            report["text_columns_trimmed"] = trimmed
            report["actions"].append(
                f"Removed extra spaces in {trimmed} text column(s), so values like "
                "'Lahore ' and 'Lahore' are treated as the same thing."
            )

    # --------------------------------
    # 4. Numbers stored as text
    # --------------------------------

    if convert_numbers:

        converted = 0

        for column in cleaned_df.columns:

            series = cleaned_df[column]

            if pd.api.types.is_numeric_dtype(
                series
            ) or pd.api.types.is_datetime64_any_dtype(series):
                continue

            # A column of plain whole numbers that never repeats is an ID such
            # as an order number, so it is left as text. A money or decimal
            # column can also be all-unique, and that one should be converted.
            if len(cleaned_df) > 0 and series.nunique(dropna=True) == len(cleaned_df):

                sample = series.dropna().astype(str).head(50)

                if sample.str.fullmatch(r"\d+").mean() > 0.9:
                    continue

            if is_number_text(series):
                cleaned_df[column] = clean_number_text(series)
                converted += 1

        if converted > 0:
            report["number_columns_converted"] = converted
            report["actions"].append(
                f"Converted {converted} column(s) from text to numbers, so they can "
                "be averaged, charted and used in predictions."
            )

    # --------------------------------
    # 5. Convert text dates into real dates
    # --------------------------------

    if convert_dates:

        converted = 0

        for column in cleaned_df.columns:

            series = cleaned_df[column]

            if pd.api.types.is_datetime64_any_dtype(
                series
            ) or pd.api.types.is_numeric_dtype(series):
                continue

            if looks_like_date(series):

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cleaned_df[column] = pd.to_datetime(series, errors="coerce")

                converted += 1

        if converted > 0:
            report["date_columns_converted"] = converted
            report["actions"].append(
                f"Recognised {converted} column(s) as dates, which makes "
                "time based charts and forecasts possible."
            )

    # --------------------------------
    # 6. Infinite values
    # --------------------------------

    numeric_columns = cleaned_df.select_dtypes(include="number").columns

    infinite_count = 0

    for column in numeric_columns:
        found = int(np.isinf(cleaned_df[column].to_numpy(dtype="float64", na_value=np.nan)).sum())
        if found > 0:
            cleaned_df[column] = cleaned_df[column].replace([np.inf, -np.inf], np.nan)
            infinite_count += found

    if infinite_count > 0:
        report["infinite_values_fixed"] = infinite_count
        report["actions"].append(
            f"Found {infinite_count:,} infinite values (usually caused by dividing "
            "by zero) and treated them as missing."
        )

    # --------------------------------
    # 7. Fill missing numeric values
    # --------------------------------

    if fill_missing:

        filled_numeric = 0
        filled_text = 0

        for column in cleaned_df.select_dtypes(include="number").columns:

            missing_count = int(cleaned_df[column].isnull().sum())

            if missing_count > 0:

                median_value = cleaned_df[column].median()

                if pd.isna(median_value):
                    median_value = 0

                cleaned_df[column] = cleaned_df[column].fillna(median_value)

                report["missing_filled"] += missing_count
                filled_numeric += missing_count

        if filled_numeric > 0:
            report["actions"].append(
                f"Filled {filled_numeric:,} empty number cells with the median "
                "(the middle value of that column)."
            )

        # --------------------------------
        # 8. Fill missing text values
        # --------------------------------

        for column in cleaned_df.columns:

            if pd.api.types.is_numeric_dtype(
                cleaned_df[column]
            ) or pd.api.types.is_datetime64_any_dtype(cleaned_df[column]):
                continue

            missing_count = int(cleaned_df[column].isnull().sum())

            if missing_count > 0:

                mode_values = cleaned_df[column].mode()

                if len(mode_values) > 0:
                    replacement = mode_values[0]
                else:
                    replacement = "Unknown"

                cleaned_df[column] = cleaned_df[column].fillna(replacement)

                report["missing_filled"] += missing_count
                filled_text += missing_count

        if filled_text > 0:
            report["actions"].append(
                f"Filled {filled_text:,} empty text cells with the most common "
                "value in that column."
            )

    # --------------------------------
    # 9. Final numbers
    # --------------------------------

    report["missing_after"] = int(cleaned_df.isnull().sum().sum())
    report["rows_after"] = int(cleaned_df.shape[0])
    report["columns_after"] = int(cleaned_df.shape[1])
    report["duplicates_after"] = int(cleaned_df.duplicated().sum())

    if not report["actions"]:
        report["actions"].append(
            "Nothing needed fixing. Your dataset was already in good shape."
        )

    return cleaned_df, report
