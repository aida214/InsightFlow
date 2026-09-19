"""
insights.py
-----------
Turns the numbers in a dataset into short sentences that anyone can read.

Every insight is a dictionary:
    {"icon": "...", "title": "...", "text": "..."}
"""

import pandas as pd

from modules.profiler import pretty_name


def make_insight(icon, title, text, priority=5):
    """priority 1 is the most important, 9 the least. The app shows the top ones."""

    return {"icon": icon, "title": title, "text": text, "priority": priority}


def count_outliers(series):
    """Count unusually high or low values using the standard IQR rule."""

    values = series.dropna()

    if len(values) < 10:
        return 0, None, None

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    gap = q3 - q1

    if gap == 0:
        return 0, None, None

    low_limit = q1 - 1.5 * gap
    high_limit = q3 + 1.5 * gap

    outliers = values[(values < low_limit) | (values > high_limit)]

    return int(len(outliers)), float(low_limit), float(high_limit)


def generate_insights(df, date_columns=None):
    """Generate simple, human-readable insights from the dataset."""

    insights = []

    if date_columns is None:
        date_columns = []

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # --------------------------------
    # Dataset size
    # --------------------------------

    insights.append(
        make_insight(
            "📦",
            "Size of your data",
            f"You have {len(df):,} records and {len(df.columns)} columns. "
            f"{len(numeric_columns)} of those columns hold numbers and "
            f"{len(categorical_columns)} hold text or categories.",
            6,
        )
    )

    # --------------------------------
    # Missing values
    # --------------------------------

    missing = int(df.isnull().sum().sum())

    if missing == 0:
        insights.append(
            make_insight(
                "✨",
                "Nothing is missing",
                "Every cell in the table has a value, so no information had to "
                "be guessed or filled in.",
                8,
            )
        )
    else:
        worst_column = df.isnull().sum().idxmax()
        worst_count = int(df.isnull().sum().max())

        insights.append(
            make_insight(
                "⚠️",
                "Some information is missing",
                f"There are {missing:,} empty cells. The column with the most gaps "
                f"is {pretty_name(worst_column)} with {worst_count:,} empty values.",
                1,
            )
        )

    # --------------------------------
    # Duplicates
    # --------------------------------

    duplicates = int(df.duplicated().sum())

    if duplicates == 0:
        insights.append(
            make_insight(
                "🔁",
                "No repeated rows",
                "No two rows are exact copies of each other, so nothing is "
                "being counted twice.",
                8,
            )
        )
    else:
        insights.append(
            make_insight(
                "🔁",
                "Repeated rows found",
                f"{duplicates:,} rows are exact copies of another row. "
                "They can make totals and averages look bigger than they really are.",
                1,
            )
        )

    # --------------------------------
    # Numbers: average, range and outliers
    # --------------------------------

    for column in numeric_columns[:4]:

        series = df[column]
        label = pretty_name(column)

        average = series.mean()
        median = series.median()
        minimum = series.min()
        maximum = series.max()

        text = (
            f"The average is {average:,.2f} and the middle value is {median:,.2f}. "
            f"Values run from {minimum:,.2f} up to {maximum:,.2f}."
        )

        outlier_count, low_limit, high_limit = count_outliers(series)

        if outlier_count > 0:
            share = outlier_count / len(series) * 100
            text += (
                f" About {outlier_count:,} records ({share:,.1f}%) are unusually "
                f"high or low, meaning they sit outside the normal range of "
                f"{low_limit:,.2f} to {high_limit:,.2f}."
            )

        insights.append(make_insight("📊", label, text, 4))

    # --------------------------------
    # Categories
    # --------------------------------

    for column in categorical_columns[:3]:

        unique_values = df[column].nunique(dropna=True)

        if unique_values <= 1:
            continue

        if unique_values > 30:
            insights.append(
                make_insight(
                    "🔤",
                    pretty_name(column),
                    f"This column has {unique_values:,} different values, so it is "
                    "probably a name or an ID rather than a group you can compare.",
                    7,
                )
            )
            continue

        counts = df[column].value_counts()
        top_value = counts.index[0]
        top_count = int(counts.iloc[0])
        share = top_count / len(df) * 100

        text = (
            f"The most common value is '{top_value}', appearing in "
            f"{top_count:,} records ({share:,.1f}% of the data), "
            f"out of {unique_values:,} different values in total."
        )

        if share > 70:
            text += (
                " Because one value covers most of the data, this column is very "
                "unbalanced and comparisons based on it can be misleading."
            )

        insights.append(make_insight("🏆", pretty_name(column), text, 3))

    # --------------------------------
    # Strongest connection between numbers
    # --------------------------------

    if len(numeric_columns) >= 2:

        matrix = df[numeric_columns].corr()

        best_pair = None
        best_value = 0

        for first in numeric_columns:
            for second in numeric_columns:
                if first == second:
                    continue
                value = matrix.loc[first, second]
                if pd.notna(value) and abs(value) > abs(best_value):
                    best_value = value
                    best_pair = (first, second)

        if best_pair is not None and abs(best_value) >= 0.3:

            if best_value > 0:
                direction = "rise together"
            else:
                direction = "move in opposite directions"

            insights.append(
                make_insight(
                    "🔗",
                    "Strongest connection",
                    f"{pretty_name(best_pair[0])} and {pretty_name(best_pair[1])} "
                    f"{direction} (correlation {best_value:.2f}). "
                    "Keep in mind that a connection does not prove that one "
                    "causes the other, only that they move together in this data.",
                    2,
                )
            )

    # --------------------------------
    # Time range
    # --------------------------------

    for column in date_columns[:1]:

        values = pd.to_datetime(df[column], errors="coerce").dropna()

        if values.empty:
            continue

        days = (values.max() - values.min()).days

        insights.append(
            make_insight(
                "📅",
                "Time covered",
                f"Your data runs from {values.min().date()} to "
                f"{values.max().date()}, which is about {days:,} days.",
                3,
            )
        )

    # --------------------------------
    # Useless columns
    # --------------------------------

    constant_columns = [
        column for column in df.columns if df[column].nunique(dropna=True) <= 1
    ]

    if constant_columns:
        names = ", ".join(pretty_name(column) for column in constant_columns[:5])

        insights.append(
            make_insight(
                "🗑️",
                "Columns with only one value",
                f"{names} contain the same value in every row, so they add nothing "
                "to the analysis and can safely be removed.",
                2,
            )
        )

    # Most important first, so the app can show a short top list
    insights.sort(key=lambda item: item["priority"])

    return insights
