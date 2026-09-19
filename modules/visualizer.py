"""
visualizer.py
-------------
Builds the charts and writes a short, plain-English summary for each one.

The summaries are calculated from the real numbers, so the app still
explains every chart even when no AI key is available.
"""

import importlib.util

import pandas as pd
import plotly.express as px

from modules.profiler import pretty_name


# Trendlines need statsmodels. If it is missing we simply skip the line.
HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None

COLORS = ["#0F8B8D", "#1B4965", "#E4903A", "#7A5EA7", "#C2555B", "#4FA1A3"]


# =========================================================
# STYLING
# =========================================================


def style_figure(fig, x_title="", y_title=""):
    """Give every chart the same clean look."""

    fig.update_layout(
        template="plotly_white",
        height=430,
        title_font_size=20,
        title_font_color="#12232E",
        font=dict(family="Inter, Segoe UI, sans-serif", size=14, color="#3C4A57"),
        xaxis_title=x_title,
        yaxis_title=y_title,
        margin=dict(l=60, r=40, t=70, b=55),
        colorway=COLORS,
        hoverlabel=dict(bgcolor="white", font_size=13),
    )

    return fig


def strength_of_link(correlation):
    """Describe a correlation number in normal words."""

    value = abs(correlation)

    if value < 0.2:
        return "almost no connection"
    if value < 0.4:
        return "a weak connection"
    if value < 0.6:
        return "a moderate connection"
    if value < 0.8:
        return "a strong connection"

    return "a very strong connection"


# =========================================================
# 1. NUMERIC DISTRIBUTION
# =========================================================


def distribution_chart(df, column):
    """Histogram of one number column."""

    data = df[column].dropna()

    if data.empty:
        return None

    statistics = {
        "column": column,
        "records": int(len(data)),
        "average": round(float(data.mean()), 2),
        "median": round(float(data.median()), 2),
        "minimum": round(float(data.min()), 2),
        "maximum": round(float(data.max()), 2),
        "25_percentile": round(float(data.quantile(0.25)), 2),
        "75_percentile": round(float(data.quantile(0.75)), 2),
        "standard_deviation": round(float(data.std()), 2),
    }

    label = pretty_name(column)

    fig = px.histogram(
        df,
        x=column,
        nbins=20,
        marginal="box",
        title=f"How {label} is spread out",
        color_discrete_sequence=[COLORS[0]],
    )

    style_figure(fig, label, "Number of records")

    summary = (
        f"Most {label} values sit between "
        f"{statistics['25_percentile']:,.2f} and {statistics['75_percentile']:,.2f}. "
        f"The average is {statistics['average']:,.2f} and the middle value is "
        f"{statistics['median']:,.2f}, "
        f"with everything falling between {statistics['minimum']:,.2f} and "
        f"{statistics['maximum']:,.2f}."
    )

    gap = statistics["average"] - statistics["median"]
    spread = statistics["standard_deviation"]

    if spread > 0 and gap > 0.5 * spread:
        summary += " A few unusually high values are pulling the average upwards."
    elif spread > 0 and gap < -0.5 * spread:
        summary += " A few unusually low values are pulling the average downwards."

    return {
        "title": f"How {label} is spread out",
        "description": (
            f"Each bar counts how many records fall into a range of {label}. "
            "Taller bars mean more records."
        ),
        "chart_type": "Numeric distribution",
        "statistics": statistics,
        "summary": summary,
        "figure": fig,
    }


# =========================================================
# 2. AVERAGE NUMBER BY CATEGORY
# =========================================================


def category_average_chart(df, category, value):
    """Average of a number column for each group."""

    grouped = (
        df.groupby(category)[value]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    if grouped.empty:
        return None

    category_data = []

    for _, row in grouped.iterrows():
        category_data.append(
            {
                "category": str(row[category]),
                "average": round(float(row[value]), 2),
            }
        )

    highest_row = grouped.iloc[0]
    lowest_row = grouped.iloc[-1]

    statistics = {
        "category_column": category,
        "value_column": value,
        "highest_category": str(highest_row[category]),
        "highest_average": round(float(highest_row[value]), 2),
        "lowest_category": str(lowest_row[category]),
        "lowest_average": round(float(lowest_row[value]), 2),
        "categories": category_data,
    }

    category_label = pretty_name(category)
    value_label = pretty_name(value)

    fig = px.bar(
        grouped,
        x=value,
        y=category,
        orientation="h",
        title=f"Average {value_label} for each {category_label}",
        color_discrete_sequence=[COLORS[0]],
    )

    fig.update_layout(yaxis=dict(categoryorder="total ascending"))

    style_figure(fig, f"Average {value_label}", category_label)

    difference = statistics["highest_average"] - statistics["lowest_average"]

    summary = (
        f"{statistics['highest_category']} has the highest average {value_label} "
        f"({statistics['highest_average']:,.2f}), while "
        f"{statistics['lowest_category']} has the lowest "
        f"({statistics['lowest_average']:,.2f}). "
        f"That is a gap of {difference:,.2f}"
    )

    if statistics["lowest_average"] > 0:
        percent = difference / statistics["lowest_average"] * 100
        summary += f", which means the top group is about {percent:,.0f}% higher."
    else:
        summary += "."

    return {
        "title": f"Average {value_label} for each {category_label}",
        "description": (
            f"Longer bars mean a higher average {value_label} for that "
            f"{category_label}. Only the top 10 groups are shown."
        ),
        "chart_type": "Category comparison",
        "statistics": statistics,
        "summary": summary,
        "figure": fig,
    }


# =========================================================
# 3. HOW OFTEN EACH CATEGORY APPEARS
# =========================================================


def category_count_chart(df, category):
    """Count how many records belong to each group."""

    counts = df[category].value_counts().head(10).reset_index()
    counts.columns = [category, "count"]

    if counts.empty:
        return None

    category_data = []

    for _, row in counts.iterrows():
        category_data.append(
            {
                "category": str(row[category]),
                "records": int(row["count"]),
            }
        )

    most_common = counts.iloc[0]
    total = int(len(df))
    share = int(most_common["count"]) / total * 100 if total > 0 else 0

    statistics = {
        "category_column": category,
        "total_records": total,
        "most_common_category": str(most_common[category]),
        "most_common_count": int(most_common["count"]),
        "most_common_share_percent": round(share, 1),
        "categories": category_data,
    }

    label = pretty_name(category)

    fig = px.bar(
        counts,
        x="count",
        y=category,
        orientation="h",
        title=f"How many records for each {label}",
        color_discrete_sequence=[COLORS[1]],
    )

    fig.update_layout(yaxis=dict(categoryorder="total ascending"))

    style_figure(fig, "Number of records", label)

    summary = (
        f"{statistics['most_common_category']} is the most common {label}. "
        f"It appears {statistics['most_common_count']:,} times, which is about "
        f"{statistics['most_common_share_percent']:,.1f}% of all "
        f"{total:,} records."
    )

    if len(counts) > 1:
        smallest = counts.iloc[-1]
        summary += (
            f" The smallest group shown is {smallest[category]} with only "
            f"{int(smallest['count']):,} records."
        )

    return {
        "title": f"How many records for each {label}",
        "description": (
            f"This counts how often each {label} appears in your data. "
            "Only the 10 most common groups are shown."
        ),
        "chart_type": "Category frequency",
        "statistics": statistics,
        "summary": summary,
        "figure": fig,
    }


# =========================================================
# 4. RELATIONSHIP BETWEEN TWO NUMBERS
# =========================================================


def relationship_chart(df, x, y):
    """Scatter plot showing whether two numbers move together."""

    data = df[[x, y]].dropna()

    if data.empty or len(data) < 3:
        return None

    correlation = data[x].corr(data[y])

    if pd.isna(correlation):
        return None

    statistics = {
        "x_variable": x,
        "y_variable": y,
        "records": int(len(data)),
        "correlation": round(float(correlation), 3),
        "x_average": round(float(data[x].mean()), 2),
        "y_average": round(float(data[y].mean()), 2),
        "x_minimum": round(float(data[x].min()), 2),
        "x_maximum": round(float(data[x].max()), 2),
        "y_minimum": round(float(data[y].min()), 2),
        "y_maximum": round(float(data[y].max()), 2),
    }

    x_label = pretty_name(x)
    y_label = pretty_name(y)

    if HAS_STATSMODELS:
        fig = px.scatter(
            data,
            x=x,
            y=y,
            title=f"{x_label} compared with {y_label}",
            trendline="ols",
            trendline_color_override=COLORS[2],
            color_discrete_sequence=[COLORS[0]],
            opacity=0.7,
        )
    else:
        fig = px.scatter(
            data,
            x=x,
            y=y,
            title=f"{x_label} compared with {y_label}",
            color_discrete_sequence=[COLORS[0]],
            opacity=0.7,
        )

    style_figure(fig, x_label, y_label)

    if correlation > 0:
        direction = f"when {x_label} goes up, {y_label} usually goes up as well"
    else:
        direction = f"when {x_label} goes up, {y_label} usually goes down"

    summary = (
        f"There is {strength_of_link(correlation)} between {x_label} and {y_label} "
        f"(correlation {statistics['correlation']}). In simple terms, {direction}. "
        f"This is based on {statistics['records']:,} records."
    )

    if abs(correlation) < 0.2:
        summary += " Knowing one of them tells you very little about the other."

    return {
        "title": f"{x_label} compared with {y_label}",
        "description": (
            "Every dot is one record. If the dots form a clear upward or downward "
            "pattern, the two values are connected."
        ),
        "chart_type": "Relationship between two numeric variables",
        "statistics": statistics,
        "summary": summary,
        "figure": fig,
    }


# =========================================================
# 5. CORRELATION HEATMAP
# =========================================================


def correlation_chart(df, numeric_columns):
    """Show which number columns are connected to each other."""

    data = df[numeric_columns].dropna()

    if len(data) < 3:
        return None

    matrix = data.corr().round(2)

    if matrix.empty:
        return None

    labels = [pretty_name(column) for column in matrix.columns]

    fig = px.imshow(
        matrix.values,
        x=labels,
        y=labels,
        text_auto=True,
        zmin=-1,
        zmax=1,
        color_continuous_scale=["#C2555B", "#F3F5F7", "#0F8B8D"],
        title="Which numbers move together",
    )

    style_figure(fig, "", "")
    fig.update_layout(height=460)

    # Find the strongest pair (ignoring a column compared with itself)
    strongest_pair = None
    strongest_value = 0

    for first in matrix.columns:
        for second in matrix.columns:
            if first == second:
                continue
            value = matrix.loc[first, second]
            if abs(value) > abs(strongest_value):
                strongest_value = value
                strongest_pair = (first, second)

    statistics = {
        "columns": list(matrix.columns),
        "correlations": matrix.to_dict(),
    }

    if strongest_pair is not None:
        summary = (
            f"The strongest connection is between {pretty_name(strongest_pair[0])} "
            f"and {pretty_name(strongest_pair[1])} (correlation {strongest_value}), "
            f"which is {strength_of_link(strongest_value)}. "
            "Green squares mean the two values rise together, red squares mean one "
            "rises while the other falls."
        )
    else:
        summary = "No clear connection was found between the number columns."

    return {
        "title": "Which numbers move together",
        "description": (
            "Each square compares two number columns. Values close to 1 or -1 mean "
            "a strong connection, values close to 0 mean no connection."
        ),
        "chart_type": "Correlation heatmap",
        "statistics": statistics,
        "summary": summary,
        "figure": fig,
    }


# =========================================================
# 6. CHANGE OVER TIME
# =========================================================


def time_chart(df, date_column, value_column):
    """Show how a number changed over time."""

    data = df[[date_column, value_column]].dropna().copy()

    if data.empty:
        return None

    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.dropna()

    if len(data) < 4:
        return None

    days = (data[date_column].max() - data[date_column].min()).days

    if days > 730:
        period = "M"
        period_name = "month"
    elif days > 120:
        period = "W"
        period_name = "week"
    else:
        period = "D"
        period_name = "day"

    grouped = (
        data.groupby(data[date_column].dt.to_period(period))[value_column]
        .mean()
        .reset_index()
    )

    grouped[date_column] = grouped[date_column].dt.to_timestamp()

    if len(grouped) < 3:
        return None

    date_label = pretty_name(date_column)
    value_label = pretty_name(value_column)

    fig = px.line(
        grouped,
        x=date_column,
        y=value_column,
        markers=True,
        title=f"Average {value_label} over time",
        color_discrete_sequence=[COLORS[0]],
    )

    style_figure(fig, date_label, f"Average {value_label}")

    first_value = float(grouped[value_column].iloc[0])
    last_value = float(grouped[value_column].iloc[-1])

    best_row = grouped.loc[grouped[value_column].idxmax()]

    statistics = {
        "date_column": date_column,
        "value_column": value_column,
        "grouped_by": period_name,
        "periods": int(len(grouped)),
        "first_value": round(first_value, 2),
        "last_value": round(last_value, 2),
        "highest_period": str(best_row[date_column].date()),
        "highest_value": round(float(best_row[value_column]), 2),
    }

    if first_value != 0:
        change = (last_value - first_value) / abs(first_value) * 100
        movement = "risen" if change >= 0 else "fallen"
        change_text = (
            f" Overall it has {movement} by about {abs(change):,.0f}% "
            "from the first period to the last."
        )
    else:
        change_text = ""

    summary = (
        f"Grouped by {period_name}, the average {value_label} started at "
        f"{first_value:,.2f} and ended at {last_value:,.2f}. "
        f"The highest point was {statistics['highest_value']:,.2f} on "
        f"{statistics['highest_period']}.{change_text}"
    )

    return {
        "title": f"Average {value_label} over time",
        "description": (
            f"Values are grouped by {period_name} so the overall trend is easier "
            "to see than the raw day to day movement."
        ),
        "chart_type": "Trend over time",
        "statistics": statistics,
        "summary": summary,
        "figure": fig,
    }


# =========================================================
# MAIN FUNCTION
# =========================================================


def generate_visualizations(df, date_columns=None):
    """
    Create a set of useful charts for any dataset.
    Returns a list of dictionaries the app can display.
    """

    charts = []

    if date_columns is None:
        date_columns = []

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Skip text columns where nearly every row is different (names, IDs)
    useful_categories = []

    for column in categorical_columns:
        unique_values = df[column].nunique(dropna=True)
        if 1 < unique_values <= 30:
            useful_categories.append(column)

    # 1. Distribution of the first few number columns
    for column in numeric_columns[:3]:
        chart = distribution_chart(df, column)
        if chart:
            charts.append(chart)

    # 2. Average number by category
    if useful_categories and numeric_columns:
        chart = category_average_chart(df, useful_categories[0], numeric_columns[0])
        if chart:
            charts.append(chart)

    # 3. How often each category appears
    for column in useful_categories[:2]:
        chart = category_count_chart(df, column)
        if chart:
            charts.append(chart)

    # 4. Relationship between the two most connected number columns
    if len(numeric_columns) >= 2:

        best_pair = (numeric_columns[0], numeric_columns[1])
        best_score = -1

        matrix = df[numeric_columns].corr()

        for first in numeric_columns:
            for second in numeric_columns:
                if first == second:
                    continue
                value = matrix.loc[first, second]
                if pd.notna(value) and abs(value) > best_score:
                    best_score = abs(value)
                    best_pair = (first, second)

        chart = relationship_chart(df, best_pair[0], best_pair[1])
        if chart:
            charts.append(chart)

    # 5. Correlation heatmap
    if len(numeric_columns) >= 3:
        chart = correlation_chart(df, numeric_columns)
        if chart:
            charts.append(chart)

    # 6. Trend over time
    if date_columns and numeric_columns:
        chart = time_chart(df, date_columns[0], numeric_columns[0])
        if chart:
            charts.append(chart)

    return charts


# =========================================================
# SINGLE COLUMN CHART (used by the column explorer)
# =========================================================


def make_column_chart(df, column):
    """Draw the most suitable chart for one single column."""

    label = pretty_name(column)
    series = df[column].dropna()

    if series.empty:
        return None

    if pd.api.types.is_numeric_dtype(series):
        fig = px.histogram(
            df,
            x=column,
            nbins=20,
            title=f"How {label} is spread out",
            color_discrete_sequence=[COLORS[0]],
        )
        return style_figure(fig, label, "Number of records")

    if pd.api.types.is_datetime64_any_dtype(series):
        counts = series.dt.to_period("M").value_counts().sort_index().reset_index()
        counts.columns = [column, "count"]
        counts[column] = counts[column].dt.to_timestamp()

        fig = px.bar(
            counts,
            x=column,
            y="count",
            title=f"Records per month in {label}",
            color_discrete_sequence=[COLORS[1]],
        )
        return style_figure(fig, label, "Number of records")

    counts = series.value_counts().head(15).reset_index()
    counts.columns = [column, "count"]

    fig = px.bar(
        counts,
        x="count",
        y=column,
        orientation="h",
        title=f"Most common values in {label}",
        color_discrete_sequence=[COLORS[3]],
    )

    fig.update_layout(yaxis=dict(categoryorder="total ascending"))

    return style_figure(fig, "Number of records", label)


# =========================================================
# CUSTOM CHART BUILDER
# =========================================================


def build_custom_chart(df, chart_type, x_column, y_column=None, color_column=None):
    """Build the chart the user chose in the 'Build your own chart' section."""

    x_label = pretty_name(x_column)
    y_label = pretty_name(y_column) if y_column else ""

    color = color_column if color_column not in (None, "None") else None

    if chart_type == "Bar chart":

        if y_column:
            grouped = df.groupby(x_column)[y_column].mean().reset_index()
            fig = px.bar(
                grouped,
                x=x_column,
                y=y_column,
                color=color,
                title=f"Average {y_label} by {x_label}",
                color_discrete_sequence=COLORS,
            )
        else:
            counts = df[x_column].value_counts().head(20).reset_index()
            counts.columns = [x_column, "count"]
            fig = px.bar(
                counts,
                x=x_column,
                y="count",
                title=f"Number of records by {x_label}",
                color_discrete_sequence=COLORS,
            )
            y_label = "Number of records"

    elif chart_type == "Line chart":

        data = df.sort_values(by=x_column)
        fig = px.line(
            data,
            x=x_column,
            y=y_column,
            color=color,
            markers=True,
            title=f"{y_label} over {x_label}",
            color_discrete_sequence=COLORS,
        )

    elif chart_type == "Scatter plot":

        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            color=color,
            opacity=0.7,
            title=f"{x_label} compared with {y_label}",
            color_discrete_sequence=COLORS,
        )

    elif chart_type == "Box plot":

        fig = px.box(
            df,
            x=x_column,
            y=y_column,
            color=color,
            title=f"Spread of {y_label} by {x_label}",
            color_discrete_sequence=COLORS,
        )

    elif chart_type == "Pie chart":

        counts = df[x_column].value_counts().head(10).reset_index()
        counts.columns = [x_column, "count"]
        fig = px.pie(
            counts,
            names=x_column,
            values="count",
            hole=0.45,
            title=f"Share of records by {x_label}",
            color_discrete_sequence=COLORS,
        )
        return style_figure(fig)

    else:  # Histogram

        fig = px.histogram(
            df,
            x=x_column,
            color=color,
            nbins=20,
            title=f"How {x_label} is spread out",
            color_discrete_sequence=COLORS,
        )
        y_label = "Number of records"

    return style_figure(fig, x_label, y_label)


# =========================================================
# LARGE DATASETS
# =========================================================


MAX_CHART_ROWS = 50000


def sample_for_charts(df, limit=MAX_CHART_ROWS):
    """
    Charts stay responsive on big files by drawing a random sample.
    Returns the data to chart and whether sampling happened, so the app
    can tell the user instead of quietly showing part of the picture.
    """

    if len(df) > limit:
        return df.sample(limit, random_state=42), True

    return df, False
