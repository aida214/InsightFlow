"""
InsightFlow
-----------
Upload a dataset and get it cleaned, explained, charted and predicted,
in language anyone can understand.

Run it with:  streamlit run app.py
"""

import io

import numpy as np
import pandas as pd
import streamlit as st

from modules.profiler import get_dataset_profile, get_column_table, pretty_name
from modules.quality import (
    missing_report,
    type_report,
    outlier_report,
    quality_score,
)
from modules.cleaner import clean_dataset, make_names_unique
from modules.visualizer import (
    generate_visualizations,
    build_custom_chart,
    make_column_chart,
    sample_for_charts,
)
from modules.insights import generate_insights
from modules.predictor import (
    sklearn_available,
    get_possible_targets,
    train_model,
    predict_one,
    predict_dataset,
    make_importance_chart,
    make_actual_vs_predicted_chart,
    make_confusion_chart,
    forecast_trend,
)
from modules.ai_insights import (
    ai_available,
    generate_ai_insight,
    answer_question,
    build_data_summary,
)
from modules.report_generator import build_text_summary, build_html_report


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="InsightFlow",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# STYLING
# =========================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
    color: #12232E;
    letter-spacing: -0.3px;
}

.hero {
    background: linear-gradient(120deg, #12232E 0%, #17494D 58%, #0F8B8D 100%);
    padding: 34px 38px;
    border-radius: 18px;
    margin-bottom: 22px;
}

.hero h1 {
    color: #FFFFFF;
    font-size: 2.4rem;
    margin: 0 0 8px 0;
}

.hero p {
    color: #CFE3E4;
    font-size: 1.05rem;
    line-height: 1.6;
    margin: 0;
    max-width: 62ch;
}

.stat-card {
    background: #FFFFFF;
    border: 1px solid #E4E9EE;
    border-left: 4px solid #0F8B8D;
    border-radius: 12px;
    padding: 16px 18px;
    height: 100%;
}

.stat-card .value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.75rem;
    font-weight: 600;
    color: #12232E;
    line-height: 1.2;
}

.stat-card .label {
    font-size: 0.85rem;
    color: #5A6B7B;
    margin-top: 2px;
}

.stat-card .note {
    font-size: 0.78rem;
    color: #8A99A6;
    margin-top: 6px;
}

.insight-card {
    background: #FFFFFF;
    border: 1px solid #E4E9EE;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
}

.insight-card h4 {
    margin: 0 0 6px 0;
    font-size: 1rem;
}

.insight-card p {
    margin: 0;
    color: #44525E;
    line-height: 1.6;
    font-size: 0.95rem;
}

.takeaway {
    background: #F0F7F7;
    border-left: 4px solid #0F8B8D;
    border-radius: 10px;
    padding: 14px 18px;
    color: #22333F;
    line-height: 1.65;
    font-size: 0.97rem;
}

.ai-box {
    background: #FCF5EC;
    border-left: 4px solid #E4903A;
    border-radius: 10px;
    padding: 14px 18px;
    color: #3B2F22;
    line-height: 1.65;
    font-size: 0.97rem;
    margin-top: 10px;
}

.score-card {
    background: linear-gradient(120deg, #12232E 0%, #17494D 100%);
    border-radius: 14px;
    padding: 22px 26px;
    color: #FFFFFF;
}

.score-card .score {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
}

.score-card .label {
    color: #CFE3E4;
    font-size: 1rem;
    margin-top: 4px;
}

.file-card {
    background: #FFFFFF;
    border: 1px solid #E4E9EE;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    padding: 10px 18px;
    border-radius: 10px 10px 0 0;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SMALL HELPERS
# =========================================================


def show_hero(subtitle):
    st.markdown(
        f"""
<div class="hero">
    <h1>📊 InsightFlow</h1>
    <p>{subtitle}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def stat_card(container, value, label, note=""):
    note_html = f"<div class='note'>{note}</div>" if note else ""

    container.markdown(
        f"""
<div class="stat-card">
    <div class="value">{value}</div>
    <div class="label">{label}</div>
    {note_html}
</div>
""",
        unsafe_allow_html=True,
    )


def insight_card(icon, title, text):
    st.markdown(
        f"""
<div class="insight-card">
    <h4>{icon} {title}</h4>
    <p>{text}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def takeaway_box(text):
    st.markdown(f"<div class='takeaway'>📌 {text}</div>", unsafe_allow_html=True)


def ai_box(text):
    st.markdown(f"<div class='ai-box'>🤖 {text}</div>", unsafe_allow_html=True)


def readable_size(byte_count):
    if byte_count < 1024:
        return f"{byte_count} B"
    if byte_count < 1024 * 1024:
        return f"{byte_count / 1024:,.1f} KB"
    return f"{byte_count / (1024 * 1024):,.1f} MB"


# =========================================================
# CACHED WORK
# =========================================================


@st.cache_data(show_spinner=False)
def load_data(file_bytes, file_name):
    """Read an uploaded CSV or Excel file into a dataframe."""

    buffer = io.BytesIO(file_bytes)

    if file_name.lower().endswith(".csv"):
        try:
            data = pd.read_csv(buffer)
        except UnicodeDecodeError:
            buffer.seek(0)
            data = pd.read_csv(buffer, encoding="latin-1")
    else:
        data = pd.read_excel(buffer)

    # Two columns with the same name break pandas, so number the repeats
    # straight away, before anything else touches the data.
    original_names = data.columns.tolist()
    unique_names = make_names_unique(original_names)
    data.columns = unique_names

    return data, unique_names != original_names


@st.cache_data(show_spinner=False)
def build_sample_data():
    """A small, deliberately messy dataset so the app can be tried instantly."""

    random = np.random.default_rng(7)
    size = 400

    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        random.integers(0, 365, size), unit="D"
    )

    cities = random.choice(
        ["Peshawar", "Lahore", "Karachi", "Islamabad", "Quetta"],
        size,
        p=[0.25, 0.25, 0.20, 0.20, 0.10],
    )

    products = random.choice(["Latte", "Espresso", "Cold Brew", "Tea", "Pastry"], size)

    units = random.integers(1, 12, size)
    price = np.round(random.normal(4.5, 1.2, size).clip(1.5, None), 2)
    delivery_days = random.integers(1, 8, size)
    rating = np.round((5.4 - 0.22 * delivery_days + random.normal(0, 0.4, size)), 1)
    rating = rating.clip(1, 5)

    data = pd.DataFrame(
        {
            "Order Date": dates.astype(str),
            "City": cities,
            "Product": products,
            "Units Sold": units,
            "Unit Price": [f"$ {value:,.2f}" for value in price],
            "Revenue": np.round(units * price, 2),
            "Delivery Days": delivery_days,
            "Customer Rating": rating,
        }
    )

    # Add realistic mess: gaps, stray spaces and repeated rows
    data.loc[random.choice(size, 25, replace=False), "Customer Rating"] = np.nan
    data.loc[random.choice(size, 12, replace=False), "City"] = None

    data["City"] = data["City"].apply(
        lambda value: value + " "
        if isinstance(value, str) and random.random() < 0.2
        else value
    )

    data = pd.concat([data, data.sample(15, random_state=3)], ignore_index=True)

    return data


@st.cache_data(show_spinner=False)
def run_cleaning(df, duplicates, names, trim, numbers, dates, fill):
    return clean_dataset(
        df,
        remove_duplicates=duplicates,
        tidy_column_names=names,
        trim_text=trim,
        convert_numbers=numbers,
        convert_dates=dates,
        fill_missing=fill,
    )


@st.cache_data(show_spinner=False)
def to_excel_bytes(df):
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Clean data")

    return buffer.getvalue()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 📊 InsightFlow")
    st.caption("Turn raw data into clear insights, visualizations and predictions.")

    st.divider()

    source = st.radio(
        "Where is your data?",
        ["Upload a file", "Use the sample data"],
        key="source",
    )

    uploaded_file = None

    if source == "Upload a file":
        uploaded_file = st.file_uploader(
            "CSV or Excel file",
            type=["csv", "xlsx", "xls"],
        )

    st.divider()

    st.markdown("#### Cleaning steps")
    st.caption("Switch any step off if you would rather keep the raw values.")

    option_duplicates = st.checkbox("Remove repeated rows", value=True)
    option_names = st.checkbox("Tidy the column names", value=True)
    option_trim = st.checkbox("Remove extra spaces in text", value=True)
    option_numbers = st.checkbox("Convert numbers stored as text", value=True)
    option_dates = st.checkbox("Detect date columns", value=True)
    option_fill = st.checkbox("Fill in missing values", value=True)

    st.divider()

    st.markdown("#### AI explanations")

    if ai_available():
        use_ai = st.checkbox("Let AI explain the charts", value=False)
        st.caption("An API key was found, so the AI features are ready.")
    else:
        use_ai = False
        st.caption(
            "AI insights are unavailable. Add your API key to enable AI-powered "
            "explanations. Everything else works without it."
        )


# =========================================================
# GET THE DATA
# =========================================================

df = None
file_name = ""
file_size = ""

if source == "Use the sample data":

    df = build_sample_data()
    file_name = "sample_coffee_sales.csv"
    file_size = "built in"
    repeated_names = False

    if st.session_state.get("current_file") != file_name:
        st.session_state["current_file"] = file_name
        st.session_state["analyzed"] = True
        st.session_state.pop("model_result", None)

elif uploaded_file is not None:

    try:
        df, repeated_names = load_data(uploaded_file.getvalue(), uploaded_file.name)
        file_name = uploaded_file.name
        file_size = readable_size(len(uploaded_file.getvalue()))

    except Exception as error:
        show_hero("That file could not be read.")
        st.error(
            "We couldn't read this file. Check that it is a valid CSV or Excel "
            "file and that it is not open in another program."
        )
        st.caption(f"Technical detail: {error}")
        st.stop()

    if st.session_state.get("current_file") != file_name:
        st.session_state["current_file"] = file_name
        st.session_state["analyzed"] = False
        st.session_state.pop("model_result", None)


# --------------------------------
# Welcome screen
# --------------------------------

if df is None:

    show_hero(
        "Upload a spreadsheet and InsightFlow cleans it, explains what is inside, "
        "draws the charts that matter and predicts what happens next. "
        "No statistics knowledge needed."
    )

    left, middle, right = st.columns(3)

    stat_card(left, "🧹", "Clean", "Repeats removed, gaps filled, types fixed")
    stat_card(middle, "📈", "Understand", "Charts and findings written in plain English")
    stat_card(right, "🔮", "Predict", "Models and forecasts, explained simply")

    st.write("")
    st.info(
        "👈 Choose a file in the sidebar, or pick **Use the sample data** to see "
        "how everything works straight away."
    )

    st.stop()


if df.empty or df.shape[1] == 0:
    show_hero("That file has no data in it.")
    st.warning("The file loaded, but there are no rows or columns to analyse.")
    st.stop()


# --------------------------------
# Upload confirmation screen
# --------------------------------

if not st.session_state.get("analyzed", False):

    show_hero("Turn raw data into clear insights, visualizations and predictions.")

    st.markdown(
        f"""
<div class="file-card">
    <h4>📄 {file_name}</h4>
    <p style="color:#5A6B7B;margin:0;">
        {file_size} &middot; {df.shape[0]:,} rows &middot; {df.shape[1]:,} columns
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.dataframe(df.head(5), use_container_width=True)

    if st.button("Analyze my data →", type="primary", key="analyze_button"):
        st.session_state["analyzed"] = True
        st.rerun()

    st.stop()


# =========================================================
# CLEAN AND PROFILE
# =========================================================

with st.spinner("Reading, checking and cleaning your data..."):

    try:
        cleaned_df, report = run_cleaning(
            df,
            option_duplicates,
            option_names,
            option_trim,
            option_numbers,
            option_dates,
            option_fill,
        )

    except Exception as error:
        show_hero("The cleaning step could not finish.")
        st.error(
            "InsightFlow couldn't clean this dataset. You can switch cleaning "
            "steps off in the sidebar and try again."
        )
        st.caption(f"Technical detail: {error}")
        st.stop()

    raw_profile = get_dataset_profile(df)
    profile = get_dataset_profile(cleaned_df)
    score = quality_score(df)

date_columns = profile["datetime_columns"]
numeric_columns = profile["numeric_columns"]
category_columns = profile["categorical_columns"]


show_hero(
    f"<b>{file_name}</b> &middot; {raw_profile['rows']:,} records across "
    f"{raw_profile['columns']} columns. "
    "Here is everything InsightFlow found inside it."
)


(
    tab_overview,
    tab_quality,
    tab_charts,
    tab_insights,
    tab_predict,
    tab_export,
) = st.tabs(
    [
        "📋 Overview",
        "🧹 Data quality",
        "📊 Visualizations",
        "💡 Insights",
        "🤖 Prediction",
        "📥 Export",
    ]
)


# =========================================================
# TAB 1: OVERVIEW
# =========================================================

with tab_overview:

    st.subheader("Your dataset at a glance")

    one, two, three, four = st.columns(4)

    stat_card(one, f"{raw_profile['rows']:,}", "Records", "One row per record")
    stat_card(two, f"{raw_profile['columns']:,}", "Columns", "Things measured")
    stat_card(
        three,
        f"{raw_profile['missing_values']:,}",
        "Empty cells",
        f"{raw_profile['missing_percent']}% of the whole table",
    )
    stat_card(
        four,
        f"{raw_profile['duplicate_rows']:,}",
        "Repeated rows",
        "Exact copies of another row",
    )

    st.write("")

    five, six, seven, eight = st.columns(4)

    stat_card(five, f"{len(numeric_columns)}", "Number columns", "Things you can total")
    stat_card(six, f"{len(category_columns)}", "Category columns", "Things you can group by")
    stat_card(seven, f"{len(date_columns)}", "Date columns", "Used for trends")
    stat_card(eight, f"{raw_profile['memory_kb']:,.0f} KB", "Size in memory")

    st.write("")

    st.markdown("#### What this means")

    messages = [
        f"Each of the {raw_profile['rows']:,} rows is one record, and each of the "
        f"{raw_profile['columns']} columns describes something about it."
    ]

    if raw_profile["missing_values"] == 0:
        messages.append("Nothing is missing, so the data is complete.")
    else:
        messages.append(
            f"{raw_profile['missing_values']:,} cells are empty "
            f"({raw_profile['missing_percent']}% of the table). "
            "The Data quality tab shows which columns and what was done about it."
        )

    if raw_profile["duplicate_rows"] > 0:
        messages.append(
            f"{raw_profile['duplicate_rows']:,} rows are exact copies of another row, "
            "which would make your totals too high if left in."
        )

    if raw_profile["id_like_columns"]:
        names = ", ".join(pretty_name(c) for c in raw_profile["id_like_columns"][:3])
        messages.append(
            f"{names} has a different value in every row, so it is most likely an "
            "identity code rather than something to analyse."
        )

    st.write(" ".join(messages))

    st.divider()

    st.markdown("#### A look at the actual rows")

    row_count = st.slider("How many rows to show", 5, 50, 10, key="preview_rows")
    st.dataframe(df.head(row_count), use_container_width=True)

    st.divider()

    st.markdown("#### What each column contains")
    st.caption(
        "'Different values' counts how many unique entries a column has. "
        "A low number usually means a category, a high number means a measurement "
        "or an ID."
    )

    st.dataframe(get_column_table(df), use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("#### Look at one column closely")

    chosen_column = st.selectbox(
        "Pick a column",
        list(cleaned_df.columns),
        key="explore_column",
        format_func=pretty_name,
    )

    series = cleaned_df[chosen_column]

    box_one, box_two, box_three = st.columns(3)

    if pd.api.types.is_numeric_dtype(series):
        stat_card(box_one, f"{series.mean():,.2f}", "Average")
        stat_card(box_two, f"{series.median():,.2f}", "Middle value")
        stat_card(
            box_three,
            f"{series.min():,.2f} – {series.max():,.2f}",
            "Lowest to highest",
        )
    elif pd.api.types.is_datetime64_any_dtype(series):
        stat_card(box_one, str(series.min().date()), "Earliest date")
        stat_card(box_two, str(series.max().date()), "Latest date")
        stat_card(box_three, f"{(series.max() - series.min()).days:,}", "Days covered")
    else:
        top_values = series.value_counts()
        stat_card(box_one, f"{series.nunique(dropna=True):,}", "Different values")
        stat_card(
            box_two,
            str(top_values.index[0]) if len(top_values) > 0 else "-",
            "Most common value",
        )
        stat_card(box_three, f"{int(series.isnull().sum()):,}", "Empty cells")

    st.write("")

    try:
        column_figure = make_column_chart(cleaned_df, chosen_column)

        if column_figure is not None:
            st.plotly_chart(column_figure, use_container_width=True)

    except Exception:
        st.caption("This column cannot be charted on its own.")


# =========================================================
# TAB 2: DATA QUALITY
# =========================================================

with tab_quality:

    st.subheader("How healthy is this data?")

    score_column, reason_column = st.columns([1, 2])

    with score_column:
        st.markdown(
            f"""
<div class="score-card">
    <div class="score">{score['score']}<span style="font-size:1.2rem;">/100</span></div>
    <div class="label">{score['label']}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with reason_column:
        st.markdown("**Why this score**")
        for reason in score["reasons"]:
            st.markdown(f"- {reason}")
        st.caption(
            "The score drops for empty cells, repeated rows and columns whose "
            "contents do not match their type. It is a guide, not a verdict."
        )

    st.divider()

    # --------------------------------
    # Missing values
    # --------------------------------

    st.markdown("#### Missing values")

    missing_table = missing_report(df)

    if missing_table.empty:
        st.success("No empty cells were found anywhere in this dataset.")
    else:
        st.dataframe(missing_table, use_container_width=True, hide_index=True)
        st.caption(
            "Filling a gap with the median or the most common value keeps the "
            "shape of the data honest, but those cells are estimates, not facts."
        )

    # --------------------------------
    # Duplicates
    # --------------------------------

    st.markdown("#### Repeated rows")

    if raw_profile["duplicate_rows"] == 0:
        st.success("No two rows are exact copies of each other.")

    elif report.get("duplicates_kept", 0) > 0:
        st.info(
            f"{raw_profile['duplicate_rows']:,} repeated rows detected, and kept. "
            "Every column here holds only a few different values, so identical "
            "rows are almost certainly separate real records rather than "
            "accidental copies. Deleting them would throw away most of your data."
        )

    elif report["duplicates_removed"] > 0:
        st.warning(
            f"{report['duplicates_removed']:,} repeated rows detected and removed. "
            "Switch off 'Remove repeated rows' in the sidebar to keep them."
        )

    else:
        st.warning(
            f"{raw_profile['duplicate_rows']:,} repeated rows detected. They are "
            "being kept because the 'Remove repeated rows' step is switched off."
        )

        with st.expander("Show me the repeated rows"):
            st.dataframe(
                df[df.duplicated(keep=False)].head(20), use_container_width=True
            )

    # --------------------------------
    # Type problems
    # --------------------------------

    if repeated_names:
        st.info(
            "Two or more columns shared the same name. The repeats were numbered "
            "when the file was read, so they can be told apart."
        )

    st.markdown("#### Columns stored the wrong way")

    type_table = type_report(df)

    if type_table.empty:
        st.success("Every column holds the kind of value it should.")
    else:
        st.dataframe(type_table, use_container_width=True, hide_index=True)

    # --------------------------------
    # Outliers
    # --------------------------------

    st.markdown("#### Unusual values")

    outlier_table = outlier_report(cleaned_df)

    if outlier_table.empty:
        st.success("No unusually high or low numbers stand out.")
    else:
        st.dataframe(outlier_table, use_container_width=True, hide_index=True)
        st.info(
            "These values are unusual, but unusual does not mean incorrect. "
            "A single very large order can be completely real. Nothing is deleted "
            "automatically, so look at these rows before deciding."
        )

        with st.expander("What counts as unusual?"):
            st.write(
                "InsightFlow uses the IQR rule. It takes the middle half of the "
                "values, measures how wide that middle is, and flags anything "
                "sitting more than one and a half times that width outside it."
            )

    st.divider()

    # --------------------------------
    # Before and after
    # --------------------------------

    st.markdown("#### Before and after cleaning")

    comparison = pd.DataFrame(
        {
            "": ["Rows", "Columns", "Empty cells", "Repeated rows"],
            "Before": [
                f"{report['rows_before']:,}",
                f"{report['columns_before']:,}",
                f"{report['missing_before']:,}",
                f"{report['duplicates_before']:,}",
            ],
            "After": [
                f"{report['rows_after']:,}",
                f"{report['columns_after']:,}",
                f"{report['missing_after']:,}",
                f"{report.get('duplicates_after', 0):,}",
            ],
        }
    )

    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("**Changes made**")

    for action in report["actions"]:
        st.markdown(f"- ✓ {action}")

    if report["name_changes"]:
        with st.expander("Which column names changed"):
            st.dataframe(
                pd.DataFrame(
                    report["name_changes"], columns=["Original name", "New name"]
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    st.markdown("#### Your cleaned data")
    st.dataframe(cleaned_df.head(15), use_container_width=True)

    left, right = st.columns(2)

    left.download_button(
        label="⬇️ Download cleaned data (CSV)",
        data=cleaned_df.to_csv(index=False),
        file_name="insightflow_cleaned_data.csv",
        mime="text/csv",
        use_container_width=True,
        key="clean_csv_quality",
    )

    try:
        right.download_button(
            label="⬇️ Download cleaned data (Excel)",
            data=to_excel_bytes(cleaned_df),
            file_name="insightflow_cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="clean_xlsx_quality",
        )
    except Exception:
        right.caption("Install openpyxl to download an Excel copy.")


# =========================================================
# TAB 3: VISUALIZATIONS
# =========================================================

with tab_charts:

    st.subheader("Recommended visualizations")
    st.caption(
        "These charts are chosen from the kinds of columns you have. Each one "
        "says what it shows and what it means, in normal sentences."
    )

    chart_data, was_sampled = sample_for_charts(cleaned_df)

    if was_sampled:
        st.info(
            f"Showing a sample of {len(chart_data):,} rows out of "
            f"{len(cleaned_df):,} so the charts stay responsive. The shape of the "
            "data is preserved."
        )

    with st.spinner("Drawing your charts..."):
        try:
            charts = generate_visualizations(chart_data, date_columns)
        except Exception as error:
            st.error(
                "The automatic charts could not be created for this dataset. "
                "You can still build your own below."
            )
            st.caption(f"Technical detail: {error}")
            charts = []

    st.session_state["charts_ready"] = len(charts)

    if not charts:
        st.info(
            "There is not enough numeric or category information in this file to "
            "draw charts automatically. Try the chart builder below."
        )

    for index, chart in enumerate(charts):

        st.markdown(f"### {chart['title']}")
        st.caption(chart["description"])

        st.plotly_chart(chart["figure"], use_container_width=True)

        takeaway_box(chart["summary"])

        if use_ai:

            text_key = f"ai_text_{index}"

            if st.button("🤖 Ask AI for a deeper read", key=f"ai_button_{index}"):
                with st.spinner("InsightFlow AI is reading the numbers..."):
                    st.session_state[text_key] = generate_ai_insight(
                        chart["chart_type"], chart["statistics"]
                    )

            if st.session_state.get(text_key):
                ai_box(st.session_state[text_key])

        st.divider()

    # --------------------------------
    # Chart builder
    # --------------------------------

    st.markdown("### Build your own chart")
    st.caption("Pick the columns you are curious about and InsightFlow draws it.")

    chart_type = st.selectbox(
        "Chart type",
        [
            "Bar chart",
            "Line chart",
            "Scatter plot",
            "Box plot",
            "Pie chart",
            "Histogram",
        ],
        key="custom_type",
    )

    all_columns = list(cleaned_df.columns)

    column_one, column_two, column_three = st.columns(3)

    x_column = column_one.selectbox(
        "Horizontal axis",
        all_columns,
        key="custom_x",
        format_func=pretty_name,
    )

    needs_y = chart_type in ["Line chart", "Scatter plot", "Box plot"]
    optional_y = chart_type == "Bar chart"

    y_column = None

    if needs_y or optional_y:

        y_options = list(numeric_columns) if numeric_columns else list(all_columns)

        if optional_y:
            y_options = ["Just count the rows"] + y_options

        y_choice = column_two.selectbox(
            "Vertical axis",
            y_options,
            key="custom_y",
            format_func=lambda name: name
            if name == "Just count the rows"
            else pretty_name(name),
        )

        y_column = None if y_choice == "Just count the rows" else y_choice

    else:
        column_two.caption("This chart only needs one column.")

    color_column = column_three.selectbox(
        "Split by colour (optional)",
        ["None"] + all_columns,
        key="custom_color",
        format_func=lambda name: name if name == "None" else pretty_name(name),
    )

    if st.button("Draw the chart", key="draw_custom"):

        try:
            custom_figure = build_custom_chart(
                chart_data, chart_type, x_column, y_column, color_column
            )
            st.plotly_chart(custom_figure, use_container_width=True)

        except Exception as error:
            st.warning(
                "Those columns do not work together for this chart type. "
                "Try a number column on the vertical axis, or a column with fewer "
                "different values."
            )
            st.caption(f"Technical detail: {error}")


# =========================================================
# TAB 4: INSIGHTS
# =========================================================

with tab_insights:

    st.subheader("The main things worth knowing")
    st.caption("Everything below is calculated directly from your cleaned data.")

    try:
        insights = generate_insights(cleaned_df, date_columns)
    except Exception as error:
        insights = []
        st.warning("The key findings could not be generated for this dataset.")
        st.caption(f"Technical detail: {error}")

    top_insights = insights[:5]
    rest_insights = insights[5:]

    left_column, right_column = st.columns(2)

    for position, insight in enumerate(top_insights):

        target_column = left_column if position % 2 == 0 else right_column

        with target_column:
            insight_card(insight["icon"], insight["title"], insight["text"])

    if rest_insights:

        with st.expander(f"Show {len(rest_insights)} more findings"):

            more_left, more_right = st.columns(2)

            for position, insight in enumerate(rest_insights):

                target_column = more_left if position % 2 == 0 else more_right

                with target_column:
                    insight_card(insight["icon"], insight["title"], insight["text"])

    if len(numeric_columns) >= 2:

        with st.expander("What does 'connection' mean here?"):
            st.write(
                "A connection, or correlation, means two columns tend to move "
                "together. If advertising spend and sales both rise at the same "
                "time, they are connected. That does not prove advertising caused "
                "the sales. Something else, such as the season, may be driving "
                "both."
            )

    if ai_available():

        st.divider()
        st.markdown("#### An overall summary from the AI")

        if st.button("🤖 Summarise my whole dataset", key="ai_summary"):
            with st.spinner("Reading the whole dataset..."):
                st.session_state["ai_overall"] = answer_question(
                    "Give me a short overall summary of this dataset: what it is "
                    "about, the three most useful things it shows, and one thing "
                    "to be careful about.",
                    build_data_summary(cleaned_df, profile),
                )

        if st.session_state.get("ai_overall"):
            ai_box(st.session_state["ai_overall"])

        st.divider()
        st.markdown("#### Ask your own question")

        question = st.text_input(
            "Your question",
            placeholder="Which city brings in the most revenue?",
            key="user_question",
        )

        if st.button("Ask InsightFlow AI", key="ask_button") and question.strip():
            with st.spinner("Thinking about your data..."):
                st.session_state["ai_answer"] = answer_question(
                    question, build_data_summary(cleaned_df, profile)
                )

        if st.session_state.get("ai_answer"):
            ai_box(st.session_state["ai_answer"])

        with st.expander("What the AI actually sees"):
            st.caption("A summary is sent, never your raw rows.")
            st.code(build_data_summary(cleaned_df, profile))

    else:
        st.divider()
        st.info(
            "AI insights are unavailable. Add your API key to enable AI-powered "
            "explanations. Put it in `.streamlit/secrets.toml` as "
            "`OPENAI_API_KEY = \"your-key\"`."
        )


# =========================================================
# TAB 5: PREDICTION
# =========================================================

with tab_predict:

    st.subheader("Predict what happens next")

    if not sklearn_available():

        st.warning(
            "The prediction features need scikit-learn. "
            "Install it with: pip install scikit-learn"
        )

    else:

        st.markdown("#### 1. Teach a model to predict one column")
        st.caption(
            "Choose the column you want to predict. InsightFlow uses the other "
            "columns to learn the pattern, compares two different models, and "
            "scores them on records they have never seen."
        )

        targets = get_possible_targets(cleaned_df)

        if not targets:
            st.info(
                "No column in this dataset can be predicted. A target needs either "
                "many different numbers, or between 2 and 10 repeating groups."
            )

        else:

            target = st.selectbox(
                "What should the model predict?",
                targets,
                key="target_column",
                format_func=pretty_name,
            )

            problem_hint = (
                "This looks like a number, so InsightFlow will predict a value."
                if pd.api.types.is_numeric_dtype(cleaned_df[target])
                and cleaned_df[target].nunique() > 10
                else "This looks like a group, so InsightFlow will predict a label."
            )

            st.caption(problem_hint)

            if st.button("🚀 Train the models", type="primary", key="train_button"):

                with st.spinner("Learning the patterns in your data..."):
                    st.session_state["model_result"] = train_model(cleaned_df, target)

            result = st.session_state.get("model_result")

            if result is not None and result.get("target") != target:
                st.info("You changed the column. Train again to update the model.")

            elif result is not None and not result["ok"]:
                st.warning(result["message"])

            elif result is not None and result["ok"]:

                st.success(
                    f"Trained and tested. The better of the two models was "
                    f"{result['best_model_name']}."
                )

                if result.get("sampled"):
                    st.caption(
                        "A random sample of 20,000 rows was used for training to "
                        "keep it quick."
                    )

                metric_columns = st.columns(len(result["metrics"]) + 1)

                for position, (name, value) in enumerate(result["metrics"].items()):
                    stat_card(metric_columns[position], f"{value:,.3f}", name)

                stat_card(
                    metric_columns[-1],
                    f"{result['testing_rows']:,}",
                    "Records used for testing",
                    f"{result['training_rows']:,} used for learning",
                )

                st.write("")
                takeaway_box(result["explanation"] + " " + result["quality"])

                with st.expander("What do these scores mean?"):
                    for name in result["metrics"]:
                        st.markdown(
                            f"**{name}** — {result['metric_help'].get(name, '')}"
                        )

                st.markdown("#### Model comparison")

                st.dataframe(
                    result["comparison"], use_container_width=True, hide_index=True
                )

                st.caption(
                    f"Based on {result['main_metric']} on this test set, "
                    f"{result['best_model_name']} scored highest among the models "
                    "tested. A different split of the same data could rank them "
                    "differently."
                )

                left_side, right_side = st.columns(2)

                with left_side:

                    importance_figure = make_importance_chart(result)

                    if importance_figure is not None:
                        st.plotly_chart(importance_figure, use_container_width=True)
                        st.caption(
                            "Taken from the Random Forest model. A higher bar means "
                            "the model leaned on that column more. It does not mean "
                            "that column causes the outcome."
                        )
                    else:
                        st.caption("This model does not report column importance.")

                with right_side:

                    if result["problem"] == "number":
                        st.plotly_chart(
                            make_actual_vs_predicted_chart(result),
                            use_container_width=True,
                        )
                        st.caption(
                            "The closer the dots sit to the dashed line, the better "
                            "the prediction."
                        )
                    else:
                        st.plotly_chart(
                            make_confusion_chart(result), use_container_width=True
                        )
                        st.caption(
                            "The diagonal from top left to bottom right holds the "
                            "correct answers. Everything else is a mistake."
                        )

                # --------------------------------
                # What if
                # --------------------------------

                st.divider()
                st.markdown("#### 2. Try your own prediction")
                st.caption(
                    "Change the values below and InsightFlow predicts the result "
                    "for that made up record. Anything you do not set is filled "
                    "with the typical value from your data."
                )

                if not result["importance"].empty:
                    top_features = result["importance"]["raw_column"].head(6).tolist()
                else:
                    top_features = (
                        result["numeric_features"] + result["categorical_features"]
                    )[:6]

                with st.form("what_if_form"):

                    form_columns = st.columns(2)
                    user_values = {}

                    for position, feature in enumerate(top_features):

                        holder = form_columns[position % 2]
                        feature_series = cleaned_df[feature]

                        if pd.api.types.is_numeric_dtype(feature_series):
                            user_values[feature] = holder.number_input(
                                pretty_name(feature),
                                value=float(feature_series.median()),
                                key=f"whatif_{feature}",
                            )
                        else:
                            options = sorted(
                                feature_series.dropna().astype(str).unique().tolist()
                            )[:50]

                            user_values[feature] = holder.selectbox(
                                pretty_name(feature),
                                options,
                                key=f"whatif_{feature}",
                            )

                    submitted = st.form_submit_button("🔮 Predict this record")

                if submitted:

                    try:
                        prediction, confidence = predict_one(
                            result, user_values, cleaned_df
                        )

                        if result["problem"] == "number":
                            message = (
                                f"🎯 Predicted {pretty_name(target)}: "
                                f"<b>{float(prediction):,.2f}</b>. "
                                "Based on the test results, expect it to be off by "
                                f"around {result['metrics']['MAE']:,.2f} either way."
                            )
                        else:
                            message = (
                                f"🎯 Predicted {pretty_name(target)}: "
                                f"<b>{prediction}</b>."
                            )
                            if confidence is not None:
                                message += (
                                    f" The model is about {confidence:,.0f}% sure "
                                    "of this answer."
                                )

                        takeaway_box(message)

                    except Exception as error:
                        st.warning(
                            "That record could not be predicted. Try different "
                            "values."
                        )
                        st.caption(f"Technical detail: {error}")

                # --------------------------------
                # Limitations
                # --------------------------------

                st.markdown("#### ⚠️ Important")

                for line in result["limitations"]:
                    st.markdown(f"- {line}")

        # --------------------------------
        # Forecast
        # --------------------------------

        st.divider()
        st.markdown("#### 3. Forecast a trend over time")

        if not date_columns or not numeric_columns:
            st.info(
                "A forecast needs one date column and one number column. "
                "This dataset does not have both."
            )

        else:

            forecast_one, forecast_two, forecast_three, forecast_four = st.columns(4)

            date_column = forecast_one.selectbox(
                "Date column",
                date_columns,
                key="forecast_date",
                format_func=pretty_name,
            )

            value_column = forecast_two.selectbox(
                "What to forecast",
                numeric_columns,
                key="forecast_value",
                format_func=pretty_name,
            )

            period_name = forecast_three.selectbox(
                "Group by", ["Day", "Week", "Month"], index=2, key="forecast_period"
            )

            steps = forecast_four.slider("Steps ahead", 1, 12, 6, key="forecast_steps")

            if st.button("📈 Forecast", key="forecast_button"):

                with st.spinner("Projecting the trend..."):
                    try:
                        forecast_result = forecast_trend(
                            cleaned_df, date_column, value_column, period_name, steps
                        )
                    except Exception as error:
                        forecast_result = {
                            "ok": False,
                            "message": f"The forecast could not be built: {error}",
                        }

                st.session_state["forecast_result"] = forecast_result

            forecast_result = st.session_state.get("forecast_result")

            if forecast_result is not None:

                if not forecast_result["ok"]:
                    st.warning(forecast_result["message"])

                else:
                    st.plotly_chart(
                        forecast_result["figure"], use_container_width=True
                    )
                    takeaway_box(forecast_result["message"])
                    st.caption("⚠️ " + forecast_result["warning"])

                    st.dataframe(
                        forecast_result["table"],
                        use_container_width=True,
                        hide_index=True,
                    )


# =========================================================
# TAB 6: EXPORT
# =========================================================

with tab_export:

    st.subheader("Take your results with you")

    model_result = st.session_state.get("model_result")

    try:
        export_insights = generate_insights(cleaned_df, date_columns)
    except Exception:
        export_insights = []

    one, two = st.columns(2)

    with one:
        st.markdown("**Cleaned dataset**")
        st.caption("Your file with the repeats removed and the gaps filled.")

        st.download_button(
            label="⬇️ Cleaned data (CSV)",
            data=cleaned_df.to_csv(index=False),
            file_name="insightflow_cleaned_data.csv",
            mime="text/csv",
            use_container_width=True,
            key="clean_csv_export",
        )

        try:
            st.download_button(
                label="⬇️ Cleaned data (Excel)",
                data=to_excel_bytes(cleaned_df),
                file_name="insightflow_cleaned_data.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
                key="clean_xlsx_export",
            )
        except Exception:
            st.caption("Install openpyxl to download an Excel copy.")

    with two:
        st.markdown("**Analysis summary**")
        st.caption("A short text file with the numbers, findings and model scores.")

        summary_text = build_text_summary(
            cleaned_df, profile, report, export_insights, model_result
        )

        st.download_button(
            label="⬇️ Summary (TXT)",
            data=summary_text,
            file_name="insightflow_summary.txt",
            mime="text/plain",
            use_container_width=True,
            key="summary_txt",
        )

        with st.expander("Preview the summary"):
            st.code(summary_text)

    st.divider()

    st.markdown("**Full report**")
    st.caption(
        "One HTML file holding the overview, the cleaning actions, the charts and "
        "the findings. Open it in any browser, or print it to PDF."
    )

    if st.button("📄 Generate report", key="build_report"):

        with st.spinner("Building your report..."):

            try:
                chart_data, _ = sample_for_charts(cleaned_df)
                report_charts = generate_visualizations(chart_data, date_columns)
            except Exception:
                report_charts = []

            try:
                st.session_state["html_report"] = build_html_report(
                    cleaned_df,
                    profile,
                    report,
                    export_insights,
                    report_charts,
                    model_result,
                    score,
                    file_name,
                )
            except Exception as error:
                st.session_state["html_report"] = None
                st.warning("The report could not be built.")
                st.caption(f"Technical detail: {error}")

    if st.session_state.get("html_report"):

        st.success("Your report is ready.")

        st.download_button(
            label="⬇️ Download report (HTML)",
            data=st.session_state["html_report"],
            file_name="insightflow_report.html",
            mime="text/html",
            key="report_html",
        )

    st.divider()

    st.markdown("**Prediction results**")

    if model_result and model_result.get("ok"):

        st.caption(
            f"Your dataset with a predicted {pretty_name(model_result['target'])} "
            "column added to every row."
        )

        try:
            predicted_df = predict_dataset(model_result, cleaned_df)

            st.download_button(
                label="⬇️ Predictions (CSV)",
                data=predicted_df.to_csv(index=False),
                file_name="insightflow_predictions.csv",
                mime="text/csv",
                key="predictions_csv",
            )

            st.dataframe(predicted_df.head(10), use_container_width=True)

        except Exception as error:
            st.warning("The predictions could not be applied to every row.")
            st.caption(f"Technical detail: {error}")

    else:
        st.caption(
            "Train a model in the Prediction tab and the results will appear here."
        )

    st.divider()

    st.caption(
        "Charts can also be saved one at a time: hover over any chart and use the "
        "camera icon in its toolbar to download it as an image."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "InsightFlow · cleaned, explained and predicted from your own file. "
    "Filled values and forecasts are estimates, so check anything important "
    "before acting on it."
)
