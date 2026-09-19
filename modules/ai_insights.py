"""
ai_insights.py
--------------
Optional OpenAI-powered features for InsightFlow.

The app works perfectly well without an API key.
Charts, statistics, cleaning, predictions, and built-in insights
do not depend on OpenAI.
"""

import os

import streamlit as st

# =========================================================
# OPENAI IMPORT
# =========================================================

try:
    from openai import OpenAI

    OPENAI_READY = True

except ImportError:
    OPENAI_READY = False


# =========================================================
# MODELS
# =========================================================

MODEL = "gpt-5-mini"
BACKUP_MODEL = "gpt-4o-mini"


# =========================================================
# API KEY
# =========================================================


def get_api_key():
    """
    Look for the OpenAI API key in Streamlit secrets first,
    then in the environment variables.
    """

    try:
        key = st.secrets["OPENAI_API_KEY"]

        if key:
            return key

    except Exception:
        pass

    return os.environ.get("OPENAI_API_KEY", "")


def ai_available():
    """
    Check whether the OpenAI package and API key are available.

    Note:
    This does NOT guarantee that the API account has credits.
    """

    return OPENAI_READY and bool(get_api_key())


# =========================================================
# ONE SHARED FUNCTION FOR EVERY AI CALL
# =========================================================


@st.cache_data(show_spinner=False)
def ask_openai(prompt):
    """
    Send a prompt to OpenAI.

    The function safely handles:
    - Missing OpenAI package
    - Missing API key
    - Invalid API key
    - No API credits
    - API errors
    - Empty responses

    The rest of InsightFlow continues working even if AI fails.
    """

    # -----------------------------------------------------
    # OpenAI package missing
    # -----------------------------------------------------

    if not OPENAI_READY:
        return (
            "AI is unavailable because the OpenAI package is not installed. "
            "The rest of InsightFlow is still working normally."
        )

    # -----------------------------------------------------
    # API key missing
    # -----------------------------------------------------

    api_key = get_api_key()

    if not api_key:
        return (
            "AI explanations are unavailable because no API key is configured. "
            "Your charts and built-in InsightFlow analysis are still available."
        )

    # -----------------------------------------------------
    # Create OpenAI client
    # -----------------------------------------------------

    try:
        client = OpenAI(api_key=api_key)

    except Exception:
        return (
            "AI is currently unavailable. "
            "Your charts and built-in InsightFlow analysis are still working."
        )

    # -----------------------------------------------------
    # Try primary model
    # -----------------------------------------------------

    try:

        response = client.responses.create(model=MODEL, input=prompt)

        result = response.output_text

        if result and result.strip():
            return result.strip()

    except Exception as error:

        error_text = str(error).lower()

        # -------------------------------------------------
        # No API credits
        # -------------------------------------------------

        if (
            "insufficient_quota" in error_text
            or "credit_balance_exhausted" in error_text
            or "quota" in error_text
            or "billing" in error_text
        ):
            return (
                "AI explanation is currently unavailable because the "
                "OpenAI API has no remaining credits. Your charts and "
                "built-in insights are still working normally."
            )

        # -------------------------------------------------
        # Continue to backup model for other errors
        # -------------------------------------------------

        pass

    # -----------------------------------------------------
    # Try backup model
    # -----------------------------------------------------

    try:

        response = client.chat.completions.create(
            model=BACKUP_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )

        result = response.choices[0].message.content

        if result and result.strip():
            return result.strip()

    except Exception as error:

        error_text = str(error).lower()

        # -------------------------------------------------
        # No API credits
        # -------------------------------------------------

        if (
            "insufficient_quota" in error_text
            or "credit_balance_exhausted" in error_text
            or "quota" in error_text
            or "billing" in error_text
        ):
            return (
                "AI explanation is currently unavailable because the "
                "OpenAI API has no remaining credits. Your charts and "
                "built-in insights are still working normally."
            )

    # -----------------------------------------------------
    # General fallback
    # -----------------------------------------------------

    return (
        "AI explanation is temporarily unavailable. "
        "Your charts and built-in InsightFlow analysis are still available."
    )


# =========================================================
# CHART EXPLANATIONS
# =========================================================


def generate_ai_insight(chart_type, chart_data):
    """
    Explain one chart using only the numbers behind it.
    """

    prompt = f"""
You are InsightFlow AI, a data analyst talking to someone with no
background in statistics.

Chart type:
{chart_type}

Data:
{chart_data}

Explain what this data actually shows.

Rules:
- Use only the supplied numbers.
- Do not invent anything.
- Do not give generic statements.
- Mention the important numbers.
- Compare high and low values when useful.
- Say what it means in practice for the person who owns this data.
- Do not explain how to read the chart.
- Do not say that the chart is useful.
- Do not use headings or bullet points.
- Write 2 to 4 short sentences in simple English.
- Return only the explanation.
"""

    return ask_openai(prompt)


# =========================================================
# QUESTION ANSWERING
# =========================================================


def answer_question(question, data_summary):
    """
    Answer a user question about the dataset.
    """

    prompt = f"""
You are InsightFlow AI. Answer a question about the dataset described below.

Dataset summary:
{data_summary}

Question:
{question}

Rules:
- Answer only from the summary above.
- If the summary does not contain the answer, say exactly what extra
  information would be needed.
- Mention real numbers from the summary where they help.
- Never invent column names or values.
- Use simple English and keep it under 120 words.
- Return only the answer.
"""

    return ask_openai(prompt)


# =========================================================
# DATASET SUMMARY USED FOR QUESTIONS
# =========================================================


def build_data_summary(df, profile):
    """
    Turn the dataset into a short text description
    that the AI can read.
    """

    lines = []

    # -----------------------------------------------------
    # Basic dataset information
    # -----------------------------------------------------

    lines.append(f"Rows: {profile['rows']}, " f"Columns: {profile['columns']}")

    lines.append(f"Missing values: {profile['missing_values']}")

    lines.append(f"Duplicate rows: {profile['duplicate_rows']}")

    lines.append("")

    # -----------------------------------------------------
    # Columns and data types
    # -----------------------------------------------------

    lines.append("Columns and types:")

    for column in df.columns:

        lines.append(f"- {column}: {df[column].dtype}")

    # -----------------------------------------------------
    # Numeric statistics
    # -----------------------------------------------------

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if numeric_columns:

        lines.append("")

        lines.append("Statistics for number columns:")

        lines.append(df[numeric_columns].describe().round(2).to_string())

    # -----------------------------------------------------
    # Categorical information
    # -----------------------------------------------------

    categorical_columns = df.select_dtypes(include=["object", "category"]).columns

    for column in categorical_columns[:5]:

        if df[column].nunique(dropna=True) <= 20:

            lines.append("")

            lines.append(f"Most common values in {column}:")

            lines.append(df[column].value_counts().head(5).to_string())

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    return "\n".join(lines)
