# InsightFlow 📊

Turn raw data into clear insights, visualizations and predictions.

Upload a CSV or Excel file and InsightFlow checks it, cleans it, explains what is
inside it, draws the charts that matter and predicts what happens next — all in
language someone with no statistics background can follow.

---

## Overview

Most data tools assume you already know what a median, a correlation or an R²
score is. InsightFlow assumes you don't. Every number it shows comes with a
sentence saying what it means for your data, and every automatic decision it
makes is listed openly so nothing changes behind your back.

It works with any tabular dataset — sales, customers, students, housing,
finance, marketing — with no code changes. No column names, targets or
categories are hardcoded anywhere.

## Features

**Data quality**
- A 0–100 health score with the reasons behind it
- Missing values per column, with a recommended action for each
- Repeated rows, shown before anything is removed
- Columns stored the wrong way: numbers kept as text (`$ 1,200.50`, `45%`),
  dates kept as text, columns with a single value, ID-like columns
- Unusual values found with the IQR rule, never deleted automatically

**Cleaning**
- Every step is optional and switchable from the sidebar
- Repeated rows removed — unless the data is all categories, where identical
  rows are genuine records and are kept instead
- Column names tidied, with the original names kept and shown
- Extra spaces trimmed, text numbers converted, text dates recognised
- Infinite values treated as missing
- Gaps filled with the median (numbers) or the most common value (text)
- A before/after table and a list of exactly what changed

**Visualizations**
- Charts chosen automatically from the kinds of columns you have:
  distributions, category comparisons, frequencies, relationships, a
  correlation heatmap and a trend over time
- Each chart explains what it shows and what it means, calculated from the
  real numbers
- A chart builder for bar, line, scatter, box, pie and histogram
- Large files are sampled for charting, and you are told when that happens

**Insights**
- The five most important findings first, the rest behind one click
- Averages, ranges, outliers, dominant categories, strongest connections,
  time covered and columns that are not worth keeping
- Correlation is never described as causation

**Prediction**
- Pick any column to predict; classification or regression is detected for you
- Two models are trained and compared (Linear/Logistic Regression vs Random
  Forest), all preprocessing inside a scikit-learn `Pipeline` so nothing leaks
  from the test set
- Regression: R², MAE, RMSE. Classification: accuracy, precision, recall, F1,
  and ROC-AUC for two-group problems — each with a plain-English explanation
- Feature importance, actual-vs-predicted, confusion matrix
- Type your own values and get a single prediction back
- A straight-line forecast for any date + number pair
- An "Important" section on what the model cannot tell you

**Export**
- Cleaned data as CSV or Excel
- A text summary of the whole analysis
- A full HTML report with charts embedded (print it to PDF from the browser)
- Every row with its prediction attached, as CSV

**AI layer (optional)**
- Deeper explanations of any chart, an overall summary, and free-text questions
- Only a summary of the dataset is sent, never the raw rows
- Without an API key the app works exactly as normal, with its own written
  explanations everywhere

## Tech stack

| Area | Tools |
|---|---|
| Interface | Streamlit |
| Data | pandas, numpy |
| Charts | Plotly |
| Machine learning | scikit-learn |
| Trendlines | statsmodels |
| Files | openpyxl |
| AI (optional) | OpenAI API |

## How it works

```
Upload → Profile → Check quality → Clean → Visualize → Insights → Predict → Export
```

1. **Upload** – CSV or Excel, read safely, repeated column names fixed on the spot
2. **Profile** – size, types, dates, IDs, constants
3. **Check quality** – missing values, repeats, wrong types, outliers, a score
4. **Clean** – only the steps you leave switched on, every change reported
5. **Visualize** – charts picked for your columns, each one explained
6. **Insights** – the findings that matter, ranked
7. **Predict** – models compared, scored, explained, and usable on new values
8. **Export** – cleaned data, summary, report, predictions

## Installation

```bash
git clone <your-repo-url>
cd InsightFlow
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

Then open the address Streamlit prints (usually `http://localhost:8501`).
Pick **Use the sample data** in the sidebar to see everything working before
uploading your own file.

## AI setup (optional)

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-key-here"
```

Or set an environment variable instead:

```bash
export OPENAI_API_KEY="your-key-here"
```

`.streamlit/secrets.toml` is already in `.gitignore`. No key is ever written
into the source code, and the app never crashes when the key is missing,
invalid or out of quota — it simply says AI insights are unavailable.

## Project structure

```
InsightFlow/
├── app.py                        the interface: sidebar, tabs, layout, styling
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml               theme
│   └── secrets.toml.example      copy to secrets.toml and add your key
└── modules/
    ├── profiler.py               describes the dataset and every column
    ├── quality.py                missing values, type problems, outliers, score
    ├── cleaner.py                the cleaning pipeline and its report
    ├── visualizer.py             charts plus their written explanations
    ├── insights.py               rule-based findings, ranked by importance
    ├── predictor.py              model training, comparison and forecasting
    ├── ai_insights.py            optional OpenAI layer
    └── report_generator.py       HTML report and text summary
```

## Screenshots

_Add screenshots here once you have run the app: the Overview tab, the Data
quality score, a chart with its explanation, and the Prediction tab._

## Future improvements

- Save a session and come back to it later
- Clustering, to group similar records without a target column
- Seasonal forecasting rather than a straight trend line
- Compare two datasets side by side
- More file types: JSON, Parquet, Google Sheets

## A note on honesty

Filled values are estimates, not facts. A connection between two columns does
not prove that one causes the other. Model scores come from one test split and
real future performance may be lower. InsightFlow says all of this inside the
app, where it matters, rather than only here.
