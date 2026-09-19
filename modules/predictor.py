"""
predictor.py
------------
Makes predictions from the dataset.

Two separate things happen here:
1. Models learn to predict one column from the other columns, are compared
   against each other, and are scored on data they have never seen.
2. A simple trend line projects a number into the future.

Everything is wrapped in a scikit-learn Pipeline, so the filling of missing
values and the encoding of categories are learned on the training data only.
That is what stops the model from quietly cheating.
"""

import numpy as np
import pandas as pd
import plotly.express as px

from modules.profiler import pretty_name
from modules.visualizer import COLORS, style_figure


# scikit-learn is optional, so the rest of the app still works without it
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_READY = True

except ImportError:
    SKLEARN_READY = False


MAX_CATEGORIES = 20
MAX_ROWS = 20000


def sklearn_available():
    return SKLEARN_READY


# =========================================================
# WHAT CAN BE PREDICTED
# =========================================================


def get_problem_type(series):
    """Decide whether we are predicting a number or a group."""

    values = series.dropna()

    if values.empty:
        return None

    unique_values = values.nunique()

    if unique_values < 2:
        return None

    if pd.api.types.is_numeric_dtype(values) and unique_values > 10:
        return "number"

    if 2 <= unique_values <= 10:
        return "group"

    return None


def get_possible_targets(df):
    """List the columns that make sense to predict."""

    targets = []

    for column in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[column]):
            continue

        if get_problem_type(df[column]) is not None:
            targets.append(column)

    return targets


def get_feature_columns(df, target):
    """Split the usable input columns into numbers and categories."""

    numeric_features = []
    categorical_features = []

    for column in df.columns:

        if column == target:
            continue

        series = df[column]

        # Dates are used in the forecast section instead
        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        if pd.api.types.is_numeric_dtype(series):

            if series.nunique(dropna=True) <= 1:
                continue

            # A whole number that is different in every row is an ID
            if series.nunique(dropna=True) == len(df) and not (
                pd.api.types.is_float_dtype(series)
            ):
                continue

            numeric_features.append(column)

        else:
            unique_values = series.nunique(dropna=True)

            if 1 < unique_values <= MAX_CATEGORIES:
                categorical_features.append(column)

    return numeric_features, categorical_features


# =========================================================
# THE PIPELINE
# =========================================================


def make_encoder():
    """OneHotEncoder changed its argument name between scikit-learn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(model, numeric_features, categorical_features):
    """Filling, scaling and encoding all happen inside the model."""

    steps = []

    if numeric_features:
        steps.append(
            (
                "numbers",
                Pipeline(
                    [
                        ("fill", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        steps.append(
            (
                "categories",
                Pipeline(
                    [
                        ("fill", SimpleImputer(strategy="most_frequent")),
                        ("encode", make_encoder()),
                    ]
                ),
                categorical_features,
            )
        )

    prepare = ColumnTransformer(steps, remainder="drop")

    return Pipeline([("prepare", prepare), ("model", model)])


# =========================================================
# FEATURE IMPORTANCE
# =========================================================


def importance_table(pipeline, numeric_features, categorical_features):
    """
    One category column becomes several encoded columns.
    This adds their scores back together so real column names are shown.
    """

    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    try:
        encoded_names = pipeline.named_steps["prepare"].get_feature_names_out()
    except Exception:
        return pd.DataFrame()

    all_features = list(numeric_features) + list(categorical_features)
    scores = {column: 0.0 for column in all_features}

    for encoded_name, score in zip(encoded_names, model.feature_importances_):

        name = str(encoded_name)

        if "__" in name:
            name = name.split("__", 1)[1]

        matched = None

        for column in all_features:
            if name == column or name.startswith(column + "_"):
                if matched is None or len(column) > len(matched):
                    matched = column

        if matched is not None:
            scores[matched] += float(score)

    table = pd.DataFrame(
        {
            "Column": [pretty_name(column) for column in scores.keys()],
            "raw_column": list(scores.keys()),
            "Importance": [round(value * 100, 1) for value in scores.values()],
        }
    )

    return table.sort_values("Importance", ascending=False).reset_index(drop=True)


# =========================================================
# SCORING
# =========================================================


def score_regression(y_true, predictions):
    return {
        "R²": round(float(r2_score(y_true, predictions)), 3),
        "MAE": round(float(mean_absolute_error(y_true, predictions)), 3),
        "RMSE": round(float(mean_squared_error(y_true, predictions)) ** 0.5, 3),
    }


def score_classification(model, x_test, y_true, predictions):

    scores = {
        "Accuracy": round(float(accuracy_score(y_true, predictions)), 3),
        "Precision": round(
            float(precision_score(y_true, predictions, average="weighted", zero_division=0)),
            3,
        ),
        "Recall": round(
            float(recall_score(y_true, predictions, average="weighted", zero_division=0)),
            3,
        ),
        "F1": round(
            float(f1_score(y_true, predictions, average="weighted", zero_division=0)), 3
        ),
    }

    # ROC-AUC only makes sense for a two group problem
    classes = list(pd.Series(y_true).unique())

    if len(classes) == 2 and hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(x_test)[:, 1]
            positive = model.named_steps["model"].classes_[1]
            actual = (pd.Series(y_true).to_numpy() == positive).astype(int)
            scores["ROC-AUC"] = round(float(roc_auc_score(actual, probabilities)), 3)
        except Exception:
            pass

    return scores


METRIC_HELP = {
    "R²": "How much of the variation in the target the model explains. "
    "1.00 is perfect, 0.00 is no better than always guessing the average, "
    "and a negative value is worse than that.",
    "MAE": "The average size of the mistake, in the same units as the target. "
    "Smaller is better.",
    "RMSE": "Like MAE, but large mistakes are punished more heavily. "
    "Smaller is better.",
    "Accuracy": "Out of every 100 records, how many the model labelled correctly.",
    "Precision": "When the model picks a group, how often it is right. "
    "High precision means few false alarms.",
    "Recall": "Out of all the records that really belong to a group, how many the "
    "model found. High recall means few misses.",
    "F1": "One number that balances precision and recall. Useful when one group "
    "is much bigger than the other.",
    "ROC-AUC": "How well the model separates the two groups, from 0.50 (random "
    "guessing) to 1.00 (perfect separation).",
}


LIMITATIONS = [
    "A prediction is an estimate based on this dataset only. It is not a guarantee.",
    "If the data reflects a bias, such as one group being under-recorded, the "
    "model will repeat that bias.",
    "Small datasets give unstable results. The same model trained on slightly "
    "different rows can score very differently.",
    "An important column influencing the prediction does not mean it causes the "
    "outcome. The two often just move together.",
    "Scores come from one test split of your data, so real future performance "
    "may be lower.",
]


# =========================================================
# TRAINING
# =========================================================


def train_model(df, target):
    """Train and compare models that predict the target column."""

    if not SKLEARN_READY:
        return {
            "ok": False,
            "target": target,
            "message": "scikit-learn is not installed. Run: pip install scikit-learn",
        }

    problem = get_problem_type(df[target])

    if problem is None:
        return {
            "ok": False,
            "target": target,
            "message": (
                f"'{pretty_name(target)}' cannot be predicted. It needs either many "
                "different numbers, or between 2 and 10 repeating groups. A column "
                "with a single value, or with a different value in every row, has "
                "no pattern to learn."
            ),
        }

    data = df.dropna(subset=[target]).copy()

    sampled = False

    if len(data) > MAX_ROWS:
        data = data.sample(MAX_ROWS, random_state=42)
        sampled = True

    numeric_features, categorical_features = get_feature_columns(data, target)

    if not numeric_features and not categorical_features:
        return {
            "ok": False,
            "target": target,
            "message": (
                "There are no other usable columns to predict from. A prediction "
                "needs at least one other number or category column."
            ),
        }

    if len(data) < 30:
        return {
            "ok": False,
            "target": target,
            "message": (
                f"Only {len(data)} complete records were found. At least 30 are "
                "needed before a model can be trusted even a little."
            ),
        }

    features = data[numeric_features + categorical_features]
    target_values = data[target]

    stratify = None

    if problem == "group" and target_values.value_counts().min() >= 2:
        stratify = target_values

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target_values,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
    except Exception as error:
        return {
            "ok": False,
            "target": target,
            "message": f"The data could not be split for testing: {error}",
        }

    # --------------------------------
    # The candidate models
    # --------------------------------

    if problem == "number":
        candidates = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=150, random_state=42, n_jobs=-1
            ),
        }
        main_metric = "R²"
    else:
        candidates = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(
                n_estimators=150, random_state=42, n_jobs=-1
            ),
        }
        main_metric = "F1"

    trained = {}
    rows = []

    for name, model in candidates.items():

        pipeline = build_pipeline(model, numeric_features, categorical_features)

        try:
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)

            if problem == "number":
                scores = score_regression(y_test, predictions)
            else:
                scores = score_classification(pipeline, x_test, y_test, predictions)

        except Exception:
            # One model failing should never stop the others
            continue

        trained[name] = {"pipeline": pipeline, "scores": scores, "predictions": predictions}
        rows.append({"Model": name, **scores})

    if not trained:
        return {
            "ok": False,
            "target": target,
            "message": (
                "No model could be trained on this data. This usually means the "
                "columns are too sparse or too unusual to learn from."
            ),
        }

    comparison = pd.DataFrame(rows)

    best_name = max(trained, key=lambda name: trained[name]["scores"].get(main_metric, 0))
    best = trained[best_name]

    result = {
        "ok": True,
        "target": target,
        "problem": problem,
        "best_model_name": best_name,
        "pipeline": best["pipeline"],
        "metrics": best["scores"],
        "comparison": comparison,
        "main_metric": main_metric,
        "predictions": best["predictions"],
        "y_test": y_test,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "training_rows": int(len(x_train)),
        "testing_rows": int(len(x_test)),
        "sampled": sampled,
        "metric_help": METRIC_HELP,
        "limitations": LIMITATIONS,
    }

    # Feature importance comes from the Random Forest, which always reports it
    if "Random Forest" in trained:
        result["importance"] = importance_table(
            trained["Random Forest"]["pipeline"], numeric_features, categorical_features
        )
        result["importance_from"] = "Random Forest"
    else:
        result["importance"] = pd.DataFrame()
        result["importance_from"] = None

    # --------------------------------
    # Plain English verdict
    # --------------------------------

    if problem == "number":

        r2 = best["scores"]["R²"]
        mae = best["scores"]["MAE"]
        average = float(pd.Series(y_test).mean())

        if r2 < 0:
            result["explanation"] = (
                f"The best model, {best_name}, could not find a usable pattern for "
                f"{pretty_name(target)}. Its guesses were worse than simply using "
                f"the average every time, and were off by {mae:,.2f} on average."
            )
            result["quality"] = (
                "The other columns do not explain this value. More relevant "
                "columns, or more records, would be needed."
            )
        else:
            error_share = (mae / abs(average) * 100) if average != 0 else 0

            result["explanation"] = (
                f"{best_name} explains about {r2 * 100:,.0f}% of the differences in "
                f"{pretty_name(target)}. On records it had never seen, its guess was "
                f"off by {mae:,.2f} on average, which is roughly "
                f"{error_share:,.0f}% of the average value ({average:,.2f})."
            )

            if r2 >= 0.75:
                result["quality"] = "That is a strong result for this kind of data."
            elif r2 >= 0.4:
                result["quality"] = (
                    "That is a reasonable result. Treat single predictions as "
                    "rough estimates rather than facts."
                )
            else:
                result["quality"] = (
                    "That is a weak result. The pattern exists but it is faint, so "
                    "individual predictions will often be wrong."
                )

    else:

        accuracy = best["scores"]["Accuracy"]
        labels = sorted(pd.Series(y_test).unique().tolist(), key=str)

        result["labels"] = labels
        result["confusion"] = confusion_matrix(y_test, best["predictions"], labels=labels)

        baseline = float(pd.Series(y_test).value_counts(normalize=True).max())

        result["explanation"] = (
            f"{best_name} labelled about {accuracy * 100:,.0f} records out of every "
            f"100 correctly, on data it had never seen before."
        )

        if accuracy >= baseline + 0.1:
            result["quality"] = (
                "That is clearly better than always guessing the most common answer, "
                f"which would be right {baseline * 100:,.0f}% of the time."
            )
        else:
            result["quality"] = (
                "That is not much better than always guessing the most common answer "
                f"({baseline * 100:,.0f}% correct), so the other columns carry little "
                "useful information about this target."
            )

    return result


# =========================================================
# CHARTS FOR THE MODEL
# =========================================================


def make_importance_chart(result, top=10):
    """Show which columns matter most for the prediction."""

    table = result["importance"]

    if table.empty:
        return None

    fig = px.bar(
        table.head(top),
        x="Importance",
        y="Column",
        orientation="h",
        title="Which columns influence the prediction most",
        color_discrete_sequence=[COLORS[0]],
    )

    fig.update_layout(yaxis=dict(categoryorder="total ascending"))

    return style_figure(fig, "Share of the decision (%)", "")


def make_actual_vs_predicted_chart(result):
    """Compare what really happened with what the model predicted."""

    data = pd.DataFrame(
        {
            "Actual": np.asarray(result["y_test"], dtype=float),
            "Predicted": np.asarray(result["predictions"], dtype=float),
        }
    )

    fig = px.scatter(
        data,
        x="Actual",
        y="Predicted",
        opacity=0.7,
        title="Real values compared with predicted values",
        color_discrete_sequence=[COLORS[0]],
    )

    lowest = float(min(data["Actual"].min(), data["Predicted"].min()))
    highest = float(max(data["Actual"].max(), data["Predicted"].max()))

    fig.add_shape(
        type="line",
        x0=lowest,
        y0=lowest,
        x1=highest,
        y1=highest,
        line=dict(color=COLORS[2], dash="dash"),
    )

    return style_figure(
        fig,
        f"Real {pretty_name(result['target'])}",
        f"Predicted {pretty_name(result['target'])}",
    )


def make_confusion_chart(result):
    """Show how often each group was predicted correctly."""

    labels = [str(label) for label in result["labels"]]

    fig = px.imshow(
        result["confusion"],
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale=["#F3F5F7", COLORS[0]],
        title="Correct and incorrect predictions",
    )

    return style_figure(fig, "Model predicted", "Real answer")


# =========================================================
# PREDICTING A SINGLE NEW RECORD
# =========================================================


def predict_one(result, user_values, df):
    """Predict the target for one record the user typed in."""

    row = {}

    for column in result["numeric_features"] + result["categorical_features"]:

        if column in user_values:
            row[column] = user_values[column]

        else:
            series = df[column]

            if pd.api.types.is_numeric_dtype(series):
                row[column] = series.median()
            else:
                modes = series.mode()
                row[column] = modes[0] if len(modes) > 0 else "Unknown"

    one_row = pd.DataFrame([row])

    pipeline = result["pipeline"]
    prediction = pipeline.predict(one_row)[0]

    confidence = None

    if result["problem"] == "group" and hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(one_row)[0]
            confidence = float(max(probabilities)) * 100
        except Exception:
            confidence = None

    return prediction, confidence


def predict_dataset(result, df):
    """Predict every row, so the user can download the results."""

    columns = result["numeric_features"] + result["categorical_features"]
    predictions = result["pipeline"].predict(df[columns])

    output = df.copy()
    output[f"predicted_{result['target']}"] = predictions

    return output


# =========================================================
# TREND FORECAST
# =========================================================


PERIOD_CODES = {"Day": "D", "Week": "W", "Month": "M"}


def forecast_trend(df, date_column, value_column, period_name="Month", steps=6):
    """Fit a straight trend line through the past and project it forward."""

    data = df[[date_column, value_column]].dropna().copy()

    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.dropna()

    if data.empty:
        return {"ok": False, "message": "This column does not contain usable dates."}

    code = PERIOD_CODES.get(period_name, "M")

    grouped = data.groupby(data[date_column].dt.to_period(code))[value_column].mean()

    if len(grouped) < 4:
        return {
            "ok": False,
            "message": (
                f"Only {len(grouped)} {period_name.lower()}s of history were found. "
                "At least 4 are needed for a trend. Try a smaller time step."
            ),
        }

    history = grouped.reset_index()
    history[date_column] = history[date_column].dt.to_timestamp()
    history["Type"] = "History"

    positions = np.arange(len(history))
    slope, intercept = np.polyfit(positions, history[value_column].astype(float), 1)

    future_positions = np.arange(len(history), len(history) + steps)
    future_values = slope * future_positions + intercept

    future_periods = [grouped.index[-1] + step for step in range(1, steps + 1)]
    future_dates = [period.to_timestamp() for period in future_periods]

    forecast = pd.DataFrame(
        {
            date_column: future_dates,
            value_column: future_values,
            "Type": "Forecast",
        }
    )

    combined = pd.concat([history, forecast], ignore_index=True)

    value_label = pretty_name(value_column)

    fig = px.line(
        combined,
        x=date_column,
        y=value_column,
        color="Type",
        markers=True,
        line_dash="Type",
        title=f"{value_label}: history and forecast",
        color_discrete_sequence=[COLORS[0], COLORS[2]],
    )

    style_figure(fig, pretty_name(date_column), value_label)

    last_real = float(history[value_column].iloc[-1])
    last_forecast = float(future_values[-1])

    average_value = float(history[value_column].mean())
    is_flat = abs(slope) < abs(average_value) * 0.005

    if is_flat:
        movement = "stayed more or less flat"
    elif slope > 0:
        movement = f"been rising by about {abs(slope):,.2f} per {period_name.lower()}"
    else:
        movement = f"been falling by about {abs(slope):,.2f} per {period_name.lower()}"

    message = (
        f"Over the past {len(history)} {period_name.lower()}s, {value_label} has "
        f"{movement}. If that pattern continues, it would reach roughly "
        f"{last_forecast:,.2f} in {steps} {period_name.lower()}s, compared with "
        f"{last_real:,.2f} today."
    )

    table = forecast.copy()
    table[date_column] = table[date_column].dt.date
    table[value_column] = table[value_column].round(2)

    return {
        "ok": True,
        "figure": fig,
        "message": message,
        "table": table[[date_column, value_column]],
        "warning": (
            "A trend line assumes the future behaves like the past. "
            "It cannot know about holidays, price changes or one off events."
        ),
    }
