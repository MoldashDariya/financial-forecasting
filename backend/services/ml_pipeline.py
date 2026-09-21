# The forecasting pipeline used by the backend. It repeats the methodology of the notebooks (notebooks/ folder)
# for one uploaded company:
#   feature engineering -> 4 feature scenarios (S1-S4) -> tuned ML models + time-series models (ARIMA, SARIMA, Prophet, LSTM)
#   -> ensembles -> ranking -> recursive forecast until the end of 2032 -> sanity check of the forecast -> JSON in data/results.
# Entry point: run_pipeline() at the bottom of the file
import json
import multiprocessing as mp
import re
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

# Optional packages. The pipeline checks what is installed and skips the models it can't run
# (the reason ends up in model_skip_reason instead of crashing the whole upload)
ARIMA_AVAILABLE = importlib.util.find_spec("statsmodels") is not None

PROPHET_AVAILABLE = importlib.util.find_spec("prophet") is not None

TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None

try:
    # xgboost needs libomp on macOS, if the import fails we just run without XGBoost
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
    XGBOOST_IMPORT_ERROR = None
except Exception:
    XGBOOST_AVAILABLE = False
    XGBOOST_IMPORT_ERROR = "xgboost could not be imported. On macOS this usually means libomp is missing."

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"
MACRO_PATH = DATA_DIR / "macro_features.csv"
# 16 quarters = 4 years: length of the robustness check and of the short forecast.
# The full forecast goes from 2026 Q1 to the end of 2032
FORECAST_STEPS = 16
FORECAST_START_DATE_2026_Q1 = pd.Timestamp("2026-03-31")
FORECAST_END_DATE_2032 = pd.Timestamp("2032-12-31")
# LSTM is trained in a separate process and killed after 4 minutes, so one upload can't hang forever
LSTM_TIMEOUT_SECONDS = 240

# A numeric column whose name contains one of these words is treated as a macro variable,
# every other numeric column is a company (micro) indicator
CORE_MACRO_HINTS = {"cpi", "dgs10", "brent", "nasdaq", "sp500", "gdp", "inflation", "interest", "rate"}
# Models are ranked by sMAPE first, then MAPE, RMSE, MAE (same rule as in the notebooks)
METRIC_SORT_ORDER = ["sMAPE", "MAPE", "RMSE", "MAE"]
TARGET_COL = "revenue"
GROUP_COL = "company"
DATE_COL = "date"
CAT_FEATURES = ["company"]

# In the notebooks these companies did better with the raw revenue target,
# all the other companies use the log-target models (see preferred_target in run_pipeline)
RAW_TARGET_COMPANIES = {
    "apple",
    "microsoft",
    "amazon",
    "gamestop",
    "intel",
    "jpmorgan",
    "jp morgan",
    "jpmorgan chase",
    "jpmorgan chase & co",
}

# Feature groups. Lags 1, 2, 4 and 8 were chosen in the notebook by a lag-importance screening
# (random forest importance + permutation importance)
BASELINE_FEATURES = ["lag_1", "lag_2", "lag_4", "lag_8"]
REVENUE_HISTORY_FEATURES = ["roll_mean_4", "roll_std_4", "roll_mean_8", "roll_std_8", "growth_1"]
SEASONALITY_FEATURES = ["quarter", "sin_q", "cos_q"]
LOG_BASELINE_FEATURES = ["log_lag_1", "log_lag_2", "log_lag_4", "log_lag_8"]
LOG_REVENUE_HISTORY_FEATURES = [
    "log_roll_mean_4",
    "log_roll_std_4",
    "log_roll_mean_8",
    "log_roll_std_8",
    "log_growth_1",
]
MODEL_FEATURES_NO_MACRO = BASELINE_FEATURES + REVENUE_HISTORY_FEATURES + SEASONALITY_FEATURES
MODEL_FEATURES_LOG_TARGET = LOG_BASELINE_FEATURES + LOG_REVENUE_HISTORY_FEATURES + SEASONALITY_FEATURES

# Hyper-parameter grids of the 'full' profile
RF_GRID = {
    "model__n_estimators": [300, 600],
    "model__max_depth": [4, 8, None],
    "model__min_samples_leaf": [2, 4],
    "model__max_features": ["sqrt", 0.8],
}

XGB_GRID = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [2, 3, 4],
    "model__learning_rate": [0.03, 0.05],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
}

SVR_GRID = {
    "model__kernel": ["rbf"],
    "model__C": [1, 10, 100],
    "model__epsilon": [0.01, 0.1, 0.5],
    "model__gamma": ["scale", "auto"],
}

# Tiny grids for the 'balanced' (fast) profile, one or two combinations per model
BALANCED_RF_GRID = {
    "model__n_estimators": [300],
    "model__max_depth": [8, None],
    "model__min_samples_leaf": [2],
    "model__max_features": ["sqrt"],
}

BALANCED_XGB_GRID = {
    "model__n_estimators": [200],
    "model__max_depth": [3],
    "model__learning_rate": [0.05],
    "model__subsample": [0.9],
    "model__colsample_bytree": [0.9],
}

BALANCED_SVR_GRID = {
    "model__kernel": ["rbf"],
    "model__C": [10],
    "model__epsilon": [0.1],
    "model__gamma": ["scale"],
}

# Ensemble = ARIMA + best Random Forest + best XGBoost of a scenario, weighted by 1/sMAPE
SCENARIO_ENSEMBLE_MODEL_NAME = "ARIMA+RF+XGBoost Ensemble"
SCENARIO_ENSEMBLE_TARGET_TYPE = "scenario-specific ARIMA + RF + XGBoost ensemble"


# Which models can run in this environment. Served by /api/pipeline/status and stored in every result
def pipeline_dependency_status(model_skip_reason=None):
    model_skip_reason = dict(model_skip_reason or {})
    return {
        "variant": "notebook-parity backend pipeline",
        "core_runtime_models": {
            "Linear Regression": True,
            "Random Forest": True,
            "SVR": True,
            "XGBoost": XGBOOST_AVAILABLE,
            "ARIMA": ARIMA_AVAILABLE,
            "SARIMA": ARIMA_AVAILABLE,
            "Prophet": PROPHET_AVAILABLE,
            "LSTM": TENSORFLOW_AVAILABLE,
        },
        "optional_dependency_status": {
            "statsmodels_for_arima_sarima": ARIMA_AVAILABLE,
            "prophet": PROPHET_AVAILABLE,
            "tensorflow_for_lstm": TENSORFLOW_AVAILABLE,
            "xgboost": XGBOOST_AVAILABLE,
        },
        "xgboost_import_error": XGBOOST_IMPORT_ERROR,
        "model_skip_reason": model_skip_reason,
    }


# Reads numbers from messy exports: non-breaking spaces, comma decimals, stray symbols. Gives a float or NaN
def _safe_float(x):
    if pd.isna(x):
        return np.nan
    text = str(x).strip().replace("\u00a0", "").replace(" ", "")
    text = text.replace(",", ".")
    text = re.sub(r"[^0-9.\-]", "", text)
    if text in ("", ".", "-", "-."):
        return np.nan
    try:
        return float(text)
    except Exception:
        return np.nan


def _clean_name(value, fallback):
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:60] or fallback


def _normalize_company_name(value):
    return str(value).strip().lower()


# Quarter-end dates that have to be forecast: from 2026-03-31 (or the quarter after the last data point if that is later)
# until 2032-12-31
def future_quarter_dates(last_date=None, end_date=FORECAST_END_DATE_2032, start_date=FORECAST_START_DATE_2026_Q1):
    end_date = pd.Timestamp(end_date)
    next_start = pd.Timestamp(start_date)
    if last_date is not None and pd.notna(last_date):
        next_after_history = pd.Timestamp(last_date) + pd.DateOffset(months=3)
        next_start = max(next_start, next_after_history)

    out = []
    current = next_start
    while current <= end_date:
        out.append(pd.Timestamp(current))
        current = pd.Timestamp(current) + pd.DateOffset(months=3)
    return out


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# points where the true value is ~0 are ignored (division by zero)
def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-9
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# sMAPE (symmetric MAPE) is the main ranking metric. The 1e-9 only avoids 0/0
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-9
    return float(np.mean(2 * np.abs(y_pred - y_true) / denom) * 100)


def metric_row(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


def _metric_sort_key(row):
    return (
        row.get("sMAPE", 1e9) if row.get("sMAPE") is not None else 1e9,
        row.get("MAPE", 1e9) if row.get("MAPE") is not None else 1e9,
        row.get("RMSE", 1e9) if row.get("RMSE") is not None else 1e9,
        row.get("MAE", 1e9) if row.get("MAE") is not None else 1e9,
    )


def _select_best_component_row(rows, name_prefix):
    candidates = [
        row for row in rows
        if str(row.get("model", "")).startswith(name_prefix)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=_metric_sort_key)[0]


# Ensemble of one scenario. Weights are 1/sMAPE of ARIMA, the best RF and the best XGBoost (normalised to sum to 1).
# Predictions are combined on the dates all three models have, and clipped at 0
def _build_scenario_ensemble_rows(scenario_name, scenario_rows, scenario_predictions, ts_rows, ts_predictions):
    arima_row = next((row for row in ts_rows if row.get("scenario") == "TS" and row.get("model") == "ARIMA"), None)
    rf_row = _select_best_component_row(scenario_rows, "Random Forest")
    xgb_row = _select_best_component_row(scenario_rows, "XGBoost")
    if not arima_row or not rf_row or not xgb_row:
        return None, []

    weights_raw = {
        "ARIMA": 1.0 / max(float(arima_row.get("sMAPE", 1e9) or 1e9), 1e-8),
        "RF": 1.0 / max(float(rf_row.get("sMAPE", 1e9) or 1e9), 1e-8),
        "XGBoost": 1.0 / max(float(xgb_row.get("sMAPE", 1e9) or 1e9), 1e-8),
    }
    weight_sum = sum(weights_raw.values())
    if weight_sum <= 0:
        weights = {"ARIMA": 1 / 3, "RF": 1 / 3, "XGBoost": 1 / 3}
    else:
        weights = {key: value / weight_sum for key, value in weights_raw.items()}

    arima_predictions = {
        row["date"]: row
        for row in ts_predictions
        if row.get("scenario") == "TS" and row.get("model") == "ARIMA"
    }
    rf_predictions = {
        row["date"]: row
        for row in scenario_predictions
        if row.get("scenario") == scenario_name and row.get("model") == rf_row.get("model")
    }
    xgb_predictions = {
        row["date"]: row
        for row in scenario_predictions
        if row.get("scenario") == scenario_name and row.get("model") == xgb_row.get("model")
    }

    common_dates = sorted(set(arima_predictions) & set(rf_predictions) & set(xgb_predictions))
    if not common_dates:
        return None, []

    ensemble_predictions = []
    y_true, y_pred = [], []
    for date in common_dates:
        arima_pred = float(arima_predictions[date]["predicted"])
        rf_pred = float(rf_predictions[date]["predicted"])
        xgb_pred = float(xgb_predictions[date]["predicted"])
        actual = float(rf_predictions[date]["actual"])
        predicted = max(
            weights["ARIMA"] * arima_pred +
            weights["RF"] * rf_pred +
            weights["XGBoost"] * xgb_pred,
            0.0,
        )
        ensemble_predictions.append(
            {
                "scenario": scenario_name,
                "model": SCENARIO_ENSEMBLE_MODEL_NAME,
                "date": date,
                "actual": actual,
                "predicted": predicted,
                "split": rf_predictions[date].get("split"),
                "component_models": {
                    "ARIMA": "ARIMA",
                    "RF": rf_row.get("model"),
                    "XGBoost": xgb_row.get("model"),
                },
                "ensemble_weights": weights,
            }
        )
        y_true.append(actual)
        y_pred.append(predicted)

    metric_payload = metric_row(y_true, y_pred)
    ensemble_row = {
        "scenario": scenario_name,
        "model_family": "ensemble",
        "model": SCENARIO_ENSEMBLE_MODEL_NAME,
        "target_type": SCENARIO_ENSEMBLE_TARGET_TYPE,
        "n_test_points": int(len(y_true)),
        **metric_payload,
    }
    return ensemble_row, ensemble_predictions


def load_company_csv(csv_path):
    """
    Universal parser:
    - clean CSV with date/revenue + optional extra columns
    - wide Refinitiv-like exports (auto-detect date and revenue rows)
    """
    try:
        clean = pd.read_csv(csv_path)
        lower = {str(c).lower().strip(): c for c in clean.columns}
        if "date" in lower and "revenue" in lower:
            out = clean.rename(columns={lower["date"]: DATE_COL, lower["revenue"]: TARGET_COL}).copy()
            out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce", dayfirst=True)
            for col in out.columns:
                if col != DATE_COL:
                    out[col] = out[col].map(_safe_float)
            out = out.dropna(subset=[DATE_COL, TARGET_COL]).sort_values(DATE_COL)
            out[GROUP_COL] = "uploaded_company"
            return out.reset_index(drop=True)
    except Exception:
        pass

    # Not a plain date/revenue file, so assume a Refinitiv-style export: one row per indicator, one column per quarter.
    # Look for the row with the most parseable dates (row 17 is tried first), then find the revenue row by its label
    raw = pd.read_csv(csv_path, header=None, sep=None, engine="python", on_bad_lines="skip")
    possible_date_rows = [17] + list(range(min(len(raw), 30)))
    best_row, best_dates, best_count = None, None, 0
    for idx in possible_date_rows:
        if idx >= len(raw):
            continue
        dates = pd.to_datetime(raw.iloc[idx, 1:], errors="coerce", dayfirst=True)
        count = int(dates.notna().sum())
        if count > best_count:
            best_row, best_dates, best_count = idx, dates, count
    if best_row is None or best_count < 4:
        raise ValueError("Could not detect quarterly dates. Upload clean date/revenue CSV or Refinitiv-style CSV.")

    # default: row 19 (where revenue normally is in these exports), replaced below if a label matches
    revenue_row = 19 if len(raw) > 19 else None
    label_col = raw.iloc[:, 0].astype(str).str.lower()
    matches = label_col[label_col.str.contains("revenue|sales|business total|total revenue", regex=True, na=False)]
    if len(matches) > 0:
        revenue_row = int(matches.index[0])
    if revenue_row is None:
        raise ValueError("Could not detect revenue row.")

    out = pd.DataFrame(
        {
            DATE_COL: best_dates,
            TARGET_COL: raw.iloc[revenue_row, 1:].map(_safe_float).values,
            GROUP_COL: "uploaded_company",
        }
    )

    used_names = set()
    for i in range(len(raw)):
        if i in {best_row, revenue_row}:
            continue
        values = raw.iloc[i, 1:].map(_safe_float)
        if values.notna().sum() < max(6, best_count // 3):
            continue
        name = _clean_name(raw.iloc[i, 0], f"feature_{i}")
        if name in used_names or name in {DATE_COL, TARGET_COL, GROUP_COL}:
            name = f"{name}_{i}"
        used_names.add(name)
        out[name] = values.values

    out = out.dropna(subset=[DATE_COL, TARGET_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return out


def _ensure_quarter_fields(df):
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out = out.dropna(subset=[DATE_COL]).sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)
    out["year"] = out[DATE_COL].dt.year.astype(int)
    out["quarter"] = out[DATE_COL].dt.quarter.astype(int)
    out["fiscal_year"] = out["year"]
    out["fiscal_quarter"] = out["quarter"]
    out["fiscal_period"] = out["fiscal_year"].astype(str) + "Q" + out["fiscal_quarter"].astype(str)
    return out


# Splits the columns into macro (name matches CORE_MACRO_HINTS) and micro (all other numeric columns)
def _infer_macro_columns(df):
    cols = []
    for c in df.columns:
        if c in {DATE_COL, TARGET_COL, GROUP_COL, "year", "quarter"}:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        lc = c.lower()
        if any(h in lc for h in CORE_MACRO_HINTS):
            cols.append(c)
    return cols


def _infer_micro_columns(df, macro_cols):
    cols = []
    for c in df.columns:
        if c in {DATE_COL, TARGET_COL, GROUP_COL, "year", "quarter"} or c in macro_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


# Revenue-only features: lags, rolling mean/std, growth and seasonality, each in a raw and a log version.
# Everything is shifted, so a row only sees the past (no data leakage)
def add_no_macro_feature_columns(df, target_col=TARGET_COL, group_col=GROUP_COL):
    out = _ensure_quarter_fields(df)
    grouped_revenue = out.groupby(group_col)[target_col]
    log_revenue = np.log1p(out[target_col].clip(lower=0))
    grouped_log_revenue = log_revenue.groupby(out[group_col])

    for lag in [1, 2, 4, 8]:
        out[f"lag_{lag}"] = grouped_revenue.shift(lag)
        out[f"log_lag_{lag}"] = grouped_log_revenue.shift(lag)

    # Notebook methodology: rolling stats are based on shifted history to avoid leakage.
    shifted = grouped_revenue.shift(1)
    shifted_log = grouped_log_revenue.shift(1)
    for window in [4, 8]:
        out[f"roll_mean_{window}"] = shifted.groupby(out[group_col]).transform(lambda s: s.rolling(window).mean())
        out[f"roll_std_{window}"] = shifted.groupby(out[group_col]).transform(lambda s: s.rolling(window).std())
        out[f"log_roll_mean_{window}"] = shifted_log.groupby(out[group_col]).transform(lambda s: s.rolling(window).mean())
        out[f"log_roll_std_{window}"] = shifted_log.groupby(out[group_col]).transform(lambda s: s.rolling(window).std())

    # Notebook methodology uses previous-quarter change, shifted to avoid leakage.
    growth_1 = grouped_revenue.pct_change(1)
    out["growth_1"] = growth_1.groupby(out[group_col]).shift(1)

    log_growth_1 = grouped_log_revenue.diff(1)
    out["log_growth_1"] = log_growth_1.groupby(out[group_col]).shift(1)

    # quarter as sin/cos, so Q4 and Q1 are 'close' for the model
    q = out["fiscal_quarter"].astype(int)
    out["sin_q"] = np.sin(2 * np.pi * q / 4)
    out["cos_q"] = np.cos(2 * np.pi * q / 4)

    all_model_features = sorted(set(MODEL_FEATURES_NO_MACRO + MODEL_FEATURES_LOG_TARGET))
    out[all_model_features] = out[all_model_features].replace([np.inf, -np.inf], np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


# Macro variables enter with a 1-quarter lag, i.e. the value that is known when the forecast is made
def add_macro_lag_features(df, macro_cols):
    out = df.copy()
    grouped = out.groupby(GROUP_COL, sort=False)
    macro_features = []
    for col in macro_cols:
        lag_col = f"{col}_lag_1q"
        out[lag_col] = grouped[col].shift(1)
        macro_features.append(lag_col)
    return out, macro_features


# Company indicators from the export: lag 1, lag 4, quarter-on-quarter growth and margin (indicator / revenue),
# all shifted by one quarter
def add_micro_features(df, micro_cols, revenue_col=TARGET_COL):
    out = df.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True).copy()
    grouped = out.groupby(GROUP_COL, sort=False)
    feature_cols = []
    for col in micro_cols:
        lag1 = f"{col}_lag_1"
        lag4 = f"{col}_lag_4"
        qoq = f"{col}_growth_qoq"
        margin = f"{col}_margin_lag1"
        out[lag1] = grouped[col].shift(1)
        out[lag4] = grouped[col].shift(4)
        raw_growth = grouped[col].pct_change()
        out[qoq] = raw_growth.groupby(out[GROUP_COL]).shift(1)
        raw_margin = out[col] / out[revenue_col].replace(0, np.nan)
        out[margin] = raw_margin.groupby(out[GROUP_COL]).shift(1)
        feature_cols.extend([lag1, lag4, qoq, margin])
    return out.replace([np.inf, -np.inf], np.nan), feature_cols


def to_log_target(y):
    return np.log1p(np.asarray(y, dtype=float))


# Duan smearing. exp(prediction in log space) is a median, so it is too low on average.
# Multiplying by the mean of exp(training residuals) removes that bias
def duan_smearing_factor(y_train_log, y_train_log_pred):
    residuals = np.asarray(y_train_log, dtype=float) - np.asarray(y_train_log_pred, dtype=float)
    return float(np.mean(np.exp(residuals)))


# clip to +-50 so expm1 can't overflow, and never return a negative revenue
def from_log_target(y_log, smearing_factor=1.0):
    arr = np.asarray(y_log, dtype=float)
    arr = np.clip(arr, -50, 50)
    safe_smearing = float(smearing_factor) if np.isfinite(smearing_factor) and smearing_factor > 0 else 1.0
    revenue_pred = np.expm1(arr) * safe_smearing
    revenue_pred = np.where(np.isfinite(revenue_pred), revenue_pred, np.nan)
    return np.maximum(revenue_pred, 0)


def fit_log_target_model(model, X_train, y_train_log):
    fitted = clone(model)
    fitted.fit(X_train, y_train_log)
    pred_train_log = fitted.predict(X_train)
    smearing = duan_smearing_factor(y_train_log, pred_train_log)
    if not np.isfinite(smearing) or smearing <= 0:
        smearing = 1.0
    return fitted, smearing


# Numbers: median imputer (+ scaler). Company column: one-hot. Trees don't need scaling (scale_numeric=False)
def make_preprocess(scale_numeric=True, numeric_features=None):
    numeric_features = list(numeric_features or MODEL_FEATURES_NO_MACRO)
    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features), ("cat", categorical_transformer, CAT_FEATURES)],
        remainder="drop",
    )


def make_ml_pipeline_for_features(model_estimator, numeric_features, scale_numeric):
    return Pipeline(steps=[("preprocess", make_preprocess(scale_numeric=scale_numeric, numeric_features=numeric_features)), ("model", model_estimator)])


def make_base_models(numeric_features):
    models = {
        "Linear Regression": make_ml_pipeline_for_features(LinearRegression(), numeric_features, scale_numeric=True),
        "Random Forest": make_ml_pipeline_for_features(
            RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1), numeric_features, scale_numeric=False
        ),
        "SVR": make_ml_pipeline_for_features(SVR(), numeric_features, scale_numeric=True),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = make_ml_pipeline_for_features(
            XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
            ),
            numeric_features,
            scale_numeric=False,
        )
    return models


def get_param_grid_for_model(model_name, runtime_profile="full"):
    if runtime_profile == "balanced":
        if model_name == "Linear Regression":
            return {}
        if model_name == "Random Forest":
            return BALANCED_RF_GRID
        if model_name == "SVR":
            return BALANCED_SVR_GRID
        if model_name == "XGBoost":
            return BALANCED_XGB_GRID
        return {}

    if model_name == "Linear Regression":
        return {}
    if model_name == "Random Forest":
        return RF_GRID
    if model_name == "SVR":
        return SVR_GRID
    if model_name == "XGBoost":
        return XGB_GRID
    return {}


# Holdout split: the last n quarters of every company are the test set (time order kept, no shuffling)
def split_last_n_by_company(df, n_test=8, group_col=GROUP_COL, date_col=DATE_COL):
    train_parts, test_parts = [], []
    for company, company_df in df.groupby(group_col):
        company_df = company_df.sort_values(date_col)
        if len(company_df) <= n_test:
            continue
        train_parts.append(company_df.iloc[:-n_test])
        test_parts.append(company_df.iloc[-n_test:])
    if not train_parts:
        raise ValueError(f"Not enough rows for holdout split n_test={n_test}.")
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


# Rolling-origin (walk-forward) cross-validation used for tuning: every fold validates on the next `horizon` quarters
# and trains only on data before them
def rolling_origin_splits_grouped(df, n_splits=3, horizon=4, min_train=20, group_col=GROUP_COL, date_col=DATE_COL):
    grouped = {company: g.sort_values(date_col).reset_index(drop=True) for company, g in df.groupby(group_col) if len(g) >= min_train + horizon}
    if not grouped:
        return
    max_possible = min((len(g) - min_train) // horizon for g in grouped.values())
    n_splits_eff = min(n_splits, max_possible)
    for fold in range(n_splits_eff):
        train_parts, valid_parts = [], []
        for _, g in grouped.items():
            valid_end = len(g) - (n_splits_eff - fold - 1) * horizon
            valid_start = valid_end - horizon
            train_start = max(0, valid_start - min_train)
            train_parts.append(g.iloc[train_start:valid_start])
            valid_parts.append(g.iloc[valid_start:valid_end])
        yield fold + 1, pd.concat(train_parts, ignore_index=True), pd.concat(valid_parts, ignore_index=True)


def evaluate_cv_model(model, df, feature_cols, log_target=False):
    rows = []
    for fold, fold_train, fold_valid in rolling_origin_splits_grouped(df):
        X_train = fold_train[CAT_FEATURES + feature_cols]
        X_valid = fold_valid[CAT_FEATURES + feature_cols]
        if log_target:
            y_train_log = to_log_target(fold_train[TARGET_COL])
            fitted, smearing = fit_log_target_model(model, X_train, y_train_log)
            pred = from_log_target(fitted.predict(X_valid), smearing_factor=smearing)
        else:
            fitted = clone(model)
            fitted.fit(X_train, fold_train[TARGET_COL])
            pred = np.maximum(fitted.predict(X_valid), 0)
        row = metric_row(fold_valid[TARGET_COL], pred)
        row["fold"] = fold
        rows.append(row)
    return pd.DataFrame(rows)


# Grid search with the rolling CV above. Returns the model with the best average (sMAPE, MAPE, RMSE, MAE)
def tune_model(model_name, base_model, param_grid, df_for_tuning, feature_cols, log_target=False):
    rows = []
    for params in ParameterGrid(param_grid):
        model = clone(base_model).set_params(**params)
        cv_metrics = evaluate_cv_model(model, df_for_tuning, feature_cols=feature_cols, log_target=log_target)
        if cv_metrics.empty:
            continue
        avg = cv_metrics[["MAE", "RMSE", "MAPE", "sMAPE"]].mean().to_dict()
        rows.append({"model": model_name, "params": params, **avg})
    if not rows:
        return clone(base_model), pd.DataFrame()
    results = pd.DataFrame(rows).sort_values(METRIC_SORT_ORDER).reset_index(drop=True)
    best_params = results.loc[0, "params"]
    return clone(base_model).set_params(**best_params), results


# One scenario, all ML models. Tune on the training part only, test on the last n_test quarters,
# then refit on ALL data (fitted_refs) for the future forecast. Every model runs twice: raw revenue and log target
def evaluate_holdout_models(
    df,
    raw_feature_cols,
    log_feature_cols,
    n_test=8,
    scenario_name="S1",
    payload_context=None,
    runtime_profile="full",
):
    raw_feature_cols = [col for col in raw_feature_cols if col in df.columns and df[col].notna().any()]
    log_feature_cols = [col for col in log_feature_cols if col in df.columns and df[col].notna().any()]

    train_df, test_df = split_last_n_by_company(df, n_test=n_test)
    raw_models = make_base_models(raw_feature_cols)
    log_models = make_base_models(log_feature_cols)

    tuned_raw_models = {}
    tuned_log_models = {}
    tuning_rows = []
    for name, model in raw_models.items():
        param_grid = get_param_grid_for_model(name, runtime_profile=runtime_profile)
        tuned_raw, tuning = tune_model(name, model, param_grid, train_df, feature_cols=raw_feature_cols, log_target=False)
        log_model = log_models[name]
        tuned_log, tuning_log = tune_model(
            f"{name} log-target",
            log_model,
            param_grid,
            train_df,
            feature_cols=log_feature_cols,
            log_target=True,
        )
        tuned_raw_models[name] = tuned_raw
        tuned_log_models[f"{name} log-target"] = tuned_log
        if not tuning.empty:
            tuning_rows.extend(tuning.to_dict(orient="records"))
        if not tuning_log.empty:
            tuning_rows.extend(tuning_log.to_dict(orient="records"))

    payload_context = dict(payload_context or {})
    all_rows, prediction_rows, fitted_refs = [], [], {}

    X_train_raw = train_df[CAT_FEATURES + raw_feature_cols]
    X_test_raw = test_df[CAT_FEATURES + raw_feature_cols]
    X_train_log = train_df[CAT_FEATURES + log_feature_cols]
    X_test_log = test_df[CAT_FEATURES + log_feature_cols]
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    for model_name, model in tuned_raw_models.items():
        fitted = clone(model).fit(X_train_raw, y_train)
        pred = fitted.predict(X_test_raw)
        row = {
            "scenario": scenario_name,
            "model_family": "machine_learning",
            "model": model_name,
            "target_type": "raw revenue",
            "n_test_points": int(len(y_test)),
            **metric_row(y_test, pred),
        }
        all_rows.append(row)
        fitted_refs[(scenario_name, model_name)] = {
            "model": clone(model).fit(df[CAT_FEATURES + raw_feature_cols], df[TARGET_COL]),
            "log_target": False,
            "smearing": 1.0,
            "feature_cols": list(raw_feature_cols),
            **payload_context,
        }
        for d, actual, p in zip(test_df[DATE_COL], y_test, pred):
            prediction_rows.append(
                {
                    "scenario": scenario_name,
                    "model": model_name,
                    "date": str(pd.to_datetime(d).date()),
                    "actual": float(actual),
                    "predicted": float(p),
                    "split": f"last_{n_test}q",
                }
            )

    for model_name, model in tuned_log_models.items():
        y_train_log = to_log_target(y_train)
        fitted, smearing = fit_log_target_model(model, X_train_log, y_train_log)
        pred = from_log_target(fitted.predict(X_test_log), smearing_factor=smearing)
        row = {
            "scenario": scenario_name,
            "model_family": "machine_learning",
            "model": model_name,
            "target_type": "log1p revenue + Duan smearing",
            "n_test_points": int(len(y_test)),
            **metric_row(y_test, pred),
        }
        all_rows.append(row)
        full_fitted, full_smearing = fit_log_target_model(
            model,
            df[CAT_FEATURES + log_feature_cols],
            to_log_target(df[TARGET_COL]),
        )
        fitted_refs[(scenario_name, model_name)] = {
            "model": full_fitted,
            "log_target": True,
            "smearing": full_smearing,
            "feature_cols": list(log_feature_cols),
            **payload_context,
        }
        for d, actual, p in zip(test_df[DATE_COL], y_test, pred):
            prediction_rows.append(
                {
                    "scenario": scenario_name,
                    "model": model_name,
                    "date": str(pd.to_datetime(d).date()),
                    "actual": float(actual),
                    "predicted": float(p),
                    "split": f"last_{n_test}q",
                }
            )

    return all_rows, prediction_rows, fitted_refs, tuning_rows, train_df, test_df


# Tries a few small ARIMA orders and keeps the one with the lowest AIC
def _fit_best_arima_model(values, candidate_orders):
    from statsmodels.tsa.arima.model import ARIMA

    best_fit = None
    best_order = None
    best_aic = np.inf
    series = np.asarray(values, dtype=float)
    for order in candidate_orders:
        try:
            fitted = ARIMA(series, order=order).fit()
        except Exception:
            continue
        aic = float(getattr(fitted, "aic", np.inf))
        if not np.isfinite(aic):
            aic = np.inf
        if best_fit is None or aic < best_aic:
            best_fit = fitted
            best_order = order
            best_aic = aic
    return best_fit, best_order


# Same for SARIMA (seasonal period 4 = quarters)
def _fit_best_sarima_model(values, candidate_specs):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    best_fit = None
    best_spec = None
    best_aic = np.inf
    series = np.asarray(values, dtype=float)
    for order, seasonal_order in candidate_specs:
        try:
            fitted = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        except Exception:
            continue
        aic = float(getattr(fitted, "aic", np.inf))
        if not np.isfinite(aic):
            aic = np.inf
        if best_fit is None or aic < best_aic:
            best_fit = fitted
            best_spec = (order, seasonal_order)
            best_aic = aic
    return best_fit, best_spec


# Walk-forward splits of a single series (Prophet and LSTM tuning)
def _rolling_series_splits(values, n_splits=2, horizon=4, min_train=16):
    series = pd.Series(values).dropna().astype(float).reset_index(drop=True)
    max_possible = (len(series) - min_train) // horizon
    n_splits_eff = min(n_splits, max_possible)
    if n_splits_eff <= 0:
        return []

    splits = []
    for fold in range(n_splits_eff):
        valid_end = len(series) - (n_splits_eff - fold - 1) * horizon
        valid_start = valid_end - horizon
        train = series.iloc[:valid_start].copy()
        valid = series.iloc[valid_start:valid_end].copy()
        splits.append((fold + 1, train, valid))
    return splits


def _build_prophet_grid():
    return [
        {
            "changepoint_prior_scale": cps,
            "seasonality_prior_scale": sps,
            "seasonality_mode": mode,
            "fourier_order": fourier,
        }
        for cps in [0.01, 0.05, 0.1]
        for sps in [1.0, 5.0]
        for mode in ["additive", "multiplicative"]
        for fourier in [2, 3]
    ]


def _prophet_forecast(train_dates, train_values, steps, params, log_target=False):
    from prophet import Prophet

    train_prophet = pd.DataFrame({"ds": pd.to_datetime(train_dates), "y": np.asarray(train_values, dtype=float)})
    if log_target:
        train_prophet["y"] = np.log1p(train_prophet["y"].clip(lower=0))

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        seasonality_mode=params["seasonality_mode"],
    )
    # note: the seasonality is called "quarterly" but period=365.25 days is one year. Same setting as in the notebooks
    model.add_seasonality(
        name="quarterly",
        period=365.25,
        fourier_order=params["fourier_order"],
    )
    model.fit(train_prophet)

    future_dates = pd.date_range(start=pd.to_datetime(train_dates).max(), periods=steps + 1, freq="QE")[1:]
    future = pd.DataFrame({"ds": future_dates})
    forecast = model.predict(future)["yhat"].values
    if log_target:
        forecast = np.maximum(np.expm1(forecast), 0)
    else:
        forecast = np.maximum(np.asarray(forecast, dtype=float), 0)
    return forecast, model


def _fit_best_prophet_params(train_dates, train_values, log_target=False):
    if not PROPHET_AVAILABLE:
        return None

    best_params = None
    best_score = None
    values = np.asarray(train_values, dtype=float)
    dates = pd.to_datetime(train_dates)
    for params in _build_prophet_grid():
        fold_scores = []
        for _, fold_train_vals, fold_valid_vals in _rolling_series_splits(values, n_splits=2, horizon=4, min_train=16):
            train_end = len(fold_train_vals)
            fold_train_dates = dates[:train_end]
            try:
                pred, _ = _prophet_forecast(fold_train_dates, fold_train_vals.values, steps=len(fold_valid_vals), params=params, log_target=log_target)
            except Exception:
                continue
            fold_scores.append(metric_row(fold_valid_vals.values, pred))
        if not fold_scores:
            continue
        avg = {
            key: float(np.mean([row[key] for row in fold_scores if row.get(key) is not None]))
            for key in ["MAE", "RMSE", "MAPE", "sMAPE"]
        }
        score = _metric_sort_key(avg)
        if best_score is None or score < best_score:
            best_score = score
            best_params = params
    return best_params


def _make_lstm_supervised(values, lookback):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        y.append(values[i])
    return np.asarray(X), np.asarray(y)


def _build_lstm_grid():
    return [
        {"lookback": 4, "units": 16, "dropout": 0.1, "batch_size": 16, "epochs": 8},
    ]


# Small LSTM: looks back 4 quarters, predicts the next one, values scaled to [0, 1], a few epochs with early stopping.
# Multi-step forecast = feed its own predictions back in
def _lstm_forecast(train_values, steps, params, log_target=False):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
    from sklearn.preprocessing import MinMaxScaler

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    values = np.asarray(train_values, dtype=float)
    if log_target:
        values = np.log1p(np.clip(values, 0, None))

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()

    lookback = params["lookback"]
    X, y = _make_lstm_supervised(scaled, lookback)
    if len(X) < 8:
        raise ValueError("Too few observations for LSTM")

    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(params["units"]),
        Dropout(params["dropout"]),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    validation_split = 0.2 if len(X) >= 12 else 0.0
    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    model.fit(
        X,
        y,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=0,
        callbacks=[early_stop] if validation_split > 0 else [],
        shuffle=False,
        validation_split=validation_split,
    )

    history = scaled.tolist()
    preds_scaled = []
    for _ in range(steps):
        x_input = np.asarray(history[-lookback:]).reshape((1, lookback, 1))
        next_scaled = float(model.predict(x_input, verbose=0)[0, 0])
        preds_scaled.append(next_scaled)
        history.append(next_scaled)

    pred = scaler.inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).flatten()
    if log_target:
        pred = np.maximum(np.expm1(pred), 0)
    return np.maximum(np.asarray(pred, dtype=float), 0), model, scaler


def _lstm_forecast_worker(result_queue, train_values, steps, params, log_target):
    try:
        pred, _, scaler = _lstm_forecast(train_values, steps=steps, params=params, log_target=log_target)
        result_queue.put(
            {
                "ok": True,
                "predictions": [float(x) for x in pred],
                "scaler_min": float(scaler.data_min_[0]),
                "scaler_max": float(scaler.data_max_[0]),
            }
        )
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)})


# TensorFlow runs in a forked child process. If it hangs or crashes the API stays alive.
# note: 'fork' doesn't exist on Windows
def _lstm_forecast_with_timeout(train_values, steps, params, log_target=False, timeout_seconds=LSTM_TIMEOUT_SECONDS):
    values = [float(x) for x in np.asarray(train_values, dtype=float)]
    ctx = mp.get_context("fork")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_lstm_forecast_worker,
        args=(result_queue, values, int(steps), dict(params), bool(log_target)),
    )
    print(f"[LSTM] Starting TensorFlow worker with {timeout_seconds}s timeout.", flush=True)
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        print(f"[LSTM] Timeout reached; stopping TensorFlow worker pid={process.pid}.", flush=True)
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        return None, f"LSTM timed out after {timeout_seconds}s."
    if process.exitcode not in (0, None):
        return None, f"LSTM worker exited with code {process.exitcode}."
    if result_queue.empty():
        return None, "LSTM worker produced no result."
    result = result_queue.get()
    if not result.get("ok"):
        return None, result.get("error", "LSTM failed.")
    return result, None


# note: not called at the moment. evaluate_time_series_models() just takes the only entry of _build_lstm_grid()
def _fit_best_lstm_params(train_values, log_target=False):
    if not TENSORFLOW_AVAILABLE:
        return None
    values = np.asarray(train_values, dtype=float)
    best_params = None
    best_score = None
    for params in _build_lstm_grid():
        fold_scores = []
        for _, fold_train_vals, fold_valid_vals in _rolling_series_splits(values, n_splits=1, horizon=4, min_train=20):
            try:
                result, error = _lstm_forecast_with_timeout(
                    fold_train_vals.values,
                    steps=len(fold_valid_vals),
                    params=params,
                    log_target=log_target,
                )
                if error:
                    continue
                pred = np.asarray(result["predictions"], dtype=float)
            except Exception:
                continue
            fold_scores.append(metric_row(fold_valid_vals.values, pred))
        if not fold_scores:
            continue
        avg = {
            key: float(np.mean([row[key] for row in fold_scores if row.get(key) is not None]))
            for key in ["MAE", "RMSE", "MAPE", "sMAPE"]
        }
        score = _metric_sort_key(avg)
        if best_score is None or score < best_score:
            best_score = score
            best_params = params
    return best_params


# Time-series models (ARIMA, SARIMA, Prophet, LSTM and their log versions) on one company's revenue alone (scenario 'TS').
# Same holdout as the ML models so the metrics can be compared. Every model has its own try/except:
# if one fails the reason goes to model_skip_reason and the others still run
def evaluate_time_series_models(df, n_test=8, include_lstm=True, runtime_profile="full"):
    model_skip_reason = {}
    all_rows, prediction_rows, fitted_refs = [], [], {}
    if len(df) <= n_test:
        model_skip_reason["ARIMA"] = f"Insufficient history for holdout n_test={n_test}."
        model_skip_reason["ARIMA log-target"] = f"Insufficient history for holdout n_test={n_test}."
        model_skip_reason["SARIMA"] = f"Insufficient history for holdout n_test={n_test}."
        model_skip_reason["SARIMA log-target"] = f"Insufficient history for holdout n_test={n_test}."
        model_skip_reason["Prophet"] = f"Insufficient history for holdout n_test={n_test}."
        model_skip_reason["Prophet log-target"] = f"Insufficient history for holdout n_test={n_test}."
        model_skip_reason["LSTM"] = f"Insufficient history for holdout n_test={n_test}."
        model_skip_reason["LSTM log-target"] = f"Insufficient history for holdout n_test={n_test}."
        return all_rows, prediction_rows, fitted_refs, model_skip_reason

    ts_df = df[[DATE_COL, TARGET_COL, GROUP_COL, "quarter"]].dropna(subset=[DATE_COL, TARGET_COL]).sort_values(DATE_COL).reset_index(drop=True)
    train_df = ts_df.iloc[:-n_test].copy()
    test_df = ts_df.iloc[-n_test:].copy()
    y_train = train_df[TARGET_COL].astype(float).values
    y_test = test_df[TARGET_COL].astype(float).values

    arima_candidate_orders = [(1, 1, 1), (1, 0, 0), (0, 1, 1), (2, 1, 1)]
    sarima_candidate_specs = [
        ((1, 1, 1), (0, 0, 0, 4)),
        ((1, 1, 1), (1, 0, 0, 4)),
        ((1, 1, 1), (0, 1, 1, 4)),
        ((2, 1, 1), (1, 0, 0, 4)),
    ]

    # ARIMA
    if not ARIMA_AVAILABLE:
        model_skip_reason["ARIMA"] = "statsmodels not installed."
        model_skip_reason["ARIMA log-target"] = "statsmodels not installed."
        model_skip_reason["SARIMA"] = "statsmodels not installed."
        model_skip_reason["SARIMA log-target"] = "statsmodels not installed."
    else:
        try:
            arima_fit, arima_order = _fit_best_arima_model(y_train, arima_candidate_orders)
            if arima_fit is None:
                model_skip_reason["ARIMA"] = "ARIMA failed to fit."
            else:
                pred = np.asarray(arima_fit.forecast(steps=n_test), dtype=float)
                pred = np.maximum(pred, 0)
                all_rows.append(
                    {
                        "scenario": "TS",
                        "model_family": "time_series",
                        "model": "ARIMA",
                        "target_type": "revenue time series",
                        "n_test_points": int(len(y_test)),
                        **metric_row(y_test, pred),
                    }
                )
                for d, actual, p in zip(test_df[DATE_COL], y_test, pred):
                    prediction_rows.append(
                        {
                            "scenario": "TS",
                            "model": "ARIMA",
                            "date": str(pd.to_datetime(d).date()),
                            "actual": float(actual),
                            "predicted": float(p),
                            "split": f"last_{n_test}q",
                        }
                    )
                full_fit, _ = _fit_best_arima_model(ts_df[TARGET_COL].astype(float).values, [arima_order])
                fitted_refs[("TS", "ARIMA")] = {
                    "model_family": "time_series",
                    "ts_type": "arima",
                    "model": full_fit,
                    "order": arima_order,
                    "log_target": False,
                    "smearing": 1.0,
                }
        except Exception:
            model_skip_reason["ARIMA"] = "ARIMA unavailable or incompatible at runtime."

        # ARIMA log-target
        try:
            train_log = to_log_target(y_train)
            arima_log_fit, arima_log_order = _fit_best_arima_model(train_log, arima_candidate_orders)
            if arima_log_fit is None:
                model_skip_reason["ARIMA log-target"] = "ARIMA log-target failed to fit."
            else:
                train_fitted_log = np.asarray(arima_log_fit.fittedvalues, dtype=float)
                usable_len = min(len(train_log), len(train_fitted_log))
                smearing = duan_smearing_factor(train_log[-usable_len:], train_fitted_log[-usable_len:]) if usable_len else 1.0
                if not np.isfinite(smearing) or smearing <= 0:
                    smearing = 1.0
                pred = from_log_target(arima_log_fit.forecast(steps=n_test), smearing_factor=smearing)
                all_rows.append(
                    {
                        "scenario": "TS",
                        "model_family": "time_series",
                        "model": "ARIMA log-target",
                        "target_type": "log1p revenue",
                        "n_test_points": int(len(y_test)),
                        **metric_row(y_test, pred),
                    }
                )
                for d, actual, p in zip(test_df[DATE_COL], y_test, pred):
                    prediction_rows.append(
                        {
                            "scenario": "TS",
                            "model": "ARIMA log-target",
                            "date": str(pd.to_datetime(d).date()),
                            "actual": float(actual),
                            "predicted": float(p),
                            "split": f"last_{n_test}q",
                        }
                    )
                full_log = to_log_target(ts_df[TARGET_COL].astype(float).values)
                full_fit, _ = _fit_best_arima_model(full_log, [arima_log_order])
                full_fitted_log = np.asarray(full_fit.fittedvalues, dtype=float) if full_fit is not None else np.asarray([])
                full_usable_len = min(len(full_log), len(full_fitted_log))
                full_smearing = duan_smearing_factor(full_log[-full_usable_len:], full_fitted_log[-full_usable_len:]) if full_usable_len else 1.0
                if not np.isfinite(full_smearing) or full_smearing <= 0:
                    full_smearing = 1.0
                fitted_refs[("TS", "ARIMA log-target")] = {
                    "model_family": "time_series",
                    "ts_type": "arima_log",
                    "model": full_fit,
                    "order": arima_log_order,
                    "log_target": True,
                    "smearing": full_smearing,
                }
        except Exception:
            model_skip_reason["ARIMA log-target"] = "ARIMA log-target unavailable or incompatible at runtime."

        # SARIMA
        try:
            sarima_fit, sarima_spec = _fit_best_sarima_model(y_train, sarima_candidate_specs)
            if sarima_fit is None:
                model_skip_reason["SARIMA"] = "SARIMA failed to fit."
            else:
                pred = np.asarray(sarima_fit.forecast(steps=n_test), dtype=float)
                pred = np.maximum(pred, 0)
                all_rows.append(
                    {
                        "scenario": "TS",
                        "model_family": "time_series",
                        "model": "SARIMA",
                        "target_type": "seasonal revenue time series",
                        "n_test_points": int(len(y_test)),
                        **metric_row(y_test, pred),
                    }
                )
                for d, actual, p in zip(test_df[DATE_COL], y_test, pred):
                    prediction_rows.append(
                        {
                            "scenario": "TS",
                            "model": "SARIMA",
                            "date": str(pd.to_datetime(d).date()),
                            "actual": float(actual),
                            "predicted": float(p),
                            "split": f"last_{n_test}q",
                        }
                    )
                full_fit, _ = _fit_best_sarima_model(ts_df[TARGET_COL].astype(float).values, [sarima_spec])
                fitted_refs[("TS", "SARIMA")] = {
                    "model_family": "time_series",
                    "ts_type": "sarima",
                    "model": full_fit,
                    "order": sarima_spec[0],
                    "seasonal_order": sarima_spec[1],
                    "log_target": False,
                    "smearing": 1.0,
                }
        except Exception:
            model_skip_reason["SARIMA"] = "SARIMA unavailable or incompatible at runtime."

        # SARIMA log-target
        try:
            train_log = to_log_target(y_train)
            sarima_log_fit, sarima_log_spec = _fit_best_sarima_model(train_log, sarima_candidate_specs)
            if sarima_log_fit is None:
                model_skip_reason["SARIMA log-target"] = "SARIMA log-target failed to fit."
            else:
                train_fitted_log = np.asarray(sarima_log_fit.fittedvalues, dtype=float)
                usable_len = min(len(train_log), len(train_fitted_log))
                smearing = duan_smearing_factor(train_log[-usable_len:], train_fitted_log[-usable_len:]) if usable_len else 1.0
                if not np.isfinite(smearing) or smearing <= 0:
                    smearing = 1.0
                pred = from_log_target(sarima_log_fit.forecast(steps=n_test), smearing_factor=smearing)
                all_rows.append(
                    {
                        "scenario": "TS",
                        "model_family": "time_series",
                        "model": "SARIMA log-target",
                        "target_type": "log1p revenue",
                        "n_test_points": int(len(y_test)),
                        **metric_row(y_test, pred),
                    }
                )
                for d, actual, p in zip(test_df[DATE_COL], y_test, pred):
                    prediction_rows.append(
                        {
                            "scenario": "TS",
                            "model": "SARIMA log-target",
                            "date": str(pd.to_datetime(d).date()),
                            "actual": float(actual),
                            "predicted": float(p),
                            "split": f"last_{n_test}q",
                        }
                    )
                full_log = to_log_target(ts_df[TARGET_COL].astype(float).values)
                full_fit, _ = _fit_best_sarima_model(full_log, [sarima_log_spec])
                full_fitted_log = np.asarray(full_fit.fittedvalues, dtype=float) if full_fit is not None else np.asarray([])
                full_usable_len = min(len(full_log), len(full_fitted_log))
                full_smearing = duan_smearing_factor(full_log[-full_usable_len:], full_fitted_log[-full_usable_len:]) if full_usable_len else 1.0
                if not np.isfinite(full_smearing) or full_smearing <= 0:
                    full_smearing = 1.0
                fitted_refs[("TS", "SARIMA log-target")] = {
                    "model_family": "time_series",
                    "ts_type": "sarima_log",
                    "model": full_fit,
                    "order": sarima_log_spec[0],
                    "seasonal_order": sarima_log_spec[1],
                    "log_target": True,
                    "smearing": full_smearing,
                }
        except Exception:
            model_skip_reason["SARIMA log-target"] = "SARIMA log-target unavailable or incompatible at runtime."

    # Prophet
    if not PROPHET_AVAILABLE:
        model_skip_reason["Prophet"] = "Package not installed."
        model_skip_reason["Prophet log-target"] = "Package not installed."
    else:
        try:
            best_params = _fit_best_prophet_params(train_df[DATE_COL].values, y_train, log_target=False)
            if best_params is None:
                model_skip_reason["Prophet"] = "No valid Prophet parameter set fit successfully."
            else:
                pred, prophet_model = _prophet_forecast(train_df[DATE_COL].values, y_train, steps=n_test, params=best_params, log_target=False)
                pred = np.maximum(np.asarray(pred, dtype=float), 0)
                all_rows.append(
                    {
                        "scenario": "TS",
                        "model_family": "time_series",
                        "model": "Prophet",
                        "target_type": "date + revenue",
                        "n_test_points": int(len(y_test)),
                        **metric_row(y_test, pred),
                    }
                )
                for d, actual, p in zip(test_df[DATE_COL], y_test, pred):
                    prediction_rows.append(
                        {
                            "scenario": "TS",
                            "model": "Prophet",
                            "date": str(pd.to_datetime(d).date()),
                            "actual": float(actual),
                            "predicted": float(p),
                            "split": f"last_{n_test}q",
                        }
                    )
                _, prophet_full = _prophet_forecast(ts_df[DATE_COL].values, ts_df[TARGET_COL].astype(float).values, steps=1, params=best_params, log_target=False)
                fitted_refs[("TS", "Prophet")] = {
                    "model_family": "time_series",
                    "ts_type": "prophet",
                    "model": prophet_full,
                    "params": best_params,
                    "log_target": False,
                    "smearing": 1.0,
                }
        except Exception:
            model_skip_reason["Prophet"] = "Prophet unavailable or incompatible at runtime."

        try:
            best_params_log = _fit_best_prophet_params(train_df[DATE_COL].values, y_train, log_target=True)
            if best_params_log is None:
                model_skip_reason["Prophet log-target"] = "No valid Prophet log-target parameter set fit successfully."
            else:
                pred, prophet_log_model = _prophet_forecast(train_df[DATE_COL].values, y_train, steps=n_test, params=best_params_log, log_target=True)
                all_rows.append(
                    {
                        "scenario": "TS",
                        "model_family": "time_series",
                        "model": "Prophet log-target",
                        "target_type": "log1p revenue",
                        "n_test_points": int(len(y_test)),
                        **metric_row(y_test, pred),
                    }
                )
                for d, actual, p in zip(test_df[DATE_COL], y_test, pred):
                    prediction_rows.append(
                        {
                            "scenario": "TS",
                            "model": "Prophet log-target",
                            "date": str(pd.to_datetime(d).date()),
                            "actual": float(actual),
                            "predicted": float(p),
                            "split": f"last_{n_test}q",
                        }
                    )
                _, prophet_log_full = _prophet_forecast(ts_df[DATE_COL].values, ts_df[TARGET_COL].astype(float).values, steps=1, params=best_params_log, log_target=True)
                fitted_refs[("TS", "Prophet log-target")] = {
                    "model_family": "time_series",
                    "ts_type": "prophet_log",
                    "model": prophet_log_full,
                    "params": best_params_log,
                    "log_target": True,
                    "smearing": 1.0,
                }
        except Exception:
            model_skip_reason["Prophet log-target"] = "Prophet log-target unavailable or incompatible at runtime."

    # LSTM
    if not include_lstm:
        model_skip_reason["LSTM"] = f"Skipped in {runtime_profile} runtime profile."
        model_skip_reason["LSTM log-target"] = f"Skipped in {runtime_profile} runtime profile."
    elif not TENSORFLOW_AVAILABLE:
        model_skip_reason["LSTM"] = "TensorFlow not installed."
        model_skip_reason["LSTM log-target"] = "TensorFlow not installed."
    else:
        if len(y_train) <= 8:
            model_skip_reason["LSTM"] = "Insufficient history for LSTM lookback."
            model_skip_reason["LSTM log-target"] = "Insufficient history for LSTM lookback."
        else:
            try:
                best_lstm_params = _build_lstm_grid()[0]
                lstm_result, lstm_error = _lstm_forecast_with_timeout(
                    y_train,
                    steps=n_test,
                    params=best_lstm_params,
                    log_target=False,
                )
                if lstm_error:
                    raise TimeoutError(lstm_error)
                pred = np.asarray(lstm_result["predictions"], dtype=float)
                all_rows.append(
                    {
                        "scenario": "TS",
                        "model_family": "time_series",
                        "model": "LSTM",
                        "target_type": "revenue sequence",
                        "n_test_points": int(len(y_test)),
                        **metric_row(y_test, pred),
                    }
                )
                for d, actual, p in zip(test_df[DATE_COL], y_test, pred):
                    prediction_rows.append(
                        {
                            "scenario": "TS",
                            "model": "LSTM",
                            "date": str(pd.to_datetime(d).date()),
                            "actual": float(actual),
                            "predicted": float(p),
                            "split": f"last_{n_test}q",
                        }
                    )
                fitted_refs[("TS", "LSTM")] = {
                    "model_family": "time_series",
                    "ts_type": "lstm",
                    "params": dict(best_lstm_params),
                    "lookback": int(best_lstm_params["lookback"]),
                    "scaler_min": float(lstm_result["scaler_min"]),
                    "scaler_max": float(lstm_result["scaler_max"]),
                    "log_target": False,
                    "smearing": 1.0,
                }
            except TimeoutError as exc:
                model_skip_reason["LSTM"] = str(exc)
            except Exception:
                model_skip_reason["LSTM"] = "TensorFlow unavailable or incompatible at runtime."

            model_skip_reason["LSTM log-target"] = "Skipped to keep LSTM bounded to one 4-minute attempt during upload."

    return all_rows, prediction_rows, fitted_refs, model_skip_reason


# Features of the newest (future) row, built from the history + the predictions made so far
def _build_future_feature_row(work_df, feature_cols, macro_lag_cols, micro_cols):
    engineered = add_no_macro_feature_columns(work_df)
    if micro_cols:
        engineered, _ = add_micro_features(engineered, micro_cols=micro_cols)
    row = engineered.iloc[[-1]].copy()

    for col in macro_lag_cols:
        if col not in row.columns:
            row[col] = np.nan
        if pd.isna(row[col].iloc[0]):
            source_col = col.replace("_lag_1q", "")
            if source_col in work_df.columns and work_df[source_col].notna().any():
                row[col] = work_df[source_col].dropna().iloc[-1]
    row = row.reindex(columns=CAT_FEATURES + feature_cols)
    return row


# Long-horizon forecast. Time-series models forecast all quarters in one go.
# ML models go recursively: add an empty future quarter, rebuild the features, predict, write the prediction into `revenue`,
# repeat. Errors pile up over 28 quarters, that's what _check_forecast_stability is for
def recursive_forecast_company(history_df, fitted_payload, feature_cols, macro_lag_cols, micro_cols, steps=FORECAST_STEPS, future_dates=None):
    if fitted_payload.get("model_family") == "time_series":
        ts_type = fitted_payload.get("ts_type")
        base = history_df.sort_values(DATE_COL).reset_index(drop=True)
        last_date = base[DATE_COL].max()
        resolved_future_dates = list(future_dates or future_quarter_dates(last_date=last_date))
        steps = len(resolved_future_dates) if resolved_future_dates else steps
        out = []
        if ts_type == "arima":
            pred = np.asarray(fitted_payload["model"].forecast(steps=steps), dtype=float)
            pred = np.maximum(pred, 0)
            for future_date, value in zip(resolved_future_dates, pred):
                out.append({"date": str(future_date.date()), "predicted_revenue": float(value)})
            return out
        if ts_type == "arima_log":
            pred_log = np.asarray(fitted_payload["model"].forecast(steps=steps), dtype=float)
            pred = from_log_target(pred_log, smearing_factor=fitted_payload.get("smearing", 1.0))
            for future_date, value in zip(resolved_future_dates, pred):
                out.append({"date": str(future_date.date()), "predicted_revenue": float(value)})
            return out
        if ts_type == "sarima":
            pred = np.asarray(fitted_payload["model"].forecast(steps=steps), dtype=float)
            pred = np.maximum(pred, 0)
            for future_date, value in zip(resolved_future_dates, pred):
                out.append({"date": str(future_date.date()), "predicted_revenue": float(value)})
            return out
        if ts_type == "sarima_log":
            pred_log = np.asarray(fitted_payload["model"].forecast(steps=steps), dtype=float)
            pred = from_log_target(pred_log, smearing_factor=fitted_payload.get("smearing", 1.0))
            for future_date, value in zip(resolved_future_dates, pred):
                out.append({"date": str(future_date.date()), "predicted_revenue": float(value)})
            return out
        if ts_type == "prophet":
            pred = fitted_payload["model"].predict(pd.DataFrame({"ds": resolved_future_dates}))["yhat"].values
            pred = np.maximum(np.asarray(pred, dtype=float), 0)
            for dt, pv in zip(resolved_future_dates, pred):
                out.append({"date": str(pd.Timestamp(dt).date()), "predicted_revenue": float(pv)})
            return out
        if ts_type == "prophet_log":
            pred_log = fitted_payload["model"].predict(pd.DataFrame({"ds": resolved_future_dates}))["yhat"].values
            pred = np.maximum(np.expm1(np.asarray(pred_log, dtype=float)), 0)
            for dt, pv in zip(resolved_future_dates, pred):
                out.append({"date": str(pd.Timestamp(dt).date()), "predicted_revenue": float(pv)})
            return out
        if ts_type == "lstm":
            result, error = _lstm_forecast_with_timeout(
                base[TARGET_COL].astype(float).values,
                steps=steps,
                params=fitted_payload["params"],
                log_target=False,
            )
            if error:
                return []
            for future_date, value in zip(resolved_future_dates, result["predictions"]):
                out.append({"date": str(future_date.date()), "predicted_revenue": float(value)})
            return out
        if ts_type == "lstm_log":
            result, error = _lstm_forecast_with_timeout(
                base[TARGET_COL].astype(float).values,
                steps=steps,
                params=fitted_payload["params"],
                log_target=True,
            )
            if error:
                return []
            for future_date, value in zip(resolved_future_dates, result["predictions"]):
                out.append({"date": str(future_date.date()), "predicted_revenue": float(value)})
            return out

    work = history_df.copy().sort_values(DATE_COL).reset_index(drop=True)
    work = _ensure_quarter_fields(work)
    out = []
    last_date = work[DATE_COL].max()
    resolved_future_dates = list(future_dates or future_quarter_dates(last_date=last_date))
    model = fitted_payload["model"]
    log_target = fitted_payload["log_target"]
    smearing = fitted_payload["smearing"]

    for future_date in resolved_future_dates:
        new_row = {GROUP_COL: work[GROUP_COL].iloc[0], DATE_COL: future_date, TARGET_COL: np.nan}
        for col in micro_cols:
            if col in work.columns:
                new_row[col] = np.nan
        for lag_col in macro_lag_cols:
            source_col = lag_col.replace("_lag_1q", "")
            if source_col in work.columns and work[source_col].notna().any():
                new_row[source_col] = float(work[source_col].dropna().iloc[-1])

        work = pd.concat([work, pd.DataFrame([new_row])], ignore_index=True, sort=False)
        work = _ensure_quarter_fields(work)
        X_future = _build_future_feature_row(work, feature_cols, macro_lag_cols, micro_cols)
        pred = float(model.predict(X_future)[0])
        if log_target:
            pred = float(from_log_target([pred], smearing_factor=smearing)[0])
        else:
            pred = float(pred) if np.isfinite(pred) else np.nan
            if np.isfinite(pred):
                pred = float(np.maximum(pred, 0))
        work.loc[work.index[-1], TARGET_COL] = pred
        out.append({"date": str(pd.to_datetime(future_date).date()), "year": int(future_date.year), "quarter": int(future_date.quarter), "predicted_revenue": pred})
    return out


# Sanity rules for a long recursive forecast: not empty / NaN / negative, not below 0.2x the historical minimum,
# not above 2x the historical maximum or the last value, doesn't end near zero, the first 16 quarters stay within
# 0.55x-1.8x, no >80% growth that history doesn't support, direction doesn't fight the recent trend, not flat.
# Returns (passed, reason). It only reports, it doesn't change which model is deployed
def _check_forecast_stability(forecast_rows, historical_values, target_type):
    vals = np.asarray([r["predicted_revenue"] for r in forecast_rows], dtype=float)
    hist = np.asarray(historical_values, dtype=float)
    hist = hist[np.isfinite(hist)]
    stability_horizon = min(FORECAST_STEPS, len(vals))
    stability_vals = vals[:stability_horizon] if stability_horizon else vals
    if vals.size == 0:
        return False, "Rejected: empty recursive forecast."
    if not np.all(np.isfinite(vals)):
        return False, "Rejected: forecast contains NaN or inf."

    if np.any(vals < 0):
        return False, "Rejected: forecast contains negative values."

    if hist.size > 0:
        positive_hist = hist[hist > 0]
        hist_min = float(np.min(positive_hist)) if positive_hist.size else float(np.min(hist))
        hist_max = float(np.max(hist))
        hist_last = float(hist[-1])
        min_floor = 0.2 * hist_min
        max_cap = 2.0 * hist_max
        last_cap = 2.0 * max(hist_last, 1e-9)
        if np.any(vals < min_floor):
            return False, f"Rejected: forecast drops below 0.2x historical min ({min_floor:.6g})."
        if np.any(vals > last_cap):
            return False, f"Rejected: forecast exceeds 2.0x last historical value ({last_cap:.6g})."
        if np.any(vals > max_cap):
            return False, f"Rejected: forecast exceeds 2.0x historical max ({max_cap:.6g})."
        near_zero_floor = max(1e-6, 0.02 * hist_max)
        if vals[-1] < near_zero_floor:
            return False, f"Rejected: forecast ends near zero (last={vals[-1]:.6g}, floor={near_zero_floor:.6g})."

        # Rule 6 + 7: growth guards over the full 16Q horizon.
        first_fc = float(max(stability_vals[0], 1e-9))
        last_fc = float(max(stability_vals[-1], 1e-9))
        growth_16q = (last_fc / first_fc) - 1.0
        ratio_16q = last_fc / first_fc
        if ratio_16q > 1.8 or ratio_16q < 0.55:
            return False, f"Rejected: first-{stability_horizon}Q forecast ratio out of bounds ({ratio_16q:.4g})."

        hist_ref_idx = stability_horizon + 1
        hist_ref = hist[-hist_ref_idx] if len(hist) >= hist_ref_idx else hist[0]
        hist_ref = float(max(hist_ref, 1e-9))
        hist_growth_16q = (hist_last / hist_ref) - 1.0
        if abs(growth_16q) > 0.80 and abs(hist_growth_16q) < (0.6 * abs(growth_16q)):
            return False, (
                f"Rejected: absolute first-{stability_horizon}Q growth is greater than 80% and not supported by historical growth "
                f"(forecast={growth_16q:.4g}, historical={hist_growth_16q:.4g})."
            )

        # Rule 8: forecast direction contradicts recent history too strongly.
        recent_hist = hist[-8:] if len(hist) >= 8 else hist
        hist_trend = float(recent_hist[-1] - recent_hist[0]) if len(recent_hist) >= 2 else 0.0
        fc_trend = float(vals[-1] - vals[0]) if len(vals) >= 2 else 0.0
        if (
            np.sign(hist_trend) != 0
            and np.sign(fc_trend) != 0
            and np.sign(hist_trend) != np.sign(fc_trend)
            and abs(fc_trend) > 0.25 * max(abs(vals[0]), 1e-9)
            and abs(hist_trend) > 0.10 * max(abs(recent_hist[0]), 1e-9)
        ):
            return False, "Rejected: forecast direction contradicts recent historical trend too strongly."

    # Flatness guard: very low variation over the recursive horizon.
    mean_abs = float(max(np.mean(np.abs(vals)), 1e-9))
    rel_range = float((np.max(vals) - np.min(vals)) / mean_abs)
    coef_var = float(np.std(vals) / mean_abs)
    if len(vals) >= 8 and rel_range < 0.02 and coef_var < 0.01:
        return False, "Rejected: forecast is unnaturally flat."

    return True, "Accepted: long-horizon recursive forecast passed sanity checks."


# empty hook, left from debugging
def _debug_candidate(model_name, scenario, candidate, forecast_rows, reason):
    return None


# Main entry point. Steps:
# 1) load the CSV and merge the macro table  2) engineer features  3) build the 4 scenarios
# 4) tune + test all ML models per scenario  5) time-series models  6) ensembles  7) optional 16-quarter robustness check
# 8) rank everything, pick the deployment model, forecast until 2032, run the stability check
# 9) save JSON files to data/results.
# runtime_profile: 'balanced' = small grids, no LSTM;  'full_safe' = big grids + 16Q check, no LSTM;  'full' = everything
def run_pipeline(csv_path, company_name="uploaded_company", runtime_profile="full"):
    if runtime_profile not in {"balanced", "full_safe", "full"}:
        raise ValueError("runtime_profile must be 'balanced', 'full_safe', or 'full'.")

    include_lstm = runtime_profile == "full"
    enable_16q_robustness = runtime_profile in {"full_safe", "full"}
    company_name = Path(company_name).stem
    raw = load_company_csv(csv_path)
    raw[GROUP_COL] = company_name
    raw = _ensure_quarter_fields(raw)

    if MACRO_PATH.exists():
        macro_df = pd.read_csv(MACRO_PATH)
        if {"date"}.issubset(macro_df.columns):
            macro_df["date"] = pd.to_datetime(macro_df["date"], errors="coerce")
            macro_df["year"] = macro_df["date"].dt.year
            macro_df["quarter"] = macro_df["date"].dt.quarter
            merge_cols = [c for c in macro_df.columns if c not in {"date"}]
            macro_df = macro_df[merge_cols].drop_duplicates(["year", "quarter"])
            raw = raw.merge(macro_df, on=["year", "quarter"], how="left")

    macro_base_cols = _infer_macro_columns(raw)
    micro_base_cols = _infer_micro_columns(raw, macro_base_cols)

    full_df = add_no_macro_feature_columns(raw)
    full_df, macro_lag_cols = add_macro_lag_features(full_df, macro_base_cols)
    full_df, micro_feature_cols = add_micro_features(full_df, micro_base_cols)

    def _derived_micro_features(raw_cols):
        out = []
        for base_col in raw_cols:
            out.extend(
                [
                    f"{base_col}_lag_1",
                    f"{base_col}_lag_4",
                    f"{base_col}_growth_qoq",
                    f"{base_col}_margin_lag1",
                ]
            )
        return [c for c in out if c in full_df.columns]

    # The four feature sets compared in the diploma:
    # S1 revenue features only, S2 + company (micro) indicators, S3 + macro, S4 + micro + macro
    scenario_defs = [
        {"scenario": "S1", "name": "engineered_only", "include_macro": False, "include_micro": False},
        {"scenario": "S2", "name": "engineered_plus_micro", "include_macro": False, "include_micro": True},
        {"scenario": "S3", "name": "engineered_plus_macro", "include_macro": True, "include_micro": False},
        {"scenario": "S4", "name": "engineered_plus_micro_plus_macro", "include_macro": True, "include_micro": True},
    ]

    scenario_data = {}
    scenario_final = {}
    scenario_skip_reason = {}
    attempted_scenarios = [sc["scenario"] for sc in scenario_defs]
    successful_scenarios = []
    skipped_scenarios = []
    min_rows = None
    for sc in scenario_defs:
        macro_for_sc = list(macro_lag_cols) if sc["include_macro"] else []
        micro_raw_for_sc = list(micro_base_cols) if sc["include_micro"] else []
        micro_features_for_sc = _derived_micro_features(micro_raw_for_sc)
        if sc["include_macro"]:
            numeric_features = list(MODEL_FEATURES_NO_MACRO + macro_for_sc)
            log_features = list(MODEL_FEATURES_LOG_TARGET + macro_for_sc)
        elif sc["include_micro"]:
            numeric_features = list(MODEL_FEATURES_NO_MACRO + micro_features_for_sc)
            log_features = list(MODEL_FEATURES_LOG_TARGET + micro_features_for_sc)
        else:
            numeric_features = list(MODEL_FEATURES_NO_MACRO)
            log_features = list(MODEL_FEATURES_LOG_TARGET)
        if sc["include_macro"] and sc["include_micro"]:
            numeric_features = list(MODEL_FEATURES_NO_MACRO + macro_for_sc + micro_features_for_sc)
            log_features = list(MODEL_FEATURES_LOG_TARGET + macro_for_sc + micro_features_for_sc)
        features_for_sc = [c for c in sorted(set(numeric_features + log_features)) if c in full_df.columns]

        rows_before = int(len(full_df))
        # Keep notebook-like coverage for scenario datasets: models handle sparse engineered
        # features via imputers, so we only require a valid target row here.
        df_sc = full_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

        rows_after = int(len(df_sc))
        n_test_sc = 8 if rows_after >= 24 else max(4, rows_after // 4) if rows_after > 0 else 0
        n_train_sc = max(0, rows_after - n_test_sc)
        scenario_data[sc["scenario"]] = df_sc
        scenario_final[sc["scenario"]] = {
            "features": features_for_sc,
            "macro_lag_cols": macro_for_sc,
            "micro_raw_cols": micro_raw_for_sc,
            "log_features": [c for c in log_features if c in full_df.columns and full_df[c].notna().any()],
            "numeric_features": [c for c in numeric_features if c in full_df.columns and full_df[c].notna().any()],
        }

        if rows_after <= 0:
            scenario_skip_reason[sc["scenario"]] = "No usable rows after dropna for required feature set."
            skipped_scenarios.append(sc["scenario"])
            continue
        if len(df_sc) > 0:
            min_rows = len(df_sc) if min_rows is None else min(min_rows, len(df_sc))
    if min_rows is None:
        raise ValueError("No usable rows after feature engineering.")
    # S4 (the richest table) is the reference for the last date and the historical values
    model_df = scenario_data.get("S4", full_df)
    if model_df.empty:
        model_df = max(scenario_data.values(), key=len)

    test_size = 8 if min_rows >= 24 else max(4, min_rows // 4)
    rows_8q, preds_8q, fitted_refs, tuning_rows = [], [], {}, []
    for sc in scenario_defs:
        df_sc = scenario_data[sc["scenario"]]
        sc_final = scenario_final[sc["scenario"]]
        if len(df_sc) < test_size + 1:
            if sc["scenario"] not in scenario_skip_reason:
                scenario_skip_reason[sc["scenario"]] = f"Insufficient rows for split: rows_after_dropna={len(df_sc)}, n_test={test_size}."
                skipped_scenarios.append(sc["scenario"])
            continue
        r, p, f, t, _, _ = evaluate_holdout_models(
            df_sc,
            sc_final["numeric_features"],
            sc_final["log_features"],
            n_test=test_size,
            scenario_name=sc["scenario"],
            payload_context={"macro_lag_cols": sc_final["macro_lag_cols"], "micro_raw_cols": sc_final["micro_raw_cols"]},
            runtime_profile=runtime_profile,
        )
        successful_scenarios.append(sc["scenario"])
        rows_8q.extend(r)
        preds_8q.extend(p)
        fitted_refs.update(f)
        tuning_rows.extend(t)

    ts_df = raw.dropna(subset=[DATE_COL, TARGET_COL]).sort_values(DATE_COL).reset_index(drop=True).copy()
    ts_df[GROUP_COL] = company_name
    ts_df = _ensure_quarter_fields(ts_df)
    scenario_data["TS"] = ts_df
    ts_rows_8q, ts_preds_8q, ts_fitted_refs, model_skip_reason = evaluate_time_series_models(
        ts_df,
        n_test=test_size,
        include_lstm=include_lstm,
        runtime_profile=runtime_profile,
    )
    for r in ts_rows_8q:
        r["split"] = f"last_{test_size}q"
    rows_8q.extend(ts_rows_8q)
    preds_8q.extend(ts_preds_8q)
    fitted_refs.update(ts_fitted_refs)

    for scenario_name in [sc["scenario"] for sc in scenario_defs if sc["scenario"] in successful_scenarios]:
        scenario_rows = [row for row in rows_8q if row.get("scenario") == scenario_name]
        scenario_predictions = [row for row in preds_8q if row.get("scenario") == scenario_name]
        ensemble_row, ensemble_predictions = _build_scenario_ensemble_rows(
            scenario_name,
            scenario_rows,
            scenario_predictions,
            ts_rows_8q,
            ts_preds_8q,
        )
        if ensemble_row and ensemble_predictions:
            rows_8q.append(ensemble_row)
            preds_8q.extend(ensemble_predictions)

    rows_16q, preds_16q = [], []
    robustness_enabled = enable_16q_robustness and min_rows >= 40
    if robustness_enabled:
        for sc in scenario_defs:
            df_sc = scenario_data[sc["scenario"]]
            sc_final = scenario_final[sc["scenario"]]
            if len(df_sc) < 17:
                continue
            r, p, _, _, _, _ = evaluate_holdout_models(
                df_sc,
                sc_final["numeric_features"],
                sc_final["log_features"],
                n_test=16,
                scenario_name=sc["scenario"],
                payload_context={"macro_lag_cols": sc_final["macro_lag_cols"], "micro_raw_cols": sc_final["micro_raw_cols"]},
                runtime_profile=runtime_profile,
            )
            for rr in r:
                rr["split"] = "last_16q"
            for pp in p:
                pp["split"] = "last_16q"
            rows_16q.extend(r)
            preds_16q.extend(p)
    for r in rows_8q:
        r["split"] = f"last_{test_size}q"

    metrics_8q = list(rows_8q)
    metrics_16q = list(rows_16q)
    if not metrics_8q:
        raise ValueError("No model metrics were produced.")
    # raw-target models for the companies in RAW_TARGET_COMPANIES, log-target models for all the others.
    # The time-series rows (TS) always take part in the ranking
    preferred_target = "raw" if _normalize_company_name(company_name) in RAW_TARGET_COMPANIES else "log"
    def _is_selected_metric(row):
        if row.get("scenario") == "TS":
            return True
        target_type = str(row.get("target_type", "")).lower()
        if preferred_target == "raw":
            return "raw revenue" in target_type
        return "log1p" in target_type

    selected_metrics_8q = [row for row in metrics_8q if _is_selected_metric(row)]
    ranking_pool = selected_metrics_8q if selected_metrics_8q else metrics_8q
    ranked = sorted(ranking_pool, key=_metric_sort_key)
    best = ranked[0]
    best_model_name = best["model"]
    best_ts_row = None
    ts_ranked = sorted(
        [row for row in metrics_8q if row.get("scenario") == "TS"],
        key=_metric_sort_key,
    )
    if ts_ranked:
        best_ts_row = ts_ranked[0]

    # Strict notebook-style deployment:
    # choose the top-ranked selected-target model/scenario directly,
    # then surface stability diagnostics separately instead of changing the winner.
    deployment_best = next(
        (row for row in ranked if (row.get("scenario"), row.get("model")) in fitted_refs),
        best,
    )
    deployment_selected_model = deployment_best.get("model")
    deployment_selected_scenario = deployment_best.get("scenario", "S1")
    if deployment_best is best:
        deployment_selection_reason = "Notebook-style selection: top-ranked selected-target model on last-8-quarter comparison."
    else:
        deployment_selection_reason = (
            "Notebook-style ranking kept for comparison, but deployment uses the highest-ranked refittable model "
            "because the top-ranked winner has no recursive forecast payload."
        )
    forecast_full = []
    deployment_future_dates = future_quarter_dates(last_date=model_df[DATE_COL].max())
    historical_values = model_df[TARGET_COL].values
    deployment_stability = {"passed": None, "reason": "No deployment forecast generated."}

    selected_payload = fitted_refs.get((deployment_selected_scenario, deployment_selected_model))
    if selected_payload is not None:
        forecast_full = recursive_forecast_company(
            scenario_data.get(deployment_selected_scenario, full_df),
            selected_payload,
            selected_payload.get("feature_cols", []),
            selected_payload.get("macro_lag_cols", []),
            selected_payload.get("micro_raw_cols", []),
            steps=len(deployment_future_dates),
            future_dates=deployment_future_dates,
        )
        ok, reason = _check_forecast_stability(
            forecast_full,
            historical_values,
            deployment_best.get("target_type", "raw revenue"),
        )
        deployment_stability = {"passed": bool(ok), "reason": reason}
        _debug_candidate(
            deployment_selected_model,
            deployment_selected_scenario,
            deployment_best,
            forecast_full,
            f"Notebook-style selection; stability check={'Accepted' if ok else reason}",
        )
    else:
        deployment_selection_reason = "Notebook-style winner could not be refit from stored payload."

    if forecast_full:
        direction = "flat"
        first_val = float(forecast_full[0]["predicted_revenue"])
        last_val = float(forecast_full[-1]["predicted_revenue"])
        if last_val > first_val * 1.01:
            direction = "up"
        elif last_val < first_val * 0.99:
            direction = "down"
    else:
        direction = "unavailable"

    # Extra forecast lines for the Forecast page: the best model of every scenario (and the best time-series model),
    # each one forecast until 2032
    scenario_future_forecasts = {}
    scenario_future_dates = future_quarter_dates(last_date=model_df[DATE_COL].max())

    scenario_comparison = []
    for sc in scenario_defs:
        sc_rows = [r for r in ranking_pool if r.get("scenario") == sc["scenario"]]
        if not sc_rows:
            continue
        best_sc = sorted(sc_rows, key=_metric_sort_key)[0]
        scenario_payload = fitted_refs.get((sc["scenario"], best_sc.get("model")))
        scenario_forecast_points = []
        if scenario_payload is not None:
            try:
                scenario_forecast_points = recursive_forecast_company(
                    scenario_data.get(sc["scenario"], full_df),
                    scenario_payload,
                    scenario_payload.get("feature_cols", []),
                    scenario_payload.get("macro_lag_cols", []),
                    scenario_payload.get("micro_raw_cols", []),
                    steps=len(scenario_future_dates),
                    future_dates=scenario_future_dates,
                )
            except Exception:
                scenario_forecast_points = []
        scenario_future_forecasts[sc["scenario"]] = {
            "model": best_sc.get("model"),
            "scenario": sc["scenario"],
            "forecast_points": scenario_forecast_points,
            "forecast_horizon_end": str(pd.Timestamp(scenario_forecast_points[-1]["date"]).date()) if scenario_forecast_points else None,
        }
        scenario_comparison.append(
            {
                "scenario": sc["scenario"],
                "scenario_name": sc["name"],
                "best_model": best_sc.get("model"),
                "model_family": best_sc.get("model_family"),
                "MAE": best_sc.get("MAE"),
                "RMSE": best_sc.get("RMSE"),
                "MAPE": best_sc.get("MAPE"),
                "sMAPE": best_sc.get("sMAPE"),
                "split": best_sc.get("split"),
            }
        )
    if best_ts_row is not None:
        ts_payload = fitted_refs.get(("TS", best_ts_row.get("model")))
        ts_forecast_points = []
        if ts_payload is not None:
            try:
                ts_forecast_points = recursive_forecast_company(
                    scenario_data["TS"],
                    ts_payload,
                    ts_payload.get("feature_cols", []),
                    ts_payload.get("macro_lag_cols", []),
                    ts_payload.get("micro_raw_cols", []),
                    steps=len(scenario_future_dates),
                    future_dates=scenario_future_dates,
                )
            except Exception:
                ts_forecast_points = []
        scenario_future_forecasts["TS"] = {
            "model": best_ts_row.get("model"),
            "scenario": "TS",
            "forecast_points": ts_forecast_points,
            "forecast_horizon_end": str(pd.Timestamp(ts_forecast_points[-1]["date"]).date()) if ts_forecast_points else None,
        }
    scenario_comparison = sorted(scenario_comparison, key=_metric_sort_key)
    top_models = [{"rank": i + 1, **r} for i, r in enumerate(ranked[:3])]
    models_attempted = sorted({f"{r.get('scenario')}::{r.get('model')}" for r in metrics_8q if r.get("scenario") and r.get("model")})
    successful_scenarios = sorted(set(successful_scenarios))
    skipped_scenarios = sorted(set(skipped_scenarios) - set(successful_scenarios))

    forecast_16q = forecast_full[:FORECAST_STEPS]
    forecast_horizon_end = str(pd.Timestamp(deployment_future_dates[-1]).date()) if forecast_full else None
    result = {
        "company": company_name,
        "runtime_profile": runtime_profile,
        "rows_original": int(len(raw)),
        "rows_modeling": int(len(model_df)),
        "feature_engineering": {
            "lag_logic": ["lag_1", "lag_2", "lag_4", "lag_8"],
            "rolling_logic": "roll_mean_4/8 and roll_std_4/8 from revenue.shift(1).rolling(window)",
            "growth_logic": "growth_1 = pct_change(1).shift(1); log_growth_1 = log diff shifted by 1",
            "macro_lag_logic": "macro_feature_lag_1q = macro_feature.shift(1) by company",
            "micro_logic": "for each micro feature: lag_1, lag_4, growth_qoq shift(1), margin_lag1",
            "active_macro_features": macro_lag_cols,
            "active_micro_features": micro_feature_cols,
        },
        "validation_split": {"primary": f"split_last_n_by_company(n_test={test_size})", "robustness_16q_enabled": robustness_enabled},
        "model_comparison_logic": "Rank by sMAPE, then MAPE, then RMSE, then MAE",
        "log_target_handling": "log1p target with Duan smearing on inverse transform",
        "selected_target_policy": "raw-target company list from notebook" if preferred_target == "raw" else "log-target company policy from notebook",
        "recursive_forecasting_logic": "Step-by-step: append future quarter row, rebuild features, predict, write predicted revenue back, repeat",
        "pipeline_environment": pipeline_dependency_status(model_skip_reason),
        "all_model_metrics": ranking_pool,
        "all_model_metrics_8q": metrics_8q,
        "all_model_metrics_16q": metrics_16q,
        "top_3_models_for_this_company": top_models,
        "best_model_for_this_company": best_model_name,
        "best_model": best_model_name,
        "best_time_series_model": best_ts_row.get("model") if best_ts_row else None,
        "top_models": top_models,
        "leaderboard": ranked,
        "scenario_comparison": scenario_comparison,
        "scenario_future_forecasts": scenario_future_forecasts,
        "scenario_skip_reason": scenario_skip_reason if scenario_skip_reason else {},
        "model_skip_reason": model_skip_reason if model_skip_reason else {},
        "scenario_coverage": {
            "attempted": attempted_scenarios,
            "successful": successful_scenarios,
            "skipped": skipped_scenarios,
        },
        "forecast_start_date": str(FORECAST_START_DATE_2026_Q1.date()),
        "forecast_end_date": str(FORECAST_END_DATE_2032.date()),
        "forecast_horizon_quarters": len(forecast_full),
        "models_attempted": models_attempted,
        "deployment_selected_model": deployment_selected_model,
        "deployment_selected_scenario": deployment_selected_scenario,
        "deployment_selection_reason": deployment_selection_reason,
        "deployment_stability_check": deployment_stability,
        "frontend_notice": (
            "No stable forecast until 2032 is available. Validation metrics are still shown, "
            "but long-horizon projection should not be used."
            if not forecast_full
            else None
        ),
        "validation_predictions": preds_8q,
        "robustness_validation_predictions_16q": preds_16q,
        "future_forecast_steps": len(forecast_full),
        "future_forecast_preview": forecast_full[:5],
        "future_forecast_full": forecast_full,
        "future_forecast_until_2032": {
            "forecast_start_date": str(FORECAST_START_DATE_2026_Q1.date()),
            "forecast_end_date": str(FORECAST_END_DATE_2032.date()),
            "horizon_quarters": len(forecast_full),
            "selected_for_deployment": {
                "model": deployment_selected_model,
                "scenario": deployment_selected_scenario,
            },
            "forecast_points": forecast_full if forecast_full else [],
            "direction": direction if forecast_full else "unavailable",
        },
        "future_forecast_16q": {
            "horizon_quarters": FORECAST_STEPS,
            "selected_for_deployment": {
                "model": deployment_selected_model,
                "scenario": deployment_selected_scenario,
            },
            "forecast_points": forecast_16q if forecast_16q else [],
            "direction": direction if forecast_full else "unavailable",
            "forecast_horizon_end": str(pd.Timestamp(forecast_16q[-1]["date"]).date()) if forecast_16q else None,
        },
        "future_forecast_horizon_end": forecast_horizon_end,
        "tuning_summary_rows": tuning_rows,
    }

    # Everything is also saved to data/results/<company>_*.json. main.py reuses these files as a cache
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = _clean_name(company_name, "uploaded_company")
    metrics_path = RESULTS_DIR / f"{safe}_full_metrics.json"
    predictions_path = RESULTS_DIR / f"{safe}_validation_predictions.json"
    forecast_path = RESULTS_DIR / f"{safe}_forecast_until_2032.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=4, default=str)
    with open(predictions_path, "w") as f:
        json.dump(result["validation_predictions"], f, indent=4, default=str)
    with open(forecast_path, "w") as f:
        json.dump(result["future_forecast_full"], f, indent=4, default=str)
    result["metrics_file"] = str(metrics_path)
    result["predictions_file"] = str(predictions_path)
    result["future_forecast_file"] = str(forecast_path)
    return result
