# Forecasting a company's financial indicators with time series and machine learning

Diploma project, Kazakh-British Technical University (School of Information Technology and Engineering, program 6B06101 Information Systems), 2026.

The project tries to forecast the **quarterly revenue** of 21 public companies. It has two parts that belong together:

1. **Research notebooks** (`notebooks/`): data cleaning, exploratory analysis, feature engineering, model training and comparison, forecasts until 2032.
2. **A web app** (`src/` + `backend/`): a React dashboard and a FastAPI backend that show the results of the notebooks, and can also train the models on a new company CSV that a user uploads.

## What was done

* **Data.** Quarterly data from 2008 to 2025 for 21 companies from seven industries (tech, finance, energy, aerospace, consumer goods, hospitality, travel). Company figures come from Refinitiv (LSEG) and Yahoo Finance, the macro series from FRED. Revenue is measured in billions of USD.
* **Macro variables.** CPI, 10-year Treasury yield (DGS10), Brent oil, NASDAQ Composite and S&P 500. They are used with a one-quarter lag, and for the future they come from separate macro forecasts (see the third notebook).
* **Features.** Revenue lags (1, 2, 4, 8 quarters), rolling mean and std, growth, sin/cos of the quarter. Everything computed from revenue is shifted, so a quarter never sees its own or later values. Optional micro indicators (operating cash flow, total assets, EBITDA, gross profit, income before extraordinary items).
* **Models.** Linear Regression, Random Forest, XGBoost, SVR, ARIMA, SARIMA, Prophet, LSTM, plus a stacked ARIMA + RF + XGBoost ensemble (weights are 1/sMAPE from out-of-fold predictions).
* **Target.** Companies with big, stable revenue (Apple, Microsoft, Amazon, GameStop, Intel, JPMorgan) use raw revenue. All others use log revenue with Duan smearing when converting back.
* **Evaluation.** Rolling-origin cross-validation for tuning, the last 8 quarters as holdout, a 16-quarter holdout as a robustness check. Metrics: sMAPE (main), MAPE, RMSE, MAE.
* **Four scenarios.** S1 revenue features only, S2 + micro indicators, S3 + macro forecasts, S4 micro + macro. Forecasts are recursive, 28 quarters from 2026 Q1 to 2032 Q4.

## What came out (short version, details are in the written report)

* Adding the macro forecasts (S3) gave the lowest error on average, and the improvement over S1 is statistically significant (Wilcoxon signed-rank test, p < 0.05). It helps most for macro-sensitive companies (banks, energy, big tech).
* Micro indicators only help a few companies, on the whole S2 is not significantly better than S1. S4 (everything at once) suffers from multicollinearity.
* There is no significant difference between the best ML model and the best time-series model per company, so neither family wins everywhere. The best model depends on the company.
* With a 16-quarter holdout the ranking of the scenarios stays the same, but errors are noticeably higher, which is expected for recursive multi-step forecasts.
* The forecasts up to 2032 are meant as indicative scenarios, not as real predictions (short history, deterministic macro path, only lightly tuned LSTM).

## Repository layout

```
notebooks/
  Diplomwork_before_linear_regression.ipynb              main notebook (version with per-scenario ensembles and a per-company summary)
  Diplomwork_before_linear_regression_16q_robustness.ipynb   variant with the original ensemble section + the 16-quarter robustness check
  macro_indicators_arima_forecasting.ipynb               forecasts of CPI / DGS10 / Brent / NASDAQ / S&P 500 used as features
src/                      React frontend (Vite, Tailwind, Recharts)
  pages/                  Home, EDA, Models, Forecast, My Data, Admin DB
  utils/dataHelpers.js    all dashboard data + the code that merges backend results into it
  diploma_dashboard_data.json, forecast_page_data.json   results of the notebooks, bundled with the frontend
backend/
  main.py                 FastAPI app: auth, uploads, SQLite, results API
  services/ml_pipeline.py the model pipeline that runs on uploaded CSVs
  ml_pipeline/run_pipeline.py   small wrapper around it
  data/macro_features.csv, data/results/   macro table and saved model results of the diploma companies
docs/NOTEBOOK_PARITY_VARIANT.md   notes about how close the backend is to the notebooks
```

## Running the web app

You need Node.js 18+ and Python 3.10+.

Backend (from the repository root):

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8012
```

`prophet` and `tensorflow` at the bottom of `requirements.txt` are optional and heavy. If they fail to install, delete those two lines, the backend then just skips Prophet and LSTM. If TensorFlow complains about NumPy, install `numpy<2` first.

Frontend (second terminal, also from the repository root):

```bash
npm install
npm run dev
```

Open the address that Vite prints (usually http://localhost:5173). In development Vite forwards `/api` to the backend on port 8012, see `vite.config.js`.

The dashboard opens with the bundled diploma results, so it also works with the backend switched off (you just can't log in or upload then).

### Accounts

On the first start the backend creates two demo accounts in a local SQLite file (`backend/data/app.db`, not committed):

| user | password | role |
| --- | --- | --- |
| `admin` | `admin123` | admin, sees all uploads and the database page |
| `user` | `user123` | normal user |

These are only defaults for local testing. Before running it anywhere public, set `DIPLOMA_ADMIN_PASSWORD` and `DIPLOMA_USER_PASSWORD` in the environment before the first start. You can also register your own account on the login screen.

### Uploading a company

Log in, use the upload button and pick a CSV with quarterly revenue (the same format as the LSEG exports used in the project). If the file is one of the diploma companies, its saved results are used and the answer is instant. For a new company the pipeline trains and tunes all models on the spot, which can take several minutes.

## Running the notebooks

The notebooks were written on the authors' laptops and read the company CSVs from a local folder. **The raw data is not in this repository**: the company files are exports from a licensed vendor (LSEG / Refinitiv), so they can't be redistributed. That means:

* the outputs of the last run (tables and plots) are saved inside the notebooks, you can read the whole analysis on GitHub without running anything;
* to re-run them you need your own copy of the data in a folder called `company data` (or change `COMPANY_DATA_DIR` in the first cell). CPI, DGS10, Brent and NASDAQ are public FRED series (fred.stlouisfed.org), the S&P 500 file is a vendor export.

Suggested order: `macro_indicators_arima_forecasting` first (it writes the macro forecasts), then the main notebook. Install `xgboost`, `statsmodels`, `scikit-learn`, `prophet` and `tensorflow` (the last two are only needed for Prophet and LSTM).

## Things worth knowing

* `backend/data/raw` and `backend/data/processed` (vendor exports) and the SQLite database are deliberately left out of the repo, see `.gitignore`.
* The auth part is kept simple: session tokens never expire and CORS only allows the local dev addresses. That is fine for a demo, but not for a public deployment.
* For uploaded companies the "QoQ growth" cards on the home page actually show year-over-year growth, and the PACF plot is only an approximation. The bundled diploma companies use the real values from the notebooks.

## Authors

Author: D. Moldash.

Supervisor: Ayagoz Imansakipova.

