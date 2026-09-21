Notebook-parity variant

This duplicate project is intended to move closer to the original notebook methodology than the main project.

What changed in this copy

- The backend time-series path in `backend/services/ml_pipeline.py` now includes:
  - `ARIMA`
  - `ARIMA log-target`
  - `SARIMA`
  - `Prophet` when installed
  - `LSTM` when installed and compatible
- The backend exposes `best_time_series_model` separately.
- Validation predictions are no longer duplicated.

What is still required for fuller notebook parity

- Install optional packages used by the notebook:
  - `prophet`
  - `tensorflow`
- Keep package versions stable across runs.
- If you want exact notebook outputs, port any remaining notebook-only experiments and selection rules that are not yet in the backend.

Suggested run command

`uvicorn backend.main:app --reload --port 8012`

Optional package install commands

Use the project virtual environment if you have one. From the project root:

`python3 -m pip install prophet`

`python3 -m pip install tensorflow`

If TensorFlow has NumPy compatibility issues on your machine, install a compatible NumPy first:

`python3 -m pip install "numpy<2"`

Then reinstall TensorFlow:

`python3 -m pip install --upgrade tensorflow`

Suggested next parity steps

1. Add Prophet log-target and LSTM log-target variants.
2. Mirror the notebook tuning grids exactly.
3. Rebuild bundled JSON from this backend once parity is acceptable.
