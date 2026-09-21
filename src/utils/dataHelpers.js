// All the dashboard data lives in this file, in module-level variables.
// The pages import the exported `let`s (hist, COMPANIES, modelPerf, ...). They are plain variables and not React state:
// syncExports() re-assigns them after every change and App.jsx bumps `dataVersion` so the pages redraw.
// The data comes from diploma_dashboard_data.json / forecast_page_data.json (results of the notebooks),
// or from the CSV files the user uploads (results come from the backend)
import bundledDashboardData from "../diploma_dashboard_data.json";
import bundledForecastPageData from "../forecast_page_data.json";

function createEmptyData() {
  return {
    historical: [],
    model_perf: [],
    forecast: {},
    forecast_by_model: {},
    company_metrics: {},
    qoq_stats: {},
    top_models_by_company: {},
    company_scenario_metrics: {},
    validation_predictions_by_company: {},
    eda: {},
    sections: {},
    meta: {
      companies: [],
      forecast_models: [],
      n_models: 0,
      best_rmse: null,
      best_mape: null,
      period: "—",
    },
    forecastPage: {
      companies: [],
      data: {},
    },
  };
}

// runtimeCompanies = one dataset per company, only filled once the user uploads something.
// runtimeData = the merged object all charts read
let runtimeCompanies = {};
let runtimeData = createEmptyData();

// scenario codes used by the backend / notebooks -> keys used in forecast_page_data.json
const SCENARIO_KEY_BY_CODE = {
  S1: "scenario_1",
  S2: "scenario_2",
  S3: "scenario_3",
  S4: "scenario_4",
  TS: "time_series",
};

export let hist = [];
export let recent = [];
export let modelPerf = [];
export let forecastMap = {};
export let forecastByModel = {};
export let coMetrics = {};
export let qoqStats = {};
export let qoqData = [];
export let topModelsByCompany = {};
export let companyScenarioMetrics = {};
export let edaByCompany = {};
export let COMPANY_SECTIONS = {};
export let COMPANIES = [];
export let META = createEmptyData().meta;
export let FORECAST_MODELS = [];

// copies runtimeData into the exported variables. Has to run after every change of runtimeData
function syncExports() {
  hist = runtimeData.historical;
  recent = hist.slice(-16);
  modelPerf = runtimeData.model_perf;
  forecastMap = runtimeData.forecast;
  forecastByModel = runtimeData.forecast_by_model;
  coMetrics = runtimeData.company_metrics;
  qoqStats = runtimeData.qoq_stats;
  topModelsByCompany = runtimeData.top_models_by_company;
  companyScenarioMetrics = runtimeData.company_scenario_metrics;
  edaByCompany = runtimeData.eda;
  COMPANY_SECTIONS = runtimeData.sections;
  COMPANIES = runtimeData.meta.companies ?? [];
  META = runtimeData.meta;
  FORECAST_MODELS = runtimeData.meta.forecast_models ?? [];
  qoqData = buildQoqData(COMPANIES, hist);
}

function cloneSerializable(value) {
  return JSON.parse(JSON.stringify(value));
}

function buildBundledDataState() {
  const bundled = cloneSerializable(bundledDashboardData);
  bundled.forecastPage = {
    companies: cloneSerializable(bundledForecastPageData.companies ?? []),
    data: cloneSerializable(bundledForecastPageData.data ?? {}),
    scenarioLabels: cloneSerializable(bundledForecastPageData.scenario_labels ?? {}),
  };
  return bundled;
}

// The bundled JSON has a different shape than the backend's answer. This rebuilds an object that looks like the backend's
// model_results (leaderboard, scenario comparison, forecasts), so the Forecast page needs only one code path
function buildBundledModelResults(company, state) {
  const baseMetrics = state.company_scenario_metrics?.[company] ?? [];
  const validationPredictions = state.validation_predictions_by_company?.[company] ?? [];
  if (!baseMetrics.length) return null;

  const topModels = state.top_models_by_company?.[company] ?? [];
  const bestOverall =
    state.best_model_per_company?.[company] ??
    topModels[0] ??
    baseMetrics[0] ??
    null;
  const companyForecastData = state.forecastPage?.data?.[company] ?? {};

  const scenarioComparison = ["S1", "S2", "S3", "S4", "TS"]
    .map((scenarioCode) => {
      const rows = baseMetrics
        .filter((row) => row.scenario_code === scenarioCode)
        .sort((left, right) => (left.rank ?? Infinity) - (right.rank ?? Infinity));
      const bestRow = rows[0];
      if (!bestRow) return null;
      return {
        scenario: scenarioCode,
        best_model: bestRow.model,
        sMAPE: bestRow.smape,
        MAPE: bestRow.mape,
      };
    })
    .filter(Boolean);

  const bestTimeSeriesRow =
    baseMetrics
      .filter((row) => row.scenario_code === "TS")
      .sort((left, right) => (left.rank ?? Infinity) - (right.rank ?? Infinity))[0] ?? null;

  const attemptedScenarios = ["S1", "S2", "S3", "S4", "TS"];
  const successfulScenarios = attemptedScenarios.filter((scenarioCode) =>
    scenarioComparison.some((row) => row.scenario === scenarioCode)
  );

  const deploymentScenarioKey = SCENARIO_KEY_BY_CODE[bestOverall?.scenario_code] ?? null;
  const deploymentScenarioData =
    (deploymentScenarioKey && companyForecastData[deploymentScenarioKey]) ||
    companyForecastData.best_time_series ||
    companyForecastData.time_series ||
    null;
  const deploymentForecastPoints = (deploymentScenarioData?.forecast ?? []).map((row) => ({
    date: row.quarter ?? null,
    year: parseQuarterLabel(row.quarter) != null ? Math.floor(parseQuarterLabel(row.quarter) / 4) : null,
    quarter: parseQuarterLabel(row.quarter) != null ? (parseQuarterLabel(row.quarter) % 4) + 1 : null,
    predicted_revenue: toNumber(row.forecast ?? row.predicted),
  }));

  const scenarioFutureForecasts = {
    S1: {
      model: companyForecastData.scenario_1?.model ?? null,
      forecast_points: (companyForecastData.scenario_1?.forecast ?? []).map((row) => ({
        date: row.quarter ?? null,
        year: parseQuarterLabel(row.quarter) != null ? Math.floor(parseQuarterLabel(row.quarter) / 4) : null,
        quarter: parseQuarterLabel(row.quarter) != null ? (parseQuarterLabel(row.quarter) % 4) + 1 : null,
        predicted_revenue: toNumber(row.forecast ?? row.predicted),
      })),
    },
    S2: {
      model: companyForecastData.scenario_2?.model ?? null,
      forecast_points: (companyForecastData.scenario_2?.forecast ?? []).map((row) => ({
        date: row.quarter ?? null,
        year: parseQuarterLabel(row.quarter) != null ? Math.floor(parseQuarterLabel(row.quarter) / 4) : null,
        quarter: parseQuarterLabel(row.quarter) != null ? (parseQuarterLabel(row.quarter) % 4) + 1 : null,
        predicted_revenue: toNumber(row.forecast ?? row.predicted),
      })),
    },
    S3: {
      model: companyForecastData.scenario_3?.model ?? null,
      forecast_points: (companyForecastData.scenario_3?.forecast ?? []).map((row) => ({
        date: row.quarter ?? null,
        year: parseQuarterLabel(row.quarter) != null ? Math.floor(parseQuarterLabel(row.quarter) / 4) : null,
        quarter: parseQuarterLabel(row.quarter) != null ? (parseQuarterLabel(row.quarter) % 4) + 1 : null,
        predicted_revenue: toNumber(row.forecast ?? row.predicted),
      })),
    },
    S4: {
      model: companyForecastData.scenario_4?.model ?? null,
      forecast_points: (companyForecastData.scenario_4?.forecast ?? []).map((row) => ({
        date: row.quarter ?? null,
        year: parseQuarterLabel(row.quarter) != null ? Math.floor(parseQuarterLabel(row.quarter) / 4) : null,
        quarter: parseQuarterLabel(row.quarter) != null ? (parseQuarterLabel(row.quarter) % 4) + 1 : null,
        predicted_revenue: toNumber(row.forecast ?? row.predicted),
      })),
    },
    TS: {
      model: companyForecastData.best_time_series?.model ?? companyForecastData.time_series?.model ?? null,
      forecast_points: ((companyForecastData.best_time_series?.forecast ?? companyForecastData.time_series?.forecast) ?? []).map((row) => ({
        date: row.quarter ?? null,
        year: parseQuarterLabel(row.quarter) != null ? Math.floor(parseQuarterLabel(row.quarter) / 4) : null,
        quarter: parseQuarterLabel(row.quarter) != null ? (parseQuarterLabel(row.quarter) % 4) + 1 : null,
        predicted_revenue: toNumber(row.forecast ?? row.predicted),
      })),
    },
  };

  return {
    all_model_metrics: baseMetrics,
    leaderboard: baseMetrics.map((row) => ({
      scenario: row.scenario_code,
      model: row.model,
      model_family: row.model_family,
      MAE: row.mae ?? row.MAE,
      RMSE: row.rmse ?? row.RMSE,
      MAPE: row.mape ?? row.MAPE,
      sMAPE: row.smape ?? row.sMAPE,
      rank: row.rank,
    })),
    top_models: topModels.map((row) => ({
      ...row,
      scenario: row.scenario_code ?? row.scenario,
      MAPE: row.mape,
      sMAPE: row.smape,
    })),
    scenario_comparison: scenarioComparison,
    scenario_future_forecasts: scenarioFutureForecasts,
    scenario_coverage: {
      attempted: attemptedScenarios,
      successful: successfulScenarios,
      skipped: attemptedScenarios.filter((scenarioCode) => !successfulScenarios.includes(scenarioCode)),
    },
    scenario_skip_reason: {},
    model_skip_reason: {},
    best_model: bestTimeSeriesRow?.model ?? null,
    best_time_series_model: bestTimeSeriesRow?.model ?? null,
    deployment_selected_model: bestOverall?.model ?? null,
    deployment_selected_scenario: bestOverall?.scenario_code ?? null,
    deployment_selection_reason:
      bestOverall?.scenario && bestOverall?.smape != null
        ? `${bestOverall.scenario} won with sMAPE ${Number(bestOverall.smape).toFixed(2)}%.`
        : "Selected from bundled diploma results.",
    future_forecast_until_2032: {
      forecast_points: deploymentForecastPoints,
    },
    future_forecast_horizon_end: deploymentForecastPoints.at(-1)?.date ?? null,
    validation_predictions: validationPredictions,
  };
}

function buildBundledCompanyDataset(company) {
  const bundledState = buildBundledDataState();
  const historyRows = (bundledState.historical ?? [])
    .filter((row) => row?.[company] != null)
    .map((row) => ({
      date: row.label,
      year: row.year,
      q: row.q,
      label: row.label,
      revenue: toNumber(row[company]),
    }))
    .filter((row) => row.revenue != null);

  return {
    company,
    historyRows: sortQuarterRows(historyRows),
    modelResults: buildBundledModelResults(company, bundledState),
    companyMetrics: bundledState.company_metrics?.[company] ?? {},
    allMetrics: bundledState.company_scenario_metrics?.[company] ?? [],
    topModels: bundledState.top_models_by_company?.[company] ?? [],
    deploymentForecast: bundledState.forecast?.[company] ?? [],
    deploymentModel: bundledState.best_model_per_company?.[company]?.model ?? null,
    forecastScenarios: bundledState.forecastPage?.data?.[company] ?? {},
    eda: bundledState.eda?.[company] ?? {},
  };
}

// fills every missing key with an empty value, so the pages don't crash on an incomplete payload
function normalizeDashboardState(payload) {
  const dashboard = cloneSerializable(payload?.dashboard ?? payload ?? {});
  const forecastPageSource =
    payload?.forecastPage ??
    dashboard.forecastPage ??
    {
      companies: bundledForecastPageData.companies ?? [],
      data: bundledForecastPageData.data ?? {},
      scenarioLabels:
        bundledForecastPageData.scenario_labels ??
        bundledForecastPageData.scenarioLabels ??
        {},
    };

  dashboard.historical ??= [];
  dashboard.model_perf ??= [];
  dashboard.forecast ??= {};
  dashboard.forecast_by_model ??= {};
  dashboard.company_metrics ??= {};
  dashboard.qoq_stats ??= {};
  dashboard.top_models_by_company ??= {};
  dashboard.company_scenario_metrics ??= {};
  dashboard.validation_predictions_by_company ??= {};
  dashboard.eda ??= {};
  dashboard.sections ??= {};
  dashboard.meta ??= createEmptyData().meta;
  dashboard.forecastPage = {
    companies: cloneSerializable(forecastPageSource.companies ?? []),
    data: cloneSerializable(forecastPageSource.data ?? {}),
    scenarioLabels: cloneSerializable(
      forecastPageSource.scenarioLabels ?? forecastPageSource.scenario_labels ?? {}
    ),
  };

  return dashboard;
}

function toNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

// 'apple_final.csv' -> 'Apple Final'
function normalizeCompanyName(value) {
  const raw = String(value ?? "").trim();
  if (!raw) return "Uploaded Company";
  return raw
    .replace(/\.(csv|xlsx|xls)$/i, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\w\S*/g, (word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase());
}

// date -> { year, q, label: 'Q3 2025' }. Uses UTC on purpose, so the time zone can't push a date into the previous quarter
function getQuarterFromDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  const month = date.getUTCMonth();
  const quarter = Math.floor(month / 3) + 1;
  return {
    year: date.getUTCFullYear(),
    q: quarter,
    label: `Q${quarter} ${date.getUTCFullYear()}`,
  };
}

function formatQuarterLabel(value, fallback = "—") {
  const quarter = getQuarterFromDate(value);
  return quarter?.label ?? fallback;
}

function sortQuarterRows(rows) {
  return [...rows].sort((a, b) => {
    const aIndex = a.year * 4 + (a.q - 1);
    const bIndex = b.year * 4 + (b.q - 1);
    return aIndex - bIndex;
  });
}

// same rule as in the backend: lower sMAPE wins, then MAPE, RMSE, MAE
function pickBetterMetricRow(current, candidate) {
  if (!current) return candidate;
  const score = (row) => [
    row?.sMAPE ?? Infinity,
    row?.MAPE ?? Infinity,
    row?.RMSE ?? Infinity,
    row?.MAE ?? Infinity,
  ];
  const left = score(current);
  const right = score(candidate);
  for (let index = 0; index < left.length; index += 1) {
    if (right[index] < left[index]) return candidate;
    if (right[index] > left[index]) return current;
  }
  return current;
}

function average(values) {
  const filtered = values.filter((value) => Number.isFinite(value));
  if (!filtered.length) return null;
  return filtered.reduce((sum, value) => sum + value, 0) / filtered.length;
}

// Plain sample autocorrelation up to lag 8.
// note: buildEdaPayload uses it for both ACF and PACF, so for uploaded data the PACF is only an approximation
// (the notebooks use statsmodels for the real one)
function buildAutocorrelation(series, fieldName) {
  const values = series.map((row) => row.revenue).filter((value) => Number.isFinite(value));
  if (values.length < 3) return [];

  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const denominator = values.reduce((sum, value) => sum + (value - mean) ** 2, 0);
  if (!denominator) return [];

  const maxLag = Math.min(8, values.length - 1);
  const rows = [];
  for (let lag = 1; lag <= maxLag; lag += 1) {
    let numerator = 0;
    for (let index = lag; index < values.length; index += 1) {
      numerator += (values[index] - mean) * (values[index - lag] - mean);
    }
    rows.push({
      lag,
      [fieldName]: numerator / denominator,
    });
  }
  return rows;
}

// EDA of an uploaded company, calculated in the browser. Simplified version:
// 'trend' is a 4-quarter moving average and the seasonal part is what is left over, so seasonal and residual are the same series here.
// ADF is not calculated (null). The proper STL / ADF results for the diploma companies are in the notebooks and in the bundled data
function buildEdaPayload(historyRows) {
  const sortedRows = sortQuarterRows(historyRows);
  const revenues = sortedRows.map((row) => row.revenue);

  const decomposition = sortedRows.map((row, index) => {
    const window = revenues.slice(Math.max(0, index - 3), index + 1);
    const trend = average(window);
    const seasonal = trend != null && row.revenue != null ? row.revenue - trend : null;
    return {
      label: row.label,
      original: row.revenue,
      trend,
      seasonal,
      residual: seasonal,
    };
  });

  const rolling4 = sortedRows.map((row, index) => {
    const window = revenues.slice(Math.max(0, index - 3), index + 1);
    const mean = average(window);
    const variance =
      mean == null || window.length < 2
        ? null
        : average(window.map((value) => (value - mean) ** 2));
    return {
      label: row.label,
      mean,
      std: variance == null ? null : Math.sqrt(variance),
    };
  });

  return {
    decomposition,
    rolling4,
    acf: buildAutocorrelation(sortedRows, "acf"),
    pacf: buildAutocorrelation(sortedRows, "pacf"),
    adf: null,
  };
}

// One scenario in the shape the Forecast page draws: metrics, backtest (actual vs predicted on the holdout) and the future forecast
function buildScenarioForecastPayload(modelResults, scenarioRow, metricRow) {
  if (!scenarioRow || !metricRow) return null;

  const backtest = (modelResults.validation_predictions ?? [])
    .filter(
      (row) =>
        row?.scenario === scenarioRow.scenario &&
        row?.model === metricRow.model
    )
    .map((row) => {
      const quarter = getQuarterFromDate(row.date);
      return {
        quarter: quarter?.label ?? row.date,
        actual: toNumber(row.actual),
        predicted: toNumber(row.predicted),
      };
    });

  const scenarioFuturePoints =
    modelResults.scenario_future_forecasts?.[scenarioRow.scenario]?.forecast_points ??
    [];
  const forecast = scenarioFuturePoints.map((row) => ({
    quarter:
      row.year != null && row.quarter != null
        ? `Q${row.quarter} ${row.year}`
        : formatQuarterLabel(row.date, row.date),
    predicted: toNumber(row.predicted_revenue),
  }));

  return {
    model: metricRow.model,
    model_family: metricRow.model_family ?? scenarioRow.model_family ?? null,
    target_type: metricRow.target_type ?? null,
    metrics: {
      mae: toNumber(metricRow.MAE),
      rmse: toNumber(metricRow.RMSE),
      mape: toNumber(metricRow.MAPE),
      smape: toNumber(metricRow.sMAPE),
    },
    backtest,
    forecast,
  };
}

// Turns the backend answer of one upload into a company dataset:
// history, metrics per model, forecast until 2032, the scenario views and the EDA
function buildCompanyDataset(payload) {
  const company = normalizeCompanyName(payload.company ?? payload.filename);
  const cleanRows = Array.isArray(payload.clean_rows) ? payload.clean_rows : [];

  const historyRows = cleanRows
    .map((row) => {
      const quarter = getQuarterFromDate(row.date);
      const revenue = toNumber(row.revenue);
      if (!quarter || revenue == null) return null;
      return {
        date: row.date,
        year: quarter.year,
        q: quarter.q,
        label: quarter.label,
        revenue,
      };
    })
    .filter(Boolean);

  const modelResults = payload.model_results ?? {};
  const allMetrics = Array.isArray(modelResults.all_model_metrics)
    ? modelResults.all_model_metrics
    : Array.isArray(modelResults.leaderboard)
      ? modelResults.leaderboard
      : [];

  const bestByModel = {};
  allMetrics.forEach((row) => {
    if (!row?.model) return;
    bestByModel[row.model] = pickBetterMetricRow(bestByModel[row.model], row);
  });

  const companyMetrics = Object.fromEntries(
    Object.entries(bestByModel).map(([model, row]) => [
      model,
      {
        rmse: toNumber(row.RMSE),
        mae: toNumber(row.MAE),
        mape: toNumber(row.MAPE),
        smape: toNumber(row.sMAPE),
      },
    ])
  );

  const deploymentForecastPoints =
    modelResults.future_forecast_until_2032?.forecast_points ??
    modelResults.future_forecast_full ??
    modelResults.future_forecast_16q?.forecast_points ??
    [];
  const deploymentForecast = deploymentForecastPoints.map((row) => ({
    quarter: row.year != null && row.quarter != null ? `Q${row.quarter} ${row.year}` : formatQuarterLabel(row.date, row.date),
    forecast: toNumber(row.predicted_revenue),
  }));

  const deploymentModel = modelResults.deployment_selected_model;
  const scenarioMetricsByCode = Object.fromEntries(
    (modelResults.scenario_comparison ?? []).map((row) => [row.scenario, row])
  );
  const metricLookup = Object.fromEntries(
    allMetrics.map((row) => [`${row.scenario}::${row.model}`, row])
  );

  const forecastScenarios = {
    scenario_1: buildScenarioForecastPayload(
      modelResults,
      scenarioMetricsByCode.S1,
      metricLookup[`S1::${scenarioMetricsByCode.S1?.best_model}`]
    ),
    scenario_2: buildScenarioForecastPayload(
      modelResults,
      scenarioMetricsByCode.S2,
      metricLookup[`S2::${scenarioMetricsByCode.S2?.best_model}`]
    ),
    scenario_3: buildScenarioForecastPayload(
      modelResults,
      scenarioMetricsByCode.S3,
      metricLookup[`S3::${scenarioMetricsByCode.S3?.best_model}`]
    ),
    scenario_4: buildScenarioForecastPayload(
      modelResults,
      scenarioMetricsByCode.S4,
      metricLookup[`S4::${scenarioMetricsByCode.S4?.best_model}`]
    ),
    time_series: buildScenarioForecastPayload(
      modelResults,
      {
        scenario: "TS",
        model_family: "time_series",
      },
      metricLookup[`TS::${modelResults.best_time_series_model}`] ??
        metricLookup[`TS::${modelResults.best_model}`] ??
        metricLookup[`TS::ARIMA`] ??
        null
    ),
  };

  return {
    company,
    historyRows: sortQuarterRows(historyRows),
    modelResults,
    companyMetrics,
    allMetrics,
    topModels: modelResults.top_models ?? [],
    deploymentForecast,
    deploymentModel,
    forecastScenarios,
    eda: buildEdaPayload(historyRows),
  };
}

// one row per quarter, one column per company (the shape recharts wants)
function buildHistoricalTable(companyDatasets) {
  const byQuarter = new Map();

  companyDatasets.forEach(({ company, historyRows }) => {
    historyRows.forEach((row) => {
      const existing = byQuarter.get(row.label) ?? {
        label: row.label,
        year: row.year,
        q: row.q,
      };
      existing[company] = row.revenue;
      byQuarter.set(row.label, existing);
    });
  });

  return sortQuarterRows([...byQuarter.values()]);
}

// note: despite the name this compares a quarter with the row 4 quarters back (slice(4) / index), i.e. it is year-over-year growth.
// The EDA page shows it as "Year-over-Year Growth", that part is fine. But buildQoqStats reuses it for the "QoQ" cards on the home page,
// so for uploaded companies those cards are YoY too (the bundled JSON has real quarter-over-quarter numbers)
function buildQoqData(companies, historicalRows) {
  if (historicalRows.length <= 4) return [];
  return historicalRows.slice(4).map((row, index) => {
    const previous = historicalRows[index];
    const result = { label: row.label };
    companies.forEach((company) => {
      const currentValue = row[company];
      const previousValue = previous[company];
      result[company] =
        currentValue != null && previousValue != null && previousValue !== 0
          ? +(((currentValue / previousValue) - 1) * 100).toFixed(1)
          : null;
    });
    return result;
  });
}

function buildQoqStats(companies, qoqData) {
  return Object.fromEntries(
    companies.map((company) => {
      const values = qoqData
        .map((row) => row[company])
        .filter((value) => Number.isFinite(value));
      return [
        company,
        {
          avg: average(values),
          max: values.length ? Math.max(...values) : null,
          min: values.length ? Math.min(...values) : null,
        },
      ];
    })
  );
}

// Average of every model's metrics over all companies (table on the Models page).
// Sorted by RMSE, the first row gets the `best` flag
function buildModelPerf(companyDatasets) {
  const byModel = new Map();

  companyDatasets.forEach(({ company, companyMetrics, allMetrics }) => {
    Object.entries(companyMetrics).forEach(([model, metrics]) => {
      const existing = byModel.get(model) ?? {
        model,
        cat: "Machine Learning",
        rmseValues: [],
        maeValues: [],
        mapeValues: [],
        smapeValues: [],
        companies: [],
      };

      const modelFamily =
        allMetrics.find((row) => row.model === model)?.model_family ?? null;
      existing.cat = modelFamily === "time_series" ? "Time Series" : "Machine Learning";
      existing.rmseValues.push(metrics.rmse);
      existing.maeValues.push(metrics.mae);
      existing.mapeValues.push(metrics.mape);
      existing.smapeValues.push(metrics.smape);
      existing.companies.push(company);
      byModel.set(model, existing);
    });
  });

  const rows = [...byModel.values()].map((row) => ({
    model: row.model,
    cat: row.cat,
    rmse: average(row.rmseValues),
    mae: average(row.maeValues),
    mape: average(row.mapeValues),
    smape: average(row.smapeValues),
    best: false,
  }));

  rows.sort(
    (left, right) =>
      (left.rmse ?? Infinity) - (right.rmse ?? Infinity) ||
      (left.mape ?? Infinity) - (right.mape ?? Infinity)
  );

  if (rows[0]) rows[0].best = true;
  return rows;
}

// Merges all company datasets into the one object the pages read (same keys as diploma_dashboard_data.json)
function buildDashboardData(companyDatasets) {
  const companies = companyDatasets.map((item) => item.company);
  const historical = buildHistoricalTable(companyDatasets);
  const qoqData = buildQoqData(companies, historical);
  const qoqStatsPayload = buildQoqStats(companies, qoqData);
  const modelPerfPayload = buildModelPerf(companyDatasets);

  const bestRmse = average(
    modelPerfPayload.map((row) => row.rmse).filter((value) => Number.isFinite(value))
  );
  const bestMape = average(
    modelPerfPayload.map((row) => row.mape).filter((value) => Number.isFinite(value))
  );

  const allModelNames = [...new Set(modelPerfPayload.map((row) => row.model))];
  const period =
    historical.length > 0
      ? `${historical[0].label} – ${historical[historical.length - 1].label}`
      : "—";

  return {
    historical,
    model_perf: modelPerfPayload,
    forecast: Object.fromEntries(
      companyDatasets.map(({ company, deploymentForecast }) => [company, deploymentForecast])
    ),
    forecast_by_model: Object.fromEntries(
      companyDatasets.map(({ company, deploymentModel, deploymentForecast }) => [
        company,
        deploymentModel ? { [deploymentModel]: deploymentForecast } : {},
      ])
    ),
    company_metrics: Object.fromEntries(
      companyDatasets.map(({ company, companyMetrics }) => [company, companyMetrics])
    ),
    qoq_stats: qoqStatsPayload,
    top_models_by_company: Object.fromEntries(
      companyDatasets.map(({ company, topModels }) => [company, topModels])
    ),
    company_scenario_metrics: Object.fromEntries(
      companyDatasets.map(({ company, allMetrics }) => [company, allMetrics])
    ),
    validation_predictions_by_company: Object.fromEntries(
      companyDatasets.map(({ company, modelResults }) => [
        company,
        Array.isArray(modelResults?.validation_predictions)
          ? modelResults.validation_predictions
          : [],
      ])
    ),
    eda: Object.fromEntries(companyDatasets.map(({ company, eda }) => [company, eda])),
    sections: companies.length ? { "Uploaded Companies": companies } : {},
    meta: {
      companies,
      forecast_models: allModelNames,
      n_models: modelPerfPayload.length,
      best_rmse: bestRmse != null ? bestRmse.toFixed(2) : "—",
      best_mape: bestMape != null ? bestMape.toFixed(2) : "—",
      period,
    },
    forecastPage: {
      companies,
      data: Object.fromEntries(
        companyDatasets.map(({ company, forecastScenarios }) => [company, forecastScenarios])
      ),
    },
  };
}

// First upload on top of the bundled data: the companies that are shown right now are converted to runtime datasets,
// so the new upload is added next to them and doesn't replace them
function hydrateRuntimeCompaniesFromCurrentData() {
  if (Object.keys(runtimeCompanies).length > 0 || !COMPANIES.length) return;

  const hydratedEntries = COMPANIES.map((company) => {
    const modelResults = getCompanyModelResults(company);
    const historyRows = getCompanyHistory(company).map((row) => ({
      date: row.date ?? row.label,
      year: row.year,
      q: row.q,
      label: row.label,
      revenue: toNumber(row[company]),
    })).filter((row) => row.revenue != null);

    return [
      company,
      {
        company,
        historyRows,
        modelResults,
        companyMetrics: coMetrics[company] ?? {},
        allMetrics: companyScenarioMetrics[company] ?? [],
        topModels: topModelsByCompany[company] ?? [],
        deploymentForecast: forecastMap[company] ?? [],
        deploymentModel: modelResults?.deployment_selected_model ?? null,
        forecastScenarios: runtimeData.forecastPage.data[company] ?? {},
        eda: edaByCompany[company] ?? {},
      },
    ];
  });

  runtimeCompanies = Object.fromEntries(hydratedEntries);
}

// Back to the bundled diploma results (or to an empty dashboard)
export function resetDashboardData({ useBundled = true } = {}) {
  runtimeCompanies = {};
  runtimeData = useBundled ? buildBundledDataState() : createEmptyData();
  syncExports();
}

// Use the dashboard JSON from GET /api/dashboard/default (same content as the bundled files)
export function applyDashboardPayload(payload) {
  runtimeCompanies = {};
  runtimeData = normalizeDashboardState(payload);
  syncExports();
  return runtimeData;
}

function appendRuntimeCompany(companyDataset) {
  delete runtimeCompanies[companyDataset.company];
  runtimeCompanies[companyDataset.company] = companyDataset;
}

function uniqueRuntimeCompanyName(name) {
  if (!runtimeCompanies[name]) return name;

  let index = 2;
  let nextName = `${name} ${index}`;
  while (runtimeCompanies[nextName]) {
    index += 1;
    nextName = `${name} ${index}`;
  }
  return nextName;
}

// Adds one upload to the dashboard. Two cases: the file was recognised as a diploma company (its bundled results are reused),
// or it is a new company and the backend sent fresh model results. A duplicate name gets a ' 2', ' 3', ... suffix
export function ingestBackendUpload(payload) {
  if (payload?.source === "bundled_dashboard_company" && payload?.bundled_company && !payload?.model_results) {
    hydrateRuntimeCompaniesFromCurrentData();
    const companyDataset = buildBundledCompanyDataset(payload.bundled_company);
    companyDataset.company = uniqueRuntimeCompanyName(normalizeCompanyName(payload.filename ?? payload.company));
    appendRuntimeCompany(companyDataset);
    runtimeData = buildDashboardData(Object.values(runtimeCompanies));
    syncExports();
    return companyDataset.company;
  }

  hydrateRuntimeCompaniesFromCurrentData();
  const companyDataset = buildCompanyDataset(payload);
  companyDataset.company = uniqueRuntimeCompanyName(companyDataset.company);
  appendRuntimeCompany(companyDataset);
  runtimeData = buildDashboardData(Object.values(runtimeCompanies));
  syncExports();
  return companyDataset.company;
}

export function hasDashboardData() {
  return COMPANIES.length > 0;
}

function normalizeForecastRows(rows) {
  return (rows ?? []).map((row) => ({
    quarter: row.quarter,
    forecast: row.forecast ?? row.predicted ?? null,
  }));
}

export function hasPerModelForecastSeries(company) {
  const companyRows = forecastByModel[company];
  if (!companyRows || typeof companyRows !== "object") return false;
  return Object.values(companyRows).some((value) => Array.isArray(value) && value.length > 0);
}

export function getCompanyHistory(company) {
  return hist.filter((row) => row[company] != null);
}

export function getLatestPoint(company) {
  const series = getCompanyHistory(company);
  return series[series.length - 1] ?? null;
}

export function getSameQuarterPrevYear(company, point) {
  if (!point) return null;
  return hist.find((row) => row.year === point.year - 1 && row.q === point.q) ?? null;
}

// 'Q3 2025' or '2025-Q3' -> one number (year * 4 + quarter - 1) that can be sorted and compared
export function parseQuarterLabel(label) {
  const normalized = String(label).trim();
  let match = /^Q(\d)\s+(\d{4})$/.exec(normalized);
  if (match) {
    const quarter = Number(match[1]);
    const year = Number(match[2]);
    return year * 4 + (quarter - 1);
  }

  match = /^(\d{4})-Q(\d)$/.exec(normalized);
  if (match) {
    const year = Number(match[1]);
    const quarter = Number(match[2]);
    return year * 4 + (quarter - 1);
  }

  return null;
}

export function getForecastRows(company) {
  return normalizeForecastRows(forecastMap[company]);
}

export function getForecastByModel(company, model) {
  const modelRows = normalizeForecastRows(forecastByModel[company]?.[model]);
  if (modelRows.length > 0) return modelRows;
  return normalizeForecastRows(forecastMap[company]);
}

export function getTopScenarioModels(company, limit = 3) {
  const rows = companyScenarioMetrics[company] ?? [];
  if (!rows.length) return [];

  const bestByModel = new Map();
  rows.forEach((row) => {
    const existing = bestByModel.get(row.model);
    bestByModel.set(row.model, pickBetterMetricRow(existing, row));
  });

  return [...bestByModel.values()]
    .sort((left, right) => (left.RMSE ?? Infinity) - (right.RMSE ?? Infinity))
    .slice(0, limit)
    .map((row) => row.model);
}

export function getEDA(company) {
  return edaByCompany[company] ?? {};
}

export function getForecastScenarioEntries(company) {
  const companyData = runtimeData.forecastPage.data[company] ?? {};
  return [
    { key: "scenario_1", label: "Scenario 1", data: companyData.scenario_1 ?? null },
    { key: "scenario_2", label: "Scenario 2", data: companyData.scenario_2 ?? null },
    { key: "scenario_3", label: "Scenario 3", data: companyData.scenario_3 ?? null },
    { key: "scenario_4", label: "Scenario 4", data: companyData.scenario_4 ?? null },
    {
      key: "time_series",
      label: "Time Series",
      data: companyData.best_time_series ?? companyData.time_series ?? null,
    },
  ];
}

export function getCompanyDataset(company) {
  return runtimeCompanies[company] ?? null;
}

// Model results of one company: from the upload if there is one, otherwise rebuilt from the bundled JSON.
// The rebuild below is almost a copy of buildBundledModelResults (they could be merged into one function)
export function getCompanyModelResults(company) {
  if (runtimeCompanies[company]?.modelResults) {
    return runtimeCompanies[company].modelResults;
  }

  const baseMetrics = companyScenarioMetrics[company] ?? [];
  const validationPredictions = runtimeData.validation_predictions_by_company?.[company] ?? [];
  if (!baseMetrics.length) return null;

  const topModels = topModelsByCompany[company] ?? [];
  const bestOverall =
    runtimeData.best_model_per_company?.[company] ??
    topModels[0] ??
    baseMetrics[0] ??
    null;
  const companyForecastData = runtimeData.forecastPage.data[company] ?? {};

  const scenarioComparison = ["S1", "S2", "S3", "S4", "TS"]
    .map((scenarioCode) => {
      const rows = baseMetrics
        .filter((row) => row.scenario_code === scenarioCode)
        .sort((left, right) => (left.rank ?? Infinity) - (right.rank ?? Infinity));
      const bestRow = rows[0];
      if (!bestRow) return null;
      return {
        scenario: scenarioCode,
        best_model: bestRow.model,
        sMAPE: bestRow.smape,
        MAPE: bestRow.mape,
      };
    })
    .filter(Boolean);

  const bestTimeSeriesRow =
    baseMetrics
      .filter((row) => row.scenario_code === "TS")
      .sort((left, right) => (left.rank ?? Infinity) - (right.rank ?? Infinity))[0] ?? null;

  const deploymentScenarioKey = SCENARIO_KEY_BY_CODE[bestOverall?.scenario_code] ?? null;
  const deploymentScenarioData =
    (deploymentScenarioKey && companyForecastData[deploymentScenarioKey]) ||
    companyForecastData.best_time_series ||
    companyForecastData.time_series ||
    null;
  const deploymentForecastPoints = (deploymentScenarioData?.forecast ?? []).map((row) => ({
    date: row.quarter ?? null,
    year: parseQuarterLabel(row.quarter) != null ? Math.floor(parseQuarterLabel(row.quarter) / 4) : null,
    quarter: parseQuarterLabel(row.quarter) != null ? (parseQuarterLabel(row.quarter) % 4) + 1 : null,
    predicted_revenue: toNumber(row.forecast ?? row.predicted),
  }));

  const attemptedScenarios = ["S1", "S2", "S3", "S4", "TS"];
  const successfulScenarios = attemptedScenarios.filter((scenarioCode) =>
    scenarioComparison.some((row) => row.scenario === scenarioCode)
  );

  return {
    all_model_metrics: baseMetrics,
    leaderboard: baseMetrics.map((row) => ({
      scenario: row.scenario_code,
      model: row.model,
      model_family: row.model_family,
      MAE: row.mae ?? row.MAE,
      RMSE: row.rmse ?? row.RMSE,
      MAPE: row.mape ?? row.MAPE,
      sMAPE: row.smape ?? row.sMAPE,
      rank: row.rank,
    })),
    top_models: topModels.map((row) => ({
      ...row,
      scenario: row.scenario_code ?? row.scenario,
      MAPE: row.mape,
      sMAPE: row.smape,
    })),
    scenario_comparison: scenarioComparison,
    scenario_coverage: {
      attempted: attemptedScenarios,
      successful: successfulScenarios,
      skipped: attemptedScenarios.filter((scenarioCode) => !successfulScenarios.includes(scenarioCode)),
    },
    scenario_skip_reason: {},
    model_skip_reason: {},
    best_model: bestTimeSeriesRow?.model ?? null,
    deployment_selected_model: bestOverall?.model ?? null,
    deployment_selected_scenario: bestOverall?.scenario_code ?? null,
    deployment_selection_reason:
      bestOverall?.scenario && bestOverall?.smape != null
        ? `${bestOverall.scenario} won with sMAPE ${Number(bestOverall.smape).toFixed(2)}%.`
        : "Selected from bundled diploma results.",
    future_forecast_until_2032: {
      selected_for_deployment: { model: bestOverall?.model ?? null },
      forecast_points: deploymentForecastPoints,
      forecast_end_date: deploymentForecastPoints.at(-1)?.date ?? null,
    },
    future_forecast_horizon_end: deploymentForecastPoints.at(-1)?.date ?? null,
    validation_predictions: validationPredictions,
    frontend_notice: "Showing bundled diploma analysis. Upload CSV files to replace it with runtime backend results.",
  };
}

// revenue is stored in billions of USD
export function fmt(value, digits = 1) {
  return value == null ? "—" : `$${Number(value).toFixed(digits)}B`;
}

export function pct(value, digits = 1) {
  return value == null ? "—" : `${Number(value).toFixed(digits)}%`;
}

// load the bundled data once, when this module is imported for the first time
resetDashboardData();
