// Builders for the chart data. Each function returns an array of points like { label, actual, forecast }
// that recharts can draw directly
import {
  hist,
  COMPANIES,
  getCompanyHistory,
  getForecastRows,
  getForecastByModel,
  getEDA,
  parseQuarterLabel,
} from "./dataHelpers";

/** Single forecast series (no per-model split in JSON) for model comparison chart. */
export function buildSharedForecastComparisonData(company) {
  const histSlice = getCompanyHistory(company).slice(-12).map((r) => ({
    label: r.label,
    actual: r[company],
  }));

  const fRows = getForecastRows(company).map((f) => ({
    label: f.quarter,
    actual: null,
    forecast: f.forecast,
  }));

  return [...histSlice, ...fRows];
}

export function buildForecastChartData(company) {
  const histSlice = getCompanyHistory(company).slice(-8).map((r) => ({
    label: r.label,
    actual: r[company],
    forecast: null,
  }));

  const fRows = getForecastRows(company).map((f) => ({
    label: f.quarter,
    actual: null,
    forecast: f.forecast,
  }));

  return [...histSlice, ...fRows];
}

// history of the last 8 quarters + the forecast of every company in one table
export function buildForecastEnsemble() {
  const histSlice = hist.slice(-8).map((r) => ({
    label: r.label,
    ...Object.fromEntries(COMPANIES.map((co) => [co, r[co]])),
    type: "historical",
  }));

  const futureLabelSet = new Set();
  COMPANIES.forEach((co) => {
    getForecastRows(co).forEach((f) => futureLabelSet.add(f.quarter));
  });

  const futureLabels = [...futureLabelSet].sort(
    (a, b) => (parseQuarterLabel(a) ?? 0) - (parseQuarterLabel(b) ?? 0)
  );

  const fRows = futureLabels.map((label) => {
    const row = { type: "forecast", label };
    COMPANIES.forEach((co) => {
      const f = getForecastRows(co).find((x) => x.quarter === label);
      row[co] = f?.forecast ?? null;
    });
    return row;
  });

  return [...histSlice, ...fRows];
}

// Build chart data for a specific model + company (historical + model forecast)
export function buildModelForecastChartData(company, model) {
  const histSlice = getCompanyHistory(company).slice(-12).map((r) => ({
    label: r.label,
    actual: r[company],
    forecast: null,
  }));

  const fRows = getForecastByModel(company, model).map((f) => ({
    label: f.quarter,
    actual: null,
    forecast: f.forecast,
  }));

  return [...histSlice, ...fRows];
}

// Build multi-model comparison chart for a company (all 5 models vs actual)
export function buildMultiModelChartData(company, models) {
  const histSlice = getCompanyHistory(company).slice(-12).map((r) => ({
    label: r.label,
    actual: r[company],
  }));

  // Collect all forecast labels
  const allForecastRows = {};
  models.forEach((mdl) => {
    getForecastByModel(company, mdl).forEach((f) => {
      if (!allForecastRows[f.quarter]) allForecastRows[f.quarter] = {};
      allForecastRows[f.quarter][mdl] = f.forecast;
    });
  });

  const sortedLabels = Object.keys(allForecastRows).sort(
    (a, b) => (parseQuarterLabel(a) ?? 0) - (parseQuarterLabel(b) ?? 0)
  );

  const fRows = sortedLabels.map((label) => ({
    label,
    actual: null,
    ...allForecastRows[label],
  }));

  return [...histSlice, ...fRows];
}

// The functions below read the EDA object of a company. They accept both key styles
// (the frontend names and the names used in the notebook export), that's why every field has an `??` fallback
export function decompose(company) {
  const rows = getEDA(company).decomposition ?? [];
  return rows.map((row) => ({
    label: row.label ?? row.quarter ?? null,
    original: row.original ?? row.original_revenue ?? null,
    trend: row.trend ?? null,
    seasonal: row.seasonal ?? null,
    residual: row.residual ?? row.residuals ?? null,
  }));
}

export function rollingStats(company) {
  const rows = getEDA(company).rolling4 ?? getEDA(company).rolling_mean_std_4q ?? [];
  return rows.map((row) => ({
    label: row.label ?? row.quarter ?? null,
    mean: row.mean ?? row.rolling_mean ?? null,
    std: row.std ?? row.rolling_std ?? null,
  }));
}

export function getACF(company) {
  const acf = getEDA(company).acf ?? [];
  if (Array.isArray(acf)) return acf;
  if (acf?.original_revenue) {
    return acf.original_revenue.map((row) => ({
      lag: row.lag,
      acf: row.acf ?? row.value ?? null,
    }));
  }
  return [];
}

export function getPACF(company) {
  const pacf = getEDA(company).pacf ?? [];
  if (Array.isArray(pacf)) return pacf;
  if (pacf?.original_revenue) {
    return pacf.original_revenue.map((row) => ({
      lag: row.lag,
      pacf: row.pacf ?? row.value ?? null,
    }));
  }
  return [];
}
