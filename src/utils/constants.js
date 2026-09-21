// Brand colors of the companies (used for lines, bars and the company buttons)
export const C = {
  Amazon: "#FF9900",
  Apple: "#111827",
  Intel: "#0071C5",
  JPMorgan: "#005EB8",
  Microsoft: "#00B4D8",
  NVIDIA: "#76B900",
  Zoom: "#2D8CFF",
  AMD: "#ED1C24",
  "Bank of America": "#012169",
  BlackBerry: "#000000",
  Boeing: "#0033A1",
  Chevron: "#003A8F",
  Exxon: "#F01716",
  GameStop: "#E60012",
  Gap: "#000066",
  Hilton: "#003A8F",
  Marriott: "#B41F2E",
  Peloton: "#000000",
  "Stitch Fix": "#4B0082",
  Tripadvisor: "#34E0A1",
  Visa: "#1A1F71",
};

const COMPANY_FALLBACKS = [
  "#0ea5e9",
  "#f59e0b",
  "#22c55e",
  "#6366f1",
  "#ef4444",
  "#14b8a6",
  "#8b5cf6",
  "#eab308",
  "#10b981",
  "#f97316",
];

// Uploaded companies have no brand color: hash the name into one of the fallback colors,
// so the same company always gets the same color
export function companyColor(company) {
  if (C[company]) return C[company];
  const key = String(company ?? "");
  let hash = 0;
  for (let i = 0; i < key.length; i += 1) {
    hash = (hash * 31 + key.charCodeAt(i)) >>> 0;
  }
  return COMPANY_FALLBACKS[hash % COMPANY_FALLBACKS.length];
}

// one color per model, keys have to match the model names that the backend and the notebooks use
export const MODEL_COLORS = {
  LSTM: "#6366f1",
  ARIMA: "#f59e0b",
  XGBoost: "#10b981",
  "XGBoost log-target": "#10b981",
  "Random Forest": "#3b82f6",
  "Random Forest log-target": "#2563eb",
  "Random Forest (tuned pooled)": "#3b82f6",
  "Random Forest (separate plain)": "#2563eb",
  "Random Forest (separate log)": "#1d4ed8",
  "Linear Regression": "#64748b",
  "Linear Regression log-target": "#475569",
  Prophet: "#ec4899",
  SVR: "#8b5cf6",
  "SVR log-target": "#7c3aed",
  "Seasonal Naive": "#94a3b8",
  "Seasonal Naive (lag_4)": "#94a3b8",
};

export const FORECAST_TAB_MODELS = [
  "LSTM",
  "ARIMA",
  "XGBoost",
  "Random Forest (tuned pooled)",
  "Linear Regression",
];
