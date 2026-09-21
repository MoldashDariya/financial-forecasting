import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, RevenueTooltip, SectionTitle } from "../components/SharedUI";
import { companyColor, MODEL_COLORS } from "../utils/constants";
import {
  COMPANIES,
  getCompanyHistory,
  getCompanyModelResults,
  getForecastScenarioEntries,
} from "../utils/dataHelpers";

// short names so the model fits in the scenario cards
const MODEL_SHORT = {
  ARIMA: "ARIMA",
  "ARIMA log-target": "ARIMA (log)",
  SARIMA: "SARIMA",
  "SARIMA log-target": "SARIMA (log)",
  Prophet: "Prophet",
  "Prophet log-target": "Prophet (log)",
  LSTM: "LSTM",
  "LSTM log-target": "LSTM (log)",
  SVR: "SVR",
  "SVR log-target": "SVR (log)",
  "Linear Regression": "Lin. Reg.",
  "Linear Regression log-target": "Lin. Reg. (log)",
  "Random Forest": "RF",
  "Random Forest log-target": "RF (log)",
  XGBoost: "XGBoost",
  "XGBoost log-target": "XGBoost (log)",
};

function quarterToIndex(label) {
  const value = String(label ?? "").trim();
  let match = /^Q(\d)\s+(\d{4})$/.exec(value);
  if (match) return Number(match[2]) * 4 + (Number(match[1]) - 1);

  match = /^(\d{4})-Q(\d)$/.exec(value);
  if (match) return Number(match[1]) * 4 + (Number(match[2]) - 1);

  return null;
}

function sortQuarterRows(rows) {
  return [...rows].sort((left, right) => (quarterToIndex(left.label ?? left.quarter) ?? 0) - (quarterToIndex(right.label ?? right.quarter) ?? 0));
}

// One line chart: full history (actual) + the holdout predictions (backtest) + the future forecast.
// Rows are merged by quarter label so a quarter that has both an actual and a prediction becomes a single point
function buildScenarioChartData(historyRows, scenario, company) {
  const history = (historyRows ?? []).map((row) => ({
    label: row.label,
    actual: row[company] ?? null,
    forecast: null,
  }));

  const backtestRows = (scenario?.backtest ?? []).map((row) => ({
    label: row.quarter,
    actual: row.actual ?? null,
    forecast: row.predicted ?? null,
  }));

  const futureRows = (scenario?.forecast ?? []).map((row) => ({
    label: row.quarter,
    actual: null,
    forecast: row.predicted ?? row.forecast ?? null,
  }));

  const rowsByLabel = new Map();
  [...history, ...backtestRows, ...futureRows].forEach((row) => {
    const existing = rowsByLabel.get(row.label) ?? { label: row.label, actual: null, forecast: null };
    rowsByLabel.set(row.label, {
      ...existing,
      actual: row.actual ?? existing.actual,
      forecast: row.forecast ?? existing.forecast,
    });
  });

  return sortQuarterRows([...rowsByLabel.values()]);
}

function NumberCell({ label, value }) {
  return (
    <div className="rounded-2xl border border-slate-100 bg-slate-50 p-4">
      <p className="text-xs text-slate-500 mb-1">{label}</p>
      <p className="text-lg font-semibold text-slate-800">{value}</p>
    </div>
  );
}

function TopModelsCard({ topModels }) {
  const tones = [
    "border-amber-200 bg-amber-50 text-amber-700",
    "border-slate-200 bg-slate-50 text-slate-600",
    "border-yellow-200 bg-yellow-50 text-yellow-700",
  ];

  return (
    <Card>
      <SectionTitle>Top 3 Models</SectionTitle>
      <div className="space-y-3">
        {topModels.slice(0, 3).map((item, index) => (
          <div key={`${item.model}-${item.scenario}-${index}`} className="rounded-2xl border border-slate-100 p-4">
            <div className="flex items-start gap-3">
              <span className={`min-w-12 rounded-full border px-3 py-1 text-sm font-semibold ${tones[index] ?? tones[1]}`}>
                #{index + 1}
              </span>
              <p className="text-sm leading-6 text-slate-700">
                <span className="font-semibold text-slate-900">{item.model}</span>
                {` | Scenario ${item.scenario} | sMAPE ${(item.sMAPE ?? 0).toFixed(2)}% | MAPE ${(item.MAPE ?? 0).toFixed(2)}% | ${item.model_family ?? "machine_learning"}`}
              </p>
            </div>
          </div>
        ))}
        {!topModels.length && <p className="text-sm text-slate-500">No top model rows returned by the backend.</p>}
      </div>
    </Card>
  );
}

function ScenarioComparisonCard({ rows }) {
  const featureSetMap = {
    S1: "Base",
    S2: "Base + Micro",
    S3: "Base + Macro",
    S4: "Full Features",
    TS: "Time-series baseline",
  };

  return (
    <Card>
      <SectionTitle>Scenario Comparison</SectionTitle>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-slate-500">
              <th className="pb-3 font-medium">Scenario</th>
              <th className="pb-3 font-medium">Feature set</th>
              <th className="pb-3 font-medium">Model</th>
              <th className="pb-3 font-medium">sMAPE</th>
              <th className="pb-3 font-medium">MAPE</th>
            </tr>
          </thead>
          <tbody>
            {["S1", "S2", "S3", "S4", "TS"].map((scenario) => {
              const row = rows.find((item) => item.scenario === scenario);
              if (!row && scenario === "TS") return null;
              return (
                <tr key={scenario} className="border-t border-slate-100 text-slate-700">
                  <td className="py-3 font-medium">{scenario}</td>
                  <td className="py-3">{featureSetMap[scenario]}</td>
                  <td className="py-3">{row?.best_model ?? "—"}</td>
                  <td className="py-3">{row?.sMAPE != null ? `${Number(row.sMAPE).toFixed(2)}%` : "—"}</td>
                  <td className="py-3">{row?.MAPE != null ? `${Number(row.MAPE).toFixed(2)}%` : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function StatusCard({ title, lines }) {
  return (
    <Card>
      <SectionTitle>{title}</SectionTitle>
      <div className="space-y-2">
        {lines.map((line) => (
          <p key={line} className="text-sm text-slate-600">
            {line}
          </p>
        ))}
      </div>
    </Card>
  );
}

// Shows, for the selected company: top 3 models, the best model of each scenario (S1-S4 + time series),
// and a chart with the forecast of the selected scenario
export default function ForecastPage() {
  const [selCo, setSelCo] = useState(COMPANIES[0] ?? "");
  const [selScenarioKey, setSelScenarioKey] = useState("scenario_1");

  useEffect(() => {
    if (!COMPANIES.includes(selCo)) {
      setSelCo(COMPANIES[0] ?? "");
    }
  }, [selCo]);

  const modelResults = useMemo(() => getCompanyModelResults(selCo) ?? {}, [selCo]);
  const historyRows = useMemo(() => getCompanyHistory(selCo), [selCo]);
  const scenarioEntries = useMemo(() => getForecastScenarioEntries(selCo), [selCo]);

  useEffect(() => {
    const selectedExists = scenarioEntries.some((entry) => entry.key === selScenarioKey && entry.data);
    if (!selectedExists) {
      const firstAvailable = scenarioEntries.find((entry) => entry.data);
      if (firstAvailable) setSelScenarioKey(firstAvailable.key);
    }
  }, [scenarioEntries, selScenarioKey]);

  const selectedScenario =
    scenarioEntries.find((entry) => entry.key === selScenarioKey && entry.data) ??
    scenarioEntries.find((entry) => entry.data) ??
    null;

  const scenarioChartData = useMemo(
    () => buildScenarioChartData(historyRows, selectedScenario?.data ?? null, selCo),
    [historyRows, selectedScenario, selCo]
  );

  const coverage = modelResults.scenario_coverage ?? {};
  const skipReason = modelResults.scenario_skip_reason ?? {};

  if (!COMPANIES.length) return null;

  return (
    <div className="space-y-6">
      <div className="flex gap-2 flex-wrap">
        {COMPANIES.map((company) => (
          <button
            key={company}
            type="button"
            onClick={() => setSelCo(company)}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all ${
              selCo === company
                ? "text-white shadow"
                : "bg-white text-slate-600 border border-slate-200 hover:border-slate-300"
            }`}
            style={selCo === company ? { background: companyColor(company) } : {}}
          >
            {company}
          </button>
        ))}
      </div>

      <Card>
        <SectionTitle>Model Comparison</SectionTitle>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          <TopModelsCard topModels={modelResults.top_models ?? []} />
          <ScenarioComparisonCard rows={modelResults.scenario_comparison ?? []} />
        </div>
        <div className="mt-4">
          <StatusCard
            title="Scenario Coverage"
            lines={[
              `Attempted: ${(coverage.attempted ?? []).join(", ") || "—"}`,
              `Successful: ${(coverage.successful ?? []).join(", ") || "—"}`,
              `Skipped: ${(coverage.skipped ?? []).join(", ") || "—"}`,
              ...Object.entries(skipReason).map(([key, value]) => `${key}: ${value}`),
            ]}
          />
        </div>
      </Card>

      <Card>
        <SectionTitle>Scenario View — {selCo}</SectionTitle>
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-5 gap-3 mt-4">
          {scenarioEntries.map(({ key, label, data }) => (
            <div
              key={key}
              onClick={() => data && setSelScenarioKey(key)}
              role="button"
              tabIndex={data ? 0 : -1}
              onKeyDown={(event) => {
                if (data && (event.key === "Enter" || event.key === " ")) {
                  event.preventDefault();
                  setSelScenarioKey(key);
                }
              }}
              className={`text-left p-4 rounded-xl border transition-all ${
                selScenarioKey === key
                  ? "border-indigo-300 bg-indigo-50"
                  : "border-slate-200 bg-slate-50 hover:bg-white"
              } ${data ? "" : "opacity-50 cursor-not-allowed"}`}
            >
              <p className="text-xs text-slate-500 mb-1">{label}</p>
              <p className="text-sm font-bold text-slate-800">
                {MODEL_SHORT[data?.model] ?? data?.model ?? "No scenario result"}
              </p>
              <p className="text-xs text-slate-500 mt-2">
                {data?.metrics?.rmse != null ? `RMSE ${data.metrics.rmse.toFixed(3)}` : "Awaiting forecast output"}
              </p>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mt-4 mb-4">
          <NumberCell label="Selected model" value={selectedScenario?.data?.model ?? "—"} />
          <NumberCell label="Model family" value={selectedScenario?.data?.model_family ?? "—"} />
          <NumberCell
            label="RMSE"
            value={selectedScenario?.data?.metrics?.rmse?.toFixed?.(4) ?? "—"}
          />
          <NumberCell
            label="MAPE"
            value={selectedScenario?.data?.metrics?.mape?.toFixed?.(2) ?? "—"}
          />
          <NumberCell
            label="Scenario horizon end"
            value={
              selectedScenario?.data?.forecast?.at(-1)?.quarter ??
              (selectedScenario?.data ? "Backtest only" : "Metrics only")
            }
          />
        </div>

        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={scenarioChartData} margin={{ top: 4, right: 20, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="label" tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} tickFormatter={(value) => `$${value}B`} />
            <Tooltip content={<RevenueTooltip />} />
            <Legend iconType="circle" iconSize={8} />
            <Line
              type="monotone"
              dataKey="actual"
              name="Actual"
              stroke="#94a3b8"
              strokeWidth={2}
              dot={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="forecast"
              name="Predicted / Forecast"
              stroke={MODEL_COLORS[selectedScenario?.data?.model] ?? "#6366f1"}
              strokeWidth={2.5}
              dot={{ r: 3 }}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
}
