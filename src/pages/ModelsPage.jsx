import { useEffect, useState } from "react";
import {
  BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from "recharts";
import { Card, SectionTitle } from "../components/SharedUI";
import { MODEL_COLORS } from "../utils/constants";
import { COMPANIES, modelPerf, coMetrics, companyScenarioMetrics } from "../utils/dataHelpers";

// Table with the average error of every model over all companies. Click a row to see that model company by company
export default function ModelsPage() {
  const [selModel, setSelModel] = useState(modelPerf[0]?.model ?? "");

  useEffect(() => {
    if (!modelPerf.some((row) => row.model === selModel)) {
      setSelModel(modelPerf[0]?.model ?? "");
    }
  }, [selModel]);

  // Metrics of one model for one company. If the company has no direct entry, take its best scenario row of that model
  const getMetricRow = (company, model) => {
    const direct = coMetrics[company]?.[model];
    if (direct) return direct;

    const scenarioRow = (companyScenarioMetrics[company] ?? [])
      .filter((row) => row.model === model)
      .sort(
        (left, right) =>
          (left.smape ?? Infinity) - (right.smape ?? Infinity) ||
          (left.mape ?? Infinity) - (right.mape ?? Infinity) ||
          (left.rmse ?? Infinity) - (right.rmse ?? Infinity) ||
          (left.mae ?? Infinity) - (right.mae ?? Infinity)
      )[0];

    return scenarioRow
      ? {
          rmse: scenarioRow.rmse ?? null,
          mae: scenarioRow.mae ?? null,
          mape: scenarioRow.mape ?? null,
          smape: scenarioRow.smape ?? null,
        }
      : null;
  };

  const barData = COMPANIES.map((co) => {
    const metrics = getMetricRow(co, selModel);
    return {
      company: co,
      RMSE: metrics?.rmse ?? null,
      MAE: metrics?.mae ?? null,
      MAPE: metrics?.mape ?? null,
      sMAPE: metrics?.smape ?? null,
    };
  });
  const hasCompanyMetricData = barData.some(
    (row) => row.RMSE != null || row.MAE != null || row.MAPE != null || row.sMAPE != null
  );

  return (
    <div className="space-y-6">
      <Card>
        <SectionTitle>Models</SectionTitle>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-100">
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Model</th>
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Category</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">MAE</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">RMSE</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">MAPE</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">sMAPE</th>
              </tr>
            </thead>
            <tbody>
              {modelPerf.map((m) => (
                <tr
                  key={m.model}
                  onClick={() => setSelModel(m.model)}
                  className={`border-b border-slate-50 cursor-pointer transition-colors ${
                    selModel === m.model ? "bg-indigo-50" : "hover:bg-slate-50"
                  }`}
                >
                  <td className="py-2.5 px-3">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-2 h-2 rounded-full"
                        style={{ background: MODEL_COLORS[m.model] ?? "#94a3b8" }}
                      />
                      <span className="font-medium text-slate-800">{m.model}</span>
                    </div>
                  </td>
                  <td className="py-2.5 px-3 text-slate-500">{m.cat}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-700">{m.mae?.toFixed(2)}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-700">{m.rmse?.toFixed(2)}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-700">{m.mape?.toFixed(2)}%</td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-700">{m.smape?.toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <SectionTitle>RMSE and MAE by Company — {selModel}</SectionTitle>
          {hasCompanyMetricData ? (
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={barData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="company" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 9 }} />
                <Tooltip />
                <Legend iconSize={8} />
                <Bar dataKey="RMSE" fill="#6366f1" radius={[4, 4, 0, 0]} />
                <Bar dataKey="MAE" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[320px] rounded-2xl border border-dashed border-slate-200 bg-slate-50 flex items-center justify-center p-6 text-center">
              <p className="max-w-md text-sm text-slate-500">
                Per-company RMSE and MAE are not available for <span className="font-semibold text-slate-700">{selModel}</span> in the current dataset. This row exists only as an aggregated model summary.
              </p>
            </div>
          )}
        </Card>

        <Card>
          <SectionTitle>MAPE and sMAPE by Company — {selModel}</SectionTitle>
          {hasCompanyMetricData ? (
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={barData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="company" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 9 }} />
                <Tooltip formatter={(v) => [`${Number(v).toFixed(2)}%`, ""]} />
                <Legend iconSize={8} />
                <Bar dataKey="MAPE" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="sMAPE" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[320px] rounded-2xl border border-dashed border-slate-200 bg-slate-50 flex items-center justify-center p-6 text-center">
              <p className="max-w-md text-sm text-slate-500">
                Per-company MAPE and sMAPE are not available for <span className="font-semibold text-slate-700">{selModel}</span> in the current dataset. This row exists only as an aggregated model summary.
              </p>
            </div>
          )}
        </Card>
      </div>

    </div>
  );
}
