import { useEffect, useMemo, useState } from "react";
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, ComposedChart,
} from "recharts";
import { Card, SectionTitle } from "../components/SharedUI";
import { companyColor } from "../utils/constants";
import {
  COMPANIES, qoqData, getEDA,
} from "../utils/dataHelpers";
import { decompose, rollingStats, getACF, getPACF } from "../utils/chartHelpers";

// Exploratory analysis of one company: stationarity test (ADF), decomposition, rolling mean/std, ACF / PACF and YoY growth.
// For the diploma companies the numbers come from the notebooks, for uploads a simplified version is calculated in dataHelpers.js
export default function EDAPage() {
  const [selCo, setSelCo] = useState(COMPANIES[0] ?? "");

  useEffect(() => {
    if (!COMPANIES.includes(selCo)) {
      setSelCo(COMPANIES[0] ?? "");
    }
  }, [selCo]);

  const decompData = useMemo(() => decompose(selCo), [selCo]);
  const rollData = useMemo(() => rollingStats(selCo), [selCo]);
  const acfData = useMemo(() => getACF(selCo), [selCo]);
  const pacfData = useMemo(() => getPACF(selCo), [selCo]);
  const adfRow = getEDA(selCo).adf;

  return (
    <div className="space-y-6">
      {/* Company Selector */}
      <div className="flex gap-2 flex-wrap">
        {COMPANIES.map((co) => (
          <button
            key={co}
            onClick={() => setSelCo(co)}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all ${
              selCo === co
                ? "text-white shadow"
                : "bg-white text-slate-600 border border-slate-200"
            }`}
            style={selCo === co ? { background: companyColor(co) } : {}}
          >
            {co}
          </button>
        ))}
      </div>

      {/* ADF Stationarity */}
      <Card>
        <SectionTitle>Stationarity Test (ADF) — {selCo}</SectionTitle>
        {adfRow && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              {
                label: "Before ADF Stat",
                val: adfRow.beforeADF != null ? adfRow.beforeADF.toFixed(3) : "—",
              },
              {
                label: "Before p-value",
                val:
                  adfRow.beforeP == null
                    ? "—"
                    : adfRow.beforeP < 0.0001
                      ? "<0.0001"
                      : adfRow.beforeP.toFixed(4),
                badge:
                  adfRow.beforeP == null
                    ? undefined
                    : adfRow.beforeP < 0.05
                      ? "Stationary"
                      : "Non-Stationary",
                ok: adfRow.beforeP != null && adfRow.beforeP < 0.05,
              },
              {
                label: "After ADF Stat",
                val: adfRow.afterADF != null ? adfRow.afterADF.toFixed(3) : "—",
              },
              {
                label: "After p-value",
                val:
                  adfRow.afterP == null
                    ? "—"
                    : adfRow.afterP < 0.0001
                      ? "<0.0001"
                      : adfRow.afterP.toFixed(4),
                badge:
                  adfRow.afterP == null
                    ? undefined
                    : adfRow.afterP < 0.05
                      ? "Stationary"
                      : "Non-Stationary",
                ok: adfRow.afterP != null && adfRow.afterP < 0.05,
              },
            ].map(({ label, val, badge, ok }) => (
              <div key={label} className="bg-slate-50 rounded-xl p-4">
                <p className="text-xs text-slate-500 mb-1">{label}</p>
                <p className="font-bold text-slate-800 text-lg">{val}</p>
                {badge && (
                  <span
                    className={`mt-2 inline-block text-xs font-semibold px-2 py-0.5 rounded-full ${
                      ok
                        ? "bg-emerald-100 text-emerald-700"
                        : "bg-red-100 text-red-700"
                    }`}
                  >
                    {badge}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Decomposition */}
      <Card>
        <SectionTitle>Time Series Decomposition — {selCo}</SectionTitle>
        <div className="space-y-4">
          {[
            { key: "original", label: "Original", color: companyColor(selCo) },
            { key: "trend", label: "Trend", color: "#6366f1" },
            { key: "seasonal", label: "Seasonal", color: "#f59e0b" },
            { key: "residual", label: "Residual", color: "#64748b" },
          ].map(({ key, label, color }) => (
            <div key={key}>
              <p className="text-xs font-semibold text-slate-500 mb-1">{label}</p>
              <ResponsiveContainer width="100%" height={112}>
                <LineChart
                  data={decompData}
                  margin={{ top: 4, right: 8, bottom: 0, left: 0 }}
                >
                  <XAxis dataKey="label" hide />
                  <YAxis tick={{ fontSize: 8 }} width={38} tickFormatter={(v) => `${v}`} />
                  <Tooltip
                    formatter={(value) =>
                      value != null
                        ? [`$${Number(value).toFixed(2)}B`, label]
                        : ["—", label]
                    }
                    labelFormatter={(period) => period}
                    contentStyle={{
                      borderRadius: 12,
                      border: "1px solid #e2e8f0",
                      fontSize: 12,
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey={key}
                    name={label}
                    stroke={color}
                    dot={false}
                    strokeWidth={1.5}
                    connectNulls
                  />
                  <ReferenceLine y={0} stroke="#e2e8f0" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
      </Card>

      {/* Rolling Stats */}
      <Card>
        <SectionTitle>
          Rolling Mean &amp; Std Dev (4-Quarter Window) — {selCo}
        </SectionTitle>
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart
            data={rollData}
            margin={{ top: 4, right: 20, bottom: 4, left: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="label" tick={{ fontSize: 9 }} interval={7} />
            <YAxis tick={{ fontSize: 9 }} tickFormatter={(v) => `$${v}B`} />
            <Tooltip
              formatter={(v, n) =>
                v != null ? [`$${Number(v).toFixed(2)}B`, n] : ["—", n]
              }
              labelFormatter={(period) => period}
              contentStyle={{
                borderRadius: 12,
                border: "1px solid #e2e8f0",
                fontSize: 12,
              }}
            />
            <Legend iconSize={8} />
            <Line
              type="monotone"
              dataKey="mean"
              name="Rolling Mean"
              stroke={companyColor(selCo)}
              dot={false}
              strokeWidth={2}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="std"
              name="Rolling Std"
              stroke="#94a3b8"
              dot={false}
              strokeWidth={1.5}
              strokeDasharray="4 2"
              connectNulls
            />
          </ComposedChart>
        </ResponsiveContainer>
      </Card>

      {/* ACF / PACF */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <SectionTitle>ACF</SectionTitle>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart
              data={acfData}
              margin={{ top: 4, right: 8, bottom: 4, left: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis
                dataKey="lag"
                tick={{ fontSize: 10 }}
                label={{ value: "Lag", position: "insideBottom", offset: -2, fontSize: 10 }}
              />
              <YAxis tick={{ fontSize: 10 }} domain={[-1, 1]} />
              <Tooltip
                formatter={(value, name) => [
                  value != null ? Number(value).toFixed(3) : "—",
                  name,
                ]}
                labelFormatter={(lag) => `Lag ${lag}`}
                contentStyle={{
                  borderRadius: 12,
                  border: "1px solid #e2e8f0",
                  fontSize: 12,
                }}
              />
              <ReferenceLine y={0.2} stroke="#ef4444" strokeDasharray="3 3" />
              <ReferenceLine y={-0.2} stroke="#ef4444" strokeDasharray="3 3" />
              <Bar dataKey="acf" name="ACF" fill="#6366f1" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <SectionTitle>PACF</SectionTitle>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart
              data={pacfData}
              margin={{ top: 4, right: 8, bottom: 4, left: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis
                dataKey="lag"
                tick={{ fontSize: 10 }}
                label={{ value: "Lag", position: "insideBottom", offset: -2, fontSize: 10 }}
              />
              <YAxis tick={{ fontSize: 10 }} domain={[-1, 1]} />
              <Tooltip
                formatter={(value, name) => [
                  value != null ? Number(value).toFixed(3) : "—",
                  name,
                ]}
                labelFormatter={(lag) => `Lag ${lag}`}
                contentStyle={{
                  borderRadius: 12,
                  border: "1px solid #e2e8f0",
                  fontSize: 12,
                }}
              />
              <ReferenceLine y={0.2} stroke="#ef4444" strokeDasharray="3 3" />
              <ReferenceLine y={-0.2} stroke="#ef4444" strokeDasharray="3 3" />
              <Bar dataKey="pacf" name="PACF" fill="#f59e0b" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* YoY Growth */}
      <Card>
        <SectionTitle>Year-over-Year Growth (%)</SectionTitle>
        <ResponsiveContainer width="100%" height={360}>
          <LineChart
            data={qoqData}
            margin={{ top: 4, right: 20, bottom: 4, left: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="label" tick={{ fontSize: 9 }} interval={7} />
            <YAxis tick={{ fontSize: 9 }} tickFormatter={(v) => `${v}%`} />
            <Tooltip
              formatter={(v, n) =>
                v != null ? [`${Number(v).toFixed(1)}%`, n] : ["—", n]
              }
              labelFormatter={(period) => period}
              contentStyle={{
                borderRadius: 12,
                border: "1px solid #e2e8f0",
                fontSize: 12,
              }}
            />
            <Legend iconType="circle" iconSize={8} />
            <ReferenceLine y={0} stroke="#94a3b8" />
            {COMPANIES.map((co) => (
              <Line
                key={co}
                type="monotone"
                dataKey={co}
                stroke={companyColor(co)}
                dot={false}
                strokeWidth={2}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
}
