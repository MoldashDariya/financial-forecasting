import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from "recharts";
import { Card, SectionTitle, RevenueTooltip } from "../components/SharedUI";
import { companyColor } from "../utils/constants";
import {
  hist, recent, qoqStats, COMPANIES, COMPANY_SECTIONS, META,
  getLatestPoint, getSameQuarterPrevYear, fmt, pct,
} from "../utils/dataHelpers";

// Overview: headline numbers, one KPI card per company (latest revenue + growth vs the same quarter last year)
// and two revenue charts
export default function HomePage() {
  const companyListText =
    COMPANIES.length <= 1
      ? COMPANIES[0] ?? "uploaded company"
      : `${COMPANIES.slice(0, -1).join(", ")} and ${COMPANIES[COMPANIES.length - 1]}`;

  const kpis = COMPANIES.map((co) => {
    const latest = getLatestPoint(co);
    const prev = getSameQuarterPrevYear(co, latest);
    const cur = latest?.[co];
    const p = prev?.[co];
    const growth =
      cur != null && p != null && p !== 0
        ? +((cur / p - 1) * 100).toFixed(1)
        : null;
    return { co, cur, growth, label: latest?.label ?? "—" };
  });

  // Cards are grouped by the sections from the JSON (e.g. tech, banks, ...), companies without a section go to 'Other'
  const kpiByCompany = Object.fromEntries(kpis.map((item) => [item.co, item]));
  const sectionedKpis = Object.entries(COMPANY_SECTIONS)
    .map(([section, companies]) => ({
      section,
      items: (companies ?? [])
        .map((co) => kpiByCompany[co])
        .filter(Boolean),
    }))
    .filter((section) => section.items.length > 0);
  const covered = new Set(sectionedKpis.flatMap((s) => s.items.map((item) => item.co)));
  const otherItems = kpis.filter((item) => !covered.has(item.co));
  if (otherItems.length > 0) {
    sectionedKpis.push({ section: "Other", items: otherItems });
  }

  return (
    <div className="space-y-6">
      {/* Hero */}
      <div className="bg-gradient-to-br from-slate-800 to-slate-700 rounded-2xl p-7 text-white">
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-3xl font-extrabold mb-2">Financial Forecasting</h1>
            <p className="text-slate-300 text-sm max-w-md">
              Quarterly revenue analysis &amp; ML forecasting for {companyListText} ·{" "}
              {META?.period ?? "2008–2025"}
            </p>
          </div>
          <div className="bg-slate-700 rounded-xl px-5 py-3 text-center">
            <p className="text-2xl font-bold text-amber-400">{META?.n_models ?? 0}</p>
            <p className="text-slate-400 text-xs mt-1">Models Compared</p>
          </div>
        </div>

        <div className="mt-6 grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: "RMSE", val: META?.best_rmse != null && META.best_rmse !== "—" ? `$${META.best_rmse}B` : "—" },
            { label: "MAPE", val: META?.best_mape != null && META.best_mape !== "—" ? `${META.best_mape}%` : "—" },
            { label: "Period", val: META.period },
          ].map(({ label, val }) => (
            <div key={label} className="bg-slate-600 rounded-xl p-3 text-center">
              <p className="text-xs text-slate-400 mb-1">{label}</p>
              <p className="font-bold text-white text-sm">{val}</p>
            </div>
          ))}
        </div>
      </div>

      {/* KPI Cards by Section */}
      <div className="space-y-5">
        {sectionedKpis.map(({ section, items }) => (
          <div key={section} className="space-y-3">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
              {section}
            </h3>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {items.map(({ co, cur, growth, label }) => (
                <Card key={co}>
                  <div className="flex items-center gap-2 mb-3">
                    <div className="w-3 h-3 rounded-full" style={{ background: companyColor(co) }} />
                    <span className="font-semibold text-slate-700 text-sm">{co}</span>
                  </div>
                  <p className="text-2xl font-bold text-slate-800">{fmt(cur)}</p>
                  <p className="text-xs text-slate-500 mt-1">
                    Latest Actual Revenue · {label}
                  </p>
                  {growth != null && (
                    <span
                      className={`inline-block mt-2 text-xs font-semibold px-2 py-0.5 rounded-full ${
                        growth >= 0
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-red-100 text-red-700"
                      }`}
                    >
                      {growth >= 0 ? "▲" : "▼"} {Math.abs(growth)}% YoY
                    </span>
                  )}
                </Card>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Full Revenue Chart */}
      <Card>
        <SectionTitle>
          Quarterly Revenue — All Companies ({META?.period ?? "2008–2025"})
        </SectionTitle>
        <ResponsiveContainer width="100%" height={440}>
          <LineChart data={hist} margin={{ top: 4, right: 20, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="label" tick={{ fontSize: 10 }} interval={7} />
            <YAxis tick={{ fontSize: 10 }} tickFormatter={(v) => `$${v}B`} />
            <Tooltip content={<RevenueTooltip />} />
            <Legend iconType="circle" iconSize={8} />
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

      {/* QoQ Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <SectionTitle>QoQ Growth Statistics (%)</SectionTitle>
          <div className="space-y-3">
            {COMPANIES.map((co) => {
              const s = qoqStats[co];
              return (
                <div key={co} className="flex items-center gap-3">
                  <div
                    className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                    style={{ background: companyColor(co) }}
                  />
                  <span className="text-sm text-slate-600 w-24">{co}</span>
                  <div className="flex-1 grid grid-cols-3 gap-2 text-xs text-right">
                    <div>
                      <p className="text-slate-400">Avg</p>
                      <p className="font-semibold text-slate-700">{pct(s?.avg)}</p>
                    </div>
                    <div>
                      <p className="text-slate-400">Max</p>
                      <p className="font-semibold text-emerald-600">{pct(s?.max)}</p>
                    </div>
                    <div>
                      <p className="text-slate-400">Min</p>
                      <p className="font-semibold text-red-500">{pct(s?.min)}</p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>

        <Card>
          <SectionTitle>Recent 16 Quarters</SectionTitle>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart
              data={recent}
              margin={{ top: 4, right: 8, bottom: 4, left: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="label" tick={{ fontSize: 9 }} interval={3} />
              <YAxis tick={{ fontSize: 9 }} tickFormatter={(v) => `$${v}B`} />
              <Tooltip content={<RevenueTooltip />} />
              <Legend iconType="circle" iconSize={7} />
              {COMPANIES.map((co) => (
                <Bar
                  key={co}
                  dataKey={co}
                  fill={companyColor(co)}
                  radius={[3, 3, 0, 0]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}
