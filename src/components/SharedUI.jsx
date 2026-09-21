// Small building blocks used by all pages
export const Card = ({ children, className = "" }) => (
  <div
    className={`bg-white rounded-2xl shadow-sm border border-slate-100 p-5 ${className}`}
  >
    {children}
  </div>
);

export const SectionTitle = ({ children }) => (
  <h2 className="text-xl font-bold text-slate-800 mb-4">{children}</h2>
);

// Tooltip for the revenue charts. Values are in billions of USD
export const RevenueTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-slate-200 rounded-xl shadow-lg p-3 text-sm">
      <p className="font-semibold text-slate-700 mb-2">{label}</p>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center gap-2 mb-1">
          <span
            className="w-2 h-2 rounded-full inline-block"
            style={{ background: p.color }}
          />
          <span className="text-slate-600">{p.dataKey}:</span>
          <span className="font-medium text-slate-800">
            {p.value == null ? "—" : `$${p.value.toFixed(1)}B`}
          </span>
        </div>
      ))}
    </div>
  );
};
