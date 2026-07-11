import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Cell, Legend, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { predictionsApi } from "../api/client";

const PALETTE = ["#4fd1c5", "#e8a33d", "#e2574c", "#6ea8fe", "#c792ea", "#f78c6c", "#82aaff", "#ffcb6b"];

export default function AnalyticsPage() {
  const { t } = useTranslation();
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);

  useEffect(() => {
    predictionsApi.list(1, 100).then((res) => {
      setItems(res.items);
      setTotal(res.total);
    });
  }, []);

  const classDistribution = useMemo(() => {
    const counts = {};
    items.forEach((p) => { counts[p.predicted_label] = (counts[p.predicted_label] ?? 0) + 1; });
    return Object.entries(counts).map(([name, value]) => ({ name, value }));
  }, [items]);

  const confidenceOverTime = useMemo(
    () =>
      [...items]
        .reverse()
        .map((p, idx) => ({ index: idx + 1, confidence: Math.round(p.confidence_calibrated * 1000) / 10 })),
    [items]
  );

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="font-mono text-sm uppercase tracking-wide text-muted">{t("analytics.title")}</h1>
        <span className="font-mono text-xs text-muted">
          {t("analytics.totalPredictions")}: <span className="text-primary">{total}</span>
        </span>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border border-line bg-panel p-5">
          <p className="mb-3 font-mono text-xs uppercase tracking-wide text-muted">{t("analytics.classDistribution")}</p>
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={classDistribution} dataKey="value" nameKey="name" innerRadius={55} outerRadius={85} paddingAngle={2}>
                {classDistribution.map((entry, idx) => (
                  <Cell key={entry.name} fill={PALETTE[idx % PALETTE.length]} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ background: "var(--bp-bg-panel-raised)", border: "1px solid var(--bp-line)", borderRadius: 6, fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 11, color: "var(--bp-text-muted)" }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="rounded-lg border border-line bg-panel p-5">
          <p className="mb-3 font-mono text-xs uppercase tracking-wide text-muted">{t("analytics.confidenceOverTime")}</p>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={confidenceOverTime}>
              <XAxis dataKey="index" tick={{ fill: "var(--bp-text-muted)", fontSize: 10 }} axisLine={{ stroke: "var(--bp-line)" }} tickLine={false} />
              <YAxis unit="%" tick={{ fill: "var(--bp-text-muted)", fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: "var(--bp-bg-panel-raised)", border: "1px solid var(--bp-line)", borderRadius: 6, fontSize: 12 }} formatter={(v) => `${v}%`} />
              <Line type="monotone" dataKey="confidence" stroke="var(--bp-accent-signal)" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
