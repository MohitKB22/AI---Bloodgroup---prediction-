import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useTranslation } from "react-i18next";

export default function ProbabilityChart({ rawProbabilities, calibratedProbabilities }) {
  const { t } = useTranslation();

  const data = Object.keys(calibratedProbabilities)
    .map((label) => ({
      label,
      calibrated: Math.round(calibratedProbabilities[label] * 1000) / 10,
      raw: Math.round((rawProbabilities[label] ?? 0) * 1000) / 10,
    }))
    .sort((a, b) => b.calibrated - a.calibrated);

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--bp-line)" vertical={false} />
        <XAxis dataKey="label" tick={{ fill: "var(--bp-text-muted)", fontSize: 11, fontFamily: "IBM Plex Mono" }} axisLine={{ stroke: "var(--bp-line)" }} tickLine={false} />
        <YAxis tick={{ fill: "var(--bp-text-muted)", fontSize: 11, fontFamily: "IBM Plex Mono" }} axisLine={false} tickLine={false} unit="%" />
        <Tooltip
          contentStyle={{ background: "var(--bp-bg-panel-raised)", border: "1px solid var(--bp-line)", borderRadius: 6, fontSize: 12 }}
          labelStyle={{ color: "var(--bp-text-primary)" }}
          formatter={(value) => `${value}%`}
        />
        <Legend
          formatter={(value) => (value === "calibrated" ? t("result.calibratedConfidence") : t("result.rawConfidence"))}
          wrapperStyle={{ fontSize: 11, color: "var(--bp-text-muted)" }}
        />
        <Bar dataKey="raw" fill="var(--bp-accent-uncertainty)" fillOpacity={0.55} radius={[3, 3, 0, 0]} />
        <Bar dataKey="calibrated" fill="var(--bp-accent-signal)" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
