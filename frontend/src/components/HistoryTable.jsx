import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ArrowRight } from "lucide-react";

export default function HistoryTable({ items }) {
  const { t } = useTranslation();

  if (!items?.length) {
    return <p className="rounded-lg border border-line bg-panel-raised p-6 text-center text-sm text-muted">{t("history.empty")}</p>;
  }

  return (
    <div className="overflow-hidden rounded-lg border border-line">
      <table className="w-full text-left text-sm">
        <thead className="bg-panel-raised font-mono text-xs uppercase tracking-wide text-muted">
          <tr>
            <th className="px-4 py-3">{t("result.predictedLabel")}</th>
            <th className="px-4 py-3">{t("result.calibratedConfidence")}</th>
            <th className="px-4 py-3">Model</th>
            <th className="px-4 py-3">Date</th>
            <th className="px-4 py-3" />
          </tr>
        </thead>
        <tbody className="divide-y divide-line bg-panel">
          {items.map((item) => (
            <tr key={item.id} className="transition-colors hover:bg-panel-raised">
              <td className="px-4 py-3 font-mono font-semibold text-signal">{item.predicted_label}</td>
              <td className="px-4 py-3 font-mono text-primary">{Math.round(item.confidence_calibrated * 100)}%</td>
              <td className="px-4 py-3 font-mono text-xs text-muted">{item.model_version}</td>
              <td className="px-4 py-3 text-xs text-muted">{new Date(item.created_at).toLocaleString()}</td>
              <td className="px-4 py-3 text-right">
                <Link to={`/history/${item.id}`} className="inline-flex items-center gap-1 text-xs text-signal hover:underline">
                  {t("history.viewDetail")} <ArrowRight size={12} />
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
