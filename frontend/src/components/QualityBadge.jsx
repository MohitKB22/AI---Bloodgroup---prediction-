import { useTranslation } from "react-i18next";
import { CheckCircle2, XCircle } from "lucide-react";

function scoreColor(score) {
  if (score >= 70) return "text-signal border-signal/40 bg-signal/10";
  if (score >= 45) return "text-uncertainty border-uncertainty/40 bg-uncertainty/10";
  return "text-critical border-critical/40 bg-critical/10";
}

export default function QualityBadge({ quality }) {
  const { t } = useTranslation();
  if (!quality) return null;

  return (
    <div className="rounded-lg border border-line bg-panel-raised p-4">
      <div className="flex items-center justify-between">
        <span className="font-mono text-xs uppercase tracking-wide text-muted">{t("result.qualityScore")}</span>
        <span className={`rounded-full border px-2.5 py-0.5 font-mono text-xs font-semibold ${scoreColor(quality.overall_score)}`}>
          {Math.round(quality.overall_score)}/100
        </span>
      </div>

      <div className="mt-3 space-y-1.5">
        {Object.entries(quality.sub_scores).map(([key, value]) => (
          <div key={key} className="flex items-center gap-2 text-xs">
            <span className="w-40 shrink-0 text-muted">{key.replaceAll("_", " ")}</span>
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-void">
              <div className="h-full rounded-full bg-signal" style={{ width: `${Math.min(100, value)}%` }} />
            </div>
            <span className="w-8 text-right font-mono text-muted">{Math.round(value)}</span>
          </div>
        ))}
      </div>

      <div className="mt-3 border-t border-line pt-3">
        {quality.warnings?.length ? (
          <ul className="space-y-1.5">
            {quality.warnings.map((w, i) => (
              <li key={i} className="flex items-start gap-1.5 text-xs text-uncertainty">
                <XCircle size={13} className="mt-0.5 shrink-0" /> {w}
              </li>
            ))}
          </ul>
        ) : (
          <p className="flex items-center gap-1.5 text-xs text-signal">
            <CheckCircle2 size={13} /> {t("result.noWarnings")}
          </p>
        )}
      </div>
    </div>
  );
}
