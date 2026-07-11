import { useTranslation } from "react-i18next";
import { Download } from "lucide-react";

import ConfidenceDial from "./ConfidenceDial";
import ProbabilityChart from "./ProbabilityChart";
import QualityBadge from "./QualityBadge";
import GradCamViewer from "./GradCamViewer";
import { predictionsApi } from "../api/client";

export default function ResultPanel({ result }) {
  const { t } = useTranslation();
  if (!result) return null;

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-line bg-panel p-5">
        <div className="flex flex-col items-center gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="font-mono text-xs uppercase tracking-wide text-muted">{t("result.predictedLabel")}</p>
            <p className="mt-1 font-mono text-4xl font-bold text-signal">{result.predicted_label}</p>
            <p className="mt-2 max-w-sm text-xs text-muted">{result.disclaimer}</p>
          </div>
          <ConfidenceDial calibrated={result.confidence_calibrated} raw={result.confidence_raw} />
        </div>
      </div>

      <div className="rounded-lg border border-line bg-panel p-5">
        <ProbabilityChart rawProbabilities={result.raw_probabilities} calibratedProbabilities={result.calibrated_probabilities} />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <QualityBadge quality={result.quality} />
        <GradCamViewer assets={result.explainability} />
      </div>

      {result.ensemble_weights && Object.keys(result.ensemble_weights).length > 0 && (
        <div className="rounded-lg border border-line bg-panel-raised p-4">
          <p className="mb-2 font-mono text-xs uppercase tracking-wide text-muted">{t("result.ensembleComposition")}</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(result.ensemble_weights).map(([name, weight]) => (
              <span key={name} className="rounded-full border border-line bg-void px-3 py-1 font-mono text-xs text-primary">
                {name}: {(weight * 100).toFixed(1)}%
              </span>
            ))}
          </div>
        </div>
      )}

      <a
        href={predictionsApi.reportUrl(result.id)}
        target="_blank"
        rel="noreferrer"
        className="flex w-fit items-center gap-2 rounded-md border border-signal/40 bg-signal/10 px-4 py-2 font-mono text-xs text-signal transition-colors hover:bg-signal/20"
      >
        <Download size={14} /> {t("result.downloadReport")}
      </a>
    </div>
  );
}
