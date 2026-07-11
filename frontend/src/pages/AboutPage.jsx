import { useTranslation } from "react-i18next";
import { AlertTriangle, FlaskConical, GitBranch, ShieldAlert } from "lucide-react";

export default function AboutPage() {
  const { t } = useTranslation();

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-6">
      <h1 className="font-mono text-sm uppercase tracking-wide text-muted">{t("about.title")}</h1>

      <div className="flex gap-3 rounded-lg border border-critical/30 bg-critical/10 p-4 text-sm text-critical">
        <ShieldAlert size={20} className="mt-0.5 shrink-0" />
        <p>
          This is a research prototype, not a medical device. Fingerprint-based blood group prediction is a
          statistically contested hypothesis without robust, independently validated scientific support. Predictions
          from this system must never be used for medical, clinical, or blood-donation/transfusion decisions. Always
          rely on certified laboratory blood typing for any real-world decision.
        </p>
      </div>

      <section className="space-y-2 rounded-lg border border-line bg-panel p-5 text-sm text-muted">
        <p className="flex items-center gap-2 font-mono text-xs uppercase tracking-wide text-primary">
          <FlaskConical size={14} className="text-signal" /> What this platform is
        </p>
        <p>
          BloodPrint is an engineering demonstration of a production-style ML platform: an ensemble of EfficientNetV2,
          Vision Transformer, and ConvNeXt models with transfer learning, calibrated confidence, and Grad-CAM /
          attention-rollout explainability, served through an authenticated API with a full dashboard on top. The
          interesting engineering problems here are the ones any deployed classifier faces &mdash; leakage-safe evaluation,
          calibration, explainability, and honest uncertainty reporting &mdash; independent of whether the underlying
          prediction target turns out to be real.
        </p>
      </section>

      <section className="space-y-2 rounded-lg border border-line bg-panel p-5 text-sm text-muted">
        <p className="flex items-center gap-2 font-mono text-xs uppercase tracking-wide text-primary">
          <AlertTriangle size={14} className="text-uncertainty" /> Known limitations
        </p>
        <ul className="list-inside list-disc space-y-1">
          <li>The core hypothesis (fingerprint ridge patterns correlate with ABO blood group) is not well-established in the scientific literature.</li>
          <li>With 8 balanced classes, a chance baseline is ~12.5% accuracy &mdash; treat any reported accuracy in that neighborhood as evidence of no real signal, not a bug.</li>
          <li>Calibration reshapes confidence to match empirical accuracy on the evaluation set; it cannot manufacture predictive validity that isn&apos;t there.</li>
          <li>Explainability heatmaps show what the model attended to, not that the attended-to region is medically meaningful.</li>
        </ul>
      </section>

      <section className="space-y-2 rounded-lg border border-line bg-panel p-5 text-sm text-muted">
        <p className="flex items-center gap-2 font-mono text-xs uppercase tracking-wide text-primary">
          <GitBranch size={14} className="text-signal" /> Full documentation
        </p>
        <p>See MODEL_CARD.md and README.md in the repository for the complete model card, evaluation methodology, and setup instructions.</p>
      </section>
    </div>
  );
}
