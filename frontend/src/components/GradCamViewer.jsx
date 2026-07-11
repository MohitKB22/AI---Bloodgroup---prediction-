import { useState } from "react";
import { useTranslation } from "react-i18next";

export default function GradCamViewer({ assets }) {
  const { t } = useTranslation();
  const [activeIdx, setActiveIdx] = useState(0);

  if (!assets?.length) return null;
  const active = assets[activeIdx];

  return (
    <div className="rounded-lg border border-line bg-panel-raised p-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="font-mono text-xs uppercase tracking-wide text-muted">{t("result.explainability")}</span>
        <div className="flex gap-1">
          {assets.map((asset, idx) => (
            <button
              key={`${asset.backbone}-${asset.method}`}
              onClick={() => setActiveIdx(idx)}
              className={`rounded px-2 py-1 font-mono text-[10px] transition-colors ${
                idx === activeIdx ? "bg-signal/20 text-signal" : "text-muted hover:text-primary"
              }`}
            >
              {asset.backbone}
            </button>
          ))}
        </div>
      </div>
      <img src={active.overlay_url} alt={`${active.method} overlay for ${active.backbone}`} className="w-full rounded border border-line" />
      <p className="mt-2 text-center font-mono text-[10px] text-muted">
        {active.method === "grad_cam" ? "Grad-CAM" : "Attention rollout"} &middot; {active.backbone}
      </p>
    </div>
  );
}
