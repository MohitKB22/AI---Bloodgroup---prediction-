import { useTranslation } from "react-i18next";

function polarToCartesian(cx, cy, r, angleDeg) {
  const rad = (angleDeg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy - r * Math.sin(rad) };
}

function describeArc(cx, cy, r, startAngle, endAngle) {
  const start = polarToCartesian(cx, cy, r, startAngle);
  const end = polarToCartesian(cx, cy, r, endAngle);
  const largeArcFlag = Math.abs(startAngle - endAngle) <= 180 ? 0 : 1;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 0 ${end.x} ${end.y}`;
}

/**
 * Renders calibrated confidence as the primary filled arc (the number the
 * model actually stands behind after temperature scaling) with the raw,
 * pre-calibration softmax confidence marked as a separate tick -- so the
 * gap between "what the network output" and "what we're actually claiming"
 * is always visible, not just the flattering one of the two numbers.
 */
export default function ConfidenceDial({ calibrated, raw, size = 220 }) {
  const { t } = useTranslation();
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 24;
  const strokeWidth = 14;

  const calibratedAngle = 180 - Math.max(0, Math.min(1, calibrated)) * 180;
  const rawAngle = 180 - Math.max(0, Math.min(1, raw)) * 180;

  const trackPath = describeArc(cx, cy, r, 180, 0);
  const valuePath = describeArc(cx, cy, r, 180, calibratedAngle);
  const rawOuter = polarToCartesian(cx, cy, r + strokeWidth / 2 + 5, rawAngle);
  const rawInner = polarToCartesian(cx, cy, r - strokeWidth / 2 - 5, rawAngle);

  return (
    <div className="flex flex-col items-center">
      <svg viewBox={`0 0 ${size} ${size / 2 + 36}`} className="w-full max-w-[240px]">
        <path d={trackPath} stroke="var(--bp-line)" strokeWidth={strokeWidth} fill="none" strokeLinecap="round" />
        <path d={valuePath} stroke="var(--bp-accent-signal)" strokeWidth={strokeWidth} fill="none" strokeLinecap="round" />
        <line
          x1={rawInner.x} y1={rawInner.y} x2={rawOuter.x} y2={rawOuter.y}
          stroke="var(--bp-accent-uncertainty)" strokeWidth={3}
        />
        <text x={cx} y={cy - 6} textAnchor="middle" fontSize={size * 0.17} fontWeight={600} className="fill-current text-primary" fontFamily="'IBM Plex Mono', monospace">
          {Math.round(calibrated * 100)}%
        </text>
        <text x={cx} y={cy + 16} textAnchor="middle" fontSize={size * 0.06} className="fill-current text-muted" fontFamily="'IBM Plex Mono', monospace">
          {t("result.calibratedConfidence").toUpperCase()}
        </text>
      </svg>
      <div className="mt-1 flex items-center gap-4 font-mono text-[11px] text-muted">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-full bg-signal" /> {t("result.calibratedConfidence")}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-0.5 bg-uncertainty" /> {t("result.rawConfidence")}: {Math.round(raw * 100)}%
        </span>
      </div>
    </div>
  );
}
