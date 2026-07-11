import { useTranslation } from "react-i18next";
import { AlertTriangle } from "lucide-react";

export default function DisclaimerBanner() {
  const { t } = useTranslation();

  return (
    <div className="flex items-center gap-2 border-b border-critical/30 bg-critical/10 px-4 py-2 text-xs text-critical">
      <AlertTriangle size={14} className="shrink-0" />
      <span>{t("disclaimerBanner")}</span>
    </div>
  );
}
