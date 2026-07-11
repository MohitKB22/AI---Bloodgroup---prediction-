import { useTranslation } from "react-i18next";
import { Languages } from "lucide-react";

const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "es", label: "Espa\u00f1ol" },
  { code: "fr", label: "Fran\u00e7ais" },
  { code: "hi", label: "\u0939\u093f\u0928\u094d\u0926\u0940" },
];

export default function LanguageSwitcher() {
  const { i18n } = useTranslation();

  return (
    <div className="flex items-center gap-1.5 rounded-md border border-line bg-panel-raised px-2 py-1.5 text-xs text-muted">
      <Languages size={14} />
      <select
        value={i18n.language?.slice(0, 2) ?? "en"}
        onChange={(e) => i18n.changeLanguage(e.target.value)}
        className="bg-transparent text-primary focus:outline-none"
      >
        {LANGUAGES.map((lang) => (
          <option key={lang.code} value={lang.code} className="bg-panel text-primary">
            {lang.label}
          </option>
        ))}
      </select>
    </div>
  );
}
