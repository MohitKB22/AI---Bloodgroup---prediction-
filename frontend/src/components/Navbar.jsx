import { NavLink } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Activity, BarChart3, Droplet, History, Info } from "lucide-react";

import LanguageSwitcher from "./LanguageSwitcher";
import ThemeToggle from "./ThemeToggle";

const linkClass = ({ isActive }) =>
  `flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors ${
    isActive ? "bg-panel-raised text-signal" : "text-muted hover:text-primary"
  }`;

export default function Navbar() {
  const { t } = useTranslation();

  return (
    <header className="flex items-center justify-between border-b border-line bg-panel px-4 py-3">
      <div className="flex items-center gap-2 font-mono text-sm font-semibold tracking-tight text-primary">
        <Droplet size={18} className="text-signal" strokeWidth={2.5} />
        {t("appName")}
      </div>

      <nav className="flex items-center gap-1">
        <NavLink to="/" end className={linkClass}>
          <Activity size={14} /> {t("nav.dashboard")}
        </NavLink>
        <NavLink to="/history" className={linkClass}>
          <History size={14} /> {t("nav.history")}
        </NavLink>
        <NavLink to="/analytics" className={linkClass}>
          <BarChart3 size={14} /> {t("nav.analytics")}
        </NavLink>
        <NavLink to="/about" className={linkClass}>
          <Info size={14} /> {t("nav.about")}
        </NavLink>
      </nav>

      <div className="flex items-center gap-2">
        <LanguageSwitcher />
        <ThemeToggle />
      </div>
    </header>
  );
}
