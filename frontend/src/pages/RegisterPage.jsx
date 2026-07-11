import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Droplet } from "lucide-react";

import { useAuth } from "../context/AuthContext";
import DisclaimerBanner from "../components/DisclaimerBanner";

export default function RegisterPage() {
  const { t } = useTranslation();
  const { register } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await register(email, password, fullName || undefined);
      navigate("/", { replace: true });
    } catch (err) {
      setError(err.response?.data?.detail ?? t("auth.error"));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex min-h-screen flex-col bg-void">
      <DisclaimerBanner />
      <div className="flex flex-1 items-center justify-center px-4">
      <div className="w-full max-w-sm rounded-lg border border-line bg-panel p-8">
        <div className="mb-6 flex items-center justify-center gap-2 font-mono text-lg font-semibold text-primary">
          <Droplet className="text-signal" strokeWidth={2.5} /> BloodPrint
        </div>
        <h1 className="mb-6 text-center text-sm text-muted">{t("auth.registerTitle")}</h1>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="mb-1 block text-xs text-muted">{t("auth.fullName")}</label>
            <input
              type="text" value={fullName} onChange={(e) => setFullName(e.target.value)}
              className="w-full rounded-md border border-line bg-panel-raised px-3 py-2 text-sm text-primary focus:border-signal focus:outline-none"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-muted">{t("auth.email")}</label>
            <input
              type="email" required value={email} onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-md border border-line bg-panel-raised px-3 py-2 text-sm text-primary focus:border-signal focus:outline-none"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-muted">{t("auth.password")}</label>
            <input
              type="password" required minLength={8} value={password} onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-md border border-line bg-panel-raised px-3 py-2 text-sm text-primary focus:border-signal focus:outline-none"
            />
          </div>

          {error && <p className="text-xs text-critical">{error}</p>}

          <button
            type="submit" disabled={submitting}
            className="w-full rounded-md bg-signal py-2 font-mono text-sm font-semibold text-void transition-opacity hover:opacity-90 disabled:opacity-50"
          >
            {t("auth.registerButton")}
          </button>
        </form>

        <p className="mt-5 text-center text-xs text-muted">
          {t("auth.haveAccount")}{" "}
          <Link to="/login" className="text-signal hover:underline">{t("auth.loginLink")}</Link>
        </p>
      </div>
      </div>
    </div>
  );
}
