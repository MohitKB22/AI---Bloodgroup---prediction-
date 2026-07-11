import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Loader2 } from "lucide-react";

import UploadDropzone from "../components/UploadDropzone";
import ResultPanel from "../components/ResultPanel";
import { predictionsApi } from "../api/client";

export default function DashboardPage() {
  const { t } = useTranslation();
  const [file, setFile] = useState(null);
  const [useTta, setUseTta] = useState(true);
  const [explain, setExplain] = useState(true);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const prediction = await predictionsApi.create(file, { useTta, explain });
      setResult(prediction);
    } catch (err) {
      setError(err.response?.data?.detail ?? "Prediction failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-6">
      <div className="rounded-lg border border-line bg-panel p-5">
        <h1 className="mb-4 font-mono text-sm uppercase tracking-wide text-muted">{t("upload.title")}</h1>
        <UploadDropzone onFileSelected={setFile} disabled={loading} />

        <div className="mt-4 flex flex-wrap items-center gap-6">
          <label className="flex items-center gap-2 text-xs text-muted">
            <input type="checkbox" checked={useTta} onChange={(e) => setUseTta(e.target.checked)} className="accent-signal" />
            {t("upload.useTta")}
          </label>
          <label className="flex items-center gap-2 text-xs text-muted">
            <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} className="accent-signal" />
            {t("upload.generateExplain")}
          </label>
        </div>

        <button
          onClick={handleSubmit}
          disabled={!file || loading}
          className="mt-5 flex w-full items-center justify-center gap-2 rounded-md bg-signal py-2.5 font-mono text-sm font-semibold text-void transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? (
            <>
              <Loader2 size={16} className="animate-spin" /> {t("upload.analyzing")}
            </>
          ) : (
            t("upload.submit")
          )}
        </button>

        {error && <p className="mt-3 text-xs text-critical">{error}</p>}
      </div>

      {result && <ResultPanel result={result} />}
    </div>
  );
}
