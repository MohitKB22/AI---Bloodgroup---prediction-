import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import ResultPanel from "../components/ResultPanel";
import { predictionsApi } from "../api/client";

export default function PredictionDetailPage() {
  const { id } = useParams();
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    predictionsApi.get(id)
      .then((res) => !cancelled && setResult(res))
      .catch(() => !cancelled && setError("Prediction not found."));
    return () => { cancelled = true; };
  }, [id]);

  return (
    <div className="mx-auto max-w-3xl p-6">
      {error && <p className="text-sm text-critical">{error}</p>}
      {result ? <ResultPanel result={result} /> : !error && <p className="text-sm text-muted">Loading&hellip;</p>}
    </div>
  );
}
