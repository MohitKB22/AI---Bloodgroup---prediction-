import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronLeft, ChevronRight } from "lucide-react";

import HistoryTable from "../components/HistoryTable";
import { predictionsApi } from "../api/client";

const PAGE_SIZE = 10;

export default function HistoryPage() {
  const { t } = useTranslation();
  const [page, setPage] = useState(1);
  const [data, setData] = useState({ items: [], total: 0 });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    predictionsApi.list(page, PAGE_SIZE).then((res) => {
      if (!cancelled) setData(res);
    }).finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, [page]);

  const totalPages = Math.max(1, Math.ceil(data.total / PAGE_SIZE));

  return (
    <div className="mx-auto max-w-4xl space-y-4 p-6">
      <h1 className="font-mono text-sm uppercase tracking-wide text-muted">{t("history.title")}</h1>

      {loading ? (
        <p className="text-sm text-muted">Loading&hellip;</p>
      ) : (
        <>
          <HistoryTable items={data.items} />
          {data.total > PAGE_SIZE && (
            <div className="flex items-center justify-center gap-3 pt-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="rounded-md border border-line p-1.5 text-muted disabled:opacity-30"
              >
                <ChevronLeft size={14} />
              </button>
              <span className="font-mono text-xs text-muted">{page} / {totalPages}</span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                className="rounded-md border border-line p-1.5 text-muted disabled:opacity-30"
              >
                <ChevronRight size={14} />
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
