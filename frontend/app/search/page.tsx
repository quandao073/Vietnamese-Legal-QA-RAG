"use client";

import { useEffect, useState } from "react";
import { getPublicConfig, HttpError, search } from "@/lib/api";
import type { PublicConfig, SearchMode, SearchResponse, SearchResult } from "@/lib/types";

const MODES: SearchMode[] = ["hybrid", "dense", "bm25"];
const EXCERPT = 240;

function fmt(n?: number | null) {
  return typeof n === "number" ? n.toFixed(3) : "–";
}

function ResultRow({ r }: { r: SearchResult }) {
  const [expanded, setExpanded] = useState(false);
  const long = r.text.length > EXCERPT;
  return (
    <div className="card space-y-1">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <p className="text-sm font-semibold">{r.display_source}</p>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Điểm: {fmt(r.score)}
          {typeof r.rerank_score === "number" && ` · Rerank: ${fmt(r.rerank_score)}`}
        </p>
      </div>
      <p className="whitespace-pre-wrap text-sm text-slate-700 dark:text-slate-300">
        {expanded || !long ? r.text : `${r.text.slice(0, EXCERPT)}…`}
      </p>
      <div className="flex gap-3 text-xs">
        {long && (
          <button
            type="button"
            className="text-blue-700 hover:underline dark:text-blue-400"
            onClick={() => setExpanded((v) => !v)}
          >
            {expanded ? "Thu gọn" : "Xem thêm"}
          </button>
        )}
        {r.url && (
          <a
            href={r.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-700 hover:underline dark:text-blue-400"
          >
            Xem văn bản gốc ↗
          </a>
        )}
      </div>
    </div>
  );
}

function ResultColumn({
  title,
  data,
  error,
}: {
  title?: string;
  data: SearchResponse | null;
  error: string | null;
}) {
  return (
    <div className="min-w-0 flex-1 space-y-3">
      {title && <h2 className="text-sm font-semibold uppercase tracking-wide">{title}</h2>}
      {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
      {data && data.results.length === 0 && (
        <p className="text-sm text-slate-500">Không có kết quả.</p>
      )}
      {data?.results.map((r) => <ResultRow key={r.chunk_id} r={r} />)}
    </div>
  );
}

type Slot = { data: SearchResponse | null; error: string | null };
const EMPTY: Slot = { data: null, error: null };

export default function SearchPage() {
  const [config, setConfig] = useState<PublicConfig | null>(null);
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [topK, setTopK] = useState(5);
  const [strategy, setStrategy] = useState("article");
  const [compare, setCompare] = useState(false);
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<Record<SearchMode, Slot>>({
    hybrid: EMPTY,
    dense: EMPTY,
    bm25: EMPTY,
  });

  useEffect(() => {
    getPublicConfig()
      .then((cfg) => {
        setConfig(cfg);
        setTopK(cfg.default_top_k);
        setStrategy(cfg.default_strategy);
      })
      .catch(() => {});
  }, []);

  const run = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!q.trim()) return;
    setBusy(true);
    const modes = compare ? MODES : [mode];
    const settled = await Promise.all(
      modes.map(async (m): Promise<[SearchMode, Slot]> => {
        try {
          return [m, { data: await search({ q: q.trim(), top_k: topK, mode: m, strategy }), error: null }];
        } catch (err) {
          return [m, { data: null, error: err instanceof HttpError ? err.message : "Lỗi không xác định." }];
        }
      })
    );
    setResults({ hybrid: EMPTY, dense: EMPTY, bm25: EMPTY, ...Object.fromEntries(settled) });
    setBusy(false);
  };

  const [min, max] = config?.top_k_range ?? [3, 10];
  const strategies = config?.strategies?.length ? config.strategies : [strategy];

  return (
    <main className="mx-auto max-w-7xl space-y-4 p-4">
      <h1 className="text-xl font-semibold">Tìm kiếm & so sánh truy hồi</h1>
      <form onSubmit={run} className="card space-y-3">
        <input
          className="input"
          placeholder="Nhập truy vấn…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          aria-label="Truy vấn"
        />
        <div className="flex flex-wrap items-end gap-4 text-sm">
          <fieldset className="flex items-center gap-3" disabled={compare}>
            <legend className="label mb-1">Chế độ</legend>
            {MODES.map((m) => (
              <label key={m} className="flex items-center gap-1">
                <input
                  type="radio"
                  name="mode"
                  value={m}
                  checked={mode === m}
                  onChange={() => setMode(m)}
                />
                {m}
              </label>
            ))}
          </fieldset>
          <div>
            <label className="label" htmlFor="s-topk">
              top_k
            </label>
            <input
              id="s-topk"
              type="number"
              className="input w-20"
              min={min}
              max={max}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
          </div>
          <div>
            <label className="label" htmlFor="s-strategy">
              Chiến lược
            </label>
            <select
              id="s-strategy"
              className="input"
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
            >
              {strategies.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} />
            So sánh cả ba chế độ
          </label>
          <button type="submit" className="btn-primary ml-auto" disabled={busy || !q.trim()}>
            {busy ? "Đang tìm…" : "Tìm kiếm"}
          </button>
        </div>
      </form>

      {compare ? (
        <div className="grid gap-4 md:grid-cols-3">
          {MODES.map((m) => (
            <ResultColumn key={m} title={m} data={results[m].data} error={results[m].error} />
          ))}
        </div>
      ) : (
        <ResultColumn data={results[mode].data} error={results[mode].error} />
      )}
    </main>
  );
}
