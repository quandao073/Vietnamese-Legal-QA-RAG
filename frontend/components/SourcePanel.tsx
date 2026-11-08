"use client";

import { useEffect, useState } from "react";
import { getChunk, HttpError } from "@/lib/api";
import { citationState } from "@/lib/markdown";
import type { Chunk, ChunkResponse, Citations, Source } from "@/lib/types";

const STATE_LABEL = {
  verified: ["Đã xác minh", "text-green-700 dark:text-green-400"],
  weak: ["Khớp một phần", "text-yellow-700 dark:text-yellow-400"],
  invalid: ["Không tồn tại", "text-red-700 dark:text-red-400"],
  neutral: ["", ""],
} as const;

function ChunkView({ c }: { c: Chunk }) {
  const heading = [c.article && `Điều ${c.article}`, c.article_title].filter(Boolean).join(". ");
  return (
    <div className="space-y-1">
      {heading && <p className="text-sm font-semibold">{heading}</p>}
      <p className="whitespace-pre-wrap text-sm leading-relaxed">{c.text}</p>
    </div>
  );
}

function SourceCard({
  source,
  citations,
  active,
  onSelect,
}: {
  source: Source;
  citations?: Citations | null;
  active: boolean;
  onSelect: () => void;
}) {
  const [full, setFull] = useState<ChunkResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const state = citationState(source.idx, citations);
  const [label, labelCls] = STATE_LABEL[state];

  const loadFull = async () => {
    setLoading(true);
    setErr(null);
    try {
      setFull(await getChunk(source.chunk_id));
    } catch (e) {
      setErr(e instanceof HttpError ? e.message : "Không tải được nội dung.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      id={`source-${source.idx}`}
      onClick={onSelect}
      className={`card cursor-pointer space-y-2 transition ${
        active ? "ring-2 ring-blue-500" : ""
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <p className="text-sm font-semibold">
          <span className="mr-1 inline-flex h-5 min-w-[1.5rem] items-center justify-center rounded-md bg-slate-200 px-1 text-[11px] dark:bg-slate-700">
            {source.idx}
          </span>
          {source.display_source}
        </p>
        {label && <span className={`shrink-0 text-xs font-medium ${labelCls}`}>{label}</span>}
      </div>
      {typeof source.rerank_score === "number" && (
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Điểm xếp hạng: {source.rerank_score.toFixed(3)}
        </p>
      )}
      {full ? (
        <div className="space-y-3 border-t border-slate-200 pt-2 dark:border-slate-800">
          {full.prev && (
            <details>
              <summary className="cursor-pointer text-xs text-slate-500">Đoạn trước</summary>
              <ChunkView c={full.prev} />
            </details>
          )}
          <ChunkView c={full.chunk} />
          {full.next && (
            <details>
              <summary className="cursor-pointer text-xs text-slate-500">Đoạn sau</summary>
              <ChunkView c={full.next} />
            </details>
          )}
        </div>
      ) : (
        <p className="text-sm text-slate-700 dark:text-slate-300">{source.snippet}</p>
      )}
      {err && <p className="text-xs text-red-600 dark:text-red-400">{err}</p>}
      <div className="flex flex-wrap gap-2">
        {!full && (
          <button
            type="button"
            className="btn-ghost h-7"
            disabled={loading}
            onClick={(e) => {
              e.stopPropagation();
              loadFull();
            }}
          >
            {loading ? "Đang tải…" : "Xem toàn bộ Điều"}
          </button>
        )}
        {(source.url || full?.url) && (
          <a
            className="btn-ghost h-7"
            href={(full?.url || source.url) as string}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
          >
            Xem văn bản gốc ↗
          </a>
        )}
      </div>
    </div>
  );
}

export default function SourcePanel({
  sources,
  citations,
  activeIdx,
  onActive,
  onClose,
}: {
  sources: Source[];
  citations?: Citations | null;
  activeIdx: number | null;
  onActive: (idx: number) => void;
  onClose: () => void;
}) {
  useEffect(() => {
    if (activeIdx == null) return;
    document
      .getElementById(`source-${activeIdx}`)
      ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [activeIdx]);

  return (
    <aside
      className="fixed inset-x-0 bottom-0 z-30 flex max-h-[70vh] flex-col rounded-t-2xl border-t border-slate-200 bg-slate-50 shadow-2xl dark:border-slate-800 dark:bg-slate-950 md:static md:max-h-none md:w-96 md:shrink-0 md:rounded-none md:border-l md:border-t-0 md:shadow-none"
      aria-label="Nguồn trích dẫn"
    >
      <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-800">
        <h2 className="text-sm font-semibold">Nguồn trích dẫn ({sources.length})</h2>
        <button type="button" className="btn-ghost h-7" onClick={onClose} aria-label="Đóng">
          ✕
        </button>
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {sources.length === 0 && (
          <p className="text-sm text-slate-500">Chưa có nguồn nào cho câu trả lời này.</p>
        )}
        {sources.map((s) => (
          <SourceCard
            key={s.idx}
            source={s}
            citations={citations}
            active={s.idx === activeIdx}
            onSelect={() => onActive(s.idx)}
          />
        ))}
      </div>
    </aside>
  );
}
