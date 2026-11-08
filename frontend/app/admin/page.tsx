"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { adminIngest, adminRun, adminStats, HttpError } from "@/lib/api";
import type { AdminStats, IngestJob, IngestRequest, NegativeFeedback } from "@/lib/types";

const TOKEN_KEY = "vnlegal.admin_token";
const INVALID_TOKEN = "Mã quản trị không hợp lệ";
const POLL_MS = 2000;

function errMsg(e: unknown): string {
  if (e instanceof HttpError) return e.status === 401 ? INVALID_TOKEN : e.message;
  return "Lỗi không xác định.";
}

function fmtNum(n: number | null | undefined, suffix = "") {
  return typeof n === "number" ? `${n.toLocaleString("vi-VN")}${suffix}` : "–";
}

function fmtTime(s?: string | null) {
  if (!s) return "–";
  const d = new Date(s);
  return isNaN(d.getTime()) ? s : d.toLocaleString("vi-VN");
}

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div className="card">
      <p className="text-xs text-slate-500 dark:text-slate-400">{label}</p>
      <p className="mt-1 text-xl font-semibold">{value}</p>
    </div>
  );
}

function FeedbackRow({ f }: { f: NegativeFeedback }) {
  const [open, setOpen] = useState(false);
  const answer = f.answer_md ?? "";
  const short = answer.length > 160 ? `${answer.slice(0, 160)}…` : answer;
  return (
    <tr className="border-t border-slate-200 align-top dark:border-slate-800">
      <td className="px-2 py-2 text-xs text-slate-500">{fmtTime(f.created_at)}</td>
      <td className="px-2 py-2 text-xs">{f.reason ?? "–"}</td>
      <td className="px-2 py-2 text-sm">{f.question ?? "–"}</td>
      <td className="px-2 py-2 text-sm">
        <p className="whitespace-pre-wrap">{open ? answer : short}</p>
        {answer.length > 160 && (
          <button
            type="button"
            className="text-xs text-blue-700 hover:underline dark:text-blue-400"
            onClick={() => setOpen((v) => !v)}
          >
            {open ? "Thu gọn" : "Xem đầy đủ"}
          </button>
        )}
        {f.comment && <p className="mt-1 text-xs italic text-slate-500">Góp ý: {f.comment}</p>}
      </td>
    </tr>
  );
}

function JobView({ job }: { job: IngestJob }) {
  const p = job.progress;
  const pct = p?.total ? Math.round(((p.done ?? 0) / p.total) * 100) : null;
  return (
    <div className="space-y-2 rounded-lg border border-slate-200 p-3 text-sm dark:border-slate-800">
      <p>
        <span className="font-medium">Tiến trình {job.run_id}</span> · trạng thái:{" "}
        <b>{job.status}</b>
        {p?.stage && ` · giai đoạn: ${p.stage}`}
        {pct !== null && ` · ${p?.done}/${p?.total} (${pct}%)`}
      </p>
      {pct !== null && (
        <div className="h-2 w-full overflow-hidden rounded bg-slate-200 dark:bg-slate-800">
          <div className="h-full bg-blue-600" style={{ width: `${pct}%` }} />
        </div>
      )}
      {job.error && <p className="text-red-600 dark:text-red-400">{job.error}</p>}
      {job.log && job.log.length > 0 && (
        <pre className="max-h-40 overflow-auto rounded bg-slate-100 p-2 text-xs dark:bg-slate-900">
          {job.log.slice(-15).join("\n")}
        </pre>
      )}
    </div>
  );
}

export default function AdminPage() {
  const [token, setToken] = useState("");
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [source, setSource] = useState<IngestRequest["source"]>("all");
  const [strategy, setStrategy] = useState("article");
  const [maxDocs, setMaxDocs] = useState("");
  const [job, setJob] = useState<IngestJob | null>(null);
  const [ingestErr, setIngestErr] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);


  const load = useCallback(async (t: string) => {
    if (!t) return;
    setLoading(true);
    setError(null);
    try {
      setStats(await adminStats(t));
      localStorage.setItem(TOKEN_KEY, t);
    } catch (e) {
      setStats(null);
      setError(errMsg(e));
    } finally {
      setLoading(false);
    }
  }, []);

  // Restore a saved token and load stats once on mount.
  useEffect(() => {
    let saved = "";
    try {
      saved = localStorage.getItem(TOKEN_KEY) ?? "";
    } catch {
      /* ignore */
    }
    if (saved) {
      setToken(saved);
      load(saved);
    }
  }, [load]);

  const poll = useCallback(
    async (runId: string) => {
      try {
        const j = await adminRun(token, runId);
        setJob(j);
        if (j.status === "done" || j.status === "failed" || j.status === "succeeded") {
          load(token);
          return;
        }
      } catch (e) {
        setIngestErr(errMsg(e));
        return;
      }
      timerRef.current = setTimeout(() => poll(runId), POLL_MS);
    },
    [token, load]
  );

  useEffect(
    () => () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    },
    []
  );

  const startIngest = async (e: React.FormEvent) => {
    e.preventDefault();
    setIngestErr(null);
    try {
      const body: IngestRequest = { source, strategy };
      if (maxDocs.trim()) body.max_docs = Number(maxDocs);
      const r = await adminIngest(token, body);
      setJob({ run_id: r.run_id, status: r.status });
      poll(r.run_id);
    } catch (err) {
      setIngestErr(errMsg(err));
    }
  };

  return (
    <main className="mx-auto max-w-7xl space-y-6 p-4">
      <h1 className="text-xl font-semibold">Quản trị hệ thống</h1>

      <form
        className="card flex flex-wrap items-end gap-3"
        onSubmit={(e) => {
          e.preventDefault();
          load(token);
        }}
      >
        <div className="min-w-0 flex-1">
          <label className="label" htmlFor="admin-token">
            Mã quản trị (X-Admin-Token)
          </label>
          <input
            id="admin-token"
            type="password"
            className="input"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            autoComplete="off"
          />
        </div>
        <button type="submit" className="btn-primary" disabled={loading || !token}>
          {loading ? "Đang tải…" : "Tải thống kê"}
        </button>
      </form>
      {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}

      {stats && (
        <>
          <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Kpi label="Số văn bản" value={fmtNum(stats.n_documents)} />
            {stats.collections.map((c) => (
              <Kpi
                key={c.strategy}
                label={`Chunks (${c.strategy})`}
                value={`${fmtNum(c.n_chunks)} / ${fmtNum(c.n_docs)} văn bản`}
              />
            ))}
            <Kpi label="Truy vấn 24h" value={fmtNum(stats.queries_24h)} />
            <Kpi label="Có căn cứ 24h" value={fmtNum(stats.grounded_24h)} />
            <Kpi
              label="Tỷ lệ hài lòng"
              value={
                typeof stats.thumbs_up_ratio === "number"
                  ? `${Math.round(stats.thumbs_up_ratio * 100)}% (${stats.feedback_up}👍 / ${stats.feedback_down}👎)`
                  : "–"
              }
            />
            <Kpi
              label="Độ trễ p50 / p95"
              value={`${fmtNum(stats.latency_p50_ms, " ms")} / ${fmtNum(stats.latency_p95_ms, " ms")}`}
            />
            <Kpi
              label="Token 24h (vào / ra)"
              value={`${fmtNum(stats.tokens_24h?.input)} / ${fmtNum(stats.tokens_24h?.output)}`}
            />
          </section>

          <section className="card space-y-3">
            <h2 className="font-semibold">Nạp dữ liệu (ingest)</h2>
            <form className="flex flex-wrap items-end gap-3 text-sm" onSubmit={startIngest}>
              <div>
                <label className="label" htmlFor="ing-source">
                  Nguồn
                </label>
                <select
                  id="ing-source"
                  className="input"
                  value={source}
                  onChange={(e) => setSource(e.target.value as IngestRequest["source"])}
                >
                  <option value="all">all</option>
                  <option value="tamir">tamir</option>
                  <option value="yuitc">yuitc</option>
                </select>
              </div>
              <div>
                <label className="label" htmlFor="ing-strategy">
                  Chiến lược
                </label>
                <input
                  id="ing-strategy"
                  className="input"
                  value={strategy}
                  onChange={(e) => setStrategy(e.target.value)}
                />
              </div>
              <div>
                <label className="label" htmlFor="ing-max">
                  Số văn bản tối đa
                </label>
                <input
                  id="ing-max"
                  type="number"
                  min={1}
                  className="input w-28"
                  value={maxDocs}
                  onChange={(e) => setMaxDocs(e.target.value)}
                  placeholder="tất cả"
                />
              </div>
              <button
                type="submit"
                className="btn-primary"
                disabled={!!job && !["done", "failed", "succeeded"].includes(job.status)}
              >
                Bắt đầu nạp
              </button>
            </form>
            {ingestErr && <p className="text-sm text-red-600 dark:text-red-400">{ingestErr}</p>}
            {job && <JobView job={job} />}
          </section>

          <section className="card space-y-2 overflow-x-auto">
            <h2 className="font-semibold">Lịch sử nạp dữ liệu</h2>
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-slate-500">
                <tr>
                  <th className="px-2 py-1">Run</th>
                  <th className="px-2 py-1">Nguồn</th>
                  <th className="px-2 py-1">Chiến lược</th>
                  <th className="px-2 py-1">Trạng thái</th>
                  <th className="px-2 py-1">Văn bản</th>
                  <th className="px-2 py-1">Chunks</th>
                  <th className="px-2 py-1">Bắt đầu</th>
                  <th className="px-2 py-1">Kết thúc</th>
                </tr>
              </thead>
              <tbody>
                {stats.ingest_runs.length === 0 && (
                  <tr>
                    <td colSpan={8} className="px-2 py-2 text-slate-500">
                      Chưa có lần nạp nào.
                    </td>
                  </tr>
                )}
                {stats.ingest_runs.map((r) => (
                  <tr key={r.run_id} className="border-t border-slate-200 dark:border-slate-800">
                    <td className="px-2 py-1 font-mono text-xs">{r.run_id}</td>
                    <td className="px-2 py-1">{r.source}</td>
                    <td className="px-2 py-1">{r.strategy}</td>
                    <td className="px-2 py-1">{r.status}</td>
                    <td className="px-2 py-1">{fmtNum(r.n_docs)}</td>
                    <td className="px-2 py-1">{fmtNum(r.n_chunks)}</td>
                    <td className="px-2 py-1 text-xs">{fmtTime(r.started_at)}</td>
                    <td className="px-2 py-1 text-xs">{fmtTime(r.finished_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section className="card space-y-2 overflow-x-auto">
            <h2 className="font-semibold">Phản hồi tiêu cực gần đây</h2>
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-slate-500">
                <tr>
                  <th className="px-2 py-1">Thời gian</th>
                  <th className="px-2 py-1">Lý do</th>
                  <th className="px-2 py-1">Câu hỏi</th>
                  <th className="px-2 py-1">Câu trả lời</th>
                </tr>
              </thead>
              <tbody>
                {stats.recent_negative_feedback.length === 0 && (
                  <tr>
                    <td colSpan={4} className="px-2 py-2 text-slate-500">
                      Chưa có phản hồi tiêu cực.
                    </td>
                  </tr>
                )}
                {stats.recent_negative_feedback.slice(0, 20).map((f) => (
                  <FeedbackRow key={f.message_id} f={f} />
                ))}
              </tbody>
            </table>
          </section>
        </>
      )}
    </main>
  );
}
