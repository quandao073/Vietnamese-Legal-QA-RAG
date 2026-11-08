import { parseSseChunk } from "./sse";
import type {
  AdminStats,
  ApiError,
  ChatRequest,
  ChunkResponse,
  Citations,
  DoneEvent,
  FeedbackRequest,
  IngestJob,
  IngestRequest,
  PublicConfig,
  SearchMode,
  SearchResponse,
  Source,
  Stage,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ?? "http://localhost:8000/api/v1";

export const NETWORK_ERROR_MESSAGE = `Không kết nối được tới máy chủ. Vui lòng kiểm tra API đang chạy tại ${API_BASE}.`;

export class HttpError extends Error {
  status: number;
  code: string;
  constructor(status: number, body: Partial<ApiError>) {
    super(body.message ?? `Lỗi máy chủ (${status})`);
    this.status = status;
    this.code = body.code ?? "http_error";
  }
}

async function readError(res: Response): Promise<ApiError> {
  try {
    const body = (await res.json()) as Partial<ApiError>;
    return {
      code: body.code ?? `http_${res.status}`,
      message: body.message ?? `Lỗi máy chủ (${res.status})`,
    };
  } catch {
    return { code: `http_${res.status}`, message: `Lỗi máy chủ (${res.status})` };
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new HttpError(0, { code: "network", message: NETWORK_ERROR_MESSAGE });
  }
  if (!res.ok) throw new HttpError(res.status, await readError(res));
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export interface StreamHandlers {
  onStatus: (stage: Stage) => void;
  onSources: (sources: Source[]) => void;
  onToken: (text: string) => void;
  onCitations: (citations: Citations) => void;
  onDone: (done: DoneEvent) => void;
  onError: (error: ApiError) => void;
}

export async function streamChat(
  req: ChatRequest,
  handlers: StreamHandlers,
  signal?: AbortSignal
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(req),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") return;
    handlers.onError({ code: "network", message: NETWORK_ERROR_MESSAGE });
    return;
  }

  if (!res.ok) {
    handlers.onError(await readError(res));
    return;
  }
  if (!res.body) {
    handlers.onError({ code: "no_body", message: "Máy chủ không trả về dữ liệu." });
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;

  const dispatch = (event: string, data: string) => {
    let payload: unknown;
    try {
      payload = JSON.parse(data);
    } catch {
      return;
    }
    switch (event) {
      case "status":
        handlers.onStatus((payload as { stage: Stage }).stage);
        break;
      case "sources":
        handlers.onSources((payload as { sources: Source[] }).sources ?? []);
        break;
      case "token":
        handlers.onToken((payload as { text: string }).text ?? "");
        break;
      case "citations":
        handlers.onCitations(payload as Citations);
        break;
      case "done":
        finished = true;
        handlers.onDone(payload as DoneEvent);
        break;
      case "error":
        finished = true;
        handlers.onError(payload as ApiError);
        break;
      default:
        break;
    }
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const { events, rest } = parseSseChunk(buffer);
      buffer = rest;
      for (const ev of events) dispatch(ev.event, ev.data);
    }
    buffer += decoder.decode();
    if (buffer.trim()) {
      const { events } = parseSseChunk(buffer + "\n\n");
      for (const ev of events) dispatch(ev.event, ev.data);
    }
  } catch (e) {
    if ((e as Error).name === "AbortError") return;
    handlers.onError({ code: "stream", message: "Kết nối tới máy chủ bị gián đoạn." });
    return;
  }

  if (!finished) {
    handlers.onError({ code: "incomplete", message: "Máy chủ kết thúc luồng trả lời sớm." });
  }
}

export function search(params: {
  q: string;
  top_k?: number;
  mode?: SearchMode;
  strategy?: string;
}): Promise<SearchResponse> {
  const qs = new URLSearchParams({ q: params.q });
  if (params.top_k) qs.set("top_k", String(params.top_k));
  if (params.mode) qs.set("mode", params.mode);
  if (params.strategy) qs.set("strategy", params.strategy);
  return request<SearchResponse>(`/search?${qs.toString()}`);
}

export function getChunk(chunkId: string): Promise<ChunkResponse> {
  return request<ChunkResponse>(`/chunks/${encodeURIComponent(chunkId)}`);
}

export function postFeedback(body: FeedbackRequest): Promise<unknown> {
  return request<unknown>("/feedback", { method: "POST", body: JSON.stringify(body) });
}

export function getPublicConfig(): Promise<PublicConfig> {
  return request<PublicConfig>("/config/public");
}

function adminHeaders(token: string): HeadersInit {
  return { "X-Admin-Token": token };
}

export function adminStats(token: string): Promise<AdminStats> {
  return request<AdminStats>("/admin/stats", { headers: adminHeaders(token) });
}

export function adminIngest(
  token: string,
  body: IngestRequest
): Promise<{ run_id: string; status: string }> {
  return request("/admin/ingest", {
    method: "POST",
    headers: adminHeaders(token),
    body: JSON.stringify(body),
  });
}

export function adminRun(token: string, runId: string): Promise<IngestJob> {
  return request<IngestJob>(`/admin/ingest/${encodeURIComponent(runId)}`, {
    headers: adminHeaders(token),
  });
}
