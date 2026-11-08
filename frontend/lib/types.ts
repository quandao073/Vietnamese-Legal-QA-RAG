export type Role = "user" | "assistant";

export interface HistoryTurn {
  role: Role;
  content: string;
}

export type SearchMode = "hybrid" | "dense" | "bm25";

export interface ChatOptions {
  top_k?: number;
  use_rerank?: boolean;
  use_rewrite?: boolean;
  strategy?: string;
  mode?: SearchMode;
}

export interface ChatRequest {
  question: string;
  session_id?: string;
  history?: HistoryTurn[];
  options?: ChatOptions;
}

export type Stage = "understanding" | "retrieving" | "reranking" | "generating";

export interface Source {
  idx: number;
  chunk_id: string;
  doc_id: string;
  display_source: string;
  url?: string | null;
  snippet: string;
  rerank_score?: number | null;
}

export interface Citations {
  verified: number[];
  weak: number[];
  invalid: number[];
}

export type CitationState = "verified" | "weak" | "invalid" | "neutral";

export interface DoneEvent {
  message_id: string;
  answer_md: string;
  grounded: boolean;
  intent?: string;
  standalone_question?: string;
  sources: Source[];
  citations: Citations;
  usage?: { input_tokens?: number; output_tokens?: number } | null;
  latency_ms?: number;
  model?: string;
  prompt_version?: string;
}

export interface ApiError {
  code: string;
  message: string;
  message_id?: string;
}

export interface SearchResult {
  chunk_id: string;
  doc_id: string;
  title: string;
  display_source: string;
  article?: string | null;
  clause?: string | null;
  url?: string | null;
  text: string;
  score?: number | null;
  rerank_score?: number | null;
  debug?: Record<string, unknown> | null;
}

export interface SearchResponse {
  q: string;
  mode: SearchMode;
  strategy: string;
  results: SearchResult[];
}

export interface Chunk {
  chunk_id: string;
  doc_id: string;
  seq: number;
  title: string;
  text: string;
  article?: string | null;
  article_title?: string | null;
  clause?: string | null;
  point?: string | null;
  display_source: string;
}

export interface ChunkResponse {
  chunk: Chunk;
  prev?: Chunk | null;
  next?: Chunk | null;
  url?: string | null;
}

export interface PublicConfig {
  default_strategy: string;
  strategies: string[];
  default_top_k: number;
  top_k_range: [number, number];
  rerank_enabled: boolean;
  query_rewrite_enabled: boolean;
  modes: SearchMode[];
  max_question_chars: number;
}

export type FeedbackReason = "wrong_citation" | "incomplete" | "hallucination" | "other";

export interface FeedbackRequest {
  message_id: string;
  rating: 1 | -1;
  reason?: FeedbackReason;
  comment?: string;
}

export interface IngestRun {
  run_id: string;
  source: string;
  strategy: string;
  status: string;
  n_docs?: number | null;
  n_chunks?: number | null;
  started_at?: string | null;
  finished_at?: string | null;
}

export interface IngestJob {
  run_id: string;
  status: string;
  progress?: { stage?: string; done?: number; total?: number } | null;
  log?: string[];
  result?: Record<string, unknown> | null;
  error?: string | null;
}

export interface NegativeFeedback {
  message_id: string;
  reason?: string | null;
  comment?: string | null;
  created_at?: string | null;
  question?: string | null;
  answer_md?: string | null;
}

export interface AdminStats {
  n_documents: number;
  collections: { strategy: string; n_chunks: number; n_docs: number }[];
  queries_24h: number;
  grounded_24h: number;
  feedback_up: number;
  feedback_down: number;
  thumbs_up_ratio: number | null;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
  tokens_24h: { input: number; output: number };
  recent_negative_feedback: NegativeFeedback[];
  ingest_runs: IngestRun[];
  jobs: IngestJob[];
}

export interface IngestRequest {
  source: "tamir" | "yuitc" | "all";
  strategy: string;
  max_docs?: number;
}

/** A chat message as kept in the UI / sessionStorage. */
export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  /** assistant-only fields */
  messageId?: string;
  sources?: Source[];
  citations?: Citations;
  grounded?: boolean;
  streaming?: boolean;
  error?: ApiError | null;
  /** question that produced this assistant message, used for "Thử lại" */
  question?: string;
}
