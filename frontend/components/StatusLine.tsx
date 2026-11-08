import type { Stage } from "@/lib/types";

export const STAGE_TEXT: Record<Stage, string> = {
  understanding: "Đang phân tích câu hỏi…",
  retrieving: "Đang tìm văn bản…",
  reranking: "Đang xếp hạng nguồn…",
  generating: "Đang soạn trả lời…",
};

export default function StatusLine({ stage }: { stage: Stage | null }) {
  if (!stage) return null;
  return (
    <div
      className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400"
      role="status"
      aria-live="polite"
    >
      <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-blue-500" />
      {STAGE_TEXT[stage]}
    </div>
  );
}
