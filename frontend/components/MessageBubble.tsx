"use client";

import { Markdown } from "@/lib/markdown";
import type { ChatMessage, Stage } from "@/lib/types";
import FeedbackButtons from "./FeedbackButtons";
import StatusLine from "./StatusLine";

export default function MessageBubble({
  message,
  stage,
  onCite,
  onRetry,
}: {
  message: ChatMessage;
  stage: Stage | null;
  onCite: (messageId: string, idx: number) => void;
  onRetry: (question: string) => void;
}) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-blue-600 px-4 py-2 text-sm text-white md:max-w-[70%]">
          {message.content}
        </div>
      </div>
    );
  }

  if (message.error) {
    return (
      <div className="flex justify-start">
        <div className="max-w-[85%] space-y-2 rounded-2xl rounded-bl-sm border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-800 dark:bg-red-950/50 dark:text-red-200 md:max-w-[70%]">
          <p>{message.error.message}</p>
          {message.question && (
            <button
              type="button"
              className="btn-ghost h-7 border-red-300 text-red-700 dark:border-red-700 dark:text-red-200"
              onClick={() => onRetry(message.question as string)}
            >
              Thử lại
            </button>
          )}
        </div>
      </div>
    );
  }

  const showSources = message.grounded === false && (message.sources?.length ?? 0) > 0;

  return (
    <div className="flex justify-start">
      <div className="max-w-[95%] space-y-2 rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-4 py-3 text-sm dark:border-slate-800 dark:bg-slate-900 md:max-w-[80%]">
        {message.streaming && !message.content && <StatusLine stage={stage} />}
        {message.grounded === false && (
          <div className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-700 dark:bg-amber-900/30 dark:text-amber-200">
            Không tìm thấy căn cứ đủ trong kho văn bản
          </div>
        )}
        {message.content && (
          <Markdown
            text={message.content}
            citations={message.citations}
            onCite={(idx) => onCite(message.id, idx)}
          />
        )}
        {message.streaming && message.content && <StatusLine stage={stage} />}
        {showSources && (
          <div className="text-xs">
            <p className="mb-1 font-medium text-slate-600 dark:text-slate-400">
              Các nguồn gần nhất:
            </p>
            <ul className="space-y-1">
              {message.sources!.map((s) => (
                <li key={s.idx}>
                  <button
                    type="button"
                    className="text-left text-blue-700 hover:underline dark:text-blue-400"
                    onClick={() => onCite(message.id, s.idx)}
                  >
                    [{s.idx}] {s.display_source}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
        {!message.streaming && message.messageId && (
          <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-200 pt-2 dark:border-slate-800">
            <FeedbackButtons messageId={message.messageId} />
            {(message.sources?.length ?? 0) > 0 && (
              <button
                type="button"
                className="text-xs text-slate-500 hover:underline dark:text-slate-400"
                onClick={() => onCite(message.id, message.sources![0].idx)}
              >
                Xem {message.sources!.length} nguồn
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
