"use client";

import { useState } from "react";
import { HttpError, postFeedback } from "@/lib/api";
import type { FeedbackReason } from "@/lib/types";

const REASONS: { value: FeedbackReason; label: string }[] = [
  { value: "wrong_citation", label: "Trích dẫn sai" },
  { value: "incomplete", label: "Thiếu thông tin" },
  { value: "hallucination", label: "Bịa thông tin" },
  { value: "other", label: "Khác" },
];

export default function FeedbackButtons({ messageId }: { messageId: string }) {
  const [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);
  const [reason, setReason] = useState<FeedbackReason>("wrong_citation");
  const [comment, setComment] = useState("");
  const [error, setError] = useState<string | null>(null);

  const send = async (rating: 1 | -1) => {
    setBusy(true);
    setError(null);
    try {
      await postFeedback({
        message_id: messageId,
        rating,
        ...(rating === -1 ? { reason, comment: comment.trim() || undefined } : {}),
      });
      setSent(true);
      setOpen(false);
    } catch (e) {
      setError(e instanceof HttpError ? e.message : "Không gửi được phản hồi.");
    } finally {
      setBusy(false);
    }
  };

  if (sent) {
    return <p className="text-xs text-green-700 dark:text-green-400">Cảm ơn bạn đã phản hồi.</p>;
  }

  return (
    <div className="relative">
      <div className="flex items-center gap-1">
        <span className="mr-1 text-xs text-slate-500 dark:text-slate-400">Câu trả lời hữu ích?</span>
        <button
          type="button"
          className="btn-ghost h-7 px-2"
          aria-label="Hữu ích"
          disabled={busy}
          onClick={() => send(1)}
        >
          👍
        </button>
        <button
          type="button"
          className="btn-ghost h-7 px-2"
          aria-label="Không hữu ích"
          disabled={busy}
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          👎
        </button>
      </div>
      {error && <p className="mt-1 text-xs text-red-600 dark:text-red-400">{error}</p>}
      {open && (
        <div className="card absolute left-0 top-9 z-20 w-72 space-y-2">
          <label className="label" htmlFor={`fb-reason-${messageId}`}>
            Lý do
          </label>
          <select
            id={`fb-reason-${messageId}`}
            className="input"
            value={reason}
            onChange={(e) => setReason(e.target.value as FeedbackReason)}
          >
            {REASONS.map((r) => (
              <option key={r.value} value={r.value}>
                {r.label}
              </option>
            ))}
          </select>
          <label className="label" htmlFor={`fb-comment-${messageId}`}>
            Góp ý thêm (không bắt buộc)
          </label>
          <textarea
            id={`fb-comment-${messageId}`}
            className="input"
            rows={3}
            value={comment}
            onChange={(e) => setComment(e.target.value)}
          />
          <div className="flex justify-end gap-2">
            <button type="button" className="btn-ghost" onClick={() => setOpen(false)}>
              Đóng
            </button>
            <button type="button" className="btn-primary" disabled={busy} onClick={() => send(-1)}>
              Gửi phản hồi
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
