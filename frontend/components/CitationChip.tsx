"use client";

import type { CitationState } from "@/lib/types";

const STYLES: Record<CitationState, string> = {
  verified:
    "bg-green-100 text-green-800 border-green-300 hover:bg-green-200 dark:bg-green-900/40 dark:text-green-300 dark:border-green-700",
  weak: "bg-yellow-100 text-yellow-800 border-yellow-300 hover:bg-yellow-200 dark:bg-yellow-900/40 dark:text-yellow-300 dark:border-yellow-700",
  invalid:
    "bg-red-100 text-red-700 border-red-300 line-through hover:bg-red-200 dark:bg-red-900/40 dark:text-red-300 dark:border-red-700",
  neutral:
    "bg-slate-200 text-slate-700 border-slate-300 hover:bg-slate-300 dark:bg-slate-700 dark:text-slate-200 dark:border-slate-600",
};

const TITLES: Record<CitationState, string> = {
  verified: "Trích dẫn đã được xác minh",
  weak: "Trích dẫn khớp một phần với nguồn",
  invalid: "Trích dẫn không tồn tại, đã bị loại khỏi câu trả lời",
  neutral: "Nguồn trích dẫn",
};

export default function CitationChip({
  idx,
  state,
  onClick,
}: {
  idx: number;
  state: CitationState;
  onClick?: (idx: number) => void;
}) {
  return (
    <button
      type="button"
      title={TITLES[state]}
      aria-label={`Nguồn ${idx}: ${TITLES[state]}`}
      onClick={() => onClick?.(idx)}
      className={`mx-0.5 inline-flex h-5 min-w-[1.5rem] items-center justify-center rounded-md border px-1 align-text-bottom text-[11px] font-semibold leading-none ${STYLES[state]}`}
    >
      {idx}
    </button>
  );
}
