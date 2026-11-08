"use client";

import { useState } from "react";
import type { ChatOptions, PublicConfig, SearchMode } from "@/lib/types";

export const MODES: SearchMode[] = ["hybrid", "dense", "bm25"];

export type Settings = Required<Pick<ChatOptions, "top_k" | "use_rerank" | "strategy" | "mode">>;

export default function AdvancedSettings({
  settings,
  config,
  onChange,
}: {
  settings: Settings;
  config: PublicConfig | null;
  onChange: (s: Settings) => void;
}) {
  const [open, setOpen] = useState(false);
  const [min, max] = config?.top_k_range ?? [3, 10];
  const strategies = config?.strategies?.length ? config.strategies : [settings.strategy];
  const modes = config?.modes?.length ? config.modes : MODES;

  return (
    <div className="text-sm">
      <button
        type="button"
        className="text-xs text-slate-500 underline-offset-2 hover:underline dark:text-slate-400"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        {open ? "Ẩn cài đặt nâng cao" : "Cài đặt nâng cao"}
      </button>
      {open && (
        <div className="mt-2 grid gap-3 rounded-lg border border-slate-200 bg-white p-3 sm:grid-cols-2 dark:border-slate-800 dark:bg-slate-900">
          <div>
            <label className="label" htmlFor="adv-topk">
              Số nguồn tối đa (top_k): <b>{settings.top_k}</b>
            </label>
            <input
              id="adv-topk"
              type="range"
              min={min}
              max={max}
              step={1}
              value={settings.top_k}
              onChange={(e) => onChange({ ...settings, top_k: Number(e.target.value) })}
              className="w-full"
            />
          </div>
          <div className="flex items-end">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={settings.use_rerank}
                onChange={(e) => onChange({ ...settings, use_rerank: e.target.checked })}
              />
              Xếp hạng lại nguồn (rerank)
            </label>
          </div>
          <div>
            <label className="label" htmlFor="adv-strategy">
              Chiến lược chia đoạn
            </label>
            <select
              id="adv-strategy"
              className="input"
              value={settings.strategy}
              onChange={(e) => onChange({ ...settings, strategy: e.target.value })}
            >
              {strategies.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="label" htmlFor="adv-mode">
              Chế độ tìm kiếm
            </label>
            <select
              id="adv-mode"
              className="input"
              value={settings.mode}
              onChange={(e) => onChange({ ...settings, mode: e.target.value as SearchMode })}
            >
              {modes.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
    </div>
  );
}
