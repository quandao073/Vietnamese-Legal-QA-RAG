"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getPublicConfig, streamChat } from "@/lib/api";
import type { ChatMessage, HistoryTurn, PublicConfig, Stage } from "@/lib/types";
import AdvancedSettings, { type Settings } from "./AdvancedSettings";
import MessageBubble from "./MessageBubble";
import SampleQuestions from "./SampleQuestions";
import SourcePanel from "./SourcePanel";

const SESSION_KEY = "vnlegal.session_id";
const MESSAGES_KEY = "vnlegal.messages";
const SETTINGS_KEY = "vnlegal.settings";
const HISTORY_TURNS = 4;

const DEFAULT_SETTINGS: Settings = {
  top_k: 5,
  use_rerank: true,
  strategy: "article",
  mode: "hybrid",
};

function uuid(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`;
}

function loadJson<T>(storage: Storage, key: string, fallback: T): T {
  try {
    const raw = storage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

export default function ChatWindow() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState("");
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [config, setConfig] = useState<PublicConfig | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState<Stage | null>(null);
  const [panel, setPanel] = useState<{ messageId: string; idx: number } | null>(null);
  const [hydrated, setHydrated] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Hydrate from storage on the client.
  useEffect(() => {
    let sid = sessionStorage.getItem(SESSION_KEY);
    if (!sid) {
      sid = uuid();
      sessionStorage.setItem(SESSION_KEY, sid);
    }
    setSessionId(sid);
    const saved = loadJson<ChatMessage[]>(sessionStorage, MESSAGES_KEY, []);
    // A reload mid-stream leaves a dangling streaming bubble; drop it.
    setMessages(saved.filter((m) => !m.streaming));
    const savedSettings = loadJson<Partial<Settings> | null>(localStorage, SETTINGS_KEY, null);
    getPublicConfig()
      .then((cfg) => {
        setConfig(cfg);
        setSettings({
          top_k: cfg.default_top_k,
          use_rerank: cfg.rerank_enabled,
          strategy: cfg.default_strategy,
          mode: "hybrid",
          ...(savedSettings ?? {}),
        });
      })
      .catch(() => {
        if (savedSettings) setSettings({ ...DEFAULT_SETTINGS, ...savedSettings });
      });
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    try {
      sessionStorage.setItem(MESSAGES_KEY, JSON.stringify(messages));
    } catch {
      /* storage full or unavailable: ignore */
    }
  }, [messages, hydrated]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, stage]);

  const updateSettings = (s: Settings) => {
    setSettings(s);
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
    } catch {
      /* ignore */
    }
  };

  const patchMessage = useCallback((id: string, patch: Partial<ChatMessage>) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
  }, []);

  const send = useCallback(
    async (question: string) => {
      const q = question.trim();
      if (!q || busy) return;

      const history: HistoryTurn[] = messages
        .filter((m) => !m.error && m.content)
        .slice(-HISTORY_TURNS)
        .map((m) => ({ role: m.role, content: m.content }));

      const userMsg: ChatMessage = { id: uuid(), role: "user", content: q };
      const assistantId = uuid();
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        streaming: true,
        question: q,
      };
      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setInput("");
      setBusy(true);
      setStage("understanding");

      const controller = new AbortController();
      abortRef.current = controller;
      let text = "";

      await streamChat(
        {
          question: q,
          session_id: sessionId || undefined,
          history,
          options: {
            top_k: settings.top_k,
            use_rerank: settings.use_rerank,
            strategy: settings.strategy,
            mode: settings.mode,
          },
        },
        {
          onStatus: (s) => setStage(s),
          onSources: (sources) => patchMessage(assistantId, { sources }),
          onToken: (t) => {
            text += t;
            patchMessage(assistantId, { content: text });
          },
          onCitations: (citations) => patchMessage(assistantId, { citations }),
          onDone: (done) =>
            patchMessage(assistantId, {
              content: done.answer_md || text,
              messageId: done.message_id,
              sources: done.sources,
              citations: done.citations,
              grounded: done.grounded,
              streaming: false,
            }),
          onError: (error) =>
            patchMessage(assistantId, {
              error,
              streaming: false,
              messageId: error.message_id,
            }),
        },
        controller.signal
      );

      setStage(null);
      setBusy(false);
      abortRef.current = null;
    },
    [busy, messages, patchMessage, sessionId, settings]
  );

  const retry = (question: string) => {
    // Drop the failed assistant bubble and its question, then resend.
    setMessages((prev) => {
      const idx = prev.findLastIndex((m) => m.error && m.question === question);
      if (idx < 0) return prev;
      const start = idx > 0 && prev[idx - 1].role === "user" ? idx - 1 : idx;
      return [...prev.slice(0, start), ...prev.slice(idx + 1)];
    });
    // Let state settle before sending.
    setTimeout(() => send(question), 0);
  };

  const newConversation = () => {
    abortRef.current?.abort();
    setMessages([]);
    setPanel(null);
    setStage(null);
    setBusy(false);
    const sid = uuid();
    sessionStorage.setItem(SESSION_KEY, sid);
    setSessionId(sid);
  };

  const panelMessage = panel ? messages.find((m) => m.id === panel.messageId) : undefined;

  return (
    <div className="flex h-[calc(100vh-3.5rem)] flex-col md:flex-row">
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-2 dark:border-slate-800">
          <AdvancedSettings settings={settings} config={config} onChange={updateSettings} />
          <button type="button" className="btn-ghost h-8" onClick={newConversation}>
            Cuộc trò chuyện mới
          </button>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
          {hydrated && messages.length === 0 && (
            <SampleQuestions onPick={send} disabled={busy} />
          )}
          {messages.map((m) => (
            <MessageBubble
              key={m.id}
              message={m}
              stage={m.streaming ? stage : null}
              onCite={(messageId, idx) => setPanel({ messageId, idx })}
              onRetry={retry}
            />
          ))}
          <div ref={bottomRef} />
        </div>

        <form
          className="border-t border-slate-200 p-3 dark:border-slate-800"
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <div className="flex items-end gap-2">
            <textarea
              className="input max-h-40 min-h-[2.75rem] flex-1 resize-y"
              rows={2}
              placeholder="Nhập câu hỏi pháp luật của bạn… (Enter để gửi, Shift+Enter xuống dòng)"
              value={input}
              maxLength={config?.max_question_chars}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send(input);
                }
              }}
              disabled={busy}
              aria-label="Câu hỏi"
            />
            <button type="submit" className="btn-primary h-11 px-4" disabled={busy || !input.trim()}>
              {busy ? "Đang trả lời…" : "Gửi"}
            </button>
          </div>
        </form>
      </div>

      {panel && panelMessage && (
        <SourcePanel
          sources={panelMessage.sources ?? []}
          citations={panelMessage.citations}
          activeIdx={panel.idx}
          onActive={(idx) => setPanel({ ...panel, idx })}
          onClose={() => setPanel(null)}
        />
      )}
    </div>
  );
}
