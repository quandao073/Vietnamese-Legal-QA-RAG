"use client";

import { Fragment, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import CitationChip from "@/components/CitationChip";
import type { Citations, CitationState } from "./types";

/** Matches `[#3]` or `[3]` citation tokens. */
const CITATION_RE = /\[#?(\d{1,2})\]/g;

export function citationState(idx: number, citations?: Citations | null): CitationState {
  if (!citations) return "neutral";
  if (citations.invalid?.includes(idx)) return "invalid";
  if (citations.weak?.includes(idx)) return "weak";
  if (citations.verified?.includes(idx)) return "verified";
  return "neutral";
}

interface Ctx {
  citations?: Citations | null;
  onCite?: (idx: number) => void;
}

function renderText(text: string, ctx: Ctx, keyPrefix: string): ReactNode {
  if (!CITATION_RE.test(text)) return text;
  CITATION_RE.lastIndex = 0;
  const out: ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = CITATION_RE.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const idx = Number(m[1]);
    out.push(
      <CitationChip
        key={`${keyPrefix}-${i++}`}
        idx={idx}
        state={citationState(idx, ctx.citations)}
        onClick={ctx.onCite}
      />
    );
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return <Fragment>{out}</Fragment>;
}

/** Recursively replace citation tokens inside the children of a markdown element. */
function withCitations(children: ReactNode, ctx: Ctx, keyPrefix = "c"): ReactNode {
  if (typeof children === "string") return renderText(children, ctx, keyPrefix);
  if (Array.isArray(children)) {
    return children.map((child, i) =>
      typeof child === "string" ? (
        <Fragment key={`${keyPrefix}-${i}`}>{renderText(child, ctx, `${keyPrefix}-${i}`)}</Fragment>
      ) : (
        child
      )
    );
  }
  return children;
}

function makeComponents(ctx: Ctx): Components {
  const wrap =
    <T extends keyof JSX.IntrinsicElements>(Tag: T) =>
    // eslint-disable-next-line react/display-name
    ({ children, node: _node, ...rest }: { children?: ReactNode; node?: unknown }) => {
      const Comp = Tag as unknown as React.ElementType;
      return <Comp {...rest}>{withCitations(children, ctx)}</Comp>;
    };
  return {
    p: wrap("p"),
    li: wrap("li"),
    td: wrap("td"),
    th: wrap("th"),
    strong: wrap("strong"),
    em: wrap("em"),
    h1: wrap("h1"),
    h2: wrap("h2"),
    h3: wrap("h3"),
    h4: wrap("h4"),
    blockquote: wrap("blockquote"),
    a: ({ children, href }) => (
      <a href={href} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    ),
  };
}

export interface MarkdownProps {
  text: string;
  citations?: Citations | null;
  onCite?: (idx: number) => void;
  className?: string;
}

/**
 * Sanitized markdown: react-markdown never renders raw HTML unless rehype-raw
 * is added, which we deliberately do not do.
 */
export function Markdown({ text, citations, onCite, className }: MarkdownProps) {
  return (
    <div className={`md ${className ?? ""}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={makeComponents({ citations, onCite })}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
