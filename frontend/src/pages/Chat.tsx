import React, { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ApiError, chat as chatApi, getReport, mediaUrl } from "../api/client";
import type { AnswerKind, ChatMessage, Citation, MediaRef, Report } from "../api/types";
import { useOverlays } from "../components/overlays/OverlaysContext";

/* ==========================================================================
   Chat — ports the '<!-- ===== CHAT ===== -->' screen from
   .design_extracted/DeepVision.dc.html: a two-column layout (message thread
   + composer on the left, suggested questions / grounding note on the
   right) wired to real POST /api/chat calls. The backend returns the full
   answer in one response; we simulate the prototype's word-by-word
   "streaming" reveal client-side so the UX matches exactly.
   ========================================================================== */

const SUGGESTED_QUESTIONS = [
  "What is the main contribution of this paper?",
  "What method or approach do the authors use?",
  "What are the key results?",
  "What are the limitations of this work?",
];

/** Shown as clickable chips under the welcome message while the conversation
 *  is still empty — deliberately spans the range of what the assistant can
 *  do (a deterministic citation lookup, not just content Q&A) so a first-time
 *  user sees the point of it immediately. */
const STARTER_PROMPTS = [
  "Cite this in APA",
  "What problem does this paper solve?",
  "Explain the method step by step",
  "What are the limitations?",
];

/** A message as rendered in the thread. Extends the wire ChatMessage shape
 *  with a few UI-only fields for the simulated streaming reveal. */
interface UiMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  citations: Citation[];
  figures: MediaRef[];
  /** How this answer was produced (see AnswerKind). Null/undefined for a user
   *  message, the synthetic welcome, or while still streaming — the
   *  citation-block rendering only kicks in once the real value is known. */
  answerKind?: AnswerKind | null;
  /** true while the word-by-word reveal animation is still running (or
   *  while waiting on the network) — citations/figures stay hidden until
   *  it finishes, matching the prototype. */
  streaming: boolean;
  /** true if this assistant slot ended in a request error. */
  errored: boolean;
  /** true for the client-generated welcome message — excluded from the
   *  history sent to the backend since it was never actually answered. */
  synthetic: boolean;
  /** the user question that produced this slot, kept so a failed request
   *  can be retried without retyping. */
  retryText?: string;
}

function welcomeMessage(report: Report): UiMessage {
  const pages = report.stats.pages;
  const figures = report.stats.figures;
  return {
    id: "welcome",
    role: "assistant",
    text: `Ask me anything about this paper — I'll answer only from its ${pages} page${pages === 1 ? "" : "s"} and ${figures} figure${figures === 1 ? "" : "s"}, and cite the exact source for every claim.`,
    citations: [],
    figures: [],
    streaming: false,
    errored: false,
    synthetic: true,
  };
}

function BlinkCursor() {
  return (
    <span
      style={{
        display: "inline-block",
        width: 7,
        height: 15,
        background: "var(--accent)",
        marginLeft: 2,
        verticalAlign: "text-bottom",
        animation: "dv-blink 1s step-end infinite",
      }}
    />
  );
}

function AssistantAvatar() {
  return (
    <div
      style={{
        flex: "none",
        width: 28,
        height: 28,
        borderRadius: 8,
        background: "var(--accent)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        marginTop: 1,
      }}
    >
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--accent-fg)" strokeWidth={2}>
        <circle cx="11" cy="11" r="6" />
        <line x1="15.5" y1="15.5" x2="21" y2="21" strokeLinecap="round" />
      </svg>
    </div>
  );
}

function FigureThumb({ fig, onOpen }: { fig: MediaRef; onOpen: () => void }) {
  const [imgFailed, setImgFailed] = useState(false);
  const src = fig.chunk_id ? mediaUrl(fig.chunk_id) : null;
  const showImg = !!src && !imgFailed;

  return (
    <div
      onClick={onOpen}
      style={{
        cursor: "zoom-in",
        border: "1px solid var(--border)",
        borderRadius: 8,
        overflow: "hidden",
        width: 120,
        flex: "none",
      }}
    >
      <div
        style={{
          height: 64,
          backgroundImage: showImg
            ? undefined
            : "repeating-linear-gradient(135deg,var(--surface-2) 0 10px,var(--surface-3) 10px 20px)",
        }}
      >
        {showImg ? (
          <img
            src={src as string}
            alt={fig.label}
            onError={() => setImgFailed(true)}
            style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
          />
        ) : null}
      </div>
      <div className="dv-mono" style={{ fontSize: 9.5, color: "var(--fg-subtle)", padding: "4px 6px" }}>
        {fig.label}
      </div>
    </div>
  );
}

/** One `**Label** — value` line out of a `citation`-kind answer. Falls back to
 *  treating the whole line as the value (no copy label) if it doesn't match
 *  the expected shape, so an unexpected format still renders instead of
 *  vanishing. */
function parseCitationLines(text: string): { label: string; value: string }[] {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const m = line.match(/^\*\*(.+?)\*\*\s*[—-]\s*(.+)$/);
      return m ? { label: m[1], value: m[2] } : { label: "", value: line };
    });
}

/** Renders a `citation`-kind answer as copyable, labelled blocks instead of
 *  prose — the point is a citation the user can paste straight into a
 *  reference manager, not a sentence to read. */
function CitationBlocks({ text }: { text: string }) {
  const lines = parseCitationLines(text);
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);

  const copy = (value: string, idx: number) => {
    navigator.clipboard?.writeText(value).then(() => {
      setCopiedIdx(idx);
      setTimeout(() => setCopiedIdx((cur) => (cur === idx ? null : cur)), 1500);
    });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {lines.map((l, idx) =>
        // A line with no `**Label** —` is the answer's lead sentence, not a
        // citation. Boxing it and offering to copy it implied you could paste
        // "Here is this paper cited in the style you asked for:" into a
        // reference manager.
        !l.label ? (
          <div key={idx} style={{ color: "var(--fg)" }}>
            {l.value}
          </div>
        ) : (
        <div key={idx} style={{ border: "1px solid var(--border)", borderRadius: 9, overflow: "hidden" }}>
          {l.label && (
            <div
              className="dv-mono"
              style={{
                fontSize: 10.5,
                fontWeight: 600,
                letterSpacing: ".04em",
                color: "var(--fg-subtle)",
                padding: "6px 11px",
                borderBottom: "1px solid var(--border)",
                background: "var(--surface-2)",
              }}
            >
              {l.label.toUpperCase()}
            </div>
          )}
          <div style={{ display: "flex", alignItems: "flex-start", gap: 8, padding: "10px 11px" }}>
            <div
              className="dv-mono"
              style={{ flex: 1, fontSize: 12.5, lineHeight: 1.55, color: "var(--fg)", wordBreak: "break-word" }}
            >
              {l.value}
            </div>
            <button
              onClick={() => copy(l.value, idx)}
              style={{
                flex: "none",
                fontSize: 11,
                fontWeight: 600,
                color: copiedIdx === idx ? "var(--green)" : "var(--accent)",
                background: copiedIdx === idx ? "var(--green-soft)" : "var(--accent-soft)",
                border: `1px solid ${copiedIdx === idx ? "var(--green)" : "var(--accent-border)"}`,
                borderRadius: 7,
                padding: "4px 9px",
                cursor: "pointer",
              }}
            >
              {copiedIdx === idx ? "Copied" : "Copy"}
            </button>
          </div>
        </div>
        )
      )}
    </div>
  );
}

function Spinner() {
  return (
    <div
      style={{
        width: 22,
        height: 22,
        borderRadius: "50%",
        border: "2px solid var(--border)",
        borderTopColor: "var(--accent)",
        animation: "dv-spin .8s linear infinite",
      }}
    />
  );
}

function CenteredPanel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: 40 }}>
      <div style={{ maxWidth: 360, textAlign: "center" }}>{children}</div>
    </div>
  );
}

export default function ChatPage() {
  const { paperId } = useParams<{ paperId?: string }>();
  const navigate = useNavigate();
  const { openLightbox, openCitation } = useOverlays();

  const [report, setReport] = useState<Report | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState<string | null>(null);

  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const revealTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearRevealTimer = useCallback(() => {
    if (revealTimerRef.current) {
      clearInterval(revealTimerRef.current);
      revealTimerRef.current = null;
    }
  }, []);

  const loadReport = useCallback(() => {
    if (!paperId) return;
    localStorage.setItem("dv-last-paper", paperId);
    let cancelled = false;
    setReportLoading(true);
    setReportError(null);
    getReport(paperId)
      .then((r) => {
        if (cancelled) return;
        setReport(r);
        setMessages([welcomeMessage(r)]);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setReport(null);
        setReportError(err instanceof ApiError ? err.message : "Failed to load this paper.");
      })
      .finally(() => {
        if (!cancelled) setReportLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [paperId]);

  useEffect(() => {
    clearRevealTimer();
    setBusy(false);
    setSessionId(null);
    setInput("");
    setMessages([]);
    setReport(null);
    setReportError(null);
    if (!paperId) {
      setReportLoading(false);
      return;
    }
    const cancel = loadReport();
    return () => {
      cancel?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paperId]);

  useEffect(() => clearRevealTimer, [clearRevealTimer]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const revealAssistantMessage = useCallback(
    (placeholderId: string, full: ChatMessage) => {
      const words = full.text.length ? full.text.split(" ") : [""];
      let i = 0;
      clearRevealTimer();
      revealTimerRef.current = setInterval(() => {
        i += 1 + Math.floor(Math.random() * 2);
        const done = i >= words.length;
        const partial = words.slice(0, Math.min(i, words.length)).join(" ");
        setMessages((prev) =>
          prev.map((m) =>
            m.id === placeholderId
              ? {
                  ...m,
                  text: partial,
                  streaming: !done,
                  citations: done ? full.citations : [],
                  figures: done ? full.figures : [],
                  answerKind: done ? full.answer_kind : null,
                }
              : m
          )
        );
        if (done) {
          clearRevealTimer();
          setBusy(false);
        }
      }, 55);
    },
    [clearRevealTimer]
  );

  const sendMessage = useCallback(
    (preset?: string) => {
      const text = (preset ?? input).trim();
      if (!text || busy || !paperId || !report) return;

      const userId = `u-${Date.now()}`;
      const placeholderId = `a-${Date.now() + 1}`;

      const historyForApi: ChatMessage[] = messages
        .filter((m) => !m.synthetic && !m.errored)
        .map((m) => ({
          id: m.id,
          paper_id: paperId,
          role: m.role,
          text: m.text,
          citations: m.citations,
          figures: m.figures,
          created_at: new Date().toISOString(),
        }));

      setMessages((prev) => [
        ...prev,
        { id: userId, role: "user", text, citations: [], figures: [], streaming: false, errored: false, synthetic: false },
        { id: placeholderId, role: "assistant", text: "", citations: [], figures: [], streaming: true, errored: false, synthetic: false, retryText: text },
      ]);
      setInput("");
      setBusy(true);

      chatApi({
        paper_id: paperId,
        message: text,
        session_id: sessionId ?? undefined,
        history: historyForApi,
      })
        .then((res) => {
          setSessionId(res.session_id);
          revealAssistantMessage(placeholderId, res.message);
        })
        .catch((err: unknown) => {
          const msg = err instanceof ApiError ? err.message : "Something went wrong reaching the assistant.";
          setMessages((prev) =>
            prev.map((m) => (m.id === placeholderId ? { ...m, text: msg, streaming: false, errored: true } : m))
          );
          setBusy(false);
        });
    },
    [input, busy, paperId, report, messages, sessionId, revealAssistantMessage]
  );

  const onComposerKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // ---- no paper selected --------------------------------------------------
  if (!paperId) {
    return (
      <div style={{ display: "flex", height: "100%", minHeight: 0 }}>
        <CenteredPanel>
          <div style={{ fontSize: 15, fontWeight: 600 }}>No paper selected</div>
          <div style={{ fontSize: 13, color: "var(--fg-subtle)", marginTop: 8, lineHeight: 1.6 }}>
            Choose a paper from your library to start a grounded conversation about it.
          </div>
          <button
            onClick={() => navigate("/library")}
            style={{
              marginTop: 18,
              fontSize: 13.5,
              fontWeight: 600,
              color: "var(--accent-fg)",
              background: "var(--accent)",
              border: "none",
              borderRadius: 9,
              padding: "10px 20px",
            }}
          >
            Go to Library
          </button>
        </CenteredPanel>
      </div>
    );
  }

  // ---- loading paper -------------------------------------------------------
  if (reportLoading) {
    return (
      <div style={{ display: "flex", height: "100%", minHeight: 0 }}>
        <CenteredPanel>
          <div style={{ display: "flex", justifyContent: "center" }}>
            <Spinner />
          </div>
          <div style={{ fontSize: 13, color: "var(--fg-subtle)", marginTop: 14 }}>Loading paper…</div>
        </CenteredPanel>
      </div>
    );
  }

  // ---- error loading paper --------------------------------------------------
  if (reportError || !report) {
    return (
      <div style={{ display: "flex", height: "100%", minHeight: 0 }}>
        <CenteredPanel>
          <div style={{ fontSize: 15, fontWeight: 600, color: "var(--red)" }}>Couldn&apos;t load this paper</div>
          <div style={{ fontSize: 13, color: "var(--fg-subtle)", marginTop: 8, lineHeight: 1.6 }}>
            {reportError ?? "This paper could not be found."}
          </div>
          <div style={{ display: "flex", gap: 10, justifyContent: "center", marginTop: 18 }}>
            <button
              onClick={() => loadReport()}
              style={{
                fontSize: 13.5,
                fontWeight: 600,
                color: "var(--accent-fg)",
                background: "var(--accent)",
                border: "none",
                borderRadius: 9,
                padding: "10px 18px",
              }}
            >
              Retry
            </button>
            <button
              onClick={() => navigate("/library")}
              style={{
                fontSize: 13.5,
                fontWeight: 500,
                color: "var(--fg-muted)",
                background: "transparent",
                border: "1px solid var(--border)",
                borderRadius: 9,
                padding: "10px 18px",
              }}
            >
              Back to Library
            </button>
          </div>
        </CenteredPanel>
      </div>
    );
  }

  const paper = report.paper;
  const title = paper?.title || `Paper ${paperId}`;
  const canSend = !busy && input.trim().length > 0;

  return (
    <div style={{ display: "flex", height: "100%", minHeight: 0 }}>
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
        <div style={{ flex: "none", padding: "14px 24px", borderBottom: "1px solid var(--border)", background: "var(--surface)" }}>
          <div style={{ fontSize: 13.5, fontWeight: 600 }}>{title}</div>
          <div className="dv-mono" style={{ fontSize: 11, color: "var(--fg-subtle)", marginTop: 2 }}>
            Grounded in {report.stats.pages} pages, {report.stats.figures} figures
          </div>
        </div>

        <div ref={scrollRef} className="dv-scroll" style={{ flex: 1, overflowY: "auto", padding: 24, display: "flex", flexDirection: "column", gap: 18 }}>
          {messages.length === 0 ? (
            <div style={{ margin: "auto", fontSize: 13, color: "var(--fg-subtle)" }}>No messages yet.</div>
          ) : (
            messages.map((m) => {
              const isA = m.role === "assistant";
              const rowStyle: React.CSSProperties = {
                display: "flex",
                gap: 11,
                maxWidth: 760,
                alignSelf: isA ? "flex-start" : "flex-end",
                flexDirection: isA ? "row" : "row-reverse",
              };
              const bubbleStyle: React.CSSProperties = isA
                ? {
                    background: "var(--surface)",
                    border: m.errored ? "1px solid var(--red)" : "1px solid var(--border)",
                    borderRadius: "4px 13px 13px 13px",
                    padding: "13px 16px",
                    color: "var(--fg)",
                  }
                : {
                    background: "var(--accent)",
                    borderRadius: "13px 4px 13px 13px",
                    padding: "11px 15px",
                    color: "var(--accent-fg)",
                  };
              const hasCites = isA && !m.streaming && m.citations.length > 0;
              const hasFigs = isA && !m.streaming && m.figures.length > 0;
              const isCitationAnswer = isA && !m.streaming && !m.errored && m.answerKind === "citation";

              return (
                <div key={m.id} style={rowStyle}>
                  {isA && <AssistantAvatar />}
                  <div style={bubbleStyle}>
                    {isCitationAnswer ? (
                      <CitationBlocks text={m.text} />
                    ) : (
                      <span style={{ fontSize: 14.5, lineHeight: 1.68, whiteSpace: "pre-wrap" }}>{m.text}</span>
                    )}
                    {m.streaming && <BlinkCursor />}
                    {m.errored && (
                      <div style={{ marginTop: 8 }}>
                        <button
                          onClick={() => sendMessage(m.retryText)}
                          style={{
                            fontSize: 12,
                            fontWeight: 600,
                            color: "var(--red)",
                            background: "var(--red-soft)",
                            border: "none",
                            borderRadius: 7,
                            padding: "5px 10px",
                          }}
                        >
                          Try again
                        </button>
                      </div>
                    )}
                    {hasFigs && (
                      <div style={{ display: "flex", gap: 8, marginTop: 11, flexWrap: "wrap" }}>
                        {m.figures.map((f) => (
                          <FigureThumb key={f.id} fig={f} onOpen={() => openLightbox(f)} />
                        ))}
                      </div>
                    )}
                    {hasCites && (
                      <div
                        style={{
                          display: "flex",
                          flexWrap: "wrap",
                          gap: 6,
                          marginTop: 11,
                          paddingTop: 11,
                          borderTop: "1px solid var(--border)",
                        }}
                      >
                        {m.citations.map((c) => (
                          <button
                            key={c.id}
                            onClick={() => openCitation(c)}
                            style={{
                              display: "inline-flex",
                              alignItems: "center",
                              gap: 5,
                              fontFamily: "'IBM Plex Mono'",
                              fontSize: 11,
                              fontWeight: 500,
                              color: "var(--accent)",
                              background: "var(--accent-soft)",
                              border: "1px solid var(--accent-border)",
                              borderRadius: 20,
                              padding: "3px 10px",
                            }}
                          >
                            [{c.marker}] {c.source}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}
          {messages.length === 1 && messages[0].synthetic && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginLeft: 39, maxWidth: 760 }}>
              {STARTER_PROMPTS.map((p) => (
                <button
                  key={p}
                  onClick={() => sendMessage(p)}
                  disabled={busy}
                  style={{
                    fontSize: 12.5,
                    fontWeight: 500,
                    color: "var(--accent)",
                    background: "var(--accent-soft)",
                    border: "1px solid var(--accent-border)",
                    borderRadius: 20,
                    padding: "7px 13px",
                    opacity: busy ? 0.6 : 1,
                    cursor: busy ? "not-allowed" : "pointer",
                  }}
                >
                  {p}
                </button>
              ))}
            </div>
          )}
        </div>

        <div style={{ flex: "none", padding: "16px 24px", borderTop: "1px solid var(--border)", background: "var(--surface)" }}>
          <div style={{ display: "flex", gap: 10, alignItems: "flex-end", maxWidth: 820, margin: "0 auto" }}>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onComposerKeyDown}
              rows={1}
              placeholder="Ask about methods, results, limitations…"
              style={{
                flex: 1,
                resize: "none",
                padding: "12px 14px",
                fontSize: 14,
                lineHeight: 1.4,
                border: "1px solid var(--border-strong)",
                borderRadius: 11,
                background: "var(--surface)",
                color: "var(--fg)",
                outline: "none",
                maxHeight: 120,
              }}
            />
            <button
              onClick={() => sendMessage()}
              disabled={!canSend}
              style={{
                flex: "none",
                height: 44,
                padding: "0 20px",
                fontSize: 14,
                fontWeight: 600,
                color: "var(--accent-fg)",
                background: "var(--accent)",
                border: "none",
                borderRadius: 11,
                opacity: canSend ? 1 : 0.5,
                cursor: canSend ? "pointer" : "not-allowed",
              }}
            >
              Send
            </button>
          </div>
        </div>
      </div>

      <aside style={{ width: 270, flex: "none", borderLeft: "1px solid var(--border)", background: "var(--surface)", padding: "20px 18px", overflowY: "auto" }}>
        <div className="dv-mono" style={{ fontSize: 11, color: "var(--fg-subtle)", letterSpacing: ".04em" }}>
          SUGGESTED QUESTIONS
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 9, marginTop: 14 }}>
          {SUGGESTED_QUESTIONS.map((q) => (
            <button
              key={q}
              onClick={() => sendMessage(q)}
              disabled={busy}
              style={{
                textAlign: "left",
                fontSize: 13,
                lineHeight: 1.5,
                color: "var(--fg)",
                background: "var(--surface-2)",
                border: "1px solid var(--border)",
                borderRadius: 10,
                padding: "11px 13px",
                opacity: busy ? 0.6 : 1,
                cursor: busy ? "not-allowed" : "pointer",
              }}
            >
              {q}
            </button>
          ))}
        </div>
        <div className="dv-mono" style={{ fontSize: 11, color: "var(--fg-subtle)", letterSpacing: ".04em", marginTop: 26 }}>
          GROUNDING
        </div>
        <div style={{ fontSize: 12.5, lineHeight: 1.6, color: "var(--fg-muted)", marginTop: 10 }}>
          Content answers are grounded in this paper only, and every claim links back to the source page or figure it
          came from. It can also give you the paper&apos;s citation in APA, MLA, Chicago, IEEE, or BibTeX, and answer
          questions about the document itself — authors, page count, figures.
        </div>
      </aside>
    </div>
  );
}
