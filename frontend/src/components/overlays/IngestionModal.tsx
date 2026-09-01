import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { pollJob } from "../../api/client";
import { REPORT_STAGE, STAGE_ORDER } from "../../api/types";
import type { IngestJob, StageProgress, StageStatus } from "../../api/types";
import { useOverlays } from "./OverlaysContext";

function stageVisual(status: StageStatus, index: number) {
  const base =
    "flex:none;width:22px;height:22px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;";
  if (status === "done") {
    return {
      iconWrap: base + "background:var(--green-soft);color:var(--green)",
      icon: "✓",
      statusText: "done",
      statusStyle: "color:var(--green)",
      labelStyle: "font-size:13.5px;color:var(--fg);font-weight:500",
    };
  }
  if (status === "running") {
    return {
      iconWrap:
        base + "background:var(--accent-soft);color:var(--accent);border:2px solid var(--accent)",
      icon: (
        <span
          style={{
            display: "inline-block",
            width: 10,
            height: 10,
            borderRadius: "50%",
            border: "2px solid var(--accent)",
            borderTopColor: "transparent",
            animation: "dv-spin .7s linear infinite",
          }}
        />
      ),
      statusText: "running",
      statusStyle: "color:var(--accent);font-weight:600",
      labelStyle: "font-size:13.5px;color:var(--fg);font-weight:600",
    };
  }
  if (status === "error") {
    return {
      iconWrap: base + "background:var(--red-soft);color:var(--red)",
      icon: "!",
      statusText: "error",
      statusStyle: "color:var(--red);font-weight:600",
      labelStyle: "font-size:13.5px;color:var(--fg);font-weight:500",
    };
  }
  return {
    iconWrap: base + "background:var(--surface-2);color:var(--fg-subtle);border:1px solid var(--border)",
    icon: String(index + 1),
    statusText: "pending",
    statusStyle: "color:var(--fg-subtle)",
    labelStyle: "font-size:13.5px;color:var(--fg-subtle)",
  };
}

export function IngestionModal() {
  const { ingest, closeIngest } = useOverlays();
  const navigate = useNavigate();
  const [job, setJob] = useState<IngestJob | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setJob(null);
    if (!ingest) return;
    const controller = new AbortController();
    pollJob(ingest.jobId, {
      signal: controller.signal,
      onUpdate: (j) => setJob(j),
    }).catch(() => {
      // aborted (modal closed) — nothing to do
    });
    return () => controller.abort();
  }, [ingest?.jobId]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [job?.log.length]);

  if (!ingest) return null;

  const state = job?.state ?? "queued";
  const done = state === "done";
  const errored = state === "error";
  const running = !done && !errored;
  const percent = job?.percent ?? 0;
  // Before the first poll lands there is no job: show the full pipeline as
  // pending. Once a job exists render exactly the stages it carries — a job row
  // persisted before "Generate report" was added to STAGE_ORDER has only six,
  // and padding it out of STAGE_ORDER would leave a phantom stage stuck on
  // "pending" forever.
  const stages: StageProgress[] =
    job?.stages ?? STAGE_ORDER.map((stage) => ({ stage, status: "pending" as StageStatus }));
  const log = job?.log ?? [];

  // "Open report →" must only appear once the report genuinely exists.
  // Report generation is the last tracked stage, so gate on that stage rather
  // than on `state === "done"` — the job also finishes `done` when only the
  // report failed (the paper stays usable for chat/compare).
  // Legacy jobs (no report stage at all) predate the tracked stage and were
  // generated eagerly, so they keep the old behaviour: done ⇒ offer the report.
  const reportStage = stages.find((s) => s.stage === REPORT_STAGE);
  const reportFailed = reportStage?.status === "error";
  const reportReady = done && (reportStage ? reportStage.status === "done" : true);
  // A `done` report stage is NOT proof the model wrote it: an LLM call that
  // exceeds the local model's timeout is swallowed by the agents' draft
  // fallback, so the report is assembled from extractive drafts and every layer
  // above reports success. On the measured local model that is the common case,
  // not the edge — so say so rather than green-checking it as a clean run.
  const reportDegraded = reportReady && Boolean(reportStage?.degraded);

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 60,
        background: "rgba(15,18,28,.5)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
        animation: "dv-fade .15s ease",
      }}
    >
      <div
        style={{
          width: 640,
          maxWidth: "100%",
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: 16,
          boxShadow: "var(--shadow-lg)",
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
          maxHeight: "90vh",
        }}
      >
        <div style={{ padding: "20px 22px", borderBottom: "1px solid var(--border)" }}>
          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16 }}>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontSize: 15, fontWeight: 600 }}>Ingesting paper</div>
              <div
                style={{
                  fontSize: 12.5,
                  color: "var(--fg-muted)",
                  marginTop: 3,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {ingest.paperTitle}
              </div>
            </div>
            <button
              onClick={closeIngest}
              style={{
                flex: "none",
                color: "var(--fg-subtle)",
                background: "var(--surface-2)",
                border: "none",
                borderRadius: 8,
                width: 30,
                height: 30,
                fontSize: 15,
              }}
              aria-label="Close"
            >
              ✕
            </button>
          </div>
          <div style={{ marginTop: 16, display: "flex", alignItems: "center", gap: 12 }}>
            <div
              style={{
                flex: 1,
                height: 8,
                borderRadius: 20,
                background: "var(--surface-3)",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: "100%",
                  borderRadius: 20,
                  background: errored ? "var(--red)" : "var(--accent)",
                  width: `${percent}%`,
                  transition: "width .3s ease",
                }}
              />
            </div>
            <span
              className="dv-mono"
              style={{ fontSize: 12, fontWeight: 600, color: "var(--accent)", width: 38, textAlign: "right" }}
            >
              {percent}%
            </span>
          </div>
        </div>

        <div style={{ padding: "18px 22px", display: "flex", flexDirection: "column", gap: 2 }}>
          {stages.map((s, i) => {
            const v = stageVisual(s.status, i);
            // A degraded `done` stage must not read as a clean green success.
            const statusStyle =
              s.status === "done" && s.degraded
                ? "color:var(--amber);font-weight:600"
                : v.statusStyle;
            return (
              <div key={s.stage} style={{ display: "flex", alignItems: "center", gap: 12, padding: "8px 0" }}>
                <span style={styleFromString(v.iconWrap)}>{v.icon}</span>
                <span style={styleFromString(v.labelStyle)}>{s.stage}</span>
                <span
                  className="dv-mono"
                  style={{ marginLeft: "auto", fontSize: 11, ...styleFromString(statusStyle) }}
                >
                  {s.detail ?? v.statusText}
                </span>
              </div>
            );
          })}
        </div>

        <div
          ref={logRef}
          className="dv-scroll dv-mono"
          style={{
            margin: "0 22px",
            padding: "12px 14px",
            background: "var(--surface-2)",
            border: "1px solid var(--border)",
            borderRadius: 10,
            height: 130,
            overflowY: "auto",
            fontSize: 11.5,
            lineHeight: 1.7,
            color: "var(--fg-muted)",
          }}
        >
          {log.length === 0 && <div style={{ color: "var(--fg-subtle)" }}>waiting for pipeline to start…</div>}
          {log.map((l, i) => (
            <div key={i}>
              <span style={{ color: "var(--fg-subtle)" }}>{l.t}</span> {l.m}
            </div>
          ))}
        </div>

        <div style={{ padding: "18px 22px", display: "flex", gap: 10, justifyContent: "flex-end" }}>
          {errored && (
            <>
              <span style={{ fontSize: 12.5, color: "var(--red)", marginRight: "auto", alignSelf: "center" }}>
                {job?.error_message ?? "Ingestion failed."}
              </span>
              <button
                onClick={closeIngest}
                style={{
                  fontSize: 13.5,
                  fontWeight: 500,
                  color: "var(--fg-muted)",
                  background: "transparent",
                  border: "1px solid var(--border)",
                  borderRadius: 9,
                  padding: "9px 16px",
                }}
              >
                Close
              </button>
            </>
          )}
          {done && (
            <>
              {reportFailed && (
                <span
                  style={{
                    fontSize: 12.5,
                    color: "var(--red)",
                    marginRight: "auto",
                    alignSelf: "center",
                    lineHeight: 1.4,
                  }}
                >
                  Report generation failed
                  {reportStage?.detail ? ` — ${reportStage.detail}` : "."} The paper is
                  ingested: chat still works. Re-run it from the Library to
                  retry the report.
                </span>
              )}
              {reportDegraded && (
                <span
                  style={{
                    fontSize: 12.5,
                    color: "var(--amber)",
                    marginRight: "auto",
                    alignSelf: "center",
                    lineHeight: 1.4,
                  }}
                >
                  Report is degraded{reportStage?.detail ? ` — ${reportStage.detail}` : "."} Those
                  parts fell back to extractive drafts instead of model-written prose. A faster
                  model (Settings) or a re-run gives a full-quality report.
                </span>
              )}
              <button
                onClick={closeIngest}
                style={{
                  fontSize: 13.5,
                  fontWeight: 500,
                  color: "var(--fg-muted)",
                  background: "transparent",
                  border: "1px solid var(--border)",
                  borderRadius: 9,
                  padding: "9px 16px",
                }}
              >
                Close
              </button>
              {reportReady && (
                <button
                  onClick={() => {
                    const paperId = ingest.paperId;
                    closeIngest();
                    navigate(`/report/${paperId}`);
                  }}
                  style={{
                    fontSize: 13.5,
                    fontWeight: 600,
                    color: "var(--accent-fg)",
                    background: "var(--accent)",
                    border: "none",
                    borderRadius: 9,
                    padding: "9px 18px",
                  }}
                >
                  Open report →
                </button>
              )}
            </>
          )}
          {running && (
            <div style={{ display: "flex", alignItems: "center", gap: 9, fontSize: 12.5, color: "var(--fg-muted)" }}>
              <span
                style={{
                  width: 14,
                  height: 14,
                  borderRadius: "50%",
                  border: "2px solid var(--border-strong)",
                  borderTopColor: "var(--accent)",
                  display: "inline-block",
                  animation: "dv-spin .7s linear infinite",
                }}
              />
              Processing — you can keep working
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/** Tiny helper: turn a "prop:value;prop2:value2" string into a React style object. */
function styleFromString(css: string): React.CSSProperties {
  const out: Record<string, string> = {};
  css
    .split(";")
    .map((s) => s.trim())
    .filter(Boolean)
    .forEach((decl) => {
      const idx = decl.indexOf(":");
      if (idx === -1) return;
      const prop = decl.slice(0, idx).trim();
      const value = decl.slice(idx + 1).trim();
      const camel = prop.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      out[camel] = value;
    });
  return out as React.CSSProperties;
}
