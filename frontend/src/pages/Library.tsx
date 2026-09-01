import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { deletePaper, listPapers, mediaUrl, pollJob, rerunPaper, uploadPdf } from "../api/client";
import { ApiError } from "../api/client";
import { REPORT_STAGE } from "../api/types";
import type { PaperMeta, PaperStatus } from "../api/types";
import { useOverlays } from "../components/overlays/OverlaysContext";

/* --------------------------------------------------------------------------
   Small icons — ported verbatim (viewBox/paths) from the prototype's inline
   SVGs in the '===== LIBRARY =====' section.
   -------------------------------------------------------------------------- */

function SearchIcon() {
  return (
    <svg
      style={{ position: "absolute", left: 13, pointerEvents: "none" }}
      width={16}
      height={16}
      viewBox="0 0 24 24"
      fill="none"
      stroke="var(--fg-subtle)"
      strokeWidth={2}
    >
      <circle cx={11} cy={11} r={7} />
      <line x1={16} y1={16} x2={21} y2={21} strokeLinecap="round" />
    </svg>
  );
}

function RerunIcon() {
  return (
    <svg width={15} height={15} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
      <path d="M20 11a8 8 0 1 0-.9 4" strokeLinecap="round" />
      <path d="M20 5v6h-6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
      <path d="M12 16V4M12 4l-5 5M12 4l5 5M5 20h14" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function DeleteIcon() {
  return (
    <svg width={15} height={15} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
      <path d="M5 7h14M9 7V5h6v2M6 7l1 13h10l1-13" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

/* --------------------------------------------------------------------------
   Status badge — matches the prototype's statusBadge() factory exactly
   (fonts, colors, dot/spinner). 'queued' isn't in the mockup's fake data set
   (only ready/processing/error) — it's rendered with the same visual family
   as 'processing' but without the spin, labeled QUEUED.
   -------------------------------------------------------------------------- */
function StatusBadge({ status }: { status: PaperStatus }) {
  const base: React.CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    gap: 4,
    fontFamily: "'IBM Plex Mono'",
    fontSize: 10,
    fontWeight: 600,
    border: "1px solid var(--border)",
    padding: "2px 7px",
    borderRadius: 5,
  };

  if (status === "ready") {
    return (
      <span style={{ ...base, color: "var(--green)", background: "var(--green-soft)" }}>
        <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--green)" }} />
        READY
      </span>
    );
  }
  if (status === "processing") {
    return (
      <span style={{ ...base, color: "var(--amber)", background: "var(--amber-soft)" }}>
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            border: "2px solid var(--amber)",
            borderTopColor: "transparent",
            animation: "dv-spin .7s linear infinite",
          }}
        />
        PROCESSING
      </span>
    );
  }
  if (status === "queued") {
    return (
      <span style={{ ...base, color: "var(--fg-subtle)", background: "var(--surface-2)" }}>
        <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--fg-subtle)" }} />
        QUEUED
      </span>
    );
  }
  return (
    <span style={{ ...base, color: "var(--red)", background: "var(--red-soft)" }}>
      {"⚠"} ERROR
    </span>
  );
}

function formatAuthors(authors: string[]): string {
  if (!authors || authors.length === 0) return "Unknown authors";
  const shown = authors.slice(0, 3).join(", ");
  const extra = authors.length - 3;
  return extra > 0 ? `${shown} +${extra}` : shown;
}

function paperDateStr(p: PaperMeta): string {
  return p.published || p.updated || p.created_at.slice(0, 10);
}

function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return "Something went wrong.";
}

/* --------------------------------------------------------------------------
   Page thumbnail — the prototype always shows a diagonal-stripe placeholder
   block (its data is fake, in-memory). We try to render the real page-1
   render from paper.thumbnail_path via the shared mediaUrl() helper, and
   fall back to the prototype's exact placeholder look if there's no
   thumbnail yet or the image fails to load (e.g. paper still processing).
   -------------------------------------------------------------------------- */
function PaperThumb({ paper }: { paper: PaperMeta }) {
  const [failed, setFailed] = useState(false);
  const thumbnailPath = paper.thumbnail_path;
  const src = thumbnailPath && !failed ? mediaUrl(thumbnailPath) : null;

  return (
    <div
      style={{
        height: 120,
        position: "relative",
        background: "var(--surface-2)",
        backgroundImage:
          "repeating-linear-gradient(135deg, transparent 0 11px, var(--surface-3) 11px 22px)",
        display: "flex",
        alignItems: "flex-end",
        padding: 10,
        overflow: "hidden",
      }}
    >
      {src && (
        <img
          src={src}
          alt=""
          onError={() => setFailed(true)}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            objectFit: "cover",
          }}
        />
      )}
      {!src && (
        <span
          className="dv-mono"
          style={{
            fontSize: 10.5,
            color: "var(--fg-subtle)",
            background: "var(--surface)",
            border: "1px solid var(--border)",
            padding: "3px 7px",
            borderRadius: 5,
          }}
        >
          page-thumbnail
        </span>
      )}
      <span style={{ position: "absolute", top: 10, right: 10 }}>
        <StatusBadge status={paper.status} />
      </span>
    </div>
  );
}

const ingestMenuItemStyle: React.CSSProperties = {
  display: "flex",
  width: "100%",
  alignItems: "center",
  gap: 9,
  fontSize: 13,
  color: "var(--fg)",
  background: "transparent",
  border: "none",
  borderRadius: 7,
  padding: "9px 11px",
  textAlign: "left",
};

/* --------------------------------------------------------------------------
   Library page
   -------------------------------------------------------------------------- */
export default function LibraryPage() {
  const navigate = useNavigate();
  const { openIngest } = useOverlays();

  const [papers, setPapers] = useState<PaperMeta[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [actionError, setActionError] = useState<string | null>(null);

  const [rerunningIds, setRerunningIds] = useState<Set<string>>(new Set());
  const [deletingIds, setDeletingIds] = useState<Set<string>>(new Set());

  const [ingestMenuOpen, setIngestMenuOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const ingestMenuRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Abort controllers for in-flight rerun polls, so we can cancel on unmount.
  const pollControllers = useRef<Map<string, AbortController>>(new Map());
  const mountedRef = useRef(true);

  const loadPapers = useCallback(async (opts?: { silent?: boolean }) => {
    if (!opts?.silent) setLoading(true);
    try {
      const res = await listPapers();
      if (!mountedRef.current) return;
      setPapers(res.papers);
      setLoadError(null);
    } catch (err) {
      if (!mountedRef.current) return;
      if (!opts?.silent) setLoadError(errorMessage(err));
    } finally {
      if (mountedRef.current && !opts?.silent) setLoading(false);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    loadPapers();
    return () => {
      mountedRef.current = false;
      pollControllers.current.forEach((c) => c.abort());
      pollControllers.current.clear();
    };
  }, [loadPapers]);

  // Background papers (queued/processing) can finish outside this page's
  // control (e.g. an ingest kicked off from Search). Poll the list lightly
  // while any paper is in flight so badges stay live without a manual reload.
  const hasInFlight = useMemo(
    () => (papers ?? []).some((p) => p.status === "processing" || p.status === "queued"),
    [papers]
  );
  useEffect(() => {
    if (!hasInFlight) return;
    const iv = setInterval(() => loadPapers({ silent: true }), 4000);
    return () => clearInterval(iv);
  }, [hasInFlight, loadPapers]);

  useEffect(() => {
    if (!ingestMenuOpen) return;
    const onDocClick = (e: MouseEvent) => {
      if (ingestMenuRef.current && !ingestMenuRef.current.contains(e.target as Node)) {
        setIngestMenuOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIngestMenuOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [ingestMenuOpen]);

  const filtered = useMemo(() => {
    const all = papers ?? [];
    const lq = query.trim().toLowerCase();
    if (!lq) return all;
    return all.filter(
      (p) =>
        p.title.toLowerCase().includes(lq) ||
        p.authors.some((a) => a.toLowerCase().includes(lq)) ||
        p.arxiv_label.toLowerCase().includes(lq)
    );
  }, [papers, query]);

  const handleOpen = useCallback((p: PaperMeta) => navigate(`/report/${p.id}`), [navigate]);
  const handleChat = useCallback((p: PaperMeta) => navigate(`/chat/${p.id}`), [navigate]);

  const handleRerun = useCallback(
    async (p: PaperMeta) => {
      if (rerunningIds.has(p.id)) return;
      setActionError(null);
      setRerunningIds((prev) => new Set(prev).add(p.id));
      const controller = new AbortController();
      pollControllers.current.set(p.id, controller);
      try {
        const res = await rerunPaper(p.id);
        const job = await pollJob(res.job.id, { signal: controller.signal });
        await loadPapers({ silent: true });
        // A re-run does NOT open the ingestion modal, so the modal's report
        // failure/degradation notice never reaches the user on this path — and
        // the report stage is deliberately non-fatal, so the job still finishes
        // `done` and the card just quietly goes back to normal. Surface it here
        // or the whole point of tracking the stage is lost for re-runs, which is
        // exactly the path a user takes to *retry a failed report*.
        const reportStage = job.stages.find((s) => s.stage === REPORT_STAGE);
        if (reportStage?.status === "error") {
          setActionError(
            `Re-run finished for "${p.title}", but report generation failed` +
              (reportStage.detail ? ` — ${reportStage.detail}` : ".") +
              " The paper is ingested: chat still works."
          );
        } else if (reportStage?.status === "done" && reportStage.degraded) {
          setActionError(
            `Re-run finished for "${p.title}", but the report is degraded — ` +
              `${reportStage.detail}. Those parts fell back to extractive drafts ` +
              "rather than model-written prose; a faster model (Settings) avoids it."
          );
        }
      } catch (err) {
        if (!(err instanceof DOMException && err.name === "AbortError")) {
          setActionError(`Re-run failed for "${p.title}" — ${errorMessage(err)}`);
          await loadPapers({ silent: true }).catch(() => {});
        }
      } finally {
        pollControllers.current.delete(p.id);
        if (mountedRef.current) {
          setRerunningIds((prev) => {
            const next = new Set(prev);
            next.delete(p.id);
            return next;
          });
        }
      }
    },
    [rerunningIds, loadPapers]
  );

  const handleDelete = useCallback(
    async (p: PaperMeta) => {
      if (deletingIds.has(p.id)) return;
      const ok = window.confirm(
        `Delete "${p.title}"?\n\nThis removes its report, chat history, and extracted data. This can't be undone.`
      );
      if (!ok) return;
      setActionError(null);
      setDeletingIds((prev) => new Set(prev).add(p.id));
      try {
        await deletePaper(p.id);
        if (!mountedRef.current) return;
        setPapers((prev) => (prev ? prev.filter((x) => x.id !== p.id) : prev));
      } catch (err) {
        if (!mountedRef.current) return;
        setActionError(`Delete failed for "${p.title}" — ${errorMessage(err)}`);
      } finally {
        if (mountedRef.current) {
          setDeletingIds((prev) => {
            const next = new Set(prev);
            next.delete(p.id);
            return next;
          });
        }
      }
    },
    [deletingIds]
  );

  const handleFileSelected = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0] ?? null;
      // Reset the input so selecting the same file again still fires onChange.
      e.target.value = "";
      if (!file) return;
      setActionError(null);
      setUploading(true);
      try {
        const res = await uploadPdf(file);
        openIngest({ paperId: res.paper_id, paperTitle: file.name, jobId: res.job.id });
        await loadPapers({ silent: true });
      } catch (err) {
        setActionError(`Upload failed for "${file.name}" — ${errorMessage(err)}`);
      } finally {
        if (mountedRef.current) setUploading(false);
      }
    },
    [openIngest, loadPapers]
  );

  const count = papers?.length ?? 0;
  const showEmptyFiltered = !loading && !loadError && papers !== null && papers.length > 0 && filtered.length === 0;
  const showEmptyLibrary = !loading && !loadError && papers !== null && papers.length === 0;

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "28px 28px 60px" }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 20 }}>
        <div style={{ flex: 1, position: "relative", display: "flex", alignItems: "center", maxWidth: 380 }}>
          <SearchIcon />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter your library…"
            style={{
              width: "100%",
              padding: "10px 14px 10px 38px",
              fontSize: 13.5,
              border: "1px solid var(--border)",
              borderRadius: 9,
              background: "var(--surface)",
              color: "var(--fg)",
              outline: "none",
            }}
          />
        </div>
        <div className="dv-mono" style={{ fontSize: 12, color: "var(--fg-subtle)", marginLeft: "auto" }}>
          {loading ? "…" : count} {count === 1 ? "paper" : "papers"}
        </div>
        {/* Upload is a first-class button, not only a menu item: when it lived
            solely inside the "+ Ingest new" dropdown the button looked identical
            to the old navigate-to-Search one, so the feature read as missing. */}
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
          title="Upload a PDF from your computer (any length)"
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 7,
            fontSize: 13,
            fontWeight: 600,
            color: "var(--fg)",
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: 9,
            padding: "9px 14px",
            opacity: uploading ? 0.7 : 1,
          }}
        >
          <span style={{ display: "inline-flex", color: "var(--fg-subtle)" }}>
            <UploadIcon />
          </span>
          {uploading ? "Uploading…" : "Upload PDF"}
        </button>
        <div style={{ position: "relative" }} ref={ingestMenuRef}>
          <button
            onClick={() => setIngestMenuOpen((v) => !v)}
            disabled={uploading}
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: "var(--accent-fg)",
              background: "var(--accent)",
              border: "none",
              borderRadius: 9,
              padding: "9px 15px",
              opacity: uploading ? 0.7 : 1,
            }}
          >
            {uploading ? "Uploading…" : "+ Ingest new ▾"}
          </button>
          {ingestMenuOpen && (
            <div
              style={{
                position: "absolute",
                top: 44,
                right: 0,
                zIndex: 20,
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: 10,
                boxShadow: "var(--shadow-lg)",
                padding: 5,
                minWidth: 180,
                animation: "dv-fade .12s ease",
              }}
            >
              <button
                onClick={() => {
                  setIngestMenuOpen(false);
                  navigate("/search");
                }}
                style={ingestMenuItemStyle}
              >
                <span className="dv-mono" style={{ fontSize: 11, color: "var(--fg-subtle)", width: 14 }}>
                  ↗
                </span>
                Search arXiv
              </button>
              <button
                onClick={() => {
                  setIngestMenuOpen(false);
                  fileInputRef.current?.click();
                }}
                style={ingestMenuItemStyle}
              >
                <span style={{ display: "inline-flex", width: 14, color: "var(--fg-subtle)" }}>
                  <UploadIcon />
                </span>
                Upload PDF
              </button>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            style={{ display: "none" }}
            onChange={handleFileSelected}
          />
        </div>
      </div>

      {actionError && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            fontSize: 12.5,
            color: "var(--red)",
            background: "var(--red-soft)",
            border: "1px solid var(--border)",
            borderRadius: 9,
            padding: "10px 14px",
            marginBottom: 16,
          }}
        >
          <span style={{ flex: 1 }}>{actionError}</span>
          <button
            onClick={() => setActionError(null)}
            style={{
              flex: "none",
              background: "transparent",
              border: "none",
              color: "var(--red)",
              fontSize: 13,
              fontWeight: 600,
            }}
            aria-label="Dismiss"
          >
            ✕
          </button>
        </div>
      )}

      {loading && (
        <div style={{ textAlign: "center", padding: "80px 20px", color: "var(--fg-subtle)" }}>
          <div
            style={{
              width: 22,
              height: 22,
              margin: "0 auto 14px",
              borderRadius: "50%",
              border: "2px solid var(--border-strong)",
              borderTopColor: "var(--accent)",
              animation: "dv-spin .7s linear infinite",
            }}
          />
          <div style={{ fontSize: 13.5 }}>Loading your library…</div>
        </div>
      )}

      {!loading && loadError && (
        <div style={{ textAlign: "center", padding: "80px 20px", color: "var(--fg-subtle)" }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: "var(--red)" }}>Couldn't load your library</div>
          <div style={{ fontSize: 13, marginTop: 6 }}>{loadError}</div>
          <button
            onClick={() => loadPapers()}
            style={{
              marginTop: 16,
              fontSize: 13,
              fontWeight: 600,
              color: "var(--accent-fg)",
              background: "var(--accent)",
              border: "none",
              borderRadius: 9,
              padding: "9px 16px",
            }}
          >
            Retry
          </button>
        </div>
      )}

      {showEmptyFiltered && (
        <div style={{ textAlign: "center", padding: "80px 20px", color: "var(--fg-subtle)" }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: "var(--fg-muted)" }}>
            No papers match "{query}"
          </div>
          <div style={{ fontSize: 13, marginTop: 6 }}>
            Try a different term, or ingest a new paper from Search.
          </div>
        </div>
      )}

      {showEmptyLibrary && (
        <div style={{ textAlign: "center", padding: "80px 20px", color: "var(--fg-subtle)" }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: "var(--fg-muted)" }}>Your library is empty</div>
          <div style={{ fontSize: 13, marginTop: 6 }}>
            Search arXiv and ingest a paper to see it here.
          </div>
          <button
            onClick={() => navigate("/search")}
            style={{
              marginTop: 16,
              fontSize: 13,
              fontWeight: 600,
              color: "var(--accent-fg)",
              background: "var(--accent)",
              border: "none",
              borderRadius: 9,
              padding: "9px 16px",
            }}
          >
            Go to Search
          </button>
        </div>
      )}

      {!loading && !loadError && filtered.length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 14 }}>
          {filtered.map((p) => {
            const isRerunning = rerunningIds.has(p.id) || p.status === "processing";
            const isDeleting = deletingIds.has(p.id);
            const effectivePaper: PaperMeta = rerunningIds.has(p.id) ? { ...p, status: "processing" } : p;

            // TWO gates, deliberately not one flag.
            //
            // Chat rides on `status === 'ready'`, which
            // the backend sets the moment the embedding stage lands — chunks,
            // vectors and images are on disk, so chat genuinely works.
            //
            // "Open report" rides on `has_report`, which is true only once a
            // report row exists. Report generation is the last ingestion stage
            // and runs ~20 minutes on a local model, so there is a long, normal
            // window where a paper is chattable but has no report. Enabling the
            // button on `status` would offer a report that does not exist, and
            // that click makes GET /report start a second agent run racing the
            // ingest thread.
            const canChat = p.status === "ready" && !isRerunning;
            const canOpenReport = p.has_report && !isRerunning;
            const openReportTitle = canOpenReport
              ? "Open the generated report"
              : p.status === "error"
                ? "Ingestion failed — re-run this paper to generate its report"
                : p.has_report
                  ? "Re-running ingestion — this paper's report is being regenerated"
                  : "Report is still generating — it's the last ingestion stage and can take " +
                    "~20 minutes on a local model. Chat already works for this paper. " +
                    "If ingestion has already finished, re-run this paper to build the report.";
            const chatTitle = canChat
              ? "Ask questions about this paper"
              : p.status === "error"
                ? "Ingestion failed — re-run this paper"
                : "Still indexing — chat unlocks as soon as the embedding stage finishes";
            return (
              <div
                key={p.id}
                style={{
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                  borderRadius: 13,
                  overflow: "hidden",
                  boxShadow: "var(--shadow)",
                  display: "flex",
                  flexDirection: "column",
                  opacity: isDeleting ? 0.5 : 1,
                  transition: "opacity .15s ease",
                }}
              >
                <PaperThumb paper={effectivePaper} />
                <div style={{ padding: "14px 15px", flex: 1, display: "flex", flexDirection: "column" }}>
                  <div
                    style={{
                      fontSize: 14,
                      fontWeight: 600,
                      lineHeight: 1.35,
                      letterSpacing: "-.01em",
                      overflow: "hidden",
                      display: "-webkit-box",
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: "vertical",
                    }}
                  >
                    {p.title}
                  </div>
                  <div style={{ fontSize: 12, color: "var(--fg-muted)", marginTop: 6 }}>
                    {formatAuthors(p.authors)}
                  </div>
                  <div className="dv-mono" style={{ fontSize: 11, color: "var(--fg-subtle)", marginTop: 3 }}>
                    {paperDateStr(p)}
                  </div>
                  {p.status === "error" && p.error_message && (
                    <div style={{ fontSize: 11.5, color: "var(--red)", marginTop: 6, lineHeight: 1.4 }}>
                      {p.error_message}
                    </div>
                  )}
                  <div style={{ display: "flex", gap: 8, marginTop: "auto", paddingTop: 14 }}>
                    <button
                      onClick={() => handleOpen(p)}
                      disabled={!canOpenReport}
                      title={openReportTitle}
                      style={{
                        flex: 1,
                        fontSize: 12.5,
                        fontWeight: 600,
                        color: "var(--accent)",
                        background: "var(--accent-soft)",
                        border: "1px solid var(--accent-border)",
                        borderRadius: 8,
                        padding: "7px 0",
                        opacity: canOpenReport ? 1 : 0.5,
                        cursor: canOpenReport ? "pointer" : "not-allowed",
                      }}
                    >
                      Open report
                    </button>
                    <button
                      onClick={() => handleChat(p)}
                      disabled={!canChat}
                      title={chatTitle}
                      style={{
                        flex: 1,
                        fontSize: 12.5,
                        fontWeight: 600,
                        color: "var(--fg)",
                        background: "var(--surface-2)",
                        border: "1px solid var(--border)",
                        borderRadius: 8,
                        padding: "7px 0",
                        opacity: canChat ? 1 : 0.5,
                        cursor: canChat ? "pointer" : "not-allowed",
                      }}
                    >
                      Chat
                    </button>
                    <button
                      title="Re-run pipeline"
                      onClick={() => handleRerun(p)}
                      disabled={isRerunning || isDeleting}
                      style={{
                        flex: "none",
                        width: 34,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        background: "var(--surface-2)",
                        border: "1px solid var(--border)",
                        borderRadius: 8,
                        color: "var(--fg-muted)",
                        opacity: isRerunning || isDeleting ? 0.5 : 1,
                        cursor: isRerunning || isDeleting ? "not-allowed" : "pointer",
                      }}
                    >
                      {isRerunning ? (
                        <span
                          style={{
                            width: 12,
                            height: 12,
                            borderRadius: "50%",
                            border: "2px solid var(--border-strong)",
                            borderTopColor: "var(--fg-muted)",
                            animation: "dv-spin .7s linear infinite",
                          }}
                        />
                      ) : (
                        <RerunIcon />
                      )}
                    </button>
                    <button
                      title="Delete"
                      onClick={() => handleDelete(p)}
                      disabled={isDeleting || isRerunning}
                      style={{
                        flex: "none",
                        width: 34,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        background: "var(--surface-2)",
                        border: "1px solid var(--border)",
                        borderRadius: 8,
                        color: "var(--red)",
                        opacity: isDeleting || isRerunning ? 0.5 : 1,
                        cursor: isDeleting || isRerunning ? "not-allowed" : "pointer",
                      }}
                    >
                      <DeleteIcon />
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
