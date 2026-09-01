import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ApiError, getReport } from "../api/client";
import { SECTION_ORDER } from "../api/types";
import type { Report } from "../api/types";
import { useOverlays } from "../components/overlays/OverlaysContext";
import { downloadTextFile, printReportAsPdf, reportToMarkdown } from "./report/export";
import { SectionCard } from "./report/SectionCard";
import { ChapterPicker } from "./report/ChapterPicker";
import type { ChapterGenInfo } from "./report/ChapterPicker";
import { StudyLaunchCards } from "./report/StudyLaunchCards";

const PAGE_STYLE: React.CSSProperties = { maxWidth: 820, margin: "0 auto", padding: "34px 28px 80px" };

function StatTile({ label, value }: { label: string; value: string | number }) {
  return (
    <div style={{ flex: 1 }}>
      <div className="dv-mono" style={{ fontSize: 11, color: "var(--fg-subtle)" }}>
        {label}
      </div>
      <div style={{ fontSize: 18, fontWeight: 600, marginTop: 2 }}>{value}</div>
    </div>
  );
}

function Spinner() {
  return (
    <span
      style={{
        display: "inline-block",
        width: 14,
        height: 14,
        borderRadius: "50%",
        border: "2px solid var(--border-strong)",
        borderTopColor: "var(--accent)",
        animation: "dv-spin .7s linear infinite",
      }}
    />
  );
}

function ChapterGeneratingPanel({ info }: { info: ChapterGenInfo }) {
  return (
    <div
      style={{
        marginTop: 22,
        padding: "20px 20px",
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: 12,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <Spinner />
        <div style={{ fontSize: 13.5, fontWeight: 600 }}>
          Generating report for &ldquo;{info.chapter.title}&rdquo;…
        </div>
      </div>
      <div
        style={{
          marginTop: 14,
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
            background: "var(--accent)",
            width: `${info.percent}%`,
            transition: "width .3s ease",
          }}
        />
      </div>
      <div className="dv-mono" style={{ marginTop: 10, fontSize: 12, color: "var(--fg-muted)" }}>
        {info.percent}%{info.logLine ? ` — ${info.logLine}` : ""}
      </div>
      <div style={{ marginTop: 6, fontSize: 11.5, color: "var(--fg-subtle)" }}>
        This can take several minutes on a local model. Feel free to keep browsing — come back anytime.
      </div>
    </div>
  );
}

export default function ReportPage() {
  const { paperId } = useParams<{ paperId?: string }>();
  const navigate = useNavigate();
  const { openCitation, openLightbox } = useOverlays();

  // Whole-paper report (GET /report/{paperId}).
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Chapter scope — null chapterReport/activeChapterId means "whole paper".
  const [activeChapterId, setActiveChapterId] = useState<string | null>(null);
  const [chapterReport, setChapterReport] = useState<Report | null>(null);
  const [chapterGenInfo, setChapterGenInfo] = useState<ChapterGenInfo | null>(null);
  const [chapterError, setChapterError] = useState<string | null>(null);

  const [openSections, setOpenSections] = useState<Record<string, boolean>>({});
  const [openDeepDive, setOpenDeepDive] = useState<Record<string, boolean>>({});
  const [exportOpen, setExportOpen] = useState(false);
  const exportRef = useRef<HTMLDivElement | null>(null);

  const displayReport = chapterReport ?? report;

  const loadReport = useCallback(
    (id: string, signal?: { cancelled: boolean }) => {
      setLoading(true);
      setError(null);
      getReport(id)
        .then((r) => {
          if (signal?.cancelled) return;
          setReport(r);
          const initialOpen: Record<string, boolean> = {};
          r.sections.forEach((s) => {
            initialOpen[s.id] = s.default_open;
          });
          setOpenSections(initialOpen);
          setOpenDeepDive({});
        })
        .catch((err: unknown) => {
          if (signal?.cancelled) return;
          setReport(null);
          setError(err instanceof ApiError ? err.message : "Failed to load report.");
        })
        .finally(() => {
          if (!signal?.cancelled) setLoading(false);
        });
    },
    []
  );

  useEffect(() => {
    if (!paperId) {
      setReport(null);
      setError(null);
      setLoading(false);
      setActiveChapterId(null);
      setChapterReport(null);
      setChapterGenInfo(null);
      setChapterError(null);
      return;
    }
    localStorage.setItem("dv-last-paper", paperId);
    // Switching papers always resets the chapter scope back to whole paper.
    setActiveChapterId(null);
    setChapterReport(null);
    setChapterGenInfo(null);
    setChapterError(null);
    const signal = { cancelled: false };
    loadReport(paperId, signal);
    return () => {
      signal.cancelled = true;
    };
  }, [paperId, loadReport]);

  useEffect(() => {
    if (!exportOpen) return;
    const onDocClick = (e: MouseEvent) => {
      if (exportRef.current && !exportRef.current.contains(e.target as Node)) setExportOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setExportOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [exportOpen]);

  const orderedSections = useMemo(() => {
    if (!displayReport) return [];
    const rank = (name: string) => {
      const i = SECTION_ORDER.indexOf(name as (typeof SECTION_ORDER)[number]);
      return i === -1 ? SECTION_ORDER.length : i;
    };
    return [...displayReport.sections].sort((a, b) => rank(a.name) - rank(b.name));
  }, [displayReport]);

  function handleRetry() {
    if (!paperId) return;
    loadReport(paperId);
  }

  function handleExportMd() {
    if (!displayReport) return;
    const filename = `${displayReport.paper?.arxiv_id || displayReport.paper_id || "report"}.md`;
    downloadTextFile(filename, reportToMarkdown(displayReport), "text/markdown");
    setExportOpen(false);
  }

  function handleExportPdf() {
    if (!displayReport) return;
    printReportAsPdf(displayReport);
    setExportOpen(false);
  }

  // ---- chapter picker callbacks -------------------------------------------
  const handleWholePaper = useCallback(() => {
    setActiveChapterId(null);
    setChapterReport(null);
    setChapterGenInfo(null);
    setChapterError(null);
    if (report) {
      const initialOpen: Record<string, boolean> = {};
      report.sections.forEach((s) => {
        initialOpen[s.id] = s.default_open;
      });
      setOpenSections(initialOpen);
      setOpenDeepDive({});
    }
  }, [report]);

  const handleChapterReport = useCallback((r: Report) => {
    setChapterReport(r);
    setActiveChapterId(r.chapter_id ?? null);
    setChapterGenInfo(null);
    setChapterError(null);
    const initialOpen: Record<string, boolean> = {};
    r.sections.forEach((s) => {
      initialOpen[s.id] = s.default_open;
    });
    setOpenSections(initialOpen);
    setOpenDeepDive({});
  }, []);

  const handleChapterGenerating = useCallback((info: ChapterGenInfo | null) => {
    setChapterGenInfo(info);
    if (info) {
      setActiveChapterId(info.chapter.id);
      setChapterError(null);
    }
  }, []);

  const handleChapterError = useCallback((message: string | null) => {
    setChapterError(message);
    if (message) {
      setActiveChapterId(null);
      setChapterReport(null);
      setChapterGenInfo(null);
    }
  }, []);

  // ---- no paper selected ---------------------------------------------------
  if (!paperId) {
    return (
      <div style={PAGE_STYLE}>
        <div style={{ fontSize: 15, fontWeight: 600 }}>No paper selected</div>
        <div style={{ fontSize: 13, color: "var(--fg-subtle)", marginTop: 6 }}>
          Open a paper from your library to view its report.
        </div>
        <button
          onClick={() => navigate("/library")}
          style={{
            marginTop: 16,
            fontSize: 13.5,
            fontWeight: 600,
            color: "var(--accent-fg)",
            background: "var(--accent)",
            border: "none",
            borderRadius: 9,
            padding: "9px 16px",
          }}
        >
          Go to Library
        </button>
      </div>
    );
  }

  const paper = displayReport?.paper;
  const authors = paper?.authors ?? [];
  const shownAuthors = authors.slice(0, 5).join(", ");
  const extraAuthors = authors.length > 5 ? authors.length - 5 : 0;
  const absUrl = paper?.abs_url || paper?.pdf_url || (paper ? `https://arxiv.org/abs/${paper.arxiv_id}` : undefined);
  const titleText = paper?.title ?? (loading && !report ? "Loading…" : paperId);
  const isChapterMode = !!chapterReport && !chapterGenInfo;

  return (
    <div style={PAGE_STYLE}>
      <h1
        className="dv-serif"
        style={{ fontSize: 30, lineHeight: 1.22, fontWeight: 600, letterSpacing: "-.015em", margin: "12px 0 0" }}
      >
        {titleText}
      </h1>
      {authors.length > 0 && (
        <div style={{ fontSize: 14, color: "var(--fg-muted)", marginTop: 14 }}>
          {shownAuthors}
          {extraAuthors > 0 && (
            <span style={{ color: "var(--fg-subtle)" }}> · +{extraAuthors} author{extraAuthors > 1 ? "s" : ""}</span>
          )}
        </div>
      )}

      {isChapterMode && (
        <div style={{ marginTop: 14 }}>
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              fontSize: 12.5,
              fontWeight: 600,
              color: "var(--accent)",
              background: "var(--accent-soft)",
              border: "1px solid var(--accent)",
              borderRadius: 20,
              padding: "5px 8px 5px 12px",
            }}
          >
            Reporting on: {chapterReport?.chapter_title}
            <button
              onClick={handleWholePaper}
              aria-label="Back to whole paper"
              title="Back to whole paper"
              style={{
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                width: 18,
                height: 18,
                fontSize: 11,
                fontWeight: 700,
                color: "var(--accent)",
                background: "var(--surface)",
                border: "none",
                borderRadius: "50%",
              }}
            >
              ✕
            </button>
          </span>
        </div>
      )}

      {chapterError && (
        <div
          style={{
            marginTop: 14,
            fontSize: 12.5,
            color: "var(--red)",
            background: "var(--red-soft)",
            border: "1px solid var(--red)",
            borderRadius: 9,
            padding: "8px 12px",
          }}
        >
          Couldn't generate that chapter report: {chapterError}
        </div>
      )}

      <div style={{ display: "flex", flexWrap: "wrap", gap: 9, alignItems: "center", marginTop: 16 }}>
        {paper?.published && (
          <span className="dv-mono" style={{ fontSize: 11.5, color: "var(--fg-muted)" }}>
            Published {paper.published}
          </span>
        )}
        {paper?.published && absUrl && (
          <span style={{ width: 3, height: 3, borderRadius: "50%", background: "var(--fg-subtle)" }} />
        )}
        {absUrl && (
          <a href={absUrl} target="_blank" rel="noopener noreferrer" style={{ fontSize: 12.5, fontWeight: 500 }}>
            View on arXiv ↗
          </a>
        )}
        <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center", position: "relative" }} ref={exportRef}>
          <ChapterPicker
            paperId={paperId}
            activeChapterId={activeChapterId}
            genInfo={chapterGenInfo}
            onWholePaper={handleWholePaper}
            onChapterReport={handleChapterReport}
            onChapterGenerating={handleChapterGenerating}
            onChapterError={handleChapterError}
          />
          <button
            onClick={() => navigate(`/chat/${paperId}`)}
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: "var(--fg)",
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              borderRadius: 9,
              padding: "8px 15px",
            }}
          >
            Chat with paper
          </button>
          <button
            onClick={() => setExportOpen((v) => !v)}
            disabled={!displayReport}
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: "var(--accent-fg)",
              background: "var(--accent)",
              border: "none",
              borderRadius: 9,
              padding: "8px 16px",
              opacity: displayReport ? 1 : 0.5,
              cursor: displayReport ? "pointer" : "not-allowed",
            }}
          >
            Export ▾
          </button>
          {exportOpen && displayReport && (
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
                minWidth: 170,
                animation: "dv-fade .12s ease",
              }}
            >
              <button
                onClick={handleExportMd}
                style={{
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
                }}
              >
                <span className="dv-mono" style={{ fontSize: 11, color: "var(--fg-subtle)" }}>
                  MD
                </span>
                Export as Markdown
              </button>
              <button
                onClick={handleExportPdf}
                style={{
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
                }}
              >
                <span className="dv-mono" style={{ fontSize: 11, color: "var(--fg-subtle)" }}>
                  PDF
                </span>
                Export as PDF
              </button>
            </div>
          )}
        </div>
      </div>

      {chapterGenInfo ? (
        <ChapterGeneratingPanel info={chapterGenInfo} />
      ) : loading && !report && !chapterReport ? (
        <div
          style={{
            marginTop: 22,
            display: "flex",
            alignItems: "center",
            gap: 10,
            color: "var(--fg-muted)",
            fontSize: 13.5,
          }}
        >
          <Spinner />
          Loading report…
        </div>
      ) : error && !chapterReport ? (
        <div
          style={{
            marginTop: 22,
            background: "var(--red-soft)",
            border: "1px solid var(--red)",
            borderRadius: 12,
            padding: "16px 18px",
          }}
        >
          <div style={{ fontSize: 14, fontWeight: 600, color: "var(--red)" }}>Couldn't load report</div>
          <div style={{ fontSize: 13, color: "var(--fg-muted)", marginTop: 6 }}>{error}</div>
          <button
            onClick={handleRetry}
            style={{
              marginTop: 14,
              fontSize: 13,
              fontWeight: 600,
              color: "var(--fg)",
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 9,
              padding: "8px 15px",
            }}
          >
            Try again
          </button>
        </div>
      ) : !displayReport ? (
        <div style={{ marginTop: 22, fontSize: 13, color: "var(--fg-subtle)" }}>
          No report available for this paper.
        </div>
      ) : (
        <>
          <div
            style={{
              display: "flex",
              gap: 16,
              marginTop: 20,
              padding: "14px 16px",
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 12,
            }}
          >
            <StatTile label="PAGES" value={displayReport.stats.pages} />
            <StatTile label="FIGURES" value={displayReport.stats.figures} />
            <StatTile label="CITATIONS EXTRACTED" value={displayReport.stats.citations_extracted} />
            <StatTile label="READING TIME" value={`${displayReport.stats.reading_time_min} min`} />
          </div>

          {orderedSections.length === 0 ? (
            <div style={{ marginTop: 22, fontSize: 13, color: "var(--fg-subtle)" }}>
              No sections available for this report.
            </div>
          ) : (
            orderedSections.map((section, i) => (
              <SectionCard
                key={section.id}
                section={section}
                isFirst={i === 0}
                isOpen={!!openSections[section.id]}
                onToggleOpen={() =>
                  setOpenSections((prev) => ({ ...prev, [section.id]: !prev[section.id] }))
                }
                isDeepOpen={!!openDeepDive[section.id]}
                onToggleDeep={() =>
                  setOpenDeepDive((prev) => ({ ...prev, [section.id]: !prev[section.id] }))
                }
                figuresTotal={displayReport.stats.figures}
                onCite={openCitation}
                onOpenMedia={openLightbox}
              />
            ))
          )}

          <StudyLaunchCards paperId={paperId} />
        </>
      )}
    </div>
  );
}
