import React, { useEffect, useRef, useState } from "react";
import { ApiError, generateChapterReport, getChapterReport, getChapters, pollJob } from "../../api/client";
import type { Chapter, Report } from "../../api/types";

/** Live progress for an in-flight chapter-report generation. */
export interface ChapterGenInfo {
  chapter: Chapter;
  percent: number;
  logLine: string | null;
}

export interface ChapterPickerProps {
  paperId: string;
  /** Chapter.id currently being displayed or generated; null = whole paper. */
  activeChapterId: string | null;
  /** Non-null while a generation job is in flight. */
  genInfo: ChapterGenInfo | null;
  onWholePaper: () => void;
  onChapterReport: (report: Report) => void;
  onChapterGenerating: (info: ChapterGenInfo | null) => void;
  onChapterError: (message: string | null) => void;
}

const selectStyle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 500,
  color: "var(--fg)",
  background: "var(--surface-2)",
  border: "1px solid var(--border)",
  borderRadius: 9,
  padding: "8px 10px",
  maxWidth: 240,
};

/**
 * Chapter select — lists "Whole paper" first, then every Chapter from
 * GET /papers/{id}/chapters. The user only ever picks from this list; the
 * `<option value>` is always a Chapter.id we were given, never a typed title.
 *
 * Selecting an already-persisted chapter loads it immediately (the backend
 * returns `cached: true` synchronously); selecting a new one queues a job and
 * this component polls it with the existing `pollJob` helper — no second
 * polling loop.
 */
export function ChapterPicker({
  paperId,
  activeChapterId,
  genInfo,
  onWholePaper,
  onChapterReport,
  onChapterGenerating,
  onChapterError,
}: ChapterPickerProps) {
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [detected, setDetected] = useState(true);
  const [chaptersLoaded, setChaptersLoaded] = useState(false);
  // "this PDF has no chapter structure" and "the request itself failed" both
  // disable the picker, but they are not the same thing to a user — keep them
  // apart so the tooltip can say which one happened.
  const [loadError, setLoadError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const opRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    setChapters([]);
    setDetected(true);
    setChaptersLoaded(false);
    setLoadError(null);
    getChapters(paperId)
      .then((res) => {
        if (cancelled) return;
        setChapters(res.chapters);
        setDetected(res.detected);
      })
      .catch((err) => {
        if (cancelled) return;
        setChapters([]);
        setDetected(false);
        setLoadError(
          err instanceof ApiError || err instanceof Error
            ? err.message
            : "Couldn't reach the server."
        );
      })
      .finally(() => {
        if (!cancelled) setChaptersLoaded(true);
      });
    return () => {
      cancelled = true;
      abortRef.current?.abort();
    };
  }, [paperId]);

  async function handleChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const id = e.target.value;
    abortRef.current?.abort();
    onChapterError(null);

    if (!id) {
      onChapterGenerating(null);
      onWholePaper();
      return;
    }

    const chapter = chapters.find((c) => c.id === id);
    if (!chapter) return;

    const myOp = ++opRef.current;
    onChapterGenerating({ chapter, percent: 0, logLine: null });

    try {
      const res = await generateChapterReport(paperId, id);
      if (myOp !== opRef.current) return; // superseded by a later selection

      if (res.cached && res.report) {
        onChapterReport(res.report);
        return;
      }
      if (res.job) {
        const controller = new AbortController();
        abortRef.current = controller;
        onChapterGenerating({ chapter, percent: res.job.percent, logLine: null });

        const finalJob = await pollJob(res.job.id, {
          signal: controller.signal,
          onUpdate: (job) => {
            if (myOp !== opRef.current) return;
            onChapterGenerating({
              chapter,
              percent: job.percent,
              logLine: job.log.length ? job.log[job.log.length - 1].m : null,
            });
          },
        });
        if (myOp !== opRef.current) return;

        if (finalJob.state === "error") {
          throw new Error(finalJob.error_message || "Chapter report generation failed.");
        }
        const report = await getChapterReport(paperId, id);
        if (myOp !== opRef.current) return;
        onChapterReport(report);
        return;
      }
      throw new Error("Unexpected response from server.");
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      if (myOp !== opRef.current) return;
      // Order matters: onWholePaper() also clears any prior chapter error, so
      // reset the view *before* setting the new error, not after.
      onChapterGenerating(null);
      onWholePaper();
      onChapterError(
        err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Failed to generate chapter report."
      );
    }
  }

  const noChapters = chaptersLoaded && (!detected || chapters.length === 0);
  const busy = !!genInfo;
  const value = genInfo ? genInfo.chapter.id : activeChapterId ?? "";

  return (
    <select
      value={value}
      onChange={handleChange}
      disabled={noChapters || busy}
      title={
        loadError
          ? `Couldn't load chapters: ${loadError}`
          : noChapters
            ? "No chapters detected in this PDF"
            : undefined
      }
      aria-label="Report scope"
      style={selectStyle}
    >
      <option value="">Whole paper</option>
      {chapters.map((c) => (
        <option key={c.id} value={c.id}>
          {`${"  ".repeat(c.level - 1)}${c.index + 1}. ${c.title} (pp. ${c.page_start}-${c.page_end})`}
        </option>
      ))}
    </select>
  );
}
