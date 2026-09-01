import React, { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, ingest as apiIngest, search as apiSearch } from "../api/client";
import type { SearchDateRange, SearchResultItem, SearchSort } from "../api/types";
import { useAppStore } from "../state/AppStore";
import { useOverlays } from "../components/overlays/OverlaysContext";

/* --------------------------------------------------------------------------
   Static option lists — category values are arXiv category codes sent
   verbatim as the `category` query param; "" means "All fields" (omitted).
   Sort intentionally excludes "most_cited" — the backend only supports
   relevance + newest.
   -------------------------------------------------------------------------- */

const CATEGORY_OPTIONS: { value: string; label: string }[] = [
  { value: "", label: "All fields" },
  { value: "cs.CL", label: "cs.CL — Computation & Language" },
  { value: "cs.CV", label: "cs.CV — Computer Vision" },
  { value: "cs.LG", label: "cs.LG — Machine Learning" },
  { value: "cs.IR", label: "cs.IR — Information Retrieval" },
  { value: "cs.AI", label: "cs.AI — Artificial Intelligence" },
];

const DATE_OPTIONS: { value: SearchDateRange; label: string }[] = [
  { value: "any", label: "Any time" },
  { value: "past_month", label: "Past month" },
  { value: "past_6_months", label: "Past 6 months" },
  { value: "past_year", label: "Past year" },
  { value: "2024_2025", label: "2024–2025" },
];

const SORT_OPTIONS: { value: SearchSort; label: string }[] = [
  { value: "relevance", label: "Relevance" },
  { value: "newest", label: "Newest first" },
];

function formatDate(iso?: string | null): string {
  if (!iso) return "—";
  const d = new Date(`${iso}T00:00:00`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function formatAuthors(authors: string[]): string {
  if (!authors || authors.length === 0) return "Unknown authors";
  if (authors.length <= 3) return authors.join(", ");
  return `${authors.slice(0, 3).join(", ")} et al.`;
}

const filterChipStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 7,
  fontSize: 12.5,
  color: "var(--fg-muted)",
  background: "var(--surface)",
  border: "1px solid var(--border)",
  borderRadius: 8,
  padding: "0 10px 0 12px",
};

const filterSelectStyle: React.CSSProperties = {
  border: "none",
  background: "transparent",
  color: "var(--fg)",
  fontSize: 12.5,
  fontWeight: 500,
  padding: "8px 4px",
  outline: "none",
};

const catTagStyle: React.CSSProperties = {
  fontFamily: "'IBM Plex Mono', ui-monospace, monospace",
  fontSize: 11,
  fontWeight: 500,
  color: "var(--fg-muted)",
  background: "var(--surface-2)",
  border: "1px solid var(--border)",
  padding: "2px 8px",
  borderRadius: 6,
};

interface FilterState {
  category: string;
  dateRange: SearchDateRange;
  sort: SearchSort;
}

export default function SearchPage() {
  const navigate = useNavigate();
  const { openIngest } = useOverlays();

  const [query, setQuery] = useState("");
  const [filters, setFilters] = useState<FilterState>({ category: "", dateRange: "any", sort: "relevance" });

  const [hasSearched, setHasSearched] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [resultCount, setResultCount] = useState(0);
  const [searchedQuery, setSearchedQuery] = useState("");

  const [ingestingId, setIngestingId] = useState<string | null>(null);
  const [ingestErrors, setIngestErrors] = useState<Record<string, string>>({});

  const performSearch = useCallback(
    async (overrides?: Partial<FilterState & { q: string }>) => {
      const q = (overrides?.q ?? query).trim();
      const effective: FilterState = {
        category: overrides?.category ?? filters.category,
        dateRange: overrides?.dateRange ?? filters.dateRange,
        sort: overrides?.sort ?? filters.sort,
      };

      if (!q) {
        setValidationError("Enter a search term to look up arXiv.");
        return;
      }
      setValidationError(null);
      setLoading(true);
      setError(null);
      try {
        const res = await apiSearch({
          q,
          category: effective.category || undefined,
          date_range: effective.dateRange,
          sort: effective.sort,
          limit: 30,
        });
        setResults(res.results);
        setResultCount(res.count);
        setSearchedQuery(res.query);
        setHasSearched(true);
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Something went wrong while searching.");
      } finally {
        setLoading(false);
      }
    },
    [query, filters]
  );

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    void performSearch();
  }

  function updateFilter<K extends keyof FilterState>(key: K, value: FilterState[K]) {
    setFilters((prev) => ({ ...prev, [key]: value }));
    if (hasSearched) void performSearch({ [key]: value } as Partial<FilterState>);
  }

  async function handleIngest(item: SearchResultItem) {
    const paperId = item.paper.id;
    setIngestingId(paperId);
    setIngestErrors((prev) => {
      if (!(paperId in prev)) return prev;
      const next = { ...prev };
      delete next[paperId];
      return next;
    });
    try {
      const res = await apiIngest(item.paper.arxiv_id);
      openIngest({ paperId: res.paper_id, paperTitle: item.paper.title, jobId: res.job.id });
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "Could not start ingestion.";
      setIngestErrors((prev) => ({ ...prev, [paperId]: msg }));
    } finally {
      setIngestingId(null);
    }
  }

  return (
    <div style={{ maxWidth: 920, margin: "0 auto", padding: "30px 28px 60px" }}>
      <form onSubmit={handleSubmit} style={{ display: "flex", gap: 10, alignItems: "stretch" }}>
        <div style={{ flex: 1, position: "relative", display: "flex", alignItems: "center" }}>
          <svg
            style={{ position: "absolute", left: 15, pointerEvents: "none" }}
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--fg-subtle)"
            strokeWidth={2}
          >
            <circle cx="11" cy="11" r="7" />
            <line x1="16" y1="16" x2="21" y2="21" strokeLinecap="round" />
          </svg>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search arXiv — titles, authors, abstracts…"
            style={{
              width: "100%",
              padding: "14px 16px 14px 44px",
              fontSize: 15,
              border: "1px solid var(--border-strong)",
              borderRadius: 11,
              background: "var(--surface)",
              color: "var(--fg)",
              outline: "none",
            }}
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          style={{
            padding: "0 24px",
            fontSize: 14.5,
            fontWeight: 600,
            background: "var(--accent)",
            color: "var(--accent-fg)",
            border: "none",
            borderRadius: 11,
            opacity: loading ? 0.7 : 1,
          }}
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

      {validationError && (
        <div style={{ fontSize: 12.5, color: "var(--red)", marginTop: 8 }}>{validationError}</div>
      )}

      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 14, alignItems: "center" }}>
        <div style={filterChipStyle}>
          Category
          <select
            value={filters.category}
            onChange={(e) => updateFilter("category", e.target.value)}
            style={filterSelectStyle}
          >
            {CATEGORY_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        <div style={filterChipStyle}>
          Date
          <select
            value={filters.dateRange}
            onChange={(e) => updateFilter("dateRange", e.target.value as SearchDateRange)}
            style={filterSelectStyle}
          >
            {DATE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        <div style={filterChipStyle}>
          Sort by
          <select
            value={filters.sort}
            onChange={(e) => updateFilter("sort", e.target.value as SearchSort)}
            style={filterSelectStyle}
          >
            {SORT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        {hasSearched && !loading && !error && (
          <div style={{ marginLeft: "auto", fontSize: 12.5, color: "var(--fg-subtle)" }}>
            {resultCount} result{resultCount === 1 ? "" : "s"}
          </div>
        )}
      </div>

      {/* ---- states ---------------------------------------------------- */}

      {loading && (
        <div
          style={{
            marginTop: 20,
            display: "flex",
            flexDirection: "column",
            gap: 12,
          }}
        >
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              style={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: 13,
                padding: "17px 18px",
                height: 120,
                animation: "dv-pulse 1.4s ease-in-out infinite",
              }}
            />
          ))}
        </div>
      )}

      {!loading && error && (
        <div
          style={{
            marginTop: 20,
            background: "var(--red-soft)",
            border: "1px solid var(--red)",
            borderRadius: 13,
            padding: "18px 20px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 16,
          }}
        >
          <div>
            <div style={{ fontSize: 13.5, fontWeight: 600, color: "var(--red)" }}>Search failed</div>
            <div style={{ fontSize: 12.5, color: "var(--fg-muted)", marginTop: 3 }}>{error}</div>
          </div>
          <button
            onClick={() => void performSearch()}
            style={{
              flex: "none",
              fontSize: 13,
              fontWeight: 600,
              color: "var(--red)",
              background: "var(--surface)",
              border: "1px solid var(--red)",
              borderRadius: 8,
              padding: "7px 14px",
            }}
          >
            Retry
          </button>
        </div>
      )}

      {!loading && !error && !hasSearched && (
        <div
          style={{
            marginTop: 40,
            textAlign: "center",
            color: "var(--fg-subtle)",
            padding: "40px 20px",
          }}
        >
          <div style={{ fontSize: 14, fontWeight: 500, color: "var(--fg-muted)" }}>
            Search arXiv to get started
          </div>
          <div style={{ fontSize: 13, marginTop: 6, maxWidth: 420, marginLeft: "auto", marginRight: "auto" }}>
            Look up papers by title, author, or abstract, then ingest the ones you want in your workspace.
          </div>
        </div>
      )}

      {!loading && !error && hasSearched && results.length === 0 && (
        <div
          style={{
            marginTop: 40,
            textAlign: "center",
            color: "var(--fg-subtle)",
            padding: "40px 20px",
          }}
        >
          <div style={{ fontSize: 14, fontWeight: 500, color: "var(--fg-muted)" }}>
            No papers found for &ldquo;{searchedQuery}&rdquo;
          </div>
          <div style={{ fontSize: 13, marginTop: 6 }}>Try a different search term or adjust your filters.</div>
        </div>
      )}

      {!loading && !error && results.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 20 }}>
          {results.map((item) => {
            const p = item.paper;
            // Same two-signal split as the Library card: `item.ingested` means
            // the paper's content is indexed (that's what the "Ingested" badge
            // reports), `p.has_report` means a report row actually exists.
            // Report generation is the last ingestion stage and runs ~20 min on
            // a local model, so offering "Open report" off `ingested` would
            // navigate to a report that isn't written yet (GET /report answers
            // 409 "still generating").
            const canOpenReport = p.has_report;
            const ingesting = ingestingId === p.id;
            const ingestError = ingestErrors[p.id];
            const pdfHref = p.pdf_url ?? p.abs_url ?? null;

            return (
              <div
                key={p.id}
                style={{
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                  borderRadius: 13,
                  padding: "17px 18px",
                  boxShadow: "var(--shadow)",
                }}
              >
                <div style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
                      <a
                        href={p.abs_url ?? undefined}
                        target={p.abs_url ? "_blank" : undefined}
                        rel={p.abs_url ? "noopener noreferrer" : undefined}
                        style={{
                          fontSize: 16,
                          fontWeight: 600,
                          lineHeight: 1.35,
                          letterSpacing: "-.01em",
                          color: "var(--fg)",
                        }}
                      >
                        {p.title}
                      </a>
                      {p.ingested && (
                        <span
                          style={{
                            flex: "none",
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 4,
                            fontSize: 11,
                            fontWeight: 600,
                            color: "var(--green)",
                            background: "var(--green-soft)",
                            padding: "3px 8px",
                            borderRadius: 20,
                            marginTop: 2,
                          }}
                        >
                          <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--green)" }} />
                          Ingested
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: 13, color: "var(--fg-muted)", marginTop: 5 }}>
                      {formatAuthors(p.authors)} ·{" "}
                      <span className="dv-mono" style={{ fontSize: 12 }}>
                        {formatDate(p.published)}
                      </span>
                    </div>
                    {p.categories.length > 0 && (
                      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 9 }}>
                        {p.categories.map((c) => (
                          <span key={c} className="dv-mono" style={catTagStyle}>
                            {c}
                          </span>
                        ))}
                      </div>
                    )}
                    <p style={{ fontSize: 13.5, lineHeight: 1.6, color: "var(--fg-muted)", margin: "11px 0 0" }}>
                      {p.abstract}
                    </p>
                    <div style={{ display: "flex", gap: 9, marginTop: 14, alignItems: "center", flexWrap: "wrap" }}>
                      {p.ingested ? (
                        <button
                          onClick={() => navigate(`/report/${p.id}`)}
                          disabled={!canOpenReport}
                          title={
                            canOpenReport
                              ? "Open the generated report"
                              : "Report is still generating — it's the last ingestion stage and can " +
                                "take ~20 minutes on a local model. Chat already works."
                          }
                          style={{
                            fontSize: 13,
                            fontWeight: 600,
                            color: "var(--accent)",
                            background: "var(--accent-soft)",
                            border: "1px solid var(--accent-border)",
                            borderRadius: 8,
                            padding: "7px 14px",
                            opacity: canOpenReport ? 1 : 0.5,
                            cursor: canOpenReport ? "pointer" : "not-allowed",
                          }}
                        >
                          Open report
                        </button>
                      ) : (
                        <button
                          onClick={() => void handleIngest(item)}
                          disabled={ingesting}
                          style={{
                            fontSize: 13,
                            fontWeight: 600,
                            color: "var(--accent-fg)",
                            background: "var(--accent)",
                            border: "none",
                            borderRadius: 8,
                            padding: "7px 16px",
                            opacity: ingesting ? 0.7 : 1,
                          }}
                        >
                          {ingesting ? "Starting…" : "Ingest"}
                        </button>
                      )}
                      <a
                        href={pdfHref ?? undefined}
                        target={pdfHref ? "_blank" : undefined}
                        rel={pdfHref ? "noopener noreferrer" : undefined}
                        style={{
                          fontSize: 13,
                          fontWeight: 500,
                          color: pdfHref ? "var(--fg-muted)" : "var(--fg-subtle)",
                          background: "transparent",
                          border: "1px solid var(--border)",
                          borderRadius: 8,
                          padding: "7px 14px",
                          pointerEvents: pdfHref ? "auto" : "none",
                          opacity: pdfHref ? 1 : 0.5,
                        }}
                      >
                        arXiv PDF ↗
                      </a>
                      {ingestError && (
                        <span style={{ fontSize: 12, color: "var(--red)" }}>{ingestError}</span>
                      )}
                    </div>
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
