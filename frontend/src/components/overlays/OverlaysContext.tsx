import React, { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { Citation, MediaRef } from "../../api/types";
import { IngestionModal } from "./IngestionModal";
import { FigureLightbox } from "./FigureLightbox";
import { CitationPopover } from "./CitationPopover";

export interface IngestOverlayState {
  paperId: string;
  paperTitle: string;
  jobId: string;
}

export interface OverlaysApi {
  /** Current ingestion-modal target, or null when closed. */
  ingest: IngestOverlayState | null;
  /** Open the ingestion modal for a just-started (or in-progress) job; the
   *  modal polls GET /jobs/{jobId} itself via client.pollJob. */
  openIngest: (args: IngestOverlayState) => void;
  closeIngest: () => void;

  /** Current lightbox figure/table, or null when closed. */
  lightbox: MediaRef | null;
  openLightbox: (media: MediaRef) => void;
  closeLightbox: () => void;

  /** Current citation popover target, or null when closed. */
  citation: Citation | null;
  openCitation: (citation: Citation) => void;
  closeCitation: () => void;
}

const OverlaysContext = createContext<OverlaysApi | undefined>(undefined);

export function OverlaysProvider({ children }: { children: React.ReactNode }) {
  const [ingest, setIngest] = useState<IngestOverlayState | null>(null);
  const [lightbox, setLightbox] = useState<MediaRef | null>(null);
  const [citation, setCitation] = useState<Citation | null>(null);

  const openIngest = useCallback((args: IngestOverlayState) => setIngest(args), []);
  const closeIngest = useCallback(() => setIngest(null), []);
  const openLightbox = useCallback((media: MediaRef) => setLightbox(media), []);
  const closeLightbox = useCallback(() => setLightbox(null), []);
  const openCitation = useCallback((c: Citation) => setCitation(c), []);
  const closeCitation = useCallback(() => setCitation(null), []);

  const value = useMemo<OverlaysApi>(
    () => ({
      ingest,
      openIngest,
      closeIngest,
      lightbox,
      openLightbox,
      closeLightbox,
      citation,
      openCitation,
      closeCitation,
    }),
    [ingest, openIngest, closeIngest, lightbox, openLightbox, closeLightbox, citation, openCitation, closeCitation]
  );

  return (
    <OverlaysContext.Provider value={value}>
      {children}
      <IngestionModal />
      <FigureLightbox />
      <CitationPopover />
    </OverlaysContext.Provider>
  );
}

export function useOverlays(): OverlaysApi {
  const ctx = useContext(OverlaysContext);
  if (!ctx) throw new Error("useOverlays must be used within an OverlaysProvider");
  return ctx;
}
