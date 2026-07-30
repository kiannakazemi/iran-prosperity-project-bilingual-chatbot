import { useState, useRef, useEffect, useCallback } from "react";
import { Document, Page, pdfjs } from "react-pdf";
// Text layer styles position the (invisible) selectable text spans exactly
// over the rasterized canvas so users can highlight and copy real text from
// the PDF. Without this CSS the spans land at default positions and the
// selection box no longer matches the visible glyphs.
import "react-pdf/dist/Page/TextLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

const BASE_PAGE_WIDTH = 440; // Fallback only; real base is measured from the panel.
const ZOOM_MIN = 0.5;
const ZOOM_MAX = 2.5;
const ZOOM_STEP = 0.25;
// Default aspect ratio (width / height) — rough A4, used until we've
// measured the real page aspect from the PDF metadata.
const DEFAULT_PAGE_ASPECT = 595 / 842;
// Extra vertical space per page wrapper for the "Page N" label + margin.
const PAGE_LABEL_SPACE = 24;
// Breathing room inside the pdf-body when fit-sizing a page: a bit of
// horizontal margin so the canvas doesn't touch the scrollbar, and
// enough vertical room for the container's top/bottom padding, the
// page-number label, and the inter-page margin so one page can fill
// the viewport without the next page peeking in.
const FIT_HORIZONTAL_BUFFER = 32;
const FIT_VERTICAL_BUFFER = 60;

// Default zoom — 100% leaves big empty borders that shrink the actual page
// content; 125% fills the panel comfortably without users having to click
// the + button each time they open the PDF.
const DEFAULT_ZOOM = 1.25;

export default function PdfPanel({ file, pageNumber, barTitle, onClose }) {
  const [numPages, setNumPages] = useState(null);
  const [currentPage, setCurrentPage] = useState(pageNumber || 1);
  const [zoom, setZoom] = useState(DEFAULT_ZOOM);
  // Controlled value for the editable page-number input. It tracks
  // `currentPage` while the field is not focused; while the user is typing,
  // it holds whatever they've typed (so partial values like "1" -> "17"
  // don't get yanked back to currentPage on each keystroke).
  const [pageInput, setPageInput] = useState(String(pageNumber || 1));
  const [pageInputFocused, setPageInputFocused] = useState(false);
  // Per-page aspect ratios (width / height) harvested from the PDF's own
  // metadata. Having these means every page wrapper is pre-sized to its
  // final height from the moment the document mounts, so the document's
  // total height doesn't change as individual canvases finish rendering.
  // That stability is what lets scrollIntoView actually land on the right
  // page — without it, later pages' offsetTop drifts while earlier pages
  // are still rendering (which is why target page 159 was landing near
  // page 18 in the longer Persian PDF).
  const [pageAspects, setPageAspects] = useState({});
  // Dynamically computed "100% zoom" page width — sized so one page
  // fits the pdf-body viewport in both dimensions. Recomputed whenever
  // the panel is resized or the PDF's aspect is discovered.
  const [fitWidth, setFitWidth] = useState(BASE_PAGE_WIDTH);
  const containerRef = useRef(null);
  const pageRefs = useRef({});

  // Measure the pdf-body and compute the largest page width that lets
  // one full page fit the viewport without the neighbour pages
  // poking into view. A ResizeObserver keeps this fresh as the panel
  // is dragged wider/narrower or the window is resized.
  useEffect(() => {
    const body = containerRef.current;
    if (!body) return;
    const measure = () => {
      const cw = body.clientWidth;
      const ch = body.clientHeight;
      if (!cw || !ch) return;
      const aspect = pageAspects[1] ?? DEFAULT_PAGE_ASPECT;
      const availW = cw - FIT_HORIZONTAL_BUFFER;
      const availH = ch - FIT_VERTICAL_BUFFER;
      if (availW <= 0 || availH <= 0) return;
      // Width bounded by both the available width and the width
      // derived from available height via the page aspect ratio.
      const w = Math.floor(Math.min(availW, availH * aspect));
      if (w > 0) setFitWidth(w);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(body);
    return () => ro.disconnect();
  }, [pageAspects]);

  const onDocumentLoadSuccess = useCallback(async (pdf) => {
    // Read every page's dimensions in parallel. getPage() reads metadata
    // only (no rasterization), so even for 200-page PDFs this resolves in
    // a few hundred ms at most.
    try {
      const aspects = await Promise.all(
        Array.from({ length: pdf.numPages }, async (_, i) => {
          const page = await pdf.getPage(i + 1);
          const vp = page.getViewport({ scale: 1 });
          return vp.width / vp.height;
        })
      );
      const map = {};
      aspects.forEach((a, i) => { map[i + 1] = a; });
      setPageAspects(map);
    } catch {
      /* fall back to DEFAULT_PAGE_ASPECT for every page */
    }
    // Set numPages *after* aspects so that the first render with page
    // wrappers already has the correct pre-sized heights.
    setNumPages(pdf.numPages);
  }, []);

  // Scroll to the target page. Because every wrapper is pre-sized via
  // pageAspects, the document layout is stable enough that a smooth
  // scroll lands on (or very near) the right page. But the reserved
  // heights are only *approximately* equal to the rendered heights —
  // small rounding differences in the page-number label area can
  // accumulate to tens of pixels of drift for pages deep in the
  // document. So after the animation we do an instantaneous
  // fine-tune to snap exactly onto the target's top.
  useEffect(() => {
    if (!pageNumber || !numPages) return;

    let cancelled = false;
    let attempts = 0;
    const maxAttempts = 60; // ~3s of retrying is plenty once layout is stable

    function tryScroll() {
      if (cancelled) return;
      const el = pageRefs.current[pageNumber];
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "start" });
        setCurrentPage(pageNumber);
        // Fine-tune: after the smooth-scroll animation has had time to
        // finish, measure the target's actual position relative to the
        // scroll container and snap to it if there's any residual
        // drift. We retry a few times because canvases may keep
        // rendering briefly after the scroll lands, which could nudge
        // the target by another pixel or two.
        const container = containerRef.current;
        if (!container) return;
        const snapToTarget = (retriesLeft) => {
          if (cancelled) return;
          const target = pageRefs.current[pageNumber];
          if (!target) return;
          const cRect = container.getBoundingClientRect();
          const eRect = target.getBoundingClientRect();
          const delta = eRect.top - cRect.top;
          if (Math.abs(delta) > 1) {
            container.scrollTop += delta;
          }
          if (retriesLeft > 0) {
            setTimeout(() => snapToTarget(retriesLeft - 1), 250);
          }
        };
        // Start correcting after the browser's smooth scroll has had
        // time to finish (~500–600ms is typical).
        setTimeout(() => snapToTarget(3), 650);
        return;
      }
      attempts++;
      if (attempts < maxAttempts) {
        setTimeout(() => requestAnimationFrame(tryScroll), 50);
      }
    }

    requestAnimationFrame(tryScroll);
    return () => { cancelled = true; };
  }, [pageNumber, numPages]);

  // Track which page is visible while scrolling
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !numPages) return;

    let ticking = false;
    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        const containerTop = container.scrollTop;
        const containerMid = containerTop + container.clientHeight / 3;
        let closest = 1;
        let closestDist = Infinity;
        for (let p = 1; p <= numPages; p++) {
          const el = pageRefs.current[p];
          if (!el) continue;
          const dist = Math.abs(el.offsetTop - containerMid);
          if (dist < closestDist) {
            closestDist = dist;
            closest = p;
          }
        }
        setCurrentPage(closest);
        ticking = false;
      });
    };
    container.addEventListener("scroll", onScroll, { passive: true });
    return () => container.removeEventListener("scroll", onScroll);
  }, [numPages]);

  const goToPage = useCallback((p) => {
    const el = pageRefs.current[p];
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
      setCurrentPage(p);
    }
  }, []);

  // Keep the input mirror in sync with the page that's actually showing,
  // but ONLY when the user isn't actively typing into it. Otherwise we'd
  // overwrite their keystrokes every time they scrolled even a pixel.
  useEffect(() => {
    if (!pageInputFocused) {
      setPageInput(String(currentPage));
    }
  }, [currentPage, pageInputFocused]);

  // Parse the input, clamp to [1, numPages], jump there, and re-sync the
  // visible text. Called on Enter and on blur. Invalid input (empty, NaN,
  // out of range and unclampable) silently reverts to the current page.
  const commitPageInput = useCallback(() => {
    const n = parseInt(pageInput, 10);
    if (Number.isFinite(n) && numPages) {
      const clamped = Math.max(1, Math.min(numPages, n));
      if (clamped !== currentPage) {
        goToPage(clamped);
      }
      setPageInput(String(clamped));
    } else {
      setPageInput(String(currentPage));
    }
  }, [pageInput, numPages, currentPage, goToPage]);

  const handlePageInputKeyDown = useCallback((e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commitPageInput();
      e.currentTarget.blur();
    } else if (e.key === "Escape") {
      // Cancel: revert and unfocus.
      setPageInput(String(currentPage));
      e.currentTarget.blur();
    }
  }, [commitPageInput, currentPage]);

  const zoomIn = useCallback(() => {
    setZoom((z) => Math.min(ZOOM_MAX, Math.round((z + ZOOM_STEP) * 100) / 100));
  }, []);

  const zoomOut = useCallback(() => {
    setZoom((z) => Math.max(ZOOM_MIN, Math.round((z - ZOOM_STEP) * 100) / 100));
  }, []);

  const pageWidth = Math.round(fitWidth * zoom);

  const fileName = file?.split("/").pop()?.replace(/_/g, " ").replace(".pdf", "") || "Document";

  return (
    <div className="pdf-panel">
      <div className="pdf-header">
        <div className="pdf-header-brand">
          <img
            src={`${import.meta.env.BASE_URL}images/ipp-azadi-tower-logo.png`}
            alt=""
            className="pdf-header-logo"
            width={28}
            height={28}
          />
          <div className="pdf-header-titles">
            <div className="pdf-bar-title">{barTitle}</div>
            <div className="pdf-file-subtitle" title={fileName}>
              {fileName}
            </div>
          </div>
        </div>
        <button type="button" className="pdf-close" onClick={onClose} aria-label="Close PDF panel">
          ×
        </button>
      </div>

      <div className="pdf-nav">
        <div className="pdf-nav-spacer" aria-hidden />
        <div className="pdf-nav-pages">
          <button
            type="button"
            disabled={currentPage <= 1}
            onClick={() => goToPage(currentPage - 1)}
            aria-label="Previous page"
          >
            ‹
          </button>
          <span className="pdf-page-indicator">
            <span className="pdf-page-label">Page</span>
            <input
              type="text"
              inputMode="numeric"
              className="pdf-page-input"
              value={pageInput}
              onChange={(e) => setPageInput(e.target.value.replace(/[^\d]/g, ""))}
              onFocus={(e) => {
                setPageInputFocused(true);
                // Select all so the user can immediately type a new number
                // without having to clear the field first.
                e.target.select();
              }}
              onBlur={() => {
                setPageInputFocused(false);
                commitPageInput();
              }}
              onKeyDown={handlePageInputKeyDown}
              aria-label="Current page — type a number and press Enter to jump"
              disabled={!numPages}
            />
            {numPages ? <span className="pdf-page-total">/ {numPages}</span> : null}
          </span>
          <button
            type="button"
            disabled={currentPage >= (numPages || 1)}
            onClick={() => goToPage(currentPage + 1)}
            aria-label="Next page"
          >
            ›
          </button>
        </div>
        <div className="pdf-nav-zoom">
          <button
            type="button"
            className="pdf-zoom-btn"
            disabled={zoom <= ZOOM_MIN}
            onClick={zoomOut}
            aria-label="Zoom out"
          >
            −
          </button>
          <span className="pdf-zoom-pct" title="Zoom level">
            {Math.round(zoom * 100)}%
          </span>
          <button
            type="button"
            className="pdf-zoom-btn"
            disabled={zoom >= ZOOM_MAX}
            onClick={zoomIn}
            aria-label="Zoom in"
          >
            +
          </button>
        </div>
      </div>

      <div className="pdf-body" ref={containerRef}>
        <Document
          file={file}
          onLoadSuccess={onDocumentLoadSuccess}
          loading={<div className="pdf-loading">Loading PDF…</div>}
        >
          {numPages && Array.from({ length: numPages }, (_, i) => i + 1).map((p) => {
            const aspect = pageAspects[p] ?? DEFAULT_PAGE_ASPECT;
            const reservedHeight = Math.round(pageWidth / aspect) + PAGE_LABEL_SPACE;
            return (
              <div
                key={p}
                ref={(el) => { pageRefs.current[p] = el; }}
                className="pdf-page-wrapper"
                style={{ minHeight: reservedHeight }}
              >
                <Page
                  pageNumber={p}
                  width={pageWidth}
                  renderAnnotationLayer={false}
                  /* renderTextLayer=true draws an invisible selectable text
                     layer on top of the canvas, so users can highlight and
                     copy real text from the PDF. */
                  renderTextLayer={true}
                />
                <div className="pdf-page-number">Page {p}</div>
              </div>
            );
          })}
        </Document>
      </div>
    </div>
  );
}
