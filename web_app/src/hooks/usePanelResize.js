import { useState, useCallback } from "react";

const STORAGE_SIDEBAR = "ipp-sidebar-width";
const STORAGE_PDF = "ipp-pdf-width";

const SIDEBAR = { min: 200, max: 520, def: 280 };
const PDF = { min: 280, max: 900, def: 480 };

function readStored(key, fallback, min, max) {
  try {
    const v = localStorage.getItem(key);
    if (v != null) {
      const n = parseInt(v, 10);
      if (!Number.isNaN(n)) return Math.min(max, Math.max(min, n));
    }
  } catch {
    /* ignore */
  }
  return fallback;
}

/**
 * Draggable panel widths for sidebar + PDF; persists to localStorage.
 * When isRtl is true (Persian), mouse-delta signs are flipped so dragging
 * still widens/narrows the panel under the handle (flex order mirrors in RTL).
 */
export function usePanelResize(isRtl = false) {
  const [sidebarWidth, setSidebarWidth] = useState(() =>
    readStored(STORAGE_SIDEBAR, SIDEBAR.def, SIDEBAR.min, SIDEBAR.max)
  );
  const [pdfPanelWidth, setPdfPanelWidth] = useState(() =>
    readStored(STORAGE_PDF, PDF.def, PDF.min, PDF.max)
  );

  const onSidebarResizeStart = useCallback(
    (e) => {
      e.preventDefault();
      const startX = e.clientX;
      const startW = sidebarWidth;
      let lastW = startW;

      const move = (ev) => {
        const dx = ev.clientX - startX;
        const delta = isRtl ? -dx : dx;
        lastW = Math.min(SIDEBAR.max, Math.max(SIDEBAR.min, startW + delta));
        setSidebarWidth(lastW);
      };

      const up = () => {
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        try {
          localStorage.setItem(STORAGE_SIDEBAR, String(lastW));
        } catch {
          /* ignore */
        }
      };

      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    },
    [sidebarWidth, isRtl]
  );

  const onPdfResizeStart = useCallback(
    (e) => {
      e.preventDefault();
      const startX = e.clientX;
      const startW = pdfPanelWidth;
      let lastW = startW;

      const move = (ev) => {
        const dx = ev.clientX - startX;
        const delta = isRtl ? dx : -dx;
        lastW = Math.min(PDF.max, Math.max(PDF.min, startW + delta));
        setPdfPanelWidth(lastW);
      };

      const up = () => {
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        try {
          localStorage.setItem(STORAGE_PDF, String(lastW));
        } catch {
          /* ignore */
        }
      };

      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    },
    [pdfPanelWidth, isRtl]
  );

  return {
    sidebarWidth,
    pdfPanelWidth,
    onSidebarResizeStart,
    onPdfResizeStart,
  };
}
