import { useEffect } from "react";
import { PDFS } from "../config";

function downloadName(filePath) {
  const base = filePath.split("/").pop() || "document.pdf";
  return base;
}

export default function DocumentsModal({ open, onClose, lang, t }) {
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      document.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);

  if (!open) return null;

  const dir = lang === "fa" ? "rtl" : "ltr";
  const list = [PDFS.en, PDFS.fa];

  return (
    <div
      className="docs-modal-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="docs-modal-title"
      onClick={onClose}
    >
      <div className="docs-modal" dir={dir} onClick={(e) => e.stopPropagation()}>
        <div className="docs-modal-header">
          <h2 id="docs-modal-title">{t.documents}</h2>
          <button type="button" className="docs-modal-close" onClick={onClose} aria-label={t.closeDocuments}>
            ×
          </button>
        </div>
        <p className="docs-modal-intro">{t.documentsIntro}</p>
        <ul className="docs-modal-list">
          {list.map((pdf) => (
            <li key={pdf.file} className="docs-modal-row">
              <div className="docs-modal-meta">
                <span className="docs-modal-label">{pdf.label}</span>
                <span className="docs-modal-pages">
                  {pdf.pages} {t.pdfPages}
                </span>
              </div>
              <a
                className="docs-modal-download"
                href={pdf.file}
                download={downloadName(pdf.file)}
              >
                {t.downloadPdf}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
