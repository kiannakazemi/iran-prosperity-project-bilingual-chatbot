import { Document, Page, pdfjs } from "react-pdf";
import { PDFS } from "../config";

pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

function PdfThumb({ file }) {
  return (
    <Document file={file} loading={<div className="thumb-placeholder" />}>
      <Page
        pageNumber={1}
        width={80}
        renderAnnotationLayer={false}
        renderTextLayer={false}
      />
    </Document>
  );
}

export default function WelcomeScreen({ lang, t, onOpenPdf }) {
  return (
    <div className="welcome">
      <div className="welcome-header">
        <div className="welcome-icon-ring">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"
              stroke="var(--ipp-accent)" strokeWidth="1.5" fill="var(--ipp-accent-muted)" />
          </svg>
        </div>
        <h1 className="welcome-title">{t.welcome}</h1>
        <p className="welcome-sub">{t.welcomeSub}</p>
      </div>

      <div className="welcome-label">{t.documents}</div>

      <div className="doc-cards">
        {Object.entries(PDFS).map(([key, pdf]) => (
          <button
            key={key}
            className="doc-card"
            onClick={() => onOpenPdf(pdf.file, 1)}
          >
            <div className="doc-card-thumb">
              <PdfThumb file={pdf.file} />
            </div>
            <div className="doc-card-body">
              <div className="doc-card-meta">PDF / {pdf.pages} {t.pdfPages}</div>
              <div className="doc-card-name">{pdf.label}</div>
              <p className="doc-card-desc">
                {key === "en"
                  ? "The Emergency Phase Booklet from the Iran Prosperity Project."
                  : "کتابچه مرحله اضطراری پروژه شکوفایی ایران."}
              </p>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
