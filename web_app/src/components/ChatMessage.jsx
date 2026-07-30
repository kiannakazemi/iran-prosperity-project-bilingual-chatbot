import { useState, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import { PDFS } from "../config";
import { markdownToPlainText } from "../utils/markdownToPlainText";

/* ── Inline SVG icons ───────────────────────────────── */

const ICONS = {
  copy: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="9" y="9" width="13" height="13" rx="2" stroke="currentColor" strokeWidth="2" />
      <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" stroke="currentColor" strokeWidth="2" />
    </svg>
  ),
  check: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M20 6L9 17l-5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  edit: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M17 3a2.83 2.83 0 114 4L7.5 20.5 2 22l1.5-5.5L17 3z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  thumbUp: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3H14zM7 22H4a2 2 0 01-2-2v-7a2 2 0 012-2h3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  thumbDown: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M10 15v4a3 3 0 003 3l4-9V2H6.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3H10zM17 2h3a2 2 0 012 2v7a2 2 0 01-2 2h-3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  regenerate: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M1 4v6h6M23 20v-6h-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  warning: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 9v4M12 17h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
};

/* ── Source helpers (unchanged) ──────────────────────── */

function mergeSources(sources) {
  if (!sources?.length) return [];

  // Key by both section name AND language so Persian/English sections don't merge
  const bySection = {};
  for (const s of sources) {
    const key = `${s.language ?? "en"}::${s.white_paper}`;
    if (!bySection[key]) bySection[key] = { white_paper: s.white_paper, language: s.language ?? "en", pages: [] };
    bySection[key].pages.push(s.page_start ?? 0, s.page_end ?? s.page_start ?? 0);
  }

  const merged = [];
  for (const { white_paper, language, pages } of Object.values(bySection)) {
    const unique = [...new Set(pages)].sort((a, b) => a - b);
    let rangeStart = unique[0];
    let rangeEnd = unique[0];

    for (let i = 1; i < unique.length; i++) {
      if (unique[i] <= rangeEnd + 1) {
        rangeEnd = unique[i];
      } else {
        merged.push({ white_paper, language, page_start: rangeStart, page_end: rangeEnd });
        rangeStart = unique[i];
        rangeEnd = unique[i];
      }
    }
    merged.push({ white_paper, language, page_start: rangeStart, page_end: rangeEnd });
  }
  return merged;
}

function SourceBadge({ source, t, onOpenPdf }) {
  const ps = (source.page_start ?? 0) + 1;
  const pe = (source.page_end ?? source.page_start ?? 0) + 1;
  const pageLabel = ps === pe ? `${t.page}${ps}` : `${t.pages}${ps}-${pe}`;

  const pdfKey = source.language === "fa" ? "fa" : "en";
  const pdf = PDFS[pdfKey] || PDFS.en;

  return (
    <button
      className="source-badge"
      onClick={() => onOpenPdf(pdf.file, ps)}
      title={`${source.white_paper} — ${pageLabel}`}
    >
      <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
        <path
          d="M4 1h6l4 4v10H4a1 1 0 01-1-1V2a1 1 0 011-1z"
          stroke="currentColor"
          strokeWidth="1.5"
        />
        <path d="M10 1v4h4" stroke="currentColor" strokeWidth="1.5" />
      </svg>
      <span>{source.white_paper}</span>
      <span className="source-page">{pageLabel}</span>
    </button>
  );
}

/* ── Main component ─────────────────────────────────── */

export default function ChatMessage({
  message,
  index,
  lang,
  t,
  onOpenPdf,
  onEditMessage,
  onRegenerate,
  isLastAssistant,
  isStreaming: chatIsStreaming,
}) {
  const isUser = message.role === "user";

  const [copied, setCopied] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState("");
  const [liked, setLiked] = useState(null); // null | "up" | "down"

  const cleanSources = !isUser ? mergeSources(message.sources) : [];

  /* ── Copy handler ──────────────────────────────────── */

  const handleCopy = useCallback(async () => {
    if (!message.content) return;
    try {
      await navigator.clipboard.writeText(markdownToPlainText(message.content));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* ignore */
    }
  }, [message.content]);

  /* ── Edit handlers (user messages) ─────────────────── */

  const handleStartEdit = useCallback(() => {
    setEditText(message.content);
    setIsEditing(true);
  }, [message.content]);

  const handleCancelEdit = useCallback(() => {
    setIsEditing(false);
    setEditText("");
  }, []);

  const handleSaveEdit = useCallback(() => {
    const trimmed = editText.trim();
    if (!trimmed) return;
    setIsEditing(false);
    onEditMessage(index, trimmed);
  }, [editText, index, onEditMessage]);

  const handleEditKeyDown = useCallback(
    (e) => {
      if (e.key === "Escape") {
        handleCancelEdit();
      } else if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSaveEdit();
      }
    },
    [handleCancelEdit, handleSaveEdit]
  );

  /* ── User message ──────────────────────────────────── */

  if (isUser) {
    if (isEditing) {
      return (
        <div className="message message-user">
          <div className="user-edit-wrap">
            <textarea
              className="user-edit-textarea"
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              onKeyDown={handleEditKeyDown}
              dir={lang === "fa" ? "rtl" : "ltr"}
              autoFocus
            />
            <div className="user-edit-actions">
              <button
                type="button"
                className="edit-save-btn"
                onClick={handleSaveEdit}
                disabled={!editText.trim()}
              >
                {t.saveEdit}
              </button>
              <button
                type="button"
                className="edit-cancel-btn"
                onClick={handleCancelEdit}
              >
                {t.cancelEdit}
              </button>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="message message-user">
        <div className="user-bubble-wrap">
          <div className="user-bubble">{message.content}</div>
          {message.content && (
            <div className="msg-actions msg-actions--user">
              <button
                type="button"
                className={`msg-action-btn${copied ? " msg-action-btn--active" : ""}`}
                onClick={handleCopy}
                title={copied ? t.copied : t.copy}
              >
                {copied ? ICONS.check : ICONS.copy}
              </button>
              <button
                type="button"
                className="msg-action-btn"
                onClick={handleStartEdit}
                title={t.edit}
                disabled={chatIsStreaming}
              >
                {ICONS.edit}
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  /* ── Assistant message ─────────────────────────────── */

  // Unified error card. Any backend failure (connection refused, HTTP 5xx,
  // missing body, malformed stream, empty response, explicit server error)
  // sets `isError: true` upstream, and we render the same compact card
  // instead of the regular markdown body / sources / action buttons.
  if (message.isError) {
    return (
      <div className="message message-assistant">
        <div className="error-card" role="alert" aria-live="polite">
          <span className="error-card-icon">{ICONS.warning}</span>
          <div className="error-card-text">
            <div className="error-card-title">{t.errorTitle}</div>
            <div className="error-card-body">{t.errorBody}</div>
          </div>
        </div>
      </div>
    );
  }

  const canShowActions = Boolean(message.content?.trim()) && !message.streaming;

  return (
    <div className="message message-assistant">
      <div className="assistant-answer markdown-body">
        <ReactMarkdown>{message.content}</ReactMarkdown>
        {message.streaming && <span className="cursor-blink">▊</span>}
      </div>

      {cleanSources.length > 0 && !message.streaming && (
        <div className="message-sources">
          <span className="sources-label">{t.sources}:</span>
          {cleanSources.map((s, i) => (
            <SourceBadge key={i} source={s} t={t} onOpenPdf={onOpenPdf} />
          ))}
        </div>
      )}

      {canShowActions && (
        <div className="msg-actions msg-actions--assistant">
          <button
            type="button"
            className={`msg-action-btn${copied ? " msg-action-btn--active" : ""}`}
            onClick={handleCopy}
            title={copied ? t.copied : t.copy}
          >
            {copied ? ICONS.check : ICONS.copy}
          </button>
          <button
            type="button"
            className={`msg-action-btn${liked === "up" ? " msg-action-btn--active" : ""}`}
            onClick={() => setLiked((prev) => (prev === "up" ? null : "up"))}
            title={t.like}
          >
            {ICONS.thumbUp}
          </button>
          <button
            type="button"
            className={`msg-action-btn${liked === "down" ? " msg-action-btn--active" : ""}`}
            onClick={() => setLiked((prev) => (prev === "down" ? null : "down"))}
            title={t.dislike}
          >
            {ICONS.thumbDown}
          </button>
          {isLastAssistant && (
            <button
              type="button"
              className="msg-action-btn"
              onClick={onRegenerate}
              title={t.regenerate}
              disabled={chatIsStreaming}
            >
              {ICONS.regenerate}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
