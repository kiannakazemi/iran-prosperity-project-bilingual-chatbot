import { useState, useRef, useEffect, useCallback } from "react";

/** Max textarea height (px); then scroll inside — same idea as ChatGPT / Claude */
const TEXTAREA_MAX_HEIGHT = 200;

function adjustTextareaHeight(el) {
  if (!el) return;
  el.style.height = "auto";
  const scrollH = el.scrollHeight;
  const next = Math.min(scrollH, TEXTAREA_MAX_HEIGHT);
  el.style.height = `${next}px`;
  el.style.overflowY = scrollH > TEXTAREA_MAX_HEIGHT ? "auto" : "hidden";
}

export default function ChatInput({ lang, t, isStreaming, onSend, chatId }) {
  const [text, setText] = useState("");
  const inputRef = useRef(null);

  const syncHeight = useCallback(() => {
    adjustTextareaHeight(inputRef.current);
  }, []);

  useEffect(() => {
    inputRef.current?.focus();
  }, [isStreaming, chatId]);

  useEffect(() => {
    syncHeight();
  }, [text, syncHeight]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!text.trim() || isStreaming) return;
    onSend(text.trim());
    setText("");
    setTimeout(() => {
      inputRef.current?.focus();
      adjustTextareaHeight(inputRef.current);
    }, 0);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="chat-input-wrapper">
      <form className="chat-input-form" onSubmit={handleSubmit}>
        <textarea
          ref={inputRef}
          className="chat-input"
          placeholder={t.askPlaceholder}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          dir={lang === "fa" ? "rtl" : "ltr"}
          aria-label={t.askPlaceholder}
        />
        <button
          type="submit"
          className="send-btn"
          disabled={!text.trim() || isStreaming}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path
              d="M12 19V5M5 12l7-7 7 7"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </form>
      <p className="disclaimer">{t.disclaimer}</p>
    </div>
  );
}
