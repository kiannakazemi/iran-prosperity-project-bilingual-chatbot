const IPP_URL = {
  en: "https://www.iranprosperityproject.com/en",
  fa: "https://www.iranprosperityproject.com/fa",
};

export default function Sidebar({
  lang,
  t,
  chatHistory,
  activeChatId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  onOpenDocuments,
}) {
  return (
    <aside className="sidebar">
      <nav className="sidebar-nav">
        <button type="button" className="nav-item new-chat-btn" onClick={onNewChat}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M8 2v12M2 8h12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
          {t.newChat}
        </button>
        <button type="button" className="nav-item nav-docs" onClick={onOpenDocuments}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M4 2h6l3 3v9H4a1 1 0 01-1-1V3a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.4" />
            <path d="M10 2v3h3" stroke="currentColor" strokeWidth="1.4" />
          </svg>
          {t.documents}
        </button>
      </nav>

      <div className="sidebar-divider" />

      <div className="sidebar-section-label">{t.chats}</div>
      <div className="chat-list">
        {chatHistory.map((chat) => (
          <div
            key={chat.id}
            className={`chat-item ${chat.id === activeChatId ? "active" : ""}`}
            onClick={() => onSelectChat(chat.id)}
          >
            <span className="chat-title">{chat.title || t.newChat}</span>
            <button
              className="delete-btn"
              onClick={(e) => {
                e.stopPropagation();
                onDeleteChat(chat.id);
              }}
              title={t.deleteChat}
            >
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            </button>
          </div>
        ))}
      </div>

      <div className="sidebar-footer">
        <a
          href={IPP_URL[lang] || IPP_URL.en}
          target="_blank"
          rel="noopener noreferrer"
          className="footer-link"
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
            <path d="M6 3H3v10h10v-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
            <path d="M9 2h5v5M14 2L7 9" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          iranprosperityproject.com
        </a>
      </div>
    </aside>
  );
}
