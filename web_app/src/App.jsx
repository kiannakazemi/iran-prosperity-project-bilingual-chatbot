import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { i18n, API_BASE } from "./config";
import { useChat } from "./hooks/useChat";
import { loadSessionChats, saveSessionChats } from "./utils/sessionChatStore";
import { usePanelResize } from "./hooks/usePanelResize";
import Sidebar from "./components/Sidebar";
import WelcomeScreen from "./components/WelcomeScreen";
import ChatMessage from "./components/ChatMessage";
import ChatInput from "./components/ChatInput";
import PdfPanel from "./components/PdfPanel";
import DocumentsModal from "./components/DocumentsModal";
import LangToggle from "./components/LangToggle";
import "./styles/theme.css";
import "./styles/app.css";

export default function App() {
  const initialSession = useMemo(() => loadSessionChats(), []);
  const [lang] = useState(() => {
    // Path-based routing: /assistant/en  → English
    //                     /assistant/fa  → Persian
    //                     /assistant/    → defaults to English
    // We strip the configured Vite base prefix (e.g. "/assistant/") off the
    // pathname and look at the first remaining segment.
    const base = import.meta.env.BASE_URL;
    const path = window.location.pathname;
    const relative = path.startsWith(base) ? path.slice(base.length) : path.replace(/^\//, "");
    const segment = relative.split("/")[0];
    return segment === "fa" ? "fa" : "en";
  });
  const [chatHistory, setChatHistory] = useState(initialSession.chatHistory);
  const [activeChatId, setActiveChatId] = useState(initialSession.activeChatId);
  const [pdfView, setPdfView] = useState(null);
  const [documentsOpen, setDocumentsOpen] = useState(false);
  const chatEndRef = useRef(null);
  const messagesCache = useRef({ ...initialSession.messagesMap });

  const t = i18n[lang];

  const activePack =
    initialSession.activeChatId &&
    initialSession.messagesMap[initialSession.activeChatId];
  const {
    messages, isStreaming, conversationId,
    sendMessage, editAndResend, regenerate, clearChat, loadChat,
  } = useChat({
    initialMessages: activePack?.messages ?? [],
    initialConversationId: activePack?.conversationId ?? null,
  });

  const isRtl = lang === "fa";
  const {
    sidebarWidth,
    pdfPanelWidth,
    onSidebarResizeStart,
    onPdfResizeStart,
  } = usePanelResize(isRtl);

  // Auto-scroll on new messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Keep in-memory cache + sessionStorage (tab session only; survives refresh)
  useEffect(() => {
    if (activeChatId) {
      messagesCache.current[activeChatId] = {
        messages: [...messages],
        conversationId,
      };
    }
    const validIds = new Set(chatHistory.map((c) => c.id));
    const pruned = {};
    for (const id of validIds) {
      const entry = messagesCache.current[id];
      if (entry) pruned[id] = entry;
    }
    messagesCache.current = pruned;
    saveSessionChats({
      chatHistory,
      activeChatId,
      messagesMap: pruned,
    });
  }, [messages, activeChatId, conversationId, chatHistory]);

  // Sync conversationId back to chat history
  useEffect(() => {
    if (conversationId && activeChatId) {
      setChatHistory((prev) =>
        prev.map((c) =>
          c.id === activeChatId ? { ...c, conversationId } : c
        )
      );
    }
  }, [conversationId, activeChatId]);

  // Auto-title: ask the LLM to generate a short title after first exchange
  useEffect(() => {
    if (messages.length < 2 || !activeChatId) return;
    const firstUser = messages.find((m) => m.role === "user");
    const firstAssistant = messages.find(
      (m) => m.role === "assistant" && !m.streaming && m.content
    );
    if (!firstUser || !firstAssistant) return;

    const chat = chatHistory.find((c) => c.id === activeChatId);
    if (!chat || chat.title) return;

    // Set a temporary title immediately so we don't fire twice
    setChatHistory((prev) =>
      prev.map((c) =>
        c.id === activeChatId ? { ...c, title: "…" } : c
      )
    );

    // Ask the backend to generate a proper title
    fetch(`${API_BASE}/api/generate-title`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: firstUser.content,
        answer: firstAssistant.content.slice(0, 300),
      }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.title) {
          setChatHistory((prev) =>
            prev.map((c) =>
              c.id === activeChatId ? { ...c, title: data.title } : c
            )
          );
        }
      })
      .catch(() => {
        // Fallback: use first few words of the question
        setChatHistory((prev) =>
          prev.map((c) =>
            c.id === activeChatId && c.title === "…"
              ? { ...c, title: firstUser.content.slice(0, 40) }
              : c
          )
        );
      });
  }, [messages, activeChatId, chatHistory]);

  const handleNewChat = useCallback(() => {
    setPdfView(null);
    // Don't create another chat if we're already on an empty conversation
    if (activeChatId && messages.length === 0) return;

    const first = chatHistory[0];
    if (first && !activeChatId) {
      const cached = messagesCache.current[first.id];
      const firstEmpty = !cached || cached.messages.length === 0;
      if (firstEmpty) {
        if (cached) loadChat(cached.messages, cached.conversationId);
        else clearChat();
        setActiveChatId(first.id);
        return;
      }
    }

    clearChat();
    const newChat = {
      id: crypto.randomUUID(),
      title: "",
      conversationId: null,
    };
    setChatHistory((prev) => [newChat, ...prev]);
    setActiveChatId(newChat.id);
  }, [clearChat, loadChat, activeChatId, messages.length, chatHistory]);

  const handleSelectChat = useCallback(
    (id) => {
      if (id === activeChatId) return;
      const cached = messagesCache.current[id];
      if (cached) {
        loadChat(cached.messages, cached.conversationId);
      } else {
        clearChat();
      }
      setActiveChatId(id);
    },
    [activeChatId, loadChat, clearChat]
  );

  const handleDeleteChat = useCallback(
    (id) => {
      delete messagesCache.current[id];
      setChatHistory((prev) => prev.filter((c) => c.id !== id));
      if (activeChatId === id) {
        setActiveChatId(null);
        clearChat();
      }
    },
    [activeChatId, clearChat]
  );

  const handleSend = useCallback(
    (text) => {
      if (!activeChatId) {
        const newChat = {
          id: crypto.randomUUID(),
          title: "",
          conversationId: null,
        };
        setChatHistory((prev) => [newChat, ...prev]);
        setActiveChatId(newChat.id);
        setTimeout(() => sendMessage(text), 50);
      } else {
        sendMessage(text);
      }
    },
    [activeChatId, sendMessage]
  );

  const handleOpenPdf = useCallback((file, page) => {
    setPdfView({ file, page, key: Date.now() });
  }, []);

  const toggleLang = useCallback(() => {
    // Navigate to the sibling-language path under the configured Vite base.
    // e.g. /assistant/en  ⇄  /assistant/fa
    const next = lang === "en" ? "fa" : "en";
    window.location.href = `${import.meta.env.BASE_URL}${next}`;
  }, [lang]);

  const dir = lang === "fa" ? "rtl" : "ltr";

  useEffect(() => {
    document.body.dir = dir;
    document.body.lang = lang;
  }, [dir, lang]);

  return (
    <div
      className={`app ${pdfView ? "with-pdf" : ""}`}
      dir={dir}
      style={{
        "--sidebar-width": `${sidebarWidth}px`,
        "--pdf-panel-width": `${pdfPanelWidth}px`,
      }}
    >
      <header className="top-bar">
        <div className="top-bar-left">
          <img src={`${import.meta.env.BASE_URL}images/ipp-azadi-tower-logo.png`} alt="Iran Prosperity Project" className="top-bar-logo" />
          <span className="top-bar-title">{t.welcome}</span>
        </div>
        <LangToggle lang={lang} onToggle={toggleLang} />
      </header>

      <div className="app-body">
        <Sidebar
          lang={lang}
          t={t}
          chatHistory={chatHistory}
          activeChatId={activeChatId}
          onNewChat={handleNewChat}
          onSelectChat={handleSelectChat}
          onDeleteChat={handleDeleteChat}
          onOpenDocuments={() => setDocumentsOpen(true)}
        />

        <div
          className="resize-handle"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize chat history panel"
          onMouseDown={onSidebarResizeStart}
        />

        <main className="main-panel">
          <div className="chat-area">
            {messages.length === 0 ? (
              <WelcomeScreen lang={lang} t={t} onOpenPdf={handleOpenPdf} />
            ) : (
              <div className="messages-scroll">
                {messages.map((msg, i) => {
                  const isLastAssistant =
                    msg.role === "assistant" &&
                    !messages.slice(i + 1).some((m) => m.role === "assistant");
                  return (
                    <ChatMessage
                      key={i}
                      message={msg}
                      index={i}
                      lang={lang}
                      t={t}
                      onOpenPdf={handleOpenPdf}
                      onEditMessage={editAndResend}
                      onRegenerate={regenerate}
                      isLastAssistant={isLastAssistant}
                      isStreaming={isStreaming}
                    />
                  );
                })}
                {isStreaming && messages[messages.length - 1]?.content === "" && (
                  <div className="thinking">{t.thinking}</div>
                )}
                <div ref={chatEndRef} />
              </div>
            )}
          </div>

          <ChatInput
            lang={lang}
            t={t}
            isStreaming={isStreaming}
            onSend={handleSend}
            chatId={activeChatId}
          />
        </main>

        {pdfView && (
          <>
            <div
              className="resize-handle"
              role="separator"
              aria-orientation="vertical"
              aria-label="Resize PDF panel"
              onMouseDown={onPdfResizeStart}
            />
            <PdfPanel
              key={pdfView.key}
              file={pdfView.file}
              pageNumber={pdfView.page}
              barTitle={t.pdfBarTitle}
              onClose={() => setPdfView(null)}
            />
          </>
        )}
      </div>

      <DocumentsModal
        open={documentsOpen}
        onClose={() => setDocumentsOpen(false)}
        lang={lang}
        t={t}
      />
    </div>
  );
}
