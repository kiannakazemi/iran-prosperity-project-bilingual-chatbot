/**
 * Persists demo chat UI state for the current browser tab only.
 * Survives refresh; cleared when the tab/window is closed (sessionStorage).
 */

const KEY = "ipp-demo-chat-session";
const VERSION = 1;

/**
 * @returns {{ chatHistory: Array, activeChatId: string|null, messagesMap: Record<string, { messages: Array, conversationId: string|null }> }}
 */
export function loadSessionChats() {
  const empty = {
    chatHistory: [],
    activeChatId: null,
    messagesMap: {},
  };
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return empty;
    const data = JSON.parse(raw);
    if (data.v !== VERSION || !Array.isArray(data.chatHistory)) return empty;
    const chatHistory = data.chatHistory;
    const messagesMap = data.messagesMap && typeof data.messagesMap === "object"
      ? data.messagesMap
      : {};
    let activeChatId = data.activeChatId ?? null;
    if (
      activeChatId &&
      !chatHistory.some((c) => c.id === activeChatId)
    ) {
      activeChatId = chatHistory[0]?.id ?? null;
    }
    return { chatHistory, activeChatId, messagesMap };
  } catch {
    return empty;
  }
}

export function saveSessionChats({
  chatHistory,
  activeChatId,
  messagesMap,
}) {
  try {
    sessionStorage.setItem(
      KEY,
      JSON.stringify({
        v: VERSION,
        chatHistory,
        activeChatId,
        messagesMap,
      })
    );
  } catch {
    /* quota or private mode */
  }
}
