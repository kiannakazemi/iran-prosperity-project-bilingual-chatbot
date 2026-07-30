import { useState, useCallback, useRef, useEffect } from "react";
import { API_BASE } from "../config";

/**
 * @param {{ initialMessages?: Array, initialConversationId?: string|null }} [opts]
 *        Restores the open tab after refresh (sessionStorage in App).
 */
export function useChat(opts = {}) {
  const { initialMessages = [], initialConversationId = null } = opts;
  const [messages, setMessages] = useState(() => initialMessages);
  const [isStreaming, setIsStreaming] = useState(false);
  const [conversationId, setConversationId] = useState(
    () => initialConversationId ?? null
  );
  const abortRef = useRef(null);
  const convIdRef = useRef(conversationId);

  useEffect(() => {
    convIdRef.current = conversationId;
  }, [conversationId]);

  const streamResponse = useCallback(async (question) => {
    setIsStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          conversation_id: convIdRef.current,
        }),
        signal: controller.signal,
      });

      // Any non-2xx HTTP status (500, 502, 503, 504, etc.) → unified error UI.
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      // Some browsers/proxies can return a response without a body — guard so
      // we don't blow up on res.body.getReader() and end up with a generic
      // crash instead of the error card.
      if (!res.body) {
        throw new Error("Empty response body");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let eventType = "";
      let fullText = "";
      let sources = [];
      let finalConvId = convIdRef.current;
      let provider = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              if (eventType === "sources") {
                sources = data.sources || [];
                setMessages((prev) => {
                  const updated = [...prev];
                  const last = updated[updated.length - 1];
                  if (last?.role === "assistant") {
                    updated[updated.length - 1] = { ...last, sources };
                  }
                  return updated;
                });
              } else if (eventType === "replace") {
                fullText = data.text || fullText;
                setMessages((prev) => {
                  const updated = [...prev];
                  const last = updated[updated.length - 1];
                  if (last?.role === "assistant") {
                    updated[updated.length - 1] = { ...last, content: fullText };
                  }
                  return updated;
                });
              } else if (eventType === "token") {
                fullText += data.text || "";
                setMessages((prev) => {
                  const updated = [...prev];
                  const last = updated[updated.length - 1];
                  if (last?.role === "assistant") {
                    updated[updated.length - 1] = { ...last, content: fullText };
                  }
                  return updated;
                });
              } else if (eventType === "done") {
                finalConvId = data.conversation_id || finalConvId;
                provider = data.provider || "";
              } else if (eventType === "error") {
                // Backend explicitly signaled an error (classifier failure,
                // LLM API failure, retriever crash, etc.). Bail out so the
                // catch block renders the unified error card.
                throw new Error(data.message || "Server error");
              }
            } catch (parseErr) {
              // If the backend sent an explicit `error` event, propagate
              // (whether the throw came from the branch above or from
              // JSON.parse failing on a malformed error payload). For any
              // other event type, a malformed SSE line is non-fatal — skip.
              if (eventType === "error") throw parseErr;
              /* otherwise: skip malformed JSON */
            }
          }
        }
      }

      // If the stream ended without ever emitting any text, treat as an error
      // (the backend likely crashed mid-pipeline before producing output).
      if (!fullText.trim()) {
        throw new Error("Empty response from server");
      }

      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last?.role === "assistant") {
          const noAnswer =
            /don['']t have enough information|اطلاعات کافی/.test(fullText);
          updated[updated.length - 1] = {
            ...last,
            content: fullText,
            sources: noAnswer ? [] : sources,
            provider,
            streaming: false,
          };
        }
        return updated;
      });

      if (finalConvId) setConversationId(finalConvId);
    } catch (err) {
      if (err.name !== "AbortError") {
        // Unified error path. Every backend failure — connection refused,
        // 5xx response, missing body, malformed stream, explicit server
        // error event, empty stream — lands here. The UI renders a single
        // error card based on `isError: true`; the text comes from the i18n
        // config so it stays in the user's selected language.
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last?.role === "assistant") {
            updated[updated.length - 1] = {
              ...last,
              content: "",
              sources: [],
              provider: "",
              streaming: false,
              isError: true,
            };
          }
          return updated;
        });
      }
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
    }
  }, []);

  const sendMessage = useCallback(
    (question) => {
      if (!question.trim() || isStreaming) return;

      setMessages((prev) => [
        ...prev,
        { role: "user", content: question, sources: [] },
        { role: "assistant", content: "", sources: [], provider: "", streaming: true },
      ]);

      streamResponse(question);
    },
    [isStreaming, streamResponse]
  );

  const editAndResend = useCallback(
    (messageIndex, newText) => {
      if (!newText.trim() || isStreaming) return;

      setMessages((prev) => [
        ...prev.slice(0, messageIndex),
        { role: "user", content: newText, sources: [] },
        { role: "assistant", content: "", sources: [], provider: "", streaming: true },
      ]);

      streamResponse(newText);
    },
    [isStreaming, streamResponse]
  );

  const regenerate = useCallback(() => {
    if (isStreaming) return;

    let lastQuestion = null;

    setMessages((prev) => {
      for (let i = prev.length - 1; i >= 0; i--) {
        if (prev[i].role === "user") {
          lastQuestion = prev[i].content;
          return [
            ...prev.slice(0, i + 1),
            { role: "assistant", content: "", sources: [], provider: "", streaming: true },
          ];
        }
      }
      return prev;
    });

    if (lastQuestion) streamResponse(lastQuestion);
  }, [isStreaming, streamResponse]);

  const loadChat = useCallback((savedMessages, savedConvId) => {
    if (abortRef.current) abortRef.current.abort();
    setMessages(savedMessages || []);
    setConversationId(savedConvId || null);
    setIsStreaming(false);
  }, []);

  const clearChat = useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
    setMessages([]);
    setConversationId(null);
    setIsStreaming(false);
  }, []);

  return {
    messages,
    isStreaming,
    conversationId,
    sendMessage,
    editAndResend,
    regenerate,
    clearChat,
    loadChat,
  };
}
