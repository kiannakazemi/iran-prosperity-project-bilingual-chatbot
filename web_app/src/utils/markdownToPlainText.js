/**
 * Turn common chatbot Markdown into plain text for the clipboard
 * (no **, ##, ---, etc.).
 */
export function markdownToPlainText(src) {
  if (!src || typeof src !== "string") return "";
  let s = src.replace(/\r\n/g, "\n");

  // [label](url) -> label
  s = s.replace(/\[([^\]]*)\]\([^)]*\)/g, "$1");

  // Bold ** and __ (repeat to catch adjacent runs)
  for (let i = 0; i < 8 && /\*\*[^*]+\*\*/.test(s); i++) {
    s = s.replace(/\*\*([^*]+)\*\*/g, "$1");
  }
  for (let i = 0; i < 8 && /__[^_]+__/.test(s); i++) {
    s = s.replace(/__([^_]+)__/g, "$1");
  }

  // Headings ### at line start
  s = s.replace(/^#{1,6}\s+/gm, "");

  // Horizontal rules
  s = s.replace(/^---+$/gm, "");

  // Inline code `like this`
  s = s.replace(/`([^`]+)`/g, "$1");

  s = s.replace(/\n{3,}/g, "\n\n");
  return s.trim();
}
