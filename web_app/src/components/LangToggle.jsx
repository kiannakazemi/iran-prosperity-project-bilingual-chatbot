export default function LangToggle({ lang, onToggle }) {
  return (
    <button className="lang-toggle" onClick={onToggle} title="Switch language / تغییر زبان">
      {lang === "en" ? "فارسی" : "English"}
    </button>
  );
}
