import React from "react";
import { useLocation } from "react-router-dom";
import { useAppStore } from "../state/AppStore";

const TITLES: Record<string, [string, string]> = {
  search: ["Search", "Query arXiv and ingest papers into your workspace"],
  library: ["Library", "Your ingested papers and their processing status"],
  report: ["Report", "Interactive multimodal summary"],
  study: ["Study", "Spaced-repetition flashcards and self-test quizzes"],
  chat: ["Chat", "Ask questions grounded in one paper"],
  settings: ["Settings", "Providers, keys and extraction controls"],
};

function screenMetaFor(pathname: string): [string, string] {
  const key = pathname.split("/").filter(Boolean)[0] ?? "search";
  return TITLES[key] ?? ["DeepVision", ""];
}

export function Topbar() {
  const location = useLocation();
  const { theme, toggleTheme, health, healthError } = useAppStore();
  const [title, subtitle] = screenMetaFor(location.pathname);

  const online = !!health?.llm_online;
  let dotColor = "var(--fg-subtle)";
  let statusText = "Checking model…";
  if (health) {
    dotColor = online ? "var(--green)" : "var(--red)";
    statusText = online ? "Local model online" : "Model offline";
  } else if (healthError) {
    dotColor = "var(--red)";
    statusText = "Backend unreachable";
  }

  return (
    <header
      style={{
        flex: "none",
        height: 60,
        borderBottom: "1px solid var(--border)",
        background: "var(--surface)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 24px",
      }}
    >
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 15, fontWeight: 600, letterSpacing: "-.01em" }}>{title}</div>
        <div style={{ fontSize: 12, color: "var(--fg-subtle)", marginTop: 1 }}>{subtitle}</div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 7,
            fontSize: 12,
            color: "var(--fg-muted)",
            padding: "6px 11px",
            border: "1px solid var(--border)",
            borderRadius: 8,
          }}
        >
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: dotColor, display: "inline-block" }} />
          {statusText}
        </div>
        <button
          onClick={toggleTheme}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 7,
            fontSize: 12.5,
            fontWeight: 500,
            color: "var(--fg)",
            background: "var(--surface-2)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            padding: "7px 12px",
          }}
        >
          {theme === "dark" ? "☀ Light" : "☾ Dark"}
        </button>
      </div>
    </header>
  );
}
