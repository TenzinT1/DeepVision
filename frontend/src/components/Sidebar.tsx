import React from "react";
import { useLocation, useNavigate } from "react-router-dom";

interface NavItem {
  key: string;
  label: string;
  path: string;
  icon: React.ReactNode;
}

const NAV_ITEMS: NavItem[] = [
  {
    key: "search",
    label: "Search",
    path: "/search",
    icon: (
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
        <circle cx="11" cy="11" r="7" />
        <line x1="16" y1="16" x2="21" y2="21" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    key: "library",
    label: "Library",
    path: "/library",
    icon: (
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
        <rect x="3" y="4" width="6" height="16" rx="1" />
        <rect x="10" y="4" width="5" height="16" rx="1" />
        <rect x="16.5" y="6" width="5" height="14" rx="1" transform="rotate(-8 19 13)" />
      </svg>
    ),
  },
  {
    key: "report",
    label: "Report",
    path: "/report",
    icon: (
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
        <rect x="5" y="3" width="14" height="18" rx="2" />
        <line x1="8.5" y1="8" x2="15.5" y2="8" strokeLinecap="round" />
        <line x1="8.5" y1="12" x2="15.5" y2="12" strokeLinecap="round" />
        <line x1="8.5" y1="16" x2="12.5" y2="16" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    key: "study",
    label: "Study",
    path: "/study",
    icon: (
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
        <rect x="3.5" y="8" width="13" height="9" rx="1.5" transform="rotate(-7 10 12.5)" />
        <rect x="7" y="6" width="13" height="9" rx="1.5" />
      </svg>
    ),
  },
  {
    key: "chat",
    label: "Chat",
    path: "/chat",
    icon: (
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
        <path d="M4 5h16v11H9l-4 3.5V16H4z" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    key: "settings",
    label: "Settings",
    path: "/settings",
    icon: (
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
        <circle cx="12" cy="12" r="3.2" />
        <path
          d="M12 3v2.5M12 18.5V21M3 12h2.5M18.5 12H21M5.6 5.6l1.8 1.8M16.6 16.6l1.8 1.8M18.4 5.6l-1.8 1.8M7.4 16.6l-1.8 1.8"
          strokeLinecap="round"
        />
      </svg>
    ),
  },
];

export function Sidebar() {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <aside
      style={{
        width: 236,
        flex: "none",
        background: "var(--surface)",
        borderRight: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        position: "sticky",
        top: 0,
        height: "100vh",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 11, padding: "20px 20px 18px" }}>
        <div
          style={{
            width: 34,
            height: 34,
            borderRadius: 9,
            background: "var(--accent)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flex: "none",
          }}
        >
          <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="var(--accent-fg)" strokeWidth={2}>
            <circle cx="11" cy="11" r="6" />
            <line x1="15.5" y1="15.5" x2="21" y2="21" strokeLinecap="round" />
            <circle cx="11" cy="11" r="2" fill="var(--accent-fg)" stroke="none" />
          </svg>
        </div>
        <div style={{ lineHeight: 1.1 }}>
          <div style={{ fontWeight: 700, fontSize: 16, letterSpacing: "-.01em" }}>DeepVision</div>
          <div className="dv-mono" style={{ fontSize: 10, color: "var(--fg-subtle)", letterSpacing: ".04em", marginTop: 2 }}>
            RESEARCH ASSISTANT
          </div>
        </div>
      </div>

      <nav style={{ padding: "6px 12px", display: "flex", flexDirection: "column", gap: 2 }}>
        {NAV_ITEMS.map((item) => {
          const active = location.pathname === item.path || location.pathname.startsWith(item.path + "/");
          return (
            <button
              key={item.key}
              onClick={() => {
                // Report & Chat are per-paper views — route to the last-opened
                // paper, or to Library to pick one, instead of a dead-end.
                if (item.path === "/report" || item.path === "/chat") {
                  const last = localStorage.getItem("dv-last-paper");
                  navigate(last ? `${item.path}/${last}` : "/library");
                } else {
                  navigate(item.path);
                }
              }}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 11,
                width: "100%",
                padding: "9px 12px",
                border: "none",
                borderRadius: 9,
                fontSize: 13.5,
                fontWeight: active ? 600 : 500,
                textAlign: "left",
                background: active ? "var(--accent-soft)" : "transparent",
                color: active ? "var(--accent)" : "var(--fg-muted)",
              }}
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div style={{ marginTop: "auto", padding: "14px 16px", borderTop: "1px solid var(--border)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "0 3px" }}>
          <span style={{ fontSize: 11.5, color: "var(--fg-subtle)" }}>v0.9 · local build</span>
        </div>
      </div>
    </aside>
  );
}
