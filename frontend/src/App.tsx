import React from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppStoreProvider } from "./state/AppStore";
import { OverlaysProvider } from "./components/overlays/OverlaysContext";
import { Sidebar } from "./components/Sidebar";
import { Topbar } from "./components/Topbar";
import SearchPage from "./pages/Search";
import LibraryPage from "./pages/Library";
import ReportPage from "./pages/Report";
import StudyPage from "./pages/Study";
import ChatPage from "./pages/Chat";
import SettingsPage from "./pages/Settings";

function Shell() {
  return (
    <div style={{ display: "flex", minHeight: "100vh", width: "100%", background: "var(--bg)", color: "var(--fg)" }}>
      <Sidebar />
      <main style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", height: "100vh", overflow: "hidden" }}>
        <Topbar />
        <div className="dv-scroll" style={{ flex: 1, overflowY: "auto", overflowX: "hidden" }}>
          <Routes>
            <Route path="/" element={<Navigate to="/search" replace />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/library" element={<LibraryPage />} />
            <Route path="/report/:paperId?" element={<ReportPage />} />
            <Route path="/study" element={<StudyPage />} />
            <Route path="/chat/:paperId?" element={<ChatPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<Navigate to="/search" replace />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppStoreProvider>
        <OverlaysProvider>
          <Shell />
        </OverlaysProvider>
      </AppStoreProvider>
    </BrowserRouter>
  );
}
