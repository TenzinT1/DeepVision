import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { health as fetchHealth } from "../api/client";
import type { HealthResponse } from "../api/types";

export type Theme = "light" | "dark";

const THEME_KEY = "dv-theme";

function readStoredTheme(): Theme {
  if (typeof window === "undefined") return "light";
  const stored = window.localStorage.getItem(THEME_KEY);
  if (stored === "light" || stored === "dark") return stored;
  if (window.matchMedia?.("(prefers-color-scheme: dark)").matches) return "dark";
  return "light";
}

export interface AppStoreApi {
  /** Current theme; also stamped onto <html data-theme>. */
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;

  /** Latest /health snapshot, or null before the first successful fetch. */
  health: HealthResponse | null;
  healthError: boolean;
  refreshHealth: () => Promise<void>;
}

const AppStoreContext = createContext<AppStoreApi | undefined>(undefined);

export function AppStoreProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(() => readStoredTheme());
  const [healthState, setHealthState] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    window.localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const setTheme = useCallback((next: Theme) => setThemeState(next), []);
  const toggleTheme = useCallback(
    () => setThemeState((prev) => (prev === "dark" ? "light" : "dark")),
    []
  );


  const refreshHealth = useCallback(async () => {
    try {
      const h = await fetchHealth();
      setHealthState(h);
      setHealthError(false);
    } catch {
      setHealthState(null);
      setHealthError(true);
    }
  }, []);

  useEffect(() => {
    refreshHealth();
    const iv = setInterval(refreshHealth, 30_000);
    return () => clearInterval(iv);
  }, [refreshHealth]);

  const value = useMemo<AppStoreApi>(
    () => ({
      theme,
      setTheme,
      toggleTheme,
      health: healthState,
      healthError,
      refreshHealth,
    }),
    [
      theme,
      setTheme,
      toggleTheme,
      healthState,
      healthError,
      refreshHealth,
    ]
  );

  return <AppStoreContext.Provider value={value}>{children}</AppStoreContext.Provider>;
}

export function useAppStore(): AppStoreApi {
  const ctx = useContext(AppStoreContext);
  if (!ctx) throw new Error("useAppStore must be used within an AppStoreProvider");
  return ctx;
}
