import { useCallback, useEffect, useState } from "react";

/** Light/dark, and the single place that decides which.
 *
 * The `dark` class on <html> is the one source of truth -- Tailwind is configured
 * `darkMode: ["class"]`, so the CSS follows it, and `useChartTheme` reads the same class so
 * the charts can't disagree with the page they're drawn on.
 *
 * The OS preference is the starting point, not the rule. Once someone uses the toggle, that
 * choice is stored and wins: a person who wants this one app light while the rest of their
 * machine is dark shouldn't have it flipped back by an OS event.
 */

export type Theme = "light" | "dark";

const STORAGE_KEY = "dataclean.theme";

function storedTheme(): Theme | null {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null; // private browsing; fall back to the OS preference
  }
}

function systemTheme(): Theme {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function resolveTheme(): Theme {
  return storedTheme() ?? systemTheme();
}

export function applyTheme(theme: Theme): void {
  document.documentElement.classList.toggle("dark", theme === "dark");
}

/** Called from main.tsx before React mounts, so the first paint is already the right colour
 * rather than flashing white and then correcting itself. */
export function initTheme(): void {
  applyTheme(resolveTheme());
}

export function useTheme(): { theme: Theme; setTheme: (next: Theme) => void } {
  const [theme, setThemeState] = useState<Theme>(resolveTheme);

  // Only while the user hasn't chosen: after that their choice is the answer, and following
  // the OS would silently overrule it.
  useEffect(() => {
    if (storedTheme() !== null) return;
    const media = window.matchMedia?.("(prefers-color-scheme: dark)");
    const onChange = () => {
      const next = systemTheme();
      setThemeState(next);
      applyTheme(next);
    };
    media?.addEventListener("change", onChange);
    return () => media?.removeEventListener("change", onChange);
  }, [theme]);

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next);
    applyTheme(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Nothing to do -- the theme still applies for this page, it just won't be remembered.
    }
  }, []);

  return { theme, setTheme };
}
