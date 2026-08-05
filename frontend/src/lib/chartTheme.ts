import { useEffect, useState } from "react";

/** Chart colour tokens.
 *
 * Validated against this app's own surfaces (#ffffff light, #0f1729 dark) rather than
 * assumed: all three categorical slots clear the lightness band, the chroma floor, adjacent
 * colour-blind separation (worst ΔE 9.2 light / 9.4 dark, target ≥ 8) and the normal-vision
 * floor. Aqua sits at 2.82:1 on the light surface, below the 3:1 bar — which is why every
 * chart here ships a table view and direct labels rather than relying on hue alone.
 *
 * Three categorical slots, deliberately. Past three, adjacent pairs stop clearing the
 * all-pairs floors, and no chart in this app needs more: the busiest has two lines plus a
 * reference. Anything that would need a fourth becomes a table.
 */

export interface ChartTheme {
  series: [string, string, string];
  /** Sequential ramp, light → dark, for magnitude (confusion matrix). One hue, never a rainbow. */
  sequential: string[];
  /** Diverging poles + neutral midpoint, for correlation. Warm/cool so the ends read opposite. */
  divergingLow: string;
  divergingHigh: string;
  divergingMid: string;
  grid: string;
  axis: string;
  /** Axis and tick text. Data colours never carry text. */
  mutedText: string;
  secondaryText: string;
  surface: string;
  good: string;
  critical: string;
  isDark: boolean;
}

const LIGHT: ChartTheme = {
  series: ["#2a78d6", "#eb6834", "#1baf7a"],
  sequential: ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95"],
  divergingLow: "#2a78d6",
  divergingHigh: "#e34948",
  divergingMid: "#f0efec",
  grid: "#e1e0d9",
  axis: "#c3c2b7",
  mutedText: "#898781",
  secondaryText: "#52514e",
  surface: "#ffffff",
  good: "#0ca30c",
  critical: "#d03b3b",
  isDark: false,
};

const DARK: ChartTheme = {
  series: ["#3987e5", "#d95926", "#199e70"],
  sequential: ["#0d366b", "#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"],
  divergingLow: "#3987e5",
  divergingHigh: "#e66767",
  divergingMid: "#383835",
  grid: "#2c2c2a",
  axis: "#383835",
  mutedText: "#898781",
  secondaryText: "#c3c2b7",
  surface: "#0f1729",
  good: "#0ca30c",
  critical: "#d03b3b",
  isDark: true,
};

/** Charts follow the app's surface, and nothing else.
 *
 * Tailwind is configured `darkMode: ["class"]`, so the page is dark exactly when the `dark`
 * class is on <html> -- the OS preference doesn't move it. Reading the OS preference here as
 * a fallback is how the correlation heatmap ended up painting its near-zero cells charcoal on
 * a white page: the machine was in dark mode, the app was not, and the palettes disagreed
 * about which surface they were sitting on. One source of truth, the same one the CSS uses.
 */
function detectDark(): boolean {
  if (typeof document === "undefined") return false;
  return document.documentElement.classList.contains("dark");
}

/** Recharts needs real colour strings, not CSS variables, so the palette is resolved in JS
 * and re-resolved when the theme changes. */
export function useChartTheme(): ChartTheme {
  const [dark, setDark] = useState(detectDark);

  useEffect(() => {
    const observer = new MutationObserver(() => setDark(detectDark()));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  return dark ? DARK : LIGHT;
}

/** Picks a step from a ramp for a 0–1 value. */
export function rampColor(ramp: string[], fraction: number): string {
  if (!Number.isFinite(fraction)) return ramp[0];
  const index = Math.min(ramp.length - 1, Math.max(0, Math.round(fraction * (ramp.length - 1))));
  return ramp[index];
}

/** Ink that stays legible on a filled cell, chosen by the fill's luminance -- the one case
 * where text sits inside a data colour. */
export function inkOn(background: string): string {
  const hex = background.replace("#", "");
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const channel = (c: number) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const luminance = 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
  return luminance > 0.45 ? "#0b0b0b" : "#ffffff";
}

export const AXIS_TICK_STYLE = { fontSize: 11, fontVariantNumeric: "tabular-nums" } as const;
