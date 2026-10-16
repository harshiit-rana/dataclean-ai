import { describe, expect, it } from "vitest";
import CSS from "../styles/globals.css?raw";

/** Text contrast in both themes, from the colour tokens themselves (#101). jsdom cannot paint a page, so axe's own
 * contrast rule cannot run in the suite; the tokens are what every coloured piece of text is made of. WCAG AA asks
 * 4.5:1 for normal text. */

function tokens(block: ":root" | ".dark"): Record<string, string> {
  const start = CSS.indexOf(`${block} {`);
  const body = CSS.slice(start, CSS.indexOf("}", start));
  const found: Record<string, string> = {};
  for (const match of body.matchAll(/--([a-z-]+):\s*([^;]+);/g)) found[match[1]] = match[2].trim();
  return found;
}

function hslToRgb(value: string): [number, number, number] {
  const [h, s, l] = value.split(/\s+/).map((part) => parseFloat(part));
  const sat = s / 100;
  const light = l / 100;
  const k = (n: number) => (n + h / 30) % 12;
  const a = sat * Math.min(light, 1 - light);
  const f = (n: number) => light - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [f(0), f(8), f(4)];
}

function luminance(value: string): number {
  const channel = (c: number) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const [r, g, b] = hslToRgb(value).map(channel);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

// (text token, background token): every pairing the app uses for text.
const PAIRS: [string, string][] = [
  ["foreground", "background"],
  ["muted-foreground", "background"],
  ["muted-foreground", "muted"],
  ["primary", "background"],
  ["primary-foreground", "primary"],
  ["destructive", "background"],
  ["destructive-foreground", "destructive"],
];

describe("Theme contrast (#101)", () => {
  for (const theme of [":root", ".dark"] as const) {
    for (const [text, background] of PAIRS) {
      it(`${theme === ":root" ? "light" : "dark"}: ${text} on ${background} meets AA`, () => {
        const t = tokens(theme);
        expect(contrast(t[text], t[background])).toBeGreaterThanOrEqual(4.5);
      });
    }
  }
});
