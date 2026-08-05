import { renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { inkOn, rampColor, useChartTheme } from "./chartTheme";

describe("rampColor", () => {
  const ramp = ["#a", "#b", "#c", "#d", "#e"];

  it("maps the ends of the range to the ends of the ramp", () => {
    expect(rampColor(ramp, 0)).toBe("#a");
    expect(rampColor(ramp, 1)).toBe("#e");
  });

  it("maps the middle to the middle", () => {
    expect(rampColor(ramp, 0.5)).toBe("#c");
  });

  it("clamps out-of-range values rather than indexing off the end", () => {
    expect(rampColor(ramp, -3)).toBe("#a");
    expect(rampColor(ramp, 42)).toBe("#e");
  });

  it("survives NaN", () => {
    // A confusion matrix row with no rows in it divides by zero upstream; a crash here
    // would take down the whole results page over an empty cell.
    expect(rampColor(ramp, Number.NaN)).toBe("#a");
  });
});

describe("inkOn", () => {
  it("uses dark ink on a light fill", () => {
    expect(inkOn("#cde2fb")).toBe("#0b0b0b");
  });

  it("uses light ink on a dark fill", () => {
    expect(inkOn("#0d366b")).toBe("#ffffff");
  });

  it("is decided by luminance, not by which channel is biggest", () => {
    // Pure green is far brighter than pure blue at the same nominal intensity, so a naive
    // check on the raw values would put white text on it and fail contrast.
    expect(inkOn("#00ff00")).toBe("#0b0b0b");
    expect(inkOn("#0000ff")).toBe("#ffffff");
  });

  it("accepts a hex string without the leading hash", () => {
    expect(inkOn("ffffff")).toBe("#0b0b0b");
  });
});

describe("useChartTheme", () => {
  afterEach(() => {
    document.documentElement.classList.remove("dark");
    vi.restoreAllMocks();
  });

  it("follows the app's surface, not the operating system", () => {
    // Found live: the machine was in dark mode, the app renders light regardless (nothing
    // ever adds the `dark` class), and the correlation heatmap painted its near-zero cells
    // charcoal on a white page. Charts have to read the same signal the CSS reads.
    vi.spyOn(window, "matchMedia").mockReturnValue({
      matches: true,
      addEventListener: () => {},
      removeEventListener: () => {},
    } as unknown as MediaQueryList);

    const { result } = renderHook(() => useChartTheme());
    expect(result.current.isDark).toBe(false);
    // A pale mid-point is what makes "pale is none" true on a white page.
    expect(result.current.divergingMid).toBe("#f0efec");
  });

  it("goes dark when the app does", () => {
    document.documentElement.classList.add("dark");
    const { result } = renderHook(() => useChartTheme());
    expect(result.current.isDark).toBe(true);
  });
});
