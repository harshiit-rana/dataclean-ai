import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { initTheme, resolveTheme, useTheme } from "./theme";

function mockSystem(dark: boolean) {
  const listeners = new Set<() => void>();
  vi.spyOn(window, "matchMedia").mockReturnValue({
    matches: dark,
    addEventListener: (_: string, fn: () => void) => listeners.add(fn),
    removeEventListener: (_: string, fn: () => void) => listeners.delete(fn),
  } as unknown as MediaQueryList);
  return { fire: () => listeners.forEach((fn) => fn()) };
}

describe("theme", () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.classList.remove("dark");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("starts from the operating system when nothing has been chosen", () => {
    mockSystem(true);
    expect(resolveTheme()).toBe("dark");
  });

  it("applies the class before React mounts, so there's no flash of the wrong colour", () => {
    mockSystem(true);
    initTheme();
    expect(document.documentElement.classList.contains("dark")).toBe(true);
  });

  it("remembers a choice and lets it beat the operating system", () => {
    // The point of storing it: someone who wants this one app light while their machine is
    // dark must not be flipped back on the next OS event.
    mockSystem(true);
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setTheme("light"));

    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(resolveTheme()).toBe("light");
  });

  it("follows the operating system until the toggle is used", () => {
    const system = mockSystem(false);
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe("light");

    vi.spyOn(window, "matchMedia").mockReturnValue({
      matches: true,
      addEventListener: () => {},
      removeEventListener: () => {},
    } as unknown as MediaQueryList);
    act(() => system.fire());

    expect(result.current.theme).toBe("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
  });

  it("stops following the operating system once the toggle is used", () => {
    const system = mockSystem(false);
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setTheme("dark"));

    act(() => system.fire()); // the machine changes its mind; the user's choice stands
    expect(result.current.theme).toBe("dark");
  });

  it("still works when storage is unavailable", () => {
    // Private browsing throws on both calls. The theme should still apply for this page --
    // it just won't be remembered.
    mockSystem(false);
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("denied");
    });
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("denied");
    });

    const { result } = renderHook(() => useTheme());
    act(() => result.current.setTheme("dark"));
    expect(document.documentElement.classList.contains("dark")).toBe(true);
  });
});
