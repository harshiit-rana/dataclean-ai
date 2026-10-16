import { useEffect, useRef } from "react";
import { useLocation } from "react-router-dom";

/** Moves keyboard focus to the new page's main heading after a navigation (#101).
 *
 * A single-page app swaps the content without the browser's own page load, so focus stayed on
 * the button that navigated -- often gone from the page, which leaves focus on <body>, and a
 * keyboard user starts again from the very top with nothing announced. Found in the keyboard
 * walkthrough: approving fixes on the review page left focus on the body of the cleaning page.
 * Not on the first load, where the browser's own starting point is the right one. */
export function RouteFocus() {
  const { pathname } = useLocation();
  const first = useRef(true);

  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    // After the new page has rendered its heading.
    const frame = window.requestAnimationFrame(() => {
      const target = document.querySelector<HTMLElement>("main h1") ?? document.querySelector<HTMLElement>("main");
      if (!target) return;
      if (!target.hasAttribute("tabindex")) target.setAttribute("tabindex", "-1");
      target.focus({ preventScroll: false });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [pathname]);

  return null;
}
