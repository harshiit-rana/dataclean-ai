import axe from "axe-core";

/** Accessibility violations axe finds in `container`, as readable lines ("rule: what, on which element").
 *
 * Colour contrast is left out: jsdom does not lay out or paint, so axe cannot measure it here. The theme's
 * colour tokens are checked for contrast separately (theme-contrast.test.ts). */
export async function axeViolations(container: Element): Promise<string[]> {
  const results = await axe.run(container, {
    rules: { "color-contrast": { enabled: false } },
  });
  return results.violations.flatMap((v) =>
    v.nodes.map((n) => `${v.id}: ${v.help} -- ${n.target.join(" ")}`)
  );
}
