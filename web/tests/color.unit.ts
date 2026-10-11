import { test, expect } from "@playwright/test";
import { rgb } from "d3-color";
import { divergingScale, inkScale, textOn, type ResolvedTheme } from "../src/lib/color";

/** The WCAG contrast ratio of two opaque colors. */
function contrast(a: string, b: string): number {
  const luminance = (color: string) => {
    const { r, g, b: blue } = rgb(color);
    const linear = (c: number) => (c / 255 <= 0.04045 ? c / 255 / 12.92 : ((c / 255 + 0.055) / 1.055) ** 2.4);
    return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(blue);
  };
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

test("text on every data color is readable in both themes", () => {
  for (const theme of ["light", "dark"] as ResolvedTheme[]) {
    const ink = inkScale(1, theme);
    const diverging = divergingScale(1, theme);
    for (let i = 0; i <= 200; i++) {
      for (const fill of [ink(i / 200), diverging(i / 100 - 1)]) {
        expect(contrast(textOn(fill), fill), `${theme} ${fill}`).toBeGreaterThanOrEqual(4.5);
      }
    }
  }
  // Whichever of black and white is farther: on mid grey, black.
  expect(textOn("#777777")).toBe("#000000");
  expect(textOn("#555555")).toBe("#ffffff");
});
