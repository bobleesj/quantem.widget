import { expect, test } from "vitest";
import { scanPosition, upsampledTransform } from "./upsampledImage";

const FACTOR = 3;

test("each pixel of a 3x image names the scan position it was made from", () => {
  expect(scanPosition(5, 11, FACTOR)).toEqual([1, 3]);
  expect(scanPosition(2, 2, FACTOR)).toEqual([0, 0]);
});

test("under any pan and zoom, a scan position lands on the same screen point on the 3x canvas as on the overlay", () => {
  const cssWidth = 400, scanCols = 128;
  for (const [panX, panY, zoom] of [[0, 0, 1], [-37.5, 12, 2.5], [10, -4, 0.75]]) {
    const t = upsampledTransform(panX, panY, zoom, FACTOR);
    for (const col of [0, 17, 127.5]) {
      const overlayCss = (panX + zoom * col) * cssWidth / scanCols;                         // overlay: scan units
      const imageCss = (t.translateX + t.scale * col * FACTOR) * cssWidth / (scanCols * FACTOR);  // image: 3x backing
      expect(imageCss).toBeCloseTo(overlayCss, 9);
    }
    for (const row of [0, 31]) expect(t.translateY + t.scale * row * FACTOR).toBeCloseTo(FACTOR * (panY + zoom * row), 9);
  }
});
