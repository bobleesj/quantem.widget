import { expect, test } from "vitest";
import { binImage, livePreviewImages, scanPosition, upsampledTransform } from "./upsampledImage";

const FACTOR = 3;

test("a 3x image bins to the scan grid and each image pixel names the scan position it was made from", () => {
  const rows = 2, cols = 4;
  const image = new Float32Array(rows * FACTOR * cols * FACTOR);
  for (let r = 0; r < rows * FACTOR; r++)
    for (let c = 0; c < cols * FACTOR; c++) image[r * cols * FACTOR + c] = 10 * Math.floor(r / FACTOR) + Math.floor(c / FACTOR);
  expect([...binImage(image, rows, cols, FACTOR)]).toEqual([0, 1, 2, 3, 10, 11, 12, 13]);
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

test("the live Direct preview at 3x keeps full resolution and a scan-grid copy, NaN below the measured rows", () => {
  const rows = 2, cols = 2, total = 4;
  const phase = new Float32Array(rows * FACTOR * cols * FACTOR).map((_, i) => i);
  const preview = { rows, cols, total, mode: "Direct", upsample: FACTOR, phase: new DataView(phase.buffer),
                    roi: new DataView(new Float32Array(rows * cols).buffer) };
  const images = livePreviewImages(preview, true)!;
  expect(images.upsampled).toMatchObject({ factor: FACTOR, rows: total * FACTOR, cols: cols * FACTOR });
  expect(images.upsampled!.data.length).toBe(total * FACTOR * cols * FACTOR);
  expect(images.upsampled!.data[0]).toBe(0);
  expect(Number.isNaN(images.upsampled!.data[rows * FACTOR * cols * FACTOR])).toBe(true);
  const grid = new Float32Array(images.image.buffer);
  expect(grid.length).toBe(total * cols);
  expect(grid[0]).toBeCloseTo((0 + 1 + 2 + 6 + 7 + 8 + 12 + 13 + 14) / 9, 5);
  expect(Number.isNaN(grid[rows * cols])).toBe(true);
  expect(livePreviewImages(preview, false)!.upsampled).toBeNull();     // the virtual detector stays on the grid
  expect(livePreviewImages({ ...preview, upsample: 2 }, true)).toBeNull();  // the bytes do not match 2x
});
