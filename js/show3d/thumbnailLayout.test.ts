import { expect, test } from "vitest";
import {
  cellBox, cellsInWindow, deviceBox, frameAtPosition, imageBox, positionOfFrame, scrollToShow, stripGrid,
  stripOverscan, stripWindow, stripWindowStale,
} from "./thumbnailLayout";

test("grid: 1, 2 or 3 columns share the inner width; rows are whole pixels", () => {
  expect(stripGrid(10, 2, 262)).toEqual({ cols: 2, cellWidth: 131, rowHeight: 131, rows: 5, totalHeight: 655 });
  const three = stripGrid(10, 3, 262);
  expect(three.cols).toBe(3);
  expect(three.rowHeight).toBe(87);
  expect(three.rows).toBe(4);
  expect(stripGrid(10, 7, 262).cols).toBe(3);
  expect(stripGrid(10, 0, 262).cols).toBe(2);   // unset: two across
  expect(stripGrid(0, 1, 262).totalHeight).toBe(0);
});

test("newest first reverses the order and is its own inverse", () => {
  expect(frameAtPosition(0, 120, true)).toBe(119);
  expect(frameAtPosition(119, 120, true)).toBe(0);
  expect(frameAtPosition(5, 120, false)).toBe(5);
  for (const frame of [0, 7, 119]) expect(frameAtPosition(positionOfFrame(frame, 120, true), 120, true)).toBe(frame);
});

test("cells tile the width with no gap", () => {
  const grid = stripGrid(9, 3, 250);
  const a = cellBox(0, grid), b = cellBox(1, grid), c = cellBox(2, grid);
  expect(a.x + a.width).toBeCloseTo(b.x);
  expect(b.x + b.width).toBeCloseTo(c.x);
  expect(c.x + c.width).toBeCloseTo(250);
  expect(cellBox(4, grid).y).toBe(grid.rowHeight);
});

test("image box: square fills, wide and tall images are centred", () => {
  const cell = { x: 0, y: 0, width: 131, height: 131 };
  expect(imageBox(cell, 1)).toEqual(cell);
  const wide = imageBox(cell, 2);
  expect(wide.width).toBe(131);
  expect(wide.height).toBeCloseTo(65.5);
  expect(wide.y).toBeCloseTo(32.75);
  const tall = imageBox(cell, 0.5);
  expect(tall.height).toBe(131);
  expect(tall.x).toBeCloseTo(32.75);
  // a fraction of a pixel off square still fills (no seam)
  expect(imageBox({ x: 0, y: 0, width: 131.4, height: 131 }, 1).width).toBe(131.4);
});

test("window: the viewport plus overscan, clamped to the content", () => {
  const overscan = stripOverscan(600, 131);
  expect(overscan).toBe(600);
  expect(stripWindow(0, 600, 20000, overscan)).toEqual({ top: 0, height: 1800 });
  expect(stripWindow(5000, 600, 20000, overscan)).toEqual({ top: 4400, height: 1800 });
  expect(stripWindow(19400, 600, 20000, overscan)).toEqual({ top: 18200, height: 1800 });
  // short content: the window is the whole content
  expect(stripWindow(0, 600, 400, overscan)).toEqual({ top: 0, height: 400 });
  // the canvas's device-pixel limit caps the height, never below the viewport
  expect(stripWindow(5000, 600, 20000, overscan, 1000).height).toBe(1000);
  expect(stripWindow(5000, 600, 20000, overscan, 300).height).toBe(600);
});

test("window moves only after half an overscan, or when the view left it", () => {
  const current = stripWindow(5000, 600, 20000, 600);
  expect(stripWindowStale(current, 5000, 600, 20000, 600)).toBe(false);
  expect(stripWindowStale(current, 5299, 600, 20000, 600)).toBe(false);
  expect(stripWindowStale(current, 5300, 600, 20000, 600)).toBe(true);
  expect(stripWindowStale(current, 1000, 600, 20000, 600)).toBe(true);
  expect(stripWindowStale(null, 0, 600, 20000, 600)).toBe(true);
  // more content (a streamed frame) changes nothing until the window would be another height
  expect(stripWindowStale(stripWindow(0, 600, 400, 600), 0, 600, 531, 600)).toBe(true);
});

test("cells in the window: whole rows overlapping it, in strip order", () => {
  const grid = stripGrid(120, 2, 262);   // rows of 131 px
  const cells = cellsInWindow({ top: 0, height: 300 }, 120, grid, false);
  expect(cells.map(c => c.frame)).toEqual([0, 1, 2, 3, 4, 5]);   // rows 0..2 (262..393 overlaps 300)
  const mid = cellsInWindow({ top: 131 * 10 + 5, height: 131 }, 120, grid, false);
  expect(mid.map(c => c.frame)).toEqual([20, 21, 22, 23]);
  const newest = cellsInWindow({ top: 0, height: 131 }, 120, grid, true);
  expect(newest.map(c => c.frame)).toEqual([119, 118]);
  // a last row with one cell
  const odd = stripGrid(5, 2, 262);
  expect(cellsInWindow({ top: 0, height: 10000 }, 5, odd, false).map(c => c.frame)).toEqual([0, 1, 2, 3, 4]);
  expect(cellsInWindow({ top: 0, height: 10000 }, 5, odd, true).map(c => c.frame)).toEqual([4, 3, 2, 1, 0]);
  expect(cellsInWindow({ top: 0, height: 0 }, 5, odd, false)).toEqual([]);
});

test("device boxes share rounded edges relative to the window top", () => {
  const grid = stripGrid(9, 3, 262);
  const a = deviceBox(cellBox(0, grid), 0, 1.5);
  const b = deviceBox(cellBox(1, grid), 0, 1.5);
  expect(a.x + a.width).toBe(b.x);
  const below = deviceBox(cellBox(3, grid), grid.rowHeight, 2);
  expect(below.y).toBe(0);
  expect(below.height).toBe(grid.rowHeight * 2);
});

test("scroll to show the selected frame: centred when out of view, null when in view", () => {
  const grid = stripGrid(120, 2, 262);
  expect(scrollToShow(0, grid, 0, 600)).toBeNull();
  expect(scrollToShow(40, grid, 0, 600)).toBe(Math.round(131 * 20 - (600 - 131) / 2));
  expect(scrollToShow(119, grid, 0, 600)).toBe(grid.totalHeight - 600);
  expect(scrollToShow(positionOfFrame(119, 120, true), grid, 0, 600)).toBeNull();
});
