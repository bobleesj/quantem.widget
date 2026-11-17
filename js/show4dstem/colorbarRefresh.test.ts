// @ts-expect-error Vitest raw source import is not part of the widget bundle.
import source from "./index.tsx?raw";
import ts from "typescript";
import {expect, it, vi} from "vitest";
import {drawColorbar} from "../figure";

// The DP renderer caches the colorbar range of each new pattern and then
// redraws the UI overlay itself, in the same call: no ROI, zoom or Colorbar
// toggle changes, yet the already-visible labels must follow the new range.
// The same renderer draws a streamed pattern from an animation frame; there
// the overlay is redrawn only while the colorbar shows. Find the renderer in
// the widget source and run its range-and-redraw tail against a recording canvas.
const tree = ts.createSourceFile("index.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
let colormapEffect = "";
let assignsLatestOverlay = false;
function visit(node: ts.Node) {
  if (ts.isVariableDeclaration(node) && node.name.getText(tree) === "renderDpPattern" && node.initializer
      && node.initializer.getText(tree).includes("dpColorbarVminRef.current = vmin")) {
    colormapEffect = node.initializer.getText(tree);
  }
  if (ts.isExpressionStatement(node) && node.getText(tree) === "drawDpUiRef.current = drawDpUi;") {
    assignsLatestOverlay = true;
  }
  ts.forEachChild(node, visit);
}
visit(tree);

it("redraws visible colorbar labels after a new explicit range and after a new data range", () => {
  expect(assignsLatestOverlay).toBe(true);
  const tail = colormapEffect.slice(colormapEffect.indexOf("dpColorbarVminRef.current = vmin"));
  // the cached range is written first, then the overlay that reads it is redrawn
  expect(tail.indexOf("dpColorbarVmaxRef.current = vmax")).toBeGreaterThan(0);
  expect(tail.indexOf("drawDpUiRef.current?.()")).toBeGreaterThan(tail.indexOf("dpColorbarVmaxRef.current = vmax"));
  const body = tail.slice(0, tail.indexOf("drawDpUiRef.current?.()") + "drawDpUiRef.current?.();".length);
  const refreshAfterRender = new Function("vmin", "vmax", "dpColorbarVminRef", "dpColorbarVmaxRef", "drawDpUiRef", "paintDpCanvas",
    "fromFrame", "showDpColorbar", body);

  const fillText = vi.fn();
  const ctx = {fillRect() {}, strokeRect() {}, fillText} as unknown as CanvasRenderingContext2D;
  const lut = new Uint8Array(256 * 3);
  const minRef = {current: 0};
  const maxRef = {current: 1};
  const drawDpUi = () => drawColorbar(ctx, 256, 256, lut, minRef.current, maxRef.current, false);
  for (const [min, max] of [[-10, 10], [-2, 2], [-7, 4]]) {
    fillText.mockClear();
    refreshAfterRender(min, max, minRef, maxRef, {current: drawDpUi}, () => {}, false, false);
    expect(fillText.mock.calls.map(call => call[0])).toEqual([max.toFixed(2), min.toFixed(2)]);
  }
  // a streamed pattern, drawn from an animation frame: the labels follow while the colorbar shows
  fillText.mockClear();
  refreshAfterRender(-3, 3, minRef, maxRef, {current: drawDpUi}, () => {}, true, false);
  expect(fillText).not.toHaveBeenCalled();
  refreshAfterRender(-3, 3, minRef, maxRef, {current: drawDpUi}, () => {}, true, true);
  expect(fillText.mock.calls.map(call => call[0])).toEqual(["3.00", "-3.00"]);
});
