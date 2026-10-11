// @ts-expect-error Vitest raw source import is not part of the widget bundle.
import source from "./index.tsx?raw";
import * as React from "react";
import {act} from "react";
import {createRoot} from "react-dom/client";
import ts from "typescript";
import {expect, it, vi} from "vitest";
import {drawColorbar} from "../figure";

// Mount the widget's real overlay-refresh effect with a recording canvas.
// The DP renderer has updated its cached range and version; no ROI, zoom or
// Colorbar toggle changes. Its already-visible labels must follow that render.
const tree = ts.createSourceFile("index.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
let effect = "";
function visit(node: ts.Node) {
  if (ts.isCallExpression(node) && node.expression.getText(tree) === "React.useEffect"
      && node.arguments[0]?.getText(tree) === "() => drawDpUi()") {
    effect = node.getText(tree);
  }
  ts.forEachChild(node, visit);
}
visit(tree);
const useRefresh = new Function("React", "drawDpUi", "dpOffscreenVersion", effect);

it("redraws visible colorbar labels after a new explicit range and after a new data range", () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  const fillText = vi.fn();
  const ctx = {fillRect() {}, strokeRect() {}, fillText} as unknown as CanvasRenderingContext2D;
  const lut = new Uint8Array(256 * 3);
  let range = [-10, 10];
  const drawDpUi = () => drawColorbar(ctx, 256, 256, lut, range[0], range[1], false);
  function Harness({version}: {version: number}) {
    useRefresh(React, drawDpUi, version);
    return null;
  }
  const mount = document.createElement("div");
  const root = createRoot(mount);
  try {
    for (const [version, min, max] of [[1, -10, 10], [2, -2, 2], [3, -7, 4]]) {
      range = [min, max];
      fillText.mockClear();
      act(() => root.render(React.createElement(Harness, {version})));
      expect(fillText.mock.calls.map(call => call[0])).toEqual([max.toFixed(2), min.toFixed(2)]);
    }
  } finally {
    act(() => root.unmount());
    vi.unstubAllGlobals();
  }
});
