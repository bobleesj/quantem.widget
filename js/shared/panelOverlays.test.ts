import { describe, expect, it } from "vitest";

import { overlayGeometry, updateOverlayFromDrag, type PanelOverlaySpec } from "./panelOverlays";

// A rectangle as Python's panel_overlays validator stores it: shape "rect", sorted corners in data pixels.
const RECT: PanelOverlaySpec = { shape: "rect", coords: "data", row0: 0, col0: 4, row1: 10, col1: 12 };

describe("updateOverlayFromDrag", () => {
  it("moves a rectangle by the drag and keeps its size", () => {
    const moved = updateOverlayFromDrag(RECT, "move", 5, 8, 8, 6, 64, 32);
    expect([moved.row0, moved.col0, moved.row1, moved.col1]).toEqual([3, 2, 13, 10]);
    // a hairline thinner than a pixel (a relative box on a small image, a box on a binned preview) stays a hairline
    const hairline = updateOverlayFromDrag({ ...RECT, row0: 5, row1: 5.5 }, "move", 5, 8, 7, 8, 64, 32);
    expect([hairline.row0, hairline.row1]).toEqual([7, 7.5]);
    const relative: PanelOverlaySpec = { shape: "rect", coords: "relative", row0: 0.5, col0: 0, row1: 0.502, col1: 1 };
    const still = updateOverlayFromDrag(relative, "move", 50, 10, 50, 10, 64, 100);
    expect([still.row0, still.row1]).toEqual([0.5, 0.502]);
  });

  it("keeps a pixel of height and width when an edge is dragged past the image border", () => {
    // the bottom edge dragged above the image top used to give row0 == row1 == 0: a closed rectangle
    const closedAbove = updateOverlayFromDrag(RECT, "resize", 10, 8, -5, 8, 64, 32, "b");
    expect([closedAbove.row0, closedAbove.row1]).toEqual([0, 1]);
    // the right edge of a rectangle at the right border, dragged past it and back over the left edge
    const atRight: PanelOverlaySpec = { ...RECT, col0: 60, col1: 64 };
    const pastRight = updateOverlayFromDrag(atRight, "resize", 5, 60, 5, 90, 64, 32, "l");
    expect([pastRight.col0, pastRight.col1]).toEqual([63, 64]);
  });

  it("resizes inside the image exactly to the pointer", () => {
    const resized = updateOverlayFromDrag(RECT, "resize", 10, 12, 20, 30, 64, 32, "br");
    expect([resized.row0, resized.col0, resized.row1, resized.col1]).toEqual([0, 4, 20, 30]);
  });
});

describe("overlayGeometry", () => {
  it("reads a rect by its corners and a circle by center and radius", () => {
    expect(overlayGeometry(RECT, 64, 32)).toEqual({ shape: "rect", row0: 0, col0: 4, row1: 10, col1: 12 });
    expect(overlayGeometry({ shape: "circle", coords: "data", row: 8, col: 9, radius: 3 }, 64, 32))
      .toEqual({ shape: "circle", row: 8, col: 9, radius: 3 });
  });
});
