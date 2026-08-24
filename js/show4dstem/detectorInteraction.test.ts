import { describe, expect, it } from "vitest";
import { clampDetectorCenter, resizeDetectorFromPointer } from "./detectorInteraction";

describe("Show4DSTEM detector interaction geometry", () => {
  it("keeps subpixel detector centers while clamping to the diffraction plane", () => {
    expect(clampDetectorCenter(12.25, 18.75, 48, 48)).toEqual({
      row: 12.25,
      col: 18.75,
    });
    expect(clampDetectorCenter(-2, 50, 48, 48)).toEqual({ row: 0, col: 47 });
  });

  it("resizes circle, square, and rectangle detectors from the live pointer", () => {
    const common = {
      centerRow: 10,
      centerCol: 10,
      pointerRow: 13,
      pointerCol: 14,
      radius: 8,
      radiusInner: 3,
    };

    expect(resizeDetectorFromPointer({ ...common, mode: "circle" })).toEqual({ radius: 5 });
    expect(resizeDetectorFromPointer({ ...common, mode: "square" })).toEqual({ radius: 4 });
    expect(resizeDetectorFromPointer({ ...common, mode: "rect" })).toEqual({
      width: 8,
      height: 6,
    });
    expect(resizeDetectorFromPointer({
      ...common,
      mode: "rect",
      aspectRatio: 2,
      preserveAspect: true,
    })).toEqual({ width: 12, height: 6 });
  });

  it("keeps annular inner and outer radii ordered during live resizing", () => {
    const common = {
      mode: "annular" as const,
      centerRow: 10,
      centerCol: 10,
      pointerRow: 10,
      pointerCol: 12,
      radius: 10,
      radiusInner: 4,
    };

    expect(resizeDetectorFromPointer(common)).toEqual({ radius: 5 });
    expect(resizeDetectorFromPointer({
      ...common,
      pointerCol: 25,
      resizeInner: true,
    })).toEqual({ radiusInner: 9 });
  });
});
