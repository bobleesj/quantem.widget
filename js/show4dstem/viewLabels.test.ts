import { describe, expect, it } from "vitest";
import { SCALE_BAR_CORNER, positionLabelAnchor, shortScanName, titleNamesScan } from "./viewLabels";

describe("viewer labels", () => {
  it("prints a scan by its number", () => {
    expect(shortScanName("series_41")).toBe("41");
    expect(shortScanName("series_41_rot90")).toBe("41_rot90");
    expect(shortScanName("tilt 3")).toBe("tilt 3");
    expect(shortScanName("my_series_2")).toBe("my_series_2");
  });

  it("puts the position label beside the crosshair", () => {
    expect(positionLabelAnchor(100, 100, 60, 400, 400)).toEqual({ x: 116, y: 96 });
  });

  it("keeps the label inside the image", () => {
    expect(positionLabelAnchor(395, 5, 60, 400, 400)).toEqual({ x: 336, y: 13 });
  });

  it("lifts the label out of the scale bar's corner", () => {
    const anchor = positionLabelAnchor(390, 395, 60, 400, 400);
    expect(anchor.y).toBe(400 - SCALE_BAR_CORNER.height);
    expect(positionLabelAnchor(100, 395, 60, 400, 400).y).toBe(391);
  });

  it("knows when the title already names the scan", () => {
    expect(titleNamesScan("Live · series_13 · 262,144/262,144 ✓ · 5 kept", "series_13")).toBe(true);
    expect(titleNamesScan("Live · series_13 · 5 kept", "series_1")).toBe(false);
    expect(titleNamesScan("4D-STEM Explorer", "series_1")).toBe(false);
    expect(titleNamesScan(undefined, "series_1")).toBe(false);
  });
});
