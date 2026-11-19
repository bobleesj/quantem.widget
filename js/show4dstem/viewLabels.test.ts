import { describe, expect, it } from "vitest";
import { SCALE_BAR_CORNER, isLiveFrame, keptScanCount, positionLabelAnchor, shortScanName, titleNamesScan } from "./viewLabels";
import shortScanNameExamples from "./shortScanName.examples.json";

describe("viewer labels", () => {
  it("prints scan names as the Python title does (shared examples)", () => {
    for (const [name, short] of shortScanNameExamples) expect(shortScanName(name)).toBe(short);
  });

  it("counts a Live frame as no kept scan", () => {
    expect(isLiveFrame("live_12")).toBe(true);
    expect(isLiveFrame("series_12")).toBe(false);
    expect(isLiveFrame("live_view")).toBe(false);
    expect(keptScanCount(["live_3"], 1)).toBe(0);
    expect(keptScanCount(["series_1", "series_2", "live_3"], 3)).toBe(2);
    expect(keptScanCount(undefined, 2)).toBe(2);
  });

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
    const tilt = "maped_20261118-222131_7of7_x0.0_y0.0mrad";
    expect(titleNamesScan(`Live · ${tilt} · 262,144/262,144 ✓ · 7 kept`, tilt)).toBe(true);
    expect(titleNamesScan(`Live · ${tilt} · 7 kept`, "maped_20261118-222131_5of7_x-8.5_y-14.7mrad")).toBe(false);
  });
});
