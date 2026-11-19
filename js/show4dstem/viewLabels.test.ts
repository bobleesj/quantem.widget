import { describe, expect, it } from "vitest";
import { diffractionPositionLabel, SCALE_BAR_CORNER, isLiveFrame, keptScanCount, positionLabelAnchor, shortScanName, splitLiveFrameTitle, sourcePathForView, titleNamesScan, virtualImageGridLabel } from "./viewLabels";
import shortScanNameExamples from "./shortScanName.examples.json";

describe("viewer labels", () => {
  it("isolates the live frame number while preserving scan progress verbatim", () => {
    expect(splitLiveFrameTitle("Live · frame 9")).toEqual({ frame: "9", detail: "" });
    expect(splitLiveFrameTitle("Live · frame 10 · 1,024/16,384")).toEqual({
      frame: "10",
      detail: " · 1,024/16,384",
    });
    expect(splitLiveFrameTitle("Live · series_10 · 1,024/16,384")).toBeNull();
    expect(splitLiveFrameTitle("Waiting for acquisition")).toBeNull();
  });

  it("shows the retained full field while Region waits for its own geometry", () => {
    expect(virtualImageGridLabel(1, 262144, 512, false)).toBe("512×512");
    expect(virtualImageGridLabel(93, 116, 512, true)).toBe("93×116");
    expect(virtualImageGridLabel(180, 120, null, false)).toBe("180×120");
  });

  it("labels a point with specimen coordinates only when the rendered pattern belongs to that frame", () => {
    const displayed = new DataView(new ArrayBuffer(4));
    const pending = new DataView(new ArrayBuffer(4));
    const point = {point:true,region:[34,52,35,53],dp:displayed};
    expect(diffractionPositionLabel(point, displayed, 0, 0)).toBe("Point diffraction at (34, 52)");
    expect(diffractionPositionLabel({...point, dp:pending}, displayed, 0, 0)).toBe("Point diffraction");
    expect(diffractionPositionLabel({...point, dp:undefined}, displayed, 0, 0)).toBe("Point diffraction");
    expect(diffractionPositionLabel(undefined, displayed, 4, 9)).toBe("DP at (4, 9)");
  });
  it("names the preview's saved file and never carries a retained or stale file over", () => {
    const retainedFolder = "/scans";
    const retainedFile = "scan-1.qem";
    const preview = { scan: "series_2", sourceReceivedUtc: "2026-11-19T08:41:43.371+00:00" };
    const oldSave = { scan: "series_1", received_utc: "2026-11-19T08:41:14.941+00:00",
      folder: retainedFolder, file: retainedFile };
    const currentSave = { ...oldSave, scan: preview.scan, received_utc: preview.sourceReceivedUtc, file: "scan-2.qem" };

    expect(sourcePathForView(preview, oldSave, retainedFolder, retainedFile)).toBe("");
    expect(sourcePathForView(preview, { ...currentSave, received_utc: oldSave.received_utc }, retainedFolder, retainedFile)).toBe("");
    expect(sourcePathForView(preview, currentSave, retainedFolder, retainedFile)).toBe("/scans/scan-2.qem");
    expect(sourcePathForView(preview, { ...currentSave, file: "not saved (live)" }, retainedFolder, retainedFile)).toBe("not saved (live)");
    expect(sourcePathForView(null, currentSave, retainedFolder, retainedFile)).toBe("/scans/scan-1.qem");
    expect(sourcePathForView(null, currentSave, retainedFolder, retainedFile, false)).toBe("");
  });

  it("keeps a retained Multiple dataset's receipt off the incoming heading before preview rows", () => {
    const retained = { title: "Live · 3 · 262,144/262,144 ✓ · 2 kept", retainedScan: "series_3_grid", showingPreview: false };
    const heading = { ...retained, title: "Live · 4 · 3,840/262,144 · 2 kept" };
    const folder = "/scans";
    const file = "20261119-124800-3.qem";
    const preview = { scan: "series_4", sourceReceivedUtc: "2026-11-19T12:48:10+00:00" };
    const receipt = { scan: preview.scan, received_utc: preview.sourceReceivedUtc, folder, file: "20261119-124810-4.qem" };

    expect(sourcePathForView(null, undefined, folder, file, true, heading)).toBe("not saved (live)");
    expect(sourcePathForView(preview, { ...receipt, scan: "series_3" }, folder, file, true, heading)).toBe("not saved (live)");
    expect(sourcePathForView(preview, receipt, folder, file, true, heading)).toBe("/scans/20261119-124810-4.qem");
    expect(sourcePathForView(preview, receipt, folder, file, true, retained)).toBe("/scans/20261119-124800-3.qem");
    expect(sourcePathForView(null, undefined, folder, file, false, retained)).toBe("");
  });

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
    expect(titleNamesScan("Live · tilt 7/7 · 262,144/262,144 ✓ · 7 kept", "tilt 7/7")).toBe(true);
    expect(titleNamesScan("Live · tilt 7/7 · 7 kept", "tilt 7/70")).toBe(false);
  });
});
