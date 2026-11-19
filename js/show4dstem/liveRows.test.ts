import { describe, expect, it } from "vitest";
import {
  binRows, createLiveRows, floatValues, liveImageAvailable, liveInfo, liveShownImage, sameLiveInfo, sampledValues,
  type LiveFrame, type LiveInfo,
} from "./liveRows";

const rows = (scan: string, image: string, first: number, values: number[], extra: object = {}) =>
  [{ type: "live_image_rows", scan, image, first_row: first, rows: values.length / 2, cols: 2, total: 3, ...extra },
   new Float32Array(values)] as const;

describe("live image rows", () => {
  it("keeps the actual completed Direct scan only within its Full Live generation", () => {
    const live = createLiveRows();
    const metadata = { scan: "live_4", total: 3, cols: 2, detRows: 4, detCols: 4, mode: "Direct" };
    const preview = liveInfo({ ...metadata, completed_direct_generation: "first" })!;
    live.write(...rows("live_2", "Direct", 0, [10, 11, 12, 13, 14, 15],
      { completed_direct_generation: "first" }));
    const phase = live.get("Direct")!;
    expect(liveImageAvailable(phase, preview)).toBe(true);
    expect(phase.scan).toBe("live_2");
    expect(liveImageAvailable(phase, { ...preview, frozen: true })).toBe(true);
    const failed = liveInfo({ ...metadata, completed_direct_generation: "first", completed_direct_failed: true,
      frozen: true })!;
    expect(failed.completedDirectFailed).toBe(true);
    expect(sameLiveInfo({ ...preview, frozen: true }, failed)).toBe(false);
    expect(liveImageAvailable(phase, failed)).toBe(true); // prior complete phase remains attributed to live_2
    expect(liveImageAvailable(phase, { ...preview, completedGeneration: "second" })).toBe(false);
    expect(liveImageAvailable(phase, liveInfo(metadata)!)).toBe(false);
    expect(sameLiveInfo(preview, { ...preview, completedGeneration: "second" })).toBe(false);
  });

  it("shows the detector in Region without clearing the Full Direct preference", () => {
    const live = createLiveRows();
    live.write(...rows("live_1", "Direct", 0, [1, 2, 3, 4, 5, 6]));
    const full = { scan: "live_2", total: 3, cols: 2, detRows: 4, detCols: 4, mode: "Direct" };
    const crop: LiveFrame = { ...full, mode: "ROI", region: [0, 0, 3, 2] };
    const region = liveInfo(crop)!;
    const image = live.get("Direct", 3, 2);
    const available = (preview: LiveInfo) => (name: string) =>
      name === "roi" || (name === "Direct" && liveImageAvailable(image, preview));

    expect(sameLiveInfo(region, liveInfo({ ...full, mode: "ROI" }))).toBe(false);
    expect(liveInfo({ ...crop, point: true })?.regional).toBeUndefined();
    expect(liveShownImage("Direct", "Direct", available(region))).toBe("roi");
    expect(liveShownImage("Direct", "roi", available(liveInfo({ ...full, scan: "live_3" })!))).toBe("Direct");
  });

  it.each(["COM_row", "COM_col", "DPC", "Direct"])("holds %s pixels until the next Live frame has rows", name => {
    const live = createLiveRows();
    live.write(...rows("live_1", name, 0, [1, 2, 3, 4, 5, 6]));
    const image = live.get(name, 3, 2)!;
    const next = liveInfo({ scan: "live_2", total: 3, cols: 2, detRows: 4, detCols: 4 })!;
    expect(liveImageAvailable(image, next)).toBe(true);
    expect(liveImageAvailable(image, { ...next, scan: "series_1" })).toBe(false);
    expect(sameLiveInfo(next, { ...next, frozen: true })).toBe(false);
    // the rotation DPC was integrated with travels with the frame (2026-11-18): shown beside Flip
    expect(liveInfo({ scan: "live_2", total: 3, cols: 2, detRows: 4, detCols: 4, dpc_rotation_deg: 3.9 })!.dpcRotationDeg).toBe(3.9);
    expect(next.dpcRotationDeg).toBeNull();
    expect(sameLiveInfo(next, { ...next, dpcRotationDeg: 183.9 })).toBe(false);
    live.write(...rows("live_2", name, 0, [7, 8]));
    expect(Array.from(image.data)).toEqual([7, 8, 3, 4, 5, 6]);
  });

  it("writes each message's rows where they belong and remembers which rows changed", () => {
    const live = createLiveRows();
    expect(live.write(...rows("a", "COM_row", 0, [1, 2]))).toBe(true);
    expect(live.write(...rows("a", "COM_row", 1, [3, 4]))).toBe(true);
    const image = live.get("COM_row", 3, 2)!;
    expect(Array.from(image.data.subarray(0, 4))).toEqual([1, 2, 3, 4]);
    expect(Number.isNaN(image.data[4])).toBe(true);
    expect(image.grid).toBe(image.data);
    expect(image.measured).toBe(2);
    expect(live.takeDirty("COM_row")).toEqual([0, 2]);
    expect(live.takeDirty("COM_row")).toBeNull();
    live.write(...rows("a", "COM_row", 2, [5, 6]));
    expect(live.takeDirty("COM_row")).toEqual([2, 3]);
    expect(live.version()).toBe(3);
  });

  it("draws a new scan over the scan before, in the same array", () => {
    const live = createLiveRows();
    live.write(...rows("a", "DPC", 0, [1, 2, 3, 4, 5, 6]));
    const before = live.get("DPC")!.data;
    live.write(...rows("b", "DPC", 0, [7, 8]));
    const image = live.get("DPC")!;
    expect(image.data).toBe(before);
    expect(image.scan).toBe("b");
    expect(image.measured).toBe(1);
    expect(Array.from(image.data)).toEqual([7, 8, 3, 4, 5, 6]);
  });

  it("takes the image to draw beneath a scan without counting it as measured", () => {
    const live = createLiveRows();
    live.write(...rows("b", "DPC", 0, [9, 9, 9, 9, 9, 9], { under: true }));
    expect(live.get("DPC")!.measured).toBe(0);
    live.write(...rows("b", "DPC", 0, [1, 2]));
    expect(Array.from(live.get("DPC")!.data)).toEqual([1, 2, 9, 9, 9, 9]);
    expect(live.get("DPC")!.measured).toBe(1);
  });

  it("recenters only the measured raw CoM prefix and preserves the image underneath", () => {
    const live = createLiveRows();
    const scale = 0.17000000178813934;
    live.write(...rows("live_8", "COM_row", 0, [8, 9, 4, 5, 2, NaN], { under: true }));
    live.write(...rows("live_8", "COM_row", 0, [3.125, 1.125], { com_centering: [2.125, scale] }));
    expect(Array.from(live.get("COM_row")!.data)).toEqual([scale, -scale, 4, 5, 2, NaN]);
    live.takeDirty("COM_row");

    const next = { com_centering: [3.5625, scale] };
    expect(live.write(...rows("live_8", "COM_row", 2, [9.5, 0.5], next))).toBe(false);
    expect(live.write(...rows("live_9", "COM_row", 1, [9.5, 0.5], next))).toBe(false);
    expect(live.write(...rows("live_8", "COM_row", 1, [9.5, 0.5], next))).toBe(true);

    // Exact NumPy float32 (samples - samples.mean()) * 0.17.
    expect(Array.from(live.get("COM_row")!.data)).toEqual([
      -0.07437500357627869, -0.4143750071525574, 1.009374976158142, -0.5206249952316284, 2, NaN,
    ]);
    expect(live.get("COM_row")!.measured).toBe(2);
    expect(live.takeDirty("COM_row")).toEqual([0, 2]);

    // Conventional centered rows are still accepted after a protocol change.
    expect(live.write(...rows("live_8", "COM_row", 0, [1, 2, 3, 4]))).toBe(true);
    expect(Array.from(live.get("COM_row")!.data)).toEqual([1, 2, 3, 4, 2, NaN]);
    expect(live.write(...rows("live_8", "COM_row", 2, [9.5, 0.5], next))).toBe(false);
  });

  it("keeps an upsampled image at full resolution and its mean on the scan grid, row by row", () => {
    const live = createLiveRows();
    const message = { type: "live_image_rows", scan: "a", image: "Direct", first_row: 1, rows: 1, cols: 2, total: 2, upsample: 2 };
    expect(live.write(message, new Float32Array([1, 3, 10, 10, 5, 7, 20, 20]))).toBe(true);
    const image = live.get("Direct", 2, 2)!;
    expect(image.data.length).toBe(16);
    expect(Array.from(image.data.subarray(8))).toEqual([1, 3, 10, 10, 5, 7, 20, 20]);
    expect(Array.from(image.grid.subarray(2))).toEqual([4, 15]);
    expect(Number.isNaN(image.grid[0])).toBe(true);
    expect(live.takeDirty("Direct")).toEqual([1, 2]);
  });

  it("starts again when the size or the upsampling changes, and refuses rows that do not fit", () => {
    const live = createLiveRows();
    live.write(...rows("a", "Direct", 0, [1, 2, 3, 4, 5, 6]));
    live.write({ type: "live_image_rows", scan: "b", image: "Direct", first_row: 0, rows: 1, cols: 2, total: 3, upsample: 2 },
      new Float32Array(8).fill(2));
    expect(live.get("Direct")!.factor).toBe(2);
    expect(Number.isNaN(live.get("Direct")!.data[8])).toBe(true);
    expect(live.get("Direct", 4, 2)).toBeNull();
    expect(live.write(...rows("b", "Direct", 3, [1, 2]))).toBe(false);
    expect(live.write(...rows("b", "Direct", 0, [1, 2], { upsample: 2 }))).toBe(false);   // too few values for 2 x
    expect(live.write({ ...rows("b", "x", 0, [1, 2])[0], rows: 2 }, new Float32Array([1, 2]))).toBe(false);
    expect(live.write(rows("b", "x", 0, [1, 2])[0], null)).toBe(false);
  });

  it("forgets every image when the scans were cleared", () => {
    const live = createLiveRows();
    live.write(...rows("a", "Direct", 0, [1, 2]));
    live.clear();
    expect(live.get("Direct")).toBeNull();
    expect(live.takeDirty("Direct")).toBeNull();
  });

  it("writes the whole images of a frame that still carries its pixels", () => {
    const live = createLiveRows();
    const bytes = (values: number[]) => new DataView(new Float32Array(values).buffer);
    const frame = { scan: "a", rows: 1, cols: 2, total: 2, detRows: 4, detCols: 4, mode: "Direct", upsample: 1,
                    phase: bytes([1, 2]), roi: bytes([3, 4]), com_row: bytes([5, 6]), dpc: bytes([7]) };
    expect(live.writeFrame(frame)).toBe(true);
    expect(Array.from(live.get("Direct")!.data.subarray(0, 2))).toEqual([1, 2]);
    expect(Array.from(live.get("roi")!.data.subarray(0, 2))).toEqual([3, 4]);
    expect(Array.from(live.get("COM_row")!.data.subarray(0, 2))).toEqual([5, 6]);
    expect(live.get("DPC")).toBeNull();                                   // its bytes do not match the rows
    expect(live.writeFrame({ ...frame, phase: undefined, roi: undefined, com_row: undefined, dpc: undefined })).toBe(false);
    expect(live.writeFrame({})).toBe(false);
  });
});

describe("the scan arriving", () => {
  it("is described without its pixels, and is the same while only rows arrive", () => {
    const frame = { scan: "a", rows: 8, cols: 4, total: 8, detRows: 2, detCols: 2, mode: "Direct", com_unit: "mrad" };
    const info = liveInfo(frame)!;
    expect(info).toEqual({ scan: "a", total: 8, cols: 4, detRows: 2, detCols: 2, mode: "Direct", comUnit: "mrad",
      images: [], diffractionOnly: false, sourceId: "", frozen: false, dpcRotationDeg: null,
      completedGeneration: undefined, waitingForDirectStripes: false });
    expect(sameLiveInfo(info, liveInfo({ ...frame, rows: 3 }))).toBe(true);
    expect(sameLiveInfo(info, liveInfo({ ...frame, scan: "b" }))).toBe(false);
    expect(liveInfo({})).toBeNull();
    expect(sameLiveInfo(null, null)).toBe(true);
    expect(sameLiveInfo(info, null)).toBe(false);
    const multiple = liveInfo({ ...frame, diffraction_only: true, source_id: "arriving" })!;
    expect(multiple).toMatchObject({ diffractionOnly: true, sourceId: "arriving" });
    expect(sameLiveInfo(multiple, info)).toBe(false);
    expect(sameLiveInfo(multiple, liveInfo({ ...frame, diffraction_only: true, source_id: "previous" }))).toBe(false);
    const visits = liveInfo({ ...frame, visit_path: true });
    expect(visits?.visitPath).toBe(true);
    expect(sameLiveInfo(info, visits)).toBe(false);
  });

  it("advertises only known live COM and DPC products as they become available", () => {
    const frame = { scan: "a", total: 8, cols: 4, detRows: 2, detCols: 2 };
    const before = liveInfo(frame)!;
    const preview = liveInfo({ ...frame, images: ["DPC", "COM_col", "COM_row", "DPC", "unknown"] })!;
    expect(before.images).toEqual([]);
    expect(preview.images).toEqual(["COM_row", "COM_col", "DPC"]);
    expect(sameLiveInfo(before, preview)).toBe(false);
    expect(liveShownImage("COM_col", "roi", name => preview.images.includes(name))).toBe("COM_col");
  });

  it("uses the preview's own positive row and column sampling when supplied", () => {
    const frame = { scan: "a", total: 8, cols: 4, detRows: 2, detCols: 2 };
    const sampling = liveInfo({ ...frame, scan_sampling_A: [0.22, 0.31] })!;
    expect(sampling.scanSamplingA).toEqual([0.22, 0.31]);
    expect(sameLiveInfo(sampling, liveInfo({ ...frame, scan_sampling_A: [0.22, 0.31] }))).toBe(true);
    expect(sameLiveInfo(sampling, liveInfo({ ...frame, scan_sampling_A: [0.22, 0.62] }))).toBe(false);
    expect(sameLiveInfo(sampling, liveInfo(frame))).toBe(false);
    expect(liveInfo({ ...frame, scan_sampling_A: [0.22, 0] })?.scanSamplingA).toBeUndefined();
    expect(liveInfo({ ...frame, scan_sampling_A: [NaN, 0.31] })?.scanSamplingA).toBeUndefined();
  });

  it("shows the chosen image when the scan sends it, else the one before, else the detector image", () => {
    const sends = (names: string[]) => (name: string) => names.includes(name);
    expect(liveShownImage("Direct", "roi", sends(["roi", "Direct"]))).toBe("Direct");
    expect(liveShownImage("COM_row", "Direct", sends(["roi", "Direct"]))).toBe("Direct");
    expect(liveShownImage("Direct", "COM_row", sends(["roi"]))).toBe("roi");
    expect(liveShownImage("roi", "Direct", sends(["roi", "Direct"]))).toBe("roi");
    expect(liveShownImage("Direct", null, sends(["roi"]))).toBe("roi");
  });
});

describe("helpers", () => {
  it("reads float32 values from an aligned buffer in place and copies an unaligned one", () => {
    const values = new Float32Array([1, 2, 3]);
    expect(floatValues(new DataView(values.buffer))!.buffer).toBe(values.buffer);
    const shifted = new Uint8Array(13);
    shifted.set(new Uint8Array(values.buffer), 1);
    expect(Array.from(floatValues(new DataView(shifted.buffer, 1, 12))!)).toEqual([1, 2, 3]);
    expect(Array.from(floatValues(values.buffer as ArrayBuffer)!)).toEqual([1, 2, 3]);
    expect(floatValues(undefined)).toBeNull();
  });

  it("bins only the rows asked for", () => {
    const data = new Float32Array([1, 3, 5, 7, 1, 3, 5, 7, 2, 2, 4, 4, 2, 2, 4, 4]);
    const grid = new Float32Array(4).fill(-1);
    binRows(data, grid, 2, 2, 1, 2);
    expect(Array.from(grid)).toEqual([-1, -1, 2, 4]);
  });

  it("samples a large image evenly and leaves a small one as it is", () => {
    const data = new Float32Array(100).map((_, index) => index);
    expect(sampledValues(data, 100, 200, null)).toBe(data);
    expect(sampledValues(data, 10, 200, null).length).toBe(10);
    const samples = sampledValues(data, 100, 10, null);
    expect(Array.from(samples)).toEqual([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]);
    expect(sampledValues(data, 50, 10, samples)).toBe(samples);
    expect(samples[9]).toBe(45);
  });
});
