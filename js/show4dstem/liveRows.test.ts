import { describe, expect, it } from "vitest";
import {
  binRows, createLiveRows, floatValues, liveInfo, liveShownImage, sameLiveInfo, sampledValues,
} from "./liveRows";

const rows = (scan: string, image: string, first: number, values: number[], extra: object = {}) =>
  [{ type: "live_image_rows", scan, image, first_row: first, rows: values.length / 2, cols: 2, total: 3, ...extra },
   new Float32Array(values)] as const;

describe("live image rows", () => {
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
      images: [], diffractionOnly: false, sourceId: "" });
    expect(sameLiveInfo(info, liveInfo({ ...frame, rows: 3 }))).toBe(true);
    expect(sameLiveInfo(info, liveInfo({ ...frame, scan: "b" }))).toBe(false);
    expect(liveInfo({})).toBeNull();
    expect(sameLiveInfo(null, null)).toBe(true);
    expect(sameLiveInfo(info, null)).toBe(false);
    const multiple = liveInfo({ ...frame, diffraction_only: true, source_id: "arriving" })!;
    expect(multiple).toMatchObject({ diffractionOnly: true, sourceId: "arriving" });
    expect(sameLiveInfo(multiple, info)).toBe(false);
    expect(sameLiveInfo(multiple, liveInfo({ ...frame, diffraction_only: true, source_id: "previous" }))).toBe(false);
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
