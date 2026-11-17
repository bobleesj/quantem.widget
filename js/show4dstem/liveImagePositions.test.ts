import { describe, expect, it } from "vitest";
import { createLiveImage, positionValues } from "./liveImagePositions";

const batch = (scan: string, generation: number, start: number, values: number[], total = 6) =>
  [{ type: "live_image_positions", scan, generation, start, count: values.length, total }, new Float32Array(values)] as const;

describe("live image positions", () => {
  it("writes each batch at its start, in raster order", () => {
    const live = createLiveImage();
    expect(live.write(...batch("a", 1, 0, [1, 2]))).toBe(true);
    expect(live.write(...batch("a", 1, 2, [3]))).toBe(true);
    expect(Array.from(live.data()!.subarray(0, 3))).toEqual([1, 2, 3]);
    expect(live.received()).toBe(3);
    expect(Number.isNaN(live.data()![3])).toBe(true);
    expect(live.version()).toBe(2);
  });

  it("starts the first scan on the image under it", () => {
    const live = createLiveImage();
    live.write(...batch("a", 1, 0, [1]), () => new Float32Array([9, 9, 9, 9, 9, 9]));
    expect(Array.from(live.data()!)).toEqual([1, 9, 9, 9, 9, 9]);
  });

  it("draws a new scan over the scan before, never over nothing", () => {
    const live = createLiveImage();
    live.write(...batch("a", 1, 0, [1, 2, 3, 4, 5, 6]));
    live.write(...batch("b", 1, 0, [7, 8]));
    expect(live.scan()).toBe("b");
    expect(live.received()).toBe(2);
    expect(Array.from(live.data()!)).toEqual([7, 8, 3, 4, 5, 6]);
    expect(live.moved()).toBe(false);
  });

  it("a moved detector area starts again at position 0 and says so", () => {
    const live = createLiveImage();
    live.write(...batch("a", 1, 0, [1, 2, 3, 4]));
    live.write(...batch("a", 2, 0, [10, 20, 30]));
    expect(live.received()).toBe(3);
    expect(live.moved()).toBe(true);
    expect(Array.from(live.data()!.subarray(0, 4))).toEqual([10, 20, 30, 4]);
    live.write(...batch("b", 2, 0, [5]));
    expect(live.moved()).toBe(false);
  });

  it("a scan of another size starts a new image", () => {
    const live = createLiveImage();
    live.write(...batch("a", 1, 0, [1, 2, 3, 4, 5, 6]));
    live.write(...batch("b", 1, 0, [7], 4));
    expect(live.data()!.length).toBe(4);
    expect(live.data()![0]).toBe(7);
    expect(Number.isNaN(live.data()![1])).toBe(true);
  });

  it("refuses a batch outside the scan and cuts one that runs past its end", () => {
    const live = createLiveImage();
    expect(live.write(...batch("a", 1, 6, [1]))).toBe(false);
    expect(live.write(...batch("a", 1, -1, [1]))).toBe(false);
    expect(live.write(...batch("", 1, 0, [1]))).toBe(false);
    expect(live.write(...batch("a", 1, 5, [1, 2, 3]))).toBe(true);
    expect(live.received()).toBe(6);
    expect(live.data()![5]).toBe(1);
  });

  it("reads float32 values from an unaligned buffer", () => {
    const bytes = new Uint8Array(1 + 8);
    new Uint8Array(new Float32Array([1.5, -2]).buffer).forEach((byte, index) => { bytes[index + 1] = byte; });
    const view = new DataView(bytes.buffer, 1, 8);
    expect(Array.from(positionValues(view, 2)!)).toEqual([1.5, -2]);
    expect(Array.from(positionValues(view, 1)!)).toEqual([1.5]);
    expect(positionValues(undefined, 2)).toBeNull();
    expect(positionValues(new DataView(new ArrayBuffer(2)), 2)).toBeNull();
  });
});

describe("the rows that changed", () => {
  it("are the rows of the batches written since the last take", () => {
    const live = createLiveImage();
    expect(live.takeDirty(2)).toBeNull();
    live.write(...batch("a", 1, 1, [1, 2]));
    live.write(...batch("a", 1, 3, [3]));
    expect(live.takeDirty(2)).toEqual([0, 2]);
    expect(live.takeDirty(2)).toBeNull();
    live.write(...batch("a", 1, 4, [4, 5]));
    expect(live.takeDirty(2)).toEqual([2, 3]);
    expect(live.generation()).toBe(1);
  });
});
