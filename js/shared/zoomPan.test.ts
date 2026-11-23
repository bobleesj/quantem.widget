import { describe, expect, it } from "vitest";
import { contrastRange, readAt } from "./zoomPan";

describe("hover readout", () => {
  it("reads the pixel under a place given as fractions, and its place in nm", () => {
    const data = new Float32Array([0, 1, 2, 3, 4, 5]);  // 2 rows × 3 cols
    const read = readAt(data, 2, 3, [0.9, 0.6], 2.0);
    expect([read.row, read.col, read.value]).toEqual([1, 2, 5]);
    expect(read.nm).toEqual([0.2, 0.4]);
    expect(readAt(data, 2, 3, [1.5, -1]).value).toBe(2);  // clamped inside the field
  });
});

describe("the view's contrast", () => {
  it("gives images shown together one range, and a change map a range centred on zero", () => {
    const before = Float32Array.from({ length: 101 }, (_, index) => index);  // 0..100
    const after = Float32Array.from({ length: 101 }, (_, index) => index + 100);  // 100..200
    const [low, high] = contrastRange([before, after], { low: 10, high: 90, cmap: null, fft: 35 });
    expect(low).toBeGreaterThan(15); expect(low).toBeLessThan(25);
    expect(high).toBeGreaterThan(175); expect(high).toBeLessThan(185);
    const change = Float32Array.from({ length: 201 }, (_, index) => index - 50);  // −50..150
    const [lower, upper] = contrastRange([change], { low: 10, high: 90, cmap: null, fft: 35 }, true);
    expect(lower).toBe(-upper);
  });
});
