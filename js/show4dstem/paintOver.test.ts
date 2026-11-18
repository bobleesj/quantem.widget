import { describe, expect, it } from "vitest";

import { createPaintOver, paintedOver } from "./paintOver";

function image(values: number[]): DataView {
  return new DataView(new Float32Array(values).buffer);
}

function values(view: DataView | null): number[] {
  return Array.from(new Float32Array(view!.buffer, view!.byteOffset, view!.byteLength / 4));
}

describe("paint over the last scan", () => {
  it("takes the measured rows from the new scan and the rest from the last image of its kind", () => {
    const paint = createPaintOver();
    paint.over("BF", 3, 2, image([1, 1, 1, 1, 1, 1]));
    const drawn = paint.over("BF", 3, 2, image([5, 6, NaN, NaN, NaN, NaN]));
    expect(values(drawn)).toEqual([5, 6, 1, 1, 1, 1]);
    // more rows of the same scan: its own rows stay, the old ones are replaced row by row
    expect(values(paint.over("BF", 3, 2, image([5, 6, 7, 8, NaN, NaN])))).toEqual([5, 6, 7, 8, 1, 1]);
  });

  it("keeps each kind apart", () => {
    const paint = createPaintOver();
    paint.over("BF", 1, 2, image([1, 1]));
    paint.over("ADF", 1, 2, image([9, 9]));
    expect(values(paint.over("ADF", 1, 2, image([3, NaN])))).toEqual([3, 9]);
    expect(values(paint.over("BF", 1, 2, image([NaN, 4])))).toEqual([1, 4]);
  });

  it("starts a scan of another shape from nothing", () => {
    const paint = createPaintOver();
    paint.over("BF", 2, 2, image([1, 1, 1, 1]));
    const drawn = paint.over("BF", 1, 4, image([2, NaN, NaN, NaN]));
    expect(values(drawn)).toEqual([2, NaN, NaN, NaN]);
  });

  it("returns a whole image as it is", () => {
    const paint = createPaintOver();
    const whole = image([1, 2]);
    expect(paint.over("BF", 1, 2, whole)).toBe(whole);
    const first = image([NaN, 3]);
    expect(createPaintOver().over("BF", 1, 2, first)).toBe(first);
  });

  it("starts over after clear", () => {
    const paint = createPaintOver();
    paint.over("BF", 1, 2, image([1, 1]));
    paint.clear();
    expect(values(paint.over("BF", 1, 2, image([2, NaN])))).toEqual([2, NaN]);
  });
});

describe("a viewer not embedded", () => {
  it("draws each image as it came", () => {
    const paint = createPaintOver();
    paintedOver(false, paint, "BF", 1, 2, image([1, 1]));
    const arriving = image([2, NaN]);
    expect(paintedOver(false, paint, "BF", 1, 2, arriving)).toBe(arriving);
    expect(values(paintedOver(true, paint, "BF", 1, 2, image([1, 1])))).toEqual([1, 1]);
    expect(values(paintedOver(true, paint, "BF", 1, 2, image([2, NaN])))).toEqual([2, 1]);
  });
});
