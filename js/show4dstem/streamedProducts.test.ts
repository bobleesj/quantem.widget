import { describe, expect, it } from "vitest";
import { createStreamedProducts, streamedProductsOf } from "./streamedProducts";

describe("streamed product images", () => {
  it("keeps each scan's image by its identity and shows a blank for a scan without one", () => {
    const store = createStreamedProducts();
    expect(store.write({ label: "Direct", source: "a", rows: 2, cols: 3 }, new Float32Array([1, 2, 3, 4, 5, 6]))).toBe(true);
    expect(Array.from(store.image("Direct", "a", 2, 3))).toEqual([1, 2, 3, 4, 5, 6]);
    expect(store.image("Direct", "b", 2, 3).every(Number.isNaN)).toBe(true);
    expect(store.image("COM_row", "a", 2, 3).every(Number.isNaN)).toBe(true);
    expect(store.image("Direct", "a", 3, 3).every(Number.isNaN)).toBe(true);   // another size: not this image
  });

  it("changes its version only when an image arrives or a scan is forgotten", () => {
    const store = createStreamedProducts();
    const before = store.version();
    expect(store.write({ label: "Direct", source: "a", rows: 1, cols: 2 }, new Float32Array(1))).toBe(false);
    expect(store.version()).toBe(before);
    store.write({ label: "Direct", source: "a", rows: 1, cols: 2 }, new Float32Array([7, 8]));
    store.write({ label: "Direct", source: "b", rows: 1, cols: 2 }, new Float32Array([9, 10]));
    const written = store.version();
    store.keep(["a", "b"]);
    expect(store.version()).toBe(written);
    store.keep(["b"]);
    expect(store.version()).toBe(written + 1);
    expect(store.has("Direct", "a")).toBe(false);
    expect(Array.from(store.image("Direct", "b", 1, 2))).toEqual([9, 10]);
  });

  it("is one store per widget model", () => {
    const model = {};
    expect(streamedProductsOf(model)).toBe(streamedProductsOf(model));
    expect(streamedProductsOf({})).not.toBe(streamedProductsOf(model));
  });
});
