import { describe, expect, it, vi } from "vitest";
import { source112MeanDelta } from "./source112MeanDelta";

describe("source112 mean display deltas", () => {
  it("reuses owned displays and normalizes new counts exactly once per update", () => {
    const rawPatterns = [[100, 20, 1], [200, 40, 2]];
    const counts = new Uint32Array(2);
    const display = new Float32Array([NaN, 999999]);
    const owned = {} as GPUBuffer;
    const previous = [owned];
    const events: string[] = [];
    let last = new Uint32Array(3);
    const source = {
      integrate: vi.fn((mask: Uint32Array) => {
        events.push("integrate");
        let added = 0, removed = 0;
        for (let q = 0; q < mask.length; q++) {
          if (mask[q] && !last[q]) added++;
          if (!mask[q] && last[q]) removed++;
        }
        rawPatterns.forEach((pattern, scan) => {
          counts[scan] = pattern.reduce((sum, value, q) => sum + (mask[q] ? value : 0), 0);
        });
        last = mask.slice();
        return { added, removed, full: false };
      }),
      normalizeDisplayBuffers: vi.fn((buffers: GPUBuffer[], area: number) => {
        events.push("normalize");
        expect(buffers).toBe(previous);
        counts.forEach((value, scan) => { display[scan] = value / area; });
      }),
      imageBuffersF32: vi.fn(() => { throw new Error("Unexpected preliminary conversion"); }),
    };
    const effective = new Uint32Array([1, 1, 0]);
    const first = source112MeanDelta(source, effective, previous);
    expect(first).toEqual({ buffers: previous, path: "delta", addedPixels: 2, removedPixels: 0 });
    expect(first.buffers[0]).toBe(owned);
    expect(Array.from(counts)).toEqual([120, 240]);
    expect(Array.from(display)).toEqual([60, 120]);
    expect(effective).toEqual(new Uint32Array([1, 1, 0]));

    // Change both the count image and its divisor; stale display values cannot
    // be incremented or divided again to obtain this independent result.
    source112MeanDelta(source, new Uint32Array([0, 0, 1]), previous);
    expect(Array.from(counts)).toEqual([1, 2]);
    expect(Array.from(display)).toEqual([1, 2]);
    expect(source.normalizeDisplayBuffers.mock.calls.map(call => call[1])).toEqual([2, 1]);
    expect(source.imageBuffersF32).not.toHaveBeenCalled();
    expect(events).toEqual(["integrate", "normalize", "integrate", "normalize"]);
  });

  it("retains the empty-mask divisor and does not normalize failed updates", () => {
    const previous = [{} as GPUBuffer];
    const source = {
      integrate: vi.fn(() => ({ added: 0, removed: 1, full: false })),
      normalizeDisplayBuffers: vi.fn(),
    };
    source112MeanDelta(source, new Uint32Array(3), previous);
    expect(source.normalizeDisplayBuffers).toHaveBeenCalledExactlyOnceWith(previous, 1);
    source.integrate.mockImplementationOnce(() => { throw new Error("Source closed"); });
    expect(() => source112MeanDelta(source, new Uint32Array(3), previous)).toThrow("Source closed");
    expect(source.normalizeDisplayBuffers).toHaveBeenCalledTimes(1);
  });
});
