import { describe, expect, it } from "vitest";

import { COLORMAPS, applyColormap } from "../display/colormaps";
import {
  displayColormap,
  frameSignFlipped,
  shownHistogramBins,
  shownHistogramWindow,
  shownLimits,
  shownStats,
  shownValue,
  signFlippedLut,
  storedClipPercent,
  toggledFrameMark,
} from "./flipSign";

/** Gray LUT entry the CPU colormap reference picks for each value (red channel of a gray ramp). */
function grayLevels(data: Float32Array, lut: Uint8Array, vmin: number, vmax: number): number[] {
  const rgba = new Uint8ClampedArray(data.length * 4);
  applyColormap(data, rgba, lut, vmin, vmax);
  return Array.from(data, (_, pixel) => rgba[pixel * 4]);
}

describe("flip_sign display", () => {
  // --- the picture ----------------------------------------------------------------------------

  it("draws the negated frame between the negated limits by reading the colormap backwards", () => {
    // A ramp with one bright blob and values beyond both limits, so clipping is covered too.
    const stored = Float32Array.from({ length: 4096 }, (_, pixel) => -0.7 + pixel / 2048 + (pixel === 1000 ? 5 : 0));
    const negated = stored.map(value => -value);
    const [vmin, vmax] = [-0.4, 1.1];
    const gray = COLORMAPS.gray;
    const flipped = displayColormap("gray", true).lut;
    const shown = grayLevels(stored, flipped, vmin, vmax);
    const plain = grayLevels(stored, gray, vmin, vmax);
    // exact mirror of the picture on screen: level L becomes 255 - L, pixel by pixel
    expect(shown).toEqual(plain.map(level => 255 - level));
    // and the same picture as colormapping -value between [-vmax, -vmin], to within the one level
    // that truncation (trunc(255 t) against trunc(255 (1 - t))) leaves between the two
    const limits = shownLimits(vmin, vmax, true);
    const reference = grayLevels(negated, gray, limits.vmin, limits.vmax);
    const worst = Math.max(...shown.map((level, pixel) => Math.abs(level - reference[pixel])));
    expect(worst).toBeLessThanOrEqual(1);
    expect(shown[1000]).toBe(0);                    // the bright blob is the darkest pixel
    expect(shown[0]).toBe(255);                     // the lowest stored value is the brightest
  });

  it("keeps the stored table and its engine name when off, and caches one flipped table per colormap", () => {
    expect(displayColormap("viridis", false)).toEqual({ name: "viridis", lut: COLORMAPS.viridis });
    const flipped = displayColormap("viridis", true);
    expect(flipped.name).not.toBe("viridis");
    expect(displayColormap("viridis", true).lut).toBe(flipped.lut);
    expect(Array.from(signFlippedLut(flipped.lut))).toEqual(Array.from(COLORMAPS.viridis));
    expect(Array.from(flipped.lut.subarray(0, 3))).toEqual(Array.from(COLORMAPS.viridis.subarray(255 * 3)));
    // an unknown name falls back the way the viewer does
    expect(displayColormap("no-such-map", true, "plasma").lut).toBe(displayColormap("plasma", true).lut);
    expect(displayColormap(undefined, false).name).toBe("inferno");
  });

  // --- the numbers beside it ------------------------------------------------------------------

  it("negates the limits, statistics and cursor value", () => {
    expect(shownLimits(-2, 5, true)).toEqual({ vmin: -5, vmax: 2 });
    expect(shownLimits(-2, 5, false)).toEqual({ vmin: -2, vmax: 5 });
    expect(shownStats({ mean: 1.5, min: -2, max: 5, std: 0.25, panel: 3 }, true)).toEqual({ mean: -1.5, min: -5, max: 2, std: 0.25, panel: 3 });
    expect(shownValue(3.25, true)).toBe(-3.25);
    expect(Object.is(shownValue(0, true), 0)).toBe(true);
    expect(shownValue(3.25, false)).toBe(3.25);
  });

  it("mirrors the histogram and maps its clip handles back to stored percentages", () => {
    const bins = Array.from({ length: 256 }, (_, bin) => bin);
    expect(shownHistogramBins(bins, true)).toEqual([...bins].reverse());
    expect(shownHistogramBins(bins, false)).toBe(bins);
    expect(shownHistogramBins(null, true)).toBeNull();
    const shown = shownHistogramWindow(-2, 6, 10, 75, true);
    expect(shown).toEqual({ dataMin: -6, dataMax: 2, vminPct: 25, vmaxPct: 90 });
    // the handles sit on the negated stored limits: stored window [-1.2, 4] is shown as [-4, 1.2]
    const span = shown.dataMax - shown.dataMin;
    expect(shown.dataMin + span * shown.vminPct / 100).toBeCloseTo(-4, 12);
    expect(shown.dataMin + span * shown.vmaxPct / 100).toBeCloseTo(1.2, 12);
    expect(storedClipPercent(shown.vminPct, shown.vmaxPct, true)).toEqual([10, 75]);
    expect(storedClipPercent(10, 75, false)).toEqual([10, 75]);
  });

  // --- one frame ------------------------------------------------------------------------------

  it("shows a frame negated when exactly one of the stack switch and its own entry is set", () => {
    const flips = [false, false, true, false];
    expect(flips.map((_, frame) => frameSignFlipped(false, flips, frame))).toEqual([false, false, true, false]);
    // the stack switch inverts every frame: the flipped frame is back to its stored sign
    expect(flips.map((_, frame) => frameSignFlipped(true, flips, frame))).toEqual([true, true, false, true]);
    // a list shorter than the stack (a frame appended a moment ago) and no list at all read as not flipped
    expect(frameSignFlipped(false, flips, 4)).toBe(false);
    expect(frameSignFlipped(true, null, 0)).toBe(true);
    expect(frameSignFlipped(false, undefined, 0)).toBe(false);
  });

  it("toggles one frame of a mark list and keeps one entry per frame", () => {
    expect(toggledFrameMark([false, false, false, false], 4, 2)).toEqual([false, false, true, false]);
    expect(toggledFrameMark([false, false, true, false], 4, 2)).toEqual([false, false, false, false]);
    expect(toggledFrameMark([true], 3, 1)).toEqual([true, true, false]);     // a short list grows to the stack
    expect(toggledFrameMark(null, 2, 0)).toEqual([true, false]);
    expect(toggledFrameMark([true, false, true], 2, 1)).toEqual([true, true]);   // and a long one is cut to it
  });
});
