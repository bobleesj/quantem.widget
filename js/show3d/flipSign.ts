/**
 * Show3D `flip_sign`: display every frame as -value without touching the frame data.
 *
 * With display limits [vmin, vmax] a pixel maps to t = (v - vmin) / (vmax - vmin) and colour
 * lut[t]. The negated frame shown between the negated limits [-vmax, -vmin] maps to
 * (-v + vmax) / (vmax - vmin) = 1 - t, so the exact negative picture is the same frame, the same
 * limits and the colormap table read backwards. Signed log scale keeps this: slog(-v) = -slog(v).
 * A flip therefore swaps one 256-entry table on the GPU and never rewrites a frame texture; the
 * numbers beside the image (limits, histogram, statistics, cursor value) are negated when shown.
 */

import { COLORMAPS } from "../display/colormaps";

const SIGN_FLIPPED_SUFFIX = "|flip_sign";
const signFlippedLuts = new Map<string, Uint8Array>();

/** `lut` (256 RGB entries) read backwards: entry i takes the colour of entry 255 - i. */
export function signFlippedLut(lut: Uint8Array): Uint8Array {
  const flipped = new Uint8Array(lut.length);
  const entries = lut.length / 3;
  for (let entry = 0; entry < entries; entry++) {
    const source = (entries - 1 - entry) * 3;
    flipped[entry * 3] = lut[source];
    flipped[entry * 3 + 1] = lut[source + 1];
    flipped[entry * 3 + 2] = lut[source + 2];
  }
  return flipped;
}

/**
 * The colormap table a frame is drawn with, and the name the GPU engine caches it under.
 * The flipped table has its own name, so the engine holds both and a toggle only rebinds.
 */
export function displayColormap(name: string | null | undefined, flipSign: boolean, fallback = "inferno"): { name: string; lut: Uint8Array } {
  const known = name && COLORMAPS[name] ? name : (COLORMAPS[fallback] ? fallback : "inferno");
  if (!flipSign) return { name: known, lut: COLORMAPS[known] };
  const flippedName = `${known}${SIGN_FLIPPED_SUFFIX}`;
  let lut = signFlippedLuts.get(flippedName);
  if (!lut) {
    lut = signFlippedLut(COLORMAPS[known]);
    signFlippedLuts.set(flippedName, lut);
  }
  return { name: flippedName, lut };
}

/** A shown value: -value when flipped (0 stays 0, never -0). */
export function shownValue(value: number, flipSign: boolean): number {
  return flipSign ? 0 - value : value;
}

/** Shown display limits: [-vmax, -vmin] when flipped. */
export function shownLimits(vmin: number, vmax: number, flipSign: boolean): { vmin: number; vmax: number } {
  return flipSign ? { vmin: 0 - vmax, vmax: 0 - vmin } : { vmin, vmax };
}

/** Shown statistics: mean negated, min and max swapped and negated, std unchanged. */
export function shownStats<Stats extends { mean: number; min: number; max: number; std: number }>(stats: Stats, flipSign: boolean): Stats {
  return flipSign ? { ...stats, mean: 0 - stats.mean, min: 0 - stats.max, max: 0 - stats.min } : stats;
}

/** Histogram bins of the shown values: the stored histogram mirrored about the centre of its range. */
export function shownHistogramBins(bins: number[] | null, flipSign: boolean): number[] | null {
  return flipSign && bins ? [...bins].reverse() : bins;
}

/**
 * Where the shown histogram sits: its range negated ([-dataMax, -dataMin]) and the clip handles
 * mirrored in percent, so the handles stay on the same stored values.
 */
export function shownHistogramWindow(
  dataMin: number,
  dataMax: number,
  vminPct: number,
  vmaxPct: number,
  flipSign: boolean,
): { dataMin: number; dataMax: number; vminPct: number; vmaxPct: number } {
  if (!flipSign) return { dataMin, dataMax, vminPct, vmaxPct };
  return { dataMin: 0 - dataMax, dataMax: 0 - dataMin, vminPct: 100 - vmaxPct, vmaxPct: 100 - vminPct };
}

/** Clip handles moved on the shown histogram, back in the stored-value percent the traits hold. */
export function storedClipPercent(shownMinPct: number, shownMaxPct: number, flipSign: boolean): [number, number] {
  return flipSign ? [100 - shownMaxPct, 100 - shownMinPct] : [shownMinPct, shownMaxPct];
}

/**
 * Whether frame `frame` is shown negated: the stack switch (`flip_sign`) or the frame's own entry
 * in `frame_flip_signs`, not both. Two negations cancel, so the stack switch still inverts every frame.
 */
export function frameSignFlipped(flipSign: boolean, frameFlipSigns: readonly boolean[] | null | undefined, frame: number): boolean {
  return flipSign !== !!frameFlipSigns?.[frame];
}

/** A per-frame mark list (`frame_flip_signs`) of `frames` entries with frame `frame` toggled. */
export function toggledFrameMark(marks: readonly boolean[] | null | undefined, frames: number, frame: number): boolean[] {
  return Array.from({ length: frames }, (_, index) => !!marks?.[index] !== (index === frame));
}
