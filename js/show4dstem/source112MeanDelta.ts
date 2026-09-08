import type { Source112ResidentSet } from "../.generated/engine/detector/compute/webgpu/source112";

/** Integrate exact counts, then paint their mean directly or refresh float displays. */
export function source112MeanDelta(
  source: Pick<Source112ResidentSet, "integrate" | "normalizeDisplayBuffers">,
  mask: Uint32Array,
  previous: GPUBuffer[],
  paintMean?: (area: number) => boolean,
) {
  const delta = source.integrate(mask);
  // The caller supplies the effective detector mask, including native validity.
  const area = Math.max(1, mask.reduce((sum, value) => sum + (value ? 1 : 0), 0));
  let painted = false;
  try { painted = paintMean?.(area) ?? false; }
  finally { if (!painted) source.normalizeDisplayBuffers(previous, area); }
  return {
    buffers: previous,
    path: "delta" as const,
    addedPixels: delta.added,
    removedPixels: delta.removed,
  };
}

export type CompareCountImages = {
  images: ReadonlyMap<number, import("../colormaps").Uint32ImageView>;
  isCurrent: () => boolean;
  refreshFloat: () => void;
};
export type CompareGpuRenderer = (counts?: CompareCountImages | null) => number;

/** Retain a current borrowed image for repaints, or refresh its float fallback.
 * A replaced/disposed source is discarded without touching its retired buffers.
 */
export function countImagesForRender(counts: CompareCountImages | null, shared: boolean): CompareCountImages | null {
  if (!counts?.isCurrent()) return null;
  if (shared) return counts;
  counts.refreshFloat();
  return null;
}
