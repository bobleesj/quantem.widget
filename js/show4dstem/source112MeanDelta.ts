import type { Source112ResidentSet } from "../.generated/engine/detector/compute/webgpu/source112";

/** Refresh borrowed mean displays once from authoritative source112 counts. */
export function source112MeanDelta(
  source: Pick<Source112ResidentSet, "integrate" | "normalizeDisplayBuffers">,
  mask: Uint32Array,
  previous: GPUBuffer[],
) {
  const delta = source.integrate(mask);
  // The caller supplies the effective detector mask, including native validity.
  const area = Math.max(1, mask.reduce((sum, value) => sum + (value ? 1 : 0), 0));
  source.normalizeDisplayBuffers(previous, area);
  return {
    buffers: previous,
    path: "delta" as const,
    addedPixels: delta.added,
    removedPixels: delta.removed,
  };
}
