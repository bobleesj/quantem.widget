// CPU colormap (Float32 -> RGBA via 256-entry LUT): the float32 normalize and
// log the WGSL display shaders repeat, and the Canvas2D paint built on them.

/**
 * Map a display value to [0, 1] exactly as the WGSL display_normalize does, in
 * float32: NaN and -inf go to 0, +inf to 1, an empty window to 0.5, and a
 * window whose float32 span overflows is halved first (exact), so
 * (value - low) / (high - low) is computed the same way for every window.
 * Every step is a correctly rounded float32 operation on both paths (the WGSL
 * divides through display_divide, not the GPU's inexact `/`), which is what
 * lets the Canvas2D and WebGPU paths choose the same LUT entry for a value that
 * sits on a level edge.
 */
export function displayNormalize(value: number, low: number, high: number): number {
  const f32 = Math.fround;
  if (!Number.isFinite(value)) return value === Infinity ? 1 : 0;
  if (!(high > low)) return 0.5;
  const scale = Number.isFinite(f32(high - low)) ? 1 : 0.5;
  const span = f32(f32(high * scale) - f32(low * scale));
  return Math.min(1, Math.max(0, f32(f32(f32(value * scale) - f32(low * scale)) / span)));
}

/** Signed log1p in float32, the WGSL log_scale branch: log(1 + v), or -log(1 - v) below zero. */
export function displayLog(value: number): number {
  const f32 = Math.fround;
  return value >= 0 ? f32(Math.log(f32(1 + value))) : f32(-Math.log(f32(1 - value)));
}

/** Apply colormap LUT to float data, writing into an RGBA Uint8ClampedArray. */
export function applyColormap(
  data: Float32Array,
  rgba: Uint8ClampedArray,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
): void {
  const low = Math.fround(vmin);
  const high = Math.fround(vmax);
  for (let i = 0; i < data.length; i++) {
    const entry = Math.min(255, Math.trunc(Math.fround(displayNormalize(data[i], low, high) * 255))) * 3;
    const pixel = i * 4;
    rgba[pixel] = lut[entry];
    rgba[pixel + 1] = lut[entry + 1];
    rgba[pixel + 2] = lut[entry + 2];
    rgba[pixel + 3] = 255;
  }
}

/** Create an offscreen canvas with colormapped data. Returns null if context unavailable. */
export function renderToOffscreen(
  data: Float32Array,
  width: number,
  height: number,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
): HTMLCanvasElement | null {
  const offscreen = document.createElement("canvas");
  offscreen.width = width;
  offscreen.height = height;
  const ctx = offscreen.getContext("2d");
  if (!ctx) return null;
  const imgData = ctx.createImageData(width, height);
  applyColormap(data, imgData.data, lut, vmin, vmax);
  ctx.putImageData(imgData, 0, 0);
  return offscreen;
}

/** Render colormapped data to a reusable offscreen canvas + ImageData (avoids per-frame allocation). */
export function renderToOffscreenReuse(
  data: Float32Array,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
  offscreen: HTMLCanvasElement,
  imgData: ImageData,
): void {
  applyColormap(data, imgData.data, lut, vmin, vmax);
  offscreen.getContext("2d")!.putImageData(imgData, 0, 0);
}
