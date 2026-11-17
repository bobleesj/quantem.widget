/**
 * Nearest-neighbour copy of a source region onto a 2D canvas with the WebGPU path's texel choice:
 * output pixel x shows source column floor((2x + 1) sw / 2 dw), rows alike (WGSL nearest_texel).
 * The browser's drawImage breaks ties differently, so at a zoom where a pixel centre lands exactly
 * on a texel edge (256 px drawn at 450) the CPU path showed the neighbouring texel along whole rows
 * and columns.
 *
 * Applies only to an unscaled, unrotated, whole-pixel placement: the context's transform must be a
 * translation by whole pixels and the destination rectangle whole pixels. Returns false otherwise,
 * and the caller draws with drawImage.
 */
export function drawNearestExact(
  ctx: CanvasRenderingContext2D,
  source: HTMLCanvasElement | OffscreenCanvas,
  sx: number, sy: number, sw: number, sh: number,
  dx: number, dy: number, dw: number, dh: number,
): boolean {
  const transform = typeof ctx.getTransform === "function" ? ctx.getTransform() : null;
  if (!transform) return false;
  const ox = transform.e + dx;
  const oy = transform.f + dy;
  const whole = [sx, sy, sw, sh, ox, oy, dw, dh].every(Number.isInteger);
  if (transform.a !== 1 || transform.d !== 1 || transform.b !== 0 || transform.c !== 0 || !whole || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) {
    return false;
  }
  const sourceCtx = source.getContext("2d") as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
  if (!sourceCtx) return false;
  const pixels = new Uint32Array(sourceCtx.getImageData(sx, sy, sw, sh).data.buffer);
  const out = ctx.createImageData(dw, dh);
  const target = new Uint32Array(out.data.buffer);
  const columns = new Int32Array(dw);
  for (let x = 0; x < dw; x++) columns[x] = Math.min(Math.floor(((2 * x + 1) * sw) / (2 * dw)), sw - 1);
  for (let y = 0; y < dh; y++) {
    const row = Math.min(Math.floor(((2 * y + 1) * sh) / (2 * dh)), sh - 1) * sw;
    const base = y * dw;
    for (let x = 0; x < dw; x++) target[base + x] = pixels[row + columns[x]];
  }
  ctx.putImageData(out, ox, oy);
  return true;
}
