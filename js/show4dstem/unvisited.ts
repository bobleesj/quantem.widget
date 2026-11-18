import type { UpsampledImage } from "./upsampledImage";

/** The positions a scan's path never visited, as Python sends them (`unvisited_bytes`: one byte per scan position,
 *  1 = never visited), or null when every position was measured. A scan taken along a sparse path (a star, random
 *  points) is kept on its grid with something at every position; the viewer draws the unvisited ones blank. */
export function unvisitedMask(bytes: DataView | null | undefined, rows: number, cols: number): Uint8Array | null {
  if (!bytes || !(rows * cols > 0) || bytes.byteLength !== rows * cols) return null;
  return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

/** `image` (float32, rows x cols) with the unvisited positions NaN: a new image, or `image` itself without a mask. */
export function blankUnvisited(image: DataView | null, mask: Uint8Array | null, rows: number, cols: number): DataView | null {
  const pixels = rows * cols;
  if (!image || !mask || mask.length !== pixels || image.byteLength !== pixels * 4) return image;
  const values = new Float32Array(image.buffer.slice(image.byteOffset, image.byteOffset + pixels * 4));
  for (let i = 0; i < pixels; i++) if (mask[i]) values[i] = NaN;
  return new DataView(values.buffer);
}

/** An image `factor` times the scan grid (an upsampled Direct phase) with the pixels of unvisited positions NaN. */
export function blankUnvisitedDetail(image: UpsampledImage | null, mask: Uint8Array | null,
                                     rows: number, cols: number): UpsampledImage | null {
  if (!image || !mask || mask.length !== rows * cols || image.rows !== rows * image.factor
      || image.cols !== cols * image.factor) return image;
  const data = new Float32Array(image.data);
  for (let row = 0; row < image.rows; row++) {
    const scanRow = Math.floor(row / image.factor) * cols;
    for (let col = 0; col < image.cols; col++) {
      if (mask[scanRow + Math.floor(col / image.factor)]) data[row * image.cols + col] = NaN;
    }
  }
  return { ...image, data };
}
