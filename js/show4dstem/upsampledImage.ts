// An image k times the scan grid (an upsampled direct ptychography phase) drawn at full resolution in the
// image panel. Its canvas holds k x k pixels per scan position at the panel's size; the pointer overlay stays in
// scan positions, so a click still picks the scan position under it (image pixel / k).

export type UpsampledImage = { data: Float32Array; factor: number; rows: number; cols: number };

/** The mean over each k x k block: the image on the scan grid, for tools that read scan positions. */
export function binImage(data: Float32Array, rows: number, cols: number, factor: number): Float32Array {
  const binned = new Float32Array(rows * cols);
  const width = cols * factor;
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      let sum = 0;
      for (let r = 0; r < factor; r++) {
        const start = (row * factor + r) * width + col * factor;
        for (let c = 0; c < factor; c++) sum += data[start + c];
      }
      binned[row * cols + col] = sum / (factor * factor);
    }
  }
  return binned;
}

/** The scan position under one pixel of the k x image. */
export function scanPosition(imageRow: number, imageCol: number, factor: number): [number, number] {
  return [Math.floor(imageRow / factor), Math.floor(imageCol / factor)];
}

/** The image canvas transform that keeps a k x image under the scan-unit overlay's pan and zoom. */
export function upsampledTransform(panX: number, panY: number, zoom: number, factor: number) {
  return { translateX: panX * factor, translateY: panY * factor, scale: zoom };
}

type Preview = {
  rows?: number; cols?: number; total?: number; mode?: string; upsample?: number;
  phase?: DataView; roi?: DataView;
};

/**
 * The live row preview on the scan grid (NaN below the rows measured so far) and, for a Direct phase sent k x
 * the scan grid (``upsample``), the same rows at full resolution. Null when the preview is incomplete.
 */
export function livePreviewImages(preview: Preview | undefined, direct: boolean):
    { image: DataView; upsampled: UpsampledImage | null } | null {
  if (!preview?.rows || !preview.cols || !preview.total || preview.rows > preview.total) return null;
  const { rows, cols, total } = preview;
  const factor = direct && preview.mode === "Direct" ? Math.max(1, Math.round(preview.upsample ?? 1)) : 1;
  const data = direct && preview.mode === "Direct" ? preview.phase : preview.roi;
  if (data?.byteLength !== rows * factor * cols * factor * 4) return null;
  const measured = new Float32Array(rows * factor * cols * factor);
  new Uint8Array(measured.buffer).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  const pixels = new Float32Array(total * cols).fill(NaN);
  pixels.set(factor > 1 ? binImage(measured, rows, cols, factor) : measured);
  if (factor === 1) return { image: new DataView(pixels.buffer), upsampled: null };
  const full = new Float32Array(total * factor * cols * factor).fill(NaN);
  full.set(measured);
  return { image: new DataView(pixels.buffer),
           upsampled: { data: full, factor, rows: total * factor, cols: cols * factor } };
}
