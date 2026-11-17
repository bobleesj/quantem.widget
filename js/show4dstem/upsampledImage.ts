// An image k times the scan grid (an upsampled direct ptychography phase) drawn at full resolution in the
// image panel. Its canvas holds k x k pixels per scan position at the panel's size; the pointer overlay stays in
// scan positions, so a click still picks the scan position under it (image pixel / k).

export type UpsampledImage = { data: Float32Array; factor: number; rows: number; cols: number };

/** The scan position under one pixel of the k x image. */
export function scanPosition(imageRow: number, imageCol: number, factor: number): [number, number] {
  return [Math.floor(imageRow / factor), Math.floor(imageCol / factor)];
}

/** The image canvas transform that keeps a k x image under the scan-unit overlay's pan and zoom. */
export function upsampledTransform(panX: number, panY: number, zoom: number, factor: number) {
  return { translateX: panX * factor, translateY: panY * factor, scale: zoom };
}
