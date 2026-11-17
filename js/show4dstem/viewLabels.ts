/** A scan's name as the viewer prints it: its number ("41" for "series_41"); any other name unchanged. */
export function shortScanName(name: string): string {
  return name.replace(/^series_(\d+)/, "$1");
}

/** The scale bar and its label hold the image's bottom-right corner (CSS px). */
export const SCALE_BAR_CORNER = { width: 150, height: 52 };

/** Where the scan position's "(row, col)" label is drawn (left edge and text baseline, CSS px): to the
 *  right of the crosshair and just above it, kept inside the image and out of the scale bar's corner. */
export function positionLabelAnchor(
  crosshairX: number, crosshairY: number, textWidth: number, width: number, height: number, offset = 16,
): { x: number; y: number } {
  const x = Math.min(width - textWidth - 4, crosshairX + offset);
  let y = Math.max(13, crosshairY - 4);
  const barLeft = width - SCALE_BAR_CORNER.width;
  const barTop = height - SCALE_BAR_CORNER.height;
  if (x + textWidth > barLeft && y > barTop) y = Math.max(13, barTop);
  return { x, y };
}

/** True when the title already names this scan as one of its words ("Live · series_13 · …" names "series_13", not "series_1"). */
export function titleNamesScan(title: string | undefined, name: string | undefined): boolean {
  return Boolean(title && name) && title!.split(/[^A-Za-z0-9_]+/).includes(name!);
}
