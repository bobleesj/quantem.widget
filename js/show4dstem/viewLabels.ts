/** A MAPED tilt's scan name: series time, place in the series, tilt in mrad and the 0/90 pair's rotation. */
const MAPED_TILT = /^maped_\d{8}-\d{6}_(\d+)of(\d+)_x-?[\d.]+_y-?[\d.]+mrad(?:_rot(\d+(?:\.\d+)?))?$/;

/** A scan's name as the viewer prints it: its number ("41" for "series_41"), a MAPED tilt's place ("tilt 7/7");
 *  any other name unchanged. */
export function shortScanName(name: string): string {
  const tilt = MAPED_TILT.exec(name);
  if (tilt) return `tilt ${tilt[1]}/${tilt[2]}` + (tilt[3] ? ` rot${tilt[3]}` : "");
  return name.replace(/^series_(\d+)/, "$1");
}

/** True for a Live frame ("live_<n>"): one at a time, each replacing the last, never kept. */
export function isLiveFrame(name: string | undefined): boolean {
  return /^live_\d+$/.test(name ?? "");
}

/** The scans a viewer keeps: its scans less the Live frame, which is never kept. */
export function keptScanCount(names: string[] | undefined, count: number): number {
  return Math.max(0, count - (names ?? []).filter(isLiveFrame).length);
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

/** True when the title already names this scan, as whole words ("Live · series_13 · …" names "series_13", not "series_1";
 *  a MAPED tilt's name keeps its dots and dashes: "maped_20261118-222131_7of7_x0.0_y0.0mrad"). */
export function titleNamesScan(title: string | undefined, name: string | undefined): boolean {
  return Boolean(title && name) && ` ${title!.replace(/\s+/g, " ")} `.includes(` ${name!} `);
}
