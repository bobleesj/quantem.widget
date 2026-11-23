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

/** Separate the changing frame number from the Live heading so its width can be held steady. */
export function splitLiveFrameTitle(title: string): { frame: string; detail: string } | null {
  const match = /^Live · frame (\d+)( · .*|)$/.exec(title);
  return match ? { frame: match[1], detail: match[2] } : null;
}

/** A stopped transient frame cannot label a retained comparison. */
export function retainedComparisonTitle(title: string, showingRetainedComparison: boolean): string {
  return showingRetainedComparison && splitLiveFrameTitle(title) ? "Retained scans" : title;
}

/** The region map shows its retained full field until a regional image arrives. */
export function virtualImageGridLabel(
  rows: number, cols: number, regionMapSize: number | null, showingRegion: boolean,
): string {
  if (regionMapSize !== null && !showingRegion) return `${regionMapSize}×${regionMapSize}`;
  return `${rows}×${cols}`;
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

export type PreviewSource = {
  scan?: string;
  received_utc?: string;
  file?: string;
  folder?: string;
};

/** Match the scan segment of the Python or browser Live title, including a retained scan's grid. */
function liveTitleNamesScan(title: string, scan: string | undefined): boolean {
  if (!scan) return false;
  const name = title.split(" · ", 2)[1];
  if (isLiveFrame(scan)) return name === `frame ${scan.slice("live_".length)}`;
  return name === shortScanName(scan) || name === shortScanName(scan.replace(/_grid$/, ""));
}

function incomingSaveStatus(title: string): string {
  const progress = title.split(" · ", 3)[2];
  const counts = /^(\d[\d,]*)\/(\d[\d,]*)$/.exec(progress ?? "");
  return counts && counts[1] !== counts[2] ? "not saved (live)" : "";
}

/** The file follows the scan named in the heading, even before the first preview rows are drawn. */
export function sourcePathForView(
  preview: { scan: string; sourceReceivedUtc?: string } | null,
  previewSource: PreviewSource | undefined,
  retainedFolder: string,
  retainedFile: string,
  retainedMatches = true,
  heading?: { title: string; retainedScan?: string; showingPreview: boolean },
): string {
  const liveTitle = heading?.title.startsWith("Live · ") ? heading.title : "";
  if (preview && (!heading || heading.showingPreview || (liveTitle && liveTitleNamesScan(liveTitle, preview.scan)))) {
    if (!preview.sourceReceivedUtc || preview.scan !== previewSource?.scan
        || preview.sourceReceivedUtc !== previewSource.received_utc) return incomingSaveStatus(liveTitle);
    retainedFolder = previewSource.folder ?? "";
    retainedFile = previewSource.file ?? "";
  } else if (!retainedMatches || (liveTitle && !liveTitleNamesScan(liveTitle, heading?.retainedScan))) {
    return incomingSaveStatus(liveTitle);
  }
  if (!retainedFile) return "";
  if (retainedFile.startsWith("not saved")) return retainedFile;
  return retainedFolder ? `${retainedFolder}/${retainedFile}` : retainedFile;
}

/** The fast diffraction trait has no scan ID. Only a point frame carrying the displayed
 * pattern in its own payload can identify that pattern's specimen position. */
export function diffractionPositionLabel(
  frame: {point?: boolean; region?: number[]; dp?: DataView} | null | undefined,
  displayed: DataView | null,
  row: number, col: number,
): string {
  if (!frame?.point) return `DP at (${Math.round(row)}, ${Math.round(col)})`;
  const area = frame.region;
  if (displayed && displayed === frame.dp && area?.length === 4 &&
      area.every(n => Number.isInteger(n) && n >= 0) && area[2] === area[0]+1 && area[3] === area[1]+1)
    return `Point diffraction at (${area[0]}, ${area[1]})`;
  return "Point diffraction";
}
