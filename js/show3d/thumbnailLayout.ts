// Layout of Show3D's thumbnail strip: cells in rows of 1, 2 or 3, the canvas window that is
// drawn (the visible part of the strip plus some overscan), and the cells inside it. Pure math,
// shared by the strip component and its tests.

export type StripGrid = {
  cols: number;
  /** Width of one column in CSS px (the strip's inner width shared evenly; may be fractional). */
  cellWidth: number;
  /** Height of one row in CSS px (whole pixels). */
  rowHeight: number;
  rows: number;
  /** Height of every row together: the scrolled content's height. */
  totalHeight: number;
};

export function stripColumns(columns: number): number {
  return Math.max(1, Math.min(3, Math.round(columns) || 2));
}

export function stripGrid(frames: number, columns: number, innerWidth: number): StripGrid {
  const cols = stripColumns(columns);
  const width = Math.max(cols, innerWidth);
  const cellWidth = width / cols;
  const rowHeight = Math.max(1, Math.floor(cellWidth));
  const rows = Math.ceil(Math.max(0, frames) / cols);
  return { cols, cellWidth, rowHeight, rows, totalHeight: rows * rowHeight };
}

/** The frame shown at strip position `position` (0 at the top left); newest first reverses the order. */
export function frameAtPosition(position: number, frames: number, newestFirst: boolean): number {
  return newestFirst ? frames - 1 - position : position;
}

/** The strip position of `frame` (the same reversal: it is its own inverse). */
export function positionOfFrame(frame: number, frames: number, newestFirst: boolean): number {
  return newestFirst ? frames - 1 - frame : frame;
}

export type Box = { x: number; y: number; width: number; height: number };

/** The cell at `position`, in CSS px from the top left of the scrolled content. */
export function cellBox(position: number, grid: StripGrid): Box {
  const row = Math.floor(position / grid.cols);
  const col = position % grid.cols;
  const x0 = col * grid.cellWidth;
  return { x: x0, y: row * grid.rowHeight, width: (col + 1) * grid.cellWidth - x0, height: grid.rowHeight };
}

/**
 * The image inside a cell: `aspect` (width / height) fitted and centred. An image within a pixel
 * of filling the cell fills it, so no seam of background shows between square thumbnails.
 */
export function imageBox(cell: Box, aspect: number): Box {
  const a = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  let width = cell.width;
  let height = width / a;
  if (height > cell.height) { height = cell.height; width = height * a; }
  if (cell.width - width < 1) width = cell.width;
  if (cell.height - height < 1) height = cell.height;
  return { x: cell.x + (cell.width - width) / 2, y: cell.y + (cell.height - height) / 2, width, height };
}

/** The drawn part of the strip: CSS px from the top of the scrolled content. */
export type StripWindow = { top: number; height: number };

/** Overscan above and below the viewport: one viewport, at least two rows. */
export function stripOverscan(viewport: number, rowHeight: number): number {
  return Math.max(Math.round(viewport), 2 * rowHeight, 1);
}

/**
 * The window to draw for a scroll position: the viewport plus `overscan` on either side, clamped
 * to the content, and no taller than `maxHeight` (the canvas's device-pixel limit over the
 * pixel ratio). The canvas sits at the window's top inside the scrolled content, so it scrolls with
 * it on the compositor and is drawn again only when the window moves.
 */
export function stripWindow(scrollTop: number, viewport: number, totalHeight: number, overscan: number, maxHeight = Infinity): StripWindow {
  const total = Math.max(0, totalHeight);
  const view = Math.max(0, viewport);
  const height = Math.min(total, Math.max(view, Math.min(view + 2 * overscan, maxHeight)));
  const top = Math.max(0, Math.min(total - height, Math.round(scrollTop - (height - view) / 2)));
  return { top, height: Math.round(height) };
}

/** True when the current window no longer covers the viewport well: draw a new one. */
export function stripWindowStale(current: StripWindow | null, scrollTop: number, viewport: number, totalHeight: number, overscan: number, maxHeight = Infinity): boolean {
  if (!current) return true;
  const next = stripWindow(scrollTop, viewport, totalHeight, overscan, maxHeight);
  if (next.height !== current.height) return true;
  // the viewport left the window, or the window drifted half an overscan from where it would be placed now
  if (scrollTop < current.top || scrollTop + viewport > current.top + current.height) return true;
  return Math.abs(next.top - current.top) >= Math.max(1, overscan / 2);
}

export type WindowCell = { frame: number; position: number; cell: Box };

/** The cells that overlap the window, in strip order. */
export function cellsInWindow(window: StripWindow, frames: number, grid: StripGrid, newestFirst: boolean): WindowCell[] {
  if (frames <= 0 || window.height <= 0) return [];
  const firstRow = Math.max(0, Math.floor(window.top / grid.rowHeight));
  const lastRow = Math.min(grid.rows - 1, Math.ceil((window.top + window.height) / grid.rowHeight) - 1);
  const cells: WindowCell[] = [];
  for (let row = firstRow; row <= lastRow; row++) {
    for (let col = 0; col < grid.cols; col++) {
      const position = row * grid.cols + col;
      if (position >= frames) break;
      cells.push({ frame: frameAtPosition(position, frames, newestFirst), position, cell: cellBox(position, grid) });
    }
  }
  return cells;
}

/** A box in device pixels of a canvas whose top is at `windowTop`: edges rounded, so neighbours share them. */
export function deviceBox(box: Box, windowTop: number, dpr: number): Box {
  const x0 = Math.round(box.x * dpr);
  const y0 = Math.round((box.y - windowTop) * dpr);
  const x1 = Math.round((box.x + box.width) * dpr);
  const y1 = Math.round((box.y + box.height - windowTop) * dpr);
  return { x: x0, y: y0, width: Math.max(0, x1 - x0), height: Math.max(0, y1 - y0) };
}

/** scrollTop that brings `position`'s row into view (centred), or null when it is already in view. */
export function scrollToShow(position: number, grid: StripGrid, scrollTop: number, viewport: number): number | null {
  const top = Math.floor(position / grid.cols) * grid.rowHeight;
  if (top >= scrollTop && top + grid.rowHeight <= scrollTop + viewport) return null;
  return Math.max(0, Math.min(Math.max(0, grid.totalHeight - viewport), Math.round(top - (viewport - grid.rowHeight) / 2)));
}
