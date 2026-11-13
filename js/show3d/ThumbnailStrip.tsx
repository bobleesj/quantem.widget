import * as React from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { DriftChart, type DriftReadout } from "./DriftChart";
import {
  cellsInWindow, deviceBox, imageBox, positionOfFrame, scrollToShow, stripGrid, stripOverscan, stripWindow,
  stripWindowStale, type StripWindow,
} from "./thumbnailLayout";

/** CSS width of the strip. The thumbnails are square and share it: 1, 2 or 3 across. */
export const THUMBNAIL_STRIP_WIDTH = 264;
const INNER_WIDTH = THUMBNAIL_STRIP_WIDTH - 2; // inside the 1 px strip border
const HEADER_HEIGHT = 30;
// The strip canvas is at most this many device pixels tall (WebGPU's default texture size limit).
const MAX_CANVAS_DEVICE_PX = 8192;

export type ThumbnailStripColors = { text: string; textMuted: string; border: string; accent: string; controlBg: string };

/** One thumbnail image to draw, in device pixels of the strip canvas. */
export type ThumbnailCellRect = { frame: number; x: number; y: number; width: number; height: number };

export type ThumbnailStripProps = {
  frames: number;
  currentFrame: number;
  labels?: string[];
  columns: number;
  onColumnsChange: (columns: number) => void;
  onHide: () => void;
  /** Show `frame` in the viewer while the pointer rests on it; null restores the selected frame. */
  onHover: (frame: number | null) => void;
  /** Move the frame slider to `frame`. */
  onSelect: (frame: number) => void;
  /** The viewer's height in CSS px: the strip is exactly as tall and scrolls inside. */
  height: number;
  /** Width / height of one thumbnail image (one panel); a non-square image is centred in its square cell. */
  imageAspect: number;
  /**
   * Draw `cells` into the strip canvas (already sized in device pixels) in one pass; cells not listed are
   * black. False when nothing could be drawn yet (no GPU engine).
   */
  paint: (canvas: HTMLCanvasElement, cells: ThumbnailCellRect[]) => boolean;
  /** Free the strip canvas's GPU textures (the strip closed). */
  release: (canvas: HTMLCanvasElement) => void;
  /** Filled with the strip's redraw, for the viewer to call when only the display transform changed. */
  redrawRef?: React.MutableRefObject<(() => void) | null>;
  /** Changes when more frames become drawable (resident on the GPU): the strip draws again. */
  retryKey: number | string;
  /** Why no thumbnail can be drawn (no WebGPU, a color stack); null when they can. */
  unavailableReason: string | null;
  colors: ThumbnailStripColors;
  /** Frames picked for Find drift (stack indices); ⌘/Ctrl-click toggles one, Shift-click picks a range. */
  chosen?: number[];
  onChoose?: (frames: number[]) => void;
  /** Runs Find drift on the picked frames (every frame when fewer than two are picked). */
  onFindDrift?: () => void;
  /** The button's words while it runs, or the last result's status. */
  driftStatus?: string;
  driftBusy?: boolean;
  /** The last Find drift result, drawn as a chart under the thumbnails. */
  drift?: DriftReadout | null;
  /** The newest frame first (at the top), as a live stream needs; frame numbers and picking are unchanged. */
  newestFirst?: boolean;
};

/**
 * A scrollable strip of square frame thumbnails beside the viewer, drawn on ONE WebGPU canvas. The
 * scrolled content keeps the strip's full height (for the scrollbar); the canvas covers only the
 * visible part plus an overscan, scrolls with the content, and is drawn again when the view moves
 * past the overscan or the display changes. Cells inside the canvas window carry the DOM for hover,
 * clicks, labels and outlines. Hover previews a frame, click selects it; the header sets 1, 2 or 3 across.
 */
export function ThumbnailStrip(props: ThumbnailStripProps) {
  const { frames, currentFrame, labels, columns, height, imageAspect, retryKey, colors } = props;
  const newestFirst = !!props.newestFirst;
  const chosen = React.useMemo(() => new Set(props.chosen ?? []), [props.chosen]);
  const lastChosen = React.useRef<number | null>(null);
  /** ⌘/Ctrl-click toggles a frame for Find drift; Shift-click adds the range from the last one; a plain click selects. */
  const handleClick = (event: React.MouseEvent, frame: number) => {
    if (props.onChoose && (event.metaKey || event.ctrlKey)) {
      const next = new Set(chosen);
      if (next.has(frame)) next.delete(frame); else next.add(frame);
      lastChosen.current = frame;
      props.onChoose([...next].sort((a, b) => a - b));
      return;
    }
    if (props.onChoose && event.shiftKey) {
      const from = lastChosen.current ?? currentFrame;
      const next = new Set(chosen);
      for (let f = Math.min(from, frame); f <= Math.max(from, frame); f++) next.add(f);
      lastChosen.current = frame;
      props.onChoose([...next].sort((a, b) => a - b));
      return;
    }
    props.onSelect(frame);
  };
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const pointerInside = React.useRef(false);
  // While the strip scrolls, thumbnails pass under the pointer: previewing each would redraw the big image dozens of
  // times a second. Previews wait until the scrolling stops and the pointer moves over a thumbnail.
  const scrollingUntil = React.useRef(0);
  const hovered = React.useRef<number | null>(null);
  const scrolling = () => performance.now() < scrollingUntil.current;
  const preview = (frame: number) => {
    if (scrolling() || hovered.current === frame) return;
    hovered.current = frame;
    props.onHover(frame);
  };

  const totalHeight = Math.max(160, Math.round(height));
  // The scroll area's own size: the scrollbar takes width, the header and drift chart take height.
  const [view, setView] = React.useState({ width: INNER_WIDTH, height: Math.max(1, totalHeight - HEADER_HEIGHT) });
  React.useLayoutEffect(() => {
    const root = scrollRef.current;
    if (!root) return;
    const measure = () => {
      const width = root.clientWidth || INNER_WIDTH;
      const viewHeight = root.clientHeight || Math.max(1, totalHeight - HEADER_HEIGHT);
      setView(prev => (prev.width === width && prev.height === viewHeight ? prev : { width, height: viewHeight }));
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(root);
    return () => observer.disconnect();
  }, [props.unavailableReason, totalHeight]);

  const grid = stripGrid(frames, columns, view.width);
  const cols = grid.cols;
  const aspect = Number.isFinite(imageAspect) && imageAspect > 0 ? imageAspect : 1;
  const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
  const overscan = stripOverscan(view.height, grid.rowHeight);
  const maxWindow = Math.floor(MAX_CANVAS_DEVICE_PX / dpr);

  // The drawn window: a ref for the scroll handler (no React render per scroll event), state for the cell DOM.
  const windowRef = React.useRef<StripWindow | null>(null);
  const [win, setWin] = React.useState<StripWindow>({ top: 0, height: 0 });

  // Everything redraw reads, current at every call (the viewer calls it from its own paint loop).
  const latest = React.useRef({ grid, frames, newestFirst, aspect, dpr, view, overscan, maxWindow, paint: props.paint, release: props.release, unavailable: props.unavailableReason });
  latest.current = { grid, frames, newestFirst, aspect, dpr, view, overscan, maxWindow, paint: props.paint, release: props.release, unavailable: props.unavailableReason };

  /** Draw the cells of the current window into the canvas. */
  const redraw = React.useCallback(() => {
    const canvas = canvasRef.current;
    const w = windowRef.current;
    const s = latest.current;
    if (!canvas || !w || s.unavailable || w.height <= 0) return;
    const deviceWidth = Math.max(1, Math.round(s.view.width * s.dpr));
    const deviceHeight = Math.max(1, Math.round(w.height * s.dpr));
    if (canvas.width !== deviceWidth) canvas.width = deviceWidth;
    if (canvas.height !== deviceHeight) canvas.height = deviceHeight;
    const cells = cellsInWindow(w, s.frames, s.grid, s.newestFirst).map(({ frame, cell }) => ({
      frame, ...deviceBox(imageBox(cell, s.aspect), w.top, s.dpr),
    }));
    s.paint(canvas, cells);
  }, []);

  /** Move the window when the view left it (or is about to); true when it moved. */
  const placeWindow = React.useCallback((force: boolean) => {
    const root = scrollRef.current;
    const s = latest.current;
    if (!root) return false;
    const scrollTop = root.scrollTop;
    if (!force && !stripWindowStale(windowRef.current, scrollTop, s.view.height, s.grid.totalHeight, s.overscan, s.maxWindow)) return false;
    const next = stripWindow(scrollTop, s.view.height, s.grid.totalHeight, s.overscan, s.maxWindow);
    windowRef.current = next;
    // position the canvas now, in the same task as its new pixels, not on React's next render
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.style.top = `${next.top}px`;
      canvas.style.height = `${next.height}px`;
    }
    setWin(prev => (prev.top === next.top && prev.height === next.height ? prev : next));
    return true;
  }, []);

  // A new layout, more frames, more resident frames: place the window again and draw it.
  React.useLayoutEffect(() => {
    if (props.unavailableReason) return;
    placeWindow(true);
    redraw();
  }, [frames, cols, grid.rowHeight, view.width, view.height, newestFirst, aspect, dpr, retryKey, props.unavailableReason, placeWindow, redraw]);

  // The viewer redraws the strip with its own image (a histogram drag, a new colormap or scale).
  React.useEffect(() => {
    const slot = props.redrawRef;
    if (!slot) return;
    slot.current = redraw;
    return () => { if (slot.current === redraw) slot.current = null; };
  }, [props.redrawRef, redraw]);

  // The strip closed: give the canvas's textures back.
  React.useEffect(() => {
    const canvas = canvasRef.current;
    return () => { if (canvas) latest.current.release(canvas); };
  }, [props.unavailableReason]);

  const handleScroll = () => {
    scrollingUntil.current = performance.now() + 150;
    if (placeWindow(false)) redraw();
  };

  // Keep the selected frame in view (playback, slider) unless the pointer is browsing the strip.
  React.useEffect(() => {
    const root = scrollRef.current;
    if (pointerInside.current || !root) return;
    const target = scrollToShow(positionOfFrame(currentFrame, frames, newestFirst), grid, root.scrollTop, view.height);
    if (target === null) return;
    root.scrollTop = target;
    if (placeWindow(false)) redraw();
  }, [currentFrame, cols]);

  const segment = (n: number) => ({
    width: 24, height: 22, display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: 11, cursor: "pointer", userSelect: "none" as const,
    color: cols === n ? "#fff" : colors.text, bgcolor: cols === n ? colors.accent : "transparent",
    borderLeft: n > 1 ? `1px solid ${colors.border}` : "none",
  });

  return (
    <Box
      data-testid="show3d-thumbnail-strip"
      sx={{ width: THUMBNAIL_STRIP_WIDTH, height: totalHeight, flex: "0 0 auto", display: "flex", flexDirection: "column", border: `1px solid ${colors.border}`, boxSizing: "border-box" }}
      onPointerEnter={() => { pointerInside.current = true; }}
      onPointerLeave={() => { pointerInside.current = false; hovered.current = null; props.onHover(null); }}
    >
      <Box sx={{ height: HEADER_HEIGHT, flex: "0 0 auto", display: "flex", alignItems: "center", gap: 0.75, px: "6px", borderBottom: `1px solid ${colors.border}` }}>
        <Typography sx={{ fontSize: 11, fontWeight: 600, color: colors.text }}>Frames</Typography>
        <Typography sx={{ fontSize: 11, color: colors.textMuted, flex: 1 }}>{chosen.size ? `${chosen.size} of ${frames}` : frames}</Typography>
        {props.onFindDrift && (
          <Box role="button" aria-label="Find drift" data-testid="show3d-find-drift"
            title={"Find drift: register the picked frames (⌘/Ctrl-click, Shift-click; all when none) on the GPU, " +
              "to 1/100 px, and fit the drift velocity"}
            onClick={props.driftBusy ? undefined : props.onFindDrift}
            sx={{ px: "6px", height: 20, display: "flex", alignItems: "center", fontSize: 11, fontWeight: 600, whiteSpace: "nowrap",
              border: `1px solid ${colors.accent}`, color: props.driftBusy ? colors.textMuted : colors.accent,
              cursor: props.driftBusy ? "progress" : "pointer", "&:hover": { bgcolor: props.driftBusy ? "transparent" : colors.controlBg } }}>
            {props.driftBusy ? "Finding…" : "Find drift"}
          </Box>
        )}
        <Box role="group" aria-label="Thumbnails across" sx={{ display: "flex", border: `1px solid ${colors.border}` }}>
          {[1, 2, 3].map(n => (
            <Box key={n} role="button" aria-label={`${n} thumbnail column${n > 1 ? "s" : ""}`} aria-pressed={cols === n}
              title={`${n} across`} sx={segment(n)} onClick={() => props.onColumnsChange(n)}>{n}</Box>
          ))}
        </Box>
        <Box role="button" aria-label="Hide thumbnails" title="Hide thumbnails" onClick={props.onHide}
          sx={{ width: 22, height: 22, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, color: colors.textMuted, cursor: "pointer", "&:hover": { color: colors.text } }}>×</Box>
      </Box>
      {props.unavailableReason ? (
        <Typography sx={{ fontSize: 11, color: colors.textMuted, p: 1.25, lineHeight: 1.4 }}>{props.unavailableReason}</Typography>
      ) : (
        <Box ref={scrollRef} onScroll={handleScroll}
          sx={{ flex: "1 1 auto", minHeight: 0, overflowY: "auto", overscrollBehavior: "contain", scrollbarWidth: "thin", position: "relative" }}>
          {/* The full content height keeps the scrollbar honest; only the window inside it is drawn. */}
          <Box sx={{ position: "relative", width: "100%", height: grid.totalHeight, bgcolor: "#000" }}>
            <canvas
              ref={canvasRef}
              data-testid="show3d-thumbnail-canvas"
              style={{ position: "absolute", left: 0, top: win.top, width: "100%", height: win.height, display: "block", pointerEvents: "none" }}
            />
            {cellsInWindow(win, frames, grid, newestFirst).map(({ frame, cell }) => {
              const selected = frame === currentFrame;
              // frames count from 1 as the slider does; a label that is only the 0-based index adds nothing
              const custom = labels?.[frame] && labels[frame] !== String(frame) ? labels[frame] : "";
              return (
                <Box key={frame} data-frame={frame} aria-label={`Frame ${frame + 1} thumbnail`}
                  title={custom ? `Frame ${frame + 1}: ${custom}` : `Frame ${frame + 1}`}
                  onPointerEnter={() => preview(frame)}
                  onPointerMove={() => preview(frame)}
                  onClick={(event) => handleClick(event, frame)}
                  sx={{
                    position: "absolute", left: cell.x, top: cell.y, width: cell.width, height: cell.height, cursor: "pointer",
                    outline: selected ? `2px solid ${colors.accent}` : "none", outlineOffset: -2,
                    "&:hover": selected ? {} : { outline: "1px solid rgba(255,255,255,0.7)", outlineOffset: -1 },
                  }}>
                  <Box sx={{
                    position: "absolute", left: 3, bottom: 3, px: 0.5, maxWidth: cell.width - 6,
                    fontSize: 10, lineHeight: "14px", color: "#fff", bgcolor: selected ? colors.accent : "rgba(0,0,0,0.55)",
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", pointerEvents: "none",
                  }}>
                    {cols < 3 && custom ? `${frame + 1} · ${custom}` : frame + 1}
                  </Box>
                  {chosen.has(frame) && (
                    <Box aria-label={`Frame ${frame + 1} picked for Find drift`} sx={{
                      position: "absolute", right: 3, top: 3, width: 16, height: 16, borderRadius: "50%", fontSize: 11,
                      lineHeight: "16px", textAlign: "center", color: "#fff", bgcolor: colors.accent, pointerEvents: "none",
                    }}>✓</Box>
                  )}
                </Box>
              );
            })}
          </Box>
        </Box>
      )}
      {(props.drift || props.driftStatus) && (
        <DriftChart drift={props.drift ?? null} status={props.driftStatus ?? ""} width={INNER_WIDTH} colors={colors} />
      )}
    </Box>
  );
}
