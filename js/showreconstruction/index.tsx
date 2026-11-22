import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import Slider from "../shared/Slider";
import Switch from "@mui/material/Switch";
import Menu from "@mui/material/Menu";
import SkipPreviousIcon from "@mui/icons-material/SkipPrevious";
import SkipNextIcon from "@mui/icons-material/SkipNext";
import NavigateBeforeIcon from "@mui/icons-material/NavigateBefore";
import NavigateNextIcon from "@mui/icons-material/NavigateNext";
import { useTheme, WIDGET_UI_FONT, WIDGET_TYPOGRAPHY } from "../theme";
import { scaleBarGeometry } from "../figure";
import { compactButton, controlRow, sliderStyles, switchStyles, SPACING } from "../controlStyles";
import { COLORMAP_NAMES, COLORMAPS } from "../display/luts";
import { GPUColormapEngine } from "../display/colormaps";
import { createColormapEngine, createCPUColormapEngine, type DisplayColormapEngine } from "../display/cpuColormap";
import { onGPULost } from "../display/device";
import { PlayPauseButton } from "../PlayPauseButton";
import { InfoTooltip } from "../shared/InfoTooltip";
import { RenderPathBadge } from "../shared/RenderPathBadge";
import { applyHannWindow2D, fftshift, getDisplayFFT, zeroPadTopLeft, type DisplayFFT } from "../display/fft";
import { bySize, ComplexCanvas, SizeControl, type ViewSize } from "../shared/viewSize";
import { contrastRange, ControlsToggle, CornerInset, DEFAULT_DISPLAY, NO_ZOOM, readAt, useViewLink, ViewControls, ViewLink, ZoomFrame, type Display } from "../shared/zoomPan";
import { appendCheckpoint, EMPTY, floatsOf, indexAt, mergeStacks, stackFromTrait, type Stack } from "./stacks";
import { disagreement } from "./tiles";

// ShowReconstruction: an iterative reconstruction's whole history in one view. The story (training and held-out
// against total epoch, each run a band) drives every panel through one epoch stepper; the slice stepper drives the
// object and the depth marker. Everything is drawn here from the synced arrays, with no kernel round trip; images go
// through the package's colormap engine (WebGPU, or its Canvas2D twin), as in Show2D.

type Record = { [name: string]: any; epoch: number };
type Change = { key: string; before: string; after: string };
type Stage = { name: string; start: number; changes: string; changed?: Change[]; refused?: string };
type Series = { name: string; color: string; points: [number, number][]; dashed?: boolean };

// Three muted hues for every series; a second series of the same quantity is dashed in the same hue, not a new color.
// The theme accent is kept for what is selected (the committed epoch, the line, the active control).
const HUES = ["#6f8fb3", "#8c8c8c", "#b09a72"];
const BAR_COLOR = "#8c8c8c";

function floats(view: DataView | null | undefined): Float32Array {
  if (!view || view.byteLength === 0) return new Float32Array(0);
  return new Float32Array(view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength));
}

// Plain numbers, never scientific notation: a value smaller than the digits asked for gets the digits it needs.
function plain(value: number | null | undefined, digits = 4): string {
  if (value == null || !Number.isFinite(value)) return "–";
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && magnitude < 10 ** -digits) digits = Math.min(10, Math.ceil(-Math.log10(magnitude)) + 1);
  return value.toFixed(digits);
}

// Enough decimals that two different values never read alike (at least digits, at most 9): "kept" never sits beside two
// equal-looking held-out scores.
function pairDigits(a: number | null | undefined, b: number | null | undefined, digits = 5): number {
  if (a == null || b == null || a === b || !Number.isFinite(a) || !Number.isFinite(b)) return digits;
  while (digits < 9 && a.toFixed(digits) === b.toFixed(digits)) digits++;
  return digits;
}

// A change as an arrow and its size: ▼ down, ▲ up.
function change(delta: number | null | undefined, digits = 5): string {
  if (delta == null || !Number.isFinite(delta)) return "";
  return `${delta <= 0 ? "▼" : "▲"}${plain(Math.abs(delta), digits)}`;
}

// Round ticks inside [low, high] at a 1, 2 or 5 step, about count of them, with the decimals the step needs.
function niceTicks(low: number, high: number, count = 3): { value: number; label: string }[] {
  const span = high - low;
  if (!(span > 0) || !Number.isFinite(span)) return [{ value: low, label: plain(low, 3) }];
  const raw = span / count, magnitude = 10 ** Math.floor(Math.log10(raw)), norm = raw / magnitude;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * magnitude;
  const decimals = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  const ticks: { value: number; label: string }[] = [];
  for (let value = Math.ceil(low / step - 1e-9) * step; value <= high + step * 1e-9; value += step) ticks.push({ value, label: value.toFixed(decimals) });
  return ticks;
}

function seconds(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "–";
  return value >= 120 ? `${Math.round(value / 60)} min` : `${Math.round(value)} s`;
}

// ---------------------------------------------------------------------------------------------------------------
// Charts and images

// Charts are drawn at the width they occupy, so their text keeps one size from a phone to a wide notebook instead of
// shrinking or growing with a fixed viewBox.
function useWidth(fallback: number): [React.RefCallback<HTMLDivElement>, number] {
  const [width, setWidth] = React.useState(fallback);
  const observer = React.useRef<ResizeObserver | null>(null);
  // A callback ref: the element can be replaced (a view first draws a "reading the runs" line, then its real root), and
  // an observer set once on the first element would keep the view at its first width when the window narrows or turns.
  const ref = React.useCallback((element: HTMLDivElement | null) => {
    observer.current?.disconnect();
    observer.current = null;
    if (!element) return;
    observer.current = new ResizeObserver(([entry]) => { const next = Math.round(entry.contentRect.width); if (next > 0) setWidth(next); });
    observer.current.observe(element);
  }, []);
  return [ref, width];
}

const TEXT = 11;
const EMPTY_FRAME = new Float32Array(0);
const GAP = SPACING.XS;  // the one spacing step between panels, rows and controls
const HAIR = 2;  // between the images of one group (before, after, change; a probe's tiles; the two cuts): they read as one, groups sit GAP apart
const ON_IMAGE: React.CSSProperties = { position: "absolute", fontSize: TEXT, color: "#fff", textShadow: "0 0 3px #000, 0 0 2px #000", pointerEvents: "none", lineHeight: "14px" };
const PEAK_COLORS = { heavy: "#d8bf8a", light: "#9fb4cc" };  // two muted tones, readable on gray

// The typical column spacing, the median nearest-neighbour distance between peaks: how near a dragged end must come
// to a column to snap onto it (a third of it), so the reach follows the lattice, not a fixed pixel count.
function columnSpacing(peaks: number[][]): number {
  if (peaks.length < 2) return 0;
  const nearest = peaks.map(([r, c], index) => {
    let best = Infinity;
    peaks.forEach(([r2, c2], other) => { if (other !== index) best = Math.min(best, Math.hypot(r - r2, c - c2)); });
    return best;
  }).sort((a, b) => a - b);
  return nearest[Math.floor(nearest.length / 2)];
}

// A run's changes, structured: the stage's own list, or its text "key before → after, key before → after" split.
function stageChanges(stage: Stage): Change[] {
  if (stage.changed && stage.changed.length) return stage.changed;
  return (stage.changes || "").split(/,\s*|\s·\s/).filter(Boolean).map((item) => {
    const match = item.match(/^(.*?)\s+(\S+)\s*(?:→|->)\s*(\S+)$/);
    return match ? { key: match[1], before: match[2], after: match[3] } : { key: item, before: "", after: "" };
  });
}

function nearestPeak(peaks: number[][], row: number, col: number, reach: number): number {
  let best = -1, distance = reach;
  peaks.forEach(([r, c], index) => { const d = Math.hypot(r - row, c - col); if (d <= distance) { distance = d; best = index; } });
  return best;
}

function LineChart({ series, stages, cursor, bars, log, height, xLabel, yLabel, colors, onPick, stageNames, legend = true, marks, snapX, tooltip, onHoverX, keyItems, outer, fitTo }: {
  series: Series[]; stages: Stage[]; cursor: number | null; bars?: { value: number; label: string }[]; log?: boolean;
  height: number; xLabel: string; yLabel: string; colors: any; onPick?: (epoch: number) => void; stageNames?: boolean; legend?: boolean;
  marks?: { x: number; color: string }[];  // short ticks on the x axis, e.g. where the profile line crosses a column
  snapX?: number[];  // hover lands on these x (the draws), never between them
  tooltip?: (x: number) => React.ReactNode;  // what the hover shows at x
  onHoverX?: (x: number | null) => void;  // the x under the pointer as it moves, null when it leaves
  keyItems?: Series[];  // a shorter legend than one entry a series
  outer?: boolean;  // height includes the legend row, for a chart filling a given box
  fitTo?: Series[];  // the series the y range is fitted to (others may run past it, clipped); all when not given
}) {
  const [hover, setHoverState] = React.useState<number | null>(null);
  const clip = React.useMemo(() => `plot-${Math.random().toString(36).slice(2)}`, []);  // the plot area: a line past the fitted range is cut at its edge. Unique on the page: every widget is its own React root, so useId repeats across cells and a chart would clip to another's box
  const setHover = (x: number | null) => { setHoverState(x); if (onHoverX) onHoverX(x); };
  const [box, width] = useWidth(320);
  const keyBox = React.useRef<HTMLDivElement>(null);
  const [keyHeight, setKeyHeight] = React.useState(15);
  React.useLayoutEffect(() => { const next = keyBox.current ? keyBox.current.offsetHeight : 0; if (next !== keyHeight) setKeyHeight(next); });
  const left = yLabel ? 54 : 34, right = 8, top = 4, bottom = 16;
  const all = series.flatMap((one) => one.points).filter(([, y]) => Number.isFinite(y) && (!log || y > 0));
  const key = legend && (series.length > 1 || (bars || []).length > 0) ? (
    <div ref={keyBox} style={{ display: "flex", flexWrap: "wrap", gap: "0 12px", fontSize: TEXT, lineHeight: "15px", paddingLeft: left }}>
      {(keyItems || series).map((one) => (
        <span key={one.name} style={{ whiteSpace: "nowrap" }}>
          <svg width={16} height={8} style={{ marginRight: 4 }}><line x1={0} x2={16} y1={4} y2={4} stroke={one.color} strokeWidth={2} strokeDasharray={one.dashed ? "4 2" : undefined} /></svg>{one.name}
        </span>
      ))}
      {(bars || []).map((bar) => (
        <span key={bar.label} style={{ whiteSpace: "nowrap", color: colors.textMuted }}>
          <svg width={16} height={8} style={{ marginRight: 4 }}><line x1={0} x2={16} y1={4} y2={4} stroke={BAR_COLOR} strokeWidth={1.5} strokeDasharray="2 2" /></svg>{bar.label} {plain(bar.value, 2)}
        </span>
      ))}
    </div>
  ) : null;
  if (all.length === 0) return <div ref={box} style={{ fontSize: TEXT, color: colors.textMuted }}>no {yLabel} yet</div>;
  if (outer && key) height = Math.max(40, height - keyHeight);
  const fitted = fitTo ? fitTo.flatMap((one) => one.points).filter(([, y]) => Number.isFinite(y) && (!log || y > 0)) : all;
  const values = (fitted.length ? fitted : all).map(([, y]) => y).concat((bars || []).map((bar) => bar.value));
  const xs = all.map(([x]) => x);
  const x0 = Math.min(0, ...xs), x1 = Math.max(1, ...xs);
  const f = (y: number) => (log ? Math.log10(y) : y);
  let y0 = Math.min(...values.map(f)), y1 = Math.max(...values.map(f));
  if (y1 - y0 < 1e-12) { y0 -= 0.5; y1 += 0.5; }
  const pad = 0.06 * (y1 - y0); y0 -= pad; y1 += pad;
  const sx = (x: number) => left + ((x - x0) / (x1 - x0)) * (width - left - right);
  const sy = (y: number) => top + (1 - (f(y) - y0) / (y1 - y0)) * (height - top - bottom);
  const ticks = log ? niceTicks(10 ** y0, 10 ** y1, Math.max(2, Math.floor((height - top - bottom) / 28))).filter((tick) => tick.value > 0)
    : niceTicks(y0, y1, Math.max(2, Math.floor((height - top - bottom) / 28)));
  const at = (event: React.MouseEvent<SVGSVGElement>) => {  // the x under the pointer: a run boundary within 6 px, else the nearest draw
    const px = event.clientX - event.currentTarget.getBoundingClientRect().left;
    const x = Math.min(x1, Math.max(x0, x0 + ((px - left) / (width - left - right)) * (x1 - x0)));
    const boundary = stages.slice(1).find((stage) => Math.abs(sx(stage.start) - px) <= 6);
    if (boundary) return boundary.start;
    if (snapX && snapX.length) return snapX.reduce((best, value) => (Math.abs(value - x) < Math.abs(best - x) ? value : best), snapX[0]);
    return Math.round(x);
  };
  const pick = (event: React.MouseEvent<SVGSVGElement>) => { const x = at(event); if (tooltip) setHoverState(x); if (onPick) onPick(x); };  // a click commits; on touch the tooltip stays to read
  const hoverPx = hover != null ? sx(hover) : 0;
  return (
    <div ref={box} style={{ minWidth: 0, position: "relative" }}>
      {key}
      <svg width={width} height={height} style={{ display: "block", cursor: onPick ? "pointer" : "default" }} onClick={pick}
        onMouseMove={tooltip || onHoverX ? (event) => { const x = at(event); if (x !== hover) setHover(x); } : undefined} onMouseLeave={() => setHover(null)}>
        {stages.map((stage, index) => {
          const end = index + 1 < stages.length ? stages[index + 1].start : x1;
          const band = Math.max(0, sx(end) - sx(stage.start));
          return (
            <g key={stage.name + index}>
              <rect x={sx(stage.start)} y={top} width={band} height={height - top - bottom}
                fill={colors.text} opacity={index % 2 ? 0.08 : 0.03} />
              {stageNames && index > 0 && <path d={`M${sx(stage.start) - 4},${top} L${sx(stage.start) + 4},${top} L${sx(stage.start)},${top + 6} Z`} fill={colors.textMuted} />}
              {stageNames && band > 6.5 * stage.name.length + 6 && <text x={sx(stage.start) + 3} y={top + 11} fontSize={TEXT} fill={colors.textMuted}>{stage.name}</text>}
            </g>
          );
        })}
        {(bars || []).map((bar) => (
          <line key={bar.label} x1={left} x2={width - right} y1={sy(bar.value)} y2={sy(bar.value)} stroke={BAR_COLOR} strokeDasharray="2 2" strokeWidth={1} />
        ))}
        <clipPath id={clip}><rect x={left} y={top} width={Math.max(0, width - left - right)} height={Math.max(0, height - top - bottom)} /></clipPath>
        {series.map((one) => {
          const points = one.points.filter(([, y]) => Number.isFinite(y) && (!log || y > 0));
          if (points.length === 0) return null;
          const d = points.map(([x, y], index) => `${index ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join("");
          return <path key={one.name} d={d} fill="none" stroke={one.color} strokeWidth={1.6} strokeDasharray={one.dashed ? "5 3" : undefined} clipPath={`url(#${clip})`} />;
        })}
        {(marks || []).map((mark, index) => (
          <line key={index} x1={sx(mark.x)} x2={sx(mark.x)} y1={height - bottom} y2={height - bottom - 6} stroke={mark.color} strokeWidth={2} />
        ))}
        {cursor != null && <line x1={sx(cursor)} x2={sx(cursor)} y1={top} y2={height - bottom} stroke={colors.accent} strokeWidth={1.5} />}
        {hover != null && <line x1={hoverPx} x2={hoverPx} y1={top} y2={height - bottom} stroke={colors.textMuted} strokeWidth={1} strokeDasharray="3 2" />}
        <line x1={left} x2={left} y1={top} y2={height - bottom} stroke={colors.border} />
        <line x1={left} x2={width - right} y1={height - bottom} y2={height - bottom} stroke={colors.border} />
        {ticks.map((tick) => (
          <text key={tick.label} x={left - 4} y={sy(tick.value) + 4} fontSize={TEXT} fill={colors.textMuted} textAnchor="end">{tick.label}</text>
        ))}
        <text x={left} y={height - 3} fontSize={TEXT} fill={colors.textMuted}>{plain(x0, 0)}</text>
        <text x={width - right} y={height - 3} fontSize={TEXT} fill={colors.textMuted} textAnchor="end">{plain(x1, 0)}</text>
        {width - left - right > 6 * xLabel.length + 70 && <text x={(left + width) / 2} y={height - 3} fontSize={TEXT} fill={colors.textMuted} textAnchor="middle">{xLabel}</text>}
        <text x={9} y={(top + height - bottom) / 2} fontSize={TEXT} fill={colors.textMuted} textAnchor="middle"
          transform={`rotate(-90 9 ${(top + height - bottom) / 2})`}>{yLabel}</text>
      </svg>
      {hover != null && tooltip && (
        <div style={{ position: "absolute", top: legend ? 16 : 0, ...(hoverPx < width / 2 ? { left: hoverPx + 8 } : { right: width - hoverPx + 8 }),
          maxWidth: Math.max(140, Math.min(300, (hoverPx < width / 2 ? width - hoverPx : hoverPx) - 12)), pointerEvents: "none", zIndex: 5,
          background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 4, padding: "3px 6px", fontSize: TEXT, lineHeight: "15px",
          boxShadow: "0 1px 4px rgba(0,0,0,0.3)" }}>{tooltip(hover)}</div>
      )}
    </div>
  );
}

// One engine for the widget's images, and a slot each: the image is uploaded once per new array and recolored by the
// engine on a range or colormap change, as Show2D draws.
const Engine = React.createContext<{ engine: DisplayColormapEngine | null; slot: () => number; smooth: boolean }>({ engine: null, slot: () => 0, smooth: false });

function ImageCanvas({ data, rows, cols, low: givenLow, high: givenHigh, cmap: givenCmap, line, lineColor, pixelA, barA, height, keepAspect, overlay, children, inset, zoomable, unit = "rad", rangeFrom, symmetric = false }: {
  data: Float32Array; rows: number; cols: number; low: number; high: number; cmap: string;
  line?: number[]; lineColor?: string; pixelA?: number; barA?: number /* any length: draw the scale bar (its length is chosen as Show2D chooses it) */; height?: number; keepAspect?: boolean;
  overlay?: React.ReactNode;  // drawn in image pixels over the image
  children?: React.ReactNode;  // HTML over the image (handles), positioned in percent
  inset?: React.ReactNode;  // a picture in the image's corner (its spectrum), outside the zoom
  zoomable?: boolean;  // a field image of its view: wheel zoom, drag pan, double-click reset (linked across the view), the view's contrast and colormap, a hover readout
  unit?: string;  // the readout's unit
  rangeFrom?: Float32Array[];  // the images whose percentiles set this one's contrast once the view's contrast is changed (before and after share one range); its own data when not given
  symmetric?: boolean;  // a change map: its contrast symmetric about zero
}) {
  const { engine, slot: nextSlot, smooth } = React.useContext(Engine);
  const link = React.useContext(ViewLink);
  const linked = zoomable && link ? link : null;
  const display = linked ? linked.display : DEFAULT_DISPLAY;
  const custom = display.low !== DEFAULT_DISPLAY.low || display.high !== DEFAULT_DISPLAY.high;
  const [low, high] = React.useMemo(() => {  // the view's contrast: the given range when left as it was, else percentiles of rangeFrom (one range for images
    if (!linked || !custom || !data.length) return [givenLow, givenHigh];  // shown together: before with after), a diverging map symmetric about zero
    return contrastRange(rangeFrom && rangeFrom.length ? rangeFrom : [data], display, symmetric);
  }, [linked, custom, data, rangeFrom, symmetric, display.low, display.high, givenLow, givenHigh]);
  const cmap = linked && display.cmap && givenCmap === "gray" ? display.cmap : givenCmap;
  const [hover, setHover] = React.useState<[number, number] | null>(null);
  const ref = React.useRef<HTMLCanvasElement>(null);
  const slot = React.useRef(-1);
  if (slot.current < 0) slot.current = nextSlot();
  const uploaded = React.useRef<{ engine: DisplayColormapEngine | null; data: Float32Array | null; rows: number; cols: number }>({ engine: null, data: null, rows: 0, cols: 0 });
  React.useEffect(() => {
    const canvas = ref.current;
    if (!engine || !canvas || rows === 0 || cols === 0 || data.length < rows * cols) return;
    let cancelled = false;
    const before = uploaded.current;
    if (before.engine !== engine || before.data !== data || before.rows !== rows || before.cols !== cols) {
      engine.uploadData(slot.current, data.length === rows * cols ? data : data.subarray(0, rows * cols), cols, rows);
      uploaded.current = { engine, data, rows, cols };
    }
    engine.renderSlotsToImageBitmapAsync([slot.current], [{ vmin: low, vmax: high }], false, cmap, COLORMAPS[cmap] || COLORMAPS.gray).then((bitmaps) => {
      const bitmap = bitmaps && bitmaps[0];
      if (!bitmap) return;
      if (!cancelled) {
        if (canvas.width !== cols || canvas.height !== rows) { canvas.width = cols; canvas.height = rows; }
        canvas.getContext("2d")?.drawImage(bitmap, 0, 0);
      }
      bitmap.close();
    }).catch((error) => console.error("[ShowReconstruction] colormap failed", error));
    return () => { cancelled = true; };
  }, [engine, data, rows, cols, low, high, cmap]);
  const scale = linked ? linked.zoom.scale : 1;
  // Show2D's scale bar (figure.ts: about 60 px rounded to a nice length, 5 px thick, a 16 px label centred over it, 12 px
  // in from the corner), bottom left since the spectrum inset holds the bottom right; a small image gets a smaller bar.
  const boxWidth = height && keepAspect && rows ? height * cols / rows : 0;
  const roomy = (height || 0) >= 160;
  const bar = pixelA && barA && boxWidth > 0 ? scaleBarGeometry({ x: 0, y: 0, width: boxWidth, height: height! }, scale * boxWidth / cols, pixelA, "Å",
    { position: "bottom-left", maxBarFraction: 0.4, margin: roomy ? 12 : 6, barThickness: roomy ? 5 : 3 }) : null;
  const picture = <>
    <canvas ref={ref} style={{ width: "100%", height: "100%", display: "block", imageRendering: smooth ? "auto" : "pixelated" }} />
    <svg viewBox={`0 0 ${cols} ${rows}`} preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
      {line && line.length === 4 && <line x1={line[1]} y1={line[0]} x2={line[3]} y2={line[2]} stroke={lineColor || "#fff"} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />}
      {overlay}
    </svg>
    {children}
  </>;
  const read = hover && linked ? readAt(data, rows, cols, hover, pixelA) : null;
  return (
    <div style={{ position: "relative", ...(height && keepAspect ? { height, aspectRatio: `${cols} / ${rows}`, maxWidth: "100%", margin: "0 auto" }
      : height ? { width: "100%", height } : { width: "100%", aspectRatio: `${cols} / ${rows}` }) }}>
      {linked ? <ZoomFrame onHover={setHover}>{picture}</ZoomFrame> : picture}
      {bar && <div style={{ position: "absolute", left: bar.barX, top: bar.barY, width: bar.barPx, height: bar.barHeight, background: "#fff", boxShadow: "1px 1px 2px rgba(0,0,0,0.7)", pointerEvents: "none" }} />}
      {bar && <div style={{ ...ON_IMAGE, left: bar.barX + bar.barPx / 2, bottom: height! - bar.barY + (roomy ? 4 : 2), transform: "translateX(-50%)", whiteSpace: "nowrap",
        fontSize: roomy ? 16 : 12, lineHeight: 1, textShadow: "1px 1px 2px rgba(0,0,0,0.7)" }}>{bar.label}</div>}
      {inset}
      {read && Number.isFinite(read.value) && <div style={{ ...ON_IMAGE, background: "rgba(0,0,0,0.6)", padding: "0 3px", borderRadius: 2, textShadow: "none", right: 2, top: 2, fontSize: 10 }}>
        {plain(read.value, 3)}{unit ? ` ${unit}` : ""} · {read.nm ? `${plain(read.nm[0], 2)}, ${plain(read.nm[1], 2)} nm` : `${read.row}, ${read.col} px`}</div>}
    </div>
  );
}

function profileAlong(slice: Float32Array, rows: number, cols: number, line: number[], pixelA: number): [number, number][] {
  if (line.length !== 4 || slice.length < rows * cols) return [];
  const [r0, c0, r1, c1] = line;
  const length = Math.hypot(r1 - r0, c1 - c0);
  const steps = Math.max(2, Math.round(length * 2));
  const out: [number, number][] = [];
  for (let step = 0; step <= steps; step++) {  // bilinear along the line, display only
    const t = step / steps, r = r0 + t * (r1 - r0), c = c0 + t * (c1 - c0);
    const ri = Math.min(rows - 2, Math.max(0, Math.floor(r))), ci = Math.min(cols - 2, Math.max(0, Math.floor(c)));
    const fr = r - ri, fc = c - ci;
    const value = slice[ri * cols + ci] * (1 - fr) * (1 - fc) + slice[ri * cols + ci + 1] * (1 - fr) * fc
      + slice[(ri + 1) * cols + ci] * fr * (1 - fc) + slice[(ri + 1) * cols + ci + 1] * fr * fc;
    out.push([t * length * pixelA, value]);
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------------
// The dashboard

// A box that takes the space left in its panel column (flex share `grow`, at least `min` px tall) and hands its size to
// what it draws, so the columns of a row end level with no empty band under a short panel.
function Fill({ grow = 1, min, children }: { grow?: number; min: number; children: (width: number, height: number) => React.ReactNode }) {
  const ref = React.useRef<HTMLDivElement>(null);
  const [size, setSize] = React.useState<[number, number]>([300, min]);
  React.useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      const next: [number, number] = [Math.floor(entry.contentRect.width), Math.floor(entry.contentRect.height)];
      if (next[0] > 0 && next[1] > 0) setSize(next);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return <div style={{ flex: `${grow} 1 0`, minHeight: min, position: "relative", minWidth: 0 }}>
    <div ref={ref} style={{ position: "absolute", inset: 0 }}>{children(size[0], Math.max(min, size[1]))}</div>
  </div>;
}

// The power spectrum of a field (or a square region of it), through the package's display FFT (WebGPU, or its CPU twin)
// as Show2D takes it: mean removed, Show2D's Hann window at the region's own size against edge streaks, zero padded to
// a power of two of at least 256 (padding past the box only interpolates, so a small box still reads smoothly),
// log10 of |F|^2, centred and cropped to ±2.6 1/Å; the ring marks 50 pm (2 1/Å). Each request waits for the next
// animation frame and is dropped when a newer one arrives, so a region drag stays live.
type Spectrum = { image: Float32Array; size: number; low: number; high: number; ringPx: number; edgeQ: number;
  spotPm: number | null; spotPx: number };  // the finest lattice spot: its spacing in pm and its radius in spectrum pixels (finestSpot)

// The finest lattice spot of a log10 power spectrum (size × size, centred, dq 1/Å a pixel): the largest radius whose
// ring holds a pixel clearly over the ring's median, from 4 Å inward. "Clearly" is measured on the spectrum itself:
// in the outermost fifth of the radii (no lattice there on real data) a ring's peak stands about 1.7 decades over its
// median (window tails and noise), so a spot must stand 0.7 decade (5 ×) over that level, and at least 2 decades.
// With a fixed 2 decades the read jumped from 58 to 39 pm between two nearly equal pictures (21 Nov 2026): the floor
// itself reached 2. It reads how far the lattice's information reaches, in the picture's own unit; a read of the
// picture, not a resolution, and near the floor it moves by one lattice spot between like pictures.
function finestSpot(image: Float32Array, size: number, dq: number): { pm: number | null; px: number } {
  const half = size / 2, rings: number[][] = Array.from({ length: Math.floor(half) }, () => []);
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) {
    const radius = Math.round(Math.hypot(r - half, c - half));
    if (radius < rings.length) rings[radius].push(image[r * size + c]);
  }
  const contrast = rings.map((ring) => { if (ring.length < 8) return 0; const sorted = Float32Array.from(ring).sort(); return sorted[sorted.length - 1] - sorted[sorted.length >> 1]; });
  const outer = contrast.slice(Math.floor(0.8 * contrast.length)).sort((a, b) => a - b);
  const needed = Math.max(2, (outer.length ? outer[outer.length >> 1] : 0) + 0.7);
  for (let radius = Math.floor(0.8 * contrast.length) - 1; radius * dq >= 0.25; radius--) if (contrast[radius] >= needed) return { pm: 100 / (radius * dq), px: radius };
  return { pm: null, px: 0 };
}

function useSpectrum(data: Float32Array, rows: number, cols: number, region: { row: number; col: number; size: number } | null, pixelA: number): Spectrum | null {
  const [spectrum, setSpectrum] = React.useState<Spectrum | null>(null);
  const fft = React.useRef<DisplayFFT | null>(null);
  const generation = React.useRef(0);
  React.useEffect(() => {
    if (!rows || !cols || data.length < rows * cols) { setSpectrum(null); return; }
    const mine = ++generation.current;
    (async () => {
      await new Promise((resolve) => requestAnimationFrame(resolve));
      if (mine !== generation.current) return;
      const engine = fft.current || (fft.current = await getDisplayFFT());
      const r0 = region ? region.row : 0, c0 = region ? region.col : 0, h = region ? region.size : rows, w = region ? region.size : cols;
      const crop = new Float32Array(h * w);
      let mean = 0;
      for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) { const value = data[(r0 + r) * cols + c0 + c]; crop[r * w + c] = value; mean += value; }
      mean /= h * w;
      for (let index = 0; index < crop.length; index++) crop[index] -= mean;
      applyHannWindow2D(crop, w, h);
      let n = 256; while (n < Math.max(h, w)) n <<= 1;
      const out = await engine.fft2D(zeroPadTopLeft(crop, w, h, n, n), new Float32Array(n * n), n, n, false);
      if (mine !== generation.current) return;
      fftshift(out.real, n, n); fftshift(out.imag, n, n);
      const dq = 1 / (n * pixelA);
      const half = Math.max(4, Math.min(n / 2, Math.round(2.6 / dq)));
      const size = 2 * half, image = new Float32Array(size * size);
      for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) {
        const index = (n / 2 - half + r) * n + (n / 2 - half + c);
        image[r * size + c] = Math.log10(out.real[index] ** 2 + out.imag[index] ** 2 + 1e-12);
      }
      const sorted = Float32Array.from(image).sort();
      const spot = finestSpot(image, size, dq);
      setSpectrum({ image, size, low: sorted[Math.floor(0.5 * sorted.length)], high: sorted[Math.floor(0.9995 * (sorted.length - 1))], ringPx: 2 / dq <= half ? 2 / dq : 0, edgeQ: half * dq,
        spotPm: spot.pm, spotPx: spot.px });
    })().catch((error) => console.error("[ShowReconstruction] spectrum failed", error));
  }, [data, rows, cols, region, pixelA]);
  return spectrum;
}

// ---------------------------------------------------------------------------------------------------------------
// The field: one image with its line and spectrum region, shared by the dashboard and the compare view

type Region = { row: number; col: number; size: number };

// The volume along a line, every slice: slices down, the line across (bilinear, display only); null without a line.
function cutAlong(volume: Float32Array, slicesCount: number, rows: number, cols: number, line: number[] | null | undefined, pixelA: number) {
  if (!line || line.length !== 4 || !slicesCount || volume.length < slicesCount * rows * cols) return null;
  const rowsOut: [number, number][][] = [];
  for (let index = 0; index < slicesCount; index++) rowsOut.push(profileAlong(volume.subarray(index * rows * cols, (index + 1) * rows * cols), rows, cols, line, pixelA));
  const width = rowsOut[0].length, image = new Float32Array(slicesCount * width);
  rowsOut.forEach((row, index) => row.forEach(([, value], column) => { image[index * width + column] = value; }));
  return { image, width, lengthA: rowsOut[0][width - 1][0] };
}

// The columns a line passes within the snap reach, as distance along it in Å, for ticks on the profile.
function lineCrossings(line: number[] | null | undefined, peaks: number[][], reach: number, pixelA: number) {
  if (!line || line.length !== 4 || !peaks.length) return [];
  const [r0, c0, r1, c1] = line, length = Math.hypot(r1 - r0, c1 - c0) || 1, near = reach || 3;
  const out: { x: number; color: string }[] = [];
  for (const [row, col, heavy] of peaks) {
    const along = ((row - r0) * (r1 - r0) + (col - c0) * (c1 - c0)) / length;
    const off = Math.abs((row - r0) * (c1 - c0) - (col - c0) * (r1 - r0)) / length;
    if (along >= 0 && along <= length && off <= near) out.push({ x: along * pixelA, color: heavy ? PEAK_COLORS.heavy : PEAK_COLORS.light });
  }
  return out;
}

// One field image the reader edits: drag the line or its ends (an end dropped near a column snaps to it, the target
// ring lit while dragging), drag elsewhere to draw a square spectrum region, drag inside the region to move it. The
// line under a drag goes to onDragLine (so profiles follow live) and the dropped line to onLine.
function FieldView({ data, rows, cols, low, high, pixelA, height, line, editLine, onDragLine, onLine, peaks, showPeaks, region, onRegion, colors, rangeFrom, inset }: {
  data: Float32Array; rows: number; cols: number; low: number; high: number; pixelA: number; height: number;
  line: number[] | null | undefined; editLine: boolean; onDragLine: (line: number[] | null) => void; onLine: (line: number[]) => void;
  peaks: number[][]; showPeaks: boolean; region: Region | null; onRegion: (region: Region | null) => void; colors: any;
  rangeFrom?: Float32Array[];  // the images shown beside it that share its contrast once the view's contrast is changed
  inset?: React.ReactNode;  // its spectrum, in its corner
}) {
  const spacing = React.useMemo(() => columnSpacing(peaks), [peaks]);
  const reach = spacing / 3;
  // part 0 and 1: a line end, 2: the whole line, 3: a new region drawn from its corner, 4: the region moved
  const drag = React.useRef<{ part: number; row: number; col: number; start: number[]; line: number[] | null } | null>(null);
  const [snapTarget, setSnapTarget] = React.useState(-1);
  const box = React.useRef<HTMLDivElement>(null);
  const toPixel = (event: React.PointerEvent) => {
    const rect = box.current!.getBoundingClientRect();
    return [((event.clientY - rect.top) / rect.height) * rows, ((event.clientX - rect.left) / rect.width) * cols, rect.width / cols];
  };
  const clampRow = (value: number) => Math.min(rows - 1, Math.max(0, value)), clampCol = (value: number) => Math.min(cols - 1, Math.max(0, value));
  const largest = Math.min(256, rows, cols);
  const hasLine = editLine && !!line && line.length === 4;
  const onDown = (event: React.PointerEvent) => {
    const [row, col, scale] = toPixel(event);
    const grab = 12 / scale;  // 12 screen px around a handle or the line picks it
    let part = -1;
    if (hasLine) {
      const [r0, c0, r1, c1] = line!;
      const along = Math.max(0, Math.min(1, ((row - r0) * (r1 - r0) + (col - c0) * (c1 - c0)) / Math.max(1e-9, (r1 - r0) ** 2 + (c1 - c0) ** 2)));
      part = Math.hypot(row - r0, col - c0) < grab ? 0 : Math.hypot(row - r1, col - c1) < grab ? 1
        : Math.hypot(row - (r0 + along * (r1 - r0)), col - (c0 + along * (c1 - c0))) < grab ? 2 : -1;
    }
    if (part < 0) part = region && row >= region.row && row < region.row + region.size && col >= region.col && col < region.col + region.size ? 4 : 3;
    event.preventDefault();
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    drag.current = { part, row, col, start: part === 4 ? [region!.row, region!.col, region!.size] : part === 3 ? [row, col] : [...line!], line: null };
  };
  const onMove = (event: React.PointerEvent) => {
    const held = drag.current;
    if (!held) return;
    const [row, col] = toPixel(event);
    if (held.part === 3) {  // a square from the pressed corner toward the pointer
      const size = Math.round(Math.min(largest, Math.max(Math.abs(row - held.row), Math.abs(col - held.col))));
      if (size < 8) return;
      const top = row < held.row ? held.row - size : held.row, leftCol = col < held.col ? held.col - size : held.col;
      onRegion({ row: Math.round(Math.min(rows - size, Math.max(0, top))), col: Math.round(Math.min(cols - size, Math.max(0, leftCol))), size });
      return;
    }
    if (held.part === 4) {
      const [r0, c0, size] = held.start;
      onRegion({ row: Math.round(Math.min(rows - size, Math.max(0, r0 + row - held.row))), col: Math.round(Math.min(cols - size, Math.max(0, c0 + col - held.col))), size });
      return;
    }
    const [r0, c0, r1, c1] = held.start;
    let next: number[];
    if (held.part === 2) {  // the whole line, kept inside the image
      const dr = Math.min(rows - 1 - Math.max(r0, r1), Math.max(-Math.min(r0, r1), row - held.row));
      const dc = Math.min(cols - 1 - Math.max(c0, c1), Math.max(-Math.min(c0, c1), col - held.col));
      next = [r0 + dr, c0 + dc, r1 + dr, c1 + dc];
    } else {
      next = held.part === 0 ? [clampRow(row), clampCol(col), r1, c1] : [r0, c0, clampRow(row), clampCol(col)];
    }
    setSnapTarget(held.part === 2 || !reach ? -1 : nearestPeak(peaks, next[held.part * 2], next[held.part * 2 + 1], reach));
    held.line = next;
    onDragLine(next);
  };
  const onUp = () => {
    const held = drag.current;
    if (!held) return;
    drag.current = null;
    if (held.part >= 3) return;
    let next = held.line || held.start;
    if (held.part !== 2 && snapTarget >= 0) {
      const [row, col] = peaks[snapTarget];
      next = held.part === 0 ? [row, col, next[2], next[3]] : [next[0], next[1], row, col];
    }
    onLine(next.map((value) => Math.round(value * 100) / 100));
    onDragLine(null); setSnapTarget(-1);
  };
  const ringRadius = Math.max(1.5, (spacing || 12) * 0.28);
  const overlay = <>
    {showPeaks && peaks.map(([row, col, heavy], index) => (
      <circle key={index} cx={col + 0.5} cy={row + 0.5} r={ringRadius} fill="none" stroke={index === snapTarget ? colors.accent : heavy ? PEAK_COLORS.heavy : PEAK_COLORS.light}
        strokeWidth={index === snapTarget ? 2.5 : 1} opacity={index === snapTarget ? 1 : 0.75} vectorEffect="non-scaling-stroke" />
    ))}
    {region && <rect x={region.col} y={region.row} width={region.size} height={region.size} fill="none" stroke="#fff" strokeDasharray="4 3" strokeWidth={1.2} vectorEffect="non-scaling-stroke" />}
  </>;
  const handle = (row: number, col: number, key: string) => (
    <div key={key} style={{ position: "absolute", left: `${(100 * (col + 0.5)) / cols}%`, top: `${(100 * (row + 0.5)) / rows}%`, width: 10, height: 10,
      marginLeft: -5, marginTop: -5, borderRadius: "50%", border: `2px solid ${colors.accent}`, background: "rgba(0,0,0,0.35)", boxSizing: "border-box", pointerEvents: "none" }} />
  );
  return (
    <div ref={box} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}
      style={{ touchAction: "none", userSelect: "none", cursor: "crosshair", width: "fit-content", maxWidth: "100%", margin: "0 auto" }}>
      <ImageCanvas data={data} rows={rows} cols={cols} low={low} high={high} cmap="gray" line={hasLine ? line! : undefined} lineColor={colors.accent} pixelA={pixelA} barA={10}
        height={height} keepAspect overlay={overlay} zoomable rangeFrom={rangeFrom} inset={inset}>
        {hasLine && [handle(line![0], line![1], "start"), handle(line![2], line![3], "end")]}
      </ImageCanvas>
    </div>
  );
}

// What every spectrum carries: the dashed ring at 50 pm, and a thin ring through its finest lattice spot.
function spectrumRings(spectrum: Spectrum) {
  const centre = spectrum.size / 2;
  return <>
    {spectrum.ringPx > 0 && <circle cx={centre} cy={centre} r={spectrum.ringPx} fill="none" stroke="#fff" opacity={0.7} strokeDasharray="4 3" strokeWidth={1} vectorEffect="non-scaling-stroke" />}
    {spectrum.spotPx > 0 && <circle cx={centre} cy={centre} r={spectrum.spotPx} fill="none" stroke="#7ddc8a" opacity={0.8} strokeWidth={1} vectorEffect="non-scaling-stroke" />}
  </>;
}
const SPOT_WHY = "finest spot: the smallest spacing whose lattice spot stands clearly (5 ×) over the spectrum's own outer floor (the thin green ring); smaller is finer. A read of the picture, not a resolution; like pictures can differ by one spot.";
function spotText(spectrum: Spectrum | null): string {
  return spectrum && spectrum.spotPm != null ? `spots to ${Math.round(spectrum.spotPm)} pm` : "";
}

// The spectrum inside its image: an inset in the bottom-right corner, dragged anywhere on the image, its side a percent
// of the image's set in "controls" (FFT size; 0 hides it). A separate panel beside every image cost a column of width.
function SpectrumInset({ spectrum, side, label, range }: { spectrum: Spectrum | null; side: number; label?: string; range?: number[] | null }) {
  const link = React.useContext(ViewLink);
  const size = Math.round(side * (link ? link.display.fft : DEFAULT_DISPLAY.fft) / 100);
  if (!spectrum || size < 24) return null;
  return (
    <CornerInset size={size} title={`spectrum${label ? ` of the ${label}` : ""}: Hann-windowed power, log scale, the dashed ring at 50 pm${spotText(spectrum) ? `; ${spotText(spectrum)} (${SPOT_WHY})` : ""}. Drag to move, double-click to send back; its size is "FFT size" in controls.`}>
      <ImageCanvas data={spectrum.image} rows={spectrum.size} cols={spectrum.size} low={range ? range[0] : spectrum.low} high={range ? range[1] : spectrum.high} cmap="gray" height={size} keepAspect
        overlay={spectrumRings(spectrum)} />
      {size >= 90 && <div style={{ ...ON_IMAGE, left: 3, bottom: 1 }}>{spotText(spectrum) || "50 pm ring"}</div>}
    </CornerInset>
  );
}

// ---------------------------------------------------------------------------------------------------------------
// Controls

const LABEL = { ...WIDGET_TYPOGRAPHY.label, fontSize: 10 };  // the siblings' control label

// Show3D's playback row over a list of real positions (the epochs with a checkpoint, the slices), so every step lands
// on data: first and previous, play or pause, next and last, a three-thumb slider (loop start, position shown, loop
// end), the count, and a "…" menu with fps, Loop and Bounce. Loop wraps from the loop's end to its start; Bounce turns at both ends;
// with neither, playing stops at the end.
function Stepper({ positions, value, onChange, label, colors, name }: {
  positions: number[]; value: number; onChange: (position: number) => void; label?: React.ReactNode; colors: any; name: string;
}) {
  const index = Math.max(0, positions.reduce((best, position, at) => (position <= value ? at : best), 0));
  const last = Math.max(0, positions.length - 1);
  const [range, setRange] = React.useState<[number, number]>([0, last]);
  const [playing, setPlaying] = React.useState(false);
  const [fps, setFps] = React.useState(4);
  const [loop, setLoop] = React.useState(true);
  const [bounce, setBounce] = React.useState(false);
  const [menu, setMenu] = React.useState<HTMLElement | null>(null);
  const direction = React.useRef<1 | -1>(1);
  const low = Math.min(range[0], last), high = Math.min(Math.max(range[1], low), last);
  React.useEffect(() => { setRange(([start, end]) => [Math.min(start, last), end >= last - 1 ? last : Math.min(end, last)]); }, [last]);  // a range to the end keeps following new checkpoints
  const go = (at: number) => onChange(positions[Math.max(0, Math.min(last, at))]);
  const here = React.useRef(index);
  here.current = index;
  React.useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      const at = here.current;
      let next = at < low || at > high ? low : at + direction.current;
      if (next > high || next < low) {
        if (bounce) { direction.current = direction.current === 1 ? -1 : 1; next = Math.max(low, Math.min(high, at + direction.current)); }
        else if (loop) next = direction.current === 1 ? low : high;
        else { setPlaying(false); return; }
      }
      go(next);
    }, 1000 / fps);
    return () => window.clearInterval(timer);
  }, [playing, fps, low, high, loop, bounce, positions]);
  const icon = { color: colors.textMuted, p: 0.25 };
  const pair = { display: "flex", alignItems: "center", gap: `${GAP}px`, flexShrink: 0 } as const;
  return (
    <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: `${GAP}px`, mt: `${GAP}px` }}>
      <Box sx={{ ...controlRow, flexWrap: "wrap", border: `1px solid ${colors.border}`, bgcolor: colors.controlBg, gap: `${GAP}px` }}>
        <Box sx={{ display: "flex", flexShrink: 0 }}>
          <IconButton size="small" sx={icon} aria-label={`First ${name}`} title={`First ${name} of the range`} disabled={index <= low} onClick={() => go(low)}><SkipPreviousIcon sx={{ fontSize: 18 }} /></IconButton>
          <IconButton size="small" sx={icon} aria-label={`Previous ${name}`} title={`Previous ${name} (←)`} disabled={index <= 0} onClick={() => go(index - 1)}><NavigateBeforeIcon sx={{ fontSize: 18 }} /></IconButton>
          <PlayPauseButton playing={playing} color={colors.accent} disabled={last === 0} onToggle={() => { direction.current = 1; setPlaying(!playing); }} />
          <IconButton size="small" sx={icon} aria-label={`Next ${name}`} title={`Next ${name} (→)`} disabled={index >= last} onClick={() => go(index + 1)}><NavigateNextIcon sx={{ fontSize: 18 }} /></IconButton>
          <IconButton size="small" sx={icon} aria-label={`Last ${name}`} title={`Last ${name} of the range`} disabled={index >= high} onClick={() => go(high)}><SkipNextIcon sx={{ fontSize: 18 }} /></IconButton>
        </Box>
        <Slider size="small" min={0} max={last} step={1} value={[low, Math.min(Math.max(index, low), high), high]} disableSwap
          getAriaLabel={(thumb) => [`Loop start, ${name}`, `${name} shown`, `Loop end, ${name}`][thumb]}
          onChange={(_, values, thumb) => { const [start, at, end] = values as number[]; if (thumb === 1) go(at); else setRange([start, end]); }}
          sx={{ ...sliderStyles.small, width: 96, mx: "6px", flexShrink: 0 }} />
        <span style={{ ...LABEL, color: colors.textMuted, minWidth: `${String(positions.length).length * 2 + 1}ch`, textAlign: "right" }}>{index + 1}/{positions.length}</span>
        <Button size="small" sx={{ ...compactButton, minWidth: 0, px: 0.6, color: colors.textMuted }} aria-label={`Playback options, ${name}s`} aria-haspopup="menu"
          title={`${fps} fps${bounce ? " · bounce" : loop ? " · loop" : ""}`} onClick={(event) => setMenu(event.currentTarget)}>…</Button>
        <Menu anchorEl={menu} open={Boolean(menu)} onClose={() => setMenu(null)} slotProps={{ paper: { sx: { bgcolor: colors.bg, color: colors.text, border: `1px solid ${colors.border}` } } }}>
          <Box sx={{ px: 1.5, py: 0.5, display: "flex", flexDirection: "column", gap: `${GAP}px` }}>
            <Box sx={pair}>
              <span style={{ ...LABEL, color: colors.textMuted, minWidth: 40 }}>fps</span>
              <Slider value={fps} min={1} max={30} step={1} onChange={(_, rate) => setFps(rate as number)} size="small" valueLabelDisplay="auto"
                sx={{ ...sliderStyles.small, width: 80, flexShrink: 0 }} aria-label={`Playback ${name}s per second`} />
              <span style={{ ...LABEL, color: colors.textMuted, minWidth: 16 }}>{fps}</span>
            </Box>
            <Box sx={pair}>
              <span style={{ ...LABEL, color: colors.textMuted, minWidth: 40 }}>Loop</span>
              <Switch size="small" checked={loop} onChange={() => setLoop(!loop)} sx={switchStyles.small} slotProps={{ input: { "aria-label": `Toggle loop playback, ${name}s` } }} />
            </Box>
            <Box sx={pair}>
              <span style={{ ...LABEL, color: colors.textMuted, minWidth: 40 }}>Bounce</span>
              <Switch size="small" checked={bounce} onChange={() => setBounce(!bounce)} sx={switchStyles.small} slotProps={{ input: { "aria-label": `Toggle bounce playback, ${name}s` } }} />
            </Box>
          </Box>
        </Menu>
      </Box>
      {label && <span style={{ fontSize: TEXT, whiteSpace: "nowrap" }}>{label}</span>}
    </Box>
  );
}

// A panel: its title, the labels a reader needs beside it (units, which slice or epoch), and an (i) holding how to read
// and use it, so no instruction sits in the layout.
function Panel({ title, hint, info, children, colors, theme, area }: {
  title: string; hint?: React.ReactNode; info?: React.ReactNode; children: React.ReactNode; colors: any; theme: "light" | "dark"; area?: string;
}) {
  return (
    <div style={{ gridArea: area, borderRadius: 4, padding: `${GAP}px 6px 6px`, minWidth: 0, background: colors.bgAlt, display: "flex", flexDirection: "column" }}>
      <div style={{ fontSize: 12, lineHeight: "18px", marginBottom: 2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
        <b>{title}</b>{hint && <span style={{ color: colors.textMuted, fontSize: TEXT }}> · {hint}</span>}
        {info && <InfoTooltip text={info} theme={theme} toggleOnClick />}
      </div>
      {children}
    </div>
  );
}

// A number against its pass bar, ✓ or ⚠; nothing when the run has no such number.
// What a health number means and its limit, in plain words, for its hover.
const MEANING: { [name: string]: string } = {
  r99_A: "the radius holding 99 % of the beam at its sharpest plane",
  copies_percent: "the probe's power sitting on the lattice's other sites (copies of the probe)",
  far_power_percent: "the probe's power more than 5 Å from its core",
};

// A number against its limit: ⚠ past it, with the limit and the meaning in plain words on hover; nothing when the run
// has no such number.
function Health({ label, value, unit, bar, colors, digits, name }: { label: string; value: number | undefined; unit: string; bar?: [number, string]; colors: any; digits: number; name?: string }) {
  if (value == null || !Number.isFinite(value)) return null;
  let mark = "", tone = colors.text, why = name && MEANING[name] ? `${label}: ${MEANING[name]}` : "";
  if (bar) {
    const pass = bar[1] === "above" ? value >= bar[0] : value <= bar[0];
    mark = pass ? "" : " ⚠"; tone = colors.warn;
    why = `${label} ${plain(value, digits)}${unit}, ${pass ? "within" : "past"} its limit (${bar[1]} ${plain(bar[0], digits)}${unit})${why ? `. ${MEANING[name!]}` : ""}`;
  }
  return <span title={why} style={{ whiteSpace: "nowrap", cursor: why ? "help" : undefined }}>{label} <b>{plain(value, digits)}{unit}</b>{mark && <span style={{ color: tone }}>{mark}</span>}</span>;
}

// A run's changed settings in plain words, the settings of one learned feature said once: "+ vibration path" for the
// path's rate, knot spacing and priors turned on together.
const FEATURES: [string, string][] = [["scan.vibration", "vibration path"], ["scan.line", "line offsets"], ["object.absorbing", "absorption"], ["object.diffuse", "diffuse scattering"]];
function plainChanges(items: Change[]): string {
  const said = new Set<string>(), out: string[] = [];
  for (const item of items) {
    const feature = FEATURES.find(([prefix]) => item.key.startsWith(prefix));
    if (feature) {
      if (said.has(feature[1])) continue;
      said.add(feature[1]);
      const turnedOn = items.some((one) => one.key.startsWith(feature[0]) && sig(one.before) === "none");
      out.push(`${turnedOn ? "+" : "changed"} ${feature[1]}`);
      continue;
    }
    out.push(plainChange(item));
  }
  return out.join(", ");
}

// Three significant figures for a header, a non-number as it is ("None" as "none").
function sig(value: string): string {
  if (value === "None" || value === "null") return "none";
  const number = Number(value);
  return value.trim() !== "" && Number.isFinite(number) ? String(Number(number.toPrecision(3))) : value;
}

// A setting's change in plain words: "slab 400 → 500 Å", "probe 8 → 10 modes", "scan vibration rate 0.006 (from none)".
const PLAIN_CHANGES: { [key: string]: (before: string, after: string) => string } = {
  "object.thickness_A": (before, after) => `slab ${before} → ${after} Å`,
  "object.slices": (before, after) => `slices ${before} → ${after}`,
  "probe.modes": (before, after) => `probe ${before} → ${after} modes`,
};
const UNITS: [RegExp, string][] = [[/_lr$/, " rate"], [/_A$/, " Å"], [/_px$/, " px"], [/_mrad$/, " mrad"], [/_nm$/, " nm"], [/_kV$/, " kV"]];
function plainChange(item: Change): string {
  const before = sig(item.before), after = sig(item.after);
  if (PLAIN_CHANGES[item.key]) return PLAIN_CHANGES[item.key](before, after);
  const learned = (value: string) => /tensor|array|\[|\(/.test(value) || value.length > 16;  // a learned array, not a number a person typed
  if (learned(item.after) || learned(item.before)) {
    const name = item.key.split(".").slice(1).join(" ").replace(/_(px|A|mrad|nm)$/, "").replace(/_/g, " ").replace(/ px$/, "");
    return before === "none" ? `+ ${name} learned` : after === "none" ? `− ${name}` : `${name} changed`;
  }
  const [part, ...rest] = item.key.split(".");
  let words = rest.join(" "), unit = "";
  for (const [pattern, name] of UNITS) if (pattern.test(words)) { words = words.replace(pattern, ""); unit = name === " rate" ? "" : name; if (name === " rate") words += " rate"; break; }
  words = `${part === "object" || part === "probe" || part === "scan" || part === "detector" ? `${part} ` : ""}${words.replace(/_/g, " ")}`;
  return before === "none" ? `${words} ${after}${unit} (from none)` : `${words} ${before} → ${after}${unit}`;
}

// A compact text button, contained when on (a choice or an action in a control row).
function toggleButton(on: boolean, label: string, onClick: () => void) {
  return <Button size="small" sx={{ ...compactButton, fontSize: TEXT, lineHeight: 1.4 }} variant={on ? "contained" : "text"} aria-pressed={on} onClick={onClick}>{label}</Button>;
}

// The siblings' labelled switch (Columns, Smooth, Loop, ...).
function switchControl(label: string, checked: boolean, onChange: () => void, help: string, colors: any) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: `${GAP}px` }}>
      <span style={{ ...LABEL, color: colors.textMuted }} title={help}>{label}</span>
      <Switch size="small" checked={checked} onChange={onChange} sx={switchStyles.small} slotProps={{ input: { "aria-label": label } }} />
    </Box>
  );
}

// The colormap engine for a view's images, WebGPU when the browser has a hardware adapter, else the Canvas2D twin; a
// lost device moves every image to the Canvas2D engine, which re-uploads on the next paint.
function useEngine(smooth: boolean) {
  const [engine, setEngine] = React.useState<DisplayColormapEngine | null>(null);
  const slots = React.useRef(0);
  React.useEffect(() => {
    let disposed = false, current: DisplayColormapEngine | null = null;
    createColormapEngine().then((made) => { if (disposed) made.destroy(); else { current = made; setEngine(made); } })
      .catch((error) => console.error("[ShowReconstruction] colormap engine failed to start", error));
    const stop = onGPULost(() => {
      if (disposed || !(current instanceof GPUColormapEngine)) return;
      const lost = current; current = createCPUColormapEngine(); setEngine(current); lost.destroy();
    });
    return () => { disposed = true; stop(); current?.destroy(); };
  }, []);
  return React.useMemo(() => ({ engine, slot: () => slots.current++, smooth }), [engine, smooth]);
}

// A view's panel interactions: one zoom its field images share, its contrast and colormap, and the "controls" toggle
// and row that set them (hidden by default, so the view's look stays clean).
const MAPS = ["gray", ...COLORMAP_NAMES.filter((name) => ["viridis", "magma", "inferno", "cividis"].includes(name))];
function useControls(colors: any, smooth: boolean, setSmooth: (on: boolean) => void) {
  const [display, setDisplay] = React.useState<Display>(DEFAULT_DISPLAY);
  const [open, setOpen] = React.useState(false);
  const link = useViewLink(display);
  const toggle = <ControlsToggle open={open} onToggle={() => setOpen(!open)} colors={colors} />;
  const row = open ? <ViewControls display={display} setDisplay={setDisplay} smooth={smooth} setSmooth={setSmooth} zoomed={link.zoom.scale > 1}
    onReset={() => link.setZoom(NO_ZOOM)} colors={colors} maps={MAPS} /> : null;
  return { link, toggle, row };
}

// The calm palette with the pass and warn tones, in the theme.
function usePalette() {
  const { colors: base, themeInfo } = useTheme();
  const theme = themeInfo.theme;
  const colors = React.useMemo(() => ({ ...base, good: theme === "dark" ? "#4caf50" : "#1a7f37", warn: theme === "dark" ? "#f2a900" : "#b35c00" }), [base, theme]);
  return { colors, theme };
}

// ---------------------------------------------------------------------------------------------------------------
// The run models in the browser

// One run model's data: its traits, and the checkpoints its messages add while it trains. One store a model, shared
// by every view of it (the run cell, ptycho("name"), the dashboard, compare), so a frame is held once.
type RunData = {
  title: string; pixelA: number; sliceA: number; bars: { [name: string]: [number, string] }; status: { [name: string]: any };
  views: number; viewsLabel: string; records: Record[]; earlier: Record[]; stages: Stage[]; panel: string;
  projections: Stack; volumes: Stack; probes: Stack; strongest: Stack; apertures: Stack; depthRows: number[][]; depthEpochs: number[];
  phaseRange: number[]; peaks: number[][]; line: number[]; lineGiven: number[];
  path: Float32Array; pathShape: number[]; pathStep: number; lineOffsets: number[][]; tilt: Float32Array; tiltShape: number[];
  ends: EndItem[]; bandPerA: number[];
};

// One array of a run's before or after state (ShowReconstruction.set_ends).
type EndItem = { which: string; kind: string; shape: number[]; data: Float32Array; slice: number; slices: number; depth_A?: number[] | null; length_A?: number; shares?: number[] };

const HISTORY = ["records", "projections", "volumes", "probes", "strongest", "apertures", "depth_rows"];
const SHOWN = ["title", "pixel_size_A", "slice_A", "bars", "status", "views", "views_label", "earlier", "stages", "panel", "phase_range",
  "profile_line", "profile_line_given", "peaks", "path", "path_shape", "path_step_ms", "line_offsets_A", "tilt_map", "tilt_map_shape", "ends", "ends_items", "band_per_A"];

class RunStore {
  model: any;
  listeners = new Set<() => void>();
  records: Record[] = [];
  projections: Stack = EMPTY; volumes: Stack = EMPTY; probes: Stack = EMPTY; strongest: Stack = EMPTY; apertures: Stack = EMPTY;
  depthRows: number[][] = []; depthEpochs: number[] = [];
  cached: RunData | null = null;
  pending = false;

  constructor(model: any) {
    this.model = model;
    this.readHistory();
    const reread = () => {  // a flush changes several traits at once: read them once
      if (this.pending) return;
      this.pending = true;
      queueMicrotask(() => { this.pending = false; this.readHistory(); this.bump(); });
    };
    for (const name of HISTORY) model.on(`change:${name}`, reread);
    for (const name of SHOWN) model.on(`change:${name}`, () => this.bump());
    model.on("msg:custom", (content: any, buffers: (DataView | ArrayBuffer)[]) => this.onMessage(content, buffers));
    model.send({ kind: "hello" });  // the checkpoints sent before this view existed
  }

  readHistory() {
    const get = (name: string) => this.model.get(name);
    this.records = get("records") || [];
    this.projections = stackFromTrait(get("projections"), get("projection_shapes"), get("snapshot_epochs"));
    this.volumes = stackFromTrait(get("volumes"), get("volume_shapes"), get("volume_epochs"));
    this.probes = stackFromTrait(get("probes"), get("probe_shapes"), get("probe_epochs"), get("probe_shares"));
    this.strongest = stackFromTrait(get("strongest"), get("strongest_shapes"), get("strongest_epochs"));
    // a draw whose probe is saved whole keeps its strongest mode only there (its first mode): add it back to the stack
    const missing = this.probes.epochs.map((epoch, index) => [epoch, index]).filter(([epoch]) => !this.strongest.epochs.includes(epoch));
    if (missing.length) {
      const entries = this.strongest.epochs.map((epoch, index) => ({ epoch, frame: this.strongest.frames[index], shape: this.strongest.shapes[index] }));
      for (const [epoch, index] of missing) {
        const shape = this.probes.shapes[index], size = shape.slice(1).reduce((product, length) => product * length, 1);
        entries.push({ epoch, frame: this.probes.frames[index].subarray(0, size), shape: shape.slice(1) });
      }
      entries.sort((a, b) => a.epoch - b.epoch);
      this.strongest = { epochs: entries.map((one) => one.epoch), frames: entries.map((one) => one.frame), shapes: entries.map((one) => one.shape), extras: entries.map(() => null) };
    }
    this.apertures = stackFromTrait(get("apertures"), get("aperture_shapes"), get("aperture_epochs"));
    this.depthRows = get("depth_rows") || [];
    this.depthEpochs = get("depth_epochs") || [];
  }

  onMessage(content: any, buffers: (DataView | ArrayBuffer)[]) {
    if (!content || content.kind !== "checkpoint") return;
    const epoch = Number(content.record.epoch);
    this.records = [...this.records.filter((row) => row.epoch !== epoch), content.record].sort((a, b) => a.epoch - b.epoch);
    let next = 0;
    for (const name of ["projections", "volumes", "probes", "apertures"] as const) {
      const part = content[name];
      if (!part) continue;
      const frame = floatsOf(buffers[next++]);
      this[name] = appendCheckpoint(this[name], epoch, frame, part.shape, part.epochs, part.extra);
      if (name === "probes" && content.strongest) {
        const size = part.shape.slice(1).reduce((product: number, length: number) => product * length, 1);  // mode 1: (rows, cols), or (2, rows, cols) complex
        this.strongest = appendCheckpoint(this.strongest, epoch, frame.subarray(0, size), content.strongest.shape, content.strongest.epochs);
      }
    }
    if (content.depth) {
      const kept = this.depthEpochs.map((value, index) => [value, this.depthRows[index]] as [number, number[]]).filter(([value]) => value !== epoch);
      kept.push([epoch, content.depth]);
      this.depthEpochs = kept.map(([value]) => value); this.depthRows = kept.map(([, row]) => row);
    }
    this.bump();
  }

  bump() { this.cached = null; this.listeners.forEach((listener) => listener()); }

  subscribe(listener: () => void) { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; }

  data(): RunData {
    if (this.cached) return this.cached;
    const get = (name: string) => this.model.get(name);
    this.cached = {
      title: get("title") || "", pixelA: get("pixel_size_A") || 1, sliceA: get("slice_A") || 0, bars: get("bars") || {}, status: get("status") || {},
      views: get("views") || 1, viewsLabel: get("views_label") || "", records: this.records, earlier: get("earlier") || [], stages: get("stages") || [],
      panel: get("panel") || "", projections: this.projections, volumes: this.volumes, probes: this.probes, strongest: this.strongest, apertures: this.apertures,
      depthRows: this.depthRows, depthEpochs: this.depthEpochs, phaseRange: get("phase_range") || [0, 1], peaks: get("peaks") || [],
      line: get("profile_line") || [], lineGiven: get("profile_line_given") || [], path: floats(get("path")), pathShape: get("path_shape") || [0, 0],
      pathStep: get("path_step_ms") || 0, lineOffsets: get("line_offsets_A") || [], tilt: floats(get("tilt_map")), tiltShape: get("tilt_map_shape") || [0, 0, 0],
      ends: endItems(get("ends"), get("ends_items") || []), bandPerA: get("band_per_A") || [],
    };
    return this.cached;
  }
}

// The ends' arrays as views into their one buffer.
function endItems(bytes: DataView | null | undefined, items: any[]): EndItem[] {
  const all = floats(bytes);
  let offset = 0;
  return items.map((item) => {
    const size = (item.shape || []).reduce((product: number, length: number) => product * length, 1);
    const data = all.subarray(offset, offset + size);
    offset += size;
    return { ...item, data };
  });
}

const STORES = new WeakMap<object, RunStore>();
function storeOf(model: any): RunStore {
  let store = STORES.get(model);
  if (!store) { store = new RunStore(model); STORES.set(model, store); }
  return store;
}

// The models a trait references (widget_serialization's "IPY_MODEL_<id>"), resolved through the widget manager: the
// dashboard's runs, compare's sources.
function useReferenced(trait: string): any[] {
  const model = useModel() as any;
  const [value] = useModelState<string[] | string | null>(trait);
  const refs = value == null ? [] : Array.isArray(value) ? value : [value];
  const [models, setModels] = React.useState<any[]>([]);
  const key = (refs || []).join();
  React.useEffect(() => {
    let cancelled = false;
    Promise.all((refs || []).map((ref) => model.widget_manager.get_model(String(ref).replace(/^IPY_MODEL_/, ""))))
      .then((found) => { if (!cancelled) setModels(found); })
      .catch((error) => console.error("[ShowReconstruction] a referenced run model could not be found", error));
    return () => { cancelled = true; };
  }, [model, key]);
  return key ? models : [];
}

// Each model's data, redrawn when any of them changes.
function useStores(models: any[]): RunData[] {
  const stores = React.useMemo(() => models.map(storeOf), [models]);
  const [, setVersion] = React.useState(0);
  React.useEffect(() => {
    const stops = stores.map((store) => store.subscribe(() => setVersion((value) => value + 1)));
    return () => stops.forEach((stop) => stop());
  }, [stores]);
  return stores.map((store) => store.data());
}

// Several runs as one story: records and frames in order, a later run's start (the earlier run's end) given way.
function mergeRuns(runs: RunData[]): RunData | null {
  if (!runs.length) return null;
  const last = runs[runs.length - 1];
  const withPath = [...runs].reverse().find((run) => run.pathShape[0] > 0);
  const withOffsets = [...runs].reverse().find((run) => run.lineOffsets.length > 0);
  const withTilt = [...runs].reverse().find((run) => run.tiltShape[0] > 0);
  return {
    ...last,
    records: runs.flatMap((run, index) => run.records.filter((row) => index === 0 || !row.start)).sort((a, b) => a.epoch - b.epoch),
    earlier: [], stages: runs.flatMap((run) => run.stages),
    projections: mergeStacks(runs.map((run) => run.projections)), volumes: mergeStacks(runs.map((run) => run.volumes)),
    probes: mergeStacks(runs.map((run) => run.probes)), strongest: mergeStacks(runs.map((run) => run.strongest)), apertures: mergeStacks(runs.map((run) => run.apertures)),
    depthRows: runs.flatMap((run) => run.depthRows), depthEpochs: runs.flatMap((run) => run.depthEpochs),
    path: withPath ? withPath.path : new Float32Array(0), pathShape: withPath ? withPath.pathShape : [0, 0], pathStep: withPath ? withPath.pathStep : 0,
    lineOffsets: withOffsets ? withOffsets.lineOffsets : [], tilt: withTilt ? withTilt.tilt : new Float32Array(0), tiltShape: withTilt ? withTilt.tiltShape : [0, 0, 0],
    peaks: [...runs].reverse().find((run) => run.peaks.length)?.peaks || [],
    line: [...runs].reverse().find((run) => run.line.length)?.line || [], lineGiven: [...runs].reverse().find((run) => run.lineGiven.length)?.lineGiven || [],
  };
}

// ---------------------------------------------------------------------------------------------------------------
// Shared reads

function recordAt(records: Record[], epoch: number, key?: string): Record | null {
  return records.reduce((best: Record | null, row) => (row.epoch <= epoch && (!key || row[key] != null) ? row : best), null);
}

function frameOf(stack: Stack, epoch: number) {
  const index = indexAt(stack, epoch);
  return index < 0 ? null : { data: stack.frames[index], shape: stack.shapes[index], epoch: stack.epochs[index], extra: stack.extras[index] };
}

// The display range of a field: its 1st to 99.8th percentile.
function rangeOf(data: Float32Array): [number, number] {
  if (!data.length) return [0, 1];
  const step = Math.max(1, Math.floor(data.length / 100000));
  const sample = new Float32Array(Math.ceil(data.length / step));
  for (let index = 0, out = 0; index < data.length; index += step) sample[out++] = data[index];
  sample.sort();
  return [sample[Math.floor(0.01 * sample.length)], sample[Math.floor(0.998 * (sample.length - 1))]];
}

// A label on a dark plate, readable over any image.
const PLATE: React.CSSProperties = { background: "rgba(0,0,0,0.6)", padding: "0 3px", borderRadius: 2, textShadow: "none" };

// A Gaussian of σ px over a field (separable, edges clamped): display only, the change map's smoothing.
function gaussianBlur(image: Float32Array, rows: number, cols: number, sigma: number): Float32Array {
  const radius = Math.ceil(3 * sigma), kernel: number[] = [];
  let total = 0;
  for (let offset = -radius; offset <= radius; offset++) { const weight = Math.exp(-(offset * offset) / (2 * sigma * sigma)); kernel.push(weight); total += weight; }
  for (let index = 0; index < kernel.length; index++) kernel[index] /= total;
  const across = new Float32Array(image.length), out = new Float32Array(image.length);
  for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
    let sum = 0;
    for (let offset = -radius; offset <= radius; offset++) sum += kernel[offset + radius] * image[row * cols + Math.min(cols - 1, Math.max(0, col + offset))];
    across[row * cols + col] = sum;
  }
  for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
    let sum = 0;
    for (let offset = -radius; offset <= radius; offset++) sum += kernel[offset + radius] * across[Math.min(rows - 1, Math.max(0, row + offset)) * cols + col];
    out[row * cols + col] = sum;
  }
  return out;
}

// B - A of two fields of one shape, each with its mean removed (a constant phase offset is arbitrary), and the
// symmetric span holding 99.5 % of it.
function differenceOf(after: Float32Array, before: Float32Array) {
  const length = Math.min(after.length, before.length);
  let meanAfter = 0, meanBefore = 0;
  for (let index = 0; index < length; index++) { meanAfter += after[index]; meanBefore += before[index]; }
  meanAfter /= length || 1; meanBefore /= length || 1;
  const image = new Float32Array(length);
  for (let index = 0; index < length; index++) image[index] = after[index] - meanAfter - (before[index] - meanBefore);
  const step = Math.max(1, Math.floor(length / 100000));
  const magnitudes: number[] = [];
  for (let index = 0; index < length; index += step) magnitudes.push(Math.abs(image[index]));
  magnitudes.sort((a, b) => a - b);
  return { image, span: magnitudes[Math.floor(0.995 * (magnitudes.length - 1))] || 1e-6 };
}

// A probe frame's modes as the views draw them. A complex frame is (modes, 2, rows, cols) (single: (2, rows, cols)):
// the real and imaginary parts, drawn hue phase, brightness amplitude; an intensity frame is (modes, rows, cols) (single:
// (rows, cols)), drawn as its amplitude, the square root, in magma.
type ModeImage = { re: Float32Array | null; im: Float32Array | null; amplitude: Float32Array; high: number; rows: number; cols: number };
function probeModes(frame: Float32Array | undefined, shape: number[] | undefined, single = false): ModeImage[] {
  if (!frame || !shape || !shape.length) return [];
  const complex = shape.length === (single ? 3 : 4);
  const full = single ? [1, ...shape] : shape;
  const [modes, rows, cols] = complex ? [full[0], full[2], full[3]] : [full[0], full[1], full[2]];
  if (!rows || !cols) return [];
  return Array.from({ length: modes }, (_, mode) => {
    if (complex) {
      const offset = mode * 2 * rows * cols;
      const re = frame.subarray(offset, offset + rows * cols), im = frame.subarray(offset + rows * cols, offset + 2 * rows * cols);
      const amplitude = new Float32Array(rows * cols);
      let high = 0; for (let index = 0; index < amplitude.length; index++) { amplitude[index] = Math.hypot(re[index], im[index]); high = Math.max(high, amplitude[index]); }
      return { re, im, amplitude, high: high || 1, rows, cols };
    }
    const amplitude = Float32Array.from(frame.subarray(mode * rows * cols, (mode + 1) * rows * cols), (value) => Math.sqrt(Math.max(0, value)));
    let high = 0; for (const value of amplitude) high = Math.max(high, value);
    return { re: null, im: null, amplitude, high: high || 1, rows, cols };
  });
}

// One mode at size px tall, true proportions: complex-wave colors when the phase is kept, else its amplitude.
function ModeTile({ mode, size, label, outline }: { mode: ModeImage; size: number; label?: string; outline?: string }) {
  const width = Math.round(size * mode.cols / mode.rows);
  return (
    <div style={{ position: "relative", width, height: size, flex: "none" }}>
      {mode.re && mode.im ? <ComplexCanvas re={mode.re} im={mode.im} rows={mode.rows} cols={mode.cols} width={width} height={size} />
        : <ImageCanvas data={mode.amplitude} rows={mode.rows} cols={mode.cols} low={0} high={mode.high} cmap="magma" height={size} keepAspect />}
      {outline && <div style={{ position: "absolute", inset: 0, border: `2px solid ${outline}`, pointerEvents: "none" }} />}
      {label && <div style={{ ...ON_IMAGE, left: 2, bottom: 0, fontSize: size >= 80 ? TEXT : 10 }}>{label}</div>}
    </div>
  );
}

// The probe as a panel, sized with the object: its strongest mode at size (complex-wave colors, its share), its
// Fourier-space amplitude (the aperture) at size beside it, and the other modes in three rows at a third of size.
function ProbeView({ frame, shape, shares, aperture, size: wanted, maxWidth, label, colors, box }: {
  frame?: Float32Array; shape?: number[]; shares?: number[]; aperture?: { data: Float32Array; shape: number[] } | null; size: number; maxWidth?: number; label?: string; colors: any;
  box?: [number, number];  // fill this width × height: one mode as large as it allows (with the aperture beside it when the box is wide), every mode a thumbnail under it; hover a thumbnail to see it large
}) {
  const modes = React.useMemo(() => probeModes(frame, shape), [frame, shape?.join()]);
  const [looking, setLooking] = React.useState<number | null>(null);
  const across = (aperture ? 2 : 1) + Math.ceil(Math.max(0, modes.length - 1) / 3) / 3;  // widths of size the panel spans
  const size = maxWidth ? Math.max(40, Math.min(wanted, Math.floor((maxWidth - 6) / across))) : wanted;
  const apertureHigh = React.useMemo(() => { let high = 0; if (aperture) for (const value of aperture.data) high = Math.max(high, value); return high || 1; }, [aperture?.data]);
  if (!modes.length) return <div style={{ height: size, fontSize: TEXT, color: colors.textMuted }}>{label ? `${label}: ` : ""}no probe kept</div>;
  const share = (index: number) => (shares && shares[index] != null ? (shares[index] < 0.001 ? "< 0.1 %" : `${(100 * shares[index]).toFixed(shares[index] < 0.1 ? 1 : 0)} %`) : "");
  const small = Math.max(16, Math.floor((size - 2 * HAIR) / 3));
  const others = modes.slice(1);
  const looked = looking != null && looking < modes.length ? looking : 0;  // hover looks: the hovered mode takes the big tile, leaving goes back to the strongest
  const apertureTile = (side: number) => aperture && aperture.shape.length === 2 ? (
    <div style={{ position: "relative", width: Math.round(side * aperture.shape[1] / aperture.shape[0]), height: side, flex: "none" }}>
      <ImageCanvas data={aperture.data} rows={aperture.shape[0]} cols={aperture.shape[1]} low={0} high={apertureHigh} cmap="gray" height={side} keepAspect />
      {side >= 60 && <div style={{ ...ON_IMAGE, left: 2, bottom: 0, fontSize: 10 }}>aperture, |F ψ|</div>}
    </div>) : null;
  if (box && box[0] >= 60 && box[1] >= 60) {
    const [width, height] = box;
    const pair = !!aperture && width >= 1.5 * height;  // room for the aperture beside the mode at full height
    const thumbs = modes.length + (aperture && !pair ? 1 : 0);
    const thumb = Math.max(20, Math.min(56, Math.floor((width + HAIR) / Math.min(thumbs, 8)) - HAIR));
    const perRow = Math.max(1, Math.floor((width + HAIR) / (thumb + HAIR)));
    const strip = Math.ceil(thumbs / perRow) * (thumb + HAIR);
    const large = Math.max(40, Math.min(pair ? Math.floor((width - HAIR) / 2) : width, height - strip));
    const showAperture = !pair && looking === -1;
    let side = 0;  // the thumbnails' size when they stand beside a full-height tile, 0 when they go under it
    if (!pair && width - height - HAIR >= 2 * (32 + HAIR)) for (let t = 72; t >= 28 && !side; t -= 4) {
      if (Math.floor((width - height) / (t + HAIR)) * Math.floor((height + HAIR) / (t + HAIR)) >= thumbs) side = t;
    }
    const thumbTiles = (t: number) => <>
      {modes.map((mode, index) => <div key={index} onMouseEnter={() => setLooking(index)} title={`mode ${index + 1} · ${share(index)}`} style={{ cursor: "zoom-in" }}>
        <ModeTile mode={mode} size={t} outline={!showAperture && looked === index ? colors.accent : undefined} label={t >= 40 ? `${index + 1} · ${share(index)}` : `${index + 1}`} /></div>)}
      {aperture && !pair && <div onMouseEnter={() => setLooking(-1)} title="the aperture: the probe's Fourier-space amplitude" style={{ cursor: "zoom-in", outline: showAperture ? `2px solid ${colors.accent}` : "none", outlineOffset: -2 }}>{apertureTile(t)}</div>}
    </>;
    if (side) return (
      <div style={{ display: "flex", gap: HAIR, width, flex: "none" }} onMouseLeave={() => setLooking(null)}>
        {showAperture ? apertureTile(height) : <ModeTile mode={modes[looked]} size={height} label={`${label ? `${label} · ` : ""}mode ${looked + 1} · ${share(looked)}`} />}
        <div style={{ display: "flex", flexWrap: "wrap", alignContent: "flex-start", gap: HAIR, flex: "1 1 0", minWidth: 0 }}>{thumbTiles(side)}</div>
      </div>
    );
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: HAIR, width, flex: "none" }} onMouseLeave={() => setLooking(null)}>
        <div style={{ display: "flex", gap: HAIR }}>
          {showAperture ? apertureTile(large) : <ModeTile mode={modes[looked]} size={large} label={`${label ? `${label} · ` : ""}mode ${looked + 1} · ${share(looked)}`} />}
          {pair && apertureTile(large)}
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: HAIR }}>{thumbTiles(thumb)}</div>
      </div>
    );
  }
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: HAIR, alignItems: "flex-start", flex: "0 1 auto", minWidth: 0 }} onMouseLeave={() => setLooking(null)}>
      <ModeTile mode={modes[looked]} size={size} label={`${label ? `${label} · ` : ""}mode ${looked + 1} · ${share(looked)}`} />
      {apertureTile(size)}
      {others.length > 0 && <div style={{ display: "grid", gridAutoFlow: "column", gridTemplateRows: `repeat(${Math.min(3, others.length)}, ${small}px)`, gap: HAIR }}>
        {others.map((mode, index) => <div key={index} onMouseEnter={() => setLooking(index + 1)} title={`mode ${index + 2} · ${share(index + 1)}: hover to see it large`} style={{ cursor: "zoom-in" }}><ModeTile mode={mode} size={small} outline={looked === index + 1 ? colors.accent : undefined} label={small >= 72 ? `mode ${index + 2} · ${share(index + 1)}` : small >= 50 ? `#${index + 2} · ${share(index + 1)}` : `#${index + 2}`} /></div>)}
      </div>}
    </div>
  );
}

function header(colors: any): React.CSSProperties {
  return { fontFamily: WIDGET_UI_FONT, fontSize: 12, color: colors.text, background: colors.bg, border: `1px solid ${colors.border}`,
    padding: GAP, borderRadius: 4, maxWidth: "100%", boxSizing: "border-box", fontVariantNumeric: "tabular-nums", outline: "none" };
}

// ---------------------------------------------------------------------------------------------------------------
// The run view: what this run did to the object and the probe


// One state of the run's ends as the run view compares them: its slices (a few depths), the cut along the line, and
// for a first run's before, its projection and probe.
function endsOf(data: RunData | null, which: string) {
  const items = data ? data.ends.filter((item) => item.which === which) : [];
  return { slices: items.filter((item) => item.kind === "slice"), cut: items.find((item) => item.kind === "cut") || null,
    projection: items.find((item) => item.kind === "projection") || null, probe: items.find((item) => item.kind === "probe") || null,
    aperture: items.find((item) => item.kind === "aperture") || null };
}

// The run view: what this line did, before (the run's start, the end of the run it continued from) against after (its
// end, or the epoch the filmstrip shows), on one display range: the object (projection or a slice), its spectrum, the
// change, the probe's modes, and the panel of what the run changed. Both states are drawn at once, so a saved,
// non-interactive notebook shows them.
function RunView() {
  const { colors, theme } = usePalette();
  const model = useModel();
  const models = React.useMemo(() => [model], [model]);
  const data = useStores(models)[0];
  const parent = useStores(useReferenced("parent"))[0] || null;
  const [sliderEpoch, setSliderEpoch] = useModelState<number>("slider_epoch");
  const [smooth, setSmooth] = useModelState<boolean>("smooth");
  const [savedLine, setSavedLine] = useModelState<number[]>("profile_line");
  const [sliceIndex, setSliceIndex] = useModelState<number>("slice_index");
  const [shown, setShown] = useModelState<string>("shown");
  const [size, setSize] = useModelState<ViewSize>("size");
  const controls = useControls(colors, !!smooth, setSmooth);
  const [root, rootWidth] = useWidth(1000);
  const SIDE = Math.min(bySize(size, 110, 200, 360), Math.max(90, Math.floor((rootWidth - 2 * HAIR - 14) / 3)));  // the before, after and change images, px tall; narrower on a phone
  const STRIP = bySize(size, 22, 30, 40);  // the filmstrip's thumbnails
  const PROBE = Math.round(0.6 * SIDE);  // the probe's strongest mode and aperture, sized with the object
  const CURVE = bySize(size, 44, 52, 60);  // the objective: a thin strip, never a big panel
  const engineContext = useEngine(!!smooth);
  const [more, setMore] = React.useState(false);  // browser only: never saved open
  const [preview, setPreview] = React.useState<number | null>(null);
  const stage = data.stages[0];
  const start = stage ? stage.start : 0;
  const records = data.records;
  const last = records.length ? records[records.length - 1] : ({ epoch: start } as Record);
  const positions = data.projections.epochs.length ? data.projections.epochs : records.map((row) => row.epoch);
  const committed = sliderEpoch >= 0 ? Math.min(sliderEpoch, last.epoch) : last.epoch;
  const epoch = preview ?? committed;
  const previewing = preview != null && preview !== committed;
  const setEpoch = (value: number) => { setPreview(null); setSliderEpoch(value >= last.epoch ? -1 : value); };
  const step = (direction: number) => {
    const index = Math.max(0, positions.reduce((best, value, at) => (value <= committed ? at : best), 0));
    const next = positions[Math.max(0, Math.min(positions.length - 1, index + direction))];
    if (next != null) setEpoch(next);
  };
  const runEpochs = Number(data.status.epochs) || Math.max(0, last.epoch - start);
  const after = endsOf(data, "after");
  const before = parent ? endsOf(parent, "after") : endsOf(data, "before");
  // the object, before and after: the projection (after at the filmstrip's epoch), or a slice (the run's end)
  const slicing = shown === "slice" && after.slices.length > 0;
  const chosen = slicing ? after.slices[sliceIndex >= 0 && sliceIndex < after.slices.length ? sliceIndex : Math.floor(after.slices.length / 2)] : null;
  const fraction = chosen ? (chosen.slice + 0.5) / chosen.slices : 0;
  const beforeSlice = chosen && before.slices.length ? before.slices.reduce((best, item) => (Math.abs((item.slice + 0.5) / item.slices - fraction) < Math.abs((best.slice + 0.5) / best.slices - fraction) ? item : best)) : null;
  const parentEnd = parent && parent.projections.frames.length ? { data: parent.projections.frames[parent.projections.frames.length - 1], shape: parent.projections.shapes[parent.projections.shapes.length - 1] } : null;
  const ownStart = records.length && records[0].start && data.projections.epochs[0] === records[0].epoch ? { data: data.projections.frames[0], shape: data.projections.shapes[0] } : null;
  const startFrame = parentEnd || ownStart;  // a run's start is its parent's end, stored once in the parent
  const afterProjection = frameOf(data.projections, epoch);
  const beforeImage = slicing ? (beforeSlice ? { data: beforeSlice.data, shape: beforeSlice.shape } : null)
    : startFrame || (before.projection ? { data: before.projection.data, shape: before.projection.shape } : null);
  const afterImage = slicing ? { data: chosen!.data, shape: chosen!.shape } : afterProjection ? { data: afterProjection.data, shape: afterProjection.shape } : null;
  const pair = React.useMemo(() => [beforeImage, afterImage].filter(Boolean).map((image) => image!.data), [beforeImage?.data, afterImage?.data]);  // before and after: one range, also once the contrast is changed
  const range = React.useMemo(() => {  // one display range for both states
    const ranges = [beforeImage, afterImage].filter(Boolean).map((image) => rangeOf(image!.data));
    return ranges.length ? [Math.min(...ranges.map((one) => one[0])), Math.max(...ranges.map((one) => one[1]))] : [0, 1];
  }, [beforeImage?.data, afterImage?.data]);
  const sliceWidth = (item: EndItem | null) => (item && item.depth_A ? item.depth_A[1] - item.depth_A[0] : null);
  const unequalSlices = slicing && beforeSlice && chosen && sliceWidth(beforeSlice) != null && sliceWidth(chosen) != null && Math.abs(sliceWidth(beforeSlice)! - sliceWidth(chosen)!) > 0.01 * sliceWidth(chosen)!
    ? `${plain(sliceWidth(beforeSlice)!, 1)} and ${plain(sliceWidth(chosen)!, 1)} Å` : "";  // a reslice: the two slices hold different thicknesses
  const [rawChange, setRawChange] = React.useState(false);  // the change is drawn smoothed over 2 px by default, said on the image; a click shows it raw
  const moved = React.useMemo(() => {
    if (!beforeImage || !afterImage || beforeImage.shape.join() !== afterImage.shape.join()) return null;
    const [height, width] = afterImage.shape;
    const smoothed = (image: Float32Array) => (rawChange ? image : gaussianBlur(image, height, width, 2));
    return differenceOf(smoothed(afterImage.data), smoothed(beforeImage.data));
  }, [beforeImage?.data, afterImage?.data, rawChange]);
  const [rows, cols] = afterImage ? afterImage.shape : [0, 0];
  const beforeSpectrum = useSpectrum(beforeImage ? beforeImage.data : EMPTY_FRAME, beforeImage ? beforeImage.shape[0] : 0, beforeImage ? beforeImage.shape[1] : 0, null, data.pixelA);
  const afterSpectrum = useSpectrum(afterImage ? afterImage.data : EMPTY_FRAME, rows, cols, null, data.pixelA);
  const spectrumRange = beforeSpectrum && afterSpectrum ? [Math.min(beforeSpectrum.low, afterSpectrum.low), Math.max(beforeSpectrum.high, afterSpectrum.high)] : null;
  // the probe, before and after
  const parentProbe = parent && parent.probes.frames.length ? { data: parent.probes.frames[parent.probes.frames.length - 1], shape: parent.probes.shapes[parent.probes.shapes.length - 1],
    shares: (parent.probes.extras[parent.probes.extras.length - 1] as number[]) || [] } : null;
  const beforeProbe = parentProbe || (before.probe ? { data: before.probe.data, shape: before.probe.shape, shares: before.probe.shares || [] } : null);
  const afterFrame = frameOf(data.probes, epoch);
  const afterProbe = afterFrame ? { data: afterFrame.data, shape: afterFrame.shape, shares: (afterFrame.extra as number[]) || [] } : null;
  const parentAperture = parent && parent.apertures.frames.length ? { data: parent.apertures.frames[parent.apertures.frames.length - 1], shape: parent.apertures.shapes[parent.apertures.shapes.length - 1] } : null;
  const beforeAperture = parentAperture || (before.aperture ? { data: before.aperture.data, shape: before.aperture.shape } : null);
  const afterApertureFrame = frameOf(data.apertures, epoch);
  const afterAperture = afterApertureFrame ? { data: afterApertureFrame.data, shape: afterApertureFrame.shape } : null;
  const scored = records.filter((row) => row.held_out != null);
  const startScore = records.length && records[0].start ? records[0].held_out : null;
  const now = recordAt(records, epoch, "held_out");
  const nowScore = now && !(now.start && scored.length > 1 && now.epoch < epoch) ? now.held_out : null;
  const delta = startScore != null && nowScore != null && now !== records[0] ? nowScore - startScore : null;
  const atEpoch = recordAt(records, epoch, "r99_A") || ({} as Record);
  const imageStart = records.length && records[0].start && records[0].image_score != null ? Number(records[0].image_score) : null;  // held-out says the patterns fit; the image score says the picture got better
  const imageRecord = recordAt(records, epoch, "image_score");
  const imageNow = imageRecord && !(imageRecord.start && imageRecord.epoch < epoch) ? Number(imageRecord.image_score) : null;
  const imageDelta = imageStart != null && imageNow != null && imageRecord !== records[0] ? imageNow - imageStart : null;
  const finalDelta = startScore != null && scored.length && scored[scored.length - 1] !== records[0] ? scored[scored.length - 1].held_out - startScore : null;
  const verdict = stage && stage.refused ? "refused" : finalDelta == null ? "" : finalDelta > 0 ? "worse than its start" : "kept";  // what became of the run, in plain words
  const muted = { color: colors.textMuted } as React.CSSProperties;
  const changes = stage ? stageChanges(stage) : [];
  const changesText = plainChanges(changes);
  const atText = afterProjection && !slicing ? (afterProjection.epoch === last.epoch ? "after" : `epoch ${afterProjection.epoch - start}`) : "after";
  const depthText = (item: EndItem | null) => item ? (item.depth_A ? `${plain(item.depth_A[0], 0)}-${plain(item.depth_A[1], 0)} Å` : `slice ${item.slice + 1}/${item.slices}`) : "";
  const onKey = (event: React.KeyboardEvent) => {
    if ((event.target as HTMLElement).closest(".MuiSlider-root")) return;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") { step(event.key === "ArrowLeft" ? -1 : 1); event.preventDefault(); }
  };
  const runSeries = (key: string, name: string, color: string, dashed?: boolean): Series => ({ name, color, dashed,
    points: records.filter((row) => row[key] != null).map((row) => [row.epoch, Number(row[key])] as [number, number]) });
  const image = (picture: { data: Float32Array; shape: number[] } | null, label: string, extra: string | undefined, spectrum: Spectrum | null) => (
    <div style={{ position: "relative", flex: "none", height: SIDE, width: picture ? Math.round(SIDE * picture.shape[1] / picture.shape[0]) : SIDE, background: colors.bgAlt }}>
      {picture ? <ImageCanvas data={picture.data} rows={picture.shape[0]} cols={picture.shape[1]} low={range[0]} high={range[1]} cmap="gray" height={SIDE} keepAspect
        pixelA={data.pixelA} barA={label === "before" ? 10 : undefined} zoomable rangeFrom={pair} inset={<SpectrumInset spectrum={spectrum} side={SIDE} label={label} range={spectrumRange} />} /> : <div style={{ ...muted, fontSize: TEXT, padding: 4 }}>no {label} state</div>}
      <div style={{ ...ON_IMAGE, left: 3, top: 1 }}>{label}{extra ? ` · ${extra}` : ""}</div>
    </div>
  );
  const third = (height: number) => {
    const panel = data.panel;
    if (panel === "path" && (data.pathShape[0] > 0 || data.lineOffsets.length)) {
      const pathOf = (run: RunData | null, dashed: boolean, color?: string): Series[] => {
        if (!run || !run.pathShape[0]) return [];
        const [lines, knots] = run.pathShape;
        return Array.from({ length: Math.min(3, lines) }, (_, index) => ({ name: `${dashed ? "before" : "after"} line ${index + 1}`, color: color || HUES[index], dashed,
          points: Array.from({ length: knots }, (_, knot) => [knot * (run.pathStep || 1), run.path[index * knots + knot]] as [number, number]) }));
      };
      const offsetsOf = (run: RunData | null, dashed: boolean): Series[] => (run && run.lineOffsets.length ? [0, 1].map((axis) => ({ name: `${dashed ? "before" : "after"} ${axis ? "col" : "row"}`,
        color: dashed ? colors.textMuted : axis ? HUES[2] : HUES[0], dashed, points: run.lineOffsets.map((pair, line) => [line, pair[axis]] as [number, number]) })) : []);
      const path = [...pathOf(parent, true, colors.textMuted), ...pathOf(data, false)];
      const offsets = [...offsetsOf(parent, true), ...offsetsOf(data, false)];
      const both = path.length && offsets.length;
      const key = [{ name: "before", color: colors.textMuted, dashed: true, points: [] }, { name: "after", color: HUES[0], points: [] }];
      return <>
        {path.length > 0 && <LineChart series={path} keyItems={key} stages={[]} cursor={null} height={both ? Math.floor(height / 2) : height} outer xLabel={data.pathStep ? "ms along the line" : "knot along the line"} yLabel="path Å" colors={colors} />}
        {offsets.length > 0 && <LineChart series={offsets} keyItems={key} stages={[]} cursor={null} height={both ? Math.ceil(height / 2) : height} outer xLabel="scan line" yLabel="line offset Å" colors={colors} />}
      </>;
    }
    if (panel === "shares") {
      const startShares = beforeProbe ? beforeProbe.shares : [];
      const count = Math.max(startShares.length, ...records.map((row) => (Array.isArray(row.probe_shares) ? row.probe_shares.length : 0)));
      const old = startShares.length || count;
      const series: Series[] = Array.from({ length: count }, (_, mode) => ({ name: `mode ${mode + 1}`, color: mode >= old ? HUES[2] : mode === 0 ? HUES[0] : HUES[1], dashed: mode >= old,
        points: [...(startShares[mode] != null ? [[0, 100 * startShares[mode]] as [number, number]] : []),
          ...records.filter((row) => Array.isArray(row.probe_shares) && row.probe_shares[mode] != null).map((row) => [row.epoch - start, 100 * row.probe_shares[mode]] as [number, number])] }));
      return <LineChart series={series} keyItems={[{ name: "mode 1", color: HUES[0], points: [] }, { name: "modes 2+", color: HUES[1], points: [] },
        ...(count > old ? [{ name: "new modes", color: HUES[2], dashed: true, points: [] }] : [])]} stages={[]} cursor={epoch - start} height={height} outer log
        xLabel="epoch of the run (0: before)" yLabel="share %" colors={colors} onPick={(value) => setEpoch(value + start)} snapX={positions.map((value) => value - start)} />;
    }
    if (panel === "depth" && (before.cut || after.cut)) {
      const cuts = [["before", before.cut], ["after", after.cut]] as [string, EndItem | null][];
      const low = data.phaseRange[0], high = data.phaseRange[1];
      return <div style={{ display: "flex", flexDirection: "column", gap: HAIR }}>
        {cuts.map(([label, cut]) => <div key={label} style={{ position: "relative", height: Math.floor((height - 2) / 2), background: colors.bgAlt }}>
          {cut && <ImageCanvas data={cut.data} rows={cut.shape[0]} cols={cut.shape[1]} low={low} high={high} cmap="gray" height={Math.floor((height - 2) / 2)} />}
          <div style={{ ...ON_IMAGE, left: 3, top: 1 }}>{label}: {cut ? `${cut.shape[0]} slices, beam down` : "none kept"}</div>
          {cut && cut.length_A && <div style={{ ...ON_IMAGE, right: 3, bottom: 1 }}>{plain(cut.length_A, 0)} Å along the line</div>}
        </div>)}
      </div>;
    }
    if (panel === "tilt") {
      const scans = Array.isArray(last.tilt_mrad) ? last.tilt_mrad.length : 0;
      const series: Series[] = [];
      for (let scan = 0; scan < scans; scan++) for (const axis of [0, 1]) series.push({ name: `scan ${scan + 1} ${axis ? "col" : "row"}`, color: HUES[scan % HUES.length], dashed: axis === 1,
        points: records.filter((row) => Array.isArray(row.tilt_mrad) && row.tilt_mrad[scan]).map((row) => [row.epoch - start, Number(row.tilt_mrad[scan][axis])] as [number, number]) });
      return <LineChart series={series} stages={[]} cursor={epoch - start} height={height} outer xLabel="epoch of the run" yLabel="tilt mrad" colors={colors} />;
    }
    return null;
  };
  const earlier = { name: "earlier runs", color: colors.textMuted, dashed: true, points: data.earlier.filter((row) => row.held_out != null).map((row) => [row.epoch, Number(row.held_out)] as [number, number]) };
  const objective = <LineChart series={[earlier, runSeries("train", "training", HUES[1], true), runSeries("held_out", "held-out", HUES[0])]}
    keyItems={[{ name: "held-out", color: HUES[0], points: [] }, { name: "training", color: HUES[1], dashed: true, points: [] }, { name: "earlier runs", color: colors.textMuted, dashed: true, points: [] }]}
    stages={stage ? [stage] : []} cursor={epoch} height={CURVE + 15} outer log xLabel="" yLabel="" colors={colors} onPick={setEpoch} snapX={positions} />;
  const hasThird = third(SIDE) !== null;
  const panelTitle = { path: "the scan's path, before and after", shares: "each mode's power", depth: "cut along the line, before and after", tilt: "tilt" }[data.panel] || "";
  const stripRange = React.useMemo(() => rangeOf(data.projections.frames.length ? data.projections.frames[data.projections.frames.length - 1] : EMPTY_FRAME), [data.projections]);
  const thumbnail = (value: number, index: number) => {
    const frame = data.projections.frames[index], shape = data.projections.shapes[index];
    const score = recordAt(records, value, "held_out");
    return (
      <div key={value} title={`epoch ${value - start} of the run${score && score.epoch === value ? ` · held-out ${plain(score.held_out, 5)}` : ""}`}
        onMouseEnter={() => setPreview(value)} onClick={() => setEpoch(value)}
        style={{ position: "relative", height: STRIP, flex: "none", cursor: "pointer", outline: value === committed ? `2px solid ${colors.accent}` : value === epoch ? `1px dashed ${colors.textMuted}` : "none", outlineOffset: -2 }}>
        <ImageCanvas data={frame} rows={shape[0]} cols={shape[1]} low={stripRange[0]} high={stripRange[1]} cmap="gray" height={STRIP} keepAspect />
        <div style={{ ...ON_IMAGE, left: 2, top: 0, fontSize: 9, lineHeight: "10px" }}>{value - start}</div>
      </div>
    );
  };
  const objectWidth = 3 * (afterImage ? Math.round(SIDE * cols / rows) : SIDE) + 2 * HAIR + GAP;
  return (
    <Engine.Provider value={engineContext}><ViewLink.Provider value={controls.link}>
    <div ref={root} tabIndex={0} onKeyDown={onKey} style={header(colors)}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: `0 ${3 * GAP}px`, lineHeight: "18px", marginBottom: 2 }}>
        <span style={{ fontWeight: 600, fontSize: 13, whiteSpace: "nowrap" }}>"{stage ? stage.name : "run"}"{verdict && <span style={{ fontWeight: 600, color: verdict === "kept" ? colors.good : colors.warn }}> {verdict}</span>} · epoch {epoch - start}/{runEpochs}
          <span style={{ fontWeight: 400, ...muted }}> ({epoch} of {start + runEpochs})</span></span>
        {data.status.seconds_per_epoch != null && data.status.left_s > 0.5 && <span style={{ ...muted, whiteSpace: "nowrap" }}>{plain(data.status.seconds_per_epoch, 1)} s/epoch{data.status.left_s > 0.5 ? ` · ${seconds(data.status.left_s)} left` : ""}</span>}
        <span style={{ whiteSpace: "nowrap" }}>held-out {plain(startScore, pairDigits(startScore, nowScore))} → <b>{plain(nowScore, pairDigits(startScore, nowScore))}</b>{delta != null && <span style={{ color: delta <= 0 ? colors.good : colors.warn }}> {change(delta)}</span>}</span>
        {imageNow != null && <span style={{ whiteSpace: "nowrap" }} title="image score: repetition × detail of the projected phase over the field's central 80 %, higher is better">image {imageStart != null && imageDelta != null ? <>{plain(imageStart, 3)} → </> : null}<b>{plain(imageNow, 3)}</b>
          {imageDelta != null && Math.abs(imageDelta) >= 0.0005 && <span style={{ color: imageDelta >= 0 ? colors.good : colors.warn }}> {imageDelta >= 0 ? "▲" : "▼"}{Math.abs(imageDelta).toFixed(3)}</span>}</span>}
        <Health label="probe radius" name="r99_A" value={atEpoch.r99_A} unit=" Å" bar={data.bars?.r99_A} colors={colors} digits={1} />
        <Health label="probe copies" name="copies_percent" value={atEpoch.copies_percent} unit=" %" bar={data.bars?.copies_percent} colors={colors} digits={1} />
        {changesText && <span title={changesText} style={{ ...muted, flex: "1 1 120px", minWidth: 0, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{changesText}</span>}
        {previewing && <span style={muted}>preview</span>}
        <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: GAP }}>
          <RenderPathBadge colors={colors} />
          {controls.toggle}<SizeControl size={size} onSize={setSize} colors={colors} />
          <InfoTooltip theme={theme} toggleOnClick text={"What this run changed. Before: the run's start, the end of the run it continued from; after: its end, or the epoch the filmstrip shows (a slice is kept at the run's end). Both on one display range, the change after − before beside them (red up, blue down, means removed), their spectra (the dashed ring at 50 pm), the probe's modes with their power shares, and the panel of what the run changed. Projection or Slice picks the object; a slice is kept at a quarter, half and three quarters through the slab. Filmstrip: the projection at every epoch; hover to look, click to keep, leave to go back; arrow keys step. S M L: the view's size, and the size every later view opens at. A saved notebook keeps, for each run: its projection at every epoch (the newest run) or at its draws (earlier runs), its probe's modes at its end and its strongest mode at every draw, and slices with the cut along the line in place of the whole volume (three, a quarter, half and three quarters through the slab, for the newest run; the middle one for earlier runs); the running kernel keeps everything."} />
          {toggleButton(more, "more", () => setMore(!more))}
        </span>
      </div>
      {controls.row}
      <div style={{ display: "flex", flexWrap: rootWidth < objectWidth + 260 ? "wrap" : "nowrap", gap: GAP, alignItems: "flex-start" }}>
        <div style={{ flex: "0 1 auto", minWidth: 0, maxWidth: "100%" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 2, height: 18, fontSize: TEXT }}>
            {toggleButton(!slicing, "Projection", () => setShown("projection"))}
            {after.slices.length > 0 && toggleButton(slicing, "Slice", () => setShown("slice"))}
            {slicing && after.slices.map((item, index) => <span key={index}>{toggleButton(item === chosen, depthText(item), () => setSliceIndex(index))}</span>)}
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: HAIR }}>
            {image(beforeImage, "before", slicing ? depthText(beforeSlice) : undefined, beforeSpectrum)}
            {image(afterImage, atText, slicing ? depthText(chosen) : undefined, afterSpectrum)}
            <div style={{ position: "relative", flex: "none", height: SIDE, width: afterImage ? Math.round(SIDE * cols / rows) : SIDE, background: colors.bgAlt, cursor: "pointer" }}
              title={rawChange ? "the change pixel by pixel: click to smooth it over 2 px" : "the change smoothed over 2 px (a Gaussian of σ 2 px on both states, display only): click for the raw pixels"}
              onClick={() => setRawChange(!rawChange)}>
              {moved && <ImageCanvas data={moved.image} rows={rows} cols={cols} low={-moved.span} high={moved.span} cmap="RdBu_r" height={SIDE} keepAspect zoomable symmetric pixelA={data.pixelA} />}
              <div style={{ ...ON_IMAGE, ...PLATE, left: 3, top: 1 }}>{SIDE >= 150 ? `after − before${rawChange ? "" : ", smoothed 2 px"}` : "change"}</div>
              {unequalSlices && <div style={{ ...ON_IMAGE, ...PLATE, left: 3, top: 16, color: "#f2b04e" }} title="the run changed the slicing: a slice's phase grows with its thickness, so this map mixes that with what the run changed">slices of {unequalSlices}</div>}
              {moved && <div style={{ ...ON_IMAGE, ...PLATE, left: 3, bottom: 1 }}>±{plain(moved.span, 2)} rad</div>}
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: GAP, marginTop: HAIR, maxWidth: "100%", overflowX: "auto" }}>
          <IconButton size="small" sx={{ color: colors.textMuted, p: 0.25 }} aria-label="Previous epoch" title="Previous epoch (←)" onClick={() => step(-1)}><NavigateBeforeIcon sx={{ fontSize: 18 }} /></IconButton>
          <IconButton size="small" sx={{ color: colors.textMuted, p: 0.25 }} aria-label="Next epoch" title="Next epoch (→)" onClick={() => step(1)}><NavigateNextIcon sx={{ fontSize: 18 }} /></IconButton>
          <span style={{ fontSize: TEXT, color: colors.textMuted }}>epochs</span>
          <div style={{ display: "flex", gap: 0 }} onMouseLeave={() => setPreview(null)}>{data.projections.epochs.map(thumbnail)}</div>
        </div>
        </div>
        {hasThird && <div style={{ flex: "1 1 240px", minWidth: 220 }}>
          <div style={{ fontSize: TEXT, lineHeight: "18px", height: 18, ...muted, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{panelTitle}</div>
          <Fill min={SIDE}>{(_, height) => third(height)}</Fill>
        </div>}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: `${GAP}px ${3 * GAP}px`, marginTop: GAP }}>
        <ProbeView frame={beforeProbe?.data} shape={beforeProbe?.shape} shares={beforeProbe?.shares} aperture={beforeAperture} size={PROBE} label="before" colors={colors} />
        <ProbeView frame={afterProbe?.data} shape={afterProbe?.shape} shares={afterProbe?.shares} aperture={afterAperture} size={PROBE}
          label={afterFrame && afterFrame.epoch !== last.epoch ? `epoch ${afterFrame.epoch - start}` : "after"} colors={colors} />
      <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", flex: "1 1 220px", minWidth: 0 }}>
        <div style={{ minWidth: 200 }} title="held-out (solid) and training (dashed) of this run, earlier runs faint, log scale">{objective}</div>
      </div>
      </div>
      {more && <DetailGrid data={data} epoch={epoch} previewing={previewing} sliceIndex={-1} setSliceIndex={() => undefined} savedLine={savedLine} setSavedLine={setSavedLine}
        smooth={!!smooth} setSmooth={setSmooth} colors={colors} theme={theme} width={rootWidth} size={size} />}
    </div>
    </ViewLink.Provider></Engine.Provider>
  );
}

// ---------------------------------------------------------------------------------------------------------------
// The object, depth and probe panels at an epoch: the dashboard's lower row, and a run view's "more"

function DetailGrid({ data, epoch, previewing, sliceIndex, setSliceIndex, savedLine, setSavedLine, smooth, setSmooth, colors, theme, width, size }: {
  data: RunData; epoch: number; previewing: boolean; sliceIndex: number; setSliceIndex: (value: number) => void; savedLine: number[]; setSavedLine: (line: number[]) => void;
  smooth: boolean; setSmooth: (value: boolean) => void; colors: any; theme: "light" | "dark"; width: number; size?: string;
}) {
  const volume = frameOf(data.volumes, epoch);
  const keptList = volume ? [] : endsOf(data, "after").slices;  // a saved notebook keeps three slices and the cut, not the volume
  const keptCut = volume ? null : endsOf(data, "after").cut;
  const fromEnds = !volume && keptList.length > 0;
  const [slicesCount, rows, cols] = volume ? volume.shape : fromEnds ? [keptList[0].slices, keptList[0].shape[0], keptList[0].shape[1]] : [0, 0, 0];
  const slicePositions = volume ? Array.from({ length: slicesCount }, (_, index) => index) : keptList.map((item) => item.slice);
  const slice = slicePositions.length ? (sliceIndex >= 0 ? slicePositions.reduce((best, one) => (Math.abs(one - sliceIndex) < Math.abs(best - sliceIndex) ? one : best), slicePositions[0])
    : slicePositions[Math.floor(slicePositions.length / 2)]) : 0;
  const sliceData = React.useMemo(() => (volume ? volume.data.subarray(slice * rows * cols, (slice + 1) * rows * cols)
    : keptList.find((item) => item.slice === slice)?.data || EMPTY_FRAME), [volume?.data, keptList, slice, rows, cols]);
  const projection = frameOf(data.projections, epoch);
  const projectionRange = React.useMemo(() => rangeOf(projection ? projection.data : EMPTY_FRAME), [projection?.data]);
  const [showProjection, setShowProjection] = React.useState(!volume);
  const [showPeaks, setShowPeaks] = React.useState(true);
  const [dragLine, setDragLine] = React.useState<number[] | null>(null);
  const [region, setRegion] = React.useState<Region | null>(null);
  const given = data.lineGiven;
  const line = dragLine || (savedLine && savedLine.length === 4 ? savedLine : data.line);
  const projecting = showProjection || (!volume && !fromEnds);
  const shown = projecting && projection ? { data: projection.data, rows: projection.shape[0], cols: projection.shape[1], low: projectionRange[0], high: projectionRange[1], epoch: projection.epoch }
    : { data: sliceData, rows, cols, low: data.phaseRange[0], high: data.phaseRange[1], epoch: volume ? volume.epoch : epoch };
  const spectrum = useSpectrum(shown.data, shown.rows, shown.cols, region, data.pixelA);
  const columns = data.peaks;
  const spacing = React.useMemo(() => columnSpacing(columns), [columns]);
  const crossings = React.useMemo(() => lineCrossings(line, columns, spacing / 3, data.pixelA), [line, columns, spacing, data.pixelA]);
  const cut = React.useMemo(() => (volume ? cutAlong(volume.data, slicesCount, rows, cols, line, data.pixelA)
    : keptCut ? { image: keptCut.data, width: keptCut.shape[1], lengthA: keptCut.length_A || 0 } : null), [volume?.data, keptCut, line, slicesCount, rows, cols, data.pixelA]);
  const probe = frameOf(data.probes, epoch);
  const shares = (probe?.extra as number[] | null) || [];
  const apertureFrame = frameOf(data.apertures, epoch);
  const [probeBox, probeWidth] = useWidth(320);
  const depthIndex = data.depthEpochs.length ? Math.max(0, data.depthEpochs.reduce((best, value, index) => (value <= epoch ? index : best), 0)) : -1;
  const depthCanvas = React.useMemo(() => {
    const draws = data.depthRows.length, slices = Math.max(0, ...data.depthRows.map((row) => row.length));
    if (!draws || !slices) return null;
    const image = new Float32Array(slices * draws).fill(NaN);  // slices down, draws across
    let low = Infinity, high = -Infinity;
    data.depthRows.forEach((row, draw) => row.forEach((value, s) => { image[s * draws + draw] = value; if (Number.isFinite(value)) { low = Math.min(low, value); high = Math.max(high, value); } }));
    return { image, draws, slices, low, high };
  }, [data.depthRows]);
  const muted = { color: colors.textMuted } as React.CSSProperties;
  const kept = (value: number | undefined) => previewing ? <span style={muted}> (preview{value != null && value !== epoch ? `, kept ${value}` : ""})</span>
    : value != null && value !== epoch ? <span style={muted}> kept epoch {value}</span> : null;
  const large = size === "large";
  const wide = width >= 900 ? (large ? 2 : 3) : width >= 520 ? 2 : 1;  // large: the object and the probe share the first row, the depth spans the next
  // the object first, then the depth cut, then the probe: in that order across a row and down a phone. The three share
  // one image height; the object's column is exactly its image, the others share what is left, so no panel holds a band.
  const areas = wide === 3 ? `"object depth probe"` : wide === 2 ? `"object depth" "probe probe"` : `"object" "depth" "probe"`;
  const aspect = (shown.rows || 1) / (shown.cols || 1);
  const imageHeight = Math.max(80, wide === 1 ? Math.min(bySize(size, 260, 420, 560), Math.floor((width - 26) * aspect))
    : Math.min(bySize(size, 180, 300, 560), Math.floor((width / wide - 26) * aspect)));
  const objectColumn = Math.round(imageHeight / aspect) + 12;  // the image and the panel's padding
  const columns3 = wide === 1 ? "minmax(0, 1fr)" : wide === 2 ? `minmax(0, ${objectColumn}px) minmax(0, 1fr)` : `minmax(0, ${objectColumn}px) minmax(0, 1fr) minmax(0, 1fr)`;  // minmax(0, ...): a fixed column would hold the view at its old width when the window narrows
  const FOOT = 24;  // one line under each panel's picture: the object's switches, the depth's band, the probe's numbers
  const foot: React.CSSProperties = { display: "flex", flexWrap: "wrap", alignItems: "center", gap: `0 ${3 * GAP}px`, minHeight: FOOT, marginTop: HAIR, fontSize: TEXT, overflow: "hidden" };
  const besideObject = wide > 1;  // the depth panel stands beside the object and ends where it ends
  const stripHeight = 56;
  const probeSize = imageHeight;  // sized with the object, narrowed by ProbeView to its panel
  const number = (key: string, name: string, unit: string, digits: number) => {  // the probe's health as numbers at the epoch shown, no small plots
    const now = recordAt(data.records, epoch, key);
    return <span key={key}><Health label={name} name={key} value={now ? now[key] : undefined} unit={unit} bar={data.bars?.[key]} colors={colors} digits={digits} /></span>;
  };
  const sliceRange = (index: number) => data.sliceA ? `, ${plain(index * data.sliceA, 0)}-${plain((index + 1) * data.sliceA, 0)} Å` : "";
  return (
    <div style={{ display: "grid", gridTemplateColumns: columns3, gridTemplateAreas: areas, gap: GAP, marginTop: GAP }}>
      <Panel area="object" title="Object" colors={colors} theme={theme}
        hint={<>{projecting ? "projection" : `slice ${slice + 1}/${slicesCount}${sliceRange(slice)}`}{kept(shown.epoch)}</>}
        info="The phase of one slice, or the projected phase. Drag the line or its ends; an end dropped near a column snaps to it, and the line is saved with the notebook. Drag elsewhere on the image to take the spectrum of a square region. Rings: the atomic columns found, tan heavy and blue-gray light; ticks on the profile mark where the line passes one. Spectrum: Hann-windowed power, log scale, ±2.6 1/Å, the dashed ring at 50 pm. Slices are kept at each run's end.">
        <div style={{ display: "flex" }}>
          <FieldView data={shown.data} rows={shown.rows} cols={shown.cols} low={shown.low} high={shown.high} pixelA={data.pixelA} height={imageHeight}
            line={line} editLine={!projecting && !!volume} onDragLine={setDragLine} onLine={setSavedLine} peaks={columns} showPeaks={showPeaks} region={region} onRegion={setRegion} colors={colors}
            inset={<SpectrumInset spectrum={spectrum} side={imageHeight} label={region ? `${plain(region.size * data.pixelA, 0)} Å box` : "whole field"} />} />
        </div>
        <div style={foot}>
          {(volume || fromEnds) && <Box sx={{ display: "flex" }}>{toggleButton(!projecting, "Slice", () => setShowProjection(false))}{toggleButton(projecting, "Projection", () => setShowProjection(true))}</Box>}
          {columns.length > 0 && switchControl("Columns", showPeaks, () => setShowPeaks(!showPeaks), "Rings on the atomic columns found", colors)}
          {switchControl("Smooth", smooth, () => setSmooth(!smooth), "CSS bilinear interpolation. Same data, the browser smooths visually.", colors)}
          {given.length === 4 && savedLine && savedLine.length === 4 && savedLine.join() !== given.join() && toggleButton(false, "Reset line", () => setSavedLine([...given]))}
          {region && toggleButton(false, "Whole field", () => setRegion(null))}
        </div>
        {!projecting && slicePositions.length > 1 && <Stepper positions={slicePositions} value={slice} onChange={setSliceIndex} name="slice" colors={colors} />}
        {!projecting && <LineChart series={[{ name: "on the line", color: HUES[0], points: profileAlong(sliceData, rows, cols, line || [], data.pixelA) }]} stages={[]} cursor={null}
          height={70} xLabel="Å along the line" yLabel="rad" colors={colors} legend={false} marks={showPeaks ? crossings : []} />}
      </Panel>
      <Panel area="depth" title="Depth" hint={<>{slicesCount ? `${slicesCount} slices${data.sliceA ? ` of ${plain(data.sliceA, 1)} Å` : ""}` : "no volume kept"}{kept(volume?.epoch)}</>} colors={colors} theme={theme}
        info="Top: the volume cut along the line through every slice, the beam entering at the top; the accent line marks the slice shown. Bottom: lattice power per slice (down) against draw (across), the accent line at the slice shown, the dashed line at the epoch shown; the specimen's faces form while vacuum slices stay dark.">
        <div style={{ display: "flex", flexDirection: "column", gap: HAIR }}>
          {cut && <div style={{ position: "relative" }}>
            <ImageCanvas data={cut.image} rows={slicesCount} cols={cut.width} low={data.phaseRange[0]} high={data.phaseRange[1]} cmap="gray" line={[slice + 0.5, 0, slice + 0.5, cut.width]} lineColor={colors.accent}
              height={besideObject ? Math.max(60, imageHeight - (depthCanvas ? stripHeight + HAIR : 0)) : Math.min(180, Math.round(0.5 * imageHeight))} />
            <div style={{ ...ON_IMAGE, left: 3, top: 1 }}>cut along the line, beam down</div>
            <div style={{ ...ON_IMAGE, right: 3, bottom: 1 }}>{plain(cut.lengthA, 0)} Å along the line</div>
          </div>}
          {depthCanvas && <div style={{ position: "relative" }}>
            <ImageCanvas data={depthCanvas.image} rows={depthCanvas.slices} cols={depthCanvas.draws} low={depthCanvas.low} high={depthCanvas.high} cmap="magma"
              line={slicesCount ? [slice + 0.5, 0, slice + 0.5, depthCanvas.draws] : undefined} lineColor={colors.accent} height={stripHeight}
              overlay={depthIndex >= 0 ? <line x1={depthIndex + 0.5} x2={depthIndex + 0.5} y1={0} y2={depthCanvas.slices} stroke="#fff" strokeWidth={1} strokeDasharray="3 2" opacity={0.7} vectorEffect="non-scaling-stroke" /> : null} />
            <div style={{ ...ON_IMAGE, left: 3, top: 1 }}>lattice power, each slice over the epochs</div>
            <div style={{ ...ON_IMAGE, right: 3, bottom: 1 }}>epochs {data.depthEpochs[0]}-{data.depthEpochs[data.depthEpochs.length - 1]}</div>
          </div>}
        </div>
        <div style={{ ...foot, color: colors.textMuted }}>{depthCanvas ? `lattice band ${data.bandPerA.length === 2 ? `${plain(data.bandPerA[0], 2)} to ${plain(data.bandPerA[1], 2)} 1/Å` : "of the crystal"}` : ""}</div>
      </Panel>
      <Panel area="probe" title="Probe" hint={<>modes at focus{kept(probe?.epoch)}</>} colors={colors} theme={theme}
        info="The probe's natural (orthogonal) modes at focus, in complex-wave colors (hue the phase, brightness the amplitude), cropped to where their intensity lives, each with its share of the power; beside the strongest, the Fourier-space amplitude (the aperture). Numbers: the radius holding 99 % of the probe's power (r99), its power on the lattice's sites (copies) and beyond 5 Å, each against its pass bar, with its curve over every epoch.">
        <div ref={probeBox}>
          <ProbeView frame={probe?.data} shape={probe?.shape} shares={shares} aperture={apertureFrame ? { data: apertureFrame.data, shape: apertureFrame.shape } : null} size={probeSize} maxWidth={probeWidth} colors={colors}
            box={[probeWidth, wide === 2 ? Math.min(imageHeight, 220) : wide === 1 ? Math.min(imageHeight, probeWidth) : imageHeight]} />
        </div>
        <div style={foot}>
          {number("r99_A", "probe radius", " Å", 1)}
          {number("copies_percent", "probe copies", " %", 1)}
          {number("far_power_percent", "power beyond 5 Å", " %", 1)}
        </div>
      </Panel>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------------------
// The stage tiles and the report

// What each stage bought: one tile a run at its end, images only, edge to edge: its projected phase, then (fft) the
// projection's spectrum (the display FFT, the dashed ring at 50 pm), then its strongest probe mode (complex colours), all
// at one height. At small (inset) the tile is the projection alone with the spectrum and the probe in its corners.
// Badges on the images: the stage's name top-left; its held-out and the signed change against the run it continued
// from bottom-left (▼ better in green, ▲ worse in amber), its image score and change just above; its verdict top-right
// (★ the best kept, ✓ kept, ▲ worse than its start, ✕ tried and refused, dimmed), with ≠ when held-out and image score
// disagree. The full numbers and its settings on hover. perRow fixes the stages a row; else as many as fit at about
// size px an image. Hover looks at a run's end, click keeps it (onLook, onPick), when given. No image here zooms: a tile
// is a summary, and a probe or spectrum never shares an object's zoom.
type TileEnd = {
  name: string; endEpoch: number; start: number; projection: { data: Float32Array; shape: number[] } | null; strongest: { data: Float32Array; shape: number[] } | null;
  score: number | null; imageScore: number | null; refused: string; changes: string; pixelA: number; slice: EndItem | null;
};

// What a stage tile holds beside the object's projection, each a chip the reader switches: its spectrum, its middle
// slice, its strongest probe mode at the picture's size. A part switched off is not lost: the probe becomes a small
// inset in the projection's corner. The tiles trait keeps the choice ("" is the view's own default; "image" none).
const TILE_PARTS: [string, string, string][] = [["fft", "FFT", "the projection's spectrum, with the 50 pm ring and its finest spot"],
  ["slice", "Slice", "the middle slice kept of the run, at its depth"], ["probe", "Probe", "the strongest probe mode at the picture's size; off, it is a small inset in the picture's corner"]];
function tileParts(tiles: string | undefined, fallback: string): Set<string> {
  const text = tiles === "all" ? "fft,probe" : tiles || fallback;  // "all": the first spelling of FFT and probe
  return new Set(text.split(",").map((part) => part.trim()).filter((part) => TILE_PARTS.some(([key]) => key === part)));
}
function TilePartsControl({ parts, onParts, colors }: { parts: Set<string>; onParts: (tiles: string) => void; colors: any }) {
  const toggle = (key: string) => { const next = new Set(parts); if (next.has(key)) next.delete(key); else next.add(key); onParts([...next].join(",") || "image"); };
  return (
    <span style={{ display: "inline-flex", border: `1px solid ${colors.border}`, borderRadius: 3, overflow: "hidden", flex: "none" }} role="group" aria-label="What each tile holds">
      {TILE_PARTS.map(([key, text, why]) => (
        <button key={key} type="button" aria-pressed={parts.has(key)} title={why} onClick={() => toggle(key)}
          style={{ border: "none", padding: "0 6px", font: "inherit", fontSize: 10, lineHeight: "16px", cursor: "pointer", whiteSpace: "nowrap",
            background: parts.has(key) ? colors.accent : "transparent", color: parts.has(key) ? "#fff" : colors.textMuted }}>{text}</button>))}
    </span>
  );
}

// Rows of equal length: the tiles a row for count tiles when a row holds at most most (eight are 4 + 4, never 6 + 2).
function evenRows(count: number, most: number): number {
  return Math.max(1, Math.ceil(count / Math.ceil(count / Math.max(1, most))));
}

function StageTiles({ runs, size, inset = false, fft = false, slice = false, probeCorner = false, perRow: fixed, shown, onLook, onPick, colors }: {
  runs: RunData[]; size: number; inset?: boolean; fft?: boolean; perRow?: number;
  shown?: Stage | null; onLook?: (epoch: number | null) => void; onPick?: (epoch: number) => void; colors: any;
  slice?: boolean;  // the run's middle kept slice beside its projection
  probeCorner?: boolean;  // the probe stays a small inset in the projection's corner while the spectrum keeps its own picture
}) {
  const [row, rowWidth] = useWidth(1000);
  const ends: TileEnd[] = runs.map((run, index) => {
    const stage = run.stages[0], endRecord = recordAt(run.records, Infinity, "held_out"), scored = recordAt(run.records, Infinity, "image_score");
    const endEpoch = run.records.length ? run.records[run.records.length - 1].epoch : stage ? stage.start : 0;
    return { name: stage ? stage.name : `run ${index + 1}`, endEpoch, start: stage ? stage.start : 0, projection: frameOf(run.projections, endEpoch), strongest: frameOf(run.strongest, endEpoch),
      score: endRecord ? endRecord.held_out : null, imageScore: scored ? Number(scored.image_score) : null,
      refused: stage && stage.refused ? stage.refused : "", changes: stage ? plainChanges(stageChanges(stage)) : "", pixelA: run.pixelA,
      slice: ((kept) => (kept.length ? kept[Math.floor(kept.length / 2)] : null))(endsOf(run, "after").slices) };
  });
  const startScores = runs.map((run) => (run.records.length && run.records[0].start ? run.records[0].held_out : null));
  const best = ends.reduce((found: number, end, index) => (!end.refused && end.score != null && (found < 0 || end.score < ends[found].score!) ? index : found), -1);
  inset = inset || (!fixed && rowWidth < 520);  // a phone: the probe in the projection's corner, so several stages fit a row
  const across = inset ? 1 : 1 + (slice ? 1 : 0) + (fft ? 1 : 0) + (probeCorner ? 0 : 1);  // images a tile is wide
  const most = Math.max(1, Math.min(ends.length, Math.floor(rowWidth / (across * size * 0.9))));  // the stages a row can hold at this size, a tenth under it allowed so four parts still go two a row
  const perRow = fixed || evenRows(ends.length, most);
  const image = Math.min(fixed ? Infinity : Math.round(size * 2), Math.floor(rowWidth / perRow / across));  // the images grow to fill the row, a short row only so far
  return (
    <div ref={row} style={{ display: "flex", flexWrap: "wrap", gap: 0 }} onMouseLeave={onLook ? () => onLook(null) : undefined}>
      {ends.map((end, index) => {
        const before = ends.slice(0, index).filter((one) => !one.refused).pop();
        const previous = end.refused || !before ? startScores[index] : before.score;
        const delta = end.score != null && previous != null ? end.score - previous : null;
        const scoreDelta = end.imageScore != null && before && before.imageScore != null ? end.imageScore - before.imageScore : null;
        return <StageTile key={index} end={end} image={image} across={across} inset={inset} fft={fft} slice={slice} probeCorner={probeCorner} best={index === best} delta={delta} scoreDelta={scoreDelta}
          against={end.refused || !before ? "its start" : "the run before"} chosen={!!shown && shown.name === end.name && shown.start === end.start}
          onLook={onLook} onPick={onPick} colors={colors} />;
      })}
    </div>
  );
}

function StageTile({ end, image, across, inset, fft, slice, probeCorner, best, delta, scoreDelta, against, chosen, onLook, onPick, colors }: {
  end: TileEnd; image: number; across: number; inset: boolean; fft: boolean; slice?: boolean; probeCorner?: boolean; best: boolean; delta: number | null; scoreDelta: number | null; against: string; chosen: boolean;
  onLook?: (epoch: number | null) => void; onPick?: (epoch: number) => void; colors: any;
}) {
  const range = React.useMemo(() => (end.projection ? rangeOf(end.projection.data) : [0, 1]), [end.projection?.data]);
  const sliceRange = React.useMemo(() => (end.slice ? rangeOf(end.slice.data) : [0, 1]), [end.slice?.data]);
  const [rows, cols] = end.projection ? end.projection.shape : [0, 0];
  const spectrum = useSpectrum(fft && end.projection ? end.projection.data : EMPTY_FRAME, fft ? rows : 0, fft ? cols : 0, null, end.pixelA);
  const strongest = React.useMemo(() => (end.strongest ? probeModes(end.strongest.data, end.strongest.shape, true)[0] : null), [end.strongest?.data]);
  const [mark, markWhy, tone] = end.refused ? ["✕", "tried and refused", colors.textMuted] : best ? ["★", "the best kept run", "#f2c14e"]
    : delta != null && delta > 0 ? ["▲", "worse than its start", colors.warn] : ["✓", "kept", colors.good];
  const differ = disagreement(delta, scoreDelta);
  const why = `${end.name}: ${markWhy}; held-out ${plain(end.score, 5)}${delta != null ? ` (${change(delta)} against ${against})` : ""}`
    + `${end.imageScore != null ? `; image score ${plain(end.imageScore, 3)}${scoreDelta != null ? ` (${scoreDelta >= 0 ? "+" : "−"}${plain(Math.abs(scoreDelta), 3)})` : ""}` : ""}`
    + `${differ ? `; ≠ ${differ}` : ""}; epochs ${end.start}-${end.endEpoch}${end.changes ? `; ${end.changes}` : ""}; ptycho("${end.name}") shows its view`;
  const font = image >= 110 ? TEXT : 10;
  const badge = (style: React.CSSProperties, text: React.ReactNode) => <div style={{ ...ON_IMAGE, ...PLATE, fontSize: font, lineHeight: "13px", ...style }}>{text}</div>;
  const small = Math.round(image * 0.3);  // the corner insets leave the picture its middle and bottom
  const spectrumImage = (side: number) => spectrum && <ImageCanvas data={spectrum.image} rows={spectrum.size} cols={spectrum.size} low={spectrum.low} high={spectrum.high} cmap="gray" height={side} keepAspect
    overlay={spectrumRings(spectrum)} />;
  return (
    <div title={why} onMouseEnter={end.refused || !onLook ? undefined : () => onLook(end.endEpoch)} onClick={end.refused || !onPick ? undefined : () => onPick(end.endEpoch)}
      style={{ position: "relative", display: "flex", width: across * image, height: image, cursor: end.refused || !onPick ? "default" : "pointer", background: "#000", opacity: end.refused ? 0.5 : 1 }}>
      <div style={{ position: "relative", width: image, height: image, flex: "none" }}>
        {end.projection && <ImageCanvas data={end.projection.data} rows={rows} cols={cols} low={range[0]} high={range[1]} cmap="gray" height={image} keepAspect />}
        {inset && fft && spectrum && <div style={{ position: "absolute", left: 0, top: 16, width: small, height: small, boxShadow: "0 0 0 1px #000" }}>{spectrumImage(small)}</div>}
        {(inset || probeCorner) && strongest && <div style={{ position: "absolute", right: 0, top: 16, width: small, height: small, boxShadow: "0 0 0 1px #000" }}><ModeTile mode={strongest} size={small} /></div>}
      </div>
      {!inset && slice && <div style={{ position: "relative", width: image, height: image, flex: "none", background: "#111" }}>
        {end.slice && <ImageCanvas data={end.slice.data} rows={end.slice.shape[0]} cols={end.slice.shape[1]} low={sliceRange[0]} high={sliceRange[1]} cmap="gray" height={image} keepAspect />}
        {badge({ left: 2, bottom: 2 }, end.slice ? `slice ${end.slice.slice + 1}/${end.slice.slices}${end.slice.depth_A ? ` · ${plain(end.slice.depth_A[0], 0)}-${plain(end.slice.depth_A[1], 0)} Å` : ""}` : "no slice kept")}</div>}
      {!inset && fft && <div style={{ position: "relative", width: image, height: image, flex: "none" }} title={SPOT_WHY}>{spectrumImage(image)}
        {spotText(spectrum) && badge({ left: 2, bottom: 2 }, spotText(spectrum))}</div>}
      {!inset && !probeCorner && strongest && <ModeTile mode={strongest} size={image} />}
      {badge({ left: 2, top: 2, maxWidth: across * image - 44, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }, end.name)}
      {badge({ right: 2, top: 2, fontSize: image >= 110 ? 13 : 11 }, <><span style={{ color: tone }}>{mark}</span>{differ && <span style={{ color: "#f2b04e" }} title={differ}> ≠</span>}</>)}
      {badge({ left: 2, bottom: 2 }, <>{plain(end.score, delta != null && end.score != null ? pairDigits(end.score, end.score - delta) : 5)}{delta != null && <span style={{ color: end.refused ? "#ccc" : delta <= 0 ? "#7ddc8a" : "#f2b04e" }}> {change(delta)}</span>}</>)}
      {end.imageScore != null && badge({ right: 2, bottom: 2 }, <><span style={{ opacity: 0.75 }}>image </span>{plain(end.imageScore, 3)}{scoreDelta != null && Math.abs(scoreDelta) >= 0.0005 && <span style={{ color: end.refused ? "#ccc" : scoreDelta >= 0 ? "#7ddc8a" : "#f2b04e" }}> {scoreDelta >= 0 ? "▲" : "▼"}{Math.abs(scoreDelta).toFixed(3)}</span>}</>)}
      {chosen && !end.refused && <div style={{ position: "absolute", inset: 0, border: `2px solid ${colors.accent}`, pointerEvents: "none" }} />}
    </div>
  );
}

// ptycho.report(): what each stage bought, the stage tiles alone, larger.
function ReportView() {
  const { colors, theme } = usePalette();
  const runs = useStores(useReferenced("runs"));
  const [title] = useModelState<string>("title");
  const [size, setSize] = useModelState<ViewSize>("size");
  const engineContext = useEngine(false);
  const [tiles, setTiles] = useModelState<string>("tiles");
  const parts = tileParts(tiles === "fft" ? "fft" : tiles, "fft,probe");
  return (
    <Engine.Provider value={engineContext}>
    <div style={header(colors)}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: GAP, lineHeight: "18px", marginBottom: GAP }}>
        <span style={{ flex: "1 1 200px", minWidth: 0 }}><b style={{ fontSize: 13 }}>What each stage bought</b><span style={{ color: colors.textMuted }}> · {title}</span></span>
        <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: GAP }}>
          <TilePartsControl parts={parts} onParts={setTiles} colors={colors} />
          <RenderPathBadge colors={colors} /><SizeControl size={size} onSize={setSize} colors={colors} />
          <InfoTooltip theme={theme} toggleOnClick text={"One tile a run, in the order they ran: its projected phase and strongest probe mode at its end (hue the phase, brightness the amplitude; at small the probe sits in the projection's corner). Top-left its name; bottom-left its held-out score (lower is better) and the change against the run it continued from (▼ better, ▲ worse); above it its image score (repetition × detail of the projected phase, higher is better) and its change; top-right ★ the best kept run, ✓ kept, ▲ worse than its start, ✕ tried and refused (dimmed), and ≠ when held-out and image score move in opposite directions. Each tile: the projected phase, its spectrum (the dashed ring at 50 pm), the strongest probe mode. Hover a tile for its numbers and settings. ptycho(\"name\") shows a run's view; ptycho.report(table=True) gives the numbers."} />
        </span>
      </div>
      <StageTiles runs={runs} size={bySize(size, 110, 170, 280)} fft={parts.has("fft")} slice={parts.has("slice")} probeCorner={!parts.has("probe")} colors={colors} />
    </div>
    </Engine.Provider>
  );
}

// ---------------------------------------------------------------------------------------------------------------
// The dashboard: every run as one story

function DashboardView() {
  const { colors, theme } = usePalette();
  const runModels = useReferenced("runs");
  const runs = useStores(runModels);
  const data = mergeRuns(runs.filter((run) => !(run.stages[0] && run.stages[0].refused)));  // the story is the line of kept runs; a refused stage is a dimmed tile
  const [title] = useModelState<string>("title");
  const [viewsCount] = useModelState<number>("views");
  const [viewsLabel] = useModelState<string>("views_label");
  const [sliderEpoch, setSliderEpoch] = useModelState<number>("slider_epoch");
  const [sliceIndex, setSliceIndex] = useModelState<number>("slice_index");
  const [savedLine, setSavedLine] = useModelState<number[]>("profile_line");
  const [smooth, setSmooth] = useModelState<boolean>("smooth");
  const [size, setSize] = useModelState<ViewSize>("size");
  const engineContext = useEngine(!!smooth);
  const controls = useControls(colors, !!smooth, setSmooth);
  const [preview, setPreview] = React.useState<number | null>(null);
  const [reference] = useModelState<number[]>("reference_tilt_mrad");  // the direct fit's tilt, the tilt's reference
  const [training, setTraining] = React.useState(false);  // the training objective's saw-tooth, behind a click: the held-out is the story
  const [tiles, setTiles] = useModelState<string>("tiles");  // before the early return below: a hook after it breaks the render once the runs arrive
  const [root, rootWidth] = useWidth(1000);
  const muted = { color: colors.textMuted } as React.CSSProperties;
  if (!data) return <div ref={root} style={header(colors)}>{title} · {runModels.length ? "reading the runs" : "no run yet"}</div>;
  const history = data.records;
  const last = history.length ? history[history.length - 1] : ({ epoch: 0 } as Record);
  const committed = sliderEpoch >= 0 ? Math.min(sliderEpoch, last.epoch) : last.epoch;
  const epoch = preview ?? committed;
  const previewing = preview != null && preview !== committed;
  const setEpoch = (value: number) => { setPreview(null); setSliderEpoch(value >= last.epoch ? -1 : value); };
  const positions = Array.from(new Set(data.projections.epochs.length ? data.projections.epochs : history.map((row) => row.epoch))).sort((a, b) => a - b);
  const stages = data.stages;
  const stageAt = (value: number) => stages.reduce((best: Stage | null, stage) => (stage.start <= value ? stage : best), null);
  const stageNow = stageAt(Math.max(0, epoch - 0.5)) || stages[0];
  const atEpoch = recordAt(history, epoch, "held_out") || ({} as Record);
  const health = recordAt(history, epoch, "r99_A") || ({} as Record);
  const heldOutLabel = (viewsCount || 1) > 1 ? "held-out (mean)" : "held-out";
  const viewsText = (viewsCount || 1) <= 1 ? "single scan" : `joint${viewsLabel || viewsCount === 2 ? ` ${viewsLabel || "0/90"}` : ""} · ${viewsCount} scans`;
  const series = (name: string, color: string, dashed?: boolean, label?: string): Series => ({
    name: label || name, color, dashed, points: history.filter((row) => row[name] != null).map((row) => [row.epoch, Number(row[name])] as [number, number]) });
  const storyTip = (value: number) => {
    const row = recordAt(history, value);
    const run = stageAt(value);
    const list = run ? stageChanges(run) : [];
    return <>
      <div><b>{run ? run.name : "no run"}</b> · epoch {value}</div>
      {row && <div>training {plain(row.train, 5)} · {heldOutLabel} <b>{plain(row.held_out, 5)}</b></div>}
      {row && Array.isArray(row.held_out_per_view) && <div style={muted}>{row.held_out_per_view.map((one: number, index: number) => `scan ${index + 1} ${plain(one, 5)}`).join(" · ")}</div>}
      {run && (list.length ? plainChanges(list).split(", ").map((text, index) => <div key={index}>{text}</div>)
        : <div style={muted}>{run.start === 0 ? "the first run" : "no settings changed"}</div>)}
    </>;
  };
  // Each stage starts its rates over, so held-out rises at a stage's first epochs and falls: the story is each stage's
  // end, a step line; the curve within a stage is drawn faint.
  const stepPoints: [number, number][] = [];
  stages.forEach((stage, index) => {
    const end = index + 1 < stages.length ? stages[index + 1].start : last.epoch;
    const row = recordAt(history.filter((one) => one.epoch <= end), end, "held_out");
    if (!row) return;
    stepPoints.push([stage.start, Number(row.held_out)], [end, Number(row.held_out)]);
  });
  const steps: Series = { name: `${heldOutLabel} at each stage's end`, color: HUES[0], points: stepPoints };
  const scans = Array.isArray(last.tilt_mrad) ? last.tilt_mrad.length : 0;
  const tiltSeries: Series[] = [];
  for (let scan = 0; scan < scans; scan++) for (const axis of [0, 1]) tiltSeries.push({ name: `scan ${scan + 1} ${axis ? "col" : "row"}`, color: HUES[scan % HUES.length], dashed: axis === 1,
    points: history.filter((row) => Array.isArray(row.tilt_mrad) && row.tilt_mrad[scan]).map((row) => [row.epoch, Number(row.tilt_mrad[scan][axis])] as [number, number]) });
  const referenceText = reference && reference.length === 2 ? `direct ptychography ${reference.map((value) => `${value >= 0 ? "+" : ""}${plain(value, 2)}`).join(", ")}` : "";
  if (reference && reference.length === 2 && scans) for (const axis of [0, 1]) tiltSeries.push({ name: `${referenceText} ${axis ? "col" : "row"}`, color: colors.textMuted, dashed: true,
    points: [[0, reference[axis]], [last.epoch, reference[axis]]] });
  const tiltMoves = tiltSeries.slice(0, 2 * scans).some((one) => { const values = one.points.map(([, y]) => y); return values.length > 1 && Math.max(...values) - Math.min(...values) > 0.5; });  // shown only when training moved it by half a mrad or more
  const [pathLines, pathKnots] = data.pathShape;
  const pathSeries: Series[] = Array.from({ length: Math.min(3, pathLines) }, (_, index) => ({ name: `line ${index + 1}`, color: HUES[index],
    points: Array.from({ length: pathKnots }, (_, knot) => [knot * (data.pathStep || 1), data.path[index * pathKnots + knot]] as [number, number]) }));
  const hasMap = data.tiltShape[0] > 0;
  const side = [tiltMoves && "tilt", hasMap && "map", pathSeries.length > 0 && "scan"].filter(Boolean) as string[];
  const narrow = rootWidth < 700;
  const storyColumns = !side.length ? "minmax(0, 1fr)" : narrow ? side.map(() => "minmax(0, 1fr)").join(" ") : `minmax(0, 2.4fr) ${side.map(() => "minmax(0, 1fr)").join(" ")}`;
  const storyAreas = !side.length ? `"story"` : narrow ? `"${side.map(() => "story").join(" ")}" "${side.join(" ")}"` : `"story ${side.join(" ")}"`;
  const tileImage = bySize(size, 96, 120, 170);  // the stage strip's images at their smallest, grown to fill the row
  const parts = tileParts(tiles, "probe");
  return (
    <Engine.Provider value={engineContext}><ViewLink.Provider value={controls.link}>
    <div ref={root} tabIndex={0} style={header(colors)}
      onKeyDown={(event) => {
        if ((event.target as HTMLElement).closest(".MuiSlider-root")) return;
        if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
          const index = Math.max(0, positions.reduce((found, value, at) => (value <= committed ? at : found), 0));
          const next = positions[Math.max(0, Math.min(positions.length - 1, index + (event.key === "ArrowLeft" ? -1 : 1)))];
          if (next != null) setEpoch(next);
          event.preventDefault();
        }
      }}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: `0 ${3 * GAP}px`, lineHeight: "18px", marginBottom: GAP }}>
        <span style={{ fontWeight: 600, fontSize: 13, overflowWrap: "anywhere" }} title={[data.status.seconds_per_epoch != null ? `${plain(data.status.seconds_per_epoch, 1)} s an epoch` : "", data.status.gpu_GB != null ? `GPU memory ${plain(data.status.gpu_GB, 1)} GB at most` : ""].filter(Boolean).join(" · ")}>{title} · {viewsText} · {stageNow ? `"${stageNow.name}"` : "no run yet"} · epoch {epoch}/{last.epoch}</span>
        <Health label={heldOutLabel} value={atEpoch.held_out} unit="" colors={colors} digits={5} />
        <Health label="image score" value={(recordAt(history, epoch, "image_score") || ({} as Record)).image_score} unit="" colors={colors} digits={3} />
        <Health label="probe radius" name="r99_A" value={health.r99_A} unit=" Å" bar={data.bars?.r99_A} colors={colors} digits={1} />
        <Health label="probe copies" name="copies_percent" value={health.copies_percent} unit=" %" bar={data.bars?.copies_percent} colors={colors} digits={1} />
        {Array.isArray(atEpoch.tilt_mrad) && <span style={{ ...muted, whiteSpace: "nowrap" }}>tilt {atEpoch.tilt_mrad.map((pair: number[]) => `${plain(pair[0], 2)}, ${plain(pair[1], 2)}`).join(" · ")} mrad{referenceText ? ` (${referenceText})` : ""}</span>}
        {previewing && <span style={muted}>preview, epoch {epoch}</span>}
        <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: GAP }}><TilePartsControl parts={parts} onParts={setTiles} colors={colors} /><RenderPathBadge colors={colors} />{controls.toggle}<SizeControl size={size} onSize={setSize} colors={colors} /></span>
      </div>
      {controls.row}
      <StageTiles runs={runs} size={tileImage} fft={parts.has("fft")} slice={parts.has("slice")} probeCorner={!parts.has("probe") || size === "small"} shown={stageNow} onLook={setPreview} onPick={setEpoch} colors={colors} />
      <DetailGrid data={data} epoch={epoch} previewing={previewing} sliceIndex={sliceIndex} setSliceIndex={setSliceIndex} savedLine={savedLine} setSavedLine={setSavedLine}
        smooth={!!smooth} setSmooth={setSmooth} colors={colors} theme={theme} width={rootWidth} size={size} />
      <div style={{ display: "grid", gridTemplateColumns: storyColumns, gridTemplateAreas: storyAreas, gap: GAP, marginTop: GAP }}>
        <Panel area="story" title="Held-out" hint={<>{previewing ? <span style={muted}>epoch {epoch} (preview) · </span> : null}<span style={{ cursor: "pointer", textDecoration: "underline dotted" }} onClick={() => setTraining(!training)}>{training ? "hide training" : "show training"}</span></>} colors={colors} theme={theme}
          info="Held-out against total epoch (lower is better): the solid step is each stage's end and sets the axis, the faint curve the epochs within a stage, which rise at a stage's start because each stage starts its rates over, then fall (cut at the axis' edge where it runs past); the training objective on a click. Each run is a shaded band, and a triangle marks where a run starts. Hover to preview an epoch in every panel and read that run's changes (old value grey, new in color); click or tap to keep that epoch. Ticks on the epoch axis mark epochs that keep a full volume. The tiles above are each run's end: hover to look, click to keep; a dimmed tile is a stage tried and refused. A saved notebook keeps, for each run: its projection at every epoch (the newest run) or at its draws (earlier runs), its probe's modes at its end and its strongest mode at every draw, and slices with the cut along the line in place of the whole volume (three, a quarter, half and three quarters through the slab, for the newest run; the middle one for earlier runs); the running kernel keeps everything.">
          <LineChart series={[...(training ? [{ ...series("train", colors.border, true, "training"), }] : []), { ...series("held_out", colors.border, false, "within a stage") }, steps]}
            keyItems={[{ name: `${heldOutLabel} at each stage's end`, color: HUES[0], points: [] }, { name: "within a stage", color: colors.border, points: [] },
              ...(training ? [{ name: "training", color: colors.border, dashed: true, points: [] }] : [])]}
            stages={stages} cursor={committed} stageNames fitTo={[steps]}
            height={bySize(size, 110, 150, 200)} xLabel="total epoch" yLabel={heldOutLabel} colors={colors} onPick={setEpoch} snapX={positions} tooltip={storyTip}
            onHoverX={setPreview} marks={data.volumes.epochs.map((value) => ({ x: value, color: colors.textMuted }))} />
          <Stepper positions={positions} value={committed} onChange={setEpoch} name="epoch" colors={colors}
            label={<>epoch {committed} of {last.epoch}{stageAt(committed) ? ` · ${stageAt(committed)!.name}` : ""}</>} />
        </Panel>
        {tiltMoves && <Panel area="tilt" title="Specimen tilt" hint="learned, mrad" colors={colors} theme={theme} info="The specimen's tilt as training learns it (row solid, column dashed) against total epoch, beside the direct ptychography fit it started from (grey dashed); shown only because it moved by half a mrad or more.">
          <Fill min={narrow ? 110 : 56}>{(_, height) => <LineChart series={tiltSeries} keyItems={[{ name: "learned (row, col dashed)", color: HUES[0], points: [] },
            ...(referenceText ? [{ name: referenceText, color: colors.textMuted, dashed: true, points: [] }] : [])]} stages={stages} cursor={epoch} height={height} outer xLabel="total epoch" yLabel="mrad" colors={colors} />}</Fill>
        </Panel>}
        {hasMap && <Panel area="map" title="Tilt map" colors={colors} theme={theme} info="Local tilt across the field: one arrow a cell, its length scaled to the longest arrow.">
          {(() => {
            const [cells] = data.tiltShape; let high = 1e-9;
            for (let index = 0; index < cells * cells; index++) high = Math.max(high, Math.hypot(data.tilt[2 * index], data.tilt[2 * index + 1]));
            return <>
              <svg viewBox="0 0 100 100" style={{ width: "100%", height: 110, display: "block" }} preserveAspectRatio="xMidYMid meet">
                <rect x={0} y={0} width={100} height={100} fill="none" stroke={colors.border} vectorEffect="non-scaling-stroke" />
                {Array.from({ length: cells * cells }, (_, index) => {
                  const r = Math.floor(index / cells), c = index % cells, cy = (r + 0.5) * 100 / cells, cx = (c + 0.5) * 100 / cells, scale = 0.45 * 100 / cells / high;
                  return <line key={index} x1={cx} y1={cy} x2={cx + data.tilt[2 * index + 1] * scale} y2={cy + data.tilt[2 * index] * scale} stroke={HUES[0]} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />;
                })}
              </svg>
              <div style={{ fontSize: TEXT, lineHeight: "16px", textAlign: "center", ...muted }}>longest {plain(high, 2)} mrad</div>
            </>;
          })()}
        </Panel>}
        {pathSeries.length > 0 && <Panel area="scan" title="Vibration path" hint="along the fast scan, Å" colors={colors} theme={theme} info="The beam's vibration against the specimen, learned: how far it moved (Å) along each of the first three scan lines as the line was scanned.">
          <Fill min={narrow ? 110 : 56}>{(_, height) => <LineChart series={pathSeries} stages={[]} cursor={null} height={height} outer xLabel={data.pathStep ? "ms along the scan line" : "along the scan line"} yLabel="Å" colors={colors} />}</Fill>
        </Panel>}
      </div>
    </div>
    </ViewLink.Provider></Engine.Provider>
  );
}

// ---------------------------------------------------------------------------------------------------------------
// Compare: two to four runs side by side

type RunMeta = {
  name: string; source: number; items: any[]; slice_A: number; pixel_A: number; low: number; high: number;
  power: number[] | null; record: { [name: string]: any }; changed: Change[]; epochs: number | null;
};

// One column's arrays: from its run model's saved after state (three slices, the cut), last projection, probe and
// aperture, and its lattice power per slice; or from the compare's own arrays for a run without a model.
type ColumnData = {
  slices: EndItem[]; count: number; cut: EndItem | null; projection: { data: Float32Array; shape: number[] } | null;
  probe: { data: Float32Array; shape: number[]; shares: number[] } | null; aperture: { data: Float32Array; shape: number[] } | null; power: number[] | null;
};

function columnOf(run: RunMeta, own: EndItem[], source: RunData | null): ColumnData {
  if (source) {
    const after = endsOf(source, "after");
    const last = (stack: Stack) => (stack.frames.length ? { data: stack.frames[stack.frames.length - 1], shape: stack.shapes[stack.shapes.length - 1] } : null);
    const probe = last(source.probes);
    return { slices: after.slices, count: after.slices.length ? after.slices[0].slices : 0, cut: after.cut, projection: last(source.projections),
      probe: probe ? { ...probe, shares: (source.probes.extras[source.probes.extras.length - 1] as number[]) || [] } : null,
      aperture: last(source.apertures), power: source.depthRows.length ? source.depthRows[source.depthRows.length - 1] : null };
  }
  const find = (kind: string) => own.find((item) => item.kind === kind) || null;
  const slices = own.filter((item) => item.kind === "slice");
  const probe = find("probe"), projection = find("projection"), aperture = find("aperture");
  return { slices, count: slices.length ? slices[0].slices : 0, cut: find("cut"), projection: projection ? { data: projection.data, shape: projection.shape } : null,
    probe: probe ? { data: probe.data, shape: probe.shape, shares: probe.shares || [] } : null, aperture: aperture ? { data: aperture.data, shape: aperture.shape } : null, power: run.power };
}

// What a person set differently from the first run, as "probe.modes 8 → 10, object.thickness_A 400 → 450".
function differs(run: RunMeta): string {
  return plainChanges(run.changed) || "same settings";
}

function CompareReconstruction() {
  const { colors, theme } = usePalette();
  const [title] = useModelState<string>("title");
  const [runs] = useModelState<RunMeta[]>("runs");
  const [arrayBytes] = useModelState<DataView>("arrays");
  const [bandBytes] = useModelState<number[]>("band_per_A");  // the band the runs' lattice power reads
  const [size, setSize] = useModelState<ViewSize>("size");
  const [peaks] = useModelState<number[][]>("peaks");
  const [line] = useModelState<number[]>("profile_line");
  const [smooth, setSmooth] = useModelState<boolean>("smooth");
  const engineContext = useEngine(!!smooth);
  const controls = useControls(colors, !!smooth, setSmooth);
  const all = runs || [];
  const sources = useStores(useReferenced("sources"));
  const ready = all.every((run) => run.source < 0 || sources[run.source]);
  const columns = React.useMemo(() => {
    const own = endItems(arrayBytes, all.flatMap((run) => run.items));
    let offset = 0;
    return all.map((run) => {
      const items = own.slice(offset, offset + run.items.length);
      offset += run.items.length;
      return columnOf(run, items, run.source >= 0 ? sources[run.source] || null : null);
    });
  }, [arrayBytes, all, ready, ...sources]);
  const [showPeaks, setShowPeaks] = React.useState(true);
  const [region, setRegion] = React.useState<Region | null>(null);
  // the depths kept, in Å from the entrance face: each column shows its kept slice nearest the depth shown
  const depthOf = (item: EndItem, run: RunMeta) => (item.depth_A ? (item.depth_A[0] + item.depth_A[1]) / 2 : (item.slice + 0.5) * (run.slice_A || 1));  // a run of unknown thickness: in slices
  const depths = React.useMemo(() => Array.from(new Set(columns.flatMap((column, index) => column.slices.map((item) => Math.round(depthOf(item, all[index])))))).sort((a, b) => a - b), [columns]);
  const [depth, setDepth] = React.useState<number | null>(null);
  const depthA = depth ?? depths[Math.floor(depths.length / 2)] ?? 0;
  const sliceOf = (column: ColumnData, run: RunMeta): EndItem | null => column.slices.length ? column.slices.reduce((best, item) => (Math.abs(depthOf(item, run) - depthA) < Math.abs(depthOf(best, run) - depthA) ? item : best)) : null;
  const shownSlices = all.map((run, index) => sliceOf(columns[index], run)?.data).filter((one): one is Float32Array => !!one);  // the columns' slices: one contrast once it is changed
  const difference = React.useMemo(() => {  // B − A on the projected phase
    if (columns.length !== 2 || !columns[0].projection || !columns[1].projection || columns[0].projection.shape.join() !== columns[1].projection.shape.join()) return null;
    const [rows, cols] = columns[0].projection.shape;
    return { ...differenceOf(columns[1].projection.data, columns[0].projection.data), rows, cols };
  }, [columns]);
  const [grid, gridWidth] = useWidth(1000);
  const perRow = gridWidth < 600 ? Math.min(2, all.length) : all.length;
  const columnWidth = (gridWidth - GAP * (perRow - 1)) / Math.max(1, perRow) - 12;
  const muted = { color: colors.textMuted } as React.CSSProperties;
  const number = (run: RunMeta, key: string, label: string, unit: string, digits: number) => {
    const value = run.record[key], base = all[0].record[key];
    if (value == null) return null;
    const delta = run !== all[0] && base != null ? Number(value) - Number(base) : null;
    return <span key={key} style={{ whiteSpace: "nowrap" }}>{label} <b>{plain(Number(value), digits)}{unit}</b>{delta != null && <span style={muted}> {delta >= 0 ? "+" : ""}{plain(delta, digits)}</span>}</span>;
  };
  if (!all.length) return null;
  const tallest = bySize(size, 220, 300, 520);
  const first = columns[0]?.slices[0];
  const imageHeight = Math.max(80, Math.min(tallest, Math.floor(columnWidth * (columnWidth * 0.4 - GAP >= 110 ? bySize(size, 0.6, 0.6, 0.98) : 1) * (first ? first.shape[0] / first.shape[1] : 1))));  // the slice takes 0.6 of the column with the probe beside it; a narrow column (or large) gives the slice the whole column, the probe under it
  // each run's raw lattice power per slice ÷ its slice thickness² (phase per slice grows with the thickness, its power with
  // the square), over the strongest of any run: a uniform crystal reads the same height however it is sliced
  const perA2 = (value: number, run: RunMeta) => value / (run.slice_A || 1) ** 2;
  const powerMax = Math.max(1e-30, ...columns.flatMap((column, index) => (column.power || []).map((value) => perA2(value, all[index]))));
  const bandText = bandBytes && bandBytes.length === 2 ? `${plain(bandBytes[0], 2)} to ${plain(bandBytes[1], 2)} 1/Å` : "the lattice's band";
  const unknownThickness = all.some((run) => !run.slice_A);
  return (
    <Engine.Provider value={engineContext}><ViewLink.Provider value={controls.link}>
    <div style={{ fontFamily: WIDGET_UI_FONT, fontSize: 12, color: colors.text, background: colors.bg, border: `1px solid ${colors.border}`,
      padding: GAP, borderRadius: 4, maxWidth: "100%", boxSizing: "border-box", fontVariantNumeric: "tabular-nums" }}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: `${GAP}px ${2 * GAP}px`, marginBottom: GAP }}>
        <span style={{ fontSize: 13, fontWeight: 600 }}>{all.map((run) => run.name).join(" vs ")}{title && <span style={{ fontWeight: 400, ...muted }}> · {title}</span>}</span>
        <RenderPathBadge colors={colors} />
        {controls.toggle}<SizeControl size={size} onSize={setSize} colors={colors} />
        <InfoTooltip theme={theme} toggleOnClick text={"One column a run, the first the reference: its numbers, then each other run's difference to it (muted ±). Each column shows its slice nearest the depth shown, from the slices a saved notebook keeps of each run (a quarter, half and three quarters through its slab for the newest run, the middle one for earlier runs; its own slicing, no resampling), its spectrum (linked region), its cut along the line through every slice, and its probe: the strongest mode in complex-wave colors (hue the phase, brightness the amplitude), the aperture (Fourier-space amplitude) and the other modes with their shares. Contrast: each run from its own floor (1st percentile) over one span per Å of slice thickness, since phase per slice grows with thickness. Below: lattice power per slice (spectrum power in the lattice's band, per Å² of slice) against depth, and for two runs B − A on the projected phase."} />
        <Box sx={{ ...controlRow, flexWrap: "wrap", border: `1px solid ${colors.border}`, bgcolor: colors.controlBg, gap: `${GAP}px`, ml: "auto" }}>
          {(peaks || []).length > 0 && switchControl("Columns", showPeaks, () => setShowPeaks(!showPeaks), "Rings on the atomic columns found", colors)}
          {switchControl("Smooth", !!smooth, () => setSmooth(!smooth), "CSS bilinear interpolation. Same data, the browser smooths visually.", colors)}
          {region && toggleButton(false, "Whole field", () => setRegion(null))}
        </Box>
      </div>
      {controls.row}
      <div ref={grid} style={{ display: "grid", gridTemplateColumns: `repeat(${perRow}, minmax(0, 1fr))`, gap: GAP }}>
        {all.map((run, index) => (
          <CompareColumn key={run.name + index} run={run} column={columns[index]} item={sliceOf(columns[index], run)} shownSlices={shownSlices} tallest={tallest} imageHeight={imageHeight} line={line}
            peaks={peaks || []} showPeaks={showPeaks} region={region} onRegion={setRegion} width={columnWidth} colors={colors} first={all[0]}
            numbers={[number(run, "held_out", "held-out", "", 5), number(run, "copies_percent", "copies", " %", 2), number(run, "r99_A", "r99", " Å", 2), number(run, "spots_outer", "outer spots", "", 0)]} />
        ))}
      </div>
      {depths.length > 1 && <Stepper positions={depths} value={depthA} onChange={setDepth} name="depth" colors={colors}
        label={<>depth {plain(depthA, 0)} Å <span style={muted}>· {all.map((run, index) => { const item = sliceOf(columns[index], run); return item ? `${run.name} slice ${item.slice + 1}/${item.slices}` : run.name; }).join(" · ")}</span></>} />}
      <div style={{ display: "grid", gridTemplateColumns: difference && gridWidth >= 600 ? "minmax(0, 1fr) auto" : "minmax(0, 1fr)", gap: GAP, marginTop: GAP }}>
        <Panel title="Lattice power" hint={`${bandText}, per Å² of slice`} colors={colors} theme={theme}
          info={`Each run's spectrum power in the lattice's band (${bandText}, the band the stage table's crystal thickness reads) for each slice, at the slice's centre depth, divided by the slice thickness squared (a slice's phase grows with its thickness, its power with the square, so differently sliced runs compare: a uniform crystal reads one height), relative to the strongest slice of any run; the line marks the depth shown.${unknownThickness ? " A run with no slab thickness is drawn per slice, its depth in slices." : ""}`}>
          <LineChart series={all.map((run, index) => ({ name: run.name, color: HUES[index % HUES.length], dashed: index >= HUES.length,
            points: (columns[index].power || []).map((value, slice) => [(slice + 0.5) * (run.slice_A || 1), perA2(value, run) / powerMax] as [number, number]) }))}
            stages={[]} cursor={depthA} height={difference ? Math.min(imageHeight, 200) : 120} outer xLabel={unknownThickness ? "depth (Å, or slices)" : "depth (Å)"} yLabel="relative per Å²" colors={colors}
            onPick={(value) => setDepth(depths.reduce((best, one) => Math.abs(one - value) < Math.abs(best - value) ? one : best, depths[0]))} />
        </Panel>
        {difference && <Panel title={`${all[1].name} − ${all[0].name}`} hint={<>projected phase, ±{plain(difference.span, 2)} rad</>} colors={colors} theme={theme}
          info="The second run's projected phase minus the first's, each with its mean removed (a constant phase offset is arbitrary); blue below zero, red above, white at zero.">
          <ImageCanvas data={difference.image} rows={difference.rows} cols={difference.cols} low={-difference.span} high={difference.span} cmap="RdBu_r" height={Math.min(imageHeight, 200)} keepAspect zoomable symmetric />
        </Panel>}
      </div>
    </div>
    </ViewLink.Provider></Engine.Provider>
  );
}

// One run's column: what it is and how its numbers differ from the first, its slice at the shared depth with the line
// and the spectrum beside it, the cut along the line, and its probe.
function CompareColumn({ run, column, item, imageHeight, line, peaks, showPeaks, region, onRegion, width, colors, numbers, first, shownSlices }: {
  run: RunMeta; column: ColumnData; item: EndItem | null; shownSlices: Float32Array[]; tallest: number; imageHeight: number; line: number[] | null | undefined; peaks: number[][]; showPeaks: boolean;
  region: Region | null; onRegion: (region: Region | null) => void; width: number; colors: any; numbers: React.ReactNode[]; first: RunMeta;
}) {
  const [rows, cols] = item ? item.shape : [0, 0];
  const data = item ? item.data : EMPTY_FRAME;
  const spectrum = useSpectrum(data, rows, cols, region, run.pixel_A);
  const beside = Math.floor(width - 12 - (rows ? imageHeight * cols / rows : 0) - GAP);  // what the column has left beside the slice: the cut and the probe go there, else under it
  const stacked = beside < 110;
  const cutHeight = Math.round(imageHeight / 4);
  const changed = run === first ? "reference" : differs(run);
  const ignore = () => undefined;
  return (
    <div style={{ borderRadius: 4, padding: `${GAP}px 6px 6px`, minWidth: 0, background: colors.bgAlt, display: "flex", flexDirection: "column", gap: GAP }}>
      <div style={{ lineHeight: "16px" }}>
        <div title={changed} style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}><b>{run.name}</b><span style={{ color: colors.textMuted, fontSize: TEXT }}> · {changed}</span></div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: `0 ${2 * GAP}px`, fontSize: TEXT }}>{numbers}</div>
        {item && <div style={{ fontSize: TEXT, color: colors.textMuted }}>slice {item.slice + 1}/{item.slices}, {plain(item.slice * run.slice_A, 0)}-{plain((item.slice + 1) * run.slice_A, 0)} Å · {plain(run.slice_A, 1)} Å slices</div>}
      </div>
      <div style={{ display: "flex", flexWrap: stacked ? "wrap" : "nowrap", gap: GAP, alignItems: "flex-start" }}>
        <div style={{ flex: "none", maxWidth: "100%" }}>
          <FieldView data={data} rows={rows} cols={cols} low={run.low} high={run.high} pixelA={run.pixel_A} height={imageHeight}
            line={line} editLine={false} onDragLine={ignore} onLine={ignore} peaks={peaks} showPeaks={showPeaks} region={region} onRegion={onRegion} colors={colors} rangeFrom={shownSlices}
            inset={<SpectrumInset spectrum={spectrum} side={imageHeight} label={region ? `${plain(region.size * run.pixel_A, 0)} Å box` : "whole field"} />} />
        </div>
        {column.probe && <ProbeView frame={column.probe.data} shape={column.probe.shape} shares={column.probe.shares} aperture={column.aperture}
          size={Math.round(0.75 * imageHeight)} maxWidth={width} colors={colors} box={stacked ? undefined : [beside, imageHeight]} />}
      </div>
      {column.cut && <div style={{ position: "relative" }}>
        <ImageCanvas data={column.cut.data} rows={column.cut.shape[0]} cols={column.cut.shape[1]} low={run.low} high={run.high} cmap="gray"
          line={item ? [item.slice + 0.5, 0, item.slice + 0.5, column.cut.shape[1]] : undefined} lineColor={colors.accent} height={cutHeight} />
        <div style={{ ...ON_IMAGE, left: 3, bottom: 1 }}>cut along the line, {column.cut.shape[0]} slices</div>
      </div>}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------
// Direct ptychography at a glance (ShowDirect)

// A scale bar of a round length near a quarter of the field: 1, 2 or 5 × a power of ten, in Å.
function niceBar(fieldA: number): number {
  const raw = fieldA / 4, magnitude = 10 ** Math.floor(Math.log10(raw)), norm = raw / magnitude;
  return (norm < 2 ? 1 : norm < 5 ? 2 : 5) * magnitude;
}

// The scan region as the draggable box keeps it (square_region in Python): whole scan positions, square, an even side
// of at least 2, inside the scan.
export function squareRegion(region: number[], rows: number, cols: number): number[] {
  const [r0, r1, c0, c1] = region.map((value) => Math.round(value));
  const side = Math.max(2, Math.floor(Math.min(r1 - r0, c1 - c0, rows, cols) / 2) * 2);
  const top = Math.min(Math.max(0, r0), rows - side), left = Math.min(Math.max(0, c0), cols - side);
  return [top, top + side, left, left + side];
}

// The region's box over the whole scan, dragged: inside it moves, near its lower-right corner it resizes, kept square on
// whole scan positions; the dropped box is written to the model (region), which Python reads back.
function RegionBox({ region, rows, cols, width, height, onDrag, onDrop, colors }: {
  region: number[]; rows: number; cols: number; width: number; height: number; onDrag: (region: number[] | null) => void; onDrop: (region: number[]) => void; colors: any;
}) {
  const box = React.useRef<HTMLDivElement>(null);
  const drag = React.useRef<{ part: "move" | "size"; row: number; col: number; start: number[]; now: number[] } | null>(null);
  const at = (event: React.PointerEvent) => {
    const rect = box.current!.getBoundingClientRect();
    return [((event.clientY - rect.top) / rect.height) * rows, ((event.clientX - rect.left) / rect.width) * cols];
  };
  const [r0, r1, c0, c1] = region;
  const scale = width / cols;
  const onDown = (event: React.PointerEvent) => {
    const [row, col] = at(event);
    const near = 10 / scale;
    const part = Math.abs(row - r1) < near && Math.abs(col - c1) < near ? "size" : row >= r0 && row <= r1 && col >= c0 && col <= c1 ? "move" : null;
    if (!part) return;
    event.preventDefault();
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    drag.current = { part, row, col, start: [...region], now: [...region] };
  };
  const onMove = (event: React.PointerEvent) => {
    const held = drag.current;
    if (!held) return;
    const [row, col] = at(event);
    const [s0, s1, t0, t1] = held.start;
    const next = held.part === "move"
      ? [s0 + row - held.row, s1 + row - held.row, t0 + col - held.col, t1 + col - held.col]
      : (() => { const side = Math.max(row - s0, col - t0); return [s0, s0 + side, t0, t0 + side]; })();
    held.now = held.part === "move" ? (() => { const side = s1 - s0; const top = Math.min(Math.max(0, Math.round(next[0])), rows - side), left = Math.min(Math.max(0, Math.round(next[2])), cols - side); return [top, top + side, left, left + side]; })()
      : squareRegion(next, rows, cols);
    onDrag(held.now);
  };
  const onUp = () => {
    const held = drag.current;
    if (!held) return;
    drag.current = null;
    onDrag(null);
    onDrop(held.now);
  };
  return (
    <div ref={box} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}
      style={{ position: "absolute", left: 0, top: 0, width, height, touchAction: "none", cursor: drag.current ? "grabbing" : "default" }}>
      <div style={{ position: "absolute", left: c0 * scale, top: r0 * (height / rows), width: (c1 - c0) * scale, height: (r1 - r0) * (height / rows),
        border: `2px solid ${colors.accent}`, boxSizing: "border-box", cursor: "move" }} title="drag to move; drag the corner to resize (kept square)">
        <div style={{ position: "absolute", right: -6, bottom: -6, width: 12, height: 12, background: colors.accent, borderRadius: 2, cursor: "nwse-resize" }} />
      </div>
    </div>
  );
}

function DirectView() {
  const { colors, theme } = usePalette();
  const [title] = useModelState<string>("title");
  const [bytes] = useModelState<DataView>("images");
  const [shapes] = useModelState<number[][]>("image_shapes");
  const [scanA] = useModelState<number>("scan_sampling_A");
  const [upsample] = useModelState<number>("upsample");
  const [region, setRegion] = useModelState<number[]>("region");
  const [phaseRegion] = useModelState<number[]>("phase_region");
  const [readout] = useModelState<{ [name: string]: { tilt_mrad?: number[]; defocus_nm?: number } }>("readout");
  const [dark, setDark] = useModelState<boolean>("dark");
  const [editable] = useModelState<boolean>("editable");
  const [origin] = useModelState<number[]>("origin");  // the virtual images' first scan position: a crop about the region, or (0, 0)
  const [size, setSize] = useModelState<ViewSize>("size");
  const [root, rootWidth] = useWidth(1000);
  const [dragging, setDragging] = React.useState<number[] | null>(null);
  const [copied, setCopied] = React.useState(false);
  const [smoothDirect, setSmoothDirect] = React.useState(false);
  const controls = useControls(colors, smoothDirect, setSmoothDirect);
  const engineContext = useEngine(smoothDirect);
  const images = React.useMemo(() => {
    const all = floats(bytes);
    let offset = 0;
    return (shapes || []).map((shape) => { const length = shape[0] * shape[1]; const data = all.subarray(offset, offset + length); offset += length; return { data, rows: shape[0], cols: shape[1] }; });
  }, [bytes, shapes]);
  const [bright, darkField, phase] = images;
  const virtual = dark ? darkField : bright;
  const virtualRange = React.useMemo(() => rangeOf(virtual ? virtual.data : EMPTY_FRAME), [virtual?.data]);
  const other = phase ? null : dark ? bright : darkField;  // the whole scan alone: both fields side by side, the box dragged on the first and drawn on the second
  const otherRange = React.useMemo(() => rangeOf(other ? other.data : EMPTY_FRAME), [other?.data]);
  const phaseRange = React.useMemo(() => rangeOf(phase ? phase.data : EMPTY_FRAME), [phase?.data]);
  const phaseA = scanA / Math.max(1, upsample || 1);
  const spectrum = useSpectrum(phase ? phase.data : EMPTY_FRAME, phase ? phase.rows : 0, phase ? phase.cols : 0, null, phaseA);
  const across = bright ? bright.cols / bright.rows + (phase ? phase.cols / phase.rows : other ? other.cols / other.rows : 0) : 2;
  const height = Math.min(bySize(size, 200, 420, 640), Math.max(120, Math.floor((rootWidth - 24) / across)));  // side by side
  const muted = { color: colors.textMuted } as React.CSSProperties;
  if (!bright) return null;
  const shown = dragging || region || [0, bright.rows, 0, bright.cols];
  const [r0, r1, c0, c1] = shown;
  const [p0, , q0] = phaseRegion || [0, bright.rows, 0, bright.cols];
  const [o0, o1] = origin || [0, 0];
  const cropped = o0 !== 0 || o1 !== 0 || (phaseRegion && bright.rows === phaseRegion[1] - phaseRegion[0] && !editable);
  const text = `scan_region = (${r0}, ${r1}, ${c0}, ${c1})`;
  const box = (top: number, left: number, rows: number, cols: number) => <rect x={left} y={top} width={cols} height={rows} fill="none" stroke={colors.accent} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />;
  const optics = (name: string) => {
    const one = readout && readout[name];
    if (!one) return null;
    const tilt = one.tilt_mrad ? `tilt ${one.tilt_mrad.map((value) => `${value >= 0 ? "+" : ""}${plain(value, 2)}`).join(", ")} mrad` : "";
    const focus = one.defocus_nm != null ? `defocus ${one.defocus_nm >= 0 ? "+" : ""}${plain(one.defocus_nm, 1)} nm` : "";
    return <span key={name} style={{ whiteSpace: "nowrap" }}><b>{name}</b> {[tilt, focus].filter(Boolean).join(", ")}</span>;
  };
  const picture = (image: { data: Float32Array; rows: number; cols: number }, range: number[], pixelA: number, label: string, overlay: React.ReactNode, children?: React.ReactNode, zoomable = false, inset?: React.ReactNode) => {
    const width = Math.round(height * image.cols / image.rows);
    return (
      <div style={{ position: "relative", height, width, flex: "none" }}>
        <ImageCanvas data={image.data} rows={image.rows} cols={image.cols} low={range[0]} high={range[1]} cmap="gray" height={height} keepAspect pixelA={pixelA}
          barA={niceBar(image.cols * pixelA)} overlay={overlay} zoomable={zoomable} unit={label.includes("phase") ? "rad" : "counts"} inset={inset} />
        <div style={{ ...ON_IMAGE, ...PLATE, left: 3, top: 2 }}>{label}</div>
        {children}
      </div>
    );
  };
  const copy = () => { navigator.clipboard?.writeText(text).then(() => { setCopied(true); window.setTimeout(() => setCopied(false), 1200); }).catch(() => undefined); };
  return (
    <Engine.Provider value={engineContext}><ViewLink.Provider value={controls.link}>
    <div ref={root} style={header(colors)}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: `0 ${3 * GAP}px`, lineHeight: "18px", marginBottom: GAP }}>
        <span style={{ fontWeight: 600, fontSize: 13 }}>{title || "Direct ptychography"}<span style={{ fontWeight: 400, ...muted }}> · region {r1 - r0} × {c1 - c0} positions at ({r0}, {c0})</span></span>
        {editable && <span style={{ whiteSpace: "nowrap" }}><code style={{ fontSize: 12, userSelect: "all" }} title="the region, as code: copy it into the cell to keep it">{text}</code>{" "}
          <button type="button" onClick={copy} style={{ border: `1px solid ${colors.border}`, background: "transparent", color: colors.textMuted, fontSize: 10, borderRadius: 3, cursor: "pointer" }}>{copied ? "copied" : "copy"}</button></span>}
        {optics("region")}
        {optics("whole scan")}
        <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: GAP }}>
          {toggleButton(!dark, "Bright field", () => setDark(false))}{toggleButton(!!dark, "Dark field", () => setDark(true))}
          <RenderPathBadge colors={colors} />{controls.toggle}<SizeControl size={size} onSize={setSize} colors={colors} />
          <InfoTooltip theme={theme} toggleOnClick text={editable
            ? "The whole scan's virtual bright field (or annular dark field) and the region to reconstruct: drag the box to move it, drag its corner to resize it (kept square, on whole scan positions, an even side). The dropped box is whole_scan.scan_region, which whole_scan.fit() fits; typing whole_scan.scan_region = (...) and showing again moves the box. The line above is the region as code, to copy into the cell."
            : "Left: the whole scan's virtual bright field (or annular dark field), the box the scan positions reconstructed and scored. Middle: direct ptychography's phase of the region around them, the same box drawn on it. Right: that phase's spectrum (Hann-windowed power, log scale, ±2.6 1/Å, the dashed ring at 50 pm). The line above: the tilt and defocus direct ptychography fitted on the region, beside the whole scan's; the iterative reconstruction starts from the region's."} />
        </span>
      </div>
      {controls.row}
      <div style={{ display: "flex", flexWrap: "wrap", gap: HAIR, alignItems: "flex-start" }}>
        {picture(virtual!, virtualRange, scanA, `${cropped ? "around the region" : "whole scan"}, ${dark ? "dark field" : "bright field"}`, editable ? null : box(r0 - o0, c0 - o1, r1 - r0, c1 - c0),
          editable ? <RegionBox region={shown} rows={virtual!.rows} cols={virtual!.cols} width={Math.round(height * virtual!.cols / virtual!.rows)} height={height}
            onDrag={setDragging} onDrop={(next) => setRegion(next)} colors={colors} /> : undefined)}
        {other && picture(other, otherRange, scanA, `whole scan, ${dark ? "bright field" : "dark field"}`, box(r0 - o0, c0 - o1, r1 - r0, c1 - c0))}
        {phase && picture(phase, phaseRange, phaseA, "region, direct phase", box((r0 - p0) * (upsample || 1), (c0 - q0) * (upsample || 1), (r1 - r0) * (upsample || 1), (c1 - c0) * (upsample || 1)), undefined, true, <SpectrumInset spectrum={spectrum} side={height} label="region's phase" />)}
      </div>
    </div>
    </ViewLink.Provider></Engine.Provider>
  );
}

// One bundle, three views: a run's, the dashboard, and the compare view.
function Root() {
  const [kind] = useModelState<string>("kind");
  const [view] = useModelState<string>("view");
  if (kind === "direct") return <DirectView />;
  return kind === "compare" ? <CompareReconstruction /> : view === "dashboard" ? <DashboardView /> : view === "report" ? <ReportView /> : <RunView />;
}

export default { render: createRender(Root) };
