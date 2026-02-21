/**
 * Show1D - Interactive 1D data viewer for spectra, profiles, and time series.
 * Self-contained widget with all utilities inlined.
 *
 * Features:
 * - Scroll to zoom, drag to pan, double-click to reset
 * - Multi-trace overlay with distinct colors and legend
 * - Cursor crosshair with snapped value readout
 * - Calibrated X/Y axes with nice tick values
 * - Log scale, grid lines, stats bar
 * - Publication-quality figure export
 * - Automatic theme detection (light/dark mode)
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Switch from "@mui/material/Switch";
import Button from "@mui/material/Button";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Tooltip from "@mui/material/Tooltip";
import "./styles.css";
import { useTheme } from "../theme";
import { roundToNiceValue } from "../scalebar";
import { extractFloat32, formatNumber, downloadBlob } from "../format";
import { findDataRange } from "../stats";

// ============================================================================
// UI Styles (per-widget, matching Show3D/Show4DSTEM)
// ============================================================================
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const },
};

const SPACING = {
  XS: 4,
  SM: 8,
  MD: 12,
  LG: 16,
};

const switchStyles = {
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};

const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: "6px",
  px: 1,
  py: 0.5,
  width: "fit-content",
};

const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

const container = {
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "monospace", overflow: "visible" },
};

// ============================================================================
// Constants
// ============================================================================
const DPR = window.devicePixelRatio || 1;
const RESIZE_HIT_AREA_PX = 10;
const UNFOCUSED_ALPHA = 0.2;

/** Snap a coordinate to the nearest pixel boundary for crisp 1px lines. */
function snap(v: number): number {
  return Math.round(v) + 0.5;
}
const DEFAULT_CANVAS_W = 500;
const DEFAULT_CANVAS_H = 300;
const MARGIN = { top: 12, right: 16, bottom: 48, left: 60 };
const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
const AXIS_TICK_PX = 4;
const MAX_TICKS_Y = 8;
const TICK_LABEL_WIDTH_PX = 55;
const Y_PAD_FRAC = 0.05;

// ============================================================================
// InfoTooltip (per-widget, matching Show3D)
// ============================================================================
function InfoTooltip({ text, theme = "dark" }: { text: React.ReactNode; theme?: "light" | "dark" }) {
  const isDark = theme === "dark";
  const content = typeof text === "string"
    ? <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>{text}</Typography>
    : text;
  return (
    <Tooltip
      title={content}
      arrow
      placement="bottom"
      componentsProps={{
        tooltip: {
          sx: {
            bgcolor: isDark ? "#333" : "#fff",
            color: isDark ? "#ddd" : "#333",
            border: `1px solid ${isDark ? "#555" : "#ccc"}`,
            maxWidth: 280,
            p: 1,
          },
        },
        arrow: {
          sx: {
            color: isDark ? "#333" : "#fff",
            "&::before": { border: `1px solid ${isDark ? "#555" : "#ccc"}` },
          },
        },
      }}
    >
      <Typography
        component="span"
        sx={{
          fontSize: 12,
          color: isDark ? "#888" : "#666",
          cursor: "help",
          ml: 0.5,
          "&:hover": { color: isDark ? "#aaa" : "#444" },
        }}
      >
        ⓘ
      </Typography>
    </Tooltip>
  );
}

function KeyboardShortcuts({ items }: { items: [string, string][] }) {
  return (
    <Box component="table" sx={{ borderCollapse: "collapse", "& td": { py: 0.25, fontSize: 11, lineHeight: 1.3, verticalAlign: "top" }, "& td:first-of-type": { pr: 1.5, opacity: 0.7, fontFamily: "monospace", fontSize: 10, whiteSpace: "nowrap" } }}>
      <tbody>
        {items.map(([key, desc], i) => (
          <tr key={i}><td>{key}</td><td>{desc}</td></tr>
        ))}
      </tbody>
    </Box>
  );
}

// ============================================================================
// Tick computation
// ============================================================================
function computeTicks(min: number, max: number, maxTicks: number = MAX_TICKS_Y): number[] {
  const range = max - min;
  if (range <= 0 || !isFinite(range)) return [min];
  const step = roundToNiceValue(range / maxTicks);
  if (step <= 0 || !isFinite(step)) return [min, max];
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 0.001; v += step) {
    if (v >= min - step * 0.001) ticks.push(v);
  }
  if (ticks.length === 0) ticks.push(min, max);
  return ticks;
}

function computeLogTicks(min: number, max: number): number[] {
  const logMin = Math.floor(Math.log10(Math.max(min, 1e-30)));
  const logMax = Math.ceil(Math.log10(Math.max(max, 1e-30)));
  const ticks: number[] = [];
  for (let e = logMin; e <= logMax; e++) {
    const val = Math.pow(10, e);
    if (val >= min && val <= max) ticks.push(val);
  }
  if (ticks.length === 0) ticks.push(min, max);
  return ticks;
}

// ============================================================================
// Main Component
// ============================================================================
function Show1D() {
  // Theme
  const { themeInfo, colors } = useTheme();
  const isDark = themeInfo.theme === "dark";

  // Model state (synced from Python)
  const [yData] = useModelState<DataView>("y_bytes");
  const [xData] = useModelState<DataView>("x_bytes");
  const [nTraces] = useModelState<number>("n_traces");
  const [nPoints] = useModelState<number>("n_points");
  const [traceLabels] = useModelState<string[]>("labels");
  const [traceColors] = useModelState<string[]>("colors");
  const [title] = useModelState<string>("title");
  const [xLabel] = useModelState<string>("x_label");
  const [yLabel] = useModelState<string>("y_label");
  const [xUnit] = useModelState<string>("x_unit");
  const [yUnit] = useModelState<string>("y_unit");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [showStats] = useModelState<boolean>("show_stats");
  const [showLegend, setShowLegend] = useModelState<boolean>("show_legend");
  const [showGrid, setShowGrid] = useModelState<boolean>("show_grid");
  const [showControls] = useModelState<boolean>("show_controls");
  const [lineWidth] = useModelState<number>("line_width");
  const [statsMean] = useModelState<number[]>("stats_mean");
  const [statsMin] = useModelState<number[]>("stats_min");
  const [statsMax] = useModelState<number[]>("stats_max");
  const [statsStd] = useModelState<number[]>("stats_std");
  const [focusedTrace, setFocusedTrace] = useModelState<number>("focused_trace");

  // Canvas refs
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const uiCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const containerRef = React.useRef<HTMLDivElement>(null);

  // Canvas size
  const [canvasW, setCanvasW] = React.useState(DEFAULT_CANVAS_W);
  const [canvasH, setCanvasH] = React.useState(DEFAULT_CANVAS_H);

  // View state (data coordinate ranges)
  const [xMin, setXMin] = React.useState(0);
  const [xMax, setXMax] = React.useState(1);
  const [yMin, setYMin] = React.useState(0);
  const [yMax, setYMax] = React.useState(1);

  // Data ranges (for reset)
  const xDataRangeRef = React.useRef({ min: 0, max: 1 });
  const yDataRangeRef = React.useRef({ min: 0, max: 1 });

  // Extracted data
  const tracesRef = React.useRef<Float32Array[]>([]);
  const xValuesRef = React.useRef<Float32Array | null>(null);

  // Cursor state
  const [cursorInfo, setCursorInfo] = React.useState<{
    canvasX: number;
    canvasY: number;
    dataX: number;
    dataY: number;
    traceIdx: number;
    label: string;
    color: string;
  } | null>(null);

  // Drag state
  const dragRef = React.useRef<{
    active: boolean;
    wasDrag: boolean;
    startX: number;
    startY: number;
    startXMin: number;
    startXMax: number;
    startYMin: number;
    startYMax: number;
  } | null>(null);

  // Resize handle state
  const resizeDragRef = React.useRef<{
    active: boolean;
    startX: number;
    startY: number;
    startW: number;
    startH: number;
  } | null>(null);

  // Export
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);

  // Plot area dimensions
  const plotW = canvasW - MARGIN.left - MARGIN.right;
  const plotH = canvasH - MARGIN.top - MARGIN.bottom;

  // ========================================================================
  // Data extraction
  // ========================================================================
  React.useEffect(() => {
    if (!yData || yData.byteLength < 4 || nTraces < 1 || nPoints < 1) {
      tracesRef.current = [];
      return;
    }
    const allY = extractFloat32(yData);
    if (!allY) {
      tracesRef.current = [];
      return;
    }
    const traces: Float32Array[] = [];
    for (let t = 0; t < nTraces; t++) {
      const offset = t * nPoints;
      traces.push(allY.slice(offset, offset + nPoints));
    }
    tracesRef.current = traces;

    // X values
    if (xData && xData.byteLength >= 4) {
      xValuesRef.current = extractFloat32(xData);
    } else {
      xValuesRef.current = null;
    }

    // Compute data ranges
    let gXMin = 0;
    let gXMax = nPoints - 1;
    if (xValuesRef.current && xValuesRef.current.length > 0) {
      const xRange = findDataRange(xValuesRef.current);
      gXMin = xRange.min;
      gXMax = xRange.max;
    }
    if (gXMin === gXMax) gXMax = gXMin + 1;

    let gYMin = Infinity;
    let gYMax = -Infinity;
    for (const trace of traces) {
      const r = findDataRange(trace);
      if (r.min < gYMin) gYMin = r.min;
      if (r.max > gYMax) gYMax = r.max;
    }
    if (!isFinite(gYMin)) gYMin = 0;
    if (!isFinite(gYMax)) gYMax = 1;
    if (gYMin === gYMax) { gYMin -= 0.5; gYMax += 0.5; }

    // Pad Y range
    const yPad = (gYMax - gYMin) * Y_PAD_FRAC;
    gYMin -= yPad;
    gYMax += yPad;

    xDataRangeRef.current = { min: gXMin, max: gXMax };
    yDataRangeRef.current = { min: gYMin, max: gYMax };

    setXMin(gXMin);
    setXMax(gXMax);
    setYMin(gYMin);
    setYMax(gYMax);
  }, [yData, xData, nTraces, nPoints]);

  // ========================================================================
  // Coordinate transforms
  // ========================================================================
  const dataToCanvasX = React.useCallback(
    (dx: number) => MARGIN.left + ((dx - xMin) / (xMax - xMin)) * plotW,
    [xMin, xMax, plotW],
  );
  const dataToCanvasY = React.useCallback(
    (dy: number) => {
      if (logScale) {
        const lMin = Math.log10(Math.max(yMin, 1e-30));
        const lMax = Math.log10(Math.max(yMax, 1e-30));
        const lVal = Math.log10(Math.max(dy, 1e-30));
        return MARGIN.top + plotH - ((lVal - lMin) / (lMax - lMin || 1)) * plotH;
      }
      return MARGIN.top + plotH - ((dy - yMin) / (yMax - yMin || 1)) * plotH;
    },
    [yMin, yMax, plotH, logScale],
  );
  const canvasToDataX = React.useCallback(
    (cx: number) => xMin + ((cx - MARGIN.left) / plotW) * (xMax - xMin),
    [xMin, xMax, plotW],
  );
  const canvasToDataY = React.useCallback(
    (cy: number) => {
      const frac = (MARGIN.top + plotH - cy) / plotH;
      if (logScale) {
        const lMin = Math.log10(Math.max(yMin, 1e-30));
        const lMax = Math.log10(Math.max(yMax, 1e-30));
        return Math.pow(10, lMin + frac * (lMax - lMin));
      }
      return yMin + frac * (yMax - yMin);
    },
    [yMin, yMax, plotH, logScale],
  );

  // ========================================================================
  // Reset view
  // ========================================================================
  const resetView = React.useCallback(() => {
    setXMin(xDataRangeRef.current.min);
    setXMax(xDataRangeRef.current.max);
    setYMin(yDataRangeRef.current.min);
    setYMax(yDataRangeRef.current.max);
  }, []);

  // ========================================================================
  // Main canvas render
  // ========================================================================
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = canvasW * DPR;
    canvas.height = canvasH * DPR;
    ctx.scale(DPR, DPR);

    // Background
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f8f8f8";
    ctx.fillRect(0, 0, canvasW, canvasH);

    if (plotW <= 0 || plotH <= 0) return;

    const traces = tracesRef.current;
    const xVals = xValuesRef.current;

    // Grid
    if (showGrid) {
      ctx.strokeStyle = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);

      // X grid
      const xTicks = computeTicks(xMin, xMax, Math.max(3, Math.floor(plotW / TICK_LABEL_WIDTH_PX)));
      for (const tv of xTicks) {
        const cx = snap(dataToCanvasX(tv));
        if (cx >= MARGIN.left && cx <= MARGIN.left + plotW) {
          ctx.beginPath();
          ctx.moveTo(cx, MARGIN.top);
          ctx.lineTo(cx, MARGIN.top + plotH);
          ctx.stroke();
        }
      }

      // Y grid
      const yTicks = logScale ? computeLogTicks(Math.max(yMin, 1e-30), yMax) : computeTicks(yMin, yMax);
      for (const tv of yTicks) {
        const cy = snap(dataToCanvasY(tv));
        if (cy >= MARGIN.top && cy <= MARGIN.top + plotH) {
          ctx.beginPath();
          ctx.moveTo(MARGIN.left, cy);
          ctx.lineTo(MARGIN.left + plotW, cy);
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);
    }

    // Axes (pixel-snapped for crisp lines)
    ctx.strokeStyle = isDark ? "#666" : "#999";
    ctx.lineWidth = 1;
    const axisLeft = snap(MARGIN.left);
    const axisBottom = snap(MARGIN.top + plotH);
    ctx.beginPath();
    ctx.moveTo(axisLeft, MARGIN.top);
    ctx.lineTo(axisLeft, axisBottom);
    ctx.lineTo(MARGIN.left + plotW, axisBottom);
    ctx.stroke();

    // X ticks + labels
    const xTicks = computeTicks(xMin, xMax, Math.max(3, Math.floor(plotW / TICK_LABEL_WIDTH_PX)));
    ctx.fillStyle = isDark ? "#aaa" : "#555";
    ctx.font = `11px ${FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const tv of xTicks) {
      const cx = snap(dataToCanvasX(tv));
      if (cx >= MARGIN.left && cx <= MARGIN.left + plotW) {
        ctx.beginPath();
        ctx.moveTo(cx, MARGIN.top + plotH);
        ctx.lineTo(cx, MARGIN.top + plotH + AXIS_TICK_PX);
        ctx.stroke();
        ctx.fillText(formatNumber(tv), cx, MARGIN.top + plotH + AXIS_TICK_PX + 2);
      }
    }

    // Y ticks + labels
    const yTicks = logScale ? computeLogTicks(Math.max(yMin, 1e-30), yMax) : computeTicks(yMin, yMax);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const tv of yTicks) {
      const cy = snap(dataToCanvasY(tv));
      if (cy >= MARGIN.top && cy <= MARGIN.top + plotH) {
        ctx.beginPath();
        ctx.moveTo(MARGIN.left - AXIS_TICK_PX, cy);
        ctx.lineTo(MARGIN.left, cy);
        ctx.stroke();
        ctx.fillText(formatNumber(tv), MARGIN.left - AXIS_TICK_PX - 2, cy);
      }
    }

    // X axis label
    if (xLabel || xUnit) {
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.font = `12px ${FONT}`;
      ctx.fillStyle = isDark ? "#999" : "#666";
      let lbl = xLabel || "";
      if (xUnit) lbl += lbl ? ` (${xUnit})` : xUnit;
      ctx.fillText(lbl, MARGIN.left + plotW / 2, canvasH - 6);
    }

    // Y axis label (rotated)
    if (yLabel || yUnit) {
      ctx.save();
      ctx.translate(12, MARGIN.top + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.font = `12px ${FONT}`;
      ctx.fillStyle = isDark ? "#999" : "#666";
      let lbl = yLabel || "";
      if (yUnit) lbl += lbl ? ` (${yUnit})` : yUnit;
      ctx.fillText(lbl, 0, 0);
      ctx.restore();
    }

    // Clip to plot area for traces
    ctx.save();
    ctx.beginPath();
    ctx.rect(MARGIN.left, MARGIN.top, plotW, plotH);
    ctx.clip();

    // Draw traces (unfocused first, then focused on top)
    const hasFocus = focusedTrace >= 0 && focusedTrace < traces.length;
    const drawTrace = (t: number, alpha: number, lw: number) => {
      const trace = traces[t];
      const color = (traceColors && traceColors[t]) || "#4fc3f7";
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < trace.length; i++) {
        const xv = xVals ? xVals[i] : i;
        const yv = trace[i];
        if (!isFinite(yv) || (logScale && yv <= 0)) continue;
        const cx = dataToCanvasX(xv);
        const cy = dataToCanvasY(yv);
        if (!started) { ctx.moveTo(cx, cy); started = true; }
        else ctx.lineTo(cx, cy);
      }
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    };

    const baseLW = lineWidth || 1.5;
    if (hasFocus) {
      // Draw unfocused traces first (dimmed)
      for (let t = 0; t < traces.length; t++) {
        if (t !== focusedTrace) drawTrace(t, UNFOCUSED_ALPHA, baseLW);
      }
      // Draw focused trace on top (full opacity, thicker)
      drawTrace(focusedTrace, 1.0, baseLW * 1.5);
    } else {
      // No focus — all traces at full opacity
      for (let t = 0; t < traces.length; t++) {
        drawTrace(t, 1.0, baseLW);
      }
    }

    ctx.restore();

    // Resize handle triangle (bottom-right corner)
    const rhSize = 10;
    const rhX = canvasW;
    const rhY = canvasH;
    ctx.beginPath();
    ctx.moveTo(rhX, rhY);
    ctx.lineTo(rhX - rhSize, rhY);
    ctx.lineTo(rhX, rhY - rhSize);
    ctx.closePath();
    ctx.fillStyle = isDark ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.15)";
    ctx.fill();
  }, [canvasW, canvasH, xMin, xMax, yMin, yMax, yData, xData, nTraces, nPoints, traceColors, lineWidth, logScale, showGrid, xLabel, yLabel, xUnit, yUnit, isDark, dataToCanvasX, dataToCanvasY, plotW, plotH, focusedTrace]);

  // ========================================================================
  // UI overlay (crosshair, legend)
  // ========================================================================
  React.useEffect(() => {
    const uiCanvas = uiCanvasRef.current;
    if (!uiCanvas) return;
    const ctx = uiCanvas.getContext("2d");
    if (!ctx) return;

    uiCanvas.width = canvasW * DPR;
    uiCanvas.height = canvasH * DPR;
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, canvasW, canvasH);

    if (plotW <= 0 || plotH <= 0) return;

    // Crosshair
    if (cursorInfo) {
      const { canvasX, canvasY } = cursorInfo;
      if (canvasX >= MARGIN.left && canvasX <= MARGIN.left + plotW &&
          canvasY >= MARGIN.top && canvasY <= MARGIN.top + plotH) {
        ctx.strokeStyle = isDark ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.25)";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);

        // Vertical (pixel-snapped)
        const snapCX = snap(canvasX);
        ctx.beginPath();
        ctx.moveTo(snapCX, MARGIN.top);
        ctx.lineTo(snapCX, MARGIN.top + plotH);
        ctx.stroke();

        // Horizontal (pixel-snapped)
        const snapCY = snap(canvasY);
        ctx.beginPath();
        ctx.moveTo(MARGIN.left, snapCY);
        ctx.lineTo(MARGIN.left + plotW, snapCY);
        ctx.stroke();

        ctx.setLineDash([]);

        // Snap dot
        ctx.fillStyle = cursorInfo.color;
        ctx.beginPath();
        ctx.arc(canvasX, dataToCanvasY(cursorInfo.dataY), 4, 0, Math.PI * 2);
        ctx.fill();

        // Readout box
        const xStr = formatNumber(cursorInfo.dataX);
        const yStr = formatNumber(cursorInfo.dataY);
        let readout = `${xStr}, ${yStr}`;
        if (cursorInfo.label) readout = `${cursorInfo.label}: ${readout}`;

        ctx.font = `10px monospace`;
        const textW = ctx.measureText(readout).width;
        const boxPad = 4;
        const boxW = textW + boxPad * 2;
        const boxH = 16;
        let boxX = canvasX + 10;
        let boxY = canvasY - boxH - 6;
        if (boxX + boxW > MARGIN.left + plotW) boxX = canvasX - boxW - 10;
        if (boxY < MARGIN.top) boxY = canvasY + 10;

        ctx.fillStyle = isDark ? "rgba(30,30,30,0.9)" : "rgba(255,255,255,0.9)";
        ctx.fillRect(boxX, boxY, boxW, boxH);
        ctx.strokeStyle = isDark ? "#555" : "#ccc";
        ctx.lineWidth = 1;
        ctx.strokeRect(boxX, boxY, boxW, boxH);
        ctx.fillStyle = isDark ? "#eee" : "#333";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(readout, boxX + boxPad, boxY + boxH / 2);
      }
    }

    // Legend (with focus state)
    const hasFocusLegend = focusedTrace >= 0 && focusedTrace < nTraces;
    if (showLegend && nTraces > 1 && traceLabels && traceLabels.length > 0) {
      const entryH = 14;
      const lineLen = 16;
      const gap = 4;
      const legendPad = 6;
      let maxLabelW = 0;
      for (let t = 0; t < nTraces; t++) {
        const lbl = traceLabels[t] || `Trace ${t + 1}`;
        ctx.font = hasFocusLegend && t === focusedTrace ? `bold 11px ${FONT}` : `11px ${FONT}`;
        const w = ctx.measureText(lbl).width;
        if (w > maxLabelW) maxLabelW = w;
      }
      const legendW = legendPad * 2 + lineLen + gap + maxLabelW;
      const legendH = legendPad * 2 + nTraces * entryH;
      const lx = MARGIN.left + plotW - legendW - 8;
      const ly = MARGIN.top + 8;

      ctx.fillStyle = isDark ? "rgba(30,30,30,0.85)" : "rgba(255,255,255,0.85)";
      ctx.fillRect(lx, ly, legendW, legendH);
      ctx.strokeStyle = isDark ? "#555" : "#ccc";
      ctx.lineWidth = 1;
      ctx.strokeRect(lx, ly, legendW, legendH);

      for (let t = 0; t < nTraces; t++) {
        const ey = ly + legendPad + t * entryH + entryH / 2;
        const color = (traceColors && traceColors[t]) || "#4fc3f7";
        const isFocused = hasFocusLegend && t === focusedTrace;
        const dimmed = hasFocusLegend && !isFocused;

        // Color line
        ctx.globalAlpha = dimmed ? UNFOCUSED_ALPHA : 1.0;
        ctx.strokeStyle = color;
        ctx.lineWidth = isFocused ? 3 : 2;
        ctx.beginPath();
        ctx.moveTo(lx + legendPad, ey);
        ctx.lineTo(lx + legendPad + lineLen, ey);
        ctx.stroke();

        // Label
        ctx.font = isFocused ? `bold 11px ${FONT}` : `11px ${FONT}`;
        ctx.fillStyle = isDark ? "#ddd" : "#333";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(traceLabels[t] || `Trace ${t + 1}`, lx + legendPad + lineLen + gap, ey);
        ctx.globalAlpha = 1.0;
      }
    }
  }, [canvasW, canvasH, cursorInfo, showLegend, nTraces, traceLabels, traceColors, isDark, dataToCanvasY, plotW, plotH, focusedTrace]);

  // ========================================================================
  // Mouse handlers
  // ========================================================================
  const handleWheel = React.useCallback(
    (e: WheelEvent) => {
      e.preventDefault();
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      // Only zoom if inside plot area
      if (mouseX < MARGIN.left || mouseX > MARGIN.left + plotW ||
          mouseY < MARGIN.top || mouseY > MARGIN.top + plotH) return;

      const factor = e.deltaY > 0 ? 1.1 : 1 / 1.1;
      const dx = canvasToDataX(mouseX);
      const dy = canvasToDataY(mouseY);

      setXMin((prev) => dx - (dx - prev) * factor);
      setXMax((prev) => dx + (prev - dx) * factor);
      setYMin((prev) => dy - (dy - prev) * factor);
      setYMax((prev) => dy + (prev - dy) * factor);
    },
    [plotW, plotH, canvasToDataX, canvasToDataY],
  );

  // Wheel prevention (passive: false)
  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener("wheel", handleWheel, { passive: false });
    return () => el.removeEventListener("wheel", handleWheel);
  }, [handleWheel]);

  const handleMouseDown = React.useCallback(
    (e: React.MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      // Resize handle check
      if (mx >= canvasW - RESIZE_HIT_AREA_PX && my >= canvasH - RESIZE_HIT_AREA_PX) {
        resizeDragRef.current = {
          active: true,
          startX: e.clientX,
          startY: e.clientY,
          startW: canvasW,
          startH: canvasH,
        };
        return;
      }

      // Pan start
      if (mx >= MARGIN.left && mx <= MARGIN.left + plotW &&
          my >= MARGIN.top && my <= MARGIN.top + plotH) {
        dragRef.current = {
          active: true,
          wasDrag: false,
          startX: e.clientX,
          startY: e.clientY,
          startXMin: xMin,
          startXMax: xMax,
          startYMin: yMin,
          startYMax: yMax,
        };
      }
    },
    [canvasW, canvasH, plotW, plotH, xMin, xMax, yMin, yMax],
  );

  const handleMouseMove = React.useCallback(
    (e: React.MouseEvent) => {
      // Resize drag
      if (resizeDragRef.current?.active) {
        const rd = resizeDragRef.current;
        const newW = Math.max(200, rd.startW + (e.clientX - rd.startX));
        const newH = Math.max(100, rd.startH + (e.clientY - rd.startY));
        setCanvasW(newW);
        setCanvasH(newH);
        return;
      }

      // Pan drag
      if (dragRef.current?.active) {
        const d = dragRef.current;
        const dxPx = e.clientX - d.startX;
        const dyPx = e.clientY - d.startY;
        if (Math.abs(dxPx) > 3 || Math.abs(dyPx) > 3) d.wasDrag = true;
        const xRange = d.startXMax - d.startXMin;
        const yRange = d.startYMax - d.startYMin;
        const dxData = -(dxPx / plotW) * xRange;
        const dyData = (dyPx / plotH) * yRange;
        setXMin(d.startXMin + dxData);
        setXMax(d.startXMax + dxData);
        setYMin(d.startYMin + dyData);
        setYMax(d.startYMax + dyData);
        setCursorInfo(null);
        return;
      }

      // Cursor tracking
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      if (mx < MARGIN.left || mx > MARGIN.left + plotW ||
          my < MARGIN.top || my > MARGIN.top + plotH) {
        setCursorInfo(null);
        return;
      }

      const traces = tracesRef.current;
      const xVals = xValuesRef.current;
      if (traces.length === 0 || nPoints < 1) {
        setCursorInfo(null);
        return;
      }

      const cursorDataX = canvasToDataX(mx);

      // Find nearest index
      let nearestIdx = 0;
      if (xVals) {
        let bestDist = Infinity;
        for (let i = 0; i < xVals.length; i++) {
          const dist = Math.abs(xVals[i] - cursorDataX);
          if (dist < bestDist) { bestDist = dist; nearestIdx = i; }
        }
      } else {
        nearestIdx = Math.round(cursorDataX);
        nearestIdx = Math.max(0, Math.min(nPoints - 1, nearestIdx));
      }

      // Find nearest trace at this index
      const actualX = xVals ? xVals[nearestIdx] : nearestIdx;
      const snapCX = dataToCanvasX(actualX);
      let bestTraceIdx = 0;
      let bestDist = Infinity;
      for (let t = 0; t < traces.length; t++) {
        const val = traces[t][nearestIdx];
        if (!isFinite(val)) continue;
        const traceCY = dataToCanvasY(val);
        const dist = Math.abs(traceCY - my);
        if (dist < bestDist) {
          bestDist = dist;
          bestTraceIdx = t;
        }
      }

      const snapVal = traces[bestTraceIdx][nearestIdx];
      setCursorInfo({
        canvasX: snapCX,
        canvasY: dataToCanvasY(snapVal),
        dataX: actualX,
        dataY: snapVal,
        traceIdx: bestTraceIdx,
        label: (traceLabels && traceLabels[bestTraceIdx]) || "",
        color: (traceColors && traceColors[bestTraceIdx]) || "#4fc3f7",
      });
    },
    [plotW, plotH, nPoints, canvasToDataX, dataToCanvasX, dataToCanvasY, traceLabels, traceColors],
  );

  const handleMouseUp = React.useCallback(() => {
    // Click (not drag) → toggle focus on nearest trace
    if (dragRef.current?.active && !dragRef.current.wasDrag && cursorInfo) {
      const newFocus = cursorInfo.traceIdx;
      setFocusedTrace(focusedTrace === newFocus ? -1 : newFocus);
    }
    dragRef.current = null;
    resizeDragRef.current = null;
  }, [cursorInfo, focusedTrace, setFocusedTrace]);

  const handleMouseLeave = React.useCallback(() => {
    dragRef.current = null;
    resizeDragRef.current = null;
    setCursorInfo(null);
  }, []);

  const handleDoubleClick = React.useCallback(() => {
    resetView();
  }, [resetView]);

  // ========================================================================
  // Keyboard
  // ========================================================================
  const handleKeyDown = React.useCallback(
    (e: React.KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        resetView();
      }
      if (e.key === "Escape") {
        e.preventDefault();
        setFocusedTrace(-1);
      }
    },
    [resetView, setFocusedTrace],
  );

  // ========================================================================
  // Export
  // ========================================================================
  const handleExportPNG = React.useCallback(() => {
    setExportAnchor(null);
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (blob) downloadBlob(blob, `${title || "show1d"}.png`);
    });
  }, [title]);

  const handleExportFigure = React.useCallback(() => {
    setExportAnchor(null);
    const traces = tracesRef.current;
    const xVals = xValuesRef.current;
    if (traces.length === 0) return;

    // Render a publication-quality figure on a white background
    const scale = 4;
    const figW = canvasW * scale;
    const figH = canvasH * scale;
    const offscreen = document.createElement("canvas");
    offscreen.width = figW;
    offscreen.height = figH;
    const ctx = offscreen.getContext("2d");
    if (!ctx) return;

    ctx.scale(scale, scale);

    // White background
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvasW, canvasH);

    // Grid
    if (showGrid) {
      ctx.strokeStyle = "rgba(0,0,0,0.08)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      const xTicks = computeTicks(xMin, xMax, Math.max(3, Math.floor(plotW / TICK_LABEL_WIDTH_PX)));
      for (const tv of xTicks) {
        const cx = dataToCanvasX(tv);
        if (cx >= MARGIN.left && cx <= MARGIN.left + plotW) {
          ctx.beginPath();
          ctx.moveTo(cx, MARGIN.top);
          ctx.lineTo(cx, MARGIN.top + plotH);
          ctx.stroke();
        }
      }
      const yTicks = logScale ? computeLogTicks(Math.max(yMin, 1e-30), yMax) : computeTicks(yMin, yMax);
      for (const tv of yTicks) {
        const cy = dataToCanvasY(tv);
        if (cy >= MARGIN.top && cy <= MARGIN.top + plotH) {
          ctx.beginPath();
          ctx.moveTo(MARGIN.left, cy);
          ctx.lineTo(MARGIN.left + plotW, cy);
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);
    }

    // Axes
    ctx.strokeStyle = "#999";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, MARGIN.top);
    ctx.lineTo(MARGIN.left, MARGIN.top + plotH);
    ctx.lineTo(MARGIN.left + plotW, MARGIN.top + plotH);
    ctx.stroke();

    // X ticks + labels
    const xTicks = computeTicks(xMin, xMax, Math.max(3, Math.floor(plotW / TICK_LABEL_WIDTH_PX)));
    ctx.fillStyle = "#555";
    ctx.font = `10px ${FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const tv of xTicks) {
      const cx = dataToCanvasX(tv);
      if (cx >= MARGIN.left && cx <= MARGIN.left + plotW) {
        ctx.beginPath();
        ctx.moveTo(cx, MARGIN.top + plotH);
        ctx.lineTo(cx, MARGIN.top + plotH + AXIS_TICK_PX);
        ctx.stroke();
        ctx.fillText(formatNumber(tv), cx, MARGIN.top + plotH + AXIS_TICK_PX + 2);
      }
    }

    // Y ticks + labels
    const yTicks = logScale ? computeLogTicks(Math.max(yMin, 1e-30), yMax) : computeTicks(yMin, yMax);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const tv of yTicks) {
      const cy = dataToCanvasY(tv);
      if (cy >= MARGIN.top && cy <= MARGIN.top + plotH) {
        ctx.beginPath();
        ctx.moveTo(MARGIN.left - AXIS_TICK_PX, cy);
        ctx.lineTo(MARGIN.left, cy);
        ctx.stroke();
        ctx.fillText(formatNumber(tv), MARGIN.left - AXIS_TICK_PX - 2, cy);
      }
    }

    // X axis label
    if (xLabel || xUnit) {
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.font = `11px ${FONT}`;
      ctx.fillStyle = "#666";
      let lbl = xLabel || "";
      if (xUnit) lbl += lbl ? ` (${xUnit})` : xUnit;
      ctx.fillText(lbl, MARGIN.left + plotW / 2, canvasH - 6);
    }

    // Y axis label
    if (yLabel || yUnit) {
      ctx.save();
      ctx.translate(12, MARGIN.top + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.font = `11px ${FONT}`;
      ctx.fillStyle = "#666";
      let lbl = yLabel || "";
      if (yUnit) lbl += lbl ? ` (${yUnit})` : yUnit;
      ctx.fillText(lbl, 0, 0);
      ctx.restore();
    }

    // Title
    if (title) {
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.font = `bold 13px ${FONT}`;
      ctx.fillStyle = "#333";
      ctx.fillText(title, canvasW / 2, 2);
    }

    // Traces
    ctx.save();
    ctx.beginPath();
    ctx.rect(MARGIN.left, MARGIN.top, plotW, plotH);
    ctx.clip();

    for (let t = 0; t < traces.length; t++) {
      const trace = traces[t];
      const color = (traceColors && traceColors[t]) || "#4fc3f7";
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth || 1.5;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < trace.length; i++) {
        const xv = xVals ? xVals[i] : i;
        const yv = trace[i];
        if (!isFinite(yv) || (logScale && yv <= 0)) continue;
        const cx = dataToCanvasX(xv);
        const cy = dataToCanvasY(yv);
        if (!started) { ctx.moveTo(cx, cy); started = true; }
        else ctx.lineTo(cx, cy);
      }
      ctx.stroke();
    }

    ctx.restore();

    // Legend on export figure
    if (showLegend && traces.length > 1 && traceLabels && traceLabels.length > 0) {
      ctx.font = `10px ${FONT}`;
      const entryH = 14;
      const lineLen = 16;
      const gap = 4;
      const legendPad = 6;
      let maxLabelW = 0;
      for (let t = 0; t < traces.length; t++) {
        const lbl = traceLabels[t] || `Trace ${t + 1}`;
        const w = ctx.measureText(lbl).width;
        if (w > maxLabelW) maxLabelW = w;
      }
      const legendW = legendPad * 2 + lineLen + gap + maxLabelW;
      const legendH = legendPad * 2 + traces.length * entryH;
      const lx = MARGIN.left + plotW - legendW - 8;
      const ly = MARGIN.top + 8;

      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.fillRect(lx, ly, legendW, legendH);
      ctx.strokeStyle = "#ccc";
      ctx.lineWidth = 1;
      ctx.strokeRect(lx, ly, legendW, legendH);

      for (let t = 0; t < traces.length; t++) {
        const ey = ly + legendPad + t * entryH + entryH / 2;
        const color = (traceColors && traceColors[t]) || "#4fc3f7";
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(lx + legendPad, ey);
        ctx.lineTo(lx + legendPad + lineLen, ey);
        ctx.stroke();
        ctx.fillStyle = "#333";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(traceLabels[t] || `Trace ${t + 1}`, lx + legendPad + lineLen + gap, ey);
      }
    }

    offscreen.toBlob((blob) => {
      if (blob) downloadBlob(blob, `${title || "show1d"}_figure.png`);
    });
  }, [canvasW, canvasH, xMin, xMax, yMin, yMax, traceColors, traceLabels, lineWidth, logScale, showGrid, showLegend, xLabel, yLabel, xUnit, yUnit, title, dataToCanvasX, dataToCanvasY, plotW, plotH]);

  // ========================================================================
  // JSX
  // ========================================================================

  return (
    <Box
      className="show1d-root"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      sx={{ ...container.root, bgcolor: colors.bg, color: colors.text }}
    >
      {/* Header row (Show3D pattern) */}
      <Typography
        variant="caption"
        sx={{
          ...typography.label,
          color: colors.accent,
          mb: `${SPACING.XS}px`,
          display: "block",
        }}
      >
        {title || "Plot"}
        <InfoTooltip
          theme={themeInfo.theme}
          text={
            <KeyboardShortcuts
              items={[
                ["Scroll", "Zoom in/out"],
                ["Drag", "Pan"],
                ["Click", "Focus trace"],
                ["Esc", "Unfocus all"],
                ["R", "Reset view"],
                ["Dbl-click", "Reset view"],
              ]}
            />
          }
        />
      </Typography>

      {/* Controls row (between title and canvas, Show3D pattern) */}
      {showControls && (
        <Box sx={{ display: "flex", alignItems: "center", gap: "4px", mb: `${SPACING.XS}px`, height: 28 }}>
          <Typography sx={{ ...typography.labelSmall, color: colors.text }}>
            Log:
          </Typography>
          <Switch
            size="small"
            checked={logScale}
            onChange={(_, v) => setLogScale(v)}
            sx={switchStyles.small}
          />

          <Typography sx={{ ...typography.labelSmall, color: colors.text, ml: "2px" }}>
            Grid:
          </Typography>
          <Switch
            size="small"
            checked={showGrid}
            onChange={(_, v) => setShowGrid(v)}
            sx={switchStyles.small}
          />

          <Typography sx={{ ...typography.labelSmall, color: colors.text, ml: "2px" }}>
            Legend:
          </Typography>
          <Switch
            size="small"
            checked={showLegend}
            onChange={(_, v) => setShowLegend(v)}
            sx={switchStyles.small}
          />

          <Box sx={{ flex: 1 }} />

          <Button size="small" sx={compactButton} onClick={resetView}>Reset</Button>
          <Button size="small" sx={{ ...compactButton, color: colors.accent }} onClick={(e) => setExportAnchor(e.currentTarget)}>Export</Button>
          <Menu
            anchorEl={exportAnchor}
            open={!!exportAnchor}
            onClose={() => setExportAnchor(null)}
            {...upwardMenuProps}
          >
            <MenuItem onClick={handleExportFigure} sx={{ fontSize: 12 }}>
              Figure (publication PNG)
            </MenuItem>
            <MenuItem onClick={handleExportPNG} sx={{ fontSize: 12 }}>
              PNG
            </MenuItem>
          </Menu>
        </Box>
      )}

      {/* Canvas container */}
      <Box
        ref={containerRef}
        sx={{
          position: "relative",
          width: canvasW,
          height: canvasH,
          border: `1px solid ${colors.border}`,
          cursor: resizeDragRef.current?.active ? "nwse-resize" : dragRef.current?.active ? "grabbing" : "crosshair",
          bgcolor: isDark ? "#1a1a1a" : "#f8f8f8",
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onDoubleClick={handleDoubleClick}
      >
        <canvas
          ref={canvasRef}
          style={{
            width: canvasW,
            height: canvasH,
            position: "absolute",
            top: 0,
            left: 0,
          }}
        />
        <canvas
          ref={uiCanvasRef}
          style={{
            width: canvasW,
            height: canvasH,
            position: "absolute",
            top: 0,
            left: 0,
            pointerEvents: "none",
          }}
        />
      </Box>

      {/* Stats bar (Show3D format, focus-aware) */}
      {showStats && statsMean && statsMean.length > 0 && (() => {
        const showIndices = focusedTrace >= 0 && focusedTrace < nTraces
          ? [focusedTrace]
          : Array.from({ length: nTraces }, (_, i) => i);
        return (
          <Box
            sx={{
              display: "flex",
              gap: SPACING.MD + "px",
              px: 1,
              py: 0.25,
              border: `1px solid ${colors.border}`,
              borderTop: "none",
              bgcolor: colors.bg,
              width: "fit-content",
              flexWrap: "wrap",
            }}
          >
            {showIndices.map((t) => {
              const label = (traceLabels && traceLabels[t]) || `Trace ${t + 1}`;
              const color = (traceColors && traceColors[t]) || "#4fc3f7";
              return (
                <Box key={t} sx={{ display: "flex", gap: "4px", alignItems: "center" }}>
                  <Box sx={{ width: 8, height: 8, bgcolor: color, flexShrink: 0 }} />
                  <Typography sx={{ ...typography.value, color: colors.text }}>
                    {nTraces > 1 ? `${label}: ` : ""}
                    Mean <span style={{ color }}>{formatNumber(statsMean[t] ?? 0)}</span>
                    {"  "}Min <span style={{ color }}>{formatNumber(statsMin[t] ?? 0)}</span>
                    {"  "}Max <span style={{ color }}>{formatNumber(statsMax[t] ?? 0)}</span>
                    {"  "}Std <span style={{ color }}>{formatNumber(statsStd[t] ?? 0)}</span>
                  </Typography>
                </Box>
              );
            })}
          </Box>
        );
      })()}

    </Box>
  );
}

export const render = createRender(Show1D);
