/// <reference types="@webgpu/types" />
/**
 * Show3D - Interactive 3D stack viewer with playback controls.
 * Self-contained widget with all utilities inlined.
 *
 * Features:
 * - Scroll to zoom, double-click to reset
 * - Adjustable ROI size via slider
 * - FPS slider control
 * - WebGPU-accelerated FFT with default 3x zoom
 * - Equal-sized FFT and histogram panels
 * - Automatic theme detection (light/dark mode)
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Slider from "@mui/material/Slider";
import IconButton from "@mui/material/IconButton";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Button from "@mui/material/Button";
import Menu from "@mui/material/Menu";
import Tooltip from "@mui/material/Tooltip";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import FastForwardIcon from "@mui/icons-material/FastForward";
import StopIcon from "@mui/icons-material/Stop";
import "./styles.css";
import { useTheme } from "../theme";
import { drawScaleBarHiDPI, drawFFTScaleBarHiDPI, drawColorbar, roundToNiceValue, exportFigure } from "../scalebar";
import { extractFloat32, formatNumber, downloadBlob, downloadDataView } from "../format";
import { computeHistogramFromBytes } from "../histogram";
import { findDataRange, applyLogScale, applyLogScaleInPlace, percentileClip, sliderRange, computeStats } from "../stats";

// ============================================================================
// UI Styles - component styling helpers (matching Show4DSTEM)
// ============================================================================
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const },
};

const SPACING = {
  XS: 4,    // Extra small gap
  SM: 8,    // Small gap (default between elements)
  MD: 12,   // Medium gap (between control groups)
  LG: 16,   // Large gap (between major sections)
};

const controlPanel = {
  select: { minWidth: 90, fontSize: 11, "& .MuiSelect-select": { py: 0.5 } },
};

const switchStyles = {
  small: { '& .MuiSwitch-thumb': { width: 12, height: 12 }, '& .MuiSwitch-switchBase': { padding: '4px' } },
};

const sliderStyles = {
  small: {
    "& .MuiSlider-thumb": { width: 12, height: 12 },
    "& .MuiSlider-rail": { height: 3 },
    "& .MuiSlider-track": { height: 3 },
  },
};

// Container styles matching Show4DSTEM
const container = {
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "monospace", overflow: "visible" },
  imageBox: { bgcolor: "#000", border: "1px solid #444", overflow: "hidden", position: "relative" as const },
};

// Control row style - bordered container for each row (matching Show4DSTEM)
const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: "6px",
  px: 1,
  py: 0.5,
  width: "fit-content",
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

// Compact button style for Reset (matching Show4DSTEM)
const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  "&.Mui-disabled": {
    color: "#666",
    borderColor: "#444",
  },
};

import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen, renderToOffscreenReuse } from "../colormaps";

// Info tooltip component (matching Show4DSTEM)
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

const DPR = window.devicePixelRatio || 1;
const RESIZE_HIT_AREA_PX = 10;
const CIRCLE_HANDLE_ANGLE = 0.707; // cos(45°)

// ROI drawing
function drawROI(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  shape: "circle" | "square" | "rectangle" | "annular",
  radius: number,
  width: number,
  height: number,
  activeColor: string,
  inactiveColor: string,
  active: boolean = false,
  innerRadius: number = 0
): void {
  const strokeColor = active ? activeColor : inactiveColor;
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 2;
  if (shape === "circle") {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
  } else if (shape === "square") {
    ctx.strokeRect(x - radius, y - radius, radius * 2, radius * 2);
  } else if (shape === "rectangle") {
    ctx.strokeRect(x - width / 2, y - height / 2, width, height);
  } else if (shape === "annular") {
    // Outer circle
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
    // Inner circle (cyan)
    ctx.strokeStyle = active ? "#0ff" : inactiveColor;
    ctx.beginPath();
    ctx.arc(x, y, innerRadius, 0, Math.PI * 2);
    ctx.stroke();
    // Annular fill
    ctx.fillStyle = (active ? activeColor : inactiveColor) + "15";
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.arc(x, y, innerRadius, 0, Math.PI * 2, true);
    ctx.fill();
    ctx.strokeStyle = strokeColor;
  }
  if (active) {
    ctx.beginPath();
    ctx.moveTo(x - 5, y);
    ctx.lineTo(x + 5, y);
    ctx.moveTo(x, y - 5);
    ctx.lineTo(x, y + 5);
    ctx.stroke();
  }
}

function drawResizeHandle(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  isDragging: boolean,
  isHovering: boolean,
  isInner: boolean = false
): void {
  const handleRadius = 5;
  let fill: string;
  if (isDragging) {
    fill = "rgba(0, 200, 255, 1)";
  } else if (isHovering) {
    fill = "rgba(255, 100, 100, 1)";
  } else {
    fill = isInner ? "rgba(0, 220, 255, 0.8)" : "rgba(0, 255, 0, 0.8)";
  }
  ctx.beginPath();
  ctx.arc(x, y, handleRadius, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

// ============================================================================
// Histogram Component
// ============================================================================

interface HistogramProps {
  data: Float32Array | null;
  colormap?: string;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
  dataMin?: number;
  dataMax?: number;
}

function Histogram({
  data,
  colormap: _colormap,
  vminPct,
  vmaxPct,
  onRangeChange,
  width = 110,
  height = 40,
  theme = "dark",
  dataMin = 0,
  dataMax = 1,
}: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const bins = React.useMemo(() => computeHistogramFromBytes(data), [data]);

  // Theme-aware colors
  const colors = theme === "dark" ? {
    bg: "#1a1a1a",
    barActive: "#888",
    barInactive: "#444",
    border: "#333",
  } : {
    bg: "#f0f0f0",
    barActive: "#666",
    barInactive: "#bbb",
    border: "#ccc",
  };

  // Draw histogram (vertical gray bars)
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Clear with theme background
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, width, height);

    // Reduce to fewer bins for cleaner display
    const displayBins = 64;
    const binRatio = Math.floor(bins.length / displayBins);
    const reducedBins: number[] = [];
    for (let i = 0; i < displayBins; i++) {
      let sum = 0;
      for (let j = 0; j < binRatio; j++) {
        sum += bins[i * binRatio + j] || 0;
      }
      reducedBins.push(sum / binRatio);
    }

    // Normalize
    const maxVal = Math.max(...reducedBins, 0.001);
    const barWidth = width / displayBins;

    // Calculate which bins are in the clipped range
    const vminBin = Math.floor((vminPct / 100) * displayBins);
    const vmaxBin = Math.floor((vmaxPct / 100) * displayBins);

    // Draw histogram bars
    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      const x = i * barWidth;

      // Bars inside range are highlighted, outside are dimmed
      const inRange = i >= vminBin && i <= vmaxBin;
      ctx.fillStyle = inRange ? colors.barActive : colors.barInactive;
      ctx.fillRect(x + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
    }

  }, [bins, vminPct, vmaxPct, width, height, colors]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas
        ref={canvasRef}
        style={{ width, height, border: `1px solid ${colors.border}` }}
      />
      <Slider
        value={[vminPct, vmaxPct]}
        onChange={(_, v) => {
          const [newMin, newMax] = v as number[];
          onRangeChange(Math.min(newMin, newMax - 1), Math.max(newMax, newMin + 1));
        }}
        min={0}
        max={100}
        size="small"
        valueLabelDisplay="auto"
        valueLabelFormat={(pct) => {
          const val = dataMin + (pct / 100) * (dataMax - dataMin);
          return val >= 1000 ? val.toExponential(1) : val.toFixed(1);
        }}
        sx={{
          width,
          py: 0,
          "& .MuiSlider-thumb": { width: 8, height: 8 },
          "& .MuiSlider-rail": { height: 2 },
          "& .MuiSlider-track": { height: 2 },
          "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
        }}
      />
    </Box>
  );
}

import { WebGPUFFT, getWebGPUFFT, fft2d, fftshift, computeMagnitude, autoEnhanceFFT, nextPow2 } from "../webgpu-fft";

/** Find the local peak in FFT magnitude near a clicked position with sub-pixel refinement. */
function findFFTPeak(mag: Float32Array, width: number, height: number, col: number, row: number, radius: number): { row: number; col: number } {
  const c0 = Math.max(0, Math.floor(col) - radius);
  const r0 = Math.max(0, Math.floor(row) - radius);
  const c1 = Math.min(width - 1, Math.floor(col) + radius);
  const r1 = Math.min(height - 1, Math.floor(row) + radius);
  let bestCol = Math.round(col), bestRow = Math.round(row), bestVal = -Infinity;
  for (let ir = r0; ir <= r1; ir++) {
    for (let ic = c0; ic <= c1; ic++) {
      const val = mag[ir * width + ic];
      if (val > bestVal) { bestVal = val; bestCol = ic; bestRow = ir; }
    }
  }
  const wc0 = Math.max(0, bestCol - 1), wc1 = Math.min(width - 1, bestCol + 1);
  const wr0 = Math.max(0, bestRow - 1), wr1 = Math.min(height - 1, bestRow + 1);
  let sumW = 0, sumWC = 0, sumWR = 0;
  for (let ir = wr0; ir <= wr1; ir++) {
    for (let ic = wc0; ic <= wc1; ic++) {
      const w = mag[ir * width + ic];
      sumW += w; sumWC += w * ic; sumWR += w * ir;
    }
  }
  if (sumW > 0) return { row: sumWR / sumW, col: sumWC / sumW };
  return { row: bestRow, col: bestCol };
}

const FFT_SNAP_RADIUS = 5;

/** Sample intensity values along a line using bilinear interpolation. */
function sampleSingleLine(data: Float32Array, w: number, h: number, row0: number, col0: number, row1: number, col1: number): Float32Array {
  const dc = col1 - col0;
  const dr = row1 - row0;
  const len = Math.sqrt(dc * dc + dr * dr);
  const n = Math.max(2, Math.ceil(len));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const c = col0 + t * dc;
    const r = row0 + t * dr;
    const ci = Math.floor(c), ri = Math.floor(r);
    const cf = c - ci, rf = r - ri;
    const c0c = Math.max(0, Math.min(w - 1, ci));
    const c1c = Math.max(0, Math.min(w - 1, ci + 1));
    const r0c = Math.max(0, Math.min(h - 1, ri));
    const r1c = Math.max(0, Math.min(h - 1, ri + 1));
    out[i] = data[r0c * w + c0c] * (1 - cf) * (1 - rf) +
             data[r0c * w + c1c] * cf * (1 - rf) +
             data[r1c * w + c0c] * (1 - cf) * rf +
             data[r1c * w + c1c] * cf * rf;
  }
  return out;
}

/** Sample intensity along a line, averaging over profileWidth perpendicular pixels. */
function sampleLineProfile(data: Float32Array, w: number, h: number, row0: number, col0: number, row1: number, col1: number, profileWidth: number = 1): Float32Array {
  if (profileWidth <= 1) return sampleSingleLine(data, w, h, row0, col0, row1, col1);
  const dc = col1 - col0;
  const dr = row1 - row0;
  const len = Math.sqrt(dc * dc + dr * dr);
  if (len < 1e-8) return sampleSingleLine(data, w, h, row0, col0, row1, col1);
  const perpR = -dc / len;
  const perpC = dr / len;
  const half = (profileWidth - 1) / 2;
  let accumulated: Float32Array | null = null;
  for (let k = 0; k < profileWidth; k++) {
    const off = -half + k;
    const vals = sampleSingleLine(data, w, h, row0 + off * perpR, col0 + off * perpC, row1 + off * perpR, col1 + off * perpC);
    if (!accumulated) {
      accumulated = vals;
    } else {
      for (let i = 0; i < vals.length; i++) accumulated[i] += vals[i];
    }
  }
  if (accumulated) for (let i = 0; i < accumulated.length; i++) accumulated[i] /= profileWidth;
  return accumulated || new Float32Array(0);
}

// ============================================================================
// Constants
// ============================================================================
const CANVAS_TARGET_SIZE = 400;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;

const ROI_SHAPES = ["none", "circle", "square", "rectangle", "annular"] as const;
type RoiShape = typeof ROI_SHAPES[number];

/** Extract a single frame from the playback buffer (zero-copy subarray). */
function getFrameFromBuffer(
  buffer: Float32Array | null,
  bufStart: number,
  bufCount: number,
  nSlices: number,
  frameIdx: number,
  frameSize: number,
): Float32Array | null {
  if (!buffer || bufCount === 0) return null;
  let offset = frameIdx - bufStart;
  if (offset < 0) offset += nSlices;
  if (offset < 0 || offset >= bufCount) return null;
  const start = offset * frameSize;
  const end = start + frameSize;
  if (end > buffer.length) return null;
  return buffer.subarray(start, end);
}

/** Fused single-pass render: optional log scale + normalize + colormap → RGBA.
 *  Eliminates multiple data passes during playback for maximum frame rate. */
function renderFramePlayback(
  data: Float32Array,
  rgba: Uint8ClampedArray,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
  logScale: boolean,
): void {
  const range = vmax - vmin;
  const invRange = range > 0 ? 255 / range : 0;
  if (logScale) {
    for (let i = 0; i < data.length; i++) {
      const v = Math.log1p(Math.max(0, data[i]));
      const idx = v <= vmin ? 0 : v >= vmax ? 255 : ((v - vmin) * invRange) | 0;
      const j = i << 2;
      const k = idx * 3;
      rgba[j] = lut[k];
      rgba[j + 1] = lut[k + 1];
      rgba[j + 2] = lut[k + 2];
      rgba[j + 3] = 255;
    }
  } else {
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      const idx = v <= vmin ? 0 : v >= vmax ? 255 : ((v - vmin) * invRange) | 0;
      const j = i << 2;
      const k = idx * 3;
      rgba[j] = lut[k];
      rgba[j + 1] = lut[k + 1];
      rgba[j + 2] = lut[k + 2];
      rgba[j + 3] = 255;
    }
  }
}

// ============================================================================
// Main Component
// ============================================================================
function Show3D() {
  // Theme detection
  const { themeInfo, colors: baseColors } = useTheme();
  const themeColors = {
    ...baseColors,
    accentGreen: themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a",
    accentYellow: themeInfo.theme === "dark" ? "#ff0" : "#b08800",
  };

  // Theme-aware select style (matching Show4DSTEM)
  const themedSelect = {
    ...controlPanel.select,
    bgcolor: themeColors.controlBg,
    color: themeColors.text,
    "& .MuiSelect-select": { py: 0.5 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
  };

  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: { sx: { bgcolor: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}` } },
  };

  // Model state (synced with Python)
  const [sliceIdx, setSliceIdx] = useModelState<number>("slice_idx");
  const [nSlices] = useModelState<number>("n_slices");
  const [width] = useModelState<number>("width");
  const [height] = useModelState<number>("height");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [labels] = useModelState<string[]>("labels");
  const [title] = useModelState<string>("title");
  const [dimLabel] = useModelState<string>("dim_label");
  const [cmap, setCmap] = useModelState<string>("cmap");

  // Playback
  const [playing, setPlaying] = useModelState<boolean>("playing");
  const [reverse, setReverse] = useModelState<boolean>("reverse");
  const [boomerang, setBoomerang] = useModelState<boolean>("boomerang");
  const [fps, setFps] = useModelState<number>("fps");
  const [loop, setLoop] = useModelState<boolean>("loop");
  const [loopStart, setLoopStart] = useModelState<number>("loop_start");
  const [loopEnd, setLoopEnd] = useModelState<number>("loop_end");
  const [bookmarkedFrames, setBookmarkedFrames] = useModelState<number[]>("bookmarked_frames");
  const [playbackPath] = useModelState<number[]>("playback_path");

  // Boomerang direction ref (avoids stale closure in setInterval)
  const bounceDirRef = React.useRef<1 | -1>(1);

  // Stats
  const [showStats] = useModelState<boolean>("show_stats");
  const [statsMean] = useModelState<number>("stats_mean");
  const [statsMin] = useModelState<number>("stats_min");
  const [statsMax] = useModelState<number>("stats_max");
  const [statsStd] = useModelState<number>("stats_std");

  // Display options
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [percentileLow] = useModelState<number>("percentile_low");
  const [percentileHigh] = useModelState<number>("percentile_high");
  const [dataMin] = useModelState<number>("data_min");
  const [dataMax] = useModelState<number>("data_max");
  // Scale bar
  const [pixelSize] = useModelState<number>("pixel_size");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");

  // Customization
  const [imageWidthPxTrait] = useModelState<number>("image_width_px");

  // Timestamps
  const [timestamps] = useModelState<number[]>("timestamps");
  const [timestampUnit] = useModelState<string>("timestamp_unit");
  const [currentTimestamp] = useModelState<number>("current_timestamp");
  // ROI
  const [roiActive, setRoiActive] = useModelState<boolean>("roi_active");
  const [roiShape, setRoiShape] = useModelState<RoiShape>("roi_shape");
  const [roiRow, setRoiRow] = useModelState<number>("roi_row");
  const [roiCol, setRoiCol] = useModelState<number>("roi_col");
  const [roiRadius, setRoiRadius] = useModelState<number>("roi_radius");
  const [roiRadiusInner, setRoiRadiusInner] = useModelState<number>("roi_radius_inner");
  const [roiWidth, setRoiWidth] = useModelState<number>("roi_width");
  const [roiHeight, setRoiHeight] = useModelState<number>("roi_height");
  const [roiMean] = useModelState<number>("roi_mean");
  const [roiMin] = useModelState<number>("roi_min");
  const [roiMax] = useModelState<number>("roi_max");
  const [roiStd] = useModelState<number>("roi_std");
  const [roiPlotData] = useModelState<DataView>("roi_plot_data");

  // FFT
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [showPlayback] = useModelState<boolean>("show_playback");

  // Export
  const [, setGifExportRequested] = useModelState<boolean>("_gif_export_requested");
  const [gifData] = useModelState<DataView>("_gif_data");
  const [, setZipExportRequested] = useModelState<boolean>("_zip_export_requested");
  const [zipData] = useModelState<DataView>("_zip_data");
  const [exporting, setExporting] = React.useState(false);
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);

  // Playback buffer (sliding prefetch)
  const [bufferBytes] = useModelState<DataView>("_buffer_bytes");
  const [bufferStart] = useModelState<number>("_buffer_start");
  const [bufferCount] = useModelState<number>("_buffer_count");
  const [, setPrefetchRequest] = useModelState<number>("_prefetch_request");

  // Canvas refs
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const overlayRef = React.useRef<HTMLCanvasElement>(null);
  const uiRef = React.useRef<HTMLCanvasElement>(null);
  const canvasContainerRef = React.useRef<HTMLDivElement>(null);
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const fftOverlayRef = React.useRef<HTMLCanvasElement>(null);

  // Local state
  const [isDraggingROI, setIsDraggingROI] = React.useState(false);
  const [isDraggingResize, setIsDraggingResize] = React.useState(false);
  const [isDraggingResizeInner, setIsDraggingResizeInner] = React.useState(false);
  const [isHoveringResize, setIsHoveringResize] = React.useState(false);
  const [isHoveringResizeInner, setIsHoveringResizeInner] = React.useState(false);
  const [zoom, setZoom] = React.useState(1);
  const [panX, setPanX] = React.useState(0);
  const [panY, setPanY] = React.useState(0);
  const [isDraggingPan, setIsDraggingPan] = React.useState(false);
  const [panStart, setPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);
  const [mainCanvasSize, setMainCanvasSize] = React.useState(CANVAS_TARGET_SIZE);
  const [isResizingMain, setIsResizingMain] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number, y: number, size: number } | null>(null);
  const rawFrameDataRef = React.useRef<Float32Array | null>(null);
  const initialCanvasSizeRef = React.useRef<number>(imageWidthPxTrait > 0 ? imageWidthPxTrait : CANVAS_TARGET_SIZE);

  // Cursor readout state
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number } | null>(null);
  const [showRoiPlot, setShowRoiPlot] = React.useState(true);
  const roiPlotCanvasRef = React.useRef<HTMLCanvasElement>(null);

  // Lens (magnifier inset)
  const [showLens, setShowLens] = React.useState(false);
  const [lensPos, setLensPos] = React.useState<{ row: number; col: number } | null>(null);
  const [lensMag, setLensMag] = React.useState(4);
  const [lensDisplaySize, setLensDisplaySize] = React.useState(128);
  const [lensAnchor, setLensAnchor] = React.useState<{ x: number; y: number } | null>(null);
  const [isDraggingLens, setIsDraggingLens] = React.useState(false);
  const lensCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const lensDragStartRef = React.useRef<{ mx: number; my: number; ax: number; ay: number } | null>(null);
  const [isResizingLens, setIsResizingLens] = React.useState(false);
  const [isHoveringLensEdge, setIsHoveringLensEdge] = React.useState(false);
  const lensResizeStartRef = React.useRef<{ my: number; startSize: number } | null>(null);

  // Reusable rendering buffers (avoid per-frame allocation)
  const mainOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const mainImgDataRef = React.useRef<ImageData | null>(null);
  const logBufferRef = React.useRef<Float32Array | null>(null);

  // Playback buffer refs (double-buffer: current + next to avoid overwrite stalls)
  const bufferRef = React.useRef<Float32Array | null>(null);
  const bufferStartRef = React.useRef(0);
  const bufferCountRef = React.useRef(0);
  const nextBufferRef = React.useRef<Float32Array | null>(null);
  const nextBufferStartRef = React.useRef(0);
  const nextBufferCountRef = React.useRef(0);
  const prefetchPendingRef = React.useRef(false);
  const playbackIdxRef = React.useRef(0);
  const [displaySliceIdx, setDisplaySliceIdx] = React.useState(sliceIdx);
  const [localStats, setLocalStats] = React.useState<{ mean: number; min: number; max: number; std: number } | null>(null);

  // WebGPU FFT state
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);
  const fftOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);

  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) { gpuFFTRef.current = fft; setGpuReady(true); }
    });
  }, []);

  // Parse incoming playback buffer (double-buffer to avoid overwrite stalls)
  React.useEffect(() => {
    if (!bufferBytes || bufferBytes.byteLength === 0) return;
    const parsed = extractFloat32(bufferBytes);
    if (!parsed) return;
    if (!bufferRef.current || bufferCountRef.current === 0) {
      // No active buffer — use as current (initial load)
      bufferRef.current = parsed;
      bufferStartRef.current = bufferStart;
      bufferCountRef.current = bufferCount;
    } else {
      // Active buffer exists — store as next (prefetch)
      nextBufferRef.current = parsed;
      nextBufferStartRef.current = bufferStart;
      nextBufferCountRef.current = bufferCount;
    }
    prefetchPendingRef.current = false;
  }, [bufferBytes, bufferStart, bufferCount]);

  // Sync displaySliceIdx with model when not playing
  React.useEffect(() => {
    if (!playing) setDisplaySliceIdx(sliceIdx);
  }, [sliceIdx, playing]);

  // Histogram state for main image
  const [imageVminPct, setImageVminPct] = React.useState(0);
  const [imageVmaxPct, setImageVmaxPct] = React.useState(100);
  const [imageHistogramData, setImageHistogramData] = React.useState<Float32Array | null>(null);
  const [imageDataRange, setImageDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });

  // Histogram state for FFT
  const [fftVminPct, setFftVminPct] = React.useState(0);
  const [fftVmaxPct, setFftVmaxPct] = React.useState(100);
  const [fftHistogramData, setFftHistogramData] = React.useState<Float32Array | null>(null);
  const [fftDataRange, setFftDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [fftStats, setFftStats] = React.useState<{ mean: number; min: number; max: number; std: number }>({ mean: 0, min: 0, max: 0, std: 0 });
  const [fftColormap, setFftColormap] = React.useState("inferno");
  const [fftLogScale, setFftLogScale] = React.useState(false);
  const [fftAuto, setFftAuto] = React.useState(true);  // Auto: mask DC + 99.9% clipping
  const [fftShowColorbar, setFftShowColorbar] = React.useState(false);
  const [showColorbar, setShowColorbar] = React.useState(false);

  // FFT d-spacing measurement
  const [fftClickInfo, setFftClickInfo] = React.useState<{
    row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  } | null>(null);
  const fftClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const fftMagCacheRef = React.useRef<Float32Array | null>(null);

  // FFT zoom/pan state
  const [fftZoom, setFftZoom] = React.useState(1);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);
  const fftContainerRef = React.useRef<HTMLDivElement>(null);

  // Line profile state
  const [profileActive, setProfileActive] = React.useState(false);
  const [profileLine, setProfileLine] = useModelState<{row: number; col: number}[]>("profile_line");
  const [profileWidth, setProfileWidth] = useModelState<number>("profile_width");
  const [profileData, setProfileData] = React.useState<Float32Array | null>(null);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const profilePoints = profileLine || [];
  const [profileHeight, setProfileHeight] = React.useState(76);
  const [isResizingProfile, setIsResizingProfile] = React.useState(false);
  const [profileResizeStart, setProfileResizeStart] = React.useState<{ y: number; height: number } | null>(null);
  const profileBaseImageRef = React.useRef<ImageData | null>(null);
  const profileLayoutRef = React.useRef<{ padLeft: number; plotW: number; padTop: number; plotH: number; gMin: number; gMax: number; totalDist: number; xUnit: string } | null>(null);

  // Sync sizes from Python and set initial minimum
  React.useEffect(() => {
    if (imageWidthPxTrait > 0) {
      setMainCanvasSize(imageWidthPxTrait);
      // Only set initial size on first load (when ref is still default)
      if (initialCanvasSizeRef.current === CANVAS_TARGET_SIZE) {
        initialCanvasSizeRef.current = imageWidthPxTrait;
      }
    }
  }, [imageWidthPxTrait]);

  // Calculate display scale
  const displayScale = mainCanvasSize / Math.max(width, height);
  const canvasW = Math.round(width * displayScale);
  const canvasH = Math.round(height * displayScale);
  const effectiveLoopEnd = loopEnd < 0 ? nSlices - 1 : loopEnd;

  // Initialize reusable offscreen canvas + ImageData (resized when dimensions change)
  React.useEffect(() => {
    if (width <= 0 || height <= 0) return;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    mainOffscreenRef.current = canvas;
    mainImgDataRef.current = canvas.getContext("2d")!.createImageData(width, height);
    logBufferRef.current = new Float32Array(width * height);
  }, [width, height]);

  // Prevent page scroll on canvas containers (but don't stop propagation so React handlers work)
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const el1 = canvasContainerRef.current;
    const el2 = fftContainerRef.current;
    el1?.addEventListener("wheel", preventDefault, { passive: false });
    el2?.addEventListener("wheel", preventDefault, { passive: false });
    return () => {
      el1?.removeEventListener("wheel", preventDefault);
      el2?.removeEventListener("wheel", preventDefault);
    };
  }, [showFft]);

  // Stop playback when FFT is toggled on (FFT doesn't update during playback)
  React.useEffect(() => {
    if (showFft && playing) setPlaying(false);
  }, [showFft]);

  // Sync boomerang direction ref with reverse state
  React.useEffect(() => {
    bounceDirRef.current = reverse ? -1 : 1;
  }, [reverse]);

  // All playback params as a single ref (avoids stale closures in rAF loop)
  const pathIdxRef = React.useRef(0);
  const playRef = React.useRef({
    fps, reverse, boomerang, loop, loopStart, loopEnd: effectiveLoopEnd,
    nSlices, width, height, displayScale, canvasW, canvasH,
    logScale, autoContrast, percentileLow, percentileHigh,
    dataMin, dataMax, cmap, imageVminPct, imageVmaxPct,
    zoom, panX, panY, playbackPath,
  });
  React.useEffect(() => {
    playRef.current = {
      fps, reverse, boomerang, loop, loopStart, loopEnd: effectiveLoopEnd,
      nSlices, width, height, displayScale, canvasW, canvasH,
      logScale, autoContrast, percentileLow, percentileHigh,
      dataMin, dataMax, cmap, imageVminPct, imageVmaxPct,
      zoom, panX, panY, playbackPath,
    };
  }, [fps, reverse, boomerang, loop, loopStart, effectiveLoopEnd,
    nSlices, width, height, displayScale, canvasW, canvasH,
    logScale, autoContrast, percentileLow, percentileHigh,
    dataMin, dataMax, cmap, imageVminPct, imageVmaxPct,
    zoom, panX, panY, playbackPath]);

  // Playback logic — rAF-driven, zero React re-renders in hot path
  React.useEffect(() => {
    if (!playing) {
      // Playback stopped — sync final position to Python
      if (playbackIdxRef.current !== sliceIdx && bufferRef.current) {
        setSliceIdx(playbackIdxRef.current);
      }
      setLocalStats(null);
      bufferRef.current = null;
      bufferCountRef.current = 0;
      nextBufferRef.current = null;
      nextBufferCountRef.current = 0;
      prefetchPendingRef.current = false;
      return;
    }

    // === PLAYBACK START ===
    playbackIdxRef.current = sliceIdx;
    pathIdxRef.current = 0;
    bounceDirRef.current = playRef.current.reverse ? -1 : 1;
    let lastFrameTime = 0;
    let lastUIUpdate = 0;
    let animId: number;

    const tick = (now: number) => {
      const c = playRef.current;
      const intervalMs = 1000 / c.fps;

      // First tick — just record time
      if (lastFrameTime === 0) {
        lastFrameTime = now;
        lastUIUpdate = now;
        animId = requestAnimationFrame(tick);
        return;
      }

      const elapsed = now - lastFrameTime;
      if (elapsed < intervalMs) {
        animId = requestAnimationFrame(tick);
        return;
      }
      lastFrameTime = now - (elapsed % intervalMs);

      // Advance frame
      let next: number;
      if (c.playbackPath && c.playbackPath.length > 0) {
        // Custom playback path
        const pp = c.playbackPath;
        let pi = pathIdxRef.current;
        if (c.boomerang) {
          pi += bounceDirRef.current;
          if (pi >= pp.length) { bounceDirRef.current = -1; pi = pp.length - 2; }
          if (pi < 0) { bounceDirRef.current = 1; pi = 1; }
        } else {
          pi += (c.reverse ? -1 : 1);
          if (pi >= pp.length) { if (!c.loop) { setPlaying(false); return; } pi = 0; }
          if (pi < 0) { if (!c.loop) { setPlaying(false); return; } pi = pp.length - 1; }
        }
        pi = Math.max(0, Math.min(pp.length - 1, pi));
        pathIdxRef.current = pi;
        next = pp[pi];
      } else {
        const rangeStart = c.loop ? Math.max(0, Math.min(c.loopStart, c.nSlices - 1)) : 0;
        const rangeEnd = c.loop ? Math.max(rangeStart, Math.min(c.loopEnd, c.nSlices - 1)) : c.nSlices - 1;
        const prev = playbackIdxRef.current;

        if (c.boomerang) {
          next = prev + bounceDirRef.current;
          if (next > rangeEnd) { bounceDirRef.current = -1; next = prev - 1 >= rangeStart ? prev - 1 : prev; }
          else if (next < rangeStart) { bounceDirRef.current = 1; next = prev + 1 <= rangeEnd ? prev + 1 : prev; }
        } else {
          next = prev + (c.reverse ? -1 : 1);
          if (c.reverse) {
            if (next < rangeStart) { if (!c.loop) { setPlaying(false); return; } next = rangeEnd; }
          } else {
            if (next > rangeEnd) { if (!c.loop) { setPlaying(false); return; } next = rangeStart; }
          }
        }
      }

      // Try buffer path (zero round-trip) with double-buffer swap
      const frameSize = c.width * c.height;
      let frame = getFrameFromBuffer(bufferRef.current, bufferStartRef.current, bufferCountRef.current, c.nSlices, next, frameSize);
      if (!frame && nextBufferRef.current) {
        // Current buffer doesn't have this frame — swap to next buffer
        bufferRef.current = nextBufferRef.current;
        bufferStartRef.current = nextBufferStartRef.current;
        bufferCountRef.current = nextBufferCountRef.current;
        nextBufferRef.current = null;
        nextBufferCountRef.current = 0;
        frame = getFrameFromBuffer(bufferRef.current, bufferStartRef.current, bufferCountRef.current, c.nSlices, next, frameSize);
      }
      if (!frame) {
        // Buffer not ready yet — keep requesting frames
        animId = requestAnimationFrame(tick);
        return;
      }

      playbackIdxRef.current = next;
      rawFrameDataRef.current = frame;

      // Render frame — fused single-pass when possible
      const lut = COLORMAPS[c.cmap] || COLORMAPS.inferno;
      if (mainOffscreenRef.current && mainImgDataRef.current) {
        let vmin: number, vmax: number;
        if (c.autoContrast) {
          // Auto-contrast needs per-frame percentile (2 passes), but no stats
          if (c.logScale && logBufferRef.current) {
            applyLogScaleInPlace(frame, logBufferRef.current);
            ({ vmin, vmax } = percentileClip(logBufferRef.current, c.percentileLow, c.percentileHigh));
            renderToOffscreenReuse(logBufferRef.current, lut, vmin, vmax, mainOffscreenRef.current, mainImgDataRef.current);
          } else {
            ({ vmin, vmax } = percentileClip(frame, c.percentileLow, c.percentileHigh));
            renderToOffscreenReuse(frame, lut, vmin, vmax, mainOffscreenRef.current, mainImgDataRef.current);
          }
        } else {
          // Global range + slider — fused single-pass render (fastest path)
          if (c.logScale) {
            const logMin = Math.log1p(Math.max(0, c.dataMin));
            const logMax = Math.log1p(Math.max(0, c.dataMax));
            ({ vmin, vmax } = sliderRange(logMin, logMax, c.imageVminPct, c.imageVmaxPct));
          } else {
            ({ vmin, vmax } = sliderRange(c.dataMin, c.dataMax, c.imageVminPct, c.imageVmaxPct));
          }
          renderFramePlayback(frame, mainImgDataRef.current.data, lut, vmin, vmax, c.logScale);
          mainOffscreenRef.current.getContext("2d")!.putImageData(mainImgDataRef.current, 0, 0);
        }

        // Draw to display canvas
        const canvas = canvasRef.current;
        if (canvas) {
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.imageSmoothingEnabled = false;
            ctx.clearRect(0, 0, c.canvasW, c.canvasH);
            ctx.save();
            ctx.translate(c.panX, c.panY);
            ctx.scale(c.zoom, c.zoom);
            ctx.drawImage(mainOffscreenRef.current, 0, 0, c.width * c.displayScale, c.height * c.displayScale);
            ctx.restore();
          }
        }
      }

      // Throttled UI updates — 10 FPS for slider/stats (avoids costly MUI re-renders)
      if (now - lastUIUpdate > 100) {
        lastUIUpdate = now;
        setDisplaySliceIdx(next);
        setLocalStats(computeStats(frame));
      }

      // Prefetch at 25% buffer consumed — only if no next buffer is already queued
      if (!prefetchPendingRef.current && !nextBufferRef.current && bufferCountRef.current > 0) {
        let idxInBuffer = next - bufferStartRef.current;
        if (idxInBuffer < 0) idxInBuffer += c.nSlices;
        if (idxInBuffer >= Math.floor(bufferCountRef.current / 4)) {
          const prefetchStart = (bufferStartRef.current + bufferCountRef.current) % c.nSlices;
          prefetchPendingRef.current = true;
          setPrefetchRequest(prefetchStart);
        }
      }

      animId = requestAnimationFrame(tick);
    };

    animId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing]);

  // Update frame ref when frame changes
  React.useEffect(() => {
    const parsed = extractFloat32(frameBytes);
    if (!parsed || parsed.length === 0) return;
    rawFrameDataRef.current = parsed;
  }, [frameBytes]);

  // Update histogram data (reflects log scale state, debounced during playback)
  React.useEffect(() => {
    const raw = rawFrameDataRef.current;
    if (!raw || raw.length === 0 || playing) return;
    const data = logScale ? applyLogScale(raw) : raw;
    setImageHistogramData(data);
    setImageDataRange(findDataRange(data));
  }, [frameBytes, playing, logScale]);

  // Data effect: normalize + colormap → reusable offscreen canvas, then draw
  React.useEffect(() => {
    const frameData = rawFrameDataRef.current;
    if (!frameData || frameData.length === 0) return;
    if (!mainOffscreenRef.current || !mainImgDataRef.current) return;

    // Apply log scale using reusable buffer
    const processed = logScale && logBufferRef.current
      ? applyLogScaleInPlace(frameData, logBufferRef.current)
      : frameData;

    // Compute vmin/vmax
    let vmin: number, vmax: number;
    if (autoContrast) {
      ({ vmin, vmax } = percentileClip(processed, percentileLow, percentileHigh));
    } else {
      const { min: pMin, max: pMax } = findDataRange(processed);
      ({ vmin, vmax } = sliderRange(pMin, pMax, imageVminPct, imageVmaxPct));
    }

    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    renderToOffscreenReuse(processed, lut, vmin, vmax, mainOffscreenRef.current, mainImgDataRef.current);

    // Draw to main canvas
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);
    ctx.drawImage(mainOffscreenRef.current, 0, 0, width * displayScale, height * displayScale);
    ctx.restore();
  }, [frameBytes, width, height, cmap, displayScale, canvasW, canvasH, imageVminPct, imageVmaxPct, logScale, autoContrast, percentileLow, percentileHigh]);

  // Draw effect: only zoom/pan changes — cheap, just drawImage from cached offscreen
  React.useEffect(() => {
    if (!mainOffscreenRef.current || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);
    ctx.drawImage(mainOffscreenRef.current, 0, 0, width * displayScale, height * displayScale);
    ctx.restore();
  }, [zoom, panX, panY]);

  // Render overlay (ROI only) — HiDPI aware
  React.useEffect(() => {
    if (!overlayRef.current) return;
    const ctx = overlayRef.current.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, canvasW, canvasH);
    if (roiActive) {
      const screenX = roiCol * displayScale * zoom + panX;
      const screenY = roiRow * displayScale * zoom + panY;
      const screenRadius = roiRadius * displayScale * zoom;
      const screenWidth = roiWidth * displayScale * zoom;
      const screenHeight = roiHeight * displayScale * zoom;
      const screenRadiusInner = roiRadiusInner * displayScale * zoom;
      drawROI(ctx, screenX, screenY, roiShape || "circle", screenRadius, screenWidth, screenHeight, themeColors.accentYellow, themeColors.accentGreen, isDraggingROI, screenRadiusInner);
      // Draw resize handle(s)
      const shape = roiShape || "circle";
      if (shape === "rectangle") {
        drawResizeHandle(ctx, screenX + screenWidth / 2, screenY + screenHeight / 2, isDraggingResize, isHoveringResize);
      } else if (shape === "annular") {
        const outerOffset = screenRadius * CIRCLE_HANDLE_ANGLE;
        drawResizeHandle(ctx, screenX + outerOffset, screenY + outerOffset, isDraggingResize, isHoveringResize);
        const innerOffset = screenRadiusInner * CIRCLE_HANDLE_ANGLE;
        drawResizeHandle(ctx, screenX + innerOffset, screenY + innerOffset, isDraggingResizeInner, isHoveringResizeInner, true);
      } else if (shape === "circle") {
        const offset = screenRadius * CIRCLE_HANDLE_ANGLE;
        drawResizeHandle(ctx, screenX + offset, screenY + offset, isDraggingResize, isHoveringResize);
      } else if (shape === "square") {
        drawResizeHandle(ctx, screenX + screenRadius, screenY + screenRadius, isDraggingResize, isHoveringResize);
      }
    }

    // Line profile overlay
    if (profileActive && profilePoints.length > 0) {
      const toScreenX = (col: number) => col * displayScale * zoom + panX;
      const toScreenY = (row: number) => row * displayScale * zoom + panY;

      // Draw point A
      const ax = toScreenX(profilePoints[0].col);
      const ay = toScreenY(profilePoints[0].row);
      ctx.fillStyle = themeColors.accent;
      ctx.beginPath();
      ctx.arc(ax, ay, 4, 0, Math.PI * 2);
      ctx.fill();

      if (profilePoints.length === 2) {
        const bx = toScreenX(profilePoints[1].col);
        const by = toScreenY(profilePoints[1].row);

        // Draw band when profile width > 1
        if (profileWidth > 1) {
          const dc = profilePoints[1].col - profilePoints[0].col;
          const dr = profilePoints[1].row - profilePoints[0].row;
          const lineLen = Math.sqrt(dc * dc + dr * dr);
          if (lineLen > 0) {
            const halfW = (profileWidth - 1) / 2;
            const perpR = -dc / lineLen * halfW;
            const perpC = dr / lineLen * halfW;
            ctx.fillStyle = themeColors.accent + "20";
            ctx.strokeStyle = themeColors.accent;
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(toScreenX(profilePoints[0].col + perpC), toScreenY(profilePoints[0].row + perpR));
            ctx.lineTo(toScreenX(profilePoints[1].col + perpC), toScreenY(profilePoints[1].row + perpR));
            ctx.lineTo(toScreenX(profilePoints[1].col - perpC), toScreenY(profilePoints[1].row - perpR));
            ctx.lineTo(toScreenX(profilePoints[0].col - perpC), toScreenY(profilePoints[0].row - perpR));
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }

        ctx.strokeStyle = themeColors.accent;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = themeColors.accent;
        ctx.beginPath();
        ctx.arc(bx, by, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [roiActive, roiShape, roiRow, roiCol, roiRadius, roiRadiusInner, roiWidth, roiHeight, isDraggingROI, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner, canvasW, canvasH, displayScale, zoom, panX, panY, themeColors, profileActive, profilePoints, profileWidth]);

  // Lens inset rendering
  React.useEffect(() => {
    const lensCanvas = lensCanvasRef.current;
    if (lensCanvas) {
      const lctx = lensCanvas.getContext("2d");
      if (lctx) lctx.clearRect(0, 0, lensCanvas.width, lensCanvas.height);
    }
    if (!showLens || !lensPos || !rawFrameDataRef.current) return;
    if (!lensCanvas) return;
    const ctx = lensCanvas.getContext("2d");
    if (!ctx) return;

    const raw = rawFrameDataRef.current;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const processed = logScale ? applyLogScale(raw) : raw;
    let vmin: number, vmax: number;
    if (imageDataRange.min !== imageDataRange.max) {
      ({ vmin, vmax } = sliderRange(imageDataRange.min, imageDataRange.max, imageVminPct, imageVmaxPct));
    } else if (autoContrast) {
      ({ vmin, vmax } = percentileClip(processed, percentileLow, percentileHigh));
    } else {
      const r = findDataRange(processed);
      vmin = r.min; vmax = r.max;
    }

    const regionSize = Math.max(4, Math.round(lensDisplaySize / lensMag));
    const lensSize = lensDisplaySize;
    const margin = 12;
    const half = Math.floor(regionSize / 2);
    const r0 = lensPos.row - half;
    const c0 = lensPos.col - half;

    const regionCanvas = document.createElement("canvas");
    regionCanvas.width = regionSize;
    regionCanvas.height = regionSize;
    const rctx = regionCanvas.getContext("2d");
    if (!rctx) return;
    const imgData = rctx.createImageData(regionSize, regionSize);
    const range = vmax - vmin || 1;
    for (let dr = 0; dr < regionSize; dr++) {
      for (let dc = 0; dc < regionSize; dc++) {
        const sr = r0 + dr;
        const sc = c0 + dc;
        const idx = (dr * regionSize + dc) * 4;
        if (sr < 0 || sr >= height || sc < 0 || sc >= width) {
          imgData.data[idx] = 0; imgData.data[idx + 1] = 0; imgData.data[idx + 2] = 0; imgData.data[idx + 3] = 255;
        } else {
          const val = processed[sr * width + sc];
          const t = Math.max(0, Math.min(1, (val - vmin) / range));
          const li = Math.round(t * 255);
          imgData.data[idx] = lut[li * 3]; imgData.data[idx + 1] = lut[li * 3 + 1]; imgData.data[idx + 2] = lut[li * 3 + 2]; imgData.data[idx + 3] = 255;
        }
      }
    }
    rctx.putImageData(imgData, 0, 0);

    ctx.save();
    ctx.scale(DPR, DPR);
    const lx = lensAnchor ? lensAnchor.x : margin;
    const ly = lensAnchor ? lensAnchor.y : canvasH - lensSize - margin - 20;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(regionCanvas, lx, ly, lensSize, lensSize);
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 2;
    ctx.strokeRect(lx, ly, lensSize, lensSize);
    const cx = lx + lensSize / 2;
    const cy = ly + lensSize / 2;
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy);
    ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8);
    ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.font = "10px monospace";
    ctx.fillText(`${lensMag}×`, lx + 4, ly + lensSize - 4);
    ctx.restore();
  }, [showLens, lensPos, cmap, logScale, autoContrast, imageDataRange, imageVminPct, imageVmaxPct, width, height, canvasH, themeColors, lensMag, lensDisplaySize, lensAnchor, percentileLow, percentileHigh, frameBytes, sliceIdx, displaySliceIdx]);

  // ROI sparkline plot
  React.useEffect(() => {
    const canvas = roiPlotCanvasRef.current;
    if (!canvas || !showRoiPlot || !roiActive) return;
    const plotW = canvasW;
    const plotH = 76;
    canvas.width = Math.round(plotW * DPR);
    canvas.height = Math.round(plotH * DPR);
    canvas.style.width = `${plotW}px`;
    canvas.style.height = `${plotH}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, plotW, plotH);

    if (!roiPlotData || roiPlotData.byteLength < 4) return;
    const values = extractFloat32(roiPlotData);
    if (values.length === 0) return;
    let min = values[0], max = values[0];
    for (let i = 1; i < values.length; i++) {
      if (values[i] < min) min = values[i];
      if (values[i] > max) max = values[i];
    }
    const range = max - min || 1;
    const padY = 14;
    const drawH = plotH - padY * 2;

    // Draw plot line
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
      const x = (i / (values.length - 1)) * plotW;
      const y = padY + drawH - ((values[i] - min) / range) * drawH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw current frame marker
    const activeIdx = playing ? displaySliceIdx : sliceIdx;
    const mx = (activeIdx / (values.length - 1)) * plotW;
    ctx.strokeStyle = themeColors.textMuted;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(mx, padY);
    ctx.lineTo(mx, padY + drawH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Current value dot
    if (activeIdx >= 0 && activeIdx < values.length) {
      const cy = padY + drawH - ((values[activeIdx] - min) / range) * drawH;
      ctx.fillStyle = themeColors.accent;
      ctx.beginPath();
      ctx.arc(mx, cy, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Y-axis labels
    ctx.fillStyle = themeColors.textMuted;
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText(formatNumber(max), 2, padY - 2);
    ctx.fillText(formatNumber(min), 2, padY + drawH + 10);
  }, [roiPlotData, roiActive, showRoiPlot, canvasW, themeColors, sliceIdx, displaySliceIdx, playing]);

  // Compute profile data when points/width/frame change
  React.useEffect(() => {
    if (profilePoints.length === 2 && rawFrameDataRef.current) {
      const p0 = profilePoints[0], p1 = profilePoints[1];
      const data = rawFrameDataRef.current;
      setProfileData(sampleLineProfile(data, width, height, p0.row, p0.col, p1.row, p1.col, profileWidth));
      if (!profileActive) setProfileActive(true);
    } else {
      setProfileData(null);
    }
  }, [profilePoints, profileWidth, frameBytes]);

  // Render profile sparkline
  React.useEffect(() => {
    const canvas = profileCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvasW;
    const cssH = profileHeight;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.scale(dpr, dpr);

    const isDark = themeInfo.theme === "dark";
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f0f0f0";
    ctx.fillRect(0, 0, cssW, cssH);

    if (!profileData || profileData.length < 2) {
      ctx.font = "10px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.fillStyle = isDark ? "#555" : "#999";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Click two points on the image to draw a profile", cssW / 2, cssH / 2);
      return;
    }

    const padLeft = 40;
    const padRight = 8;
    const padTop = 6;
    const padBottom = 18;
    const plotW = cssW - padLeft - padRight;
    const plotH = cssH - padTop - padBottom;

    let gMin = Infinity, gMax = -Infinity;
    for (let i = 0; i < profileData.length; i++) {
      if (profileData[i] < gMin) gMin = profileData[i];
      if (profileData[i] > gMax) gMax = profileData[i];
    }
    const range = gMax - gMin || 1;

    // Draw profile line
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < profileData.length; i++) {
      const x = padLeft + (i / (profileData.length - 1)) * plotW;
      const y = padTop + plotH - ((profileData[i] - gMin) / range) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // X-axis: calibrated distance
    let totalDist = profileData.length - 1;
    let xUnit = "px";
    if (profilePoints.length === 2) {
      const dx = profilePoints[1].col - profilePoints[0].col;
      const dy = profilePoints[1].row - profilePoints[0].row;
      const distPx = Math.sqrt(dx * dx + dy * dy);
      if (pixelSize > 0) {
        const pixelSizeAngstrom = pixelSize * 10;
        const distA = distPx * pixelSizeAngstrom;
        if (distA >= 10) { totalDist = distA / 10; xUnit = "nm"; }
        else { totalDist = distA; xUnit = "Å"; }
      } else {
        totalDist = distPx;
      }
    }

    // Draw x-axis ticks
    const tickY = padTop + plotH;
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    const idealTicks = Math.max(2, Math.floor(plotW / 70));
    const tickStep = roundToNiceValue(totalDist / idealTicks);
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textBaseline = "top";
    const ticks: number[] = [];
    for (let v = 0; v <= totalDist + tickStep * 0.01; v += tickStep) {
      if (v > totalDist * 1.001) break;
      ticks.push(v);
    }
    for (let i = 0; i < ticks.length; i++) {
      const v = ticks[i];
      const frac = totalDist > 0 ? v / totalDist : 0;
      const x = padLeft + frac * plotW;
      ctx.beginPath(); ctx.moveTo(x, tickY); ctx.lineTo(x, tickY + 3); ctx.stroke();
      ctx.textAlign = frac < 0.05 ? "left" : frac > 0.95 ? "right" : "center";
      const valStr = v % 1 === 0 ? v.toFixed(0) : v.toFixed(1);
      ctx.fillText(i === ticks.length - 1 ? `${valStr} ${xUnit}` : valStr, x, tickY + 4);
    }

    // Y-axis min/max labels
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(formatNumber(gMax), padLeft - 3, padTop);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatNumber(gMin), padLeft - 3, padTop + plotH);

    // Draw axis lines
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // Save base rendering + layout for hover overlay
    profileBaseImageRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    profileLayoutRef.current = { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit };
  }, [profileData, profilePoints, pixelSize, canvasW, themeInfo.theme, themeColors.accent, profileHeight]);

  // Profile hover handler — draws crosshair + value readout
  const handleProfileMouseMove = React.useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    const layout = profileLayoutRef.current;
    if (!canvas || !base || !layout) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const cssX = e.clientX - rect.left;
    const { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit } = layout;
    const range = gMax - gMin || 1;

    ctx.putImageData(base, 0, 0);
    if (cssX < padLeft || cssX > padLeft + plotW) return;
    const frac = (cssX - padLeft) / plotW;

    const dpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Vertical crosshair
    ctx.strokeStyle = themeInfo.theme === "dark" ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(cssX, padTop);
    ctx.lineTo(cssX, padTop + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Dot on profile line + value
    if (profileData && profileData.length >= 2) {
      const dataIdx = Math.min(profileData.length - 1, Math.max(0, Math.round(frac * (profileData.length - 1))));
      const val = profileData[dataIdx];
      const y = padTop + plotH - ((val - gMin) / range) * plotH;
      ctx.fillStyle = themeColors.accent;
      ctx.beginPath();
      ctx.arc(cssX, y, 3, 0, Math.PI * 2);
      ctx.fill();

      // Value readout label
      const dist = frac * totalDist;
      const label = `${formatNumber(val)}  @  ${dist.toFixed(1)} ${xUnit}`;
      const isDark = themeInfo.theme === "dark";
      ctx.font = "bold 9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      const textW = ctx.measureText(label).width;
      const labelX = Math.min(cssX + 6, padLeft + plotW - textW - 2);
      const labelY = padTop + 2;
      ctx.fillStyle = isDark ? "rgba(0,0,0,0.7)" : "rgba(255,255,255,0.8)";
      ctx.fillRect(labelX - 2, labelY - 1, textW + 4, 11);
      ctx.fillStyle = isDark ? "#fff" : "#000";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(label, labelX, labelY);
    }

    ctx.restore();
  }, [profileData, themeInfo.theme, themeColors.accent]);

  const handleProfileMouseLeave = React.useCallback(() => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    if (!canvas || !base) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(base, 0, 0);
  }, []);

  // Profile height resize
  React.useEffect(() => {
    if (!isResizingProfile) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!profileResizeStart) return;
      const delta = e.clientY - profileResizeStart.y;
      setProfileHeight(Math.max(40, Math.min(300, profileResizeStart.height + delta)));
    };
    const handleMouseUp = () => {
      setIsResizingProfile(false);
      setProfileResizeStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingProfile, profileResizeStart]);

  // Render HiDPI scale bar + zoom indicator + colorbar
  React.useEffect(() => {
    if (!uiRef.current) return;
    const ctx = uiRef.current.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, uiRef.current.width, uiRef.current.height);
    if (scaleBarVisible) {
      const pixelSizeAngstrom = pixelSize * 10;
      const unit = pixelSizeAngstrom > 0 ? "Å" as const : "px" as const;
      const pxSize = pixelSizeAngstrom > 0 ? pixelSizeAngstrom : 1;
      drawScaleBarHiDPI(uiRef.current, DPR, zoom, pxSize, unit, width);
    }
    if (showColorbar) {
      const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
      const { vmin, vmax } = sliderRange(imageDataRange.min, imageDataRange.max, imageVminPct, imageVmaxPct);
      const cssW = uiRef.current.width / DPR;
      const cssH = uiRef.current.height / DPR;
      ctx.save();
      ctx.scale(DPR, DPR);
      drawColorbar(ctx, cssW, cssH, lut, vmin, vmax, logScale);
      ctx.restore();
    }
  }, [pixelSize, scaleBarVisible, width, canvasW, canvasH, displayScale, zoom, showColorbar, cmap, imageDataRange, imageVminPct, imageVmaxPct, logScale]);

  // Compute FFT magnitude (expensive, async — only re-run on data/GPU changes)
  const fftMagRef = React.useRef<Float32Array | null>(null);
  const [fftMagVersion, setFftMagVersion] = React.useState(0);

  React.useEffect(() => {
    if (!showFft || !rawFrameDataRef.current || playing) return;
    let cancelled = false;

    const data = rawFrameDataRef.current;

    const computeFFT = async () => {
      let real: Float32Array, imag: Float32Array;

      if (gpuReady && gpuFFTRef.current) {
        const result = await gpuFFTRef.current.fft2D(data.slice(), new Float32Array(data.length), width, height, false);
        real = result.real;
        imag = result.imag;
      } else {
        real = data.slice();
        imag = new Float32Array(data.length);
        fft2d(real, imag, width, height, false);
      }

      if (cancelled) return;
      fftshift(real, width, height);
      fftshift(imag, width, height);

      fftMagRef.current = computeMagnitude(real, imag);
      fftMagCacheRef.current = fftMagRef.current;
      setFftMagVersion(v => v + 1);
    };

    computeFFT();
    return () => { cancelled = true; };
  }, [showFft, frameBytes, playing, width, height, gpuReady]);

  // Process FFT magnitude → histogram + colormap rendering (cheap, sync)
  React.useEffect(() => {
    const mag = fftMagRef.current;
    if (!showFft || !mag) return;

    let displayMin: number, displayMax: number;
    if (fftAuto) {
      ({ min: displayMin, max: displayMax } = autoEnhanceFFT(mag, width, height));
    } else {
      ({ min: displayMin, max: displayMax } = findDataRange(mag));
    }

    const displayData = fftLogScale ? applyLogScale(mag) : mag;
    if (fftLogScale) {
      displayMin = Math.log1p(displayMin);
      displayMax = Math.log1p(displayMax);
    }

    setFftHistogramData(displayData);
    setFftDataRange({ min: displayMin, max: displayMax });
    setFftStats(computeStats(displayData));

    const { vmin, vmax } = sliderRange(displayMin, displayMax, fftVminPct, fftVmaxPct);
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
    const offscreen = renderToOffscreen(displayData, width, height, lut, vmin, vmax);
    if (!offscreen) return;

    fftOffscreenRef.current = offscreen;

    if (fftCanvasRef.current) {
      const ctx = fftCanvasRef.current.getContext("2d");
      if (ctx) {
        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, canvasW, canvasH);
        ctx.save();
        ctx.translate(fftPanX, fftPanY);
        ctx.scale(fftZoom, fftZoom);
        ctx.drawImage(offscreen, 0, 0, canvasW, canvasH);
        ctx.restore();
      }
    }
  }, [showFft, fftMagVersion, fftLogScale, fftAuto, fftVminPct, fftVmaxPct, fftColormap, width, height, canvasW, canvasH]);

  // Redraw cached FFT with zoom/pan (cheap — no recomputation)
  React.useEffect(() => {
    if (!showFft || !fftCanvasRef.current || !fftOffscreenRef.current) return;
    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.save();
    ctx.translate(fftPanX, fftPanY);
    ctx.scale(fftZoom, fftZoom);
    ctx.drawImage(fftOffscreenRef.current, 0, 0, canvasW, canvasH);
    ctx.restore();
  }, [showFft, fftZoom, fftPanX, fftPanY, canvasW, canvasH]);

  // Render FFT overlay (reciprocal-space scale bar + colorbar)
  React.useEffect(() => {
    const overlay = fftOverlayRef.current;
    if (!overlay || !showFft) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    overlay.width = Math.round(canvasW * DPR);
    overlay.height = Math.round(canvasH * DPR);
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Reciprocal-space scale bar (pixelSize is in nm, convert to Å)
    if (pixelSize > 0) {
      const pixelSizeAngstrom = pixelSize * 10;
      const fftPixelSize = 1 / (width * pixelSizeAngstrom);
      drawFFTScaleBarHiDPI(overlay, DPR, fftZoom, fftPixelSize, width);
    }

    // FFT colorbar
    if (fftShowColorbar && fftDataRange.min !== fftDataRange.max) {
      const { vmin, vmax } = sliderRange(fftDataRange.min, fftDataRange.max, fftVminPct, fftVmaxPct);
      const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
      ctx.save();
      ctx.scale(DPR, DPR);
      const cssW = overlay.width / DPR;
      const cssH = overlay.height / DPR;
      drawColorbar(ctx, cssW, cssH, lut, vmin, vmax, fftLogScale);
      ctx.restore();
    }

    // D-spacing crosshair marker
    if (fftClickInfo) {
      ctx.save();
      ctx.scale(DPR, DPR);
      const screenX = fftPanX + fftZoom * (fftClickInfo.col / width * canvasW);
      const screenY = fftPanY + fftZoom * (fftClickInfo.row / height * canvasH);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.shadowColor = "rgba(0, 0, 0, 0.6)";
      ctx.shadowBlur = 2;
      ctx.lineWidth = 1.5;
      const r = 8;
      ctx.beginPath();
      ctx.moveTo(screenX - r, screenY); ctx.lineTo(screenX - 3, screenY);
      ctx.moveTo(screenX + 3, screenY); ctx.lineTo(screenX + r, screenY);
      ctx.moveTo(screenX, screenY - r); ctx.lineTo(screenX, screenY - 3);
      ctx.moveTo(screenX, screenY + 3); ctx.lineTo(screenX, screenY + r);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(screenX, screenY, 4, 0, Math.PI * 2);
      ctx.stroke();
      if (fftClickInfo.dSpacing != null) {
        const d = fftClickInfo.dSpacing;
        const label = d >= 10 ? `d = ${(d / 10).toFixed(2)} nm` : `d = ${d.toFixed(2)} Å`;
        ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
        ctx.fillStyle = "white";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(label, screenX + 10, screenY - 4);
      }
      ctx.restore();
    }
  }, [showFft, fftZoom, fftPanX, fftPanY, canvasW, canvasH, pixelSize, width, height, fftDataRange, fftVminPct, fftVmaxPct, fftColormap, fftLogScale, fftShowColorbar, fftClickInfo]);

  // Mouse handlers
  const handleWheel = (e: React.WheelEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * zoomFactor));
    const zoomRatio = newZoom / zoom;
    setZoom(newZoom);
    setPanX(mouseX - (mouseX - panX) * zoomRatio);
    setPanY(mouseY - (mouseY - panY) * zoomRatio);
  };

  const handleDoubleClick = () => {
    setZoom(1);
    setPanX(0);
    setPanY(0);
  };

  const handleExportPng = async () => {
    setExportAnchor(null);
    if (!canvasRef.current) return;
    const blob = await new Promise<Blob>((resolve) =>
      canvasRef.current!.toBlob((b) => resolve(b!), "image/png"));
    const label = labels?.[sliceIdx] || String(sliceIdx);
    downloadBlob(blob, "show3d_frame_" + label + ".png");
  };

  const handleExportPngAll = () => {
    setExportAnchor(null);
    setExporting(true);
    setZipExportRequested(true);
  };

  const handleExportGif = () => {
    setExportAnchor(null);
    setExporting(true);
    setGifExportRequested(true);
  };

  const handleCopy = async () => {
    if (!canvasRef.current) return;
    try {
      const blob = await new Promise<Blob | null>(resolve => canvasRef.current!.toBlob(resolve, "image/png"));
      if (!blob) return;
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    } catch {
      // Fallback: download if clipboard API unavailable
      canvasRef.current.toBlob((b) => {
        if (b) {
          const label = labels?.[sliceIdx] || String(sliceIdx);
          downloadBlob(b, "show3d_frame_" + label + ".png");
        }
      }, "image/png");
    }
  };

  // Export publication-quality figure
  const handleExportFigure = (withColorbar: boolean) => {
    setExportAnchor(null);
    const frameData = rawFrameDataRef.current;
    if (!frameData) return;

    const processed = logScale ? applyLogScale(frameData) : frameData;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;

    let vmin: number, vmax: number;
    if (autoContrast) {
      ({ vmin, vmax } = percentileClip(processed, percentileLow, percentileHigh));
    } else {
      const { min: pMin, max: pMax } = findDataRange(processed);
      ({ vmin, vmax } = sliderRange(pMin, pMax, imageVminPct, imageVmaxPct));
    }

    const offscreen = renderToOffscreen(processed, width, height, lut, vmin, vmax);
    if (!offscreen) return;

    // pixelSize is in nm, convert to Å for exportFigure
    const pixelSizeAngstrom = pixelSize > 0 ? pixelSize * 10 : 0;

    const figCanvas = exportFigure({
      imageCanvas: offscreen,
      title: title || undefined,
      lut,
      vmin,
      vmax,
      logScale,
      pixelSize: pixelSizeAngstrom > 0 ? pixelSizeAngstrom : undefined,
      showColorbar: withColorbar,
      showScaleBar: pixelSizeAngstrom > 0,
      drawAnnotations: (ctx) => {
        if (roiActive) {
          const shape = roiShape as "circle" | "square" | "rectangle" | "annular";
          drawROI(ctx, roiCol, roiRow, shape, roiRadius, roiWidth, roiHeight, "#ff4444", "#ff4444", false, roiRadiusInner);
        }
      },
    });

    figCanvas.toBlob((blob) => {
      if (blob) {
        const label = labels?.[sliceIdx] || String(sliceIdx);
        downloadBlob(blob, `show3d_figure_${label}.png`);
      }
    }, "image/png");
  };

  // Download GIF when data arrives from Python
  React.useEffect(() => {
    if (!gifData || gifData.byteLength === 0) return;
    downloadDataView(gifData, "show3d_animation.gif", "image/gif");
    setExporting(false);
  }, [gifData]);

  // Download ZIP when data arrives from Python
  React.useEffect(() => {
    if (!zipData || zipData.byteLength === 0) return;
    downloadDataView(zipData, "show3d_frames.zip", "application/zip");
    setExporting(false);
  }, [zipData]);

  const clickStartRef = React.useRef<{ x: number; y: number } | null>(null);

  const screenToImg = (e: React.MouseEvent): { imgCol: number; imgRow: number } => {
    const canvas = canvasRef.current;
    if (!canvas) return { imgCol: 0, imgRow: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const screenX = (e.clientX - rect.left) * scaleX;
    const screenY = (e.clientY - rect.top) * scaleY;
    return {
      imgCol: (screenX - panX) / (displayScale * zoom),
      imgRow: (screenY - panY) / (displayScale * zoom),
    };
  };

  const isNearResizeHandle = (imgCol: number, imgRow: number): boolean => {
    if (!roiActive) return false;
    const hitArea = RESIZE_HIT_AREA_PX / (displayScale * zoom);
    if (roiShape === "rectangle") {
      const hx = roiCol + roiWidth / 2;
      const hy = roiRow + roiHeight / 2;
      return Math.sqrt((imgCol - hx) ** 2 + (imgRow - hy) ** 2) < hitArea;
    }
    if (roiShape === "circle" || roiShape === "annular") {
      const offset = roiRadius * CIRCLE_HANDLE_ANGLE;
      const hx = roiCol + offset;
      const hy = roiRow + offset;
      return Math.sqrt((imgCol - hx) ** 2 + (imgRow - hy) ** 2) < hitArea;
    }
    if (roiShape === "square") {
      const hx = roiCol + roiRadius;
      const hy = roiRow + roiRadius;
      return Math.sqrt((imgCol - hx) ** 2 + (imgRow - hy) ** 2) < hitArea;
    }
    return false;
  };

  const isNearResizeHandleInner = (imgCol: number, imgRow: number): boolean => {
    if (!roiActive || roiShape !== "annular") return false;
    const offset = roiRadiusInner * CIRCLE_HANDLE_ANGLE;
    const hx = roiCol + offset;
    const hy = roiRow + offset;
    const hitArea = RESIZE_HIT_AREA_PX / (displayScale * zoom);
    return Math.sqrt((imgCol - hx) ** 2 + (imgRow - hy) ** 2) < hitArea;
  };

  const handleCanvasMouseDown = (e: React.MouseEvent) => {
    clickStartRef.current = { x: e.clientX, y: e.clientY };
    // Check if clicking on lens inset for drag or resize
    if (showLens) {
      const rect = canvasContainerRef.current?.getBoundingClientRect();
      if (rect) {
        const cssX = e.clientX - rect.left;
        const cssY = e.clientY - rect.top;
        const margin = 12;
        const lx = lensAnchor ? lensAnchor.x : margin;
        const ly = lensAnchor ? lensAnchor.y : canvasH - lensDisplaySize - margin - 20;
        if (cssX >= lx && cssX <= lx + lensDisplaySize && cssY >= ly && cssY <= ly + lensDisplaySize) {
          const edgeHit = 8;
          const nearEdge = cssX - lx < edgeHit || lx + lensDisplaySize - cssX < edgeHit ||
                           cssY - ly < edgeHit || ly + lensDisplaySize - cssY < edgeHit;
          if (nearEdge) {
            setIsResizingLens(true);
            lensResizeStartRef.current = { my: e.clientY, startSize: lensDisplaySize };
          } else {
            setIsDraggingLens(true);
            lensDragStartRef.current = { mx: e.clientX, my: e.clientY, ax: lx, ay: ly };
          }
          return;
        }
      }
    }
    if (roiActive) {
      const { imgCol, imgRow } = screenToImg(e);
      if (isNearResizeHandleInner(imgCol, imgRow)) {
        setIsDraggingResizeInner(true);
        return;
      }
      if (isNearResizeHandle(imgCol, imgRow)) {
        setIsDraggingResize(true);
        return;
      }
      setIsDraggingROI(true);
      updateROI(e);
    } else {
      setIsDraggingPan(true);
      setPanStart({ x: e.clientX, y: e.clientY, pX: panX, pY: panY });
    }
  };

  const handleCanvasMouseMove = (e: React.MouseEvent) => {
    // Cursor readout: convert screen position to image pixel coordinates
    const canvas = canvasRef.current;
    if (canvas && rawFrameDataRef.current) {
      const rect = canvas.getBoundingClientRect();
      const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
      const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
      const imageCanvasX = (mouseCanvasX - panX) / zoom;
      const imageCanvasY = (mouseCanvasY - panY) / zoom;
      const imgX = Math.floor(imageCanvasX / displayScale);
      const imgY = Math.floor(imageCanvasY / displayScale);
      if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
        const rawData = rawFrameDataRef.current;
        setCursorInfo({ row: imgY, col: imgX, value: rawData[imgY * width + imgX] });
        if (showLens) setLensPos({ row: imgY, col: imgX });
      } else {
        setCursorInfo(null);
        if (showLens) setLensPos(null);
      }
    }

    // Lens edge hover detection
    if (showLens) {
      const rect2 = canvasContainerRef.current?.getBoundingClientRect();
      if (rect2) {
        const cssX2 = e.clientX - rect2.left;
        const cssY2 = e.clientY - rect2.top;
        const margin = 12;
        const lx = lensAnchor ? lensAnchor.x : margin;
        const ly = lensAnchor ? lensAnchor.y : canvasH - lensDisplaySize - margin - 20;
        const inside = cssX2 >= lx && cssX2 <= lx + lensDisplaySize && cssY2 >= ly && cssY2 <= ly + lensDisplaySize;
        const edgeHit = 8;
        const nearEdge = inside && (cssX2 - lx < edgeHit || lx + lensDisplaySize - cssX2 < edgeHit ||
                                     cssY2 - ly < edgeHit || ly + lensDisplaySize - cssY2 < edgeHit);
        setIsHoveringLensEdge(nearEdge);
      }
    } else {
      setIsHoveringLensEdge(false);
    }

    // Lens drag
    if (isDraggingLens && lensDragStartRef.current) {
      const dx = e.clientX - lensDragStartRef.current.mx;
      const dy = e.clientY - lensDragStartRef.current.my;
      setLensAnchor({ x: lensDragStartRef.current.ax + dx, y: lensDragStartRef.current.ay + dy });
      return;
    }

    // Lens resize drag
    if (isResizingLens && lensResizeStartRef.current) {
      const dy = e.clientY - lensResizeStartRef.current.my;
      setLensDisplaySize(Math.max(64, Math.min(256, lensResizeStartRef.current.startSize + dy)));
      return;
    }

    // Resize handle dragging
    if (isDraggingResizeInner) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e);
      const newR = Math.sqrt((ic - roiCol) ** 2 + (ir - roiRow) ** 2);
      setRoiRadiusInner(Math.max(1, Math.min(roiRadius - 1, Math.round(newR))));
      return;
    }
    if (isDraggingResize) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e);
      if (roiShape === "rectangle") {
        setRoiWidth(Math.max(2, Math.round(Math.abs(ic - roiCol) * 2)));
        setRoiHeight(Math.max(2, Math.round(Math.abs(ir - roiRow) * 2)));
      } else {
        const newR = roiShape === "square" ? Math.max(Math.abs(ic - roiCol), Math.abs(ir - roiRow)) : Math.sqrt((ic - roiCol) ** 2 + (ir - roiRow) ** 2);
        const minR = roiShape === "annular" ? roiRadiusInner + 1 : 1;
        setRoiRadius(Math.max(minR, Math.round(newR)));
      }
      return;
    }

    // Hover state for resize handles
    if (roiActive && !isDraggingROI && !isDraggingPan) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e);
      setIsHoveringResizeInner(isNearResizeHandleInner(ic, ir));
      setIsHoveringResize(isNearResizeHandle(ic, ir));
    }

    if (isDraggingROI) {
      updateROI(e);
    } else if (isDraggingPan && panStart) {
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - panStart.x) * scaleX;
      const dy = (e.clientY - panStart.y) * scaleY;
      setPanX(panStart.pX + dx);
      setPanY(panStart.pY + dy);
    }
  };

  const handleCanvasMouseUp = (e: React.MouseEvent) => {
    // Profile click capture
    if (profileActive && clickStartRef.current) {
      const dx = e.clientX - clickStartRef.current.x;
      const dy = e.clientY - clickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const canvas = canvasRef.current;
        if (canvas && rawFrameDataRef.current) {
          const rect = canvas.getBoundingClientRect();
          const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
          const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
          const imgX = (mouseCanvasX - panX) / zoom / displayScale;
          const imgY = (mouseCanvasY - panY) / zoom / displayScale;
          if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
            const pt = { row: imgY, col: imgX };
            if (profilePoints.length === 0 || profilePoints.length === 2) {
              setProfileLine([pt]);
              setProfileData(null);
            } else {
              const p0 = profilePoints[0];
              setProfileLine([p0, pt]);
              setProfileData(sampleLineProfile(rawFrameDataRef.current, width, height, p0.row, p0.col, pt.row, pt.col, profileWidth));
            }
          }
        }
      }
    }
    clickStartRef.current = null;
    setIsDraggingROI(false);
    setIsDraggingResize(false);
    setIsDraggingResizeInner(false);
    setIsDraggingLens(false);
    lensDragStartRef.current = null;
    setIsResizingLens(false);
    lensResizeStartRef.current = null;
    setIsDraggingPan(false);
    setPanStart(null);
  };

  const handleCanvasMouseLeave = () => {
    setCursorInfo(null);
    if (showLens) setLensPos(null);
    setIsDraggingROI(false);
    setIsDraggingResize(false);
    setIsDraggingResizeInner(false);
    setIsDraggingLens(false);
    lensDragStartRef.current = null;
    setIsResizingLens(false);
    lensResizeStartRef.current = null;
    setIsHoveringLensEdge(false);
    setIsHoveringResize(false);
    setIsHoveringResizeInner(false);
    setIsDraggingPan(false);
    setPanStart(null);
  };

  // FFT mouse handlers
  const [isFftDragging, setIsFftDragging] = React.useState(false);
  const [fftPanStart, setFftPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);

  const handleFftWheel = (e: React.WheelEvent) => {
    const canvas = fftCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, fftZoom * zoomFactor));
    const zoomRatio = newZoom / fftZoom;
    setFftZoom(newZoom);
    setFftPanX(mouseX - (mouseX - fftPanX) * zoomRatio);
    setFftPanY(mouseY - (mouseY - fftPanY) * zoomRatio);
  };

  // Convert FFT canvas mouse position to FFT image pixel coordinates
  const fftScreenToImg = (e: React.MouseEvent): { col: number; row: number } | null => {
    const canvas = fftCanvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;
    const imgCol = ((mouseX - fftPanX) / fftZoom) / canvasW * width;
    const imgRow = ((mouseY - fftPanY) / fftZoom) / canvasH * height;
    if (imgCol >= 0 && imgCol < width && imgRow >= 0 && imgRow < height) {
      return { col: imgCol, row: imgRow };
    }
    return null;
  };

  const handleFftMouseDown = (e: React.MouseEvent) => {
    fftClickStartRef.current = { x: e.clientX, y: e.clientY };
    setIsFftDragging(true);
    setFftPanStart({ x: e.clientX, y: e.clientY, pX: fftPanX, pY: fftPanY });
  };

  const handleFftMouseMove = (e: React.MouseEvent) => {
    if (isFftDragging && fftPanStart) {
      const canvas = fftCanvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - fftPanStart.x) * scaleX;
      const dy = (e.clientY - fftPanStart.y) * scaleY;
      setFftPanX(fftPanStart.pX + dx);
      setFftPanY(fftPanStart.pY + dy);
    }
  };

  const handleFftMouseUp = (e: React.MouseEvent) => {
    // Click detection for d-spacing measurement
    if (fftClickStartRef.current) {
      const dx = e.clientX - fftClickStartRef.current.x;
      const dy = e.clientY - fftClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const pos = fftScreenToImg(e);
        if (pos) {
          let imgCol = pos.col;
          let imgRow = pos.row;
          if (fftMagCacheRef.current) {
            const snapped = findFFTPeak(fftMagCacheRef.current, width, height, imgCol, imgRow, FFT_SNAP_RADIUS);
            imgCol = snapped.col;
            imgRow = snapped.row;
          }
          const halfW = Math.floor(width / 2);
          const halfH = Math.floor(height / 2);
          const dcol = imgCol - halfW;
          const drow = imgRow - halfH;
          const distPx = Math.sqrt(dcol * dcol + drow * drow);
          if (distPx < 1) {
            setFftClickInfo(null);
          } else {
            let spatialFreq: number | null = null;
            let dSpacing: number | null = null;
            if (pixelSize > 0) {
              const pixelSizeAngstrom = pixelSize * 10;
              const paddedW = nextPow2(width);
              const paddedH = nextPow2(height);
              const binC = ((Math.round(imgCol) - halfW) % width + width) % width;
              const binR = ((Math.round(imgRow) - halfH) % height + height) % height;
              const freqC = binC <= paddedW / 2 ? binC / (paddedW * pixelSizeAngstrom) : (binC - paddedW) / (paddedW * pixelSizeAngstrom);
              const freqR = binR <= paddedH / 2 ? binR / (paddedH * pixelSizeAngstrom) : (binR - paddedH) / (paddedH * pixelSizeAngstrom);
              spatialFreq = Math.sqrt(freqC * freqC + freqR * freqR);
              dSpacing = spatialFreq > 0 ? 1 / spatialFreq : null;
            }
            setFftClickInfo({ row: imgRow, col: imgCol, distPx, spatialFreq, dSpacing });
          }
        }
      }
      fftClickStartRef.current = null;
    }
    setIsFftDragging(false);
    setFftPanStart(null);
  };

  const handleFftReset = () => {
    setFftZoom(1);
    setFftPanX(0);
    setFftPanY(0);
    setFftClickInfo(null);
  };

  const fftNeedsReset = fftZoom !== 1 || fftPanX !== 0 || fftPanY !== 0;

  const updateROI = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const screenX = (e.clientX - rect.left) * scaleX;
    const screenY = (e.clientY - rect.top) * scaleY;
    const x = Math.floor((screenX - panX) / (displayScale * zoom));
    const y = Math.floor((screenY - panY) / (displayScale * zoom));
    setRoiCol(Math.max(0, Math.min(width - 1, x)));
    setRoiRow(Math.max(0, Math.min(height - 1, y)));
  };

  // Resize handlers
  const handleMainResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizingMain(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: mainCanvasSize });
  };

  React.useEffect(() => {
    if (!isResizingMain) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeStart) return;
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);
      // Minimum is the initial size, maximum is 800px
      setMainCanvasSize(Math.max(initialCanvasSizeRef.current, Math.min(800, resizeStart.size + delta)));
    };
    const handleMouseUp = () => {
      setIsResizingMain(false);
      setResizeStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingMain, resizeStart]);

  // Keyboard
  const handleKeyDown = (e: React.KeyboardEvent) => {
    switch (e.key) {
      case " ":
        e.preventDefault();
        if (!showFft) setPlaying(!playing);
        break;
      case "ArrowLeft": {
        e.preventDefault();
        const lo = loop ? Math.max(0, loopStart) : 0;
        setSliceIdx(Math.max(lo, sliceIdx - 1));
        break;
      }
      case "ArrowRight": {
        e.preventDefault();
        const hi = loop ? Math.min(effectiveLoopEnd, nSlices - 1) : nSlices - 1;
        setSliceIdx(Math.min(hi, sliceIdx + 1));
        break;
      }
      case "Home":
        e.preventDefault();
        setSliceIdx(loop ? Math.max(0, loopStart) : 0);
        break;
      case "End":
        e.preventDefault();
        setSliceIdx(loop ? Math.min(effectiveLoopEnd, nSlices - 1) : nSlices - 1);
        break;
      case "r":
      case "R":
        handleDoubleClick();
        break;
      case "c":
      case "C":
        if (cursorInfo) {
          navigator.clipboard.writeText(`(${cursorInfo.row}, ${cursorInfo.col}, ${cursorInfo.value})`);
        }
        break;
    }
  };

  // Check if view needs reset
  const needsReset = zoom !== 1 || panX !== 0 || panY !== 0;

  return (
    <Box className="show3d-root" tabIndex={0} onKeyDown={handleKeyDown} sx={{ ...container.root, bgcolor: themeColors.bg, color: themeColors.text }}>
      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        <Box>
          {/* Title row */}
          <Typography variant="caption" sx={{ ...typography.label, color: themeColors.accent, mb: `${SPACING.XS}px`, display: "block" }}>
            {title || "Image"}
            <InfoTooltip text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT: Show power spectrum (Fourier transform) alongside image.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Profile: Click two points on image to draw a line intensity profile.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Lens: Magnifier inset that follows the cursor.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Scale: Linear or logarithmic intensity mapping.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto: Percentile-based contrast (clips outliers). FFT Auto masks DC + clips to 99.9th.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>ROI: Region of Interest — click to place, drag to move. Tracks mean intensity across frames.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Loop: Loop playback. Drag end markers on slider for loop range.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Bounce: Ping-pong playback — alternates forward and reverse.</Typography>
              <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
              <KeyboardShortcuts items={[["Space", "Play / Pause"], ["← / →", `Prev / Next ${dimLabel.toLowerCase()}`], ["Home / End", `First / Last ${dimLabel.toLowerCase()}`], ["R", "Reset zoom"], ["C", "Copy cursor coords"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]]} />
            </Box>} theme={themeInfo.theme} />
          </Typography>
          {/* Controls row */}
          <Box sx={{ display: "flex", alignItems: "center", gap: "4px", mb: `${SPACING.XS}px`, height: 28, maxWidth: canvasW }}>
            <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT:</Typography>
            <Switch checked={showFft} onChange={(e) => setShowFft(e.target.checked)} size="small" sx={switchStyles.small} />
            <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Profile:</Typography>
            <Switch checked={profileActive} onChange={(e) => { const on = e.target.checked; setProfileActive(on); if (on) { setRoiActive(false); setRoiShape("none"); } else { setProfileLine([]); setProfileData(null); } }} size="small" sx={switchStyles.small} />
            <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Lens:</Typography>
            <Switch checked={showLens} onChange={() => { if (!showLens) { setShowLens(true); setLensPos({ row: Math.floor(height / 2), col: Math.floor(width / 2) }); } else { setShowLens(false); setLensPos(null); } }} size="small" sx={switchStyles.small} />
            <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>ROI:</Typography>
            <Switch checked={roiActive} onChange={(e) => { const on = e.target.checked; if (on) { setRoiActive(true); if (!roiShape || roiShape === "none") setRoiShape("circle"); setProfileActive(false); setProfileLine([]); setProfileData(null); setRoiCol(Math.floor(width / 2)); setRoiRow(Math.floor(height / 2)); } else { setRoiActive(false); setRoiShape("none"); } }} size="small" sx={switchStyles.small} />
            <Box sx={{ flex: 1 }} />
            <Box sx={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={(e) => setExportAnchor(e.currentTarget)} disabled={exporting}>{exporting ? "..." : "Export"}</Button>
              <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                <MenuItem onClick={() => handleExportFigure(true)} sx={{ fontSize: 12 }}>Figure + colorbar</MenuItem>
                <MenuItem onClick={() => handleExportFigure(false)} sx={{ fontSize: 12 }}>Figure</MenuItem>
                <MenuItem onClick={handleExportPng} sx={{ fontSize: 12 }}>PNG (current frame)</MenuItem>
                <MenuItem onClick={handleExportPngAll} sx={{ fontSize: 12 }}>PNG (all frames .zip)</MenuItem>
                <MenuItem onClick={handleExportGif} sx={{ fontSize: 12 }}>GIF (fps: {fps})</MenuItem>
              </Menu>
              <Button size="small" sx={compactButton} onClick={handleCopy}>Copy</Button>
              <Button size="small" sx={compactButton} disabled={!needsReset} onClick={handleDoubleClick}>Reset</Button>
            </Box>
          </Box>
          <Box
            ref={canvasContainerRef}
            sx={{ ...container.imageBox, width: canvasW, height: canvasH, cursor: isHoveringLensEdge ? "nwse-resize" : (isHoveringResize || isDraggingResize || isHoveringResizeInner || isDraggingResizeInner) ? "nwse-resize" : (roiActive || profileActive) ? "crosshair" : "grab" }}
            onMouseDown={handleCanvasMouseDown}
            onMouseMove={handleCanvasMouseMove}
            onMouseUp={handleCanvasMouseUp}
            onMouseLeave={handleCanvasMouseLeave}
            onWheel={handleWheel}
            onDoubleClick={handleDoubleClick}
          >
            <canvas ref={canvasRef} width={canvasW} height={canvasH} style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }} />
            <canvas ref={overlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} />
            <canvas ref={uiRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} />
            <canvas ref={lensCanvasRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} />
            {/* Cursor readout overlay */}
            {cursorInfo && (
              <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                  ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                </Typography>
              </Box>
            )}
            <Box onMouseDown={handleMainResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, borderRadius: "0 0 4px 0", "&:hover": { opacity: 1 } }} />
          </Box>
          {/* Statistics bar - right below the image */}
          {showStats && (
            <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", maxWidth: canvasW, boxSizing: "border-box" }}>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.mean : statsMean)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.min : statsMin)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.max : statsMax)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.std : statsStd)}</Box></Typography>
              {roiActive && !localStats && (
                <>
                  <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 14 }} />
                  <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>ROI <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(roiMean)}</Box></Typography>
                  <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(roiMin)}</Box></Typography>
                  <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(roiMax)}</Box></Typography>
                  <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(roiStd)}</Box></Typography>
                </>
              )}
            </Box>
          )}
          {/* Line profile sparkline */}
          {profileActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, maxWidth: canvasW, boxSizing: "border-box" }}>
              <canvas
                ref={profileCanvasRef}
                onMouseMove={handleProfileMouseMove}
                onMouseLeave={handleProfileMouseLeave}
                style={{ width: canvasW, height: profileHeight, display: "block", border: `1px solid ${themeColors.border}`, borderBottom: "none", cursor: "crosshair" }}
              />
              <div
                onMouseDown={(e) => { e.preventDefault(); setIsResizingProfile(true); setProfileResizeStart({ y: e.clientY, height: profileHeight }); }}
                style={{ width: canvasW, height: 4, cursor: "ns-resize", borderLeft: `1px solid ${themeColors.border}`, borderRight: `1px solid ${themeColors.border}`, borderBottom: `1px solid ${themeColors.border}`, background: `linear-gradient(to bottom, ${themeColors.border}, transparent)` }}
              />
            </Box>
          )}
          {/* ROI sparkline plot */}
          {roiActive && showRoiPlot && roiPlotData && roiPlotData.byteLength >= 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, maxWidth: canvasW, boxSizing: "border-box" }}>
              <canvas
                ref={roiPlotCanvasRef}
                style={{ width: canvasW, height: 76, display: "block", border: `1px solid ${themeColors.border}` }}
              />
            </Box>
          )}
          {/* Image Controls - two rows with histogram on right (like Show4DSTEM) */}
          <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: canvasW, boxSizing: "border-box" }}>
            {/* Left: two rows of controls */}
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
              {/* Row 1: Scale + Auto + Color */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Scale:</Typography>
                <Select value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="linear">Lin</MenuItem>
                  <MenuItem value="log">Log</MenuItem>
                </Select>
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Auto:</Typography>
                <Switch checked={autoContrast} onChange={(e) => setAutoContrast(e.target.checked)} size="small" sx={switchStyles.small} />
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Colorbar:</Typography>
                <Switch checked={showColorbar} onChange={(e) => setShowColorbar(e.target.checked)} size="small" sx={switchStyles.small} />
              </Box>
              {/* Row 2: Color + zoom indicator */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Color:</Typography>
                <Select size="small" value={cmap} onChange={(e) => setCmap(e.target.value)} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }}>
                  {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                </Select>
                {zoom !== 1 && (
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.accent, fontWeight: "bold" }}>{zoom.toFixed(1)}x</Typography>
                )}
              </Box>
            </Box>
            {/* Right: Histogram spanning both rows */}
            <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
              <Histogram
                data={imageHistogramData}
                colormap={cmap}
                vminPct={imageVminPct}
                vmaxPct={imageVmaxPct}
                onRangeChange={(min, max) => { setImageVminPct(min); setImageVmaxPct(max); }}
                width={110}
                height={58}
                theme={themeInfo.theme === "dark" ? "dark" : "light"}
                dataMin={imageDataRange.min}
                dataMax={imageDataRange.max}
              />
            </Box>
          </Box>
          {/* Lens settings row (when Lens is active) */}
          {showLens && (
            <Box sx={{ mt: `${SPACING.XS}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, width: "fit-content" }}>
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Lens {lensMag}×</Typography>
                <Slider value={lensMag} min={2} max={8} step={1} onChange={(_, v) => setLensMag(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35 }} />
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>{lensDisplaySize}px</Typography>
                <Slider value={lensDisplaySize} min={64} max={256} step={16} onChange={(_, v) => setLensDisplaySize(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35 }} />
              </Box>
            </Box>
          )}
          {/* ROI settings row (when ROI is active) */}
          {roiActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, width: "fit-content" }}>
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>ROI:</Typography>
                <Select
                  size="small"
                  value={roiShape || "circle"}
                  onChange={(e) => { setRoiShape(e.target.value as RoiShape); }}
                  MenuProps={themedMenuProps}
                  sx={{ ...themedSelect, minWidth: 70 }}
                >
                  {(["circle", "square", "rectangle", "annular"] as const).map((shape) => (<MenuItem key={shape} value={shape}>{shape.charAt(0).toUpperCase() + shape.slice(1)}</MenuItem>))}
                </Select>
                {roiShape === "rectangle" && (
                  <>
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>W</Typography>
                    <Slider value={roiWidth} min={5} max={width} onChange={(_, v) => setRoiWidth(v as number)} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>H</Typography>
                    <Slider value={roiHeight} min={5} max={height} onChange={(_, v) => setRoiHeight(v as number)} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                  </>
                )}
                {roiShape === "annular" && (
                  <>
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Inner</Typography>
                    <Slider value={roiRadiusInner} min={1} max={roiRadius - 1} onChange={(_, v) => setRoiRadiusInner(v as number)} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Outer</Typography>
                    <Slider value={roiRadius} min={roiRadiusInner + 1} max={Math.max(width, height)} onChange={(_, v) => setRoiRadius(v as number)} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                  </>
                )}
                {roiShape !== "rectangle" && roiShape !== "annular" && (
                  <>
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Size</Typography>
                    <Slider value={roiRadius} min={5} max={Math.max(width, height)} onChange={(_, v) => setRoiRadius(v as number)} size="small" sx={{ ...sliderStyles.small, width: 50 }} />
                  </>
                )}
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Plot:</Typography>
                <Switch checked={showRoiPlot} onChange={(e) => setShowRoiPlot(e.target.checked)} size="small" sx={switchStyles.small} />
              </Box>
            </Box>
          )}
        </Box>

        {/* FFT Panel - same size as main image, canvas-aligned with spacer */}
        {showFft && (
          <Box sx={{ width: canvasW }}>
            {/* Spacer — matches main panel title row height for canvas alignment */}
            <Box sx={{ mb: `${SPACING.XS}px`, height: 16 }} />
            {/* Controls row — matches main panel controls row height */}
            <Stack direction="row" justifyContent="flex-end" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
              <Button size="small" sx={compactButton} disabled={!fftNeedsReset} onClick={handleFftReset}>Reset</Button>
            </Stack>
            {/* FFT Canvas - same size as main image */}
            <Box
              ref={fftContainerRef}
              sx={{ ...container.imageBox, width: canvasW, height: canvasH, cursor: "grab" }}
              onMouseDown={handleFftMouseDown}
              onMouseMove={handleFftMouseMove}
              onMouseUp={handleFftMouseUp}
              onMouseLeave={() => { fftClickStartRef.current = null; setIsFftDragging(false); setFftPanStart(null); }}
              onWheel={handleFftWheel}
              onDoubleClick={handleFftReset}
            >
              <canvas ref={fftCanvasRef} width={canvasW} height={canvasH} style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }} />
              <canvas ref={fftOverlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} />
            </Box>
            {/* FFT Statistics bar */}
            {showStats && (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, flexWrap: "wrap" }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.mean)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.min)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.max)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.std)}</Box></Typography>
                {fftClickInfo && (
                  <>
                    <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 14 }} />
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                      {fftClickInfo.dSpacing != null ? (
                        <>d = <Box component="span" sx={{ color: themeColors.accent, fontWeight: "bold" }}>{fftClickInfo.dSpacing >= 10 ? `${(fftClickInfo.dSpacing / 10).toFixed(2)} nm` : `${fftClickInfo.dSpacing.toFixed(2)} Å`}</Box>{" | |g| = "}<Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.spatialFreq!.toFixed(4)} Å⁻¹</Box></>
                      ) : (
                        <>dist = <Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.distPx.toFixed(1)} px</Box></>
                      )}
                    </Typography>
                  </>
                )}
              </Box>
            )}
            {/* FFT Controls - two rows with histogram on right (like Show4DSTEM) */}
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: canvasW, boxSizing: "border-box" }}>
              {/* Left: two rows of controls */}
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: Scale + Auto */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Scale:</Typography>
                  <Select value={fftLogScale ? "log" : "linear"} onChange={(e) => setFftLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Auto:</Typography>
                  <Switch checked={fftAuto} onChange={(e) => setFftAuto(e.target.checked)} size="small" sx={switchStyles.small} />
                </Box>
                {/* Row 2: Color + Colorbar */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Color:</Typography>
                  <Select value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={themedMenuProps}>
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Colorbar:</Typography>
                  <Switch checked={fftShowColorbar} onChange={(e) => setFftShowColorbar(e.target.checked)} size="small" sx={switchStyles.small} />
                </Box>
              </Box>
              {/* Right: Histogram spanning both rows */}
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
                <Histogram
                  data={fftHistogramData}
                  vminPct={fftVminPct}
                  vmaxPct={fftVmaxPct}
                  onRangeChange={(min, max) => { setFftVminPct(min); setFftVmaxPct(max); }}
                  width={110}
                  height={58}
                  theme={themeInfo.theme}
                  dataMin={fftDataRange.min}
                  dataMax={fftDataRange.max}
                />
              </Box>
            </Box>
          </Box>
        )}
      </Stack>

      {/* Playback controls - two rows, constrained to image width */}
      {/* Row 1: Transport controls + position slider (with loop range handles when Loop is ON) */}
      {(() => { const activeIdx = playing ? displaySliceIdx : sliceIdx; return (<>
      <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, width: canvasW, maxWidth: canvasW, boxSizing: "border-box" }}>
        {!showFft && (
        <Stack direction="row" spacing={0} sx={{ flexShrink: 0, mr: 0.5 }}>
          <IconButton size="small" onClick={() => { setReverse(true); setPlaying(true); }} sx={{ color: reverse && playing ? themeColors.accent : themeColors.textMuted, p: 0.25 }}>
            <FastRewindIcon sx={{ fontSize: 18 }} />
          </IconButton>
          <IconButton size="small" onClick={() => setPlaying(!playing)} sx={{ color: themeColors.accent, p: 0.25 }}>
            {playing ? <PauseIcon sx={{ fontSize: 18 }} /> : <PlayArrowIcon sx={{ fontSize: 18 }} />}
          </IconButton>
          <IconButton size="small" onClick={() => { setReverse(false); setPlaying(true); }} sx={{ color: !reverse && playing ? themeColors.accent : themeColors.textMuted, p: 0.25 }}>
            <FastForwardIcon sx={{ fontSize: 18 }} />
          </IconButton>
          <IconButton size="small" onClick={() => { setPlaying(false); setSliceIdx(loop ? Math.max(0, loopStart) : 0); }} sx={{ color: themeColors.textMuted, p: 0.25 }}>
            <StopIcon sx={{ fontSize: 16 }} />
          </IconButton>
        </Stack>
        )}
        {loop && !showFft ? (
          <Slider
            value={[loopStart, activeIdx, effectiveLoopEnd]}
            onChange={(_, v) => {
              const vals = v as number[];
              setLoopStart(vals[0]);
              if (playing) setPlaying(false);
              setSliceIdx(vals[1]);
              setLoopEnd(vals[2]);
            }}
            disableSwap
            min={0}
            max={nSlices - 1}
            size="small"
            valueLabelDisplay="auto"
            valueLabelFormat={(v) => `${v + 1}`}
            marks={bookmarkedFrames.map(f => ({ value: f }))}
            sx={{
              ...sliderStyles.small,
              flex: 1,
              minWidth: 40,
              "& .MuiSlider-thumb[data-index='0']": { width: 8, height: 8, bgcolor: themeColors.textMuted },
              "& .MuiSlider-thumb[data-index='1']": { width: 12, height: 12 },
              "& .MuiSlider-thumb[data-index='2']": { width: 8, height: 8, bgcolor: themeColors.textMuted },
              "& .MuiSlider-mark": { bgcolor: themeColors.accent, width: 4, height: 4, borderRadius: "50%", top: "50%", transform: "translate(-50%, -50%)" },
              "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
            }}
          />
        ) : (
          <Slider value={activeIdx} min={0} max={nSlices - 1} onChange={(_, v) => { if (playing) setPlaying(false); setSliceIdx(v as number); }} size="small" marks={bookmarkedFrames.map(f => ({ value: f }))} sx={{ ...sliderStyles.small, flex: 1, minWidth: 40, "& .MuiSlider-mark": { bgcolor: themeColors.accent, width: 4, height: 4, borderRadius: "50%", top: "50%", transform: "translate(-50%, -50%)" } }} />
        )}
        <Typography sx={{ ...typography.value, color: themeColors.textMuted, minWidth: 28, textAlign: "right", flexShrink: 0 }}>{activeIdx + 1}/{nSlices}{timestamps && timestamps.length > 0 && activeIdx < timestamps.length && ` (${formatNumber(timestamps[activeIdx])} ${timestampUnit})`}</Typography>
      </Box>
      {/* Row 2: FPS, Loop, Bounce, Bookmark (hidden when FFT is on unless show_playback) */}
      {(!showFft || showPlayback) && (<Box sx={{ ...controlRow, mt: `${SPACING.XS}px`, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, width: canvasW, maxWidth: canvasW, boxSizing: "border-box" }}>
        <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>fps</Typography>
        <Slider value={fps} min={1} max={60} step={1} onChange={(_, v) => setFps(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35, flexShrink: 0 }} />
        <Typography sx={{ ...typography.label, color: themeColors.textMuted, minWidth: 14, flexShrink: 0 }}>{Math.round(fps)}</Typography>
        <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>Loop</Typography>
        <Switch size="small" checked={loop} onChange={() => setLoop(!loop)} sx={{ ...switchStyles.small, flexShrink: 0 }} />
        <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>Bounce</Typography>
        <Switch size="small" checked={boomerang} onChange={() => setBoomerang(!boomerang)} sx={{ ...switchStyles.small, flexShrink: 0 }} />
        <Tooltip title="Bookmark current frame" arrow>
          <IconButton size="small" onClick={() => {
            const set = new Set(bookmarkedFrames);
            if (set.has(activeIdx)) { set.delete(activeIdx); } else { set.add(activeIdx); }
            setBookmarkedFrames(Array.from(set).sort((a, b) => a - b));
          }} sx={{ color: bookmarkedFrames.includes(activeIdx) ? themeColors.accent : themeColors.textMuted, p: 0.25, flexShrink: 0 }}>
            <Typography sx={{ fontSize: 14, lineHeight: 1 }}>{bookmarkedFrames.includes(activeIdx) ? "\u2605" : "\u2606"}</Typography>
          </IconButton>
        </Tooltip>
        {loop && (loopStart > 0 || (loopEnd >= 0 && loopEnd < nSlices - 1)) && (
          <IconButton size="small" onClick={() => { setLoopStart(0); setLoopEnd(-1); }} sx={{ color: themeColors.textMuted, p: 0.25, flexShrink: 0 }} title="Reset loop range">
            <Typography sx={{ fontSize: 10, lineHeight: 1 }}>Reset</Typography>
          </IconButton>
        )}
      </Box>)}
      </>); })()}


    </Box>
  );
}

export const render = createRender(Show3D);
