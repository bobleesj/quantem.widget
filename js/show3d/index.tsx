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
import { drawScaleBarHiDPI } from "../scalebar";
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
  gap: `${SPACING.SM}px`,
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

// ROI drawing
function drawROI(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  shape: "circle" | "square" | "rectangle",
  radius: number,
  width: number,
  height: number,
  activeColor: string,
  inactiveColor: string,
  active: boolean = false
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

// ============================================================================
// Histogram Component
// ============================================================================

interface HistogramProps {
  data: Float32Array | null;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
  dataMin?: number;
  dataMax?: number;
}

/**
 * Histogram component with integrated vmin/vmax slider.
 * Shows data distribution with adjustable clipping.
 */
function Histogram({
  data,
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

import { WebGPUFFT, getWebGPUFFT, fft2d, fftshift, computeMagnitude, autoEnhanceFFT } from "../webgpu-fft";

// ============================================================================
// Constants
// ============================================================================
const CANVAS_TARGET_SIZE = 400;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;

const ROI_SHAPES = ["none", "circle", "square", "rectangle"] as const;
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
    accentGreen: themeInfo.theme === "dark" ? "#0f0" : "#0a0",
    accentYellow: themeInfo.theme === "dark" ? "#ff0" : "#cc0",
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
  const [roiWidth, setRoiWidth] = useModelState<number>("roi_width");
  const [roiHeight, setRoiHeight] = useModelState<number>("roi_height");
  const [roiMean] = useModelState<number>("roi_mean");

  // FFT
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");

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

  // Local state
  const [isDraggingROI, setIsDraggingROI] = React.useState(false);
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

  // FFT zoom/pan state
  const [fftZoom, setFftZoom] = React.useState(1);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);
  const fftContainerRef = React.useRef<HTMLDivElement>(null);

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

  // Sync boomerang direction ref with reverse state
  React.useEffect(() => {
    bounceDirRef.current = reverse ? -1 : 1;
  }, [reverse]);

  // All playback params as a single ref (avoids stale closures in rAF loop)
  const playRef = React.useRef({
    fps, reverse, boomerang, loop, loopStart, loopEnd: effectiveLoopEnd,
    nSlices, width, height, displayScale, canvasW, canvasH,
    logScale, autoContrast, percentileLow, percentileHigh,
    dataMin, dataMax, cmap, imageVminPct, imageVmaxPct,
    zoom, panX, panY,
  });
  React.useEffect(() => {
    playRef.current = {
      fps, reverse, boomerang, loop, loopStart, loopEnd: effectiveLoopEnd,
      nSlices, width, height, displayScale, canvasW, canvasH,
      logScale, autoContrast, percentileLow, percentileHigh,
      dataMin, dataMax, cmap, imageVminPct, imageVmaxPct,
      zoom, panX, panY,
    };
  }, [fps, reverse, boomerang, loop, loopStart, effectiveLoopEnd,
    nSlices, width, height, displayScale, canvasW, canvasH,
    logScale, autoContrast, percentileLow, percentileHigh,
    dataMin, dataMax, cmap, imageVminPct, imageVmaxPct,
    zoom, panX, panY]);

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
      const rangeStart = c.loop ? Math.max(0, Math.min(c.loopStart, c.nSlices - 1)) : 0;
      const rangeEnd = c.loop ? Math.max(rangeStart, Math.min(c.loopEnd, c.nSlices - 1)) : c.nSlices - 1;
      const prev = playbackIdxRef.current;

      let next: number;
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

  // Update frame ref and histogram when frame changes (debounce histogram during playback)
  React.useEffect(() => {
    const parsed = extractFloat32(frameBytes);
    if (!parsed || parsed.length === 0) return;
    rawFrameDataRef.current = parsed;
    if (!playing) {
      setImageHistogramData(parsed);
      setImageDataRange(findDataRange(parsed));
    }
  }, [frameBytes, playing]);

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

  // Render overlay (ROI only)
  React.useEffect(() => {
    if (!overlayRef.current) return;
    const ctx = overlayRef.current.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvasW, canvasH);
    if (roiActive) {
      const screenX = roiCol * displayScale * zoom + panX;
      const screenY = roiRow * displayScale * zoom + panY;
      const screenRadius = roiRadius * displayScale * zoom;
      const screenWidth = roiWidth * displayScale * zoom;
      const screenHeight = roiHeight * displayScale * zoom;
      drawROI(ctx, screenX, screenY, roiShape || "circle", screenRadius, screenWidth, screenHeight, themeColors.accentYellow, themeColors.accentGreen, isDraggingROI);
    }
  }, [roiActive, roiShape, roiRow, roiCol, roiRadius, roiWidth, roiHeight, isDraggingROI, canvasW, canvasH, displayScale, zoom, panX, panY, themeColors]);

  // Render HiDPI scale bar + zoom indicator
  React.useEffect(() => {
    if (!uiRef.current) return;
    if (scaleBarVisible) {
      const pixelSizeAngstrom = pixelSize * 10;
      const unit = pixelSizeAngstrom > 0 ? "Å" as const : "px" as const;
      const pxSize = pixelSizeAngstrom > 0 ? pixelSizeAngstrom : 1;
      drawScaleBarHiDPI(uiRef.current, DPR, zoom, pxSize, unit, width);
    } else {
      const ctx = uiRef.current.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, uiRef.current.width, uiRef.current.height);
    }
  }, [pixelSize, scaleBarVisible, width, canvasW, canvasH, displayScale, zoom]);

  // Compute FFT and cache the rendered offscreen canvas (skip during playback)
  React.useEffect(() => {
    if (!showFft || !rawFrameDataRef.current || playing) return;

    const data = rawFrameDataRef.current;

    const computeFFT = async () => {
      let real: Float32Array, imag: Float32Array;

      if (gpuReady && gpuFFTRef.current) {
        // WebGPU path
        const result = await gpuFFTRef.current.fft2D(data.slice(), new Float32Array(data.length), width, height, false);
        real = result.real;
        imag = result.imag;
      } else {
        // CPU fallback
        real = data.slice();
        imag = new Float32Array(data.length);
        fft2d(real, imag, width, height, false);
      }

      fftshift(real, width, height);
      fftshift(imag, width, height);

      const mag = computeMagnitude(real, imag);

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

      const fftStatsResult = computeStats(displayData);

      setFftHistogramData(displayData);
      setFftDataRange({ min: displayMin, max: displayMax });
      setFftStats(fftStatsResult);

      // Apply histogram slider range on top of percentile clipping
      const { vmin, vmax } = sliderRange(displayMin, displayMax, fftVminPct, fftVmaxPct);

      const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
      const offscreen = renderToOffscreen(displayData, width, height, lut, vmin, vmax);
      if (!offscreen) return;

      // Cache the rendered FFT image
      fftOffscreenRef.current = offscreen;

      // Trigger initial draw
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
    };

    computeFFT();
  }, [showFft, frameBytes, playing, width, height, canvasW, canvasH, fftVminPct, fftVmaxPct, fftColormap, fftLogScale, fftAuto, gpuReady]);

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

  // Mouse handlers
  const handleWheel = (e: React.WheelEvent) => {
    const canvas = overlayRef.current;
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

  const handleCanvasMouseDown = (e: React.MouseEvent) => {
    if (roiActive) {
      setIsDraggingROI(true);
      updateROI(e);
    } else {
      setIsDraggingPan(true);
      setPanStart({ x: e.clientX, y: e.clientY, pX: panX, pY: panY });
    }
  };

  const handleCanvasMouseMove = (e: React.MouseEvent) => {
    // Cursor readout: convert screen position to image pixel coordinates
    const canvas = overlayRef.current;
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
      } else {
        setCursorInfo(null);
      }
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

  const handleCanvasMouseUp = () => {
    setIsDraggingROI(false);
    setIsDraggingPan(false);
    setPanStart(null);
  };

  const handleCanvasMouseLeave = () => {
    setCursorInfo(null);
    setIsDraggingROI(false);
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

  const handleFftMouseDown = (e: React.MouseEvent) => {
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

  const handleFftMouseUp = () => {
    setIsFftDragging(false);
    setFftPanStart(null);
  };

  const handleFftReset = () => {
    setFftZoom(1);
    setFftPanX(0);
    setFftPanY(0);
  };

  const fftNeedsReset = fftZoom !== 1 || fftPanX !== 0 || fftPanY !== 0;

  const updateROI = (e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const canvas = overlayRef.current;
    if (!canvas) return;
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
        setPlaying(!playing);
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
    }
  };

  // Check if view needs reset
  const needsReset = zoom !== 1 || panX !== 0 || panY !== 0;

  return (
    <Box className="show3d-root" tabIndex={0} onKeyDown={handleKeyDown} sx={{ ...container.root, bgcolor: themeColors.bg, color: themeColors.text }}>
      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        <Box>
          {/* Header row with title, FFT toggle, and Reset button - matching Show4DSTEM */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
            <Typography variant="caption" sx={{ ...typography.label }}>{title || "Image"}</Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
              <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT:</Typography>
              <Switch checked={showFft} onChange={(e) => setShowFft(e.target.checked)} size="small" sx={switchStyles.small} />
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={(e) => setExportAnchor(e.currentTarget)} disabled={exporting}>{exporting ? "Exporting..." : "Export"}</Button>
              <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                <MenuItem onClick={handleExportPng} sx={{ fontSize: 12 }}>PNG (current frame)</MenuItem>
                <MenuItem onClick={handleExportPngAll} sx={{ fontSize: 12 }}>PNG (all frames .zip)</MenuItem>
                <MenuItem onClick={handleExportGif} sx={{ fontSize: 12 }}>GIF (fps: {fps})</MenuItem>
              </Menu>
              <Button size="small" sx={compactButton} disabled={!needsReset} onClick={handleDoubleClick}>Reset</Button>
            </Stack>
          </Stack>
          <Box
            ref={canvasContainerRef}
            sx={{ ...container.imageBox, width: canvasW, height: canvasH, cursor: roiActive ? "crosshair" : "grab" }}
            onMouseDown={handleCanvasMouseDown}
            onMouseMove={handleCanvasMouseMove}
            onMouseUp={handleCanvasMouseUp}
            onMouseLeave={handleCanvasMouseLeave}
            onWheel={handleWheel}
            onDoubleClick={handleDoubleClick}
          >
            <canvas ref={canvasRef} width={canvasW} height={canvasH} style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }} />
            <canvas ref={overlayRef} width={canvasW} height={canvasH} style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }} />
            <canvas ref={uiRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} />
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
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>ROI <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(roiMean)}</Box></Typography>
              )}
              {cursorInfo && (
                <>
                  <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 14 }} />
                  <Typography sx={{ fontSize: 11, color: themeColors.textMuted, fontFamily: "monospace" }}>
                    ({cursorInfo.row}, {cursorInfo.col}) <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(cursorInfo.value)}</Box>
                  </Typography>
                </>
              )}
            </Box>
          )}
          {/* Image Controls - two rows with histogram on right (like Show4DSTEM) */}
          <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: canvasW, boxSizing: "border-box" }}>
            {/* Left: two rows of controls */}
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
              {/* Row 1: Scale + Auto + Color */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Scale:<InfoTooltip text="Linear or logarithmic intensity mapping. Log emphasizes low-intensity features." theme={themeInfo.theme} /></Typography>
                <Select value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="linear">Lin</MenuItem>
                  <MenuItem value="log">Log</MenuItem>
                </Select>
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Auto:<InfoTooltip text="Auto-contrast using percentile-based clipping. Ignores extreme outliers for better contrast." theme={themeInfo.theme} /></Typography>
                <Switch checked={autoContrast} onChange={(e) => setAutoContrast(e.target.checked)} size="small" sx={switchStyles.small} />
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Color:</Typography>
                <Select size="small" value={cmap} onChange={(e) => setCmap(e.target.value)} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }}>
                  {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                </Select>
              </Box>
              {/* Row 2: ROI dropdown + size controls + zoom indicator */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>ROI:<InfoTooltip text="Region of Interest. Click on the image to place and drag. Tracks mean intensity across frames." theme={themeInfo.theme} /></Typography>
                <Select
                  size="small"
                  value={roiActive ? (roiShape || "circle") : "none"}
                  onChange={(e) => {
                    const val = e.target.value as RoiShape;
                    if (val === "none") {
                      setRoiActive(false);
                    } else {
                      setRoiActive(true);
                      setRoiShape(val);
                      if (!roiActive) { setRoiCol(Math.floor(width / 2)); setRoiRow(Math.floor(height / 2)); }
                    }
                  }}
                  MenuProps={themedMenuProps}
                  sx={{ ...themedSelect, minWidth: 70 }}
                >
                  {ROI_SHAPES.map((shape) => (<MenuItem key={shape} value={shape}>{shape.charAt(0).toUpperCase() + shape.slice(1)}</MenuItem>))}
                </Select>
                {roiActive && roiShape === "rectangle" && (
                  <>
                    <Typography sx={{ ...typography.label, color: themeColors.textMuted }}>W</Typography>
                    <Slider value={roiWidth} min={5} max={width} onChange={(_, v) => setRoiWidth(v as number)} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                    <Typography sx={{ ...typography.label, color: themeColors.textMuted }}>H</Typography>
                    <Slider value={roiHeight} min={5} max={height} onChange={(_, v) => setRoiHeight(v as number)} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                  </>
                )}
                {roiActive && roiShape !== "rectangle" && (
                  <>
                    <Typography sx={{ ...typography.label, color: themeColors.textMuted }}>Size</Typography>
                    <Slider value={roiRadius} min={5} max={Math.max(width, height)} onChange={(_, v) => setRoiRadius(v as number)} size="small" sx={{ ...sliderStyles.small, width: 50 }} />
                  </>
                )}
                {zoom !== 1 && (
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.accent, fontWeight: "bold" }}>{zoom.toFixed(1)}x</Typography>
                )}
              </Box>
            </Box>
            {/* Right: Histogram spanning both rows */}
            <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
              <Histogram
                data={imageHistogramData}
                vminPct={imageVminPct}
                vmaxPct={imageVmaxPct}
                onRangeChange={(min, max) => { setImageVminPct(min); setImageVmaxPct(max); }}
                width={110}
                height={58}
                theme={themeInfo.theme}
                dataMin={imageDataRange.min}
                dataMax={imageDataRange.max}
              />
            </Box>
          </Box>
        </Box>

        {/* FFT Panel - same size as main image, matching Show4DSTEM layout */}
        {showFft && (
          <Box sx={{ width: canvasW }}>
            {/* FFT Header with Reset button - matching Show4DSTEM */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
              <Typography variant="caption" sx={{ ...typography.label }}>FFT{gpuReady ? " (GPU)" : " (CPU)"}</Typography>
              <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
                <Button size="small" sx={compactButton} disabled={!fftNeedsReset} onClick={handleFftReset}>Reset</Button>
              </Stack>
            </Stack>
            {/* FFT Canvas - same size as main image */}
            <Box
              ref={fftContainerRef}
              sx={{ ...container.imageBox, width: canvasW, height: canvasH, cursor: "grab" }}
              onMouseDown={handleFftMouseDown}
              onMouseMove={handleFftMouseMove}
              onMouseUp={handleFftMouseUp}
              onMouseLeave={handleFftMouseUp}
              onWheel={handleFftWheel}
              onDoubleClick={handleFftReset}
            >
              <canvas ref={fftCanvasRef} width={canvasW} height={canvasH} style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }} />
            </Box>
            {/* FFT Statistics bar */}
            {showStats && (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2 }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.mean)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.min)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.max)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.std)}</Box></Typography>
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
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Auto:<InfoTooltip text="Auto-enhance FFT display. When ON: (1) Masks DC component at center - replaced with average of 4 neighbors. (2) Clips display to 99.9 percentile to exclude outliers. When OFF: shows raw FFT with full dynamic range." theme={themeInfo.theme} /></Typography>
                  <Switch checked={fftAuto} onChange={(e) => setFftAuto(e.target.checked)} size="small" sx={switchStyles.small} />
                </Box>
                {/* Row 2: Color */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Color:</Typography>
                  <Select value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={themedMenuProps}>
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
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
        {loop ? (
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
      {/* Row 2: FPS, Loop, Bounce, Bookmark */}
      <Box sx={{ ...controlRow, mt: `${SPACING.XS}px`, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, width: canvasW, maxWidth: canvasW, boxSizing: "border-box" }}>
        <InfoTooltip text={<KeyboardShortcuts items={[["Space", "Play / Pause"], ["← / →", "Prev / Next frame"], ["Home / End", "First / Last frame"], ["R", "Reset zoom"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]]} />} theme={themeInfo.theme} />
        <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>fps</Typography>
        <Slider value={fps} min={1} max={60} step={1} onChange={(_, v) => setFps(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35, flexShrink: 0 }} />
        <Typography sx={{ ...typography.label, color: themeColors.textMuted, minWidth: 14, flexShrink: 0 }}>{Math.round(fps)}</Typography>
        <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>Loop<InfoTooltip text="Loop playback. Drag the small end markers on the slider to set loop range." theme={themeInfo.theme} /></Typography>
        <Switch size="small" checked={loop} onChange={() => setLoop(!loop)} sx={{ ...switchStyles.small, flexShrink: 0 }} />
        <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>Bounce<InfoTooltip text="Ping-pong playback. Alternates forward and reverse at loop boundaries." theme={themeInfo.theme} /></Typography>
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
      </Box>
      </>); })()}


    </Box>
  );
}

export const render = createRender(Show3D);
