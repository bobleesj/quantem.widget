/**
 * ShowComplex - Interactive complex-valued image viewer.
 *
 * Features:
 * - 5 display modes: amplitude, phase, HSV, real, imaginary
 * - Phase colorwheel inset for phase/HSV modes
 * - Scroll to zoom, double-click to reset
 * - WebGPU-accelerated FFT
 * - Scale bar, histogram, statistics
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Tooltip from "@mui/material/Tooltip";
import { useTheme } from "../theme";
import { drawScaleBarHiDPI, drawColorbar, exportFigure } from "../scalebar";
import { extractFloat32, formatNumber, downloadBlob } from "../format";
import { computeHistogramFromBytes } from "../histogram";
import { findDataRange, applyLogScale, percentileClip, sliderRange } from "../stats";
import { getWebGPUFFT, WebGPUFFT, fft2d, fftshift, computeMagnitude, autoEnhanceFFT, nextPow2 } from "../webgpu-fft";
import { COLORMAPS, COLORMAP_NAMES, applyColormap, renderToOffscreen } from "../colormaps";
import { ControlCustomizer } from "../control-customizer";
import { computeToolVisibility } from "../tool-parity";
import "./showcomplex.css";

// ============================================================================
// Helper components (per-widget, not shared)
// ============================================================================

function InfoTooltip({ text, theme = "dark" }: { text: React.ReactNode; theme?: "light" | "dark" }) {
  const isDark = theme === "dark";
  const content = typeof text === "string"
    ? <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>{text}</Typography>
    : text;
  return (
    <Tooltip
      title={content}
      arrow placement="bottom"
      componentsProps={{
        tooltip: { sx: { bgcolor: isDark ? "#333" : "#fff", color: isDark ? "#ddd" : "#333", border: `1px solid ${isDark ? "#555" : "#ccc"}`, maxWidth: 280, p: 1 } },
        arrow: { sx: { color: isDark ? "#333" : "#fff", "&::before": { border: `1px solid ${isDark ? "#555" : "#ccc"}` } } },
      }}
    >
      <Typography component="span" sx={{ fontSize: 12, color: isDark ? "#888" : "#666", cursor: "help", ml: 0.5, "&:hover": { color: isDark ? "#aaa" : "#444" } }}>ⓘ</Typography>
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
// Style constants
// ============================================================================

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const DPR = window.devicePixelRatio || 1;
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
};
const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
};
const switchStyles = {
  small: { "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-switchBase": { padding: "4px" } },
};
const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  "&.Mui-disabled": { color: "#666", borderColor: "#444" },
};
const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

const DEFAULT_CANVAS_SIZE = 500;

type DisplayMode = "amplitude" | "phase" | "hsv" | "real" | "imag";

// ============================================================================
// HSV rendering
// ============================================================================

function renderHSV(
  real: Float32Array, imag: Float32Array,
  rgba: Uint8ClampedArray,
  ampMin: number, ampMax: number,
): void {
  const ampRange = ampMax > ampMin ? ampMax - ampMin : 1;
  for (let i = 0; i < real.length; i++) {
    const phase = Math.atan2(imag[i], real[i]);
    const amp = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
    const h = (phase + Math.PI) / (2 * Math.PI); // [0, 1]
    const v = Math.max(0, Math.min(1, (amp - ampMin) / ampRange));
    // HSV→RGB with S=1
    const hi = Math.floor(h * 6) % 6;
    const f = h * 6 - Math.floor(h * 6);
    const q = v * (1 - f);
    const t = v * f;
    let r: number, g: number, b: number;
    switch (hi) {
      case 0: r = v; g = t; b = 0; break;
      case 1: r = q; g = v; b = 0; break;
      case 2: r = 0; g = v; b = t; break;
      case 3: r = 0; g = q; b = v; break;
      case 4: r = t; g = 0; b = v; break;
      default: r = v; g = 0; b = q; break;
    }
    const j = i * 4;
    rgba[j] = r * 255;
    rgba[j + 1] = g * 255;
    rgba[j + 2] = b * 255;
    rgba[j + 3] = 255;
  }
}

// ============================================================================
// Phase colorwheel drawing
// ============================================================================

function drawPhaseColorwheel(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number,
  radius: number,
): void {
  // Draw filled circle with hue sweep
  for (let angle = 0; angle < 360; angle += 1) {
    const rad = (angle * Math.PI) / 180;
    const rad2 = ((angle + 2) * Math.PI) / 180;
    // phase = angle mapped to [-pi, pi] → hue
    const h = angle / 360;
    const hi = Math.floor(h * 6) % 6;
    const f = h * 6 - Math.floor(h * 6);
    const q = 1 - f;
    let r: number, g: number, b: number;
    switch (hi) {
      case 0: r = 1; g = f; b = 0; break;
      case 1: r = q; g = 1; b = 0; break;
      case 2: r = 0; g = 1; b = f; break;
      case 3: r = 0; g = q; b = 1; break;
      case 4: r = f; g = 0; b = 1; break;
      default: r = 1; g = 0; b = q; break;
    }
    ctx.fillStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, radius, rad, rad2);
    ctx.closePath();
    ctx.fill();
  }

  // White center gradient for "brightness = amplitude"
  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
  grad.addColorStop(0, "rgba(255,255,255,0.8)");
  grad.addColorStop(0.5, "rgba(255,255,255,0.2)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fill();

  // Border
  ctx.strokeStyle = "rgba(255,255,255,0.6)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();

  // Labels
  ctx.fillStyle = "white";
  ctx.shadowColor = "rgba(0,0,0,0.7)";
  ctx.shadowBlur = 2;
  ctx.font = "10px -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("0", cx + radius + 10, cy);
  ctx.fillText("π", cx - radius - 8, cy);
  ctx.fillText("π/2", cx, cy - radius - 8);
  ctx.fillText("-π/2", cx, cy + radius + 8);
  ctx.shadowBlur = 0;
}

// ============================================================================
// Histogram component
// ============================================================================

interface HistogramProps {
  data: Float32Array | null;
  colormap: string;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
  dataMin?: number;
  dataMax?: number;
}

function Histogram({ data, colormap: _colormap, vminPct, vmaxPct, onRangeChange, width = 110, height = 40, theme = "dark", dataMin = 0, dataMax = 1 }: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const bins = React.useMemo(() => computeHistogramFromBytes(data), [data]);
  const isDark = theme === "dark";
  const colors = isDark ? { bg: "#1a1a1a", barActive: "#888", barInactive: "#444", border: "#333" } : { bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc" };

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, width, height);
    const displayBins = 64;
    const binRatio = Math.floor(bins.length / displayBins);
    const reducedBins: number[] = [];
    for (let i = 0; i < displayBins; i++) {
      let sum = 0;
      for (let j = 0; j < binRatio; j++) sum += bins[i * binRatio + j] || 0;
      reducedBins.push(sum / binRatio);
    }
    const maxVal = Math.max(...reducedBins, 0.001);
    const barWidth = width / displayBins;
    const vminBin = Math.floor((vminPct / 100) * displayBins);
    const vmaxBin = Math.floor((vmaxPct / 100) * displayBins);
    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      ctx.fillStyle = (i >= vminBin && i <= vmaxBin) ? colors.barActive : colors.barInactive;
      ctx.fillRect(i * barWidth + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
    }
  }, [bins, vminPct, vmaxPct, width, height, colors]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas ref={canvasRef} style={{ width, height, border: `1px solid ${colors.border}` }} />
      <Slider
        value={[vminPct, vmaxPct]}
        onChange={(_, v) => { const [lo, hi] = v as number[]; onRangeChange(Math.min(lo, hi - 1), Math.max(hi, lo + 1)); }}
        min={0} max={100} size="small" valueLabelDisplay="auto"
        valueLabelFormat={(pct) => { const val = dataMin + (pct / 100) * (dataMax - dataMin); return val >= 1000 ? val.toExponential(1) : val.toFixed(1); }}
        sx={{ width, py: 0, "& .MuiSlider-thumb": { width: 8, height: 8 }, "& .MuiSlider-rail": { height: 2 }, "& .MuiSlider-track": { height: 2 }, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" } }}
      />
    </Box>
  );
}

// ============================================================================
// Main Component
// ============================================================================

function ShowComplex2D() {
  // Theme
  const { themeInfo, colors: tc } = useTheme();
  const themeColors = tc;

  const themedSelect = {
    fontSize: 10,
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

  // Model state
  const [width] = useModelState<number>("width");
  const [height] = useModelState<number>("height");
  const [realBytes] = useModelState<DataView>("real_bytes");
  const [imagBytes] = useModelState<DataView>("imag_bytes");
  const [title] = useModelState<string>("title");
  const [displayMode, setDisplayMode] = useModelState<string>("display_mode");
  const [cmap, setCmap] = useModelState<string>("cmap");

  // Display options
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [percentileLow] = useModelState<number>("percentile_low");
  const [percentileHigh] = useModelState<number>("percentile_high");

  // Scale bar
  const [pixelSize] = useModelState<number>("pixel_size");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");

  // UI
  const [showStats] = useModelState<boolean>("show_stats");
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [showControls] = useModelState<boolean>("show_controls");
  const [imageWidthPx] = useModelState<number>("image_width_px");

  // Stats
  const [statsMean] = useModelState<number>("stats_mean");
  const [statsMin] = useModelState<number>("stats_min");
  const [statsMax] = useModelState<number>("stats_max");
  const [statsStd] = useModelState<number>("stats_std");

  // Tool visibility
  const [disabledTools, setDisabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools, setHiddenTools] = useModelState<string[]>("hidden_tools");

  const toolVisibility = React.useMemo(
    () => computeToolVisibility("ShowComplex2D", disabledTools, hiddenTools),
    [disabledTools, hiddenTools],
  );

  const hideDisplay = toolVisibility.isHidden("display");
  const hideHistogram = toolVisibility.isHidden("histogram");
  const hideFft = toolVisibility.isHidden("fft");
  const hideStats = toolVisibility.isHidden("stats");
  const hideExport = toolVisibility.isHidden("export");
  const hideView = toolVisibility.isHidden("view");

  const lockDisplay = toolVisibility.isLocked("display");
  const lockHistogram = toolVisibility.isLocked("histogram");
  const lockFft = toolVisibility.isLocked("fft");
  const lockStats = toolVisibility.isLocked("stats");
  const lockExport = toolVisibility.isLocked("export");
  const lockView = toolVisibility.isLocked("view");

  const effectiveShowFft = showFft && !hideFft;

  // Canvas refs
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const uiCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const containerRef = React.useRef<HTMLDivElement>(null);

  // FFT refs
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const fftContainerRef = React.useRef<HTMLDivElement>(null);

  // Zoom/pan
  const [zoom, setZoom] = React.useState(1);
  const [panX, setPanX] = React.useState(0);
  const [panY, setPanY] = React.useState(0);
  const [isDragging, setIsDragging] = React.useState(false);
  const dragRef = React.useRef<{ startX: number; startY: number; startPanX: number; startPanY: number; wasDrag: boolean }>({
    startX: 0, startY: 0, startPanX: 0, startPanY: 0, wasDrag: false,
  });

  // Canvas sizing
  const [canvasW, setCanvasW] = React.useState(DEFAULT_CANVAS_SIZE);
  const [canvasH, setCanvasH] = React.useState(DEFAULT_CANVAS_SIZE);
  const [isResizing, setIsResizing] = React.useState(false);
  const resizeRef = React.useRef<{ startX: number; startY: number; startW: number; startH: number }>({ startX: 0, startY: 0, startW: 0, startH: 0 });

  // Histogram state
  const [histData, setHistData] = React.useState<Float32Array | null>(null);
  const [histRange, setHistRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [vminPct, setVminPct] = React.useState(0);
  const [vmaxPct, setVmaxPct] = React.useState(100);

  // FFT state
  const [gpuFFT, setGpuFFT] = React.useState<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);
  const [fftZoom, setFftZoom] = React.useState(3);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);
  const fftMagRef = React.useRef<Float32Array | null>(null);
  const [fftMagVersion, setFftMagVersion] = React.useState(0);
  const [fftHistData, setFftHistData] = React.useState<Float32Array | null>(null);
  const [fftHistRange, setFftHistRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [fftVminPct, setFftVminPct] = React.useState(0);
  const [fftVmaxPct, setFftVmaxPct] = React.useState(100);
  const [fftLogScale, setFftLogScale] = React.useState(true);
  const [fftAuto, setFftAuto] = React.useState(true);
  const [fftColormap] = React.useState("inferno");

  // Hover cursor
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: string } | null>(null);
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);

  // Data refs
  const realDataRef = React.useRef<Float32Array | null>(null);
  const imagDataRef = React.useRef<Float32Array | null>(null);
  const displayDataRef = React.useRef<Float32Array | null>(null);

  // ============================================================================
  // Init GPU FFT
  // ============================================================================
  React.useEffect(() => {
    let cancelled = false;
    getWebGPUFFT().then((fft) => {
      if (!cancelled && fft) { setGpuFFT(fft); setGpuReady(true); }
    }).catch(() => {});
    return () => { cancelled = true; };
  }, []);

  // ============================================================================
  // Extract real/imag data
  // ============================================================================
  React.useEffect(() => {
    const r = extractFloat32(realBytes);
    const im = extractFloat32(imagBytes);
    if (!r || !im || r.length === 0) return;
    realDataRef.current = r;
    imagDataRef.current = im;
  }, [realBytes, imagBytes]);

  // ============================================================================
  // Compute display data + histogram based on display mode
  // ============================================================================
  React.useEffect(() => {
    const r = realDataRef.current;
    const im = imagDataRef.current;
    if (!r || !im) return;

    let data: Float32Array;
    const mode = displayMode as DisplayMode;

    if (mode === "amplitude" || mode === "hsv") {
      data = new Float32Array(r.length);
      for (let i = 0; i < r.length; i++) {
        data[i] = Math.sqrt(r[i] * r[i] + im[i] * im[i]);
      }
    } else if (mode === "phase") {
      data = new Float32Array(r.length);
      for (let i = 0; i < r.length; i++) {
        data[i] = Math.atan2(im[i], r[i]);
      }
    } else if (mode === "real") {
      data = r;
    } else {
      data = im;
    }

    if (logScale && mode !== "phase") {
      data = applyLogScale(data);
    }

    displayDataRef.current = data;
    setHistData(data);
    setHistRange(findDataRange(data));
  }, [realBytes, imagBytes, displayMode, logScale]);

  // ============================================================================
  // Canvas sizing
  // ============================================================================
  React.useEffect(() => {
    if (!width || !height) return;
    const targetW = imageWidthPx > 0 ? imageWidthPx : DEFAULT_CANVAS_SIZE;
    const scale = targetW / width;
    setCanvasW(Math.round(width * scale));
    setCanvasH(Math.round(height * scale));
  }, [width, height, imageWidthPx]);

  // ============================================================================
  // Main canvas rendering
  // ============================================================================
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !width || !height) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const r = realDataRef.current;
    const im = imagDataRef.current;
    const dispData = displayDataRef.current;
    if (!r || !im) return;

    const mode = displayMode as DisplayMode;

    // Create offscreen canvas at native resolution
    const offscreen = document.createElement("canvas");
    offscreen.width = width;
    offscreen.height = height;
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;
    const imgData = offCtx.createImageData(width, height);

    if (mode === "hsv") {
      const range = histRange;
      renderHSV(r, im, imgData.data, range.min, range.max);
    } else {
      if (!dispData) return;
      const lut = COLORMAPS[mode === "phase" ? "hsv" : cmap] || COLORMAPS.inferno;
      let vmin: number, vmax: number;
      if (autoContrast && mode !== "phase") {
        const pc = percentileClip(dispData, percentileLow, percentileHigh);
        vmin = pc.vmin;
        vmax = pc.vmax;
      } else if (mode === "phase") {
        vmin = -Math.PI;
        vmax = Math.PI;
      } else {
        ({ vmin, vmax } = sliderRange(histRange.min, histRange.max, vminPct, vmaxPct));
      }
      applyColormap(dispData, imgData.data, lut, vmin, vmax);
    }

    offCtx.putImageData(imgData, 0, 0);

    // Draw with zoom/pan (same pattern as Show2D)
    ctx.imageSmoothingEnabled = zoom < 4;
    ctx.clearRect(0, 0, canvasW, canvasH);

    if (zoom !== 1 || panX !== 0 || panY !== 0) {
      ctx.save();
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      ctx.translate(cx + panX, cy + panY);
      ctx.scale(zoom, zoom);
      ctx.translate(-cx, -cy);
      ctx.drawImage(offscreen, 0, 0, width, height, 0, 0, canvasW, canvasH);
      ctx.restore();
    } else {
      ctx.drawImage(offscreen, 0, 0, width, height, 0, 0, canvasW, canvasH);
    }
  }, [realBytes, imagBytes, displayMode, cmap, logScale, autoContrast, percentileLow, percentileHigh,
      vminPct, vmaxPct, zoom, panX, panY, canvasW, canvasH, width, height, histRange]);

  // ============================================================================
  // UI overlay canvas (scale bar, colorwheel, colorbar)
  // ============================================================================
  React.useEffect(() => {
    const canvas = uiCanvasRef.current;
    if (!canvas || !width || !height) return;
    canvas.width = Math.round(canvasW * DPR);
    canvas.height = Math.round(canvasH * DPR);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.scale(DPR, DPR);

    const mode = displayMode as DisplayMode;

    // Scale bar
    if (scaleBarVisible && pixelSize > 0) {
      drawScaleBarHiDPI(canvas, DPR, zoom, pixelSize, "Å", width);
    }

    // Zoom indicator (when no scale bar)
    if (!scaleBarVisible || pixelSize <= 0) {
      ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
      ctx.shadowBlur = 2;
      ctx.fillStyle = "white";
      ctx.font = "16px -apple-system, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${zoom.toFixed(1)}×`, 12, canvasH - 12);
      ctx.shadowBlur = 0;
    }

    // Phase colorwheel for phase/HSV modes
    if (mode === "phase" || mode === "hsv") {
      const cwRadius = 25;
      const cwX = 12 + cwRadius;
      const cwY = 12 + cwRadius;
      drawPhaseColorwheel(ctx, cwX, cwY, cwRadius);
    }

    // Colorbar (for non-HSV modes)
    if (mode !== "hsv" && displayDataRef.current) {
      const dispData = displayDataRef.current;
      const lut = COLORMAPS[mode === "phase" ? "hsv" : cmap] || COLORMAPS.inferno;
      let vmin: number, vmax: number;
      if (autoContrast && mode !== "phase") {
        const pc = percentileClip(dispData, percentileLow, percentileHigh);
        vmin = pc.vmin;
        vmax = pc.vmax;
      } else if (mode === "phase") {
        vmin = -Math.PI;
        vmax = Math.PI;
      } else {
        ({ vmin, vmax } = sliderRange(histRange.min, histRange.max, vminPct, vmaxPct));
      }
      drawColorbar(ctx, canvasW, canvasH, lut, vmin, vmax, logScale && mode !== "phase");
    }

    ctx.restore();
  }, [displayMode, cmap, zoom, canvasW, canvasH, width, height, pixelSize, scaleBarVisible,
      logScale, autoContrast, vminPct, vmaxPct, histRange, percentileLow, percentileHigh]);

  // ============================================================================
  // FFT computation (async)
  // ============================================================================
  React.useEffect(() => {
    if (!effectiveShowFft || !displayDataRef.current || !width || !height) return;
    let cancelled = false;
    const data = displayDataRef.current;
    const pw = nextPow2(width);
    const ph = nextPow2(height);
    const n = pw * ph;
    const padR = new Float32Array(n);
    const padI = new Float32Array(n);
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < width; col++) {
        padR[row * pw + col] = data[row * width + col];
      }
    }

    const computeFFT = async () => {
      let fReal: Float32Array, fImag: Float32Array;
      if (gpuFFT) {
        const result = await gpuFFT.fft2D(padR, padI, pw, ph, false);
        if (cancelled) return;
        fReal = result.real;
        fImag = result.imag;
      } else {
        fft2d(padR, padI, pw, ph, false);
        if (cancelled) return;
        fReal = padR;
        fImag = padI;
      }
      fftshift(fReal, pw, ph);
      fftshift(fImag, pw, ph);
      const mag = computeMagnitude(fReal, fImag);
      if (cancelled) return;
      fftMagRef.current = mag;
      setFftMagVersion((v) => v + 1);
    };
    computeFFT();
    return () => { cancelled = true; };
  }, [effectiveShowFft, realBytes, imagBytes, displayMode, logScale, width, height, gpuReady]);

  // FFT rendering (cheap, sync)
  React.useEffect(() => {
    const mag = fftMagRef.current;
    if (!effectiveShowFft || !mag) return;
    const pw = nextPow2(width);
    const ph = nextPow2(height);

    let processedMag: Float32Array;
    let vmin: number, vmax: number;

    if (fftAuto) {
      const enhanced = autoEnhanceFFT(mag, pw, ph);
      processedMag = fftLogScale ? applyLogScale(mag) : mag;
      const enh2 = fftLogScale ? autoEnhanceFFT(processedMag, pw, ph) : enhanced;
      vmin = enh2.min;
      vmax = enh2.max;
    } else {
      processedMag = fftLogScale ? applyLogScale(mag) : mag;
      const range = findDataRange(processedMag);
      ({ vmin, vmax } = sliderRange(range.min, range.max, fftVminPct, fftVmaxPct));
    }

    setFftHistData(processedMag);
    setFftHistRange(findDataRange(processedMag));

    // Render FFT (same pattern as Show2D)
    const fftCanvas = fftCanvasRef.current;
    if (!fftCanvas) return;
    const ctx = fftCanvas.getContext("2d");
    if (!ctx) return;

    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
    const offscreen = renderToOffscreen(processedMag, pw, ph, lut, vmin, vmax);
    if (!offscreen) return;

    ctx.imageSmoothingEnabled = fftZoom < 4;
    ctx.clearRect(0, 0, canvasW, canvasH);
    const scaleX = canvasW / pw;
    const scaleY = canvasH / ph;
    ctx.save();
    ctx.translate(canvasW / 2 + fftPanX, canvasH / 2 + fftPanY);
    ctx.scale(fftZoom * scaleX, fftZoom * scaleY);
    ctx.translate(-pw / 2, -ph / 2);
    ctx.drawImage(offscreen, 0, 0);
    ctx.restore();
  }, [effectiveShowFft, fftMagVersion, fftLogScale, fftAuto, fftVminPct, fftVmaxPct, fftColormap,
      fftZoom, fftPanX, fftPanY, width, height, canvasW, canvasH]);

  // ============================================================================
  // Mouse handlers
  // ============================================================================
  const handleMouseDown = React.useCallback((e: React.MouseEvent) => {
    if (lockView) return;
    dragRef.current = { startX: e.clientX, startY: e.clientY, startPanX: panX, startPanY: panY, wasDrag: false };
    setIsDragging(true);
  }, [panX, panY, lockView]);

  const handleMouseMove = React.useCallback((e: React.MouseEvent) => {
    // Cursor info — same coordinate conversion as Show2D
    const canvas = canvasRef.current;
    if (canvas && width && height) {
      const rect = canvas.getBoundingClientRect();
      const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
      const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      const imageCanvasX = (mouseCanvasX - cx - panX) / zoom + cx;
      const imageCanvasY = (mouseCanvasY - cy - panY) / zoom + cy;
      const displayScale = canvasW / width;
      const col = Math.floor(imageCanvasX / displayScale);
      const row = Math.floor(imageCanvasY / displayScale);
      if (col >= 0 && col < width && row >= 0 && row < height) {
        const r = realDataRef.current;
        const im = imagDataRef.current;
        if (r && im) {
          const idx = row * width + col;
          const re = r[idx];
          const ims = im[idx];
          const amp = Math.sqrt(re * re + ims * ims);
          const phase = Math.atan2(ims, re);
          setCursorInfo({ row, col, value: `amp=${formatNumber(amp)} phase=${phase.toFixed(2)}` });
        }
      } else {
        setCursorInfo(null);
      }
    }

    if (isDragging) {
      const dx = e.clientX - dragRef.current.startX;
      const dy = e.clientY - dragRef.current.startY;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) dragRef.current.wasDrag = true;
      setPanX(dragRef.current.startPanX + dx);
      setPanY(dragRef.current.startPanY + dy);
    }
  }, [isDragging, canvasW, canvasH, width, height, zoom, panX, panY]);

  const handleMouseUp = React.useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleMouseLeave = React.useCallback(() => {
    setIsDragging(false);
    setCursorInfo(null);
  }, []);

  const handleWheel = React.useCallback((e: WheelEvent) => {
    e.preventDefault();
    if (lockView) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();

    // Mouse position in canvas pixel coordinates
    const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);

    // Canvas center
    const cx = canvasW / 2;
    const cy = canvasH / 2;

    // Mouse position in image-canvas space (undo zoom+pan transform)
    const mouseImageX = (mouseCanvasX - cx - panX) / zoom + cx;
    const mouseImageY = (mouseCanvasY - cy - panY) / zoom + cy;

    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * factor));

    // New pan to keep mouse on the same image point
    const newPanX = mouseCanvasX - (mouseImageX - cx) * newZoom - cx;
    const newPanY = mouseCanvasY - (mouseImageY - cy) * newZoom - cy;

    setZoom(newZoom);
    setPanX(newPanX);
    setPanY(newPanY);
  }, [zoom, panX, panY, canvasW, canvasH, lockView]);

  const handleDoubleClick = React.useCallback(() => {
    if (lockView) return;
    setZoom(1);
    setPanX(0);
    setPanY(0);
  }, [lockView]);

  // Prevent scroll on canvas
  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener("wheel", handleWheel, { passive: false });
    return () => el.removeEventListener("wheel", handleWheel);
  }, [handleWheel]);

  // ============================================================================
  // Keyboard shortcuts
  // ============================================================================
  const handleKeyDown = React.useCallback((e: React.KeyboardEvent) => {
    if (e.key === "r" || e.key === "R") {
      if (!lockView) { setZoom(1); setPanX(0); setPanY(0); }
    }
    if (e.key === "f" || e.key === "F") {
      if (!lockFft) setShowFft(!showFft);
    }
  }, [lockView, lockFft, showFft]);

  // ============================================================================
  // Resize handle
  // ============================================================================
  const handleResizeMouseDown = React.useCallback((e: React.MouseEvent) => {
    if (lockView) return;
    e.stopPropagation();
    e.preventDefault();
    resizeRef.current = { startX: e.clientX, startY: e.clientY, startW: canvasW, startH: canvasH };
    setIsResizing(true);
  }, [canvasW, canvasH, lockView]);

  React.useEffect(() => {
    if (!isResizing) return;
    const onMove = (e: MouseEvent) => {
      const dx = e.clientX - resizeRef.current.startX;
      const dy = e.clientY - resizeRef.current.startY;
      const delta = Math.max(dx, dy);
      const newW = Math.max(200, resizeRef.current.startW + delta);
      const aspect = height / width;
      setCanvasW(newW);
      setCanvasH(Math.round(newW * aspect));
    };
    const onUp = () => setIsResizing(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, [isResizing, width, height]);

  // ============================================================================
  // Export figure
  // ============================================================================
  const handleExportFigure = React.useCallback((withColorbar: boolean) => {
    setExportAnchor(null);
    if (lockExport) return;
    const r = realDataRef.current;
    const im = imagDataRef.current;
    const dispData = displayDataRef.current;
    if (!r || !im || !width || !height) return;

    const mode = displayMode as DisplayMode;

    // Create offscreen canvas at native resolution
    const offscreen = document.createElement("canvas");
    offscreen.width = width;
    offscreen.height = height;
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;
    const imgData = offCtx.createImageData(width, height);

    let vmin: number, vmax: number;
    let lut: Uint8Array | undefined;

    if (mode === "hsv") {
      // Custom HSV rendering
      const range = histRange;
      renderHSV(r, im, imgData.data, range.min, range.max);
      vmin = range.min;
      vmax = range.max;
    } else {
      if (!dispData) return;
      lut = COLORMAPS[mode === "phase" ? "hsv" : cmap] || COLORMAPS.inferno;
      if (autoContrast && mode !== "phase") {
        const pc = percentileClip(dispData, percentileLow, percentileHigh);
        vmin = pc.vmin;
        vmax = pc.vmax;
      } else if (mode === "phase") {
        vmin = -Math.PI;
        vmax = Math.PI;
      } else {
        ({ vmin, vmax } = sliderRange(histRange.min, histRange.max, vminPct, vmaxPct));
      }
      applyColormap(dispData, imgData.data, lut, vmin, vmax);
    }

    offCtx.putImageData(imgData, 0, 0);

    const figCanvas = exportFigure({
      imageCanvas: offscreen,
      title: title || undefined,
      lut: mode !== "hsv" ? lut : undefined,
      vmin,
      vmax,
      logScale: logScale && mode !== "phase",
      pixelSize: pixelSize > 0 ? pixelSize : undefined,
      showColorbar: withColorbar && mode !== "hsv",
      showScaleBar: pixelSize > 0,
    });

    figCanvas.toBlob((blob) => {
      if (blob) downloadBlob(blob, `showcomplex_${mode}.png`);
    }, "image/png");
  }, [displayMode, cmap, logScale, autoContrast, percentileLow, percentileHigh,
      vminPct, vmaxPct, histRange, width, height, title, pixelSize, lockExport]);

  const handleExport = React.useCallback(() => {
    setExportAnchor(null);
    if (lockExport) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (blob) downloadBlob(blob, `showcomplex2d_${(displayMode as string)}.png`);
    }, "image/png");
  }, [displayMode, lockExport]);

  const handleCopy = React.useCallback(async () => {
    if (lockExport) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, "image/png"));
      if (!blob) return;
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    } catch {
      canvas.toBlob((b) => { if (b) downloadBlob(b, `showcomplex2d_${(displayMode as string)}.png`); }, "image/png");
    }
  }, [displayMode, lockExport]);

  const needsReset = zoom !== 1 || panX !== 0 || panY !== 0;

  // ============================================================================
  // Render
  // ============================================================================
  const borderColor = themeColors.border;
  const mode = displayMode as DisplayMode;
  const isColormapEnabled = mode === "amplitude" || mode === "real" || mode === "imag";

  return (
    <Box className="showcomplex-root" tabIndex={0} onKeyDown={handleKeyDown}
      sx={{ p: 2, bgcolor: themeColors.bg, color: themeColors.text, overflow: "visible" }}>
      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        <Box>
        {/* Title */}
        <Typography variant="caption" sx={{ ...typography.label, color: themeColors.accent, mb: `${SPACING.XS}px`, display: "block" }}>
          {title || "ShowComplex2D"}
          <InfoTooltip theme={themeInfo.theme} text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
            <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT: Show power spectrum alongside image.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Mode: Display mode — amplitude, phase, HSV (phase→hue), real, or imaginary part.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto: Percentile-based contrast (clips outliers). Disabled for phase/HSV modes.</Typography>
            <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
            <KeyboardShortcuts items={[
              ["Scroll", "Zoom in/out"],
              ["Drag", "Pan image"],
              ["Dbl-click", "Reset view"],
              ["R", "Reset zoom & pan"],
            ]} />
          </Box>} />
          <ControlCustomizer
            widgetName="ShowComplex2D"
            hiddenTools={hiddenTools}
            setHiddenTools={setHiddenTools}
            disabledTools={disabledTools}
            setDisabledTools={setDisabledTools}
            themeColors={themeColors}
          />
        </Typography>

        {/* Controls row */}
        <Stack direction="row" alignItems="center" spacing={`${SPACING.SM}px`} sx={{ mb: `${SPACING.XS}px`, height: 28, maxWidth: canvasW }}>
          {!hideFft && (
            <>
              <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT:</Typography>
              <Switch checked={showFft} onChange={(e) => { if (!lockFft) setShowFft(e.target.checked); }} disabled={lockFft} size="small" sx={switchStyles.small} />
            </>
          )}
          <Box sx={{ flex: 1 }} />
          {!hideView && (
            <Button size="small" sx={compactButton} disabled={lockView || !needsReset} onClick={handleDoubleClick}>Reset</Button>
          )}
          {!hideExport && (
            <>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={lockExport} onClick={(e) => { if (!lockExport) setExportAnchor(e.currentTarget); }}>Export</Button>
              <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                <MenuItem disabled={lockExport} onClick={() => handleExportFigure(true)} sx={{ fontSize: 12 }}>Figure + colorbar</MenuItem>
                <MenuItem disabled={lockExport} onClick={() => handleExportFigure(false)} sx={{ fontSize: 12 }}>Figure</MenuItem>
                <MenuItem disabled={lockExport} onClick={handleExport} sx={{ fontSize: 12 }}>PNG</MenuItem>
              </Menu>
              <Button size="small" sx={compactButton} disabled={lockExport} onClick={handleCopy}>Copy</Button>
            </>
          )}
        </Stack>

        {/* Canvas */}
        <Box ref={containerRef}
          sx={{ position: "relative", bgcolor: "#000", border: `1px solid ${borderColor}`, cursor: isDragging ? "grabbing" : "grab" }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          onDoubleClick={handleDoubleClick}
        >
          <canvas ref={canvasRef} width={canvasW} height={canvasH} style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }} />
          <canvas ref={uiCanvasRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} />
          {cursorInfo && (
            <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
              <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                ({cursorInfo.row}, {cursorInfo.col}) {cursorInfo.value}
              </Typography>
            </Box>
          )}
          {!hideView && (
            <Box onMouseDown={handleResizeMouseDown} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: lockView ? "default" : "nwse-resize", opacity: lockView ? 0.3 : 0.6, pointerEvents: lockView ? "none" : "auto", background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, borderRadius: "0 0 4px 0", "&:hover": { opacity: lockView ? 0.3 : 1 } }} />
          )}
        </Box>

        {/* Stats bar */}
        {!hideStats && showStats && (
          <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", maxWidth: canvasW, boxSizing: "border-box", overflow: "hidden", whiteSpace: "nowrap", opacity: lockStats ? 0.5 : 1 }}>
            <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMean)}</Box></Typography>
            <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMin)}</Box></Typography>
            <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMax)}</Box></Typography>
            <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsStd)}</Box></Typography>
          </Box>
        )}

        {/* Controls: two rows left + histogram right (matches Show2D layout) */}
        {showControls && (
          <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, maxWidth: canvasW, boxSizing: "border-box" }}>
            <Box sx={{ display: "flex", gap: `${SPACING.SM}px` }}>
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: Scale + Color + Mode */}
                {!hideDisplay && (
                  <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                    <Select disabled={lockDisplay} value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45 }} MenuProps={themedMenuProps}>
                      <MenuItem value="linear">Lin</MenuItem>
                      <MenuItem value="log">Log</MenuItem>
                    </Select>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                    <Select disabled={lockDisplay || !isColormapEnabled} size="small" value={cmap} onChange={(e) => setCmap(e.target.value)} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 60, opacity: isColormapEnabled ? 1 : 0.4 }}>
                      {COLORMAP_NAMES.filter(n => n !== "hsv").map((name) => (
                        <MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>
                      ))}
                    </Select>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Mode:</Typography>
                    <Select disabled={lockDisplay} size="small" value={displayMode} onChange={(e) => setDisplayMode(e.target.value)} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 90 }}>
                      <MenuItem value="amplitude">Amplitude</MenuItem>
                      <MenuItem value="phase">Phase</MenuItem>
                      <MenuItem value="hsv">HSV</MenuItem>
                      <MenuItem value="real">Real</MenuItem>
                      <MenuItem value="imag">Imaginary</MenuItem>
                    </Select>
                  </Box>
                )}
                {/* Row 2: Auto + zoom indicator */}
                {!hideDisplay && (
                  <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto:</Typography>
                    <Switch checked={autoContrast} onChange={() => { if (!lockDisplay) setAutoContrast(!autoContrast); }} disabled={lockDisplay || mode === "phase" || mode === "hsv"} size="small" sx={switchStyles.small} />
                    {zoom !== 1 && (
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.accent, fontWeight: "bold" }}>{zoom.toFixed(1)}x</Typography>
                    )}
                  </Box>
                )}
              </Box>
              {!hideHistogram && (
                <Box sx={{ opacity: lockHistogram ? 0.5 : 1, pointerEvents: lockHistogram ? "none" : "auto" }}>
                  <Histogram
                    data={histData} colormap={mode === "phase" ? "hsv" : cmap}
                    vminPct={vminPct} vmaxPct={vmaxPct}
                    onRangeChange={(lo, hi) => { if (!lockHistogram) { setVminPct(lo); setVmaxPct(hi); } }}
                    width={110} height={58}
                    theme={themeInfo.theme}
                    dataMin={histRange.min} dataMax={histRange.max}
                  />
                </Box>
              )}
            </Box>
          </Box>
        )}
        </Box>

        {/* FFT Panel — side panel (same layout as Show2D) */}
        {effectiveShowFft && (
          <Box sx={{ width: canvasW }}>
            {/* Spacer — matches main panel title row height */}
            <Box sx={{ mb: `${SPACING.XS}px`, height: 16 }} />
            {/* Controls row — matches main panel controls row height */}
            <Stack direction="row" justifyContent="flex-end" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
              {!hideView && (
                <Button size="small" sx={compactButton} disabled={lockView || (fftZoom === 3 && fftPanX === 0 && fftPanY === 0)} onClick={() => { setFftZoom(3); setFftPanX(0); setFftPanY(0); }}>Reset</Button>
              )}
            </Stack>
            <Box ref={fftContainerRef} sx={{ position: "relative", border: `1px solid ${borderColor}`, bgcolor: "#000", cursor: lockView ? "default" : "grab", width: canvasW, height: canvasH }}>
              <canvas ref={fftCanvasRef} width={canvasW} height={canvasH} style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }} />
            </Box>
            {/* FFT controls + histogram */}
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px` }}>
              <Box sx={{ display: "flex", gap: `${SPACING.SM}px` }}>
                <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                  <Box sx={{ ...controlRow, border: `1px solid ${borderColor}`, bgcolor: themeColors.controlBg, opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                    <Select disabled={lockDisplay} value={fftLogScale ? "log" : "linear"} onChange={(e) => setFftLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45 }} MenuProps={themedMenuProps}>
                      <MenuItem value="linear">Lin</MenuItem>
                      <MenuItem value="log">Log</MenuItem>
                    </Select>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto:</Typography>
                    <Switch checked={fftAuto} onChange={(e) => { if (!lockDisplay) setFftAuto(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} />
                  </Box>
                </Box>
                {!hideHistogram && (
                  <Box sx={{ opacity: lockHistogram ? 0.5 : 1, pointerEvents: lockHistogram ? "none" : "auto" }}>
                    <Histogram
                      data={fftHistData} colormap={fftColormap}
                      vminPct={fftVminPct} vmaxPct={fftVmaxPct}
                      onRangeChange={(lo, hi) => { if (!lockHistogram) { setFftVminPct(lo); setFftVmaxPct(hi); setFftAuto(false); } }}
                      width={110} height={40}
                      theme={themeInfo.theme}
                      dataMin={fftHistRange.min} dataMax={fftHistRange.max}
                    />
                  </Box>
                )}
              </Box>
            </Box>
          </Box>
        )}
      </Stack>
    </Box>
  );
}

export const render = createRender(ShowComplex2D);
