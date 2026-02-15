/**
 * Show2D - Static 2D image viewer with gallery support.
 * 
 * Features:
 * - Single image or gallery mode with configurable columns
 * - Scroll to zoom, double-click to reset
 * - WebGPU-accelerated FFT with default 3x zoom
 * - Equal-sized FFT and histogram panels
 * - Click to select image in gallery mode
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Tooltip from "@mui/material/Tooltip";
import { useTheme } from "../theme";
import { drawScaleBarHiDPI, roundToNiceValue } from "../scalebar";
import { extractBytes, extractFloat32, formatNumber, downloadBlob } from "../format";
import { computeHistogramFromBytes } from "../histogram";
import { findDataRange, applyLogScale, percentileClip, sliderRange, computeStats } from "../stats";

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

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};
import { getWebGPUFFT, WebGPUFFT, fft2d, fftshift, computeMagnitude, autoEnhanceFFT, nextPow2 } from "../webgpu-fft";
import { COLORMAPS, COLORMAP_NAMES, applyColormap, renderToOffscreen } from "../colormaps";
import "./show2d.css";

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;

const DPR = window.devicePixelRatio || 1;

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
  const colors = theme === "dark" ? { bg: "#1a1a1a", barActive: "#888", barInactive: "#444", border: "#333" } : { bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc" };

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
        onChange={(_, v) => { const [newMin, newMax] = v as number[]; onRangeChange(Math.min(newMin, newMax - 1), Math.max(newMax, newMin + 1)); }}
        min={0} max={100} size="small" valueLabelDisplay="auto"
        valueLabelFormat={(pct) => { const val = dataMin + (pct / 100) * (dataMax - dataMin); return val >= 1000 ? val.toExponential(1) : val.toFixed(1); }}
        sx={{ width, py: 0, "& .MuiSlider-thumb": { width: 8, height: 8 }, "& .MuiSlider-rail": { height: 2 }, "& .MuiSlider-track": { height: 2 }, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" } }}
      />
    </Box>
  );
}

// ============================================================================
// Line profile sampling (bilinear interpolation along line)
// ============================================================================
function sampleLineProfile(data: Float32Array, w: number, h: number, row0: number, col0: number, row1: number, col1: number): Float32Array {
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

// ============================================================================
// FFT peak finder (snap to Bragg spot with sub-pixel centroid refinement)
// ============================================================================
function findFFTPeak(mag: Float32Array, width: number, height: number, col: number, row: number, radius: number): { row: number; col: number } {
  // Find brightest pixel in search window
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
  // Sub-pixel refinement via weighted centroid in 3×3 window
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

// ============================================================================
// Types
// ============================================================================
type ZoomState = { zoom: number; panX: number; panY: number };

// ============================================================================
// Constants
// ============================================================================
const SINGLE_IMAGE_TARGET = 500;
const GALLERY_IMAGE_TARGET = 300;
const DEFAULT_FFT_ZOOM = 3;
const DEFAULT_ZOOM_STATE: ZoomState = { zoom: 1, panX: 0, panY: 0 };

// ============================================================================
// Main Component
// ============================================================================
// Show4DSTEM-style UI constants
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
};
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };
const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
};
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
const switchStyles = {
  small: { "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-switchBase": { padding: "4px" } },
};

function Show2D() {
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
  const [nImages] = useModelState<number>("n_images");
  const [width] = useModelState<number>("width");
  const [height] = useModelState<number>("height");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [labels] = useModelState<string[]>("labels");
  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [ncols] = useModelState<number>("ncols");

  // Display options
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");

  // Customization
  const [imageWidthPx] = useModelState<number>("image_width_px");

  // Scale bar
  const [pixelSizeAngstrom] = useModelState<number>("pixel_size_angstrom");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");

  // UI visibility
  const [showControls] = useModelState<boolean>("show_controls");
  const [showStats] = useModelState<boolean>("show_stats");
  const [statsMean] = useModelState<number[]>("stats_mean");
  const [statsMin] = useModelState<number[]>("stats_min");
  const [statsMax] = useModelState<number[]>("stats_max");
  const [statsStd] = useModelState<number[]>("stats_std");

  // Analysis Panels (FFT + Histogram)
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");

  // Selection
  const [selectedIdx, setSelectedIdx] = useModelState<number>("selected_idx");

  // Canvas refs
  const canvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const overlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const imageContainerRefs = React.useRef<(HTMLDivElement | null)[]>([]);
  const fftContainerRefs = React.useRef<(HTMLDivElement | null)[]>([]);
  const singleFftContainerRef = React.useRef<HTMLDivElement>(null);
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const [canvasReady, setCanvasReady] = React.useState(0);  // Trigger re-render when refs attached

  // Zoom/Pan state - per-image when not linked, shared when linked
  const [zoomStates, setZoomStates] = React.useState<Map<number, ZoomState>>(new Map());
  const [linkedZoomState, setLinkedZoomState] = React.useState<ZoomState>(DEFAULT_ZOOM_STATE);
  const [linkedZoom, setLinkedZoom] = React.useState(false);  // Link zoom across gallery images
  const [isDraggingPan, setIsDraggingPan] = React.useState(false);
  const [panStart, setPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);

  // Helper to get zoom state for an image
  const getZoomState = React.useCallback((idx: number): ZoomState => {
    if (linkedZoom) return linkedZoomState;
    return zoomStates.get(idx) || DEFAULT_ZOOM_STATE;
  }, [linkedZoom, linkedZoomState, zoomStates]);

  // Helper to set zoom state for an image
  const setZoomState = React.useCallback((idx: number, state: ZoomState) => {
    if (linkedZoom) {
      setLinkedZoomState(state);
    } else {
      setZoomStates(prev => new Map(prev).set(idx, state));
    }
  }, [linkedZoom]);

  // FFT zoom/pan state (single mode)
  const [fftZoom, setFftZoom] = React.useState(DEFAULT_FFT_ZOOM);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);
  const [isDraggingFftPan, setIsDraggingFftPan] = React.useState(false);
  const [fftPanStart, setFftPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);

  // Histogram state for main image (single mode)
  const [imageVminPct, setImageVminPct] = React.useState(0);
  const [imageVmaxPct, setImageVmaxPct] = React.useState(100);
  const [imageHistogramData, setImageHistogramData] = React.useState<Float32Array | null>(null);
  const [imageDataRange, setImageDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });

  // FFT display state (single mode)
  const [fftVminPct, setFftVminPct] = React.useState(0);
  const [fftVmaxPct, setFftVmaxPct] = React.useState(100);
  const [fftHistogramData, setFftHistogramData] = React.useState<Float32Array | null>(null);
  const [fftDataRange, setFftDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [fftColormap, setFftColormap] = React.useState("inferno");
  const [fftScaleMode, setFftScaleMode] = React.useState<"linear" | "log" | "power">("log");
  const [fftAuto, setFftAuto] = React.useState(true);
  const [fftStats, setFftStats] = React.useState<number[] | null>(null);

  // Cursor readout state
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number } | null>(null);

  // Colorbar state (single image mode only)
  const [showColorbar, setShowColorbar] = React.useState(false);

  // FFT d-spacing measurement
  const [fftClickInfo, setFftClickInfo] = React.useState<{
    row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  } | null>(null);
  const fftClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const fftOverlayRef = React.useRef<HTMLCanvasElement>(null);

  // Line profile state
  const [profileActive, setProfileActive] = React.useState(false);
  const [profileLine, setProfileLine] = useModelState<{ row: number; col: number }[]>("profile_line");
  const [profileData, setProfileData] = React.useState<Float32Array | null>(null);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);

  // Sync profile points from model state
  const profilePoints = profileLine || [];
  const setProfilePoints = (pts: { row: number; col: number }[]) => setProfileLine(pts);

  // FFT zoom/pan state (gallery mode — per-image or linked)
  const [galleryFftStates, setGalleryFftStates] = React.useState<Map<number, ZoomState>>(new Map());
  const [linkedFftZoomState, setLinkedFftZoomState] = React.useState<ZoomState>({ zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 });
  const [fftPanningIdx, setFftPanningIdx] = React.useState<number | null>(null);
  const getGalleryFftState = React.useCallback((idx: number) => {
    if (linkedZoom) return linkedFftZoomState;
    return galleryFftStates.get(idx) || { zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 };
  }, [linkedZoom, linkedFftZoomState, galleryFftStates]);
  const setGalleryFftState = React.useCallback((idx: number, state: ZoomState) => {
    if (linkedZoom) {
      setLinkedFftZoomState(state);
    } else {
      setGalleryFftStates(prev => new Map(prev).set(idx, state));
    }
  }, [linkedZoom]);

  // Resizable state (gallery starts smaller)
  const [canvasSize, setCanvasSize] = React.useState(nImages > 1 ? GALLERY_IMAGE_TARGET : SINGLE_IMAGE_TARGET);

  // Sync initial sizes from traits
  React.useEffect(() => {
    if (imageWidthPx > 0) setCanvasSize(imageWidthPx);
  }, [imageWidthPx]);

  const [isResizingCanvas, setIsResizingCanvas] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number, y: number, size: number } | null>(null);

  // WebGPU FFT
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);
  const rawDataRef = React.useRef<Float32Array[] | null>(null);

  // Inline FFT refs for gallery mode
  const fftCanvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const fftOffscreensRef = React.useRef<(HTMLCanvasElement | null)[]>([]);

  // Cached FFT magnitude for single image mode (avoids recomputing on zoom/pan)
  const fftMagCacheRef = React.useRef<Float32Array | null>(null);
  const [fftMagVersion, setFftMagVersion] = React.useState(0);

  // Layout calculations
  const isGallery = nImages > 1;
  const displayScale = canvasSize / Math.max(width, height);
  const canvasW = Math.round(width * displayScale);
  const canvasH = Math.round(height * displayScale);
  const floatsPerImage = width * height;

  // Extract raw float32 bytes and parse into Float32Arrays
  const allFloats = React.useMemo(() => extractFloat32(frameBytes), [frameBytes]);

  // Initialize WebGPU FFT on mount
  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) {
        gpuFFTRef.current = fft;
        setGpuReady(true);
      }
    });
  }, []);

  const [dataReady, setDataReady] = React.useState(false);

  // Keep inline FFT ref arrays in sync with nImages
  React.useEffect(() => {
    fftCanvasRefs.current = fftCanvasRefs.current.slice(0, nImages);
    fftOffscreensRef.current = fftOffscreensRef.current.slice(0, nImages);
  }, [nImages]);

  // Parse frame data and store raw floats for FFT
  React.useEffect(() => {
    if (!allFloats || allFloats.length === 0) return;
    const dataArrays: Float32Array[] = [];
    for (let i = 0; i < nImages; i++) {
      const start = i * floatsPerImage;
      const imageData = allFloats.subarray(start, start + floatsPerImage);
      dataArrays.push(new Float32Array(imageData));
    }
    rawDataRef.current = dataArrays;
    setDataReady(true);

    // Compute histogram data for single image mode
    if (nImages === 1 && dataArrays[0]) {
      const d = dataArrays[0];
      setImageHistogramData(d);
      setImageDataRange(findDataRange(d));
    }
  }, [allFloats, nImages, floatsPerImage]);

  // Prevent page scroll when scrolling on canvases (must use native listener with passive: false)
  // In gallery mode, only block scroll on the selected image (or all if linkedZoom)
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const elements: (HTMLElement | null)[] = isGallery
      ? (linkedZoom
          ? [...imageContainerRefs.current, ...fftContainerRefs.current]
          : [imageContainerRefs.current[selectedIdx], fftContainerRefs.current[selectedIdx]])
      : [imageContainerRefs.current[0], singleFftContainerRef.current];
    elements.forEach(el => el?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => elements.forEach(el => el?.removeEventListener("wheel", preventDefault));
  }, [canvasReady, showFft, nImages, isGallery, selectedIdx, linkedZoom]);

  // -------------------------------------------------------------------------
  // Render Images (JS-side normalization for instant Log/Auto toggle)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!dataReady || !rawDataRef.current || rawDataRef.current.length === 0) return;

    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;

    for (let i = 0; i < nImages; i++) {
      const canvas = canvasRefs.current[i];
      if (!canvas) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;

      // Get raw float32 data for this image
      const rawData = rawDataRef.current[i];
      if (!rawData) continue;

      // Apply log scale if enabled
      const processed = logScale ? applyLogScale(rawData) : rawData;

      // Compute min/max for normalization
      let vmin: number, vmax: number;
      if (!isGallery && imageDataRange.min !== imageDataRange.max) {
        ({ vmin, vmax } = sliderRange(imageDataRange.min, imageDataRange.max, imageVminPct, imageVmaxPct));
      } else if (autoContrast) {
        ({ vmin, vmax } = percentileClip(processed, 2, 98));
      } else {
        const r = findDataRange(processed);
        vmin = r.min;
        vmax = r.max;
      }

      // Create offscreen canvas at native resolution
      const offscreen = renderToOffscreen(processed, width, height, lut, vmin, vmax);
      if (!offscreen) continue;

      // Draw to display canvas with proper scaling
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Get per-image zoom state (inline to avoid callback dependency issues)
      const zs = linkedZoom ? linkedZoomState : (zoomStates.get(i) || DEFAULT_ZOOM_STATE);
      const { zoom, panX, panY } = zs;

      // Apply zoom/pan: zoom from center, then apply pan offset
      if (zoom !== 1 || panX !== 0 || panY !== 0) {
        ctx.save();
        // Translate to center, scale, translate back, then apply pan
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
    }
  }, [dataReady, nImages, width, height, cmap, displayScale, isGallery, canvasW, canvasH, canvasReady, linkedZoom, linkedZoomState, zoomStates, logScale, autoContrast, imageVminPct, imageVmaxPct, imageDataRange]);

  // -------------------------------------------------------------------------
  // Render Overlays (scale bar, colorbar, zoom indicator)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    for (let i = 0; i < nImages; i++) {
      const overlay = overlayRefs.current[i];
      if (!overlay) continue;
      const ctx = overlay.getContext("2d");
      if (!ctx) continue;

      if (scaleBarVisible) {
        const zs = linkedZoom ? linkedZoomState : (zoomStates.get(i) || DEFAULT_ZOOM_STATE);
        const unit = pixelSizeAngstrom > 0 ? "Å" as const : "px" as const;
        const pxSize = pixelSizeAngstrom > 0 ? pixelSizeAngstrom : 1;
        drawScaleBarHiDPI(overlay, DPR, zs.zoom, pxSize, unit, width);
      } else {
        ctx.clearRect(0, 0, overlay.width, overlay.height);
      }

      // Colorbar (single image mode only)
      if (showColorbar && !isGallery && rawDataRef.current?.[0]) {
        const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
        const cssW = overlay.width / DPR;
        const cssH = overlay.height / DPR;
        ctx.save();
        ctx.scale(DPR, DPR);

        const barW = 12;
        const barH = Math.round(cssH * 0.6);
        const barX = cssW - barW - 12;
        const barY = Math.round((cssH - barH) / 2);

        // Determine vmin/vmax matching the image rendering exactly
        const processed = logScale ? applyLogScale(rawDataRef.current[0]) : rawDataRef.current[0];
        let vmin: number, vmax: number;
        if (imageDataRange.min !== imageDataRange.max) {
          ({ vmin, vmax } = sliderRange(imageDataRange.min, imageDataRange.max, imageVminPct, imageVmaxPct));
        } else if (autoContrast) {
          ({ vmin, vmax } = percentileClip(processed, 2, 98));
        } else {
          const r = findDataRange(processed);
          vmin = r.min;
          vmax = r.max;
        }

        // Draw gradient strip (bottom=vmin, top=vmax)
        for (let row = 0; row < barH; row++) {
          const t = 1 - row / (barH - 1); // 0 at bottom, 1 at top
          const lutIdx = Math.round(t * 255);
          const r = lut[lutIdx * 3];
          const g = lut[lutIdx * 3 + 1];
          const b = lut[lutIdx * 3 + 2];
          ctx.fillStyle = `rgb(${r},${g},${b})`;
          ctx.fillRect(barX, barY + row, barW, 1);
        }

        // Border
        ctx.strokeStyle = "rgba(255,255,255,0.5)";
        ctx.lineWidth = 1;
        ctx.strokeRect(barX, barY, barW, barH);

        // Labels with drop shadow
        ctx.shadowColor = "rgba(0, 0, 0, 0.7)";
        ctx.shadowBlur = 2;
        ctx.shadowOffsetX = 1;
        ctx.shadowOffsetY = 1;
        ctx.font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
        ctx.fillStyle = "white";
        ctx.textAlign = "right";
        ctx.textBaseline = "bottom";
        ctx.fillText(formatNumber(vmax), barX - 4, barY + 6);
        ctx.textBaseline = "top";
        ctx.fillText(formatNumber(vmin), barX - 4, barY + barH - 4);
        if (logScale) {
          ctx.textBaseline = "middle";
          ctx.fillText("log", barX - 4, barY + barH / 2);
        }

        ctx.restore();
      }

      // Line profile overlay
      if (profileActive && profilePoints.length > 0) {
        const zs = linkedZoom ? linkedZoomState : (zoomStates.get(i) || DEFAULT_ZOOM_STATE);
        const { zoom, panX, panY } = zs;
        ctx.save();
        ctx.scale(DPR, DPR);

        // Transform image coords to screen coords
        const cx = canvasW / 2;
        const cy = canvasH / 2;
        const toScreenX = (ix: number) => (ix * displayScale - cx) * zoom + cx + panX;
        const toScreenY = (iy: number) => (iy * displayScale - cy) * zoom + cy + panY;

        // Draw point A
        const ax = toScreenX(profilePoints[0].col);
        const ay = toScreenY(profilePoints[0].row);
        ctx.fillStyle = themeColors.accent;
        ctx.beginPath();
        ctx.arc(ax, ay, 4, 0, Math.PI * 2);
        ctx.fill();

        // Draw line and point B if complete
        if (profilePoints.length === 2) {
          const bx = toScreenX(profilePoints[1].col);
          const by = toScreenY(profilePoints[1].row);
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

        ctx.restore();
      }
    }
  }, [nImages, pixelSizeAngstrom, scaleBarVisible, selectedIdx, isGallery, canvasW, canvasH, width, displayScale, linkedZoom, linkedZoomState, zoomStates, dataReady, showColorbar, cmap, imageDataRange, imageVminPct, imageVmaxPct, autoContrast, logScale, profileActive, profilePoints, themeColors]);

  // -------------------------------------------------------------------------
  // Auto-compute profile when profile_line is set (e.g. from Python)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (profilePoints.length === 2 && rawDataRef.current) {
      const imgIdx = isGallery ? selectedIdx : 0;
      const raw = rawDataRef.current[imgIdx];
      if (raw) {
        const p0 = profilePoints[0], p1 = profilePoints[1];
        const sampled = sampleLineProfile(raw, width, height, p0.row, p0.col, p1.row, p1.col);
        setProfileData(sampled);
        if (!profileActive) setProfileActive(true);
      }
    }
  }, [profilePoints, dataReady]);

  // -------------------------------------------------------------------------
  // Render sparkline for line profile
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const canvas = profileCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvasW;
    const cssH = 76;
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

    const padTop = 6;
    const padBottom = 18;
    const plotH = cssH - padTop - padBottom;

    // Find data range
    let pMin = profileData[0], pMax = profileData[0];
    for (let i = 1; i < profileData.length; i++) {
      if (profileData[i] < pMin) pMin = profileData[i];
      if (profileData[i] > pMax) pMax = profileData[i];
    }
    const range = pMax - pMin || 1;

    // Draw profile line
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < profileData.length; i++) {
      const x = (i / (profileData.length - 1)) * cssW;
      const y = padTop + plotH - ((profileData[i] - pMin) / range) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Compute total distance for x-axis
    let totalDist = profileData.length - 1;
    let xUnit = "px";
    if (profilePoints.length === 2) {
      const dx = profilePoints[1].col - profilePoints[0].col;
      const dy = profilePoints[1].row - profilePoints[0].row;
      const distPx = Math.sqrt(dx * dx + dy * dy);
      if (pixelSizeAngstrom > 0) {
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
    const idealTicks = Math.max(2, Math.floor(cssW / 70));
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
      const x = frac * cssW;
      ctx.beginPath(); ctx.moveTo(x, tickY); ctx.lineTo(x, tickY + 3); ctx.stroke();
      ctx.textAlign = frac < 0.05 ? "left" : frac > 0.95 ? "right" : "center";
      const valStr = v % 1 === 0 ? v.toFixed(0) : v.toFixed(1);
      ctx.fillText(i === ticks.length - 1 ? `${valStr} ${xUnit}` : valStr, x, tickY + 4);
    }

    // Draw y-axis min/max labels
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(formatNumber(pMax), 2, 1);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatNumber(pMin), 2, padTop + plotH - 1);
  }, [profileData, canvasW, themeInfo.theme, themeColors.accent, profilePoints, pixelSizeAngstrom]);

  // -------------------------------------------------------------------------
  // Compute FFT magnitude (cached — only recomputes when data changes)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!showFft || isGallery || !rawDataRef.current) return;
    if (!rawDataRef.current[selectedIdx]) return;

    const computeFFT = async () => {
      const data = rawDataRef.current![selectedIdx];
      const real = data.slice();
      const imag = new Float32Array(data.length);

      let mag: Float32Array;
      if (gpuFFTRef.current && gpuReady) {
        const { real: fReal, imag: fImag } = await gpuFFTRef.current.fft2D(real, imag, width, height, false);
        fftshift(fReal, width, height);
        fftshift(fImag, width, height);
        mag = computeMagnitude(fReal, fImag);
      } else {
        fft2d(real, imag, width, height, false);
        fftshift(real, width, height);
        fftshift(imag, width, height);
        mag = computeMagnitude(real, imag);
      }
      fftMagCacheRef.current = mag;
      setFftMagVersion(v => v + 1);
    };

    computeFFT();
  }, [showFft, isGallery, selectedIdx, width, height, gpuReady, dataReady]);

  // Clear FFT measurement when image or FFT state changes
  React.useEffect(() => { setFftClickInfo(null); }, [selectedIdx, showFft]);

  // -------------------------------------------------------------------------
  // Render FFT display from cached magnitude (zoom/pan/colormap changes)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!showFft || isGallery || !fftCanvasRef.current || !fftMagCacheRef.current) return;

    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const fftMag = fftMagCacheRef.current;
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;

    // Apply scale mode
    const magnitude = new Float32Array(fftMag.length);
    for (let i = 0; i < fftMag.length; i++) {
      if (fftScaleMode === "log") {
        magnitude[i] = Math.log1p(fftMag[i]);
      } else if (fftScaleMode === "power") {
        magnitude[i] = Math.pow(fftMag[i], 0.5);
      } else {
        magnitude[i] = fftMag[i];
      }
    }

    let displayMin: number, displayMax: number;
    if (fftAuto) {
      ({ min: displayMin, max: displayMax } = autoEnhanceFFT(magnitude, width, height));
    } else {
      ({ min: displayMin, max: displayMax } = findDataRange(magnitude));
    }

    const { mean, std } = computeStats(magnitude);
    setFftStats([mean, displayMin, displayMax, std]);

    // Store histogram data
    setFftHistogramData(magnitude.slice());
    setFftDataRange({ min: displayMin, max: displayMax });

    // Apply histogram slider clipping
    const { vmin, vmax } = sliderRange(displayMin, displayMax, fftVminPct, fftVmaxPct);
    const offscreen = renderToOffscreen(magnitude, width, height, lut, vmin, vmax);
    if (!offscreen) return;

    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.save();

    const centerOffsetX = (canvasW - canvasW * fftZoom) / 2 + fftPanX;
    const centerOffsetY = (canvasH - canvasH * fftZoom) / 2 + fftPanY;

    ctx.translate(centerOffsetX, centerOffsetY);
    ctx.scale(fftZoom, fftZoom);
    ctx.drawImage(offscreen, 0, 0, width, height, 0, 0, canvasW, canvasH);
    ctx.restore();
  }, [showFft, isGallery, fftMagVersion, canvasW, canvasH, fftZoom, fftPanX, fftPanY, fftVminPct, fftVmaxPct, fftColormap, fftScaleMode, fftAuto, width, height]);

  // -------------------------------------------------------------------------
  // Render FFT overlay (d-spacing marker)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const overlay = fftOverlayRef.current;
    if (!overlay) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!fftClickInfo) return;
    ctx.save();
    ctx.scale(DPR, DPR);
    const centerOffsetX = (canvasW - canvasW * fftZoom) / 2 + fftPanX;
    const centerOffsetY = (canvasH - canvasH * fftZoom) / 2 + fftPanY;
    const screenX = centerOffsetX + fftZoom * (fftClickInfo.col / width * canvasW);
    const screenY = centerOffsetY + fftZoom * (fftClickInfo.row / height * canvasH);
    // Crosshair with gap
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
    // D-spacing label
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
  }, [fftClickInfo, canvasW, canvasH, fftZoom, fftPanX, fftPanY, width, height]);

  // -------------------------------------------------------------------------
  // Render inline FFTs for gallery mode
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!showFft || !isGallery || !rawDataRef.current) return;
    if (rawDataRef.current.length === 0) return;

    let cancelled = false;

    const computeAllFFTs = async () => {
      const lut = COLORMAPS.inferno;

      for (let idx = 0; idx < nImages; idx++) {
        if (cancelled) return;

        const data = rawDataRef.current![idx];
        if (!data) continue;

        const real = data.slice();
        const imag = new Float32Array(data.length);
        let fReal: Float32Array;
        let fImag: Float32Array;

        if (gpuFFTRef.current && gpuReady) {
          const result = await gpuFFTRef.current.fft2D(real, imag, width, height, false);
          fReal = result.real;
          fImag = result.imag;
        } else {
          fft2d(real, imag, width, height, false);
          fReal = real;
          fImag = imag;
        }

        if (cancelled) return;

        fftshift(fReal, width, height);
        fftshift(fImag, width, height);

        const mag = computeMagnitude(fReal, fImag);
        const logData = applyLogScale(mag);
        const { min, max } = findDataRange(logData);

        const offscreen = renderToOffscreen(logData, width, height, lut, min, max);
        if (!offscreen) continue;
        fftOffscreensRef.current[idx] = offscreen;

        // Draw to visible canvas with default 3x zoom centered
        const canvas = fftCanvasRefs.current[idx];
        if (!canvas) continue;
        const ctx = canvas.getContext("2d");
        if (!ctx) continue;

        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, canvasW, canvasH);
        ctx.save();
        const zoom = DEFAULT_FFT_ZOOM;
        const centerOffsetX = (canvasW - canvasW * zoom) / 2;
        const centerOffsetY = (canvasH - canvasH * zoom) / 2;
        ctx.translate(centerOffsetX, centerOffsetY);
        ctx.scale(zoom, zoom);
        ctx.drawImage(offscreen, 0, 0, width, height, 0, 0, canvasW, canvasH);
        ctx.restore();
      }
    };

    computeAllFFTs();

    return () => { cancelled = true; };
  }, [showFft, isGallery, nImages, width, height, gpuReady, canvasW, canvasH, dataReady]);

  // Re-render gallery FFTs when zoom/pan changes (without recomputing FFT)
  React.useEffect(() => {
    if (!showFft || !isGallery) return;
    for (let idx = 0; idx < nImages; idx++) {
      const offscreen = fftOffscreensRef.current[idx];
      const canvas = fftCanvasRefs.current[idx];
      if (!offscreen || !canvas) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;
      const { zoom, panX, panY } = getGalleryFftState(idx);
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvasW, canvasH);
      ctx.save();
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      ctx.translate(cx + panX, cy + panY);
      ctx.scale(zoom, zoom);
      ctx.translate(-cx, -cy);
      ctx.drawImage(offscreen, 0, 0, width, height, 0, 0, canvasW, canvasH);
      ctx.restore();
    }
  }, [showFft, isGallery, nImages, canvasW, canvasH, width, height, galleryFftStates, linkedZoom, linkedFftZoomState]);

  // -------------------------------------------------------------------------
  // Mouse Handlers for Zoom/Pan
  // -------------------------------------------------------------------------
  const handleWheel = (e: React.WheelEvent, idx: number) => {
    // In gallery mode, only allow zoom on the selected image (unless linked)
    if (isGallery && idx !== selectedIdx && !linkedZoom) return;
    e.preventDefault(); // Prevent page scroll when zooming

    const canvas = canvasRefs.current[idx];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    
    // Get current zoom state
    const zs = getZoomState(idx);
    
    // Mouse position relative to canvas (in canvas pixel coordinates)
    const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
    
    // Canvas center
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    
    // Mouse position relative to the current view (accounting for pan and zoom)
    // The transformation is: translate(cx + panX, cy + panY) -> scale(zoom) -> translate(-cx, -cy)
    // So a point on screen at (screenX, screenY) maps to image space as:
    // imageX = (screenX - cx - panX) / zoom + cx
    const mouseImageX = (mouseCanvasX - cx - zs.panX) / zs.zoom + cx;
    const mouseImageY = (mouseCanvasY - cy - zs.panY) / zs.zoom + cy;

    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zs.zoom * zoomFactor));
    
    // Calculate new pan to keep the mouse position fixed on the same image point
    // After zoom: screenX = (imageX - cx) * newZoom + cx + newPanX
    // We want screenX to stay at mouseCanvasX, so:
    // newPanX = mouseCanvasX - (imageX - cx) * newZoom - cx
    const newPanX = mouseCanvasX - (mouseImageX - cx) * newZoom - cx;
    const newPanY = mouseCanvasY - (mouseImageY - cy) * newZoom - cy;

    setZoomState(idx, { zoom: newZoom, panX: newPanX, panY: newPanY });
  };

  const handleDoubleClick = (idx: number) => {
    setZoomState(idx, DEFAULT_ZOOM_STATE);
  };

  // Reset all zoom states to default
  const handleResetAll = () => {
    setZoomStates(new Map());
    setLinkedZoomState(DEFAULT_ZOOM_STATE);
    setGalleryFftStates(new Map());
    setLinkedFftZoomState({ zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 });
    setFftZoom(DEFAULT_FFT_ZOOM);
    setFftPanX(0);
    setFftPanY(0);
    setProfileActive(false);
    setProfilePoints([]);
    setProfileData(null);
    setFftClickInfo(null);
  };

  // FFT zoom/pan handlers
  const handleFftWheel = (e: React.WheelEvent) => {
    e.preventDefault(); // Prevent page scroll when zooming FFT
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    setFftZoom(Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, fftZoom * zoomFactor)));
  };

  const handleFftDoubleClick = () => {
    setFftZoom(DEFAULT_FFT_ZOOM);
    setFftPanX(0);
    setFftPanY(0);
    setFftClickInfo(null);
  };

  const handleFftMouseDown = (e: React.MouseEvent) => {
    fftClickStartRef.current = { x: e.clientX, y: e.clientY };
    setIsDraggingFftPan(true);
    setFftPanStart({ x: e.clientX, y: e.clientY, pX: fftPanX, pY: fftPanY });
  };

  const handleFftMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingFftPan || !fftPanStart) return;
    const dx = e.clientX - fftPanStart.x;
    const dy = e.clientY - fftPanStart.y;
    setFftPanX(fftPanStart.pX + dx);
    setFftPanY(fftPanStart.pY + dy);
  };

  const handleFftMouseUp = (e: React.MouseEvent) => {
    // Click detection for d-spacing measurement
    if (fftClickStartRef.current) {
      const dx = e.clientX - fftClickStartRef.current.x;
      const dy = e.clientY - fftClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const canvas = fftCanvasRef.current;
        if (canvas) {
          const rect = canvas.getBoundingClientRect();
          const mouseX = e.clientX - rect.left;
          const mouseY = e.clientY - rect.top;
          const cOffX = (canvasW - canvasW * fftZoom) / 2 + fftPanX;
          const cOffY = (canvasH - canvasH * fftZoom) / 2 + fftPanY;
          let imgCol = ((mouseX - cOffX) / fftZoom) / canvasW * width;
          let imgRow = ((mouseY - cOffY) / fftZoom) / canvasH * height;
          if (imgCol >= 0 && imgCol < width && imgRow >= 0 && imgRow < height) {
            // Snap to nearest Bragg spot (local max in FFT magnitude)
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
              if (pixelSizeAngstrom > 0) {
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
      }
      fftClickStartRef.current = null;
    }
    setIsDraggingFftPan(false);
    setFftPanStart(null);
  };

  const handleFftMouseLeave = () => {
    fftClickStartRef.current = null;
    setIsDraggingFftPan(false);
    setFftPanStart(null);
  };

  // Gallery FFT zoom/pan handlers (only selected image's FFT responds)
  const handleGalleryFftWheel = (e: React.WheelEvent, idx: number) => {
    if (isGallery && idx !== selectedIdx && !linkedZoom) return;
    e.preventDefault(); // Prevent page scroll when zooming FFT
    const zs = getGalleryFftState(idx);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    setGalleryFftState(idx, { ...zs, zoom: Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zs.zoom * zoomFactor)) });
  };

  const handleGalleryFftMouseDown = (e: React.MouseEvent, idx: number) => {
    if (isGallery && idx !== selectedIdx) {
      setSelectedIdx(idx);
      return; // Select first, don't start panning
    }
    const zs = getGalleryFftState(idx);
    setFftPanningIdx(idx);
    setIsDraggingFftPan(true);
    setFftPanStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
  };

  const handleGalleryFftMouseMove = (e: React.MouseEvent, idx: number) => {
    if (!isDraggingFftPan || !fftPanStart || fftPanningIdx !== idx) return;
    const dx = e.clientX - fftPanStart.x;
    const dy = e.clientY - fftPanStart.y;
    const zs = getGalleryFftState(idx);
    setGalleryFftState(idx, { ...zs, panX: fftPanStart.pX + dx, panY: fftPanStart.pY + dy });
  };

  const handleGalleryFftMouseUp = () => {
    setIsDraggingFftPan(false);
    setFftPanStart(null);
    setFftPanningIdx(null);
  };

  // Track which image is being panned
  const [panningIdx, setPanningIdx] = React.useState<number | null>(null);
  const clickStartRef = React.useRef<{ x: number; y: number } | null>(null);

  const handleMouseDown = (e: React.MouseEvent, idx: number) => {
    const zs = getZoomState(idx);
    if (isGallery && idx !== selectedIdx) {
      setSelectedIdx(idx);
      return; // Only select, don't start panning
    }
    clickStartRef.current = { x: e.clientX, y: e.clientY };
    setIsDraggingPan(true);
    setPanningIdx(idx);
    setPanStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
  };

  const handleMouseMove = (e: React.MouseEvent, idx: number) => {
    // Cursor readout: convert screen position to image pixel coordinates
    const canvas = canvasRefs.current[idx];
    if (canvas && rawDataRef.current) {
      const rect = canvas.getBoundingClientRect();
      const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
      const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
      const zs = linkedZoom ? linkedZoomState : (zoomStates.get(idx) || DEFAULT_ZOOM_STATE);
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      const imageCanvasX = (mouseCanvasX - cx - zs.panX) / zs.zoom + cx;
      const imageCanvasY = (mouseCanvasY - cy - zs.panY) / zs.zoom + cy;
      const imgX = Math.floor(imageCanvasX / displayScale);
      const imgY = Math.floor(imageCanvasY / displayScale);
      if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
        const rawData = rawDataRef.current[idx];
        if (rawData) setCursorInfo({ row: imgY, col: imgX, value: rawData[imgY * width + imgX] });
      } else {
        setCursorInfo(null);
      }
    }

    // Panning
    if (!isDraggingPan || !panStart || panningIdx === null) return;
    if (idx !== panningIdx) return;
    if (!canvas) return;
    const rect2 = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect2.width;
    const scaleY = canvas.height / rect2.height;
    const dx = (e.clientX - panStart.x) * scaleX;
    const dy = (e.clientY - panStart.y) * scaleY;

    const zs = getZoomState(idx);
    setZoomState(idx, { ...zs, panX: panStart.pX + dx, panY: panStart.pY + dy });
  };

  const handleMouseUp = (e: React.MouseEvent, idx: number) => {
    // Detect click (vs drag) for profile mode
    if (profileActive && clickStartRef.current) {
      const dx = e.clientX - clickStartRef.current.x;
      const dy = e.clientY - clickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        // It's a click — compute image coordinates
        const canvas = canvasRefs.current[idx];
        if (canvas && rawDataRef.current) {
          const rect = canvas.getBoundingClientRect();
          const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
          const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
          const zs = linkedZoom ? linkedZoomState : (zoomStates.get(idx) || DEFAULT_ZOOM_STATE);
          const cx = canvasW / 2;
          const cy = canvasH / 2;
          const imgX = ((mouseCanvasX - cx - zs.panX) / zs.zoom + cx) / displayScale;
          const imgY = ((mouseCanvasY - cy - zs.panY) / zs.zoom + cy) / displayScale;
          if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
            const pt = { row: imgY, col: imgX };
            if (profilePoints.length === 0 || profilePoints.length === 2) {
              // Start new line
              setProfilePoints([pt]);
              setProfileData(null);
            } else {
              // Complete the line
              const p0 = profilePoints[0];
              setProfilePoints([p0, pt]);
              const imgIdx = isGallery ? selectedIdx : 0;
              const raw = rawDataRef.current[imgIdx];
              if (raw) {
                setProfileData(sampleLineProfile(raw, width, height, p0.row, p0.col, pt.row, pt.col));
              }
            }
          }
        }
      }
    }
    clickStartRef.current = null;
    setIsDraggingPan(false);
    setPanStart(null);
    setPanningIdx(null);
  };

  const handleMouseLeave = (idx: number) => {
    setCursorInfo(null);
    if (panningIdx === idx) {
      setIsDraggingPan(false);
      setPanStart(null);
      setPanningIdx(null);
    }
  };

  // -------------------------------------------------------------------------
  // Export handler
  const handleExport = React.useCallback(() => {
    const canvas = canvasRefs.current[isGallery ? selectedIdx : 0];
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (!blob) return;
      downloadBlob(blob, `show2d_${labels?.[selectedIdx] || "image"}.png`);
    }, "image/png");
  }, [isGallery, selectedIdx, labels]);

  // Resize Handlers
  // -------------------------------------------------------------------------
  const handleCanvasResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizingCanvas(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: canvasSize });
  };

  React.useEffect(() => {
    if (!isResizingCanvas) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeStart) return;
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);
      const newSize = Math.max(100, Math.min(600, resizeStart.size + delta));
      setCanvasSize(newSize);
    };

    const handleMouseUp = () => {
      setIsResizingCanvas(false);
      setResizeStart(null);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingCanvas, resizeStart]);

  // -------------------------------------------------------------------------
  // Keyboard shortcuts
  // -------------------------------------------------------------------------
  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Number keys 1-9 select gallery images (avoids arrow key conflicts with Jupyter)
    if (isGallery && e.key >= "1" && e.key <= "9") {
      const idx = parseInt(e.key) - 1;
      if (idx < nImages) { e.preventDefault(); setSelectedIdx(idx); }
      return;
    }
    switch (e.key) {
      case "ArrowLeft":
        if (isGallery) { e.preventDefault(); setSelectedIdx(Math.max(0, selectedIdx - 1)); }
        break;
      case "ArrowRight":
        if (isGallery) { e.preventDefault(); setSelectedIdx(Math.min(nImages - 1, selectedIdx + 1)); }
        break;
      case "r":
      case "R":
        handleResetAll();
        break;
    }
  };

  // -------------------------------------------------------------------------
  // Render (Show3D-style layout)
  // -------------------------------------------------------------------------
  const needsReset = getZoomState(isGallery ? selectedIdx : 0).zoom !== 1 || getZoomState(isGallery ? selectedIdx : 0).panX !== 0 || getZoomState(isGallery ? selectedIdx : 0).panY !== 0;
  const statsIdx = isGallery ? selectedIdx : 0;

  // Calibrated cursor position
  const calibratedUnit = pixelSizeAngstrom > 0 ? (Math.max(height, width) * pixelSizeAngstrom >= 10 ? "nm" : "Å") : "";
  const calibratedFactor = calibratedUnit === "nm" ? pixelSizeAngstrom / 10 : pixelSizeAngstrom;

  return (
    <Box className="show2d-root" tabIndex={0} onKeyDown={handleKeyDown} sx={{ p: 2, bgcolor: themeColors.bg, color: themeColors.text, overflow: "visible" }}>
      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        {/* Main panel */}
        <Box>
          {/* Header row: Title | FFT, Export, Reset */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28, maxWidth: isGallery ? ncols * canvasW + (ncols - 1) * 8 : canvasW }}>
            <Typography variant="caption" sx={{ ...typography.label, color: themeColors.text }}>
              {title || (isGallery ? "Gallery" : "Image")}
              <InfoTooltip text={<KeyboardShortcuts items={isGallery ? [["← / →", "Prev / Next image"], ["1 – 9", "Select image"], ["R", "Reset zoom"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]] : [["R", "Reset zoom"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]]} />} theme={themeInfo.theme} />
            </Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
              <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT:</Typography>
              <Switch checked={showFft} onChange={(e) => setShowFft(e.target.checked)} size="small" sx={switchStyles.small} />
              {!isGallery && (
                <>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Profile:</Typography>
                  <Switch checked={profileActive} onChange={(e) => { setProfileActive(e.target.checked); if (!e.target.checked) { setProfilePoints([]); setProfileData(null); } }} size="small" sx={switchStyles.small} />
                </>
              )}
              <Button size="small" sx={compactButton} disabled={!needsReset} onClick={handleResetAll}>Reset</Button>
              <Button size="small" sx={compactButton} onClick={handleExport}>Export</Button>
            </Stack>
          </Stack>

          {isGallery ? (
            /* Gallery mode */
            <Box sx={{ display: "grid", gridTemplateColumns: `repeat(${ncols}, auto)`, gap: 1 }}>
              {Array.from({ length: nImages }).map((_, i) => (
                <Box key={i} sx={{ cursor: i === selectedIdx ? "grab" : "pointer" }}>
                  <Box
                    ref={(el: HTMLDivElement | null) => { imageContainerRefs.current[i] = el; }}
                    sx={{ position: "relative", bgcolor: "#000", border: `2px solid ${i === selectedIdx ? themeColors.accent : themeColors.border}`, borderRadius: 0 }}
                    onMouseDown={(e) => handleMouseDown(e, i)}
                    onMouseMove={(e) => handleMouseMove(e, i)}
                    onMouseUp={(e) => handleMouseUp(e, i)}
                    onMouseLeave={() => handleMouseLeave(i)}
                    onWheel={(i === selectedIdx || linkedZoom) ? (e) => handleWheel(e, i) : undefined}
                    onDoubleClick={() => handleDoubleClick(i)}
                  >
                    <canvas
                      ref={(el) => { if (el && canvasRefs.current[i] !== el) { canvasRefs.current[i] = el; setCanvasReady(c => c + 1); } }}
                      width={canvasW} height={canvasH}
                      style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }}
                    />
                    <canvas
                      ref={(el) => { overlayRefs.current[i] = el; }}
                      width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)}
                      style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }}
                    />
                    <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, borderRadius: "0 0 4px 0", "&:hover": { opacity: 1 } }} />
                  </Box>
                  <Typography sx={{ fontSize: 10, color: themeColors.textMuted, textAlign: "center", mt: 0.25 }}>
                    {labels?.[i] || `Image ${i + 1}`}
                  </Typography>
                  {showFft && (
                    <Box
                      ref={(el: HTMLDivElement | null) => { fftContainerRefs.current[i] = el; }}
                      sx={{ mt: 0.5, border: `2px solid ${i === selectedIdx ? themeColors.accent : themeColors.border}`, borderRadius: 0, bgcolor: "#000", cursor: "grab" }}
                      onWheel={(i === selectedIdx || linkedZoom) ? (e) => handleGalleryFftWheel(e, i) : undefined}
                      onDoubleClick={() => setGalleryFftState(i, { zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 })}
                      onMouseDown={(e) => handleGalleryFftMouseDown(e, i)}
                      onMouseMove={(e) => handleGalleryFftMouseMove(e, i)}
                      onMouseUp={handleGalleryFftMouseUp}
                      onMouseLeave={handleGalleryFftMouseUp}
                    >
                      <canvas
                        ref={(el) => { fftCanvasRefs.current[i] = el; }}
                        width={canvasW} height={canvasH}
                        style={{ width: canvasW, height: canvasH, imageRendering: "pixelated", display: "block" }}
                      />
                    </Box>
                  )}
                </Box>
              ))}
            </Box>
          ) : (
            /* Single image mode */
            <Box
              ref={(el: HTMLDivElement | null) => { imageContainerRefs.current[0] = el; }}
              sx={{ position: "relative", bgcolor: "#000", border: `1px solid ${themeColors.border}`, cursor: profileActive ? "crosshair" : "grab" }}
              onMouseDown={(e) => handleMouseDown(e, 0)}
              onMouseMove={(e) => handleMouseMove(e, 0)}
              onMouseUp={(e) => handleMouseUp(e, 0)}
              onMouseLeave={() => handleMouseLeave(0)}
              onWheel={(e) => handleWheel(e, 0)}
              onDoubleClick={() => handleDoubleClick(0)}
            >
              <canvas
                ref={(el) => { if (el && canvasRefs.current[0] !== el) { canvasRefs.current[0] = el; setCanvasReady(c => c + 1); } }}
                width={canvasW} height={canvasH}
                style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }}
              />
              <canvas
                ref={(el) => { overlayRefs.current[0] = el; }}
                width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)}
                style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }}
              />
              <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, borderRadius: "0 0 4px 0", "&:hover": { opacity: 1 } }} />
            </Box>
          )}

          {/* Stats bar - right below canvas (Show3D style) */}
          {showStats && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", maxWidth: isGallery ? ncols * canvasW + (ncols - 1) * 8 : canvasW, boxSizing: "border-box", overflow: "hidden", whiteSpace: "nowrap" }}>
              {isGallery && (
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>{labels?.[statsIdx] || `#${statsIdx + 1}`}</Typography>
              )}
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMean?.[statsIdx] ?? 0)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMin?.[statsIdx] ?? 0)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMax?.[statsIdx] ?? 0)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsStd?.[statsIdx] ?? 0)}</Box></Typography>
              {cursorInfo && (
                <>
                  <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 14 }} />
                  <Typography sx={{ fontSize: 11, color: themeColors.textMuted, fontFamily: "monospace" }}>
                    ({cursorInfo.row}, {cursorInfo.col}){pixelSizeAngstrom > 0 && <Box component="span" sx={{ opacity: 0.7 }}>{` = (${(cursorInfo.row * calibratedFactor).toFixed(1)}, ${(cursorInfo.col * calibratedFactor).toFixed(1)} ${calibratedUnit})`}</Box>} <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(cursorInfo.value)}</Box>
                  </Typography>
                </>
              )}
            </Box>
          )}

          {/* Line profile sparkline — always reserve space when profile is active */}
          {profileActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, maxWidth: canvasW, boxSizing: "border-box" }}>
              <canvas
                ref={profileCanvasRef}
                style={{ width: canvasW, height: 76, display: "block", border: `1px solid ${themeColors.border}` }}
              />
            </Box>
          )}

          {/* Controls: two rows left + histogram right (Show3D style) */}
          {showControls && (
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, maxWidth: isGallery ? ncols * canvasW + (ncols - 1) * 8 : canvasW, boxSizing: "border-box" }}>
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: Scale + Color */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                  <Select value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45 }} MenuProps={themedMenuProps}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                  <Select size="small" value={cmap} onChange={(e) => setCmap(e.target.value)} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 60 }}>
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
                </Box>
                {/* Row 2: Auto + Link Zoom (gallery) + zoom indicator */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto:<InfoTooltip text="Auto-contrast: clips display to 2nd–98th percentile to improve visibility of subtle features. When OFF, uses full data range." theme={themeInfo.theme} /></Typography>
                  <Switch checked={autoContrast} onChange={() => setAutoContrast(!autoContrast)} size="small" sx={switchStyles.small} />
                  {!isGallery && (
                    <>
                      <Typography sx={{ ...typography.label, fontSize: 10 }}>Colorbar:</Typography>
                      <Switch checked={showColorbar} onChange={() => setShowColorbar(!showColorbar)} size="small" sx={switchStyles.small} />
                    </>
                  )}
                  {isGallery && (
                    <>
                      <Typography sx={{ ...typography.label, fontSize: 10 }}>Link Zoom</Typography>
                      <Switch checked={linkedZoom} onChange={() => setLinkedZoom(!linkedZoom)} size="small" sx={switchStyles.small} />
                    </>
                  )}
                  {getZoomState(isGallery ? selectedIdx : 0).zoom !== 1 && (
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.accent, fontWeight: "bold" }}>{getZoomState(isGallery ? selectedIdx : 0).zoom.toFixed(1)}x</Typography>
                  )}
                </Box>
              </Box>
              {/* Right: Histogram spanning both rows */}
              {!isGallery && imageHistogramData && (
                <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
                  <Histogram data={imageHistogramData} colormap={cmap} vminPct={imageVminPct} vmaxPct={imageVmaxPct} onRangeChange={(min, max) => { setImageVminPct(min); setImageVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme === "dark" ? "dark" : "light"} dataMin={imageDataRange.min} dataMax={imageDataRange.max} />
                </Box>
              )}
            </Box>
          )}
        </Box>

        {/* FFT Panel - side-by-side (Show3D style, single mode only) */}
        {showFft && !isGallery && (
          <Box sx={{ width: canvasW }}>
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
              <Typography variant="caption" sx={{ ...typography.label, color: themeColors.text }}>FFT{gpuReady ? " (GPU)" : " (CPU)"}</Typography>
              <Button size="small" sx={compactButton} disabled={fftZoom === DEFAULT_FFT_ZOOM && fftPanX === 0 && fftPanY === 0} onClick={handleFftDoubleClick}>Reset</Button>
            </Stack>
            <Box
              ref={singleFftContainerRef}
              sx={{ position: "relative", bgcolor: "#000", border: `1px solid ${themeColors.border}`, cursor: "crosshair", width: canvasW, height: canvasH }}
              onWheel={handleFftWheel}
              onDoubleClick={handleFftDoubleClick}
              onMouseDown={handleFftMouseDown}
              onMouseMove={handleFftMouseMove}
              onMouseUp={handleFftMouseUp}
              onMouseLeave={handleFftMouseLeave}
            >
              <canvas ref={fftCanvasRef} width={canvasW} height={canvasH} style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }} />
              <canvas ref={fftOverlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} />
              <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, borderRadius: "0 0 4px 0", "&:hover": { opacity: 1 } }} />
            </Box>
            {/* FFT Stats Bar */}
            {fftStats && fftStats.length === 4 && (
              <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2 }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[0])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[1])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[2])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[3])}</Box></Typography>
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
            {/* FFT Controls - Two rows with histogram on right */}
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
              {/* Left: Two rows of controls */}
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: Scale + Auto */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                  <Select value={fftScaleMode} onChange={(e) => setFftScaleMode(e.target.value as "linear" | "log" | "power")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                    <MenuItem value="power">Pow</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto:<InfoTooltip text="Auto-enhance FFT display. When ON: masks DC component at center and clips to 99.9th percentile. When OFF: shows raw FFT with full dynamic range." theme={themeInfo.theme} /></Typography>
                  <Switch checked={fftAuto} onChange={(e) => setFftAuto(e.target.checked)} size="small" sx={switchStyles.small} />
                </Box>
                {/* Row 2: Color */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                  <Select value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
                </Box>
              </Box>
              {/* Right: Histogram spanning both rows */}
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
                {fftHistogramData && (
                  <Histogram data={fftHistogramData} colormap={fftColormap} vminPct={fftVminPct} vmaxPct={fftVmaxPct} onRangeChange={(min, max) => { setFftVminPct(min); setFftVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme === "dark" ? "dark" : "light"} dataMin={fftDataRange.min} dataMax={fftDataRange.max} />
                )}
              </Box>
            </Box>
          </Box>
        )}
      </Stack>
    </Box>
  );
}

export const render = createRender(Show2D);
