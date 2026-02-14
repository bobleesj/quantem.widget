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
import { useTheme } from "../theme";

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};
import { getWebGPUFFT, WebGPUFFT, fft2d, fftshift } from "../webgpu-fft";
import { COLORMAPS, COLORMAP_NAMES } from "../colormaps";
import "./show2d.css";

function formatNumber(val: number, decimals: number = 2): string {
  if (val === 0) return "0";
  if (Math.abs(val) >= 1000 || Math.abs(val) < 0.01) return val.toExponential(decimals);
  return val.toFixed(decimals);
}

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;

// ============================================================================
// Inline utilities
// ============================================================================
function extractBytes(dataView: DataView | ArrayBuffer | Uint8Array): Uint8Array {
  if (dataView instanceof Uint8Array) return dataView;
  if (dataView instanceof ArrayBuffer) return new Uint8Array(dataView);
  if (dataView && "buffer" in dataView) {
    return new Uint8Array(dataView.buffer, dataView.byteOffset, dataView.byteLength);
  }
  return new Uint8Array(0);
}

function roundToNiceValue(value: number): number {
  if (value <= 0) return 1;
  const magnitude = Math.pow(10, Math.floor(Math.log10(value)));
  const normalized = value / magnitude;
  if (normalized < 1.5) return magnitude;
  if (normalized < 3.5) return 2 * magnitude;
  if (normalized < 7.5) return 5 * magnitude;
  return 10 * magnitude;
}

function formatScaleLabel(value: number, unit: "Å" | "px"): string {
  const nice = roundToNiceValue(value);
  if (unit === "Å") {
    if (nice >= 10) return `${Math.round(nice / 10)} nm`;
    return nice >= 1 ? `${Math.round(nice)} Å` : `${nice.toFixed(2)} Å`;
  }
  return nice >= 1 ? `${Math.round(nice)} px` : `${nice.toFixed(1)} px`;
}

const DPR = window.devicePixelRatio || 1;

function drawScaleBarHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  zoom: number,
  pixelSize: number,
  unit: "Å" | "px",
  imageWidth: number,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const scaleX = cssWidth / imageWidth;
  const effectiveZoom = zoom * scaleX;

  const targetBarPx = 60;
  const barThickness = 5;
  const fontSize = 16;
  const margin = 12;

  const targetPhysical = (targetBarPx / effectiveZoom) * pixelSize;
  const nicePhysical = roundToNiceValue(targetPhysical);
  const barPx = (nicePhysical / pixelSize) * effectiveZoom;

  const barY = cssHeight - margin;
  const barX = cssWidth - barPx - margin;

  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  ctx.fillStyle = "white";
  ctx.fillRect(barX, barY, barPx, barThickness);

  const label = formatScaleLabel(nicePhysical, unit);
  ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
  ctx.fillStyle = "white";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(label, barX + barPx / 2, barY - 4);

  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText(`${zoom.toFixed(1)}×`, margin, cssHeight - margin + barThickness);

  ctx.restore();
}

// ============================================================================
// Histogram Component (ported from Show3D)
// ============================================================================
function computeHistogramFromBytes(data: Float32Array | null, numBins = 256): number[] {
  if (!data || data.length === 0) return new Array(numBins).fill(0);
  const bins = new Array(numBins).fill(0);
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (isFinite(v)) { if (v < min) min = v; if (v > max) max = v; }
  }
  if (!isFinite(min) || !isFinite(max) || min === max) return bins;
  const range = max - min;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (isFinite(v)) bins[Math.min(numBins - 1, Math.floor(((v - min) / range) * numBins))]++;
  }
  const maxCount = Math.max(...bins);
  if (maxCount > 0) for (let i = 0; i < numBins; i++) bins[i] /= maxCount;
  return bins;
}

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
// Types
// ============================================================================
type ZoomState = { zoom: number; panX: number; panY: number };

// ============================================================================
// Constants
// ============================================================================
const SINGLE_IMAGE_TARGET = 400;
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

  // Layout calculations
  const isGallery = nImages > 1;
  const displayScale = canvasSize / Math.max(width, height);
  const canvasW = Math.round(width * displayScale);
  const canvasH = Math.round(height * displayScale);
  const floatsPerImage = width * height;

  // Extract raw float32 bytes and parse into Float32Arrays
  const allFloats = React.useMemo(() => {
    const bytes = extractBytes(frameBytes);
    if (!bytes || bytes.length === 0) return null;
    // Create Float32Array from the raw bytes
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }, [frameBytes]);

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
      let min = Infinity, max = -Infinity;
      for (let i = 0; i < d.length; i++) {
        if (d[i] < min) min = d[i];
        if (d[i] > max) max = d[i];
      }
      setImageHistogramData(d);
      setImageDataRange({ min, max });
    }
  }, [allFloats, nImages, floatsPerImage]);

  // Prevent page scroll when scrolling on the active image canvas or FFT
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();

    // In gallery mode:
    // - If linkedZoom is ON, prevent scroll on ALL images (since all are zoomable)
    // - If linkedZoom is OFF, prevent scroll only on the selected image
    // In single mode: prevent scroll on the single image
    const targets: (HTMLCanvasElement | null)[] = [];

    if (isGallery) {
      if (linkedZoom) {
        // Add all available canvases
        targets.push(...canvasRefs.current);
      } else {
        // Add only selected
        targets.push(canvasRefs.current[selectedIdx]);
      }
    } else {
      targets.push(canvasRefs.current[0]);
    }

    // Add FFT canvases when visible (only selected in gallery mode)
    if (showFft) {
      if (isGallery) {
        if (linkedZoom) {
          targets.push(...fftCanvasRefs.current);
        } else {
          targets.push(fftCanvasRefs.current[selectedIdx]);
        }
      } else if (fftCanvasRef.current) {
        targets.push(fftCanvasRef.current);
      }
    }

    targets.forEach(t => t?.addEventListener("wheel", preventDefault, { passive: false }));

    return () => {
      targets.forEach(t => t?.removeEventListener("wheel", preventDefault));
    };
  }, [nImages, canvasReady, selectedIdx, isGallery, linkedZoom, dataReady, showFft]);

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
      let processed: Float32Array;
      if (logScale) {
        processed = new Float32Array(rawData.length);
        for (let j = 0; j < rawData.length; j++) {
          processed[j] = Math.log1p(Math.max(0, rawData[j]));
        }
      } else {
        processed = rawData;
      }

      // Compute min/max for normalization
      let vmin = processed[0];
      let vmax = processed[0];
      if (!isGallery && imageDataRange.min !== imageDataRange.max) {
        // Single mode: use histogram slider range
        const dataRange = imageDataRange.max - imageDataRange.min;
        vmin = imageDataRange.min + (imageVminPct / 100) * dataRange;
        vmax = imageDataRange.min + (imageVmaxPct / 100) * dataRange;
      } else if (autoContrast) {
        const sorted = Float32Array.from(processed).sort((a, b) => a - b);
        const p2Idx = Math.floor(sorted.length * 0.02);
        const p98Idx = Math.floor(sorted.length * 0.98);
        vmin = sorted[p2Idx];
        vmax = sorted[p98Idx];
      } else {
        for (let j = 1; j < processed.length; j++) {
          if (processed[j] < vmin) vmin = processed[j];
          if (processed[j] > vmax) vmax = processed[j];
        }
      }

      const range = vmax > vmin ? vmax - vmin : 1;

      // Create offscreen canvas at native resolution
      const offscreen = document.createElement("canvas");
      offscreen.width = width;
      offscreen.height = height;
      const offCtx = offscreen.getContext("2d");
      if (!offCtx) continue;

      const imgData = offCtx.createImageData(width, height);
      const rgba = imgData.data;

      for (let j = 0; j < processed.length; j++) {
        const clipped = Math.max(vmin, Math.min(vmax, processed[j]));
        const normalized = Math.floor(((clipped - vmin) / range) * 255);
        const k = j * 4;
        const lutIdx = normalized * 3;
        rgba[k] = lut[lutIdx];
        rgba[k + 1] = lut[lutIdx + 1];
        rgba[k + 2] = lut[lutIdx + 2];
        rgba[k + 3] = 255;
      }
      offCtx.putImageData(imgData, 0, 0);

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
  // Render Overlays (scale bar, selection, zoom indicator)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    for (let i = 0; i < nImages; i++) {
      const overlay = overlayRefs.current[i];
      if (!overlay) continue;

      if (scaleBarVisible) {
        const zs = linkedZoom ? linkedZoomState : (zoomStates.get(i) || DEFAULT_ZOOM_STATE);
        const unit = pixelSizeAngstrom > 0 ? "Å" as const : "px" as const;
        const pxSize = pixelSizeAngstrom > 0 ? pixelSizeAngstrom : 1;
        drawScaleBarHiDPI(overlay, DPR, zs.zoom, pxSize, unit, width);
      } else {
        const ctx = overlay.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, overlay.width, overlay.height);
      }
    }
  }, [nImages, pixelSizeAngstrom, scaleBarVisible, selectedIdx, isGallery, canvasW, canvasH, width, displayScale, linkedZoom, linkedZoomState, zoomStates, dataReady]);

  // -------------------------------------------------------------------------
  // Render FFT with WebGPU
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!showFft || isGallery || !fftCanvasRef.current || !rawDataRef.current) return;
    if (!rawDataRef.current[selectedIdx]) return;

    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const renderFFT = async (fftMag: Float32Array) => {
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

      // Auto mode: mask DC + 99.9% percentile clipping
      let displayMin: number, displayMax: number;
      if (fftAuto) {
        const centerIdx = Math.floor(height / 2) * width + Math.floor(width / 2);
        const neighbors = [
          magnitude[centerIdx - 1],
          magnitude[centerIdx + 1],
          magnitude[centerIdx - width],
          magnitude[centerIdx + width]
        ];
        magnitude[centerIdx] = neighbors.reduce((a, b) => a + b, 0) / 4;
        const sorted = magnitude.slice().sort((a, b) => a - b);
        displayMin = sorted[0];
        displayMax = sorted[Math.floor(sorted.length * 0.999)];
      } else {
        displayMin = Infinity;
        displayMax = -Infinity;
        for (let i = 0; i < magnitude.length; i++) {
          if (magnitude[i] < displayMin) displayMin = magnitude[i];
          if (magnitude[i] > displayMax) displayMax = magnitude[i];
        }
      }

      // Compute stats
      let sum = 0;
      for (let i = 0; i < magnitude.length; i++) sum += magnitude[i];
      const mean = sum / magnitude.length;
      let sumSq = 0;
      for (let i = 0; i < magnitude.length; i++) {
        const diff = magnitude[i] - mean;
        sumSq += diff * diff;
      }
      setFftStats([mean, displayMin, displayMax, Math.sqrt(sumSq / magnitude.length)]);

      // Store histogram data
      setFftHistogramData(magnitude.slice());
      setFftDataRange({ min: displayMin, max: displayMax });

      // Apply histogram slider clipping
      const dataRange = displayMax - displayMin;
      const vmin = displayMin + (fftVminPct / 100) * dataRange;
      const vmax = displayMin + (fftVmaxPct / 100) * dataRange;
      const range = vmax > vmin ? vmax - vmin : 1;

      const offscreen = document.createElement("canvas");
      offscreen.width = width;
      offscreen.height = height;
      const offCtx = offscreen.getContext("2d");
      if (!offCtx) return;

      const imgData = offCtx.createImageData(width, height);
      for (let i = 0; i < magnitude.length; i++) {
        const v = Math.round(((magnitude[i] - vmin) / range) * 255);
        const j = i * 4;
        const lutIdx = Math.max(0, Math.min(255, v)) * 3;
        imgData.data[j] = lut[lutIdx];
        imgData.data[j + 1] = lut[lutIdx + 1];
        imgData.data[j + 2] = lut[lutIdx + 2];
        imgData.data[j + 3] = 255;
      }
      offCtx.putImageData(imgData, 0, 0);

      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvasW, canvasH);
      ctx.save();

      const centerOffsetX = (canvasW - canvasW * fftZoom) / 2 + fftPanX;
      const centerOffsetY = (canvasH - canvasH * fftZoom) / 2 + fftPanY;

      ctx.translate(centerOffsetX, centerOffsetY);
      ctx.scale(fftZoom, fftZoom);
      ctx.drawImage(offscreen, 0, 0, width, height, 0, 0, canvasW, canvasH);
      ctx.restore();
    };

    const computeFFT = async () => {
      const data = rawDataRef.current![selectedIdx];
      const real = data.slice();
      const imag = new Float32Array(data.length);

      if (gpuFFTRef.current && gpuReady) {
        // WebGPU FFT
        const { real: fReal, imag: fImag } = await gpuFFTRef.current.fft2D(real, imag, width, height, false);
        fftshift(fReal, width, height);
        fftshift(fImag, width, height);

        const mag = new Float32Array(width * height);
        for (let i = 0; i < mag.length; i++) {
          mag[i] = Math.sqrt(fReal[i] ** 2 + fImag[i] ** 2);
        }
        await renderFFT(mag);
      } else {
        // CPU fallback
        fft2d(real, imag, width, height, false);
        fftshift(real, width, height);
        fftshift(imag, width, height);

        const mag = new Float32Array(width * height);
        for (let i = 0; i < mag.length; i++) {
          mag[i] = Math.sqrt(real[i] ** 2 + imag[i] ** 2);
        }
        await renderFFT(mag);
      }
    };

    computeFFT();
  }, [showFft, isGallery, selectedIdx, width, height, gpuReady, canvasW, canvasH, fftZoom, fftPanX, fftPanY, dataReady, fftVminPct, fftVmaxPct, fftColormap, fftScaleMode, fftAuto]);

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

        const mag = new Float32Array(width * height);
        for (let i = 0; i < mag.length; i++) {
          mag[i] = Math.sqrt(fReal[i] ** 2 + fImag[i] ** 2);
        }

        // Log scale and normalize
        let min = Infinity, max = -Infinity;
        const logData = new Float32Array(mag.length);
        for (let i = 0; i < mag.length; i++) {
          logData[i] = Math.log(1 + mag[i]);
          if (logData[i] < min) min = logData[i];
          if (logData[i] > max) max = logData[i];
        }

        // Render to offscreen canvas at native resolution
        const offscreen = document.createElement("canvas");
        offscreen.width = width;
        offscreen.height = height;
        const offCtx = offscreen.getContext("2d");
        if (!offCtx) continue;

        const imgData = offCtx.createImageData(width, height);
        const range = max - min || 1;
        for (let i = 0; i < logData.length; i++) {
          const v = Math.floor(((logData[i] - min) / range) * 255);
          const j = i * 4;
          imgData.data[j] = lut[v * 3];
          imgData.data[j + 1] = lut[v * 3 + 1];
          imgData.data[j + 2] = lut[v * 3 + 2];
          imgData.data[j + 3] = 255;
        }
        offCtx.putImageData(imgData, 0, 0);
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
    // In gallery mode, only allow zoom on the selected image
    if (isGallery && idx !== selectedIdx) return;
    
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
    setImageVminPct(0);
    setImageVmaxPct(100);
    setFftVminPct(0);
    setFftVmaxPct(100);
    setFftColormap("inferno");
    setFftScaleMode("log");
    setFftAuto(true);
  };

  // FFT zoom/pan handlers
  const handleFftWheel = (e: React.WheelEvent) => {
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    setFftZoom(Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, fftZoom * zoomFactor)));
  };

  const handleFftDoubleClick = () => {
    setFftZoom(DEFAULT_FFT_ZOOM);
    setFftPanX(0);
    setFftPanY(0);
  };

  const handleFftMouseDown = (e: React.MouseEvent) => {
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

  const handleFftMouseUp = () => {
    setIsDraggingFftPan(false);
    setFftPanStart(null);
  };

  // Gallery FFT zoom/pan handlers (only selected image's FFT responds)
  const handleGalleryFftWheel = (e: React.WheelEvent, idx: number) => {
    if (isGallery && idx !== selectedIdx) return;
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

  const handleMouseDown = (e: React.MouseEvent, idx: number) => {
    const zs = getZoomState(idx);
    if (isGallery && idx !== selectedIdx) {
      setSelectedIdx(idx);
      return; // Only select, don't start panning
    }
    setIsDraggingPan(true);
    setPanningIdx(idx);
    setPanStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
  };

  const handleMouseMove = (e: React.MouseEvent, idx: number) => {
    if (!isDraggingPan || !panStart || panningIdx === null) return;
    if (idx !== panningIdx) return;

    const canvas = canvasRefs.current[idx];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const dx = (e.clientX - panStart.x) * scaleX;
    const dy = (e.clientY - panStart.y) * scaleY;

    const zs = getZoomState(idx);
    setZoomState(idx, { ...zs, panX: panStart.pX + dx, panY: panStart.pY + dy });
  };

  const handleMouseUp = () => {
    setIsDraggingPan(false);
    setPanStart(null);
    setPanningIdx(null);
  };

  // -------------------------------------------------------------------------
  // Export handler
  const handleExport = React.useCallback(() => {
    const canvas = canvasRefs.current[isGallery ? selectedIdx : 0];
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (!blob) return;
      const link = document.createElement("a");
      link.download = `show2d_${labels?.[selectedIdx] || "image"}.png`;
      link.href = URL.createObjectURL(blob);
      link.click();
      URL.revokeObjectURL(link.href);
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
  // Render (Show3D-style layout)
  // -------------------------------------------------------------------------
  const needsReset = getZoomState(isGallery ? selectedIdx : 0).zoom !== 1 || getZoomState(isGallery ? selectedIdx : 0).panX !== 0 || getZoomState(isGallery ? selectedIdx : 0).panY !== 0;
  const statsIdx = isGallery ? selectedIdx : 0;

  return (
    <Box className="show2d-root" sx={{ p: 2, bgcolor: themeColors.bg, color: themeColors.text, overflow: "visible" }}>
      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        {/* Main panel */}
        <Box>
          {/* Header row: Title | FFT, Export, Reset */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28, maxWidth: isGallery ? ncols * canvasW + (ncols - 1) * 8 : canvasW }}>
            <Typography variant="caption" sx={{ ...typography.label, color: themeColors.text }}>{title || (isGallery ? "Gallery" : "Image")}</Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
              <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT:</Typography>
              <Switch checked={showFft} onChange={() => setShowFft(!showFft)} size="small" sx={switchStyles.small} />
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={handleExport}>EXPORT</Button>
              <Button size="small" sx={compactButton} disabled={!needsReset} onClick={handleResetAll}>RESET</Button>
            </Stack>
          </Stack>

          {isGallery ? (
            /* Gallery mode */
            <Box sx={{ display: "grid", gridTemplateColumns: `repeat(${ncols}, auto)`, gap: 1 }}>
              {Array.from({ length: nImages }).map((_, i) => (
                <Box key={i} sx={{ cursor: i === selectedIdx ? "grab" : "pointer" }}>
                  <Box
                    sx={{ position: "relative", bgcolor: "#000", border: `1px solid ${i === selectedIdx ? themeColors.accent : themeColors.border}` }}
                    onMouseDown={(e) => handleMouseDown(e, i)}
                    onMouseMove={(e) => handleMouseMove(e, i)}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                    onWheel={(e) => handleWheel(e, i)}
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
                      sx={{ mt: 0.5, border: `1px solid ${i === selectedIdx ? themeColors.accent : themeColors.border}`, bgcolor: "#000", cursor: "grab" }}
                      onWheel={(e) => handleGalleryFftWheel(e, i)}
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
              sx={{ position: "relative", bgcolor: "#000", border: `1px solid ${themeColors.border}`, cursor: "grab" }}
              onMouseDown={(e) => handleMouseDown(e, 0)}
              onMouseMove={(e) => handleMouseMove(e, 0)}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
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
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", maxWidth: isGallery ? ncols * canvasW + (ncols - 1) * 8 : canvasW, boxSizing: "border-box" }}>
              {isGallery && (
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>{labels?.[statsIdx] || `#${statsIdx + 1}`}</Typography>
              )}
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMean?.[statsIdx] ?? 0)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMin?.[statsIdx] ?? 0)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMax?.[statsIdx] ?? 0)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsStd?.[statsIdx] ?? 0)}</Box></Typography>
            </Box>
          )}

          {/* Controls: two rows left + histogram right (Show3D style) */}
          {showControls && (
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, maxWidth: isGallery ? ncols * canvasW + (ncols - 1) * 8 : canvasW, boxSizing: "border-box" }}>
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: Scale + Color */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                  <Select value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45 }} MenuProps={upwardMenuProps}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                  <Select size="small" value={cmap} onChange={(e) => setCmap(e.target.value)} MenuProps={upwardMenuProps} sx={{ ...themedSelect, minWidth: 60 }}>
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
                </Box>
                {/* Row 2: Auto + Link Zoom (gallery) + zoom indicator */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto:</Typography>
                  <Switch checked={autoContrast} onChange={() => setAutoContrast(!autoContrast)} size="small" sx={switchStyles.small} />
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
              <Typography variant="caption" sx={{ ...typography.label, color: themeColors.text }}>FFT</Typography>
              <Button size="small" sx={compactButton} disabled={fftZoom === DEFAULT_FFT_ZOOM && fftPanX === 0 && fftPanY === 0} onClick={handleFftDoubleClick}>Reset</Button>
            </Stack>
            <Box
              sx={{ position: "relative", bgcolor: "#000", border: `1px solid ${themeColors.border}`, cursor: "grab", width: canvasW, height: canvasH }}
              onWheel={handleFftWheel}
              onDoubleClick={handleFftDoubleClick}
              onMouseDown={handleFftMouseDown}
              onMouseMove={handleFftMouseMove}
              onMouseUp={handleFftMouseUp}
              onMouseLeave={handleFftMouseUp}
            >
              <canvas ref={fftCanvasRef} width={canvasW} height={canvasH} style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }} />
              <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, borderRadius: "0 0 4px 0", "&:hover": { opacity: 1 } }} />
            </Box>
            {/* FFT Stats Bar */}
            {fftStats && fftStats.length === 4 && (
              <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2 }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[0])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[1])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[2])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[3])}</Box></Typography>
              </Box>
            )}
            {/* FFT Controls - Two rows with histogram on right */}
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
              {/* Left: Two rows of controls */}
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: Scale + Auto */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                  <Select value={fftScaleMode} onChange={(e) => setFftScaleMode(e.target.value as "linear" | "log" | "power")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={upwardMenuProps}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                    <MenuItem value="power">Pow</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto:</Typography>
                  <Switch checked={fftAuto} onChange={(e) => setFftAuto(e.target.checked)} size="small" sx={switchStyles.small} />
                </Box>
                {/* Row 2: Color */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                  <Select value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={upwardMenuProps}>
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
