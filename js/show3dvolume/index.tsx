/// <reference types="@webgpu/types" />
/**
 * Show3DVolume - Orthogonal slice viewer for 3D volumetric data.
 *
 * Three side-by-side canvases showing XY, XZ, YZ planes with sliders.
 * All slicing done in JS from raw float32 volume data for instant response.
 *
 * Self-contained widget with all utilities inlined (matching Show3D pattern).
 */
import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Slider from "@mui/material/Slider";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import FastForwardIcon from "@mui/icons-material/FastForward";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import StopIcon from "@mui/icons-material/Stop";
import "./show3dvolume.css";
import { useTheme } from "../theme";
import { VolumeRenderer, CameraState, DEFAULT_CAMERA } from "../webgl-volume";

// ============================================================================
// UI Styles (matching Show3D exactly)
// ============================================================================
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const },
};

const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };

const controlPanel = {
  select: { minWidth: 90, fontSize: 11, "& .MuiSelect-select": { py: 0.5 } },
};

const switchStyles = {
  small: { "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-switchBase": { padding: "4px" } },
};

const sliderStyles = {
  small: {
    "& .MuiSlider-thumb": { width: 12, height: 12 },
    "& .MuiSlider-rail": { height: 3 },
    "& .MuiSlider-track": { height: 3 },
  },
};

const container = {
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "monospace", overflow: "visible" },
  imageBox: { bgcolor: "#000", border: "1px solid #444", overflow: "hidden", position: "relative" as const },
};

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

const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  "&.Mui-disabled": { color: "#666", borderColor: "#444" },
};

import { COLORMAPS, COLORMAP_NAMES } from "../colormaps";

// Formatting (matching Show3D)
function formatNumber(val: number, decimals: number = 2): string {
  if (val === 0) return "0";
  if (Math.abs(val) >= 1000 || Math.abs(val) < 0.01) return val.toExponential(decimals);
  return val.toFixed(decimals);
}

import { WebGPUFFT, getWebGPUFFT, fft2d, fftshift, nextPow2 } from "../webgpu-fft";

// Extract bytes from DataView (matching Show3D)
function extractBytes(dataView: DataView | ArrayBuffer | Uint8Array): Uint8Array {
  if (dataView instanceof Uint8Array) return dataView;
  if (dataView instanceof ArrayBuffer) return new Uint8Array(dataView);
  if (dataView && "buffer" in dataView) return new Uint8Array(dataView.buffer, dataView.byteOffset, dataView.byteLength);
  return new Uint8Array(0);
}

// ============================================================================
// Zoom constants (matching Show3D)
// ============================================================================
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;

// ============================================================================
// Slice extraction from flat float32 buffer
// ============================================================================
function extractXY(vol: Float32Array, nx: number, ny: number, _nz: number, z: number): Float32Array {
  const start = z * ny * nx;
  return vol.subarray(start, start + ny * nx);
}

function extractXZ(vol: Float32Array, nx: number, ny: number, nz: number, y: number): Float32Array {
  const out = new Float32Array(nz * nx);
  for (let z = 0; z < nz; z++) {
    const srcOffset = z * ny * nx + y * nx;
    for (let x = 0; x < nx; x++) out[z * nx + x] = vol[srcOffset + x];
  }
  return out;
}

function extractYZ(vol: Float32Array, nx: number, ny: number, nz: number, x: number): Float32Array {
  const out = new Float32Array(nz * ny);
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) out[z * ny + y] = vol[z * ny * nx + y * nx + x];
  }
  return out;
}

// ============================================================================
// Constants
// ============================================================================
type ZoomState = { zoom: number; panX: number; panY: number };
const DEFAULT_ZOOM: ZoomState = { zoom: 1, panX: 0, panY: 0 };
const CANVAS_TARGET = 400;
const AXES = ["xy", "xz", "yz"] as const;

// ============================================================================
// Main Component
// ============================================================================
function Show3DVolume() {
  // Theme detection
  const { themeInfo, colors: baseColors } = useTheme();
  const tc = {
    ...baseColors,
    accentGreen: themeInfo.theme === "dark" ? "#0f0" : "#0a0",
    accentYellow: themeInfo.theme === "dark" ? "#ff0" : "#cc0",
  };

  // Initialize WebGPU FFT
  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) { gpuFFTRef.current = fft; setGpuReady(true); }
    });
  }, []);

  const themedSelect = {
    ...controlPanel.select,
    bgcolor: tc.controlBg,
    color: tc.text,
    "& .MuiSelect-select": { py: 0.5 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: tc.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: tc.accent },
  };

  // Model state
  const [nx] = useModelState<number>("nx");
  const [ny] = useModelState<number>("ny");
  const [nz] = useModelState<number>("nz");
  const [volumeBytes] = useModelState<DataView>("volume_bytes");
  const [sliceX, setSliceX] = useModelState<number>("slice_x");
  const [sliceY, setSliceY] = useModelState<number>("slice_y");
  const [sliceZ, setSliceZ] = useModelState<number>("slice_z");
  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [showControls] = useModelState<boolean>("show_controls");
  const [showStats] = useModelState<boolean>("show_stats");
  const [showCrosshair, setShowCrosshair] = useModelState<boolean>("show_crosshair");
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [dimLabels] = useModelState<string[]>("dim_labels");
  const [statsMean] = useModelState<number[]>("stats_mean");
  const [statsMin] = useModelState<number[]>("stats_min");
  const [statsMax] = useModelState<number[]>("stats_max");
  const [statsStd] = useModelState<number[]>("stats_std");

  // Canvas refs
  const canvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const overlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);

  // FFT state
  const [fftColormap, setFftColormap] = React.useState("inferno");
  const [fftLogScale, setFftLogScale] = React.useState(false);
  const [fftAuto, setFftAuto] = React.useState(true);
  const [fftZooms, setFftZooms] = React.useState<ZoomState[]>([DEFAULT_ZOOM, DEFAULT_ZOOM, DEFAULT_ZOOM]);
  const [fftDragAxis, setFftDragAxis] = React.useState<number | null>(null);
  const [fftDragStart, setFftDragStart] = React.useState<{ x: number; y: number; pX: number; pY: number } | null>(null);
  const fftCanvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const fftOffscreenRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);

  // Zoom/pan per axis
  const [zooms, setZooms] = React.useState<ZoomState[]>([DEFAULT_ZOOM, DEFAULT_ZOOM, DEFAULT_ZOOM]);
  const [dragAxis, setDragAxis] = React.useState<number | null>(null);
  const [dragStart, setDragStart] = React.useState<{ x: number; y: number; pX: number; pY: number } | null>(null);

  // Canvas resize (matching Show2D pattern)
  const [canvasTarget, setCanvasTarget] = React.useState(CANVAS_TARGET);
  const [isResizing, setIsResizing] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number; y: number; size: number } | null>(null);

  // Playback state
  const [playing, setPlaying] = React.useState(false);
  const [playAxis, setPlayAxis] = React.useState<number>(0); // 0=Z(XY), 1=Y(XZ), 2=X(YZ), 3=All
  const [reverse, setReverse] = React.useState(false);
  const [fps, setFps] = React.useState(5);
  const [loop, setLoop] = React.useState(true);
  const playIntervalRef = React.useRef<number | null>(null);
  const [boomerang, setBoomerang] = React.useState(false);
  const bounceDirRef = React.useRef<1 | -1>(1);
  const [loopStarts, setLoopStarts] = React.useState([0, 0, 0]);
  const [loopEnds, setLoopEnds] = React.useState([-1, -1, -1]);

  // 3D volume renderer state
  const volumeCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const volumeRendererRef = React.useRef<VolumeRenderer | null>(null);
  const [camera, setCamera] = React.useState<CameraState>(DEFAULT_CAMERA);
  const [volumeDrag, setVolumeDrag] = React.useState<{
    button: number; x: number; y: number; yaw: number; pitch: number; panX: number; panY: number;
  } | null>(null);
  const [webglSupported, setWebglSupported] = React.useState(true);
  const [volumeOpacity, setVolumeOpacity] = React.useState(0.5);
  const [volumeBrightness, setVolumeBrightness] = React.useState(1.0);
  const [volumeCanvasSize, setVolumeCanvasSize] = React.useState(300);
  const [volumeResizing, setVolumeResizing] = React.useState(false);
  const volumeResizeStartRef = React.useRef<{ x: number; y: number; size: number } | null>(null);
  const [showSlicePlanes, setShowSlicePlanes] = React.useState(false);

  // Parse volume data
  const allFloats = React.useMemo(() => {
    const bytes = extractBytes(volumeBytes);
    if (!bytes || bytes.length === 0) return null;
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }, [volumeBytes]);

  // Slice dimensions: [xy: ny x nx], [xz: nz x nx], [yz: nz x ny]
  const sliceDims: [number, number][] = React.useMemo(() => [[ny, nx], [nz, nx], [nz, ny]], [nx, ny, nz]);

  // Canvas sizes
  const canvasSizes = React.useMemo(() => {
    return sliceDims.map(([h, w]) => {
      const scale = canvasTarget / Math.max(w, h);
      return { w: Math.round(w * scale), h: Math.round(h * scale), scale };
    });
  }, [sliceDims, canvasTarget]);

  // Prevent page scroll on canvases
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    canvasRefs.current.forEach(c => c?.addEventListener("wheel", preventDefault, { passive: false }));
    fftCanvasRefs.current.forEach(c => c?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => {
      canvasRefs.current.forEach(c => c?.removeEventListener("wheel", preventDefault));
      fftCanvasRefs.current.forEach(c => c?.removeEventListener("wheel", preventDefault));
    };
  }, [allFloats, showFft]);

  // Sync boomerang direction ref with reverse state
  React.useEffect(() => {
    bounceDirRef.current = reverse ? -1 : 1;
  }, [reverse]);

  // -------------------------------------------------------------------------
  // 3D Volume Renderer — init, upload, render
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const canvas = volumeCanvasRef.current;
    if (!canvas) return;
    if (!VolumeRenderer.isSupported()) { setWebglSupported(false); return; }
    try {
      const renderer = new VolumeRenderer(canvas);
      volumeRendererRef.current = renderer;
    } catch { setWebglSupported(false); }
    return () => { volumeRendererRef.current?.dispose(); volumeRendererRef.current = null; };
  }, []);

  // Upload volume data
  React.useEffect(() => {
    const renderer = volumeRendererRef.current;
    if (!renderer || !allFloats || allFloats.length === 0) return;
    renderer.uploadVolume(allFloats, nx, ny, nz);
  }, [allFloats, nx, ny, nz]);

  // Upload colormap
  React.useEffect(() => {
    const renderer = volumeRendererRef.current;
    if (!renderer) return;
    renderer.uploadColormap(COLORMAPS[cmap] || COLORMAPS.inferno);
  }, [cmap]);

  // Render 3D volume
  React.useEffect(() => {
    const renderer = volumeRendererRef.current;
    if (!renderer || !allFloats || allFloats.length === 0) return;
    // Parse background color from theme
    const bgHex = tc.bg;
    const r = parseInt(bgHex.slice(1, 3), 16) / 255;
    const g = parseInt(bgHex.slice(3, 5), 16) / 255;
    const b = parseInt(bgHex.slice(5, 7), 16) / 255;
    renderer.render({
      sliceX, sliceY, sliceZ, nx, ny, nz,
      opacity: volumeOpacity, brightness: volumeBrightness,
      showSlicePlanes,
    }, camera, [r, g, b]);
  }, [allFloats, sliceX, sliceY, sliceZ, nx, ny, nz, cmap, camera, volumeOpacity, volumeBrightness, volumeCanvasSize, tc.bg, showSlicePlanes]);

  // Prevent scroll on volume canvas
  React.useEffect(() => {
    const canvas = volumeCanvasRef.current;
    if (!canvas || !webglSupported) return;
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    canvas.addEventListener("wheel", preventDefault, { passive: false });
    return () => canvas.removeEventListener("wheel", preventDefault);
  }, [webglSupported]);

  // -------------------------------------------------------------------------
  // 3D Volume mouse handlers
  // -------------------------------------------------------------------------
  const handleVolumeMouseDown = (e: React.MouseEvent) => {
    setVolumeDrag({
      button: e.button, x: e.clientX, y: e.clientY,
      yaw: camera.yaw, pitch: camera.pitch, panX: camera.panX, panY: camera.panY,
    });
    e.preventDefault();
  };

  const handleVolumeMouseMove = (e: React.MouseEvent) => {
    if (!volumeDrag) return;
    const dx = e.clientX - volumeDrag.x;
    const dy = e.clientY - volumeDrag.y;
    if (volumeDrag.button === 0 && !e.shiftKey) {
      // Left drag = rotate
      setCamera(prev => ({
        ...prev,
        yaw: volumeDrag.yaw + dx * 0.005,
        pitch: Math.max(-Math.PI * 0.49, Math.min(Math.PI * 0.49, volumeDrag.pitch - dy * 0.005)),
      }));
    } else {
      // Right drag or shift+drag = pan
      const sens = 0.003 * camera.distance;
      setCamera(prev => ({
        ...prev,
        panX: volumeDrag.panX + dx * sens,
        panY: volumeDrag.panY - dy * sens,
      }));
    }
  };

  const handleVolumeMouseUp = () => setVolumeDrag(null);

  const handleVolumeWheel = (e: React.WheelEvent) => {
    const factor = e.deltaY > 0 ? 1.1 : 0.9;
    setCamera(prev => ({ ...prev, distance: Math.max(0.5, Math.min(10, prev.distance * factor)) }));
  };

  const handleVolumeDoubleClick = () => setCamera(DEFAULT_CAMERA);

  // -------------------------------------------------------------------------
  // 3D Volume canvas resize
  // -------------------------------------------------------------------------
  const handleVolumeResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation(); e.preventDefault();
    setVolumeResizing(true);
    volumeResizeStartRef.current = { x: e.clientX, y: e.clientY, size: volumeCanvasSize };
  };

  React.useEffect(() => {
    if (!volumeResizing) return;
    const onMove = (e: MouseEvent) => {
      const start = volumeResizeStartRef.current;
      if (!start) return;
      const delta = Math.max(e.clientX - start.x, e.clientY - start.y);
      setVolumeCanvasSize(Math.max(300, Math.min(800, start.size + delta)));
    };
    const onUp = () => { setVolumeResizing(false); volumeResizeStartRef.current = null; };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => { document.removeEventListener("mousemove", onMove); document.removeEventListener("mouseup", onUp); };
  }, [volumeResizing]);

  const cameraChanged = camera.yaw !== DEFAULT_CAMERA.yaw || camera.pitch !== DEFAULT_CAMERA.pitch || camera.distance !== DEFAULT_CAMERA.distance || camera.panX !== DEFAULT_CAMERA.panX || camera.panY !== DEFAULT_CAMERA.panY;

  // Any zoom active?
  const needsReset = zooms.some(z => z.zoom !== 1 || z.panX !== 0 || z.panY !== 0) || fftZooms.some(z => z.zoom !== 1 || z.panX !== 0 || z.panY !== 0) || cameraChanged;

  // -------------------------------------------------------------------------
  // Render slices
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!allFloats || allFloats.length === 0) return;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const sliceData = [
      extractXY(allFloats, nx, ny, nz, sliceZ),
      extractXZ(allFloats, nx, ny, nz, sliceY),
      extractYZ(allFloats, nx, ny, nz, sliceX),
    ];
    for (let a = 0; a < 3; a++) {
      const canvas = canvasRefs.current[a];
      if (!canvas) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;
      const [sliceH, sliceW] = sliceDims[a];
      const { w: cw, h: ch } = canvasSizes[a];
      let processed = sliceData[a];
      if (logScale) {
        const tmp = new Float32Array(processed.length);
        for (let i = 0; i < processed.length; i++) tmp[i] = Math.log1p(Math.max(0, processed[i]));
        processed = tmp;
      }
      let vmin = processed[0], vmax = processed[0];
      if (autoContrast) {
        const sorted = Float32Array.from(processed).sort((a, b) => a - b);
        vmin = sorted[Math.floor(sorted.length * 0.02)];
        vmax = sorted[Math.floor(sorted.length * 0.98)];
      } else {
        for (let i = 1; i < processed.length; i++) {
          if (processed[i] < vmin) vmin = processed[i];
          if (processed[i] > vmax) vmax = processed[i];
        }
      }
      const range = vmax - vmin || 1;
      const offscreen = document.createElement("canvas");
      offscreen.width = sliceW;
      offscreen.height = sliceH;
      const offCtx = offscreen.getContext("2d")!;
      const imgData = offCtx.createImageData(sliceW, sliceH);
      const rgba = imgData.data;
      for (let i = 0; i < processed.length; i++) {
        const v = Math.max(0, Math.min(255, Math.round(((processed[i] - vmin) / range) * 255)));
        const k = i * 4;
        rgba[k] = lut[v * 3];
        rgba[k + 1] = lut[v * 3 + 1];
        rgba[k + 2] = lut[v * 3 + 2];
        rgba[k + 3] = 255;
      }
      offCtx.putImageData(imgData, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, cw, ch);
      const zs = zooms[a];
      if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
        ctx.save();
        const cx = cw / 2, cy = ch / 2;
        ctx.translate(cx + zs.panX, cy + zs.panY);
        ctx.scale(zs.zoom, zs.zoom);
        ctx.translate(-cx, -cy);
        ctx.drawImage(offscreen, 0, 0, sliceW, sliceH, 0, 0, cw, ch);
        ctx.restore();
      } else {
        ctx.drawImage(offscreen, 0, 0, sliceW, sliceH, 0, 0, cw, ch);
      }
    }
  }, [allFloats, sliceX, sliceY, sliceZ, nx, ny, nz, cmap, logScale, autoContrast, zooms, sliceDims, canvasSizes]);

  // -------------------------------------------------------------------------
  // Render overlays (crosshair lines)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!allFloats) return;
    const crossPositions: [number, number][] = [
      [sliceX, sliceY],
      [sliceX, sliceZ],
      [sliceY, sliceZ],
    ];
    for (let a = 0; a < 3; a++) {
      const overlay = overlayRefs.current[a];
      if (!overlay) continue;
      const ctx = overlay.getContext("2d");
      if (!ctx) continue;
      const { w: cw, h: ch, scale } = canvasSizes[a];
      ctx.clearRect(0, 0, cw, ch);
      if (showCrosshair) {
        const zs = zooms[a];
        const [dataX, dataY] = crossPositions[a];
        const cx = cw / 2, cy = ch / 2;
        let canvasX = dataX * scale;
        let canvasY = dataY * scale;
        if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
          canvasX = (canvasX - cx) * zs.zoom + cx + zs.panX;
          canvasY = (canvasY - cy) * zs.zoom + cy + zs.panY;
        }
        ctx.strokeStyle = tc.accentYellow + "80";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(canvasX, 0); ctx.lineTo(canvasX, ch); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, canvasY); ctx.lineTo(cw, canvasY); ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }, [allFloats, sliceX, sliceY, sliceZ, zooms, showCrosshair, tc, sliceDims, canvasSizes]);

  // -------------------------------------------------------------------------
  // FFT computation and caching
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!showFft || !allFloats || allFloats.length === 0) return;

    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;

    const computeAllFFTs = async () => {
      const sliceData = [
        extractXY(allFloats, nx, ny, nz, sliceZ),
        extractXZ(allFloats, nx, ny, nz, sliceY),
        extractYZ(allFloats, nx, ny, nz, sliceX),
      ];
      const dims: [number, number][] = [[ny, nx], [nz, nx], [nz, ny]];

      for (let a = 0; a < 3; a++) {
        const data = sliceData[a];
        const [sliceH, sliceW] = dims[a];
        // Pad to power-of-2 so fftshift works correctly on the full result
        const pw = nextPow2(sliceW), ph = nextPow2(sliceH);
        const paddedSize = pw * ph;
        let real: Float32Array, imag: Float32Array;

        if (gpuReady && gpuFFTRef.current) {
          // Pad input manually so GPU FFT doesn't crop
          const padReal = new Float32Array(paddedSize);
          const padImag = new Float32Array(paddedSize);
          for (let y = 0; y < sliceH; y++) for (let x = 0; x < sliceW; x++) padReal[y * pw + x] = data[y * sliceW + x];
          const result = await gpuFFTRef.current.fft2D(padReal, padImag, pw, ph, false);
          real = result.real; imag = result.imag;
        } else {
          real = new Float32Array(paddedSize);
          imag = new Float32Array(paddedSize);
          for (let y = 0; y < sliceH; y++) for (let x = 0; x < sliceW; x++) real[y * pw + x] = data[y * sliceW + x];
          fft2d(real, imag, pw, ph, false);
        }

        fftshift(real, pw, ph);
        fftshift(imag, pw, ph);

        const mag = new Float32Array(paddedSize);
        for (let i = 0; i < paddedSize; i++) mag[i] = Math.sqrt(real[i] ** 2 + imag[i] ** 2);

        let displayMin: number, displayMax: number;
        if (fftAuto) {
          const centerIdx = Math.floor(ph / 2) * pw + Math.floor(pw / 2);
          const neighbors = [
            mag[Math.max(0, centerIdx - 1)], mag[Math.min(mag.length - 1, centerIdx + 1)],
            mag[Math.max(0, centerIdx - pw)], mag[Math.min(mag.length - 1, centerIdx + pw)],
          ];
          mag[centerIdx] = neighbors.reduce((s, v) => s + v, 0) / 4;
          const sorted = mag.slice().sort((x, y) => x - y);
          displayMin = sorted[0];
          displayMax = sorted[Math.floor(sorted.length * 0.999)];
        } else {
          displayMin = mag[0]; displayMax = mag[0];
          for (let i = 1; i < mag.length; i++) { if (mag[i] < displayMin) displayMin = mag[i]; if (mag[i] > displayMax) displayMax = mag[i]; }
        }

        const displayData = new Float32Array(paddedSize);
        for (let i = 0; i < paddedSize; i++) displayData[i] = fftLogScale ? Math.log(1 + mag[i]) : mag[i];
        if (fftLogScale) { displayMin = Math.log(1 + displayMin); displayMax = Math.log(1 + displayMax); }

        const range = displayMax > displayMin ? displayMax - displayMin : 1;
        const offscreen = document.createElement("canvas");
        offscreen.width = pw; offscreen.height = ph;
        const offCtx = offscreen.getContext("2d");
        if (!offCtx) continue;

        const imgData = offCtx.createImageData(pw, ph);
        const rgba = imgData.data;
        for (let i = 0; i < paddedSize; i++) {
          const clipped = Math.max(displayMin, Math.min(displayMax, displayData[i]));
          const v = Math.floor(((clipped - displayMin) / range) * 255);
          const k = i * 4;
          rgba[k] = lut[v * 3]; rgba[k + 1] = lut[v * 3 + 1]; rgba[k + 2] = lut[v * 3 + 2]; rgba[k + 3] = 255;
        }
        offCtx.putImageData(imgData, 0, 0);
        fftOffscreenRefs.current[a] = offscreen;

        const canvas = fftCanvasRefs.current[a];
        if (canvas) {
          const ctx = canvas.getContext("2d");
          if (ctx) {
            const { w: cw, h: ch } = canvasSizes[a];
            ctx.imageSmoothingEnabled = false;
            ctx.clearRect(0, 0, cw, ch);
            const zs = fftZooms[a];
            if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
              ctx.save();
              const cx = cw / 2, cy = ch / 2;
              ctx.translate(cx + zs.panX, cy + zs.panY); ctx.scale(zs.zoom, zs.zoom); ctx.translate(-cx, -cy);
              ctx.drawImage(offscreen, 0, 0, pw, ph, 0, 0, cw, ch);
              ctx.restore();
            } else {
              ctx.drawImage(offscreen, 0, 0, pw, ph, 0, 0, cw, ch);
            }
          }
        }
      }
    };

    computeAllFFTs();
  }, [showFft, allFloats, sliceX, sliceY, sliceZ, nx, ny, nz, fftColormap, fftLogScale, fftAuto, gpuReady, canvasSizes, fftZooms]);

  // Redraw cached FFT with zoom/pan (cheap -- no recomputation)
  React.useEffect(() => {
    if (!showFft) return;
    for (let a = 0; a < 3; a++) {
      const canvas = fftCanvasRefs.current[a];
      const offscreen = fftOffscreenRefs.current[a];
      if (!canvas || !offscreen) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;
      const { w: cw, h: ch } = canvasSizes[a];
      const ow = offscreen.width, oh = offscreen.height;
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, cw, ch);
      const zs = fftZooms[a];
      if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
        ctx.save();
        const cx = cw / 2, cy = ch / 2;
        ctx.translate(cx + zs.panX, cy + zs.panY); ctx.scale(zs.zoom, zs.zoom); ctx.translate(-cx, -cy);
        ctx.drawImage(offscreen, 0, 0, ow, oh, 0, 0, cw, ch);
        ctx.restore();
      } else {
        ctx.drawImage(offscreen, 0, 0, ow, oh, 0, 0, cw, ch);
      }
    }
  }, [showFft, fftZooms, canvasSizes]);

  // -------------------------------------------------------------------------
  // Playback logic (matching Show3D pattern)
  // -------------------------------------------------------------------------
  const sliceSettersRef = React.useRef<((v: number) => void)[]>([setSliceZ, setSliceY, setSliceX]);
  sliceSettersRef.current = [setSliceZ, setSliceY, setSliceX];
  const effectiveLoopEnds = React.useMemo(
    () => loopEnds.map((end, i) => {
      const max = [nz - 1, ny - 1, nx - 1][i];
      return end < 0 ? max : Math.min(end, max);
    }),
    [loopEnds, nx, ny, nz],
  );
  React.useEffect(() => {
    if (!playing) return;
    const intervalMs = 1000 / fps;

    if (playAxis === 3) {
      // "All" mode: advance all 3 axes simultaneously
      playIntervalRef.current = window.setInterval(() => {
        const dir = boomerang ? bounceDirRef.current : (reverse ? -1 : 1);
        // Check if any axis would go out of range
        let shouldBounce = false;
        for (let a = 0; a < 3; a++) {
          const next = sliceValuesRef.current[a] + dir;
          if (next > effectiveLoopEnds[a] || next < loopStarts[a]) { shouldBounce = true; break; }
        }
        if (boomerang && shouldBounce) {
          bounceDirRef.current = (-bounceDirRef.current) as 1 | -1;
        }
        const finalDir = boomerang ? bounceDirRef.current : dir;
        for (let a = 0; a < 3; a++) {
          const start = loopStarts[a];
          const end = effectiveLoopEnds[a];
          let next = sliceValuesRef.current[a] + finalDir;
          if (next > end) next = loop || boomerang ? start : end;
          else if (next < start) next = loop || boomerang ? end : start;
          sliceSettersRef.current[a](next);
          sliceValuesRef.current[a] = next;
        }
        if (!loop && !boomerang && shouldBounce) setPlaying(false);
      }, intervalMs);
    } else {
      // Single axis mode
      const axis = playAxis;
      const start = loopStarts[axis];
      const end = effectiveLoopEnds[axis];
      const setter = sliceSettersRef.current[axis];
      playIntervalRef.current = window.setInterval(() => {
        setter((prev: number) => {
          if (boomerang) {
            const next = prev + bounceDirRef.current;
            if (next > end) { bounceDirRef.current = -1; return prev - 1 >= start ? prev - 1 : prev; }
            if (next < start) { bounceDirRef.current = 1; return prev + 1 <= end ? prev + 1 : prev; }
            return next;
          }
          let next = prev + (reverse ? -1 : 1);
          if (reverse) {
            if (next < start) { if (!loop) setPlaying(false); return loop ? end : start; }
          } else {
            if (next > end) { if (!loop) setPlaying(false); return loop ? start : end; }
          }
          return next;
        });
      }, intervalMs);
    }
    return () => {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
        playIntervalRef.current = null;
      }
    };
  }, [playing, fps, reverse, boomerang, loop, playAxis, loopStarts, effectiveLoopEnds]);

  // -------------------------------------------------------------------------
  // Zoom/Pan handlers (matching Show3D)
  // -------------------------------------------------------------------------
  const handleWheel = (e: React.WheelEvent, axis: number) => {
    const canvas = canvasRefs.current[axis];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const zs = zooms[axis];
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const cx = canvas.width / 2, cy = canvas.height / 2;
    const imgX = (mouseX - cx - zs.panX) / zs.zoom + cx;
    const imgY = (mouseY - cy - zs.panY) / zs.zoom + cy;
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zs.zoom * factor));
    const newPanX = mouseX - (imgX - cx) * newZoom - cx;
    const newPanY = mouseY - (imgY - cy) * newZoom - cy;
    setZooms(prev => { const next = [...prev]; next[axis] = { zoom: newZoom, panX: newPanX, panY: newPanY }; return next; });
  };

  const handleDoubleClick = (axis: number) => {
    setZooms(prev => { const next = [...prev]; next[axis] = DEFAULT_ZOOM; return next; });
  };

  const handleMouseDown = (e: React.MouseEvent, axis: number) => {
    const zs = zooms[axis];
    setDragAxis(axis);
    setDragStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
  };

  const handleMouseMove = (e: React.MouseEvent, axis: number) => {
    if (dragAxis !== axis || !dragStart) return;
    const canvas = canvasRefs.current[axis];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dx = (e.clientX - dragStart.x) * (canvas.width / rect.width);
    const dy = (e.clientY - dragStart.y) * (canvas.height / rect.height);
    setZooms(prev => { const next = [...prev]; next[axis] = { ...prev[axis], panX: dragStart.pX + dx, panY: dragStart.pY + dy }; return next; });
  };

  const handleMouseUp = () => { setDragAxis(null); setDragStart(null); };

  const handleResetAll = () => {
    setZooms([DEFAULT_ZOOM, DEFAULT_ZOOM, DEFAULT_ZOOM]);
    setFftZooms([DEFAULT_ZOOM, DEFAULT_ZOOM, DEFAULT_ZOOM]);
    setCamera(DEFAULT_CAMERA);
    setVolumeOpacity(0.5);
    setVolumeBrightness(1.0);
    setLoopStarts([0, 0, 0]);
    setLoopEnds([-1, -1, -1]);
  };

  // -------------------------------------------------------------------------
  // FFT Zoom/Pan handlers
  // -------------------------------------------------------------------------
  const handleFftWheel = (e: React.WheelEvent, axis: number) => {
    const canvas = fftCanvasRefs.current[axis];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const zs = fftZooms[axis];
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const cx = canvas.width / 2, cy = canvas.height / 2;
    const imgX = (mouseX - cx - zs.panX) / zs.zoom + cx;
    const imgY = (mouseY - cy - zs.panY) / zs.zoom + cy;
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zs.zoom * factor));
    const newPanX = mouseX - (imgX - cx) * newZoom - cx;
    const newPanY = mouseY - (imgY - cy) * newZoom - cy;
    setFftZooms(prev => { const next = [...prev]; next[axis] = { zoom: newZoom, panX: newPanX, panY: newPanY }; return next; });
  };

  const handleFftDoubleClick = (axis: number) => {
    setFftZooms(prev => { const next = [...prev]; next[axis] = DEFAULT_ZOOM; return next; });
  };

  const handleFftMouseDown = (e: React.MouseEvent, axis: number) => {
    const zs = fftZooms[axis];
    setFftDragAxis(axis);
    setFftDragStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
  };

  const handleFftMouseMove = (e: React.MouseEvent, axis: number) => {
    if (fftDragAxis !== axis || !fftDragStart) return;
    const canvas = fftCanvasRefs.current[axis];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dx = (e.clientX - fftDragStart.x) * (canvas.width / rect.width);
    const dy = (e.clientY - fftDragStart.y) * (canvas.height / rect.height);
    setFftZooms(prev => { const next = [...prev]; next[axis] = { ...prev[axis], panX: fftDragStart.pX + dx, panY: fftDragStart.pY + dy }; return next; });
  };

  const handleFftMouseUp = () => { setFftDragAxis(null); setFftDragStart(null); };

  const handleFftResetAll = () => { setFftZooms([DEFAULT_ZOOM, DEFAULT_ZOOM, DEFAULT_ZOOM]); };

  const fftNeedsReset = fftZooms.some(z => z.zoom !== 1 || z.panX !== 0 || z.panY !== 0);

  // -------------------------------------------------------------------------
  // Canvas resize (matching Show2D)
  // -------------------------------------------------------------------------
  const handleResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizing(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: canvasTarget });
  };

  React.useEffect(() => {
    if (!isResizing) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeStart) return;
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);
      const newSize = Math.max(300, Math.min(600, resizeStart.size + delta));
      setCanvasTarget(newSize);
    };
    const handleMouseUp = () => {
      setIsResizing(false);
      setResizeStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing, resizeStart]);

  // -------------------------------------------------------------------------
  // Labels and setters
  // -------------------------------------------------------------------------
  const dl = dimLabels || ["Z", "Y", "X"];
  const axisLabels = [
    `${dl[1]}${dl[2]} (${dl[0]}=${sliceZ})`,
    `${dl[0]}${dl[2]} (${dl[1]}=${sliceY})`,
    `${dl[0]}${dl[1]} (${dl[2]}=${sliceX})`,
  ];
  const sliceValues = [sliceZ, sliceY, sliceX];
  const sliceValuesRef = React.useRef(sliceValues);
  sliceValuesRef.current = sliceValues;
  const sliceMaxes = [nz - 1, ny - 1, nx - 1];
  const sliceSetters = [
    (_: Event, v: number | number[]) => setSliceZ(v as number),
    (_: Event, v: number | number[]) => setSliceY(v as number),
    (_: Event, v: number | number[]) => setSliceX(v as number),
  ];

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <Box className="show3dvolume-root" sx={{ ...container.root, bgcolor: tc.bg, color: tc.text }}>
      {/* 3D Volume Renderer */}
      <Box sx={{ mb: `${SPACING.LG}px` }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
          <Typography variant="caption" sx={{ ...typography.label }}>
            {title || "Volume 3D"}
          </Typography>
          <Stack direction="row" alignItems="center" spacing={0.5}>
            <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>FFT:</Typography>
            <Switch checked={showFft} onChange={(e) => setShowFft(e.target.checked)} size="small" sx={switchStyles.small} />
            <Button size="small" sx={compactButton} disabled={!cameraChanged} onClick={handleVolumeDoubleClick}>Reset View</Button>
          </Stack>
        </Stack>
        {webglSupported ? (
          <Box
            sx={{
              ...container.imageBox,
              width: volumeCanvasSize,
              height: volumeCanvasSize,
              cursor: volumeDrag ? "grabbing" : "grab",
            }}
            onMouseDown={handleVolumeMouseDown}
            onMouseMove={handleVolumeMouseMove}
            onMouseUp={handleVolumeMouseUp}
            onMouseLeave={handleVolumeMouseUp}
            onWheel={handleVolumeWheel}
            onDoubleClick={handleVolumeDoubleClick}
            onContextMenu={(e) => e.preventDefault()}
          >
            <canvas
              ref={volumeCanvasRef}
              width={volumeCanvasSize}
              height={volumeCanvasSize}
              style={{ width: volumeCanvasSize, height: volumeCanvasSize, display: "block" }}
            />
            {/* Resize handle */}
            <Box
              onMouseDown={handleVolumeResizeStart}
              sx={{
                position: "absolute", bottom: 2, right: 2, width: 12, height: 12,
                cursor: "nwse-resize", opacity: 0.4,
                background: `linear-gradient(135deg, transparent 50%, ${tc.textMuted} 50%)`,
                "&:hover": { opacity: 1 },
              }}
            />
          </Box>
        ) : (
          <Box sx={{
            ...container.imageBox, width: volumeCanvasSize, height: 80,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <Typography sx={{ ...typography.label, color: tc.textMuted, px: 2, textAlign: "center" }}>
              WebGL 2 not available. 3D volume rendering requires a WebGL 2 capable browser.
            </Typography>
          </Box>
        )}
        {/* Volume rendering controls */}
        {webglSupported && (
          <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg }}>
            <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Opacity:</Typography>
            <Slider
              value={volumeOpacity} min={0} max={1} step={0.01}
              onChange={(_, v) => setVolumeOpacity(v as number)}
              size="small" sx={{ ...sliderStyles.small, width: 60 }}
            />
            <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 24 }}>{volumeOpacity.toFixed(2)}</Typography>
            <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Bright:</Typography>
            <Slider
              value={volumeBrightness} min={0.1} max={3} step={0.1}
              onChange={(_, v) => setVolumeBrightness(v as number)}
              size="small" sx={{ ...sliderStyles.small, width: 60 }}
            />
            <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 24 }}>{volumeBrightness.toFixed(1)}</Typography>
            <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Planes:</Typography>
            <Switch checked={showSlicePlanes} onChange={(e) => setShowSlicePlanes(e.target.checked)} size="small" sx={switchStyles.small} />
          </Box>
        )}
      </Box>
      {/* Slice canvases row */}
      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        {AXES.map((_, a) => {
          const { w: cw, h: ch } = canvasSizes[a];
          return (
            <Box key={a}>
              {/* Header row matching Show3D */}
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
                <Typography variant="caption" sx={{ ...typography.label }}>{a === 0 && title ? title : axisLabels[a]}</Typography>
                {a === 0 && (
                  <Button size="small" sx={compactButton} disabled={!needsReset} onClick={handleResetAll}>Reset</Button>
                )}
              </Stack>
              {/* Canvas with plane-colored border */}
              <Box
                sx={{ ...container.imageBox, width: cw, height: ch, cursor: "grab", borderColor: ["#4d80ff", "#4dff66", "#ff4d4d"][a] }}
                onMouseDown={(e) => handleMouseDown(e, a)}
                onMouseMove={(e) => handleMouseMove(e, a)}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                onWheel={(e) => handleWheel(e, a)}
                onDoubleClick={() => handleDoubleClick(a)}
              >
                <canvas
                  ref={(el) => { canvasRefs.current[a] = el; }}
                  width={cw}
                  height={ch}
                  style={{ width: cw, height: ch, imageRendering: "pixelated" }}
                />
                <canvas
                  ref={(el) => { overlayRefs.current[a] = el; }}
                  width={cw}
                  height={ch}
                  style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}
                />
                {/* Resize handle */}
                <Box
                  onMouseDown={handleResizeStart}
                  sx={{
                    position: "absolute", bottom: 2, right: 2, width: 12, height: 12,
                    cursor: "nwse-resize", opacity: 0.4,
                    background: `linear-gradient(135deg, transparent 50%, ${tc.textMuted} 50%)`,
                    "&:hover": { opacity: 1 },
                  }}
                />
              </Box>
              {/* Stats bar */}
              {showStats && (
                <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: tc.bgAlt, display: "flex", gap: 2 }}>
                  {[
                    { label: "Mean", value: statsMean?.[a] },
                    { label: "Min", value: statsMin?.[a] },
                    { label: "Max", value: statsMax?.[a] },
                    { label: "Std", value: statsStd?.[a] },
                  ].map(({ label, value }) => (
                    <Typography key={label} sx={{ fontSize: 11, color: tc.textMuted }}>
                      {label} <Box component="span" sx={{ color: tc.accent }}>{value !== undefined ? formatNumber(value) : "-"}</Box>
                    </Typography>
                  ))}
                </Box>
              )}
              {/* FFT canvas (inline, below stats) */}
              {showFft && (
                <Box sx={{ mt: `${SPACING.SM}px` }}>
                  <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 20 }}>
                    <Typography variant="caption" sx={{ ...typography.label, fontSize: 10 }}>
                      FFT {["XY", "XZ", "YZ"][a]} {gpuReady ? "(GPU)" : "(CPU)"}
                    </Typography>
                    {a === 0 && fftNeedsReset && (
                      <Button size="small" sx={compactButton} onClick={handleFftResetAll}>Reset</Button>
                    )}
                  </Stack>
                  <Box
                    sx={{ ...container.imageBox, width: cw, height: ch, cursor: "grab", borderColor: ["#4d80ff", "#4dff66", "#ff4d4d"][a] }}
                    onMouseDown={(e) => handleFftMouseDown(e, a)}
                    onMouseMove={(e) => handleFftMouseMove(e, a)}
                    onMouseUp={handleFftMouseUp}
                    onMouseLeave={handleFftMouseUp}
                    onWheel={(e) => handleFftWheel(e, a)}
                    onDoubleClick={() => handleFftDoubleClick(a)}
                  >
                    <canvas
                      ref={(el) => { fftCanvasRefs.current[a] = el; }}
                      width={cw}
                      height={ch}
                      style={{ width: cw, height: ch, imageRendering: "pixelated" }}
                    />
                  </Box>
                  {fftZooms[a].zoom !== 1 && (
                    <Typography sx={{ ...typography.label, fontSize: 10, color: tc.accent, fontWeight: "bold", mt: 0.25, textAlign: "right" }}>
                      {fftZooms[a].zoom.toFixed(1)}x
                    </Typography>
                  )}
                </Box>
              )}
              {/* Slider row (always at bottom) — 3-thumb when loop on, single when off */}
              <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg, width: cw, maxWidth: cw, boxSizing: "border-box" }}>
                <Typography sx={{ ...typography.labelSmall, color: tc.textMuted, flexShrink: 0 }}>{dl[a]}</Typography>
                {loop ? (
                  <Slider
                    value={[loopStarts[a], sliceValues[a], effectiveLoopEnds[a]]}
                    onChange={(_, v) => {
                      const vals = v as number[];
                      setLoopStarts(prev => { const next = [...prev]; next[a] = vals[0]; return next; });
                      [setSliceZ, setSliceY, setSliceX][a](vals[1]);
                      setLoopEnds(prev => { const next = [...prev]; next[a] = vals[2]; return next; });
                    }}
                    disableSwap
                    min={0}
                    max={sliceMaxes[a]}
                    size="small"
                    valueLabelDisplay="auto"
                    valueLabelFormat={(v) => `${v}`}
                    sx={{
                      ...sliderStyles.small,
                      flex: 1,
                      minWidth: 40,
                      "& .MuiSlider-thumb[data-index='0']": { width: 8, height: 8, bgcolor: tc.textMuted },
                      "& .MuiSlider-thumb[data-index='1']": { width: 12, height: 12 },
                      "& .MuiSlider-thumb[data-index='2']": { width: 8, height: 8, bgcolor: tc.textMuted },
                      "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
                    }}
                  />
                ) : (
                  <Slider
                    value={sliceValues[a]}
                    min={0}
                    max={sliceMaxes[a]}
                    onChange={sliceSetters[a]}
                    size="small"
                    sx={{ ...sliderStyles.small, flex: 1, minWidth: 40 }}
                  />
                )}
                <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 28, textAlign: "right", flexShrink: 0 }}>
                  {sliceValues[a]}/{sliceMaxes[a]}
                </Typography>
                {zooms[a].zoom !== 1 && (
                  <Typography sx={{ ...typography.label, fontSize: 10, color: tc.accent, fontWeight: "bold" }}>{zooms[a].zoom.toFixed(1)}x</Typography>
                )}
              </Box>
            </Box>
          );
        })}
      </Stack>
      {/* FFT controls row */}
      {showFft && (
        <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg }}>
          <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>FFT Scale:</Typography>
          <Select value={fftLogScale ? "log" : "linear"} onChange={(e) => setFftLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={upwardMenuProps}>
            <MenuItem value="linear">Lin</MenuItem>
            <MenuItem value="log">Log</MenuItem>
          </Select>
          <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Color:</Typography>
          <Select value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={upwardMenuProps}>
            {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
          </Select>
          <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Auto:</Typography>
          <Switch checked={fftAuto} onChange={(e) => setFftAuto(e.target.checked)} size="small" sx={switchStyles.small} />
        </Box>
      )}
      {/* Controls row */}
      {showControls && (
        <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg }}>
          <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Scale:</Typography>
          <Select value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={upwardMenuProps}>
            <MenuItem value="linear">Lin</MenuItem>
            <MenuItem value="log">Log</MenuItem>
          </Select>
          <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Color:</Typography>
          <Select size="small" value={cmap} onChange={(e) => setCmap(e.target.value)} MenuProps={upwardMenuProps} sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }}>
            {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
          </Select>
          <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Auto:</Typography>
          <Switch checked={autoContrast} onChange={(e) => setAutoContrast(e.target.checked)} size="small" sx={switchStyles.small} />
          <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Cross:</Typography>
          <Switch checked={showCrosshair} onChange={(e) => setShowCrosshair(e.target.checked)} size="small" sx={switchStyles.small} />
        </Box>
      )}
      {/* Playback: transport + axis selector + fps + loop + bounce */}
      <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg }}>
        <Select
          value={playAxis}
          onChange={(e) => { setPlaying(false); setPlayAxis(e.target.value as number); }}
          size="small"
          sx={{ ...themedSelect, minWidth: 40, fontSize: 10 }}
          MenuProps={upwardMenuProps}
        >
          <MenuItem value={0}>{dl[0]}</MenuItem>
          <MenuItem value={1}>{dl[1]}</MenuItem>
          <MenuItem value={2}>{dl[2]}</MenuItem>
          <MenuItem value={3}>All</MenuItem>
        </Select>
        <Stack direction="row" spacing={0} sx={{ flexShrink: 0 }}>
          <IconButton size="small" onClick={() => { setReverse(true); setPlaying(true); }} sx={{ color: reverse && playing ? tc.accent : tc.textMuted, p: 0.25 }}>
            <FastRewindIcon sx={{ fontSize: 18 }} />
          </IconButton>
          <IconButton size="small" onClick={() => setPlaying(!playing)} sx={{ color: tc.accent, p: 0.25 }}>
            {playing ? <PauseIcon sx={{ fontSize: 18 }} /> : <PlayArrowIcon sx={{ fontSize: 18 }} />}
          </IconButton>
          <IconButton size="small" onClick={() => { setReverse(false); setPlaying(true); }} sx={{ color: !reverse && playing ? tc.accent : tc.textMuted, p: 0.25 }}>
            <FastForwardIcon sx={{ fontSize: 18 }} />
          </IconButton>
          <IconButton size="small" onClick={() => {
            setPlaying(false);
            if (playAxis === 3) {
              for (let a = 0; a < 3; a++) sliceSettersRef.current[a](loopStarts[a]);
            } else {
              sliceSettersRef.current[playAxis](loopStarts[playAxis]);
            }
          }} sx={{ color: tc.textMuted, p: 0.25 }}>
            <StopIcon sx={{ fontSize: 16 }} />
          </IconButton>
        </Stack>
        <Typography sx={{ ...typography.label, color: tc.textMuted, flexShrink: 0 }}>fps</Typography>
        <Slider value={fps} min={1} max={30} step={1} onChange={(_, v) => setFps(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35, flexShrink: 0 }} />
        <Typography sx={{ ...typography.label, color: tc.textMuted, minWidth: 14, flexShrink: 0 }}>{Math.round(fps)}</Typography>
        <Typography sx={{ ...typography.label, color: tc.textMuted, flexShrink: 0 }}>Loop</Typography>
        <Switch size="small" checked={loop} onChange={() => setLoop(!loop)} sx={{ ...switchStyles.small, flexShrink: 0 }} />
        <Typography sx={{ ...typography.label, color: tc.textMuted, flexShrink: 0 }}>Bounce</Typography>
        <Switch size="small" checked={boomerang} onChange={() => setBoomerang(!boomerang)} sx={{ ...switchStyles.small, flexShrink: 0 }} />
      </Box>
    </Box>
  );
}

export const render = createRender(Show3DVolume);
