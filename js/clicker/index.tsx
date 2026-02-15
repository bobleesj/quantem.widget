import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Select, { type SelectChangeEvent } from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Tooltip from "@mui/material/Tooltip";
import "./clicker.css";
import { useTheme } from "../theme";
import { drawScaleBarHiDPI } from "../scalebar";
import { extractFloat32, formatNumber, downloadBlob } from "../format";
import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen } from "../colormaps";
import { computeHistogramFromBytes } from "../histogram";
import { findDataRange, computeStats, percentileClip, sliderRange, applyLogScale } from "../stats";
import { fft2d, fftshift, computeMagnitude, autoEnhanceFFT, getWebGPUFFT, type WebGPUFFT } from "../webgpu-fft";
import JSZip from "jszip";

type MarkerShape = "circle" | "triangle" | "square" | "diamond" | "star";
type Point = { row: number; col: number; shape: MarkerShape; color: string };
type ZoomState = { zoom: number; panX: number; panY: number };

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const DRAG_THRESHOLD = 3;
const DEFAULT_ZOOM: ZoomState = { zoom: 1, panX: 0, panY: 0 };
const CANVAS_TARGET_SIZE = 600;
const GALLERY_TARGET_SIZE = 300;
const DPR = window.devicePixelRatio || 1;

const SPACING = {
  XS: 4,
  SM: 8,
  MD: 12,
  LG: 16,
};

const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" as const },
};

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
  minWidth: 0,
  px: 1,
  py: 0.25,
  "&.Mui-disabled": {
    color: "#666",
    borderColor: "#444",
  },
};

const sliderStyles = {
  small: {
    "& .MuiSlider-thumb": { width: 12, height: 12 },
    "& .MuiSlider-rail": { height: 3 },
    "& .MuiSlider-track": { height: 3 },
  },
};

const switchStyles = {
  small: { '& .MuiSwitch-thumb': { width: 12, height: 12 }, '& .MuiSwitch-switchBase': { padding: '4px' } },
};

const containerStyles = {
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "monospace", overflow: "visible" },
  imageBox: { bgcolor: "#000", border: "1px solid #444", overflow: "hidden", position: "relative" as const },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
};

const MARKER_COLORS = [
  "#f44336", // red
  "#4caf50", // green
  "#2196f3", // blue
  "#ff9800", // orange
  "#9c27b0", // purple
  "#00bcd4", // cyan
  "#ffeb3b", // yellow
  "#e91e63", // pink
  "#8bc34a", // lime
  "#ff5722", // deep orange
];

const MARKER_SHAPES: MarkerShape[] = ["circle", "triangle", "square", "diamond", "star"];

const ROI_SHAPES = ["circle", "square", "rectangle"] as const;
type RoiShape = typeof ROI_SHAPES[number];

type ROI = {
  id: number;
  mode: RoiShape;
  row: number;
  col: number;
  radius: number;
  rectW: number;
  rectH: number;
  color: string;
  opacity: number;
};

const ROI_COLORS = ["#0f0", "#ff0", "#0af", "#f0f", "#f80", "#f44"];

type RoiStats = { mean: number; std: number; min: number; max: number; count: number };

function computeRoiStats(roi: ROI, data: Float32Array, width: number, height: number): RoiStats | null {
  if (!data || data.length === 0) return null;
  let sum = 0, sumSq = 0, min = Infinity, max = -Infinity, count = 0;
  let x0: number, y0: number, x1: number, y1: number;
  if (roi.mode === "rectangle") {
    x0 = Math.max(0, Math.floor(roi.col - roi.rectW / 2));
    y0 = Math.max(0, Math.floor(roi.row - roi.rectH / 2));
    x1 = Math.min(width - 1, Math.ceil(roi.col + roi.rectW / 2));
    y1 = Math.min(height - 1, Math.ceil(roi.row + roi.rectH / 2));
  } else {
    x0 = Math.max(0, Math.floor(roi.col - roi.radius));
    y0 = Math.max(0, Math.floor(roi.row - roi.radius));
    x1 = Math.min(width - 1, Math.ceil(roi.col + roi.radius));
    y1 = Math.min(height - 1, Math.ceil(roi.row + roi.radius));
  }
  const r2 = roi.radius * roi.radius;
  for (let py = y0; py <= y1; py++) {
    for (let px = x0; px <= x1; px++) {
      if (roi.mode === "circle") {
        const dx = px - roi.col, dy = py - roi.row;
        if (dx * dx + dy * dy > r2) continue;
      }
      const val = data[py * width + px];
      sum += val; sumSq += val * val;
      if (val < min) min = val;
      if (val > max) max = val;
      count++;
    }
  }
  if (count === 0) return null;
  const mean = sum / count;
  const std = Math.sqrt(Math.max(0, sumSq / count - mean * mean));
  return { mean, std, min, max, count };
}

function findLocalMax(data: Float32Array, width: number, height: number, cc: number, cr: number, radius: number): { row: number; col: number } {
  let bestCol = cc, bestRow = cr, bestVal = -Infinity;
  const c0 = Math.max(0, cc - radius), r0 = Math.max(0, cr - radius);
  const c1 = Math.min(width - 1, cc + radius), r1 = Math.min(height - 1, cr + radius);
  for (let ir = r0; ir <= r1; ir++) {
    for (let ic = c0; ic <= c1; ic++) {
      const val = data[ir * width + ic];
      if (val > bestVal) { bestVal = val; bestCol = ic; bestRow = ir; }
    }
  }
  return { row: bestRow, col: bestCol };
}

function formatDistance(p1: Point, p2: Point, pixelSize: number): string {
  const dx = p2.col - p1.col, dy = p2.row - p1.row;
  const distPx = Math.sqrt(dx * dx + dy * dy);
  if (pixelSize > 0) {
    const distAng = distPx * pixelSize;
    return distAng >= 10 ? `${(distAng / 10).toFixed(2)} nm` : `${distAng.toFixed(2)} \u00C5`;
  }
  return `${distPx.toFixed(1)} px`;
}

function brightenColor(hex: string, amount: number): string {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  if (!m) return hex;
  const r = Math.min(255, parseInt(m[1], 16) + amount);
  const g = Math.min(255, parseInt(m[2], 16) + amount);
  const b = Math.min(255, parseInt(m[3], 16) + amount);
  return `rgb(${r},${g},${b})`;
}

function drawROI(
  ctx: CanvasRenderingContext2D,
  x: number, y: number,
  shape: RoiShape,
  radius: number, w: number, h: number,
  color: string, opacity: number,
  isActive: boolean,
  isHovered: boolean,
): void {
  ctx.save();
  const highlighted = isActive || isHovered;
  ctx.globalAlpha = highlighted ? Math.min(1, opacity + 0.2) : opacity;
  ctx.strokeStyle = highlighted ? brightenColor(color, 80) : color;
  ctx.lineWidth = isActive ? 3 : isHovered ? 2.5 : 2;
  if (highlighted) {
    ctx.shadowColor = brightenColor(color, 120);
    ctx.shadowBlur = isActive ? 8 : 5;
  }
  if (shape === "circle") {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
  } else if (shape === "square") {
    ctx.strokeRect(x - radius, y - radius, radius * 2, radius * 2);
  } else if (shape === "rectangle") {
    ctx.strokeRect(x - w / 2, y - h / 2, w, h);
  }
  if (isActive) {
    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.moveTo(x - 5, y);
    ctx.lineTo(x + 5, y);
    ctx.moveTo(x, y - 5);
    ctx.lineTo(x, y + 5);
    ctx.stroke();
  }
  ctx.restore();
}

function drawMarker(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, shape: MarkerShape, fillColor: string, strokeColor: string, opacity: number, strokeWidth: number) {
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.beginPath();
  switch (shape) {
    case "circle":
      ctx.arc(x, y, r, 0, Math.PI * 2);
      break;
    case "triangle":
      ctx.moveTo(x, y - r);
      ctx.lineTo(x + r * 0.87, y + r * 0.5);
      ctx.lineTo(x - r * 0.87, y + r * 0.5);
      ctx.closePath();
      break;
    case "square":
      ctx.rect(x - r * 0.75, y - r * 0.75, r * 1.5, r * 1.5);
      break;
    case "diamond":
      ctx.moveTo(x, y - r);
      ctx.lineTo(x + r * 0.7, y);
      ctx.lineTo(x, y + r);
      ctx.lineTo(x - r * 0.7, y);
      ctx.closePath();
      break;
    case "star": {
      const spikes = 5;
      const outerR = r;
      const innerR = r * 0.4;
      for (let s = 0; s < spikes * 2; s++) {
        const rad = (s * Math.PI) / spikes - Math.PI / 2;
        const sr = s % 2 === 0 ? outerR : innerR;
        if (s === 0) ctx.moveTo(x + sr * Math.cos(rad), y + sr * Math.sin(rad));
        else ctx.lineTo(x + sr * Math.cos(rad), y + sr * Math.sin(rad));
      }
      ctx.closePath();
      break;
    }
  }
  ctx.fillStyle = fillColor;
  ctx.fill();
  if (strokeWidth > 0) {
    ctx.clip();
    ctx.lineWidth = strokeWidth * 2;
    ctx.strokeStyle = strokeColor;
    ctx.stroke();
  }
  ctx.restore();
}

function ShapeIcon({ shape, color, size }: { shape: MarkerShape; color: string; size: number }) {
  const h = size / 2;
  const r = h * 0.8;
  let path: React.ReactNode;
  switch (shape) {
    case "circle": path = <circle cx={h} cy={h} r={r} />; break;
    case "triangle": path = <polygon points={`${h},${h - r} ${h + r * 0.87},${h + r * 0.5} ${h - r * 0.87},${h + r * 0.5}`} />; break;
    case "square": path = <rect x={h - r * 0.75} y={h - r * 0.75} width={r * 1.5} height={r * 1.5} />; break;
    case "diamond": path = <polygon points={`${h},${h - r} ${h + r * 0.7},${h} ${h},${h + r} ${h - r * 0.7},${h}`} />; break;
    case "star": {
      const pts: string[] = [];
      for (let i = 0; i < 10; i++) {
        const a = (i * Math.PI) / 5 - Math.PI / 2;
        const sr = i % 2 === 0 ? r : r * 0.4;
        pts.push(`${h + sr * Math.cos(a)},${h + sr * Math.sin(a)}`);
      }
      path = <polygon points={pts.join(" ")} />;
      break;
    }
  }
  return (
    <svg width={size} height={size} style={{ display: "block", flexShrink: 0 }}>
      <g fill={color}>{path}</g>
    </svg>
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
          sx: { bgcolor: isDark ? "#333" : "#fff", color: isDark ? "#ddd" : "#333", border: `1px solid ${isDark ? "#555" : "#ccc"}`, maxWidth: 280, p: 1 },
        },
        arrow: {
          sx: { color: isDark ? "#333" : "#fff", "&::before": { border: `1px solid ${isDark ? "#555" : "#ccc"}` } },
        },
      }}
    >
      <Typography
        component="span"
        sx={{ fontSize: 12, color: isDark ? "#888" : "#666", cursor: "help", ml: 0.5, "&:hover": { color: isDark ? "#aaa" : "#444" } }}
      >
        ⓘ
      </Typography>
    </Tooltip>
  );
}

function HistogramWidget({
  data,
  vminPct,
  vmaxPct,
  onRangeChange,
  width = 110,
  height = 40,
  theme = "dark",
  dataMin = 0,
  dataMax = 1,
}: {
  data: Float32Array | null;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
  dataMin?: number;
  dataMax?: number;
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const bins = React.useMemo(() => data ? computeHistogramFromBytes(data) : new Array(256).fill(0), [data]);
  const colors = theme === "dark" ? {
    bg: "#1a1a1a", barActive: "#888", barInactive: "#444", border: "#333",
  } : {
    bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc",
  };

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
    const reduced: number[] = [];
    for (let i = 0; i < displayBins; i++) {
      let sum = 0;
      for (let j = 0; j < binRatio; j++) sum += bins[i * binRatio + j] || 0;
      reduced.push(sum / binRatio);
    }
    const maxVal = Math.max(...reduced, 0.001);
    const barWidth = width / displayBins;
    const vminBin = Math.floor((vminPct / 100) * displayBins);
    const vmaxBin = Math.floor((vmaxPct / 100) * displayBins);
    for (let i = 0; i < displayBins; i++) {
      const barH = (reduced[i] / maxVal) * (height - 2);
      ctx.fillStyle = i >= vminBin && i <= vmaxBin ? colors.barActive : colors.barInactive;
      ctx.fillRect(i * barWidth + 0.5, height - barH, Math.max(1, barWidth - 1), barH);
    }
  }, [bins, vminPct, vmaxPct, width, height, colors]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas ref={canvasRef} style={{ width, height, border: `1px solid ${colors.border}` }} />
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

const render = createRender(() => {
  const { themeInfo, colors: tc } = useTheme();
  const accentGreen = themeInfo.theme === "dark" ? "#0f0" : "#0a0";

  const themedSelect = {
    minWidth: 50, fontSize: 11, bgcolor: tc.controlBg, color: tc.text,
    "& .MuiSelect-select": { py: 0.5, px: 1 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: tc.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: tc.accent },
  };
  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: { sx: { bgcolor: tc.controlBg, color: tc.text, border: `1px solid ${tc.border}` } },
  };

  // Model state
  const [nImages] = useModelState<number>("n_images");
  const [width] = useModelState<number>("width");
  const [height] = useModelState<number>("height");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [imgMin] = useModelState<number[]>("img_min");
  const [imgMax] = useModelState<number[]>("img_max");
  const [selectedIdx, setSelectedIdx] = useModelState<number>("selected_idx");
  const [ncols] = useModelState<number>("ncols");
  const [labels] = useModelState<string[]>("labels");
  const [scale] = useModelState<number>("scale");
  const [selectedPoints, setSelectedPoints] = useModelState<Point[] | Point[][]>("selected_points");
  const [dotSize, setDotSize] = useModelState<number>("dot_size");
  const [maxPoints, setMaxPoints] = useModelState<number>("max_points");
  const [pixelSizeAngstrom] = useModelState<number>("pixel_size_angstrom");
  const [title] = useModelState<string>("title");
  const [showStats] = useModelState<boolean>("show_stats");

  const isGallery = nImages > 1;

  // Marker styling (Python-configurable)
  const [borderWidth, setBorderWidth] = useModelState<number>("marker_border");
  const [markerOpacity, setMarkerOpacity] = useModelState<number>("marker_opacity");
  const [labelSize, setLabelSize] = useModelState<number>("label_size");
  const [labelColor, setLabelColor] = useModelState<string>("label_color");

  // Current marker style (synced to Python for state portability)
  const [currentShape, setCurrentShape] = useModelState<string>("marker_shape");
  const [currentColor, setCurrentColor] = useModelState<string>("marker_color");

  // ROI state (synced to Python via roi_list trait)
  const [rois, setRois] = useModelState<ROI[]>("roi_list");
  const [activeRoiIdx, setActiveRoiIdx] = React.useState(-1);
  const [hoveredRoiIdx, setHoveredRoiIdx] = React.useState(-1);
  const [isDraggingROI, setIsDraggingROI] = React.useState(false);
  const [newRoiShape, setNewRoiShape] = React.useState<RoiShape>("circle");
  const safeRois = rois || [];
  const roiActive = safeRois.length > 0;
  const activeRoi = activeRoiIdx >= 0 && activeRoiIdx < safeRois.length ? safeRois[activeRoiIdx] : null;

  const pushRoiHistory = React.useCallback(() => {
    roiHistoryRef.current = [...roiHistoryRef.current.slice(-49), safeRois.map(r => ({ ...r }))];
    roiRedoRef.current = [];
  }, [safeRois]);

  const updateActiveRoi = React.useCallback((updates: Partial<ROI>) => {
    pushRoiHistory();
    setRois(safeRois.map((r, i) => i === activeRoiIdx ? { ...r, ...updates } : r));
  }, [activeRoiIdx, safeRois, setRois, pushRoiHistory]);

  const undoRoi = React.useCallback(() => {
    if (roiHistoryRef.current.length === 0) return false;
    const prev = roiHistoryRef.current.pop()!;
    roiRedoRef.current.push(safeRois.map(r => ({ ...r })));
    setRois(prev);
    return true;
  }, [safeRois, setRois]);

  const redoRoi = React.useCallback(() => {
    if (roiRedoRef.current.length === 0) return false;
    const next = roiRedoRef.current.pop()!;
    roiHistoryRef.current.push(safeRois.map(r => ({ ...r })));
    setRois(next);
    return true;
  }, [safeRois, setRois]);

  // Snap-to-peak (synced to Python for state portability)
  const [snapEnabled, setSnapEnabled] = useModelState<boolean>("snap_enabled");
  const [snapRadius, setSnapRadius] = useModelState<number>("snap_radius");

  // Colormap, contrast, FFT (synced to Python)
  const [colormap, setColormap] = useModelState<string>("colormap");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");

  // Histogram slider state (local)
  const [vminPct, setVminPct] = React.useState(0);
  const [vmaxPct, setVmaxPct] = React.useState(100);

  // Point dragging state
  const draggingPointRef = React.useRef<{ idx: number; imageIdx: number } | null>(null);
  const [hoveredPointIdx, setHoveredPointIdx] = React.useState(-1);

  // ROI undo/redo history
  const roiHistoryRef = React.useRef<ROI[][]>([]);
  const roiRedoRef = React.useRef<ROI[][]>([]);

  // FFT refs
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const fftOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const fftCanvasRef = React.useRef<HTMLCanvasElement | null>(null);

  // Collapsible advanced options
  const [showAdvanced, setShowAdvanced] = React.useState(false);

  // Refs
  const canvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const overlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const uiRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const offscreenRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const canvasContainerRefs = React.useRef<(HTMLDivElement | null)[]>([]);

  const [hover, setHover] = React.useState<{
    row: number;
    col: number;
    raw?: number;
    norm?: number;
  } | null>(null);

  // Per-image zoom state
  const [zoomStates, setZoomStates] = React.useState<Map<number, ZoomState>>(new Map());
  const getZoom = React.useCallback((idx: number): ZoomState => zoomStates.get(idx) || DEFAULT_ZOOM, [zoomStates]);
  const setZoom = React.useCallback((idx: number, zs: ZoomState) => {
    setZoomStates(prev => new Map(prev).set(idx, zs));
  }, []);

  const dragRef = React.useRef<{
    startX: number;
    startY: number;
    startPanX: number;
    startPanY: number;
    dragging: boolean;
    wasDrag: boolean;
    imageIdx: number;
  } | null>(null);

  // Resize state
  const [mainCanvasSize, setMainCanvasSize] = React.useState(CANVAS_TARGET_SIZE);
  const [galleryCanvasSize, setGalleryCanvasSize] = React.useState(GALLERY_TARGET_SIZE);
  const [isResizing, setIsResizing] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number; y: number; size: number } | null>(null);
  const initialCanvasSizeRef = React.useRef<number>(CANVAS_TARGET_SIZE);

  // Sync initial size when image loads — never shrink below target
  React.useEffect(() => {
    if (width > 0 && height > 0) {
      const sz = Math.max(CANVAS_TARGET_SIZE, Math.round(Math.max(width, height) * scale));
      if (!isGallery) setMainCanvasSize(sz);
      initialCanvasSizeRef.current = CANVAS_TARGET_SIZE;
    }
  }, [width, height, scale, isGallery]);

  // Compute display dimensions
  const targetSize = isGallery ? galleryCanvasSize : mainCanvasSize;
  const displayScale = width > 0 && height > 0 ? targetSize / Math.max(width, height) : 1;
  const canvasW = width > 0 ? Math.round(width * displayScale) : targetSize;
  const canvasH = height > 0 ? Math.round(height * displayScale) : targetSize;
  const contentW = isGallery
    ? ncols * canvasW + (ncols - 1) * 8
    : showFft ? canvasW * 2 + SPACING.LG : canvasW;

  // Parse frame_bytes into per-image Float32Arrays
  const floatsPerImage = width * height;
  const perImageData = React.useMemo(() => {
    if (!frameBytes || !width || !height) return [];
    const allFloats = extractFloat32(frameBytes);
    if (!allFloats) return [];
    const result: Float32Array[] = [];
    for (let i = 0; i < nImages; i++) {
      const start = i * floatsPerImage;
      result.push(allFloats.subarray(start, start + floatsPerImage));
    }
    return result;
  }, [frameBytes, nImages, floatsPerImage, width, height]);

  // ROI pixel statistics
  const roiStats = React.useMemo(() => {
    if (!activeRoi || perImageData.length === 0) return null;
    const imgIdx = isGallery ? selectedIdx : 0;
    const data = perImageData[imgIdx];
    if (!data) return null;
    return computeRoiStats(activeRoi, data, width, height);
  }, [activeRoi, perImageData, isGallery, selectedIdx, width, height]);

  // Build offscreen canvases from float32 data (colormap + contrast)
  React.useEffect(() => {
    if (perImageData.length === 0 || !width || !height) return;
    const lut = COLORMAPS[colormap || "gray"] || COLORMAPS.gray;
    for (let i = 0; i < nImages; i++) {
      const f32 = perImageData[i];
      if (!f32) continue;
      const data = logScale ? applyLogScale(f32) : f32;
      let vmin: number, vmax: number;
      if (autoContrast) {
        ({ vmin, vmax } = percentileClip(data, 2, 98));
      } else {
        const { min: dMin, max: dMax } = findDataRange(data);
        ({ vmin, vmax } = sliderRange(dMin, dMax, vminPct, vmaxPct));
      }
      const result = renderToOffscreen(data, width, height, lut, vmin, vmax);
      offscreenRefs.current[i] = result;
    }
    offscreenRefs.current.length = nImages;
  }, [perImageData, nImages, width, height, colormap, autoContrast, logScale, vminPct, vmaxPct]);

  // Histogram data for current image
  const histogramData = React.useMemo(() => {
    const idx = isGallery ? selectedIdx : 0;
    const f32 = perImageData[idx];
    if (!f32) return null;
    return logScale ? applyLogScale(f32) : f32;
  }, [perImageData, isGallery, selectedIdx, logScale]);

  const dataRange = React.useMemo(() => {
    if (!histogramData) return { min: 0, max: 1 };
    return findDataRange(histogramData);
  }, [histogramData]);

  // Image statistics (Mean/Min/Max/Std) for the active image
  const imageStats = React.useMemo(() => {
    const idx = isGallery ? selectedIdx : 0;
    const f32 = perImageData[idx];
    if (!f32 || f32.length === 0) return null;
    return computeStats(f32);
  }, [perImageData, isGallery, selectedIdx]);

  // Per-image points helpers
  const getPointsForImage = React.useCallback((idx: number): Point[] => {
    if (!isGallery) return (selectedPoints as Point[]) || [];
    const nested = (selectedPoints as Point[][]) || [];
    return nested[idx] || [];
  }, [isGallery, selectedPoints]);

  const setPointsForImage = React.useCallback((idx: number, points: Point[]) => {
    if (!isGallery) {
      setSelectedPoints(points);
    } else {
      setSelectedPoints((prev) => {
        const nested = [...((prev as Point[][]) || [])];
        while (nested.length < nImages) nested.push([]);
        nested[idx] = points;
        return nested;
      });
    }
  }, [isGallery, nImages, setSelectedPoints]);

  // Dot size
  const size = Number.isFinite(dotSize) && dotSize > 0 ? dotSize : 12;

  // Render all canvases
  React.useEffect(() => {
    if (!width || !height || perImageData.length === 0) return;
    for (let i = 0; i < nImages; i++) {
      const canvas = canvasRefs.current[i];
      const offscreen = offscreenRefs.current[i];
      if (!canvas || !offscreen) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;

      canvas.width = canvasW;
      canvas.height = canvasH;

      const { zoom, panX, panY } = getZoom(i);
      const cx = canvasW / 2;
      const cy = canvasH / 2;

      ctx.clearRect(0, 0, canvasW, canvasH);
      ctx.save();
      ctx.imageSmoothingEnabled = false;

      ctx.translate(cx + panX, cy + panY);
      ctx.scale(zoom, zoom);
      ctx.translate(-cx, -cy);

      ctx.drawImage(offscreen, 0, 0, canvasW, canvasH);

      // Draw points for this image
      const pts = getPointsForImage(i);
      const dotRadius = (size / 2) * displayScale;
      for (let j = 0; j < pts.length; j++) {
        const p = pts[j];
        const px = (p.col / width) * canvasW;
        const py = (p.row / height) * canvasH;
        const color = p.color || MARKER_COLORS[j % MARKER_COLORS.length];
        const shape = p.shape || MARKER_SHAPES[j % MARKER_SHAPES.length];

        drawMarker(ctx, px, py, dotRadius, shape, color, tc.bg, markerOpacity, borderWidth);
      }

      ctx.restore();

    }
  }, [perImageData, width, height, canvasW, canvasH, displayScale, zoomStates, selectedPoints, size, tc.accent, tc.bg, tc.text, nImages, isGallery, selectedIdx, getZoom, getPointsForImage, markerOpacity, borderWidth]);

  // Render ROI overlays + edge tick marks
  React.useEffect(() => {
    if (!width || !height) return;
    const idx = isGallery ? selectedIdx : 0;
    for (let i = 0; i < nImages; i++) {
      const overlay = overlayRefs.current[i];
      if (!overlay) continue;
      const ctx = overlay.getContext("2d");
      if (!ctx) continue;
      overlay.width = canvasW;
      overlay.height = canvasH;
      ctx.clearRect(0, 0, canvasW, canvasH);
      if (safeRois.length > 0 && i === idx) {
        const { zoom, panX, panY } = getZoom(i);
        const cx = canvasW / 2;
        const cy = canvasH / 2;
        for (let ri = 0; ri < safeRois.length; ri++) {
          const roi = safeRois[ri];
          const screenX = ((roi.col / width) * canvasW - cx) * zoom + cx + panX;
          const screenY = ((roi.row / height) * canvasH - cy) * zoom + cy + panY;
          const screenRadius = roi.radius * displayScale * zoom;
          const screenW = roi.rectW * displayScale * zoom;
          const screenH = roi.rectH * displayScale * zoom;
          const isActive = ri === activeRoiIdx;
          const isHovered = ri === hoveredRoiIdx && !isActive;
          drawROI(ctx, screenX, screenY, roi.mode, screenRadius, screenW, screenH, roi.color, roi.opacity, isActive, isHovered);
        }
      }
      // Edge tick marks — short indicators on the 4 canvas edges tracking cursor position
      if (hover && i === idx) {
        const { zoom, panX, panY } = getZoom(i);
        const cx = canvasW / 2;
        const cy = canvasH / 2;
        const imgX = (hover.col / width) * canvasW;
        const imgY = (hover.row / height) * canvasH;
        const screenX = (imgX - cx) * zoom + cx + panX;
        const screenY = (imgY - cy) * zoom + cy + panY;
        const tickLen = 10;
        ctx.save();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
        ctx.lineWidth = 1.5;
        ctx.shadowColor = "rgba(0, 0, 0, 0.6)";
        ctx.shadowBlur = 2;
        ctx.beginPath();
        // Top edge
        ctx.moveTo(screenX, 0);
        ctx.lineTo(screenX, tickLen);
        // Bottom edge
        ctx.moveTo(screenX, canvasH);
        ctx.lineTo(screenX, canvasH - tickLen);
        // Left edge
        ctx.moveTo(0, screenY);
        ctx.lineTo(tickLen, screenY);
        // Right edge
        ctx.moveTo(canvasW, screenY);
        ctx.lineTo(canvasW - tickLen, screenY);
        ctx.stroke();
        ctx.restore();
      }
    }
  }, [safeRois, activeRoiIdx, hoveredRoiIdx, isDraggingROI, canvasW, canvasH, displayScale, width, height, nImages, isGallery, selectedIdx, getZoom, hover]);

  // Scale bar + marker labels (HiDPI UI overlay)
  React.useEffect(() => {
    if (!width || !height) return;
    const pxSize = pixelSizeAngstrom || 0;
    const unit = pxSize > 0 ? "Å" as const : "px" as const;
    const pxSizeVal = pxSize > 0 ? pxSize : 1;
    const dotRadius = (size / 2) * displayScale;
    for (let i = 0; i < nImages; i++) {
      const uiCanvas = uiRefs.current[i];
      if (!uiCanvas) continue;
      uiCanvas.width = Math.round(canvasW * DPR);
      uiCanvas.height = Math.round(canvasH * DPR);
      const { zoom, panX, panY } = getZoom(i);
      drawScaleBarHiDPI(uiCanvas, DPR, zoom, pxSizeVal, unit, width);

      // Draw marker labels on HiDPI UI canvas (pixel-independent)
      const pts = getPointsForImage(i);
      if (pts.length === 0) continue;
      const ctx = uiCanvas.getContext("2d");
      if (!ctx) continue;
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      const fontSize = (labelSize > 0 ? labelSize : Math.max(10, size * 0.9)) * DPR;
      ctx.font = `bold ${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
      ctx.fillStyle = labelColor || tc.text;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.shadowColor = "rgba(0,0,0,0.7)";
      ctx.shadowBlur = 3 * DPR;
      for (let j = 0; j < pts.length; j++) {
        const p = pts[j];
        const imgX = (p.col / width) * canvasW;
        const imgY = (p.row / height) * canvasH;
        const screenX = ((imgX - cx) * zoom + cx + panX) * DPR;
        const screenY = ((imgY - cy) * zoom + cy + panY) * DPR;
        const screenDotR = dotRadius * zoom * DPR;
        ctx.fillText(`${j + 1}`, screenX, screenY - screenDotR - 2 * DPR);
      }
    }
  }, [pixelSizeAngstrom, canvasW, canvasH, width, height, nImages, zoomStates, getZoom, selectedPoints, getPointsForImage, size, displayScale, labelSize, labelColor, tc.text]);

  // Map screen coordinates to image pixel coordinates
  const clientToImage = React.useCallback(
    (clientX: number, clientY: number, idx: number): { row: number; col: number } | null => {
      const canvas = canvasRefs.current[idx];
      if (!canvas || !width || !height) return null;
      const rect = canvas.getBoundingClientRect();
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      const { zoom, panX, panY } = getZoom(idx);

      const canvasX = ((clientX - rect.left) / rect.width) * canvasW;
      const canvasY = ((clientY - rect.top) / rect.height) * canvasH;

      const imgDisplayX = (canvasX - cx - panX) / zoom + cx;
      const imgDisplayY = (canvasY - cy - panY) / zoom + cy;

      const col = Math.floor((imgDisplayX / canvasW) * width);
      const row = Math.floor((imgDisplayY / canvasH) * height);
      if (col < 0 || row < 0 || col >= width || row >= height) return null;
      return { row, col };
    },
    [width, height, canvasW, canvasH, getZoom],
  );

  // Init GPU FFT
  React.useEffect(() => {
    getWebGPUFFT().then(fft => { if (fft) gpuFFTRef.current = fft; });
  }, []);

  // Compute FFT when toggled
  React.useEffect(() => {
    if (!showFft || !width || !height) { fftOffscreenRef.current = null; return; }
    const idx = isGallery ? selectedIdx : 0;
    const f32 = perImageData[idx];
    if (!f32) return;
    const lut = COLORMAPS[colormap || "gray"] || COLORMAPS.gray;
    const compute = async () => {
      let real: Float32Array, imag: Float32Array;
      if (gpuFFTRef.current) {
        const result = await gpuFFTRef.current.fft2D(f32.slice(), new Float32Array(f32.length), width, height, false);
        real = result.real; imag = result.imag;
      } else {
        real = f32.slice(); imag = new Float32Array(f32.length);
        fft2d(real, imag, width, height, false);
      }
      fftshift(real, width, height);
      fftshift(imag, width, height);
      const mag = computeMagnitude(real, imag);
      autoEnhanceFFT(mag, width, height);
      const logMag = applyLogScale(mag);
      const { min: logMin, max: logMax } = findDataRange(logMag);
      fftOffscreenRef.current = renderToOffscreen(logMag, width, height, lut, logMin, logMax);
      // Trigger redraw
      if (fftCanvasRef.current && fftOffscreenRef.current) {
        const ctx = fftCanvasRef.current.getContext("2d");
        if (ctx) {
          fftCanvasRef.current.width = canvasW;
          fftCanvasRef.current.height = canvasH;
          ctx.imageSmoothingEnabled = false;
          ctx.clearRect(0, 0, canvasW, canvasH);
          ctx.drawImage(fftOffscreenRef.current, 0, 0, canvasW, canvasH);
        }
      }
    };
    compute();
  }, [showFft, perImageData, isGallery, selectedIdx, width, height, colormap, canvasW, canvasH]);

  // Prevent page scroll on canvas containers
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const containers = canvasContainerRefs.current.filter(Boolean);
    containers.forEach(el => el?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => {
      containers.forEach(el => el?.removeEventListener("wheel", preventDefault));
    };
  }, [nImages]);

  // Scroll to zoom
  const handleWheel = React.useCallback(
    (e: React.WheelEvent, idx: number) => {
      e.preventDefault();
      if (isGallery && idx !== selectedIdx) return;
      const canvas = canvasRefs.current[idx];
      if (!canvas || !width || !height) return;
      const rect = canvas.getBoundingClientRect();

      const mouseX = ((e.clientX - rect.left) / rect.width) * canvasW;
      const mouseY = ((e.clientY - rect.top) / rect.height) * canvasH;

      const factor = e.deltaY < 0 ? 1.1 : 0.9;
      const prev = getZoom(idx);
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, prev.zoom * factor));
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      const wx = (mouseX - cx - prev.panX) / prev.zoom + cx;
      const wy = (mouseY - cy - prev.panY) / prev.zoom + cy;
      const newPanX = mouseX - cx - (wx - cx) * newZoom;
      const newPanY = mouseY - cy - (wy - cy) * newZoom;
      setZoom(idx, { zoom: newZoom, panX: newPanX, panY: newPanY });
    },
    [width, height, canvasW, canvasH, isGallery, selectedIdx, getZoom, setZoom],
  );

  // Track when we just switched focus (don't place a point on the same click)
  const justSwitchedRef = React.useRef(false);

  // Mouse down
  const handleMouseDown = React.useCallback(
    (e: React.MouseEvent, idx: number) => {
      if (e.button !== 0) return;
      justSwitchedRef.current = false;
      if (isGallery && idx !== selectedIdx) {
        setSelectedIdx(idx);
        justSwitchedRef.current = true;
        return;
      }
      // Check if click is near any ROI center to drag it
      if (safeRois.length > 0) {
        const coords = clientToImage(e.clientX, e.clientY, idx);
        if (coords) {
          const { zoom } = getZoom(idx);
          const hitDist = 20 / (displayScale * zoom);
          let hitIdx = -1;
          let bestDist = Infinity;
          for (let ri = 0; ri < safeRois.length; ri++) {
            const dx = coords.col - safeRois[ri].col;
            const dy = coords.row - safeRois[ri].row;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < hitDist && dist < bestDist) {
              bestDist = dist;
              hitIdx = ri;
            }
          }
          if (hitIdx >= 0) {
            pushRoiHistory();
            setActiveRoiIdx(hitIdx);
            setIsDraggingROI(true);
            return;
          }
        }
      }
      // Check if click is near any existing point to drag it
      {
        const coords = clientToImage(e.clientX, e.clientY, idx);
        if (coords) {
          const pts = getPointsForImage(idx);
          const { zoom } = getZoom(idx);
          const hitDist = 15 / (displayScale * zoom);
          let bestPtIdx = -1;
          let bestDist = Infinity;
          for (let pi = 0; pi < pts.length; pi++) {
            const dx = coords.col - pts[pi].col;
            const dy = coords.row - pts[pi].row;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < hitDist && dist < bestDist) {
              bestDist = dist;
              bestPtIdx = pi;
            }
          }
          if (bestPtIdx >= 0) {
            draggingPointRef.current = { idx: bestPtIdx, imageIdx: idx };
            return;
          }
        }
      }
      const zs = getZoom(idx);
      dragRef.current = {
        startX: e.clientX,
        startY: e.clientY,
        startPanX: zs.panX,
        startPanY: zs.panY,
        dragging: false,
        wasDrag: false,
        imageIdx: idx,
      };
    },
    [isGallery, selectedIdx, setSelectedIdx, getZoom, safeRois, clientToImage, displayScale, getPointsForImage, pushRoiHistory],
  );

  // Mouse move
  const handleMouseMove = React.useCallback(
    (e: React.MouseEvent, idx: number) => {
      // Point dragging
      if (draggingPointRef.current && draggingPointRef.current.imageIdx === idx) {
        const coords = clientToImage(e.clientX, e.clientY, idx);
        if (coords) {
          const pts = getPointsForImage(idx);
          const pi = draggingPointRef.current.idx;
          if (pi < pts.length) {
            const updated = pts.map((p, j) => j === pi ? { ...p, row: coords.row, col: coords.col } : p);
            setPointsForImage(idx, updated);
          }
        }
        return;
      }
      if (isDraggingROI && activeRoiIdx >= 0) {
        const coords = clientToImage(e.clientX, e.clientY, idx);
        if (coords) {
          setRois(safeRois.map((r, i) => i === activeRoiIdx ? { ...r, row: coords.row, col: coords.col } : r));
        }
        return;
      }
      const drag = dragRef.current;
      if (drag && drag.imageIdx === idx) {
        const dx = e.clientX - drag.startX;
        const dy = e.clientY - drag.startY;
        if (!drag.dragging && Math.abs(dx) + Math.abs(dy) > DRAG_THRESHOLD) {
          drag.dragging = true;
        }
        if (drag.dragging) {
          drag.wasDrag = true;
          const canvas = canvasRefs.current[idx];
          if (!canvas) return;
          const rect = canvas.getBoundingClientRect();
          const scaleX = canvasW / rect.width;
          const scaleY = canvasH / rect.height;
          setZoom(idx, {
            zoom: getZoom(idx).zoom,
            panX: drag.startPanX + dx * scaleX,
            panY: drag.startPanY + dy * scaleY,
          });
          return;
        }
      }

      // Hover readout (only for selected image in gallery)
      if (isGallery && idx !== selectedIdx) return;
      const p = clientToImage(e.clientX, e.clientY, idx);
      if (!p) { setHover(null); return; }
      let raw: number | undefined;
      let norm: number | undefined;
      const f32 = perImageData[idx];
      if (f32) {
        raw = f32[p.row * width + p.col];
        const min = imgMin?.[idx] ?? 0;
        const max = imgMax?.[idx] ?? 1;
        const denom = max > min ? max - min : 1;
        norm = (raw - min) / denom;
      }
      setHover({ row: p.row, col: p.col, raw, norm });

      // ROI hover detection
      if (safeRois.length > 0) {
        const { zoom } = getZoom(idx);
        const hitDist = 20 / (displayScale * zoom);
        let hitIdx = -1;
        let bestDist = Infinity;
        for (let ri = 0; ri < safeRois.length; ri++) {
          const dx = p.col - safeRois[ri].col;
          const dy = p.row - safeRois[ri].row;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < hitDist && dist < bestDist) {
            bestDist = dist;
            hitIdx = ri;
          }
        }
        setHoveredRoiIdx(hitIdx);
      } else {
        setHoveredRoiIdx(-1);
      }

      // Point hover detection
      const pts = getPointsForImage(idx);
      if (pts.length > 0) {
        const { zoom } = getZoom(idx);
        const hitDist = 15 / (displayScale * zoom);
        let bestPtIdx = -1;
        let bestDist = Infinity;
        for (let pi = 0; pi < pts.length; pi++) {
          const dx = p.col - pts[pi].col;
          const dy = p.row - pts[pi].row;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < hitDist && dist < bestDist) {
            bestDist = dist;
            bestPtIdx = pi;
          }
        }
        setHoveredPointIdx(bestPtIdx);
      } else {
        setHoveredPointIdx(-1);
      }
    },
    [clientToImage, width, canvasW, canvasH, perImageData, imgMin, imgMax, isGallery, selectedIdx, getZoom, setZoom, isDraggingROI, activeRoiIdx, safeRois, setRois, displayScale, getPointsForImage, setPointsForImage],
  );

  // Mouse up — place point
  const handleMouseUp = React.useCallback(
    (e: React.MouseEvent, idx: number) => {
      if (draggingPointRef.current) {
        draggingPointRef.current = null;
        return;
      }
      if (isDraggingROI) {
        setIsDraggingROI(false);
        return;
      }
      const drag = dragRef.current;
      dragRef.current = null;
      if (drag?.wasDrag) return;
      if (justSwitchedRef.current) { justSwitchedRef.current = false; return; }
      if (isGallery && idx !== selectedIdx) return;

      let coords = clientToImage(e.clientX, e.clientY, idx);
      if (!coords) return;
      // Snap to local intensity peak if enabled
      if (snapEnabled && perImageData[idx]) {
        const snapped = findLocalMax(perImageData[idx], width, height, coords.col, coords.row, snapRadius);
        coords = { row: snapped.row, col: snapped.col };
      }
      redoStackRef.current.set(idx, []); // clear redo on new point
      const p: Point = { row: coords.row, col: coords.col, shape: (currentShape || "circle") as MarkerShape, color: currentColor || MARKER_COLORS[0] };
      const currentPts = getPointsForImage(idx);
      const limit = Number.isFinite(maxPoints) && maxPoints > 0 ? maxPoints : 3;
      const next = [...currentPts, p];
      setPointsForImage(idx, next.length <= limit ? next : next.slice(next.length - limit));
    },
    [clientToImage, maxPoints, isGallery, selectedIdx, getPointsForImage, setPointsForImage, isDraggingROI, currentShape, currentColor, snapEnabled, snapRadius, perImageData, width, height],
  );

  // Double-click — reset zoom
  const handleDoubleClick = React.useCallback((idx: number) => {
    if (isGallery && idx !== selectedIdx) return;
    setZoom(idx, DEFAULT_ZOOM);
  }, [isGallery, selectedIdx, setZoom]);

  const handleExport = React.useCallback(async () => {
    const idx = isGallery ? selectedIdx : 0;
    const label = isGallery && labels?.[idx] ? labels[idx] : "clicker";
    const prefix = `${label}_${width}x${height}`;

    const canvasToBlob = (c: HTMLCanvasElement): Promise<Blob> =>
      new Promise((resolve) => c.toBlob((b) => resolve(b!), "image/png"));

    const zip = new JSZip();

    // Raw image (colormapped, no markers/ROIs/scale bar)
    const offscreen = offscreenRefs.current[idx];
    if (offscreen) {
      const raw = document.createElement("canvas");
      raw.width = width;
      raw.height = height;
      const rawCtx = raw.getContext("2d");
      if (rawCtx) {
        rawCtx.drawImage(offscreen, 0, 0);
        zip.file(`${prefix}_raw.png`, await canvasToBlob(raw));
      }
    }

    // Annotated image (composite: image + markers + ROIs + labels)
    const mainCanvas = canvasRefs.current[idx];
    const overlay = overlayRefs.current[idx];
    const ui = uiRefs.current[idx];
    if (mainCanvas) {
      const comp = document.createElement("canvas");
      comp.width = canvasW;
      comp.height = canvasH;
      const ctx = comp.getContext("2d");
      if (ctx) {
        ctx.drawImage(mainCanvas, 0, 0);
        if (overlay) ctx.drawImage(overlay, 0, 0);
        if (ui) ctx.drawImage(ui, 0, 0, canvasW, canvasH);
        zip.file(`${prefix}_annotated.png`, await canvasToBlob(comp));
      }
    }

    // FFT image (if active)
    const fftCanvas = fftCanvasRef.current;
    if (showFft && fftCanvas) {
      zip.file(`${prefix}_fft.png`, await canvasToBlob(fftCanvas));
    }

    const blob = await zip.generateAsync({ type: "blob" });
    downloadBlob(blob, `${prefix}.zip`);
  }, [isGallery, selectedIdx, labels, width, height, canvasW, canvasH, showFft]);

  const activeIdx = isGallery ? selectedIdx : 0;

  // Redo stack: per-image undone points
  const redoStackRef = React.useRef<Map<number, Point[]>>(new Map());

  const resetPoints = React.useCallback(() => {
    if (!isGallery) {
      setSelectedPoints([]);
    } else {
      setSelectedPoints([...Array(nImages)].map(() => []));
    }
    setHover(null);
    setDotSize(12);
    setMaxPoints(10);
    redoStackRef.current = new Map();
  }, [isGallery, nImages, setSelectedPoints, setDotSize, setMaxPoints]);

  const undoPoint = React.useCallback(() => {
    const pts = getPointsForImage(activeIdx);
    if (pts.length === 0) return;
    const removed = pts[pts.length - 1];
    const stack = redoStackRef.current.get(activeIdx) || [];
    redoStackRef.current.set(activeIdx, [...stack, removed]);
    setPointsForImage(activeIdx, pts.slice(0, -1));
  }, [activeIdx, getPointsForImage, setPointsForImage]);

  const redoPoint = React.useCallback(() => {
    const stack = redoStackRef.current.get(activeIdx) || [];
    if (stack.length === 0) return;
    const point = stack[stack.length - 1];
    redoStackRef.current.set(activeIdx, stack.slice(0, -1));
    const pts = getPointsForImage(activeIdx);
    const limit = Number.isFinite(maxPoints) && maxPoints > 0 ? maxPoints : 3;
    if (pts.length < limit) {
      setPointsForImage(activeIdx, [...pts, point]);
    }
  }, [activeIdx, getPointsForImage, setPointsForImage, maxPoints]);

  const canRedo = (redoStackRef.current.get(activeIdx) || []).length > 0;

  // Keyboard shortcuts
  const handleKeyDown = React.useCallback((e: React.KeyboardEvent) => {
    const isMeta = e.metaKey || e.ctrlKey;
    switch (e.key) {
      case "Delete":
      case "Backspace":
        e.preventDefault();
        if (activeRoi) {
          pushRoiHistory();
          const next = safeRois.filter((_, i) => i !== activeRoiIdx);
          setActiveRoiIdx(next.length === 0 ? -1 : Math.min(activeRoiIdx, next.length - 1));
          setRois(next);
        } else {
          undoPoint();
        }
        break;
      case "z":
      case "Z":
        if (isMeta && e.shiftKey) { e.preventDefault(); if (!redoRoi()) redoPoint(); }
        else if (isMeta) { e.preventDefault(); undoPoint(); if (getPointsForImage(isGallery ? selectedIdx : 0).length === 0) undoRoi(); }
        break;
      case "1": case "2": case "3": case "4": case "5": case "6": {
        const roiIdx = parseInt(e.key) - 1;
        if (roiIdx < safeRois.length) { e.preventDefault(); setActiveRoiIdx(roiIdx); }
        break;
      }
      case "ArrowLeft":
        if (activeRoi) { e.preventDefault(); const step = e.shiftKey ? 10 : 1; updateActiveRoi({ col: Math.max(0, activeRoi.col - step) }); }
        else if (isGallery) { e.preventDefault(); setSelectedIdx(Math.max(0, selectedIdx - 1)); }
        break;
      case "ArrowRight":
        if (activeRoi) { e.preventDefault(); const step = e.shiftKey ? 10 : 1; updateActiveRoi({ col: Math.min(width - 1, activeRoi.col + step) }); }
        else if (isGallery) { e.preventDefault(); setSelectedIdx(Math.min(nImages - 1, selectedIdx + 1)); }
        break;
      case "Escape":
        e.preventDefault();
        setActiveRoiIdx(-1);
        break;
      case "r":
      case "R":
        handleDoubleClick(activeIdx);
        break;
    }
  }, [undoPoint, redoPoint, undoRoi, redoRoi, safeRois, activeRoi, activeRoiIdx, updateActiveRoi, pushRoiHistory, setActiveRoiIdx, setRois, width, height, isGallery, selectedIdx, nImages, setSelectedIdx, getPointsForImage, handleDoubleClick, activeIdx]);

  // Resize handlers
  const handleResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizing(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: isGallery ? galleryCanvasSize : mainCanvasSize });
  };

  React.useEffect(() => {
    if (!isResizing) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeStart) return;
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);
      const minSize = isGallery ? 100 : initialCanvasSizeRef.current;
      const maxSize = isGallery ? 600 : 800;
      const newSize = Math.max(minSize, Math.min(maxSize, resizeStart.size + delta));
      if (isGallery) {
        setGalleryCanvasSize(newSize);
      } else {
        setMainCanvasSize(newSize);
      }
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

  const activeZoom = getZoom(activeIdx);
  const needsReset = activeZoom.zoom !== 1 || activeZoom.panX !== 0 || activeZoom.panY !== 0;
  const maxPtsVal = Number.isFinite(maxPoints) && maxPoints > 0 ? maxPoints : 3;
  const activePts = getPointsForImage(activeIdx);
  const hasAnyPoints = isGallery
    ? (selectedPoints as Point[][])?.some((pts) => pts?.length > 0)
    : (selectedPoints as Point[])?.length > 0;

  // Render a single canvas box (shared between single and gallery mode)
  const renderCanvasBox = (idx: number, showResizeHandle: boolean) => (
    <Box
      ref={(el: HTMLDivElement | null) => { canvasContainerRefs.current[idx] = el; }}
      sx={{
        ...containerStyles.imageBox,
        width: canvasW,
        height: canvasH,
        cursor: isGallery && idx !== selectedIdx
          ? "pointer"
          : isDraggingROI || draggingPointRef.current ? "grabbing"
          : hoveredRoiIdx >= 0 || hoveredPointIdx >= 0 ? "grab"
          : "crosshair",
        border: isGallery && idx === selectedIdx
          ? `3px solid ${tc.accent}`
          : containerStyles.imageBox.border,
        borderRadius: 0,
      }}
      onMouseDown={(e) => handleMouseDown(e, idx)}
      onMouseMove={(e) => handleMouseMove(e, idx)}
      onMouseUp={(e) => handleMouseUp(e, idx)}
      onMouseLeave={() => { dragRef.current = null; draggingPointRef.current = null; setHover(null); setIsDraggingROI(false); setHoveredRoiIdx(-1); setHoveredPointIdx(-1); }}
      onWheel={(e) => handleWheel(e, idx)}
      onDoubleClick={() => handleDoubleClick(idx)}
    >
      <canvas
        ref={(el) => { canvasRefs.current[idx] = el; }}
        width={canvasW}
        height={canvasH}
        style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }}
      />
      <canvas
        ref={(el) => { overlayRefs.current[idx] = el; }}
        width={canvasW}
        height={canvasH}
        style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }}
      />
      <canvas
        ref={(el) => { uiRefs.current[idx] = el; }}
        width={Math.round(canvasW * DPR)}
        height={Math.round(canvasH * DPR)}
        style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }}
      />
      {showResizeHandle && (
        <Box
          onMouseDown={handleResizeStart}
          sx={{
            position: "absolute",
            bottom: 0,
            right: 0,
            width: 16,
            height: 16,
            cursor: "nwse-resize",
            opacity: 0.6,
            background: `linear-gradient(135deg, transparent 50%, ${tc.accent} 50%)`,
            borderRadius: "0 0 4px 0",
            "&:hover": { opacity: 1 },
          }}
        />
      )}
    </Box>
  );

  return (
    <Box className="clicker-root" tabIndex={0} onKeyDown={handleKeyDown} sx={{ ...containerStyles.root, bgcolor: tc.bg, color: tc.text, outline: "none" }}>
      {/* Header row */}
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28, maxWidth: contentW }}>
        <Typography variant="caption" sx={{ ...typography.label, color: tc.text }}>
          {title || "Clicker"}
          <InfoTooltip theme={themeInfo.theme} text={<KeyboardShortcuts items={[["Click", "Place point"], ["Drag point", "Reposition"], ["\u2318Z / Ctrl+Z", "Undo"], ["\u2318\u21E7Z", "Redo"], ["\u232B / Del", "Delete ROI or undo"], ["1 \u2013 6", "Select ROI"], ["\u2190 / \u2192", "Nudge ROI / Nav gallery"], ["\u21E7 + \u2190\u2192", "Nudge ROI 10 px"], ["R", "Reset zoom"], ["Esc", "Deselect ROI"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]]} />} />
          {isGallery && labels?.[activeIdx] && (
            <Box component="span" sx={{ color: tc.textMuted, ml: 1 }}>
              {labels[activeIdx]}
            </Box>
          )}
          {activePts.length > 0 && (
            <Box component="span" sx={{ color: tc.textMuted, ml: 1 }}>
              {activePts.length}/{maxPtsVal} pts
            </Box>
          )}
        </Typography>
        <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
          <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>FFT:</Typography>
          <Switch checked={showFft} onChange={(e) => setShowFft(e.target.checked)} size="small" sx={switchStyles.small} />
          <Button size="small" sx={{ ...compactButton, color: tc.accent }} onClick={handleExport}>EXPORT</Button>
          <Button size="small" sx={compactButton} onClick={undoPoint} disabled={!activePts.length}>UNDO</Button>
          <Button size="small" sx={compactButton} onClick={redoPoint} disabled={!canRedo}>REDO</Button>
          <Button size="small" sx={compactButton} disabled={!needsReset} onClick={() => handleDoubleClick(activeIdx)}>RESET VIEW</Button>
          <Button size="small" sx={compactButton} onClick={resetPoints} disabled={!hasAnyPoints}>RESET ALL</Button>
        </Stack>
      </Stack>

      {/* Canvas area + FFT side-by-side */}
      {isGallery ? (
        <Box sx={{ display: "inline-grid", gridTemplateColumns: `repeat(${ncols}, ${canvasW}px)`, gap: 1 }}>
          {Array.from({ length: nImages }).map((_, i) => (
            <Box key={i}>
              {renderCanvasBox(i, i === selectedIdx)}
              <Typography sx={{ fontSize: 10, color: i === selectedIdx ? tc.accent : tc.textMuted, textAlign: "center", mt: 0.25 }}>
                {labels?.[i] || `Image ${i + 1}`}
              </Typography>
            </Box>
          ))}
        </Box>
      ) : (
        <Stack direction="row" spacing={`${SPACING.LG}px`}>
          {renderCanvasBox(0, true)}
          {showFft && (
            <Box sx={{ ...containerStyles.imageBox, width: canvasW, height: canvasH }}>
              <canvas
                ref={fftCanvasRef}
                width={canvasW}
                height={canvasH}
                style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }}
              />
              <Typography sx={{ position: "absolute", top: 4, left: 8, fontSize: 10, color: "#fff", textShadow: "0 0 3px #000" }}>
                FFT
              </Typography>
            </Box>
          )}
        </Stack>
      )}

      {/* Stats + Readout bar */}
      <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: tc.bgAlt, display: "flex", gap: 2, alignItems: "center", minHeight: 20, maxWidth: contentW, boxSizing: "border-box" }}>
        {showStats && imageStats ? (
          <>
            <Typography sx={{ fontSize: 11, color: tc.textMuted }}>
              Mean <Box component="span" sx={{ color: tc.accent }}>{formatNumber(imageStats.mean)}</Box>
            </Typography>
            <Typography sx={{ fontSize: 11, color: tc.textMuted }}>
              Min <Box component="span" sx={{ color: tc.accent }}>{formatNumber(imageStats.min)}</Box>
            </Typography>
            <Typography sx={{ fontSize: 11, color: tc.textMuted }}>
              Max <Box component="span" sx={{ color: tc.accent }}>{formatNumber(imageStats.max)}</Box>
            </Typography>
            <Typography sx={{ fontSize: 11, color: tc.textMuted }}>
              Std <Box component="span" sx={{ color: tc.accent }}>{formatNumber(imageStats.std)}</Box>
            </Typography>
          </>
        ) : (
          <Typography sx={{ fontSize: 11, color: tc.textMuted }}>
            {width}×{height} px
          </Typography>
        )}
        {hover && (
          <>
            <Box sx={{ borderLeft: `1px solid ${tc.border}`, height: 14 }} />
            <Typography sx={{ fontSize: 11, color: tc.textMuted, fontFamily: "monospace" }}>
              ({hover.row}, {hover.col}) <Box component="span" sx={{ color: tc.accent }}>{hover.raw !== undefined ? formatNumber(hover.raw) : ""}</Box>
            </Typography>
          </>
        )}
        <Typography sx={{ fontSize: 11, color: tc.textMuted, ml: "auto" }}>
          {width}×{height}
          {activeZoom.zoom !== 1 && (
            <Box component="span" sx={{ color: tc.accent, fontWeight: "bold", ml: 1 }}>
              {activeZoom.zoom.toFixed(1)}x
            </Box>
          )}
        </Typography>
      </Box>

      {/* Image controls + ROI basics + Histogram (Show3D layout) */}
      <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, maxWidth: contentW, boxSizing: "border-box" }}>
        {/* Left: two rows of controls */}
        <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, justifyContent: "center" }}>
          {/* Row 1: Scale + Auto + Color */}
          <Box sx={{ ...controlRow, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg }}>
            <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>Scale:<InfoTooltip text="Linear or logarithmic intensity mapping. Log emphasizes low-intensity features." theme={themeInfo.theme} /></Typography>
            <Select value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" variant="outlined" MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 50 }}>
              <MenuItem value="linear" sx={{ fontSize: 11 }}>Lin</MenuItem>
              <MenuItem value="log" sx={{ fontSize: 11 }}>Log</MenuItem>
            </Select>
            <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>Auto:<InfoTooltip text="Auto-contrast using percentile-based clipping. Ignores extreme outliers for better contrast." theme={themeInfo.theme} /></Typography>
            <Switch checked={autoContrast} onChange={(e) => setAutoContrast(e.target.checked)} size="small" sx={switchStyles.small} />
            <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>Color:</Typography>
            <Select size="small" value={colormap || "gray"} onChange={(e) => setColormap(e.target.value)} variant="outlined" MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 65 }}>
              {COLORMAP_NAMES.map((name) => (
                <MenuItem key={name} value={name} sx={{ fontSize: 11 }}>
                  {name.charAt(0).toUpperCase() + name.slice(1)}
                </MenuItem>
              ))}
            </Select>
          </Box>
          {/* Row 2: ROI basics */}
          <Box sx={{ ...controlRow, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg }}>
            <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>ROI:<InfoTooltip text="Region of Interest. Click ADD to place, then drag to reposition. Shows pixel statistics (Mean, Std, Min, Max) for the enclosed area." theme={themeInfo.theme} /></Typography>
            <Select
              size="small"
              value={activeRoi ? activeRoi.mode : newRoiShape}
              onChange={(e) => {
                const val = e.target.value as RoiShape;
                setNewRoiShape(val);
                if (activeRoi) updateActiveRoi({ mode: val });
              }}
              variant="outlined"
              MenuProps={themedMenuProps}
              sx={{ ...themedSelect, minWidth: 75 }}
            >
              {ROI_SHAPES.map((m) => (
                <MenuItem key={m} value={m} sx={{ fontSize: 11 }}>{m.charAt(0).toUpperCase() + m.slice(1)}</MenuItem>
              ))}
            </Select>
            <Button
              size="small"
              variant="outlined"
              onClick={() => {
                pushRoiHistory();
                const id = Math.max(0, ...safeRois.map(r => r.id)) + 1;
                const color = ROI_COLORS[safeRois.length % ROI_COLORS.length];
                const roi: ROI = {
                  id, mode: newRoiShape,
                  row: Math.floor(height / 2), col: Math.floor(width / 2),
                  radius: 30, rectW: 60, rectH: 40,
                  color, opacity: 0.8,
                };
                setActiveRoiIdx(safeRois.length);
                setRois([...safeRois, roi]);
              }}
              sx={{ fontSize: 10, minWidth: 0, px: 1, py: 0.25, color: tc.accent, borderColor: tc.border }}
            >
              ADD
            </Button>
          </Box>
        </Box>
        {/* Right: Histogram spanning both rows */}
        <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
          {histogramData && (
            <HistogramWidget
              data={histogramData}
              vminPct={vminPct}
              vmaxPct={vmaxPct}
              onRangeChange={(min, max) => { setVminPct(min); setVmaxPct(max); }}
              width={110}
              height={58}
              theme={themeInfo.theme}
              dataMin={dataRange.min}
              dataMax={dataRange.max}
            />
          )}
        </Box>
      </Box>

      {/* Shape + Color picker row */}
      <Box sx={{ ...controlRow, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg, mt: 0.5, maxWidth: contentW, width: "fit-content", boxSizing: "border-box" }}>
        <Box sx={{ display: "flex", gap: "3px", flexShrink: 0 }}>
          {MARKER_SHAPES.map(s => {
            const sz = 16;
            const half = sz / 2;
            const r = half * 0.7;
            const selected = s === currentShape;
            let path: React.ReactNode;
            switch (s) {
              case "circle": path = <circle cx={half} cy={half} r={r} />; break;
              case "triangle": path = <polygon points={`${half},${half - r} ${half + r * 0.87},${half + r * 0.5} ${half - r * 0.87},${half + r * 0.5}`} />; break;
              case "square": path = <rect x={half - r * 0.75} y={half - r * 0.75} width={r * 1.5} height={r * 1.5} />; break;
              case "diamond": path = <polygon points={`${half},${half - r} ${half + r * 0.7},${half} ${half},${half + r} ${half - r * 0.7},${half}`} />; break;
              case "star": {
                const pts: string[] = [];
                for (let i = 0; i < 10; i++) {
                  const angle = (i * Math.PI) / 5 - Math.PI / 2;
                  const sr = i % 2 === 0 ? r : r * 0.4;
                  pts.push(`${half + sr * Math.cos(angle)},${half + sr * Math.sin(angle)}`);
                }
                path = <polygon points={pts.join(" ")} />;
                break;
              }
            }
            return (
              <Box
                key={s}
                onClick={() => setCurrentShape(s)}
                sx={{
                  width: sz, height: sz, cursor: "pointer", borderRadius: "2px",
                  border: selected ? `2px solid ${tc.text}` : "2px solid transparent",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  "&:hover": { opacity: 0.8 },
                }}
              >
                <svg width={sz} height={sz} style={{ display: "block" }}>
                  <g fill={currentColor} stroke={tc.bg} strokeWidth={1}>{path}</g>
                </svg>
              </Box>
            );
          })}
        </Box>
        <Box sx={{ display: "flex", gap: "3px" }}>
          {MARKER_COLORS.map(c => (
            <Box
              key={c}
              onClick={() => setCurrentColor(c)}
              sx={{
                width: 16, height: 16, bgcolor: c, borderRadius: "2px", cursor: "pointer",
                border: c === currentColor ? `2px solid ${tc.text}` : "2px solid transparent",
                "&:hover": { opacity: 0.8 },
              }}
            />
          ))}
        </Box>
      </Box>

      {/* Controls row: Marker size + Max + Advanced toggle */}
      <Box sx={{ ...controlRow, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg, mt: 0.5, maxWidth: contentW, width: "fit-content", boxSizing: "border-box" }}>
        <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>Marker:</Typography>
        <Slider
          value={size}
          min={4}
          max={40}
          step={1}
          onChange={(_, v) => { if (typeof v === "number") setDotSize(v); }}
          size="small"
          sx={{ ...sliderStyles.small, width: 60 }}
        />
        <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 20 }}>{size}px</Typography>
        <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>Max:</Typography>
        <Select
          value={maxPtsVal}
          onChange={(e: SelectChangeEvent<number>) => {
            const v = Number(e.target.value);
            setMaxPoints(v);
            if (!isGallery) {
              setSelectedPoints((prev) => {
                const flat = (prev as Point[]) || [];
                return flat.length <= v ? flat : flat.slice(flat.length - v);
              });
            } else {
              setSelectedPoints((prev) => {
                const nested = ((prev as Point[][]) || []).map(pts =>
                  pts.length <= v ? pts : pts.slice(pts.length - v)
                );
                return nested;
              });
            }
          }}
          size="small"
          variant="outlined"
          MenuProps={themedMenuProps}
          sx={themedSelect}
        >
          {Array.from({ length: 20 }, (_, i) => i + 1).map(n => (
            <MenuItem key={n} value={n} sx={{ fontSize: 11 }}>{n}</MenuItem>
          ))}
        </Select>
        <Typography
          onClick={() => setSnapEnabled(!snapEnabled)}
          sx={{ ...typography.labelSmall, color: snapEnabled ? accentGreen : tc.textMuted, cursor: "pointer", userSelect: "none", fontWeight: snapEnabled ? "bold" : "normal", "&:hover": { textDecoration: "underline" } }}
        >
          {snapEnabled ? "\u25C9" : "\u25CB"} Snap<InfoTooltip text="Snap to local intensity maximum. Clicked positions jump to the brightest pixel within the search radius R. Useful for precise atom column picking on HAADF-STEM images." theme={themeInfo.theme} />
        </Typography>
        {snapEnabled && (
          <>
            <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>R:</Typography>
            <Slider value={snapRadius} min={1} max={20} step={1} onChange={(_, v) => { if (typeof v === "number") setSnapRadius(v); }} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
            <Typography sx={{ ...typography.value, color: tc.textMuted }}>{snapRadius}px</Typography>
          </>
        )}
        <Typography
          onClick={() => setShowAdvanced(!showAdvanced)}
          sx={{ ...typography.labelSmall, color: tc.accent, cursor: "pointer", userSelect: "none", "&:hover": { textDecoration: "underline" } }}
        >
          {showAdvanced ? "\u25BE Advanced" : "\u25B8 Advanced"}
        </Typography>
      </Box>

      {/* Advanced options row (collapsible) */}
      {showAdvanced && (
        <Box sx={{ ...controlRow, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg, maxWidth: contentW, width: "fit-content", boxSizing: "border-box" }}>
          <Typography sx={{ ...typography.label, color: tc.textMuted }}>Border</Typography>
          <Slider
            value={borderWidth}
            min={0}
            max={6}
            step={1}
            onChange={(_, v) => { if (typeof v === "number") setBorderWidth(v); }}
            size="small"
            sx={{ ...sliderStyles.small, width: 50 }}
          />
          <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 16 }}>{borderWidth}</Typography>
          <Typography sx={{ ...typography.label, color: tc.textMuted }}>Opacity</Typography>
          <Slider
            value={markerOpacity}
            min={0.1}
            max={1.0}
            step={0.1}
            onChange={(_, v) => { if (typeof v === "number") setMarkerOpacity(v); }}
            size="small"
            sx={{ ...sliderStyles.small, width: 50 }}
          />
          <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 20 }}>{Math.round(markerOpacity * 100)}%</Typography>
          <Typography sx={{ ...typography.label, color: tc.textMuted }}>Label</Typography>
          <Slider
            value={labelSize}
            min={0}
            max={36}
            step={1}
            onChange={(_, v) => { if (typeof v === "number") setLabelSize(v); }}
            size="small"
            sx={{ ...sliderStyles.small, width: 50 }}
          />
          <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 28 }}>{labelSize === 0 ? "Auto" : `${labelSize}px`}</Typography>
          <Typography sx={{ ...typography.label, color: tc.textMuted }}>Color</Typography>
          <Select value={labelColor} onChange={(e) => setLabelColor(e.target.value)} size="small" variant="outlined" displayEmpty MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 60 }}>
            <MenuItem value="" sx={{ fontSize: 11 }}>Auto</MenuItem>
            <MenuItem value="white" sx={{ fontSize: 11 }}>White</MenuItem>
            <MenuItem value="black" sx={{ fontSize: 11 }}>Black</MenuItem>
            <MenuItem value="#ff0" sx={{ fontSize: 11 }}>Yellow</MenuItem>
            <MenuItem value="#0f0" sx={{ fontSize: 11 }}>Green</MenuItem>
            <MenuItem value="#f00" sx={{ fontSize: 11 }}>Red</MenuItem>
            <MenuItem value="#0af" sx={{ fontSize: 11 }}>Cyan</MenuItem>
          </Select>
        </Box>
      )}

      {/* Active ROI details (only when a ROI is selected) */}
      {activeRoi && (
        <Box sx={{ ...controlRow, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg, mt: 0.5, maxWidth: contentW, width: "fit-content", boxSizing: "border-box" }}>
          <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 24 }}>
            ROI #{activeRoiIdx + 1}/{safeRois.length}
          </Typography>
          {safeRois.length > 1 && (
            <>
              <Typography
                onClick={() => setActiveRoiIdx((activeRoiIdx - 1 + safeRois.length) % safeRois.length)}
                sx={{ ...typography.labelSmall, color: tc.accent, cursor: "pointer", userSelect: "none" }}
              >&larr;</Typography>
              <Typography
                onClick={() => setActiveRoiIdx((activeRoiIdx + 1) % safeRois.length)}
                sx={{ ...typography.labelSmall, color: tc.accent, cursor: "pointer", userSelect: "none" }}
              >&rarr;</Typography>
            </>
          )}
          {activeRoi.mode === "rectangle" ? (
            <>
              <Typography sx={{ ...typography.label, color: tc.textMuted }}>W</Typography>
              <Slider value={activeRoi.rectW} min={5} max={Math.max(width, 10)} onChange={(_, v) => { if (typeof v === "number") updateActiveRoi({ rectW: v }); }} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
              <Typography sx={{ ...typography.label, color: tc.textMuted }}>H</Typography>
              <Slider value={activeRoi.rectH} min={5} max={Math.max(height, 10)} onChange={(_, v) => { if (typeof v === "number") updateActiveRoi({ rectH: v }); }} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
            </>
          ) : (
            <>
              <Typography sx={{ ...typography.label, color: tc.textMuted }}>Size</Typography>
              <Slider value={activeRoi.radius} min={5} max={Math.max(width, height, 10)} onChange={(_, v) => { if (typeof v === "number") updateActiveRoi({ radius: v }); }} size="small" sx={{ ...sliderStyles.small, width: 50 }} />
            </>
          )}
          <Box sx={{ display: "flex", gap: "2px" }}>
            {ROI_COLORS.map(c => (
              <Box
                key={c}
                onClick={() => updateActiveRoi({ color: c })}
                sx={{
                  width: 12, height: 12, bgcolor: c, cursor: "pointer",
                  border: c === activeRoi.color ? `2px solid ${tc.text}` : "1px solid transparent",
                  "&:hover": { opacity: 0.8 },
                }}
              />
            ))}
          </Box>
          <Typography sx={{ ...typography.label, color: tc.textMuted }}>Op</Typography>
          <Slider
            value={activeRoi.opacity}
            min={0.1}
            max={1.0}
            step={0.1}
            onChange={(_, v) => { if (typeof v === "number") updateActiveRoi({ opacity: v }); }}
            size="small"
            sx={{ ...sliderStyles.small, width: 40 }}
          />
          <Button
            size="small"
            variant="outlined"
            onClick={() => {
              pushRoiHistory();
              const next = safeRois.filter((_, i) => i !== activeRoiIdx);
              setActiveRoiIdx(next.length === 0 ? -1 : Math.min(activeRoiIdx, next.length - 1));
              setRois(next);
            }}
            sx={{ fontSize: 10, minWidth: 0, px: 0.5, py: 0.25, color: tc.textMuted, borderColor: tc.border }}
          >
            &times;
          </Button>
        </Box>
      )}

      {/* ROI pixel statistics */}
      {roiStats && (
        <Box sx={{ ...controlRow, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg, maxWidth: contentW, width: "fit-content", boxSizing: "border-box" }}>
          <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>ROI Stats:</Typography>
          <Typography sx={{ ...typography.value, color: tc.textMuted }}>
            Mean <Box component="span" sx={{ color: tc.accent }}>{roiStats.mean.toFixed(4)}</Box>
          </Typography>
          <Typography sx={{ ...typography.value, color: tc.textMuted }}>
            Std <Box component="span" sx={{ color: tc.accent }}>{roiStats.std.toFixed(4)}</Box>
          </Typography>
          <Typography sx={{ ...typography.value, color: tc.textMuted }}>
            Min <Box component="span" sx={{ color: tc.accent }}>{roiStats.min.toFixed(4)}</Box>
          </Typography>
          <Typography sx={{ ...typography.value, color: tc.textMuted }}>
            Max <Box component="span" sx={{ color: tc.accent }}>{roiStats.max.toFixed(4)}</Box>
          </Typography>
          <Typography sx={{ ...typography.value, color: tc.textMuted }}>
            N <Box component="span" sx={{ color: tc.accent }}>{roiStats.count}</Box>
          </Typography>
        </Box>
      )}

      {/* Selected points list */}
      {isGallery ? (
        hasAnyPoints && (
          <Box sx={{ mt: 0.5 }}>
            {Array.from({ length: nImages }).map((_, imgIdx) => {
              const pts = getPointsForImage(imgIdx);
              if (pts.length === 0) return null;
              return (
                <Box key={imgIdx} sx={{ mb: 0.5 }}>
                  <Typography sx={{ fontSize: 10, fontFamily: "monospace", color: imgIdx === selectedIdx ? tc.accent : tc.textMuted, fontWeight: "bold", lineHeight: 1.6 }}>
                    {labels?.[imgIdx] || `Image ${imgIdx + 1}`}
                  </Typography>
                  <Box sx={{ display: "grid", gridTemplateColumns: "repeat(5, auto)", gap: `0 ${SPACING.MD}px`, width: "fit-content", pl: 1 }}>
                    {pts.map((p, i) => {
                      const c = p.color || MARKER_COLORS[i % MARKER_COLORS.length];
                      return (
                        <Box key={`pt-${imgIdx}-${i}`} sx={{ display: "flex", alignItems: "center", gap: "2px", lineHeight: 1.6 }}>
                          <ShapeIcon shape={p.shape || "circle"} color={c} size={10} />
                          <Typography component="span" sx={{ fontSize: 10, fontFamily: "monospace", color: tc.textMuted }}>
                            <Box component="span" sx={{ color: c }}>{i + 1}</Box> ({p.row}, {p.col})
                            {i > 0 && (
                              <Box component="span" sx={{ color: tc.textMuted, ml: 0.5, fontSize: 9 }}>
                                {"\u2194"} {formatDistance(pts[i - 1], p, pixelSizeAngstrom || 0)}
                              </Box>
                            )}
                          </Typography>
                        </Box>
                      );
                    })}
                  </Box>
                </Box>
              );
            })}
          </Box>
        )
      ) : (
        activePts.length > 0 && (
          <Box sx={{ mt: 0.5, display: "grid", gridTemplateColumns: "repeat(5, auto)", gap: `0 ${SPACING.MD}px`, width: "fit-content" }}>
            {activePts.map((p, i) => {
              const c = p.color || MARKER_COLORS[i % MARKER_COLORS.length];
              return (
                <Box key={`pt-${p.row}-${p.col}-${i}`} sx={{ display: "flex", alignItems: "center", gap: "2px", lineHeight: 1.6 }}>
                  <ShapeIcon shape={p.shape || "circle"} color={c} size={10} />
                  <Typography component="span" sx={{ fontSize: 10, fontFamily: "monospace", color: tc.textMuted }}>
                    <Box component="span" sx={{ color: c }}>{i + 1}</Box> ({p.row}, {p.col})
                    {i > 0 && (
                      <Box component="span" sx={{ color: tc.textMuted, ml: 0.5, fontSize: 9 }}>
                        {"\u2194"} {formatDistance(activePts[i - 1], p, pixelSizeAngstrom || 0)}
                      </Box>
                    )}
                  </Typography>
                </Box>
              );
            })}
          </Box>
        )
      )}
    </Box>
  );
});

export default { render };
