import * as React from "react";
import { createRender, useModelState, useModel } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import Switch from "@mui/material/Switch";
import Tooltip from "@mui/material/Tooltip";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import StopIcon from "@mui/icons-material/Stop";
import "./styles.css";
import { useTheme } from "../theme";
import { COLORMAPS, applyColormap, renderToOffscreen } from "../colormaps";
import { drawScaleBarHiDPI, roundToNiceValue, formatScaleLabel, exportFigure } from "../scalebar";
import { findDataRange, sliderRange, computeStats, applyLogScale, percentileClip } from "../stats";
import { formatNumber, downloadBlob, downloadDataView } from "../format";
import { computeHistogramFromBytes } from "../histogram";
import { getWebGPUFFT, WebGPUFFT, fft2d, fftshift, computeMagnitude, autoEnhanceFFT } from "../webgpu-fft";

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const CANVAS_SIZE = 450;
const RESIZE_HIT_AREA_PX = 10;
const CIRCLE_HANDLE_ANGLE = 0.707;

// ============================================================================
// UI Styles
// ============================================================================
const typography = {
  label: { fontSize: 11 },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const },
};

const SPACING = {
  XS: 4,
  SM: 8,
  MD: 12,
  LG: 16,
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
  small: { '& .MuiSwitch-thumb': { width: 12, height: 12 }, '& .MuiSwitch-switchBase': { padding: '4px' } },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

// ============================================================================
// Helpers
// ============================================================================
function formatStat(value: number): string {
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs < 0.001 || abs >= 10000) return value.toExponential(2);
  if (abs < 0.01) return value.toFixed(4);
  if (abs < 1) return value.toFixed(3);
  return value.toFixed(2);
}

// ============================================================================
// HiDPI Drawing Functions
// ============================================================================

/** Draw position crosshair on high-DPI canvas */
function drawPositionMarker(
  canvas: HTMLCanvasElement,
  dpr: number,
  posRow: number,
  posCol: number,
  zoom: number,
  panX: number,
  panY: number,
  imageWidth: number,
  imageHeight: number,
  isDragging: boolean,
  snapEnabled: boolean = false,
  snapRadius: number = 5
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const scaleX = cssWidth / imageWidth;
  const scaleY = cssHeight / imageHeight;

  const screenX = posCol * zoom * scaleX + panX * scaleX;
  const screenY = posRow * zoom * scaleY + panY * scaleY;

  const crosshairSize = 12;
  const lineWidth = 1.5;

  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(255, 100, 100, 0.9)";
  ctx.lineWidth = lineWidth;

  ctx.beginPath();
  ctx.moveTo(screenX - crosshairSize, screenY);
  ctx.lineTo(screenX + crosshairSize, screenY);
  ctx.moveTo(screenX, screenY - crosshairSize);
  ctx.lineTo(screenX, screenY + crosshairSize);
  ctx.stroke();

  // Snap radius circle
  if (snapEnabled && snapRadius > 0) {
    const radiusScreenX = snapRadius * zoom * scaleX;
    const radiusScreenY = snapRadius * zoom * scaleY;
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "rgba(0, 200, 255, 0.7)";
    ctx.lineWidth = 1.2;
    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, radiusScreenX, radiusScreenY, 0, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.restore();
}

/** Draw ROI overlay on high-DPI canvas for navigation panel */
function drawNavRoiOverlay(
  canvas: HTMLCanvasElement,
  dpr: number,
  roiMode: string,
  centerX: number,
  centerY: number,
  radius: number,
  roiWidth: number,
  roiHeight: number,
  zoom: number,
  panX: number,
  panY: number,
  imageWidth: number,
  imageHeight: number,
  isDragging: boolean,
  isDraggingResize: boolean,
  isHoveringResize: boolean
) {
  if (roiMode === "off") return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const scaleX = cssWidth / imageWidth;
  const scaleY = cssHeight / imageHeight;

  const screenX = centerY * zoom * scaleX + panX * scaleX;
  const screenY = centerX * zoom * scaleY + panY * scaleY;

  const lineWidth = 2.5;
  const crosshairSize = 10;
  const handleRadius = 6;

  ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  const drawResizeHandle = (handleX: number, handleY: number) => {
    let handleFill: string;
    let handleStroke: string;
    if (isDraggingResize) {
      handleFill = "rgba(0, 200, 255, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else if (isHoveringResize) {
      handleFill = "rgba(255, 100, 100, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else {
      handleFill = "rgba(0, 255, 0, 0.8)";
      handleStroke = "rgba(255, 255, 255, 0.8)";
    }
    ctx.beginPath();
    ctx.arc(handleX, handleY, handleRadius, 0, 2 * Math.PI);
    ctx.fillStyle = handleFill;
    ctx.fill();
    ctx.strokeStyle = handleStroke;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  };

  const drawCenterCrosshair = () => {
    ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(screenX - crosshairSize, screenY);
    ctx.lineTo(screenX + crosshairSize, screenY);
    ctx.moveTo(screenX, screenY - crosshairSize);
    ctx.lineTo(screenX, screenY + crosshairSize);
    ctx.stroke();
  };

  const strokeColor = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
  const fillColor = isDragging ? "rgba(255, 255, 0, 0.12)" : "rgba(0, 255, 0, 0.12)";

  if (roiMode === "circle" && radius > 0) {
    const screenRadiusX = radius * zoom * scaleX;
    const screenRadiusY = radius * zoom * scaleY;

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusX, screenRadiusY, 0, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.fillStyle = fillColor;
    ctx.fill();

    drawCenterCrosshair();
    const handleOffsetX = screenRadiusX * CIRCLE_HANDLE_ANGLE;
    const handleOffsetY = screenRadiusY * CIRCLE_HANDLE_ANGLE;
    drawResizeHandle(screenX + handleOffsetX, screenY + handleOffsetY);

  } else if (roiMode === "square" && radius > 0) {
    const screenHalfW = radius * zoom * scaleX;
    const screenHalfH = radius * zoom * scaleY;
    const left = screenX - screenHalfW;
    const top = screenY - screenHalfH;

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();
    ctx.fillStyle = fillColor;
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);

  } else if (roiMode === "rect" && roiWidth > 0 && roiHeight > 0) {
    const screenHalfW = (roiWidth / 2) * zoom * scaleX;
    const screenHalfH = (roiHeight / 2) * zoom * scaleY;
    const left = screenX - screenHalfW;
    const top = screenY - screenHalfH;

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();
    ctx.fillStyle = fillColor;
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);
  }

  ctx.restore();
}

// ============================================================================
// KeyboardShortcuts Component
// ============================================================================
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
// InfoTooltip Component
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

// ============================================================================
// Histogram Component
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

function Histogram({
  data,
  colormap,
  vminPct,
  vmaxPct,
  onRangeChange,
  width = 120,
  height = 40,
  theme = "dark",
  dataMin = 0,
  dataMax = 1,
}: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const bins = React.useMemo(() => computeHistogramFromBytes(data), [data]);

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
      for (let j = 0; j < binRatio; j++) {
        sum += bins[i * binRatio + j] || 0;
      }
      reducedBins.push(sum / binRatio);
    }

    const maxVal = Math.max(...reducedBins, 0.001);
    const barWidth = width / displayBins;
    const vminBin = Math.floor((vminPct / 100) * displayBins);
    const vmaxBin = Math.floor((vmaxPct / 100) * displayBins);

    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      const x = i * barWidth;
      const inRange = i >= vminBin && i <= vmaxBin;
      ctx.fillStyle = inRange ? colors.barActive : colors.barInactive;
      ctx.fillRect(x + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
    }
  }, [bins, colormap, vminPct, vmaxPct, width, height, colors]);

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

// ============================================================================
// Line Profile Functions
// ============================================================================
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
// Snap-to-peak: find local intensity maximum
// ============================================================================
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

// ============================================================================
// Main Component
// ============================================================================
function Show4D() {
  const model = useModel();
  const { themeInfo, colors: themeColors } = useTheme();
  const DPR = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;

  // Themed typography — applies theme colors to module-level font sizes
  const typo = React.useMemo(() => ({
    label: { ...typography.label, color: themeColors.textMuted },
    value: { ...typography.value, color: themeColors.textMuted },
    title: { ...typography.title, color: themeColors.accent },
  }), [themeColors]);

  // ── Model State ──
  const [navRows] = useModelState<number>("nav_rows");
  const [navCols] = useModelState<number>("nav_cols");
  const [sigRows] = useModelState<number>("sig_rows");
  const [sigCols] = useModelState<number>("sig_cols");
  const [posRow, setPosRow] = useModelState<number>("pos_row");
  const [posCol, setPosCol] = useModelState<number>("pos_col");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [navImageBytes] = useModelState<DataView>("nav_image_bytes");
  const [navDataMin] = useModelState<number>("nav_data_min");
  const [navDataMax] = useModelState<number>("nav_data_max");
  const [sigDataMin] = useModelState<number>("sig_data_min");
  const [sigDataMax] = useModelState<number>("sig_data_max");
  const [roiMode, setRoiMode] = useModelState<string>("roi_mode");
  const [roiReduce, setRoiReduce] = useModelState<string>("roi_reduce");
  const [roiCenterRow] = useModelState<number>("roi_center_row");
  const [roiCenterCol] = useModelState<number>("roi_center_col");
  const [roiRadius, setRoiRadius] = useModelState<number>("roi_radius");
  const [roiWidth, setRoiWidth] = useModelState<number>("roi_width");
  const [roiHeight, setRoiHeight] = useModelState<number>("roi_height");
  const [navStats] = useModelState<number[]>("nav_stats");
  const [sigStats] = useModelState<number[]>("sig_stats");
  const [navPixelSize] = useModelState<number>("nav_pixel_size");
  const [sigPixelSize] = useModelState<number>("sig_pixel_size");
  const [navPixelUnit] = useModelState<string>("nav_pixel_unit");
  const [sigPixelUnit] = useModelState<string>("sig_pixel_unit");
  const [title] = useModelState<string>("title");
  const [snapEnabled, setSnapEnabled] = useModelState<boolean>("snap_enabled");
  const [snapRadius, setSnapRadius] = useModelState<number>("snap_radius");
  const [profileLine, setProfileLine] = useModelState<{row: number; col: number}[]>("profile_line");
  const [profileWidth, setProfileWidth] = useModelState<number>("profile_width");
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");

  // Path animation
  const [pathPlaying, setPathPlaying] = useModelState<boolean>("path_playing");
  const [pathIndex, setPathIndex] = useModelState<number>("path_index");
  const [pathLength] = useModelState<number>("path_length");
  const [pathIntervalMs] = useModelState<number>("path_interval_ms");
  const [pathLoop] = useModelState<boolean>("path_loop");

  // Export
  const [, setGifExportRequested] = useModelState<boolean>("_gif_export_requested");
  const [gifData] = useModelState<DataView>("_gif_data");
  const [exporting, setExporting] = React.useState(false);
  const [navExportAnchor, setNavExportAnchor] = React.useState<HTMLElement | null>(null);
  const [sigExportAnchor, setSigExportAnchor] = React.useState<HTMLElement | null>(null);

  // ── Local State ──
  const [localPosRow, setLocalPosRow] = React.useState(posRow + 0.5);
  const [localPosCol, setLocalPosCol] = React.useState(posCol + 0.5);
  const [isDraggingNav, setIsDraggingNav] = React.useState(false);
  const [isDraggingRoi, setIsDraggingRoi] = React.useState(false);
  const [isDraggingRoiResize, setIsDraggingRoiResize] = React.useState(false);
  const [isHoveringRoiResize, setIsHoveringRoiResize] = React.useState(false);
  const [localRoiCenterRow, setLocalRoiCenterRow] = React.useState(roiCenterRow);
  const [localRoiCenterCol, setLocalRoiCenterCol] = React.useState(roiCenterCol);

  // Signal panel drag-to-pan
  const [isDraggingSig, setIsDraggingSig] = React.useState(false);
  const [sigDragStart, setSigDragStart] = React.useState<{ x: number; y: number; panX: number; panY: number } | null>(null);

  // Independent colormaps and scales
  const [navColormap, setNavColormap] = React.useState("inferno");
  const [sigColormap, setSigColormap] = React.useState("inferno");
  const [navScaleMode, setNavScaleMode] = React.useState<"linear" | "log" | "power">("linear");
  const [sigScaleMode, setSigScaleMode] = React.useState<"linear" | "log" | "power">("linear");
  const navPowerExp = 0.5;
  const sigPowerExp = 0.5;
  const [navVminPct, setNavVminPct] = React.useState(0);
  const [navVmaxPct, setNavVmaxPct] = React.useState(100);
  const [sigVminPct, setSigVminPct] = React.useState(0);
  const [sigVmaxPct, setSigVmaxPct] = React.useState(100);

  // Zoom state
  const [navZoom, setNavZoom] = React.useState(1);
  const [navPanX, setNavPanX] = React.useState(0);
  const [navPanY, setNavPanY] = React.useState(0);
  const [sigZoom, setSigZoom] = React.useState(1);
  const [sigPanX, setSigPanX] = React.useState(0);
  const [sigPanY, setSigPanY] = React.useState(0);

  // Canvas resize state
  const [canvasSize, setCanvasSize] = React.useState(CANVAS_SIZE);
  const [isResizing, setIsResizing] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number; y: number; size: number } | null>(null);

  // Histogram data
  const [navHistogramData, setNavHistogramData] = React.useState<Float32Array | null>(null);
  const [sigHistogramData, setSigHistogramData] = React.useState<Float32Array | null>(null);

  // Line profile state
  const [profileActive, setProfileActive] = React.useState(false);
  const [profileData, setProfileData] = React.useState<Float32Array | null>(null);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const profilePoints = profileLine || [];
  const rawSigDataRef = React.useRef<Float32Array | null>(null);
  const sigClickStartRef = React.useRef<{ x: number; y: number } | null>(null);

  // FFT state
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const fftOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const fftUiRef = React.useRef<HTMLCanvasElement>(null);
  const fftOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const fftMagRef = React.useRef<Float32Array | null>(null);
  const [fftMagVersion, setFftMagVersion] = React.useState(0);
  const [fftZoom, setFftZoom] = React.useState(1);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);
  const [fftColormap, setFftColormap] = React.useState("inferno");
  const [fftLogScale, setFftLogScale] = React.useState(false);
  const [fftAuto, setFftAuto] = React.useState(true);
  const [fftVminPct, setFftVminPct] = React.useState(0);
  const [fftVmaxPct, setFftVmaxPct] = React.useState(100);
  const [fftHistogramData, setFftHistogramData] = React.useState<Float32Array | null>(null);
  const [fftDataRange, setFftDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [fftStats, setFftStats] = React.useState<{ mean: number; min: number; max: number; std: number }>({ mean: 0, min: 0, max: 0, std: 0 });
  const [isDraggingFft, setIsDraggingFft] = React.useState(false);
  const [fftDragStart, setFftDragStart] = React.useState<{ x: number; y: number; panX: number; panY: number } | null>(null);

  // ROI toggle memory
  const lastRoiModeRef = React.useRef<string>("circle");

  // Cursor readout
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number; panel: string } | null>(null);

  // Aspect-ratio-aware canvas sizes
  const navCanvasWidth = navRows > navCols ? Math.round(canvasSize * (navCols / navRows)) : canvasSize;
  const navCanvasHeight = navCols > navRows ? Math.round(canvasSize * (navRows / navCols)) : canvasSize;
  const sigCanvasWidth = sigRows > sigCols ? Math.round(canvasSize * (sigCols / sigRows)) : canvasSize;
  const sigCanvasHeight = sigCols > sigRows ? Math.round(canvasSize * (sigRows / sigCols)) : canvasSize;

  // Canvas refs (three-canvas stack per panel)
  const navCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const navOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const navUiRef = React.useRef<HTMLCanvasElement>(null);
  const navOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const navImageDataRef = React.useRef<ImageData | null>(null);
  const rawNavImageRef = React.useRef<Float32Array | null>(null);

  const sigCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const sigOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const sigUiRef = React.useRef<HTMLCanvasElement>(null);
  const sigOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const sigImageDataRef = React.useRef<ImageData | null>(null);

  // ── Sync local state ──
  React.useEffect(() => {
    if (!isDraggingNav) { setLocalPosRow(posRow + 0.5); setLocalPosCol(posCol + 0.5); }
  }, [posRow, posCol, isDraggingNav]);

  React.useEffect(() => {
    if (!isDraggingRoi && !isDraggingRoiResize) {
      setLocalRoiCenterRow(roiCenterRow);
      setLocalRoiCenterCol(roiCenterCol);
    }
  }, [roiCenterRow, roiCenterCol, isDraggingRoi, isDraggingRoiResize]);

  // ── Prevent scroll on canvases ──
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const overlays = [navOverlayRef.current, sigOverlayRef.current, fftOverlayRef.current];
    overlays.forEach(el => el?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => overlays.forEach(el => el?.removeEventListener("wheel", preventDefault));
  }, []);

  // ── GPU FFT init ──
  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) { gpuFFTRef.current = fft; setGpuReady(true); }
    });
  }, []);

  // ── Path animation timer ──
  React.useEffect(() => {
    if (!pathPlaying || pathLength === 0) return;
    const timer = setInterval(() => {
      setPathIndex((prev: number) => {
        const next = prev + 1;
        if (next >= pathLength) {
          if (pathLoop) {
            return 0;
          } else {
            setPathPlaying(false);
            return prev;
          }
        }
        return next;
      });
    }, pathIntervalMs);
    return () => clearInterval(timer);
  }, [pathPlaying, pathLength, pathIntervalMs, pathLoop, setPathIndex, setPathPlaying]);

  // ── Parse nav image bytes ──
  React.useEffect(() => {
    if (!navImageBytes) return;
    const numFloats = navImageBytes.byteLength / 4;
    const rawData = new Float32Array(navImageBytes.buffer, navImageBytes.byteOffset, numFloats);
    let storedData = rawNavImageRef.current;
    if (!storedData || storedData.length !== numFloats) {
      storedData = new Float32Array(numFloats);
      rawNavImageRef.current = storedData;
    }
    storedData.set(rawData);

    const scaledData = new Float32Array(numFloats);
    if (navScaleMode === "log") {
      for (let i = 0; i < numFloats; i++) scaledData[i] = Math.log1p(Math.max(0, rawData[i]));
    } else if (navScaleMode === "power") {
      for (let i = 0; i < numFloats; i++) scaledData[i] = Math.pow(Math.max(0, rawData[i]), navPowerExp);
    } else {
      scaledData.set(rawData);
    }
    setNavHistogramData(scaledData);
  }, [navImageBytes, navScaleMode, navPowerExp]);

  // ── Parse signal frame bytes ──
  React.useEffect(() => {
    if (!frameBytes) return;
    const rawData = new Float32Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength / 4);
    // Store raw data for profile sampling
    if (!rawSigDataRef.current || rawSigDataRef.current.length !== rawData.length) {
      rawSigDataRef.current = new Float32Array(rawData.length);
    }
    rawSigDataRef.current.set(rawData);
    const scaledData = new Float32Array(rawData.length);
    if (sigScaleMode === "log") {
      for (let i = 0; i < rawData.length; i++) scaledData[i] = Math.log1p(Math.max(0, rawData[i]));
    } else if (sigScaleMode === "power") {
      for (let i = 0; i < rawData.length; i++) scaledData[i] = Math.pow(Math.max(0, rawData[i]), sigPowerExp);
    } else {
      scaledData.set(rawData);
    }
    setSigHistogramData(scaledData);
  }, [frameBytes, sigScaleMode, sigPowerExp]);

  // ── Render nav image ──
  React.useEffect(() => {
    if (!rawNavImageRef.current || !navCanvasRef.current) return;
    const canvas = navCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rawData = rawNavImageRef.current;
    let scaled: Float32Array;
    if (navScaleMode === "log") {
      scaled = new Float32Array(rawData.length);
      for (let i = 0; i < rawData.length; i++) scaled[i] = Math.log1p(Math.max(0, rawData[i]));
    } else if (navScaleMode === "power") {
      scaled = new Float32Array(rawData.length);
      for (let i = 0; i < rawData.length; i++) scaled[i] = Math.pow(Math.max(0, rawData[i]), navPowerExp);
    } else {
      scaled = rawData;
    }

    const { min: dataMin, max: dataMax } = findDataRange(scaled);
    const { vmin, vmax } = sliderRange(dataMin, dataMax, navVminPct, navVmaxPct);

    const width = navCols;
    const height = navRows;
    const lut = COLORMAPS[navColormap] || COLORMAPS.inferno;

    let offscreen = navOffscreenRef.current;
    if (!offscreen) {
      offscreen = document.createElement("canvas");
      navOffscreenRef.current = offscreen;
    }
    if (offscreen.width !== width || offscreen.height !== height) {
      offscreen.width = width;
      offscreen.height = height;
      navImageDataRef.current = null;
    }
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;

    let imgData = navImageDataRef.current;
    if (!imgData) {
      imgData = offCtx.createImageData(width, height);
      navImageDataRef.current = imgData;
    }
    applyColormap(scaled, imgData.data, lut, vmin, vmax);
    offCtx.putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(navPanX, navPanY);
    ctx.scale(navZoom, navZoom);
    ctx.drawImage(offscreen, 0, 0);
    ctx.restore();
  }, [navImageBytes, navColormap, navVminPct, navVmaxPct, navScaleMode, navPowerExp, navZoom, navPanX, navPanY, navRows, navCols]);

  // ── Render signal frame ──
  React.useEffect(() => {
    if (!frameBytes || !sigCanvasRef.current) return;
    const canvas = sigCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rawData = new Float32Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength / 4);
    let scaled: Float32Array;
    if (sigScaleMode === "log") {
      scaled = new Float32Array(rawData.length);
      for (let i = 0; i < rawData.length; i++) scaled[i] = Math.log1p(Math.max(0, rawData[i]));
    } else if (sigScaleMode === "power") {
      scaled = new Float32Array(rawData.length);
      for (let i = 0; i < rawData.length; i++) scaled[i] = Math.pow(Math.max(0, rawData[i]), sigPowerExp);
    } else {
      scaled = rawData;
    }

    const { min: dataMin, max: dataMax } = findDataRange(scaled);
    const { vmin, vmax } = sliderRange(dataMin, dataMax, sigVminPct, sigVmaxPct);

    const width = sigCols;
    const height = sigRows;
    const lut = COLORMAPS[sigColormap] || COLORMAPS.inferno;

    let offscreen = sigOffscreenRef.current;
    if (!offscreen) {
      offscreen = document.createElement("canvas");
      sigOffscreenRef.current = offscreen;
    }
    if (offscreen.width !== width || offscreen.height !== height) {
      offscreen.width = width;
      offscreen.height = height;
      sigImageDataRef.current = null;
    }
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;

    let imgData = sigImageDataRef.current;
    if (!imgData) {
      imgData = offCtx.createImageData(width, height);
      sigImageDataRef.current = imgData;
    }
    applyColormap(scaled, imgData.data, lut, vmin, vmax);
    offCtx.putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(sigPanX, sigPanY);
    ctx.scale(sigZoom, sigZoom);
    ctx.drawImage(offscreen, 0, 0);
    ctx.restore();
  }, [frameBytes, sigColormap, sigVminPct, sigVmaxPct, sigScaleMode, sigPowerExp, sigZoom, sigPanX, sigPanY, sigRows, sigCols]);

  // ── Compute FFT from signal frame ──
  React.useEffect(() => {
    if (!showFft || !rawSigDataRef.current) return;
    let cancelled = false;
    const data = rawSigDataRef.current;
    const w = sigCols, h = sigRows;

    const computeFFT = async () => {
      let real: Float32Array, imag: Float32Array;
      if (gpuReady && gpuFFTRef.current) {
        const result = await gpuFFTRef.current.fft2D(data.slice(), new Float32Array(data.length), w, h, false);
        real = result.real;
        imag = result.imag;
      } else {
        real = data.slice();
        imag = new Float32Array(data.length);
        fft2d(real, imag, w, h, false);
      }
      if (cancelled) return;
      fftshift(real, w, h);
      fftshift(imag, w, h);
      fftMagRef.current = computeMagnitude(real, imag);
      setFftMagVersion(v => v + 1);
    };
    computeFFT();
    return () => { cancelled = true; };
  }, [showFft, frameBytes, sigRows, sigCols, gpuReady]);

  // ── Render FFT image ──
  React.useEffect(() => {
    const mag = fftMagRef.current;
    if (!showFft || !mag) return;

    const w = sigCols, h = sigRows;
    let displayMin: number, displayMax: number;
    if (fftAuto) {
      ({ min: displayMin, max: displayMax } = autoEnhanceFFT(mag, w, h));
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
    const offscreen = renderToOffscreen(displayData, w, h, lut, vmin, vmax);
    if (!offscreen) return;
    fftOffscreenRef.current = offscreen;

    if (fftCanvasRef.current) {
      const ctx = fftCanvasRef.current.getContext("2d");
      if (ctx) {
        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, sigCols, sigRows);
        ctx.save();
        ctx.translate(fftPanX, fftPanY);
        ctx.scale(fftZoom, fftZoom);
        ctx.drawImage(offscreen, 0, 0);
        ctx.restore();
      }
    }
  }, [showFft, fftMagVersion, fftLogScale, fftAuto, fftVminPct, fftVmaxPct, fftColormap, sigRows, sigCols]);

  // ── FFT zoom/pan redraw ──
  React.useEffect(() => {
    if (!showFft || !fftCanvasRef.current || !fftOffscreenRef.current) return;
    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, sigCols, sigRows);
    ctx.save();
    ctx.translate(fftPanX, fftPanY);
    ctx.scale(fftZoom, fftZoom);
    ctx.drawImage(fftOffscreenRef.current, 0, 0);
    ctx.restore();
  }, [showFft, fftZoom, fftPanX, fftPanY, sigCols, sigRows]);

  // ── FFT UI overlay ──
  React.useEffect(() => {
    if (!fftUiRef.current || !showFft) return;
    const canvas = fftUiRef.current;
    canvas.width = sigCanvasWidth * DPR;
    canvas.height = sigCanvasHeight * DPR;
    if (sigPixelSize > 0) {
      const recipPxSize = 1.0 / (sigPixelSize * sigCols);
      drawScaleBarHiDPI(canvas, DPR, fftZoom, recipPxSize, "1/" + sigPixelUnit, sigCols);
    } else {
      drawScaleBarHiDPI(canvas, DPR, fftZoom, 1, "px", sigCols);
    }
  }, [showFft, fftZoom, fftPanX, fftPanY, sigPixelSize, sigPixelUnit, sigCols, sigCanvasWidth, sigCanvasHeight]);

  // ── Nav HiDPI UI overlay ──
  React.useEffect(() => {
    if (!navUiRef.current) return;
    const navUnit = navPixelSize > 0 ? navPixelUnit : "px";
    const navPxSize = navPixelSize > 0 ? navPixelSize : 1;
    drawScaleBarHiDPI(navUiRef.current, DPR, navZoom, navPxSize, navUnit, navCols);

    if (roiMode === "off") {
      drawPositionMarker(navUiRef.current, DPR, localPosRow, localPosCol, navZoom, navPanX, navPanY, navCols, navRows, isDraggingNav, snapEnabled, snapRadius);
    } else {
      drawNavRoiOverlay(
        navUiRef.current, DPR, roiMode,
        localRoiCenterRow, localRoiCenterCol, roiRadius, roiWidth, roiHeight,
        navZoom, navPanX, navPanY, navCols, navRows,
        isDraggingRoi, isDraggingRoiResize, isHoveringRoiResize
      );
    }
  }, [navZoom, navPanX, navPanY, navPixelSize, navPixelUnit, navRows, navCols,
      localPosRow, localPosCol, isDraggingNav, snapEnabled, snapRadius,
      roiMode, localRoiCenterRow, localRoiCenterCol, roiRadius, roiWidth, roiHeight,
      isDraggingRoi, isDraggingRoiResize, isHoveringRoiResize]);

  // ── Signal HiDPI UI overlay ──
  React.useEffect(() => {
    if (!sigUiRef.current) return;
    const canvas = sigUiRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const sigUnit = sigPixelSize > 0 ? sigPixelUnit : "px";
    const sigPxSize = sigPixelSize > 0 ? sigPixelSize : 1;
    drawScaleBarHiDPI(canvas, DPR, sigZoom, sigPxSize, sigUnit, sigCols);

    // Profile line overlay
    if (profileActive && profilePoints.length > 0) {
      ctx.save();
      ctx.scale(DPR, DPR);
      const cssW = canvas.width / DPR;
      const cssH = canvas.height / DPR;
      const scaleX = cssW / sigCols;
      const scaleY = cssH / sigRows;
      const toScreenX = (col: number) => col * sigZoom * scaleX + sigPanX * scaleX;
      const toScreenY = (row: number) => row * sigZoom * scaleY + sigPanY * scaleY;

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

        // Draw line A→B
        ctx.strokeStyle = themeColors.accent;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();

        // Draw point B
        ctx.fillStyle = themeColors.accent;
        ctx.beginPath();
        ctx.arc(bx, by, 4, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }
  }, [sigZoom, sigPanX, sigPanY, sigPixelSize, sigPixelUnit, sigRows, sigCols,
      profileActive, profilePoints, profileWidth, themeColors]);

  // ── Profile computation ──
  React.useEffect(() => {
    if (profilePoints.length === 2 && rawSigDataRef.current) {
      const p0 = profilePoints[0], p1 = profilePoints[1];
      setProfileData(sampleLineProfile(rawSigDataRef.current, sigCols, sigRows, p0.row, p0.col, p1.row, p1.col, profileWidth));
      if (!profileActive) setProfileActive(true);
    } else {
      setProfileData(null);
    }
  }, [profilePoints, profileWidth, frameBytes]);

  // ── Profile sparkline rendering ──
  React.useEffect(() => {
    const canvas = profileCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = sigCanvasWidth;
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
      ctx.fillText("Click two points on the signal to draw a profile", cssW / 2, cssH / 2);
      return;
    }

    const padTop = 6;
    const padBottom = 18;
    const plotH = cssH - padTop - padBottom;

    let gMin = Infinity, gMax = -Infinity;
    for (let i = 0; i < profileData.length; i++) {
      if (profileData[i] < gMin) gMin = profileData[i];
      if (profileData[i] > gMax) gMax = profileData[i];
    }
    const range = gMax - gMin || 1;

    // Draw profile curve
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < profileData.length; i++) {
      const x = (i / (profileData.length - 1)) * cssW;
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
      if (sigPixelSize > 0) {
        totalDist = distPx * sigPixelSize;
        xUnit = sigPixelUnit;
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

    // Y-axis min/max labels
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(formatNumber(gMax), 2, 1);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatNumber(gMin), 2, padTop + plotH - 1);
  }, [profileData, profilePoints, sigPixelSize, sigPixelUnit, sigCanvasWidth, themeInfo.theme, themeColors.accent]);

  // ── Zoom handler factory ──
  const createZoomHandler = (
    setZoom: React.Dispatch<React.SetStateAction<number>>,
    setPanXFn: React.Dispatch<React.SetStateAction<number>>,
    setPanYFn: React.Dispatch<React.SetStateAction<number>>,
    zoom: number, panX: number, panY: number,
    canvasRef: React.RefObject<HTMLCanvasElement | null>
  ) => (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * zoomFactor));
    const zoomRatio = newZoom / zoom;
    setZoom(newZoom);
    setPanXFn(mouseX - (mouseX - panX) * zoomRatio);
    setPanYFn(mouseY - (mouseY - panY) * zoomRatio);
  };

  // ── Resize handle hit test ──
  const isNearRoiResizeHandle = (imgX: number, imgY: number): boolean => {
    if (roiMode === "off") return false;
    if (roiMode === "rect") {
      const halfH = (roiHeight || 10) / 2;
      const halfW = (roiWidth || 10) / 2;
      const handleX = localRoiCenterRow + halfH;
      const handleY = localRoiCenterCol + halfW;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      const cornerDist = Math.sqrt(halfW ** 2 + halfH ** 2);
      const hitArea = Math.min(RESIZE_HIT_AREA_PX / navZoom, cornerDist * 0.5);
      return dist < hitArea;
    }
    if (roiMode === "circle" || roiMode === "square") {
      const radius = roiRadius || 5;
      const offset = roiMode === "square" ? radius : radius * CIRCLE_HANDLE_ANGLE;
      const handleX = localRoiCenterRow + offset;
      const handleY = localRoiCenterCol + offset;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      const hitArea = Math.min(RESIZE_HIT_AREA_PX / navZoom, radius * 0.5);
      return dist < hitArea;
    }
    return false;
  };

  // ── Nav mouse handlers ──
  const handleNavMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = navOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenY - navPanY) / navZoom;
    const imgY = (screenX - navPanX) / navZoom;

    if (roiMode !== "off") {
      if (isNearRoiResizeHandle(imgX, imgY)) {
        setIsDraggingRoiResize(true);
        return;
      }
      setIsDraggingRoi(true);
      setLocalRoiCenterRow(imgX);
      setLocalRoiCenterCol(imgY);
      const newX = Math.round(Math.max(0, Math.min(navRows - 1, imgX)));
      const newY = Math.round(Math.max(0, Math.min(navCols - 1, imgY)));
      model.set("roi_center", [newX, newY]);
      model.save_changes();
      return;
    }

    setIsDraggingNav(true);
    let newX = Math.round(Math.max(0, Math.min(navRows - 1, imgX)));
    let newY = Math.round(Math.max(0, Math.min(navCols - 1, imgY)));
    if (snapEnabled && rawNavImageRef.current) {
      const snapped = findLocalMax(rawNavImageRef.current, navCols, navRows, newY, newX, snapRadius);
      newX = snapped.row;
      newY = snapped.col;
    }
    setLocalPosRow(newX + 0.5);
    setLocalPosCol(newY + 0.5);
    model.set("pos_row", newX);
    model.set("pos_col", newY);
    model.save_changes();
  };

  const handleNavMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = navOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenY - navPanY) / navZoom;
    const imgY = (screenX - navPanX) / navZoom;

    // Cursor readout
    const pxRow = Math.floor(imgX);
    const pxCol = Math.floor(imgY);
    if (pxRow >= 0 && pxRow < navRows && pxCol >= 0 && pxCol < navCols && rawNavImageRef.current) {
      setCursorInfo({ row: pxRow, col: pxCol, value: rawNavImageRef.current[pxRow * navCols + pxCol], panel: "nav" });
    } else {
      setCursorInfo(prev => prev?.panel === "nav" ? null : prev);
    }

    // ROI resize dragging
    if (isDraggingRoiResize) {
      const dx = Math.abs(imgX - localRoiCenterRow);
      const dy = Math.abs(imgY - localRoiCenterCol);
      if (roiMode === "rect") {
        setRoiWidth(Math.max(2, Math.round(dy * 2)));
        setRoiHeight(Math.max(2, Math.round(dx * 2)));
      } else if (roiMode === "square") {
        setRoiRadius(Math.max(1, Math.round(Math.max(dx, dy))));
      } else {
        setRoiRadius(Math.max(1, Math.round(Math.sqrt(dx ** 2 + dy ** 2))));
      }
      return;
    }

    // Hover check for resize handles
    if (!isDraggingRoi && !isDraggingNav) {
      setIsHoveringRoiResize(isNearRoiResizeHandle(imgX, imgY));
      if (roiMode !== "off") return;
    }

    // ROI center dragging
    if (isDraggingRoi) {
      setLocalRoiCenterRow(imgX);
      setLocalRoiCenterCol(imgY);
      const newX = Math.round(Math.max(0, Math.min(navRows - 1, imgX)));
      const newY = Math.round(Math.max(0, Math.min(navCols - 1, imgY)));
      model.set("roi_center", [newX, newY]);
      model.save_changes();
      return;
    }

    // Position dragging
    if (!isDraggingNav) return;
    let newX = Math.round(Math.max(0, Math.min(navRows - 1, imgX)));
    let newY = Math.round(Math.max(0, Math.min(navCols - 1, imgY)));
    if (snapEnabled && rawNavImageRef.current) {
      const snapped = findLocalMax(rawNavImageRef.current, navCols, navRows, newY, newX, snapRadius);
      newX = snapped.row;
      newY = snapped.col;
    }
    setLocalPosRow(newX + 0.5);
    setLocalPosCol(newY + 0.5);
    model.set("pos_row", newX);
    model.set("pos_col", newY);
    model.save_changes();
  };

  const handleNavMouseUp = () => {
    setIsDraggingNav(false);
    setIsDraggingRoi(false);
    setIsDraggingRoiResize(false);
  };
  const handleNavMouseLeave = () => {
    setIsDraggingNav(false);
    setIsDraggingRoi(false);
    setIsDraggingRoiResize(false);
    setIsHoveringRoiResize(false);
    setCursorInfo(prev => prev?.panel === "nav" ? null : prev);
  };
  const handleNavDoubleClick = () => { setNavZoom(1); setNavPanX(0); setNavPanY(0); };

  // ── FFT mouse handlers ──
  const handleFftMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDraggingFft(true);
    setFftDragStart({ x: e.clientX, y: e.clientY, panX: fftPanX, panY: fftPanY });
  };

  const handleFftMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDraggingFft || !fftDragStart) return;
    const canvas = fftOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    setFftPanX(fftDragStart.panX + (e.clientX - fftDragStart.x) * scaleX);
    setFftPanY(fftDragStart.panY + (e.clientY - fftDragStart.y) * scaleY);
  };

  const handleFftMouseUp = () => { setIsDraggingFft(false); setFftDragStart(null); };
  const handleFftMouseLeave = () => { setIsDraggingFft(false); setFftDragStart(null); };
  const handleFftDoubleClick = () => { setFftZoom(1); setFftPanX(0); setFftPanY(0); };

  // ── Signal mouse handlers (drag-to-pan + profile click) ──
  const handleSigMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    sigClickStartRef.current = { x: e.clientX, y: e.clientY };
    setIsDraggingSig(true);
    setSigDragStart({ x: e.clientX, y: e.clientY, panX: sigPanX, panY: sigPanY });
  };

  const handleSigMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // Cursor readout
    const canvas = sigOverlayRef.current;
    if (canvas) {
      const rect = canvas.getBoundingClientRect();
      const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
      const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
      const imgX = (screenY - sigPanY) / sigZoom;
      const imgY = (screenX - sigPanX) / sigZoom;
      const pxRow = Math.floor(imgX);
      const pxCol = Math.floor(imgY);
      if (pxRow >= 0 && pxRow < sigRows && pxCol >= 0 && pxCol < sigCols && frameBytes) {
        const raw = new Float32Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength / 4);
        setCursorInfo({ row: pxRow, col: pxCol, value: raw[pxRow * sigCols + pxCol], panel: "sig" });
      } else {
        setCursorInfo(prev => prev?.panel === "sig" ? null : prev);
      }
    }

    if (!isDraggingSig || !sigDragStart) return;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const dx = (e.clientX - sigDragStart.x) * scaleX;
    const dy = (e.clientY - sigDragStart.y) * scaleY;
    setSigPanX(sigDragStart.panX + dx);
    setSigPanY(sigDragStart.panY + dy);
  };

  const handleSigMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // Profile click capture
    if (profileActive && sigClickStartRef.current) {
      const dx = e.clientX - sigClickStartRef.current.x;
      const dy = e.clientY - sigClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const canvas = sigOverlayRef.current;
        if (canvas && rawSigDataRef.current) {
          const rect = canvas.getBoundingClientRect();
          const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
          const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
          const imgCol = (screenX - sigPanX) / sigZoom;
          const imgRow = (screenY - sigPanY) / sigZoom;
          if (imgCol >= 0 && imgCol < sigCols && imgRow >= 0 && imgRow < sigRows) {
            const pt = { row: imgRow, col: imgCol };
            if (profilePoints.length === 0 || profilePoints.length === 2) {
              setProfileLine([pt]);
              setProfileData(null);
            } else {
              const p0 = profilePoints[0];
              setProfileLine([p0, pt]);
              setProfileData(sampleLineProfile(rawSigDataRef.current, sigCols, sigRows, p0.row, p0.col, pt.row, pt.col, profileWidth));
            }
          }
        }
      }
    }
    sigClickStartRef.current = null;
    setIsDraggingSig(false);
    setSigDragStart(null);
  };
  const handleSigMouseLeave = () => {
    setIsDraggingSig(false);
    setSigDragStart(null);
    setCursorInfo(prev => prev?.panel === "sig" ? null : prev);
  };
  const handleSigDoubleClick = () => { setSigZoom(1); setSigPanX(0); setSigPanY(0); };

  // ── Canvas resize handlers ──
  const handleResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizing(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: canvasSize });
  };

  React.useEffect(() => {
    if (!isResizing) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeStart) return;
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);
      setCanvasSize(Math.max(CANVAS_SIZE, Math.min(800, resizeStart.size + delta)));
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

  // ── Nav Export Handlers ──
  const handleNavExportFigure = (withColorbar: boolean) => {
    setNavExportAnchor(null);
    if (!navCanvasRef.current) return;
    const navData = new Float32Array(navImageBytes.buffer, navImageBytes.byteOffset, navImageBytes.byteLength / 4);
    const lut = COLORMAPS[navColormap] || COLORMAPS.inferno;
    const { min: dMin, max: dMax } = findDataRange(navData);
    const offscreen = renderToOffscreen(navData, navCols, navRows, lut, dMin, dMax);
    if (!offscreen) return;
    const pixelSizeAngstrom = navPixelSize > 0 && navPixelUnit === "\u00C5" ? navPixelSize : navPixelSize > 0 && navPixelUnit === "nm" ? navPixelSize * 10 : 0;
    const figCanvas = exportFigure({
      imageCanvas: offscreen,
      title: title || "Navigation",
      lut,
      vmin: dMin,
      vmax: dMax,
      pixelSize: pixelSizeAngstrom > 0 ? pixelSizeAngstrom : undefined,
      showColorbar: withColorbar,
      showScaleBar: pixelSizeAngstrom > 0,
    });
    figCanvas.toBlob((blob) => { if (blob) downloadBlob(blob, "show4d_nav_figure.png"); }, "image/png");
  };

  const handleNavExportPng = () => {
    setNavExportAnchor(null);
    if (!navCanvasRef.current) return;
    navCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4d_nav.png"); }, "image/png");
  };

  // ── Signal Export Handlers ──
  const handleSigExportFigure = (withColorbar: boolean) => {
    setSigExportAnchor(null);
    const frameData = rawSigDataRef.current;
    if (!frameData) return;
    let processed: Float32Array;
    if (sigScaleMode === "log") {
      processed = new Float32Array(frameData.length);
      for (let i = 0; i < frameData.length; i++) processed[i] = Math.log1p(Math.max(0, frameData[i]));
    } else if (sigScaleMode === "power") {
      processed = new Float32Array(frameData.length);
      for (let i = 0; i < frameData.length; i++) processed[i] = Math.pow(Math.max(0, frameData[i]), sigPowerExp);
    } else {
      processed = frameData;
    }
    const lut = COLORMAPS[sigColormap] || COLORMAPS.inferno;
    const { min: pMin, max: pMax } = findDataRange(processed);
    const { vmin, vmax } = sliderRange(pMin, pMax, sigVminPct, sigVmaxPct);
    const offscreen = renderToOffscreen(processed, sigCols, sigRows, lut, vmin, vmax);
    if (!offscreen) return;
    const pixelSizeAngstrom = sigPixelSize > 0 && sigPixelUnit === "\u00C5" ? sigPixelSize : sigPixelSize > 0 && sigPixelUnit === "nm" ? sigPixelSize * 10 : 0;
    const figCanvas = exportFigure({
      imageCanvas: offscreen,
      title: title ? `${title} \u2014 Signal` : "Signal",
      lut,
      vmin,
      vmax,
      pixelSize: pixelSizeAngstrom > 0 ? pixelSizeAngstrom : undefined,
      showColorbar: withColorbar,
      showScaleBar: pixelSizeAngstrom > 0,
    });
    figCanvas.toBlob((blob) => { if (blob) downloadBlob(blob, "show4d_signal_figure.png"); }, "image/png");
  };

  const handleSigExportPng = () => {
    setSigExportAnchor(null);
    if (!sigCanvasRef.current) return;
    sigCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4d_signal.png"); }, "image/png");
  };

  const handleSigExportGif = () => {
    setSigExportAnchor(null);
    setExporting(true);
    setGifExportRequested(true);
  };

  // Download GIF when data arrives from Python
  React.useEffect(() => {
    if (!gifData || gifData.byteLength === 0) return;
    downloadDataView(gifData, "show4d_animation.gif", "image/gif");
    setExporting(false);
  }, [gifData]);

  // ── Keyboard shortcuts ──
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      const step = e.shiftKey ? 10 : 1;
      switch (e.key) {
        case "ArrowLeft":
          e.preventDefault();
          setPosCol(Math.max(0, posCol - step));
          break;
        case "ArrowRight":
          e.preventDefault();
          setPosCol(Math.min(navCols - 1, posCol + step));
          break;
        case "r":
        case "R":
          setNavZoom(1); setNavPanX(0); setNavPanY(0);
          setSigZoom(1); setSigPanX(0); setSigPanY(0);
          setFftZoom(1); setFftPanX(0); setFftPanY(0);
          break;
        case "t":
        case "T":
          if (roiMode === "off") {
            setRoiMode(lastRoiModeRef.current);
          } else {
            lastRoiModeRef.current = roiMode;
            setRoiMode("off");
          }
          break;
        case " ":
          if (pathLength > 0) {
            e.preventDefault();
            setPathPlaying(!pathPlaying);
          }
          break;
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [posRow, posCol, navRows, navCols, setPosRow, setPosCol, roiMode, setRoiMode, pathLength, pathPlaying, setPathPlaying]);

  // ── Theme-aware select style ──
  const themedSelect = {
    minWidth: 65,
    bgcolor: themeColors.controlBg,
    color: themeColors.text,
    fontSize: 11,
    "& .MuiSelect-select": { py: 0.5 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
  };

  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: { sx: { bgcolor: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}` } },
  };

  // ── Render ──
  return (
    <Box className="show4d-root" sx={{ p: `${SPACING.LG}px`, bgcolor: themeColors.bg, color: themeColors.text }}>
      {/* Title */}
      <Typography variant="h6" sx={{ ...typo.title, mb: `${SPACING.SM}px` }}>
        {title || "4D Explorer"}
        <InfoTooltip text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
          <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>ROI: Region of Interest on navigation image — integrates signal over enclosed area.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Snap: Snap to local intensity maximum within search radius.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT: Show power spectrum of signal image.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Profile: Click two points to draw a line intensity profile.</Typography>
          <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
          <KeyboardShortcuts items={[["← / →", "Move position"], ["Shift+←/→", "Move ×10"], ["T", "Toggle ROI on/off"], ["Space", "Play / pause path"], ["R", "Reset zoom"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]]} />
        </Box>} theme={themeInfo.theme} />
      </Typography>

      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        {/* ── LEFT COLUMN: Navigation Panel ── */}
        <Box sx={{ width: navCanvasWidth }}>
          {/* Nav Header */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
            <Typography variant="caption" sx={{ ...typo.label }}>
              Navigation ({Math.round(localPosRow)}, {Math.round(localPosCol)})
            </Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`}>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={async () => {
                if (!navCanvasRef.current) return;
                try {
                  const blob = await new Promise<Blob | null>(resolve => navCanvasRef.current!.toBlob(resolve, "image/png"));
                  if (!blob) return;
                  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
                } catch {
                  navCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4d_nav.png"); }, "image/png");
                }
              }}>COPY</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={(e) => setNavExportAnchor(e.currentTarget)} disabled={exporting}>{exporting ? "..." : "Export"}</Button>
              <Menu anchorEl={navExportAnchor} open={Boolean(navExportAnchor)} onClose={() => setNavExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                <MenuItem onClick={() => handleNavExportFigure(true)} sx={{ fontSize: 12 }}>Figure + colorbar</MenuItem>
                <MenuItem onClick={() => handleNavExportFigure(false)} sx={{ fontSize: 12 }}>Figure</MenuItem>
                <MenuItem onClick={handleNavExportPng} sx={{ fontSize: 12 }}>PNG</MenuItem>
              </Menu>
              <Button size="small" sx={compactButton} disabled={navZoom === 1 && navPanX === 0 && navPanY === 0} onClick={() => { setNavZoom(1); setNavPanX(0); setNavPanY(0); }}>Reset</Button>
            </Stack>
          </Stack>

          {/* Nav Canvas */}
          <Box sx={{ ...container.imageBox, width: navCanvasWidth, height: navCanvasHeight }}>
            <canvas ref={navCanvasRef} width={navCols} height={navRows} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
            <canvas
              ref={navOverlayRef} width={navCols} height={navRows}
              onMouseDown={handleNavMouseDown} onMouseMove={handleNavMouseMove}
              onMouseUp={handleNavMouseUp} onMouseLeave={handleNavMouseLeave}
              onWheel={createZoomHandler(setNavZoom, setNavPanX, setNavPanY, navZoom, navPanX, navPanY, navOverlayRef)}
              onDoubleClick={handleNavDoubleClick}
              style={{ position: "absolute", width: "100%", height: "100%", cursor: isHoveringRoiResize || isDraggingRoiResize ? "nwse-resize" : snapEnabled ? "cell" : "crosshair" }}
            />
            <canvas ref={navUiRef} width={navCanvasWidth * DPR} height={navCanvasHeight * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
            {cursorInfo && cursorInfo.panel === "nav" && (
              <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                  ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                </Typography>
              </Box>
            )}
            <Box onMouseDown={handleResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: 1 } }} />
          </Box>

          {/* Nav Stats Bar */}
          {navStats && navStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center" }}>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(navStats[0])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(navStats[1])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(navStats[2])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(navStats[3])}</Box></Typography>
            </Box>
          )}

          {/* Nav Controls */}
          <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
              {/* Row 1: ROI */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typo.label, fontSize: 10 }}>ROI:</Typography>
                <Select value={roiMode || "off"} onChange={(e) => { const v = e.target.value; if (v !== "off") lastRoiModeRef.current = v; setRoiMode(v); }} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="off">Off</MenuItem>
                  <MenuItem value="circle">Circle</MenuItem>
                  <MenuItem value="square">Square</MenuItem>
                  <MenuItem value="rect">Rect</MenuItem>
                </Select>
                {roiMode !== "off" && (
                  <Select value={roiReduce || "mean"} onChange={(e) => setRoiReduce(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 55, fontSize: 10 }} MenuProps={themedMenuProps}>
                    <MenuItem value="mean">Mean</MenuItem>
                    <MenuItem value="max">Max</MenuItem>
                    <MenuItem value="min">Min</MenuItem>
                    <MenuItem value="sum">Sum</MenuItem>
                  </Select>
                )}
                {roiMode !== "off" && (roiMode === "circle" || roiMode === "square") && (
                  <>
                    <Slider
                      value={roiRadius || 5}
                      onChange={(_, v) => setRoiRadius(v as number)}
                      min={1}
                      max={Math.min(navRows, navCols) / 2}
                      size="small"
                      sx={{ width: 80, mx: 1, "& .MuiSlider-thumb": { width: 14, height: 14 } }}
                    />
                    <Typography sx={{ ...typo.value, fontSize: 10, minWidth: 30 }}>
                      {Math.round(roiRadius || 5)}px
                    </Typography>
                  </>
                )}
                {roiMode === "off" && (
                  <>
                    <Typography sx={{ ...typo.label, fontSize: 10 }}>Snap:</Typography>
                    <Switch checked={snapEnabled} onChange={(_, v) => setSnapEnabled(v)} size="small" sx={switchStyles.small} />
                    {snapEnabled && (
                      <>
                        <Slider value={snapRadius} min={1} max={20} step={1} onChange={(_, v) => { if (typeof v === "number") setSnapRadius(v); }} size="small" sx={{ width: 60, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                        <Typography sx={{ ...typo.value, fontSize: 10 }}>{snapRadius}px</Typography>
                      </>
                    )}
                  </>
                )}
              </Box>
              {/* Row 2: Color + Scale */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typo.label, fontSize: 10 }}>Color:</Typography>
                <Select value={navColormap} onChange={(e) => setNavColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="inferno">Inferno</MenuItem>
                  <MenuItem value="viridis">Viridis</MenuItem>
                  <MenuItem value="plasma">Plasma</MenuItem>
                  <MenuItem value="magma">Magma</MenuItem>
                  <MenuItem value="hot">Hot</MenuItem>
                  <MenuItem value="gray">Gray</MenuItem>
                </Select>
                <Typography sx={{ ...typo.label, fontSize: 10 }}>Scale:</Typography>
                <Select value={navScaleMode} onChange={(e) => setNavScaleMode(e.target.value as "linear" | "log" | "power")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="linear">Lin</MenuItem>
                  <MenuItem value="log">Log</MenuItem>
                  <MenuItem value="power">Pow</MenuItem>
                </Select>
              </Box>
            </Box>
            <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
              <Histogram data={navHistogramData} colormap={navColormap} vminPct={navVminPct} vmaxPct={navVmaxPct} onRangeChange={(min, max) => { setNavVminPct(min); setNavVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme} dataMin={navDataMin} dataMax={navDataMax} />
            </Box>
          </Box>
        </Box>

        {/* ── RIGHT COLUMN: Signal Panel ── */}
        <Box sx={{ width: sigCanvasWidth }}>
          {/* Signal Header */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
            <Typography variant="caption" sx={{ ...typo.label }}>
              Signal
              {roiMode !== "off"
                ? <span style={{ color: themeColors.accent, marginLeft: SPACING.SM }}>(ROI {roiReduce || "mean"})</span>
                : <span style={{ color: themeColors.textMuted, marginLeft: SPACING.SM }}>at ({posRow}, {posCol})</span>
              }
            </Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
              <Typography sx={{ ...typo.label, color: themeColors.textMuted, fontSize: 10 }}>
                {navRows}×{navCols} | {sigRows}×{sigCols}
              </Typography>
              <Typography sx={{ ...typo.label, fontSize: 10 }}>FFT:</Typography>
              <Switch checked={showFft} onChange={(e) => setShowFft(e.target.checked)} size="small" sx={switchStyles.small} />
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={async () => {
                if (!sigCanvasRef.current) return;
                try {
                  const blob = await new Promise<Blob | null>(resolve => sigCanvasRef.current!.toBlob(resolve, "image/png"));
                  if (!blob) return;
                  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
                } catch {
                  sigCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4d_signal.png"); }, "image/png");
                }
              }}>COPY</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={(e) => setSigExportAnchor(e.currentTarget)} disabled={exporting}>{exporting ? "Exporting..." : "Export"}</Button>
              <Menu anchorEl={sigExportAnchor} open={Boolean(sigExportAnchor)} onClose={() => setSigExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                <MenuItem onClick={() => handleSigExportFigure(true)} sx={{ fontSize: 12 }}>Figure + colorbar</MenuItem>
                <MenuItem onClick={() => handleSigExportFigure(false)} sx={{ fontSize: 12 }}>Figure</MenuItem>
                <MenuItem onClick={handleSigExportPng} sx={{ fontSize: 12 }}>PNG (current frame)</MenuItem>
                {pathLength > 0 && <MenuItem onClick={handleSigExportGif} sx={{ fontSize: 12 }}>GIF (path animation)</MenuItem>}
              </Menu>
              <Button size="small" sx={compactButton} disabled={sigZoom === 1 && sigPanX === 0 && sigPanY === 0} onClick={() => { setSigZoom(1); setSigPanX(0); setSigPanY(0); }}>Reset</Button>
            </Stack>
          </Stack>

          {/* Signal Canvas */}
          <Box sx={{ ...container.imageBox, width: sigCanvasWidth, height: sigCanvasHeight }}>
            <canvas ref={sigCanvasRef} width={sigCols} height={sigRows} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
            <canvas
              ref={sigOverlayRef} width={sigCols} height={sigRows}
              onMouseDown={handleSigMouseDown} onMouseMove={handleSigMouseMove}
              onMouseUp={handleSigMouseUp} onMouseLeave={handleSigMouseLeave}
              onWheel={createZoomHandler(setSigZoom, setSigPanX, setSigPanY, sigZoom, sigPanX, sigPanY, sigOverlayRef)}
              onDoubleClick={handleSigDoubleClick}
              style={{ position: "absolute", width: "100%", height: "100%", cursor: profileActive ? "crosshair" : isDraggingSig ? "grabbing" : "grab" }}
            />
            <canvas ref={sigUiRef} width={sigCanvasWidth * DPR} height={sigCanvasHeight * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
            {cursorInfo && cursorInfo.panel === "sig" && (
              <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                  ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                </Typography>
              </Box>
            )}
            <Box onMouseDown={handleResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: 1 } }} />
          </Box>

          {/* Signal Stats Bar */}
          {sigStats && sigStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center" }}>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(sigStats[0])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(sigStats[1])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(sigStats[2])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(sigStats[3])}</Box></Typography>
            </Box>
          )}

          {/* Profile sparkline */}
          {profileActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, maxWidth: sigCanvasWidth, boxSizing: "border-box" }}>
              <canvas
                ref={profileCanvasRef}
                style={{ width: sigCanvasWidth, height: 76, display: "block", border: `1px solid ${themeColors.border}` }}
              />
            </Box>
          )}

          {/* Signal Controls */}
          <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
              {/* Row 1: Profile */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typo.label, fontSize: 10 }}>Profile:</Typography>
                <Switch checked={profileActive} onChange={(e) => {
                  const on = e.target.checked;
                  setProfileActive(on);
                  if (!on) { setProfileLine([]); setProfileData(null); }
                }} size="small" sx={switchStyles.small} />
                {profileActive && profileWidth > 1 && (
                  <>
                    <Typography sx={{ ...typo.value, fontSize: 10 }}>w={profileWidth}</Typography>
                    <Slider value={profileWidth} min={1} max={20} step={1} onChange={(_, v) => { if (typeof v === "number") setProfileWidth(v); }} size="small" sx={{ width: 50, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                  </>
                )}
              </Box>
              {/* Row 2: Color + Scale */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typo.label, fontSize: 10 }}>Color:</Typography>
                <Select value={sigColormap} onChange={(e) => setSigColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="inferno">Inferno</MenuItem>
                  <MenuItem value="viridis">Viridis</MenuItem>
                  <MenuItem value="plasma">Plasma</MenuItem>
                  <MenuItem value="magma">Magma</MenuItem>
                  <MenuItem value="hot">Hot</MenuItem>
                  <MenuItem value="gray">Gray</MenuItem>
                </Select>
                <Typography sx={{ ...typo.label, fontSize: 10 }}>Scale:</Typography>
                <Select value={sigScaleMode} onChange={(e) => setSigScaleMode(e.target.value as "linear" | "log" | "power")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="linear">Lin</MenuItem>
                  <MenuItem value="log">Log</MenuItem>
                  <MenuItem value="power">Pow</MenuItem>
                </Select>
              </Box>
            </Box>
            <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
              <Histogram data={sigHistogramData} colormap={sigColormap} vminPct={sigVminPct} vmaxPct={sigVmaxPct} onRangeChange={(min, max) => { setSigVminPct(min); setSigVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme} dataMin={sigDataMin} dataMax={sigDataMax} />
            </Box>
          </Box>
        </Box>

        {/* ── THIRD COLUMN: FFT Panel ── */}
        {showFft && (
          <Box sx={{ width: sigCanvasWidth }}>
            {/* FFT Header */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
              <Typography variant="caption" sx={{ ...typo.label }}>
                FFT (Signal)
              </Typography>
              <Stack direction="row" spacing={`${SPACING.SM}px`}>
                <Button size="small" sx={compactButton} onClick={() => {
                  if (!fftCanvasRef.current) return;
                  fftCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4d_fft.png"); }, "image/png");
                }}>Export</Button>
                <Button size="small" sx={compactButton} disabled={fftZoom === 1 && fftPanX === 0 && fftPanY === 0} onClick={() => { setFftZoom(1); setFftPanX(0); setFftPanY(0); }}>Reset</Button>
              </Stack>
            </Stack>

            {/* FFT Canvas */}
            <Box sx={{ ...container.imageBox, width: sigCanvasWidth, height: sigCanvasHeight }}>
              <canvas ref={fftCanvasRef} width={sigCols} height={sigRows} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
              <canvas
                ref={fftOverlayRef} width={sigCols} height={sigRows}
                onMouseDown={handleFftMouseDown} onMouseMove={handleFftMouseMove}
                onMouseUp={handleFftMouseUp} onMouseLeave={handleFftMouseLeave}
                onWheel={createZoomHandler(setFftZoom, setFftPanX, setFftPanY, fftZoom, fftPanX, fftPanY, fftOverlayRef)}
                onDoubleClick={handleFftDoubleClick}
                style={{ position: "absolute", width: "100%", height: "100%", cursor: isDraggingFft ? "grabbing" : "grab" }}
              />
              <canvas ref={fftUiRef} width={sigCanvasWidth * DPR} height={sigCanvasHeight * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
              <Box onMouseDown={handleResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: 1 } }} />
            </Box>

            {/* FFT Stats Bar */}
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center" }}>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats.mean)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats.min)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats.max)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats.std)}</Box></Typography>
            </Box>

            {/* FFT Controls */}
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typo.label, fontSize: 10 }}>Auto:</Typography>
                  <Switch checked={fftAuto} onChange={(e) => setFftAuto(e.target.checked)} size="small" sx={switchStyles.small} />
                  <Typography sx={{ ...typo.label, fontSize: 10 }}>Color:</Typography>
                  <Select value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                    <MenuItem value="inferno">Inferno</MenuItem>
                    <MenuItem value="viridis">Viridis</MenuItem>
                    <MenuItem value="plasma">Plasma</MenuItem>
                    <MenuItem value="magma">Magma</MenuItem>
                    <MenuItem value="hot">Hot</MenuItem>
                    <MenuItem value="gray">Gray</MenuItem>
                  </Select>
                  <Typography sx={{ ...typo.label, fontSize: 10 }}>Scale:</Typography>
                  <Select value={fftLogScale ? "log" : "linear"} onChange={(e) => setFftLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                  </Select>
                </Box>
              </Box>
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
                <Histogram data={fftHistogramData} colormap={fftColormap} vminPct={fftVminPct} vmaxPct={fftVmaxPct} onRangeChange={(min, max) => { setFftVminPct(min); setFftVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme} dataMin={fftDataRange.min} dataMax={fftDataRange.max} />
              </Box>
            </Box>
          </Box>
        )}
      </Stack>

      {/* Playback bar (only when path is set) */}
      {pathLength > 0 && (
        <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
          <Stack direction="row" spacing={0} sx={{ flexShrink: 0 }}>
            <IconButton size="small" onClick={() => setPathPlaying(!pathPlaying)} sx={{ color: themeColors.accent, p: 0.25 }}>
              {pathPlaying ? <PauseIcon sx={{ fontSize: 18 }} /> : <PlayArrowIcon sx={{ fontSize: 18 }} />}
            </IconButton>
            <IconButton size="small" onClick={() => { setPathPlaying(false); setPathIndex(0); }} sx={{ color: themeColors.textMuted, p: 0.25 }}>
              <StopIcon sx={{ fontSize: 16 }} />
            </IconButton>
          </Stack>
          <Slider value={pathIndex} onChange={(_, v) => { setPathPlaying(false); setPathIndex(v as number); }} min={0} max={Math.max(0, pathLength - 1)} size="small" sx={{ flex: 1, minWidth: 60, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
          <Typography sx={{ ...typo.value, minWidth: 50, textAlign: "right", flexShrink: 0 }}>{pathIndex + 1}/{pathLength}</Typography>
          <Typography sx={{ ...typo.label, fontSize: 10 }}>Loop:</Typography>
          <Switch checked={pathLoop} onChange={() => { model.set("path_loop", !pathLoop); model.save_changes(); }} size="small" sx={switchStyles.small} />
        </Box>
      )}
    </Box>
  );
}

export const render = createRender(Show4D);
