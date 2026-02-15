import * as React from "react";
import { createRender, useModelState, useModel } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Tooltip from "@mui/material/Tooltip";
import "./styles.css";
import { useTheme } from "../theme";
import { COLORMAPS, applyColormap } from "../colormaps";
import { drawScaleBarHiDPI } from "../scalebar";
import { findDataRange, sliderRange } from "../stats";
import { formatNumber } from "../format";
import { computeHistogramFromBytes } from "../histogram";

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const CANVAS_SIZE = 450;
const RESIZE_HIT_AREA_PX = 10;
const CIRCLE_HANDLE_ANGLE = 0.707;

// ============================================================================
// UI Styles
// ============================================================================
const typography = {
  label: { color: "#aaa", fontSize: 11 },
  value: { color: "#888", fontSize: 10, fontFamily: "monospace" },
  title: { color: "#0af", fontWeight: "bold" as const },
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
  posX: number,
  posY: number,
  zoom: number,
  panX: number,
  panY: number,
  imageWidth: number,
  imageHeight: number,
  isDragging: boolean
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const scaleX = cssWidth / imageWidth;
  const scaleY = cssHeight / imageHeight;

  const screenX = posY * zoom * scaleX + panX * scaleX;
  const screenY = posX * zoom * scaleY + panY * scaleY;

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
// Main Component
// ============================================================================
function Show4D() {
  const model = useModel();
  const { themeInfo, colors: themeColors } = useTheme();
  const DPR = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;

  // ── Model State ──
  const [navX] = useModelState<number>("nav_x");
  const [navY] = useModelState<number>("nav_y");
  const [sigX] = useModelState<number>("sig_x");
  const [sigY] = useModelState<number>("sig_y");
  const [posX, setPosX] = useModelState<number>("pos_x");
  const [posY, setPosY] = useModelState<number>("pos_y");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [navImageBytes] = useModelState<DataView>("nav_image_bytes");
  const [navDataMin] = useModelState<number>("nav_data_min");
  const [navDataMax] = useModelState<number>("nav_data_max");
  const [sigDataMin] = useModelState<number>("sig_data_min");
  const [sigDataMax] = useModelState<number>("sig_data_max");
  const [roiMode, setRoiMode] = useModelState<string>("roi_mode");
  const [roiReduce, setRoiReduce] = useModelState<string>("roi_reduce");
  const [roiCenterX] = useModelState<number>("roi_center_x");
  const [roiCenterY] = useModelState<number>("roi_center_y");
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

  // ── Local State ──
  const [localPosX, setLocalPosX] = React.useState(posX);
  const [localPosY, setLocalPosY] = React.useState(posY);
  const [isDraggingNav, setIsDraggingNav] = React.useState(false);
  const [isDraggingRoi, setIsDraggingRoi] = React.useState(false);
  const [isDraggingRoiResize, setIsDraggingRoiResize] = React.useState(false);
  const [isHoveringRoiResize, setIsHoveringRoiResize] = React.useState(false);
  const [localRoiCenterX, setLocalRoiCenterX] = React.useState(roiCenterX);
  const [localRoiCenterY, setLocalRoiCenterY] = React.useState(roiCenterY);

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

  // ROI toggle memory
  const lastRoiModeRef = React.useRef<string>("circle");

  // Cursor readout
  const [cursorInfo, setCursorInfo] = React.useState<{ x: number; y: number; value: number; panel: string } | null>(null);

  // Aspect-ratio-aware canvas sizes
  const navCanvasWidth = navX > navY ? Math.round(canvasSize * (navY / navX)) : canvasSize;
  const navCanvasHeight = navY > navX ? Math.round(canvasSize * (navX / navY)) : canvasSize;
  const sigCanvasWidth = sigX > sigY ? Math.round(canvasSize * (sigY / sigX)) : canvasSize;
  const sigCanvasHeight = sigY > sigX ? Math.round(canvasSize * (sigX / sigY)) : canvasSize;

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
    if (!isDraggingNav) { setLocalPosX(posX); setLocalPosY(posY); }
  }, [posX, posY, isDraggingNav]);

  React.useEffect(() => {
    if (!isDraggingRoi && !isDraggingRoiResize) {
      setLocalRoiCenterX(roiCenterX);
      setLocalRoiCenterY(roiCenterY);
    }
  }, [roiCenterX, roiCenterY, isDraggingRoi, isDraggingRoiResize]);

  // ── Prevent scroll on canvases ──
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const overlays = [navOverlayRef.current, sigOverlayRef.current];
    overlays.forEach(el => el?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => overlays.forEach(el => el?.removeEventListener("wheel", preventDefault));
  }, []);

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

    const width = navY;
    const height = navX;
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
  }, [navImageBytes, navColormap, navVminPct, navVmaxPct, navScaleMode, navPowerExp, navZoom, navPanX, navPanY, navX, navY]);

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

    const width = sigY;
    const height = sigX;
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
  }, [frameBytes, sigColormap, sigVminPct, sigVmaxPct, sigScaleMode, sigPowerExp, sigZoom, sigPanX, sigPanY, sigX, sigY]);

  // ── Nav HiDPI UI overlay ──
  React.useEffect(() => {
    if (!navUiRef.current) return;
    const navUnit = navPixelSize > 0 ? navPixelUnit : "px";
    const navPxSize = navPixelSize > 0 ? navPixelSize : 1;
    drawScaleBarHiDPI(navUiRef.current, DPR, navZoom, navPxSize, navUnit, navY);

    if (roiMode === "off") {
      drawPositionMarker(navUiRef.current, DPR, localPosX, localPosY, navZoom, navPanX, navPanY, navY, navX, isDraggingNav);
    } else {
      drawNavRoiOverlay(
        navUiRef.current, DPR, roiMode,
        localRoiCenterX, localRoiCenterY, roiRadius, roiWidth, roiHeight,
        navZoom, navPanX, navPanY, navY, navX,
        isDraggingRoi, isDraggingRoiResize, isHoveringRoiResize
      );
    }
  }, [navZoom, navPanX, navPanY, navPixelSize, navPixelUnit, navX, navY,
      localPosX, localPosY, isDraggingNav,
      roiMode, localRoiCenterX, localRoiCenterY, roiRadius, roiWidth, roiHeight,
      isDraggingRoi, isDraggingRoiResize, isHoveringRoiResize]);

  // ── Signal HiDPI UI overlay ──
  React.useEffect(() => {
    if (!sigUiRef.current) return;
    const sigUnit = sigPixelSize > 0 ? sigPixelUnit : "px";
    const sigPxSize = sigPixelSize > 0 ? sigPixelSize : 1;
    drawScaleBarHiDPI(sigUiRef.current, DPR, sigZoom, sigPxSize, sigUnit, sigY);
  }, [sigZoom, sigPanX, sigPanY, sigPixelSize, sigPixelUnit, sigX, sigY]);

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
      const handleX = localRoiCenterX + halfH;
      const handleY = localRoiCenterY + halfW;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      const cornerDist = Math.sqrt(halfW ** 2 + halfH ** 2);
      const hitArea = Math.min(RESIZE_HIT_AREA_PX / navZoom, cornerDist * 0.5);
      return dist < hitArea;
    }
    if (roiMode === "circle" || roiMode === "square") {
      const radius = roiRadius || 5;
      const offset = roiMode === "square" ? radius : radius * CIRCLE_HANDLE_ANGLE;
      const handleX = localRoiCenterX + offset;
      const handleY = localRoiCenterY + offset;
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
      setLocalRoiCenterX(imgX);
      setLocalRoiCenterY(imgY);
      const newX = Math.round(Math.max(0, Math.min(navX - 1, imgX)));
      const newY = Math.round(Math.max(0, Math.min(navY - 1, imgY)));
      model.set("roi_center", [newX, newY]);
      model.save_changes();
      return;
    }

    setIsDraggingNav(true);
    setLocalPosX(imgX);
    setLocalPosY(imgY);
    const newX = Math.round(Math.max(0, Math.min(navX - 1, imgX)));
    const newY = Math.round(Math.max(0, Math.min(navY - 1, imgY)));
    model.set("pos_x", newX);
    model.set("pos_y", newY);
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
    if (pxRow >= 0 && pxRow < navX && pxCol >= 0 && pxCol < navY && rawNavImageRef.current) {
      setCursorInfo({ x: pxCol, y: pxRow, value: rawNavImageRef.current[pxRow * navY + pxCol], panel: "nav" });
    } else {
      setCursorInfo(prev => prev?.panel === "nav" ? null : prev);
    }

    // ROI resize dragging
    if (isDraggingRoiResize) {
      const dx = Math.abs(imgX - localRoiCenterX);
      const dy = Math.abs(imgY - localRoiCenterY);
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
      setLocalRoiCenterX(imgX);
      setLocalRoiCenterY(imgY);
      const newX = Math.round(Math.max(0, Math.min(navX - 1, imgX)));
      const newY = Math.round(Math.max(0, Math.min(navY - 1, imgY)));
      model.set("roi_center", [newX, newY]);
      model.save_changes();
      return;
    }

    // Position dragging
    if (!isDraggingNav) return;
    setLocalPosX(imgX);
    setLocalPosY(imgY);
    const newX = Math.round(Math.max(0, Math.min(navX - 1, imgX)));
    const newY = Math.round(Math.max(0, Math.min(navY - 1, imgY)));
    model.set("pos_x", newX);
    model.set("pos_y", newY);
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

  // ── Signal mouse handlers (drag-to-pan only) ──
  const handleSigMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
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
      if (pxRow >= 0 && pxRow < sigX && pxCol >= 0 && pxCol < sigY && frameBytes) {
        const raw = new Float32Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength / 4);
        setCursorInfo({ x: pxCol, y: pxRow, value: raw[pxRow * sigY + pxCol], panel: "sig" });
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

  const handleSigMouseUp = () => { setIsDraggingSig(false); setSigDragStart(null); };
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

  // ── Keyboard shortcuts ──
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      const step = e.shiftKey ? 10 : 1;
      switch (e.key) {
        case "ArrowLeft":
          e.preventDefault();
          setPosY(Math.max(0, posY - step));
          break;
        case "ArrowRight":
          e.preventDefault();
          setPosY(Math.min(navY - 1, posY + step));
          break;
        case "r":
        case "R":
          setNavZoom(1); setNavPanX(0); setNavPanY(0);
          setSigZoom(1); setSigPanX(0); setSigPanY(0);
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
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [posX, posY, navX, navY, setPosX, setPosY, roiMode, setRoiMode]);

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
      <Typography variant="h6" sx={{ ...typography.title, mb: `${SPACING.SM}px` }}>
        {title || "4D Explorer"}
      </Typography>

      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        {/* ── LEFT COLUMN: Navigation Panel ── */}
        <Box sx={{ width: navCanvasWidth }}>
          {/* Nav Header */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
            <Typography variant="caption" sx={{ ...typography.label }}>
              Navigation ({Math.round(localPosX)}, {Math.round(localPosY)})
              <InfoTooltip text={<KeyboardShortcuts items={[["← / →", "Move position"], ["Shift+←/→", "Move ×10"], ["T", "Toggle ROI on/off"], ["R", "Reset zoom"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]]} />} theme={themeInfo.theme} />
            </Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`}>
              <Button size="small" sx={compactButton} disabled={navZoom === 1 && navPanX === 0 && navPanY === 0} onClick={() => { setNavZoom(1); setNavPanX(0); setNavPanY(0); }}>Reset</Button>
            </Stack>
          </Stack>

          {/* Nav Canvas */}
          <Box sx={{ ...container.imageBox, width: navCanvasWidth, height: navCanvasHeight }}>
            <canvas ref={navCanvasRef} width={navY} height={navX} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
            <canvas
              ref={navOverlayRef} width={navY} height={navX}
              onMouseDown={handleNavMouseDown} onMouseMove={handleNavMouseMove}
              onMouseUp={handleNavMouseUp} onMouseLeave={handleNavMouseLeave}
              onWheel={createZoomHandler(setNavZoom, setNavPanX, setNavPanY, navZoom, navPanX, navPanY, navOverlayRef)}
              onDoubleClick={handleNavDoubleClick}
              style={{ position: "absolute", width: "100%", height: "100%", cursor: isHoveringRoiResize || isDraggingRoiResize ? "nwse-resize" : "crosshair" }}
            />
            <canvas ref={navUiRef} width={navCanvasWidth * DPR} height={navCanvasHeight * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
            <Box onMouseDown={handleResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: 1 } }} />
          </Box>

          {/* Nav Stats Bar */}
          {navStats && navStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center" }}>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(navStats[0])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(navStats[1])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(navStats[2])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(navStats[3])}</Box></Typography>
              {cursorInfo && cursorInfo.panel === "nav" && (
                <>
                  <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 14 }} />
                  <Typography sx={{ fontSize: 11, color: themeColors.textMuted, fontFamily: "monospace" }}>
                    ({cursorInfo.x}, {cursorInfo.y}) <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(cursorInfo.value)}</Box>
                  </Typography>
                </>
              )}
            </Box>
          )}

          {/* Nav Controls */}
          <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
              {/* Row 1: ROI */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>ROI:</Typography>
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
                      max={Math.min(navX, navY) / 2}
                      size="small"
                      sx={{ width: 80, mx: 1, "& .MuiSlider-thumb": { width: 14, height: 14 } }}
                    />
                    <Typography sx={{ ...typography.value, fontSize: 10, minWidth: 30 }}>
                      {Math.round(roiRadius || 5)}px
                    </Typography>
                  </>
                )}
              </Box>
              {/* Row 2: Color + Scale */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                <Select value={navColormap} onChange={(e) => setNavColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="inferno">Inferno</MenuItem>
                  <MenuItem value="viridis">Viridis</MenuItem>
                  <MenuItem value="plasma">Plasma</MenuItem>
                  <MenuItem value="magma">Magma</MenuItem>
                  <MenuItem value="hot">Hot</MenuItem>
                  <MenuItem value="gray">Gray</MenuItem>
                </Select>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
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
            <Typography variant="caption" sx={{ ...typography.label }}>
              Signal
              {roiMode !== "off"
                ? <span style={{ color: "#0f0", marginLeft: SPACING.SM }}>(ROI {roiReduce || "mean"})</span>
                : <span style={{ color: "#888", marginLeft: SPACING.SM }}>at ({posX}, {posY})</span>
              }
              <InfoTooltip text={<KeyboardShortcuts items={[["Scroll", "Zoom"], ["Drag", "Pan"], ["Dbl-click", "Reset view"]]} />} theme={themeInfo.theme} />
            </Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`}>
              <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: 10 }}>
                {navX}×{navY} | {sigX}×{sigY}
              </Typography>
              <Button size="small" sx={compactButton} disabled={sigZoom === 1 && sigPanX === 0 && sigPanY === 0} onClick={() => { setSigZoom(1); setSigPanX(0); setSigPanY(0); }}>Reset</Button>
            </Stack>
          </Stack>

          {/* Signal Canvas */}
          <Box sx={{ ...container.imageBox, width: sigCanvasWidth, height: sigCanvasHeight }}>
            <canvas ref={sigCanvasRef} width={sigY} height={sigX} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
            <canvas
              ref={sigOverlayRef} width={sigY} height={sigX}
              onMouseDown={handleSigMouseDown} onMouseMove={handleSigMouseMove}
              onMouseUp={handleSigMouseUp} onMouseLeave={handleSigMouseLeave}
              onWheel={createZoomHandler(setSigZoom, setSigPanX, setSigPanY, sigZoom, sigPanX, sigPanY, sigOverlayRef)}
              onDoubleClick={handleSigDoubleClick}
              style={{ position: "absolute", width: "100%", height: "100%", cursor: isDraggingSig ? "grabbing" : "grab" }}
            />
            <canvas ref={sigUiRef} width={sigCanvasWidth * DPR} height={sigCanvasHeight * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
            <Box onMouseDown={handleResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: 1 } }} />
          </Box>

          {/* Signal Stats Bar */}
          {sigStats && sigStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center" }}>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(sigStats[0])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(sigStats[1])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(sigStats[2])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(sigStats[3])}</Box></Typography>
              {cursorInfo && cursorInfo.panel === "sig" && (
                <>
                  <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 14 }} />
                  <Typography sx={{ fontSize: 11, color: themeColors.textMuted, fontFamily: "monospace" }}>
                    ({cursorInfo.x}, {cursorInfo.y}) <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(cursorInfo.value)}</Box>
                  </Typography>
                </>
              )}
            </Box>
          )}

          {/* Signal Controls */}
          <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                <Select value={sigColormap} onChange={(e) => setSigColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="inferno">Inferno</MenuItem>
                  <MenuItem value="viridis">Viridis</MenuItem>
                  <MenuItem value="plasma">Plasma</MenuItem>
                  <MenuItem value="magma">Magma</MenuItem>
                  <MenuItem value="hot">Hot</MenuItem>
                  <MenuItem value="gray">Gray</MenuItem>
                </Select>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
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
      </Stack>
    </Box>
  );
}

export const render = createRender(Show4D);
