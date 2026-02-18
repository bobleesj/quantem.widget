/// <reference types="@webgpu/types" />
import * as React from "react";
import { createRender, useModelState, useModel } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Menu from "@mui/material/Menu";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Switch from "@mui/material/Switch";
import Tooltip from "@mui/material/Tooltip";
import IconButton from "@mui/material/IconButton";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import StopIcon from "@mui/icons-material/Stop";
import JSZip from "jszip";
import "./styles.css";
import { useTheme } from "../theme";
import { COLORMAPS, applyColormap, renderToOffscreen } from "../colormaps";
import { WebGPUFFT, getWebGPUFFT, fft2d, fftshift, autoEnhanceFFT } from "../webgpu-fft";
import { drawScaleBarHiDPI, drawColorbar, roundToNiceValue, formatScaleLabel, exportFigure } from "../scalebar";
import { findDataRange, sliderRange, computeStats, applyLogScale } from "../stats";
import { downloadBlob, formatNumber, downloadDataView } from "../format";
import { computeHistogramFromBytes } from "../histogram";

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;

// ============================================================================
// UI Styles - component styling helpers
// ============================================================================
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const },
};

const controlPanel = {
  select: { minWidth: 90, fontSize: 11, "& .MuiSelect-select": { py: 0.5 } },
};

const container = {
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "monospace", overflow: "visible" },
  imageBox: { bgcolor: "#000", border: "1px solid #444", overflow: "hidden", position: "relative" as const },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

const switchStyles = {
  small: { '& .MuiSwitch-thumb': { width: 12, height: 12 }, '& .MuiSwitch-switchBase': { padding: '4px' } },
  medium: { '& .MuiSwitch-thumb': { width: 14, height: 14 }, '& .MuiSwitch-switchBase': { padding: '4px' } },
};

// ============================================================================
// Layout Constants - consistent spacing throughout
// ============================================================================
const SPACING = {
  XS: 4,    // Extra small gap
  SM: 8,    // Small gap (default between elements)
  MD: 12,   // Medium gap (between control groups)
  LG: 16,   // Large gap (between major sections)
};

const CANVAS_SIZE = 450;  // Both DP and VI canvases

// Theme-aware ROI colors for DP detector overlay
interface RoiColors {
  stroke: string;
  strokeDragging: string;
  fill: string;
  fillDragging: string;
  handleFill: string;
  innerStroke: string;
  innerStrokeDragging: string;
  innerHandleFill: string;
  textColor: string;
}
const DARK_ROI_COLORS: RoiColors = {
  stroke: "rgba(0, 255, 0, 0.9)",
  strokeDragging: "rgba(255, 255, 0, 0.9)",
  fill: "rgba(0, 255, 0, 0.12)",
  fillDragging: "rgba(255, 255, 0, 0.12)",
  handleFill: "rgba(0, 255, 0, 0.8)",
  innerStroke: "rgba(0, 220, 255, 0.9)",
  innerStrokeDragging: "rgba(255, 200, 0, 0.9)",
  innerHandleFill: "rgba(0, 220, 255, 0.8)",
  textColor: "#0f0",
};
const LIGHT_ROI_COLORS: RoiColors = {
  stroke: "rgba(0, 140, 0, 0.9)",
  strokeDragging: "rgba(200, 160, 0, 0.9)",
  fill: "rgba(0, 140, 0, 0.15)",
  fillDragging: "rgba(200, 160, 0, 0.15)",
  handleFill: "rgba(0, 140, 0, 0.85)",
  innerStroke: "rgba(0, 160, 200, 0.9)",
  innerStrokeDragging: "rgba(200, 160, 0, 0.9)",
  innerHandleFill: "rgba(0, 160, 200, 0.85)",
  textColor: "#0a0",
};

// Interaction constants
const RESIZE_HIT_AREA_PX = 10;
const CIRCLE_HANDLE_ANGLE = 0.707;  // cos(45°)
// Compact button style for Reset/Export
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

// Control row style - bordered container for each row
const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
};

/** Format stat value for display (compact scientific notation for small values) */
function formatStat(value: number): string {
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs < 0.001 || abs >= 10000) {
    return value.toExponential(2);
  }
  if (abs < 0.01) return value.toFixed(4);
  if (abs < 1) return value.toFixed(3);
  return value.toFixed(2);
}


/**
 * Draw VI crosshair on high-DPI canvas (crisp regardless of image resolution)
 * Note: Does NOT clear canvas - should be called after drawScaleBarHiDPI
 */
function drawViPositionMarker(
  canvas: HTMLCanvasElement,
  dpr: number,
  posRow: number,  // Position in image coordinates
  posCol: number,
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

  // Convert image coordinates to CSS pixel coordinates
  const screenX = posCol * zoom * scaleX + panX * scaleX;
  const screenY = posRow * zoom * scaleY + panY * scaleY;

  // Simple crosshair (no circle)
  const crosshairSize = 12;
  const lineWidth = 1.5;

  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(255, 100, 100, 0.9)";
  ctx.lineWidth = lineWidth;

  // Draw crosshair lines only
  ctx.beginPath();
  ctx.moveTo(screenX - crosshairSize, screenY);
  ctx.lineTo(screenX + crosshairSize, screenY);
  ctx.moveTo(screenX, screenY - crosshairSize);
  ctx.lineTo(screenX, screenY + crosshairSize);
  ctx.stroke();

  ctx.restore();
}

/**
 * Draw VI ROI overlay on high-DPI canvas for real-space region selection
 * Note: Does NOT clear canvas - should be called after drawViPositionMarker
 */
function drawViRoiOverlayHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  roiMode: string,
  centerRow: number,
  centerCol: number,
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

  // Convert image coordinates to screen coordinates (row→screenY, col→screenX)
  const screenX = centerCol * zoom * scaleX + panX * scaleX;
  const screenY = centerRow * zoom * scaleY + panY * scaleY;

  const lineWidth = 2.5;
  const crosshairSize = 10;
  const handleRadius = 6;

  ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  // Helper to draw resize handle (purple color for VI ROI to differentiate from DP)
  const drawResizeHandle = (handleX: number, handleY: number) => {
    let handleFill: string;
    let handleStroke: string;

    if (isDraggingResize) {
      handleFill = "rgba(180, 100, 255, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else if (isHoveringResize) {
      handleFill = "rgba(220, 150, 255, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else {
      handleFill = "rgba(160, 80, 255, 0.8)";
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

  // Helper to draw center crosshair (purple/magenta for VI ROI)
  const drawCenterCrosshair = () => {
    ctx.strokeStyle = isDragging ? "rgba(255, 200, 0, 0.9)" : "rgba(180, 80, 255, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(screenX - crosshairSize, screenY);
    ctx.lineTo(screenX + crosshairSize, screenY);
    ctx.moveTo(screenX, screenY - crosshairSize);
    ctx.lineTo(screenX, screenY + crosshairSize);
    ctx.stroke();
  };

  // Purple/magenta color for VI ROI to differentiate from green DP detector
  const strokeColor = isDragging ? "rgba(255, 200, 0, 0.9)" : "rgba(180, 80, 255, 0.9)";
  const fillColor = isDragging ? "rgba(255, 200, 0, 0.15)" : "rgba(180, 80, 255, 0.15)";

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

    // Resize handle at 45° diagonal
    const handleOffsetX = screenRadiusX * CIRCLE_HANDLE_ANGLE;
    const handleOffsetY = screenRadiusY * CIRCLE_HANDLE_ANGLE;
    drawResizeHandle(screenX + handleOffsetX, screenY + handleOffsetY);

  } else if (roiMode === "square" && radius > 0) {
    // Square uses radius as half-size
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

/**
 * Draw DP crosshair on high-DPI canvas (crisp regardless of detector resolution)
 * Note: Does NOT clear canvas - should be called after drawScaleBarHiDPI
 */
function drawDpCrosshairHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  kCol: number,  // Column position in detector coordinates
  kRow: number,  // Row position in detector coordinates
  zoom: number,
  panX: number,
  panY: number,
  detWidth: number,
  detHeight: number,
  isDragging: boolean,
  roiColors: RoiColors = DARK_ROI_COLORS
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  // Use separate X/Y scale factors (canvas stretches to fill container)
  const scaleX = cssWidth / detWidth;
  const scaleY = cssHeight / detHeight;

  // Convert detector coordinates to CSS pixel coordinates
  const screenX = kCol * zoom * scaleX + panX * scaleX;
  const screenY = kRow * zoom * scaleY + panY * scaleY;
  
  // Fixed UI sizes in CSS pixels (consistent with VI crosshair)
  const crosshairSize = 18;
  const lineWidth = 3;
  const dotRadius = 6;
  
  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;
  
  ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
  ctx.lineWidth = lineWidth;
  
  // Draw crosshair
  ctx.beginPath();
  ctx.moveTo(screenX - crosshairSize, screenY);
  ctx.lineTo(screenX + crosshairSize, screenY);
  ctx.moveTo(screenX, screenY - crosshairSize);
  ctx.lineTo(screenX, screenY + crosshairSize);
  ctx.stroke();
  
  // Draw center dot
  ctx.beginPath();
  ctx.arc(screenX, screenY, dotRadius, 0, 2 * Math.PI);
  ctx.stroke();
  
  ctx.restore();
}

/**
 * Draw ROI overlay (circle, square, rect, annular) on high-DPI canvas
 * Note: Does NOT clear canvas - should be called after drawScaleBarHiDPI
 */
function drawRoiOverlayHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  roiMode: string,
  centerCol: number,
  centerRow: number,
  radius: number,
  radiusInner: number,
  roiWidth: number,
  roiHeight: number,
  zoom: number,
  panX: number,
  panY: number,
  detWidth: number,
  detHeight: number,
  isDragging: boolean,
  isDraggingResize: boolean,
  isDraggingResizeInner: boolean,
  isHoveringResize: boolean,
  isHoveringResizeInner: boolean,
  roiColors: RoiColors = DARK_ROI_COLORS
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  // Use separate X/Y scale factors (canvas stretches to fill container)
  const scaleX = cssWidth / detWidth;
  const scaleY = cssHeight / detHeight;

  // Convert detector coordinates to CSS pixel coordinates
  const screenX = centerCol * zoom * scaleX + panX * scaleX;
  const screenY = centerRow * zoom * scaleY + panY * scaleY;
  
  // Fixed UI sizes in CSS pixels
  const lineWidth = 2.5;
  const crosshairSizeSmall = 10;
  const handleRadius = 6;
  
  ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;
  
  // Helper to draw resize handle
  const drawResizeHandle = (handleX: number, handleY: number, isInner: boolean = false) => {
    let handleFill: string;
    let handleStroke: string;
    const dragging = isInner ? isDraggingResizeInner : isDraggingResize;
    const hovering = isInner ? isHoveringResizeInner : isHoveringResize;
    
    if (dragging) {
      handleFill = "rgba(0, 200, 255, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else if (hovering) {
      handleFill = "rgba(255, 100, 100, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else {
      handleFill = isInner ? roiColors.innerHandleFill : roiColors.handleFill;
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
  
  // Helper to draw center crosshair
  const drawCenterCrosshair = () => {
    ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(screenX - crosshairSizeSmall, screenY);
    ctx.lineTo(screenX + crosshairSizeSmall, screenY);
    ctx.moveTo(screenX, screenY - crosshairSizeSmall);
    ctx.lineTo(screenX, screenY + crosshairSizeSmall);
    ctx.stroke();
  };
  
  if (roiMode === "circle" && radius > 0) {
    // Use separate X/Y radii for ellipse (handles non-square detectors)
    const screenRadiusX = radius * zoom * scaleX;
    const screenRadiusY = radius * zoom * scaleY;

    // Draw ellipse (becomes circle if scaleX === scaleY)
    ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusX, screenRadiusY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Semi-transparent fill
    ctx.fillStyle = isDragging ? roiColors.fillDragging : roiColors.fill;
    ctx.fill();

    drawCenterCrosshair();

    // Resize handle at 45° diagonal
    const handleOffsetX = screenRadiusX * CIRCLE_HANDLE_ANGLE;
    const handleOffsetY = screenRadiusY * CIRCLE_HANDLE_ANGLE;
    drawResizeHandle(screenX + handleOffsetX, screenY + handleOffsetY);

  } else if (roiMode === "square" && radius > 0) {
    // Square in detector space uses same half-size in both dimensions
    const screenHalfW = radius * zoom * scaleX;
    const screenHalfH = radius * zoom * scaleY;
    const left = screenX - screenHalfW;
    const top = screenY - screenHalfH;

    ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();

    ctx.fillStyle = isDragging ? roiColors.fillDragging : roiColors.fill;
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);

  } else if (roiMode === "rect" && roiWidth > 0 && roiHeight > 0) {
    const screenHalfW = (roiWidth / 2) * zoom * scaleX;
    const screenHalfH = (roiHeight / 2) * zoom * scaleY;
    const left = screenX - screenHalfW;
    const top = screenY - screenHalfH;

    ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();

    ctx.fillStyle = isDragging ? roiColors.fillDragging : roiColors.fill;
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);

  } else if (roiMode === "annular" && radius > 0) {
    // Use separate X/Y radii for ellipses
    const screenRadiusOuterX = radius * zoom * scaleX;
    const screenRadiusOuterY = radius * zoom * scaleY;
    const screenRadiusInnerX = (radiusInner || 0) * zoom * scaleX;
    const screenRadiusInnerY = (radiusInner || 0) * zoom * scaleY;

    // Outer ellipse
    ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusOuterX, screenRadiusOuterY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Inner ellipse
    ctx.strokeStyle = isDragging ? roiColors.innerStrokeDragging : roiColors.innerStroke;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusInnerX, screenRadiusInnerY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Fill annular region
    ctx.fillStyle = isDragging ? roiColors.fillDragging : roiColors.fill;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusOuterX, screenRadiusOuterY, 0, 0, 2 * Math.PI);
    ctx.ellipse(screenX, screenY, screenRadiusInnerX, screenRadiusInnerY, 0, 0, 2 * Math.PI, true);
    ctx.fill();

    drawCenterCrosshair();

    // Outer handle at 45° diagonal
    const handleOffsetOuterX = screenRadiusOuterX * CIRCLE_HANDLE_ANGLE;
    const handleOffsetOuterY = screenRadiusOuterY * CIRCLE_HANDLE_ANGLE;
    drawResizeHandle(screenX + handleOffsetOuterX, screenY + handleOffsetOuterY);

    // Inner handle at 45° diagonal
    const handleOffsetInnerX = screenRadiusInnerX * CIRCLE_HANDLE_ANGLE;
    const handleOffsetInnerY = screenRadiusInnerY * CIRCLE_HANDLE_ANGLE;
    drawResizeHandle(screenX + handleOffsetInnerX, screenY + handleOffsetInnerY, true);
  }
  
  ctx.restore();
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

/**
 * Info tooltip component - small ⓘ icon with hover tooltip
 */
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

/**
 * Histogram component with integrated vmin/vmax slider and statistics.
 * Shows data distribution with colormap gradient and adjustable clipping.
 */
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
// Line Profile Sampling
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
// Main Component
// ============================================================================
function Show4DSTEM() {
  // Direct model access for batched updates
  const model = useModel();

  // ─────────────────────────────────────────────────────────────────────────
  // Model State (synced with Python)
  // ─────────────────────────────────────────────────────────────────────────
  const [shapeRows] = useModelState<number>("shape_rows");
  const [shapeCols] = useModelState<number>("shape_cols");
  const [detRows] = useModelState<number>("det_rows");
  const [detCols] = useModelState<number>("det_cols");

  const [posRow, setPosRow] = useModelState<number>("pos_row");
  const [posCol, setPosCol] = useModelState<number>("pos_col");
  const [roiCenterCol, setRoiCenterCol] = useModelState<number>("roi_center_col");
  const [roiCenterRow, setRoiCenterRow] = useModelState<number>("roi_center_row");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [kPixelSize] = useModelState<number>("k_pixel_size");
  const [kCalibrated] = useModelState<boolean>("k_calibrated");

  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [virtualImageBytes] = useModelState<DataView>("virtual_image_bytes");

  // ROI state
  const [roiRadius, setRoiRadius] = useModelState<number>("roi_radius");
  const [roiRadiusInner, setRoiRadiusInner] = useModelState<number>("roi_radius_inner");
  const [roiMode, setRoiMode] = useModelState<string>("roi_mode");
  const [roiWidth, setRoiWidth] = useModelState<number>("roi_width");
  const [roiHeight, setRoiHeight] = useModelState<number>("roi_height");

  // Global min/max for DP normalization (from Python)
  const [dpGlobalMin] = useModelState<number>("dp_global_min");
  const [dpGlobalMax] = useModelState<number>("dp_global_max");

  // VI min/max for normalization (from Python)
  const [viDataMin] = useModelState<number>("vi_data_min");
  const [viDataMax] = useModelState<number>("vi_data_max");

  // Detector calibration (for presets)
  const [bfRadius] = useModelState<number>("bf_radius");
  const [centerCol] = useModelState<number>("center_col");
  const [centerRow] = useModelState<number>("center_row");

  // Path animation state
  const [pathPlaying, setPathPlaying] = useModelState<boolean>("path_playing");
  const [pathIndex, setPathIndex] = useModelState<number>("path_index");
  const [pathLength] = useModelState<number>("path_length");
  const [pathIntervalMs] = useModelState<number>("path_interval_ms");
  const [pathLoop] = useModelState<boolean>("path_loop");

  // Profile line state (synced with Python)
  const [profileLine, setProfileLine] = useModelState<{row: number; col: number}[]>("profile_line");
  const [profileWidth, setProfileWidth] = useModelState<number>("profile_width");

  // Auto-detection trigger
  // ─────────────────────────────────────────────────────────────────────────
  // Local State (UI-only, not synced to Python)
  // ─────────────────────────────────────────────────────────────────────────
  const [localKCol, setLocalKCol] = React.useState(roiCenterCol);
  const [localKRow, setLocalKRow] = React.useState(roiCenterRow);
  const [localPosRow, setLocalPosRow] = React.useState(posRow);
  const [localPosCol, setLocalPosCol] = React.useState(posCol);
  const [isDraggingDP, setIsDraggingDP] = React.useState(false);
  const [isDraggingVI, setIsDraggingVI] = React.useState(false);
  const [isDraggingFFT, setIsDraggingFFT] = React.useState(false);
  const [fftDragStart, setFftDragStart] = React.useState<{ x: number, y: number, panX: number, panY: number } | null>(null);
  const [isDraggingResize, setIsDraggingResize] = React.useState(false);
  const [isDraggingResizeInner, setIsDraggingResizeInner] = React.useState(false); // For annular inner handle
  const [isHoveringResize, setIsHoveringResize] = React.useState(false);
  const [isHoveringResizeInner, setIsHoveringResizeInner] = React.useState(false);
  // VI ROI drag/resize states (same pattern as DP)
  const [isDraggingViRoi, setIsDraggingViRoi] = React.useState(false);
  const [isDraggingViRoiResize, setIsDraggingViRoiResize] = React.useState(false);
  const [isHoveringViRoiResize, setIsHoveringViRoiResize] = React.useState(false);
  // Independent colormaps for DP and VI panels
  const [showDpColorbar, setShowDpColorbar] = React.useState(false);
  const [dpColormap, setDpColormap] = React.useState("inferno");
  const [viColormap, setViColormap] = React.useState("inferno");
  // vmin/vmax percentile clipping (0-100)
  const [dpVminPct, setDpVminPct] = React.useState(0);
  const [dpVmaxPct, setDpVmaxPct] = React.useState(100);
  const [viVminPct, setViVminPct] = React.useState(0);
  const [viVmaxPct, setViVmaxPct] = React.useState(100);
  // Scale mode: "linear" | "log" | "power"
  const [dpScaleMode, setDpScaleMode] = React.useState<"linear" | "log" | "power">("linear");
  const dpPowerExp = 0.5;
  const [viScaleMode, setViScaleMode] = React.useState<"linear" | "log" | "power">("linear");
  const viPowerExp = 0.5;

  // VI ROI state (real-space region selection for summed DP) - synced with Python
  const [viRoiMode, setViRoiMode] = useModelState<string>("vi_roi_mode");
  const [viRoiCenterRow, setViRoiCenterRow] = useModelState<number>("vi_roi_center_row");
  const [viRoiCenterCol, setViRoiCenterCol] = useModelState<number>("vi_roi_center_col");
  const [viRoiRadius, setViRoiRadius] = useModelState<number>("vi_roi_radius");
  const [viRoiWidth, setViRoiWidth] = useModelState<number>("vi_roi_width");
  const [viRoiHeight, setViRoiHeight] = useModelState<number>("vi_roi_height");
  // Local VI ROI center for smooth dragging
  const [localViRoiCenterRow, setLocalViRoiCenterRow] = React.useState(viRoiCenterRow || 0);
  const [localViRoiCenterCol, setLocalViRoiCenterCol] = React.useState(viRoiCenterCol || 0);
  const [summedDpBytes] = useModelState<DataView>("summed_dp_bytes");
  const [summedDpCount] = useModelState<number>("summed_dp_count");
  const [dpStats] = useModelState<number[]>("dp_stats");  // [mean, min, max, std]
  const [viStats] = useModelState<number[]>("vi_stats");  // [mean, min, max, std]
  const [showFft, setShowFft] = React.useState(false);  // Hidden by default per feedback

  // Canvas resize state
  const [canvasSize, setCanvasSize] = React.useState(CANVAS_SIZE);
  const [isResizingCanvas, setIsResizingCanvas] = React.useState(false);
  const [resizeCanvasStart, setResizeCanvasStart] = React.useState<{ x: number; y: number; size: number } | null>(null);

  // Export
  const [, setGifExportRequested] = useModelState<boolean>("_gif_export_requested");
  const [gifData] = useModelState<DataView>("_gif_data");
  const [exporting, setExporting] = React.useState(false);
  const [dpExportAnchor, setDpExportAnchor] = React.useState<HTMLElement | null>(null);
  const [viExportAnchor, setViExportAnchor] = React.useState<HTMLElement | null>(null);

  // Cursor readout state
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number; panel: string } | null>(null);

  // DP Line profile state
  const [profileActive, setProfileActive] = React.useState(false);
  const [profileData, setProfileData] = React.useState<Float32Array | null>(null);
  const [profileHeight, setProfileHeight] = React.useState(76);
  const [isResizingProfile, setIsResizingProfile] = React.useState(false);
  const profileResizeStart = React.useRef<{ startY: number; startHeight: number } | null>(null);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const profileBaseImageRef = React.useRef<ImageData | null>(null);
  const profileLayoutRef = React.useRef<{ padLeft: number; plotW: number; padTop: number; plotH: number; gMin: number; gMax: number; totalDist: number; xUnit: string } | null>(null);
  const profilePoints = profileLine || [];
  const rawDpDataRef = React.useRef<Float32Array | null>(null);
  const dpClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const dpDragOffsetRef = React.useRef<{ dRow: number; dCol: number }>({ dRow: 0, dCol: 0 });

  // VI Line profile state
  const [viProfileActive, setViProfileActive] = React.useState(false);
  const [viProfileData, setViProfileData] = React.useState<Float32Array | null>(null);
  const [viProfilePoints, setViProfilePoints] = React.useState<Array<{ row: number; col: number }>>([]);
  const [viProfileHeight, setViProfileHeight] = React.useState(76);
  const [isResizingViProfile, setIsResizingViProfile] = React.useState(false);
  const viProfileResizeStart = React.useRef<{ startY: number; startHeight: number } | null>(null);
  const viProfileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const viProfileBaseImageRef = React.useRef<ImageData | null>(null);
  const viProfileLayoutRef = React.useRef<{ padLeft: number; plotW: number; padTop: number; plotH: number; gMin: number; gMax: number; totalDist: number; xUnit: string } | null>(null);
  const rawViDataRef = React.useRef<Float32Array | null>(null);
  const viClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const viRoiDragOffsetRef = React.useRef<{ dRow: number; dCol: number }>({ dRow: 0, dCol: 0 });

  // Theme detection
  const { themeInfo, colors: themeColors } = useTheme();
  const roiColors = themeInfo.theme === "dark" ? DARK_ROI_COLORS : LIGHT_ROI_COLORS;

  // Themed typography — applies theme colors to module-level font sizes
  const typo = React.useMemo(() => ({
    label: { ...typography.label, color: themeColors.textMuted },
    labelSmall: { ...typography.labelSmall, color: themeColors.textMuted },
    value: { ...typography.value, color: themeColors.textMuted },
    title: { ...typography.title, color: themeColors.accent },
  }), [themeColors]);

  // Compute VI canvas dimensions to respect aspect ratio of rectangular scans
  const viCanvasWidth = shapeRows > shapeCols ? Math.round(canvasSize * (shapeCols / shapeRows)) : canvasSize;
  const viCanvasHeight = shapeCols > shapeRows ? Math.round(canvasSize * (shapeRows / shapeCols)) : canvasSize;

  // Histogram data - use state to ensure re-renders (both are Float32Array now)
  const [dpHistogramData, setDpHistogramData] = React.useState<Float32Array | null>(null);
  const [viHistogramData, setViHistogramData] = React.useState<Float32Array | null>(null);

  // Parse DP frame bytes for histogram (float32 now)
  React.useEffect(() => {
    if (!frameBytes) return;
    // Parse as Float32Array since Python now sends raw float32
    const rawData = new Float32Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength / 4);
    // Store raw data for profile sampling
    if (!rawDpDataRef.current || rawDpDataRef.current.length !== rawData.length) {
      rawDpDataRef.current = new Float32Array(rawData.length);
    }
    rawDpDataRef.current.set(rawData);
    // Apply scale transformation for histogram display
    const scaledData = new Float32Array(rawData.length);
    if (dpScaleMode === "log") {
      for (let i = 0; i < rawData.length; i++) {
        scaledData[i] = Math.log1p(Math.max(0, rawData[i]));
      }
    } else if (dpScaleMode === "power") {
      for (let i = 0; i < rawData.length; i++) {
        scaledData[i] = Math.pow(Math.max(0, rawData[i]), dpPowerExp);
      }
    } else {
      scaledData.set(rawData);
    }
    setDpHistogramData(scaledData);
  }, [frameBytes, dpScaleMode, dpPowerExp]);

  // GPU FFT state
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);

  // Path animation timer
  React.useEffect(() => {
    if (!pathPlaying || pathLength === 0) return;

    const timer = setInterval(() => {
      setPathIndex((prev: number) => {
        const next = prev + 1;
        if (next >= pathLength) {
          if (pathLoop) {
            return 0;  // Loop back to start
          } else {
            setPathPlaying(false);  // Stop at end
            return prev;
          }
        }
        return next;
      });
    }, pathIntervalMs);

    return () => clearInterval(timer);
  }, [pathPlaying, pathLength, pathIntervalMs, pathLoop, setPathIndex, setPathPlaying]);

  // Keyboard shortcuts
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      const step = e.shiftKey ? 10 : 1;

      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault();
          setPosCol(Math.max(0, posCol - step));
          break;
        case 'ArrowRight':
          e.preventDefault();
          setPosCol(Math.min(shapeCols - 1, posCol + step));
          break;
        case ' ':  // Space bar
          e.preventDefault();
          if (pathLength > 0) {
            setPathPlaying(!pathPlaying);
          }
          break;
        case 'r':  // Reset view
        case 'R':
          setDpZoom(1); setDpPanX(0); setDpPanY(0);
          setViZoom(1); setViPanX(0); setViPanY(0);
          setFftZoom(1); setFftPanX(0); setFftPanY(0);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [posRow, posCol, shapeRows, shapeCols, pathPlaying, pathLength, setPosRow, setPosCol, setPathPlaying]);

  // Initialize WebGPU FFT on mount
  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) {
        gpuFFTRef.current = fft;
        setGpuReady(true);
      }
    });
  }, []);

  // Root element ref (theme-aware styling handled via CSS variables)
  const rootRef = React.useRef<HTMLDivElement>(null);

  // Zoom state
  const [dpZoom, setDpZoom] = React.useState(1);
  const [dpPanX, setDpPanX] = React.useState(0);
  const [dpPanY, setDpPanY] = React.useState(0);
  const [viZoom, setViZoom] = React.useState(1);
  const [viPanX, setViPanX] = React.useState(0);
  const [viPanY, setViPanY] = React.useState(0);
  const [fftZoom, setFftZoom] = React.useState(1);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);
  const [fftScaleMode, setFftScaleMode] = React.useState<"linear" | "log" | "power">("linear");
  const [fftColormap, setFftColormap] = React.useState("inferno");
  const [fftAuto, setFftAuto] = React.useState(true);  // Auto: mask DC + 99.9% clipping
  const [fftVminPct, setFftVminPct] = React.useState(0);
  const [fftVmaxPct, setFftVmaxPct] = React.useState(100);
  const [fftStats, setFftStats] = React.useState<number[] | null>(null);  // [mean, min, max, std]
  const [fftHistogramData, setFftHistogramData] = React.useState<Float32Array | null>(null);
  const [fftDataMin, setFftDataMin] = React.useState(0);
  const [fftDataMax, setFftDataMax] = React.useState(1);

  // Sync local state
  React.useEffect(() => {
    if (!isDraggingDP && !isDraggingResize) { setLocalKCol(roiCenterCol); setLocalKRow(roiCenterRow); }
  }, [roiCenterCol, roiCenterRow, isDraggingDP, isDraggingResize]);

  React.useEffect(() => {
    if (!isDraggingVI) { setLocalPosRow(posRow); setLocalPosCol(posCol); }
  }, [posRow, posCol, isDraggingVI]);

  // Sync VI ROI local state
  React.useEffect(() => {
    if (!isDraggingViRoi && !isDraggingViRoiResize) {
      setLocalViRoiCenterRow(viRoiCenterRow || shapeRows / 2);
      setLocalViRoiCenterCol(viRoiCenterCol || shapeCols / 2);
    }
  }, [viRoiCenterRow, viRoiCenterCol, isDraggingViRoi, isDraggingViRoiResize, shapeRows, shapeCols]);

  // Canvas refs
  const dpCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const dpOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const dpUiRef = React.useRef<HTMLCanvasElement>(null);  // High-DPI UI overlay for scale bar
  const dpOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const dpImageDataRef = React.useRef<ImageData | null>(null);
  const virtualCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const virtualOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const viUiRef = React.useRef<HTMLCanvasElement>(null);  // High-DPI UI overlay for scale bar
  const viOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const viImageDataRef = React.useRef<ImageData | null>(null);
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const fftOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const fftOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const fftImageDataRef = React.useRef<ImageData | null>(null);

  // Device pixel ratio for high-DPI UI overlays
  const DPR = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;

  // ─────────────────────────────────────────────────────────────────────────
  // Effects: Canvas Rendering & Animation
  // ─────────────────────────────────────────────────────────────────────────

  // Prevent page scroll when scrolling on canvases
  // Re-run when showFft changes since FFT canvas is conditionally rendered
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const overlays = [dpOverlayRef.current, virtualOverlayRef.current, fftOverlayRef.current];
    overlays.forEach(el => el?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => overlays.forEach(el => el?.removeEventListener("wheel", preventDefault));
  }, [showFft]);

  // Store raw data for filtering/FFT
  const rawVirtualImageRef = React.useRef<Float32Array | null>(null);
  const fftWorkRealRef = React.useRef<Float32Array | null>(null);
  const fftWorkImagRef = React.useRef<Float32Array | null>(null);
  const fftMagnitudeRef = React.useRef<Float32Array | null>(null);

  // Parse virtual image bytes into Float32Array and apply scale for histogram
  React.useEffect(() => {
    if (!virtualImageBytes) return;
    // Parse as Float32Array
    const numFloats = virtualImageBytes.byteLength / 4;
    const rawData = new Float32Array(virtualImageBytes.buffer, virtualImageBytes.byteOffset, numFloats);

    // Store a copy for filtering/FFT (rawData is a view, we need a copy)
    let storedData = rawVirtualImageRef.current;
    if (!storedData || storedData.length !== numFloats) {
      storedData = new Float32Array(numFloats);
      rawVirtualImageRef.current = storedData;
    }
    storedData.set(rawData);

    // Also store for VI profile sampling
    if (!rawViDataRef.current || rawViDataRef.current.length !== numFloats) {
      rawViDataRef.current = new Float32Array(numFloats);
    }
    rawViDataRef.current.set(rawData);

    // Apply scale transformation for histogram display
    const scaledData = new Float32Array(numFloats);
    if (viScaleMode === "log") {
      for (let i = 0; i < numFloats; i++) {
        scaledData[i] = Math.log1p(Math.max(0, rawData[i]));
      }
    } else if (viScaleMode === "power") {
      for (let i = 0; i < numFloats; i++) {
        scaledData[i] = Math.pow(Math.max(0, rawData[i]), viPowerExp);
      }
    } else {
      scaledData.set(rawData);
    }
    setViHistogramData(scaledData);
  }, [virtualImageBytes, viScaleMode, viPowerExp]);

  // Render DP with zoom (use summed DP when VI ROI is active)
  React.useEffect(() => {
    if (!dpCanvasRef.current) return;

    // Determine which bytes to display: summed DP (if VI ROI active) or single frame
    const usesSummedDp = viRoiMode && viRoiMode !== "off" && summedDpBytes && summedDpBytes.byteLength > 0;
    const sourceBytes = usesSummedDp ? summedDpBytes : frameBytes;
    if (!sourceBytes) return;

    const canvas = dpCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const lut = COLORMAPS[dpColormap] || COLORMAPS.inferno;

    // Parse data based on source (summedDp is still uint8, frame is now float32)
    let scaled: Float32Array;
    if (usesSummedDp) {
      // Summed DP is still uint8 from Python
      const bytes = new Uint8Array(sourceBytes.buffer, sourceBytes.byteOffset, sourceBytes.byteLength);
      scaled = new Float32Array(bytes.length);
      for (let i = 0; i < bytes.length; i++) {
        scaled[i] = bytes[i];
      }
    } else {
      // Frame is now float32 from Python - parse and apply scale transformation
      const rawData = new Float32Array(sourceBytes.buffer, sourceBytes.byteOffset, sourceBytes.byteLength / 4);
      scaled = new Float32Array(rawData.length);

      if (dpScaleMode === "log") {
        for (let i = 0; i < rawData.length; i++) {
          scaled[i] = Math.log1p(Math.max(0, rawData[i]));
        }
      } else if (dpScaleMode === "power") {
        for (let i = 0; i < rawData.length; i++) {
          scaled[i] = Math.pow(Math.max(0, rawData[i]), dpPowerExp);
        }
      } else {
        scaled.set(rawData);
      }
    }

    // Compute actual min/max of scaled data for normalization
    const { min: dataMin, max: dataMax } = findDataRange(scaled);

    // Apply vmin/vmax percentile clipping
    const { vmin, vmax } = sliderRange(dataMin, dataMax, dpVminPct, dpVmaxPct);

    let offscreen = dpOffscreenRef.current;
    if (!offscreen) {
      offscreen = document.createElement("canvas");
      dpOffscreenRef.current = offscreen;
    }
    const sizeChanged = offscreen.width !== detCols || offscreen.height !== detRows;
    if (sizeChanged) {
      offscreen.width = detCols;
      offscreen.height = detRows;
      dpImageDataRef.current = null;
    }
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;

    let imgData = dpImageDataRef.current;
    if (!imgData) {
      imgData = offCtx.createImageData(detCols, detRows);
      dpImageDataRef.current = imgData;
    }
    applyColormap(scaled, imgData.data, lut, vmin, vmax);
    offCtx.putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(dpPanX, dpPanY);
    ctx.scale(dpZoom, dpZoom);
    ctx.drawImage(offscreen, 0, 0);
    ctx.restore();
  }, [frameBytes, summedDpBytes, viRoiMode, detRows, detCols, dpColormap, dpVminPct, dpVmaxPct, dpScaleMode, dpPowerExp, dpZoom, dpPanX, dpPanY]);

  // Render DP overlay - just clear (ROI shapes now drawn on high-DPI UI canvas)
  React.useEffect(() => {
    if (!dpOverlayRef.current) return;
    const canvas = dpOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // All visual overlays (crosshair, ROI shapes, scale bar) are now on dpUiRef for crisp rendering
  }, [localKCol, localKRow, isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner, dpZoom, dpPanX, dpPanY, roiMode, roiRadius, roiRadiusInner, roiWidth, roiHeight, detRows, detCols]);

  // Render filtered virtual image
  React.useEffect(() => {
    if (!rawVirtualImageRef.current || !virtualCanvasRef.current) return;
    const canvas = virtualCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = shapeCols;
    const height = shapeRows;

    const renderData = (filtered: Float32Array) => {
      // Normalize and render
      // Apply scale transformation first
      let scaled = filtered;
      if (viScaleMode === "log") {
        scaled = new Float32Array(filtered.length);
        for (let i = 0; i < filtered.length; i++) {
          scaled[i] = Math.log1p(Math.max(0, filtered[i]));
        }
      } else if (viScaleMode === "power") {
        scaled = new Float32Array(filtered.length);
        for (let i = 0; i < filtered.length; i++) {
          scaled[i] = Math.pow(Math.max(0, filtered[i]), viPowerExp);
        }
      }

      // Use Python's pre-computed min/max when valid, fallback to computing from data
      let dataMin: number, dataMax: number;
      const hasValidMinMax = viDataMin !== undefined && viDataMax !== undefined && viDataMax > viDataMin;
      if (hasValidMinMax) {
        // Apply scale transform to Python's values
        if (viScaleMode === "log") {
          dataMin = Math.log1p(Math.max(0, viDataMin));
          dataMax = Math.log1p(Math.max(0, viDataMax));
        } else if (viScaleMode === "power") {
          dataMin = Math.pow(Math.max(0, viDataMin), viPowerExp);
          dataMax = Math.pow(Math.max(0, viDataMax), viPowerExp);
        } else {
          dataMin = viDataMin;
          dataMax = viDataMax;
        }
      } else {
        // Fallback: compute from scaled data
        const r = findDataRange(scaled);
        dataMin = r.min;
        dataMax = r.max;
      }

      // Apply vmin/vmax percentile clipping
      const { vmin, vmax } = sliderRange(dataMin, dataMax, viVminPct, viVmaxPct);

      const lut = COLORMAPS[viColormap] || COLORMAPS.inferno;
      let offscreen = viOffscreenRef.current;
      if (!offscreen) {
        offscreen = document.createElement("canvas");
        viOffscreenRef.current = offscreen;
      }
      const sizeChanged = offscreen.width !== width || offscreen.height !== height;
      if (sizeChanged) {
        offscreen.width = width;
        offscreen.height = height;
        viImageDataRef.current = null;
      }
      const offCtx = offscreen.getContext("2d");
      if (!offCtx) return;

      let imageData = viImageDataRef.current;
      if (!imageData) {
        imageData = offCtx.createImageData(width, height);
        viImageDataRef.current = imageData;
      }
      applyColormap(scaled, imageData.data, lut, vmin, vmax);
      offCtx.putImageData(imageData, 0, 0);

      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.translate(viPanX, viPanY);
      ctx.scale(viZoom, viZoom);
      ctx.drawImage(offscreen, 0, 0);
      ctx.restore();
    };

    if (!rawVirtualImageRef.current) return;
    renderData(rawVirtualImageRef.current);
    // Note: viDataMin/viDataMax intentionally not in deps - they arrive with virtualImageBytes
    // and we have a fallback if they're stale
  }, [virtualImageBytes, shapeRows, shapeCols, viColormap, viVminPct, viVmaxPct, viScaleMode, viPowerExp, viZoom, viPanX, viPanY]);

  // Render virtual image overlay (just clear - crosshair drawn on high-DPI UI canvas)
  React.useEffect(() => {
    if (!virtualOverlayRef.current) return;
    const canvas = virtualOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Crosshair and scale bar now drawn on high-DPI UI canvas (viUiRef)
  }, [localPosRow, localPosCol, isDraggingVI, viZoom, viPanX, viPanY, pixelSize, shapeRows, shapeCols]);

  // Compute FFT (expensive, async — only re-run on data/GPU changes)
  const fftRealRef = React.useRef<Float32Array | null>(null);
  const fftImagRef = React.useRef<Float32Array | null>(null);
  const [fftVersion, setFftVersion] = React.useState(0);

  React.useEffect(() => {
    if (!rawVirtualImageRef.current || !showFft) return;
    let cancelled = false;
    const width = shapeCols;
    const height = shapeRows;
    const sourceData = rawVirtualImageRef.current;

    if (gpuFFTRef.current && gpuReady) {
      const runGpuFFT = async () => {
        const real = sourceData.slice();
        const imag = new Float32Array(real.length);
        const { real: fReal, imag: fImag } = await gpuFFTRef.current!.fft2D(real, imag, width, height, false);
        if (cancelled) return;
        fftshift(fReal, width, height);
        fftshift(fImag, width, height);
        fftRealRef.current = fReal;
        fftImagRef.current = fImag;
        setFftVersion(v => v + 1);
      };
      runGpuFFT();
      return () => { cancelled = true; };
    } else {
      const len = sourceData.length;
      let real = fftWorkRealRef.current;
      if (!real || real.length !== len) { real = new Float32Array(len); fftWorkRealRef.current = real; }
      real.set(sourceData);
      let imag = fftWorkImagRef.current;
      if (!imag || imag.length !== len) { imag = new Float32Array(len); fftWorkImagRef.current = imag; } else { imag.fill(0); }
      fft2d(real, imag, width, height, false);
      fftshift(real, width, height);
      fftshift(imag, width, height);
      fftRealRef.current = real;
      fftImagRef.current = imag;
      setFftVersion(v => v + 1);
    }
  }, [virtualImageBytes, shapeRows, shapeCols, gpuReady, showFft]);

  // Process FFT → magnitude + histogram + colormap rendering (cheap, sync)
  React.useEffect(() => {
    if (!fftRealRef.current || !fftImagRef.current || !fftCanvasRef.current) return;
    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (!showFft) { ctx.clearRect(0, 0, canvas.width, canvas.height); return; }

    const width = shapeCols;
    const height = shapeRows;
    const real = fftRealRef.current;
    const imag = fftImagRef.current;
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;

    // Compute magnitude with scale mode
    let magnitude = fftMagnitudeRef.current;
    if (!magnitude || magnitude.length !== real.length) {
      magnitude = new Float32Array(real.length);
      fftMagnitudeRef.current = magnitude;
    }
    for (let i = 0; i < real.length; i++) {
      const mag = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
      if (fftScaleMode === "log") { magnitude[i] = Math.log1p(mag); }
      else if (fftScaleMode === "power") { magnitude[i] = Math.pow(mag, 0.5); }
      else { magnitude[i] = mag; }
    }

    let displayMin: number, displayMax: number;
    if (fftAuto) {
      ({ min: displayMin, max: displayMax } = autoEnhanceFFT(magnitude, width, height));
    } else {
      ({ min: displayMin, max: displayMax } = findDataRange(magnitude));
    }
    setFftDataMin(displayMin);
    setFftDataMax(displayMax);
    setFftStats([computeStats(magnitude).mean, displayMin, displayMax, computeStats(magnitude).std]);
    setFftHistogramData(magnitude.slice());

    // Render to offscreen canvas
    let offscreen = fftOffscreenRef.current;
    if (!offscreen) { offscreen = document.createElement("canvas"); fftOffscreenRef.current = offscreen; }
    if (offscreen.width !== width || offscreen.height !== height) {
      offscreen.width = width; offscreen.height = height; fftImageDataRef.current = null;
    }
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;
    let imgData = fftImageDataRef.current;
    if (!imgData) { imgData = offCtx.createImageData(width, height); fftImageDataRef.current = imgData; }

    const { vmin, vmax } = sliderRange(displayMin, displayMax, fftVminPct, fftVmaxPct);
    applyColormap(magnitude, imgData.data, lut, vmin, vmax);
    offCtx.putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(fftPanX, fftPanY);
    ctx.scale(fftZoom, fftZoom);
    ctx.drawImage(offscreen, 0, 0);
    ctx.restore();
  }, [showFft, fftVersion, fftScaleMode, fftAuto, fftVminPct, fftVmaxPct, fftColormap, shapeRows, shapeCols, fftZoom, fftPanX, fftPanY]);

  // Render FFT overlay with high-pass filter circle
  React.useEffect(() => {
    if (!fftOverlayRef.current) return;
    const canvas = fftOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, [fftZoom, fftPanX, fftPanY, showFft]);

  // ─────────────────────────────────────────────────────────────────────────
  // High-DPI Scale Bar UI Overlays
  // ─────────────────────────────────────────────────────────────────────────
  
  // DP scale bar + crosshair + ROI overlay + profile line (high-DPI)
  React.useEffect(() => {
    if (!dpUiRef.current) return;
    // Draw scale bar first (clears canvas)
    const kUnit = kCalibrated ? "mrad" : "px";
    drawScaleBarHiDPI(dpUiRef.current, DPR, dpZoom, kPixelSize || 1, kUnit, detCols);
    // Draw ROI overlay (circle, square, rect, annular) or point crosshair
    if (roiMode === "point") {
      drawDpCrosshairHiDPI(dpUiRef.current, DPR, localKCol, localKRow, dpZoom, dpPanX, dpPanY, detCols, detRows, isDraggingDP, roiColors);
    } else {
      drawRoiOverlayHiDPI(
        dpUiRef.current, DPR, roiMode,
        localKCol, localKRow, roiRadius, roiRadiusInner, roiWidth, roiHeight,
        dpZoom, dpPanX, dpPanY, detCols, detRows,
        isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner,
        roiColors
      );
    }

    // Profile line overlay
    if (profileActive && profilePoints.length > 0) {
      const canvas = dpUiRef.current;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.save();
        ctx.scale(DPR, DPR);
        const cssW = canvas.width / DPR;
        const cssH = canvas.height / DPR;
        const scaleX = cssW / detCols;
        const scaleY = cssH / detRows;
        const toScreenX = (col: number) => col * dpZoom * scaleX + dpPanX * scaleX;
        const toScreenY = (row: number) => row * dpZoom * scaleY + dpPanY * scaleY;

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

          // Draw line A->B
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
    }

    // Colorbar overlay
    if (showDpColorbar && rawDpDataRef.current) {
      const canvas = dpUiRef.current;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.save();
        ctx.scale(DPR, DPR);
        const cssW = canvas.width / DPR;
        const cssH = canvas.height / DPR;
        const lut = COLORMAPS[dpColormap] || COLORMAPS.inferno;
        const processed = dpScaleMode === "log" ? applyLogScale(rawDpDataRef.current) : rawDpDataRef.current;
        const { min: dMin, max: dMax } = findDataRange(processed);
        const { vmin, vmax } = sliderRange(dMin, dMax, dpVminPct, dpVmaxPct);
        drawColorbar(ctx, cssW, cssH, lut, vmin, vmax, dpScaleMode === "log");
        ctx.restore();
      }
    }
  }, [dpZoom, dpPanX, dpPanY, kPixelSize, kCalibrated, detRows, detCols, roiMode, roiRadius, roiRadiusInner, roiWidth, roiHeight, localKCol, localKRow, isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner,
      profileActive, profilePoints, profileWidth, themeColors, showDpColorbar, dpColormap, dpScaleMode, dpVminPct, dpVmaxPct, canvasSize, roiColors]);
  
  // VI scale bar + crosshair + ROI + profile lines (high-DPI)
  React.useEffect(() => {
    if (!viUiRef.current) return;
    // Draw scale bar first (clears canvas)
    drawScaleBarHiDPI(viUiRef.current, DPR, viZoom, pixelSize || 1, "Å", shapeCols);
    // Draw crosshair only when ROI is off (ROI replaces the crosshair)
    if (!viRoiMode || viRoiMode === "off") {
      drawViPositionMarker(viUiRef.current, DPR, localPosRow, localPosCol, viZoom, viPanX, viPanY, shapeCols, shapeRows, isDraggingVI);
    } else {
      // Draw VI ROI instead of crosshair
      drawViRoiOverlayHiDPI(
        viUiRef.current, DPR, viRoiMode,
        localViRoiCenterRow, localViRoiCenterCol, viRoiRadius || 5, viRoiWidth || 10, viRoiHeight || 10,
        viZoom, viPanX, viPanY, shapeCols, shapeRows,
        isDraggingViRoi, isDraggingViRoiResize, isHoveringViRoiResize
      );
    }
    // Draw VI profile lines
    if (viProfileActive && viProfilePoints.length > 0) {
      const canvas = viUiRef.current;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        const cssW = canvas.width / DPR;
        const cssH = canvas.height / DPR;
        const scaleX = cssW / shapeCols;
        const scaleY = cssH / shapeRows;
        ctx.save();
        ctx.scale(DPR, DPR);
        ctx.strokeStyle = "#a0f";
        ctx.lineWidth = 2;
        ctx.shadowColor = "rgba(0,0,0,0.5)";
        ctx.shadowBlur = 2;
        if (viProfilePoints.length >= 1) {
          const p0 = viProfilePoints[0];
          const x0 = p0.col * viZoom * scaleX + viPanX * scaleX;
          const y0 = p0.row * viZoom * scaleY + viPanY * scaleY;
          ctx.beginPath();
          ctx.arc(x0, y0, 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = "#fff";
          ctx.fillText("1", x0 + 6, y0 - 6);
        }
        if (viProfilePoints.length === 2) {
          const p0 = viProfilePoints[0], p1 = viProfilePoints[1];
          const x0 = p0.col * viZoom * scaleX + viPanX * scaleX;
          const y0 = p0.row * viZoom * scaleY + viPanY * scaleY;
          const x1 = p1.col * viZoom * scaleX + viPanX * scaleX;
          const y1 = p1.row * viZoom * scaleY + viPanY * scaleY;
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.lineTo(x1, y1);
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(x1, y1, 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = "#fff";
          ctx.fillText("2", x1 + 6, y1 - 6);
        }
        ctx.restore();
      }
    }
  }, [viZoom, viPanX, viPanY, pixelSize, shapeRows, shapeCols, localPosRow, localPosCol, isDraggingVI,
      viRoiMode, localViRoiCenterRow, localViRoiCenterCol, viRoiRadius, viRoiWidth, viRoiHeight,
      isDraggingViRoi, isDraggingViRoiResize, isHoveringViRoiResize, canvasSize, viProfileActive, viProfilePoints]);

  // ── DP Profile computation ──
  React.useEffect(() => {
    if (profilePoints.length === 2 && rawDpDataRef.current) {
      const p0 = profilePoints[0], p1 = profilePoints[1];
      setProfileData(sampleLineProfile(rawDpDataRef.current, detCols, detRows, p0.row, p0.col, p1.row, p1.col, profileWidth));
      if (!profileActive) setProfileActive(true);
    } else {
      setProfileData(null);
    }
  }, [profilePoints, profileWidth, frameBytes]);

  // ── VI Profile computation ──
  React.useEffect(() => {
    if (viProfilePoints.length === 2 && rawViDataRef.current && shapeCols > 0 && shapeRows > 0) {
      const p0 = viProfilePoints[0], p1 = viProfilePoints[1];
      setViProfileData(sampleLineProfile(rawViDataRef.current, shapeCols, shapeRows, p0.row, p0.col, p1.row, p1.col, 1));
    } else {
      setViProfileData(null);
    }
  }, [viProfilePoints, virtualImageBytes, shapeCols, shapeRows]);

  // ── Profile sparkline rendering ──
  React.useEffect(() => {
    const canvas = profileCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvasSize;
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
      ctx.fillText("Click two points on the DP to draw a profile", cssW / 2, cssH / 2);
      profileBaseImageRef.current = null;
      profileLayoutRef.current = null;
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

    // X-axis: calibrated distance
    let totalDist = profileData.length - 1;
    let xUnit = "px";
    if (profilePoints.length === 2) {
      const dx = profilePoints[1].col - profilePoints[0].col;
      const dy = profilePoints[1].row - profilePoints[0].row;
      const distPx = Math.sqrt(dx * dx + dy * dy);
      if (kCalibrated && kPixelSize > 0) {
        totalDist = distPx * kPixelSize;
        xUnit = "mrad";
      } else {
        totalDist = distPx;
      }
    }

    // Draw axes
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // Draw profile curve
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
      const label = kCalibrated ? formatScaleLabel(v, xUnit) : (v % 1 === 0 ? v.toFixed(0) : v.toFixed(1));
      ctx.fillText(i === ticks.length - 1 ? `${label} ${xUnit}` : label, x, tickY + 4);
    }

    // Y-axis min/max labels (left margin)
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(formatNumber(gMax), 2, padTop);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatNumber(gMin), 2, padTop + plotH);

    // Save base image and layout for hover
    profileBaseImageRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    profileLayoutRef.current = { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit };
  }, [profileData, profilePoints, kPixelSize, kCalibrated, themeInfo.theme, themeColors.accent, canvasSize, profileHeight]);

  // DP Profile hover handlers
  const handleProfileMouseMove = React.useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    const layout = profileLayoutRef.current;
    if (!canvas || !base || !layout || !profileData) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const cssX = e.clientX - rect.left;
    const { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit } = layout;
    const range = gMax - gMin || 1;

    // Restore base image
    ctx.putImageData(base, 0, 0);

    if (cssX < padLeft || cssX > padLeft + plotW) return;
    const frac = (cssX - padLeft) / plotW;

    const dpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Vertical crosshair
    const isDark = themeInfo.theme === "dark";
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(cssX, padTop);
    ctx.lineTo(cssX, padTop + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Dot on curve + value
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

  // DP Profile resize handlers
  React.useEffect(() => {
    if (!isResizingProfile) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!profileResizeStart.current) return;
      const deltaY = e.clientY - profileResizeStart.current.startY;
      const newHeight = Math.max(40, Math.min(300, profileResizeStart.current.startHeight + deltaY));
      setProfileHeight(newHeight);
    };
    const handleMouseUp = () => {
      setIsResizingProfile(false);
      profileResizeStart.current = null;
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingProfile]);

  // ── VI Profile sparkline rendering ──
  React.useEffect(() => {
    const canvas = viProfileCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = viCanvasWidth;
    const cssH = viProfileHeight;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.scale(dpr, dpr);

    const isDark = themeInfo.theme === "dark";
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f0f0f0";
    ctx.fillRect(0, 0, cssW, cssH);

    if (!viProfileData || viProfileData.length < 2) {
      ctx.font = "10px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.fillStyle = isDark ? "#555" : "#999";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Click two points on the VI to draw a profile", cssW / 2, cssH / 2);
      viProfileBaseImageRef.current = null;
      viProfileLayoutRef.current = null;
      return;
    }

    const padLeft = 40;
    const padRight = 8;
    const padTop = 6;
    const padBottom = 18;
    const plotW = cssW - padLeft - padRight;
    const plotH = cssH - padTop - padBottom;

    let gMin = Infinity, gMax = -Infinity;
    for (let i = 0; i < viProfileData.length; i++) {
      if (viProfileData[i] < gMin) gMin = viProfileData[i];
      if (viProfileData[i] > gMax) gMax = viProfileData[i];
    }
    const range = gMax - gMin || 1;

    // X-axis: calibrated distance
    let totalDist = viProfileData.length - 1;
    let xUnit = "px";
    if (viProfilePoints.length === 2 && pixelSize > 0) {
      const dx = viProfilePoints[1].col - viProfilePoints[0].col;
      const dy = viProfilePoints[1].row - viProfilePoints[0].row;
      const distPx = Math.sqrt(dx * dx + dy * dy);
      totalDist = distPx * pixelSize;
      xUnit = pixelSize >= 10 ? "nm" : "Å";
      if (xUnit === "nm") totalDist /= 10;
    }

    // Draw axes
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // Draw profile curve
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < viProfileData.length; i++) {
      const x = padLeft + (i / (viProfileData.length - 1)) * plotW;
      const y = padTop + plotH - ((viProfileData[i] - gMin) / range) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

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
      const label = pixelSize > 0 ? formatScaleLabel(v, xUnit) : (v % 1 === 0 ? v.toFixed(0) : v.toFixed(1));
      ctx.fillText(i === ticks.length - 1 ? `${label} ${xUnit}` : label, x, tickY + 4);
    }

    // Y-axis min/max labels (left margin)
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(formatNumber(gMax), 2, padTop);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatNumber(gMin), 2, padTop + plotH);

    // Save base image and layout for hover
    viProfileBaseImageRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    viProfileLayoutRef.current = { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit };
  }, [viProfileData, viProfilePoints, pixelSize, themeInfo.theme, themeColors.accent, viCanvasWidth, viProfileHeight]);

  // VI Profile hover handlers
  const handleViProfileMouseMove = React.useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = viProfileCanvasRef.current;
    const base = viProfileBaseImageRef.current;
    const layout = viProfileLayoutRef.current;
    if (!canvas || !base || !layout || !viProfileData) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const cssX = e.clientX - rect.left;
    const { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit } = layout;
    const range = gMax - gMin || 1;

    // Restore base image
    ctx.putImageData(base, 0, 0);

    if (cssX < padLeft || cssX > padLeft + plotW) return;
    const frac = (cssX - padLeft) / plotW;

    const dpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Vertical crosshair
    const isDark = themeInfo.theme === "dark";
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(cssX, padTop);
    ctx.lineTo(cssX, padTop + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Dot on curve + value
    const dataIdx = Math.min(viProfileData.length - 1, Math.max(0, Math.round(frac * (viProfileData.length - 1))));
    const val = viProfileData[dataIdx];
    const y = padTop + plotH - ((val - gMin) / range) * plotH;
    ctx.fillStyle = themeColors.accent;
    ctx.beginPath();
    ctx.arc(cssX, y, 3, 0, Math.PI * 2);
    ctx.fill();

    // Value readout label
    const dist = frac * totalDist;
    const label = `${formatNumber(val)}  @  ${dist.toFixed(1)} ${xUnit}`;
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

    ctx.restore();
  }, [viProfileData, themeInfo.theme, themeColors.accent]);

  const handleViProfileMouseLeave = React.useCallback(() => {
    const canvas = viProfileCanvasRef.current;
    const base = viProfileBaseImageRef.current;
    if (!canvas || !base) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(base, 0, 0);
  }, []);

  // VI Profile resize handlers
  React.useEffect(() => {
    if (!isResizingViProfile) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!viProfileResizeStart.current) return;
      const deltaY = e.clientY - viProfileResizeStart.current.startY;
      const newHeight = Math.max(40, Math.min(300, viProfileResizeStart.current.startHeight + deltaY));
      setViProfileHeight(newHeight);
    };
    const handleMouseUp = () => {
      setIsResizingViProfile(false);
      viProfileResizeStart.current = null;
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingViProfile]);

  // Generic zoom handler
  const createZoomHandler = (
    setZoom: React.Dispatch<React.SetStateAction<number>>,
    setPanX: React.Dispatch<React.SetStateAction<number>>,
    setPanY: React.Dispatch<React.SetStateAction<number>>,
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
    setPanX(mouseX - (mouseX - panX) * zoomRatio);
    setPanY(mouseY - (mouseY - panY) * zoomRatio);
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Mouse Handlers
  // ─────────────────────────────────────────────────────────────────────────

  // Helper: convert screen-pixel hit radius to image-pixel radius
  // handleRadius=6 CSS px drawn, hit area ~10 CSS px → convert to image coords
  const dpHitRadius = RESIZE_HIT_AREA_PX * Math.max(detCols, detRows) / canvasSize / dpZoom;

  // Helper: check if point is near the outer resize handle
  const isNearResizeHandle = (imgX: number, imgY: number): boolean => {
    if (roiMode === "rect") {
      // For rectangle, check near bottom-right corner
      const handleX = roiCenterCol + roiWidth / 2;
      const handleY = roiCenterRow + roiHeight / 2;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      return dist < dpHitRadius;
    }
    if ((roiMode !== "circle" && roiMode !== "square" && roiMode !== "annular") || !roiRadius) return false;
    const offset = roiMode === "square" ? roiRadius : roiRadius * CIRCLE_HANDLE_ANGLE;
    const handleX = roiCenterCol + offset;
    const handleY = roiCenterRow + offset;
    const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
    return dist < dpHitRadius;
  };

  // Helper: check if point is near the inner resize handle (annular mode only)
  const isNearResizeHandleInner = (imgX: number, imgY: number): boolean => {
    if (roiMode !== "annular" || !roiRadiusInner) return false;
    const offset = roiRadiusInner * CIRCLE_HANDLE_ANGLE;
    const handleX = roiCenterCol + offset;
    const handleY = roiCenterRow + offset;
    const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
    return dist < dpHitRadius;
  };

  // Helper: check if point is near VI ROI resize handle (same logic as DP)
  // Hit area is capped to avoid overlap with center for small ROIs
  const viHitRadius = RESIZE_HIT_AREA_PX * Math.max(shapeRows, shapeCols) / canvasSize / viZoom;
  const isNearViRoiResizeHandle = (imgX: number, imgY: number): boolean => {
    if (!viRoiMode || viRoiMode === "off") return false;
    if (viRoiMode === "rect") {
      const halfH = (viRoiHeight || 10) / 2;
      const halfW = (viRoiWidth || 10) / 2;
      const handleX = localViRoiCenterRow + halfH;
      const handleY = localViRoiCenterCol + halfW;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      const cornerDist = Math.sqrt(halfW ** 2 + halfH ** 2);
      const hitArea = Math.min(viHitRadius, cornerDist * 0.5);
      return dist < hitArea;
    }
    if (viRoiMode === "circle" || viRoiMode === "square") {
      const radius = viRoiRadius || 5;
      const offset = viRoiMode === "square" ? radius : radius * CIRCLE_HANDLE_ANGLE;
      const handleX = localViRoiCenterRow + offset;
      const handleY = localViRoiCenterCol + offset;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      // Cap hit area to 50% of radius so center remains draggable
      const hitArea = Math.min(viHitRadius, radius * 0.5);
      return dist < hitArea;
    }
    return false;
  };

  // Helper: check if point is inside the DP ROI area
  const isInsideDpRoi = (imgX: number, imgY: number): boolean => {
    if (roiMode === "point") return false;
    const dx = imgX - roiCenterCol;
    const dy = imgY - roiCenterRow;
    if (roiMode === "circle") return Math.sqrt(dx * dx + dy * dy) <= (roiRadius || 5);
    if (roiMode === "square") return Math.abs(dx) <= (roiRadius || 5) && Math.abs(dy) <= (roiRadius || 5);
    if (roiMode === "annular") { const d = Math.sqrt(dx * dx + dy * dy); return d <= (roiRadius || 20) && d >= (roiRadiusInner || 5); }
    if (roiMode === "rect") return Math.abs(dx) <= (roiWidth || 10) / 2 && Math.abs(dy) <= (roiHeight || 10) / 2;
    return false;
  };

  // Helper: check if point is inside the VI ROI area
  const isInsideViRoi = (imgX: number, imgY: number): boolean => {
    if (!viRoiMode || viRoiMode === "off") return false;
    const dx = imgY - localViRoiCenterCol;
    const dy = imgX - localViRoiCenterRow;
    if (viRoiMode === "circle") return Math.sqrt(dx * dx + dy * dy) <= (viRoiRadius || 5);
    if (viRoiMode === "square") return Math.abs(dx) <= (viRoiRadius || 5) && Math.abs(dy) <= (viRoiRadius || 5);
    if (viRoiMode === "rect") return Math.abs(dx) <= (viRoiWidth || 10) / 2 && Math.abs(dy) <= (viRoiHeight || 10) / 2;
    return false;
  };

  // Mouse handlers
  const handleDpMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    dpClickStartRef.current = { x: e.clientX, y: e.clientY };
    const canvas = dpOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenX - dpPanX) / dpZoom;
    const imgY = (screenY - dpPanY) / dpZoom;

    // When profile mode is active, don't start ROI dragging — let mouseUp handle clicks
    if (profileActive) {
      setIsDraggingDP(false);
      return;
    }

    // Check if clicking on resize handle (inner first, then outer)
    if (isNearResizeHandleInner(imgX, imgY)) {
      setIsDraggingResizeInner(true);
      return;
    }
    if (isNearResizeHandle(imgX, imgY)) {
      setIsDraggingResize(true);
      return;
    }

    setIsDraggingDP(true);
    // If clicking inside the ROI, drag with offset (grab-and-drag)
    if (roiMode !== "off" && roiMode !== "point" && isInsideDpRoi(imgX, imgY)) {
      dpDragOffsetRef.current = { dRow: imgY - roiCenterRow, dCol: imgX - roiCenterCol };
      return;
    }
    // Clicking outside ROI — teleport center to click position
    dpDragOffsetRef.current = { dRow: 0, dCol: 0 };
    setLocalKCol(imgX); setLocalKRow(imgY);
    // Use compound roi_center trait [row, col] - single observer fires in Python
    const newCol = Math.round(Math.max(0, Math.min(detCols - 1, imgX)));
    const newRow = Math.round(Math.max(0, Math.min(detRows - 1, imgY)));
    model.set("roi_active", true);
    model.set("roi_center", [newRow, newCol]);
    model.save_changes();
  };

  const handleDpMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = dpOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenX - dpPanX) / dpZoom;
    const imgY = (screenY - dpPanY) / dpZoom;

    // Cursor readout: look up raw DP value at pixel position
    const pxCol = Math.floor(imgX);
    const pxRow = Math.floor(imgY);
    if (pxCol >= 0 && pxCol < detCols && pxRow >= 0 && pxRow < detRows && frameBytes) {
      const usesSummedDp = viRoiMode && viRoiMode !== "off" && summedDpBytes && summedDpBytes.byteLength > 0;
      const sourceBytes = usesSummedDp ? summedDpBytes : frameBytes;
      if (usesSummedDp) {
        const bytes = new Uint8Array(sourceBytes.buffer, sourceBytes.byteOffset, sourceBytes.byteLength);
        setCursorInfo({ row: pxRow, col: pxCol, value: bytes[pxRow * detCols + pxCol], panel: "DP" });
      } else {
        const raw = new Float32Array(sourceBytes.buffer, sourceBytes.byteOffset, sourceBytes.byteLength / 4);
        setCursorInfo({ row: pxRow, col: pxCol, value: raw[pxRow * detCols + pxCol], panel: "DP" });
      }
    } else {
      setCursorInfo(null);
    }

    // Handle inner resize dragging (annular mode)
    if (isDraggingResizeInner) {
      const dx = Math.abs(imgX - roiCenterCol);
      const dy = Math.abs(imgY - roiCenterRow);
      const newRadius = Math.sqrt(dx ** 2 + dy ** 2);
      // Inner radius must be less than outer radius
      setRoiRadiusInner(Math.max(1, Math.min(roiRadius - 1, Math.round(newRadius))));
      return;
    }

    // Handle outer resize dragging - use model state center, not local values
    if (isDraggingResize) {
      const dx = Math.abs(imgX - roiCenterCol);
      const dy = Math.abs(imgY - roiCenterRow);
      if (roiMode === "rect") {
        // For rectangle, update width and height independently
        setRoiWidth(Math.max(2, Math.round(dx * 2)));
        setRoiHeight(Math.max(2, Math.round(dy * 2)));
      } else {
        const newRadius = roiMode === "square" ? Math.max(dx, dy) : Math.sqrt(dx ** 2 + dy ** 2);
        // For annular mode, outer radius must be greater than inner radius
        const minRadius = roiMode === "annular" ? (roiRadiusInner || 0) + 1 : 1;
        setRoiRadius(Math.max(minRadius, Math.round(newRadius)));
      }
      return;
    }

    // Check hover state for resize handles
    if (!isDraggingDP) {
      setIsHoveringResizeInner(isNearResizeHandleInner(imgX, imgY));
      setIsHoveringResize(isNearResizeHandle(imgX, imgY));
      return;
    }

    const centerCol = imgX - dpDragOffsetRef.current.dCol;
    const centerRow = imgY - dpDragOffsetRef.current.dRow;
    setLocalKCol(centerCol); setLocalKRow(centerRow);
    // Use compound roi_center trait [row, col] - single observer fires in Python
    const newCol = Math.round(Math.max(0, Math.min(detCols - 1, centerCol)));
    const newRow = Math.round(Math.max(0, Math.min(detRows - 1, centerRow)));
    model.set("roi_center", [newRow, newCol]);
    model.save_changes();
  };

  const handleDpMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // Profile click capture
    if (profileActive && dpClickStartRef.current) {
      const dx = e.clientX - dpClickStartRef.current.x;
      const dy = e.clientY - dpClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const canvas = dpOverlayRef.current;
        if (canvas && rawDpDataRef.current) {
          const rect = canvas.getBoundingClientRect();
          const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
          const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
          const imgCol = (screenX - dpPanX) / dpZoom;
          const imgRow = (screenY - dpPanY) / dpZoom;
          if (imgCol >= 0 && imgCol < detCols && imgRow >= 0 && imgRow < detRows) {
            const pt = { row: imgRow, col: imgCol };
            if (profilePoints.length === 0 || profilePoints.length === 2) {
              setProfileLine([pt]);
              setProfileData(null);
            } else {
              const p0 = profilePoints[0];
              setProfileLine([p0, pt]);
              setProfileData(sampleLineProfile(rawDpDataRef.current, detCols, detRows, p0.row, p0.col, pt.row, pt.col, profileWidth));
            }
          }
        }
      }
    }
    dpClickStartRef.current = null;
    setIsDraggingDP(false); setIsDraggingResize(false); setIsDraggingResizeInner(false);
  };
  const handleDpMouseLeave = () => {
    dpClickStartRef.current = null;
    setIsDraggingDP(false); setIsDraggingResize(false); setIsDraggingResizeInner(false);
    setIsHoveringResize(false); setIsHoveringResizeInner(false);
    setCursorInfo(prev => prev?.panel === "DP" ? null : prev);
  };
  const handleDpDoubleClick = () => { setDpZoom(1); setDpPanX(0); setDpPanY(0); };

  const handleViMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = virtualOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenY - viPanY) / viZoom;
    const imgY = (screenX - viPanX) / viZoom;

    // VI Profile mode - click to set points
    if (viProfileActive) {
      viClickStartRef.current = { x: screenX, y: screenY };
      return;
    }

    // Check if VI ROI mode is active - same logic as DP
    if (viRoiMode && viRoiMode !== "off") {
      // Check if clicking on resize handle
      if (isNearViRoiResizeHandle(imgX, imgY)) {
        setIsDraggingViRoiResize(true);
        return;
      }

      // Grab-and-drag if clicking inside VI ROI, otherwise teleport
      setIsDraggingViRoi(true);
      if (isInsideViRoi(imgX, imgY)) {
        viRoiDragOffsetRef.current = { dRow: imgX - localViRoiCenterRow, dCol: imgY - localViRoiCenterCol };
      } else {
        viRoiDragOffsetRef.current = { dRow: 0, dCol: 0 };
        setLocalViRoiCenterRow(imgX);
        setLocalViRoiCenterCol(imgY);
        setViRoiCenterRow(Math.round(Math.max(0, Math.min(shapeRows - 1, imgX))));
        setViRoiCenterCol(Math.round(Math.max(0, Math.min(shapeCols - 1, imgY))));
      }
      return;
    }

    // Regular position selection (when ROI is off)
    setIsDraggingVI(true);
    setLocalPosRow(imgX); setLocalPosCol(imgY);
    // Batch X and Y updates into a single sync
    const newX = Math.round(Math.max(0, Math.min(shapeRows - 1, imgX)));
    const newY = Math.round(Math.max(0, Math.min(shapeCols - 1, imgY)));
    model.set("pos_row", newX);
    model.set("pos_col", newY);
    model.save_changes();
  };

  const handleViMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = virtualOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenY - viPanY) / viZoom;
    const imgY = (screenX - viPanX) / viZoom;

    // Cursor readout: look up raw VI value at pixel position
    // imgX = row, imgY = col (swapped coordinate convention)
    const pxRow = Math.floor(imgX);
    const pxCol = Math.floor(imgY);
    if (pxRow >= 0 && pxRow < shapeRows && pxCol >= 0 && pxCol < shapeCols && rawVirtualImageRef.current) {
      const raw = rawVirtualImageRef.current;
      setCursorInfo({ row: pxRow, col: pxCol, value: raw[pxRow * shapeCols + pxCol], panel: "VI" });
    } else {
      setCursorInfo(prev => prev?.panel === "VI" ? null : prev);
    }

    // Handle VI ROI resize dragging (same pattern as DP)
    if (isDraggingViRoiResize) {
      const dx = Math.abs(imgX - localViRoiCenterRow);
      const dy = Math.abs(imgY - localViRoiCenterCol);
      if (viRoiMode === "rect") {
        setViRoiWidth(Math.max(2, Math.round(dy * 2)));
        setViRoiHeight(Math.max(2, Math.round(dx * 2)));
      } else if (viRoiMode === "square") {
        const newHalfSize = Math.max(dx, dy);
        setViRoiRadius(Math.max(1, Math.round(newHalfSize)));
      } else {
        // circle
        const newRadius = Math.sqrt(dx ** 2 + dy ** 2);
        setViRoiRadius(Math.max(1, Math.round(newRadius)));
      }
      return;
    }

    // Check hover state for resize handles (same as DP)
    if (!isDraggingViRoi) {
      setIsHoveringViRoiResize(isNearViRoiResizeHandle(imgX, imgY));
      if (viRoiMode && viRoiMode !== "off") return;  // Don't update position when ROI active
    }

    // Handle VI ROI center dragging (same as DP — with offset)
    if (isDraggingViRoi) {
      const centerRow = imgX - viRoiDragOffsetRef.current.dRow;
      const centerCol = imgY - viRoiDragOffsetRef.current.dCol;
      setLocalViRoiCenterRow(centerRow);
      setLocalViRoiCenterCol(centerCol);
      // Batch VI ROI center updates
      const newViX = Math.round(Math.max(0, Math.min(shapeRows - 1, centerRow)));
      const newViY = Math.round(Math.max(0, Math.min(shapeCols - 1, centerCol)));
      model.set("vi_roi_center_row", newViX);
      model.set("vi_roi_center_col", newViY);
      model.save_changes();
      return;
    }

    // Handle regular position dragging (when ROI is off)
    if (!isDraggingVI) return;
    setLocalPosRow(imgX); setLocalPosCol(imgY);
    // Batch position updates into a single sync
    const newX = Math.round(Math.max(0, Math.min(shapeRows - 1, imgX)));
    const newY = Math.round(Math.max(0, Math.min(shapeCols - 1, imgY)));
    model.set("pos_row", newX);
    model.set("pos_col", newY);
    model.save_changes();
  };

  const handleViMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // VI Profile mode - complete point selection
    if (viProfileActive && viClickStartRef.current) {
      const canvas = virtualOverlayRef.current;
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const endX = (e.clientX - rect.left) * (canvas.width / rect.width);
        const endY = (e.clientY - rect.top) * (canvas.height / rect.height);
        const dx = endX - viClickStartRef.current.x;
        const dy = endY - viClickStartRef.current.y;
        const wasDrag = Math.sqrt(dx * dx + dy * dy) > 3;

        if (!wasDrag) {
          // Click to add point
          const imgX = (endY - viPanY) / viZoom;
          const imgY = (endX - viPanX) / viZoom;
          const pt = { row: Math.round(Math.max(0, Math.min(shapeRows - 1, imgX))), col: Math.round(Math.max(0, Math.min(shapeCols - 1, imgY))) };
          if (viProfilePoints.length < 2) {
            setViProfilePoints([...viProfilePoints, pt]);
          } else {
            setViProfilePoints([pt]);
          }
        }
      }
      viClickStartRef.current = null;
    }

    setIsDraggingVI(false);
    setIsDraggingViRoi(false);
    setIsDraggingViRoiResize(false);
  };
  const handleViMouseLeave = () => {
    setIsDraggingVI(false);
    setIsDraggingViRoi(false);
    setIsDraggingViRoiResize(false);
    setIsHoveringViRoiResize(false);
    setCursorInfo(prev => prev?.panel === "VI" ? null : prev);
  };
  const handleViDoubleClick = () => { setViZoom(1); setViPanX(0); setViPanY(0); };
  const handleFftDoubleClick = () => { setFftZoom(1); setFftPanX(0); setFftPanY(0); };

  // FFT drag-to-pan handlers
  const handleFftMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDraggingFFT(true);
    setFftDragStart({ x: e.clientX, y: e.clientY, panX: fftPanX, panY: fftPanY });
  };

  const handleFftMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDraggingFFT || !fftDragStart) return;
    const canvas = fftOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const dx = (e.clientX - fftDragStart.x) * scaleX;
    const dy = (e.clientY - fftDragStart.y) * scaleY;
    setFftPanX(fftDragStart.panX + dx);
    setFftPanY(fftDragStart.panY + dy);
  };

  const handleFftMouseUp = () => { setIsDraggingFFT(false); setFftDragStart(null); };
  const handleFftMouseLeave = () => { setIsDraggingFFT(false); setFftDragStart(null); };

  // ── Canvas resize handlers ──
  const handleCanvasResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizingCanvas(true);
    setResizeCanvasStart({ x: e.clientX, y: e.clientY, size: canvasSize });
  };

  React.useEffect(() => {
    if (!isResizingCanvas) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeCanvasStart) return;
      const delta = Math.max(e.clientX - resizeCanvasStart.x, e.clientY - resizeCanvasStart.y);
      setCanvasSize(Math.max(CANVAS_SIZE, Math.min(800, resizeCanvasStart.size + delta)));
    };
    const handleMouseUp = () => {
      setIsResizingCanvas(false);
      setResizeCanvasStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingCanvas, resizeCanvasStart]);

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────

  // Export DP handler
  const handleExportDP = async () => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const zip = new JSZip();
    const metadata = {
      exported_at: new Date().toISOString(),
      type: "diffraction_pattern",
      scan_position: { row: posRow, col: posCol },
      scan_shape: { rows: shapeRows, cols: shapeCols },
      detector_shape: { rows: detRows, cols: detCols },
      roi: { mode: roiMode, center_col: roiCenterCol, center_row: roiCenterRow, radius_outer: roiRadius, radius_inner: roiRadiusInner },
      display: { colormap: dpColormap, vmin_pct: dpVminPct, vmax_pct: dpVmaxPct, scale_mode: dpScaleMode },
      calibration: { bf_radius: bfRadius, center_col: centerCol, center_row: centerRow, k_pixel_size: kPixelSize, k_calibrated: kCalibrated },
    };
    zip.file("metadata.json", JSON.stringify(metadata, null, 2));
    const canvasToBlob = (canvas: HTMLCanvasElement): Promise<Blob> => new Promise((resolve) => canvas.toBlob((blob) => resolve(blob!), 'image/png'));
    if (dpCanvasRef.current) zip.file("diffraction_pattern.png", await canvasToBlob(dpCanvasRef.current));
    const zipBlob = await zip.generateAsync({ type: "blob" });
    downloadBlob(zipBlob, `dp_export_${timestamp}.zip`);
  };

  // Export VI handler
  const handleExportVI = async () => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const zip = new JSZip();
    const metadata = {
      exported_at: new Date().toISOString(),
      scan_position: { row: posRow, col: posCol },
      scan_shape: { rows: shapeRows, cols: shapeCols },
      detector_shape: { rows: detRows, cols: detCols },
      roi: { mode: roiMode, center_col: roiCenterCol, center_row: roiCenterRow, radius_outer: roiRadius, radius_inner: roiRadiusInner },
      display: { dp_colormap: dpColormap, vi_colormap: viColormap, dp_scale_mode: dpScaleMode, vi_scale_mode: viScaleMode },
      calibration: { bf_radius: bfRadius, center_col: centerCol, center_row: centerRow, pixel_size: pixelSize, k_pixel_size: kPixelSize },
    };
    zip.file("metadata.json", JSON.stringify(metadata, null, 2));
    const canvasToBlob = (canvas: HTMLCanvasElement): Promise<Blob> => new Promise((resolve) => canvas.toBlob((blob) => resolve(blob!), 'image/png'));
    if (virtualCanvasRef.current) zip.file("virtual_image.png", await canvasToBlob(virtualCanvasRef.current));
    if (dpCanvasRef.current) zip.file("diffraction_pattern.png", await canvasToBlob(dpCanvasRef.current));
    if (fftCanvasRef.current) zip.file("fft.png", await canvasToBlob(fftCanvasRef.current));
    const zipBlob = await zip.generateAsync({ type: "blob" });
    downloadBlob(zipBlob, `4dstem_export_${timestamp}.zip`);
  };

  // ── DP Figure Export ──
  const handleDpExportFigure = (withColorbar: boolean) => {
    setDpExportAnchor(null);
    const frameData = rawDpDataRef.current;
    if (!frameData) return;
    const processed = dpScaleMode === "log" ? applyLogScale(frameData) : frameData;
    const lut = COLORMAPS[dpColormap] || COLORMAPS.inferno;
    const { min: dMin, max: dMax } = findDataRange(processed);
    const { vmin, vmax } = sliderRange(dMin, dMax, dpVminPct, dpVmaxPct);
    const offscreen = renderToOffscreen(processed, detCols, detRows, lut, vmin, vmax);
    if (!offscreen) return;
    const kPxAngstrom = kPixelSize > 0 && kCalibrated ? kPixelSize : 0;
    const figCanvas = exportFigure({
      imageCanvas: offscreen,
      title: `DP at (${posRow}, ${posCol})`,
      lut,
      vmin,
      vmax,
      logScale: dpScaleMode === "log",
      pixelSize: kPxAngstrom > 0 ? kPxAngstrom : undefined,
      showColorbar: withColorbar,
      showScaleBar: kPxAngstrom > 0,
    });
    figCanvas.toBlob((blob) => { if (blob) downloadBlob(blob, "show4dstem_dp_figure.png"); }, "image/png");
  };

  const handleDpExportPng = () => {
    setDpExportAnchor(null);
    if (!dpCanvasRef.current) return;
    dpCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4dstem_dp.png"); }, "image/png");
  };

  const handleDpExportGif = () => {
    setDpExportAnchor(null);
    setExporting(true);
    setGifExportRequested(true);
  };

  // ── VI Figure Export ──
  const handleViExportFigure = (withColorbar: boolean) => {
    setViExportAnchor(null);
    if (!virtualCanvasRef.current) return;
    const viCanvas = virtualCanvasRef.current;
    const pixelSizeAngstrom = pixelSize > 0 ? pixelSize : 0;
    const figCanvas = exportFigure({
      imageCanvas: viCanvas,
      title: "Virtual Image",
      showColorbar: withColorbar,
      showScaleBar: pixelSizeAngstrom > 0,
      pixelSize: pixelSizeAngstrom > 0 ? pixelSizeAngstrom : undefined,
    });
    figCanvas.toBlob((blob) => { if (blob) downloadBlob(blob, "show4dstem_vi_figure.png"); }, "image/png");
  };

  const handleViExportPng = () => {
    setViExportAnchor(null);
    if (!virtualCanvasRef.current) return;
    virtualCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4dstem_vi.png"); }, "image/png");
  };

  // Download GIF when data arrives from Python
  React.useEffect(() => {
    if (!gifData || gifData.byteLength === 0) return;
    downloadDataView(gifData, "show4dstem_dp_animation.gif", "image/gif");
    setExporting(false);
  }, [gifData]);


  // Theme-aware select style
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

  return (
    <Box ref={rootRef} className="show4dstem-root" sx={{ p: `${SPACING.LG}px`, bgcolor: themeColors.bg, color: themeColors.text }}>
      {/* HEADER */}
      <Typography variant="h6" sx={{ ...typo.title, mb: `${SPACING.SM}px` }}>
        4D-STEM Explorer
        <InfoTooltip text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
          <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>DP: Diffraction pattern I(kx,ky) at scan position. Drag to move ROI center.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Detector: ROI mask shape — defines which DP pixels are integrated for the virtual image.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>BF/ABF/ADF: Preset detector configurations (bright-field, annular bright-field, annular dark-field).</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Image: Virtual image — integrated intensity within detector ROI at each scan position.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT: Spatial frequency content of the virtual image. Auto masks DC + clips to 99.9th percentile.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Profile: Click two points on DP to draw a line intensity profile.</Typography>
          <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
          <KeyboardShortcuts items={[["← / →", "Move scan position"], ["Shift+←/→", "Move ×10"], ["Space", "Play / pause path"], ["R", "Reset all zoom/pan"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]]} />
        </Box>} theme={themeInfo.theme} />
      </Typography>

      {/* MAIN CONTENT: DP | VI | FFT (three columns when FFT shown) */}
      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        {/* LEFT COLUMN: DP Panel */}
        <Box sx={{ width: canvasSize }}>
          {/* DP Header */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
            <Typography variant="caption" sx={{ ...typo.label }}>
              DP at ({Math.round(localPosRow)}, {Math.round(localPosCol)})
              <span style={{ color: roiColors.textColor, marginLeft: SPACING.SM }}>k: ({Math.round(localKRow)}, {Math.round(localKCol)})</span>
            </Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
              <Typography sx={{ ...typo.label, fontSize: 10 }}>Profile:</Typography>
              <Switch checked={profileActive} onChange={(e) => {
                const on = e.target.checked;
                setProfileActive(on);
                if (!on) { setProfileLine([]); setProfileData(null); }
              }} size="small" sx={switchStyles.small} />
              <Button size="small" sx={compactButton} disabled={dpZoom === 1 && dpPanX === 0 && dpPanY === 0 && roiCenterCol === centerCol && roiCenterRow === centerRow} onClick={() => { setDpZoom(1); setDpPanX(0); setDpPanY(0); setRoiCenterCol(centerCol); setRoiCenterRow(centerRow); }}>Reset</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={async () => {
                if (!dpCanvasRef.current) return;
                try {
                  const blob = await new Promise<Blob | null>(resolve => dpCanvasRef.current!.toBlob(resolve, "image/png"));
                  if (!blob) return;
                  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
                } catch {
                  dpCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4dstem_dp.png"); }, "image/png");
                }
              }}>COPY</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={(e) => setDpExportAnchor(e.currentTarget)} disabled={exporting}>{exporting ? "..." : "Export"}</Button>
              <Menu anchorEl={dpExportAnchor} open={Boolean(dpExportAnchor)} onClose={() => setDpExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                <MenuItem onClick={() => handleDpExportFigure(true)} sx={{ fontSize: 12 }}>Figure + colorbar</MenuItem>
                <MenuItem onClick={() => handleDpExportFigure(false)} sx={{ fontSize: 12 }}>Figure</MenuItem>
                <MenuItem onClick={handleDpExportPng} sx={{ fontSize: 12 }}>PNG</MenuItem>
                <MenuItem onClick={() => { setDpExportAnchor(null); handleExportDP(); }} sx={{ fontSize: 12 }}>ZIP (PNG + metadata)</MenuItem>
                {pathLength > 0 && <MenuItem onClick={handleDpExportGif} sx={{ fontSize: 12 }}>GIF (path animation)</MenuItem>}
              </Menu>
            </Stack>
          </Stack>

          {/* DP Canvas */}
          <Box sx={{ ...container.imageBox, width: canvasSize, height: canvasSize }}>
            <canvas ref={dpCanvasRef} width={detCols} height={detRows} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
            <canvas
              ref={dpOverlayRef} width={detCols} height={detRows}
              onMouseDown={handleDpMouseDown} onMouseMove={handleDpMouseMove}
              onMouseUp={handleDpMouseUp} onMouseLeave={handleDpMouseLeave}
              onWheel={createZoomHandler(setDpZoom, setDpPanX, setDpPanY, dpZoom, dpPanX, dpPanY, dpOverlayRef)}
              onDoubleClick={handleDpDoubleClick}
              style={{ position: "absolute", width: "100%", height: "100%", cursor: isHoveringResize || isDraggingResize ? "nwse-resize" : "crosshair" }}
            />
            <canvas ref={dpUiRef} width={canvasSize * DPR} height={canvasSize * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
            {cursorInfo && cursorInfo.panel === "DP" && (
              <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                  ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                </Typography>
              </Box>
            )}
            <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: 1 } }} />
          </Box>

          {/* DP Stats Bar */}
          {dpStats && dpStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center" }}>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[0])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[1])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[2])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[3])}</Box></Typography>
              <Box sx={{ flex: 1 }} />
              <Typography component="span" onClick={() => { setRoiMode("circle"); setRoiRadius(bfRadius || 10); setRoiCenterCol(centerCol); setRoiCenterRow(centerRow); }} sx={{ color: roiColors.textColor, fontSize: 11, fontWeight: "bold", cursor: "pointer", "&:hover": { textDecoration: "underline" } }}>BF</Typography>
              <Typography component="span" onClick={() => { setRoiMode("annular"); setRoiRadiusInner((bfRadius || 10) * 0.5); setRoiRadius(bfRadius || 10); setRoiCenterCol(centerCol); setRoiCenterRow(centerRow); }} sx={{ color: "#4af", fontSize: 11, fontWeight: "bold", cursor: "pointer", "&:hover": { textDecoration: "underline" } }}>ABF</Typography>
              <Typography component="span" onClick={() => { setRoiMode("annular"); setRoiRadiusInner(bfRadius || 10); setRoiRadius(Math.min((bfRadius || 10) * 3, Math.min(detRows, detCols) / 2 - 2)); setRoiCenterCol(centerCol); setRoiCenterRow(centerRow); }} sx={{ color: "#fa4", fontSize: 11, fontWeight: "bold", cursor: "pointer", "&:hover": { textDecoration: "underline" } }}>ADF</Typography>
            </Box>
          )}

          {/* Profile sparkline */}
          {profileActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, maxWidth: canvasSize, boxSizing: "border-box" }}>
              <canvas
                ref={profileCanvasRef}
                onMouseMove={handleProfileMouseMove}
                onMouseLeave={handleProfileMouseLeave}
                style={{ width: canvasSize, height: profileHeight, display: "block", border: `1px solid ${themeColors.border}`, borderBottom: "none", cursor: "crosshair" }}
              />
              <Box
                onMouseDown={(e) => {
                  setIsResizingProfile(true);
                  profileResizeStart.current = { startY: e.clientY, startHeight: profileHeight };
                }}
                sx={{ width: canvasSize, height: 4, cursor: "ns-resize", borderTop: `1px solid ${themeColors.border}`, borderLeft: `1px solid ${themeColors.border}`, borderRight: `1px solid ${themeColors.border}`, borderBottom: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, "&:hover": { bgcolor: themeColors.accent } }}
              />
            </Box>
          )}

          {/* DP Controls - two rows with histogram on right */}
          <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
            {/* Left: two rows of controls */}
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
              {/* Row 1: Detector + slider */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typo.label, fontSize: 10 }}>Detector:</Typography>
                <Select value={roiMode || "point"} onChange={(e) => setRoiMode(e.target.value)} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="point">Point</MenuItem>
                  <MenuItem value="circle">Circle</MenuItem>
                  <MenuItem value="square">Square</MenuItem>
                  <MenuItem value="rect">Rect</MenuItem>
                  <MenuItem value="annular">Annular</MenuItem>
                </Select>
                {(roiMode === "circle" || roiMode === "square" || roiMode === "annular") && (
                  <>
                    <Slider
                      value={roiMode === "annular" ? [roiRadiusInner, roiRadius] : [roiRadius]}
                      onChange={(_, v) => {
                        if (roiMode === "annular") {
                          const [inner, outer] = v as number[];
                          setRoiRadiusInner(Math.min(inner, outer - 1));
                          setRoiRadius(Math.max(outer, inner + 1));
                        } else {
                          setRoiRadius(v as number);
                        }
                      }}
                      min={1}
                      max={Math.min(detRows, detCols) / 2}
                      size="small"
                      sx={{
                        width: roiMode === "annular" ? 100 : 70,
                        mx: 1,
                        "& .MuiSlider-thumb": { width: 14, height: 14 }
                      }}
                    />
                    <Typography sx={{ ...typo.label, fontSize: 10 }}>
                      {roiMode === "annular" ? `${Math.round(roiRadiusInner)}-${Math.round(roiRadius)}px` : `${Math.round(roiRadius)}px`}
                    </Typography>
                  </>
                )}
              </Box>
              {/* Row 2: Color + Scale + Colorbar */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typo.label, fontSize: 10 }}>Color:</Typography>
                <Select value={dpColormap} onChange={(e) => setDpColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="inferno">Inferno</MenuItem>
                  <MenuItem value="viridis">Viridis</MenuItem>
                  <MenuItem value="plasma">Plasma</MenuItem>
                  <MenuItem value="magma">Magma</MenuItem>
                  <MenuItem value="hot">Hot</MenuItem>
                  <MenuItem value="gray">Gray</MenuItem>
                </Select>
                <Typography sx={{ ...typo.label, fontSize: 10 }}>Scale:</Typography>
                <Select value={dpScaleMode} onChange={(e) => setDpScaleMode(e.target.value as "linear" | "log" | "power")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="linear">Lin</MenuItem>
                  <MenuItem value="log">Log</MenuItem>
                  <MenuItem value="power">Pow</MenuItem>
                </Select>
                <Typography sx={{ ...typo.label, fontSize: 10 }}>Colorbar:</Typography>
                <Switch checked={showDpColorbar} onChange={(e) => setShowDpColorbar(e.target.checked)} size="small" sx={switchStyles.small} />
              </Box>
            </Box>
            {/* Right: Histogram spanning both rows */}
            <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
              <Histogram data={dpHistogramData} colormap={dpColormap} vminPct={dpVminPct} vmaxPct={dpVmaxPct} onRangeChange={(min, max) => { setDpVminPct(min); setDpVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme} dataMin={dpGlobalMin} dataMax={dpGlobalMax} />
            </Box>
          </Box>
        </Box>

        {/* SECOND COLUMN: VI Panel */}
        <Box sx={{ width: viCanvasWidth }}>
          {/* VI Header */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
            <Typography variant="caption" sx={{ ...typo.label }}>Image</Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
              <Typography sx={{ ...typo.label, color: themeColors.textMuted, fontSize: 10 }}>
                {shapeRows}×{shapeCols} | {detRows}×{detCols}
              </Typography>
              <Typography sx={{ ...typo.label, fontSize: 10 }}>FFT:</Typography>
              <Switch checked={showFft} onChange={(e) => setShowFft(e.target.checked)} size="small" sx={switchStyles.small} />
              <Typography sx={{ ...typo.label, fontSize: 10 }}>Profile:</Typography>
              <Switch checked={viProfileActive} onChange={(e) => { setViProfileActive(e.target.checked); if (!e.target.checked) setViProfilePoints([]); }} size="small" sx={switchStyles.small} />
              <Button size="small" sx={compactButton} disabled={viZoom === 1 && viPanX === 0 && viPanY === 0} onClick={() => { setViZoom(1); setViPanX(0); setViPanY(0); }}>Reset</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={async () => {
                if (!virtualCanvasRef.current) return;
                try {
                  const blob = await new Promise<Blob | null>(resolve => virtualCanvasRef.current!.toBlob(resolve, "image/png"));
                  if (!blob) return;
                  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
                } catch {
                  virtualCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4dstem_vi.png"); }, "image/png");
                }
              }}>COPY</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={(e) => setViExportAnchor(e.currentTarget)}>Export</Button>
              <Menu anchorEl={viExportAnchor} open={Boolean(viExportAnchor)} onClose={() => setViExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                <MenuItem onClick={() => handleViExportFigure(true)} sx={{ fontSize: 12 }}>Figure + colorbar</MenuItem>
                <MenuItem onClick={() => handleViExportFigure(false)} sx={{ fontSize: 12 }}>Figure</MenuItem>
                <MenuItem onClick={handleViExportPng} sx={{ fontSize: 12 }}>PNG</MenuItem>
                <MenuItem onClick={() => { setViExportAnchor(null); handleExportVI(); }} sx={{ fontSize: 12 }}>ZIP (all panels + metadata)</MenuItem>
              </Menu>
            </Stack>
          </Stack>

          {/* VI Canvas */}
          <Box sx={{ ...container.imageBox, width: viCanvasWidth, height: viCanvasHeight }}>
            <canvas ref={virtualCanvasRef} width={shapeCols} height={shapeRows} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
            <canvas
              ref={virtualOverlayRef} width={shapeCols} height={shapeRows}
              onMouseDown={handleViMouseDown} onMouseMove={handleViMouseMove}
              onMouseUp={handleViMouseUp} onMouseLeave={handleViMouseLeave}
              onWheel={createZoomHandler(setViZoom, setViPanX, setViPanY, viZoom, viPanX, viPanY, virtualOverlayRef)}
              onDoubleClick={handleViDoubleClick}
              style={{ position: "absolute", width: "100%", height: "100%", cursor: "crosshair" }}
            />
            <canvas ref={viUiRef} width={viCanvasWidth * DPR} height={viCanvasHeight * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
            {cursorInfo && cursorInfo.panel === "VI" && (
              <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                  ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                </Typography>
              </Box>
            )}
            <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: 1 } }} />
          </Box>

          {/* VI Stats Bar */}
          {viStats && viStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center" }}>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[0])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[1])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[2])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[3])}</Box></Typography>
            </Box>
          )}

          {/* VI Profile sparkline */}
          {viProfileActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, maxWidth: viCanvasWidth, boxSizing: "border-box" }}>
              <canvas
                ref={viProfileCanvasRef}
                onMouseMove={handleViProfileMouseMove}
                onMouseLeave={handleViProfileMouseLeave}
                style={{ width: viCanvasWidth, height: viProfileHeight, display: "block", border: `1px solid ${themeColors.border}`, borderBottom: "none", cursor: "crosshair" }}
              />
              <Box
                onMouseDown={(e) => {
                  setIsResizingViProfile(true);
                  viProfileResizeStart.current = { startY: e.clientY, startHeight: viProfileHeight };
                }}
                sx={{ width: viCanvasWidth, height: 4, cursor: "ns-resize", borderTop: `1px solid ${themeColors.border}`, borderLeft: `1px solid ${themeColors.border}`, borderRight: `1px solid ${themeColors.border}`, borderBottom: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, "&:hover": { bgcolor: themeColors.accent } }}
              />
            </Box>
          )}

          {/* VI Controls - Two rows with histogram on right */}
          <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
            {/* Left: Two rows of controls */}
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
              {/* Row 1: ROI selector */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typo.label, fontSize: 10 }}>ROI:</Typography>
                <Select value={viRoiMode || "off"} onChange={(e) => setViRoiMode(e.target.value)} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="off">Off</MenuItem>
                  <MenuItem value="circle">Circle</MenuItem>
                  <MenuItem value="square">Square</MenuItem>
                  <MenuItem value="rect">Rect</MenuItem>
                </Select>
                {viRoiMode && viRoiMode !== "off" && (
                  <>
                    {(viRoiMode === "circle" || viRoiMode === "square") && (
                      <>
                        <Slider
                          value={viRoiRadius || 5}
                          onChange={(_, v) => setViRoiRadius(v as number)}
                          min={1}
                          max={Math.min(shapeRows, shapeCols) / 2}
                          size="small"
                          sx={{ width: 80, mx: 1 }}
                        />
                        <Typography sx={{ ...typo.value, fontSize: 10, minWidth: 30 }}>
                          {Math.round(viRoiRadius || 5)}px
                        </Typography>
                      </>
                    )}
                    {summedDpCount > 0 && (
                      <Typography sx={{ ...typo.label, fontSize: 9, color: "#a6f" }}>
                        {summedDpCount} pos
                      </Typography>
                    )}
                  </>
                )}
              </Box>
              {/* Row 2: Color + Scale */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typo.label, fontSize: 10 }}>Color:</Typography>
                <Select value={viColormap} onChange={(e) => setViColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="inferno">Inferno</MenuItem>
                  <MenuItem value="viridis">Viridis</MenuItem>
                  <MenuItem value="plasma">Plasma</MenuItem>
                  <MenuItem value="magma">Magma</MenuItem>
                  <MenuItem value="hot">Hot</MenuItem>
                  <MenuItem value="gray">Gray</MenuItem>
                </Select>
                <Typography sx={{ ...typo.label, fontSize: 10 }}>Scale:</Typography>
                <Select value={viScaleMode} onChange={(e) => setViScaleMode(e.target.value as "linear" | "log" | "power")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="linear">Lin</MenuItem>
                  <MenuItem value="log">Log</MenuItem>
                  <MenuItem value="power">Pow</MenuItem>
                </Select>
              </Box>
            </Box>
            {/* Right: Histogram spanning both rows */}
            <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
              <Histogram data={viHistogramData} colormap={viColormap} vminPct={viVminPct} vmaxPct={viVmaxPct} onRangeChange={(min, max) => { setViVminPct(min); setViVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme} dataMin={viDataMin} dataMax={viDataMax} />
            </Box>
          </Box>
        </Box>

        {/* THIRD COLUMN: FFT Panel (conditionally shown) */}
        {showFft && (
          <Box sx={{ width: viCanvasWidth }}>
            {/* FFT Header */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
              <Typography variant="caption" sx={{ ...typo.label }}>FFT</Typography>
              <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
                <Button size="small" sx={compactButton} disabled={fftZoom === 1 && fftPanX === 0 && fftPanY === 0} onClick={() => { setFftZoom(1); setFftPanX(0); setFftPanY(0); }}>Reset</Button>
              </Stack>
            </Stack>

            {/* FFT Canvas */}
            <Box sx={{ ...container.imageBox, width: viCanvasWidth, height: viCanvasHeight }}>
              <canvas ref={fftCanvasRef} width={shapeCols} height={shapeRows} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
              <canvas
                ref={fftOverlayRef} width={shapeCols} height={shapeRows}
                onMouseDown={handleFftMouseDown} onMouseMove={handleFftMouseMove}
                onMouseUp={handleFftMouseUp} onMouseLeave={handleFftMouseLeave}
                onWheel={createZoomHandler(setFftZoom, setFftPanX, setFftPanY, fftZoom, fftPanX, fftPanY, fftOverlayRef)}
                onDoubleClick={handleFftDoubleClick}
                style={{ position: "absolute", width: "100%", height: "100%", cursor: isDraggingFFT ? "grabbing" : "grab" }}
              />
              <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: 1 } }} />
            </Box>

            {/* FFT Stats Bar */}
            {fftStats && fftStats.length === 4 && (
              <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2 }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats[0])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats[1])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats[2])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats[3])}</Box></Typography>
              </Box>
            )}

            {/* FFT Controls - Two rows with histogram on right */}
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
              {/* Left: Two rows of controls */}
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: Scale + Clip */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typo.label, fontSize: 10 }}>Scale:</Typography>
                  <Select value={fftScaleMode} onChange={(e) => setFftScaleMode(e.target.value as "linear" | "log" | "power")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                    <MenuItem value="power">Pow</MenuItem>
                  </Select>
                  <Typography sx={{ ...typo.label, fontSize: 10 }}>Auto:</Typography>
                  <Switch checked={fftAuto} onChange={(e) => setFftAuto(e.target.checked)} size="small" sx={switchStyles.small} />
                </Box>
                {/* Row 2: Color */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typo.label, fontSize: 10 }}>Color:</Typography>
                  <Select value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                    <MenuItem value="inferno">Inferno</MenuItem>
                    <MenuItem value="viridis">Viridis</MenuItem>
                    <MenuItem value="plasma">Plasma</MenuItem>
                    <MenuItem value="magma">Magma</MenuItem>
                    <MenuItem value="hot">Hot</MenuItem>
                    <MenuItem value="gray">Gray</MenuItem>
                  </Select>
                </Box>
              </Box>
              {/* Right: Histogram spanning both rows */}
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
                {fftHistogramData && (
                  <Histogram data={fftHistogramData} colormap={fftColormap} vminPct={fftVminPct} vmaxPct={fftVmaxPct} onRangeChange={(min, max) => { setFftVminPct(min); setFftVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme} dataMin={fftDataMin} dataMax={fftDataMax} />
                )}
              </Box>
            </Box>
          </Box>
        )}
      </Stack>

      {/* BOTTOM CONTROLS - Path only (FFT toggle moved to VI panel) */}
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
          <Switch checked={pathLoop} onChange={(_, v) => { model.set("path_loop", v); model.save_changes(); }} size="small" sx={switchStyles.small} />
        </Box>
      )}
    </Box>
  );
}

export const render = createRender(Show4DSTEM);
