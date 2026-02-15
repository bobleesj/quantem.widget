/// <reference types="@webgpu/types" />
import * as React from "react";
import { createRender, useModelState, useModel } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Switch from "@mui/material/Switch";
import Tooltip from "@mui/material/Tooltip";
import JSZip from "jszip";
import "./styles.css";
import { useTheme } from "../theme";
import { COLORMAPS, applyColormap } from "../colormaps";
import { WebGPUFFT, getWebGPUFFT, fft2d, fftshift, autoEnhanceFFT } from "../webgpu-fft";
import { drawScaleBarHiDPI } from "../scalebar";
import { findDataRange, sliderRange, computeStats } from "../stats";
import { downloadBlob, formatNumber } from "../format";

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;

// ============================================================================
// UI Styles - component styling helpers
// ============================================================================
const typography = {
  label: { color: "#aaa", fontSize: 11 },
  labelSmall: { color: "#888", fontSize: 10 },
  value: { color: "#888", fontSize: 10, fontFamily: "monospace" },
  title: { color: "#0af", fontWeight: "bold" as const },
};

const controlPanel = {
  group: { bgcolor: "#222", px: 1.5, py: 0.5, borderRadius: 1, border: "1px solid #444", height: 32 },
  button: { color: "#888", fontSize: 10, cursor: "pointer", "&:hover": { color: "#fff" }, bgcolor: "#222", px: 1, py: 0.25, borderRadius: 0.5, border: "1px solid #444" },
  select: { minWidth: 90, bgcolor: "#333", color: "#fff", fontSize: 11, "& .MuiSelect-select": { py: 0.5 } },
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
  isDragging: boolean
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
  
  ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
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
  isHoveringResizeInner: boolean
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
      handleFill = isInner ? "rgba(0, 220, 255, 0.8)" : "rgba(0, 255, 0, 0.8)";
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
    ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
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
    ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusX, screenRadiusY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Semi-transparent fill
    ctx.fillStyle = isDragging ? "rgba(255, 255, 0, 0.12)" : "rgba(0, 255, 0, 0.12)";
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

    ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();

    ctx.fillStyle = isDragging ? "rgba(255, 255, 0, 0.12)" : "rgba(0, 255, 0, 0.12)";
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);

  } else if (roiMode === "rect" && roiWidth > 0 && roiHeight > 0) {
    const screenHalfW = (roiWidth / 2) * zoom * scaleX;
    const screenHalfH = (roiHeight / 2) * zoom * scaleY;
    const left = screenX - screenHalfW;
    const top = screenY - screenHalfH;

    ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();

    ctx.fillStyle = isDragging ? "rgba(255, 255, 0, 0.12)" : "rgba(0, 255, 0, 0.12)";
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);

  } else if (roiMode === "annular" && radius > 0) {
    // Use separate X/Y radii for ellipses
    const screenRadiusOuterX = radius * zoom * scaleX;
    const screenRadiusOuterY = radius * zoom * scaleY;
    const screenRadiusInnerX = (radiusInner || 0) * zoom * scaleX;
    const screenRadiusInnerY = (radiusInner || 0) * zoom * scaleY;

    // Outer ellipse (green)
    ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusOuterX, screenRadiusOuterY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Inner ellipse (cyan)
    ctx.strokeStyle = isDragging ? "rgba(255, 200, 0, 0.9)" : "rgba(0, 220, 255, 0.9)";
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusInnerX, screenRadiusInnerY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Fill annular region
    ctx.fillStyle = isDragging ? "rgba(255, 255, 0, 0.12)" : "rgba(0, 255, 0, 0.12)";
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

/**
 * Compute histogram from byte data (0-255).
 * Returns 256 bins normalized to 0-1 range.
 */
function computeHistogramFromBytes(data: Uint8Array | Float32Array | null, numBins = 256): number[] {
  if (!data || data.length === 0) {
    return new Array(numBins).fill(0);
  }

  const bins = new Array(numBins).fill(0);

  // For Float32Array, find min/max and bin accordingly
  if (data instanceof Float32Array) {
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (isFinite(v)) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    if (!isFinite(min) || !isFinite(max) || min === max) {
      return bins;
    }
    const range = max - min;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (isFinite(v)) {
        const binIdx = Math.min(numBins - 1, Math.floor(((v - min) / range) * numBins));
        bins[binIdx]++;
      }
    }
  } else {
    // Uint8Array - values are already 0-255
    for (let i = 0; i < data.length; i++) {
      const binIdx = Math.min(numBins - 1, data[i]);
      bins[binIdx]++;
    }
  }

  // Normalize bins to 0-1
  const maxCount = Math.max(...bins);
  if (maxCount > 0) {
    for (let i = 0; i < numBins; i++) {
      bins[i] /= maxCount;
    }
  }

  return bins;
}

interface HistogramProps {
  data: Uint8Array | Float32Array | null;
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

  // Cursor readout state
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number; panel: string } | null>(null);

  // Theme detection
  const { themeInfo, colors: themeColors } = useTheme();

  // Compute VI canvas dimensions to respect aspect ratio of rectangular scans
  // The longer dimension gets CANVAS_SIZE, the shorter scales proportionally
  const viCanvasWidth = shapeRows > shapeCols ? Math.round(CANVAS_SIZE * (shapeCols / shapeRows)) : CANVAS_SIZE;
  const viCanvasHeight = shapeCols > shapeRows ? Math.round(CANVAS_SIZE * (shapeRows / shapeCols)) : CANVAS_SIZE;

  // Histogram data - use state to ensure re-renders (both are Float32Array now)
  const [dpHistogramData, setDpHistogramData] = React.useState<Float32Array | null>(null);
  const [viHistogramData, setViHistogramData] = React.useState<Float32Array | null>(null);

  // Parse DP frame bytes for histogram (float32 now)
  React.useEffect(() => {
    if (!frameBytes) return;
    // Parse as Float32Array since Python now sends raw float32
    const rawData = new Float32Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength / 4);
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

  // Render FFT (WebGPU when available, CPU fallback)
  React.useEffect(() => {
    if (!rawVirtualImageRef.current || !fftCanvasRef.current) return;
    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (!showFft) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const width = shapeCols;
    const height = shapeRows;
    const sourceData = rawVirtualImageRef.current;
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;

    // Helper to render magnitude to canvas
    const renderMagnitude = (real: Float32Array, imag: Float32Array) => {
      // Compute magnitude (log or linear)
      let magnitude = fftMagnitudeRef.current;
      if (!magnitude || magnitude.length !== real.length) {
        magnitude = new Float32Array(real.length);
        fftMagnitudeRef.current = magnitude;
      }
      for (let i = 0; i < real.length; i++) {
        const mag = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
        if (fftScaleMode === "log") {
          magnitude[i] = Math.log1p(mag);
        } else if (fftScaleMode === "power") {
          magnitude[i] = Math.pow(mag, 0.5);  // gamma = 0.5
        } else {
          magnitude[i] = mag;
        }
      }

      let displayMin: number, displayMax: number;
      if (fftAuto) {
        ({ min: displayMin, max: displayMax } = autoEnhanceFFT(magnitude, width, height));
      } else {
        ({ min: displayMin, max: displayMax } = findDataRange(magnitude));
      }
      setFftDataMin(displayMin);
      setFftDataMax(displayMax);

      const { mean, std } = computeStats(magnitude);
      setFftStats([mean, displayMin, displayMax, std]);

      // Store histogram data (copy of magnitude for histogram component)
      setFftHistogramData(magnitude.slice());

      let offscreen = fftOffscreenRef.current;
      if (!offscreen) {
        offscreen = document.createElement("canvas");
        fftOffscreenRef.current = offscreen;
      }
      const sizeChanged = offscreen.width !== width || offscreen.height !== height;
      if (sizeChanged) {
        offscreen.width = width;
        offscreen.height = height;
        fftImageDataRef.current = null;
      }
      const offCtx = offscreen.getContext("2d");
      if (!offCtx) return;

      let imgData = fftImageDataRef.current;
      if (!imgData) {
        imgData = offCtx.createImageData(width, height);
        fftImageDataRef.current = imgData;
      }
      const rgba = imgData.data;

      // Apply histogram slider range on top of percentile clipping
      const { vmin, vmax } = sliderRange(displayMin, displayMax, fftVminPct, fftVmaxPct);

      applyColormap(magnitude, rgba, lut, vmin, vmax);
      offCtx.putImageData(imgData, 0, 0);

      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.translate(fftPanX, fftPanY);
      ctx.scale(fftZoom, fftZoom);
      ctx.drawImage(offscreen, 0, 0);
      ctx.restore();
    };

    // Try WebGPU first, fall back to CPU
    if (gpuFFTRef.current && gpuReady) {
      // WebGPU path (async)
      let isCancelled = false;
      const runGpuFFT = async () => {
        const real = sourceData.slice();
        const imag = new Float32Array(real.length);
        
        const { real: fReal, imag: fImag } = await gpuFFTRef.current!.fft2D(real, imag, width, height, false);
        if (isCancelled) return;
        
        // Shift in CPU (TODO: move to GPU shader)
        fftshift(fReal, width, height);
        fftshift(fImag, width, height);
        
        renderMagnitude(fReal, fImag);
      };
      runGpuFFT();
      return () => { isCancelled = true; };
    } else {
      // CPU fallback (sync)
      const len = sourceData.length;
      let real = fftWorkRealRef.current;
      if (!real || real.length !== len) {
        real = new Float32Array(len);
        fftWorkRealRef.current = real;
      }
      real.set(sourceData);
      let imag = fftWorkImagRef.current;
      if (!imag || imag.length !== len) {
        imag = new Float32Array(len);
        fftWorkImagRef.current = imag;
      } else {
        imag.fill(0);
      }
      fft2d(real, imag, width, height, false);
      fftshift(real, width, height);
      fftshift(imag, width, height);
      renderMagnitude(real, imag);
    }
  }, [virtualImageBytes, shapeRows, shapeCols, fftColormap, fftZoom, fftPanX, fftPanY, gpuReady, showFft, fftScaleMode, fftAuto, fftVminPct, fftVmaxPct]);

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
  
  // DP scale bar + crosshair + ROI overlay (high-DPI)
  React.useEffect(() => {
    if (!dpUiRef.current) return;
    // Draw scale bar first (clears canvas)
    const kUnit = kCalibrated ? "mrad" : "px";
    drawScaleBarHiDPI(dpUiRef.current, DPR, dpZoom, kPixelSize || 1, kUnit, detCols);
    // Draw ROI overlay (circle, square, rect, annular) or point crosshair
    if (roiMode === "point") {
      drawDpCrosshairHiDPI(dpUiRef.current, DPR, localKCol, localKRow, dpZoom, dpPanX, dpPanY, detCols, detRows, isDraggingDP);
    } else {
      drawRoiOverlayHiDPI(
        dpUiRef.current, DPR, roiMode,
        localKCol, localKRow, roiRadius, roiRadiusInner, roiWidth, roiHeight,
        dpZoom, dpPanX, dpPanY, detCols, detRows,
        isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner
      );
    }
  }, [dpZoom, dpPanX, dpPanY, kPixelSize, kCalibrated, detRows, detCols, roiMode, roiRadius, roiRadiusInner, roiWidth, roiHeight, localKCol, localKRow, isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner]);
  
  // VI scale bar + crosshair + ROI (high-DPI)
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
  }, [viZoom, viPanX, viPanY, pixelSize, shapeRows, shapeCols, localPosRow, localPosCol, isDraggingVI,
      viRoiMode, localViRoiCenterRow, localViRoiCenterCol, viRoiRadius, viRoiWidth, viRoiHeight,
      isDraggingViRoi, isDraggingViRoiResize, isHoveringViRoiResize]);
  
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

  // Helper: check if point is near the outer resize handle
  const isNearResizeHandle = (imgX: number, imgY: number): boolean => {
    if (roiMode === "rect") {
      // For rectangle, check near bottom-right corner
      const handleX = roiCenterCol + roiWidth / 2;
      const handleY = roiCenterRow + roiHeight / 2;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      return dist < RESIZE_HIT_AREA_PX / dpZoom;
    }
    if ((roiMode !== "circle" && roiMode !== "square" && roiMode !== "annular") || !roiRadius) return false;
    const offset = roiMode === "square" ? roiRadius : roiRadius * CIRCLE_HANDLE_ANGLE;
    const handleX = roiCenterCol + offset;
    const handleY = roiCenterRow + offset;
    const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
    return dist < RESIZE_HIT_AREA_PX / dpZoom;
  };

  // Helper: check if point is near the inner resize handle (annular mode only)
  const isNearResizeHandleInner = (imgX: number, imgY: number): boolean => {
    if (roiMode !== "annular" || !roiRadiusInner) return false;
    const offset = roiRadiusInner * CIRCLE_HANDLE_ANGLE;
    const handleX = roiCenterCol + offset;
    const handleY = roiCenterRow + offset;
    const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
    return dist < RESIZE_HIT_AREA_PX / dpZoom;
  };

  // Helper: check if point is near VI ROI resize handle (same logic as DP)
  // Hit area is capped to avoid overlap with center for small ROIs
  const isNearViRoiResizeHandle = (imgX: number, imgY: number): boolean => {
    if (!viRoiMode || viRoiMode === "off") return false;
    if (viRoiMode === "rect") {
      const halfH = (viRoiHeight || 10) / 2;
      const halfW = (viRoiWidth || 10) / 2;
      const handleX = localViRoiCenterRow + halfH;
      const handleY = localViRoiCenterCol + halfW;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      const cornerDist = Math.sqrt(halfW ** 2 + halfH ** 2);
      const hitArea = Math.min(RESIZE_HIT_AREA_PX / viZoom, cornerDist * 0.5);
      return dist < hitArea;
    }
    if (viRoiMode === "circle" || viRoiMode === "square") {
      const radius = viRoiRadius || 5;
      const offset = viRoiMode === "square" ? radius : radius * CIRCLE_HANDLE_ANGLE;
      const handleX = localViRoiCenterRow + offset;
      const handleY = localViRoiCenterCol + offset;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      // Cap hit area to 50% of radius so center remains draggable
      const hitArea = Math.min(RESIZE_HIT_AREA_PX / viZoom, radius * 0.5);
      return dist < hitArea;
    }
    return false;
  };

  // Mouse handlers
  const handleDpMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = dpOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenX - dpPanX) / dpZoom;
    const imgY = (screenY - dpPanY) / dpZoom;

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

    setLocalKCol(imgX); setLocalKRow(imgY);
    // Use compound roi_center trait [row, col] - single observer fires in Python
    const newCol = Math.round(Math.max(0, Math.min(detCols - 1, imgX)));
    const newRow = Math.round(Math.max(0, Math.min(detRows - 1, imgY)));
    model.set("roi_center", [newRow, newCol]);
    model.save_changes();
  };

  const handleDpMouseUp = () => {
    setIsDraggingDP(false); setIsDraggingResize(false); setIsDraggingResizeInner(false);
  };
  const handleDpMouseLeave = () => {
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

    // Check if VI ROI mode is active - same logic as DP
    if (viRoiMode && viRoiMode !== "off") {
      // Check if clicking on resize handle
      if (isNearViRoiResizeHandle(imgX, imgY)) {
        setIsDraggingViRoiResize(true);
        return;
      }

      // Otherwise, move ROI center to click position (same as DP)
      setIsDraggingViRoi(true);
      setLocalViRoiCenterRow(imgX);
      setLocalViRoiCenterCol(imgY);
      setViRoiCenterRow(Math.round(Math.max(0, Math.min(shapeRows - 1, imgX))));
      setViRoiCenterCol(Math.round(Math.max(0, Math.min(shapeCols - 1, imgY))));
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

    // Handle VI ROI center dragging (same as DP)
    if (isDraggingViRoi) {
      setLocalViRoiCenterRow(imgX);
      setLocalViRoiCenterCol(imgY);
      // Batch VI ROI center updates
      const newViX = Math.round(Math.max(0, Math.min(shapeRows - 1, imgX)));
      const newViY = Math.round(Math.max(0, Math.min(shapeCols - 1, imgY)));
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

  const handleViMouseUp = () => {
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
      <Typography variant="h6" sx={{ ...typography.title, mb: `${SPACING.SM}px` }}>
        4D-STEM Explorer
      </Typography>

      {/* MAIN CONTENT: DP | VI | FFT (three columns when FFT shown) */}
      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        {/* LEFT COLUMN: DP Panel */}
        <Box sx={{ width: CANVAS_SIZE }}>
          {/* DP Header */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
            <Typography variant="caption" sx={{ ...typography.label }}>
              DP at ({Math.round(localPosRow)}, {Math.round(localPosCol)})
              <span style={{ color: "#0f0", marginLeft: SPACING.SM }}>k: ({Math.round(localKRow)}, {Math.round(localKCol)})</span>
              <InfoTooltip text="Diffraction Pattern: 2D detector image I(kx,ky) at scan position (row,col). The ROI mask M(kx,ky) defines which pixels are integrated for the virtual image. Drag to move ROI center, scroll to zoom, double-click to reset." theme={themeInfo.theme} />
            </Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`}>
              <Button size="small" sx={compactButton} disabled={dpZoom === 1 && dpPanX === 0 && dpPanY === 0 && roiCenterCol === centerCol && roiCenterRow === centerRow} onClick={() => { setDpZoom(1); setDpPanX(0); setDpPanY(0); setRoiCenterCol(centerCol); setRoiCenterRow(centerRow); }}>Reset</Button>
              <Button size="small" sx={compactButton} onClick={handleExportDP}>Export</Button>
            </Stack>
          </Stack>

          {/* DP Canvas */}
          <Box sx={{ ...container.imageBox, width: CANVAS_SIZE, height: CANVAS_SIZE }}>
            <canvas ref={dpCanvasRef} width={detCols} height={detRows} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
            <canvas
              ref={dpOverlayRef} width={detCols} height={detRows}
              onMouseDown={handleDpMouseDown} onMouseMove={handleDpMouseMove}
              onMouseUp={handleDpMouseUp} onMouseLeave={handleDpMouseLeave}
              onWheel={createZoomHandler(setDpZoom, setDpPanX, setDpPanY, dpZoom, dpPanX, dpPanY, dpOverlayRef)}
              onDoubleClick={handleDpDoubleClick}
              style={{ position: "absolute", width: "100%", height: "100%", cursor: isHoveringResize || isDraggingResize ? "nwse-resize" : "crosshair" }}
            />
            <canvas ref={dpUiRef} width={CANVAS_SIZE * DPR} height={CANVAS_SIZE * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
          </Box>

          {/* DP Stats Bar */}
          {dpStats && dpStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center" }}>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[0])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[1])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[2])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[3])}</Box></Typography>
              {cursorInfo && cursorInfo.panel === "DP" && (
                <>
                  <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 14 }} />
                  <Typography sx={{ fontSize: 11, color: themeColors.textMuted, fontFamily: "monospace" }}>
                    ({cursorInfo.row}, {cursorInfo.col}) <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(cursorInfo.value)}</Box>
                  </Typography>
                </>
              )}
            </Box>
          )}

          {/* DP Controls - two rows with histogram on right */}
          <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
            {/* Left: two rows of controls */}
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
              {/* Row 1: Detector + slider */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Detector:</Typography>
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
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>
                      {roiMode === "annular" ? `${Math.round(roiRadiusInner)}-${Math.round(roiRadius)}px` : `${Math.round(roiRadius)}px`}
                    </Typography>
                  </>
                )}
              </Box>
              {/* Row 2: Presets + Color + Scale */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography component="span" onClick={() => { setRoiMode("circle"); setRoiRadius(bfRadius || 10); setRoiCenterCol(centerCol); setRoiCenterRow(centerRow); }} sx={{ color: "#4f4", fontSize: 11, fontWeight: "bold", cursor: "pointer", "&:hover": { textDecoration: "underline" } }}>BF</Typography>
                <Typography component="span" onClick={() => { setRoiMode("annular"); setRoiRadiusInner((bfRadius || 10) * 0.5); setRoiRadius(bfRadius || 10); setRoiCenterCol(centerCol); setRoiCenterRow(centerRow); }} sx={{ color: "#4af", fontSize: 11, fontWeight: "bold", cursor: "pointer", "&:hover": { textDecoration: "underline" } }}>ABF</Typography>
                <Typography component="span" onClick={() => { setRoiMode("annular"); setRoiRadiusInner(bfRadius || 10); setRoiRadius(Math.min((bfRadius || 10) * 3, Math.min(detRows, detCols) / 2 - 2)); setRoiCenterCol(centerCol); setRoiCenterRow(centerRow); }} sx={{ color: "#fa4", fontSize: 11, fontWeight: "bold", cursor: "pointer", "&:hover": { textDecoration: "underline" } }}>ADF</Typography>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                <Select value={dpColormap} onChange={(e) => setDpColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="inferno">Inferno</MenuItem>
                  <MenuItem value="viridis">Viridis</MenuItem>
                  <MenuItem value="plasma">Plasma</MenuItem>
                  <MenuItem value="magma">Magma</MenuItem>
                  <MenuItem value="hot">Hot</MenuItem>
                  <MenuItem value="gray">Gray</MenuItem>
                </Select>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                <Select value={dpScaleMode} onChange={(e) => setDpScaleMode(e.target.value as "linear" | "log" | "power")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="linear">Lin</MenuItem>
                  <MenuItem value="log">Log</MenuItem>
                  <MenuItem value="power">Pow</MenuItem>
                </Select>
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
            <Typography variant="caption" sx={{ ...typography.label }}>Image<InfoTooltip text="Virtual image: Integrated intensity within detector ROI at each scan position. Computed as Σ(DP × mask) for each (row,col). Double-click to select position." theme={themeInfo.theme} /></Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
              <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: 10 }}>
                {shapeRows}×{shapeCols} | {detRows}×{detCols}
              </Typography>
              <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT:</Typography>
              <Switch checked={showFft} onChange={(e) => setShowFft(e.target.checked)} size="small" sx={switchStyles.small} />
              <Button size="small" sx={compactButton} disabled={viZoom === 1 && viPanX === 0 && viPanY === 0} onClick={() => { setViZoom(1); setViPanX(0); setViPanY(0); }}>Reset</Button>
              <Button size="small" sx={compactButton} onClick={handleExportVI}>Export</Button>
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
          </Box>

          {/* VI Stats Bar */}
          {viStats && viStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center" }}>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[0])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[1])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[2])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[3])}</Box></Typography>
              {cursorInfo && cursorInfo.panel === "VI" && (
                <>
                  <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 14 }} />
                  <Typography sx={{ fontSize: 11, color: themeColors.textMuted, fontFamily: "monospace" }}>
                    ({cursorInfo.row}, {cursorInfo.col}) <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(cursorInfo.value)}</Box>
                  </Typography>
                </>
              )}
            </Box>
          )}

          {/* VI Controls - Two rows with histogram on right */}
          <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
            {/* Left: Two rows of controls */}
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
              {/* Row 1: ROI selector */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>ROI:</Typography>
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
                        <Typography sx={{ ...typography.value, fontSize: 10, minWidth: 30 }}>
                          {Math.round(viRoiRadius || 5)}px
                        </Typography>
                      </>
                    )}
                    {summedDpCount > 0 && (
                      <Typography sx={{ ...typography.label, fontSize: 9, color: "#a6f" }}>
                        {summedDpCount} pos
                      </Typography>
                    )}
                  </>
                )}
              </Box>
              {/* Row 2: Color + Scale */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                <Select value={viColormap} onChange={(e) => setViColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="inferno">Inferno</MenuItem>
                  <MenuItem value="viridis">Viridis</MenuItem>
                  <MenuItem value="plasma">Plasma</MenuItem>
                  <MenuItem value="magma">Magma</MenuItem>
                  <MenuItem value="hot">Hot</MenuItem>
                  <MenuItem value="gray">Gray</MenuItem>
                </Select>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
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
              <Typography variant="caption" sx={{ ...typography.label }}>FFT<InfoTooltip text="Fast Fourier Transform: Shows spatial frequency content of the virtual image. Center = low frequencies (large features), edges = high frequencies (fine detail). Useful for detecting periodic structures and scan artifacts." theme={themeInfo.theme} /></Typography>
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
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                  <Select value={fftScaleMode} onChange={(e) => setFftScaleMode(e.target.value as "linear" | "log" | "power")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                    <MenuItem value="power">Pow</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto:<InfoTooltip text="Auto-enhance FFT display. When ON: (1) Masks DC component at center - DC = F(0,0) = Σ(image), replaced with average of 4 neighbors. (2) Clips display to 99.9 percentile to exclude outliers. When OFF: shows raw FFT with full dynamic range." theme={themeInfo.theme} /></Typography>
                  <Switch checked={fftAuto} onChange={(e) => setFftAuto(e.target.checked)} size="small" sx={switchStyles.small} />
                </Box>
                {/* Row 2: Color */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
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
        <Stack direction="row" spacing={`${SPACING.MD}px`} sx={{ mt: `${SPACING.LG}px` }}>
          <Box className="show4dstem-control-group" sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px` }}>
            <Typography sx={{ ...typography.label }}>Path:</Typography>
            <Typography component="span" onClick={() => { setPathPlaying(false); setPathIndex(0); }} sx={{ color: themeColors.textMuted, fontSize: 14, cursor: "pointer", "&:hover": { color: "#fff" }, px: 0.5 }} title="Stop">⏹</Typography>
            <Typography component="span" onClick={() => setPathPlaying(!pathPlaying)} sx={{ color: pathPlaying ? "#0f0" : "#888", fontSize: 14, cursor: "pointer", "&:hover": { color: "#fff" }, px: 0.5 }} title={pathPlaying ? "Pause" : "Play"}>{pathPlaying ? "⏸" : "▶"}</Typography>
            <Typography sx={{ ...typography.value, minWidth: 60 }}>{pathIndex + 1}/{pathLength}</Typography>
            <Slider value={pathIndex} onChange={(_, v) => { setPathPlaying(false); setPathIndex(v as number); }} min={0} max={Math.max(0, pathLength - 1)} size="small" sx={{ width: 100 }} />
          </Box>
        </Stack>
      )}
    </Box>
  );
}

export const render = createRender(Show4DSTEM);
