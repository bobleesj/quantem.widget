/**
 * Edit2D - Interactive crop/pad/mask tool for 2D images.
 *
 * Features:
 * - Crop mode: draggable crop rectangle with corner/edge handles
 * - Mask mode: brush, rectangle, ellipse, and threshold tools
 * - Semi-transparent overlays for crop dim and mask visualization
 * - Scroll to zoom, double-click to reset
 * - Automatic theme detection (light/dark mode)
 */

import * as React from "react";
import { createRender, useModelState, useModel } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Menu from "@mui/material/Menu";
import Switch from "@mui/material/Switch";
import Button from "@mui/material/Button";
import Tooltip from "@mui/material/Tooltip";
import IconButton from "@mui/material/IconButton";
import Slider from "@mui/material/Slider";
import NavigateBeforeIcon from "@mui/icons-material/NavigateBefore";
import NavigateNextIcon from "@mui/icons-material/NavigateNext";
import TuneIcon from "@mui/icons-material/Tune";
import "./edit2d.css";
import { useTheme } from "../theme";
import { drawScaleBarHiDPI, exportFigure } from "../scalebar";
import { extractFloat32, formatNumber, downloadBlob } from "../format";
import { computeHistogramFromBytes } from "../histogram";
import { findDataRange, applyLogScale, percentileClip, sliderRange } from "../stats";
import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen } from "../colormaps";

// ============================================================================
// UI Styles (matching Show3D/Show4DSTEM)
// ============================================================================
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const },
};

const SPACING = {
  XS: 4,
  SM: 8,
  MD: 12,
  LG: 16,
};

const switchStyles = {
  small: { '& .MuiSwitch-thumb': { width: 12, height: 12 }, '& .MuiSwitch-switchBase': { padding: '4px' } },
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
  "&.Mui-disabled": {
    color: "#666",
    borderColor: "#444",
  },
};

const sliderStyles = {
  small: {
    py: 0,
    "& .MuiSlider-thumb": { width: 10, height: 10 },
    "& .MuiSlider-rail": { height: 2 },
    "& .MuiSlider-track": { height: 2 },
  },
};

// ============================================================================
// InfoTooltip (matching Show3D)
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
        \u24d8
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

// ============================================================================
// Histogram (matching Show3D)
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

// ============================================================================
// Constants
// ============================================================================
const CANVAS_TARGET_SIZE = 500;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const DPR = window.devicePixelRatio || 1;
const HANDLE_SIZE = 6;

type DragMode =
  | "none"
  | "move"
  | "resize-tl"
  | "resize-tr"
  | "resize-bl"
  | "resize-br"
  | "resize-t"
  | "resize-b"
  | "resize-l"
  | "resize-r"
  | "pan";

interface DragStart {
  mouseX: number;
  mouseY: number;
  cropTop: number;
  cropLeft: number;
  cropBottom: number;
  cropRight: number;
  panX: number;
  panY: number;
}

// ============================================================================
// Hit testing (crop mode)
// ============================================================================
function nearVal(a: number, b: number, threshold: number): boolean {
  return Math.abs(a - b) < threshold;
}

function hitTestCrop(
  screenX: number,
  screenY: number,
  sLeft: number,
  sRight: number,
  sTop: number,
  sBottom: number,
): DragMode {
  const thresh = HANDLE_SIZE + 2;

  if (nearVal(screenX, sLeft, thresh) && nearVal(screenY, sTop, thresh)) return "resize-tl";
  if (nearVal(screenX, sRight, thresh) && nearVal(screenY, sTop, thresh)) return "resize-tr";
  if (nearVal(screenX, sLeft, thresh) && nearVal(screenY, sBottom, thresh)) return "resize-bl";
  if (nearVal(screenX, sRight, thresh) && nearVal(screenY, sBottom, thresh)) return "resize-br";

  if (nearVal(screenY, sTop, thresh) && screenX > sLeft + thresh && screenX < sRight - thresh) return "resize-t";
  if (nearVal(screenY, sBottom, thresh) && screenX > sLeft + thresh && screenX < sRight - thresh) return "resize-b";
  if (nearVal(screenX, sLeft, thresh) && screenY > sTop + thresh && screenY < sBottom - thresh) return "resize-l";
  if (nearVal(screenX, sRight, thresh) && screenY > sTop + thresh && screenY < sBottom - thresh) return "resize-r";

  if (screenX >= sLeft && screenX <= sRight && screenY >= sTop && screenY <= sBottom) return "move";

  return "pan";
}

function getCursorForMode(dragMode: DragMode): string {
  switch (dragMode) {
    case "resize-tl": case "resize-br": return "nwse-resize";
    case "resize-tr": case "resize-bl": return "nesw-resize";
    case "resize-t": case "resize-b": return "ns-resize";
    case "resize-l": case "resize-r": return "ew-resize";
    case "move": return "move";
    case "pan": return "grab";
    default: return "default";
  }
}

// ============================================================================
// Mask painting functions
// ============================================================================
function paintBrushCircle(
  mask: Uint8Array,
  w: number,
  h: number,
  centerCol: number,
  centerRow: number,
  radius: number,
  value: number,
): void {
  const r = Math.ceil(radius);
  const r2 = radius * radius;
  const y0 = Math.max(0, Math.floor(centerRow - r));
  const y1 = Math.min(h - 1, Math.ceil(centerRow + r));
  const x0 = Math.max(0, Math.floor(centerCol - r));
  const x1 = Math.min(w - 1, Math.ceil(centerCol + r));
  for (let row = y0; row <= y1; row++) {
    for (let col = x0; col <= x1; col++) {
      const dc = col - centerCol;
      const dr = row - centerRow;
      if (dc * dc + dr * dr <= r2) {
        mask[row * w + col] = value;
      }
    }
  }
}

function paintBrushLine(
  mask: Uint8Array,
  w: number,
  h: number,
  col0: number,
  row0: number,
  col1: number,
  row1: number,
  radius: number,
  value: number,
): void {
  const dc = col1 - col0;
  const dr = row1 - row0;
  const dist = Math.sqrt(dc * dc + dr * dr);
  const steps = Math.max(1, Math.ceil(dist / Math.max(1, radius * 0.5)));
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    paintBrushCircle(mask, w, h, col0 + t * dc, row0 + t * dr, radius, value);
  }
}

function fillRectMask(
  mask: Uint8Array,
  w: number,
  h: number,
  r0: number,
  c0: number,
  r1: number,
  c1: number,
  value: number,
): void {
  const rowMin = Math.max(0, Math.min(Math.floor(Math.min(r0, r1)), h - 1));
  const rowMax = Math.max(0, Math.min(Math.floor(Math.max(r0, r1)), h - 1));
  const colMin = Math.max(0, Math.min(Math.floor(Math.min(c0, c1)), w - 1));
  const colMax = Math.max(0, Math.min(Math.floor(Math.max(c0, c1)), w - 1));
  for (let row = rowMin; row <= rowMax; row++) {
    for (let col = colMin; col <= colMax; col++) {
      mask[row * w + col] = value;
    }
  }
}

function fillEllipseMask(
  mask: Uint8Array,
  w: number,
  h: number,
  centerRow: number,
  centerCol: number,
  radiusRow: number,
  radiusCol: number,
  value: number,
): void {
  if (radiusRow < 0.5 || radiusCol < 0.5) return;
  const rowMin = Math.max(0, Math.floor(centerRow - radiusRow));
  const rowMax = Math.min(h - 1, Math.ceil(centerRow + radiusRow));
  const colMin = Math.max(0, Math.floor(centerCol - radiusCol));
  const colMax = Math.min(w - 1, Math.ceil(centerCol + radiusCol));
  for (let row = rowMin; row <= rowMax; row++) {
    for (let col = colMin; col <= colMax; col++) {
      const dr = (row - centerRow) / radiusRow;
      const dc = (col - centerCol) / radiusCol;
      if (dr * dr + dc * dc <= 1) {
        mask[row * w + col] = value;
      }
    }
  }
}

// ============================================================================
// Main Component
// ============================================================================
function Edit2D() {
  const model = useModel();
  const { themeInfo, colors: themeColors } = useTheme();

  const themedSelect = {
    fontSize: 10,
    "& .MuiSelect-select": { py: 0.5 },
    bgcolor: themeColors.controlBg,
    color: themeColors.text,
    "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.textMuted },
    "& .MuiSvgIcon-root": { color: themeColors.textMuted },
  };
  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: {
      sx: {
        bgcolor: themeColors.controlBg,
        color: themeColors.text,
        border: `1px solid ${themeColors.border}`,
        "& .MuiMenuItem-root": { fontSize: 10, "&:hover": { bgcolor: themeColors.bgAlt } },
      },
    },
  };

  // ── Model state ──────────────────────────────────────────────────────
  const [nImages] = useModelState<number>("n_images");
  const [height] = useModelState<number>("height");
  const [width] = useModelState<number>("width");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [labels] = useModelState<string[]>("labels");
  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [selectedIdx, setSelectedIdx] = useModelState<number>("selected_idx");

  const [cropTop, setCropTop] = useModelState<number>("crop_top");
  const [cropLeft, setCropLeft] = useModelState<number>("crop_left");
  const [cropBottom, setCropBottom] = useModelState<number>("crop_bottom");
  const [cropRight, setCropRight] = useModelState<number>("crop_right");
  const [fillValue] = useModelState<number>("fill_value");

  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [showStats, setShowStats] = useModelState<boolean>("show_stats");
  const [showControls] = useModelState<boolean>("show_controls");
  const [showDisplayControlsGroup, setShowDisplayControlsGroup] = useModelState<boolean>("show_display_controls");
  const [showEditControlsGroup, setShowEditControlsGroup] = useModelState<boolean>("show_edit_controls");
  const [showHistogramGroup, setShowHistogramGroup] = useModelState<boolean>("show_histogram");
  const [disabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools] = useModelState<string[]>("hidden_tools");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [statsMean] = useModelState<number>("stats_mean");
  const [statsMin] = useModelState<number>("stats_min");
  const [statsMax] = useModelState<number>("stats_max");
  const [statsStd] = useModelState<number>("stats_std");

  // Mask model state
  const [mode, setMode] = useModelState<string>("mode");
  const [maskTool, setMaskTool] = useModelState<string>("mask_tool");
  const [brushSize, setBrushSize] = useModelState<number>("brush_size");
  const [maskAction, setMaskAction] = useModelState<string>("mask_action");

  // ── Local state ──────────────────────────────────────────────────────
  const [zoom, setZoom] = React.useState(1);
  const [panX, setPanX] = React.useState(0);
  const [panY, setPanY] = React.useState(0);
  const [canvasSize, setCanvasSize] = React.useState(CANVAS_TARGET_SIZE);
  const [dragMode, setDragMode] = React.useState<DragMode>("none");
  const [cursorStyle, setCursorStyle] = React.useState("default");
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number } | null>(null);
  const [imageVminPct, setImageVminPct] = React.useState(0);
  const [imageVmaxPct, setImageVmaxPct] = React.useState(100);
  const [imageHistogramData, setImageHistogramData] = React.useState<Float32Array | null>(null);
  const [imageDataRange, setImageDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [maskVersion, setMaskVersion] = React.useState(0);
  const [shapePreview, setShapePreview] = React.useState<{ r0: number; c0: number; r1: number; c1: number } | null>(null);
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);
  const [controlsAnchor, setControlsAnchor] = React.useState<HTMLElement | null>(null);
  const [isRootHovered, setIsRootHovered] = React.useState(false);

  // ── Refs ─────────────────────────────────────────────────────────────
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const overlayRef = React.useRef<HTMLCanvasElement>(null);
  const uiRef = React.useRef<HTMLCanvasElement>(null);
  const canvasContainerRef = React.useRef<HTMLDivElement>(null);
  const dragStartRef = React.useRef<DragStart | null>(null);
  const rawDataRef = React.useRef<Float32Array | null>(null);
  const maskRef = React.useRef<Uint8Array | null>(null);
  const isPaintingRef = React.useRef(false);
  const lastPaintPosRef = React.useRef<{ row: number; col: number } | null>(null);
  const shapeStartRef = React.useRef<{ row: number; col: number } | null>(null);
  const isDraggingShapeRef = React.useRef(false);
  const maskCanvasRef = React.useRef<HTMLCanvasElement | null>(null);

  // ── Derived values ───────────────────────────────────────────────────
  const displayScale = canvasSize / Math.max(width, height, 1);
  const canvasW = Math.round(width * displayScale);
  const canvasH = Math.round(height * displayScale);
  const dataReady = frameBytes && frameBytes.byteLength > 0 && width > 0 && height > 0;
  const needsReset = zoom !== 1 || panX !== 0 || panY !== 0;
  const cropH = cropBottom - cropTop;
  const cropW = cropRight - cropLeft;
  const hasPadding = cropTop < 0 || cropLeft < 0 || cropBottom > height || cropRight > width;

  const disabledToolSet = React.useMemo(
    () =>
      new Set(
        (disabledTools || [])
          .map((name) => String(name).trim().toLowerCase())
          .filter(Boolean),
      ),
    [disabledTools],
  );
  const hiddenToolSet = React.useMemo(
    () =>
      new Set(
        (hiddenTools || [])
          .map((name) => String(name).trim().toLowerCase())
          .filter(Boolean),
      ),
    [hiddenTools],
  );

  const hideAll = hiddenToolSet.has("all");
  const hideMode = hideAll || hiddenToolSet.has("mode");
  const hideEdit = hideAll || hiddenToolSet.has("edit");
  const hideDisplay = hideAll || hiddenToolSet.has("display");
  const hideHistogram = hideAll || hiddenToolSet.has("histogram");
  const hideStats = hideAll || hiddenToolSet.has("stats");
  const hideNavigation = hideAll || hiddenToolSet.has("navigation");
  const hideExport = hideAll || hiddenToolSet.has("export");
  const hideView = hideAll || hiddenToolSet.has("view");

  const lockAll = disabledToolSet.has("all") || hideAll;
  const lockMode = lockAll || hideMode || disabledToolSet.has("mode");
  const lockEdit = lockAll || hideEdit || disabledToolSet.has("edit");
  const lockDisplay = lockAll || hideDisplay || disabledToolSet.has("display");
  const lockHistogram = lockAll || hideHistogram || disabledToolSet.has("histogram");
  const lockStats = lockAll || hideStats || disabledToolSet.has("stats");
  const lockNavigation = lockAll || hideNavigation || disabledToolSet.has("navigation");
  const lockExport = lockAll || hideExport || disabledToolSet.has("export");
  const lockView = lockAll || hideView || disabledToolSet.has("view");

  const wantDisplayControlsGroup = showDisplayControlsGroup && !hideDisplay;
  const wantEditControlsGroup = showEditControlsGroup && !hideEdit;
  const wantHistogramGroup = showHistogramGroup && !hideHistogram;
  const showLeftControlGroups = wantDisplayControlsGroup || wantEditControlsGroup;
  const showAnyControlGroups = showLeftControlGroups || wantHistogramGroup;

  // ── Mask coverage ───────────────────────────────────────────────────
  const maskCoverage = React.useMemo(() => {
    if (mode !== "mask" || !maskRef.current) return { count: 0, pct: 0 };
    let count = 0;
    const m = maskRef.current;
    for (let i = 0; i < m.length; i++) {
      if (m[i] > 0) count++;
    }
    return { count, pct: m.length > 0 ? (count / m.length) * 100 : 0 };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, maskVersion, width, height]);

  // ── Initialize mask ref ─────────────────────────────────────────────
  React.useEffect(() => {
    if (width <= 0 || height <= 0) return;
    const size = width * height;
    if (!maskRef.current || maskRef.current.length !== size) {
      maskRef.current = new Uint8Array(size);
    }
  }, [width, height]);

  // ── Sync mask from Python (on init or external update) ──────────────
  React.useEffect(() => {
    if (width <= 0 || height <= 0) return;
    const maskBytesView = model.get("mask_bytes") as DataView | undefined;
    const size = width * height;
    if (maskBytesView && maskBytesView.byteLength === size) {
      maskRef.current = new Uint8Array(maskBytesView.buffer, maskBytesView.byteOffset, maskBytesView.byteLength).slice();
      setMaskVersion((v) => v + 1);
    }
    // Only run on init
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, height]);

  // ── Sync mask to Python ─────────────────────────────────────────────
  const syncMaskToPython = React.useCallback(() => {
    if (!maskRef.current) return;
    const copy = maskRef.current.slice();
    model.set("mask_bytes", new DataView(copy.buffer));
    model.save_changes();
  }, [model]);

  // ── Extract float32 data ─────────────────────────────────────────────
  React.useEffect(() => {
    if (!dataReady) return;
    const allFloats = extractFloat32(frameBytes);
    if (!allFloats) return;
    const frameSize = width * height;
    const idx = Math.min(selectedIdx, nImages - 1);
    const offset = idx * frameSize;
    if (offset + frameSize <= allFloats.length) {
      rawDataRef.current = allFloats.subarray(offset, offset + frameSize);
    }
  }, [frameBytes, width, height, selectedIdx, nImages, dataReady]);

  // ── Histogram ────────────────────────────────────────────────────────
  React.useEffect(() => {
    if (!rawDataRef.current) return;
    const raw = rawDataRef.current;
    const processed = logScale ? applyLogScale(raw) : raw;
    const { min, max } = findDataRange(processed);
    setImageDataRange({ min, max });
    setImageHistogramData(processed);
  }, [frameBytes, selectedIdx, logScale, dataReady]);

  // ── Threshold mask (when threshold tool active) ─────────────────────
  React.useEffect(() => {
    if (lockEdit) return;
    if (mode !== "mask" || maskTool !== "threshold" || !rawDataRef.current || !maskRef.current) return;
    const data = rawDataRef.current;
    const processed = logScale ? applyLogScale(data) : data;
    const { vmin, vmax } = sliderRange(imageDataRange.min, imageDataRange.max, imageVminPct, imageVmaxPct);
    const mask = maskRef.current;
    for (let i = 0; i < data.length; i++) {
      const v = processed[i];
      mask[i] = (v >= vmin && v <= vmax) ? 255 : 0;
    }
    setMaskVersion((v) => v + 1);
    syncMaskToPython();
  }, [lockEdit, mode, maskTool, imageVminPct, imageVmaxPct, imageDataRange, logScale, syncMaskToPython]);

  // ── Prevent page scroll on wheel ────────────────────────────────────
  React.useEffect(() => {
    const el = canvasContainerRef.current;
    if (!el) return;
    const prevent = (e: WheelEvent) => e.preventDefault();
    el.addEventListener("wheel", prevent, { passive: false });
    return () => el.removeEventListener("wheel", prevent);
  }, []);

  // ── Canvas resize handle ─────────────────────────────────────────────
  const resizeRef = React.useRef<{ startX: number; startSize: number } | null>(null);

  const handleResizeStart = React.useCallback((e: React.MouseEvent) => {
    if (lockView) return;
    e.stopPropagation();
    e.preventDefault();
    resizeRef.current = { startX: e.clientX, startSize: canvasSize };
    const onMove = (ev: MouseEvent) => {
      if (!resizeRef.current) return;
      const delta = ev.clientX - resizeRef.current.startX;
      const newSize = Math.max(200, Math.min(1200, resizeRef.current.startSize + delta));
      setCanvasSize(newSize);
    };
    const onUp = () => {
      resizeRef.current = null;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [canvasSize, lockView]);

  // ── Screen <-> Image coordinate transforms ──────────────────────────
  const toScreenX = React.useCallback((imgCol: number) => {
    const cx = canvasW / 2;
    return (imgCol * displayScale - cx) * zoom + cx + panX;
  }, [canvasW, displayScale, zoom, panX]);

  const toScreenY = React.useCallback((imgRow: number) => {
    const cy = canvasH / 2;
    return (imgRow * displayScale - cy) * zoom + cy + panY;
  }, [canvasH, displayScale, zoom, panY]);

  const toImageCol = React.useCallback((screenX: number) => {
    const cx = canvasW / 2;
    return ((screenX - cx - panX) / zoom + cx) / displayScale;
  }, [canvasW, displayScale, zoom, panX]);

  const toImageRow = React.useCallback((screenY: number) => {
    const cy = canvasH / 2;
    return ((screenY - cy - panY) / zoom + cy) / displayScale;
  }, [canvasH, displayScale, zoom, panY]);

  // ── Render main image ────────────────────────────────────────────────
  React.useEffect(() => {
    if (!dataReady || !canvasRef.current || !rawDataRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const raw = rawDataRef.current;
    const processed = logScale ? applyLogScale(raw) : raw;

    let vmin: number, vmax: number;
    if (autoContrast) {
      ({ vmin, vmax } = percentileClip(processed, 2, 98));
    } else {
      ({ vmin, vmax } = sliderRange(imageDataRange.min, imageDataRange.max, imageVminPct, imageVmaxPct));
    }

    const lut = COLORMAPS[cmap] || COLORMAPS.gray;
    const offscreen = renderToOffscreen(processed, width, height, lut, vmin, vmax);
    if (!offscreen) return;

    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.save();
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    ctx.translate(cx + panX, cy + panY);
    ctx.scale(zoom, zoom);
    ctx.translate(-cx, -cy);
    ctx.imageSmoothingEnabled = zoom < 4;
    ctx.drawImage(offscreen, 0, 0, width, height, 0, 0, canvasW, canvasH);
    ctx.restore();
  }, [dataReady, cmap, logScale, autoContrast, imageVminPct, imageVmaxPct, zoom, panX, panY, canvasW, canvasH, width, height, imageDataRange, frameBytes, selectedIdx]);

  // ── Render crop overlay (crop mode only) ────────────────────────────
  React.useEffect(() => {
    if (mode !== "crop" || !overlayRef.current) return;
    const ctx = overlayRef.current.getContext("2d");
    if (!ctx) return;

    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, canvasW, canvasH);

    const sLeft = toScreenX(cropLeft);
    const sRight = toScreenX(cropRight);
    const sTop = toScreenY(cropTop);
    const sBottom = toScreenY(cropBottom);

    // Dim overlay outside crop region
    ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
    ctx.fillRect(0, 0, canvasW, Math.max(0, sTop));
    ctx.fillRect(0, Math.min(canvasH, sBottom), canvasW, canvasH - Math.min(canvasH, sBottom));
    const stripTop = Math.max(0, sTop);
    const stripBottom = Math.min(canvasH, sBottom);
    ctx.fillRect(0, stripTop, Math.max(0, sLeft), stripBottom - stripTop);
    ctx.fillRect(Math.min(canvasW, sRight), stripTop, canvasW - Math.min(canvasW, sRight), stripBottom - stripTop);

    // Crop rectangle border
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(sLeft, sTop, sRight - sLeft, sBottom - sTop);
    ctx.setLineDash([]);

    // Corner handles
    ctx.fillStyle = "#fff";
    const hs = HANDLE_SIZE;
    for (const [hx, hy] of [[sLeft, sTop], [sRight, sTop], [sLeft, sBottom], [sRight, sBottom]]) {
      ctx.fillRect(hx - hs / 2, hy - hs / 2, hs, hs);
    }

    // Edge handles
    const sMidX = (sLeft + sRight) / 2;
    const sMidY = (sTop + sBottom) / 2;
    for (const [hx, hy, hw, hh] of [
      [sMidX, sTop, 10, 4],
      [sMidX, sBottom, 10, 4],
      [sLeft, sMidY, 4, 10],
      [sRight, sMidY, 4, 10],
    ] as [number, number, number, number][]) {
      ctx.fillRect(hx - hw / 2, hy - hh / 2, hw, hh);
    }

    // Image boundary when crop extends beyond
    if (hasPadding) {
      const imgLeft = toScreenX(0);
      const imgRight = toScreenX(width);
      const imgTop = toScreenY(0);
      const imgBottom = toScreenY(height);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.strokeRect(imgLeft, imgTop, imgRight - imgLeft, imgBottom - imgTop);
      ctx.setLineDash([]);
    }

    // Crop dimensions label
    ctx.fillStyle = "#fff";
    ctx.font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.shadowColor = "rgba(0,0,0,0.7)";
    ctx.shadowBlur = 3;
    const dimLabel = `${cropW} \u00d7 ${cropH}`;
    const labelY = Math.max(14, sTop - 4);
    ctx.fillText(dimLabel, (sLeft + sRight) / 2, labelY);
    ctx.shadowBlur = 0;
  }, [mode, cropTop, cropLeft, cropBottom, cropRight, zoom, panX, panY, canvasW, canvasH, width, height, displayScale, hasPadding, toScreenX, toScreenY, cropW, cropH]);

  // ── Render mask overlay (mask mode only) ────────────────────────────
  React.useEffect(() => {
    if (mode !== "mask" || !overlayRef.current) return;
    const ctx = overlayRef.current.getContext("2d");
    if (!ctx) return;
    const mask = maskRef.current;

    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, canvasW, canvasH);

    // Draw mask overlay
    if (mask && mask.length === width * height) {
      // Create/reuse offscreen mask canvas
      if (!maskCanvasRef.current || maskCanvasRef.current.width !== width || maskCanvasRef.current.height !== height) {
        maskCanvasRef.current = document.createElement("canvas");
        maskCanvasRef.current.width = width;
        maskCanvasRef.current.height = height;
      }
      const maskCtx = maskCanvasRef.current.getContext("2d")!;
      const maskImageData = maskCtx.createImageData(width, height);
      const pixels = maskImageData.data;
      for (let i = 0; i < mask.length; i++) {
        if (mask[i] > 0) {
          pixels[i * 4] = 255;
          pixels[i * 4 + 1] = 50;
          pixels[i * 4 + 2] = 50;
          pixels[i * 4 + 3] = 100;
        }
      }
      maskCtx.putImageData(maskImageData, 0, 0);

      // Draw with zoom/pan transform
      ctx.save();
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      ctx.translate(cx + panX, cy + panY);
      ctx.scale(zoom, zoom);
      ctx.translate(-cx, -cy);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(maskCanvasRef.current, 0, 0, width, height, 0, 0, canvasW, canvasH);
      ctx.restore();
    }

    // Draw brush cursor
    if (maskTool === "brush" && cursorInfo) {
      const sx = toScreenX(cursorInfo.col + 0.5);
      const sy = toScreenY(cursorInfo.row + 0.5);
      const sr = (brushSize / 2) * displayScale * zoom;
      ctx.strokeStyle = maskAction === "add" ? "rgba(255,100,100,0.8)" : "rgba(100,100,255,0.8)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(sx, sy, sr, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Draw shape preview while dragging
    if (shapePreview && isDraggingShapeRef.current) {
      const { r0, c0, r1, c1 } = shapePreview;
      ctx.strokeStyle = maskAction === "add" ? "rgba(255,100,100,0.8)" : "rgba(100,100,255,0.8)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      if (maskTool === "rectangle") {
        const sx0 = toScreenX(Math.min(c0, c1));
        const sy0 = toScreenY(Math.min(r0, r1));
        const sx1 = toScreenX(Math.max(c0, c1));
        const sy1 = toScreenY(Math.max(r0, r1));
        ctx.strokeRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
      } else if (maskTool === "ellipse") {
        const scx = toScreenX((c0 + c1) / 2);
        const scy = toScreenY((r0 + r1) / 2);
        const rx = Math.abs(toScreenX(c1) - toScreenX(c0)) / 2;
        const ry = Math.abs(toScreenY(r1) - toScreenY(r0)) / 2;
        ctx.beginPath();
        ctx.ellipse(scx, scy, Math.max(1, rx), Math.max(1, ry), 0, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
  }, [mode, maskVersion, maskTool, brushSize, maskAction, cursorInfo, shapePreview, zoom, panX, panY, canvasW, canvasH, width, height, displayScale, toScreenX, toScreenY]);

  // ── Render scale bar ─────────────────────────────────────────────────
  React.useEffect(() => {
    if (!uiRef.current) return;
    const ctx = uiRef.current.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, canvasW, canvasH);

    if (pixelSize > 0) {
      drawScaleBarHiDPI(uiRef.current, DPR, zoom, pixelSize, "\u00c5", width);
    }

    if (zoom !== 1) {
      ctx.fillStyle = "#fff";
      ctx.font = "bold 12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.shadowColor = "rgba(0,0,0,0.5)";
      ctx.shadowBlur = 2;
      ctx.fillText(`${zoom.toFixed(1)}\u00d7`, 12, canvasH - 12);
      ctx.shadowBlur = 0;
    }
  }, [canvasW, canvasH, zoom, pixelSize, width]);

  // ── Mask operations ─────────────────────────────────────────────────
  const handleInvertMask = React.useCallback(() => {
    if (lockEdit) return;
    if (!maskRef.current) return;
    const mask = maskRef.current;
    for (let i = 0; i < mask.length; i++) {
      mask[i] = mask[i] > 0 ? 0 : 255;
    }
    setMaskVersion((v) => v + 1);
    syncMaskToPython();
  }, [lockEdit, syncMaskToPython]);

  const handleClearMask = React.useCallback(() => {
    if (lockEdit) return;
    if (!maskRef.current) return;
    maskRef.current.fill(0);
    setMaskVersion((v) => v + 1);
    syncMaskToPython();
  }, [lockEdit, syncMaskToPython]);

  // ── Mouse handlers ───────────────────────────────────────────────────
  const getCanvasCoords = React.useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (canvasW / rect.width),
      y: (e.clientY - rect.top) * (canvasH / rect.height),
    };
  }, [canvasW, canvasH]);

  const handleMouseDown = React.useCallback((e: React.MouseEvent) => {
    const { x, y } = getCanvasCoords(e);
    const imgCol = toImageCol(x);
    const imgRow = toImageRow(y);

    // Mask mode mouse handling
    if (mode === "mask" && !lockEdit) {
      if (maskTool === "brush" && maskRef.current) {
        isPaintingRef.current = true;
        const value = maskAction === "add" ? 255 : 0;
        paintBrushCircle(maskRef.current, width, height, imgCol, imgRow, brushSize / 2, value);
        lastPaintPosRef.current = { row: imgRow, col: imgCol };
        setMaskVersion((v) => v + 1);
        return;
      }
      if ((maskTool === "rectangle" || maskTool === "ellipse") && maskRef.current) {
        shapeStartRef.current = { row: imgRow, col: imgCol };
        isDraggingShapeRef.current = true;
        setShapePreview({ r0: imgRow, c0: imgCol, r1: imgRow, c1: imgCol });
        return;
      }
      // Threshold: no mouse interaction, fall through to pan
    }

    // Crop mode mouse handling
    if (mode === "crop" && !lockEdit) {
      const sLeft = toScreenX(cropLeft);
      const sRight = toScreenX(cropRight);
      const sTop = toScreenY(cropTop);
      const sBottom = toScreenY(cropBottom);
      const hitMode = hitTestCrop(x, y, sLeft, sRight, sTop, sBottom);
      setDragMode(hitMode);
      dragStartRef.current = {
        mouseX: e.clientX,
        mouseY: e.clientY,
        cropTop,
        cropLeft,
        cropBottom,
        cropRight,
        panX,
        panY,
      };
      return;
    }

    // Pan (mask mode fallthrough for threshold / outside image)
    if (lockView) return;
    setDragMode("pan");
    dragStartRef.current = {
      mouseX: e.clientX,
      mouseY: e.clientY,
      cropTop, cropLeft, cropBottom, cropRight,
      panX, panY,
    };
  }, [getCanvasCoords, mode, lockEdit, lockView, maskTool, maskAction, brushSize, width, height, cropTop, cropLeft, cropBottom, cropRight, toScreenX, toScreenY, toImageCol, toImageRow, panX, panY]);

  const handleMouseMove = React.useCallback((e: React.MouseEvent) => {
    const { x, y } = getCanvasCoords(e);

    // Cursor readout (always)
    const imgCol = toImageCol(x);
    const imgRow = toImageRow(y);
    if (imgCol >= 0 && imgCol < width && imgRow >= 0 && imgRow < height && rawDataRef.current) {
      const col = Math.floor(imgCol);
      const row = Math.floor(imgRow);
      setCursorInfo({ row, col, value: rawDataRef.current[row * width + col] });
    } else {
      setCursorInfo(null);
    }

    // Mask mode: brush painting
    if (!lockEdit && mode === "mask" && maskTool === "brush" && isPaintingRef.current && maskRef.current) {
      const value = maskAction === "add" ? 255 : 0;
      if (lastPaintPosRef.current) {
        paintBrushLine(maskRef.current, width, height,
          lastPaintPosRef.current.col, lastPaintPosRef.current.row,
          imgCol, imgRow, brushSize / 2, value);
      } else {
        paintBrushCircle(maskRef.current, width, height, imgCol, imgRow, brushSize / 2, value);
      }
      lastPaintPosRef.current = { row: imgRow, col: imgCol };
      setMaskVersion((v) => v + 1);
      return;
    }

    // Mask mode: shape drag
    if (!lockEdit && mode === "mask" && isDraggingShapeRef.current && shapeStartRef.current) {
      setShapePreview({
        r0: shapeStartRef.current.row,
        c0: shapeStartRef.current.col,
        r1: imgRow,
        c1: imgCol,
      });
      return;
    }

    if (mode === "crop" && lockEdit && dragMode === "none") {
      setCursorStyle(lockView ? "default" : "grab");
    }

    // Crop mode: handle dragging
    if (!lockEdit && mode === "crop") {
      if (dragMode === "none" || !dragStartRef.current) {
        const sLeft = toScreenX(cropLeft);
        const sRight = toScreenX(cropRight);
        const sTop = toScreenY(cropTop);
        const sBottom = toScreenY(cropBottom);
        const hitMode = hitTestCrop(x, y, sLeft, sRight, sTop, sBottom);
        setCursorStyle(getCursorForMode(hitMode));
        return;
      }

      const ds = dragStartRef.current;
      const dx = e.clientX - ds.mouseX;
      const dy = e.clientY - ds.mouseY;
      const imgDeltaCol = dx / (displayScale * zoom);
      const imgDeltaRow = dy / (displayScale * zoom);

      if (dragMode === "pan") {
        setPanX(ds.panX + dx);
        setPanY(ds.panY + dy);
        return;
      }

      if (dragMode === "move") {
        const newTop = Math.round(ds.cropTop + imgDeltaRow);
        const newLeft = Math.round(ds.cropLeft + imgDeltaCol);
        const h = ds.cropBottom - ds.cropTop;
        const w = ds.cropRight - ds.cropLeft;
        setCropTop(newTop);
        setCropLeft(newLeft);
        setCropBottom(newTop + h);
        setCropRight(newLeft + w);
        return;
      }

      // Resize modes
      let newTop = ds.cropTop;
      let newLeft = ds.cropLeft;
      let newBottom = ds.cropBottom;
      let newRight = ds.cropRight;

      if (dragMode === "resize-t" || dragMode === "resize-tl" || dragMode === "resize-tr") {
        newTop = Math.round(ds.cropTop + imgDeltaRow);
      }
      if (dragMode === "resize-b" || dragMode === "resize-bl" || dragMode === "resize-br") {
        newBottom = Math.round(ds.cropBottom + imgDeltaRow);
      }
      if (dragMode === "resize-l" || dragMode === "resize-tl" || dragMode === "resize-bl") {
        newLeft = Math.round(ds.cropLeft + imgDeltaCol);
      }
      if (dragMode === "resize-r" || dragMode === "resize-tr" || dragMode === "resize-br") {
        newRight = Math.round(ds.cropRight + imgDeltaCol);
      }

      if (newBottom - newTop < 1) {
        if (dragMode.includes("t")) newTop = newBottom - 1;
        else newBottom = newTop + 1;
      }
      if (newRight - newLeft < 1) {
        if (dragMode.includes("l")) newLeft = newRight - 1;
        else newRight = newLeft + 1;
      }

      setCropTop(newTop);
      setCropLeft(newLeft);
      setCropBottom(newBottom);
      setCropRight(newRight);
      return;
    }

    // Pan (mask mode threshold / fallback)
    if (!lockView && dragMode === "pan" && dragStartRef.current) {
      const ds = dragStartRef.current;
      const dx = e.clientX - ds.mouseX;
      const dy = e.clientY - ds.mouseY;
      setPanX(ds.panX + dx);
      setPanY(ds.panY + dy);
    }
  }, [mode, lockEdit, lockView, maskTool, maskAction, brushSize, dragMode, getCanvasCoords, cropTop, cropLeft, cropBottom, cropRight, displayScale, zoom, toImageCol, toImageRow, toScreenX, toScreenY, width, height, setCropTop, setCropLeft, setCropBottom, setCropRight, setPanX, setPanY]);

  const handleMouseUp = React.useCallback(() => {
    // Brush: stop painting, sync
    if (isPaintingRef.current) {
      isPaintingRef.current = false;
      lastPaintPosRef.current = null;
      syncMaskToPython();
    }

    // Shape: apply and sync
    if (isDraggingShapeRef.current && shapeStartRef.current && maskRef.current && shapePreview) {
      const value = maskAction === "add" ? 255 : 0;
      const { r0, c0, r1, c1 } = shapePreview;
      if (maskTool === "rectangle") {
        fillRectMask(maskRef.current, width, height, r0, c0, r1, c1, value);
      } else if (maskTool === "ellipse") {
        const centerRow = (r0 + r1) / 2;
        const centerCol = (c0 + c1) / 2;
        const radiusRow = Math.abs(r1 - r0) / 2;
        const radiusCol = Math.abs(c1 - c0) / 2;
        fillEllipseMask(maskRef.current, width, height, centerRow, centerCol, radiusRow, radiusCol, value);
      }
      setMaskVersion((v) => v + 1);
      syncMaskToPython();
    }
    isDraggingShapeRef.current = false;
    shapeStartRef.current = null;
    setShapePreview(null);

    setDragMode("none");
    dragStartRef.current = null;
  }, [syncMaskToPython, maskAction, maskTool, width, height, shapePreview]);

  const handleMouseLeave = React.useCallback(() => {
    if (isPaintingRef.current) {
      isPaintingRef.current = false;
      lastPaintPosRef.current = null;
      syncMaskToPython();
    }
    isDraggingShapeRef.current = false;
    shapeStartRef.current = null;
    setShapePreview(null);
    setDragMode("none");
    dragStartRef.current = null;
    setCursorInfo(null);
  }, [syncMaskToPython]);

  // ── Wheel zoom ───────────────────────────────────────────────────────
  const handleWheel = React.useCallback((e: React.WheelEvent) => {
    if (lockView) return;
    const { x, y } = getCanvasCoords(e);
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * factor));

    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const mouseImageX = (x - cx - panX) / zoom + cx;
    const mouseImageY = (y - cy - panY) / zoom + cy;
    const newPanX = x - (mouseImageX - cx) * newZoom - cx;
    const newPanY = y - (mouseImageY - cy) * newZoom - cy;

    setZoom(newZoom);
    setPanX(newPanX);
    setPanY(newPanY);
  }, [lockView, getCanvasCoords, zoom, panX, panY, canvasW, canvasH]);

  const handleReset = React.useCallback(() => {
    if (lockView) return;
    setZoom(1);
    setPanX(0);
    setPanY(0);
  }, [lockView]);

  // ── Copy / Export / Figure ──────────────────────────────────────────
  const handleCopy = React.useCallback(async () => {
    if (lockExport) return;
    if (!canvasRef.current) return;
    try {
      const blob = await new Promise<Blob | null>(resolve => canvasRef.current!.toBlob(resolve, "image/png"));
      if (!blob) return;
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    } catch {
      canvasRef.current.toBlob((b) => {
        if (b) downloadBlob(b, `edit2d_${labels?.[selectedIdx] || "image"}.png`);
      }, "image/png");
    }
  }, [labels, selectedIdx, lockExport]);

  const handleExport = React.useCallback(() => {
    if (lockExport) return;
    setExportAnchor(null);
    if (!canvasRef.current) return;
    canvasRef.current.toBlob((blob) => {
      if (!blob) return;
      downloadBlob(blob, `edit2d_${labels?.[selectedIdx] || "image"}.png`);
    }, "image/png");
  }, [labels, selectedIdx, lockExport]);

  const handleExportFigure = React.useCallback((withColorbar: boolean) => {
    if (lockExport) return;
    setExportAnchor(null);
    const raw = rawDataRef.current;
    if (!raw) return;

    const processed = logScale ? applyLogScale(raw) : raw;
    const lut = COLORMAPS[cmap] || COLORMAPS.gray;

    let vmin: number, vmax: number;
    if (autoContrast) {
      ({ vmin, vmax } = percentileClip(processed, 2, 98));
    } else {
      ({ vmin, vmax } = sliderRange(imageDataRange.min, imageDataRange.max, imageVminPct, imageVmaxPct));
    }

    const offscreen = renderToOffscreen(processed, width, height, lut, vmin, vmax);
    if (!offscreen) return;

    const figCanvas = exportFigure({
      imageCanvas: offscreen,
      title: title || undefined,
      lut,
      vmin,
      vmax,
      logScale,
      pixelSize: pixelSize > 0 ? pixelSize : undefined,
      showColorbar: withColorbar,
      showScaleBar: pixelSize > 0,
    });

    figCanvas.toBlob((blob) => {
      if (blob) downloadBlob(blob, `edit2d_${labels?.[selectedIdx] || "image"}_figure.png`);
    }, "image/png");
  }, [logScale, cmap, autoContrast, imageDataRange, imageVminPct, imageVmaxPct, width, height, title, pixelSize, labels, selectedIdx, lockExport]);

  // ── Keyboard ─────────────────────────────────────────────────────────
  const handleKeyDown = React.useCallback((e: React.KeyboardEvent) => {
    if (e.key === "r" || e.key === "R") {
      if (lockView) return;
      handleReset();
      e.preventDefault();
      return;
    }

    // Mask mode shortcuts
    if (mode === "mask") {
      if (lockEdit) return;
      if (e.key === "b" || e.key === "B") { setMaskTool("brush"); e.preventDefault(); }
      else if (e.key === "t" || e.key === "T") { setMaskTool("rectangle"); e.preventDefault(); }
      else if (e.key === "e" || e.key === "E") { setMaskTool("ellipse"); e.preventDefault(); }
      else if (e.key === "i" || e.key === "I") { handleInvertMask(); e.preventDefault(); }
      else if (e.key === "c" || e.key === "C") { handleClearMask(); e.preventDefault(); }
      else if (e.key === "x" || e.key === "X") {
        setMaskAction(maskAction === "add" ? "subtract" : "add");
        e.preventDefault();
      }
      return;
    }

    // Crop mode shortcuts
    if (e.key === "Escape") {
      if (lockEdit) return;
      setCropTop(0);
      setCropLeft(0);
      setCropBottom(height);
      setCropRight(width);
      e.preventDefault();
    } else if (e.key === "ArrowLeft") {
      if (e.shiftKey && nImages > 1) {
        if (lockNavigation) return;
        setSelectedIdx(Math.max(0, selectedIdx - 1));
      } else {
        if (lockEdit) return;
        setCropLeft(cropLeft - 1);
        setCropRight(cropRight - 1);
      }
      e.preventDefault();
    } else if (e.key === "ArrowRight") {
      if (e.shiftKey && nImages > 1) {
        if (lockNavigation) return;
        setSelectedIdx(Math.min(nImages - 1, selectedIdx + 1));
      } else {
        if (lockEdit) return;
        setCropLeft(cropLeft + 1);
        setCropRight(cropRight + 1);
      }
      e.preventDefault();
    }
  }, [mode, lockView, lockEdit, lockNavigation, handleReset, handleInvertMask, handleClearMask, maskAction, height, width, cropLeft, cropRight, nImages, selectedIdx, setCropTop, setCropLeft, setCropBottom, setCropRight, setSelectedIdx, setMaskTool, setMaskAction]);

  // ── Cursor style ────────────────────────────────────────────────────
  const canvasCursor = React.useMemo(() => {
    if (dragMode !== "none") {
      return dragMode === "pan" ? "grabbing" : getCursorForMode(dragMode);
    }
    if (isPaintingRef.current) return "crosshair";
    if (mode === "mask") {
      if (lockEdit) return lockView ? "default" : "grab";
      if (maskTool === "brush") return "crosshair";
      if (maskTool === "rectangle" || maskTool === "ellipse") return "crosshair";
      return "default";
    }
    if (mode === "crop" && lockEdit) return lockView ? "default" : "grab";
    return cursorStyle;
  }, [dragMode, mode, lockEdit, lockView, maskTool, cursorStyle]);

  // ── Keyboard shortcuts for info tooltip ─────────────────────────────
  const shortcutItems: [string, string][] = mode === "mask" ? [
    ["Scroll", "Zoom in/out"],
    ["Dbl-click", "Reset view"],
    ["R", "Reset zoom"],
    ["B", "Brush tool"],
    ["T", "Rectangle tool"],
    ["E", "Ellipse tool"],
    ["X", "Toggle add/subtract"],
    ["I", "Invert mask"],
    ["C", "Clear mask"],
  ] : [
    ["Scroll", "Zoom in/out"],
    ["Dbl-click", "Reset view"],
    ["R", "Reset zoom"],
    ["Esc", "Reset crop to full image"],
    ["\u2190 / \u2192", "Nudge crop region"],
    ["\u21e7+\u2190/\u2192", "Prev/next image"],
    ["Drag handle", "Resize crop region"],
    ["Drag inside", "Move crop region"],
  ];

  // ── Display controls (shared between modes) ─────────────────────────
  const displayControlsRow = (
    <Box
      sx={{
        ...controlRow,
        border: `1px solid ${themeColors.border}`,
        bgcolor: themeColors.controlBg,
        opacity: lockDisplay ? 0.5 : 1,
        pointerEvents: lockDisplay ? "none" : "auto",
      }}
    >
      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Scale:</Typography>
      <Select disabled={lockDisplay} value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps}>
        <MenuItem value="linear">Lin</MenuItem>
        <MenuItem value="log">Log</MenuItem>
      </Select>
      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Auto:</Typography>
      <Switch checked={autoContrast} onChange={(e) => setAutoContrast(e.target.checked)} disabled={lockDisplay} size="small" sx={switchStyles.small} />
      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Color:</Typography>
      <Select disabled={lockDisplay} size="small" value={cmap} onChange={(e) => setCmap(e.target.value)} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }}>
        {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
      </Select>
    </Box>
  );

  // ── Render ───────────────────────────────────────────────────────────
  return (
    <Box
      className="edit2d-root"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onMouseEnter={() => setIsRootHovered(true)}
      onMouseLeave={() => setIsRootHovered(false)}
      sx={{ ...container.root, outline: "none", "&:focus": { outline: "none" } }}
    >
      {/* Header */}
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28, maxWidth: canvasW }}>
        <Stack direction="row" alignItems="center" spacing={0.5}>
          <Typography variant="caption" sx={{ ...typography.label, ...typography.title }}>{title || "Edit2D"}</Typography>
          {!hideMode && (
            <Stack direction="row" spacing={0} sx={{ ml: 1 }}>
              <Button
                size="small"
                disabled={lockMode}
                variant={mode === "crop" ? "contained" : "outlined"}
                onClick={() => { if (!lockMode) setMode("crop"); }}
                sx={{ ...compactButton, borderTopRightRadius: 0, borderBottomRightRadius: 0, minWidth: 40 }}
              >
                CROP
              </Button>
              <Button
                size="small"
                disabled={lockMode}
                variant={mode === "mask" ? "contained" : "outlined"}
                onClick={() => { if (!lockMode) setMode("mask"); }}
                sx={{ ...compactButton, borderTopLeftRadius: 0, borderBottomLeftRadius: 0, minWidth: 40 }}
              >
                MASK
              </Button>
            </Stack>
          )}
          <InfoTooltip theme={themeInfo.theme} text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
            <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Crop: Drag handles to resize, drag inside to move crop region.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Mask: Paint binary mask with brush, rectangle, ellipse, or threshold tools.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Fill: Value used for padded regions outside the original image bounds.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto: Percentile-based contrast — clips outliers for better visibility.</Typography>
            <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
            <KeyboardShortcuts items={shortcutItems} />
          </Box>} />
        </Stack>
        <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
          <Tooltip title="Customize visible controls" arrow placement="top">
            <IconButton
              size="small"
              onClick={(e) => setControlsAnchor(e.currentTarget)}
              sx={{
                p: 0.25,
                opacity: isRootHovered || Boolean(controlsAnchor) ? 1 : 0,
                pointerEvents: isRootHovered || Boolean(controlsAnchor) ? "auto" : "none",
                transition: "opacity 150ms ease",
                color: themeColors.textMuted,
              }}
            >
              <TuneIcon sx={{ fontSize: 16 }} />
            </IconButton>
          </Tooltip>
          <Menu
            anchorEl={controlsAnchor}
            open={Boolean(controlsAnchor)}
            onClose={() => setControlsAnchor(null)}
            anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
            transformOrigin={{ vertical: "top", horizontal: "left" }}
            sx={{ zIndex: 9999 }}
          >
            <MenuItem
              disabled={hideDisplay}
              onClick={() => { if (!hideDisplay) setShowDisplayControlsGroup((prev) => !prev); }}
              sx={{ display: "flex", justifyContent: "space-between", gap: 1.5, minWidth: 210 }}
            >
              <Typography sx={{ fontSize: 11 }}>Display controls</Typography>
              <Switch
                size="small"
                checked={showDisplayControlsGroup}
                disabled={hideDisplay}
                onClick={(e) => e.stopPropagation()}
                onChange={(e) => setShowDisplayControlsGroup(e.target.checked)}
              />
            </MenuItem>
            <MenuItem
              disabled={hideEdit}
              onClick={() => { if (!hideEdit) setShowEditControlsGroup((prev) => !prev); }}
              sx={{ display: "flex", justifyContent: "space-between", gap: 1.5, minWidth: 210 }}
            >
              <Typography sx={{ fontSize: 11 }}>Edit controls</Typography>
              <Switch
                size="small"
                checked={showEditControlsGroup}
                disabled={hideEdit}
                onClick={(e) => e.stopPropagation()}
                onChange={(e) => setShowEditControlsGroup(e.target.checked)}
              />
            </MenuItem>
            <MenuItem
              disabled={hideHistogram}
              onClick={() => { if (!hideHistogram) setShowHistogramGroup((prev) => !prev); }}
              sx={{ display: "flex", justifyContent: "space-between", gap: 1.5, minWidth: 210 }}
            >
              <Typography sx={{ fontSize: 11 }}>Histogram</Typography>
              <Switch
                size="small"
                checked={showHistogramGroup}
                disabled={hideHistogram}
                onClick={(e) => e.stopPropagation()}
                onChange={(e) => setShowHistogramGroup(e.target.checked)}
              />
            </MenuItem>
            <MenuItem
              disabled={hideStats || lockStats}
              onClick={() => { if (!(hideStats || lockStats)) setShowStats((prev) => !prev); }}
              sx={{ display: "flex", justifyContent: "space-between", gap: 1.5, minWidth: 210 }}
            >
              <Typography sx={{ fontSize: 11 }}>Stats bar</Typography>
              <Switch
                size="small"
                checked={showStats}
                disabled={hideStats || lockStats}
                onClick={(e) => e.stopPropagation()}
                onChange={(e) => setShowStats(e.target.checked)}
              />
            </MenuItem>
            <MenuItem
              onClick={() => {
                if (!hideDisplay) setShowDisplayControlsGroup(true);
                if (!hideEdit) setShowEditControlsGroup(true);
                if (!hideHistogram) setShowHistogramGroup(true);
                if (!(hideStats || lockStats)) setShowStats(true);
              }}
              sx={{ fontSize: 11, color: themeColors.accent }}
            >
              All
            </MenuItem>
            <MenuItem
              onClick={() => {
                if (!hideDisplay) setShowDisplayControlsGroup(false);
                if (!hideEdit) setShowEditControlsGroup(true);
                if (!hideHistogram) setShowHistogramGroup(false);
                if (!(hideStats || lockStats)) setShowStats(false);
              }}
              sx={{ fontSize: 11 }}
            >
              Compact
            </MenuItem>
            <MenuItem
              onClick={() => {
                if (!hideDisplay) setShowDisplayControlsGroup(false);
                if (!hideEdit) setShowEditControlsGroup(true);
                if (!hideHistogram) setShowHistogramGroup(true);
                if (!(hideStats || lockStats)) setShowStats(true);
                if (!lockMode) setMode("mask");
              }}
              sx={{ fontSize: 11 }}
            >
              Mask Focus
            </MenuItem>
            <MenuItem
              onClick={() => {
                if (!hideDisplay) setShowDisplayControlsGroup(true);
                if (!hideEdit) setShowEditControlsGroup(true);
                if (!hideHistogram) setShowHistogramGroup(false);
                if (!(hideStats || lockStats)) setShowStats(true);
                if (!lockMode) setMode("crop");
              }}
              sx={{ fontSize: 11 }}
            >
              Crop Focus
            </MenuItem>
          </Menu>
          {!hideNavigation && nImages > 1 && (
            <Stack direction="row" alignItems="center" spacing={0}>
              <IconButton size="small" onClick={() => setSelectedIdx(Math.max(0, selectedIdx - 1))} disabled={lockNavigation || selectedIdx === 0} sx={{ p: 0.25, color: themeColors.textMuted }}>
                <NavigateBeforeIcon sx={{ fontSize: 18 }} />
              </IconButton>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted, minWidth: 60, textAlign: "center" }}>
                {labels[selectedIdx] || `${selectedIdx + 1}/${nImages}`}
              </Typography>
              <IconButton size="small" onClick={() => setSelectedIdx(Math.min(nImages - 1, selectedIdx + 1))} disabled={lockNavigation || selectedIdx === nImages - 1} sx={{ p: 0.25, color: themeColors.textMuted }}>
                <NavigateNextIcon sx={{ fontSize: 18 }} />
              </IconButton>
            </Stack>
          )}
          {!hideExport && (
            <>
              <Button size="small" sx={compactButton} disabled={lockExport} onClick={handleCopy}>COPY</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={lockExport} onClick={(e) => { if (!lockExport) setExportAnchor(e.currentTarget); }}>Export</Button>
              <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                <MenuItem disabled={lockExport} onClick={() => handleExportFigure(true)} sx={{ fontSize: 12 }}>Figure + colorbar</MenuItem>
                <MenuItem disabled={lockExport} onClick={() => handleExportFigure(false)} sx={{ fontSize: 12 }}>Figure</MenuItem>
                <MenuItem disabled={lockExport} onClick={handleExport} sx={{ fontSize: 12 }}>PNG</MenuItem>
              </Menu>
            </>
          )}
          {!hideView && (
            <Button size="small" sx={compactButton} disabled={lockView || !needsReset} onClick={handleReset}>RESET</Button>
          )}
        </Stack>
      </Stack>

      {/* Canvas */}
      <Box
        ref={canvasContainerRef}
        sx={{ ...container.imageBox, width: canvasW, height: canvasH, cursor: canvasCursor }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onWheel={handleWheel}
        onDoubleClick={handleReset}
      >
        <canvas ref={canvasRef} width={canvasW} height={canvasH} style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }} />
        <canvas ref={overlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} />
        <canvas ref={uiRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} />
        {cursorInfo && (
          <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
            <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
              ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
            </Typography>
          </Box>
        )}
        {!hideView && (
          <Box
            onMouseDown={handleResizeStart}
            sx={{
              position: "absolute",
              bottom: 0,
              right: 0,
              width: 16,
              height: 16,
              cursor: lockView ? "default" : "nwse-resize",
              opacity: lockView ? 0.3 : 0.6,
              pointerEvents: lockView ? "none" : "auto",
              background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`,
              "&:hover": { opacity: lockView ? 0.3 : 1 },
            }}
          />
        )}
      </Box>

      {/* Stats bar */}
      {!hideStats && showStats && (
        <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", maxWidth: canvasW, boxSizing: "border-box", flexWrap: "wrap" }}>
          <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMean)}</Box></Typography>
          <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMin)}</Box></Typography>
          <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMax)}</Box></Typography>
          <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsStd)}</Box></Typography>
          <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 14 }} />
          {mode === "crop" ? (
            <Typography sx={{ fontSize: 11, color: themeColors.textMuted, fontFamily: "monospace" }}>
              Crop <Box component="span" sx={{ color: themeColors.accent }}>{cropW}{"\u00d7"}{cropH}</Box> at <Box component="span" sx={{ color: themeColors.accent }}>({cropTop}, {cropLeft})</Box>
              {hasPadding && <Box component="span" sx={{ color: "#ffa726", ml: 0.5 }}>(pad)</Box>}
            </Typography>
          ) : (
            <Typography sx={{ fontSize: 11, color: themeColors.textMuted, fontFamily: "monospace" }}>
              Mask: <Box component="span" sx={{ color: themeColors.accent }}>{maskCoverage.count} px ({maskCoverage.pct.toFixed(1)}%)</Box>
            </Typography>
          )}
        </Box>
      )}

      {/* Controls */}
      {showControls && showAnyControlGroups && (
        <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: canvasW, boxSizing: "border-box" }}>
          {showLeftControlGroups && (
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
              {mode === "crop" ? (
                <>
                  {wantDisplayControlsGroup && displayControlsRow}
                  {wantEditControlsGroup && (
                    <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Fill:</Typography>
                      <Typography sx={{ ...typography.value, color: themeColors.accent }}>{fillValue}</Typography>
                      {zoom !== 1 && (
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.accent, fontWeight: "bold" }}>{zoom.toFixed(1)}{"\u00d7"}</Typography>
                      )}
                    </Box>
                  )}
                </>
              ) : (
                <>
                  {wantEditControlsGroup && (
                    <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockEdit ? 0.5 : 1, pointerEvents: lockEdit ? "none" : "auto" }}>
                      {(["brush", "rectangle", "ellipse", "threshold"] as const).map((tool) => (
                        <Button
                          key={tool}
                          size="small"
                          disabled={lockEdit}
                          variant={maskTool === tool ? "contained" : "outlined"}
                          onClick={() => { if (!lockEdit) setMaskTool(tool); }}
                          sx={{ ...compactButton, minWidth: 32 }}
                        >
                          {tool === "rectangle" ? "RECT" : tool === "threshold" ? "THRESH" : tool.toUpperCase()}
                        </Button>
                      ))}
                      {maskTool === "brush" && (
                        <>
                          <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 16, mx: 0 }} />
                          <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Size:</Typography>
                          <Slider
                            value={brushSize}
                            disabled={lockEdit}
                            onChange={(_, v) => { if (!lockEdit) setBrushSize(v as number); }}
                            min={1}
                            max={100}
                            size="small"
                            sx={{ ...sliderStyles.small, width: 60 }}
                          />
                          <Typography sx={{ ...typography.value }}>{brushSize}</Typography>
                        </>
                      )}
                      <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 16, mx: 0 }} />
                      <Button
                        size="small"
                        disabled={lockEdit}
                        variant={maskAction === "add" ? "contained" : "outlined"}
                        onClick={() => { if (!lockEdit) setMaskAction("add"); }}
                        sx={{ ...compactButton, borderTopRightRadius: 0, borderBottomRightRadius: 0, minWidth: 32 }}
                      >
                        ADD
                      </Button>
                      <Button
                        size="small"
                        disabled={lockEdit}
                        variant={maskAction === "subtract" ? "contained" : "outlined"}
                        onClick={() => { if (!lockEdit) setMaskAction("subtract"); }}
                        sx={{ ...compactButton, borderTopLeftRadius: 0, borderBottomLeftRadius: 0, minWidth: 32 }}
                      >
                        SUB
                      </Button>
                    </Box>
                  )}
                  {wantEditControlsGroup && (
                    <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockEdit ? 0.5 : 1, pointerEvents: lockEdit ? "none" : "auto" }}>
                      <Button size="small" sx={compactButton} disabled={lockEdit} onClick={handleInvertMask}>INVERT</Button>
                      <Button size="small" sx={compactButton} disabled={lockEdit} onClick={handleClearMask}>CLEAR</Button>
                      <Typography sx={{ ...typography.value, color: themeColors.accent }}>{maskCoverage.pct.toFixed(1)}%</Typography>
                    </Box>
                  )}
                  {wantDisplayControlsGroup && (
                    <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Scale:</Typography>
                      <Select disabled={lockDisplay} value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps}>
                        <MenuItem value="linear">Lin</MenuItem>
                        <MenuItem value="log">Log</MenuItem>
                      </Select>
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Auto:</Typography>
                      <Switch checked={autoContrast} onChange={(e) => setAutoContrast(e.target.checked)} disabled={lockDisplay} size="small" sx={switchStyles.small} />
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Color:</Typography>
                      <Select disabled={lockDisplay} size="small" value={cmap} onChange={(e) => setCmap(e.target.value)} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }}>
                        {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                      </Select>
                    </Box>
                  )}
                </>
              )}
            </Box>
          )}
          {wantHistogramGroup && (
            <Box sx={{ display: "flex", flexDirection: "column", alignItems: showLeftControlGroups ? "flex-end" : "flex-start", justifyContent: "center", opacity: lockHistogram ? 0.5 : 1, pointerEvents: lockHistogram ? "none" : "auto" }}>
              <Histogram
                data={imageHistogramData}
                vminPct={imageVminPct}
                vmaxPct={imageVmaxPct}
                onRangeChange={(min, max) => {
                  if (lockHistogram) return;
                  if (autoContrast) setAutoContrast(false);
                  setImageVminPct(min);
                  setImageVmaxPct(max);
                }}
                width={110}
                height={58}
                theme={themeInfo.theme}
                dataMin={imageDataRange.min}
                dataMax={imageDataRange.max}
              />
            </Box>
          )}
        </Box>
      )}
    </Box>
  );
}

export const render = createRender(Edit2D);
