import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import Switch from "@mui/material/Switch";
import Tooltip from "@mui/material/Tooltip";
import { useTheme } from "../theme";
import { extractFloat32, formatNumber, downloadBlob } from "../format";
import { applyLogScale, findDataRange } from "../stats";
import { COLORMAPS, renderToOffscreen } from "../colormaps";
import { computeHistogramFromBytes } from "../histogram";
import { roundToNiceValue, formatScaleLabel } from "../scalebar";
import { computeToolVisibility } from "../tool-parity";
import { ControlCustomizer } from "../control-customizer";
import "./bin.css";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const DPR = window.devicePixelRatio || 1;
const CANVAS_SIZE = 250;
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };

const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
  section: { fontSize: 11, fontWeight: "bold" as const },
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
  "&.Mui-disabled": { color: "#666", borderColor: "#444" },
};

const switchStyles = {
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};

const sliderStyles = {
  small: {
    "& .MuiSlider-thumb": { width: 12, height: 12 },
    "& .MuiSlider-rail": { height: 3 },
    "& .MuiSlider-track": { height: 3 },
  },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function optionsForFactor(maxValue: number): number[] {
  const maxSafe = Math.max(1, Math.floor(maxValue));
  if (maxSafe <= 16) {
    return Array.from({ length: maxSafe }, (_, i) => i + 1);
  }
  const base = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128];
  const filtered = base.filter((v) => v <= maxSafe);
  if (!filtered.includes(maxSafe)) filtered.push(maxSafe);
  return filtered.sort((a, b) => a - b);
}

function stepOption(options: number[], current: number, delta: -1 | 1): number {
  if (!options.length) return current;
  const sorted = [...options].sort((a, b) => a - b);
  const idx = Math.max(0, sorted.indexOf(current));
  if (delta < 0) return sorted[Math.max(0, idx - 1)];
  return sorted[Math.min(sorted.length - 1, idx + 1)];
}

function isEditableElement(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName.toLowerCase();
  if (target.isContentEditable) return true;
  if (target.getAttribute("role") === "textbox") return true;
  return tag === "input" || tag === "textarea" || tag === "select";
}

function statsLine(stats: number[]): string {
  if (!stats || stats.length < 4) return "";
  const [mean, min, max, std] = stats;
  return `mean ${formatNumber(mean)} | min ${formatNumber(min)} | max ${formatNumber(max)} | std ${formatNumber(std)}`;
}

// ---------------------------------------------------------------------------
// InfoTooltip
// ---------------------------------------------------------------------------

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
            maxWidth: 320,
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

// ---------------------------------------------------------------------------
// InteractivePanel — reusable zoom/pan canvas panel
// ---------------------------------------------------------------------------

interface PanelProps {
  label: string;
  rows: number;
  cols: number;
  bytes: DataView;
  cmap: string;
  logScale: boolean;
  canvasSize: number;
  borderColor: string;
  textColor: string;
  mutedColor: string;
  accentColor: string;
  lockView: boolean;
  stats?: number[];
  hideStats?: boolean;
  pixelSize?: number;
  pixelUnit?: string;
  resetKey?: number;
  overlayRenderer?: (
    ctx: CanvasRenderingContext2D,
    cssW: number,
    cssH: number,
    zoom: number,
    panX: number,
    panY: number,
  ) => void;
  canvasRef?: React.RefObject<HTMLCanvasElement | null>;
}

function InteractivePanel({
  label,
  rows,
  cols,
  bytes,
  cmap,
  logScale,
  canvasSize: size,
  borderColor,
  textColor,
  mutedColor,
  accentColor,
  lockView,
  stats,
  hideStats,
  pixelSize,
  pixelUnit,
  resetKey,
  overlayRenderer,
  canvasRef: externalCanvasRef,
}: PanelProps) {
  const internalCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const canvasRef = externalCanvasRef || internalCanvasRef;
  const overlayRef = React.useRef<HTMLCanvasElement>(null);

  const [zoom, setZoom] = React.useState(1);
  const [panX, setPanX] = React.useState(0);
  const [panY, setPanY] = React.useState(0);
  const [isDragging, setIsDragging] = React.useState(false);
  const dragStart = React.useRef<{ x: number; y: number; pX: number; pY: number } | null>(null);

  // Reset zoom/pan when resetKey changes (R key pressed)
  React.useEffect(() => {
    if (resetKey === undefined || resetKey === 0) return;
    setZoom(1);
    setPanX(0);
    setPanY(0);
  }, [resetKey]);

  const parsed = React.useMemo(() => extractFloat32(bytes), [bytes]);

  // Compute aspect-correct CSS dimensions
  const cssW = React.useMemo(() => {
    if (rows <= 0 || cols <= 0) return size;
    const aspect = cols / rows;
    return aspect >= 1 ? size : Math.round(size * aspect);
  }, [rows, cols, size]);

  const cssH = React.useMemo(() => {
    if (rows <= 0 || cols <= 0) return size;
    const aspect = cols / rows;
    return aspect >= 1 ? Math.round(size / aspect) : size;
  }, [rows, cols, size]);

  // Render colormapped image
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !parsed || parsed.length === 0) return;
    if (rows <= 0 || cols <= 0) return;
    if (parsed.length !== rows * cols) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const data = logScale ? applyLogScale(parsed) : parsed;
    const range = findDataRange(data);
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const offscreen = renderToOffscreen(data, cols, rows, lut, range.min, range.max);
    if (!offscreen) return;

    canvas.width = cssW;
    canvas.height = cssH;
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.imageSmoothingEnabled = false;

    if (zoom !== 1 || panX !== 0 || panY !== 0) {
      ctx.save();
      ctx.translate(panX, panY);
      ctx.scale(zoom, zoom);
      ctx.drawImage(offscreen, 0, 0, cols, rows, 0, 0, cssW, cssH);
      ctx.restore();
    } else {
      ctx.drawImage(offscreen, 0, 0, cols, rows, 0, 0, cssW, cssH);
    }
  }, [parsed, rows, cols, cmap, logScale, cssW, cssH, zoom, panX, panY]);

  // Render overlay (BF/ADF circles on detector panels + scale bar)
  React.useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;

    overlay.width = cssW * DPR;
    overlay.height = cssH * DPR;
    overlay.style.width = `${cssW}px`;
    overlay.style.height = `${cssH}px`;

    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (overlayRenderer) {
      ctx.save();
      ctx.scale(DPR, DPR);
      overlayRenderer(ctx, cssW, cssH, zoom, panX, panY);
      ctx.restore();
    }

    // Scale bar (drawn without zoom/pan transform — stays fixed in corner)
    if (pixelSize && pixelSize > 0 && pixelUnit && cols > 0) {
      ctx.save();
      ctx.scale(DPR, DPR);

      const scaleX = cssW / cols;
      const effectiveZoom = zoom * scaleX;
      const targetBarPx = 60;
      const barThickness = 5;
      const fontSize = 16;
      const margin = 12;

      const targetPhysical = (targetBarPx / effectiveZoom) * pixelSize;
      const nicePhysical = roundToNiceValue(targetPhysical);
      const barPx = (nicePhysical / pixelSize) * effectiveZoom;

      const barY = cssH - margin;
      const barX = cssW - barPx - margin;

      ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
      ctx.shadowBlur = 2;
      ctx.shadowOffsetX = 1;
      ctx.shadowOffsetY = 1;

      ctx.fillStyle = "white";
      ctx.fillRect(barX, barY, barPx, barThickness);

      const scaleLabel = formatScaleLabel(nicePhysical, pixelUnit as "Å" | "mrad" | "px");
      ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
      ctx.fillStyle = "white";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(scaleLabel, barX + barPx / 2, barY - 4);

      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${zoom.toFixed(1)}×`, margin, cssH - margin + barThickness);

      ctx.restore();
    }
  }, [cssW, cssH, cols, overlayRenderer, zoom, panX, panY, pixelSize, pixelUnit]);

  // Wheel scroll prevention
  React.useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    overlay.addEventListener("wheel", preventDefault, { passive: false });
    return () => overlay.removeEventListener("wheel", preventDefault);
  }, []);

  // Zoom handler (cursor-centered)
  const handleWheel = React.useCallback(
    (e: React.WheelEvent<HTMLCanvasElement>) => {
      if (lockView) return;
      e.preventDefault();
      const overlay = overlayRef.current;
      if (!overlay) return;
      const rect = overlay.getBoundingClientRect();
      const mouseX = (e.clientX - rect.left);
      const mouseY = (e.clientY - rect.top);
      const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * zoomFactor));
      const zoomRatio = newZoom / zoom;
      setZoom(newZoom);
      setPanX(mouseX - (mouseX - panX) * zoomRatio);
      setPanY(mouseY - (mouseY - panY) * zoomRatio);
    },
    [lockView, zoom, panX, panY],
  );

  // Pan handlers
  const handleMouseDown = React.useCallback(
    (e: React.MouseEvent) => {
      if (lockView) return;
      setIsDragging(true);
      dragStart.current = { x: e.clientX, y: e.clientY, pX: panX, pY: panY };
    },
    [lockView, panX, panY],
  );

  const handleMouseUp = React.useCallback(() => {
    setIsDragging(false);
    dragStart.current = null;
  }, []);

  const handleDoubleClick = React.useCallback(() => {
    if (lockView) return;
    setZoom(1);
    setPanX(0);
    setPanY(0);
  }, [lockView]);

  const [isHovered, setIsHovered] = React.useState(false);
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number } | null>(null);

  // Cursor readout — compute image (row, col) and pixel value from mouse position
  const handleCursorMove = React.useCallback(
    (e: React.MouseEvent) => {
      if (isDragging) {
        // Pan while dragging
        if (dragStart.current) {
          setPanX(dragStart.current.pX + (e.clientX - dragStart.current.x));
          setPanY(dragStart.current.pY + (e.clientY - dragStart.current.y));
        }
        return;
      }
      if (!parsed || rows <= 0 || cols <= 0) { setCursorInfo(null); return; }
      const overlay = overlayRef.current;
      if (!overlay) return;
      const rect = overlay.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      // Invert zoom/pan transform to get image-space pixel
      const imgX = (mouseX - panX) / zoom;
      const imgY = (mouseY - panY) / zoom;
      const col = Math.floor((imgX / cssW) * cols);
      const row = Math.floor((imgY / cssH) * rows);
      if (row < 0 || row >= rows || col < 0 || col >= cols) { setCursorInfo(null); return; }
      const idx = row * cols + col;
      const value = idx < parsed.length ? parsed[idx] : 0;
      setCursorInfo({ row, col, value });
    },
    [isDragging, parsed, rows, cols, cssW, cssH, zoom, panX, panY],
  );

  return (
    <Box sx={{ width: cssW }}>
      <Typography
        sx={{ ...typography.labelSmall, color: textColor, mb: "2px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
      >
        {label}
      </Typography>
      <Box
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => { setIsHovered(false); setCursorInfo(null); handleMouseUp(); }}
        sx={{
          position: "relative",
          bgcolor: "#000",
          border: `1px solid ${borderColor}`,
          width: cssW,
          height: cssH,
          overflow: "hidden",
        }}
      >
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: cssW,
            height: cssH,
            imageRendering: "pixelated",
            display: "block",
          }}
        />
        <canvas
          ref={overlayRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleCursorMove}
          onMouseUp={handleMouseUp}
          onWheel={handleWheel}
          onDoubleClick={handleDoubleClick}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            cursor: isDragging ? "grabbing" : lockView ? "default" : "grab",
          }}
        />
        {/* Cursor readout (top-right) */}
        {cursorInfo && (
          <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.5)", px: 0.5, py: 0.15, pointerEvents: "none" }}>
            <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.8)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
              ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
            </Typography>
          </Box>
        )}
        {/* Stats shown on hover (bottom) */}
        {!hideStats && stats && isHovered && (
          <Box sx={{ position: "absolute", bottom: 0, left: 0, right: 0, bgcolor: "rgba(0,0,0,0.6)", px: 0.5, py: 0.25, pointerEvents: "none" }}>
            <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.85)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {statsLine(stats)}
            </Typography>
          </Box>
        )}
      </Box>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// BinWidget — main component
// ---------------------------------------------------------------------------

function BinWidget() {
  const { themeInfo, colors: themeColors } = useTheme();

  // Shape traits
  const [scanRows] = useModelState<number>("scan_rows");
  const [scanCols] = useModelState<number>("scan_cols");
  const [detRows] = useModelState<number>("det_rows");
  const [detCols] = useModelState<number>("det_cols");

  const [binnedScanRows] = useModelState<number>("binned_scan_rows");
  const [binnedScanCols] = useModelState<number>("binned_scan_cols");
  const [binnedDetRows] = useModelState<number>("binned_det_rows");
  const [binnedDetCols] = useModelState<number>("binned_det_cols");

  const [maxScanBinRow] = useModelState<number>("max_scan_bin_row");
  const [maxScanBinCol] = useModelState<number>("max_scan_bin_col");
  const [maxDetBinRow] = useModelState<number>("max_det_bin_row");
  const [maxDetBinCol] = useModelState<number>("max_det_bin_col");

  // Bin factors
  const [scanBinRow, setScanBinRow] = useModelState<number>("scan_bin_row");
  const [scanBinCol, setScanBinCol] = useModelState<number>("scan_bin_col");
  const [detBinRow, setDetBinRow] = useModelState<number>("det_bin_row");
  const [detBinCol, setDetBinCol] = useModelState<number>("det_bin_col");

  const [binMode, setBinMode] = useModelState<string>("bin_mode");
  const [edgeMode, setEdgeMode] = useModelState<string>("edge_mode");

  // Mask ratios
  const [bfRadiusRatio, setBfRadiusRatio] = useModelState<number>("bf_radius_ratio");
  const [adfInnerRatio, setAdfInnerRatio] = useModelState<number>("adf_inner_ratio");
  const [adfOuterRatio, setAdfOuterRatio] = useModelState<number>("adf_outer_ratio");

  // Calibration
  const [pixelSizeRow] = useModelState<number>("pixel_size_row");
  const [pixelSizeCol] = useModelState<number>("pixel_size_col");
  const [pixelUnit] = useModelState<string>("pixel_unit");
  const [pixelCalibrated] = useModelState<boolean>("pixel_calibrated");
  const [kPixelSizeRow] = useModelState<number>("k_pixel_size_row");
  const [kPixelSizeCol] = useModelState<number>("k_pixel_size_col");
  const [kUnit] = useModelState<string>("k_unit");
  const [kCalibrated] = useModelState<boolean>("k_calibrated");
  const [binnedPixelSizeRow] = useModelState<number>("binned_pixel_size_row");
  const [binnedPixelSizeCol] = useModelState<number>("binned_pixel_size_col");
  const [binnedKPixelSizeRow] = useModelState<number>("binned_k_pixel_size_row");
  const [binnedKPixelSizeCol] = useModelState<number>("binned_k_pixel_size_col");

  // Preview bytes
  const [originalBfBytes] = useModelState<DataView>("original_bf_bytes");
  const [originalAdfBytes] = useModelState<DataView>("original_adf_bytes");
  const [binnedBfBytes] = useModelState<DataView>("binned_bf_bytes");
  const [binnedAdfBytes] = useModelState<DataView>("binned_adf_bytes");
  const [originalMeanDpBytes] = useModelState<DataView>("original_mean_dp_bytes");
  const [binnedMeanDpBytes] = useModelState<DataView>("binned_mean_dp_bytes");

  // Detector center
  const [centerRow] = useModelState<number>("center_row");
  const [centerCol] = useModelState<number>("center_col");
  const [binnedCenterRow] = useModelState<number>("binned_center_row");
  const [binnedCenterCol] = useModelState<number>("binned_center_col");

  // Stats
  const [originalBfStats] = useModelState<number[]>("original_bf_stats");
  const [originalAdfStats] = useModelState<number[]>("original_adf_stats");
  const [binnedBfStats] = useModelState<number[]>("binned_bf_stats");
  const [binnedAdfStats] = useModelState<number[]>("binned_adf_stats");

  // Display
  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [showControls] = useModelState<boolean>("show_controls");
  const [disabledTools, setDisabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools, setHiddenTools] = useModelState<string[]>("hidden_tools");

  // Status
  const [statusMessage] = useModelState<string>("status_message");
  const [statusLevel] = useModelState<string>("status_level");

  // Local UI state
  const [canvasSize, setCanvasSize] = React.useState(CANVAS_SIZE);
  const [isResizingCanvas, setIsResizingCanvas] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number; y: number; size: number } | null>(null);
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);
  const [showShortcutHelp, setShowShortcutHelp] = React.useState(false);
  const [resetKey, setResetKey] = React.useState(0);

  // Canvas refs for export
  const origBfRef = React.useRef<HTMLCanvasElement | null>(null);
  const origAdfRef = React.useRef<HTMLCanvasElement | null>(null);
  const binnedBfRef = React.useRef<HTMLCanvasElement | null>(null);
  const binnedAdfRef = React.useRef<HTMLCanvasElement | null>(null);

  // Factor options
  const scanRowOptions = React.useMemo(() => optionsForFactor(maxScanBinRow), [maxScanBinRow]);
  const scanColOptions = React.useMemo(() => optionsForFactor(maxScanBinCol), [maxScanBinCol]);
  const detRowOptions = React.useMemo(() => optionsForFactor(maxDetBinRow), [maxDetBinRow]);
  const detColOptions = React.useMemo(() => optionsForFactor(maxDetBinCol), [maxDetBinCol]);

  // Tool visibility
  const toolVisibility = React.useMemo(
    () => computeToolVisibility("Bin", disabledTools, hiddenTools),
    [disabledTools, hiddenTools],
  );
  const hideDisplay = toolVisibility.isHidden("display");
  const hideBinning = toolVisibility.isHidden("binning");
  const hideMask = toolVisibility.isHidden("mask");
  const hidePreview = toolVisibility.isHidden("preview");
  const hideStats = toolVisibility.isHidden("stats");
  const hideExport = toolVisibility.isHidden("export");
  const lockDisplay = toolVisibility.isLocked("display");
  const lockBinning = toolVisibility.isLocked("binning");
  const lockMask = toolVisibility.isLocked("mask");
  const lockExport = toolVisibility.isLocked("export");
  const lockView = lockDisplay;

  const colormapOptions = React.useMemo(() => ["inferno", "viridis", "gray", "magma"], []);

  // Histogram data (original BF)
  const histogramData = React.useMemo(() => {
    const parsed = extractFloat32(originalBfBytes);
    if (!parsed || parsed.length === 0) return null;
    return logScale ? applyLogScale(parsed) : parsed;
  }, [originalBfBytes, logScale]);

  const histogramBins = React.useMemo(() => computeHistogramFromBytes(histogramData), [histogramData]);

  // Histogram canvas ref
  const histogramRef = React.useRef<HTMLCanvasElement>(null);

  // Draw histogram
  React.useEffect(() => {
    const canvas = histogramRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = 100;
    const h = 32;
    canvas.width = w * DPR;
    canvas.height = h * DPR;
    ctx.scale(DPR, DPR);

    const isDark = themeColors.bg === "#1e1e1e" || themeColors.bg.startsWith("#1") || themeColors.bg.startsWith("#2");
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f0f0f0";
    ctx.fillRect(0, 0, w, h);

    const displayBins = 64;
    const binRatio = Math.floor(histogramBins.length / displayBins);
    const reducedBins: number[] = [];
    for (let i = 0; i < displayBins; i++) {
      let sum = 0;
      for (let j = 0; j < binRatio; j++) sum += histogramBins[i * binRatio + j] || 0;
      reducedBins.push(sum / binRatio);
    }

    const maxVal = Math.max(...reducedBins, 0.001);
    const barWidth = w / displayBins;
    const barColor = isDark ? "#888" : "#666";

    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (h - 2);
      ctx.fillStyle = barColor;
      ctx.fillRect(i * barWidth + 0.5, h - barHeight, Math.max(1, barWidth - 1), barHeight);
    }
  }, [histogramBins, themeColors.bg]);

  // Themed styling
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
    PaperProps: {
      sx: {
        bgcolor: themeColors.controlBg,
        color: themeColors.text,
        border: `1px solid ${themeColors.border}`,
      },
    },
  };

  // Resize handle
  const handleCanvasResizeStart = React.useCallback(
    (e: React.MouseEvent) => {
      if (lockView) return;
      e.stopPropagation();
      e.preventDefault();
      setIsResizingCanvas(true);
      setResizeStart({ x: e.clientX, y: e.clientY, size: canvasSize });
    },
    [lockView, canvasSize],
  );

  React.useEffect(() => {
    if (!isResizingCanvas) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeStart) return;
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);
      setCanvasSize(Math.max(120, Math.min(600, resizeStart.size + delta)));
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

  // Detector overlay renderers
  const makeDetectorOverlay = React.useCallback(
    (cRow: number, cCol: number, dRows: number, dCols: number) =>
      (ctx: CanvasRenderingContext2D, cssW: number, cssH: number, zm: number, pX: number, pY: number) => {
        if (dRows <= 0 || dCols <= 0) return;
        const scaleX = cssW / dCols;
        const scaleY = cssH / dRows;
        const detSize = Math.min(dRows, dCols);

        // Apply same transform as the image canvas
        ctx.save();
        ctx.translate(pX, pY);
        ctx.scale(zm, zm);

        const cx = cCol * scaleX;
        const cy = cRow * scaleY;

        // BF disk
        const bfRadiusPx = bfRadiusRatio * detSize;
        const bfRx = bfRadiusPx * scaleX;
        const bfRy = bfRadiusPx * scaleY;

        ctx.lineWidth = 2 / zm;
        ctx.shadowColor = "rgba(0,0,0,0.5)";
        ctx.shadowBlur = 2 / zm;

        ctx.beginPath();
        ctx.ellipse(cx, cy, bfRx, bfRy, 0, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(0, 255, 0, 0.12)";
        ctx.fill();
        ctx.strokeStyle = "rgba(0, 255, 0, 0.9)";
        ctx.stroke();

        ctx.shadowBlur = 0;
        ctx.font = `bold ${10 / zm}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
        ctx.fillStyle = "rgba(0, 255, 0, 0.9)";
        ctx.textAlign = "center";
        ctx.fillText("BF", cx, cy - bfRy - 4 / zm);

        // ADF annulus
        const adfInnerPx = adfInnerRatio * detSize;
        const adfOuterPx = adfOuterRatio * detSize;
        const adfInnerRx = adfInnerPx * scaleX;
        const adfInnerRy = adfInnerPx * scaleY;
        const adfOuterRx = adfOuterPx * scaleX;
        const adfOuterRy = adfOuterPx * scaleY;

        ctx.shadowBlur = 2 / zm;
        ctx.beginPath();
        ctx.ellipse(cx, cy, adfOuterRx, adfOuterRy, 0, 0, 2 * Math.PI);
        ctx.ellipse(cx, cy, adfInnerRx, adfInnerRy, 0, 2 * Math.PI, 0, true);
        ctx.fillStyle = "rgba(0, 220, 255, 0.12)";
        ctx.fill();

        ctx.beginPath();
        ctx.ellipse(cx, cy, adfOuterRx, adfOuterRy, 0, 0, 2 * Math.PI);
        ctx.strokeStyle = "rgba(0, 220, 255, 0.9)";
        ctx.stroke();

        ctx.beginPath();
        ctx.ellipse(cx, cy, adfInnerRx, adfInnerRy, 0, 0, 2 * Math.PI);
        ctx.stroke();

        ctx.shadowBlur = 0;
        ctx.fillStyle = "rgba(0, 220, 255, 0.9)";
        ctx.fillText("ADF", cx, cy - adfOuterRy - 4 / zm);

        ctx.restore();
      },
    [bfRadiusRatio, adfInnerRatio, adfOuterRatio],
  );

  const origDetOverlay = React.useMemo(
    () => makeDetectorOverlay(centerRow, centerCol, detRows, detCols),
    [makeDetectorOverlay, centerRow, centerCol, detRows, detCols],
  );
  const binnedDetOverlay = React.useMemo(
    () => makeDetectorOverlay(binnedCenterRow, binnedCenterCol, binnedDetRows, binnedDetCols),
    [makeDetectorOverlay, binnedCenterRow, binnedCenterCol, binnedDetRows, binnedDetCols],
  );

  // Export
  const createComposite = React.useCallback((): HTMLCanvasElement | null => {
    const refs = [origBfRef, origAdfRef, binnedBfRef, binnedAdfRef];
    const canvases = refs.map((r) => r.current).filter((c): c is HTMLCanvasElement => c !== null);
    if (canvases.length < 4) return null;

    const gap = 4;
    const w0 = Math.max(canvases[0].width, canvases[2].width);
    const w1 = Math.max(canvases[1].width, canvases[3].width);
    const h0 = Math.max(canvases[0].height, canvases[1].height);
    const h1 = Math.max(canvases[2].height, canvases[3].height);

    const composite = document.createElement("canvas");
    composite.width = w0 + w1 + gap;
    composite.height = h0 + h1 + gap;
    const ctx = composite.getContext("2d");
    if (!ctx) return null;

    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, composite.width, composite.height);
    ctx.drawImage(canvases[0], 0, 0);
    ctx.drawImage(canvases[1], w0 + gap, 0);
    ctx.drawImage(canvases[2], 0, h0 + gap);
    ctx.drawImage(canvases[3], w0 + gap, h0 + gap);
    return composite;
  }, []);

  const handleExportPng = React.useCallback(() => {
    if (lockExport) return;
    setExportAnchor(null);
    const composite = createComposite();
    if (!composite) return;
    composite.toBlob((blob) => {
      if (blob) downloadBlob(blob, "bin_export.png");
    }, "image/png");
  }, [lockExport, createComposite]);

  const handleCopy = React.useCallback(async () => {
    if (lockExport) return;
    const composite = createComposite();
    if (!composite) return;
    composite.toBlob(async (blob) => {
      if (!blob) return;
      try {
        await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
      } catch {
        /* clipboard not available */
      }
    }, "image/png");
  }, [lockExport, createComposite]);

  // Keyboard
  const handleKeyDown = React.useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      const key = String(event.key || "").toLowerCase();
      if (isEditableElement(event.target)) return;

      if (key === "?") {
        event.preventDefault();
        setShowShortcutHelp((prev) => !prev);
        return;
      }
      if (key === "r") {
        event.preventDefault();
        setResetKey((prev) => prev + 1);
        return;
      }
      if (key === "l" && !lockDisplay) {
        event.preventDefault();
        setLogScale((prev) => !prev);
        return;
      }
      if (key === "c" && !lockDisplay) {
        event.preventDefault();
        const idx = colormapOptions.indexOf(cmap);
        const nextIdx = idx < 0 ? 0 : (idx + 1) % colormapOptions.length;
        setCmap(colormapOptions[nextIdx]);
        return;
      }
      if (key === "m" && !lockBinning) {
        event.preventDefault();
        setBinMode(binMode === "sum" ? "mean" : "sum");
        return;
      }
      if (key === "e" && !lockBinning) {
        event.preventDefault();
        const order = ["crop", "pad", "error"];
        const idx = order.indexOf(edgeMode);
        const nextIdx = idx < 0 ? 0 : (idx + 1) % order.length;
        setEdgeMode(order[nextIdx]);
        return;
      }
      if (key === "]" && !lockBinning) {
        event.preventDefault();
        setScanBinRow(stepOption(scanRowOptions, scanBinRow, +1));
        setScanBinCol(stepOption(scanColOptions, scanBinCol, +1));
        return;
      }
      if (key === "[" && !lockBinning) {
        event.preventDefault();
        setScanBinRow(stepOption(scanRowOptions, scanBinRow, -1));
        setScanBinCol(stepOption(scanColOptions, scanBinCol, -1));
        return;
      }
      if ((key === "=" || key === "+") && !lockBinning) {
        event.preventDefault();
        setDetBinRow(stepOption(detRowOptions, detBinRow, +1));
        setDetBinCol(stepOption(detColOptions, detBinCol, +1));
        return;
      }
      if (key === "-" && !lockBinning) {
        event.preventDefault();
        setDetBinRow(stepOption(detRowOptions, detBinRow, -1));
        setDetBinCol(stepOption(detColOptions, detBinCol, -1));
      }
    },
    [
      binMode, cmap, colormapOptions, detBinCol, detBinRow, detColOptions,
      detRowOptions, edgeMode, lockBinning, lockDisplay, scanBinCol, scanBinRow,
      scanColOptions, scanRowOptions, setBinMode, setCmap, setDetBinCol,
      setDetBinRow, setEdgeMode, setLogScale, setScanBinCol, setScanBinRow,
    ],
  );

  // Status color
  const statusColor =
    statusLevel === "error" ? "#ff6b6b" : statusLevel === "warn" ? "#ffb84d" : themeColors.textMuted;

  // Resize handle JSX
  const resizeHandle = (
    <Box
      onMouseDown={handleCanvasResizeStart}
      sx={{
        position: "absolute",
        bottom: 0,
        right: 0,
        width: 16,
        height: 16,
        cursor: lockView ? "default" : "nwse-resize",
        opacity: lockView ? 0.2 : 0.6,
        pointerEvents: lockView ? "none" : "auto",
        background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`,
        "&:hover": { opacity: lockView ? 0.2 : 1 },
      }}
    />
  );

  return (
    <Box
      className="bin-root"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      sx={{
        p: `${SPACING.LG}px`,
        bgcolor: themeColors.bg,
        color: themeColors.text,
        outline: "none",
      }}
    >
      {/* HEADER */}
      <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center" sx={{ mb: `${SPACING.SM}px` }}>
        <Typography sx={{ fontSize: 12, fontWeight: "bold", color: themeColors.accent, flex: 1 }}>
          {title || "Bin"}
          <ControlCustomizer
            widgetName="Bin"
            hiddenTools={hiddenTools}
            setHiddenTools={setHiddenTools}
            disabledTools={disabledTools}
            setDisabledTools={setDisabledTools}
            themeColors={themeColors}
          />
        </Typography>
        {!hideExport && (
          <>
            <Button size="small" variant="outlined" disabled={lockExport} onClick={(e) => { if (!lockExport) setExportAnchor(e.currentTarget); }} sx={{ ...compactButton, color: themeColors.accent }}>Export</Button>
            <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
              <MenuItem disabled={lockExport} onClick={handleExportPng} sx={{ fontSize: 12 }}>PNG (grid)</MenuItem>
            </Menu>
            <Button size="small" variant="outlined" disabled={lockExport} onClick={handleCopy} sx={{ ...compactButton, color: themeColors.accent }}>Copy</Button>
          </>
        )}
        <Button size="small" variant="outlined" onClick={() => setShowShortcutHelp((prev) => !prev)} sx={compactButton}>?</Button>
      </Stack>

      {/* SHORTCUT HELP */}
      {showShortcutHelp && (
        <Box sx={{ mb: `${SPACING.SM}px`, p: 0.65, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.bgAlt }}>
          <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
            ? help | L log/linear | C cycle colormap | M mean/sum | E edge mode |
            [ / ] scan bin down/up | - / + detector bin down/up | R / double-click reset zoom | scroll zoom
          </Typography>
        </Box>
      )}

      {/* 2×3 IMAGE GRID */}
      {!hidePreview && (
        <Box sx={{ position: "relative", width: "fit-content" }}>
          {/* Row 1: Original BF | Original ADF | Original DP */}
          <Stack direction="row" spacing={`${SPACING.SM}px`}>
            <InteractivePanel
              label="Original BF"
              rows={scanRows}
              cols={scanCols}
              bytes={originalBfBytes}
              cmap={cmap}
              logScale={logScale}
              canvasSize={canvasSize}
              borderColor={themeColors.border}
              textColor={themeColors.text}
              mutedColor={themeColors.textMuted}
              accentColor={themeColors.accent}
              lockView={lockView}
              stats={originalBfStats}
              hideStats={hideStats}
              pixelSize={pixelCalibrated ? pixelSizeCol : undefined}
              pixelUnit={pixelCalibrated ? pixelUnit : undefined}
              resetKey={resetKey}
              canvasRef={origBfRef}
            />
            <InteractivePanel
              label="Original ADF"
              rows={scanRows}
              cols={scanCols}
              bytes={originalAdfBytes}
              cmap={cmap}
              logScale={logScale}
              canvasSize={canvasSize}
              borderColor={themeColors.border}
              textColor={themeColors.text}
              mutedColor={themeColors.textMuted}
              accentColor={themeColors.accent}
              lockView={lockView}
              stats={originalAdfStats}
              hideStats={hideStats}
              pixelSize={pixelCalibrated ? pixelSizeCol : undefined}
              pixelUnit={pixelCalibrated ? pixelUnit : undefined}
              resetKey={resetKey}
              canvasRef={origAdfRef}
            />
            <InteractivePanel
              label="Original DP"
              rows={detRows}
              cols={detCols}
              bytes={originalMeanDpBytes}
              cmap={cmap}
              logScale={logScale}
              canvasSize={canvasSize}
              borderColor={themeColors.border}
              textColor={themeColors.text}
              mutedColor={themeColors.textMuted}
              accentColor={themeColors.accent}
              lockView={lockView}
              hideStats
              pixelSize={kCalibrated ? kPixelSizeCol : undefined}
              pixelUnit={kCalibrated ? kUnit : undefined}
              resetKey={resetKey}
              overlayRenderer={origDetOverlay}
            />
          </Stack>

          {/* Row 2: Binned BF | Binned ADF | Binned DP */}
          <Stack direction="row" spacing={`${SPACING.SM}px`} sx={{ mt: `${SPACING.SM}px` }}>
            <InteractivePanel
              label="Binned BF"
              rows={binnedScanRows}
              cols={binnedScanCols}
              bytes={binnedBfBytes}
              cmap={cmap}
              logScale={logScale}
              canvasSize={canvasSize}
              borderColor={themeColors.border}
              textColor={themeColors.text}
              mutedColor={themeColors.textMuted}
              accentColor={themeColors.accent}
              lockView={lockView}
              stats={binnedBfStats}
              hideStats={hideStats}
              pixelSize={pixelCalibrated ? binnedPixelSizeCol : undefined}
              pixelUnit={pixelCalibrated ? pixelUnit : undefined}
              resetKey={resetKey}
              canvasRef={binnedBfRef}
            />
            <InteractivePanel
              label="Binned ADF"
              rows={binnedScanRows}
              cols={binnedScanCols}
              bytes={binnedAdfBytes}
              cmap={cmap}
              logScale={logScale}
              canvasSize={canvasSize}
              borderColor={themeColors.border}
              textColor={themeColors.text}
              mutedColor={themeColors.textMuted}
              accentColor={themeColors.accent}
              lockView={lockView}
              stats={binnedAdfStats}
              hideStats={hideStats}
              pixelSize={pixelCalibrated ? binnedPixelSizeCol : undefined}
              pixelUnit={pixelCalibrated ? pixelUnit : undefined}
              resetKey={resetKey}
              canvasRef={binnedAdfRef}
            />
            <InteractivePanel
              label="Binned DP"
              rows={binnedDetRows}
              cols={binnedDetCols}
              bytes={binnedMeanDpBytes}
              cmap={cmap}
              logScale={logScale}
              canvasSize={canvasSize}
              borderColor={themeColors.border}
              textColor={themeColors.text}
              mutedColor={themeColors.textMuted}
              accentColor={themeColors.accent}
              lockView={lockView}
              hideStats
              pixelSize={kCalibrated ? binnedKPixelSizeCol : undefined}
              pixelUnit={kCalibrated ? kUnit : undefined}
              resetKey={resetKey}
              overlayRenderer={binnedDetOverlay}
            />
          </Stack>

          {/* Resize handle (bottom-right of grid) */}
          {resizeHandle}
        </Box>
      )}

      {/* CONTROLS (below images) */}
      {showControls && (
        <Box sx={{ mt: `${SPACING.XS}px`, display: "flex", flexDirection: "column", gap: "2px" }}>
          {/* Row 1: Scan + Det bins + Reduce + Edge */}
          {!hideBinning && (
            <Box sx={{ ...controlRow, gap: `${SPACING.SM}px`, width: "auto" }}>
              <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted }}>Scan:</Typography>
              <Select size="small" value={scanBinRow} onChange={(e) => setScanBinRow(Number(e.target.value))} sx={{ ...themedSelect, minWidth: 44 }} MenuProps={themedMenuProps} disabled={lockBinning}>
                {scanRowOptions.map((v) => <MenuItem key={`sr-${v}`} value={v}>{v}</MenuItem>)}
              </Select>
              <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted }}>×</Typography>
              <Select size="small" value={scanBinCol} onChange={(e) => setScanBinCol(Number(e.target.value))} sx={{ ...themedSelect, minWidth: 44 }} MenuProps={themedMenuProps} disabled={lockBinning}>
                {scanColOptions.map((v) => <MenuItem key={`sc-${v}`} value={v}>{v}</MenuItem>)}
              </Select>
              <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted }}>Det:</Typography>
              <Select size="small" value={detBinRow} onChange={(e) => setDetBinRow(Number(e.target.value))} sx={{ ...themedSelect, minWidth: 44 }} MenuProps={themedMenuProps} disabled={lockBinning}>
                {detRowOptions.map((v) => <MenuItem key={`dr-${v}`} value={v}>{v}</MenuItem>)}
              </Select>
              <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted }}>×</Typography>
              <Select size="small" value={detBinCol} onChange={(e) => setDetBinCol(Number(e.target.value))} sx={{ ...themedSelect, minWidth: 44 }} MenuProps={themedMenuProps} disabled={lockBinning}>
                {detColOptions.map((v) => <MenuItem key={`dc-${v}`} value={v}>{v}</MenuItem>)}
              </Select>
              <Select size="small" value={binMode} onChange={(e) => setBinMode(String(e.target.value))} sx={{ ...themedSelect, minWidth: 52 }} MenuProps={themedMenuProps} disabled={lockBinning}>
                <MenuItem value="mean">mean</MenuItem>
                <MenuItem value="sum">sum</MenuItem>
              </Select>
              <Select size="small" value={edgeMode} onChange={(e) => setEdgeMode(String(e.target.value))} sx={{ ...themedSelect, minWidth: 52 }} MenuProps={themedMenuProps} disabled={lockBinning}>
                <MenuItem value="crop">crop</MenuItem>
                <MenuItem value="pad">pad</MenuItem>
                <MenuItem value="error">error</MenuItem>
              </Select>
              <InfoTooltip theme={themeInfo.theme} text={
                <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
                  <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Binning Controls</Typography>
                  <Typography sx={{ fontSize: 10 }}><b>Scan:</b> bin factor for scan-space rows × cols</Typography>
                  <Typography sx={{ fontSize: 10 }}><b>Det:</b> bin factor for detector-space rows × cols</Typography>
                  <Typography sx={{ fontSize: 10, mt: 0.5, fontWeight: "bold" }}>Reduce mode</Typography>
                  <Typography sx={{ fontSize: 10 }}><b>mean:</b> average pixel values within each bin</Typography>
                  <Typography sx={{ fontSize: 10 }}><b>sum:</b> sum pixel values within each bin</Typography>
                  <Typography sx={{ fontSize: 10, mt: 0.5, fontWeight: "bold" }}>Edge mode</Typography>
                  <Typography sx={{ fontSize: 10 }}><b>crop:</b> discard remainder pixels at edges</Typography>
                  <Typography sx={{ fontSize: 10 }}><b>pad:</b> zero-pad edges to fill the last bin</Typography>
                  <Typography sx={{ fontSize: 10 }}><b>error:</b> raise error if shape not divisible</Typography>
                </Box>
              } />
            </Box>
          )}

          {/* Row 2: BF / ADF mask sliders */}
          {!hideMask && (
            <Box sx={{ ...controlRow, gap: `${SPACING.SM}px`, width: "auto" }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: "3px", minWidth: 140 }}>
                <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted, whiteSpace: "nowrap" }}>BF:{bfRadiusRatio.toFixed(3)}</Typography>
                <Slider value={bfRadiusRatio} min={0.02} max={0.5} step={0.005} onChange={(_, v) => setBfRadiusRatio(v as number)} size="small" sx={{ ...sliderStyles.small, minWidth: 60 }} disabled={lockMask} />
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: "3px", minWidth: 160 }}>
                <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted, whiteSpace: "nowrap" }}>ADF in:{adfInnerRatio.toFixed(3)}</Typography>
                <Slider value={adfInnerRatio} min={0.02} max={0.8} step={0.005} onChange={(_, v) => setAdfInnerRatio(v as number)} size="small" sx={{ ...sliderStyles.small, minWidth: 60 }} disabled={lockMask} />
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: "3px", minWidth: 160 }}>
                <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted, whiteSpace: "nowrap" }}>ADF out:{adfOuterRatio.toFixed(3)}</Typography>
                <Slider value={adfOuterRatio} min={0.05} max={0.95} step={0.005} onChange={(_, v) => setAdfOuterRatio(v as number)} size="small" sx={{ ...sliderStyles.small, minWidth: 60 }} disabled={lockMask} />
              </Box>
            </Box>
          )}

          {/* Row 2: Color + Log + Histogram */}
          {!hideDisplay && (
            <Box sx={{ ...controlRow, gap: `${SPACING.SM}px`, width: "auto" }}>
              <Select size="small" value={cmap} onChange={(e) => setCmap(String(e.target.value))} sx={{ ...themedSelect, minWidth: 60 }} MenuProps={themedMenuProps} disabled={lockDisplay}>
                {colormapOptions.map((c) => <MenuItem key={c} value={c}>{c}</MenuItem>)}
              </Select>
              <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted }}>Log:</Typography>
              <Switch checked={logScale} onChange={(_, v) => setLogScale(v)} size="small" sx={switchStyles.small} disabled={lockDisplay} />
              <canvas ref={histogramRef} style={{ width: 100, height: 32, border: `1px solid ${themeColors.border}` }} />
            </Box>
          )}

          {/* Shape + Calibration + Status — table */}
          <Box sx={{ display: "flex", flexDirection: "column", gap: "1px", mt: "2px" }}>
            <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
              shape: ({scanRows}, {scanCols}, {detRows}, {detCols}) → ({binnedScanRows}, {binnedScanCols}, {binnedDetRows}, {binnedDetCols})
            </Typography>
            <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
              scan:{"  "}({formatNumber(pixelSizeRow, 4)}, {formatNumber(pixelSizeCol, 4)}) → ({formatNumber(binnedPixelSizeRow, 4)}, {formatNumber(binnedPixelSizeCol, 4)}) {pixelUnit}/px
            </Typography>
            <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
              det:{"   "}({formatNumber(kPixelSizeRow, 4)}, {formatNumber(kPixelSizeCol, 4)}) → ({formatNumber(binnedKPixelSizeRow, 4)}, {formatNumber(binnedKPixelSizeCol, 4)}) {kUnit}/px
            </Typography>
            {statusMessage && (
              <Typography sx={{ ...typography.value, color: statusColor }}>
                {statusMessage}
              </Typography>
            )}
          </Box>
        </Box>
      )}
    </Box>
  );
}

export const render = createRender(BinWidget);
