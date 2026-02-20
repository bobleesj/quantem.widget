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
import { useTheme } from "../theme";
import { extractFloat32, formatNumber, downloadBlob } from "../format";
import { applyLogScale, findDataRange } from "../stats";
import { COLORMAPS, renderToOffscreen } from "../colormaps";
import { computeToolVisibility } from "../tool-parity";
import { ControlCustomizer } from "../control-customizer";
import "./bin.css";

const PANEL_MAX = 280;

const typography = {
  label: { fontSize: 11 },
  value: { fontSize: 11, fontFamily: "monospace" },
  section: { fontSize: 11, fontWeight: "bold" as const },
};

const sliderStyles = {
  small: {
    "& .MuiSlider-thumb": { width: 12, height: 12 },
    "& .MuiSlider-rail": { height: 3 },
    "& .MuiSlider-track": { height: 3 },
  },
};

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

function sizeForPanel(rows: number, cols: number): { width: number; height: number } {
  const h = Math.max(1, rows);
  const w = Math.max(1, cols);
  const scale = Math.min(PANEL_MAX / w, PANEL_MAX / h);
  return {
    width: Math.max(120, Math.round(w * scale)),
    height: Math.max(120, Math.round(h * scale)),
  };
}

function statsLine(stats: number[]): string {
  if (!stats || stats.length < 6) return "";
  const [mean, min, max, std, snr, contrast] = stats;
  return `mean ${formatNumber(mean)} | min ${formatNumber(min)} | max ${formatNumber(max)} | std ${formatNumber(std)} | snr ${formatNumber(snr)} | c ${formatNumber(contrast)}`;
}

interface PreviewPanelProps {
  title: string;
  rows: number;
  cols: number;
  bytes: DataView;
  stats: number[];
  cmap: string;
  logScale: boolean;
  borderColor: string;
  textColor: string;
  mutedColor: string;
  hideStats?: boolean;
  externalRef?: React.RefObject<HTMLCanvasElement | null>;
}

function PreviewPanel({
  title,
  rows,
  cols,
  bytes,
  stats,
  cmap,
  logScale,
  borderColor,
  textColor,
  mutedColor,
  hideStats,
  externalRef,
}: PreviewPanelProps) {
  const internalRef = React.useRef<HTMLCanvasElement>(null);
  const canvasRef = externalRef || internalRef;
  const parsed = React.useMemo(() => extractFloat32(bytes), [bytes]);
  const panelSize = React.useMemo(() => sizeForPanel(rows, cols), [rows, cols]);

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

    canvas.width = panelSize.width;
    canvas.height = panelSize.height;
    ctx.clearRect(0, 0, panelSize.width, panelSize.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreen, 0, 0, panelSize.width, panelSize.height);
  }, [parsed, rows, cols, cmap, logScale, panelSize]);

  return (
    <Box sx={{ minWidth: panelSize.width }}>
      <Typography sx={{ ...typography.section, color: textColor, mb: 0.5 }}>{title}</Typography>
      <Box sx={{ border: `1px solid ${borderColor}`, bgcolor: "#000", width: panelSize.width, height: panelSize.height }}>
        <canvas ref={canvasRef} style={{ width: panelSize.width, height: panelSize.height, display: "block" }} />
      </Box>
      {!hideStats && <Typography sx={{ ...typography.value, color: mutedColor, mt: 0.5, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
        {statsLine(stats)}
      </Typography>}
    </Box>
  );
}

const DPR = window.devicePixelRatio || 1;

interface DetectorPanelProps {
  title: string;
  rows: number;
  cols: number;
  bytes: DataView;
  centerRow: number;
  centerCol: number;
  bfRadiusRatio: number;
  adfInnerRatio: number;
  adfOuterRatio: number;
  cmap: string;
  logScale: boolean;
  borderColor: string;
  textColor: string;
  mutedColor: string;
}

function DetectorPanel({
  title,
  rows,
  cols,
  bytes,
  centerRow,
  centerCol,
  bfRadiusRatio,
  adfInnerRatio,
  adfOuterRatio,
  cmap,
  logScale,
  borderColor,
  textColor,
  mutedColor,
}: DetectorPanelProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const overlayRef = React.useRef<HTMLCanvasElement>(null);
  const parsed = React.useMemo(() => extractFloat32(bytes), [bytes]);
  const panelSize = React.useMemo(() => sizeForPanel(rows, cols), [rows, cols]);

  // Render colormapped mean DP
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

    canvas.width = panelSize.width;
    canvas.height = panelSize.height;
    ctx.clearRect(0, 0, panelSize.width, panelSize.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreen, 0, 0, panelSize.width, panelSize.height);
  }, [parsed, rows, cols, cmap, logScale, panelSize]);

  // Render BF/ADF mask overlay
  React.useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay || rows <= 0 || cols <= 0) return;

    const cssW = panelSize.width;
    const cssH = panelSize.height;
    overlay.width = cssW * DPR;
    overlay.height = cssH * DPR;
    overlay.style.width = `${cssW}px`;
    overlay.style.height = `${cssH}px`;

    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.save();
    ctx.scale(DPR, DPR);

    const scaleX = cssW / cols;
    const scaleY = cssH / rows;
    const cx = centerCol * scaleX;
    const cy = centerRow * scaleY;
    const detSize = Math.min(rows, cols);

    // BF disk
    const bfRadiusPx = bfRadiusRatio * detSize;
    const bfRx = bfRadiusPx * scaleX;
    const bfRy = bfRadiusPx * scaleY;

    ctx.lineWidth = 2;
    ctx.shadowColor = "rgba(0,0,0,0.5)";
    ctx.shadowBlur = 2;

    ctx.beginPath();
    ctx.ellipse(cx, cy, bfRx, bfRy, 0, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(0, 255, 0, 0.12)";
    ctx.fill();
    ctx.strokeStyle = "rgba(0, 255, 0, 0.9)";
    ctx.stroke();

    // BF label
    ctx.shadowBlur = 0;
    ctx.font = "bold 10px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = "rgba(0, 255, 0, 0.9)";
    ctx.textAlign = "center";
    ctx.fillText("BF", cx, cy - bfRy - 4);

    // ADF annulus
    const adfInnerPx = adfInnerRatio * detSize;
    const adfOuterPx = adfOuterRatio * detSize;
    const adfInnerRx = adfInnerPx * scaleX;
    const adfInnerRy = adfInnerPx * scaleY;
    const adfOuterRx = adfOuterPx * scaleX;
    const adfOuterRy = adfOuterPx * scaleY;

    ctx.shadowBlur = 2;
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

    // ADF label
    ctx.shadowBlur = 0;
    ctx.fillStyle = "rgba(0, 220, 255, 0.9)";
    ctx.fillText("ADF", cx, cy - adfOuterRy - 4);

    ctx.restore();
  }, [rows, cols, panelSize, centerRow, centerCol, bfRadiusRatio, adfInnerRatio, adfOuterRatio]);

  return (
    <Box sx={{ minWidth: panelSize.width }}>
      <Typography sx={{ ...typography.section, color: textColor, mb: 0.5 }}>{title}</Typography>
      <Box sx={{ border: `1px solid ${borderColor}`, bgcolor: "#000", width: panelSize.width, height: panelSize.height, position: "relative" }}>
        <canvas ref={canvasRef} style={{ width: panelSize.width, height: panelSize.height, display: "block" }} />
        <canvas ref={overlayRef} style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }} />
      </Box>
      <Typography sx={{ ...typography.value, color: mutedColor, mt: 0.5, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
        {rows}×{cols} px | center ({formatNumber(centerRow, 1)}, {formatNumber(centerCol, 1)})
      </Typography>
    </Box>
  );
}

function BinWidget() {
  const { colors } = useTheme();

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

  const [scanBinRow, setScanBinRow] = useModelState<number>("scan_bin_row");
  const [scanBinCol, setScanBinCol] = useModelState<number>("scan_bin_col");
  const [detBinRow, setDetBinRow] = useModelState<number>("det_bin_row");
  const [detBinCol, setDetBinCol] = useModelState<number>("det_bin_col");

  const [binMode, setBinMode] = useModelState<string>("bin_mode");
  const [edgeMode, setEdgeMode] = useModelState<string>("edge_mode");

  const [bfRadiusRatio, setBfRadiusRatio] = useModelState<number>("bf_radius_ratio");
  const [adfInnerRatio, setAdfInnerRatio] = useModelState<number>("adf_inner_ratio");
  const [adfOuterRatio, setAdfOuterRatio] = useModelState<number>("adf_outer_ratio");

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
  const [computeBackend] = useModelState<string>("compute_backend");
  const [device] = useModelState<string>("device");
  const [torchEnabled] = useModelState<boolean>("torch_enabled");

  const [originalBfBytes] = useModelState<DataView>("original_bf_bytes");
  const [originalAdfBytes] = useModelState<DataView>("original_adf_bytes");
  const [binnedBfBytes] = useModelState<DataView>("binned_bf_bytes");
  const [binnedAdfBytes] = useModelState<DataView>("binned_adf_bytes");

  const [originalMeanDpBytes] = useModelState<DataView>("original_mean_dp_bytes");
  const [binnedMeanDpBytes] = useModelState<DataView>("binned_mean_dp_bytes");
  const [centerRow] = useModelState<number>("center_row");
  const [centerCol] = useModelState<number>("center_col");
  const [binnedCenterRow] = useModelState<number>("binned_center_row");
  const [binnedCenterCol] = useModelState<number>("binned_center_col");

  const [originalBfStats] = useModelState<number[]>("original_bf_stats");
  const [originalAdfStats] = useModelState<number[]>("original_adf_stats");
  const [binnedBfStats] = useModelState<number[]>("binned_bf_stats");
  const [binnedAdfStats] = useModelState<number[]>("binned_adf_stats");

  const [statusMessage] = useModelState<string>("status_message");
  const [statusLevel] = useModelState<string>("status_level");

  const scanRowOptions = React.useMemo(() => optionsForFactor(maxScanBinRow), [maxScanBinRow]);
  const scanColOptions = React.useMemo(() => optionsForFactor(maxScanBinCol), [maxScanBinCol]);
  const detRowOptions = React.useMemo(() => optionsForFactor(maxDetBinRow), [maxDetBinRow]);
  const detColOptions = React.useMemo(() => optionsForFactor(maxDetBinCol), [maxDetBinCol]);

  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [showControls] = useModelState<boolean>("show_controls");
  const [disabledTools, setDisabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools, setHiddenTools] = useModelState<string[]>("hidden_tools");
  const [showShortcutHelp, setShowShortcutHelp] = React.useState<boolean>(false);
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);

  const origBfRef = React.useRef<HTMLCanvasElement | null>(null);
  const origAdfRef = React.useRef<HTMLCanvasElement | null>(null);
  const binnedBfRef = React.useRef<HTMLCanvasElement | null>(null);
  const binnedAdfRef = React.useRef<HTMLCanvasElement | null>(null);

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

  const colormapOptions = React.useMemo(() => ["inferno", "viridis", "gray", "magma"], []);

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

  const handleKeyDown = React.useCallback((event: React.KeyboardEvent<HTMLDivElement>) => {
    const key = String(event.key || "").toLowerCase();
    if (isEditableElement(event.target)) return;

    if (key === "?") {
      event.preventDefault();
      setShowShortcutHelp((prev) => !prev);
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
  }, [
    binMode,
    cmap,
    colormapOptions,
    detBinCol,
    detBinRow,
    detColOptions,
    detRowOptions,
    edgeMode,
    lockBinning,
    lockDisplay,
    scanBinCol,
    scanBinRow,
    scanColOptions,
    scanRowOptions,
    setBinMode,
    setDetBinCol,
    setDetBinRow,
    setEdgeMode,
    setScanBinCol,
    setScanBinRow,
  ]);

  const statusColor =
    statusLevel === "error" ? "#ff6b6b" : statusLevel === "warn" ? "#ffb84d" : colors.textMuted;

  const controlSelect = {
    fontSize: 11,
    minWidth: 80,
    bgcolor: colors.controlBg,
    color: colors.text,
    "& .MuiSelect-select": { py: 0.5 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: colors.border },
  };

  const themedMenuProps = {
    PaperProps: {
      sx: {
        bgcolor: colors.controlBg,
        color: colors.text,
        border: `1px solid ${colors.border}`,
      },
    },
  };

  return (
    <Box className="bin-root" tabIndex={0} onKeyDown={handleKeyDown} sx={{ p: 2, bgcolor: colors.bg, color: colors.text, outline: "none" }}>
      <Stack direction="row" spacing={0.8} alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
        <Typography sx={{ fontSize: 12, fontWeight: "bold", color: colors.accent }}>
        {title || "Bin: Calibration-Aware 4D-STEM Binning + BF/ADF QC"}
        <ControlCustomizer
            widgetName="Bin"
            hiddenTools={hiddenTools}
            setHiddenTools={setHiddenTools}
            disabledTools={disabledTools}
            setDisabledTools={setDisabledTools}
            themeColors={colors}
          />
        </Typography>
        {!hideExport && <>
          <Button size="small" variant="outlined" disabled={lockExport} onClick={(e) => { if (!lockExport) setExportAnchor(e.currentTarget); }} sx={{ fontSize: 10, minWidth: 60 }}>Export</Button>
          <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
            <MenuItem disabled={lockExport} onClick={handleExportPng} sx={{ fontSize: 12 }}>PNG (grid)</MenuItem>
          </Menu>
          <Button size="small" variant="outlined" disabled={lockExport} onClick={handleCopy} sx={{ fontSize: 10, minWidth: 50 }}>Copy</Button>
        </>}
        <Button size="small" variant="outlined" onClick={() => setShowShortcutHelp((prev) => !prev)} sx={{ fontSize: 10, minWidth: 86 }}>
          Shortcuts ?
        </Button>
      </Stack>
      {showShortcutHelp && (
        <Box sx={{ mb: 1, p: 0.65, border: `1px solid ${colors.border}`, bgcolor: colors.bgAlt }}>
          <Typography sx={{ ...typography.value, color: colors.textMuted }}>
            `?` help · `L` log/linear · `C` cycle colormap · `M` mean/sum · `E` edge mode ·
            `[`/`]` scan bin down/up · `-`/`+` detector bin down/up
          </Typography>
        </Box>
      )}

      {showControls && <><Stack direction="row" spacing={2} sx={{ flexWrap: "wrap", mb: 1.5 }}>
        {!hideBinning && <Box sx={{ border: `1px solid ${colors.border}`, bgcolor: colors.controlBg, p: 1, minWidth: 320, opacity: lockBinning ? 0.6 : 1 }}>
          <Typography sx={{ ...typography.section, color: colors.text, mb: 0.75 }}>Binning Controls</Typography>
          <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", rowGap: 1 }}>
            <Box>
              <Typography sx={{ ...typography.label, color: colors.textMuted }}>Scan row</Typography>
              <Select size="small" value={scanBinRow} onChange={(e) => setScanBinRow(Number(e.target.value))} sx={controlSelect} MenuProps={themedMenuProps} disabled={lockBinning}>
                {scanRowOptions.map((v) => <MenuItem key={`srow-${v}`} value={v}>{v}</MenuItem>)}
              </Select>
            </Box>
            <Box>
              <Typography sx={{ ...typography.label, color: colors.textMuted }}>Scan col</Typography>
              <Select size="small" value={scanBinCol} onChange={(e) => setScanBinCol(Number(e.target.value))} sx={controlSelect} MenuProps={themedMenuProps} disabled={lockBinning}>
                {scanColOptions.map((v) => <MenuItem key={`scol-${v}`} value={v}>{v}</MenuItem>)}
              </Select>
            </Box>
            <Box>
              <Typography sx={{ ...typography.label, color: colors.textMuted }}>Det row</Typography>
              <Select size="small" value={detBinRow} onChange={(e) => setDetBinRow(Number(e.target.value))} sx={controlSelect} MenuProps={themedMenuProps} disabled={lockBinning}>
                {detRowOptions.map((v) => <MenuItem key={`drow-${v}`} value={v}>{v}</MenuItem>)}
              </Select>
            </Box>
            <Box>
              <Typography sx={{ ...typography.label, color: colors.textMuted }}>Det col</Typography>
              <Select size="small" value={detBinCol} onChange={(e) => setDetBinCol(Number(e.target.value))} sx={controlSelect} MenuProps={themedMenuProps} disabled={lockBinning}>
                {detColOptions.map((v) => <MenuItem key={`dcol-${v}`} value={v}>{v}</MenuItem>)}
              </Select>
            </Box>
            <Box>
              <Typography sx={{ ...typography.label, color: colors.textMuted }}>Reduce</Typography>
              <Select size="small" value={binMode} onChange={(e) => setBinMode(String(e.target.value))} sx={controlSelect} MenuProps={themedMenuProps} disabled={lockBinning}>
                <MenuItem value="mean">mean</MenuItem>
                <MenuItem value="sum">sum</MenuItem>
              </Select>
            </Box>
            <Box>
              <Typography sx={{ ...typography.label, color: colors.textMuted }}>Edges</Typography>
              <Select size="small" value={edgeMode} onChange={(e) => setEdgeMode(String(e.target.value))} sx={controlSelect} MenuProps={themedMenuProps} disabled={lockBinning}>
                <MenuItem value="crop">crop</MenuItem>
                <MenuItem value="pad">pad</MenuItem>
                <MenuItem value="error">error</MenuItem>
              </Select>
            </Box>
            {!hideDisplay && <>
            <Box>
              <Typography sx={{ ...typography.label, color: colors.textMuted }}>Colormap</Typography>
              <Select size="small" value={cmap} onChange={(e) => setCmap(String(e.target.value))} sx={controlSelect} MenuProps={themedMenuProps} disabled={lockDisplay}>
                <MenuItem value="inferno">inferno</MenuItem>
                <MenuItem value="viridis">viridis</MenuItem>
                <MenuItem value="gray">gray</MenuItem>
                <MenuItem value="magma">magma</MenuItem>
              </Select>
            </Box>
            <Box>
              <Typography sx={{ ...typography.label, color: colors.textMuted }}>Display</Typography>
              <Select size="small" value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(String(e.target.value) === "log")} sx={controlSelect} MenuProps={themedMenuProps} disabled={lockDisplay}>
                <MenuItem value="linear">linear</MenuItem>
                <MenuItem value="log">log</MenuItem>
              </Select>
            </Box>
            </>}
          </Stack>
        </Box>}

        {!hideMask && <Box sx={{ border: `1px solid ${colors.border}`, bgcolor: colors.controlBg, p: 1, minWidth: 320, flex: 1, opacity: lockMask ? 0.6 : 1 }}>
          <Typography sx={{ ...typography.section, color: colors.text, mb: 0.75 }}>Mask Presets (for BF/ADF QC)</Typography>
          <Stack spacing={0.5}>
            <Typography sx={{ ...typography.label, color: colors.textMuted }}>BF radius ratio: {bfRadiusRatio.toFixed(3)}</Typography>
            <Slider
              value={bfRadiusRatio}
              min={0.02}
              max={0.5}
              step={0.005}
              onChange={(_, v) => setBfRadiusRatio(v as number)}
              size="small"
              sx={sliderStyles.small}
              disabled={lockMask}
            />
            <Typography sx={{ ...typography.label, color: colors.textMuted }}>ADF inner ratio: {adfInnerRatio.toFixed(3)}</Typography>
            <Slider
              value={adfInnerRatio}
              min={0.02}
              max={0.8}
              step={0.005}
              onChange={(_, v) => setAdfInnerRatio(v as number)}
              size="small"
              sx={sliderStyles.small}
              disabled={lockMask}
            />
            <Typography sx={{ ...typography.label, color: colors.textMuted }}>ADF outer ratio: {adfOuterRatio.toFixed(3)}</Typography>
            <Slider
              value={adfOuterRatio}
              min={0.05}
              max={0.95}
              step={0.005}
              onChange={(_, v) => setAdfOuterRatio(v as number)}
              size="small"
              sx={sliderStyles.small}
              disabled={lockMask}
            />
          </Stack>
        </Box>}
      </Stack>

      <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap", mb: 1.5 }}>
        <Box sx={{ border: `1px solid ${colors.border}`, bgcolor: colors.bgAlt, p: 1, minWidth: 320 }}>
          <Typography sx={{ ...typography.section, color: colors.text, mb: 0.5 }}>Shape + Calibration</Typography>
          <Typography sx={{ ...typography.value, color: colors.textMuted }}>
            shape: ({scanRows}, {scanCols}, {detRows}, {detCols}) → ({binnedScanRows}, {binnedScanCols}, {binnedDetRows}, {binnedDetCols})
          </Typography>
          <Typography sx={{ ...typography.value, color: colors.textMuted }}>
            real-space: ({formatNumber(pixelSizeRow, 4)}, {formatNumber(pixelSizeCol, 4)}) {pixelUnit}/px
            → ({formatNumber(binnedPixelSizeRow, 4)}, {formatNumber(binnedPixelSizeCol, 4)})
          </Typography>
          <Typography sx={{ ...typography.value, color: colors.textMuted }}>
            detector: ({formatNumber(kPixelSizeRow, 4)}, {formatNumber(kPixelSizeCol, 4)}) {kUnit}/px
            → ({formatNumber(binnedKPixelSizeRow, 4)}, {formatNumber(binnedKPixelSizeCol, 4)})
          </Typography>
          <Typography sx={{ ...typography.value, color: colors.textMuted }}>
            calibrated: real={pixelCalibrated ? "yes" : "no"}, detector={kCalibrated ? "yes" : "no"}
          </Typography>
          <Typography sx={{ ...typography.value, color: colors.textMuted }}>
            compute: {computeBackend} on {device} (torch={torchEnabled ? "enabled" : "off"})
          </Typography>
        </Box>

        <Box sx={{ border: `1px solid ${colors.border}`, bgcolor: colors.bgAlt, p: 1, minWidth: 320, flex: 1 }}>
          <Typography sx={{ ...typography.section, color: colors.text, mb: 0.5 }}>Status</Typography>
          <Typography sx={{ ...typography.value, color: statusColor }}>{statusMessage || "ready"}</Typography>
        </Box>
      </Stack></>}

      {!hidePreview && <>
        <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap", rowGap: 2 }}>
          <PreviewPanel
            title="Original BF"
            rows={scanRows}
            cols={scanCols}
            bytes={originalBfBytes}
            stats={originalBfStats}
            cmap={cmap}
            logScale={logScale}
            borderColor={colors.border}
            textColor={colors.text}
            mutedColor={colors.textMuted}
            hideStats={hideStats}
            externalRef={origBfRef}
          />
          <PreviewPanel
            title="Original ADF"
            rows={scanRows}
            cols={scanCols}
            bytes={originalAdfBytes}
            stats={originalAdfStats}
            cmap={cmap}
            logScale={logScale}
            borderColor={colors.border}
            textColor={colors.text}
            mutedColor={colors.textMuted}
            hideStats={hideStats}
            externalRef={origAdfRef}
          />
          <PreviewPanel
            title="Binned BF"
            rows={binnedScanRows}
            cols={binnedScanCols}
            bytes={binnedBfBytes}
            stats={binnedBfStats}
            cmap={cmap}
            logScale={logScale}
            borderColor={colors.border}
            textColor={colors.text}
            mutedColor={colors.textMuted}
            hideStats={hideStats}
            externalRef={binnedBfRef}
          />
          <PreviewPanel
            title="Binned ADF"
            rows={binnedScanRows}
            cols={binnedScanCols}
            bytes={binnedAdfBytes}
            stats={binnedAdfStats}
            cmap={cmap}
            logScale={logScale}
            borderColor={colors.border}
            textColor={colors.text}
            mutedColor={colors.textMuted}
            hideStats={hideStats}
            externalRef={binnedAdfRef}
          />
        </Stack>
        <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap", rowGap: 2, mt: 2 }}>
          <DetectorPanel
            title="Original Detector"
            rows={detRows}
            cols={detCols}
            bytes={originalMeanDpBytes}
            centerRow={centerRow}
            centerCol={centerCol}
            bfRadiusRatio={bfRadiusRatio}
            adfInnerRatio={adfInnerRatio}
            adfOuterRatio={adfOuterRatio}
            cmap={cmap}
            logScale={logScale}
            borderColor={colors.border}
            textColor={colors.text}
            mutedColor={colors.textMuted}
          />
          <DetectorPanel
            title="Binned Detector"
            rows={binnedDetRows}
            cols={binnedDetCols}
            bytes={binnedMeanDpBytes}
            centerRow={binnedCenterRow}
            centerCol={binnedCenterCol}
            bfRadiusRatio={bfRadiusRatio}
            adfInnerRatio={adfInnerRatio}
            adfOuterRatio={adfOuterRatio}
            cmap={cmap}
            logScale={logScale}
            borderColor={colors.border}
            textColor={colors.text}
            mutedColor={colors.textMuted}
          />
        </Stack>
      </>}
    </Box>
  );
}

export const render = createRender(BinWidget);
