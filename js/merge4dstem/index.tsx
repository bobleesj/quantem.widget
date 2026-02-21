import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import Switch from "@mui/material/Switch";
import { useTheme } from "../theme";
import { extractFloat32, formatNumber, downloadBlob } from "../format";
import { applyLogScale, findDataRange, computeStats } from "../stats";
import { COLORMAPS, renderToOffscreen } from "../colormaps";
import { computeToolVisibility } from "../tool-parity";
import { ControlCustomizer } from "../control-customizer";
import "./merge4dstem.css";

// ============================================================================
// Layout Constants — matches Show4DSTEM spacing
// ============================================================================
const SPACING = {
  XS: 4,
  SM: 8,
  MD: 12,
  LG: 16,
};

const PREVIEW_SIZE = 280;

// ============================================================================
// UI Styles — matches Show4DSTEM typography + controls
// ============================================================================
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const },
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

const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
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

type SourceInfo = {
  name: string;
  shape: number[];
  valid: boolean;
  message: string;
};

function isEditableElement(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName.toLowerCase();
  if (target.isContentEditable) return true;
  if (target.getAttribute("role") === "textbox") return true;
  return tag === "input" || tag === "textarea" || tag === "select";
}

function formatStat(value: number): string {
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs < 0.001 || abs >= 10000) return value.toExponential(2);
  if (abs < 0.01) return value.toFixed(4);
  if (abs < 1) return value.toFixed(3);
  return value.toFixed(2);
}

function Merge4DSTEMWidget() {
  const { colors } = useTheme();

  const [title] = useModelState<string>("title");
  const [nSources] = useModelState<number>("n_sources");
  const [scanRows] = useModelState<number>("scan_rows");
  const [scanCols] = useModelState<number>("scan_cols");
  const [detRows] = useModelState<number>("det_rows");
  const [detCols] = useModelState<number>("det_cols");

  const [pixelSize] = useModelState<number>("pixel_size");
  const [pixelUnit] = useModelState<string>("pixel_unit");
  const [pixelCalibrated] = useModelState<boolean>("pixel_calibrated");
  const [kPixelSize] = useModelState<number>("k_pixel_size");
  const [kUnit] = useModelState<string>("k_unit");
  const [kCalibrated] = useModelState<boolean>("k_calibrated");

  const [sourceInfoJson] = useModelState<string>("source_info_json");
  const [previewBytes] = useModelState<DataView>("preview_bytes");
  const [previewRows] = useModelState<number>("preview_rows");
  const [previewCols] = useModelState<number>("preview_cols");
  const [previewIndex, setPreviewIndex] = useModelState<number>("preview_index");

  const [merged] = useModelState<boolean>("merged");
  const [outputShapeJson] = useModelState<string>("output_shape_json");
  const [frameDimLabel] = useModelState<string>("frame_dim_label");
  const [binFactor, setBinFactor] = useModelState<number>("bin_factor");

  const [statusMessage] = useModelState<string>("status_message");
  const [statusLevel] = useModelState<string>("status_level");

  const [, setMergeRequested] = useModelState<boolean>("_merge_requested");

  const [cmap, setCmap] = useModelState<string>("cmap");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [showControls] = useModelState<boolean>("show_controls");
  const [showStats] = useModelState<boolean>("show_stats");
  const [device] = useModelState<string>("device");

  const [disabledTools, setDisabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools, setHiddenTools] = useModelState<string[]>("hidden_tools");

  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  const sources: SourceInfo[] = React.useMemo(() => {
    try {
      return JSON.parse(sourceInfoJson || "[]");
    } catch {
      return [];
    }
  }, [sourceInfoJson]);

  const outputShape: number[] = React.useMemo(() => {
    try {
      return JSON.parse(outputShapeJson || "[]");
    } catch {
      return [];
    }
  }, [outputShapeJson]);

  const toolVisibility = React.useMemo(
    () => computeToolVisibility("Merge4DSTEM", disabledTools, hiddenTools),
    [disabledTools, hiddenTools],
  );
  const hideDisplay = toolVisibility.isHidden("display");
  const hideSources = toolVisibility.isHidden("sources");
  const hideMerge = toolVisibility.isHidden("merge");
  const hidePreview = toolVisibility.isHidden("preview");
  const hideStatsGroup = toolVisibility.isHidden("stats");
  const hideExport = toolVisibility.isHidden("export");
  const lockDisplay = toolVisibility.isLocked("display");
  const lockMerge = toolVisibility.isLocked("merge");
  const lockExport = toolVisibility.isLocked("export");

  // Compute preview canvas size (aspect-preserving fit)
  const previewSize = React.useMemo(() => {
    const h = Math.max(1, previewRows);
    const w = Math.max(1, previewCols);
    const scale = Math.min(PREVIEW_SIZE / w, PREVIEW_SIZE / h);
    return {
      width: Math.max(120, Math.round(w * scale)),
      height: Math.max(120, Math.round(h * scale)),
    };
  }, [previewRows, previewCols]);

  // Preview stats
  const parsed = React.useMemo(() => extractFloat32(previewBytes), [previewBytes]);
  const previewStats = React.useMemo(() => {
    if (!parsed || parsed.length === 0) return null;
    return computeStats(parsed);
  }, [parsed]);

  // Render preview canvas
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !parsed || parsed.length === 0) return;
    if (previewRows <= 0 || previewCols <= 0) return;
    if (parsed.length !== previewRows * previewCols) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const data = logScale ? applyLogScale(parsed) : parsed;
    const range = findDataRange(data);
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const offscreen = renderToOffscreen(data, previewCols, previewRows, lut, range.min, range.max);
    if (!offscreen) return;

    canvas.width = previewSize.width;
    canvas.height = previewSize.height;
    ctx.clearRect(0, 0, previewSize.width, previewSize.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreen, 0, 0, previewSize.width, previewSize.height);
  }, [parsed, previewRows, previewCols, cmap, logScale, previewSize]);

  const handleMerge = React.useCallback(() => {
    if (lockMerge || merged) return;
    setMergeRequested(true);
  }, [lockMerge, merged, setMergeRequested]);

  const handleExportPng = React.useCallback(() => {
    if (lockExport) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (blob) downloadBlob(blob, "merge4dstem_preview.png");
    }, "image/png");
  }, [lockExport]);

  const handleCopy = React.useCallback(async () => {
    if (lockExport) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.toBlob(async (blob) => {
      if (!blob) return;
      try {
        await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
      } catch {
        /* clipboard not available */
      }
    }, "image/png");
  }, [lockExport]);

  const colormapOptions = React.useMemo(() => ["inferno", "viridis", "gray", "magma"], []);

  // Bin factor options (powers of 2 that divide both detector dims)
  const binOptions = React.useMemo(() => {
    const opts = [1];
    for (const f of [2, 4, 8, 16]) {
      if (detRows >= f && detCols >= f && detRows % f === 0 && detCols % f === 0) {
        opts.push(f);
      }
    }
    return opts;
  }, [detRows, detCols]);

  const handleKeyDown = React.useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      const key = String(event.key || "").toLowerCase();
      if (isEditableElement(event.target)) return;

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
      if (key === "m" && !lockMerge && !merged) {
        event.preventDefault();
        setMergeRequested(true);
        return;
      }
      // Arrow keys: browse sources
      if (key === "arrowleft" && nSources > 1) {
        event.preventDefault();
        setPreviewIndex(Math.max(0, previewIndex - 1));
        return;
      }
      if (key === "arrowright" && nSources > 1) {
        event.preventDefault();
        setPreviewIndex(Math.min(nSources - 1, previewIndex + 1));
        return;
      }
    },
    [cmap, colormapOptions, lockDisplay, lockMerge, merged, nSources, previewIndex,
     setCmap, setLogScale, setMergeRequested, setPreviewIndex],
  );

  const statusColor =
    statusLevel === "error" ? "#ff6b6b" : statusLevel === "warn" ? "#ffb84d" : colors.textMuted;

  // Themed select styling (inside render — depends on colors)
  const themedSelect = {
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

  // Calibration label
  const calLabel = React.useMemo(() => {
    const parts: string[] = [];
    if (pixelCalibrated) parts.push(`${formatNumber(pixelSize, 4)} ${pixelUnit}`);
    if (kCalibrated) parts.push(`${formatNumber(kPixelSize, 4)} ${kUnit}`);
    return parts.length > 0 ? parts.join(" | ") : "uncalibrated";
  }, [pixelCalibrated, pixelSize, pixelUnit, kCalibrated, kPixelSize, kUnit]);

  return (
    <Box
      className="merge4dstem-root"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      sx={{
        p: `${SPACING.LG}px`,
        bgcolor: colors.bg,
        color: colors.text,
        outline: "none",
      }}
    >
      {/* ── Header ── */}
      <Typography
        variant="h6"
        sx={{
          ...typography.title,
          color: colors.accent,
          mb: `${SPACING.SM}px`,
          fontSize: 14,
        }}
      >
        {title}
        <ControlCustomizer
          widgetName="Merge4DSTEM"
          hiddenTools={hiddenTools}
          setHiddenTools={setHiddenTools}
          disabledTools={disabledTools}
          setDisabledTools={setDisabledTools}
          themeColors={colors}
        />
      </Typography>

      {showControls && (
        <Stack direction="row" spacing={`${SPACING.LG}px`} sx={{ mb: `${SPACING.MD}px` }}>
          {/* ── Left column: Source table + info ── */}
          <Stack spacing={`${SPACING.SM}px`} sx={{ minWidth: 300 }}>
            {/* Source table */}
            {!hideSources && (
              <Box
                sx={{
                  border: `1px solid ${colors.border}`,
                  bgcolor: colors.controlBg,
                  px: 1,
                  py: 0.5,
                }}
              >
                <Box
                  component="table"
                  sx={{
                    width: "100%",
                    borderCollapse: "collapse",
                    "& td, & th": {
                      fontSize: 10,
                      fontFamily: "monospace",
                      py: 0.25,
                      px: 0.5,
                      borderBottom: `1px solid ${colors.border}`,
                      color: colors.text,
                    },
                    "& th": {
                      fontWeight: "bold",
                      textAlign: "left",
                      color: colors.textMuted,
                    },
                  }}
                >
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Name</th>
                      <th>Shape</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {sources.map((src, i) => (
                      <tr
                        key={i}
                        style={{
                          cursor: "pointer",
                          backgroundColor:
                            i === previewIndex
                              ? `${colors.accent}22`
                              : undefined,
                        }}
                        onClick={() => setPreviewIndex(i)}
                      >
                        <td>{i + 1}</td>
                        <td>{src.name}</td>
                        <td>
                          ({src.shape[0]}, {src.shape[1]}, {src.shape[2]}, {src.shape[3]})
                        </td>
                        <td style={{ color: src.valid ? "#4caf50" : "#ff6b6b" }}>
                          {src.valid ? "\u2713" : "\u2717"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </Box>
              </Box>
            )}

            {/* Info row */}
            <Box
              sx={{
                ...controlRow,
                border: `1px solid ${colors.border}`,
                bgcolor: colors.bgAlt,
                flexWrap: "wrap",
                width: "100%",
                gap: `${SPACING.MD}px`,
              }}
            >
              <Typography sx={{ ...typography.value, color: colors.textMuted }}>
                scan{" "}
                <Box component="span" sx={{ color: colors.accent }}>
                  ({scanRows}, {scanCols})
                </Box>
              </Typography>
              <Typography sx={{ ...typography.value, color: colors.textMuted }}>
                det{" "}
                <Box component="span" sx={{ color: colors.accent }}>
                  ({detRows}, {detCols})
                </Box>
              </Typography>
              <Typography sx={{ ...typography.value, color: colors.textMuted }}>
                {calLabel}
              </Typography>
              <Typography sx={{ ...typography.value, color: colors.textMuted }}>
                {frameDimLabel}{" "}
                <Box component="span" sx={{ color: colors.accent }}>
                  {nSources}
                </Box>
              </Typography>
              <Typography sx={{ ...typography.value, color: colors.textMuted }}>
                {device}
              </Typography>
            </Box>
          </Stack>

          {/* ── Right column: Preview canvas ── */}
          {!hidePreview && (
            <Stack spacing={`${SPACING.XS}px`}>
              {/* Source slider */}
              {nSources > 1 && (
                <Stack direction="row" spacing={1} alignItems="center">
                  <Typography sx={{ ...typography.labelSmall, color: colors.textMuted }}>
                    Source:
                  </Typography>
                  <Slider
                    size="small"
                    min={0}
                    max={nSources - 1}
                    step={1}
                    value={previewIndex}
                    onChange={(_, v) => setPreviewIndex(v as number)}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(v) => `${v + 1}/${nSources}`}
                    sx={{
                      ...sliderStyles.small,
                      width: Math.max(100, previewSize.width - 80),
                      color: colors.accent,
                    }}
                  />
                  <Typography sx={{ ...typography.value, color: colors.accent, minWidth: 36 }}>
                    {previewIndex + 1}/{nSources}
                  </Typography>
                </Stack>
              )}

              {/* Canvas */}
              <Box
                sx={{
                  bgcolor: "#000",
                  border: `1px solid ${colors.border}`,
                  width: previewSize.width,
                  height: previewSize.height,
                  overflow: "hidden",
                  position: "relative",
                }}
              >
                <canvas
                  ref={canvasRef}
                  style={{
                    width: previewSize.width,
                    height: previewSize.height,
                    display: "block",
                  }}
                />
              </Box>

              {/* Preview stats bar */}
              {showStats && !hideStatsGroup && previewStats && (
                <Box
                  sx={{
                    px: 1,
                    py: 0.5,
                    bgcolor: colors.bgAlt,
                    display: "flex",
                    gap: 2,
                    width: previewSize.width,
                  }}
                >
                  {(["mean", "min", "max", "std"] as const).map((key) => (
                    <Typography
                      key={key}
                      sx={{ ...typography.value, color: colors.textMuted }}
                    >
                      {key}{" "}
                      <Box component="span" sx={{ color: colors.accent }}>
                        {formatStat(previewStats[key])}
                      </Box>
                    </Typography>
                  ))}
                </Box>
              )}
            </Stack>
          )}
        </Stack>
      )}

      {/* ── Controls row: Display + Bin + Merge + Export ── */}
      {showControls && (
        <Box
          sx={{
            ...controlRow,
            border: `1px solid ${colors.border}`,
            bgcolor: colors.controlBg,
            mb: `${SPACING.SM}px`,
            flexWrap: "wrap",
            width: "100%",
          }}
        >
          {/* Display controls (inline) */}
          {!hideDisplay && (
            <>
              <Typography sx={{ ...typography.labelSmall, color: colors.textMuted }}>
                Colormap:
              </Typography>
              <Select
                size="small"
                value={cmap}
                onChange={(e) => setCmap(String(e.target.value))}
                sx={{ ...themedSelect, minWidth: 80 }}
                MenuProps={themedMenuProps}
                disabled={lockDisplay}
              >
                {colormapOptions.map((c) => (
                  <MenuItem key={c} value={c}>
                    {c}
                  </MenuItem>
                ))}
              </Select>

              <Typography sx={{ ...typography.labelSmall, color: colors.textMuted }}>
                Log:
              </Typography>
              <Switch
                size="small"
                checked={logScale}
                onChange={(e) => setLogScale(e.target.checked)}
                sx={switchStyles.small}
                disabled={lockDisplay}
              />

              <Box sx={{ borderLeft: `1px solid ${colors.border}`, height: 20, mx: 0.5 }} />
            </>
          )}

          {/* Bin factor */}
          <Typography sx={{ ...typography.labelSmall, color: colors.textMuted }}>
            Bin:
          </Typography>
          <Select
            size="small"
            value={binFactor}
            onChange={(e) => setBinFactor(Number(e.target.value))}
            sx={{ ...themedSelect, minWidth: 50 }}
            MenuProps={themedMenuProps}
            disabled={lockMerge || merged}
          >
            {binOptions.map((f) => (
              <MenuItem key={f} value={f}>
                {f}x
              </MenuItem>
            ))}
          </Select>

          <Box sx={{ borderLeft: `1px solid ${colors.border}`, height: 20, mx: 0.5 }} />

          {/* Output shape preview */}
          <Typography sx={{ ...typography.value, color: colors.textMuted }}>
            output{" "}
            <Box component="span" sx={{ color: colors.accent }}>
              ({outputShape.join(", ")})
            </Box>
          </Typography>

          <Box sx={{ borderLeft: `1px solid ${colors.border}`, height: 20, mx: 0.5 }} />

          {/* Merge button */}
          {!hideMerge && (
            <Button
              size="small"
              variant={merged ? "outlined" : "contained"}
              color={merged ? "success" : "primary"}
              disabled={lockMerge || merged}
              onClick={handleMerge}
              sx={{ ...compactButton, fontSize: 11, px: 2, minWidth: 80 }}
            >
              {merged ? "MERGED" : binFactor > 1 ? `MERGE ${binFactor}x` : "MERGE"}
            </Button>
          )}

          {/* Export + Copy */}
          {!hideExport && (
            <>
              <Box sx={{ borderLeft: `1px solid ${colors.border}`, height: 20, mx: 0.5 }} />
              <Button
                size="small"
                variant="outlined"
                disabled={lockExport}
                onClick={handleExportPng}
                sx={compactButton}
              >
                EXPORT
              </Button>
              <Button
                size="small"
                variant="outlined"
                disabled={lockExport}
                onClick={handleCopy}
                sx={compactButton}
              >
                COPY
              </Button>
            </>
          )}
        </Box>
      )}

      {/* ── Status bar ── */}
      <Box
        sx={{
          px: 1,
          py: 0.5,
          bgcolor: colors.bgAlt,
          border: `1px solid ${colors.border}`,
        }}
      >
        <Typography sx={{ ...typography.value, color: statusColor }}>
          {statusMessage || "ready"}
        </Typography>
      </Box>
    </Box>
  );
}

export const render = createRender(Merge4DSTEMWidget);
