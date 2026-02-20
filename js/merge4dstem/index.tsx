import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import { useTheme } from "../theme";
import { extractFloat32, formatNumber, downloadBlob } from "../format";
import { applyLogScale, findDataRange } from "../stats";
import { COLORMAPS, renderToOffscreen } from "../colormaps";
import { computeToolVisibility } from "../tool-parity";
import { ControlCustomizer } from "../control-customizer";
import "./merge4dstem.css";

const PREVIEW_MAX = 280;

const typography = {
  label: { fontSize: 11 },
  value: { fontSize: 11, fontFamily: "monospace" },
  section: { fontSize: 11, fontWeight: "bold" as const },
};

const switchStyles = {
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
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

function sizeForPreview(rows: number, cols: number): { width: number; height: number } {
  const h = Math.max(1, rows);
  const w = Math.max(1, cols);
  const scale = Math.min(PREVIEW_MAX / w, PREVIEW_MAX / h);
  return {
    width: Math.max(120, Math.round(w * scale)),
    height: Math.max(120, Math.round(h * scale)),
  };
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

  const [merged] = useModelState<boolean>("merged");
  const [outputShapeJson] = useModelState<string>("output_shape_json");
  const [frameDimLabel] = useModelState<string>("frame_dim_label");

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

  const previewSize = React.useMemo(
    () => sizeForPreview(previewRows, previewCols),
    [previewRows, previewCols],
  );

  // Render preview canvas
  const parsed = React.useMemo(() => extractFloat32(previewBytes), [previewBytes]);
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
      }
    },
    [cmap, colormapOptions, lockDisplay, lockMerge, merged, setCmap, setLogScale, setMergeRequested],
  );

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
    <Box
      className="merge4dstem-root"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      sx={{ p: 2, bgcolor: colors.bg, color: colors.text, outline: "none" }}
    >
      {/* Header */}
      <Stack
        direction="row"
        spacing={0.8}
        alignItems="center"
        justifyContent="space-between"
        sx={{ mb: 1 }}
      >
        <Typography sx={{ fontSize: 12, fontWeight: "bold", color: colors.accent }}>
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
        {!hideExport && (
          <>
            <Button
              size="small"
              variant="outlined"
              disabled={lockExport}
              onClick={handleExportPng}
              sx={{ fontSize: 10, minWidth: 60 }}
            >
              Export
            </Button>
            <Button
              size="small"
              variant="outlined"
              disabled={lockExport}
              onClick={handleCopy}
              sx={{ fontSize: 10, minWidth: 50 }}
            >
              Copy
            </Button>
          </>
        )}
      </Stack>

      {showControls && (
        <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap", mb: 1.5 }}>
          {/* Source table */}
          {!hideSources && (
            <Box
              sx={{
                border: `1px solid ${colors.border}`,
                bgcolor: colors.controlBg,
                p: 1,
                minWidth: 320,
              }}
            >
              <Typography sx={{ ...typography.section, color: colors.text, mb: 0.75 }}>
                Sources ({nSources})
              </Typography>
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
                  "& th": { fontWeight: "bold", textAlign: "left" },
                }}
              >
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Name</th>
                    <th>Shape</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {sources.map((src, i) => (
                    <tr key={i}>
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

          {/* Preview */}
          {!hidePreview && (
            <Box sx={{ minWidth: previewSize.width }}>
              <Typography sx={{ ...typography.section, color: colors.text, mb: 0.5 }}>
                Preview (mean DP, source 1)
              </Typography>
              <Box
                sx={{
                  border: `1px solid ${colors.border}`,
                  bgcolor: "#000",
                  width: previewSize.width,
                  height: previewSize.height,
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
              {showStats && !hideStatsGroup && (
                <Typography
                  sx={{
                    ...typography.value,
                    color: colors.textMuted,
                    mt: 0.5,
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  {previewRows}x{previewCols} px
                </Typography>
              )}
            </Box>
          )}
        </Stack>
      )}

      {showControls && (
        <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap", mb: 1.5 }}>
          {/* Info panel */}
          <Box
            sx={{
              border: `1px solid ${colors.border}`,
              bgcolor: colors.bgAlt,
              p: 1,
              minWidth: 320,
            }}
          >
            <Typography sx={{ ...typography.section, color: colors.text, mb: 0.5 }}>
              Shape + Calibration
            </Typography>
            <Typography sx={{ ...typography.value, color: colors.textMuted }}>
              scan: ({scanRows}, {scanCols}) det: ({detRows}, {detCols})
            </Typography>
            {pixelCalibrated && (
              <Typography sx={{ ...typography.value, color: colors.textMuted }}>
                real: {formatNumber(pixelSize, 4)} {pixelUnit}/px
              </Typography>
            )}
            {kCalibrated && (
              <Typography sx={{ ...typography.value, color: colors.textMuted }}>
                k-space: {formatNumber(kPixelSize, 4)} {kUnit}/px
              </Typography>
            )}
            <Typography sx={{ ...typography.value, color: colors.textMuted }}>
              {frameDimLabel} axis: {nSources} frames
            </Typography>
            <Typography sx={{ ...typography.value, color: colors.textMuted }}>
              output: ({outputShape.join(", ")})
            </Typography>
            <Typography sx={{ ...typography.value, color: colors.textMuted }}>
              device: {device}
            </Typography>
          </Box>

          {/* Display controls */}
          {!hideDisplay && (
            <Box
              sx={{
                border: `1px solid ${colors.border}`,
                bgcolor: colors.controlBg,
                p: 1,
                minWidth: 200,
                opacity: lockDisplay ? 0.6 : 1,
              }}
            >
              <Typography sx={{ ...typography.section, color: colors.text, mb: 0.75 }}>
                Display
              </Typography>
              <Stack spacing={0.5}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Typography sx={typography.label}>Colormap:</Typography>
                  <Select
                    size="small"
                    value={cmap}
                    onChange={(e) => setCmap(String(e.target.value))}
                    sx={controlSelect}
                    MenuProps={themedMenuProps}
                    disabled={lockDisplay}
                  >
                    {colormapOptions.map((c) => (
                      <MenuItem key={c} value={c}>
                        {c}
                      </MenuItem>
                    ))}
                  </Select>
                </Box>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Typography sx={typography.label}>Log scale:</Typography>
                  <Switch
                    size="small"
                    checked={logScale}
                    onChange={(e) => setLogScale(e.target.checked)}
                    sx={switchStyles.small}
                    disabled={lockDisplay}
                  />
                </Box>
              </Stack>
            </Box>
          )}

          {/* Merge button */}
          {!hideMerge && (
            <Box
              sx={{
                border: `1px solid ${colors.border}`,
                bgcolor: colors.controlBg,
                p: 1,
                minWidth: 140,
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
                alignItems: "center",
              }}
            >
              <Button
                variant="contained"
                color={merged ? "success" : "primary"}
                disabled={lockMerge || merged}
                onClick={handleMerge}
                sx={{ fontSize: 12, minWidth: 120, mb: 0.5 }}
              >
                {merged ? "MERGED" : "MERGE"}
              </Button>
              {merged && (
                <Typography
                  sx={{ ...typography.value, color: "#4caf50", textAlign: "center" }}
                >
                  {outputShape.join(" x ")}
                </Typography>
              )}
            </Box>
          )}
        </Stack>
      )}

      {/* Status bar */}
      <Box
        sx={{
          border: `1px solid ${colors.border}`,
          bgcolor: colors.bgAlt,
          p: 0.75,
          mt: 0.5,
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
