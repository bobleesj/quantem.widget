import * as React from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Switch from "@mui/material/Switch";
import Tooltip from "@mui/material/Tooltip";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import Button from "@mui/material/Button";
import Menu from "@mui/material/Menu";
import TuneIcon from "@mui/icons-material/Tune";

import {
  addToolGroup,
  compactToolLabel,
  computeToolVisibility,
  getControlPresetIds,
  getControlPresetLabel,
  getWidgetToolGroups,
  removeToolGroup,
  resolvePresetHiddenTools,
} from "./tool-parity";

type ToolSetter = React.Dispatch<React.SetStateAction<string[]>>;

type ThemeColors = {
  controlBg: string;
  text: string;
  border: string;
  textMuted?: string;
  accent?: string;
};

type ControlCustomizerProps = {
  widgetName: string;
  hiddenTools: string[];
  setHiddenTools: ToolSetter;
  disabledTools: string[];
  setDisabledTools: ToolSetter;
  themeColors: ThemeColors;
  labelOverrides?: Record<string, string>;
};

const switchStyles = {
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};

const presetButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
};

export function ControlCustomizer({
  widgetName,
  hiddenTools,
  setHiddenTools,
  disabledTools,
  setDisabledTools,
  themeColors,
  labelOverrides,
}: ControlCustomizerProps) {
  const [anchor, setAnchor] = React.useState<HTMLElement | null>(null);
  const groups = React.useMemo(
    () => getWidgetToolGroups(widgetName).filter((group) => group !== "all"),
    [widgetName],
  );
  const visibility = React.useMemo(
    () => computeToolVisibility(widgetName, disabledTools, hiddenTools),
    [widgetName, disabledTools, hiddenTools],
  );

  const setGroupVisible = React.useCallback((group: string, visible: boolean) => {
    setHiddenTools((prev) => {
      if (visible) return removeToolGroup(widgetName, prev, group);
      return addToolGroup(widgetName, prev, group);
    });
  }, [setHiddenTools, widgetName]);

  const setGroupLocked = React.useCallback((group: string, locked: boolean) => {
    setDisabledTools((prev) => {
      if (locked) return addToolGroup(widgetName, prev, group);
      return removeToolGroup(widgetName, prev, group);
    });
  }, [setDisabledTools, widgetName]);

  const applyPreset = React.useCallback((presetId: string) => {
    setHiddenTools(resolvePresetHiddenTools(widgetName, presetId));
  }, [setHiddenTools, widgetName]);

  return (
    <>
      <Tooltip title="Customize controls" arrow placement="top">
        <IconButton
          size="small"
          aria-label="Customize controls"
          onClick={(e) => setAnchor(e.currentTarget)}
          sx={{ p: 0.25, ml: 0.5, color: themeColors.text }}
        >
          <TuneIcon sx={{ fontSize: 16 }} />
        </IconButton>
      </Tooltip>
      <Menu
        anchorEl={anchor}
        open={Boolean(anchor)}
        onClose={() => setAnchor(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
        transformOrigin={{ vertical: "top", horizontal: "right" }}
        PaperProps={{
          sx: {
            bgcolor: themeColors.controlBg,
            color: themeColors.text,
            border: `1px solid ${themeColors.border}`,
            p: 0.5,
            minWidth: 280,
          },
        }}
      >
        <Box sx={{ px: 0.5, pb: 0.75 }}>
          <Typography sx={{ fontSize: 11, fontWeight: "bold", mb: 0.75 }}>Presets</Typography>
          <Box sx={{ display: "flex", gap: 0.5, flexWrap: "wrap" }}>
            {getControlPresetIds().map((presetId) => (
              <Button
                key={presetId}
                size="small"
                sx={presetButton}
                data-testid={`preset-${presetId}`}
                onClick={() => applyPreset(presetId)}
              >
                {getControlPresetLabel(presetId)}
              </Button>
            ))}
          </Box>
        </Box>
        <Divider sx={{ borderColor: themeColors.border, my: 0.5 }} />
        <Box sx={{ maxHeight: 300, overflowY: "auto", px: 0.5 }}>
          <Typography sx={{ fontSize: 11, fontWeight: "bold", mb: 0.5 }}>Per-group</Typography>
          {groups.map((group) => {
            const label = labelOverrides?.[group] ?? compactToolLabel(group);
            const hidden = visibility.isHidden(group);
            const locked = visibility.isLocked(group);
            return (
              <Box key={group} data-testid={`tool-row-${group}`} sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", py: 0.25, gap: 0.5 }}>
                <Typography sx={{ fontSize: 11 }}>{label}</Typography>
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                  <Typography sx={{ fontSize: 10, color: themeColors.textMuted ?? themeColors.text }}>Show</Typography>
                  <Switch
                    size="small"
                    checked={!hidden}
                    onChange={(e) => setGroupVisible(group, e.target.checked)}
                    inputProps={{ "aria-label": `show-${group}` }}
                    sx={switchStyles.small}
                  />
                  <Typography sx={{ fontSize: 10, color: themeColors.textMuted ?? themeColors.text }}>Lock</Typography>
                  <Switch
                    size="small"
                    checked={locked}
                    onChange={(e) => setGroupLocked(group, e.target.checked)}
                    inputProps={{ "aria-label": `lock-${group}` }}
                    sx={switchStyles.small}
                    disabled={hidden}
                  />
                </Box>
              </Box>
            );
          })}
        </Box>
      </Menu>
    </>
  );
}
