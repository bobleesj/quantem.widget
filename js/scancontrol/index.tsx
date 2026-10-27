/**
 * ScanControl: connect, set up and run STEM scans, and manage the detector's files.
 *
 * Python owns the runner and every hardware call; this panel shows the runner's status
 * (polled into the `status` trait), edits the scan settings traits and sends actions
 * ({action: "connect" | "go" | "stop" | "refresh" | "download" | "delete"}). Go moves the beam.
 * The files list is one row per scan (master + data files), selected like a Finder list:
 * click, Ctrl/Cmd-click to toggle, Shift-click for a range.
 */

import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import Switch from "@mui/material/Switch";
import TextField from "@mui/material/TextField";
import { compactButton, controlRow, switchStyles } from "../controlStyles";
import { useTheme, WIDGET_UI_FONT } from "../theme";

type Scan = { name: string; files: number; bytes: number | null; modified: string };

type Status = {
  state?: string;
  message?: string;
  dcu_ok?: boolean;
  revolon_ok?: boolean;
  high_voltage?: string | null;
  sensor?: string | null;
  microscope?: Record<string, string> | null;
  done?: number;
  total?: number;
  dropped?: number | null;
  scans?: Scan[];
  free_bytes?: number | null;
  dwell_us?: number;
  busy?: boolean;
  notice?: string;
  notice_error?: boolean;
};

const STATE_COLORS: Record<string, string> = {
  READY: "#3c9a5f",
  SCANNING: "#d98a00",
  STOPPING: "#d98a00",
  BUSY: "#5a8fd8",
  ERROR: "#d32f2f",
  NOT_CONNECTED: "#888",
};

// ANS keeps a 512x512x96x96 uint16 scan at about a tenth of its raw size (measured on ARINA data)
const QEM_RATIO = 0.1;

function gb(bytes: number | null | undefined, digits = 2): string {
  return bytes == null ? "?" : `${(bytes / 1e9).toFixed(digits)} GB`;
}

function when(iso: string): string {
  if (!iso) return "";
  const date = new Date(iso);
  return isNaN(date.getTime()) ? iso : date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function Dot({ ok, label }: { ok: boolean; label: string }) {
  return <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
    <span aria-label={`${label} ${ok ? "connected" : "not connected"}`}
      style={{ width: 9, height: 9, borderRadius: "50%", background: ok ? "#3c9a5f" : "#777" }} />
    {label}
  </span>;
}

// Microscope detectors: → out (green), ← in (amber; red for the Ceta and the screen, which
// block Go), – fixed, cannot move (grey), ✕ error (red), ? anything else (grey).
function position(name: string, state: string): [string, string] {
  const s = state.toLowerCase();
  return s === "retracted" ? ["→", "#3c9a5f"]
    : s === "inserted" ? ["←", /ceta/i.test(name) || name === "Screen" ? "#d32f2f" : "#e6a100"]
    : s === "fixed" ? ["–", "#777"]
    : s.startsWith("error") ? ["✕", "#d32f2f"] : ["?", "#777"];
}

function ScanControlView() {
  const model = useModel();
  const { colors } = useTheme();
  const [status] = useModelState<Status>("status");
  const [title] = useModelState<string>("title");
  const [sizes] = useModelState<number[]>("sizes");
  const [size, setSize] = useModelState<number>("size");
  const [binned, setBinned] = useModelState<boolean>("binned");
  const [output, setOutput] = useModelState<string>("output");
  const [save, setSave] = useModelState<boolean>("save");
  const [count, setCount] = useModelState<number>("count");
  const [forever, setForever] = useModelState<boolean>("forever");
  const [folder, setFolder] = useModelState<string>("folder");
  const [selected, setSelected] = React.useState<string[]>([]);
  const [anchor, setAnchor] = React.useState<number | null>(null);
  const [format, setFormat] = React.useState<"h5" | "qem">("h5");
  const [confirmDelete, setConfirmDelete] = React.useState(false);
  const [countText, setCountText] = React.useState(String(count));
  React.useEffect(() => setCountText(String(count)), [count]);

  const state = status?.state ?? "NOT_CONNECTED";
  const scans = status?.scans ?? [];
  const names = scans.map((scan) => scan.name);
  React.useEffect(() => setSelected((chosen) => chosen.filter((name) => names.includes(name))), [names.join("\n")]);
  const running = state === "SCANNING" || state === "STOPPING";
  const locked = running || state === "BUSY";
  const send = (content: object) => model.send(content);
  const hvOn = status?.high_voltage === "READY";
  const inserted = status?.sensor === "inserted";

  const chosen = scans.filter((scan) => selected.includes(scan.name));
  const h5Bytes = chosen.some((scan) => scan.bytes == null) ? null : chosen.reduce((sum, scan) => sum + (scan.bytes ?? 0), 0);
  const sizeText = h5Bytes == null ? "size unknown" : format === "h5" ? `${gb(h5Bytes)}` : `≈ ${gb(h5Bytes * QEM_RATIO)} (estimate)`;

  // Finder selection: click selects one, Ctrl/Cmd toggles, Shift extends from the last click.
  const pick = (index: number, event: React.MouseEvent) => {
    const name = names[index];
    if (event.shiftKey && anchor !== null) {
      const [lo, hi] = [Math.min(anchor, index), Math.max(anchor, index)];
      const range = names.slice(lo, hi + 1);
      setSelected((current) => (event.metaKey || event.ctrlKey) ? Array.from(new Set([...current, ...range])) : range);
      return;
    }
    if (event.metaKey || event.ctrlKey) {
      setSelected((current) => current.includes(name) ? current.filter((n) => n !== name) : [...current, name]);
    } else {
      setSelected([name]);
    }
    setAnchor(index);
  };
  const onKey = (event: React.KeyboardEvent) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "a") {
      event.preventDefault();
      setSelected(names);
    } else if (event.key === "Escape") {
      setSelected([]);
    }
  };

  const text = { color: colors.text, fontFamily: WIDGET_UI_FONT, fontSize: 12 };
  const field = { "& .MuiInputBase-root": { color: colors.text, fontSize: 12, bgcolor: colors.controlBg },
                  "& .MuiOutlinedInput-notchedOutline": { borderColor: colors.border } };
  const button = { ...compactButton, color: colors.text, borderColor: colors.border };
  const box = { border: `1px solid ${colors.border}`, borderRadius: 1, p: 1, mb: 1 };
  const cell = { padding: "3px 8px", whiteSpace: "nowrap" as const };

  return <Box sx={{ ...text, bgcolor: colors.bg, p: 1.5, maxWidth: 780 }}>
    <Box sx={{ display: "flex", alignItems: "baseline", gap: 1.5, mb: 1, flexWrap: "wrap" }}>
      <strong style={{ fontSize: 15, color: colors.accent }}>{title}</strong>
      <Dot ok={!!status?.dcu_ok} label="Detector" />
      <Dot ok={!!status?.revolon_ok} label="Scan controller" />
    </Box>

    <Box role="status" sx={{ ...box, borderLeft: `4px solid ${STATE_COLORS[state] ?? "#888"}` }}>
      <strong>{state.replace("_", " ")}</strong> &nbsp; {status?.message}
      {!!status?.total && <span style={{ color: colors.textMuted }}> &nbsp; ({status.done ?? 0} of {status.total} done
        {status?.dropped != null && `, ${status.dropped} frames dropped`})</span>}
      {status?.notice && <div style={{ color: status.notice_error ? "#d32f2f" : colors.textMuted, marginTop: 4 }}>{status.notice}</div>}
    </Box>

    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
      <Button size="small" variant="contained" disabled={locked} onClick={() => send({ action: "connect" })}
        sx={{ ...compactButton, fontSize: 13, px: 2.5 }}>Connect</Button>
      <span style={{ color: colors.textMuted }}>connects the detector and the scan controller and sets the detector up</span>
    </Box>

    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1, flexWrap: "wrap" }}>
      <Dot ok={hvOn} label={`HV ${status?.high_voltage ?? "?"}`} />
      <Button size="small" variant="outlined" sx={button} disabled={locked || !status?.dcu_ok}
        onClick={() => send({ action: "high_voltage", on: !hvOn })}>{hvOn ? "HV off" : "HV on"}</Button>
      <Dot ok={inserted} label={`Detector ${status?.sensor ?? "?"}`} />
      <Button size="small" variant="outlined" sx={button} disabled={locked || !status?.dcu_ok}
        onClick={() => send({ action: inserted ? "retract" : "insert" })}>{inserted ? "Retract" : "Insert"}</Button>
    </Box>

    <Box sx={{ ...box }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
        <strong>Microscope</strong>
        {status?.microscope?.error && <span style={{ color: colors.textMuted }}>{status.microscope.error}</span>}
        {!status?.microscope && <span style={{ color: colors.textMuted }}>not read yet</span>}
        <Button size="small" sx={{ ...button, ml: "auto" }} disabled={running} onClick={() => send({ action: "microscope" })}>Check</Button>
      </Box>
      {status?.microscope && !status.microscope.error &&
        <table style={{ borderCollapse: "collapse" }}>
          <thead><tr style={{ color: colors.textMuted, textAlign: "left" }}>
            <th style={cell}>Detector</th><th style={{ ...cell, textAlign: "center" }}>Position</th><th style={cell}>State</th>
          </tr></thead>
          <tbody>{Object.entries(status.microscope).map(([name, state]) => {
            const [symbol, color] = position(name, state);
            return <tr key={name} style={{ borderTop: `1px solid ${colors.border}` }}>
              <td style={cell}>{name}</td>
              <td style={{ ...cell, textAlign: "center", color, fontSize: 22, fontWeight: 900, lineHeight: 1 }}>{symbol}</td>
              <td style={{ ...cell, color }}>{state}</td>
            </tr>;
          })}</tbody>
        </table>}
    </Box>

    <Box sx={{ ...controlRow, ...box, width: "auto" }}>
      <span>Size</span>
      <Select size="small" value={size} disabled={locked} sx={field} onChange={(e) => setSize(Number(e.target.value))}>
        {(sizes ?? []).map((side) => <MenuItem key={side} value={side}>{side} x {side}</MenuItem>)}
      </Select>
      <span>Binning</span>
      <Select size="small" value={binned ? "2x2" : "1x1"} disabled={locked} sx={field}
        onChange={(e) => setBinned(e.target.value === "2x2")}>
        <MenuItem value="2x2">2x2 (120 kHz)</MenuItem>
        <MenuItem value="1x1">unbinned (30 kHz)</MenuItem>
      </Select>
      <span>Output</span>
      <Select size="small" value={output} disabled={locked} sx={field} onChange={(e) => setOutput(String(e.target.value))}>
        <MenuItem value="rdma">RDMA</MenuItem>
        <MenuItem value="stream">normal stream</MenuItem>
      </Select>
      <span>Save HDF5</span>
      <Switch size="small" checked={save} disabled={locked} sx={switchStyles.small} onChange={(e) => setSave(e.target.checked)} />
      <span style={{ color: colors.textMuted }}>dwell {(status?.dwell_us ?? 0).toFixed(2)} µs</span>
    </Box>

    <Box sx={{ ...controlRow, ...box, width: "auto" }}>
      <span>Scans</span>
      <TextField size="small" value={countText} disabled={locked || forever} sx={{ ...field, width: 80 }}
        inputProps={{ inputMode: "numeric", "aria-label": "Scans per Go" }}
        onChange={(e) => setCountText(e.target.value)}
        onBlur={() => { const n = Math.round(Number(countText)); if (n >= 1) setCount(n); else setCountText(String(count)); }} />
      <span>Run forever</span>
      <Switch size="small" checked={forever} disabled={locked} sx={switchStyles.small} onChange={(e) => setForever(e.target.checked)} />
      <Button size="small" variant="contained" disabled={state !== "READY" || !hvOn || !inserted} onClick={() => send({ action: "go" })}
        title="Moves the electron beam" sx={{ ...compactButton, fontSize: 13, px: 2.5, bgcolor: "#c62828", "&:hover": { bgcolor: "#b71c1c" } }}>
        Go
      </Button>
      <Button size="small" variant="outlined" sx={button} disabled={!running} onClick={() => send({ action: "stop" })}>Stop</Button>
      <span style={{ color: "#d32f2f", fontSize: 11 }}>Go moves the beam</span>
    </Box>

    <Box sx={box}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap", mb: 0.5 }}>
        <strong>Scans on the detector</strong>
        <span style={{ color: colors.textMuted }}>{scans.length} scans · free {gb(status?.free_bytes, 0)}</span>
        <Box sx={{ flex: 1 }} />
        <Button size="small" sx={button} disabled={!scans.length} onClick={() => setSelected(names)}>Select all</Button>
        <Button size="small" sx={button} disabled={!selected.length} onClick={() => setSelected([])}>Select none</Button>
        <Button size="small" sx={button} disabled={locked || !status?.dcu_ok} onClick={() => send({ action: "refresh" })}>Refresh</Button>
      </Box>
      <Box tabIndex={0} onKeyDown={onKey} aria-label="Scans on the detector"
        sx={{ maxHeight: 240, overflowY: "auto", border: `1px solid ${colors.border}`, borderRadius: 1, mb: 1, outline: "none", userSelect: "none" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead style={{ position: "sticky", top: 0, background: colors.bgAlt, color: colors.textMuted, textAlign: "left" }}>
            <tr><th style={cell}>Name</th><th style={{ ...cell, textAlign: "right" }}>Size</th><th style={cell}>Files</th><th style={cell}>Modified</th></tr>
          </thead>
          <tbody>
            {scans.length === 0 && <tr><td colSpan={4} style={{ ...cell, color: colors.textMuted }}>No files</td></tr>}
            {scans.map((scan, index) => {
              const on = selected.includes(scan.name);
              return <tr key={scan.name} aria-selected={on} onMouseDown={(event) => pick(index, event)}
                style={{ cursor: "default", background: on ? colors.accent : index % 2 ? colors.bgAlt : "transparent",
                         color: on ? "#fff" : colors.text }}>
                <td style={cell}>{scan.name}</td>
                <td style={{ ...cell, textAlign: "right" }}>{gb(scan.bytes)}</td>
                <td style={cell}>{scan.files}</td>
                <td style={cell}>{when(scan.modified)}</td>
              </tr>;
            })}
          </tbody>
        </table>
      </Box>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
        <span>Save as</span>
        <Select size="small" value={format} sx={field} onChange={(e) => setFormat(e.target.value as "h5" | "qem")}>
          <MenuItem value="h5">.h5 (detector files)</MenuItem>
          <MenuItem value="qem">.qem (ANS, quantem.gpu)</MenuItem>
        </Select>
        <TextField size="small" value={folder} placeholder="Download folder" disabled={locked}
          sx={{ ...field, flex: 1, minWidth: 220 }} onChange={(e) => setFolder(e.target.value)} />
      </Box>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap", mt: 1 }}>
        <span style={{ color: colors.textMuted }}>{selected.length} selected · {selected.length ? sizeText : "0 GB"}</span>
        <Box sx={{ flex: 1 }} />
        <Button size="small" variant="contained" disabled={locked || !selected.length || !folder}
          sx={{ ...compactButton, fontSize: 12, px: 2 }}
          onClick={() => send({ action: "download", scans: selected, format })}>Download {selected.length || ""}</Button>
        <Button size="small" variant="outlined" sx={{ ...button, color: "#d32f2f" }} disabled={locked || !selected.length}
          onClick={() => setConfirmDelete(true)}>Delete {selected.length || ""}</Button>
      </Box>
      {confirmDelete && <Box role="alert" sx={{ display: "flex", alignItems: "center", gap: 1, mt: 1, p: 1, bgcolor: colors.bgAlt }}>
        <span style={{ flex: 1 }}>Delete {selected.length} scan(s) from the detector? This cannot be undone.</span>
        <Button size="small" sx={{ ...button, color: "#d32f2f" }} onClick={() => { send({ action: "delete", scans: selected }); setConfirmDelete(false); }}>Delete</Button>
        <Button size="small" sx={button} onClick={() => setConfirmDelete(false)}>Cancel</Button>
      </Box>}
    </Box>
  </Box>;
}

export default { render: createRender(ScanControlView) };
