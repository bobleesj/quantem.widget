import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import { useTheme, WIDGET_UI_FONT } from "../theme";
import { SPACING } from "../controlStyles";
import { COLORMAPS } from "../display/luts";
import { GPUColormapEngine } from "../display/colormaps";
import { createColormapEngine, createCPUColormapEngine, type DisplayColormapEngine } from "../display/cpuColormap";
import { onGPULost } from "../display/device";
import { applyHannWindow2D, fftshift, getDisplayFFT, zeroPadTopLeft, type DisplayFFT } from "../display/fft";
import { InfoTooltip } from "../shared/InfoTooltip";
import { bySize, ComplexCanvas, SizeControl, type ViewSize } from "../shared/viewSize";
import { useViewLink, ViewLink, ZoomFrame } from "../shared/zoomPan";
import { contrastOf, groupsOf, plain, signed, type Candidate } from "./search";

// ShowSearch: a parameter search in one view. The geometry candidates as a grid of thumbnails on thickness × focus,
// every later candidate and finishing stage as a strip, each candidate's held-out against epoch (convergence), and the
// hovered or chosen candidate's projection, spectrum, probe and depth cut. Hover looks, click chooses, leaving goes back.
// Drawn in the browser from the synced arrays and the messages that add to them, with no kernel round trip; images go
// through the package's colormap engine (WebGPU, or its Canvas2D twin), as in ShowReconstruction.

type Stage = { name: string; verdict: string; held_out: number | null; start: number | null; control: number | null; flag: string; thickness_nm: number | null; focus_nm: number | null };
type Pick = { stage: string; name: string; held_out: number; paired_SE: number | null; flags: string[] };
type Frame = { data: Float32Array; shape: number[] };

const TEXT = 11;
const GAP = SPACING.XS;
const STAGE_COLORS: { [stage: string]: string } = { geometry: "#8c8c8c", slices: "#6f8fb3", refocus: "#b09a72", refinement: "#9a7fb0" };
const PLATE: React.CSSProperties = { background: "rgba(0,0,0,0.6)", padding: "0 3px", borderRadius: 2, textShadow: "none" };  // a label readable over any image, inside its own tile
const ON_IMAGE: React.CSSProperties = { position: "absolute", fontSize: TEXT, color: "#fff", textShadow: "0 0 3px #000, 0 0 2px #000", pointerEvents: "none", lineHeight: "13px", whiteSpace: "nowrap" };
const MARKS: { [mark: string]: [string, string] } = { best: ["★", "the pick of its stage: lowest held-out"], tie: ["▲", "within 2 standard errors of the pick"] };

function floatsOf(view: DataView | ArrayBuffer | null | undefined): Float32Array {
  if (!view || view.byteLength === 0) return new Float32Array(0);
  if (view instanceof ArrayBuffer) return new Float32Array(view.slice(0));
  return new Float32Array(view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength));
}

function clock(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "";
  const whole = Math.round(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

// ---------------------------------------------------------------------------------------------------------------
// The arrays and curves in the browser: the synced traits, plus what the messages added since

class Store {
  frames = new Map<string, Frame>();
  curves: { [name: string]: [number, number][] } = {};
  listeners = new Set<() => void>();
  version = 0;
  constructor(public model: any) {
    this.read();
    model.on("change:images", () => { this.read(); this.bump(); });
    model.on("change:image_index", () => { this.read(); this.bump(); });
    model.on("change:curves", () => { this.read(); this.bump(); });
    model.on("msg:custom", (content: any, buffers: (DataView | ArrayBuffer)[]) => this.onMessage(content, buffers));
    model.send({ kind: "hello" });  // the arrays sent before this view existed
  }

  read() {
    const values = floatsOf(this.model.get("images"));
    let offset = 0;
    for (const entry of (this.model.get("image_index") || []) as { name: string; kind: string; shape: number[] }[]) {
      const size = entry.shape.reduce((a, b) => a * b, 1);
      this.frames.set(`${entry.name}/${entry.kind}`, { data: values.subarray(offset, offset + size), shape: entry.shape });
      offset += size;
    }
    const synced = this.model.get("curves") || {};
    for (const name of Object.keys(synced)) {
      const merged = new Map<number, number>((this.curves[name] || []).map(([epoch, value]) => [epoch, value]));
      for (const [epoch, value] of synced[name]) merged.set(epoch, value);
      this.curves[name] = [...merged.entries()].sort((a, b) => a[0] - b[0]);
    }
  }

  onMessage(content: any, buffers: (DataView | ArrayBuffer)[]) {
    if (!content) return;
    if (content.kind === "epoch") {
      const points = (this.curves[content.name] || []).filter(([epoch]) => epoch !== content.point[0]);
      points.push(content.point);
      this.curves[content.name] = points.sort((a, b) => a[0] - b[0]);
    } else if (content.kind === "arrays") {
      content.entries.forEach((entry: { name: string; kind: string; shape: number[] }, index: number) => {
        this.frames.set(`${entry.name}/${entry.kind}`, { data: floatsOf(buffers[index]), shape: entry.shape });
      });
    } else return;
    this.bump();
  }

  bump() { this.version++; this.listeners.forEach((listener) => listener()); }
  subscribe(listener: () => void) { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; }
}

const STORES = new WeakMap<object, Store>();
function useStore(): Store {
  const model = useModel();
  let store = STORES.get(model);
  if (!store) { store = new Store(model); STORES.set(model, store); }
  const [, setVersion] = React.useState(0);
  React.useEffect(() => store!.subscribe(() => setVersion((value) => value + 1)), [store]);
  return store;
}

// ---------------------------------------------------------------------------------------------------------------
// Images

const Engine = React.createContext<{ engine: DisplayColormapEngine | null; slot: () => number }>({ engine: null, slot: () => 0 });

function useEngine() {
  const [engine, setEngine] = React.useState<DisplayColormapEngine | null>(null);
  const slots = React.useRef(0);
  React.useEffect(() => {
    let disposed = false, current: DisplayColormapEngine | null = null;
    createColormapEngine().then((made) => { if (disposed) made.destroy(); else { current = made; setEngine(made); } })
      .catch((error) => console.error("[ShowSearch] colormap engine failed to start", error));
    const stop = onGPULost(() => {
      if (disposed || !(current instanceof GPUColormapEngine)) return;
      const lost = current; current = createCPUColormapEngine(); setEngine(current); lost.destroy();
    });
    return () => { disposed = true; stop(); current?.destroy(); };
  }, []);
  return React.useMemo(() => ({ engine, slot: () => slots.current++ }), [engine]);
}

// One image at its native pixels, stretched into width × height (the depth cut) or square; contrast from its own
// percentiles unless low and high are given.
function Picture({ data, rows, cols, width, height, cmap = "gray", low, high, transform }: {
  data: Float32Array; rows: number; cols: number; width: number; height: number; cmap?: string; low?: number; high?: number; transform?: (value: number) => number;
}) {
  const { engine, slot: nextSlot } = React.useContext(Engine);
  const ref = React.useRef<HTMLCanvasElement>(null);
  const slot = React.useRef(-1);
  if (slot.current < 0) slot.current = nextSlot();
  const shown = React.useMemo(() => (transform ? Float32Array.from(data, transform) : data), [data, transform]);
  const range = React.useMemo(() => (low != null && high != null ? [low, high] : contrastOf(shown)), [shown, low, high]);
  React.useEffect(() => {
    const canvas = ref.current;
    if (!engine || !canvas || rows === 0 || cols === 0 || shown.length < rows * cols) return;
    let cancelled = false;
    engine.uploadData(slot.current, shown.subarray(0, rows * cols), cols, rows);
    engine.renderSlotsToImageBitmapAsync([slot.current], [{ vmin: range[0], vmax: range[1] }], false, cmap, COLORMAPS[cmap] || COLORMAPS.gray).then((bitmaps) => {
      const bitmap = bitmaps && bitmaps[0];
      if (!bitmap) return;
      if (!cancelled) {
        if (canvas.width !== cols || canvas.height !== rows) { canvas.width = cols; canvas.height = rows; }
        canvas.getContext("2d")?.drawImage(bitmap, 0, 0);
      }
      bitmap.close();
    }).catch((error) => console.error("[ShowSearch] colormap failed", error));
    return () => { cancelled = true; };
  }, [engine, shown, rows, cols, range[0], range[1], cmap]);
  return <canvas ref={ref} style={{ width, height, display: "block", imageRendering: "pixelated" }} />;
}

// The power spectrum of a projection through the package's display FFT, as ShowReconstruction draws it: mean removed,
// Hann window, padded to a power of two of at least 256, log10 |F|², centred and cropped to ±2.6 1/Å.
type Spectrum = { image: Float32Array; size: number; low: number; high: number };
function useSpectrum(frame: Frame | undefined, pixelA: number): Spectrum | null {
  const [spectrum, setSpectrum] = React.useState<Spectrum | null>(null);
  const fft = React.useRef<DisplayFFT | null>(null);
  const generation = React.useRef(0);
  React.useEffect(() => {
    if (!frame || frame.shape.length !== 2) { setSpectrum(null); return; }
    const mine = ++generation.current;
    const [rows, cols] = frame.shape;
    (async () => {
      const engine = fft.current || (fft.current = await getDisplayFFT());
      const crop = Float32Array.from(frame.data.subarray(0, rows * cols));
      let mean = 0; for (const value of crop) mean += value; mean /= crop.length;
      for (let index = 0; index < crop.length; index++) crop[index] -= mean;
      applyHannWindow2D(crop, cols, rows);
      let n = 256; while (n < Math.max(rows, cols)) n <<= 1;
      const out = await engine.fft2D(zeroPadTopLeft(crop, cols, rows, n, n), new Float32Array(n * n), n, n, false);
      if (mine !== generation.current) return;
      fftshift(out.real, n, n); fftshift(out.imag, n, n);
      const half = Math.max(4, Math.min(n / 2, Math.round(2.6 * n * pixelA)));
      const size = 2 * half, image = new Float32Array(size * size);
      for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) {
        const index = (n / 2 - half + r) * n + (n / 2 - half + c);
        image[r * size + c] = Math.log10(out.real[index] ** 2 + out.imag[index] ** 2 + 1e-12);
      }
      const sorted = Float32Array.from(image).sort();
      setSpectrum({ image, size, low: sorted[Math.floor(0.5 * sorted.length)], high: sorted[Math.floor(0.9995 * (sorted.length - 1))] });
    })().catch((error) => console.error("[ShowSearch] spectrum failed", error));
  }, [frame, pixelA]);
  return spectrum;
}

// ---------------------------------------------------------------------------------------------------------------
// Tiles

// One candidate or stage: its projection edge to edge, the held-out as a badge, its mark in the corner. Hover looks,
// click chooses; a queued candidate is a gray square, a running one shows ● and its epochs.
function Tile({ frame, size, badge, caption, mark, markWhy, running, queued, dim, dashed, active, chosen, colors, why, onHover, onPick }: {
  frame?: Frame; size: number; badge: string; caption?: string; mark?: string; markWhy?: string; running?: string; queued?: boolean; dim?: boolean; dashed?: boolean;
  active?: boolean; chosen?: boolean; colors: any; why: string; onHover: (on: boolean) => void; onPick: () => void;
}) {
  return (
    <div title={why} onMouseEnter={() => onHover(true)} onMouseLeave={() => onHover(false)} onClick={onPick}
      style={{ position: "relative", width: size, height: size, flex: "none", cursor: "pointer", background: queued ? colors.border : "#000", opacity: dim ? 0.45 : 1,
        outline: chosen ? `2px solid ${colors.accent}` : active ? `2px solid ${colors.textMuted}` : dashed ? `1px dashed ${colors.textMuted}` : "none", outlineOffset: -2 }}>
      {frame && frame.shape.length === 2 && <Picture data={frame.data} rows={frame.shape[0]} cols={frame.shape[1]} width={size} height={size} />}
      {caption && <div style={{ ...ON_IMAGE, ...PLATE, left: 2, top: 2, fontSize: 10 }}>{caption}</div>}
      {running && <div style={{ ...ON_IMAGE, ...PLATE, right: 2, top: 2 }}>● {running}</div>}
      <div style={{ ...ON_IMAGE, ...PLATE, left: 2, bottom: 2, fontSize: size < 64 ? 10 : TEXT }} title={markWhy}>{mark && <span style={{ color: mark === "★" ? "#f2c14e" : "#fff" }}>{mark} </span>}{badge}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------------------
// Convergence: held-out against epoch, one line a candidate, of the stage in view

function Curves({ series, hovered, width, height, colors }: {
  series: { name: string; color: string; points: [number, number][] }[]; hovered: string | null; width: number; height: number; colors: any;
}) {
  const left = 52, right = 6, top = 4, bottom = 16;
  const all = series.flatMap((one) => one.points);
  if (all.length === 0) return <div style={{ width, height, fontSize: TEXT, color: colors.textMuted }}>no epochs yet</div>;
  const xs = all.map(([x]) => x);
  const x0 = Math.min(...xs), x1 = Math.max(x0 + 1, ...xs);
  const after = all.filter(([x]) => x > x0).map(([, y]) => y);  // the start's held-out sits far above: the range follows training
  let y0 = Math.min(...all.map(([, y]) => y)), y1 = Math.max(...(after.length ? after : all.map(([, y]) => y)));
  if (y1 - y0 < 1e-9) { y0 -= 1e-4; y1 += 1e-4; }
  const pad = 0.05 * (y1 - y0); y0 -= pad; y1 += pad;
  const sx = (x: number) => left + ((x - x0) / (x1 - x0)) * (width - left - right);
  const sy = (y: number) => top + (1 - (Math.min(y, y1) - y0) / (y1 - y0)) * (height - top - bottom);
  const step = (y1 - y0) / 3, magnitude = 10 ** Math.floor(Math.log10(step));
  const nice = [1, 2, 5, 10].map((value) => value * magnitude).find((value) => value >= step) || step;
  const decimals = Math.min(5, Math.max(0, -Math.floor(Math.log10(nice) + 1e-9)));
  const ticks: number[] = [];
  for (let value = Math.ceil(y0 / nice) * nice; value <= y1; value += nice) ticks.push(value);
  const ordered = [...series].sort((a, b) => (a.name === hovered ? 1 : 0) - (b.name === hovered ? 1 : 0));
  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      {ticks.map((value) => (
        <g key={value}>
          <line x1={left} x2={width - right} y1={sy(value)} y2={sy(value)} stroke={colors.border} strokeWidth={1} />
          <text x={left - 4} y={sy(value) + 4} fontSize={TEXT} fill={colors.textMuted} textAnchor="end">{value.toFixed(decimals)}</text>
        </g>
      ))}
      <text x={left} y={height - 3} fontSize={TEXT} fill={colors.textMuted}>{x0}</text>
      <text x={width - right} y={height - 3} fontSize={TEXT} fill={colors.textMuted} textAnchor="end">epoch {x1}</text>
      {ordered.map((one) => {
        const bold = one.name === hovered;
        const path = one.points.map(([x, y], index) => `${index ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(" ");
        return (
          <g key={one.name} opacity={hovered && !bold ? 0.45 : 1}>
            <path d={path} fill="none" stroke={bold ? colors.accent : one.color} strokeWidth={bold ? 2.2 : 1.2} />
            {one.points.length > 0 && <circle cx={sx(one.points[one.points.length - 1][0])} cy={sy(one.points[one.points.length - 1][1])} r={bold ? 3 : 2} fill={bold ? colors.accent : one.color} />}
          </g>
        );
      })}
    </svg>
  );
}

function useWidth(fallback: number): [React.RefObject<HTMLDivElement | null>, number] {
  const ref = React.useRef<HTMLDivElement>(null);
  const [width, setWidth] = React.useState(fallback);
  React.useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => { const next = Math.round(entry.contentRect.width); if (next > 0) setWidth(next); });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

function Panel({ title, hint, info, children, colors, theme, style }: {
  title: string; hint?: React.ReactNode; info?: React.ReactNode; children: React.ReactNode; colors: any; theme: "light" | "dark"; style?: React.CSSProperties;
}) {
  return (
    <div style={{ borderRadius: 4, padding: `${GAP}px 6px 6px`, minWidth: 0, background: colors.bgAlt, display: "flex", flexDirection: "column", ...style }}>
      <div style={{ fontSize: 12, lineHeight: "17px", marginBottom: 2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
        <b>{title}</b>{hint && <span style={{ color: colors.textMuted, fontSize: TEXT }}> · {hint}</span>}
        {info && <InfoTooltip text={info} theme={theme} toggleOnClick />}
      </div>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------------------
// The view

function ShowSearch() {
  const store = useStore();
  const { colors: base, themeInfo } = useTheme();
  const theme = themeInfo.theme;
  const colors = base as any;
  const engine = useEngine();
  const [title] = useModelState<string>("title");
  const [pixelA] = useModelState<number>("pixel_size_A");
  const [code] = useModelState<string>("code");
  const [rules] = useModelState<string[]>("rules");
  const [status] = useModelState<{ [key: string]: any }>("status");
  const [candidates] = useModelState<Candidate[]>("candidates");
  const [stages] = useModelState<Stage[]>("stages");
  const [picks] = useModelState<Pick[]>("picks");
  const [best] = useModelState<string>("best");
  const [selected, setSelected] = useModelState<string>("selected");
  const [size, setSize] = useModelState<ViewSize>("size");
  const objectLink = useViewLink();  // wheel zoom, drag pan and double-click reset on the shown projection
  const probeLink = useViewLink();  // and on the probe, on its own: a probe never zooms with an object image
  const [root, rootWidth] = useWidth(1000);
  const STRIP_TILE = bySize(size, 52, 64, 92);
  const SHOWN = Math.min(bySize(size, 150, 230, Math.max(230, Math.floor((rootWidth - Math.min(500, rootWidth * 0.46) - 40) / 2.5))), Math.max(110, Math.floor((rootWidth - 40) / 2)));  // the shown projection and probe; at large beside the grid; half the width on a phone
  const SMALL = Math.floor((SHOWN - GAP) / 2);  // spectrum and depth cut stacked beside them
  const [hovered, setHovered] = React.useState<string | null>(null);
  const [curvesBox, curvesWidth] = useWidth(300);

  const groups = React.useMemo(() => groupsOf(candidates || [], store.curves), [candidates, store.version]);
  const byKey = new Map(groups.map((group) => [group.key, group]));
  const shownKey = hovered || selected || best || "";
  const shownGroup = byKey.get(shownKey) || groups.find((group) => group.members.includes(shownKey));
  const shownStage = (stages || []).find((stage) => `stage:${stage.name}` === shownKey);
  const frameName = shownGroup ? shownGroup.landed : shownStage ? `stage:${shownStage.name}` : "";
  const frame = (kind: string) => store.frames.get(`${frameName}/${kind}`);
  const projection = frame("projection"), probe = frame("probe"), section = frame("section");
  const spectrum = useSpectrum(projection, pixelA || 1);

  const geometry = groups.filter((group) => group.row.stage === "geometry");
  const thicknesses = [...new Set(geometry.map((group) => group.row.thickness_nm))].sort((a, b) => a - b);
  const focuses = [...new Set(geometry.map((group) => group.row.focus_nm))].sort((a, b) => a - b);
  const rowsCount = thicknesses.length;  // the grid holds the geometry candidates only: a thicker slab the search added is a later stage, on the stages strip
  const TILE = size === "large" ? Math.max(96, Math.min(170, Math.floor((Math.min(500, rootWidth * 0.46) - 40) / Math.max(1, focuses.length))))  // large: the grid about 500 px, the shown panel beside it
    : bySize(size, 56, 76, 108);
  const tile = rowsCount > 3 ? Math.round(TILE * 0.8) : TILE;  // smaller for many thicknesses, so the view keeps its height
  const later = ["slices", "refocus", "refinement"].map((stage) => ({ stage, groups: groups.filter((group) => group.row.stage === stage) })).filter((part) => part.groups.length);
  const running = groups.find((group) => group.row.state === "running");
  const focusStage = shownGroup ? shownGroup.row.stage : running ? running.row.stage : groups.length ? groups[groups.length - 1].row.stage : "geometry";
  const series = groups.filter((group) => group.row.stage === focusStage && group.points.length)
    .map((group) => ({ name: group.key, color: STAGE_COLORS[group.row.stage] || "#8c8c8c", points: group.points }));
  const pick = (picks || []).find((one) => one.stage === focusStage);
  const overall = groups.find((group) => group.members.includes(best || ""));

  const look = (key: string) => (on: boolean) => setHovered(on ? key : null);
  const choose = (key: string) => () => { setSelected(selected === key ? "" : key); };
  const tileOf = (group: (typeof groups)[number], size: number, label?: string) => {
    const row = group.row;
    const [mark, markWhy] = MARKS[row.mark] || ["", ""];
    const epochs = group.points.length ? group.points[group.points.length - 1][0] : 0;
    return (
      <Tile key={group.key} frame={store.frames.get(`${group.landed}/projection`)} size={size} colors={colors}
        badge={plain(row.held_out)} caption={label} mark={mark} markWhy={markWhy}
        running={row.state === "running" ? `${epochs}/${group.epochs}` : undefined} queued={row.state === "queued"}
        active={hovered === group.key} chosen={selected === group.key}
        why={`${row.thickness_nm} nm · focus ${signed(row.focus_nm, 1)} nm · ${row.slices} slices · ${row.modes} modes · ${group.epochs} epochs · held-out ${plain(row.held_out)}`}
        onHover={look(group.key)} onPick={choose(group.key)} />
    );
  };
  const stageTile = (stage: Stage, size: number) => {
    const delta = stage.held_out != null && stage.start != null ? stage.held_out - stage.start : null;
    const key = `stage:${stage.name}`;
    const kept = stage.verdict === "kept";
    return (
      <Tile key={key} frame={store.frames.get(`${key}/projection`)} size={size} colors={colors} dim={!kept && stage.verdict !== "running"}
        badge={stage.verdict === "running" ? "" : delta == null ? plain(stage.held_out) : `${delta <= 0 ? "▼" : "▲"}${plain(Math.abs(delta))}`}
        running={stage.verdict === "running" ? "" : undefined} mark={stage.flag ? "⚠" : ""} markWhy={stage.flag}
        active={hovered === key} chosen={selected === key}
        why={`${stage.name}: ${stage.verdict} · held-out ${plain(stage.held_out)} against ${plain(stage.start)} at its start and ${plain(stage.control)} for a plain control${stage.flag ? ` · ${stage.flag}` : ""}`}
        onHover={look(key)} onPick={choose(key)} />
    );
  };

  const done = status?.done, total = status?.total;
  const headline = [
    status?.stage ? `${status.stage}${total ? ` ${done}/${total}` : ""}` : "",
    clock(status?.elapsed_s),
    status?.left_s != null ? `about ${Math.max(1, Math.round(status.left_s / 60))} min left` : "",
  ].filter(Boolean).join(" · ");
  const bestRow = overall?.row;
  const lastKept = [...(stages || [])].reverse().find((stage) => stage.verdict === "kept");
  const chosenGroup = byKey.get(selected || "");
  const codeLine = chosenGroup && code ? code.replace("{name}", chosenGroup.landed) : "";

  return (
    <Engine.Provider value={engine}>
      <div ref={root} style={{ fontFamily: WIDGET_UI_FONT, fontSize: 12, color: colors.text, background: colors.bg, border: `1px solid ${colors.border}`,
        padding: GAP, borderRadius: 4, maxWidth: 1100, boxSizing: "border-box", fontVariantNumeric: "tabular-nums" }}>
        <div style={{ lineHeight: "18px", overflowWrap: "anywhere" }}>
          <span style={{ float: "right", marginLeft: GAP }}><SizeControl size={size} onSize={setSize} colors={colors} /></span>
          <b>{title || "Search"}</b>{rules && rules.length > 0 && <InfoTooltip text={<div>{rules.map((rule) => <div key={rule}>{rule}</div>)}</div>} theme={theme} toggleOnClick />}{headline && <span> · {headline}</span>}{status?.note && <span style={{ color: colors.textMuted }}> · {status.note}</span>}
        </div>
        <div style={{ lineHeight: "18px", overflowWrap: "anywhere", marginBottom: GAP }}>
          {bestRow ? <>best after the {bestRow.stage} pass: {bestRow.thickness_nm} nm · focus {signed(bestRow.focus_nm, 1)} nm · {bestRow.slices} slices · {bestRow.modes} modes · held-out <b>{plain(bestRow.held_out)}</b> ★</>
            : <span style={{ color: colors.textMuted }}>no candidate landed yet</span>}
          {lastKept && <span> · after {lastKept.name} <b>{plain(lastKept.held_out)}</b></span>}
          {(picks || []).flatMap((one) => one.flags).includes("search_boundary") && <span style={{ color: colors.textMuted }}> · {bestRow && thicknesses.length && bestRow.thickness_nm <= thicknesses[0] ? "at the bottom of the thickness range" : "at the top of the thickness range"}</span>}
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: GAP, alignItems: "stretch" }}>
          <Panel title="candidates" hint="thickness × focus, held-out" colors={colors} theme={theme} style={{ flex: "none" }}
            info="Each tile is one geometry candidate's projected phase at its last epoch, badge its held-out (lower is better). ★ the pick of its stage, ▲ within 2 standard errors of it, ● running with its epochs, gray queued: every tile trained the same epochs, so their scores compare (a thicker slab the search tried later is on the stages strip). Hover a tile to see it; click to choose it.">
            <div style={{ display: "grid", gridTemplateColumns: `40px repeat(${Math.max(1, focuses.length)}, ${tile}px)`, alignItems: "center", fontSize: TEXT }}>
              <div style={{ color: colors.textMuted }}>focus</div>
              {focuses.map((focus) => <div key={focus} style={{ textAlign: "center", color: colors.textMuted }}>{signed(focus, 1)} nm</div>)}
              {thicknesses.map((thickness) => (
                <React.Fragment key={thickness}>
                  <div style={{ color: colors.textMuted }}>{thickness} nm</div>
                  {focuses.map((focus) => {
                    const group = geometry.find((one) => one.row.thickness_nm === thickness && one.row.focus_nm === focus);
                    return group ? tileOf(group, tile) : <div key={focus} />;
                  })}
                </React.Fragment>
              ))}
            </div>
          </Panel>
          <Panel title="shown" hint={shownGroup ? `${shownGroup.row.thickness_nm} nm, focus ${signed(shownGroup.row.focus_nm, 1)} nm${hovered ? "" : selected ? " (chosen)" : " (best candidate)"}` : shownStage ? shownStage.name : "hover a tile"} colors={colors} theme={theme} style={{ flex: "1 1 auto", minWidth: 0, maxWidth: "100%" }}
            info="The hovered tile, else the chosen one, else the best: its projected phase, its spectrum (log power, ±2.6 1/Å), its strongest probe mode at focus and the depth cut along its line (slices down).">
            <div style={{ display: "flex", flexWrap: "wrap", gap: GAP }}>
              <div style={{ width: SHOWN, height: SHOWN, background: "#000", position: "relative" }}>
                {projection && projection.shape.length === 2 && <ViewLink.Provider value={objectLink}><ZoomFrame><Picture data={projection.data} rows={projection.shape[0]} cols={projection.shape[1]} width={SHOWN} height={SHOWN} /></ZoomFrame></ViewLink.Provider>}
                {projection && shownGroup && pixelA > 0 && (() => {  // a stage's field has its own pixel: no bar

                  const barPx = (10 / pixelA) * (SHOWN / projection.shape[1]);
                  return barPx < SHOWN * 0.6 ? <><div style={{ position: "absolute", left: 8, bottom: 8, width: barPx, height: 3, background: "#fff" }} /><div style={{ ...ON_IMAGE, left: 8, bottom: 13 }}>1 nm</div></> : null;
                })()}
              </div>
              {(() => {  // the strongest probe mode: complex (2, rows, cols) in complex-wave colors, hue the phase and brightness the amplitude; an intensity as its amplitude
                const [rows, cols] = probe ? probe.shape.slice(-2) : [1, 1];
                const width = Math.round(SHOWN * cols / rows);
                const plane = rows * cols;
                return <div style={{ width, height: SHOWN, background: "#000", position: "relative" }}>
                  <ViewLink.Provider value={probeLink}><ZoomFrame>
                    {probe && probe.shape.length === 3 && <ComplexCanvas re={probe.data.subarray(0, plane)} im={probe.data.subarray(plane, 2 * plane)} rows={rows} cols={cols} width={width} height={SHOWN} />}
                    {probe && probe.shape.length === 2 && <Picture data={probe.data} rows={rows} cols={cols} width={width} height={SHOWN} cmap="magma" low={0} transform={Math.sqrt} high={undefined} />}
                  </ZoomFrame></ViewLink.Provider>
                  <div style={{ ...ON_IMAGE, left: 3, bottom: 1 }}>probe, strongest mode{probe && probe.shape.length === 3 ? "" : ", amplitude"}</div>
                </div>;
              })()}
              <div style={{ display: "flex", flexDirection: "column", gap: GAP }}>
                {[["spectrum", spectrum && <Picture data={spectrum.image} rows={spectrum.size} cols={spectrum.size} width={SMALL} height={SMALL} low={spectrum.low} high={spectrum.high} />],
                  ["depth", section && section.shape.length === 2 && <Picture data={section.data} rows={section.shape[0]} cols={section.shape[1]} width={SMALL} height={SMALL} />]]
                  .map(([label, content]) => (
                    <div key={label as string} style={{ width: SMALL, height: SMALL, background: "#000", position: "relative" }}>
                      {content}<div style={{ ...ON_IMAGE, left: 2, bottom: 0, fontSize: 10 }}>{label}</div>
                    </div>
                  ))}
              </div>
            </div>
            <div style={{ fontSize: TEXT, color: colors.textMuted, marginTop: 2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", maxWidth: 2 * SHOWN + SMALL + 2 * GAP }}>
              {shownGroup ? `${shownGroup.row.thickness_nm} nm · ${signed(shownGroup.row.focus_nm, 1)} nm · ${shownGroup.row.slices} slices · ${shownGroup.row.modes} modes · ${shownGroup.epochs} epochs`
                : shownStage ? `${shownStage.verdict} · start ${plain(shownStage.start)} · control ${plain(shownStage.control)}` : ""}
            </div>
          </Panel>
          <Panel title="held-out" hint={`${focusStage}, each candidate against epoch`} colors={colors} theme={theme} style={{ flex: "1 1 260px" }}
            info="Each candidate's held-out Poisson (the patterns it never trains on) at every epoch: lower is better, flat is converged. The y range follows training; the start's value sits above it. The hovered candidate is bold. Colors: geometry gray, slices blue, refocus tan.">
            <div ref={curvesBox} style={{ minWidth: 0 }}>
              <Curves series={series} hovered={shownGroup ? shownGroup.key : null} width={curvesWidth} height={Math.min(SHOWN - 18, bySize(size, 90, 110, 130))} colors={colors} />
            </div>
            <div style={{ fontSize: TEXT, lineHeight: "15px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {pick ? <>★ best of this pass <b>{plain(pick.held_out)}</b>{pick.paired_SE != null && <> · standard error {plain(pick.paired_SE)}</>}
                {groups.filter((group) => group.row.stage === focusStage && group.row.mark === "tie").length > 0 &&
                  <> · ▲ {groups.filter((group) => group.row.stage === focusStage && group.row.mark === "tie").length} within 2 standard errors</>}</>
                : <span style={{ color: colors.textMuted }}>lowest held-out wins; within 2 standard errors both continue</span>}
            </div>
          </Panel>
        </div>
        {(later.length > 0 || (stages || []).length > 0) && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: `${GAP}px ${3 * GAP}px`, marginTop: GAP, background: colors.bgAlt, borderRadius: 4, padding: `${GAP}px 6px 6px` }}>
            {later.map((part) => (
              <div key={part.stage}>
                <div style={{ fontSize: TEXT, color: colors.textMuted, lineHeight: "15px" }}><b style={{ color: colors.text }}>{part.stage}</b> · {part.stage === "refocus" ? "focus, slices (sl)" : "slices"} · held-out</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 2 }}>{part.groups.map((group) => tileOf(group, STRIP_TILE, part.stage === "refocus" ? `${signed(group.row.focus_nm, 1)} nm, ${group.row.slices} sl` : `${group.row.slices} slices`))}</div>
              </div>
            ))}
            {(stages || []).length > 0 && (
              <div>
                <div style={{ fontSize: TEXT, color: colors.textMuted, lineHeight: "15px" }}><b style={{ color: colors.text }}>stages</b> · change against its start; dim: refused</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
                  {(stages || []).map((stage) => (
                    <div key={stage.name} style={{ display: "flex", flexDirection: "column", width: STRIP_TILE }}>
                      {stageTile(stage, STRIP_TILE)}
                      <div style={{ fontSize: 10, lineHeight: "12px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={stage.name}>{stage.name}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        <div style={{ fontSize: TEXT, color: colors.textMuted, lineHeight: "16px", marginTop: 2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {codeLine ? <>chosen: <code style={{ color: colors.text, userSelect: "all" }}>{codeLine}</code></> : "hover a tile to see it, click to choose it; leaving goes back"}
        </div>
      </div>
    </Engine.Provider>
  );
}

export default { render: createRender(ShowSearch) };
