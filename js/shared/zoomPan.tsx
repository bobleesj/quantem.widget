// Show2D's panel interactions for the reconstruction views: wheel zoom about the pointer, drag to pan once zoomed,
// double-click to reset, one zoom shared by the images of a view (before and after, compare's columns: one field),
// and the view's display settings (contrast percentiles, colormap) behind its "controls" toggle. Display only: the
// arrays are never changed, and a saved page draws the same pictures.
import * as React from "react";

export type Zoom = { scale: number; x: number; y: number };  // scale ≥ 1; (x, y) the shown centre as a fraction of the field
export const NO_ZOOM: Zoom = { scale: 1, x: 0.5, y: 0.5 };
export type Display = { low: number; high: number; cmap: string | null; fft: number };  // contrast as percentiles of each image; a colormap for the grey images, null keeps them grey; the spectrum inset's side as a percent of its image's (0 hides it)
export const DEFAULT_DISPLAY: Display = { low: 1, high: 99.8, cmap: null, fft: 35 };

export const ViewLink = React.createContext<{ zoom: Zoom; setZoom: (zoom: Zoom) => void; display: Display } | null>(null);

/** One view's shared zoom and display settings, for ViewLink.Provider. */
export function useViewLink(display: Display = DEFAULT_DISPLAY) {
  const [zoom, setZoom] = React.useState<Zoom>(NO_ZOOM);
  return React.useMemo(() => ({ zoom, setZoom, display }), [zoom, display]);
}

function clamp(zoom: Zoom): Zoom {
  const scale = Math.min(16, Math.max(1, zoom.scale)), half = 0.5 / scale;
  return { scale, x: Math.min(1 - half, Math.max(half, zoom.x)), y: Math.min(1 - half, Math.max(half, zoom.y)) };
}

/** A frame whose content zooms and pans with the view's shared zoom; onHover gets the pointer's place in the field
 *  (fractions 0 to 1), null when it leaves. Without a ViewLink it draws its content as it is. */
export function ZoomFrame({ children, onHover, style }: { children: React.ReactNode; onHover?: (place: [number, number] | null) => void; style?: React.CSSProperties }) {
  const link = React.useContext(ViewLink);
  const box = React.useRef<HTMLDivElement>(null);
  const drag = React.useRef<{ x: number; y: number; zoom: Zoom } | null>(null);
  const zoom = link ? link.zoom : NO_ZOOM;
  const latest = React.useRef(zoom);
  latest.current = zoom;
  const fraction = (clientX: number, clientY: number): [number, number] => {
    const rect = box.current!.getBoundingClientRect();
    return [(clientX - rect.left) / rect.width, (clientY - rect.top) / rect.height];
  };
  React.useEffect(() => {  // a native listener: the page must not scroll while the wheel zooms the image
    const element = box.current;
    if (!element || !link) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const [fx, fy] = fraction(event.clientX, event.clientY), now = latest.current;
      const scale = Math.min(16, Math.max(1, now.scale * (event.deltaY < 0 ? 1.2 : 1 / 1.2)));
      const u = now.x + (fx - 0.5) / now.scale, v = now.y + (fy - 0.5) / now.scale;  // the field point under the pointer stays under it
      link.setZoom(clamp({ scale, x: u - (fx - 0.5) / scale, y: v - (fy - 0.5) / scale }));
    };
    element.addEventListener("wheel", onWheel, { passive: false });
    return () => element.removeEventListener("wheel", onWheel);
  }, [link?.setZoom]);
  const s = zoom.scale;
  return (
    <div ref={box} style={{ position: "absolute", inset: 0, overflow: "hidden", ...style }}
      onPointerDown={(event) => {
        if (!link || s <= 1 || event.button !== 0) return;
        event.stopPropagation(); event.preventDefault();
        (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
        drag.current = { x: event.clientX, y: event.clientY, zoom };
      }}
      onPointerMove={(event) => {
        if (onHover) { const [fx, fy] = fraction(event.clientX, event.clientY); onHover([zoom.x + (fx - 0.5) / s, zoom.y + (fy - 0.5) / s]); }
        const held = drag.current;
        if (!held || !link) return;
        const rect = box.current!.getBoundingClientRect();
        link.setZoom(clamp({ ...held.zoom, x: held.zoom.x - (event.clientX - held.x) / (rect.width * s), y: held.zoom.y - (event.clientY - held.y) / (rect.height * s) }));
      }}
      onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}
      onPointerLeave={() => onHover && onHover(null)}
      onDoubleClick={(event) => { if (link && s > 1) { event.stopPropagation(); link.setZoom(NO_ZOOM); } }}>
      <div style={{ position: "absolute", inset: 0, transformOrigin: "0 0",
        transform: s > 1 ? `translate(${(0.5 - zoom.x * s) * 100}%, ${(0.5 - zoom.y * s) * 100}%) scale(${s})` : undefined }}>
        {children}
      </div>
    </div>
  );
}

/** The value of a field at a place given as fractions, and its place in nm when the pixel is known. */
export function readAt(data: Float32Array, rows: number, cols: number, place: [number, number], pixelA?: number) {
  const row = Math.min(rows - 1, Math.max(0, Math.floor(place[1] * rows))), col = Math.min(cols - 1, Math.max(0, Math.floor(place[0] * cols)));
  return { value: data[row * cols + col], row, col, nm: pixelA ? [row * pixelA / 10, col * pixelA / 10] : null };
}

/** A view's "controls": contrast percentiles, colormap of the grey images, Smooth, and the zoom's reset. Hidden until
 *  asked for, so the view's look stays clean. */
export function ViewControls({ display, setDisplay, smooth, setSmooth, zoomed, onReset, colors, maps }: {
  display: Display; setDisplay: (display: Display) => void; smooth: boolean; setSmooth: (on: boolean) => void; zoomed: boolean; onReset: () => void; colors: any; maps: string[];
}) {
  const field = (label: string, value: number, onChange: (value: number) => void, low: number, high: number) => (
    <label style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>{label}
      <input type="range" min={low} max={high} step={0.1} value={value} onChange={(event) => onChange(Number(event.target.value))} style={{ width: 90 }} />
      <span style={{ minWidth: 34 }}>{value.toFixed(1)}</span></label>
  );
  return (
    <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "4px 12px", fontSize: 11, color: colors.textMuted, padding: "2px 4px", marginBottom: 4,
      border: `1px solid ${colors.border}`, borderRadius: 4 }}>
      <span>contrast, percentiles</span>
      {field("low", display.low, (low) => setDisplay({ ...display, low: Math.min(low, display.high - 0.5) }), 0, 50)}
      {field("high", display.high, (high) => setDisplay({ ...display, high: Math.max(high, display.low + 0.5) }), 50, 100)}
      <label style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>colormap
        <select value={display.cmap || "gray"} onChange={(event) => setDisplay({ ...display, cmap: event.target.value === "gray" ? null : event.target.value })} style={{ fontSize: 11 }}>
          {maps.map((name) => <option key={name} value={name}>{name}</option>)}
        </select></label>
      <label style={{ display: "inline-flex", alignItems: "center", gap: 4 }}><input type="checkbox" checked={smooth} onChange={() => setSmooth(!smooth)} />smooth</label>
      <button type="button" disabled={!zoomed} onClick={onReset} style={{ fontSize: 11 }}>reset zoom</button>
      <button type="button" onClick={() => setDisplay({ ...DEFAULT_DISPLAY, fft: display.fft })} style={{ fontSize: 11 }}>reset contrast</button>
      <label style={{ display: "inline-flex", alignItems: "center", gap: 4 }} title="the spectrum inset's side, as a percent of its image's; 0 hides it">FFT size
        <input type="range" min={0} max={60} step={5} value={display.fft} onChange={(event) => setDisplay({ ...display, fft: Number(event.target.value) })} style={{ width: 70 }} />
        <span style={{ minWidth: 28 }}>{display.fft} %</span></label>
      <span>wheel zooms · drag pans · double-click resets · drag the FFT</span>
    </div>
  );
}

/** The "controls" toggle a view's header carries. */
export function ControlsToggle({ open, onToggle, colors }: { open: boolean; onToggle: () => void; colors: any }) {
  return (
    <button type="button" aria-pressed={open} onClick={onToggle} title="contrast, colormap, smooth and the zoom"
      style={{ border: `1px solid ${colors.border}`, background: open ? colors.accent : "transparent", color: open ? "#fff" : colors.textMuted,
        fontSize: 10, lineHeight: "16px", padding: "0 6px", borderRadius: 3, cursor: "pointer", flex: "none" }}>controls</button>
  );
}

/** The view's contrast over images shown together: percentiles (low, high) of all of them pooled, so before and after
 *  keep one range; a change map (symmetric) takes ± the high percentile of |value|, centred on zero. */
export function contrastRange(sources: Float32Array[], display: Display, symmetric = false): [number, number] {
  const sample: number[] = [];
  for (const source of sources) { const step = Math.max(1, Math.floor(source.length / 50000)); for (let index = 0; index < source.length; index += step) sample.push(symmetric ? Math.abs(source[index]) : source[index]); }
  if (!sample.length) return [0, 1];
  sample.sort((a, b) => a - b);
  const at = (percent: number) => sample[Math.floor((percent / 100) * (sample.length - 1))];
  if (symmetric) { const span = at(display.high) || 1e-12; return [-span, span]; }
  return [at(display.low), at(display.high)];
}

/** A small picture inside its image's corner (a spectrum in a projection): bottom right until dragged, kept inside the
 *  image, double-click sends it back to the corner. It sits outside the zoom, so it stays put while the field zooms,
 *  and it takes its own pointer events, so a drag moves it and never the line or the region under it. */
export function CornerInset({ size, title, children }: { size: number; title?: string; children: React.ReactNode }) {
  const [offset, setOffset] = React.useState<[number, number]>([0, 0]);  // px in from the right and bottom edges
  const box = React.useRef<HTMLDivElement>(null);
  const held = React.useRef<{ x: number; y: number; offset: [number, number]; width: number; height: number } | null>(null);
  const stop = (event: React.SyntheticEvent) => { event.stopPropagation(); };
  const release = (event: React.PointerEvent) => { if (held.current) event.stopPropagation(); held.current = null; };
  return (
    <div ref={box} title={title}
      style={{ position: "absolute", right: offset[0], bottom: offset[1], width: size, height: size, zIndex: 2, background: "#000", boxShadow: "0 0 0 1px rgba(0,0,0,0.9)",
        cursor: "grab", touchAction: "none", userSelect: "none" }}
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        event.stopPropagation(); event.preventDefault();
        const frame = (box.current!.offsetParent as HTMLElement).getBoundingClientRect();
        (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
        held.current = { x: event.clientX, y: event.clientY, offset, width: frame.width, height: frame.height };
      }}
      onPointerMove={(event) => {
        const now = held.current;
        if (!now) return;
        event.stopPropagation();
        setOffset([Math.max(0, Math.min(now.width - size, now.offset[0] - (event.clientX - now.x))), Math.max(0, Math.min(now.height - size, now.offset[1] - (event.clientY - now.y)))]);
      }}
      onPointerUp={release} onPointerCancel={release} onClick={stop}
      onDoubleClick={(event) => { event.stopPropagation(); setOffset([0, 0]); }}>
      {children}
    </div>
  );
}
