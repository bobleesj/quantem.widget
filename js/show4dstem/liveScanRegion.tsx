import * as React from "react";
import { isLiveFrame } from "./viewLabels";

export type ScanBounds = [number, number, number, number]; // row, col, exclusive row and col
type Pixel = [number, number];
const POINT_MOVE_INTERVAL_MS = 32; // Limit queued comm messages while dragging.
type Command = {
  type: "spectra-live-region";
  enabled: boolean;
  locked: boolean;
  editingLive?: boolean;
  point?: boolean;
  regionSession?: number | null;
  size: number;
  region: ScanBounds | null;
};
type Reply = {
  type: "spectra-live-region-change";
  ready: boolean;
  region: ScanBounds | null;
  size: number;
  reason?: string;
  committed?: true;
  session_id?: number;
};
type Gesture = {
  id: number;
  session?: number;
  kind: "draw" | "resize" | "move" | "point";
  lastSent?: { time: number; bounds: ScanBounds };
  anchor: Pixel;
  offset: Pixel;
  initial: ScanBounds;
};

function validBounds(value: unknown, size: number): value is ScanBounds {
  return (
    Array.isArray(value) &&
    value.length === 4 &&
    value.every((n) => Number.isInteger(n) && n >= 0 && n <= size) &&
    value[0] < value[2] &&
    value[1] < value[3]
  );
}

function defaultBounds(size: number, center: Pixel = [size / 2, size / 2]): ScanBounds {
  const width = Math.max(1, Math.floor(size / 8));
  const start = (at: number) => Math.max(0, Math.min(size - width, Math.floor(at - width / 2)));
  const row = start(center[0]), col = start(center[1]);
  return [row, col, row + width, col + width];
}

/** Rectangle gestures on a scan grid. Both corners of a drawn rectangle are acquired pixels. */
export function scanBoxAt(gesture: Omit<Gesture, "id">, pixel: Pixel, size: number): ScanBounds {
  const [r, c] = pixel,
    [r0, c0, r1, c1] = gesture.initial;
  if (gesture.kind === "point") return [r, c, r + 1, c + 1];
  if (gesture.kind === "draw") {
    const [a, b] = gesture.anchor;
    return [Math.min(a, r), Math.min(b, c), Math.max(a, r) + 1, Math.max(b, c) + 1];
  }
  if (gesture.kind === "resize") return [r0, c0, Math.max(r0, r) + 1, Math.max(c0, c) + 1];
  const h = r1 - r0,
    w = c1 - c0;
  const top = Math.min(size - h, Math.max(0, r - gesture.offset[0]));
  const left = Math.min(size - w, Math.max(0, c - gesture.offset[1]));
  return [top, left, top + h, left + w];
}

/** Pixels the normal VI renderer actually drew for this regional frame, in raster order. */
export type RegionImage = {
  canvas: HTMLCanvasElement;
  region: unknown;
  fieldShape: unknown;
  positions: number;
};

type Source = {
  canvas: HTMLCanvasElement | null;
  scan?: string;
  rows: number;
  cols: number;
  frozen: boolean;
  region?: unknown;
};

/** Spectra's acquisition overlay. Ordinary Show4DSTEM views never enable it. The full raster is
 * copied once at its native orientation before regional Live changes the viewer's scan geometry. */
export function LiveScanRegion({
  source,
  onMapSize,
  onImageDraw,
  incomingScan,
}: {
  source: () => Source;
  incomingScan?: string;
  onMapSize: (size: number | null, point?: boolean) => void;
  onImageDraw: React.RefObject<((image: RegionImage) => void) | null>;
}) {
  const layer = React.useRef<HTMLDivElement>(null),
    box = React.useRef<HTMLDivElement>(null);
  const map = React.useRef<HTMLCanvasElement>(null),
    previewTile = React.useRef<HTMLCanvasElement>(null);
  const frame = React.useRef(0),
    gesture = React.useRef<Gesture | null>(null);
  const reference = React.useRef<HTMLCanvasElement | null>(null);
  const active = React.useRef<Command | null>(null),
    send = React.useRef<(value: Reply) => void>(() => {});
  const sourceRef = React.useRef(source);
  sourceRef.current = source;
  const bounds = React.useRef<ScanBounds>([0, 0, 1, 1]);
  const [display, setDisplay] = React.useState<{ size: number; pixels: number; locked: boolean; point: boolean } | null>(null);
  const paintedRegion = React.useRef<ScanBounds | null>(null);
  const paintBox = () => {
    const node = box.current,
      size = active.current?.size;
    if (!node || !size) return;
    const [r, c, endRow, endCol] = bounds.current;
    Object.assign(node.style, {
      top: ((r + (active.current?.point ? 0.5 : 0)) / size) * 100 + "%",
      left: ((c + (active.current?.point ? 0.5 : 0)) / size) * 100 + "%",
      width: active.current?.point ? "14px" : ((endCol - c) / size) * 100 + "%",
      height: active.current?.point ? "14px" : ((endRow - r) / size) * 100 + "%",
    });
    node.dataset.bounds = bounds.current.join(",");
  };
  const clearReference = React.useCallback(() => {
    gesture.current = null;
    active.current = null;
    reference.current = null;
    paintedRegion.current = null;
    setDisplay(null);
    onMapSize(null);
  }, [onMapSize]);

  // A normal Go acquisition replaces the specimen image. A region on the earlier
  // field must not survive into this scan, even when its dimensions match.
  React.useEffect(() => {
    if (!incomingScan || isLiveFrame(incomingScan) || !active.current) return;
    const size = active.current.size;
    clearReference();
    send.current({ type: "spectra-live-region-change", ready: false, size, region: null });
  }, [incomingScan, clearReference]);
  React.useEffect(() => {
    // Notebook twins share a DOM subtree with their controls; the microscope's remote
    // viewer lives inside a separate-origin iframe embedded by the acquisition notebook.
    if (layer.current?.closest(".spectra-gui-saved")) return;
    const scope = layer.current?.closest(".spectra-gui");
    const parentOrigin =
      window.location.ancestorOrigins?.[0] || (document.referrer ? new URL(document.referrer).origin : "");
    const configure = (command: Command, reply: (value: Reply) => void) => {
      if (
        !command ||
        command.type !== "spectra-live-region" ||
        typeof command.enabled !== "boolean" ||
        typeof command.locked !== "boolean" ||
        !Number.isInteger(command.size) ||
        command.size < 1
      )
        return;
      if (!command.enabled) {
        clearReference();
        return;
      }
      send.current = reply;
      const size = command.size;
      if (!reference.current || active.current?.size !== size) {
        const current = sourceRef.current();
        if (
          current.rows !== size ||
          current.cols !== size ||
          !current.canvas ||
          !current.frozen ||
          !isLiveFrame(current.scan) ||
          current.region ||
          current.canvas.width < size ||
          current.canvas.width % size !== 0 ||
          current.canvas.height !== current.canvas.width
        ) {
          reply({
            type: "spectra-live-region-change",
            ready: false,
            size,
            region: null,
            reason: "Run and stop a full Live frame at this size.",
          });
          clearReference();
          return;
        }
        const canvas = document.createElement("canvas");
        canvas.width = canvas.height = current.canvas.width;
        const context = canvas.getContext("2d");
        if (!context) return;
        context.drawImage(current.canvas, 0, 0);
        reference.current = canvas;
        bounds.current = defaultBounds(size);
        paintedRegion.current = null;
      }
      const session = command.editingLive ? command.regionSession : undefined;
      if (
        gesture.current &&
        (command.locked ||
          gesture.current.session !== session ||
          Boolean(command.point) !== Boolean(active.current?.point) ||
          active.current?.size !== size)
      ) {
        bounds.current = gesture.current.initial;
        gesture.current = null;
      }
      if (!gesture.current && validBounds(command.region, size)) bounds.current = command.region;
      if (command.point && !gesture.current) {
        const [r0, c0, r1, c1] = bounds.current;
        const row = Math.floor((r0 + r1) / 2),
          col = Math.floor((c0 + c1) / 2);
        bounds.current = [row, col, row + 1, col + 1];
      } else if (!gesture.current && active.current?.point && !command.point) {
        const [row, col, endRow, endCol] = bounds.current;
        if (endRow === row + 1 && endCol === col + 1) bounds.current = defaultBounds(size, [row, col]);
      }
      active.current = command;
      const pixels = reference.current.width;
      setDisplay((previous) =>
        previous?.size === size &&
        previous.pixels === pixels &&
        previous.locked === command.locked &&
        previous.point === Boolean(command.point)
          ? previous
          : { size, pixels, locked: command.locked, point: Boolean(command.point) },
      );
      onMapSize(size, Boolean(command.point));
      paintBox();
      reply({ type: "spectra-live-region-change", ready: true, size, region: bounds.current });
    };
    const local = (event: Event) =>
      configure((event as CustomEvent).detail, (value) =>
        scope?.dispatchEvent(new CustomEvent(value.type, { detail: value })),
      );
    const embedded = (event: MessageEvent) => {
      if (
        window.parent === window ||
        event.source !== window.parent ||
        !parentOrigin ||
        event.origin !== parentOrigin
      )
        return;
      configure(event.data, (value) => window.parent.postMessage(value, parentOrigin));
    };
    scope?.addEventListener("spectra-live-region", local);
    window.addEventListener("message", embedded);
    return () => {
      scope?.removeEventListener("spectra-live-region", local);
      window.removeEventListener("message", embedded);
      cancelAnimationFrame(frame.current);
      if (active.current)
        send.current({
          type: "spectra-live-region-change",
          ready: false,
          size: active.current.size,
          region: null,
        });
      onMapSize(null);
    };
  }, [onMapSize, clearReference]);

  React.useLayoutEffect(() => {
    if (!display || !map.current || !reference.current) return;
    map.current.getContext("2d")?.drawImage(reference.current, 0, 0);
    paintBox();
  }, [display]);
  React.useEffect(() => {
    if (!display) return;
    // Only the frame that produced the pixels can place them. The requested selection
    // may already be several frames ahead while a scan or transfer is finishing.
    const update = ({ canvas, region: area, fieldShape, positions }: RegionImage) => {
      const tile = previewTile.current;
      if (
        tile &&
        validBounds(area, display.size) &&
        Array.isArray(fieldShape) &&
        fieldShape[0] === display.size &&
        fieldShape[1] === display.size &&
        canvas.width === area[3] - area[1] &&
        canvas.height === area[2] - area[0] &&
        Number.isInteger(positions) &&
        positions > 0 &&
        positions <= canvas.width * canvas.height
      ) {
        const context = tile.getContext("2d");
        if (!context) return;
        if (!paintedRegion.current || area.some((v, i) => paintedRegion.current![i] !== v)) {
          // Moving without resizing still invalidates all pixels at the old location.
          tile.width = canvas.width;
          tile.height = canvas.height;
          paintedRegion.current = [...area];
        }
        const rows = Math.floor(positions / canvas.width);
        const tail = positions % canvas.width;
        if (rows) context.drawImage(canvas, 0, 0, canvas.width, rows, 0, 0, canvas.width, rows);
        if (tail) context.drawImage(canvas, 0, rows, tail, 1, 0, rows, tail, 1);
        Object.assign(tile.style, {
          display: "block",
          top: (area[0] / display.size) * 100 + "%",
          left: (area[1] / display.size) * 100 + "%",
          width: (canvas.width / display.size) * 100 + "%",
          height: (canvas.height / display.size) * 100 + "%",
        });
      }
    };
    onImageDraw.current = update;
    return () => {
      onImageDraw.current = null;
    };
  }, [display, onImageDraw]);

  function pixel(e: React.PointerEvent): Pixel {
    const r = layer.current!.getBoundingClientRect(),
      n = active.current!.size;
    const at = (v: number) => Math.max(0, Math.min(n - 1, Math.floor(v * n)));
    return [at((e.clientY - r.top) / r.height), at((e.clientX - r.left) / r.width)];
  }
  function move(e: React.PointerEvent, final = false) {
    const g = gesture.current;
    if (!g || g.id !== e.pointerId || !active.current) return;
    bounds.current = scanBoxAt(g, pixel(e), active.current.size);
    cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(paintBox);
    if (!final && g.kind === "point" && g.session !== undefined) {
      const now = performance.now();
      const prev = g.lastSent;
      if (
        !prev ||
        (now - prev.time >= POINT_MOVE_INTERVAL_MS &&
          (bounds.current[0] !== prev.bounds[0] || bounds.current[1] !== prev.bounds[1]))
      ) {
        g.lastSent = { time: now, bounds: bounds.current };
        send.current({
          type: "spectra-live-region-change",
          ready: true,
          committed: true,
          size: active.current.size,
          session_id: g.session,
          region: bounds.current,
        });
      }
    }
  }
  function begin(e: React.PointerEvent<HTMLDivElement>) {
    if (!active.current || active.current.locked || e.button !== 0 || gesture.current) return;
    const session = active.current.editingLive ? active.current.regionSession : undefined;
    if (active.current.editingLive && !Number.isSafeInteger(session)) return;
    const [r, c] = pixel(e),
      [a, b, endR, endC] = bounds.current,
      inside = r >= a && r < endR && c >= b && c < endC;
    const handle = (e.target as HTMLElement).dataset.handle;
    gesture.current = {
      id: e.pointerId,
      session: session ?? undefined,
      initial: bounds.current,
      anchor: [r, c],
      kind: active.current.point ? "point" : e.shiftKey ? "draw" : handle ? "resize" : "move",
      offset: inside ? [r - a, c - b] : [Math.floor((endR - a - 1) / 2), Math.floor((endC - b - 1) / 2)],
    };
    e.currentTarget.setPointerCapture(e.pointerId);
    e.preventDefault();
    move(e);
  }
  function finish(e: React.PointerEvent, cancelled = false) {
    const g = gesture.current,
      command = active.current;
    if (!g || g.id !== e.pointerId || !command) return;
    if (cancelled) bounds.current = g.lastSent?.bounds ?? g.initial;
    else move(e, true);
    gesture.current = null;
    paintBox();
    send.current({
      type: "spectra-live-region-change",
      ready: true,
      size: command.size,
      region: bounds.current,
      ...(!cancelled && { committed: true as const, session_id: g.session }),
    });
  }
  return (
    <div
      ref={layer}
      aria-label="Live scan region map"
      hidden={!display}
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 8,
        touchAction: "none",
        cursor: display?.locked ? "default" : "crosshair",
      }}
      onPointerDown={begin}
      onPointerMove={move}
      onPointerUp={(e) => finish(e)}
      onPointerCancel={(e) => finish(e, true)}
      onLostPointerCapture={(e) => finish(e, true)}
    >
      {display && (
        <>
          <canvas
            ref={map}
            width={display.pixels}
            height={display.pixels}
            style={{ width: "100%", height: "100%", imageRendering: "pixelated", display: "block" }}
          />
          <canvas
            ref={previewTile}
            style={{
              position: "absolute",
              display: "none",
              imageRendering: "pixelated",
              pointerEvents: "none",
            }}
          />
          <div
            ref={box}
            aria-label={display.point ? "Selected probe position" : "Selected scan region"}
            style={{
              position: "absolute",
              boxSizing: "border-box",
              borderRadius: display.point ? "50%" : 0,
              transform: display.point ? "translate(-50%, -50%)" : undefined,
              pointerEvents: display.point ? "none" : undefined,
              outline: "2px solid #9ed6ff",
              boxShadow: "inset 0 0 0 1px #245b7e",
              background: "#78acff12",
              cursor: display.locked ? "default" : "grab",
            }}
          >
            {!display.locked && !display.point && (
              <span
                aria-label="Resize scan region"
                data-handle="resize"
                style={{
                  position: "absolute",
                  right: -5,
                  bottom: -5,
                  width: 11,
                  height: 11,
                  boxSizing: "border-box",
                  background: "white",
                  border: "2px solid #176293",
                  cursor: "nwse-resize",
                }}
              />
            )}
          </div>
        </>
      )}
    </div>
  );
}
