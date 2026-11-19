type PanelView = { zoom: number; panX: number; panY: number; raf: number };
type PanelSize = Pick<HTMLCanvasElement, "width" | "height">;
type Pointer = Pick<PointerEvent, "pointerId" | "pointerType" | "button" | "buttons" | "isPrimary" | "altKey" | "shiftKey" | "clientX" | "clientY">;

export function isPanelPan(event: Pointer, tool = false): boolean {
  return event.pointerType === "mouse" && event.isPrimary && event.button === 0 && (event.altKey || tool) && !event.shiftKey;
}

function boundPanelView(view: PanelView, { width, height }: PanelSize): void {
  if (view.zoom < 1) return;
  view.panX = Math.min(0, Math.max(width * (1 - view.zoom), view.panX));
  view.panY = Math.min(0, Math.max(height * (1 - view.zoom), view.panY));
}

/** Follow the wheel anchor until doing so would expose the edge of a zoomed image. */
export function zoomPanelAt(view: PanelView, zoom: number, x: number, y: number, bounds?: PanelSize): void {
  const ratio = zoom / view.zoom;
  view.zoom = zoom;
  view.panX = x - (x - view.panX) * ratio;
  view.panY = y - (y - view.panY) * ratio;
  if (bounds) boundPanelView(view, bounds);
}

/** One pointer pans the zoomed display without changing its scientific selection. */
export function createPanelPan() {
  let drag: { id: number; x: number; y: number; scaleX: number; scaleY: number; canvas: HTMLCanvasElement } | null = null;
  return {
    get active() { return drag !== null; },
    start(event: Pointer, view: PanelView, canvas: HTMLCanvasElement) {
      const rect = canvas.getBoundingClientRect();
      if (drag || view.zoom <= 1 || rect.width <= 0 || rect.height <= 0) return false;
      drag = { id: event.pointerId, x: event.clientX, y: event.clientY,
        scaleX: canvas.width / rect.width, scaleY: canvas.height / rect.height, canvas };
      return true;
    },
    move(event: Pointer, view: PanelView) {
      if (!drag || drag.id !== event.pointerId) return false;
      if (!(event.buttons & 1) || view.zoom < 1) { drag = null; return false; }
      // Accumulate each movement from the last bounded position so reversing at an edge takes effect immediately.
      view.panX += (event.clientX - drag.x) * drag.scaleX;
      view.panY += (event.clientY - drag.y) * drag.scaleY;
      boundPanelView(view, drag.canvas);
      drag.x = event.clientX;
      drag.y = event.clientY;
      return true;
    },
    release(event: Pick<Pointer, "pointerId">) {
      if (!drag || drag.id !== event.pointerId) return false;
      drag = null;
      return true;
    },
  };
}

export function schedulePanelView(view: PanelView, commit: (view: PanelView) => void): void {
  if (view.raf) return;
  view.raf = requestAnimationFrame(() => {
    view.raf = 0;
    commit(view);
  });
}
