// Show3DSlices pointer behavior, mounted in jsdom (no WebGPU, so the slices
// paint on the Canvas2D path). A drag of the oblique cut's handle moves the cut
// once per pointer move, from one document listener bound once.
import { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

vi.mock("../display/colormaps", async (importOriginal) => {
  const original = await importOriginal<typeof import("../display/colormaps")>();
  return {
    ...original,
    createGPUColormapEngine: async () => null,
    renderToOffscreen: (_data: Float32Array, width: number, height: number) => Object.assign(document.createElement("canvas"), { width, height }),
    renderToOffscreenReuse: () => {},
  };
});

const { default: show3dslices } = await import("./index");

const NZ = 4, NY = 12, NX = 16;
const VOLUME = Float32Array.from({ length: NZ * NY * NX }, (_, index) => Math.sin(index / 7) + (index % NX) / NX);

class Model {
  values: Record<string, unknown> = {
    volume_bytes: new DataView(VOLUME.buffer), nx: NX, ny: NY, nz: NZ,
    slice_x: 8, slice_y: 6, slice_z: 2, cmap: "plasma", auto_contrast: true,
    image_vmin_pct: 0, image_vmax_pct: 100, vmin: null, vmax: null, log_scale: false, flip: false,
    dim_labels: ["slice", "row", "col"], panel_count: 1, panel_titles: [], plane_visibility: [true, true],
    pixel_size_axes: [0, 0, 0], slice_alignment: "off", show_controls: true, show_title: true,
    show_crosshair: true, view_state: {}, oblique_angle: 0, oblique_profile_line: [], page_labels: [],
    n_pages: 1, page_idx: 0, panels_per_page: 0, active_panel: 0, z_stretch: 30, fps: 30, loop: false,
  };
  writes: string[] = [];
  listeners = new Map<string, Set<() => void>>();
  get(key: string) { return this.values[key]; }
  set(key: string, value: unknown) {
    this.writes.push(key);
    this.values[key] = value;
    this.listeners.get(`change:${key}`)?.forEach(fn => fn());
  }
  save_changes = vi.fn();
  send = vi.fn();
  on(key: string, fn: () => void) {
    if (!this.listeners.has(key)) this.listeners.set(key, new Set());
    this.listeners.get(key)!.add(fn);
  }
  off(key: string, fn: () => void) { this.listeners.get(key)?.delete(fn); }
}

let el: HTMLDivElement;
let model: Model;
let cleanup: (() => void) | undefined;

async function settle() { await act(async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); }); }
function mount() {
  act(() => { cleanup = show3dslices.render({ model, el } as never) as (() => void) | undefined; });
}
function mouse(target: EventTarget, type: string, clientX: number, clientY: number, buttons = 1) {
  target.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, button: 0, buttons, clientX, clientY }));
}
const slicePanel = () => el.querySelector<HTMLCanvasElement>('canvas[data-quantem-scientific-output="show3dslices-slice-0"]')!;

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.stubGlobal("ImageData", class { constructor(readonly width: number, readonly height: number) {} });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(new Proxy({}, {
    get: (target: Record<string, unknown>, prop: string) => prop in target ? target[prop]
      : prop === "measureText" ? (text: string) => ({ width: text.length * 6 }) : () => undefined,
    set: (target: Record<string, unknown>, prop: string, value: unknown) => { target[prop] = value; return true; },
  }) as never);
  // Every canvas sits at the page origin at its own pixel size, so client and canvas coordinates agree.
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
    const width = this instanceof HTMLCanvasElement ? this.width : 300;
    const height = this instanceof HTMLCanvasElement ? this.height : 10;
    return { left: 0, top: 0, width, height, right: width, bottom: height, x: 0, y: 0, toJSON: () => ({}) };
  });
  el = document.createElement("div");
  document.body.append(el);
  model = new Model();
});
afterEach(async () => {
  act(() => cleanup?.());
  cleanup = undefined;
  await settle();
  el.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("moves the oblique cut once per pointer move of a handle drag", async () => {
  mount();
  await settle();
  // 16 x 12 voxels on a 480 x 360 canvas (30 px per voxel); the default cut runs
  // along row 6 from column 1 to 14, so its start handle sits at (30, 180).
  const canvas = slicePanel();
  act(() => mouse(canvas, "mousedown", 30, 180));
  model.writes.length = 0;
  act(() => mouse(canvas, "mousemove", 90, 120));
  expect(model.writes.filter(key => key === "oblique_profile_line")).toHaveLength(1);
  act(() => mouse(document, "mousemove", 120, 60));
  expect(model.writes.filter(key => key === "oblique_profile_line")).toHaveLength(2);
  act(() => mouse(document, "mouseup", 120, 60, 0));
  await settle();
  // The released handle sits where the pointer let go: column 4, row 2.
  expect(model.get("oblique_profile_line")).toEqual([{ row: 2, col: 4 }, { row: 6, col: 14 }]);
});

it("binds the oblique drag listeners once, not on every render", async () => {
  mount();
  await settle();
  const added = vi.spyOn(document, "addEventListener");
  for (const cmap of ["viridis", "magma", "inferno"]) {
    act(() => model.set("cmap", cmap));
    await settle();
  }
  expect(added.mock.calls.filter(([type]) => type === "mousemove")).toHaveLength(0);
});
