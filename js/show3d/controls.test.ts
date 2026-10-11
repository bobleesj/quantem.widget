// Show3D mounted in jsdom (no WebGPU, so the canvas paths run) against a model that records writes.
// The user stories: dragging a control writes its trait once, on release, not once per path or per step.
import { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

const { default: { render: renderShow3D } } = await import("./index");

const SIZE = 16;
const FRAMES = 4;

class TestImageData {
  constructor(readonly data: Uint8ClampedArray, readonly width: number, readonly height: number) {}
}

class Model {
  values: Record<string, unknown>;
  writes: [string, unknown][] = [];
  listeners = new Map<string, Set<() => void>>();
  constructor(overrides: Record<string, unknown> = {}) {
    const stack = Float32Array.from({ length: FRAMES * SIZE * SIZE }, (_, i) => Math.sin(i / 7) + (i % SIZE) / SIZE);
    this.values = {
      n_slices: FRAMES, width: SIZE, height: SIZE, n_panels: 1, slice_idx: 0, labels: [], title: "", dim_label: "Frame",
      _offline_float_stack: new DataView(stack.buffer), data_min: -1, data_max: 2, auto_vmins: [], auto_vmaxs: [],
      cmap: "gray", show_controls: true, controls_collapsed: false, show_title: true, fps: 30, loop: false,
      loop_start: 0, loop_end: -1, percentile_low: 0.5, percentile_high: 99.5, auto_contrast: false, link_contrast: true,
      link_panels: true, panel_cmaps: [], hidden_panels: [], selected_panels: [], panel_order: [], panel_titles: [],
      image_vmin_pct: 0, image_vmax_pct: 100, vmin_per_panel: [], vmax_per_panel: [], roi_list: [], roi_selected_idx: -1,
      profile_line: [], profile_width: 1, frame_rotations: [], image_rotation: 0, bookmarked_frames: [], playback_path: [],
      diff_mode: "off", avg_window: 1, compare_mode: "off", compare_pair: [0, 1], blink_fps: 2, denoise: "none",
      denoise_sigma: 4, denoise_bin: 1, denoise_modes: [], denoise_sigmas: [], denoise_bins: [], denoise_scope: "all",
      denoise_enabled: false, show_denoise: false, frequency_filter: "none", frequency_filter_enabled: false,
      frequency_filter_cutoff: 0.15, frequency_filter_center: 0.3, frequency_filter_width: 0.12, frequency_filter_modes: [],
      frequency_filter_cutoffs: [], frequency_filter_centers: [], frequency_filter_widths: [], frequency_filter_scope: "all",
      show_frequency_filter: false, show_fft: false, fft_layout: "bottom", fft_overlay_size: 0.35, fft_overlay_zoom: 1,
      fft_window: true, pixel_size: 0, pixel_unit: "px", scale_bar_visible: true, show_panel_titles: true, size: 0,
      max_cols: 4, panel_gap: 0, inter_panel_gap_px: 0, n_pages: 1, page_idx: 0, panels_per_page: 0, view_state: {},
      ...overrides,
    };
  }
  get(key: string) { return this.values[key]; }
  set(key: string, value: unknown) {
    this.values[key] = value;
    this.writes.push([key, value]);
    this.listeners.get(`change:${key}`)?.forEach(fn => fn());
  }
  save_changes = vi.fn();
  send = vi.fn();
  on(key: string, fn: () => void) {
    if (!this.listeners.has(key)) this.listeners.set(key, new Set());
    this.listeners.get(key)!.add(fn);
  }
  off(key: string, fn: () => void) { this.listeners.get(key)?.delete(fn); }
  /** Writes of `key` since the last call. */
  written(key: string): unknown[] {
    const values = this.writes.filter(([name]) => name === key).map(([, value]) => value);
    this.writes = this.writes.filter(([name]) => name !== key);
    return values;
  }
}

let el: HTMLDivElement;
let cleanup: (() => void) | undefined;
// jsdom has no pointer capture; the ring's pointer path needs it.
const captured = new Set<number>();
const pointerCapture = {
  setPointerCapture(id: number) { captured.add(id); },
  hasPointerCapture(id: number) { return captured.has(id); },
  releasePointerCapture(id: number) { captured.delete(id); },
};

async function settle() { await act(async () => { for (let i = 0; i < 30; i++) await Promise.resolve(); }); }

async function mount(model: Model) {
  act(() => { cleanup = renderShow3D({ model, el } as never) as (() => void) | undefined; });
  await settle();
}

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.stubGlobal("ImageData", TestImageData);
  vi.stubGlobal("createImageBitmap", (image: TestImageData) => Promise.resolve(Object.assign(image, { close() {} })));
  vi.stubGlobal("requestAnimationFrame", () => 0);
  vi.stubGlobal("cancelAnimationFrame", () => undefined);
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(new Proxy({}, {
    get: (target: Record<string, unknown>, prop: string) => prop in target ? target[prop]
      : prop === "measureText" ? (text: string) => ({ width: text.length * 6 })
        : prop === "getImageData" || prop === "createImageData" ? (...args: number[]) => new TestImageData(new Uint8ClampedArray(args[args.length - 2] * args[args.length - 1] * 4), args[args.length - 2], args[args.length - 1])
          : () => undefined,
    set: (target: Record<string, unknown>, prop: string, value: unknown) => { target[prop] = value; return true; },
  }) as never);
  Object.assign(Element.prototype, pointerCapture);
  captured.clear();
  el = document.createElement("div");
  document.body.append(el);
});
afterEach(async () => {
  act(() => cleanup?.());
  cleanup = undefined;
  await settle();
  el.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  for (const name of Object.keys(pointerCapture)) delete (Element.prototype as unknown as Record<string, unknown>)[name];
});

/**
 * A mouse drag as a browser delivers it: each pointer event, then its compatibility mouse event unless the
 * pointerdown was cancelled (the Pointer Events rule that keeps one gesture from running two handlers).
 */
function mouseDrag(target: Element, points: [number, number][]) {
  const fire = (type: string, [clientX, clientY]: [number, number]) => {
    const pointer = new PointerEvent(type, { bubbles: true, cancelable: true, clientX, clientY, pointerId: 1, button: 0, buttons: type === "pointerup" ? 0 : 1 });
    act(() => { target.dispatchEvent(pointer); });
    return pointer;
  };
  let compatible = true;
  const mouse = (type: string, [clientX, clientY]: [number, number], mouseTarget: EventTarget) => {
    if (compatible) act(() => { mouseTarget.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, clientX, clientY, button: 0 })); });
  };
  compatible = !fire("pointerdown", points[0]).defaultPrevented;
  mouse("mousedown", points[0], target);
  for (const point of points.slice(1, -1)) {
    fire("pointermove", point);
    mouse("mousemove", point, document);
  }
  fire("pointerup", points[points.length - 1]);
  mouse("mouseup", points[points.length - 1], document);
}

it("commits one frequency cutoff, where the ring is released, for a mouse drag of the FFT ring", async () => {
  const model = new Model({ show_fft: true, frequency_filter: "lowpass", frequency_filter_enabled: true, frequency_filter_cutoff: 0.3 });
  await mount(model);
  const ring = el.querySelector(".quantem-frequency-filter-ring")!;
  vi.spyOn(ring.parentElement!, "getBoundingClientRect").mockReturnValue({ left: 0, top: 0, width: 200, height: 200 } as DOMRect);
  model.writes = [];
  // the ring's radius is the cutoff as a fraction of Nyquist, the half side of the FFT (100 px)
  mouseDrag(ring, [[130, 100], [150, 100], [160, 100], [170, 100]]);
  await settle();
  expect(model.written("frequency_filter_cutoff")).toEqual([0.7]);
});

/** A mouse drag of the MUI slider whose hidden input has `label`, on a track 0 to 100 px wide. */
function dragSlider(label: string, positions: number[], betweenMoves: () => void = () => {}) {
  const root = el.querySelector(`input[aria-label="${label}"]`)!.closest(".MuiSlider-root")!;
  vi.spyOn(root, "getBoundingClientRect").mockReturnValue({ left: 0, width: 100, top: 0, height: 10, bottom: 10, right: 100 } as DOMRect);
  const mouse = (type: string, clientX: number, target: EventTarget) => act(() => {
    target.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, clientX, clientY: 5, button: 0, buttons: type === "mouseup" ? 0 : 1 }));
  });
  mouse("mousedown", positions[0], root);
  for (const position of positions.slice(1)) {
    mouse("mousemove", position, document);
    betweenMoves();
  }
  mouse("mouseup", positions[positions.length - 1], document);
}

it("commits the band-pass width once, on release, not at every drag step", async () => {
  const model = new Model({ frequency_filter: "bandpass", frequency_filter_enabled: true, show_frequency_filter: true });
  await mount(model);
  model.writes = [];
  dragSlider("Band-pass width as fraction of Nyquist", [20, 30, 40, 60]);
  await settle();
  // 60 % of the 0.01 to 1 track, on the 0.005 step
  expect(model.written("frequency_filter_width")).toEqual([0.605]);
});

it("keeps the current frame while a loop end is dragged", async () => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  const model = new Model({ n_slices: FRAMES, loop: true, loop_start: 0, loop_end: 3, slice_idx: 2 });
  await mount(model);
  model.writes = [];
  // the start thumb (frame 0) moves to frame 1; the current frame (2) must not be re-sent mid-drag
  const root = el.querySelector('input[data-index="0"]')!.closest(".MuiSlider-root")!;
  vi.spyOn(root, "getBoundingClientRect").mockReturnValue({ left: 0, width: 90, top: 0, height: 10, bottom: 10, right: 90 } as DOMRect);
  const mouse = (type: string, clientX: number, target: EventTarget) => act(() => {
    target.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, clientX, clientY: 5, button: 0, buttons: 1 }));
  });
  mouse("mousedown", 1, root);
  mouse("mousemove", 32, document);
  act(() => { vi.advanceTimersByTime(1000); });
  expect(model.written("loop_start")).toEqual([1]);
  expect(model.written("slice_idx")).toEqual([]);
  vi.useRealTimers();
});

it("marks More as active while denoise is on with its settings row closed", async () => {
  const model = new Model({ denoise: "gaussian", denoise_enabled: true, show_denoise: false });
  await mount(model);
  const more = el.querySelector('button[aria-label="More tools"]')!;
  const badge = more.parentElement!.querySelector(".MuiBadge-badge")!;
  expect(badge.textContent).toBe("1");
  expect(badge.classList.contains("MuiBadge-invisible")).toBe(false);
});
