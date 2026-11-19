import { afterEach, expect, it, vi } from "vitest";
import { createPanelPan, isPanelPan, schedulePanelView, zoomPanelAt } from "./panelPan";

const mouse = { pointerId: 5, pointerType: "mouse", isPrimary: true, button: 0, buttons: 1,
  altKey: false, shiftKey: false, clientX: 100, clientY: 125 };

afterEach(() => vi.unstubAllGlobals());

it("keeps plain, Shift, secondary and touch presses outside the mouse pan gesture", () => {
  expect(isPanelPan(mouse)).toBe(false);
  expect(isPanelPan({ ...mouse, altKey: true })).toBe(true);
  expect(isPanelPan(mouse, true)).toBe(true);
  expect(isPanelPan({ ...mouse, altKey: true, shiftKey: true }, true)).toBe(false);
  expect(isPanelPan({ ...mouse, pointerType: "touch" }, true)).toBe(false);
  expect(isPanelPan({ ...mouse, button: 2 }, true)).toBe(false);
});

it("translates a zoomed canvas in image pixels using only its initiating pointer", () => {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 256;
  canvas.getBoundingClientRect = () => new DOMRect(40, 50, 256, 256);
  const view = { zoom: 1, panX: -25, panY: -5, raf: 0 };
  const pan = createPanelPan();
  expect(pan.start(mouse, view, canvas)).toBe(false);
  view.zoom = 2;
  expect(pan.start(mouse, view, canvas)).toBe(true);
  expect(pan.move({ ...mouse, pointerId: 8, clientX: 130 }, view)).toBe(false);
  expect(pan.release({ pointerId: 8 })).toBe(false);
  expect(pan.move({ ...mouse, clientX: 112, clientY: 115 }, view)).toBe(true);
  expect(view).toMatchObject({ zoom: 2, panX: -1, panY: -15 });
  expect(pan.release(mouse)).toBe(true);
  expect(pan.move({ ...mouse, clientX: 150 }, view)).toBe(false);
  expect(pan.start(mouse, view, canvas)).toBe(true);
  expect(pan.move({ ...mouse, buttons: 0, clientX: 150 }, view)).toBe(false);
  expect(pan.active).toBe(false);
});

it("commits the latest pan and zoom together once per frame", () => {
  const request = vi.fn((_paint: FrameRequestCallback) => 1);
  vi.stubGlobal("requestAnimationFrame", request);
  const commit = vi.fn();
  const view = { zoom: 2, panX: -25, panY: -5, raf: 0 };
  schedulePanelView(view, commit);
  view.panX += 20;
  schedulePanelView(view, commit);
  view.zoom = 3;
  schedulePanelView(view, commit);
  expect(request).toHaveBeenCalledTimes(1);
  expect(commit).not.toHaveBeenCalled();
  request.mock.calls[0][0](16.7);
  expect(commit).toHaveBeenCalledExactlyOnceWith({ zoom: 3, panX: -5, panY: -5, raf: 0 });
});

it("pans to each viewport edge without blank bands and reverses immediately", () => {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 256;
  canvas.getBoundingClientRect = () => new DOMRect(0, 0, 256, 256);
  const view = { zoom: 1.5, panX: -40, panY: -80, raf: 0 };
  const pan = createPanelPan();
  expect(pan.start(mouse, view, canvas)).toBe(true);

  pan.move({ ...mouse, clientX: 600, clientY: -300 }, view);
  expect(view).toMatchObject({ panX: 0, panY: -128 });
  pan.move({ ...mouse, clientX: 596, clientY: -296 }, view);
  expect(view).toMatchObject({ panX: -8, panY: -124 });
  pan.move({ ...mouse, clientX: -600, clientY: 300 }, view);
  expect(view).toMatchObject({ panX: -256, panY: 0 });
});

it("keeps the wheel anchor when possible and clips opposite-edge zoom to the image bounds", () => {
  const size = { width: 512, height: 256 };
  const view = { zoom: 1, panX: 0, panY: 0, raf: 0 };
  zoomPanelAt(view, 1.5, 100, 150, size);
  expect(view).toMatchObject({ zoom: 1.5, panX: -50, panY: -75 });

  const topLeft = { zoom: 2, panX: 0, panY: 0, raf: 0 };
  zoomPanelAt(topLeft, 1.8, size.width - 1, size.height - 1, size);
  expect(topLeft).toMatchObject({ zoom: 1.8, panX: 0, panY: 0 });

  const bottomRight = { zoom: 2, panX: -size.width, panY: -size.height, raf: 0 };
  zoomPanelAt(bottomRight, 1.5, 0, 0, size);
  expect(bottomRight).toMatchObject({ zoom: 1.5, panX: -size.width / 2, panY: -size.height / 2 });
  zoomPanelAt(bottomRight, 1, 50, 60, size);
  expect(bottomRight).toMatchObject({ zoom: 1, panX: 0, panY: 0 });
  zoomPanelAt(bottomRight, 0.75, 100, 120, size);
  expect(bottomRight).toMatchObject({ zoom: 0.75, panX: 25, panY: 30 });
});
