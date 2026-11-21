import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import { LiveScanRegion, type RegionImage, type ScanBounds } from "./liveScanRegion";

describe("Live scan region", () => {
  it.each([1, 3])("captures a stopped %s× full image and keeps scan coordinates when editing", (sampling) => {
    const gui = document.createElement("div");
    gui.className = "spectra-gui";
    document.body.append(gui);
    const mount = gui.appendChild(document.createElement("div"));
    const root = createRoot(mount);
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 64 * sampling;
    const image: {
      canvas: HTMLCanvasElement;
      scan: string;
      rows: number;
      cols: number;
      frozen: boolean;
      region?: ScanBounds;
    } = { canvas, scan: "live_1", rows: 64, cols: 64, frozen: false };
    const onImageDraw: { current: ((image: RegionImage) => void) | null } = { current: null };
    const copy = vi.fn();
    const context = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation((() => ({
      drawImage: copy,
    })) as unknown as typeof HTMLCanvasElement.prototype.getContext);
    const replies: { ready: boolean; committed?: true; region: ScanBounds | null }[] = [];
    gui.addEventListener("spectra-live-region-change", (event) => replies.push((event as CustomEvent).detail));
    const command = (
      locked = false,
      region: ScanBounds | null = null,
      enabled = true,
      editingLive = false,
      regionSession = 7,
      point = false,
    ) =>
      act(() => {
        gui.dispatchEvent(
          new CustomEvent("spectra-live-region", {
            detail: {
              type: "spectra-live-region",
              enabled,
              size: 64,
              locked,
              region,
              editingLive,
              regionSession,
              point,
            },
          }),
        );
      });
    const send = (target: Element, type: string, row: number, col: number, shiftKey = false) =>
      act(() => {
        target.dispatchEvent(
          new PointerEvent(type, {
            pointerId: 1,
            button: 0,
            bubbles: true,
            clientX: col + 0.5,
            clientY: row + 0.5,
            shiftKey,
          }),
        );
      });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    try {
      const onMapSize = vi.fn();
      const render = () =>
        act(() =>
          root.render(
            createElement(LiveScanRegion, {
              source: () => image,
              incomingScan: image.scan,
              onMapSize,
              onImageDraw,
            }),
          ),
        );
      render();
      command();
      expect(replies.at(-1)?.ready).toBe(false);
      image.frozen = true;
      command();
      const map = gui.querySelector<HTMLElement>('[aria-label="Live scan region map"]')!;
      map.setPointerCapture = vi.fn();
      map.getBoundingClientRect = () => ({ left: 0, top: 0, width: 64, height: 64 }) as DOMRect;
      expect(map.hidden).toBe(false);
      expect(copy.mock.calls[0][0]).toBe(canvas);
      expect(map.querySelector("canvas")?.width).toBe(64 * sampling);
      // Reverse-direction Shift-drag includes both end pixels.
      send(map, "pointerdown", 22, 35, true);
      send(map, "pointermove", 11, 13, true);
      send(map, "pointerup", 11, 13, true);
      expect(replies.at(-1)?.region).toEqual([11, 13, 23, 36]);
      send(map, "pointerdown", 12, 15);
      send(map, "pointerup", 18, 22);
      expect(replies.at(-1)?.region).toEqual([17, 20, 29, 43]);
      const handle = gui.querySelector('[aria-label="Resize scan region"]')!;
      send(handle, "pointerdown", 28, 42);
      send(map, "pointerup", 31, 49);
      const area: ScanBounds = [17, 20, 32, 50];
      expect(replies.at(-1)?.region).toEqual(area);
      // Starting Live leaves the selection fixed. A wrong acquisition cannot paint the map.
      command(true, area);
      send(map, "pointerdown", 4, 4, true);
      send(map, "pointerup", 10, 10, true);
      expect(replies.at(-1)?.region).toEqual(area);
      const tile = map.querySelectorAll("canvas")[1];
      image.rows = canvas.height = 15;
      image.cols = canvas.width = 30;
      const paint = (region: ScanBounds, positions = canvas.width * canvas.height, fieldShape = [64, 64]) =>
        onImageDraw.current!({ canvas, region, positions, fieldShape });
      paint([0, 0, 16, 30]);
      expect(tile.style.display).toBe("none");
      image.region = area;
      paint(area);
      expect(tile.width).toBe(30);
      expect(tile.height).toBe(15);
      expect(tile.style.top).toBe("26.5625%");
      expect(tile.style.left).toBe("31.25%");
      command(false, area, true, true);
      expect(replies.at(-1)?.committed).toBeUndefined();
      expect(tile.style.display).toBe("block");
      send(map, "pointerdown", 20, 30);
      send(map, "pointermove", 45, 40);
      expect(replies.at(-1)?.committed).toBeUndefined();
      send(map, "pointerup", 100, 100);
      const next: ScanBounds = [49, 34, 64, 64];
      expect(replies.at(-1)).toMatchObject({ committed: true, session_id: 7, region: next });
      // A late frame, even with the same shape, still belongs at its own origin.
      paint(area);
      expect(tile.style.left).toBe("31.25%");
      paint(next, 32, [128, 128]);
      expect(tile.style.left).toBe("31.25%");
      paint(next, 32);
      expect(tile.style.left).toBe("53.125%");
      expect(copy.mock.calls.slice(-2).map((call) => call.slice(1))).toEqual([
        [0, 0, 30, 1, 0, 0, 30, 1],
        [0, 1, 2, 1, 0, 1, 2, 1],
      ]);
      // Cancellation rolls the selection back without submitting a new scan target.
      send(map, "pointerdown", 10, 10, true);
      send(map, "pointercancel", 20, 20, true);
      expect(replies.at(-1)?.committed).toBeUndefined();
      expect(replies.at(-1)?.region).toEqual(next);
      send(map, "pointerdown", 10, 10, true);
      send(map, "pointermove", 25, 25, true);
      command(false, next, true, true, 8);
      send(map, "pointerup", 27, 27, true);
      expect(replies.at(-1)?.committed).toBeUndefined();
      expect(replies.at(-1)?.region).toEqual(next);
      command(false, next);
      expect(tile.style.display).toBe("block");
      command(false, next, true, false, 9, true);
      const marker = gui.querySelector<HTMLElement>('[aria-label="Selected probe position"]')!;
      expect(marker.dataset.bounds).toBe("56,49,57,50");
      expect(gui.querySelector('[aria-label="Resize scan region"]')).toBeNull();
      // A press anywhere on the full map selects one pixel; Shift doesn't change that.
      send(map, "pointerdown", 12, 15, true);
      send(map, "pointerup", 12, 15, true);
      expect(replies.at(-1)).toMatchObject({ committed: true, region: [12, 15, 13, 16] });
      command(false, [12, 15, 13, 16], true, true, 9, true);
      const time = vi.spyOn(performance, "now").mockReturnValue(10);
      try {
        const before = replies.length;
        send(map, "pointerdown", 8, 9);
        expect(replies.at(-1)).toMatchObject({ committed: true, session_id: 9, region: [8, 9, 9, 10] });
        send(map, "pointermove", 17, 18);
        expect(replies.length).toBe(before + 1);
        time.mockReturnValue(91);
        send(map, "pointermove", 22, 25);
        expect(replies.at(-1)).toMatchObject({ session_id: 9, region: [22, 25, 23, 26] });
        send(map, "pointerup", 31, 4);
        expect(replies.at(-1)).toMatchObject({ session_id: 9, region: [31, 4, 32, 5] });
        expect(replies.length).toBe(before + 3);
        expect(marker.dataset.bounds).toBe("31,4,32,5");
      } finally {
        time.mockRestore();
      }
      image.scan = "series_42";
      render();
      expect(map.hidden).toBe(true);
      expect(replies.at(-1)?.ready).toBe(false);
      command();
      expect(replies.at(-1)?.ready).toBe(false);
      command(false, null, false);
    } finally {
      act(() => root.unmount());
      context.mockRestore();
      vi.unstubAllGlobals();
      gui.remove();
    }
  });
  it("ignores the local command in saved viewers in the same acquisition GUI", () => {
    const gui = document.body.appendChild(document.createElement("div"));
    gui.className = "spectra-gui";
    const saved = gui.appendChild(document.createElement("div"));
    saved.className = "spectra-gui-saved";
    const root = createRoot(saved);
    const reply = vi.fn();
    gui.addEventListener("spectra-live-region-change", reply);
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    try {
      act(() =>
        root.render(
          createElement(LiveScanRegion, {
            source: () => ({ canvas: null, rows: 64, cols: 64, frozen: false }),
            onMapSize: vi.fn(),
            onImageDraw: { current: null },
          }),
        ),
      );
      act(() => {
        gui.dispatchEvent(
          new CustomEvent("spectra-live-region", {
            detail: { type: "spectra-live-region", enabled: true, locked: false, size: 64, region: null },
          }),
        );
      });
      expect(reply).not.toHaveBeenCalled();
    } finally {
      act(() => root.unmount());
      gui.remove();
      vi.unstubAllGlobals();
    }
  });
  it("accepts embedded microscope commands only from the parent window at the embedding origin", () => {
    const mount = document.body.appendChild(document.createElement("div"));
    const root = createRoot(mount);
    const owner = { postMessage: vi.fn() };
    const parent = vi.spyOn(window, "parent", "get").mockReturnValue(owner as unknown as Window);
    const referrer = vi.spyOn(document, "referrer", "get").mockReturnValue("http://arina.test/notebooks/Start");
    const source = () => ({ canvas: null, rows: 64, cols: 64, scan: "live_1", frozen: true });
    const command = { type: "spectra-live-region", enabled: true, locked: false, size: 64, region: null };
    const dispatch = (origin: string, sender: unknown) =>
      act(() => {
        window.dispatchEvent(new MessageEvent("message", { origin, source: sender as Window, data: command }));
      });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    try {
      act(() =>
        root.render(
          createElement(LiveScanRegion, { source, onMapSize: vi.fn(), onImageDraw: { current: null } }),
        ),
      );
      dispatch("http://other.test", owner);
      dispatch("http://arina.test", window);
      expect(owner.postMessage).not.toHaveBeenCalled();
      dispatch("http://arina.test", owner);
      expect(owner.postMessage).toHaveBeenCalledWith(
        expect.objectContaining({ ready: false }),
        "http://arina.test",
      );
    } finally {
      act(() => root.unmount());
      mount.remove();
      referrer.mockRestore();
      parent.mockRestore();
      vi.unstubAllGlobals();
    }
  });
  it("draws the box at once on a stopped full Live frame the acquisition vouches for, with no frozen preview", () => {
    // 11-21, the operator: "the moment Region is on, it should draw a draggable box". The stopped Live frame stays explorable
    // (its preview withdrawn, never frozen); the panel says it was a full frame of this size.
    const gui = document.createElement("div");
    gui.className = "spectra-gui";
    document.body.append(gui);
    const root = createRoot(gui.appendChild(document.createElement("div")));
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 64;
    const context = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation((() => ({
      drawImage: vi.fn(),
    })) as unknown as typeof HTMLCanvasElement.prototype.getContext);
    const replies: { ready: boolean; region: ScanBounds | null }[] = [];
    gui.addEventListener("spectra-live-region-change", (event) => replies.push((event as CustomEvent).detail));
    const ask = (stoppedFrame: boolean) =>
      act(() => {
        gui.dispatchEvent(new CustomEvent("spectra-live-region", { detail: {
          type: "spectra-live-region", enabled: true, locked: false, size: 64, region: null, stoppedFrame } }));
      });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    try {
      act(() => root.render(createElement(LiveScanRegion, {
        source: () => ({ canvas, scan: undefined, rows: 64, cols: 64, frozen: false }),
        onMapSize: vi.fn(), onImageDraw: { current: null },
      })));
      ask(false);
      expect(replies.at(-1)).toEqual(expect.objectContaining({ ready: false }));
      ask(true);
      expect(replies.at(-1)).toEqual(expect.objectContaining({ ready: true }));
      expect(replies.at(-1)?.region).not.toBeNull();
    } finally {
      act(() => root.unmount());
      gui.remove();
      context.mockRestore();
      vi.unstubAllGlobals();
    }
  });
});
