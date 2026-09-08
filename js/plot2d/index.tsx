import * as React from "react";
import { createRoot } from "react-dom/client";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import {
  COLORMAPS,
  createGPUColormapEngine,
  GPUColormapEngine,
  renderToOffscreen,
} from "../colormaps";
import { extractBytes, downloadBlob } from "../format";

type Model = {
  get(key: string): any;
  set(key: string, value: unknown): void;
  save_changes(): void;
  on(event: string, callback: () => void): void;
  off(event: string, callback: () => void): void;
};

function render({ model, el }: { model: Model; el: HTMLElement }) {
  const host = document.createElement("div");
  host.dataset.quantemPlot2d = "true";
  host.style.cssText =
    "width:100%;min-width:260px;color:#203040;background:white;font:12px system-ui;";
  const canvas = document.createElement("canvas");
  canvas.dataset.quantemScientificOutput = "plot2d-map";
  canvas.style.cssText =
    "width:100%;display:block;touch-action:none;cursor:crosshair";
  const controls = document.createElement("div");
  controls.style.cssText =
    "display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:4px 8px";
  const reset = document.createElement("button");
  reset.textContent = "Reset View";
  const zoomIn = document.createElement("button");
  zoomIn.textContent = "Zoom In";
  const zoomOut = document.createElement("button");
  zoomOut.textContent = "Zoom Out";
  const zoomLabel = document.createElement("span");
  zoomLabel.setAttribute("aria-live", "polite");
  const save = document.createElement("button");
  save.textContent = "Save PNG";
  const status = document.createElement("span");
  status.textContent = "Preparing display…";
  const readout = document.createElement("div");
  readout.style.cssText =
    "height:22px;padding:3px 8px;font-variant-numeric:tabular-nums";
  const colorControl = document.createElement("span");
  colorControl.style.cssText = "display:inline-flex;gap:6px;align-items:center";
  const colorRoot = createRoot(colorControl);
  function updateColorControl() {
    colorRoot.render(<>
      <span>Color</span>
      <Select size="small" value={model.get("cmap")}
        inputProps={{ "aria-label": "Color map" }}
        sx={{ fontSize: 12, color: "#203040", background: "white", height: 28 }}
        MenuProps={{ PaperProps: { sx: { maxHeight: 320 } } }}
        onChange={(event) => {
          model.set("cmap", event.target.value);
          model.save_changes();
        }}>
        {Object.keys(COLORMAPS).map(name => <MenuItem key={name} value={name}>{name}</MenuItem>)}
      </Select>
    </>);
  }
  controls.append(colorControl, zoomIn, zoomOut, save, reset, zoomLabel, status);
  host.append(canvas, controls, readout);
  el.append(host);
  let engine: GPUColormapEngine | null = null;
  let bitmap: CanvasImageSource | null = null;
  let source = new Float64Array();
  let disposed = false,
    generation = 0,
    frame = 0;
  let queue = Promise.resolve();
  let bounds = (
    model.get("view_bounds").length
      ? model.get("view_bounds")
      : model.get("grid").bounds
  ).slice();
  let drag: { x: number; y: number; bounds: number[] } | null = null;
  let wheelTimer = 0;
  let readoutFrame = 0;
  let pendingReadout = "";
  function showReadout(text: string) {
    pendingReadout = text;
    if (!readoutFrame) readoutFrame = requestAnimationFrame(() => {
      readoutFrame = 0;
      readout.textContent = pendingReadout;
    });
  }
  const full = () => model.get("grid").bounds as number[];
  function size() {
    host.style.maxWidth = `${model.get("max_width") ?? 600}px`;
    schedule();
  }
  const geometry = () => ({
    width: Math.max(260, host.clientWidth),
    height: model.get("plot_height_px"),
    left: 65,
    top: 30,
    right: 18,
    bottom: 105,
  });
  const number = (value: number) => Number(value.toPrecision(4)).toString();
  const tickNumber = (value: number, span: number) =>
    number(Math.abs(value) < span * 1e-12 ? 0 : value);
  function commit() {
    model.set("view_bounds", bounds.slice());
    model.save_changes();
  }
  function clampBounds(next: number[]) {
    const original = full();
    return [0, 2].flatMap((index) => {
      const span = Math.min(
        original[index + 1] - original[index],
        Math.max(
          (original[index + 1] - original[index]) / 100,
          next[index + 1] - next[index],
        ),
      );
      const low = Math.max(
        original[index],
        Math.min(original[index + 1] - span, next[index]),
      );
      return [low, low + span];
    });
  }
  function paint() {
    frame = 0;
    if (disposed) return;
    const g = geometry(),
      width = g.width - g.left - g.right,
      height = g.height - g.top - g.bottom;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(g.width * ratio);
    canvas.height = Math.round(g.height * ratio);
    canvas.style.height = `${g.height}px`;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(ratio, ratio);
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, g.width, g.height);
    const grid = model.get("grid"),
      original = full();
    const zoom = (original[1] - original[0]) / (bounds[1] - bounds[0]);
    zoomLabel.textContent = `${number(zoom)}× · ${zoom > 1.001 ? "Drag to pan" : "Zoom in to pan"}`;
    canvas.style.cursor = drag
      ? "grabbing"
      : zoom > 1.001
        ? "grab"
        : "crosshair";
    if (bitmap) {
      ctx.save();
      ctx.translate(g.left, g.top + height);
      ctx.scale(1, -1);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(
        bitmap,
        ((bounds[0] - original[0]) / (original[1] - original[0])) * grid.cols,
        ((bounds[2] - original[2]) / (original[3] - original[2])) * grid.rows,
        ((bounds[1] - bounds[0]) / (original[1] - original[0])) * grid.cols,
        ((bounds[3] - bounds[2]) / (original[3] - original[2])) * grid.rows,
        0,
        0,
        width,
        height,
      );
      ctx.restore();
    }
    ctx.strokeStyle = "#425467";
    ctx.strokeRect(g.left, g.top, width, height);
    ctx.font = "12px system-ui";
    ctx.fillStyle = "#203040";
    ctx.textAlign = "center";
    ctx.fillText(model.get("title"), g.left + width / 2, 17);
    for (let tick = 0; tick <= 4; tick++) {
      const fraction = tick / 4;
      ctx.textAlign = "center";
      ctx.fillText(
        tickNumber(
          bounds[0] + fraction * (bounds[1] - bounds[0]),
          bounds[1] - bounds[0],
        ),
        g.left + fraction * width,
        g.top + height + 17,
      );
      ctx.textAlign = "right";
      ctx.fillText(
        tickNumber(
          bounds[2] + fraction * (bounds[3] - bounds[2]),
          bounds[3] - bounds[2],
        ),
        g.left - 7,
        g.top + height * (1 - fraction) + 4,
      );
    }
    ctx.textAlign = "center";
    ctx.fillText(model.get("x_label"), g.left + width / 2, g.top + height + 37);
    ctx.save();
    ctx.translate(15, g.top + height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(model.get("y_label"), 0, 0);
    ctx.restore();
    const lut = COLORMAPS[model.get("cmap")];
    if (lut)
      for (let i = 0; i < 256; i++) {
        ctx.fillStyle = `rgb(${lut[i * 3]},${lut[i * 3 + 1]},${lut[i * 3 + 2]})`;
        ctx.fillRect(
          g.left + (i * width) / 256,
          g.height - 52,
          width / 256 + 0.5,
          10,
        );
      }
    ctx.fillStyle = "#203040";
    ctx.textAlign = "left";
    ctx.fillText(number(model.get("vmin")), g.left, g.height - 27);
    ctx.textAlign = "right";
    ctx.fillText(number(model.get("vmax")), g.left + width, g.height - 27);
    ctx.textAlign = "center";
    ctx.fillText(model.get("colorbar_label"), g.left + width / 2, g.height - 8);
    const line = model.get("horizontal_line");
    if (line != null && line >= bounds[2] && line <= bounds[3]) {
      const pos =
        g.top + height * (1 - (line - bounds[2]) / (bounds[3] - bounds[2]));
      ctx.strokeStyle = "#e649a0";
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(g.left, pos);
      ctx.lineTo(g.left + width, pos);
      ctx.stroke();
    }
    canvas.setAttribute(
      "aria-label",
      `${model.get("title")}; ${model.get("x_label")}; ${model.get("y_label")}; ${model.get("colorbar_label")}`,
    );
    canvas.dataset.paintGeneration = String(
      Number(canvas.dataset.paintGeneration || 0) + 1,
    );
  }
  function schedule() {
    if (!frame) frame = requestAnimationFrame(paint);
  }
  const ready = createGPUColormapEngine()
    .then((value) => {
      engine = value;
      return value;
    })
    .catch(() => null);
  function prepare() {
    const current = ++generation;
    queue = queue
      .then(async () => {
        await ready;
        if (disposed || current !== generation) return;
        const grid = model.get("grid"),
          bytes = extractBytes(model.get("data_bytes"));
        source = new Float64Array(
          bytes.slice(0, grid.rows * grid.cols * 8).buffer,
        );
        const display = Float32Array.from(source),
          lut = COLORMAPS[model.get("cmap")];
        if (!lut) {
          status.textContent = "Unsupported colormap";
          return;
        }
        let next: CanvasImageSource | null = null;
        if (engine) {
          engine.uploadData(0, display, grid.cols, grid.rows);
          engine.uploadLUT(model.get("cmap"), lut);
          const rendered = await engine.renderSlotsToImageBitmapAsync(
            [0],
            [{ vmin: model.get("vmin"), vmax: model.get("vmax") }],
          );
          next = rendered?.[0] ?? null;
        }
        if (!next) {
          next = renderToOffscreen(
            display,
            grid.cols,
            grid.rows,
            lut,
            model.get("vmin"),
            model.get("vmax"),
          );
          status.textContent = "Canvas fallback · wheel to zoom";
        } else status.textContent = "WebGPU display · wheel to zoom";
        if (disposed || current !== generation) {
          if (next instanceof ImageBitmap) next.close();
          return;
        }
        if (bitmap instanceof ImageBitmap) bitmap.close();
        bitmap = next;
        schedule();
      })
      .catch((error) => {
        status.textContent = `Display error: ${String(error)}`;
      });
  }
  function plotPosition(event: MouseEvent) {
    const g = geometry(),
      rect = canvas.getBoundingClientRect();
    const col =
      (event.clientX - rect.left - g.left) / (g.width - g.left - g.right);
    const row =
      1 - (event.clientY - rect.top - g.top) / (g.height - g.top - g.bottom);
    return col >= 0 && col <= 1 && row >= 0 && row <= 1 ? [col, row] : null;
  }
  canvas.onpointerdown = (event) => {
    if (event.button !== 0 || !plotPosition(event)) return;
    event.preventDefault();
    event.stopPropagation();
    drag = { x: event.clientX, y: event.clientY, bounds: bounds.slice() };
    canvas.setPointerCapture(event.pointerId);
    schedule();
  };
  canvas.onpointermove = (event) => {
    const g = geometry(),
      rect = canvas.getBoundingClientRect();
    const width = g.width - g.left - g.right,
      height = g.height - g.top - g.bottom;
    if (drag) {
      const dx =
        ((event.clientX - drag.x) / width) * (drag.bounds[1] - drag.bounds[0]);
      const dy =
        ((event.clientY - drag.y) / height) * (drag.bounds[3] - drag.bounds[2]);
      bounds = clampBounds([
        drag.bounds[0] - dx,
        drag.bounds[1] - dx,
        drag.bounds[2] + dy,
        drag.bounds[3] + dy,
      ]);
      schedule();
    }
    const colFraction = (event.clientX - rect.left - g.left) / width;
    const rowFraction = 1 - (event.clientY - rect.top - g.top) / height;
    if (
      colFraction < 0 ||
      colFraction >= 1 ||
      rowFraction < 0 ||
      rowFraction >= 1
    ) {
      showReadout("");
      return;
    }
    const x = bounds[0] + colFraction * (bounds[1] - bounds[0]),
      y = bounds[2] + rowFraction * (bounds[3] - bounds[2]);
    const grid = model.get("grid"),
      original = full();
    const col = Math.floor(
      ((x - original[0]) / (original[1] - original[0])) * grid.cols,
    );
    const row = Math.floor(
      ((y - original[2]) / (original[3] - original[2])) * grid.rows,
    );
    const xc =
      original[0] + ((col + 0.5) * (original[1] - original[0])) / grid.cols;
    const yc =
      original[2] + ((row + 0.5) * (original[3] - original[2])) / grid.rows;
    showReadout(`Bin (${row}, ${col}) · x ${number(xc)} · y ${number(yc)} · value ${source[row * grid.cols + col]?.toPrecision(6)}`);
  };
  canvas.onpointerup =
    canvas.onpointercancel =
    canvas.onlostpointercapture =
      () => {
        if (!drag) return;
        drag = null;
        commit();
        schedule();
      };
  canvas.onpointerleave = () => {
    showReadout("");
  };
  function zoomBy(factor: number, position = [0.5, 0.5]) {
    bounds = clampBounds(
      [0, 2].flatMap((i, axis) => {
        const span = bounds[i + 1] - bounds[i];
        const anchor = bounds[i] + position[axis] * span;
        return [
          anchor - position[axis] * span * factor,
          anchor + (1 - position[axis]) * span * factor,
        ];
      }),
    );
    schedule();
  }
  canvas.addEventListener(
    "wheel",
    (event) => {
      const position = plotPosition(event);
      if (!position) return;
      event.preventDefault();
      event.stopPropagation();
      const delta =
        event.deltaY *
        (event.deltaMode === 1
          ? 16
          : event.deltaMode === 2
            ? geometry().height
            : 1);
      zoomBy(Math.exp(Math.max(-1, Math.min(1, delta * 0.002))), position);
      clearTimeout(wheelTimer);
      wheelTimer = window.setTimeout(commit, 150);
    },
    { passive: false },
  );
  reset.onclick = () => {
    clearTimeout(wheelTimer);
    drag = null;
    bounds = full().slice();
    commit();
    schedule();
  };
  canvas.ondblclick = reset.onclick as () => void;
  zoomIn.onclick = () => {
    zoomBy(1 / 1.5);
    commit();
  };
  zoomOut.onclick = () => {
    zoomBy(1.5);
    commit();
  };
  save.onclick = () =>
    canvas.toBlob((blob) => {
      if (blob) downloadBlob(blob, "plot2d.png");
    });
  const observers: [string, () => void][] = [];
  for (const key of ["data_bytes", "cmap", "vmin", "vmax"])
    observers.push([`change:${key}`, prepare]);
  for (const key of [
    "horizontal_line",
    "title",
    "x_label",
    "y_label",
    "colorbar_label",
    "plot_height_px",
  ])
    observers.push([`change:${key}`, schedule]);
  observers.push(["change:max_width", size]);
  observers.push(["change:cmap", updateColorControl]);
  observers.push([
    "change:view_bounds",
    () => {
      bounds = (
        model.get("view_bounds").length ? model.get("view_bounds") : full()
      ).slice();
      schedule();
    },
  ]);
  observers.forEach(([event, handler]) => model.on(event, handler));
  const resize = new ResizeObserver(schedule);
  resize.observe(host);
  size();
  updateColorControl();
  prepare();
  return () => {
    disposed = true;
    generation++;
    cancelAnimationFrame(frame);
    clearTimeout(wheelTimer);
    cancelAnimationFrame(readoutFrame);
    colorRoot.unmount();
    resize.disconnect();
    observers.forEach(([event, handler]) => model.off(event, handler));
    if (bitmap instanceof ImageBitmap) bitmap.close();
    void queue.finally(() => engine?.destroy());
    host.remove();
  };
}

export default { render };
