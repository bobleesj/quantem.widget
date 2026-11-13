// Frame thumbnails on the GPU.
//
// Each frame is box-filtered ONCE, on the GPU, from its resident full-size slot into a small
// float32 layer of raw values (THUMBNAIL_LONG_SIDE px on the long side). Layers live in pages:
// storage buffers of THUMBNAIL_PAGE_FRAMES layers each, added as the stack grows, so a live
// stream never reallocates or copies what is already there. The strip is one canvas: one
// instanced draw per page touched by the visible cells samples each cell's layer and applies the
// main image's display transform (log scale, vmin/vmax, colormap LUT) in the fragment shader.
// A histogram drag, a colormap or log-scale change only rewrites the small uniform and cell
// buffers; no layer is recomputed and nothing is read back to the CPU.
import type { GPUColormapEngine } from "./colormapEngine";
import { DISPLAY_NORMALIZE_WGSL } from "./displayShaders";
import { computePipeline } from "./pipelines";

/** Long side of a thumbnail layer, in texels. */
export const THUMBNAIL_LONG_SIDE = 128;
/** Layers per page buffer: a page is added when a frame index reaches the pages' capacity. */
export const THUMBNAIL_PAGE_FRAMES = 64;
/** Box filter: at most this many samples per block side (a 1024 px frame into 128 px averages all 8 x 8). */
const MAX_SAMPLES_PER_SIDE = 16;
/** Bytes per cell instance: rect (4 f32), layer (u32), pad (u32), vmin, vmax (f32). */
export const THUMBNAIL_CELL_BYTES = 32;
/** Bytes of the draw uniform. */
export const THUMBNAIL_UNIFORM_BYTES = 32;

/** Thumbnail layer size for a source region: THUMBNAIL_LONG_SIDE on the long side (never upsampled). */
export function thumbnailLayerSize(sourceWidth: number, sourceHeight: number, longSide = THUMBNAIL_LONG_SIDE): { width: number; height: number } {
  const w = Math.max(1, Math.round(sourceWidth));
  const h = Math.max(1, Math.round(sourceHeight));
  const scale = Math.min(1, longSide / Math.max(w, h));
  return { width: Math.max(1, Math.round(w * scale)), height: Math.max(1, Math.round(h * scale)) };
}

/** Pages needed to hold `frames` layers (grown in whole pages, never per frame). */
export function thumbnailPagesFor(frames: number, pageFrames = THUMBNAIL_PAGE_FRAMES): number {
  return Math.max(0, Math.ceil(Math.max(0, frames) / pageFrames));
}

/** One visible cell in device pixels of the strip canvas, and the display range its frame is drawn with. */
export type ThumbnailCellDraw = {
  frame: number;
  x: number;
  y: number;
  width: number;
  height: number;
  vmin: number;
  vmax: number;
};

/** Per-page instance runs: cells of page `page` are instances first..first+count-1. */
export type ThumbnailDrawGroup = { page: number; first: number; count: number };

/**
 * Pack the cells into the instance buffer layout, grouped by page so each page is one draw.
 * Cells whose frame has no layer (`has(frame)` false) are left out: the canvas shows black there.
 */
export function packThumbnailCells(
  cells: readonly ThumbnailCellDraw[],
  has: (frame: number) => boolean,
  pageFrames = THUMBNAIL_PAGE_FRAMES,
): { data: ArrayBuffer; count: number; groups: ThumbnailDrawGroup[] } {
  const ready = cells.filter(cell => cell.frame >= 0 && has(cell.frame));
  ready.sort((a, b) => Math.floor(a.frame / pageFrames) - Math.floor(b.frame / pageFrames));
  const data = new ArrayBuffer(Math.max(1, ready.length) * THUMBNAIL_CELL_BYTES);
  const f32 = new Float32Array(data);
  const u32 = new Uint32Array(data);
  const groups: ThumbnailDrawGroup[] = [];
  ready.forEach((cell, i) => {
    const page = Math.floor(cell.frame / pageFrames);
    const o = i * (THUMBNAIL_CELL_BYTES / 4);
    f32[o] = cell.x; f32[o + 1] = cell.y; f32[o + 2] = cell.width; f32[o + 3] = cell.height;
    u32[o + 4] = cell.frame % pageFrames;
    u32[o + 5] = 0;
    f32[o + 6] = cell.vmin; f32[o + 7] = cell.vmax;
    const last = groups[groups.length - 1];
    if (last && last.page === page) last.count++;
    else groups.push({ page, first: i, count: 1 });
  });
  return { data, count: ready.length, groups };
}

/** The draw uniform: canvas size (f32 x2), layer size (u32 x2), log scale, pad. */
export function packThumbnailUniforms(canvasWidth: number, canvasHeight: number, layerWidth: number, layerHeight: number, logScale: boolean): ArrayBuffer {
  const data = new ArrayBuffer(THUMBNAIL_UNIFORM_BYTES);
  const f32 = new Float32Array(data);
  const u32 = new Uint32Array(data);
  f32[0] = Math.max(1, canvasWidth);
  f32[1] = Math.max(1, canvasHeight);
  u32[2] = Math.max(1, layerWidth);
  u32[3] = Math.max(1, layerHeight);
  u32[4] = logScale ? 1 : 0;
  return data;
}

const DOWNSAMPLE_SHADER = /* wgsl */ `
struct Params {
  src_width: u32,     // slot row length
  src_height: u32,
  region_x0: u32,     // slot column of the panel drawn
  region_width: u32,
  layer_width: u32,
  layer_height: u32,
  layer: u32,         // layer within the page
  max_samples: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> page: array<f32>;

fn is_finite(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let lw = params.layer_width;
  let lh = params.layer_height;
  if (gid.x >= lw || gid.y >= lh) { return; }
  // the block of source pixels under this texel (an area / box filter)
  let bx0 = gid.x * params.region_width / lw;
  let bx1 = max(bx0 + 1u, (gid.x + 1u) * params.region_width / lw);
  let y0 = gid.y * params.src_height / lh;
  let y1 = max(y0 + 1u, (gid.y + 1u) * params.src_height / lh);
  let step_x = max(1u, (bx1 - bx0 + params.max_samples - 1u) / params.max_samples);
  let step_y = max(1u, (y1 - y0 + params.max_samples - 1u) / params.max_samples);
  var total = 0.0;
  var count = 0.0;
  for (var y = y0; y < y1; y += step_y) {
    let row = y * params.src_width + params.region_x0;
    for (var x = bx0; x < bx1; x += step_x) {
      let v = src[row + x];
      if (is_finite(v)) { total += v; count += 1.0; }
    }
  }
  var value = -3.4e38;   // no finite pixel in the block
  if (count > 0.0) { value = total / count; }
  page[params.layer * lw * lh + gid.y * lw + gid.x] = value;
}
`;

const DRAW_SHADER = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct Uniforms {
  canvas_size: vec2f,
  layer_size: vec2u,
  log_scale: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};
struct Cell { rect: vec4f, layer: u32, _pad: u32, vmin: f32, vmax: f32 };
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
@group(0) @binding(2) var<storage, read> page: array<f32>;
@group(0) @binding(3) var<storage, read> lut: array<u32>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) texel: vec2f,
  @location(1) @interpolate(flat) cell: u32,
};

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  // two triangles: corners (0,0) (1,0) (0,1) / (0,1) (1,0) (1,1)
  let cx = select(0.0, 1.0, vi == 1u || vi == 4u || vi == 5u);
  let cy = select(0.0, 1.0, vi == 2u || vi == 3u || vi == 5u);
  let rect = cells[ii].rect;
  let px = rect.xy + vec2f(cx, cy) * rect.zw;
  var out: VSOut;
  out.pos = vec4f(px.x / u.canvas_size.x * 2.0 - 1.0, 1.0 - px.y / u.canvas_size.y * 2.0, 0.0, 1.0);
  out.texel = vec2f(cx, cy) * vec2f(u.layer_size);
  out.cell = ii;
  return out;
}

fn texel_at(base: u32, x: u32, y: u32) -> f32 { return page[base + y * u.layer_size.x + x]; }

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let cell = cells[in.cell];
  let lw = u.layer_size.x;
  let lh = u.layer_size.y;
  let base = cell.layer * lw * lh;
  // bilinear between texel centres, as the main image's Smooth sampling
  let fx = clamp(in.texel.x - 0.5, 0.0, f32(lw - 1u));
  let fy = clamp(in.texel.y - 0.5, 0.0, f32(lh - 1u));
  let x0 = u32(floor(fx));
  let y0 = u32(floor(fy));
  let x1 = min(x0 + 1u, lw - 1u);
  let y1 = min(y0 + 1u, lh - 1u);
  let tx = fx - f32(x0);
  let ty = fy - f32(y0);
  let v00 = texel_at(base, x0, y0);
  let v10 = texel_at(base, x1, y0);
  let v01 = texel_at(base, x0, y1);
  let v11 = texel_at(base, x1, y1);
  var val = v00;
  var empty = false;
  if (min(min(v00, v10), min(v01, v11)) > -3.0e38) {
    val = mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
  } else {
    // a block without a finite pixel: nearest texel, and the low end of the colormap for an empty one
    let nx = select(x0, x1, tx >= 0.5);
    let ny = select(y0, y1, ty >= 0.5);
    val = texel_at(base, nx, ny);
    empty = val <= -3.0e38;
  }
  if (u.log_scale == 1u) {
    if (val >= 0.0) { val = log(1.0 + val); } else { val = -log(1.0 - val); }
  }
  var t = display_normalize(val, cell.vmin, cell.vmax);
  if (empty) { t = 0.0; }
  let rgb = lut[min(u32(t * 255.0), 255u)];
  return vec4f(f32(rgb & 0xFFu) / 255.0, f32((rgb >> 8u) & 0xFFu) / 255.0, f32((rgb >> 16u) & 0xFFu) / 255.0, 1.0);
}
`;

/** Where a frame's layer comes from: the panel region of its resident slot, and the stack identity. */
export type ThumbnailSource = {
  /** Changes when the stack, its geometry or the panel shown changes: every layer is rebuilt. */
  key: string;
  /** Slot column where the panel drawn starts, and its width. */
  regionX0: number;
  regionWidth: number;
};

export type ThumbnailDrawOptions = {
  cells: readonly ThumbnailCellDraw[];
  logScale: boolean;
  lutName: string;
  lut: Uint8Array;
};

/** Thumbnail layers of every resident frame, and the strip's one-canvas draw. */
export class ThumbnailRenderer {
  private readonly engine: GPUColormapEngine;
  private readonly device: GPUDevice;
  private readonly format: GPUTextureFormat;
  private downsamplePipeline: GPUComputePipeline | null = null;
  private drawPipeline: GPURenderPipeline | null = null;
  private readonly downsampleParams: GPUBuffer;
  private readonly uniforms: GPUBuffer;
  private cellBuffer: GPUBuffer | null = null;
  private cellCapacity = 0;
  private pages: GPUBuffer[] = [];
  private built: Uint8Array = new Uint8Array(0);
  private sourceKey = "";
  private layerWidth = 0;
  private layerHeight = 0;
  private readonly configured = new WeakSet<GPUCanvasContext>();
  /** Set when the pipelines fail GPU validation; every later draw is refused instead of leaving black canvases. */
  failure: string | null = null;

  constructor(engine: GPUColormapEngine) {
    this.engine = engine;
    this.device = engine.getDevice();
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.downsampleParams = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uniforms = this.device.createBuffer({ size: THUMBNAIL_UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  private ensurePipelines(): boolean {
    if (this.failure) return false;
    if (!this.downsamplePipeline) {
      this.device.pushErrorScope("validation");
      this.downsamplePipeline = computePipeline(this.device, DOWNSAMPLE_SHADER);
      const module = this.device.createShaderModule({ code: DRAW_SHADER });
      this.drawPipeline = this.device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "vs" },
        fragment: { module, entryPoint: "fs", targets: [{ format: this.format }] },
        primitive: { topology: "triangle-list" },
      });
      void this.device.popErrorScope().then(error => {
        if (!error) return;
        this.failure = error.message;
        console.error("[thumbnails] WebGPU pipeline failed validation; thumbnails are not drawn:", error.message);
      });
    }
    return Boolean(this.downsamplePipeline && this.drawPipeline);
  }

  /** Frames with a layer for the current source. */
  has(frame: number): boolean {
    return frame >= 0 && frame < this.built.length && this.built[frame] === 1;
  }

  /** A new stack, geometry or panel (another key) drops every layer and its pages. */
  useSource(key: string): void {
    if (key === this.sourceKey) return;
    this.dropLayers();
    this.sourceKey = key;
  }

  private dropLayers(): void {
    for (const page of this.pages) page.destroy();
    this.pages = [];
    this.built = new Uint8Array(0);
  }

  /** Grow to hold `frame`: whole pages are added, nothing already built moves. */
  private ensureFrame(frame: number): GPUBuffer {
    const pages = thumbnailPagesFor(frame + 1);
    const pageBytes = THUMBNAIL_PAGE_FRAMES * this.layerWidth * this.layerHeight * 4;
    while (this.pages.length < pages) {
      this.pages.push(this.device.createBuffer({ size: pageBytes, usage: GPUBufferUsage.STORAGE }));
    }
    if (this.built.length < pages * THUMBNAIL_PAGE_FRAMES) {
      const next = new Uint8Array(pages * THUMBNAIL_PAGE_FRAMES);
      next.set(this.built);
      this.built = next;
    }
    return this.pages[Math.floor(frame / THUMBNAIL_PAGE_FRAMES)];
  }

  /**
   * Box-filter resident slot `frame` into its layer, once per source. Returns true when the layer
   * exists (already built or built now), false when the slot is not a resident float32 frame.
   */
  ingest(frame: number, source: ThumbnailSource): boolean {
    const slot = this.engine.slots[frame];
    if (!slot || slot.dataKind !== "f32" || frame < 0) return false;
    const regionWidth = Math.max(1, Math.min(slot.width, Math.round(source.regionWidth)));
    const regionX0 = Math.max(0, Math.min(slot.width - regionWidth, Math.round(source.regionX0)));
    const size = thumbnailLayerSize(regionWidth, slot.height);
    this.useSource(source.key);
    if (size.width !== this.layerWidth || size.height !== this.layerHeight) {
      this.dropLayers();
      this.layerWidth = size.width;
      this.layerHeight = size.height;
    }
    if (this.has(frame)) return true;
    if (!this.ensurePipelines()) return false;
    const page = this.ensureFrame(frame);
    this.device.queue.writeBuffer(this.downsampleParams, 0, new Uint32Array([
      slot.width, slot.height, regionX0, regionWidth,
      this.layerWidth, this.layerHeight, frame % THUMBNAIL_PAGE_FRAMES, MAX_SAMPLES_PER_SIDE,
    ]));
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.downsamplePipeline!);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: this.downsamplePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.downsampleParams } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: page } },
      ],
    }));
    pass.dispatchWorkgroups(Math.ceil(this.layerWidth / 8), Math.ceil(this.layerHeight / 8));
    pass.end();
    // one submit per frame: the params uniform is rewritten for the next frame
    this.device.queue.submit([encoder.finish()]);
    this.built[frame] = 1;
    return true;
  }

  /** Draw the strip's visible cells into `context` (its canvas sized in device pixels). */
  draw(context: GPUCanvasContext, options: ThumbnailDrawOptions): boolean {
    if (!this.ensurePipelines()) return false;
    if (!this.configured.has(context)) {
      context.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
      this.configured.add(context);
    }
    const texture = context.getCurrentTexture();
    const packed = packThumbnailCells(options.cells, frame => this.has(frame));
    if (packed.count > this.cellCapacity || !this.cellBuffer) {
      this.cellBuffer?.destroy();
      this.cellCapacity = Math.max(packed.count, this.cellCapacity * 2, 64);
      this.cellBuffer = this.device.createBuffer({ size: this.cellCapacity * THUMBNAIL_CELL_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    if (packed.count > 0) this.device.queue.writeBuffer(this.cellBuffer, 0, packed.data, 0, packed.count * THUMBNAIL_CELL_BYTES);
    this.device.queue.writeBuffer(this.uniforms, 0, packThumbnailUniforms(texture.width, texture.height, this.layerWidth, this.layerHeight, options.logScale));
    const lut = this.engine.namedLutBuffer(options.lutName, options.lut);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    if (packed.count > 0) {
      pass.setPipeline(this.drawPipeline!);
      for (const group of packed.groups) {
        const page = this.pages[group.page];
        if (!page) continue;
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: this.drawPipeline!.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.uniforms } },
            { binding: 1, resource: { buffer: this.cellBuffer } },
            { binding: 2, resource: { buffer: page } },
            { binding: 3, resource: { buffer: lut } },
          ],
        }));
        pass.draw(6, group.count, 0, group.first);
      }
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  /** Free the strip canvas's textures; the next draw configures it again. */
  release(context: GPUCanvasContext): void {
    if (!this.configured.has(context)) return;
    context.unconfigure();
    this.configured.delete(context);
  }

  /** Bytes held by the layer pages. */
  get layerBytes(): number {
    return this.pages.length * THUMBNAIL_PAGE_FRAMES * this.layerWidth * this.layerHeight * 4;
  }

  destroy(): void {
    this.dropLayers();
    this.downsampleParams.destroy();
    this.uniforms.destroy();
    this.cellBuffer?.destroy();
    this.cellBuffer = null;
  }
}
