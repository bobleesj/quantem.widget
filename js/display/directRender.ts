// Direct canvas rendering: slots drawn straight into a WebGPU canvas or an
// offscreen bitmap at display size (scaled, shared-grid, per-panel and
// combined-region layouts), so a repaint never reads pixels back to the CPU.
import type { GPUColormapEngine, GPUSlot } from "./colormapEngine";
import { DIRECT_GRID_COLORMAP_SHADER, DIRECT_SLOT_COLORMAP_SHADER, DIRECT_SLOT_GPU_RANGE_SHADER, PACKED_PANEL_TRANSFORM_SHADER } from "./displayShaders";
import { flushParamsBufQueue, paramsBufQueue, type GPURanges } from "./gpuRange";
import { computePipeline, renderPipeline } from "./pipelines";

function shouldSmoothDirectSample(
  smooth: boolean | undefined,
  zoom: number,
  sourceWidth: number,
  sourceHeight: number,
  panelWidth: number,
  panelHeight: number,
): boolean {
  if (!smooth) return false;
  const scale = Math.max(1e-6, zoom);
  const projectedW = Math.max(1, panelWidth) * scale;
  const projectedH = Math.max(1, panelHeight) * scale;
  return projectedW >= Math.max(1, sourceWidth) * 0.75
    || projectedH >= Math.max(1, sourceHeight) * 0.75;
}

/** Clear color for a 0xBBGGRR packed background, the byte order the shaders' unpack_rgb reads. */
function clearColorFromRgb(bgRgb: number): GPUColorDict {
  return { r: (bgRgb & 0xFF) / 255, g: ((bgRgb >> 8) & 0xFF) / 255, b: ((bgRgb >> 16) & 0xFF) / 255, a: 1 };
}

/** Direct-to-canvas colormap passes over the engine's slots, with their pipelines and uniforms. */
export class DirectRenderer {
  private readonly engine: GPUColormapEngine;
  private readonly ranges: GPURanges;
  private readonly device: GPUDevice;
  private directGridPipeline: GPURenderPipeline | null = null;
  private directSlotPipeline: GPURenderPipeline | null = null;
  private directSlotGpuRangePipeline: GPURenderPipeline | null = null;
  private packedPanelTransformPipeline: GPUComputePipeline | null = null;
  private directGridParams = new ArrayBuffer(64);
  private directGridParamsU32 = new Uint32Array(this.directGridParams);
  private directGridParamsF32 = new Float32Array(this.directGridParams);
  private packedPanelRangesBuffer: GPUBuffer | null = null;
  private packedPanelRangesCapacity = 0;
  private packedPanelTransformsBuffer: GPUBuffer | null = null;
  private packedPanelTransformsCapacity = 0;
  private packedPanelIndicesBuffer: GPUBuffer | null = null;
  private packedPanelIndicesCapacity = 0;

  constructor(engine: GPUColormapEngine, ranges: GPURanges) {
    this.engine = engine;
    this.ranges = ranges;
    this.device = engine.getDevice();
  }

  private ensureDirectGridPipeline(format: GPUTextureFormat): void {
    this.directGridPipeline ??= renderPipeline(this.device, DIRECT_GRID_COLORMAP_SHADER, format);
  }

  private ensureDirectSlotPipeline(format: GPUTextureFormat): void {
    this.directSlotPipeline ??= renderPipeline(this.device, DIRECT_SLOT_COLORMAP_SHADER, format);
  }

  private ensureDirectSlotGpuRangePipeline(format: GPUTextureFormat): void {
    this.directSlotGpuRangePipeline ??= renderPipeline(this.device, DIRECT_SLOT_GPU_RANGE_SHADER, format);
  }

  private ensurePackedPanelTransformPipeline(): void {
    this.packedPanelTransformPipeline ??= computePipeline(this.device, PACKED_PANEL_TRANSFORM_SHADER);
  }

  /**
   * Encode the scaled colormap pass: each of the outWidth x outHeight output
   * pixels samples its nearest source pixel through `pipeline` (float32 or
   * packed uint8), writing the 32-byte scaled params into the slot's uniform.
   */
  private encodeScaledColormapPass(
    encoder: GPUCommandEncoder,
    slot: GPUSlot,
    pipeline: GPUComputePipeline,
    lutBuffer: GPUBuffer,
    range: { vmin: number; vmax: number },
    logScale: boolean,
    outWidth: number,
    outHeight: number,
  ): void {
    const params = new ArrayBuffer(32);
    const paramsU32 = new Uint32Array(params);
    const paramsF32 = new Float32Array(params);
    paramsU32[0] = slot.width;
    paramsU32[1] = slot.height;
    paramsU32[2] = outWidth;
    paramsU32[3] = outHeight;
    paramsF32[4] = range.vmin;
    paramsF32[5] = range.vmax;
    paramsU32[6] = logScale ? 1 : 0;
    paramsU32[7] = 0;
    this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);
    const computeGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: slot.paramsBuffer } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: lutBuffer } },
        { binding: 3, resource: { buffer: slot.rgbaBuffer } },
      ],
    });
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, computeGroup);
    computePass.dispatchWorkgroups(Math.ceil(outWidth / 16), Math.ceil(outHeight / 16));
    computePass.end();
  }

  /**
   * GPU colormap one slot, then blit the full-resolution RGBA buffer into a
   * smaller OffscreenCanvas. The fragment shader samples the source buffer by
   * UV, so values stay full-precision through the colormap step while playback
   * avoids creating a 4096x4096 ImageBitmap when the visible canvas is smaller.
   */
  renderSlotScaledToImageBitmap(
    idx: number,
    range: { vmin: number; vmax: number },
    logScale: boolean,
    outW: number,
    outH: number,
  ): ImageBitmap | null {
    if (!this.engine.pipeline || !this.engine.lutBuffer) return null;
    const slot = this.engine.slots[idx];
    if (!slot || slot.directOnly) return null;
    const canvasWidth = Math.max(1, Math.round(outW));
    const canvasHeight = Math.max(1, Math.round(outH));
    if (canvasWidth * canvasHeight > slot.rgbaCapacity) {
      if (slot.rgbaCapacity < slot.count) return null;
      const bitmaps = this.engine.renderSlotsToImageBitmap([idx], [range], logScale);
      return bitmaps?.[0] ?? null;
    }
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.engine.ensureScaledPipeline();
    this.engine.ensureBlitPipeline(format);
    if (!this.engine.scaledPipeline || !this.engine.blitPipeline) return null;
    const encoder = this.device.createCommandEncoder();
    this.encodeScaledColormapPass(encoder, slot, this.engine.scaledPipeline, this.engine.lutBuffer, range, logScale, canvasWidth, canvasHeight);
    const blit = this.engine.encodeBlitToOffscreen(encoder, slot.rgbaBuffer, canvasWidth, canvasHeight, format);
    if (!blit) return null;
    this.device.queue.submit([encoder.finish()]);
    blit.blitParamsBuffer.destroy();
    return blit.canvas.transferToImageBitmap();
  }

  /**
   * Queue-safe scaled render for float32 or packed uint8 slots.
   *
   * Awaiting submitted work before `transferToImageBitmap()` is required on
   * Metal; otherwise a busy queue can snapshot the render pass clear color.
   */
  async renderSlotScaledToImageBitmapAsync(
    idx: number,
    range: { vmin: number; vmax: number },
    logScale: boolean,
    outW: number,
    outH: number,
    lutName: string,
    lut: Uint8Array,
  ): Promise<ImageBitmap | null> {
    const activeLut = this.engine.namedLutBuffer(lutName, lut);
    const slot = this.engine.slots[idx];
    if (!slot || slot.directOnly) return null;
    const canvasWidth = Math.max(1, Math.round(outW));
    const canvasHeight = Math.max(1, Math.round(outH));
    if (canvasWidth * canvasHeight > slot.rgbaCapacity) return null;

    const format = navigator.gpu.getPreferredCanvasFormat();
    if (slot.dataKind === "u8") this.engine.ensureScaledUint8Pipeline();
    else this.engine.ensureScaledPipeline();
    this.engine.ensureBlitPipeline(format);
    const computePipeline = slot.dataKind === "u8" ? this.engine.scaledUint8Pipeline : this.engine.scaledPipeline;
    if (!computePipeline || !this.engine.blitPipeline) return null;
    const encoder = this.device.createCommandEncoder();
    this.encodeScaledColormapPass(encoder, slot, computePipeline, activeLut, range, logScale, canvasWidth, canvasHeight);
    const blit = this.engine.encodeBlitToOffscreen(encoder, slot.rgbaBuffer, canvasWidth, canvasHeight, format);
    if (!blit) return null;
    this.device.queue.submit([encoder.finish()]);
    blit.blitParamsBuffer.destroy();
    await this.device.queue.onSubmittedWorkDone();
    return blit.canvas.transferToImageBitmap();
  }

  renderSharedGridDirectToCanvas(
    idx: number,
    range: { vmin: number; vmax: number },
    logScale: boolean,
    ctx: GPUCanvasContext,
    opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      sourcePanelWidth?: number;
      sharedSource?: boolean;
    },
  ): boolean {
    if (!this.engine.lutBuffer) return false;
    const slot = this.engine.slots[idx];
    if (!slot) return false;
    const outW = Math.max(1, Math.round(opts.width));
    const outH = Math.max(1, Math.round(opts.height));
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ensureDirectGridPipeline(format);
    const pipeline = this.directGridPipeline;
    if (!pipeline) return false;

    const params = new ArrayBuffer(64);
    const paramsU32 = new Uint32Array(params);
    const paramsF32 = new Float32Array(params);
    paramsU32[0] = slot.width;
    paramsU32[1] = slot.height;
    paramsU32[2] = Math.max(1, Math.min(slot.width, Math.round(opts.sourcePanelWidth ?? slot.width)));
    paramsU32[3] = outW;
    paramsU32[4] = outH;
    paramsU32[5] = Math.max(1, Math.round(opts.panelCount));
    paramsU32[6] = Math.max(1, Math.round(opts.cols));
    paramsU32[7] = Math.max(1, Math.round(opts.rows));
    paramsU32[8] = logScale ? 1 : 0;
    paramsU32[9] = opts.bgRgb & 0xFFFFFF;
    paramsU32[10] = opts.sharedSource ? 1 : 0;
    paramsU32[11] = 0;
    paramsF32[12] = range.vmin;
    paramsF32[13] = range.vmax;
    paramsF32[14] = Math.max(0, opts.gap);
    paramsF32[15] = 0;
    this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

    let bindGroup = slot.directGridBindGroup;
    if (!bindGroup) {
      bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.paramsBuffer } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: this.engine.lutBuffer } },
        ],
      });
      slot.directGridBindGroup = bindGroup;
    }

    const texture = ctx.getCurrentTexture();
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  /** Queue-safe durable-frame twin of renderSharedGridDirectToCanvas. */
  async renderSharedGridToImageBitmapAsync(
    idx: number,
    range: { vmin: number; vmax: number },
    logScale: boolean,
    opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      sourcePanelWidth?: number;
      sharedSource?: boolean;
    },
  ): Promise<ImageBitmap | null> {
    const canvas = new OffscreenCanvas(
      Math.max(1, Math.round(opts.width)),
      Math.max(1, Math.round(opts.height)),
    );
    const context = canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!context) return null;
    context.configure({
      device: this.device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: "opaque",
    });
    if (!this.renderSharedGridDirectToCanvas(idx, range, logScale, context, opts)) return null;
    await this.device.queue.onSubmittedWorkDone();
    return canvas.transferToImageBitmap();
  }

  renderPanelSlotsDirectToCanvas(
    indices: number[],
    range: { vmin: number; vmax: number } | { vmin: number; vmax: number }[],
    logScale: boolean | boolean[],
    ctx: GPUCanvasContext,
    opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      transforms?: { zoom: number; panX: number; panY: number }[];
      smooth?: boolean;
    },
  ): boolean {
    const encoder = this.device.createCommandEncoder();
    const rendered = this.encodePanelSlotsDirectToCanvas(encoder, indices, range, logScale, ctx, opts);
    if (rendered) this.device.queue.submit([encoder.finish()]);
    return rendered;
  }

  private encodePanelSlotsDirectToCanvas(
    encoder: GPUCommandEncoder,
    ...[indices, range, logScale, ctx, opts]: Parameters<GPUColormapEngine["renderPanelSlotsDirectToCanvas"]>
  ): boolean {
    if (!this.engine.lutBuffer || indices.length === 0) return false;
    // Each panel's placement and range go through its slot's uniform, written before the one
    // render pass: a slot drawn twice would draw both panels with the last panel's values.
    if (new Set(indices).size !== indices.length) throw new Error("Supply one distinct resident slot for each panel.");
    const outW = Math.max(1, Math.round(opts.width));
    const outH = Math.max(1, Math.round(opts.height));
    const n = Math.max(1, Math.min(indices.length, Math.round(opts.panelCount)));
    const cols = Math.max(1, Math.round(opts.cols));
    const rows = Math.max(1, Math.round(opts.rows));
    const gap = Math.max(0, opts.gap);
    const panelW = (outW - gap * (cols - 1)) / cols;
    const panelH = (outH - gap * (rows - 1)) / rows;
    if (panelW <= 0 || panelH <= 0) return false;

    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ensureDirectSlotPipeline(format);
    const pipeline = this.directSlotPipeline;
    if (!pipeline) return false;

    const params = this.directGridParams;
    const paramsU32 = this.directGridParamsU32;
    const paramsF32 = this.directGridParamsF32;
    for (let panel = 0; panel < n; panel++) {
      const slot = this.engine.slots[indices[panel]];
      if (!slot) return false;
      const panelRange = Array.isArray(range) ? (range[panel] ?? range[0]) : range;
      const panelLogScale = Array.isArray(logScale) ? !!logScale[panel] : logScale;
      paramsU32[0] = slot.width;
      paramsU32[1] = slot.height;
      paramsU32[2] = 0;
      paramsU32[3] = slot.width;
      paramsU32[4] = Math.max(1, Math.round(panelH));
      paramsU32[5] = Math.max(1, Math.round(panelW));
      const col = panel % cols;
      const row = Math.floor(panel / cols);
      paramsF32[6] = col * (panelW + gap);
      paramsF32[7] = row * (panelH + gap);
      paramsU32[8] = panelLogScale ? 1 : 0;
      paramsU32[9] = opts.bgRgb & 0xFFFFFF;
      const transform = opts.transforms?.[panel];
      const zoomValue = Math.max(1e-6, transform?.zoom ?? 1);
      paramsF32[10] = zoomValue;
      paramsU32[11] = shouldSmoothDirectSample(opts.smooth, zoomValue, slot.width, slot.height, panelW, panelH) ? 1 : 0;
      paramsF32[12] = panelRange.vmin;
      paramsF32[13] = panelRange.vmax;
      paramsF32[14] = transform?.panX ?? 0;
      paramsF32[15] = transform?.panY ?? 0;
      this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);
      if (!slot.directSlotBindGroup) {
        slot.directSlotBindGroup = this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: slot.paramsBuffer } },
            { binding: 1, resource: { buffer: slot.dataBuffer } },
            { binding: 2, resource: { buffer: this.engine.lutBuffer } },
          ],
        });
      }
    }

    const texture = ctx.getCurrentTexture();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: clearColorFromRgb(opts.bgRgb),
      }],
    });
    pass.setPipeline(pipeline);
    for (let panel = 0; panel < n; panel++) {
      const slot = this.engine.slots[indices[panel]];
      if (!slot?.directSlotBindGroup) continue;
      const col = panel % cols;
      const row = Math.floor(panel / cols);
      const x = col * (panelW + gap);
      const y = row * (panelH + gap);
      const sx = Math.max(0, Math.floor(x));
      const sy = Math.max(0, Math.floor(y));
      const sw = Math.max(1, Math.ceil(panelW));
      const sh = Math.max(1, Math.ceil(panelH));
      pass.setViewport(x, y, panelW, panelH, 0, 1);
      pass.setScissorRect(sx, sy, Math.min(sw, outW - sx), Math.min(sh, outH - sy));
      pass.setBindGroup(0, slot.directSlotBindGroup);
      pass.draw(3);
    }
    pass.end();
    return true;
  }

  /**
   * Queue-safe offscreen twin of renderPanelSlotsDirectToCanvas.
   *
   * Browser presentation textures may be discarded after compositing. Render
   * to an OffscreenCanvas and transfer only after the GPU queue completes when
   * callers need a durable scientific frame.
   */
  async renderPanelSlotsToImageBitmapAsync(
    indices: number[],
    range: { vmin: number; vmax: number } | { vmin: number; vmax: number }[],
    logScale: boolean | boolean[],
    opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      transforms?: { zoom: number; panX: number; panY: number }[];
      smooth?: boolean;
    },
  ): Promise<ImageBitmap | null> {
    const canvas = new OffscreenCanvas(
      Math.max(1, Math.round(opts.width)),
      Math.max(1, Math.round(opts.height)),
    );
    const context = canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!context) return null;
    context.configure({
      device: this.device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: "opaque",
    });
    if (!this.renderPanelSlotsDirectToCanvas(indices, range, logScale, context, opts)) return null;
    await this.device.queue.onSubmittedWorkDone();
    return canvas.transferToImageBitmap();
  }

  renderSlotDirectWithGpuRangeToCanvas(
    idx: number,
    vminPct: number,
    vmaxPct: number,
    logScale: boolean,
    ctx: GPUCanvasContext,
    opts: {
      width: number;
      height: number;
      bgRgb: number;
      transform?: { zoom: number; panX: number; panY: number };
      smooth?: boolean;
    },
  ): boolean {
    return this.renderSlotsDirectWithGpuRangeToCanvases([idx], [ctx], vminPct, vmaxPct, logScale, opts) === 1;
  }

  /** Present distinct resident slots with one queue submission. */
  renderSlotsDirectWithGpuRangeToCanvases(
    indices: number[], contexts: GPUCanvasContext[], vminPct: number, vmaxPct: number,
    logScale: boolean,
    opts: { width: number; height: number; bgRgb: number;
      transform?: { zoom: number; panX: number; panY: number }; smooth?: boolean },
  ): number {
    if (indices.length !== contexts.length || new Set(indices).size !== indices.length) {
      throw new Error("Supply one distinct resident slot for each canvas context.");
    }
    const encoder = this.device.createCommandEncoder();
    const paramsStart = paramsBufQueue.length;
    let rendered = 0;
    try {
      indices.forEach((idx, i) => {
        if (this.encodeSlotDirectWithGpuRangeToCanvas(encoder, idx, vminPct, vmaxPct, logScale, contexts[i], opts)) rendered++;
      });
      if (rendered) this.device.queue.submit([encoder.finish()]);
      return rendered;
    } finally {
      flushParamsBufQueue(paramsStart);
    }
  }

  private encodeSlotDirectWithGpuRangeToCanvas(
    encoder: GPUCommandEncoder,
    idx: number,
    vminPct: number,
    vmaxPct: number,
    logScale: boolean,
    ctx: GPUCanvasContext,
    opts: {
      width: number;
      height: number;
      bgRgb: number;
      transform?: { zoom: number; panX: number; panY: number };
      smooth?: boolean;
    },
  ): boolean {
    if (!this.engine.lutBuffer) return false;
    const slot = this.engine.slots[idx];
    // Direct scalar presentation consumes f32. Packed u8 slots retain their
    // existing uint8-aware scaled renderer and must not be interpreted as f32.
    if (!slot || slot.dataKind !== "f32") return false;
    const outW = Math.max(1, Math.round(opts.width));
    const outH = Math.max(1, Math.round(opts.height));
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ensureDirectSlotGpuRangePipeline(format);
    const pipeline = this.directSlotGpuRangePipeline;
    if (!pipeline) return false;

    this.ranges.recordLiveRange(encoder, slot, logScale);

    const params = this.directGridParams;
    const paramsU32 = this.directGridParamsU32;
    const paramsF32 = this.directGridParamsF32;
    paramsU32[0] = slot.width;
    paramsU32[1] = slot.height;
    paramsU32[2] = 0;
    paramsU32[3] = slot.width;
    paramsU32[4] = outH;
    paramsU32[5] = outW;
    paramsF32[6] = 0;
    paramsF32[7] = 0;
    paramsU32[8] = logScale ? 1 : 0;
    paramsU32[9] = opts.bgRgb & 0xFFFFFF;
    const transform = opts.transform;
    const zoomValue = Math.max(1e-6, transform?.zoom ?? 1);
    paramsF32[10] = zoomValue;
    paramsU32[11] = shouldSmoothDirectSample(opts.smooth, zoomValue, slot.width, slot.height, outW, outH) ? 1 : 0;
    paramsF32[12] = vminPct;
    paramsF32[13] = vmaxPct;
    paramsF32[14] = transform?.panX ?? 0;
    paramsF32[15] = transform?.panY ?? 0;
    this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

    const rangeBuffer = this.ranges.ensureSlotRangeBuffer(slot);
    const bindGroup = slot.liveRange!.renderGroup ??= this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: slot.paramsBuffer } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: this.engine.lutBuffer } },
        { binding: 3, resource: { buffer: rangeBuffer } },
      ],
    });

    const texture = ctx.getCurrentTexture();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: clearColorFromRgb(opts.bgRgb),
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    return true;
  }

  private renderPackedPanelTransformComputeToCanvas(
    slot: GPUSlot,
    range: { vmin: number; vmax: number } | { vmin: number; vmax: number }[],
    logScale: boolean | boolean[],
    ctx: GPUCanvasContext,
    opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      sourcePanelWidth: number;
      transforms?: { zoom: number; panX: number; panY: number }[];
      sourcePanelIndices?: number[];
      smooth?: boolean;
    },
  ): boolean {
    if (!this.engine.lutBuffer) return false;
    const outW = Math.max(1, Math.round(opts.width));
    const outH = Math.max(1, Math.round(opts.height));
    const n = Math.max(1, Math.round(opts.panelCount));
    const cols = Math.max(1, Math.round(opts.cols));
    const rows = Math.max(1, Math.round(opts.rows));
    const gap = Math.max(0, opts.gap);
    const panelW = (outW - gap * (cols - 1)) / cols;
    const panelH = (outH - gap * (rows - 1)) / rows;
    if (panelW <= 0 || panelH <= 0) return false;
    if (slot.rgbaCapacity < outW * outH) return false;

    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ensurePackedPanelTransformPipeline();
    this.engine.ensureBlitPipeline(format);
    const pipeline = this.packedPanelTransformPipeline;
    const blitPipeline = this.engine.blitPipeline;
    if (!pipeline || !blitPipeline) return false;

    const sourcePanelW = Math.max(1, Math.min(slot.width, Math.round(opts.sourcePanelWidth)));
    const rangeBytes = n * 16;
    if (!this.packedPanelRangesBuffer || this.packedPanelRangesCapacity < rangeBytes) {
      this.packedPanelRangesBuffer?.destroy();
      this.packedPanelRangesBuffer = this.device.createBuffer({
        size: rangeBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.packedPanelRangesCapacity = rangeBytes;
    }
    if (!this.packedPanelTransformsBuffer || this.packedPanelTransformsCapacity < rangeBytes) {
      this.packedPanelTransformsBuffer?.destroy();
      this.packedPanelTransformsBuffer = this.device.createBuffer({
        size: rangeBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.packedPanelTransformsCapacity = rangeBytes;
    }
    const indexBytes = n * 4;
    if (!this.packedPanelIndicesBuffer || this.packedPanelIndicesCapacity < indexBytes) {
      this.packedPanelIndicesBuffer?.destroy();
      this.packedPanelIndicesBuffer = this.device.createBuffer({
        size: indexBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.packedPanelIndicesCapacity = indexBytes;
    }

    const packedRanges = new Float32Array(n * 4);
    const packedTransforms = new Float32Array(n * 4);
    const packedIndices = new Uint32Array(n);
    let useSmooth = false;
    let usesExplicitSourcePanels = false;
    for (let panel = 0; panel < n; panel++) {
      const panelRange = Array.isArray(range) ? (range[panel] ?? range[0]) : range;
      const panelLogScale = Array.isArray(logScale) ? !!logScale[panel] : logScale;
      const transform = opts.transforms?.[panel];
      const zoomValue = Math.max(1e-6, transform?.zoom ?? 1);
      packedRanges[panel * 4] = panelRange.vmin;
      packedRanges[panel * 4 + 1] = panelRange.vmax;
      packedRanges[panel * 4 + 2] = panelLogScale ? 1 : 0;
      packedRanges[panel * 4 + 3] = 0;
      packedTransforms[panel * 4] = zoomValue;
      packedTransforms[panel * 4 + 1] = transform?.panX ?? 0;
      packedTransforms[panel * 4 + 2] = transform?.panY ?? 0;
      packedTransforms[panel * 4 + 3] = 0;
      const sourcePanel = Math.max(0, Math.round(opts.sourcePanelIndices?.[panel] ?? panel));
      packedIndices[panel] = sourcePanel;
      usesExplicitSourcePanels = usesExplicitSourcePanels || sourcePanel !== panel;
      useSmooth = useSmooth
        || shouldSmoothDirectSample(opts.smooth, zoomValue, sourcePanelW, slot.height, panelW, panelH);
    }
    this.device.queue.writeBuffer(this.packedPanelRangesBuffer, 0, packedRanges);
    this.device.queue.writeBuffer(this.packedPanelTransformsBuffer, 0, packedTransforms);
    this.device.queue.writeBuffer(this.packedPanelIndicesBuffer, 0, packedIndices);

    const params = this.directGridParams;
    const paramsU32 = this.directGridParamsU32;
    const paramsF32 = this.directGridParamsF32;
    paramsU32[0] = slot.width;
    paramsU32[1] = slot.height;
    paramsU32[2] = sourcePanelW;
    paramsU32[3] = outW;
    paramsU32[4] = outH;
    paramsU32[5] = n;
    paramsU32[6] = cols;
    paramsU32[7] = rows;
    paramsU32[8] = 0;
    paramsU32[9] = opts.bgRgb & 0xFFFFFF;
    paramsU32[10] = useSmooth ? 1 : 0;
    paramsU32[11] = usesExplicitSourcePanels ? 1 : 0;
    paramsF32[12] = gap;
    paramsF32[13] = 0;
    paramsF32[14] = 0;
    paramsF32[15] = 0;
    this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);
    this.device.queue.writeBuffer(slot.blitParamsBuffer, 0, new Uint32Array([outW, outH]));

    const computeGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: slot.paramsBuffer } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: this.engine.lutBuffer } },
        { binding: 3, resource: { buffer: this.packedPanelRangesBuffer } },
        { binding: 4, resource: { buffer: this.packedPanelTransformsBuffer } },
        { binding: 5, resource: { buffer: this.packedPanelIndicesBuffer } },
        { binding: 6, resource: { buffer: slot.rgbaBuffer } },
      ],
    });
    const blitGroup = this.device.createBindGroup({
      layout: blitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: slot.blitParamsBuffer } },
        { binding: 1, resource: { buffer: slot.rgbaBuffer } },
      ],
    });

    const texture = ctx.getCurrentTexture();
    const encoder = this.device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, computeGroup);
    computePass.dispatchWorkgroups(Math.ceil(outW / 16), Math.ceil(outH / 16));
    computePass.end();

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: clearColorFromRgb(opts.bgRgb),
      }],
    });
    renderPass.setPipeline(blitPipeline);
    renderPass.setBindGroup(0, blitGroup);
    renderPass.draw(3);
    renderPass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  renderCombinedPanelRegionsDirectToCanvas(
    slotIdx: number,
    range: { vmin: number; vmax: number } | { vmin: number; vmax: number }[],
    logScale: boolean | boolean[],
    ctx: GPUCanvasContext,
    opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      sourcePanelWidth: number;
      transforms?: { zoom: number; panX: number; panY: number }[];
      sourcePanelIndices?: number[];
      panelLuts?: { name: string; lut: Uint8Array }[];
      smooth?: boolean;
    },
  ): boolean {
    if (!this.engine.lutBuffer) return false;
    const slot = this.engine.slots[slotIdx];
    if (!slot) return false;
    const hasActiveTransform = (opts.transforms || []).some((transform) => (
      Math.abs((transform?.zoom ?? 1) - 1) > 1e-6 ||
      Math.abs(transform?.panX ?? 0) > 1e-3 ||
      Math.abs(transform?.panY ?? 0) > 1e-3
    ));
    // The compute shortcut binds the one shared LUT. Any explicit panel LUT
    // contract must use the per-panel fragment bindings, even when all named
    // panel maps currently happen to be identical.
    const hasPanelLuts = Boolean(opts.panelLuts?.length);
    if (!hasActiveTransform && !hasPanelLuts && this.renderPackedPanelTransformComputeToCanvas(slot, range, logScale, ctx, opts)) {
      return true;
    }
    const outW = Math.max(1, Math.round(opts.width));
    const outH = Math.max(1, Math.round(opts.height));
    const n = Math.max(1, Math.round(opts.panelCount));
    const cols = Math.max(1, Math.round(opts.cols));
    const rows = Math.max(1, Math.round(opts.rows));
    const gap = Math.max(0, opts.gap);
    const panelW = (outW - gap * (cols - 1)) / cols;
    const panelH = (outH - gap * (rows - 1)) / rows;
    if (panelW <= 0 || panelH <= 0) return false;

    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ensureDirectSlotPipeline(format);
    const pipeline = this.directSlotPipeline;
    if (!pipeline) return false;

    const params = this.directGridParams;
    const paramsU32 = this.directGridParamsU32;
    const paramsF32 = this.directGridParamsF32;
    const sourcePanelW = Math.max(1, Math.min(slot.width, Math.round(opts.sourcePanelWidth)));
    while (slot.directRegionParamsBuffers.length < n) {
      slot.directRegionParamsBuffers.push(null);
      slot.directRegionBindGroups.push(null);
    }
    for (let panel = 0; panel < n; panel++) {
      let paramsBuffer = slot.directRegionParamsBuffers[panel];
      if (!paramsBuffer) {
        paramsBuffer = this.device.createBuffer({
          size: 64,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        slot.directRegionParamsBuffers[panel] = paramsBuffer;
      }
      const panelLut = opts.panelLuts?.[panel];
      const lutName = panelLut?.name || this.engine.currentLutName;
      const lutBuffer = panelLut
        ? this.engine.namedLutBuffer(panelLut.name, panelLut.lut)
        : this.engine.lutBuffer;
      if (!lutBuffer) continue;
      const panelRange = Array.isArray(range) ? (range[panel] ?? range[0]) : range;
      const panelLogScale = Array.isArray(logScale) ? !!logScale[panel] : logScale;
      const sourcePanel = Math.max(0, Math.round(opts.sourcePanelIndices?.[panel] ?? panel));
      const srcX0 = Math.min(sourcePanel * sourcePanelW, Math.max(0, slot.width - 1));
      const sourceWidth = Math.max(1, Math.min(sourcePanelW, slot.width - srcX0));
      paramsU32[0] = slot.width;
      paramsU32[1] = slot.height;
      paramsU32[2] = srcX0;
      paramsU32[3] = sourceWidth;
      paramsU32[4] = Math.max(1, Math.round(panelH));
      paramsU32[5] = Math.max(1, Math.round(panelW));
      const col = panel % cols;
      const row = Math.floor(panel / cols);
      paramsF32[6] = col * (panelW + gap);
      paramsF32[7] = row * (panelH + gap);
      paramsU32[8] = panelLogScale ? 1 : 0;
      paramsU32[9] = opts.bgRgb & 0xFFFFFF;
      paramsU32[10] = 1;
      const transform = opts.transforms?.[panel];
      const zoomValue = Math.max(1e-6, transform?.zoom ?? 1);
      paramsF32[10] = zoomValue;
      paramsU32[11] = shouldSmoothDirectSample(opts.smooth, zoomValue, sourceWidth, slot.height, panelW, panelH) ? 1 : 0;
      paramsF32[12] = panelRange.vmin;
      paramsF32[13] = panelRange.vmax;
      paramsF32[14] = transform?.panX ?? 0;
      paramsF32[15] = transform?.panY ?? 0;
      this.device.queue.writeBuffer(paramsBuffer, 0, params);
      if (
        !slot.directRegionBindGroups[panel]
        || slot.directRegionLutNames[panel] !== lutName
      ) {
        slot.directRegionBindGroups[panel] = this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: { buffer: slot.dataBuffer } },
            { binding: 2, resource: { buffer: lutBuffer } },
          ],
        });
        slot.directRegionLutNames[panel] = lutName;
      }
    }

    const texture = ctx.getCurrentTexture();
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: clearColorFromRgb(opts.bgRgb),
      }],
    });
    pass.setPipeline(pipeline);
    for (let panel = 0; panel < n; panel++) {
      const bindGroup = slot.directRegionBindGroups[panel];
      if (!bindGroup) continue;
      const col = panel % cols;
      const row = Math.floor(panel / cols);
      const x = col * (panelW + gap);
      const y = row * (panelH + gap);
      const sx = Math.max(0, Math.floor(x));
      const sy = Math.max(0, Math.floor(y));
      const sw = Math.max(1, Math.min(Math.ceil(panelW), outW - sx));
      const sh = Math.max(1, Math.min(Math.ceil(panelH), outH - sy));
      pass.setViewport(x, y, panelW, panelH, 0, 1);
      pass.setScissorRect(sx, sy, sw, sh);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  /** Queue-safe durable-frame twin of renderCombinedPanelRegionsDirectToCanvas. */
  async renderCombinedPanelRegionsToImageBitmapAsync(
    slotIdx: number,
    range: { vmin: number; vmax: number } | { vmin: number; vmax: number }[],
    logScale: boolean | boolean[],
    opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      sourcePanelWidth: number;
      transforms?: { zoom: number; panX: number; panY: number }[];
      sourcePanelIndices?: number[];
      panelLuts?: { name: string; lut: Uint8Array }[];
      smooth?: boolean;
    },
  ): Promise<ImageBitmap | null> {
    const canvas = new OffscreenCanvas(
      Math.max(1, Math.round(opts.width)),
      Math.max(1, Math.round(opts.height)),
    );
    const context = canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!context) return null;
    context.configure({
      device: this.device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: "opaque",
    });
    if (!this.renderCombinedPanelRegionsDirectToCanvas(
      slotIdx,
      range,
      logScale,
      context,
      opts,
    )) return null;
    await this.device.queue.onSubmittedWorkDone();
    return canvas.transferToImageBitmap();
  }
}
