// Volume-resident orthogonal and oblique slices for Show3DSlices: the volume
// is uploaded once and each scrub runs one slice + colormap compute pass.
import type { GPUColormapEngine } from "./colormapEngine";
import { DISPLAY_NORMALIZE_WGSL } from "./displayShaders";

// Smallest zoom of a volume slice view, Show3DSlices' MIN_ZOOM. The area
// average reads every source pixel an output pixel covers, (fullW / canvasW /
// zoom) per axis, so the clamp is what bounds that work per pixel.
export const VOLUME_SLICE_MIN_ZOOM = 0.5;

// Volume-resident orthogonal slice + colormap in ONE compute pass. The whole 3D
// volume lives on the GPU (uploaded once), in a storage buffer or, for 4k and
// row-padded stacks, a 2D texture array; per scrub only a tiny uniform (axis +
// slice index + vmin/vmax) changes, so there is NO per-frame CPU slice
// extraction and NO per-frame volume re-upload. axis: 0=XY(z fixed), 1=XZ(y
// fixed), 2=YZ(x fixed), 3=oblique cut along the alignment segment. Order
// matches the CPU path: log THEN flip.
function volumeSliceShader(source: "buffer" | "texture"): string {
  const volumeBinding = source === "buffer"
    ? "var<storage, read> vol: array<f32>"
    : "var volTex: texture_2d_array<f32>";
  const volumeLoad = source === "buffer"
    ? "vol[z * p.ny * p.nx + y * p.nx + x]"
    : "textureLoad(volTex, vec2<i32>(i32(x), i32(y)), i32(z), 0).r";
  return DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct VParams {
  nx: u32, ny: u32, nz: u32, axis: u32,
  index: u32, outW: u32, outH: u32, logScale: u32,
  flip: u32, viewMode: u32, canvasW: u32, canvasH: u32,
  vmin: f32, vmax: f32, zoom: f32, panX: f32,
  panY: f32, rowShift: f32, colShift: f32, segStartX: f32,
  segStartY: f32, segStopX: f32, segStopY: f32, _p0: f32,
};
@group(0) @binding(0) var<uniform> p: VParams;
@group(0) @binding(1) ${volumeBinding};
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> rgba: array<u32>;

fn loadVolume(z: u32, x: u32, y: u32) -> f32 {
  return ${volumeLoad};
}

fn sampleAligned(zRaw: f32, xRaw: f32, yRaw: f32) -> f32 {
  let z = u32(clamp(round(zRaw), 0.0, f32(p.nz - 1u)));
  let center = (f32(max(p.nz, 1u)) - 1.0) * 0.5;
  let x = clamp(xRaw - (f32(z) - center) * p.colShift, 0.0, f32(p.nx - 1u));
  let y = clamp(yRaw - (f32(z) - center) * p.rowShift, 0.0, f32(p.ny - 1u));
  let x0 = u32(floor(x));
  let y0 = u32(floor(y));
  let x1 = min(p.nx - 1u, x0 + 1u);
  let y1 = min(p.ny - 1u, y0 + 1u);
  let tx = x - f32(x0);
  let ty = y - f32(y0);
  let v00 = loadVolume(z, x0, y0);
  let v10 = loadVolume(z, x1, y0);
  let v01 = loadVolume(z, x0, y1);
  let v11 = loadVolume(z, x1, y1);
  return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
}

fn sampleSlice(p_axis: u32, p_index: u32, sliceX: u32, sliceY: u32) -> f32 {
  var x: f32; var y: f32; var z: f32;
  if (p_axis == 0u) { x = f32(sliceX); y = f32(sliceY); z = f32(p_index); }          // XY
  else if (p_axis == 1u) { x = f32(sliceX); y = f32(p_index); z = f32(sliceY); }     // XZ
  else if (p_axis == 2u) { x = f32(p_index); y = f32(sliceX); z = f32(sliceY); }     // YZ
  else {
    let denom = max(f32(p.canvasW - 1u), 1.0);
    let t = f32(sliceX) / denom;
    x = mix(p.segStartX, p.segStopX, t);
    y = mix(p.segStartY, p.segStopY, t);
    z = f32(sliceY);
  }
  return sampleAligned(z, x, y);
}

fn signedLog1p(v: f32) -> f32 {
  if (v >= 0.0) { return log(1.0 + v); }
  return -log(1.0 - v);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.outW || gid.y >= p.outH) { return; }
  // Full slice dims for this axis (source resolution).
  var fullW: u32; var fullH: u32;
  if (p.axis == 0u) { fullW = p.nx; fullH = p.ny; }        // XY
  else if (p.axis == 1u) { fullW = p.nx; fullH = p.nz; }   // XZ
  else if (p.axis == 2u) { fullW = p.ny; fullH = p.nz; }    // YZ
  else { fullW = max(1u, p.canvasW); fullH = p.nz; }        // oblique
  var x0: u32; var y0: u32; var x1: u32; var y1: u32;
  if (p.viewMode == 0u) {
    // AREA AVERAGE downsample: this output pixel covers the source block
    // [x0,x1) x [y0,y1). Average every covered source value, so no source pixels
    // are silently skipped when a 4k slice is displayed in a smaller panel. When
    // outW==fullW the block is 1x1 = exact native pixel.
    x0 = (gid.x * fullW) / p.outW;
    y0 = (gid.y * fullH) / p.outH;
    x1 = max(x0 + 1u, ((gid.x + 1u) * fullW) / p.outW);
    y1 = max(y0 + 1u, ((gid.y + 1u) * fullH) / p.outH);
  } else {
    let cw = max(f32(p.canvasW), 1.0);
    let ch = max(f32(p.canvasH), 1.0);
    let z = max(p.zoom, ${VOLUME_SLICE_MIN_ZOOM});
    let cx = cw * 0.5;
    let cy = ch * 0.5;
    let sx0 = (((f32(gid.x) - cx - p.panX) / z) + cx) * f32(fullW) / cw;
    let sy0 = (((f32(gid.y) - cy - p.panY) / z) + cy) * f32(fullH) / ch;
    let sx1 = (((f32(gid.x + 1u) - cx - p.panX) / z) + cx) * f32(fullW) / cw;
    let sy1 = (((f32(gid.y + 1u) - cy - p.panY) / z) + cy) * f32(fullH) / ch;
    let loX = min(sx0, sx1);
    let hiX = max(sx0, sx1);
    let loY = min(sy0, sy1);
    let hiY = max(sy0, sy1);
    if (hiX <= 0.0 || hiY <= 0.0 || loX >= f32(fullW) || loY >= f32(fullH)) {
      rgba[gid.y * p.outW + gid.x] = 0xFF000000u;
      return;
    }
    x0 = u32(clamp(floor(loX), 0.0, f32(fullW - 1u)));
    y0 = u32(clamp(floor(loY), 0.0, f32(fullH - 1u)));
    x1 = max(x0 + 1u, u32(clamp(ceil(hiX), 1.0, f32(fullW))));
    y1 = max(y0 + 1u, u32(clamp(ceil(hiY), 1.0, f32(fullH))));
  }
  var sum = 0.0; var cnt = 0.0;
  for (var yy = y0; yy < y1; yy++) {
    for (var xx = x0; xx < x1; xx++) {
      sum = sum + sampleSlice(p.axis, p.index, min(xx, fullW - 1u), min(yy, fullH - 1u));
      cnt = cnt + 1.0;
    }
  }
  var val = sum / max(cnt, 1.0);
  if (p.logScale == 1u) { val = signedLog1p(val); }
  if (p.flip == 1u) { val = -val; }
  let t = display_normalize(val, p.vmin, p.vmax);
  let li = min(u32(t * 255.0), 255u);
  rgba[gid.y * p.outW + gid.x] = lut[li] | 0xFF000000u;
}
`;
}

const VOLUME_PARAMS_BYTES = 96;

interface VolumeSliceView {
  zoom: number;
  panX: number;
  panY: number;
  canvasW: number;
  canvasH: number;
}

interface VolumeSliceAlignment {
  rowShift: number;
  colShift: number;
  segment?: {
    start: { x: number; y: number };
    stop: { x: number; y: number };
  };
}

/**
 * Volume-resident slice pipeline (Show3DSlices): volume uploaded once, slice +
 * colormap done on GPU per scrub - no per-frame CPU extract / re-upload. It
 * paints with the engine's current LUT and blit pass.
 */
export class VolumeSlices {
  private readonly engine: GPUColormapEngine;
  private readonly device: GPUDevice;
  private volumePipeline: GPUComputePipeline | null = null;
  private volumeTexturePipeline: GPUComputePipeline | null = null;
  private volumeBuffer: GPUBuffer | null = null;
  private volumeTexture: GPUTexture | null = null;
  private volTextureView: GPUTextureView | null = null;
  private volUseTexture = false;
  private volTextureWidth = 0;
  private volNx = 0;
  private volNy = 0;
  private volNz = 0;
  private volCount = 0;
  private volParamsBuffer: GPUBuffer | null = null;
  private volRgbaBuffer: GPUBuffer | null = null;
  private volRgbaCapacity = 0;
  private volParams = new ArrayBuffer(VOLUME_PARAMS_BYTES);
  private volParamsU32 = new Uint32Array(this.volParams);
  private volParamsF32 = new Float32Array(this.volParams);
  private volBlitCanvas: OffscreenCanvas | null = null;
  private volBlitContext: GPUCanvasContext | null = null;
  private volBlitFormat: GPUTextureFormat | null = null;
  private volBlitWidth = 0;
  private volBlitHeight = 0;
  private volBlitParamsBuffer: GPUBuffer | null = null;
  private volBlitParams = new Uint32Array(2);
  private volBlitBindGroup: GPUBindGroup | null = null;
  volComputeBindGroup: GPUBindGroup | null = null;
  volTextureBindGroup: GPUBindGroup | null = null;

  constructor(engine: GPUColormapEngine) {
    this.engine = engine;
    this.device = engine.getDevice();
  }

  private ensureVolumePipeline(): void {
    if (this.volumePipeline) return;
    const module = this.device.createShaderModule({ code: volumeSliceShader("buffer") });
    this.volumePipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    if (!this.volParamsBuffer) {
      this.volParamsBuffer = this.device.createBuffer({
        size: VOLUME_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
  }

  private ensureVolumeTexturePipeline(): void {
    if (this.volumeTexturePipeline) return;
    const module = this.device.createShaderModule({ code: volumeSliceShader("texture") });
    this.volumeTexturePipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    if (!this.volParamsBuffer) {
      this.volParamsBuffer = this.device.createBuffer({
        size: VOLUME_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
  }

  /**
   * Upload a 3D volume (nz, ny, nx) row-major float32 into GPU memory once.
   * 4k and row-padded large stacks use a 2D texture array to avoid
   * storage-buffer binding limits. The texture path keeps original float32
   * values; unaligned rows are padded only in the upload stride and never
   * sampled. Other shapes use the storage-buffer path.
   */
  uploadVolume(vol: Float32Array, nx: number, ny: number, nz: number): boolean {
    const rowBytes = nx * 4;
    const paddedRowBytes = Math.ceil(rowBytes / 256) * 256;
    const textureWidth = paddedRowBytes / 4;
    const canTexture = textureWidth <= this.device.limits.maxTextureDimension2D &&
      ny <= this.device.limits.maxTextureDimension2D &&
      nz <= this.device.limits.maxTextureArrayLayers;
    if (canTexture) {
      try {
        this.ensureVolumeTexturePipeline();
        const needsTexture = !this.volumeTexture || this.volCount !== vol.length ||
          this.volNx !== nx || this.volNy !== ny || this.volNz !== nz ||
          this.volTextureWidth !== textureWidth;
        if (needsTexture) {
          this.volumeTexture?.destroy();
          this.volumeTexture = this.device.createTexture({
            size: { width: textureWidth, height: ny, depthOrArrayLayers: nz },
            format: "r32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
          });
          this.volTextureView = this.volumeTexture.createView({ dimension: "2d-array" });
          this.volTextureBindGroup = null;
          this.volCount = vol.length;
        }
        const texture = this.volumeTexture;
        if (!texture) return false;
        if (paddedRowBytes === rowBytes) {
          this.device.queue.writeTexture(
            { texture },
            vol.buffer as ArrayBuffer,
            { offset: vol.byteOffset, bytesPerRow: rowBytes, rowsPerImage: ny },
            { width: nx, height: ny, depthOrArrayLayers: nz },
          );
        } else {
          const layer = new Float32Array(textureWidth * ny);
          const sliceStride = nx * ny;
          for (let z = 0; z < nz; z++) {
            const sliceStart = z * sliceStride;
            for (let row = 0; row < ny; row++) {
              const rowStart = sliceStart + row * nx;
              layer.set(vol.subarray(rowStart, rowStart + nx), row * textureWidth);
            }
            this.device.queue.writeTexture(
              { texture, origin: { x: 0, y: 0, z } },
              layer.buffer,
              { bytesPerRow: paddedRowBytes, rowsPerImage: ny },
              { width: nx, height: ny, depthOrArrayLayers: 1 },
            );
          }
        }
        this.volUseTexture = true;
        this.volumeBuffer?.destroy();
        this.volumeBuffer = null;
        this.volComputeBindGroup = null;
        this.volTextureWidth = textureWidth;
        this.volNx = nx; this.volNy = ny; this.volNz = nz;
        return true;
      } catch {
        this.volumeTexture?.destroy();
        this.volumeTexture = null;
        this.volTextureView = null;
        this.volTextureBindGroup = null;
        this.volUseTexture = false;
        this.volTextureWidth = 0;
      }
    }
    this.ensureVolumePipeline();
    this.volumeTexture?.destroy();
    this.volumeTexture = null;
    this.volTextureView = null;
    this.volTextureBindGroup = null;
    this.volTextureWidth = 0;
    const maxBind = this.device.limits.maxStorageBufferBindingSize;
    if (vol.byteLength > maxBind) return false;
    if (!this.volumeBuffer || this.volCount !== vol.length) {
      this.volumeBuffer?.destroy();
      this.volumeBuffer = this.device.createBuffer({
        size: vol.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.volCount = vol.length;
      this.volComputeBindGroup = null;
    }
    this.device.queue.writeBuffer(this.volumeBuffer, 0, vol.buffer as ArrayBuffer, vol.byteOffset, vol.byteLength);
    this.volUseTexture = false;
    this.volNx = nx; this.volNy = ny; this.volNz = nz;
    return true;
  }

  /**
   * Slice the resident volume along `axis` (0=XY, 1=XZ, 2=YZ) at `index`,
   * colormap with the current LUT + vmin/vmax (logScale/flip applied in-shader to
   * match the CPU path), and blit to an ImageBitmap. Returns null if the volume
   * isn't uploaded or the LUT/pipeline isn't ready (caller falls back to CPU).
   */
  renderVolumeSliceToImageBitmap(
    axis: number, index: number,
    range: { vmin: number; vmax: number },
    logScale: boolean, flip: boolean,
    maxOut?: number,
    view?: VolumeSliceView,
    alignment?: VolumeSliceAlignment,
  ): ImageBitmap | null {
    const texturePipeline = this.volUseTexture ? this.volumeTexturePipeline : null;
    const textureView = this.volUseTexture ? this.volTextureView : null;
    const useTexture = texturePipeline != null && textureView != null;
    const bufferPipeline = this.volumePipeline;
    const useBuffer = !useTexture && this.volumeBuffer != null && bufferPipeline != null;
    if ((!useTexture && !useBuffer) || !this.engine.lutBuffer || !this.volParamsBuffer) return null;
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.engine.ensureBlitPipeline(format);
    if (!this.engine.blitPipeline) return null;
    const nx = this.volNx, ny = this.volNy, nz = this.volNz;
    const segmentWidth = alignment?.segment
      ? Math.max(1, Math.ceil(Math.hypot(
        alignment.segment.stop.x - alignment.segment.start.x,
        alignment.segment.stop.y - alignment.segment.start.y,
      )) + 1)
      : 1;
    const fullW = axis === 0 ? nx : axis === 1 ? nx : axis === 2 ? ny : segmentWidth;
    const fullH = axis === 0 ? ny : nz;
    // If maxOut is supplied, cap the output raster while still sampling from
    // the full-resolution source. Callers that need native-pixel zoom leave it
    // undefined, so outW/outH stay at the full slice dimensions.
    const cap = maxOut && maxOut > 0 ? maxOut : Math.max(fullW, fullH);
    const scale = Math.min(1, cap / Math.max(fullW, fullH));
    const outW = view ? Math.max(1, Math.round(view.canvasW)) : Math.max(1, Math.round(fullW * scale));
    const outH = view ? Math.max(1, Math.round(view.canvasH)) : Math.max(1, Math.round(fullH * scale));
    const sliceIndex = Math.max(0, Math.min((axis === 2 ? nx : axis === 1 ? ny : nz) - 1, Math.round(index)));
    const rgbaCount = outW * outH;
    if (!this.volRgbaBuffer || this.volRgbaCapacity < rgbaCount) {
      this.volRgbaBuffer?.destroy();
      this.volRgbaBuffer = this.device.createBuffer({
        size: rgbaCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.volRgbaCapacity = rgbaCount;
      this.volBlitBindGroup = null;
      this.volComputeBindGroup = null;
      this.volTextureBindGroup = null;
    }
    const paramsBuffer = this.volParamsBuffer;
    const lutBuffer = this.engine.lutBuffer;
    const rgbaBuffer = this.volRgbaBuffer;
    if (!paramsBuffer || !lutBuffer || !rgbaBuffer) return null;
    // VParams: u32 control block + float contrast / viewport block.
    const paramsU32 = this.volParamsU32; const paramsF32 = this.volParamsF32;
    paramsU32[0] = nx; paramsU32[1] = ny; paramsU32[2] = nz; paramsU32[3] = axis;
    paramsU32[4] = sliceIndex; paramsU32[5] = outW; paramsU32[6] = outH; paramsU32[7] = logScale ? 1 : 0;
    paramsU32[8] = flip ? 1 : 0;
    paramsU32[9] = view ? 1 : 0;
    paramsU32[10] = view ? Math.max(1, Math.round(view.canvasW)) : outW;
    paramsU32[11] = view ? Math.max(1, Math.round(view.canvasH)) : outH;
    paramsF32[12] = range.vmin; paramsF32[13] = range.vmax;
    paramsF32[14] = view ? Math.max(1e-6, view.zoom) : 1;
    paramsF32[15] = view ? view.panX : 0;
    paramsF32[16] = view ? view.panY : 0;
    paramsF32[17] = alignment && Number.isFinite(alignment.rowShift) ? alignment.rowShift : 0;
    paramsF32[18] = alignment && Number.isFinite(alignment.colShift) ? alignment.colShift : 0;
    paramsF32[19] = alignment?.segment ? alignment.segment.start.x : 0;
    paramsF32[20] = alignment?.segment ? alignment.segment.start.y : 0;
    paramsF32[21] = alignment?.segment ? alignment.segment.stop.x : 0;
    paramsF32[22] = alignment?.segment ? alignment.segment.stop.y : 0;
    paramsF32[23] = 0;
    this.device.queue.writeBuffer(paramsBuffer, 0, this.volParams);
    const encoder = this.device.createCommandEncoder();
    if (useTexture) {
      if (!this.volTextureBindGroup) {
        this.volTextureBindGroup = this.device.createBindGroup({
          layout: texturePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: textureView },
            { binding: 2, resource: { buffer: lutBuffer } },
            { binding: 3, resource: { buffer: rgbaBuffer } },
          ],
        });
      }
    } else if (!this.volComputeBindGroup && bufferPipeline && this.volumeBuffer) {
      this.volComputeBindGroup = this.device.createBindGroup({
        layout: bufferPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: this.volumeBuffer } },
          { binding: 2, resource: { buffer: lutBuffer } },
          { binding: 3, resource: { buffer: rgbaBuffer } },
        ],
      });
    }
    const computePass = encoder.beginComputePass();
    if (useTexture) {
      computePass.setPipeline(texturePipeline);
      computePass.setBindGroup(0, this.volTextureBindGroup!);
    } else {
      if (!bufferPipeline || !this.volComputeBindGroup) { computePass.end(); return null; }
      computePass.setPipeline(bufferPipeline);
      computePass.setBindGroup(0, this.volComputeBindGroup);
    }
    computePass.dispatchWorkgroups(Math.ceil(outW / 16), Math.ceil(outH / 16));
    computePass.end();
    const sizeChanged = !this.volBlitCanvas || this.volBlitWidth !== outW || this.volBlitHeight !== outH;
    const formatChanged = this.volBlitFormat !== format;
    if (sizeChanged || formatChanged || !this.volBlitContext) {
      this.volBlitCanvas = new OffscreenCanvas(outW, outH);
      this.volBlitContext = this.volBlitCanvas.getContext("webgpu") as GPUCanvasContext | null;
      if (!this.volBlitContext) return null;
      this.volBlitContext.configure({ device: this.device, format, alphaMode: "opaque" });
      this.volBlitWidth = outW;
      this.volBlitHeight = outH;
      this.volBlitFormat = format;
    }
    if (!this.volBlitParamsBuffer) {
      this.volBlitParamsBuffer = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    this.volBlitParams[0] = outW;
    this.volBlitParams[1] = outH;
    this.device.queue.writeBuffer(this.volBlitParamsBuffer, 0, this.volBlitParams);
    if (!this.volBlitBindGroup) {
      this.volBlitBindGroup = this.device.createBindGroup({
        layout: this.engine.blitPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.volBlitParamsBuffer } },
          { binding: 1, resource: { buffer: rgbaBuffer } },
        ],
      });
    }
    const blitContext = this.volBlitContext;
    const blitCanvas = this.volBlitCanvas;
    if (!blitContext || !blitCanvas) return null;
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{ view: blitContext.getCurrentTexture().createView(), loadOp: "clear" as GPULoadOp, storeOp: "store" as GPUStoreOp, clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    renderPass.setPipeline(this.engine.blitPipeline);
    renderPass.setBindGroup(0, this.volBlitBindGroup);
    renderPass.draw(3);
    renderPass.end();
    this.device.queue.submit([encoder.finish()]);
    return blitCanvas.transferToImageBitmap();
  }

  /** Release the volume and its slice buffers; the engine's destroy calls this. */
  destroy(): void {
    this.volumeBuffer?.destroy(); this.volumeBuffer = null;
    this.volumeTexture?.destroy(); this.volumeTexture = null; this.volTextureView = null;
    this.volParamsBuffer?.destroy(); this.volParamsBuffer = null;
    this.volRgbaBuffer?.destroy(); this.volRgbaBuffer = null;
    this.volBlitParamsBuffer?.destroy(); this.volBlitParamsBuffer = null;
    this.volBlitCanvas = null; this.volBlitContext = null; this.volBlitFormat = null;
    this.volBlitBindGroup = null; this.volComputeBindGroup = null; this.volTextureBindGroup = null; this.volBlitWidth = 0; this.volBlitHeight = 0;
    this.volUseTexture = false;
    this.volCount = 0; this.volRgbaCapacity = 0; this.volTextureWidth = 0;
  }
}
