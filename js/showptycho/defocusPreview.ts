// Full-field phase samples stay on the hardware adapter. Only the held C10
// value changes between frames. A release always goes to DirectPtycho.
import { computePipeline } from "../display/pipelines";

const shader = `
struct Params { t: f32, sign: f32, count: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read> c: array<u32>;
@group(0) @binding(3) var<storage, read> d: array<u32>;
@group(0) @binding(4) var<storage, read_write> phase: array<f32>;
@group(0) @binding(5) var<uniform> p: Params;
fn wrap(v: vec2<f32>) -> vec2<f32> {return v - 6.283185307 * floor((v + 3.141592654) / 6.283185307);}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (2u*i >= p.count) {return;}
  let b0 = unpack2x16float(b[i]);
  let a0 = wrap(unpack2x16float(a[i]) - b0);
  let c0 = wrap(unpack2x16float(c[i]) - b0);
  let d0 = c0 + wrap(unpack2x16float(d[i]) - unpack2x16float(c[i]));
  let t = p.t;
  let v = p.sign * wrap(b0 + 0.5*t*((c0-a0) + t*((2.0*a0+4.0*c0-d0) + t*(d0-a0-3.0*c0))));
  phase[2u*i] = v.x;
  if (2u*i+1u < p.count) {phase[2u*i+1u] = v.y;}
}`;

export class DefocusPreview {
  private planes: GPUBuffer[] = [];
  private pipeline: GPUComputePipeline;
  private uniform: GPUBuffer;
  constructor(private device: GPUDevice, readonly start: number, readonly count: number,
              readonly pixels: number, readonly revision: number, readonly key: string) {
    this.pipeline = computePipeline(device, shader);
    this.uniform = device.createBuffer({size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  }
  get loaded() { return this.planes.length; }
  add(bytes: DataView) {
    const size = 4 * Math.ceil(this.pixels / 2);
    if (bytes.byteLength !== size || this.loaded >= this.count) throw new Error("Unexpected defocus preview plane.");
    const buffer = this.device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    this.planes.push(buffer);
    this.device.queue.writeBuffer(buffer, 0, bytes.buffer as ArrayBuffer, bytes.byteOffset, size);
  }
  draw(value: number, sign: number, out: GPUBuffer): boolean {
    const x = value - this.start, i = Math.min(Math.floor(x), this.count - 3);
    if (this.loaded !== this.count || x < 1 || x > this.count - 2) return false;
    const params = new ArrayBuffer(16);
    new Float32Array(params).set([x-i, sign]);
    new Uint32Array(params)[2] = this.pixels;
    this.device.queue.writeBuffer(this.uniform, 0, params);
    const buffers = [...this.planes.slice(i-1,i+3), out, this.uniform];
    const group = this.device.createBindGroup({layout: this.pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer,binding)=>({binding, resource:{buffer}}))});
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(this.pixels / 512)); pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }
  destroy() { for (const plane of this.planes) plane.destroy(); this.planes = []; this.uniform.destroy(); }
}
