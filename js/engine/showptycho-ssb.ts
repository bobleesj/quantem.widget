/// <reference types="@webgpu/types" />

import { getGPUDevice, getGPUInfo, isSoftwareGPUAdapter } from "./device";
import { readH5MasterInfo, readH5Volume } from "./h5reader";
import { decodeBslz4ToStack, type Bslz4Spec } from "./bslz4";

const SUPPORTED_SSB_SIZES = [128, 256, 512, 1024] as const;
const MAX_BF_WORKGROUPS_PER_SUBMIT = 256;
const MAX_GQK_CHUNK_BYTES = 1024 * 1024 * 1024;
// Resident G(q,k) budget. Caps how many BF pixels the session keeps resident
// so a full-BF drag on a big scan cannot device-lost a small GPU. Override:
// globalThis.__QUANTEM_SHOWPTYCHO_GQK_BUDGET_GB__ = 8.
const FULL_STACK_GPU_BUDGET_BYTES = 4.5 * 1024 * 1024 * 1024;
const GQK_GPU_BUDGET_BYTES = FULL_STACK_GPU_BUDGET_BYTES;

function gqkBudgetBytes(): number {
  const raw = (globalThis as { __QUANTEM_SHOWPTYCHO_GQK_BUDGET_GB__?: number }).__QUANTEM_SHOWPTYCHO_GQK_BUDGET_GB__;
  if (Number.isFinite(raw) && Number(raw) > 0) return Number(raw) * 1024 * 1024 * 1024;
  return GQK_GPU_BUDGET_BYTES;
}
const REDUCE_BF_GROUP = 32;
const BF_COLUMN_UNPACK_WORKGROUP_X = 32;
const BF_COLUMN_UNPACK_WORKGROUP_Y = 8;

type SupportedSsbSize = typeof SUPPORTED_SSB_SIZES[number];

function supportedSsbSize(n: number): SupportedSsbSize | null {
  return (SUPPORTED_SSB_SIZES as readonly number[]).includes(n) ? n as SupportedSsbSize : null;
}

// Resident G(q,k) storage mode.
// Public contract:
// - "herm" is Exact mode. It stores the Hermitian half-plane, which is
//   bit-identical to the old n x n storage for real-count scan FFTs.
// - "herm16" is compact preview. It keeps the same half-plane and block-quantizes
//   each BF pixel to snorm16 with one f32 scale.
type GqkMode = "herm" | "herm16";

function resolveGqkMode(): GqkMode {
  const g = globalThis as { __QUANTEM_SHOWPTYCHO_GQK_MODE__?: string; location?: Location };
  let raw = g.__QUANTEM_SHOWPTYCHO_GQK_MODE__ || "";
  try {
    const fromUrl = new URLSearchParams(g.location?.search || "").get("gqk");
    if (fromUrl) raw = fromUrl;
  } catch { /* no location in workers */ }
  const key = raw.trim().toLowerCase();
  if (
    key === "herm16" || key === "i16" || key === "quant"
    || key === "preview" || key === "compact"
  ) return "herm16";
  if (key === "herm" || key === "exact" || key === "half") return "herm";
  // Default: Hermitian half-plane. Measured bit-exact against the old n x n path
  // (real-input scan FFT gives G(-q) = conj(G(q)) down to f32 rounding) and
  // faster per reconstruct (half the G(q,k) read bandwidth) at 2x less VRAM.
  return "herm";
}

function storedPlaneFor(n: number): number {
  return n * (n / 2 + 1);
}

function gqkBytesPerValue(mode: GqkMode): number {
  return mode === "herm16" ? 4 : 8;
}

// ---------------------------------------------------------------------------
// No-server local folder source. When the viewer is opened via file:// the
// browser cannot fetch() sibling files, so the user grants the folder once
// (File System Access picker, or a webkitdirectory input) and every read
// below routes through File.slice() instead of HTTP. Range reads stay
// byte-for-byte identical to the Range-server path.
// ---------------------------------------------------------------------------
type LocalFileResolver = (path: string) => Promise<File>;
let localSourceResolver: LocalFileResolver | null = null;

function normaliseSourcePath(path: string): string {
  return path.replace(/^[.][/]/, "").replace(/^[/]+/, "");
}

let localDirHandle: FileSystemDirectoryHandle | null = null;

export function setShowPtychoLocalDirectory(handle: FileSystemDirectoryHandle): void {
  localDirHandle = handle;
  localSourceResolver = async (path: string) => {
    const parts = normaliseSourcePath(path).split("/").filter(Boolean);
    let dir: FileSystemDirectoryHandle = handle;
    for (const part of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(part);
    const fileHandle = await dir.getFileHandle(parts[parts.length - 1]);
    return fileHandle.getFile();
  };
}

export function setShowPtychoLocalFiles(files: ArrayLike<File>): void {
  const byPath = new Map<string, File>();
  const byName = new Map<string, File>();
  for (let i = 0; i < files.length; i++) {
    const file = files[i] as File & { webkitRelativePath?: string };
    const rel = String(file.webkitRelativePath || "");
    if (rel) {
      // webkitdirectory prefixes the picked folder name; strip it so paths
      // match the manifest-relative urls ("source/x.h5").
      byPath.set(rel.split("/").slice(1).join("/") || rel, file);
      byPath.set(rel, file);
    }
    byName.set(file.name, file);
  }
  localSourceResolver = async (path: string) => {
    const clean = normaliseSourcePath(path);
    const hit = byPath.get(clean) || byName.get(clean.split("/").pop() || clean);
    if (!hit) throw new Error(`local folder is missing ${clean}`);
    return hit;
  };
}

export function showPtychoHasLocalSource(): boolean {
  return localSourceResolver !== null;
}

export function showPtychoNeedsLocalSource(): boolean {
  try {
    return globalThis.location?.protocol === "file:" && !localSourceResolver;
  } catch {
    return false;
  }
}

async function readSourceBytes(url: string, byteOffset?: number, byteLength?: number): Promise<Uint8Array> {
  if (localSourceResolver) {
    const file = await localSourceResolver(url);
    const blob = byteOffset != null && byteLength != null
      ? file.slice(byteOffset, byteOffset + byteLength)
      : file;
    return new Uint8Array(await blob.arrayBuffer());
  }
  if (byteOffset != null && byteLength != null) {
    const res = await fetch(url, {
      headers: { Range: `bytes=${byteOffset}-${Math.max(byteOffset, byteOffset + byteLength - 1)}` },
    });
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    return new Uint8Array(await res.arrayBuffer());
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

// ---------------------------------------------------------------------------
// Folder save write-back. Saved review states (JPEG + saves.json) persist
// INSIDE the export folder so they survive relaunch from both the no-server
// double-click path (via the readwrite directory handle) and the CLI-served
// path (via PUT/DELETE handled by serve_sidecar_range.py). Writes are
// restricted to the saves/ subfolder on both transports.
// ---------------------------------------------------------------------------
export function showPtychoFolderWritable(): boolean {
  if (localDirHandle) return true;
  try {
    const proto = globalThis.location?.protocol || "";
    return proto === "http:" || proto === "https:";
  } catch {
    return false;
  }
}

async function savesDirHandle(): Promise<FileSystemDirectoryHandle> {
  if (!localDirHandle) throw new Error("no local directory handle");
  return localDirHandle.getDirectoryHandle("saves", { create: true });
}

export async function writeShowPtychoFolderFile(path: string, body: Blob | string): Promise<void> {
  const clean = normaliseSourcePath(path);
  if (!clean.startsWith("saves/")) throw new Error("folder writes are restricted to saves/");
  if (localDirHandle) {
    const dir = await savesDirHandle();
    const handle = await dir.getFileHandle(clean.slice("saves/".length), { create: true });
    const writable = await handle.createWritable();
    await writable.write(body);
    await writable.close();
    return;
  }
  const res = await fetch(clean, { method: "PUT", body });
  if (!res.ok) throw new Error(`${clean}: HTTP ${res.status} (serve_sidecar_range.py supports saves/ writes; plain servers do not)`);
}

export async function deleteShowPtychoFolderFile(path: string): Promise<void> {
  const clean = normaliseSourcePath(path);
  if (!clean.startsWith("saves/")) throw new Error("folder deletes are restricted to saves/");
  if (localDirHandle) {
    const dir = await savesDirHandle();
    await dir.removeEntry(clean.slice("saves/".length));
    return;
  }
  const res = await fetch(clean, { method: "DELETE" });
  if (!res.ok) throw new Error(`${clean}: HTTP ${res.status}`);
}

export async function readShowPtychoFolderJson<T>(path: string): Promise<T | null> {
  try {
    const bytes = await readSourceBytes(path);
    return JSON.parse(new TextDecoder().decode(bytes)) as T;
  } catch {
    return null;
  }
}

export async function readShowPtychoFolderBytes(path: string): Promise<Uint8Array> {
  return readSourceBytes(path);
}

type SsbCal = {
  num_bf: number;
  g_shape: [number, number, number];
  detector_shape?: [number, number];
  bf_rows?: number[];
  bf_cols?: number[];
  wavelength_A: number;
  semiangle_rad: number;
  angular_sampling_rad: [number, number];
  dc_value: [number, number];
  kx_bf: number[];
  ky_bf: number[];
  qx_1d: number[];
  qy_1d: number[];
  aperture_k: number[];
  alpha_k2: number[];
  cos2phi_k: number[];
  sin2phi_k: number[];
  rotation_angle_deg?: number;
  rotation_angle_rad?: number;
};

type H5SsbSource = {
  kind: "hdf5";
  masterUrl: string;
  dataUrls?: string[];
  chunkIndexes?: H5ChunkIndexSource[];
  decodeDtype?: "uint8" | "uint16" | "float32";
};

type BfColumnSsbSource = {
  kind: "bf_columns";
  url: string;
  dtype?: "uint4" | "uint8" | "uint16" | "float32";
  encoding?: "uint4" | "uint8" | "uint16" | "float32";
  numBf: number;
  plane: number;
  scanShape?: [number, number];
  bytesPerBf: number;
  bitsPerValue?: number;
};

type H5ChunkIndexSource = {
  path?: string;
  url?: string;
  byte_offset?: number;
  bytes?: number;
  frames: number;
  detector_shape?: [number, number];
  dtype?: string;
};

type SsbSource =
  | { kind: "g-bf"; bytes: Uint8Array | null; url: string | null }
  | { kind: "hdf5"; source: H5SsbSource }
  | { kind: "bf-columns"; source: BfColumnSsbSource };

type SsbBuffers = {
  params: GPUBuffer;
  paramsChunks: GPUBuffer[];
  aberrations: GPUBuffer;
  gqkChunks: GPUBuffer[];
  gqkScales: GPUBuffer | null;
  gqkMode: GqkMode;
  gqkResidentBytes: number;
  chunkBfCounts: number[];
  chunkCapacity: number;
  dispatchChunkCapacity: number;
  fullStack: boolean;
  stage: GPUBuffer;
  phaseStack: GPUBuffer;
  partialSum: GPUBuffer;
  partialSumSq: GPUBuffer;
  partialGroups: number;
  activeBfCount: number;
  nonzeroBfCount: number;
  activeSourceIndices: Uint32Array;
  phase: GPUBuffer;
  variance: GPUBuffer;
  bfGeom: GPUBuffer;
  bfTrig: GPUBuffer;
  qx: GPUBuffer;
  qy: GPUBuffer;
};

type SsbPipelines = {
  rows: GPUComputePipeline;
  cols: GPUComputePipeline;
  reducePartial: GPUComputePipeline;
  finalizeGroups: GPUComputePipeline;
  objSum: GPUComputePipeline;
  objFftRows: GPUComputePipeline;
  objFftCols: GPUComputePipeline;
};

type SsbBindGroups = {
  rows: GPUBindGroup[];
  cols: GPUBindGroup[];
  reducePartial: GPUBindGroup[];
  finalizeGroups: GPUBindGroup;
  objSum: GPUBindGroup[];
  objFftRows: GPUBindGroup;
  objFftCols: GPUBindGroup;
};

export type WebGPUSSBResult = {
  phase: Float32Array;
  width: number;
  height: number;
  gpuMs: number;
  bfCount: number;
  rotationDeg: number;
  loss: number | null;
  adapterInfo: string;
  softwareAdapter: boolean;
};

export type WebGPULoadProgress = {
  stage: "idle" | "device" | "pipeline" | "metadata" | "fetch" | "parse" | "decode" | "gather" | "fft" | "ready" | "error";
  message: string;
  detail?: string;
  current?: number;
  total?: number;
  percent?: number;
  activeBf?: number;
  totalBf?: number;
  sourceFrames?: number;
  elapsedMs?: number;
};

type WebGPUProgressHandler = (progress: WebGPULoadProgress) => void;

function formatF32(value: number): string {
  if (Math.abs(value) < 1e-12) return "0.0";
  if (Math.abs(value - 1) < 1e-12) return "1.0";
  if (Math.abs(value + 1) < 1e-12) return "-1.0";
  return value.toPrecision(9);
}

function makeFftConstants(n: SupportedSsbSize): string {
  const twiddles: string[] = [];
  for (let len = 2; len <= n; len *= 2) {
    const half = len / 2;
    for (let j = 0; j < half; j++) {
      const angle = 2 * Math.PI * j / len;
      twiddles.push(`vec2<f32>(${formatF32(Math.cos(angle))}, ${formatF32(Math.sin(angle))})`);
    }
  }
  const bitrev: number[] = [];
  const bits = Math.log2(n);
  for (let x = 0; x < n; x++) {
    let y = 0;
    for (let bit = 0; bit < bits; bit++) y = (y << 1) | ((x >> bit) & 1);
    bitrev.push(y);
  }
  return `
const FFT_TWIDDLES: array<vec2<f32>, ${twiddles.length}> = array<vec2<f32>, ${twiddles.length}>(
  ${twiddles.join(",\n  ")}
);
const FFT_BITREV: array<u32, ${n}> = array<u32, ${n}>(
  ${bitrev.map(value => `${value}u`).join(", ")}
);`;
}

function makeFftStages(n: SupportedSsbSize): string {
  const workgroupSize = Math.min(n, 256);
  const lines: string[] = ["  workgroupBarrier();"];
  for (let len = 2; len <= n; len *= 2) {
    const half = len / 2;
    const twiddleOffset = half - 1;
    lines.push(
      `  for (var butterfly = tid; butterfly < ${n / 2}u; butterfly = butterfly + ${workgroupSize}u) {`,
      `    let group = butterfly / ${half}u;`,
      `    let j = butterfly - group * ${half}u;`,
      `    let i0 = group * ${len}u + j;`,
      `    let i1 = i0 + ${half}u;`,
      `    let w = FFT_TWIDDLES[${twiddleOffset}u + j];`,
      "    let t = cmul(w, s[i1]);",
      "    let u = s[i0];",
      "    s[i0] = u + t;",
      "    s[i1] = u - t;",
      "  }",
      "  workgroupBarrier();",
    );
  }
  return lines.join("\n");
}

function makeSsbShader(n: SupportedSsbSize, mode: GqkMode): string {
  const workgroupSize = Math.min(n, 256);
  const half = n / 2;
  const halfW = half + 1;
  const storedPlane = storedPlaneFor(n);
  const gqkDecl = mode === "herm16"
    ? `@group(0) @binding(1) var<storage, read> gqk: array<u32>;
@group(0) @binding(13) var<storage, read> gqkScale: array<f32>;`
    : "@group(0) @binding(1) var<storage, read> gqk: array<vec2<f32>>;";
  const fetchBody = `  var r = row;
  var c = x;
  var conj_sign = 1.0;
  if (x > ${half}u) {
    r = select(${n}u - row, 0u, row == 0u);
    c = ${n}u - x;
    conj_sign = -1.0;
  }
  let idx = local_bf * ${storedPlane}u + r * ${halfW}u + c;
${mode === "herm16"
      ? `  let v = unpack2x16snorm(gqk[idx]) * gqkScale[bf_global];`
      : "  let v = gqk[idx];"}
  return vec2<f32>(v.x, v.y * conj_sign);`;
  return `
${makeFftConstants(n)}
struct Params {
  num_bf: u32,
  n: u32,
  plane: u32,
  bf_offset: u32,
  wavelength: f32,
  semiangle_rad: f32,
  ang_y_rad: f32,
  ang_x_rad: f32,
  C10: f32,
  C12: f32,
  cos2phi12: f32,
  sin2phi12: f32,
  factor: f32,
  dc_re: f32,
  dc_im: f32,
  inv_n2: f32,
  chunk_bf: u32,
  full_stack: u32,
  partial_groups: u32,
  compute_loss: u32,
  active_bf: u32,
  full_aberration: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
${gqkDecl}
@group(0) @binding(2) var<storage, read_write> stage: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> bfGeom: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> bfTrig: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> qx1d: array<f32>;
@group(0) @binding(6) var<storage, read> qy1d: array<f32>;
@group(0) @binding(7) var<storage, read_write> phaseStack: array<f32>;
@group(0) @binding(8) var<storage, read_write> partialSum: array<f32>;
@group(0) @binding(9) var<storage, read_write> partialSumSq: array<f32>;
@group(0) @binding(10) var<storage, read_write> phaseOut: array<f32>;
@group(0) @binding(11) var<storage, read_write> varianceOut: array<f32>;
@group(0) @binding(12) var<storage, read> abr: array<vec4<f32>, 14>;

var<workgroup> s: array<vec2<f32>, ${n}>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
fn fetch_g(local_bf: u32, bf_global: u32, row: u32, x: u32) -> vec2<f32> {
${fetchBody}
}
fn compute_geometry(dx: f32, dy: f32) -> vec4<f32> {
  let dx2 = dx * dx;
  let dy2 = dy * dy;
  let r2 = dx2 + dy2;
  let r = sqrt(r2);
  let alpha = r * params.wavelength;
  let alpha2 = alpha * alpha;
  let inv_r2 = select(0.0, 1.0 / r2, r2 > 1e-30);
  let cos2phi = (dx2 - dy2) * inv_r2;
  let sin2phi = 2.0 * dx * dy * inv_r2;
  let denom_num2 = (dx * params.ang_y_rad) * (dx * params.ang_y_rad) + (dy * params.ang_x_rad) * (dy * params.ang_x_rad);
  let inv_r = select(0.0, 1.0 / r, r > 1e-15);
  let denom = sqrt(denom_num2) * inv_r;
  let edge = select(1.0, (params.semiangle_rad - alpha) / denom + 0.5, denom > 1e-15);
  return vec4<f32>(alpha2, cos2phi, sin2phi, clamp(edge, 0.0, 1.0));
}
fn gamma_mul(qx: f32, qy: f32, kx: f32, ky: f32, pk: vec2<f32>, G: vec2<f32>) -> vec2<f32> {
  let m = compute_geometry(qx - kx, qy - ky);
  let p = compute_geometry(qx + kx, qy + ky);
  var chi_m: f32;
  var chi_p: f32;
  if (params.full_aberration != 0u) {
    chi_m = chi_full(qx - kx, qy - ky);
    chi_p = chi_full(qx + kx, qy + ky);
  } else {
    let cos_term_m = m.y * params.cos2phi12 + m.z * params.sin2phi12;
    let cos_term_p = p.y * params.cos2phi12 + p.z * params.sin2phi12;
    chi_m = params.factor * m.x * (params.C12 * cos_term_m + params.C10);
    chi_p = params.factor * p.x * (params.C12 * cos_term_p + params.C10);
  }
  let pm = vec2<f32>(m.w * cos(chi_m), -m.w * sin(chi_m));
  let pp = vec2<f32>(p.w * cos(chi_p), -p.w * sin(chi_p));
  let pk_conj = vec2<f32>(pk.x, -pk.y);
  let pp_conj = vec2<f32>(pp.x, -pp.y);
  let t1 = cmul(pm, pk_conj);
  let t2 = cmul(pp_conj, pk);
  var gg = t1 - t2;
  let mag_sq = gg.x * gg.x + gg.y * gg.y;
  let inv_mag = select(1e8, inverseSqrt(mag_sq), mag_sq > 1e-16);
  gg *= inv_mag;
  return vec2<f32>(G.x * gg.x + G.y * gg.y, G.y * gg.x - G.x * gg.y);
}
fn chi_full(dx: f32, dy: f32) -> f32 {
  let r2 = dx * dx + dy * dy;
  let inv_r = select(0.0, inverseSqrt(r2), r2 > 1e-30);
  let r = r2 * inv_r;
  let alpha = r * params.wavelength;
  let c1 = dx * inv_r;
  let s1 = dy * inv_r;
  let c2 = 2.0 * c1 * c1 - 1.0;
  let s2 = 2.0 * s1 * c1;
  let c3 = c1 * c2 - s1 * s2;
  let s3 = s1 * c2 + c1 * s2;
  let c4 = c2 * c2 - s2 * s2;
  let s4 = 2.0 * s2 * c2;
  let c5 = c1 * c4 - s1 * s4;
  let s5 = s1 * c4 + c1 * s4;
  let c6 = c2 * c4 - s2 * s4;
  let s6 = s2 * c4 + c2 * s4;
  let a2 = alpha * alpha;
  let a3 = a2 * alpha;
  let a4 = a2 * a2;
  let a5 = a4 * alpha;
  let a6 = a3 * a3;
  var chi = 0.0;
  chi = chi + a2 * abr[0].x;
  chi = chi + a2 * abr[1].x * (c2 * abr[1].y + s2 * abr[1].z);
  chi = chi + a3 * abr[2].x * (c1 * abr[2].y + s1 * abr[2].z);
  chi = chi + a3 * abr[3].x * (c3 * abr[3].y + s3 * abr[3].z);
  chi = chi + a4 * abr[4].x;
  chi = chi + a4 * abr[5].x * (c2 * abr[5].y + s2 * abr[5].z);
  chi = chi + a4 * abr[6].x * (c4 * abr[6].y + s4 * abr[6].z);
  chi = chi + a5 * abr[7].x * (c1 * abr[7].y + s1 * abr[7].z);
  chi = chi + a5 * abr[8].x * (c3 * abr[8].y + s3 * abr[8].z);
  chi = chi + a5 * abr[9].x * (c5 * abr[9].y + s5 * abr[9].z);
  chi = chi + a6 * abr[10].x;
  chi = chi + a6 * abr[11].x * (c2 * abr[11].y + s2 * abr[11].z);
  chi = chi + a6 * abr[12].x * (c4 * abr[12].y + s4 * abr[12].z);
  chi = chi + a6 * abr[13].x * (c6 * abr[13].y + s6 * abr[13].z);
  return (6.283185307179586 / params.wavelength) * chi;
}
fn fft_inplace(tid: u32, sign: f32) {
  let _unused_sign = sign;
${makeFftStages(n)}
}

@compute @workgroup_size(${workgroupSize})
fn ssbRows(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let row = wid.y;
  let local_bf = wid.z;
  if (local_bf >= params.chunk_bf) { return; }
  let bf = local_bf + params.bf_offset;
  if (bf >= params.active_bf) { return; }
  let bg = bfGeom[bf];
  if (bg.w == 0.0) { return; }
  let storage_bf = select(local_bf, bf, params.full_stack != 0u);
  let base = (storage_bf * params.plane) + row * ${n}u;
  let bt = bfTrig[bf];
  var chi_k: f32;
  if (params.full_aberration != 0u) {
    chi_k = chi_full(bg.x, bg.y);
  } else {
    let cos_term_k = bt.x * params.cos2phi12 + bt.y * params.sin2phi12;
    chi_k = params.factor * bg.z * (params.C12 * cos_term_k + params.C10);
  }
  let pk = vec2<f32>(bg.w * cos(chi_k), -bg.w * sin(chi_k));
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let x = off + tid;
    if (x < ${n}u) {
      var v = gamma_mul(qx1d[row], qy1d[x], bg.x, bg.y, pk, fetch_g(local_bf, bf, row, x));
      if (row == 0u && x == 0u) { v = vec2<f32>(params.dc_re, params.dc_im); }
      s[FFT_BITREV[x]] = v;
    }
  }
  fft_inplace(tid, 1.0);
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let x = off + tid;
    if (x < ${n}u) { stage[base + x] = s[x]; }
  }
}

@compute @workgroup_size(${workgroupSize})
fn ssbCols(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let col = wid.y;
  let local_bf = wid.z;
  if (local_bf >= params.chunk_bf) { return; }
  let bf = local_bf + params.bf_offset;
  if (bf >= params.active_bf) { return; }
  let bg = bfGeom[bf];
  if (bg.w == 0.0) { return; }
  let storage_bf = select(local_bf, bf, params.full_stack != 0u);
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let y = off + tid;
    if (y < ${n}u) {
      s[FFT_BITREV[y]] = stage[storage_bf * params.plane + y * ${n}u + col];
    }
  }
  fft_inplace(tid, 1.0);
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let y = off + tid;
    if (y < ${n}u) {
      let idx = storage_bf * params.plane + y * ${n}u + col;
      let v = s[y] * params.inv_n2;
      phaseStack[idx] = atan2(v.y, v.x);
    }
  }
}

@compute @workgroup_size(256)
fn reducePartialGroups(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x;
  let groups_in_chunk = (params.chunk_bf + ${REDUCE_BF_GROUP - 1}u) / ${REDUCE_BF_GROUP}u;
  let total = groups_in_chunk * params.plane;
  if (linear >= total) { return; }
  let local_group = linear / params.plane;
  let idx = linear - local_group * params.plane;
  let start_bf = params.bf_offset + local_group * ${REDUCE_BF_GROUP}u;
  let end_bf = min(params.active_bf, min(params.bf_offset + params.chunk_bf, start_bf + ${REDUCE_BF_GROUP}u));
  let global_group = start_bf / ${REDUCE_BF_GROUP}u;
  var sum = 0.0;
  var sumsq = 0.0;
  if (params.compute_loss != 0u) {
    for (var bf = start_bf; bf < end_bf; bf = bf + 1u) {
      if (bfGeom[bf].w != 0.0) {
        let storage_bf = select(bf - params.bf_offset, bf, params.full_stack != 0u);
        let a = phaseStack[storage_bf * params.plane + idx];
        sum += a;
        sumsq += a * a;
      }
    }
  } else {
    for (var bf = start_bf; bf < end_bf; bf = bf + 1u) {
      if (bfGeom[bf].w != 0.0) {
        let storage_bf = select(bf - params.bf_offset, bf, params.full_stack != 0u);
        sum += phaseStack[storage_bf * params.plane + idx];
      }
    }
  }
  let out = global_group * params.plane + idx;
  partialSum[out] = sum;
  if (params.compute_loss != 0u) {
    partialSumSq[out] = sumsq;
  }
}

@compute @workgroup_size(256)
fn finalizePartialGroups(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.plane) { return; }
  var sum = 0.0;
  var sumsq = 0.0;
  if (params.compute_loss != 0u) {
    for (var group = 0u; group < params.partial_groups; group = group + 1u) {
      let off = group * params.plane + idx;
      sum += partialSum[off];
      sumsq += partialSumSq[off];
    }
  } else {
    for (var group = 0u; group < params.partial_groups; group = group + 1u) {
      sum += partialSum[group * params.plane + idx];
    }
  }
  let denom = f32(max(params.num_bf, 1u));
  let mean = sum / denom;
  phaseOut[idx] = mean;
  varianceOut[idx] = select(0.0, sumsq / denom - mean * mean, params.compute_loss != 0u);
}

// ---------------------------------------------------------------------------
// Object-redraw fast path (ported from the CUDA SSB engine, 2026-07-17):
// mean_bf(ifft2(corrected_bf)) == ifft2(mean_bf(corrected_bf)), so the drag
// path sums gamma-corrected G(q,k) over BF pixels in Fourier space and runs
// ONE 2D inverse FFT, instead of two FFT passes + atan2 per BF pixel. This
// displays angle(mean(object)) - the same estimator as the Python
// SSB.result() reference - and skips the per-BF phase stack entirely.
// The loss/variance commit path still uses the exact per-BF pipeline above,
// because mean(angle(object)) != angle(mean(object)).
// ---------------------------------------------------------------------------
@compute @workgroup_size(256)
fn ssbObjSum(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x + gid.y * 65535u * 256u;
  if (idx >= params.plane) { return; }
  let row = idx / ${n}u;
  let x = idx - row * ${n}u;
  var acc = vec2<f32>(0.0, 0.0);
  let end_bf = min(params.active_bf, params.bf_offset + params.chunk_bf);
  for (var bf = params.bf_offset; bf < end_bf; bf = bf + 1u) {
    let bg = bfGeom[bf];
    if (bg.w == 0.0) { continue; }
    if (idx == 0u) {
      acc += vec2<f32>(params.dc_re, params.dc_im);
      continue;
    }
    let bt = bfTrig[bf];
    var chi_k: f32;
    if (params.full_aberration != 0u) {
      chi_k = chi_full(bg.x, bg.y);
    } else {
      let cos_term_k = bt.x * params.cos2phi12 + bt.y * params.sin2phi12;
      chi_k = params.factor * bg.z * (params.C12 * cos_term_k + params.C10);
    }
    let pk = vec2<f32>(bg.w * cos(chi_k), -bg.w * sin(chi_k));
    acc += gamma_mul(qx1d[row], qy1d[x], bg.x, bg.y, pk, fetch_g(bf - params.bf_offset, bf, row, x));
  }
  // stage[0..plane] is the accumulation plane for the fast path (cleared
  // before the first chunk; chunks add sequentially via separate dispatches).
  stage[idx] = stage[idx] + acc;
}

@compute @workgroup_size(${workgroupSize})
fn ssbObjFftRows(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let row = wid.y;
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let x = off + tid;
    if (x < ${n}u) { s[FFT_BITREV[x]] = stage[row * ${n}u + x]; }
  }
  fft_inplace(tid, 1.0);
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let x = off + tid;
    if (x < ${n}u) { stage[row * ${n}u + x] = s[x]; }
  }
}

@compute @workgroup_size(${workgroupSize})
fn ssbObjFftCols(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let col = wid.y;
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let y = off + tid;
    if (y < ${n}u) { s[FFT_BITREV[y]] = stage[y * ${n}u + col]; }
  }
  fft_inplace(tid, 1.0);
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let y = off + tid;
    if (y < ${n}u) {
      let v = s[y] * params.inv_n2;
      phaseOut[y * ${n}u + col] = atan2(v.y, v.x);
    }
  }
}

`;
}

function makeBufferFromBytes(
  device: GPUDevice,
  bytes: Uint8Array,
  usage: GPUBufferUsageFlags,
  label: string,
): GPUBuffer {
  const buffer = device.createBuffer({ size: bytes.byteLength, usage: usage | GPUBufferUsage.COPY_DST, label });
  const chunk = 32 * 1024 * 1024;
  for (let off = 0; off < bytes.byteLength; off += chunk) {
    const len = Math.min(chunk, bytes.byteLength - off);
    device.queue.writeBuffer(buffer, off, bytes as unknown as BufferSource, off, len);
  }
  return buffer;
}

function makeBufferFromF32(
  device: GPUDevice,
  array: Float32Array,
  usage: GPUBufferUsageFlags,
  label: string,
): GPUBuffer {
  return makeBufferFromBytes(
    device,
    new Uint8Array(array.buffer, array.byteOffset, array.byteLength),
    usage,
    label,
  );
}

function makeBufferFromU32(
  device: GPUDevice,
  array: Uint32Array,
  usage: GPUBufferUsageFlags,
  label: string,
): GPUBuffer {
  return makeBufferFromBytes(
    device,
    new Uint8Array(array.buffer, array.byteOffset, array.byteLength),
    usage,
    label,
  );
}

function makeSourceFftConstants(n: SupportedSsbSize): string {
  const twiddles: string[] = [];
  for (let len = 2; len <= n; len *= 2) {
    const half = len / 2;
    for (let j = 0; j < half; j++) {
      const angle = -2 * Math.PI * j / len;
      twiddles.push(`vec2<f32>(${formatF32(Math.cos(angle))}, ${formatF32(Math.sin(angle))})`);
    }
  }
  const bitrev: number[] = [];
  const bits = Math.log2(n);
  for (let x = 0; x < n; x++) {
    let y = 0;
    for (let bit = 0; bit < bits; bit++) y = (y << 1) | ((x >> bit) & 1);
    bitrev.push(y);
  }
  return `
const SOURCE_FFT_TWIDDLES: array<vec2<f32>, ${twiddles.length}> = array<vec2<f32>, ${twiddles.length}>(
  ${twiddles.join(",\n  ")}
);
const SOURCE_FFT_BITREV: array<u32, ${n}> = array<u32, ${n}>(
  ${bitrev.map(value => `${value}u`).join(", ")}
);`;
}

function makeSourceFftStages(n: SupportedSsbSize): string {
  const workgroupSize = Math.min(n, 256);
  const lines: string[] = ["  workgroupBarrier();"];
  for (let len = 2; len <= n; len *= 2) {
    const half = len / 2;
    const twiddleOffset = half - 1;
    lines.push(
      `  for (var butterfly = tid; butterfly < ${n / 2}u; butterfly = butterfly + ${workgroupSize}u) {`,
      `    let group = butterfly / ${half}u;`,
      `    let j = butterfly - group * ${half}u;`,
      `    let i0 = group * ${len}u + j;`,
      `    let i1 = i0 + ${half}u;`,
      `    let w = SOURCE_FFT_TWIDDLES[${twiddleOffset}u + j];`,
      "    let t = cmul_source(w, s[i1]);",
      "    let u = s[i0];",
      "    s[i0] = u + t;",
      "    s[i1] = u - t;",
      "  }",
      "  workgroupBarrier();",
    );
  }
  return lines.join("\n");
}

function makeH5GqkShader(n: SupportedSsbSize): string {
  const workgroupSize = Math.min(n, 256);
  return `
${makeSourceFftConstants(n)}
fn sample(gp: u32, mode: u32) -> u32 {
  if (mode == 3u) { let w = data[gp >> 3u]; return (w >> ((gp & 7u) * 4u)) & 0xfu; }
  if (mode == 1u) { let w = data[gp >> 2u]; return (w >> ((gp & 3u) * 8u)) & 0xffu; }
  let w = data[gp >> 1u];
  return select(w >> 16u, w & 0xffffu, (gp & 1u) == 0u);
}
fn sampleF(gp: u32, mode: u32) -> f32 {
  if (mode == 2u) { return bitcast<f32>(data[gp]); }
  return f32(sample(gp, mode));
}
fn cmul_source(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
struct GatherParams {
  start_scan: u32,
  n_scan: u32,
  det_size: u32,
  mode: u32,
  chunk_bf: u32,
  plane: u32,
  _pad0: u32,
  _pad1: u32,
};
@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read> bfIdx: array<u32>;
@group(0) @binding(2) var<storage, read_write> gqk: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> gatherParams: GatherParams;
@compute @workgroup_size(16, 16)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
  let scan = gid.x;
  let bf = gid.y;
  if (scan >= gatherParams.n_scan || bf >= gatherParams.chunk_bf) { return; }
  let globalScan = gatherParams.start_scan + scan;
  let det = bfIdx[bf];
  let v = sampleF(scan * gatherParams.det_size + det, gatherParams.mode);
  gqk[bf * gatherParams.plane + globalScan] = vec2<f32>(v, 0.0);
}
@compute @workgroup_size(${BF_COLUMN_UNPACK_WORKGROUP_X}, ${BF_COLUMN_UNPACK_WORKGROUP_Y})
fn bfColumnsToGqk(@builtin(global_invocation_id) gid: vec3<u32>) {
  let scan = gid.x;
  let bf = gid.y;
  if (scan >= gatherParams.plane || bf >= gatherParams.chunk_bf) { return; }
  let gp = bf * gatherParams.plane + scan;
  gqk[gp] = vec2<f32>(sampleF(gp, gatherParams.mode), 0.0);
}

struct FftParams {
  chunk_bf: u32,
  plane: u32,
  _pad0: u32,
  _pad1: u32,
};
@group(1) @binding(0) var<storage, read_write> fftData: array<vec2<f32>>;
@group(1) @binding(1) var<uniform> fftParams: FftParams;
var<workgroup> s: array<vec2<f32>, ${n}>;
fn source_fft_inplace(tid: u32) {
${makeSourceFftStages(n)}
}
@compute @workgroup_size(${workgroupSize})
fn fftRows(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let row = wid.y;
  let bf = wid.z;
  if (bf >= fftParams.chunk_bf) { return; }
  let base = bf * fftParams.plane + row * ${n}u;
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let x = off + tid;
    if (x < ${n}u) { s[SOURCE_FFT_BITREV[x]] = fftData[base + x]; }
  }
  source_fft_inplace(tid);
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let x = off + tid;
    if (x < ${n}u) { fftData[base + x] = s[x]; }
  }
}
@compute @workgroup_size(${workgroupSize})
fn fftCols(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let col = wid.y;
  let bf = wid.z;
  if (bf >= fftParams.chunk_bf) { return; }
  let base = bf * fftParams.plane;
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let y = off + tid;
    if (y < ${n}u) { s[SOURCE_FFT_BITREV[y]] = fftData[base + y * ${n}u + col]; }
  }
  source_fft_inplace(tid);
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let y = off + tid;
    if (y < ${n}u) { fftData[base + y * ${n}u + col] = s[y]; }
  }
}
`;
}


function makeGqkTransformShader(n: SupportedSsbSize, mode: GqkMode): string {
  const half = n / 2;
  const halfW = half + 1;
  const storedPlane = n * halfW;
  const dstDecl = mode === "herm16"
    ? "@group(0) @binding(1) var<storage, read_write> dst: array<u32>;"
    : "@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;";
  const writeBody = mode === "herm16"
    ? `let s = max(scales[tp.bf_offset + bf], 1e-30);
    dst[bf * ${storedPlane}u + row * ${halfW}u + c] = pack2x16snorm(clamp(v / s, vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0)));`
    : `dst[bf * ${storedPlane}u + row * ${halfW}u + c] = v;`;
  return `
struct TransformParams {
  chunk_bf: u32,
  bf_offset: u32,
  _pad0: u32,
  _pad1: u32,
};
@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
${dstDecl}
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;
@group(0) @binding(3) var<uniform> tp: TransformParams;
var<workgroup> wg_max: array<f32, 256>;

@compute @workgroup_size(256)
fn scaleMax(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let bf = wid.z;
  if (bf >= tp.chunk_bf) { return; }
  var m = 0.0;
  // Scan the half-plane only: the mirrored half has identical magnitudes.
  for (var i = lid.x; i < ${storedPlane}u; i = i + 256u) {
    let row = i / ${halfW}u;
    let c = i - row * ${halfW}u;
    let v = src[bf * ${n * n}u + row * ${n}u + c];
    m = max(m, max(abs(v.x), abs(v.y)));
  }
  wg_max[lid.x] = m;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (lid.x < stride) { wg_max[lid.x] = max(wg_max[lid.x], wg_max[lid.x + stride]); }
    workgroupBarrier();
  }
  if (lid.x == 0u) { scales[tp.bf_offset + bf] = max(wg_max[0u], 1e-30); }
}

@compute @workgroup_size(256)
fn compact(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let bf = gid.z;
  if (i >= ${storedPlane}u || bf >= tp.chunk_bf) { return; }
  let row = i / ${halfW}u;
  let c = i - row * ${halfW}u;
  let v = src[bf * ${n * n}u + row * ${n}u + c];
  ${writeBody}
}
`;
}

type GqkTransformResult = {
  chunks: GPUBuffer[];
  scales: GPUBuffer | null;
  residentBytes: number;
};

// Rewrite freshly built n x n complex64 G(q,k) chunks into the resident
// half-plane representation, destroying each source chunk as soon as it is
// compacted so peak memory only briefly exceeds the build peak by one
// half-plane chunk.
async function transformGqkChunks(
  device: GPUDevice,
  n: SupportedSsbSize,
  mode: GqkMode,
  gqkChunks: GPUBuffer[],
  chunkBfCounts: number[],
  activeBfCount: number,
): Promise<GqkTransformResult> {
  const storedPlane = storedPlaneFor(n);
  const bytesPer = gqkBytesPerValue(mode);
  const module = device.createShaderModule({
    code: makeGqkTransformShader(n, mode),
    label: `ShowPtycho gqk transform ${mode} ${n}`,
  });
  const scalePipe = mode === "herm16"
    ? device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "scaleMax" } })
    : null;
  const compactPipe = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "compact" } });
  const scales = device.createBuffer({
    size: Math.max(4, activeBfCount * 4),
    usage: GPUBufferUsage.STORAGE,
    label: "showptycho gqk scales",
  });
  const out: GPUBuffer[] = [];
  let residentBytes = 0;
  let bfOffset = 0;
  for (let i = 0; i < gqkChunks.length; i++) {
    const chunkBf = chunkBfCounts[i];
    const dst = device.createBuffer({
      size: chunkBf * storedPlane * bytesPer,
      usage: GPUBufferUsage.STORAGE,
      label: `showptycho gqk ${mode} chunk ${i}`,
    });
    const params = uniformU32(device, [chunkBf, bfOffset, 0, 0], `showptycho gqk transform params ${i}`);
    const enc = device.createCommandEncoder();
    if (scalePipe) {
      const bind = device.createBindGroup({
        layout: scalePipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: gqkChunks[i] } },
          { binding: 2, resource: { buffer: scales } },
          { binding: 3, resource: { buffer: params } },
        ],
      });
      const pass = enc.beginComputePass({ label: "showptycho gqk scaleMax" });
      pass.setPipeline(scalePipe);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(1, 1, chunkBf);
      pass.end();
    }
    const compactEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: gqkChunks[i] } },
      { binding: 1, resource: { buffer: dst } },
      { binding: 3, resource: { buffer: params } },
    ];
    if (mode === "herm16") compactEntries.push({ binding: 2, resource: { buffer: scales } });
    const compactBind = device.createBindGroup({
      layout: compactPipe.getBindGroupLayout(0),
      entries: compactEntries,
    });
    const pass = enc.beginComputePass({ label: "showptycho gqk compact" });
    pass.setPipeline(compactPipe);
    pass.setBindGroup(0, compactBind);
    pass.dispatchWorkgroups(Math.ceil(storedPlane / 256), 1, chunkBf);
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    params.destroy();
    gqkChunks[i].destroy();
    out.push(dst);
    residentBytes += chunkBf * storedPlane * bytesPer;
    bfOffset += chunkBf;
  }
  if (mode !== "herm16") {
    scales.destroy();
    return { chunks: out, scales: null, residentBytes };
  }
  return { chunks: out, scales, residentBytes: residentBytes + activeBfCount * 4 };
}


type H5GqkBuild = {
  gqkChunks: GPUBuffer[];
  chunkBfCounts: number[];
  fetchBytes: number;
  fetchMs: number;
  fetchWallMs: number;
  parseMs: number;
  decodeMs: number;
  gatherMs: number;
  fftMs: number;
  sourceFrames: number;
};

function dataUrlsFromSource(source: H5SsbSource): string[] {
  if (source.dataUrls && source.dataUrls.length > 0) return source.dataUrls;
  const base = source.masterUrl.replace(/_master\.h5$/, "");
  const urls: string[] = [];
  for (let n = 1; n < 10000; n++) {
    urls.push(`${base}_data_${String(n).padStart(6, "0")}.h5`);
  }
  return urls;
}

function detectorFlatPixels(cal: SsbCal, activeIndices: Uint32Array): Uint32Array {
  const detCols = Number(cal.detector_shape?.[1] || 0);
  if (!detCols || !cal.bf_rows || !cal.bf_cols) {
    throw new Error("Compressed-source ShowPtycho needs detector_shape plus bf_rows/bf_cols in cal.json.");
  }
  const out = new Uint32Array(activeIndices.length);
  for (let i = 0; i < activeIndices.length; i++) {
    const src = activeIndices[i];
    out[i] = Math.round(Number(cal.bf_rows[src] || 0)) * detCols + Math.round(Number(cal.bf_cols[src] || 0));
  }
  return out;
}

function clippedBslz4Frames(h5Chunk: Bslz4Spec, nFrames: number): Bslz4Spec {
  const safeFrames = Math.max(0, Math.min(h5Chunk.nFrames, Math.floor(nFrames)));
  if (safeFrames === h5Chunk.nFrames) return h5Chunk;
  const metaValues = safeFrames * h5Chunk.nBlocksPerFrame * 2;
  return {
    ...h5Chunk,
    nFrames: safeFrames,
    blockMeta: h5Chunk.blockMeta.subarray(0, metaValues),
  };
}

function uniformU32(device: GPUDevice, values: number[], label: string): GPUBuffer {
  const b = device.createBuffer({ size: Math.max(16, Math.ceil(values.length / 4) * 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label });
  const arr = new Uint32Array(Math.ceil(values.length / 4) * 4);
  arr.set(values.map(v => Math.max(0, Math.round(v))));
  device.queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength);
  return b;
}

async function buildH5GqkChunks(
  device: GPUDevice,
  cal: SsbCal,
  source: H5SsbSource,
  n: SupportedSsbSize,
  plane: number,
  activeSourceIndices: Uint32Array,
  storageChunkCapacity: number,
  onProgress?: WebGPUProgressHandler,
): Promise<H5GqkBuild> {
  const t0 = performance.now();
  const emit = (progress: Omit<WebGPULoadProgress, "elapsedMs" | "activeBf" | "totalBf">) => {
    onProgress?.({
      activeBf: activeSourceIndices.length,
      totalBf: cal.num_bf,
      elapsedMs: performance.now() - t0,
      ...progress,
    });
  };
  emit({
    stage: "pipeline",
    message: "Preparing WebGPU kernels",
    detail: `Building HDF5 gather and FFT kernels for ${n}x${n}`,
  });
  const module = device.createShaderModule({ code: makeH5GqkShader(n), label: `ShowPtycho HDF5 gather+FFT ${n}` });
  const gatherPipe = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "gather" } });
  const rowsPipe = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "fftRows" } });
  const colsPipe = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "fftCols" } });
  const activeFlat = detectorFlatPixels(cal, activeSourceIndices);
  const gqkChunks: GPUBuffer[] = [];
  const chunkBfCounts: number[] = [];
  const bfIndexBuffers: GPUBuffer[] = [];
  const fftParamBuffers: GPUBuffer[] = [];
  for (let bfOffset = 0; bfOffset < activeFlat.length; bfOffset += storageChunkCapacity) {
    const chunkBf = Math.min(storageChunkCapacity, activeFlat.length - bfOffset);
    const bytes = chunkBf * plane * 8;
    gqkChunks.push(device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: `showptycho transient hdf5 G(q,k) ${bfOffset}`,
    }));
    chunkBfCounts.push(chunkBf);
    bfIndexBuffers.push(makeBufferFromU32(
      device,
      activeFlat.subarray(bfOffset, bfOffset + chunkBf),
      GPUBufferUsage.STORAGE,
      `showptycho hdf5 bf index ${bfOffset}`,
    ));
    fftParamBuffers.push(uniformU32(device, [chunkBf, plane, 0, 0], `showptycho hdf5 fft params ${bfOffset}`));
  }

  const dataUrls = dataUrlsFromSource(source);
  let fetchBytes = 0;
  let fetchMs = 0;
  let fetchWallMs = 0;
  let parseMs = 0;
  let decodeMs = 0;
  let gatherMs = 0;
  let sourceFrames = 0;
  let badPixels = 0;
  try {
    emit({
      stage: "metadata",
      message: "Reading microscope acquisition metadata",
      detail: "Checking master HDF5 file and hot-pixel table",
    });
    const masterBytes = await readSourceBytes(source.masterUrl);
    const info = readH5MasterInfo(masterBytes.buffer.slice(masterBytes.byteOffset, masterBytes.byteOffset + masterBytes.byteLength) as ArrayBuffer, "showptycho-master");
    badPixels = info.badPixels.length;
  } catch (err) {
    console.warn("[showptycho] could not read HDF5 master metadata", err);
  }

  const decodeDtype = source.decodeDtype || "uint16";
  const finiteDataUrlCount = source.dataUrls && source.dataUrls.length > 0 ? source.dataUrls.length : undefined;
  const disableIndexedPrefetch =
    typeof globalThis !== "undefined"
    && (globalThis as { __QUANTEM_SHOWPTYCHO_DISABLE_H5_PREFETCH__?: boolean }).__QUANTEM_SHOWPTYCHO_DISABLE_H5_PREFETCH__ === true;
  const indexedPrefetchWindow = disableIndexedPrefetch ? 1 : h5IndexedPrefetchWindow();
  const decodeAndGather = async (
    h5Chunk: Bslz4Spec,
    srcDtype: "uint8" | "uint16" | "uint32" | "float32",
    detSize: number,
    fileIndex: number,
    chunkIndex: number,
    chunkTotal: number,
  ) => {
    const remainingFrames = plane - sourceFrames;
    if (remainingFrames <= 0) return;
    const scanChunk = clippedBslz4Frames(h5Chunk, remainingFrames);
    const decodeT = performance.now();
    emit({
      stage: "decode",
      message: "Decompressing detector frames on WebGPU",
      detail: `${scanChunk.nFrames} scan positions, ${decodeDtype}, chunk ${chunkIndex + 1}/${chunkTotal}`,
      current: fileIndex + 1,
      total: finiteDataUrlCount,
      percent: finiteDataUrlCount ? ((fileIndex + 0.35) / finiteDataUrlCount) * 70 : Math.min(75, (sourceFrames / plane) * 70 + 5),
      sourceFrames,
    });
    const dec = await decodeBslz4ToStack(scanChunk, decodeDtype, srcDtype);
    decodeMs += performance.now() - decodeT;
    if (!dec) throw new Error("WebGPU HDF5 decode failed.");
    const gatherT = performance.now();
    emit({
      stage: "gather",
      message: "Gathering selected BF pixels",
      detail: `${activeSourceIndices.length}/${cal.num_bf} BF pixels from ${scanChunk.nFrames} frames`,
      current: fileIndex + 1,
      total: finiteDataUrlCount,
      percent: finiteDataUrlCount ? ((fileIndex + 0.7) / finiteDataUrlCount) * 70 : Math.min(82, (sourceFrames / plane) * 70 + 12),
      sourceFrames,
    });
    const enc = device.createCommandEncoder();
    const gatherTemps: GPUBuffer[] = [];
    for (let i = 0; i < gqkChunks.length; i++) {
      const chunkBf = chunkBfCounts[i];
      const params = uniformU32(
        device,
        [sourceFrames, scanChunk.nFrames, detSize, dec.mode, chunkBf, plane, 0, 0],
        `showptycho hdf5 gather params ${fileIndex}-${chunkIndex}-${i}`,
      );
      gatherTemps.push(params);
      const bind = device.createBindGroup({
        layout: gatherPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: dec.buffer } },
          { binding: 1, resource: { buffer: bfIndexBuffers[i] } },
          { binding: 2, resource: { buffer: gqkChunks[i] } },
          { binding: 3, resource: { buffer: params } },
        ],
      });
      const pass = enc.beginComputePass();
      pass.setPipeline(gatherPipe);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(scanChunk.nFrames / 16), Math.ceil(chunkBf / 16));
      pass.end();
    }
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    gatherTemps.forEach(b => b.destroy());
    gatherMs += performance.now() - gatherT;
    dec.buffer.destroy();
    sourceFrames += scanChunk.nFrames;
    emit({
      stage: "gather",
      message: "Accumulating BF reducer",
      detail: `${sourceFrames}/${plane} scan positions gathered`,
      current: Math.min(sourceFrames, plane),
      total: plane,
      percent: Math.min(85, (sourceFrames / plane) * 80),
      sourceFrames,
    });
  };
  for (let fileIndex = 0; fileIndex < dataUrls.length; fileIndex++) {
    const url = dataUrls[fileIndex];
    emit({
      stage: "fetch",
      message: "Reading compressed diffraction frames",
      detail: finiteDataUrlCount
        ? `HDF5 data file ${fileIndex + 1}/${finiteDataUrlCount}`
        : `HDF5 data file ${fileIndex + 1}`,
      current: fileIndex + 1,
      total: finiteDataUrlCount,
      percent: finiteDataUrlCount ? (fileIndex / finiteDataUrlCount) * 70 : Math.min(70, (sourceFrames / plane) * 70),
      sourceFrames,
    });
    const chunkIndexSource = source.chunkIndexes?.[fileIndex];
    if (chunkIndexSource) {
      const parseT = performance.now();
      emit({
        stage: "parse",
        message: "Parsing HDF5 range index",
        detail: chunkIndexSource.path || chunkIndexSource.url || "chunk index",
        current: fileIndex + 1,
        total: finiteDataUrlCount,
        sourceFrames,
      });
      const index = await fetchChunkIndex(chunkIndexSource, url);
      parseMs += performance.now() - parseT;
      const detSize = index.detRows * index.detCols;
      if (detSize !== Number(cal.detector_shape?.[0] || 0) * Number(cal.detector_shape?.[1] || 0)) {
        throw new Error(
          `${url}: detector shape mismatch; HDF5 has ${index.detRows}x${index.detCols}, `
          + `calibration has ${(cal.detector_shape || []).join("x")}`,
        );
      }
      const framesPerChunk = indexedFramesPerChunk(
        detSize,
        decodeDtype,
        device.limits.maxStorageBufferBindingSize,
      );
      const fileFramesToRead = Math.min(index.frames, Math.max(0, plane - sourceFrames));
      if (fileFramesToRead <= 0) break;
      const chunkTotal = Math.ceil(fileFramesToRead / framesPerChunk);
      type IndexedFetchPlan = {
        h5ChunkIndex: number;
        frameStart: number;
        nFrames: number;
        frameStop: number;
        rangeStart: number;
        rangeBytes: number;
        payloadBytes: number;
      };
      type IndexedFetchResult = {
        plan: IndexedFetchPlan;
        range: Uint8Array;
        fetchMs: number;
        fetchStartMs: number;
        fetchEndMs: number;
      };
      const plans: IndexedFetchPlan[] = [];
      for (let h5ChunkIndex = 0; h5ChunkIndex < chunkTotal; h5ChunkIndex++) {
        const frameStart = h5ChunkIndex * framesPerChunk;
        const frameStop = Math.min(fileFramesToRead, frameStart + framesPerChunk);
        let rangeStart = Number.POSITIVE_INFINITY;
        let rangeEnd = 0;
        let payloadBytes = 0;
        for (let frame = frameStart; frame < frameStop; frame++) {
          rangeStart = Math.min(rangeStart, index.offsets[frame]);
          rangeEnd = Math.max(rangeEnd, index.offsets[frame] + index.sizes[frame]);
          payloadBytes += index.sizes[frame];
        }
        const rangeBytes = rangeEnd - rangeStart;
        plans.push({
          h5ChunkIndex,
          frameStart,
          nFrames: frameStop - frameStart,
          frameStop,
          rangeStart,
          rangeBytes,
          payloadBytes,
        });
      }
      const startFetch = (plan: IndexedFetchPlan): Promise<IndexedFetchResult> => {
        emit({
          stage: "fetch",
          message: "Reading compressed diffraction frames",
          detail: `range chunk ${plan.h5ChunkIndex + 1}/${chunkTotal}, ${(plan.rangeBytes / 1e6).toFixed(1)} MB`,
          current: plan.h5ChunkIndex + 1,
          total: chunkTotal,
          percent: Math.min(70, (sourceFrames / plane) * 70),
          sourceFrames,
        });
        const fetchT = performance.now();
        return fetchRangeBytes(url, plan.rangeStart, plan.rangeBytes).then((range) => ({
          plan,
          range,
          fetchMs: performance.now() - fetchT,
          fetchStartMs: fetchT,
          fetchEndMs: performance.now(),
        }));
      };
      const inFlight = new Map<number, Promise<IndexedFetchResult>>();
      let nextFetchIndex = 0;
      let fileFetchStartMs = Number.POSITIVE_INFINITY;
      let fileFetchEndMs = 0;
      const queueFetches = () => {
        while (
          nextFetchIndex < plans.length
          && inFlight.size < indexedPrefetchWindow
        ) {
          inFlight.set(nextFetchIndex, startFetch(plans[nextFetchIndex]));
          nextFetchIndex += 1;
        }
      };
      queueFetches();
      for (let planIndex = 0; planIndex < plans.length; planIndex++) {
        const pendingFetch = inFlight.get(planIndex);
        if (!pendingFetch) break;
        const { plan, range, fetchMs: chunkFetchMs, fetchStartMs, fetchEndMs } = await pendingFetch;
        inFlight.delete(planIndex);
        fetchMs += chunkFetchMs;
        fileFetchStartMs = Math.min(fileFetchStartMs, fetchStartMs);
        fileFetchEndMs = Math.max(fileFetchEndMs, fetchEndMs);
        fetchBytes += plan.payloadBytes;
        queueFetches();
        const h5Chunk = makeIndexedBslz4Spec(
          range,
          index,
          plan.frameStart,
          plan.nFrames,
          plan.rangeStart,
        );
        await decodeAndGather(h5Chunk, index.srcDtype, detSize, fileIndex, plan.h5ChunkIndex, chunkTotal);
        if (sourceFrames >= plane) break;
        queueFetches();
      }
      if (Number.isFinite(fileFetchStartMs)) fetchWallMs += Math.max(0, fileFetchEndMs - fileFetchStartMs);
      if (sourceFrames >= plane) break;
      continue;
    }
    const fetchT = performance.now();
    let bufBytes: Uint8Array;
    try {
      bufBytes = await readSourceBytes(url);
    } catch (err) {
      if (source.dataUrls && source.dataUrls.length > 0) throw err;
      break;
    }
    const buf = bufBytes.buffer.slice(bufBytes.byteOffset, bufBytes.byteOffset + bufBytes.byteLength) as ArrayBuffer;
    const fetchElapsed = performance.now() - fetchT;
    fetchMs += fetchElapsed;
    fetchWallMs += fetchElapsed;
    fetchBytes += buf.byteLength;
    const parseT = performance.now();
    emit({
      stage: "parse",
      message: "Parsing HDF5 frame index",
      detail: `${(buf.byteLength / 1e6).toFixed(1)} MB compressed block`,
      current: fileIndex + 1,
      total: finiteDataUrlCount,
      sourceFrames,
    });
    const vol = readH5Volume(buf, `showptycho-data-${fileIndex}`);
    parseMs += performance.now() - parseT;
    if (vol.detSize !== Number(cal.detector_shape?.[0] || 0) * Number(cal.detector_shape?.[1] || 0)) {
      throw new Error(
        `${url}: detector shape mismatch; HDF5 has ${vol.detRows}x${vol.detCols}, `
        + `calibration has ${(cal.detector_shape || []).join("x")}`,
      );
    }
    for (let h5ChunkIndex = 0; h5ChunkIndex < vol.chunks.length; h5ChunkIndex++) {
      if (sourceFrames >= plane) break;
      const h5Chunk = vol.chunks[h5ChunkIndex];
      await decodeAndGather(h5Chunk, vol.srcDtype, vol.detSize, fileIndex, h5ChunkIndex, vol.chunks.length);
    }
    if (sourceFrames >= plane) break;
  }
  if (sourceFrames !== plane) {
    throw new Error(`HDF5 source frames ${sourceFrames} do not match phase grid ${n}x${n}=${plane}.`);
  }

  const fftT = performance.now();
  emit({
    stage: "fft",
    message: "Running WebGPU FFT for the BF reducer",
    detail: `${gqkChunks.length} chunk${gqkChunks.length === 1 ? "" : "s"}, ${activeSourceIndices.length} active BF pixels`,
    current: activeSourceIndices.length,
    total: activeSourceIndices.length,
    percent: 90,
    sourceFrames,
  });
  const enc = device.createCommandEncoder();
  for (let i = 0; i < gqkChunks.length; i++) {
    const chunkBf = chunkBfCounts[i];
    const rowBind = device.createBindGroup({
      layout: rowsPipe.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: gqkChunks[i] } },
        { binding: 1, resource: { buffer: fftParamBuffers[i] } },
      ],
    });
    const colBind = device.createBindGroup({
      layout: colsPipe.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: gqkChunks[i] } },
        { binding: 1, resource: { buffer: fftParamBuffers[i] } },
      ],
    });
    const rows = enc.beginComputePass();
    rows.setPipeline(rowsPipe);
    rows.setBindGroup(1, rowBind);
    rows.dispatchWorkgroups(1, n, chunkBf);
    rows.end();
    const cols = enc.beginComputePass();
    cols.setPipeline(colsPipe);
    cols.setBindGroup(1, colBind);
    cols.dispatchWorkgroups(1, n, chunkBf);
    cols.end();
  }
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  const fftMs = performance.now() - fftT;
  bfIndexBuffers.forEach(b => b.destroy());
  fftParamBuffers.forEach(b => b.destroy());
  console.log(
    `[showptycho] HDF5 source prepared ${activeSourceIndices.length} BF from ${sourceFrames} frames `
    + `compressed ${(fetchBytes / 1e9).toFixed(2)} GB, hot px ${badPixels}, `
    + `fetch ${fetchMs.toFixed(1)} ms wall ${fetchWallMs.toFixed(1)} ms parse ${parseMs.toFixed(1)} ms decode ${decodeMs.toFixed(1)} ms `
    + `gather ${gatherMs.toFixed(1)} ms fft ${fftMs.toFixed(1)} ms total ${(performance.now() - t0).toFixed(1)} ms`,
  );
  (globalThis as unknown as { __showptychoH5Profile?: unknown }).__showptychoH5Profile = {
    compressedGB: +(fetchBytes / 1e9).toFixed(3),
    sourceFrames,
    activeBf: activeSourceIndices.length,
    prefetch: indexedPrefetchWindow > 1,
    prefetchWindow: indexedPrefetchWindow,
    decodeChunkMiB: Math.round(h5DecodeChunkTargetBytes(device.limits.maxStorageBufferBindingSize) / (1024 * 1024)),
    fetchMs: Math.round(fetchMs),
    fetchWallMs: Math.round(fetchWallMs),
    parseMs: Math.round(parseMs),
    decodeMs: Math.round(decodeMs),
    gatherMs: Math.round(gatherMs),
    fftMs: Math.round(fftMs),
    totalMs: Math.round(performance.now() - t0),
  };
  emit({
    stage: "ready",
    message: "WebGPU reducer ready",
    detail: `Loaded ${(fetchBytes / 1e9).toFixed(2)} GB compressed HDF5; hot pixels ${badPixels}`,
    current: activeSourceIndices.length,
    total: activeSourceIndices.length,
    percent: 100,
    sourceFrames,
  });
  return { gqkChunks, chunkBfCounts, fetchBytes, fetchMs, fetchWallMs, parseMs, decodeMs, gatherMs, fftMs, sourceFrames };
}

function makeParams(
  cal: SsbCal,
  n: SupportedSsbSize,
  c10: number,
  c12: number,
  phi12Deg: number,
  bfCount?: number,
  bfOffset = 0,
  chunkBf?: number,
  fullStack = false,
  computeLoss = true,
  activeBfCount = bfCount,
  fullAberration = false,
): ArrayBuffer {
  const plane = n * n;
  const b = new ArrayBuffer(96);
  const u = new Uint32Array(b);
  const f = new Float32Array(b);
  const phi12Rad = phi12Deg * Math.PI / 180;
  u[0] = Math.max(1, Math.min(cal.num_bf, Math.round(bfCount || cal.num_bf)));
  u[1] = n; u[2] = plane; u[3] = Math.max(0, Math.min(u[0] - 1, Math.round(bfOffset)));
  f[4] = cal.wavelength_A;
  f[5] = cal.semiangle_rad;
  f[6] = cal.angular_sampling_rad[0];
  f[7] = cal.angular_sampling_rad[1];
  f[8] = c10;
  f[9] = c12;
  f[10] = Math.cos(2 * phi12Rad);
  f[11] = Math.sin(2 * phi12Rad);
  f[12] = Math.PI / cal.wavelength_A;
  f[13] = cal.dc_value[0];
  f[14] = cal.dc_value[1];
  f[15] = 1 / plane;
  u[16] = Math.max(1, Math.min(u[0], Math.round(chunkBf || u[0])));
  u[17] = fullStack ? 1 : 0;
  u[18] = Math.max(1, Math.ceil(Math.max(1, Math.round(activeBfCount || u[0])) / REDUCE_BF_GROUP));
  u[19] = computeLoss ? 1 : 0;
  u[20] = Math.max(1, Math.round(activeBfCount || u[0]));
  u[21] = fullAberration ? 1 : 0;
  return b;
}

const ABR_INV_N_PLUS_ONE = [
  1 / 2, 1 / 2,
  1 / 3, 1 / 3,
  1 / 4, 1 / 4, 1 / 4,
  1 / 5, 1 / 5, 1 / 5,
  1 / 6, 1 / 6, 1 / 6, 1 / 6,
];
const ABR_M_VALUES = [0, 2, 1, 3, 0, 2, 4, 1, 3, 5, 0, 2, 4, 6];
const HO_LAYOUT: Array<[string, number, boolean]> = [
  ["C21", 2, true], ["C23", 3, true],
  ["C30", 4, false], ["C32", 5, true], ["C34", 6, true],
  ["C41", 7, true], ["C43", 8, true], ["C45", 9, true],
  ["C50", 10, false], ["C52", 11, true],
  ["C54", 12, true], ["C56", 13, true],
];

function packAberrations(
  c10: number,
  c12: number,
  phi12Deg: number,
  higherOrder?: Record<string, number>,
): { data: Float32Array; active: boolean } {
  const mags = new Float32Array(14);
  const angles = new Float32Array(14);
  mags[0] = c10;
  mags[1] = c12;
  angles[1] = phi12Deg * Math.PI / 180;
  let active = false;
  const ho = higherOrder || {};
  for (const [name, index, hasAngle] of HO_LAYOUT) {
    const mag = Number(ho[hasAngle ? `${name}_mag` : name] || 0);
    const angleDeg = hasAngle ? Number(ho[`${name}_angle`] || 0) : 0;
    mags[index] = mag;
    angles[index] = angleDeg * Math.PI / 180;
    if (mag !== 0) active = true;
  }
  const out = new Float32Array(14 * 4);
  for (let i = 0; i < 14; i++) {
    const theta = ABR_M_VALUES[i] * angles[i];
    out[i * 4 + 0] = mags[i] * ABR_INV_N_PLUS_ONE[i];
    out[i * 4 + 1] = Math.cos(theta);
    out[i * 4 + 2] = Math.sin(theta);
    out[i * 4 + 3] = 0;
  }
  return { data: out, active };
}

function baseRotationDeg(cal: SsbCal): number {
  if (Number.isFinite(cal.rotation_angle_deg)) return Number(cal.rotation_angle_deg);
  if (Number.isFinite(cal.rotation_angle_rad)) return Number(cal.rotation_angle_rad) * 180 / Math.PI;
  return 0;
}

function rotateBfCoord(cal: SsbCal, kx: number, ky: number, rotationDeg?: number): [number, number] {
  if (rotationDeg == null || !Number.isFinite(rotationDeg)) return [kx, ky];
  const delta = (Number(rotationDeg) - baseRotationDeg(cal)) * Math.PI / 180;
  if (Math.abs(delta) < 1e-12) return [kx, ky];
  const cosA = Math.cos(-delta);
  const sinA = Math.sin(-delta);
  return [
    kx * cosA + ky * sinA,
    -kx * sinA + ky * cosA,
  ];
}

function bfGeometryFromCoord(cal: SsbCal, kx: number, ky: number): [number, number, number, number] {
  const dx2 = kx * kx;
  const dy2 = ky * ky;
  const r2 = dx2 + dy2;
  const r = Math.sqrt(r2);
  const alpha = r * cal.wavelength_A;
  const alpha2 = alpha * alpha;
  const invR2 = r2 > 1e-30 ? 1 / r2 : 0;
  const cos2phi = (dx2 - dy2) * invR2;
  const sin2phi = 2 * kx * ky * invR2;
  const denomNum2 = (kx * cal.angular_sampling_rad[0]) ** 2 + (ky * cal.angular_sampling_rad[1]) ** 2;
  const denom = r > 1e-15 ? Math.sqrt(denomNum2) / r : 0;
  const edge = denom > 1e-15 ? (cal.semiangle_rad - alpha) / denom + 0.5 : 1;
  return [alpha2, cos2phi, sin2phi, Math.max(0, Math.min(1, edge))];
}

function collectActiveBfIndices(cal: SsbCal, numBf: number, rotationDeg?: number): Uint32Array {
  const count = Math.max(1, Math.min(cal.num_bf, Math.round(numBf)));
  const active: number[] = [];
  for (let i = 0; i < count; i++) {
    const [kx, ky] = rotateBfCoord(cal, cal.kx_bf[i], cal.ky_bf[i], rotationDeg);
    const [, , , aperture] = bfGeometryFromCoord(cal, kx, ky);
    if (aperture !== 0) active.push(i);
  }
  return new Uint32Array(active);
}

function packGeometry(cal: SsbCal, indices: Uint32Array, rotationDeg?: number): { geom: Float32Array; trig: Float32Array } {
  const count = Math.max(1, indices.length);
  const geom = new Float32Array(count * 4);
  const trig = new Float32Array(count * 2);
  if (indices.length === 0) {
    return { geom, trig };
  }
  for (let i = 0; i < indices.length; i++) {
    const src = indices[i];
    const [kx, ky] = rotateBfCoord(cal, cal.kx_bf[src], cal.ky_bf[src], rotationDeg);
    const [alpha2, cos2phi, sin2phi, aperture] = bfGeometryFromCoord(cal, kx, ky);
    geom[i * 4 + 0] = kx;
    geom[i * 4 + 1] = ky;
    geom[i * 4 + 2] = alpha2;
    geom[i * 4 + 3] = aperture;
    trig[i * 2 + 0] = cos2phi;
    trig[i * 2 + 1] = sin2phi;
  }
  return { geom, trig };
}

async function readF32(device: GPUDevice, buffer: GPUBuffer, length: number): Promise<Float32Array> {
  const read = device.createBuffer({
    size: length * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buffer, 0, read, 0, length * 4);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(read.getMappedRange().slice(0));
  read.unmap();
  read.destroy();
  return out;
}

function bytesFromDataView(view: DataView): Uint8Array {
  return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
}

async function fetchRangeBytes(url: string, byteOffset: number, byteLength: number): Promise<Uint8Array> {
  const bytes = await readSourceBytes(url, byteOffset, byteLength);
  if (bytes.byteLength < byteLength) {
    throw new Error(
      `${url}: received ${(bytes.byteLength / 1e6).toFixed(1)} MB; `
      + `need ${(byteLength / 1e6).toFixed(1)} MB.`,
    );
  }
  return bytes.byteLength === byteLength ? bytes : bytes.slice(0, byteLength);
}

function readBE32(bytes: Uint8Array, off: number): number {
  return ((bytes[off] << 24) | (bytes[off + 1] << 16) | (bytes[off + 2] << 8) | bytes[off + 3]) >>> 0;
}

function srcDtypeFromH5(dtype: string | undefined): "uint8" | "uint16" | "uint32" | "float32" {
  const raw = String(dtype || "").toLowerCase();
  if (raw.includes("float32") || raw.includes("f4")) return "float32";
  if (raw.includes("uint8") || raw.includes("u1")) return "uint8";
  if (raw.includes("uint32") || raw.includes("u4")) return "uint32";
  return "uint16";
}

function srcBytesFor(dtype: "uint8" | "uint16" | "uint32" | "float32"): number {
  return dtype === "uint8" ? 1 : dtype === "uint16" ? 2 : 4;
}

type ParsedChunkIndex = {
  offsets: number[];
  sizes: number[];
  frames: number;
  detRows: number;
  detCols: number;
  srcDtype: "uint8" | "uint16" | "uint32" | "float32";
};

async function fetchChunkIndex(index: H5ChunkIndexSource, fallbackUrl: string): Promise<ParsedChunkIndex> {
  const indexUrl = index.url || index.path;
  if (!indexUrl) throw new Error("HDF5 chunk index is missing a URL.");
  const expectedBytes = Math.max(0, Math.round(index.frames || 0)) * 16;
  const byteOffset = Math.max(0, Math.round(Number(index.byte_offset || 0)));
  const byteLength = Math.max(0, Math.round(Number(index.bytes || expectedBytes)));
  const idxBytes = byteOffset > 0 || byteLength > 0
    ? await fetchRangeBytes(indexUrl, byteOffset, byteLength || expectedBytes)
    : await readSourceBytes(indexUrl);
  const buf = idxBytes.buffer.slice(idxBytes.byteOffset, idxBytes.byteOffset + idxBytes.byteLength);
  const frames = Math.floor(buf.byteLength / 16);
  const dv = new DataView(buf);
  const offsets = new Array<number>(frames);
  const sizes = new Array<number>(frames);
  for (let i = 0; i < frames; i++) {
    offsets[i] = Number(dv.getBigUint64(i * 16, true));
    sizes[i] = Number(dv.getBigUint64(i * 16 + 8, true));
  }
  const shape = index.detector_shape || [0, 0];
  if (!shape[0] || !shape[1]) {
    throw new Error(`${fallbackUrl}: HDF5 chunk index is missing detector_shape.`);
  }
  return {
    offsets,
    sizes,
    frames,
    detRows: Number(shape[0]),
    detCols: Number(shape[1]),
    srcDtype: srcDtypeFromH5(index.dtype),
  };
}

function indexedFramesPerChunk(
  detSize: number,
  decodeDtype: "uint8" | "uint16" | "float32",
  maxStorageBindingSize?: number,
): number {
  const bytesPerPixel = decodeDtype === "float32" ? 4 : decodeDtype === "uint16" ? 2 : 1;
  return Math.max(1, Math.floor(h5DecodeChunkTargetBytes(maxStorageBindingSize) / Math.max(1, detSize * bytesPerPixel)));
}

function h5DecodeChunkTargetBytes(maxStorageBindingSize?: number): number {
  const requested = 1792 * 1024 * 1024;
  const safety = 64 * 1024 * 1024;
  if (!maxStorageBindingSize || !Number.isFinite(maxStorageBindingSize)) return requested;
  return Math.max(256 * 1024 * 1024, Math.min(requested, Math.floor(maxStorageBindingSize) - safety));
}

function h5IndexedPrefetchWindow(): number {
  const raw = typeof globalThis !== "undefined"
    ? (globalThis as { __QUANTEM_SHOWPTYCHO_H5_PREFETCH_WINDOW__?: number }).__QUANTEM_SHOWPTYCHO_H5_PREFETCH_WINDOW__
    : undefined;
  if (Number.isFinite(raw)) return Math.max(1, Math.min(6, Math.floor(Number(raw))));
  return 1;
}

function makeIndexedBslz4Spec(
  raw: Uint8Array,
  index: ParsedChunkIndex,
  startFrame: number,
  nFrames: number,
  rangeStart: number,
): Bslz4Spec {
  const detSize = index.detRows * index.detCols;
  const srcBytes = srcBytesFor(index.srcDtype);
  const firstFrameOffset = index.offsets[startFrame] - rangeStart;
  const blockBytes = readBE32(raw, firstFrameOffset + 8);
  const blockElems = blockBytes / srcBytes;
  const nBlocksPerFrame = Math.ceil(detSize / blockElems);
  const meta: number[] = [];
  for (let localFrame = 0; localFrame < nFrames; localFrame++) {
    const frame = startFrame + localFrame;
    const addr = index.offsets[frame] - rangeStart;
    let pos = 12;
    for (let block = 0; block < nBlocksPerFrame; block++) {
      const clen = readBE32(raw, addr + pos);
      meta.push(addr + pos + 4, clen);
      pos += 4 + clen;
    }
  }
  return {
    compressed: raw,
    blockMeta: new Uint32Array(meta),
    nFrames,
    nBlocksPerFrame,
    blockElems,
    detSize,
  };
}

async function fetchPackedActiveBfBytes(
  fullBytes: Uint8Array | null,
  url: string | null,
  activeIndices: Uint32Array,
  activeOffset: number,
  activeCount: number,
  bytesPerBf: number,
): Promise<Uint8Array> {
  const out = new Uint8Array(activeCount * bytesPerBf);
  if (activeIndices.length === 0) return out;
  let dstBf = 0;
  while (dstBf < activeCount) {
    const runActiveStart = activeOffset + dstBf;
    const srcStart = activeIndices[runActiveStart];
    let runBf = 1;
    while (
      dstBf + runBf < activeCount
      && activeIndices[runActiveStart + runBf] === srcStart + runBf
    ) {
      runBf += 1;
    }
    const srcByteOffset = srcStart * bytesPerBf;
    const runBytes = runBf * bytesPerBf;
    if (fullBytes && fullBytes.byteLength >= srcByteOffset + runBytes) {
      out.set(fullBytes.subarray(srcByteOffset, srcByteOffset + runBytes), dstBf * bytesPerBf);
    } else {
      if (!url) throw new Error("Missing BF-G WebGPU payload");
      out.set(await fetchRangeBytes(url, srcByteOffset, runBytes), dstBf * bytesPerBf);
    }
    dstBf += runBf;
  }
  return out;
}

function bfColumnMode(source: BfColumnSsbSource): number {
  const dtype = String(source.encoding || source.dtype || "uint16").toLowerCase();
  if (dtype === "uint4" || dtype === "u4") return 3;
  if (dtype === "uint8" || dtype === "u1") return 1;
  if (dtype === "float32" || dtype === "f4") return 2;
  return 0;
}

function bfColumnBytesPerBf(source: BfColumnSsbSource, plane: number): number {
  const declared = Number(source.bytesPerBf || 0);
  if (declared > 0) return declared;
  const mode = bfColumnMode(source);
  if (mode === 3) return Math.ceil(plane / 2);
  if (mode === 1) return plane;
  if (mode === 2) return plane * 4;
  return plane * 2;
}

async function buildBfColumnGqkChunks(
  device: GPUDevice,
  cal: SsbCal,
  source: BfColumnSsbSource,
  n: SupportedSsbSize,
  plane: number,
  activeSourceIndices: Uint32Array,
  storageChunkCapacity: number,
  onProgress?: WebGPUProgressHandler,
): Promise<H5GqkBuild> {
  const t0 = performance.now();
  const emit = (progress: Omit<WebGPULoadProgress, "elapsedMs" | "activeBf" | "totalBf">) => {
    onProgress?.({
      activeBf: activeSourceIndices.length,
      totalBf: cal.num_bf,
      elapsedMs: performance.now() - t0,
      ...progress,
    });
  };
  if (!source.url) throw new Error("BF-column source is missing a URL.");
  if (Number(source.plane) !== plane) {
    throw new Error(`BF-column source plane ${source.plane} does not match ${n}x${n}=${plane}.`);
  }
  const maxActiveSource = activeSourceIndices.reduce((max, value) => Math.max(max, value), 0);
  if (Number(source.numBf) <= maxActiveSource) {
    throw new Error(`BF-column source has ${source.numBf} BF columns; need source BF index ${maxActiveSource}.`);
  }
  emit({
    stage: "pipeline",
    message: "Preparing WebGPU kernels",
    detail: `Building BF-column unpack and FFT kernels for ${n}x${n}`,
    current: 0,
    total: activeSourceIndices.length,
    percent: 10,
    sourceFrames: plane,
  });
  const module = device.createShaderModule({ code: makeH5GqkShader(n), label: `ShowPtycho BF-column gather+FFT ${n}` });
  const unpackPipe = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "bfColumnsToGqk" } });
  const rowsPipe = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "fftRows" } });
  const colsPipe = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "fftCols" } });
  const mode = bfColumnMode(source);
  const bytesPerBf = bfColumnBytesPerBf(source, plane);
  const gqkChunks: GPUBuffer[] = [];
  const chunkBfCounts: number[] = [];
  const fftParamBuffers: GPUBuffer[] = [];
  for (let bfOffset = 0; bfOffset < activeSourceIndices.length; bfOffset += storageChunkCapacity) {
    const chunkBf = Math.min(storageChunkCapacity, activeSourceIndices.length - bfOffset);
    gqkChunks.push(device.createBuffer({
      size: chunkBf * plane * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: `showptycho transient bf-column G(q,k) ${bfOffset}`,
    }));
    chunkBfCounts.push(chunkBf);
    fftParamBuffers.push(uniformU32(device, [chunkBf, plane, 0, 0], `showptycho bf-column fft params ${bfOffset}`));
  }

  let fetchBytes = 0;
  let fetchMs = 0;
  let gatherMs = 0;
  for (let bfOffset = 0; bfOffset < activeSourceIndices.length; bfOffset += storageChunkCapacity) {
    const chunkIndex = Math.floor(bfOffset / storageChunkCapacity);
    const chunkBf = chunkBfCounts[chunkIndex];
    emit({
      stage: "fetch",
      message: "Reading detector BF columns",
      detail: `${bfOffset + chunkBf}/${activeSourceIndices.length} active BF columns, ${(chunkBf * bytesPerBf / 1e6).toFixed(1)} MB`,
      current: bfOffset + chunkBf,
      total: activeSourceIndices.length,
      percent: Math.min(70, ((bfOffset + chunkBf) / activeSourceIndices.length) * 65 + 10),
      sourceFrames: plane,
    });
    const fetchT = performance.now();
    const packed = await fetchPackedActiveBfBytes(null, source.url, activeSourceIndices, bfOffset, chunkBf, bytesPerBf);
    fetchMs += performance.now() - fetchT;
    fetchBytes += packed.byteLength;
    const gatherT = performance.now();
    const dataBuffer = makeBufferFromBytes(
      device,
      packed,
      GPUBufferUsage.STORAGE,
      `showptycho bf-column packed ${bfOffset}`,
    );
    const params = uniformU32(
      device,
      [0, 0, 0, mode, chunkBf, plane, 0, 0],
      `showptycho bf-column unpack params ${bfOffset}`,
    );
    const bind = device.createBindGroup({
      layout: unpackPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dataBuffer } },
        { binding: 2, resource: { buffer: gqkChunks[chunkIndex] } },
        { binding: 3, resource: { buffer: params } },
      ],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass({ label: "showptycho bf-column unpack" });
    pass.setPipeline(unpackPipe);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(
      Math.ceil(plane / BF_COLUMN_UNPACK_WORKGROUP_X),
      Math.ceil(chunkBf / BF_COLUMN_UNPACK_WORKGROUP_Y),
    );
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    dataBuffer.destroy();
    params.destroy();
    gatherMs += performance.now() - gatherT;
    emit({
      stage: "gather",
      message: "Preparing BF reducer",
      detail: `${bfOffset + chunkBf}/${activeSourceIndices.length} BF columns unpacked`,
      current: bfOffset + chunkBf,
      total: activeSourceIndices.length,
      percent: Math.min(82, ((bfOffset + chunkBf) / activeSourceIndices.length) * 72 + 10),
      sourceFrames: plane,
    });
  }

  const fftT = performance.now();
  emit({
    stage: "fft",
    message: "Running WebGPU FFT for the BF reducer",
    detail: `${gqkChunks.length} chunk${gqkChunks.length === 1 ? "" : "s"}, ${activeSourceIndices.length} active BF pixels`,
    current: activeSourceIndices.length,
    total: activeSourceIndices.length,
    percent: 90,
    sourceFrames: plane,
  });
  const enc = device.createCommandEncoder();
  for (let i = 0; i < gqkChunks.length; i++) {
    const chunkBf = chunkBfCounts[i];
    const rowBind = device.createBindGroup({
      layout: rowsPipe.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: gqkChunks[i] } },
        { binding: 1, resource: { buffer: fftParamBuffers[i] } },
      ],
    });
    const colBind = device.createBindGroup({
      layout: colsPipe.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: gqkChunks[i] } },
        { binding: 1, resource: { buffer: fftParamBuffers[i] } },
      ],
    });
    const rows = enc.beginComputePass();
    rows.setPipeline(rowsPipe);
    rows.setBindGroup(1, rowBind);
    rows.dispatchWorkgroups(1, n, chunkBf);
    rows.end();
    const cols = enc.beginComputePass();
    cols.setPipeline(colsPipe);
    cols.setBindGroup(1, colBind);
    cols.dispatchWorkgroups(1, n, chunkBf);
    cols.end();
  }
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  const fftMs = performance.now() - fftT;
  fftParamBuffers.forEach(b => b.destroy());
  const totalMs = performance.now() - t0;
  console.log(
    `[showptycho] BF-column source prepared ${activeSourceIndices.length} BF from ${plane} scan positions `
    + `read ${(fetchBytes / 1e9).toFixed(2)} GB ${source.encoding || source.dtype || "uint16"}, `
    + `fetch ${fetchMs.toFixed(1)} ms unpack ${gatherMs.toFixed(1)} ms fft ${fftMs.toFixed(1)} ms total ${totalMs.toFixed(1)} ms`,
  );
  (globalThis as unknown as { __showptychoBfColumnProfile?: unknown }).__showptychoBfColumnProfile = {
    transport: "bf_columns",
    encodedGB: +(fetchBytes / 1e9).toFixed(3),
    sourceFrames: plane,
    activeBf: activeSourceIndices.length,
    bytesPerBf,
    encoding: source.encoding || source.dtype || "uint16",
    fetchMs: Math.round(fetchMs),
    fetchWallMs: Math.round(fetchMs),
    parseMs: 0,
    decodeMs: 0,
    gatherMs: Math.round(gatherMs),
    fftMs: Math.round(fftMs),
    totalMs: Math.round(totalMs),
  };
  emit({
    stage: "ready",
    message: "WebGPU reducer ready",
    detail: `Loaded ${(fetchBytes / 1e9).toFixed(2)} GB detector BF columns`,
    current: activeSourceIndices.length,
    total: activeSourceIndices.length,
    percent: 100,
    sourceFrames: plane,
  });
  return {
    gqkChunks,
    chunkBfCounts,
    fetchBytes,
    fetchMs,
    fetchWallMs: fetchMs,
    parseMs: 0,
    decodeMs: 0,
    gatherMs,
    fftMs,
    sourceFrames: plane,
  };
}

function chooseChunkCapacity(nbf: number, plane: number, device: GPUDevice): number {
  const maxStorage = device.limits.maxStorageBufferBindingSize || Number.MAX_SAFE_INTEGER;
  const maxBuffer = device.limits.maxBufferSize || Number.MAX_SAFE_INTEGER;
  const safeStorage = maxStorage === Number.MAX_SAFE_INTEGER ? maxStorage : Math.floor(maxStorage * 0.75);
  const safeBuffer = maxBuffer === Number.MAX_SAFE_INTEGER ? maxBuffer : Math.floor(maxBuffer * 0.75);
  const maxBytes = Math.max(1, Math.min(safeStorage, safeBuffer, MAX_GQK_CHUNK_BYTES));
  const maxByComplexBuffer = Math.max(1, Math.floor(maxBytes / (plane * 8)));
  const override = typeof globalThis !== "undefined"
    ? (globalThis as { __QUANTEM_SHOWPTYCHO_CHUNK_BF_CAP__?: number }).__QUANTEM_SHOWPTYCHO_CHUNK_BF_CAP__
    : undefined;
  const diagnosticCap = Number.isFinite(override) ? Math.max(1, Math.floor(Number(override))) : Number.MAX_SAFE_INTEGER;
  const raw = Math.max(1, Math.min(nbf, MAX_BF_WORKGROUPS_PER_SUBMIT, maxByComplexBuffer, diagnosticCap));
  if (raw <= REDUCE_BF_GROUP) return raw;
  return Math.max(REDUCE_BF_GROUP, Math.floor(raw / REDUCE_BF_GROUP) * REDUCE_BF_GROUP);
}

function canUseFullStack(nbf: number, plane: number, device: GPUDevice): boolean {
  // The full-stack path keeps one giant stage/phase stack and is disabled for
  // the browser review contract; chunking remains safer for multi-GB folders.
  void nbf; void plane; void device;
  return false;
}

export class ShowPtychoWebGPUSSB {
  private cal: SsbCal;
  private n: SupportedSsbSize;
  private plane: number;
  private previewBfCount: number;
  private source: SsbSource;
  private device: GPUDevice | null = null;
  private buffers: SsbBuffers | null = null;
  private pipelines: SsbPipelines | null = null;
  private bindGroups: SsbBindGroups | null = null;
  private setupPromise: Promise<void> | null = null;
  private operationQueue: Promise<void> = Promise.resolve();
  private initialized = false;
  private loadedBfCount = 0;
  private geometryRotationDeg: number | null = null;
  private onProgress: WebGPUProgressHandler | null = null;

  constructor(calJson: string, gBfSource: DataView | string | { kind: "hdf5" | "bf_columns"; json: string }) {
    if (!globalThis.isSecureContext || !navigator.gpu) {
      throw new Error(
        "ShowPtycho WebGPU folder needs browser WebGPU. Open it from http://127.0.0.1, localhost, or HTTPS; LAN-IP HTTP pages cannot use WebGPU.",
      );
    }
    this.cal = JSON.parse(calJson) as SsbCal;
    const ny = Number(this.cal.g_shape?.[1] || 0);
    const nx = Number(this.cal.g_shape?.[2] || 0);
    // Accept legacy square (n x n) and rfft half-plane (n x n/2+1)
    // calibration shapes - CUDA/MPS backends store Hermitian-half G_qk and export
    // that shape. The viewer only derives the scan size n here; it rebuilds
    // its own G(q,k) from the folder source, so the backend layout is
    // irrelevant beyond n. Flattened scan positions (N, det, det) are already
    // squared by the loaders before g_shape is written.
    const squareN = ny === nx ? ny : (nx === ny / 2 + 1 ? ny : 0);
    const n = supportedSsbSize(squareN);
    if (!n) {
      throw new Error(
        `WebGPU SSB supports square ${SUPPORTED_SSB_SIZES.join("/")} G(k) or rfft half-plane G(k), got ${this.cal.g_shape?.join("x")}`,
      );
    }
    this.n = n;
    this.plane = n * n;
    this.previewBfCount = Math.max(
      1,
      Math.min(this.cal.num_bf, Math.round(Number(this.cal.num_bf || 1))),
    );
    if (typeof gBfSource === "string") {
      this.source = { kind: "g-bf", bytes: null, url: gBfSource };
    } else if (gBfSource && typeof gBfSource === "object" && "kind" in gBfSource && gBfSource.kind === "hdf5") {
      const parsed = JSON.parse(gBfSource.json) as H5SsbSource | BfColumnSsbSource;
      this.source = parsed.kind === "bf_columns"
        ? { kind: "bf-columns", source: parsed as BfColumnSsbSource }
        : { kind: "hdf5", source: parsed as H5SsbSource };
    } else if (gBfSource && typeof gBfSource === "object" && "kind" in gBfSource && gBfSource.kind === "bf_columns") {
      this.source = { kind: "bf-columns", source: JSON.parse(gBfSource.json) as BfColumnSsbSource };
    } else {
      this.source = { kind: "g-bf", bytes: bytesFromDataView(gBfSource as DataView), url: null };
    }
  }

  get readyLabel(): string {
    const src = this.source.kind === "hdf5"
      ? "compressed HDF5 source"
      : this.source.kind === "bf-columns" ? "detector BF columns" : "BF-G cache";
    return `WebGPU SSB ready: ${this.n}x${this.n}, ${this.cal.num_bf} BF pixels, ${src}`;
  }

  setProgressHandler(handler: WebGPUProgressHandler | null): void {
    this.onProgress = handler;
  }

  private emitProgress(progress: Omit<WebGPULoadProgress, "totalBf">): void {
    this.onProgress?.({
      totalBf: this.cal.num_bf,
      elapsedMs: undefined,
      ...progress,
    });
  }

  private async setup(requiredBfCount = this.cal.num_bf, rotationDeg?: number): Promise<void> {
    const capacity = this.clampBfCount(requiredBfCount, false);
    if (this.initialized && this.loadedBfCount >= capacity) {
      this.updateGeometryRotation(rotationDeg);
      return;
    }
    if (this.setupPromise) {
      await this.setupPromise;
      if (this.initialized && this.loadedBfCount >= capacity) {
        this.updateGeometryRotation(rotationDeg);
        return;
      }
    }
    this.setupPromise = this.setupOnce(capacity, rotationDeg);
    try {
      await this.setupPromise;
    } finally {
      this.setupPromise = null;
    }
    this.updateGeometryRotation(rotationDeg);
  }

  private async runExclusive<T>(operation: () => Promise<T>): Promise<T> {
    const previous = this.operationQueue;
    let release = () => {};
    this.operationQueue = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous.catch(() => undefined);
    try {
      return await operation();
    } finally {
      release();
    }
  }

  private resetGpuBuffers(): void {
    if (this.buffers) {
      this.buffers.gqkChunks.forEach(buffer => buffer.destroy());
      this.buffers.gqkScales?.destroy();
      this.buffers.paramsChunks.forEach(buffer => buffer.destroy());
      this.buffers.params.destroy();
      this.buffers.aberrations.destroy();
      this.buffers.stage.destroy();
      this.buffers.phaseStack.destroy();
      this.buffers.partialSum.destroy();
      this.buffers.partialSumSq.destroy();
      this.buffers.phase.destroy();
      this.buffers.variance.destroy();
      this.buffers.bfGeom.destroy();
      this.buffers.bfTrig.destroy();
      this.buffers.qx.destroy();
      this.buffers.qy.destroy();
    }
    this.buffers = null;
    this.pipelines = null;
    this.bindGroups = null;
    this.initialized = false;
    this.loadedBfCount = 0;
    this.geometryRotationDeg = null;
  }

  private async setupOnce(capacity: number, rotationDeg?: number): Promise<void> {
    if (this.initialized && this.loadedBfCount >= capacity) return;
    if (this.initialized || this.buffers) this.resetGpuBuffers();
    const setupT0 = performance.now();
    const n = this.n;
    const plane = this.plane;
    this.emitProgress({
      stage: "device",
      message: "Starting WebGPU for ptychography review",
      detail: `Preparing ${capacity}/${this.cal.num_bf} BF pixels at ${n}x${n}`,
      current: 0,
      total: capacity,
      percent: 0,
      activeBf: capacity,
    });
    const device = this.device || await getGPUDevice();
    if (!device) throw new Error("WebGPU device unavailable");
    if (
      device.limits.maxComputeInvocationsPerWorkgroup < Math.min(n, 256)
      || device.limits.maxComputeWorkgroupSizeX < Math.min(n, 256)
    ) {
      throw new Error(
        `WebGPU adapter supports workgroup x=${device.limits.maxComputeWorkgroupSizeX}, `
        + `invocations=${device.limits.maxComputeInvocationsPerWorkgroup}; `
        + `${n}x${n} SSB needs ${Math.min(n, 256)} threads plus ${(n * 8 / 1024).toFixed(1)} KB workgroup storage.`,
      );
    }
    this.device = device;
    const nbf = Math.max(1, Math.min(this.cal.num_bf, Math.round(capacity)));
    let activeSourceIndices = collectActiveBfIndices(this.cal, nbf, rotationDeg);
    // GPU-memory clamp: resident G(q,k) is activeBf x storedPlane x bytesPer.
    // Cap the active set so one full-BF drag cannot exceed the budget and
    // device-lost the tab on a small GPU. Uniform stride keeps BF coverage.
    const clampMode = resolveGqkMode();
    const perBfBytes = storedPlaneFor(n) * gqkBytesPerValue(clampMode);
    const budgetMaxBf = Math.max(1, Math.floor(gqkBudgetBytes() / perBfBytes));
    if (activeSourceIndices.length > budgetMaxBf) {
      const stride = activeSourceIndices.length / budgetMaxBf;
      const clamped = new Uint32Array(budgetMaxBf);
      for (let i = 0; i < budgetMaxBf; i++) clamped[i] = activeSourceIndices[Math.floor(i * stride)];
      console.warn(
        `[showptycho] BF clamped ${activeSourceIndices.length} -> ${budgetMaxBf} by GPU budget `
        + `(${(gqkBudgetBytes() / 1e9).toFixed(1)} GB, ${(perBfBytes / 1e6).toFixed(1)} MB/BF in ${clampMode} mode)`,
      );
      this.emitProgress({
        stage: "pipeline",
        message: `BF capped at ${budgetMaxBf} by GPU memory budget`,
        detail: `${(gqkBudgetBytes() / 1e9).toFixed(1)} GB budget, ${(perBfBytes / 1e6).toFixed(1)} MB per BF pixel (${clampMode})`,
        activeBf: budgetMaxBf,
      });
      activeSourceIndices = clamped;
    }
    const nonzeroBfCount = activeSourceIndices.length;
    const activeIndices = nonzeroBfCount > 0 ? activeSourceIndices : new Uint32Array([0]);
    const activeBfCount = activeIndices.length;
    const bytesPerBf = plane * Float32Array.BYTES_PER_ELEMENT * 2;
    const expectedBytes = nbf * bytesPerBf;
    const activeBytes = activeBfCount * bytesPerBf;
    const fullStack = canUseFullStack(nbf, plane, device);
    const storageChunkCapacity = fullStack ? activeBfCount : chooseChunkCapacity(activeBfCount, plane, device);
    const dispatchChunkCapacity = fullStack
      ? Math.min(activeBfCount, MAX_BF_WORKGROUPS_PER_SUBMIT)
      : storageChunkCapacity;
    const dispatchCount = Math.ceil(activeBfCount / dispatchChunkCapacity);
    const partialGroups = Math.ceil(activeBfCount / REDUCE_BF_GROUP);
    console.log(
      `[showptycho] WebGPU SSB setup start ${nbf}/${this.cal.num_bf} BF `
      + `${(expectedBytes / 1e6).toFixed(1)} MB source; `
      + `${nonzeroBfCount}/${nbf} active aperture BF `
      + `${(activeBytes / 1e6).toFixed(1)} MB upload `
      + `${fullStack ? "full-stack" : `in chunks of ${storageChunkCapacity}`}`,
    );
    console.log(
      `[showptycho] WebGPU limits buffer=${device.limits.maxBufferSize} `
      + `storageBinding=${device.limits.maxStorageBufferBindingSize} `
      + `storageBuffers=${device.limits.maxStorageBuffersPerShaderStage}`,
    );
    this.emitProgress({
      stage: "pipeline",
      message: "Allocating WebGPU BF reducer buffers",
      detail: `${activeBfCount}/${this.cal.num_bf} active BF pixels, ${(activeBytes / 1e6).toFixed(1)} MB working data`,
      current: 1,
      total: 5,
      percent: 8,
      activeBf: activeBfCount,
    });
    const { geom, trig } = packGeometry(this.cal, activeIndices, rotationDeg);
    const maxStorage = device.limits.maxStorageBufferBindingSize || 0;
    const maxBuffer = device.limits.maxBufferSize || 0;
    const largestStorage = Math.max(storageChunkCapacity * plane * 8, storageChunkCapacity * plane * 4);
    if ((maxStorage > 0 && largestStorage > maxStorage) || (maxBuffer > 0 && largestStorage > maxBuffer)) {
      throw new Error(
        `WebGPU buffer limit is too small for chunked ${nbf}x${n}x${n}: `
        + `need ${(largestStorage / 1e6).toFixed(1)} MB, `
        + `maxStorage ${(maxStorage / 1e6).toFixed(1)} MB, maxBuffer ${(maxBuffer / 1e6).toFixed(1)} MB.`,
      );
    }
    const gqkStorageMode = resolveGqkMode();
    const module = device.createShaderModule({ code: makeSsbShader(n, gqkStorageMode), label: `ShowPtycho SSB WGSL ${n} ${gqkStorageMode}` });
    const pipelines: SsbPipelines = {
      rows: device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "ssbRows" } }),
      cols: device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "ssbCols" } }),
      reducePartial: device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "reducePartialGroups" } }),
      finalizeGroups: device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "finalizePartialGroups" } }),
      objSum: device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "ssbObjSum" } }),
      objFftRows: device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "ssbObjFftRows" } }),
      objFftCols: device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "ssbObjFftCols" } }),
    };
    let gqkChunks: GPUBuffer[] = [];
    let chunkBfCounts: number[] = [];
    if (this.source.kind === "bf-columns") {
      const built = await buildBfColumnGqkChunks(
        device,
        this.cal,
        this.source.source,
        n,
        plane,
        activeIndices,
        storageChunkCapacity,
        (progress) => this.onProgress?.({
          ...progress,
          elapsedMs: progress.elapsedMs ?? performance.now() - setupT0,
        }),
      );
      gqkChunks = built.gqkChunks;
      chunkBfCounts = built.chunkBfCounts;
    } else if (this.source.kind === "hdf5") {
      const built = await buildH5GqkChunks(
        device,
        this.cal,
        this.source.source,
        n,
        plane,
        activeIndices,
        storageChunkCapacity,
        (progress) => this.onProgress?.({
          ...progress,
          elapsedMs: progress.elapsedMs ?? performance.now() - setupT0,
        }),
      );
      gqkChunks = built.gqkChunks;
      chunkBfCounts = built.chunkBfCounts;
    } else {
      for (let bfOffset = 0; bfOffset < activeBfCount; bfOffset += storageChunkCapacity) {
        const chunkBf = Math.min(storageChunkCapacity, activeBfCount - bfOffset);
        const chunkBytes = chunkBf * bytesPerBf;
        const gBytes = await fetchPackedActiveBfBytes(this.source.bytes, this.source.url, activeSourceIndices, bfOffset, chunkBf, bytesPerBf);
        if (gBytes.byteLength !== chunkBytes) {
          throw new Error(
            `G(k) chunk is ${(gBytes.byteLength / 1e6).toFixed(1)} MB; `
            + `expected ${(chunkBytes / 1e6).toFixed(1)} MB for ${chunkBf}x${n}x${n} complex64.`,
          );
        }
        this.emitProgress({
          stage: "fetch",
          message: "Reading BF-G cache",
          detail: `${bfOffset + chunkBf}/${activeBfCount} BF pixels uploaded`,
          current: bfOffset + chunkBf,
          total: activeBfCount,
          percent: Math.min(85, ((bfOffset + chunkBf) / activeBfCount) * 75),
          activeBf: activeBfCount,
          elapsedMs: performance.now() - setupT0,
        });
        gqkChunks.push(makeBufferFromBytes(device, gBytes, GPUBufferUsage.STORAGE, `showptycho g_bf chunk ${bfOffset}`));
        chunkBfCounts.push(chunkBf);
        const sourceStart = nonzeroBfCount > 0 ? activeSourceIndices[bfOffset] : 0;
        const sourceEnd = nonzeroBfCount > 0 ? activeSourceIndices[bfOffset + chunkBf - 1] : 0;
        console.log(
          `[showptycho] WebGPU SSB setup chunk ${gqkChunks.length} active ${bfOffset}-${bfOffset + chunkBf - 1} `
          + `source ${sourceStart}-${sourceEnd} `
          + `uploaded in ${(performance.now() - setupT0).toFixed(1)} ms`,
        );
      }
    }
    const transformed = await transformGqkChunks(device, n, gqkStorageMode, gqkChunks, chunkBfCounts, activeBfCount);
    gqkChunks = transformed.chunks;
    console.log(
      `[showptycho] gqk mode ${gqkStorageMode}: resident ${(transformed.residentBytes / 1e9).toFixed(2)} GB `
      + `for ${chunkBfCounts.reduce((acc, bf) => acc + bf, 0)} active BF pixels`,
    );
    const buffers: SsbBuffers = {
      params: device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "showptycho ssb params" }),
      paramsChunks: Array.from({ length: dispatchCount }, (_, index) => device.createBuffer({
        size: 96,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        label: `showptycho ssb params chunk ${index}`,
      })),
      aberrations: makeBufferFromF32(
        device,
        packAberrations(0, 0, 0).data,
        GPUBufferUsage.STORAGE,
        "showptycho aberrations",
      ),
      gqkChunks,
      gqkScales: transformed.scales,
      gqkMode: gqkStorageMode,
      gqkResidentBytes: transformed.residentBytes,
      chunkBfCounts,
      chunkCapacity: storageChunkCapacity,
      dispatchChunkCapacity,
      fullStack,
      stage: device.createBuffer({ size: storageChunkCapacity * plane * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: "showptycho ssb stage" }),
      phaseStack: device.createBuffer({ size: storageChunkCapacity * plane * 4, usage: GPUBufferUsage.STORAGE, label: "showptycho phase stack" }),
      partialSum: device.createBuffer({ size: partialGroups * plane * 4, usage: GPUBufferUsage.STORAGE, label: "showptycho phase partial sum" }),
      partialSumSq: device.createBuffer({ size: partialGroups * plane * 4, usage: GPUBufferUsage.STORAGE, label: "showptycho phase partial sumsq" }),
      partialGroups,
      activeBfCount,
      nonzeroBfCount,
      activeSourceIndices: activeIndices,
      phase: device.createBuffer({ size: plane * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, label: "showptycho phase" }),
      variance: device.createBuffer({ size: plane * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, label: "showptycho variance" }),
      bfGeom: makeBufferFromF32(device, geom, GPUBufferUsage.STORAGE, "showptycho bf geometry"),
      bfTrig: makeBufferFromF32(device, trig, GPUBufferUsage.STORAGE, "showptycho bf trig"),
      qx: makeBufferFromF32(device, new Float32Array(this.cal.qx_1d), GPUBufferUsage.STORAGE, "showptycho qx"),
      qy: makeBufferFromF32(device, new Float32Array(this.cal.qy_1d), GPUBufferUsage.STORAGE, "showptycho qy"),
    };
    const chunkBufferIndex = (index: number) => {
      const bfOffset = index * buffers.dispatchChunkCapacity;
      return Math.floor(bfOffset / buffers.chunkCapacity);
    };
    device.pushErrorScope("validation");
    const bindGroups: SsbBindGroups = {
      rows: buffers.paramsChunks.map((params, index) => {
        const entries: GPUBindGroupEntry[] = [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: buffers.gqkChunks[chunkBufferIndex(index)] } },
          { binding: 2, resource: { buffer: buffers.stage } },
          { binding: 3, resource: { buffer: buffers.bfGeom } },
          { binding: 4, resource: { buffer: buffers.bfTrig } },
          { binding: 5, resource: { buffer: buffers.qx } },
          { binding: 6, resource: { buffer: buffers.qy } },
          { binding: 12, resource: { buffer: buffers.aberrations } },
        ];
        if (buffers.gqkMode === "herm16" && buffers.gqkScales) {
          entries.push({ binding: 13, resource: { buffer: buffers.gqkScales } });
        }
        return device.createBindGroup({
          layout: pipelines.rows.getBindGroupLayout(0),
          entries,
        });
      }),
      cols: buffers.paramsChunks.map((params) => device.createBindGroup({
        layout: pipelines.cols.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 2, resource: { buffer: buffers.stage } },
          { binding: 3, resource: { buffer: buffers.bfGeom } },
          { binding: 7, resource: { buffer: buffers.phaseStack } },
        ],
      })),
      reducePartial: buffers.paramsChunks.map((params) => device.createBindGroup({
        layout: pipelines.reducePartial.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 3, resource: { buffer: buffers.bfGeom } },
          { binding: 7, resource: { buffer: buffers.phaseStack } },
          { binding: 8, resource: { buffer: buffers.partialSum } },
          { binding: 9, resource: { buffer: buffers.partialSumSq } },
        ],
      })),
      finalizeGroups: device.createBindGroup({
        layout: pipelines.finalizeGroups.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: buffers.params } },
          { binding: 8, resource: { buffer: buffers.partialSum } },
          { binding: 9, resource: { buffer: buffers.partialSumSq } },
          { binding: 10, resource: { buffer: buffers.phase } },
          { binding: 11, resource: { buffer: buffers.variance } },
        ],
      }),
      objSum: buffers.paramsChunks.map((params, index) => {
        const entries: GPUBindGroupEntry[] = [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: buffers.gqkChunks[chunkBufferIndex(index)] } },
          { binding: 2, resource: { buffer: buffers.stage } },
          { binding: 3, resource: { buffer: buffers.bfGeom } },
          { binding: 4, resource: { buffer: buffers.bfTrig } },
          { binding: 5, resource: { buffer: buffers.qx } },
          { binding: 6, resource: { buffer: buffers.qy } },
          { binding: 12, resource: { buffer: buffers.aberrations } },
        ];
        if (buffers.gqkMode === "herm16" && buffers.gqkScales) {
          entries.push({ binding: 13, resource: { buffer: buffers.gqkScales } });
        }
        return device.createBindGroup({ layout: pipelines.objSum.getBindGroupLayout(0), entries });
      }),
      objFftRows: device.createBindGroup({
        layout: pipelines.objFftRows.getBindGroupLayout(0),
        // ssbObjFftRows uses only the stage plane (sizes are compile-time
        // constants), so the auto layout exposes just binding 2.
        entries: [
          { binding: 2, resource: { buffer: buffers.stage } },
        ],
      }),
      objFftCols: device.createBindGroup({
        layout: pipelines.objFftCols.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: buffers.params } },
          { binding: 2, resource: { buffer: buffers.stage } },
          { binding: 10, resource: { buffer: buffers.phase } },
        ],
      }),
    };
    const bindError = await device.popErrorScope();
    if (bindError) {
      throw new Error(`WebGPU SSB bind group validation failed: ${bindError.message}`);
    }
    this.buffers = buffers;
    this.pipelines = pipelines;
    this.bindGroups = bindGroups;
    this.initialized = true;
    this.loadedBfCount = nbf;
    this.geometryRotationDeg = rotationDeg == null || !Number.isFinite(rotationDeg) ? baseRotationDeg(this.cal) : Number(rotationDeg);
    console.log(
      `[showptycho] WebGPU SSB setup ready ${nbf}/${this.cal.num_bf} BF `
      + `(${nonzeroBfCount} active) in ${(performance.now() - setupT0).toFixed(1)} ms`,
    );
    this.emitProgress({
      stage: "ready",
      message: "WebGPU ptychography reducer ready",
      detail: `${nbf}/${this.cal.num_bf} BF pixels prepared in ${(performance.now() - setupT0).toFixed(0)} ms`,
      current: nbf,
      total: this.cal.num_bf,
      percent: 100,
      activeBf: activeBfCount,
      elapsedMs: performance.now() - setupT0,
    });
  }

  private clampBfCount(requested: number | undefined, preview = false): number {
    if (requested == null || !Number.isFinite(requested)) {
      return preview ? this.previewBfCount : this.cal.num_bf;
    }
    return Math.max(1, Math.min(this.cal.num_bf, Math.round(requested)));
  }

  async prepareBfCount(count: number): Promise<number> {
    const requested = this.clampBfCount(count, false);
    await this.runExclusive(async () => {
      await this.setup(requested);
    });
    return this.loadedBfCount;
  }

  private updateGeometryRotation(rotationDeg?: number): void {
    const device = this.device;
    const buffers = this.buffers;
    if (!device || !buffers) return;
    const requested = rotationDeg == null || !Number.isFinite(rotationDeg) ? baseRotationDeg(this.cal) : Number(rotationDeg);
    if (this.geometryRotationDeg != null && Math.abs(requested - this.geometryRotationDeg) < 1e-9) return;
    const { geom, trig } = packGeometry(this.cal, buffers.activeSourceIndices, requested);
    device.queue.writeBuffer(buffers.bfGeom, 0, geom as unknown as BufferSource);
    device.queue.writeBuffer(buffers.bfTrig, 0, trig as unknown as BufferSource);
    this.geometryRotationDeg = requested;
  }

  async reconstruct(
    c10: number,
    c12: number,
    phi12Deg: number,
    options: {
      preview?: boolean;
      bfCount?: number;
      computeLoss?: boolean;
      rotationDeg?: number;
      higherOrder?: Record<string, number>;
    } = {},
  ): Promise<WebGPUSSBResult> {
    const bfCount = this.clampBfCount(options.bfCount, options.preview);
    const computeLoss = options.computeLoss ?? bfCount === this.cal.num_bf;
    const rotationDeg = options.rotationDeg == null || !Number.isFinite(options.rotationDeg)
      ? baseRotationDeg(this.cal)
      : Number(options.rotationDeg);
    return this.runExclusive(async () => {
      await this.setup(bfCount, rotationDeg);
      const device = this.device;
      const buffers = this.buffers;
      const pipelines = this.pipelines;
      const bindGroups = this.bindGroups;
      if (!device || !buffers || !pipelines || !bindGroups) {
        throw new Error("WebGPU SSB buffers are not ready after setup");
      }
      const aberrations = packAberrations(c10, c12, phi12Deg, options.higherOrder);
      device.queue.writeBuffer(buffers.aberrations, 0, aberrations.data as unknown as BufferSource);
      const t0 = performance.now();
      const enc = device.createCommandEncoder();
      if (buffers.fullStack) {
        // Current canUseFullStack() returns false; the chunked path below is the
        // maintained path for multi-GB browser folders.
      }
      // Object-redraw fast path (ported from the CUDA SSB engine): sum
      // corrected G over BF pixels in Fourier space, one iFFT, angle once -
      // skips two FFT passes + atan2 per BF pixel. Displays angle(mean(obj)),
      // the same estimator as the Python SSB.result() reference (verified
      // corr 0.997 vs backend, 0.9965 vs the exact browser path at identical
      // state). Loss commits always use the exact per-BF path below.
      // Opt out with globalThis.__QUANTEM_SSB_OBJ_FAST__ = false.
      const objFastEnabled = (globalThis as { __QUANTEM_SSB_OBJ_FAST__?: boolean }).__QUANTEM_SSB_OBJ_FAST__ !== false;
      if (!computeLoss && objFastEnabled) {
        const onlyChunk = (globalThis as { __QUANTEM_SSB_OBJ_ONLY_CHUNK__?: number }).__QUANTEM_SSB_OBJ_ONLY_CHUNK__;
        enc.clearBuffer(buffers.stage, 0, this.plane * 8);
        for (let bfOffset = 0; bfOffset < buffers.activeBfCount; bfOffset += buffers.dispatchChunkCapacity) {
          const chunkBf = Math.min(buffers.dispatchChunkCapacity, buffers.activeBfCount - bfOffset);
          const chunkIndex = Math.floor(bfOffset / buffers.dispatchChunkCapacity);
          if (Number.isFinite(onlyChunk) && chunkIndex !== Number(onlyChunk)) continue;
          device.queue.writeBuffer(
            buffers.paramsChunks[chunkIndex],
            0,
            makeParams(this.cal, this.n, c10, c12, phi12Deg, bfCount, bfOffset, chunkBf, buffers.fullStack, false, buffers.activeBfCount, aberrations.active),
          );
          const pass = enc.beginComputePass({ label: "showptycho ssb obj sum" });
          pass.setPipeline(pipelines.objSum);
          pass.setBindGroup(0, bindGroups.objSum[chunkIndex]);
          pass.dispatchWorkgroups(Math.ceil(this.plane / 256));
          pass.end();
        }
        device.queue.writeBuffer(
          buffers.params,
          0,
          makeParams(this.cal, this.n, c10, c12, phi12Deg, bfCount, 0, 1, buffers.fullStack, false, buffers.activeBfCount, aberrations.active),
        );
        let pass = enc.beginComputePass({ label: "showptycho ssb obj fft rows" });
        pass.setPipeline(pipelines.objFftRows);
        pass.setBindGroup(0, bindGroups.objFftRows);
        pass.dispatchWorkgroups(1, this.n);
        pass.end();
        pass = enc.beginComputePass({ label: "showptycho ssb obj fft cols" });
        pass.setPipeline(pipelines.objFftCols);
        pass.setBindGroup(0, bindGroups.objFftCols);
        pass.dispatchWorkgroups(1, this.n);
        pass.end();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        const gpuMsFast = performance.now() - t0;
        const phaseFast = await readF32(device, buffers.phase, this.plane);
        const resultFast = {
          phase: phaseFast,
          width: this.n,
          height: this.n,
          gpuMs: gpuMsFast,
          bfCount,
          rotationDeg,
          loss: null,
          adapterInfo: getGPUInfo(),
          softwareAdapter: isSoftwareGPUAdapter(),
        };
        (globalThis as unknown as { __quantemSsbLast?: unknown }).__quantemSsbLast = {
          gqkMode: buffers.gqkMode,
          residentGqkBytes: buffers.gqkResidentBytes,
          gpuMs: gpuMsFast,
          bfCount,
          activeBf: buffers.activeBfCount,
          c10, c12, phi12Deg,
          rotationDeg,
          loss: null,
          objPath: true,
          phase: phaseFast,
        };
        return resultFast;
      }
      for (let bfOffset = 0; bfOffset < buffers.activeBfCount; bfOffset += buffers.dispatchChunkCapacity) {
        const chunkBf = Math.min(buffers.dispatchChunkCapacity, buffers.activeBfCount - bfOffset);
        const chunkIndex = Math.floor(bfOffset / buffers.dispatchChunkCapacity);
        device.queue.writeBuffer(
          buffers.paramsChunks[chunkIndex],
          0,
          makeParams(
            this.cal,
            this.n,
            c10,
            c12,
            phi12Deg,
            bfCount,
            bfOffset,
            chunkBf,
            buffers.fullStack,
            computeLoss,
            buffers.activeBfCount,
            aberrations.active,
          ),
        );
        let pass = enc.beginComputePass({ label: "showptycho ssb rows" });
        pass.setPipeline(pipelines.rows);
        pass.setBindGroup(0, bindGroups.rows[chunkIndex]);
        pass.dispatchWorkgroups(1, this.n, chunkBf);
        pass.end();
        pass = enc.beginComputePass({ label: "showptycho ssb cols" });
        pass.setPipeline(pipelines.cols);
        pass.setBindGroup(0, bindGroups.cols[chunkIndex]);
        pass.dispatchWorkgroups(1, this.n, chunkBf);
        pass.end();
        pass = enc.beginComputePass({ label: "showptycho ssb reduce chunk" });
        pass.setPipeline(pipelines.reducePartial);
        pass.setBindGroup(0, bindGroups.reducePartial[chunkIndex]);
        pass.dispatchWorkgroups(Math.ceil(Math.ceil(chunkBf / REDUCE_BF_GROUP) * this.plane / 256));
        pass.end();
      }
      device.queue.writeBuffer(
        buffers.params,
        0,
        makeParams(
          this.cal,
          this.n,
          c10,
          c12,
          phi12Deg,
          bfCount,
          0,
          1,
          buffers.fullStack,
          computeLoss,
          buffers.activeBfCount,
          aberrations.active,
        ),
      );
      const pass = enc.beginComputePass({ label: "showptycho ssb reduce" });
      pass.setPipeline(pipelines.finalizeGroups);
      pass.setBindGroup(0, bindGroups.finalizeGroups);
      pass.dispatchWorkgroups(Math.ceil(this.plane / 256));
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      const gpuMs = performance.now() - t0;
      const phase = await readF32(device, buffers.phase, this.plane);
      let loss: number | null = null;
      if (computeLoss && bfCount === this.cal.num_bf) {
        const variance = await readF32(device, buffers.variance, this.plane);
        let sum = 0;
        for (let i = 0; i < variance.length; i++) sum += variance[i];
        loss = sum / variance.length;
      }
      const result = {
        phase,
        width: this.n,
        height: this.n,
        gpuMs,
        bfCount,
        rotationDeg,
        loss,
        adapterInfo: getGPUInfo(),
        softwareAdapter: isSoftwareGPUAdapter(),
      };
      (globalThis as unknown as { __quantemSsbLast?: unknown }).__quantemSsbLast = {
        gqkMode: buffers.gqkMode,
        residentGqkBytes: buffers.gqkResidentBytes,
        gpuMs,
        bfCount,
        activeBf: buffers.activeBfCount,
        c10, c12, phi12Deg,
        rotationDeg,
        loss,
        phase,
      };
      return result;
    });
  }
}
