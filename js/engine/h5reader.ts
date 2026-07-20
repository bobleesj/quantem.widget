// In-browser native Arina/HDF5 reader. Pulls the raw bitshuffle+LZ4 (bslz4) chunks
// straight out of an .h5 file with jsfive (pure-JS HDF5, no wasm) and packs them into
// the Bslz4Spec the WGSL engine decodes. This is the ONLY path that turns a user's
// .h5 file into GPU-decodable bytes with no Python and no server - the colleague opens
// the HTML, points at a folder, and every dataset decodes on the GPU.
//
// The raw chunk jsfive hands back is byte-identical to h5py read_direct_chunk (verified
// 2026-06-03 on full gold 512 Arina, frame0 = 12586 bytes, bit-exact), so the bytes the
// engine eats here are the same bytes CUDA eats. Parity is by composition: jsfive chunk
// == h5py chunk, and engine decode == CUDA decode.

import * as jsfive from "jsfive";
import type { Bslz4MaskedSumSpec, Bslz4Spec } from "./bslz4";

export interface H5Volume {
  name: string;
  detRows: number;
  detCols: number;
  detSize: number;
  blockElems: number;          // elements per bitshuffle block (read from the chunk header)
  nBlocksPerFrame: number;
  srcDtype: "uint8" | "uint16" | "uint32" | "float32";
  nFrames: number;             // frames in THIS file (one Arina data file is a scan slab)
  chunks: Bslz4Spec[];         // scan-frame chunked so each decoded buffer <= the GPU cap
  chunkScanCounts: number[];   // frame count per chunk (== spec.nFrames)
}

export interface H5MasterInfo {
  badPixels: number[];
  detectorShape?: [number, number];
  totalFrames?: number;
}

export interface H5VolumeFrameIndex {
  detRows: number;
  detCols: number;
  nFrames: number;
  srcDtype: "uint8" | "uint16" | "uint32" | "float32";
  frameOffsets: number[];
}

export interface H5BlockIndexChunk {
  startFrame: number;
  nFrames: number;
  rangeStart: number;
  rangeEnd: number;
  metaOffsetWords: number;
  metaWords: number;
}

export interface H5BlockIndexMetadata {
  detRows: number;
  detCols: number;
  nFrames: number;
  srcDtype: "uint8" | "uint16" | "uint32" | "float32";
  blockElems: number;
  nBlocksPerFrame: number;
  chunks: H5BlockIndexChunk[];
}

export interface Bslz4SelectedBlockVolume {
  name: string;
  detRows: number;
  detCols: number;
  detSize: number;
  blockElems: number;
  nFrames: number;
  srcDtype: "uint8" | "uint16" | "uint32" | "float32";
  selectedBlockIds: number[];
  chunk: Bslz4MaskedSumSpec;
}

export interface Bslz4SelectedBlockMetadata {
  detRows: number;
  detCols: number;
  nFrames: number;
  srcDtype: "uint8" | "uint16" | "uint32" | "float32";
  blockElems: number;
  selectedBlockIds: number[];
  startScan?: number;
}

// jsfive dataset path candidates, in priority order. Arina data files use entry/data/data;
// fall back to a search for the first 3D unsigned-int dataset for other layouts.
const PATH_CANDIDATES = ["entry/data/data", "entry/data", "data"];
const PIXEL_MASK_CANDIDATES = [
  "entry/instrument/detector/detectorSpecific/pixel_mask",
  "entry/instrument/detector/pixel_mask",
  "entry/instrument/detector/detectorSpecific/pixel_mask_applied",
];
const H5_BLOCK_INDEX_MAGIC = "QH5IDX01";

function readBE32(b: Uint8Array, off: number): number {
  return ((b[off] << 24) | (b[off + 1] << 16) | (b[off + 2] << 8) | b[off + 3]) >>> 0;
}

function align4(n: number): number {
  return Math.ceil(n / 4) * 4;
}

// Walk a chunked dataset's B-tree and return per-frame raw bslz4 chunk bytes, indexed by
// the scan-frame index (chunk_offset[0]). Returns each frame's ABSOLUTE byte offset in the
// file (NOT a copy) - jsfive reads the B-tree but does NOT apply the bitshuffle filter
// (filter 32008 is unknown to it), so the bytes at that offset are the native codec stream
// the WGSL decoder wants. Keeping offsets (not slicing) lets the decoder read straight out
// of the one uploaded file buffer - no 7.5 GB of chunk copies per dataset.
function frameOffsets(ds: any): number[] {
  const dobj = ds._dataobjects;
  dobj._get_chunk_params();   // cheap: caches _chunk_address / _chunk_dims, sets fh (the file buffer)
  return fastFrameOffsets(dobj.fh, dobj._chunk_address, dobj._chunk_dims, ds.shape[0]);
}

// Fast HDF5 v1 B-tree raw-data-chunk index walk. Replaces jsfive's BTreeV1RawDataChunks
// (a Map/object per node + per key, ~2.8 s across 28 Arina files) with a single recursive
// DataView descent that allocates nothing per node - the dominant load cost. Returns, per
// frame, the ABSOLUTE byte offset of its compressed chunk. Byte-identical to the jsfive walk
// (verified on real gold Arina). Format (jsfive esm/btree.js, 8-byte offsets):
//   node header = "TREE"(4) type(1) level(1) entries_used(2) lsib(8) rsib(8) = 24 B
//   per entry   = chunk_size(u32) filter_mask(u32) coord[dims]*u64 child_addr(u64)
//   dims = _chunk_dims (ndims+1 = 4 for a 3D stack); frame index = coord[0];
//   level>0 -> child_addr is a child NODE (recurse); level==0 -> raw chunk offset (emit).
function fastFrameOffsets(buffer: ArrayBuffer, chunkAddress: number, chunkDims: number, nFrames: number): number[] {
  const dv = new DataView(buffer);
  const out: number[] = new Array(nFrames);
  const coordBytes = chunkDims * 8;
  const entryStride = 8 + coordBytes + 8;   // chunk_size+filter_mask, coords, child addr
  const addrInEntry = 8 + coordBytes;
  const readU64 = (off: number) => dv.getUint32(off, true) + dv.getUint32(off + 4, true) * 4294967296;
  const walk = (nodeOffset: number): void => {
    const level = dv.getUint8(nodeOffset + 5);
    const entries = dv.getUint16(nodeOffset + 6, true);
    let p = nodeOffset + 24;   // first entry, after the 24-byte node header
    if (level === 0) {
      for (let i = 0; i < entries; i++) { out[readU64(p + 8)] = readU64(p + addrInEntry); p += entryStride; }
    } else {
      for (let i = 0; i < entries; i++) { walk(readU64(p + addrInEntry)); p += entryStride; }
    }
  };
  walk(chunkAddress);
  return out;
}

// Find the 3D detector-stack dataset inside the file (scan frames x detRows x detCols).
function findStack(file: any): any {
  for (const path of PATH_CANDIDATES) {
    try { const d = file.get(path); if (d && d.shape && d.shape.length === 3) return d; } catch { /* not present */ }
  }
  // Fallback: first 3D dataset anywhere in the tree.
  const walk = (grp: any): any => {
    for (const key of grp.keys) {
      const child = grp.get(key);
      if (child?.shape?.length === 3) return child;
      if (child?.keys) { const found = walk(child); if (found) return found; }
    }
    return null;
  };
  const found = walk(file);
  if (!found) throw new Error("no 3D detector-stack dataset found in HDF5 file");
  return found;
}

function readPixelMask(file: any): { badPixels: number[]; detectorShape?: [number, number] } {
  for (const path of PIXEL_MASK_CANDIDATES) {
    try {
      const ds = file.get(path);
      const shape = Array.isArray(ds?.shape) && ds.shape.length === 2
        ? [Number(ds.shape[0]), Number(ds.shape[1])] as [number, number]
        : undefined;
      const values = ds?.value as ArrayLike<number> | undefined;
      if (!values || !Number.isFinite(values.length)) continue;
      const badPixels: number[] = [];
      for (let i = 0; i < values.length; i++) {
        if (Number(values[i]) !== 0) badPixels.push(i);
      }
      return { badPixels, detectorShape: shape };
    } catch {
      // Try the next common Arina/Dectris metadata path.
    }
  }
  return { badPixels: [] };
}

function readScalarNumber(file: any, path: string): number | undefined {
  try {
    const value = file.get(path)?.value as ArrayLike<number> | number | undefined;
    const raw = typeof value === "number" ? value : value && value.length ? Number(value[0]) : undefined;
    return Number.isFinite(raw) && raw !== undefined ? Number(raw) : undefined;
  } catch {
    return undefined;
  }
}

export function readH5MasterInfo(buffer: ArrayBuffer, name = "master"): H5MasterInfo {
  const file = new jsfive.File(buffer, name);
  const mask = readPixelMask(file);
  const ntrigger = readScalarNumber(file, "entry/instrument/detector/detectorSpecific/ntrigger");
  const nimages = readScalarNumber(file, "entry/instrument/detector/detectorSpecific/nimages");
  const totalFrames = ntrigger ?? (nimages !== undefined && nimages > 1 ? nimages : undefined);
  return { ...mask, totalFrames };
}

// Parse one Arina/HDF5 file's raw chunks into chunked Bslz4Spec(s). framesPerChunk bounds
// the decoded GPU buffer (uint8 stack <= ~0.95 GB): detSize bytes/frame, so the default
// keeps each chunk under the 1 GB per-buffer cap.
export function readH5Volume(buffer: ArrayBuffer, name: string, framesPerChunk?: number): H5Volume {
  const file = new jsfive.File(buffer, name);
  const ds = findStack(file);
  const [nFrames, detRows, detCols] = ds.shape;
  // Arina writes uint8/uint16/uint32 detector data depending on bit-depth; jsfive reports
  // "|u1"/"<u2"/"<u4". A MAPED-merged stack is float32 ("<f4"). The element byte width drives
  // the bitshuffle plane count (8/16/32) - float32 is 4-byte, so it de-bitshuffles exactly
  // like uint32 (32 planes); only the display reinterprets the decoded 4 bytes as f32.
  const dt = String(ds.dtype);
  const srcDtype: "uint8" | "uint16" | "uint32" | "float32" =
    /f4|float32/.test(dt) ? "float32" : /u1|int8/.test(dt) ? "uint8" : /u4|int32/.test(dt) ? "uint32" : "uint16";
  const offsets = frameOffsets(ds);
  return readH5VolumeFromFrameIndex(buffer, name, { detRows, detCols, nFrames, srcDtype, frameOffsets: offsets }, framesPerChunk);
}

export function readH5VolumeFromFrameIndex(
  buffer: ArrayBuffer,
  name: string,
  index: H5VolumeFrameIndex,
  framesPerChunk?: number,
): H5Volume {
  const { detRows, detCols, nFrames, srcDtype } = index;
  const detSize = detRows * detCols;
  const offsets = index.frameOffsets;
  if (offsets.length < nFrames) {
    throw new Error(`HDF5 frame index for ${name} has ${offsets.length} offsets, expected ${nFrames}.`);
  }
  const srcBytes = srcDtype === "uint8" ? 1 : (srcDtype === "uint32" || srcDtype === "float32") ? 4 : 2;
  // Zero-copy: one GPU buffer = the WHOLE file (a view, no copy), one spec for the dataset.
  // blockMeta holds ABSOLUTE byte offsets into the file, so the decoder reads each block's
  // LZ4 stream straight from the uploaded file - no per-chunk slice, no concatenation blob.
  const fileBytes = new Uint8Array(buffer);
  // Block geometry from the first chunk's 12-byte header: bytes 8-11 (BE) are the per-block
  // uncompressed byte count; blockElems = that / element bytes.
  const blockBytes = readBE32(fileBytes, offsets[0] + 8);
  const blockElems = blockBytes / srcBytes;
  const nBlocksPerFrame = Math.ceil(detSize / blockElems);
  const defaultFramesPerChunk = Math.max(1, Math.floor((1024 * 1024 * 1024) / detSize));
  const frameStep = Math.max(1, Math.floor(framesPerChunk || defaultFramesPerChunk));
  const chunks: Bslz4Spec[] = [];
  const chunkScanCounts: number[] = [];
  for (let start = 0; start < nFrames; start += frameStep) {
    const stop = Math.min(nFrames, start + frameStep);
    const framesThisChunk = stop - start;
    let rangeStart = Number.POSITIVE_INFINITY;
    let rangeEnd = 0;
    const meta = new Uint32Array(framesThisChunk * nBlocksPerFrame * 2);
    let m = 0;
    for (let f = start; f < stop; f++) {
      const addr = offsets[f];
      rangeStart = Math.min(rangeStart, addr);
      let pos = 12;
      for (let b = 0; b < nBlocksPerFrame; b++) {
        const clen = readBE32(fileBytes, addr + pos);
        meta[m++] = addr + pos + 4;
        meta[m++] = clen;
        pos += 4 + clen;
      }
      rangeEnd = Math.max(rangeEnd, addr + pos);
    }
    for (let i = 0; i < meta.length; i += 2) {
      meta[i] -= rangeStart;
    }
    chunks.push({
      compressed: fileBytes.subarray(rangeStart, rangeEnd),
      blockMeta: meta,
      nFrames: stop - start,
      nBlocksPerFrame,
      blockElems,
      detSize,
    });
    chunkScanCounts.push(stop - start);
  }
  return { name, detRows, detCols, detSize, blockElems, nBlocksPerFrame, srcDtype, nFrames, chunks, chunkScanCounts };
}

function parseH5BlockIndex(buffer: ArrayBuffer, name: string): { meta: H5BlockIndexMetadata; blockMeta: Uint32Array } {
  const bytes = new Uint8Array(buffer);
  const magic = new TextDecoder().decode(bytes.subarray(0, 8));
  if (magic !== H5_BLOCK_INDEX_MAGIC) {
    throw new Error(`HDF5 block-index sidecar ${name} is not a ${H5_BLOCK_INDEX_MAGIC} file.`);
  }
  const dv = new DataView(buffer);
  const jsonLen = dv.getUint32(8, true);
  const blockMetaWords = dv.getUint32(12, true);
  const jsonStart = 16;
  const jsonStop = jsonStart + jsonLen;
  if (jsonStop > buffer.byteLength) {
    throw new Error(`HDF5 block-index sidecar ${name} metadata header is truncated.`);
  }
  const metaStart = align4(jsonStop);
  if (metaStart + blockMetaWords * 4 > buffer.byteLength) {
    throw new Error(`HDF5 block-index sidecar ${name} block metadata is truncated.`);
  }
  const raw = JSON.parse(new TextDecoder().decode(bytes.subarray(jsonStart, jsonStop))) as H5BlockIndexMetadata;
  const chunks = Array.from(raw.chunks || []).map((chunk) => ({
    startFrame: Math.max(0, Math.round(Number(chunk.startFrame || 0))),
    nFrames: Math.max(0, Math.round(Number(chunk.nFrames || 0))),
    rangeStart: Math.max(0, Math.round(Number(chunk.rangeStart || 0))),
    rangeEnd: Math.max(0, Math.round(Number(chunk.rangeEnd || 0))),
    metaOffsetWords: Math.max(0, Math.round(Number(chunk.metaOffsetWords || 0))),
    metaWords: Math.max(0, Math.round(Number(chunk.metaWords || 0))),
  }));
  const meta: H5BlockIndexMetadata = {
    detRows: Math.round(Number(raw.detRows)),
    detCols: Math.round(Number(raw.detCols)),
    nFrames: Math.round(Number(raw.nFrames)),
    srcDtype: raw.srcDtype,
    blockElems: Math.round(Number(raw.blockElems)),
    nBlocksPerFrame: Math.round(Number(raw.nBlocksPerFrame)),
    chunks,
  };
  const blockMeta = new Uint32Array(buffer, metaStart, blockMetaWords);
  return { meta, blockMeta };
}

export function readH5BlockIndexMetadata(buffer: ArrayBuffer, name: string): H5BlockIndexMetadata {
  return parseH5BlockIndex(buffer, name).meta;
}

export function readH5VolumeFromBlockIndex(h5Buffer: ArrayBuffer, indexBuffer: ArrayBuffer, name: string): H5Volume {
  const { meta, blockMeta } = parseH5BlockIndex(indexBuffer, name);
  const fileBytes = new Uint8Array(h5Buffer);
  const detSize = meta.detRows * meta.detCols;
  const chunks = meta.chunks.map((chunk) => ({
    compressed: fileBytes.subarray(chunk.rangeStart, chunk.rangeEnd),
    blockMeta: blockMeta.subarray(chunk.metaOffsetWords, chunk.metaOffsetWords + chunk.metaWords),
    nFrames: chunk.nFrames,
    nBlocksPerFrame: meta.nBlocksPerFrame,
    blockElems: meta.blockElems,
    detSize,
  }));
  const chunkScanCounts = meta.chunks.map((chunk) => chunk.nFrames);
  return {
    name,
    detRows: meta.detRows,
    detCols: meta.detCols,
    detSize,
    blockElems: meta.blockElems,
    nBlocksPerFrame: meta.nBlocksPerFrame,
    srcDtype: meta.srcDtype,
    nFrames: meta.nFrames,
    chunks,
    chunkScanCounts,
  };
}

export function readBslz4SelectedBlockVolume(buffer: ArrayBuffer, name: string): Bslz4SelectedBlockVolume {
  const bytes = new Uint8Array(buffer);
  const magic = new TextDecoder().decode(bytes.subarray(0, 8));
  if (magic !== "QBSLZ4S1") {
    throw new Error(`Selected-block file ${name} is not a QBSLZ4S1 file.`);
  }
  const dv = new DataView(buffer);
  const jsonLen = dv.getUint32(8, true);
  const blockMetaWords = dv.getUint32(12, true);
  const jsonStart = 16;
  const jsonStop = jsonStart + jsonLen;
  const metaStart = align4(jsonStop);
  const compressedStart = metaStart + blockMetaWords * 4;
  const meta = JSON.parse(new TextDecoder().decode(bytes.subarray(jsonStart, jsonStop))) as Bslz4SelectedBlockMetadata;
  const selectedBlockIds = Array.from(meta.selectedBlockIds || []).map((v) => Math.round(Number(v)));
  const blockMeta = new Uint32Array(buffer, metaStart, blockMetaWords);
  const compressed = bytes.subarray(compressedStart);
  const detRows = Math.round(Number(meta.detRows));
  const detCols = Math.round(Number(meta.detCols));
  const detSize = detRows * detCols;
  const nFrames = Math.round(Number(meta.nFrames));
  const blockElems = Math.round(Number(meta.blockElems));
  return {
    name,
    detRows,
    detCols,
    detSize,
    blockElems,
    nFrames,
    srcDtype: meta.srcDtype,
    selectedBlockIds,
    chunk: {
      compressed,
      blockMeta,
      nFrames,
      nBlocksPerFrame: selectedBlockIds.length,
      blockElems,
      detSize,
      startScan: Math.round(Number(meta.startScan || 0)),
      sourceStartScan: Math.round(Number(meta.startScan || 0)),
      selectedBlockIds,
    },
  };
}

export function readBslz4SelectedBlockMetadata(buffer: ArrayBuffer, name: string): Bslz4SelectedBlockMetadata {
  const bytes = new Uint8Array(buffer);
  const magic = new TextDecoder().decode(bytes.subarray(0, 8));
  if (magic !== "QBSLZ4S1") {
    throw new Error(`Selected-block file ${name} is not a QBSLZ4S1 file.`);
  }
  const jsonLen = new DataView(buffer).getUint32(8, true);
  const jsonStart = 16;
  const jsonStop = jsonStart + jsonLen;
  if (jsonStop > buffer.byteLength) {
    throw new Error(`Selected-block file ${name} metadata header is truncated.`);
  }
  const meta = JSON.parse(new TextDecoder().decode(bytes.subarray(jsonStart, jsonStop))) as Bslz4SelectedBlockMetadata;
  return {
    ...meta,
    detRows: Math.round(Number(meta.detRows)),
    detCols: Math.round(Number(meta.detCols)),
    nFrames: Math.round(Number(meta.nFrames)),
    blockElems: Math.round(Number(meta.blockElems)),
    selectedBlockIds: Array.from(meta.selectedBlockIds || []).map((v) => Math.round(Number(v))),
  };
}
