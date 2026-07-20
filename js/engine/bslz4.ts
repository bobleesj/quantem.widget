/// <reference types="@webgpu/types" />
// Offline GPU decompression of HDF5 bitshuffle+LZ4 (bslz4) 4D-STEM data.
//
// Ships the NATIVE detector codec (Dectris/Arina write bslz4) so the browser
// downloads ~6x less than raw uint16 and decompresses on WebGPU - no Python, no
// server. Two passes, both verified bit-exact vs h5py on real gold (192x192
// uint16): Pass1 LZ4-decodes each independent block, Pass2 inverts the bit
// transpose. Output is uint16-packed in [scanPos][detPixel] order - the exact
// layout Show4DSTEMCompute reads in uint16 mode, so it feeds masked_sum /
// reduce_frames with no copy.
//
// Per-frame chunk layout (one HDF5 chunk = one diffraction pattern):
//   blockMeta gives, per (frame,block), the absolute byte offset + compressed
//   length of that block's LZ4 stream within the concatenated `compressed`. Each
//   block decompresses to blockElemBytes = blockElems * elemBytes, holds
//   nbits = elemBytes*8 bit-planes of planeBytes = blockElems/8 each.

import { getGPUDevice, onGPULost } from "./device";

type Bslz4Globals = {
  __BSLZ4_PARALLEL?: boolean;
  __BSLZ4_LOW8_ONLY?: boolean;
  __BSLZ4_COOP_LOW8?: boolean;
  __BSLZ4_FRAME_LOW8?: boolean;
  __BSLZ4_LOW8_U32_SHARED?: boolean;
  __BSLZ4_WORD_LOW8?: boolean;
  __BSLZ4_SINGLE_PARSE_LOW8?: boolean;
  __BSLZ4_FRAME_SERIAL_LOW8?: boolean;
  __BSLZ4_FRAMES_PER_WG?: number;
  __BSLZ4_FRAME_WG?: number;
  __BSLZ4_PIPELINE_STAGING?: boolean;
  __BSLZ4_UPLOAD_WRITEBUFFER?: boolean;
  __BSLZ4_UPLOAD_MAPPED?: boolean;
  __BSLZ4_UPLOAD_COMBINED?: boolean;
  __BSLZ4_PROFILE_SPLIT?: boolean;
};

function bslz4Global(): Bslz4Globals {
  return typeof globalThis === "undefined" ? {} : globalThis as Bslz4Globals;
}

// A/B switch for the parallel (Strategy-D, round-based) LZ4 fused kernel vs the serial-thread-0
// fused kernel. Set globalThis.__BSLZ4_PARALLEL=true in the console to measure on real hardware
// without a rebuild. Default false until the parallel kernel is parity-verified + benchmarked.
function bslz4Parallel(): boolean {
  return bslz4Global().__BSLZ4_PARALLEL === true;
}

// Experimental V-E/V-F fast path for lossless-valid uint8 browse data. This is NOT the
// general uint16->uint8 clip path: it is only scientifically valid when an upstream
// count-range audit proves all unmasked detector counts are <=255. In that case the
// high bit-planes can be ignored, and the LZ4 stream only needs to be decoded until
// the low 8 bit-planes are materialized.
function bslz4Low8Only(): boolean {
  return bslz4Global().__BSLZ4_LOW8_ONLY === true;
}

function bslz4CoopLow8(): boolean {
  return bslz4Low8Only() && bslz4Global().__BSLZ4_COOP_LOW8 !== false;
}

function bslz4FrameLow8(): boolean {
  return bslz4CoopLow8() && bslz4Global().__BSLZ4_FRAME_LOW8 !== false;
}

function bslz4Low8U32Shared(): boolean {
  return bslz4FrameLow8() && bslz4Global().__BSLZ4_LOW8_U32_SHARED === true;
}

function bslz4WordLow8(): boolean {
  return bslz4FrameLow8() && bslz4Global().__BSLZ4_WORD_LOW8 === true;
}

function bslz4SingleParseLow8(): boolean {
  return bslz4FrameLow8() && bslz4Global().__BSLZ4_SINGLE_PARSE_LOW8 === true;
}

function bslz4FrameSerialLow8(): boolean {
  return bslz4FrameLow8() && bslz4Global().__BSLZ4_FRAME_SERIAL_LOW8 === true;
}

function bslz4FramesPerWorkgroup(): number {
  const raw = Number(bslz4Global().__BSLZ4_FRAMES_PER_WG);
  if (!Number.isFinite(raw)) return 1;
  return Math.max(1, Math.min(8, Math.round(raw)));
}

function bslz4FrameWorkgroupSize(): 8 | 16 | 32 | 64 | 128 {
  const raw = Number(bslz4Global().__BSLZ4_FRAME_WG);
  if (raw === 8 || raw === 16 || raw === 32 || raw === 64 || raw === 128) return raw;
  return 32;
}

function bslz4PipelineStaging(): boolean {
  return bslz4Global().__BSLZ4_PIPELINE_STAGING !== false;
}

function bslz4UploadWriteBuffer(): boolean {
  return bslz4Global().__BSLZ4_UPLOAD_WRITEBUFFER === true;
}

function bslz4UploadMapped(): boolean {
  return bslz4Global().__BSLZ4_UPLOAD_MAPPED === true;
}

function bslz4UploadCombined(): boolean {
  return bslz4Global().__BSLZ4_UPLOAD_COMBINED === true;
}

function bslz4ProfileSplit(): boolean {
  return bslz4Global().__BSLZ4_PROFILE_SPLIT === true;
}

// Pass1: one thread per block, LZ4-decode into the `inter` (bitshuffled) buffer.
// Blocks are independent -> embarrassingly parallel. Byte-addressed RMW because
// WGSL storage is u32-only.
const PASS1_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read_write> inter: array<u32>;
@group(0) @binding(2) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;             // totalBlocks, blockBytes
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rout(i:u32)->u32{return (inter[i>>2u]>>((i&3u)*8u))&0xffu;}
fn wout(i:u32,v:u32){let w=i>>2u;let s=(i&3u)*8u;inter[w]=(inter[w]&(~(0xffu<<s)))|((v&0xffu)<<s);}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let g=gid.x; if(g>=cfg.x){return;}
  let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u]; let base=g*cfg.y;
  var ci=coff; var di=0u;
  loop{ if(ci>=cend){break;}
    let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
    if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
    var k=0u; loop{if(k>=nlit){break;} wout(base+di+k,rraw(ci+k)); k=k+1u;} ci=ci+nlit; di=di+nlit;
    if(ci>=cend){break;}
    let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
    if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
    var j=0u; loop{if(j>=ml){break;} wout(base+di+j,rout(base+di+j-off)); j=j+1u;} di=di+ml;
  }
}`;

// Pass2: inverse bitshuffle. One thread per GROUP of 8 consecutive pixels - they
// share the same 16 plane-bytes (one byte per bit-plane), so 16 global reads feed
// 8 outputs (8x fewer reads than per-pixel -> ~4x faster, measured). Writes 4
// uint16-packed u32. Plane-major, LSB-first: pixel (group*8 + i) bit b = bit i of
// plane-byte (b*planeBytes + group_byte). cfg = nGroups, strideX, blockElems, planeBytes.
const PASS2_WGSL = `
@group(0) @binding(0) var<storage,read> inter: array<u32>;
@group(0) @binding(1) var<storage,read_write> stack: array<u32>;  // uint16-packed
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;  // nGroups, strideX, blockElems, planeBytes
fn byteAt(o:u32)->u32{return (inter[o>>2u]>>((o&3u)*8u))&0xffu;}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let grp=gid.y*cfg.y + gid.x; if(grp>=cfg.x){return;}
  let blockElems=cfg.z; let planeBytes=cfg.w; let blockBytes=blockElems*2u;
  let nBlk=__NBLK__; let framePix=__FRAMEPIX__;
  let e0=grp*8u; let frm=e0/framePix; let inFrame=e0%framePix;
  let blk=inFrame/blockElems; let groupByte=(inFrame%blockElems)>>3u;
  let pb=(frm*nBlk+blk)*blockBytes + groupByte;
  var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
  for(var b:u32=0u;b<16u;b=b+1u){
    let byte=byteAt(pb+b*planeBytes); let bit=1u<<b;
    if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
    if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
    if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
    if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
  }
  let o=grp*4u;
  stack[o]=v0|(v1<<16u); stack[o+1u]=v2|(v3<<16u); stack[o+2u]=v4|(v5<<16u); stack[o+3u]=v6|(v7<<16u);
}`;

// Pass2 (uint8): same 8-pixel-group inverse bitshuffle, but clip(0,255) and pack 4
// pixels per u32 - halves resident memory (9.6 GB vs 19 GB for full 512x512). Offline
// default: real detector counts are 0-~50, so clip is near-lossless for the signal.
const PASS2_U8_WGSL = `
@group(0) @binding(0) var<storage,read> inter: array<u32>;
@group(0) @binding(1) var<storage,read_write> stack: array<u32>;  // uint8-packed (4/u32)
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;  // nGroups, strideX, blockElems, planeBytes
fn byteAt(o:u32)->u32{return (inter[o>>2u]>>((o&3u)*8u))&0xffu;}
fn clip8(v:u32)->u32{return select(v,255u,v>255u);}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let grp=gid.y*cfg.y + gid.x; if(grp>=cfg.x){return;}
  let blockElems=cfg.z; let planeBytes=cfg.w; let blockBytes=blockElems*2u;
  let nBlk=__NBLK__; let framePix=__FRAMEPIX__;
  let e0=grp*8u; let frm=e0/framePix; let inFrame=e0%framePix;
  let blk=inFrame/blockElems; let groupByte=(inFrame%blockElems)>>3u;
  let pb=(frm*nBlk+blk)*blockBytes + groupByte;
  var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
  for(var b:u32=0u;b<16u;b=b+1u){
    let byte=byteAt(pb+b*planeBytes); let bit=1u<<b;
    if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
    if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
    if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
    if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
  }
  let o=grp*2u;
  stack[o]=clip8(v0)|(clip8(v1)<<8u)|(clip8(v2)<<16u)|(clip8(v3)<<24u);
  stack[o+1u]=clip8(v4)|(clip8(v5)<<8u)|(clip8(v6)<<16u)|(clip8(v7)<<24u);
}`;

// Pass2 (uint8 SOURCE): companion encoded from uint8 (typesize 1) -> only 8 bit
// planes, and the LZ4 block is half the bytes -> ~2x faster than the uint16-source
// path, BOTH passes. Output is uint8-packed directly (values already <= 255).
const PASS2_U8SRC_WGSL = `
@group(0) @binding(0) var<storage,read> inter: array<u32>;
@group(0) @binding(1) var<storage,read_write> stack: array<u32>;  // uint8-packed (4/u32)
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;  // nGroups, strideX, blockElems, planeBytes
fn byteAt(o:u32)->u32{return (inter[o>>2u]>>((o&3u)*8u))&0xffu;}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let grp=gid.y*cfg.y + gid.x; if(grp>=cfg.x){return;}
  let blockElems=cfg.z; let planeBytes=cfg.w; let blockBytes=blockElems;   // uint8: 1 byte/elem
  let nBlk=__NBLK__; let framePix=__FRAMEPIX__;
  let e0=grp*8u; let frm=e0/framePix; let inFrame=e0%framePix;
  let blk=inFrame/blockElems; let groupByte=(inFrame%blockElems)>>3u;
  let pb=(frm*nBlk+blk)*blockBytes + groupByte;
  var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
  for(var b:u32=0u;b<8u;b=b+1u){
    let byte=byteAt(pb+b*planeBytes); let bit=1u<<b;
    if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
    if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
    if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
    if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
  }
  let o=grp*2u;
  stack[o]=v0|(v1<<8u)|(v2<<16u)|(v3<<24u); stack[o+1u]=v4|(v5<<8u)|(v6<<16u)|(v7<<24u);
}`;

// Pass2 (float32 SOURCE): 4-byte elements -> 32 bit-planes. Same 8-pixel-group inverse
// bitshuffle as PASS2_WGSL but loops all 32 planes and reconstructs the FULL 32-bit value
// per pixel, stored as one u32/pixel. The decoded 4 bytes ARE the IEEE-754 float32 bit
// pattern; the reduce/frame shaders bitcast<f32> them. No clip, no pack - full precision.
// cfg = nGroups, strideX, blockElems, planeBytes. blockBytes = blockElems*4 (4-byte elem).
const PASS2_F32_WGSL = `
@group(0) @binding(0) var<storage,read> inter: array<u32>;
@group(0) @binding(1) var<storage,read_write> stack: array<u32>;  // float32 bit-pattern, 1/pixel
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;  // nGroups, strideX, blockElems, planeBytes
fn byteAt(o:u32)->u32{return (inter[o>>2u]>>((o&3u)*8u))&0xffu;}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let grp=gid.y*cfg.y + gid.x; if(grp>=cfg.x){return;}
  let blockElems=cfg.z; let planeBytes=cfg.w; let blockBytes=blockElems*4u;
  let nBlk=__NBLK__; let framePix=__FRAMEPIX__;
  let e0=grp*8u; let frm=e0/framePix; let inFrame=e0%framePix;
  let blk=inFrame/blockElems; let groupByte=(inFrame%blockElems)>>3u;
  let pb=(frm*nBlk+blk)*blockBytes + groupByte;
  var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
  for(var b:u32=0u;b<32u;b=b+1u){
    let byte=byteAt(pb+b*planeBytes); let bit=1u<<b;
    if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
    if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
    if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
    if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
  }
  let o=grp*8u;
  stack[o]=v0; stack[o+1u]=v1; stack[o+2u]=v2; stack[o+3u]=v3;
  stack[o+4u]=v4; stack[o+5u]=v5; stack[o+6u]=v6; stack[o+7u]=v7;
}`;

const MAX_WG = 65535;

// Copy bytes into a mapped GPU range using the widest aligned element view. V8's
// Uint8Array.set is byte-granular (~3.75 GB/s); a Float64Array view stores 8 bytes/op and is
// ~2-3x faster for multi-GB uploads. `src` is a Uint8Array over a whole file ArrayBuffer at
// byteOffset 0 and the mapped range is a fresh 4-aligned ArrayBuffer, so copy the bulk as
// f64 and the <8 trailing bytes as u8.
function copyWide(dst: ArrayBuffer, src: Uint8Array, dstOffset = 0): void {
  const len = src.byteLength, off = src.byteOffset;
  if (((off | dstOffset) & 7) === 0) {
    const n8 = len >>> 3;
    if (n8 > 0) new Float64Array(dst, dstOffset, n8).set(new Float64Array(src.buffer, off, n8));
    const tail = n8 << 3;
    if (tail < len) new Uint8Array(dst, dstOffset + tail, len - tail).set(src.subarray(tail));
    return;
  }
  new Uint8Array(dst, dstOffset, len).set(src);
}

// FUSED decode (integer source -> uint8 output, the offline default): ONE WORKGROUP per
// block. Thread 0 LZ4-decodes the block's bytes into workgroup-SHARED `sh` (byte RMW on
// shared is ~100x the old global RMW, and there is no interBuf round-trip), barrier, then
// all 64 threads inverse-bitshuffle the blockElems/8 eight-pixel groups straight from `sh`
// and write uint8-packed output coalesced to `stack`. Bit-exact with PASS1+PASS2_U8 and
// PASS1+PASS2_U8SRC: same LZ4 token loop, same plane addressing (sh[lg + b*planeBytes]),
// same pack/clip. __NPB__ = blockElems/8 = planeBytes = groups per block; __BE__ =
// blockElems; __NBITS__ = 8/16/32 source bit planes.
const FUSED_U16U8_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(2) var<storage,read_write> stack: array<u32>;  // uint8-packed (4/u32)
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;   // totalBlocks, gridX, nBlk, framePix
var<workgroup> sh: array<u32, 2048>;   // up to 8192 decoded (bitshuffled) bytes / block
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (sh[i>>2u]>>((i&3u)*8u))&0xffu;}
fn wsh(i:u32,v:u32){let w=i>>2u;let s=(i&3u)*8u;sh[w]=(sh[w]&(~(0xffu<<s)))|((v&0xffu)<<s);}
fn clip8(v:u32)->u32{return select(v,255u,v>255u);}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let g = wid.y*cfg.y + wid.x; let lx = lid.x;
  if(lx==0u && g<cfg.x){
    let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
    var ci=coff; var di=0u;
    loop{ if(ci>=cend){break;}
      let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
      if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
      var k=0u; loop{if(k>=nlit){break;} wsh(di+k,rraw(ci+k)); k=k+1u;} ci=ci+nlit; di=di+nlit;
      if(ci>=cend){break;}
      let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
      if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
      var j=0u; loop{if(j>=ml){break;} wsh(di+j,rsh(di+j-off)); j=j+1u;} di=di+ml;
    }
  }
  workgroupBarrier();
  if(g>=cfg.x){return;}
  let frm = g/cfg.z; let blk = g%cfg.z;
  let pixBase = frm*cfg.w + blk*__BE__;   // first detector pixel of this block
  let oBase = (pixBase>>3u)*2u;            // 2 u32 per 8-pixel group
  // uint8 output == clip8(value): 255 iff ANY bit >=8 is set, else the low byte. So only the
  // 8 LOW planes need a bit-transpose; planes 8..nbits-1 collapse to one OR ("any high bit set"
  // per pixel). Bit-exact for any input, and 2x (uint16) / 4x (uint32) less bitshuffle work.
  for(var lg=lx; lg<__NPB__; lg=lg+64u){
    var hi:u32=0u;
    for(var b:u32=8u;b<__NBITS__;b=b+1u){ hi = hi | rsh(lg + b*__NPB__); }
    var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
    for(var b:u32=0u;b<8u;b=b+1u){
      let byte=rsh(lg + b*__NPB__); let bit=1u<<b;
      if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
      if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
      if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
      if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
    }
    let o=oBase + lg*2u;
    stack[o]=select(v0,255u,(hi&1u)!=0u)|(select(v1,255u,(hi&2u)!=0u)<<8u)|(select(v2,255u,(hi&4u)!=0u)<<16u)|(select(v3,255u,(hi&8u)!=0u)<<24u);
    stack[o+1u]=select(v4,255u,(hi&16u)!=0u)|(select(v5,255u,(hi&32u)!=0u)<<8u)|(select(v6,255u,(hi&64u)!=0u)<<16u)|(select(v7,255u,(hi&128u)!=0u)<<24u);
  }
}`;

// FUSED low-8 decode (integer source -> uint8 output, EXPERIMENTAL): one workgroup per
// block, but thread 0 only LZ4-decodes the first blockElems bytes of the bitshuffled
// output. Those bytes are exactly the low 8 bit-planes. This is bit-exact to the
// corrected CUDA uint8 output only when a count-range audit proves every unmasked
// source value is <=255; otherwise it would return the low byte instead of clip(255).
const FUSED_LOW8_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(2) var<storage,read_write> stack: array<u32>;  // uint8-packed (4/u32)
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;   // totalBlocks, gridX, nBlk, framePix
var<workgroup> sh: array<u32, 2048>;   // low 8 bit-planes: up to 8192 bytes / block
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (sh[i>>2u]>>((i&3u)*8u))&0xffu;}
fn wsh(i:u32,v:u32){let w=i>>2u;let s=(i&3u)*8u;sh[w]=(sh[w]&(~(0xffu<<s)))|((v&0xffu)<<s);}
@compute @workgroup_size(__WG__)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let g = wid.y*cfg.y + wid.x; let lx = lid.x;
  if(lx==0u && g<cfg.x){
    let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
    var ci=coff; var di=0u;
    loop{ if(ci>=cend || di>=__BE__){break;}
      let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
      if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
      let litCopy=min(nlit,__BE__-di);
      var k=0u; loop{if(k>=litCopy){break;} wsh(di+k,rraw(ci+k)); k=k+1u;}
      ci=ci+nlit; di=di+nlit;
      if(ci>=cend || di>=__BE__){break;}
      let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
      if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
      let matchCopy=min(ml,__BE__-di);
      var j=0u; loop{if(j>=matchCopy){break;} wsh(di+j,rsh(di+j-off)); j=j+1u;}
      di=di+ml;
    }
  }
  workgroupBarrier();
  if(g>=cfg.x){return;}
  let frm = g/cfg.z; let blk = g%cfg.z;
  let pixBase = frm*cfg.w + blk*__BE__;
  let oBase = (pixBase>>3u)*2u;
  for(var lg=lx; lg<__NPB__; lg=lg+64u){
    var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
    for(var b:u32=0u;b<8u;b=b+1u){
      let byte=rsh(lg + b*__NPB__); let bit=1u<<b;
      if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
      if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
      if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
      if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
    }
    let o=oBase + lg*2u;
    stack[o]=v0|(v1<<8u)|(v2<<16u)|(v3<<24u);
    stack[o+1u]=v4|(v5<<8u)|(v6<<16u)|(v7<<24u);
  }
}`;

// FUSED cooperative low-8 decode (EXPERIMENTAL V-F): all lanes redundantly parse
// the LZ4 token stream but cooperatively copy literal and match bytes into shared
// memory. Overlapping LZ4 matches are handled with the period identity
// value[j] = output[matchStart - offset + (j % offset)], so there is no iterative
// dependency chain and no round-based rescan. The barrier is inside the token loop,
// so browser validation is part of the experiment gate.
const FUSED_COOP_LOW8_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(2) var<storage,read_write> stack: array<u32>;  // uint8-packed (4/u32)
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;   // totalBlocks, gridX, nBlk, framePix
  var<workgroup> sh: array<atomic<u32>, __SH_WORDS__>;
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (atomicLoad(&sh[i>>2u])>>((i&3u)*8u))&0xffu;}
fn wsh(i:u32,v:u32){atomicOr(&sh[i>>2u], (v&0xffu)<<((i&3u)*8u));}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let g = wid.y*cfg.y + wid.x; let lx = lid.x;
  for(var w=lx; w<__SH_WORDS__; w=w+64u){ atomicStore(&sh[w], 0u); }
  workgroupBarrier();
  if(g<cfg.x){
    let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
    var ci=coff; var di=0u; var working=true;
    loop{
      if(!working){break;}
      let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
      if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
      let litCopy=min(nlit,__BE__-di);
      for(var k=lx; k<litCopy; k=k+64u){ wsh(di+k,rraw(ci+k)); }
      ci=ci+nlit; di=di+nlit;
      workgroupBarrier();
      if(ci>=cend || di>=__BE__){
        working=false;
      } else {
        let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
        if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
        let matchCopy=min(ml,__BE__-di);
        for(var j=lx; j<matchCopy; j=j+64u){ wsh(di+j,rsh(di-off+(j%off))); }
        di=di+ml;
        workgroupBarrier();
        if(ci>=cend || di>=__BE__){ working=false; }
      }
    }
  }
  workgroupBarrier();
  if(g>=cfg.x){return;}
  let frm = g/cfg.z; let blk = g%cfg.z;
  let pixBase = frm*cfg.w + blk*__BE__;
  let oBase = (pixBase>>3u)*2u;
  for(var lg=lx; lg<__NPB__; lg=lg+64u){
    var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
    for(var b:u32=0u;b<8u;b=b+1u){
      let byte=rsh(lg + b*__NPB__); let bit=1u<<b;
      if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
      if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
      if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
      if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
    }
    let o=oBase + lg*2u;
    stack[o]=v0|(v1<<8u)|(v2<<16u)|(v3<<24u);
    stack[o+1u]=v4|(v5<<8u)|(v6<<16u)|(v7<<24u);
  }
}`;

// FUSED frame-cooperative low-8 decode (EXPERIMENTAL V-G): one workgroup per
// diffraction pattern, looping over the frame's bitshuffle blocks. This preserves
// the V-F low-byte math but cuts workgroup scheduling by nBlocksPerFrame.
const FUSED_FRAME_COOP_LOW8_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(2) var<storage,read_write> stack: array<u32>;  // uint8-packed (4/u32)
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;   // nFrames, gridX, nBlk, framePix
var<workgroup> sh: array<atomic<u32>, __SH_WORDS__>;
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (atomicLoad(&sh[i>>2u])>>((i&3u)*8u))&0xffu;}
fn wsh(i:u32,v:u32){atomicOr(&sh[i>>2u], (v&0xffu)<<((i&3u)*8u));}
@compute @workgroup_size(__WG__)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let baseFrm = (wid.y*cfg.y + wid.x) * __FPW__; let lx = lid.x;
  if(baseFrm>=cfg.x){return;}
  for(var ff=0u; ff<__FPW__; ff=ff+1u){
  let frm = baseFrm + ff;
  if(frm>=cfg.x){break;}
  for(var blk=0u; blk<cfg.z; blk=blk+1u){
    for(var w=lx; w<__SH_WORDS__; w=w+__WG__){ atomicStore(&sh[w], 0u); }
    workgroupBarrier();
    let g = frm*cfg.z + blk;
    let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
    var ci=coff; var di=0u; var working=true;
    loop{
      if(!working){break;}
      let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
      if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
      let litCopy=min(nlit,__BE__-di);
      for(var k=lx; k<litCopy; k=k+__WG__){ wsh(di+k,rraw(ci+k)); }
      ci=ci+nlit; di=di+nlit;
      workgroupBarrier();
      if(ci>=cend || di>=__BE__){
        working=false;
      } else {
        let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
        if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
        let matchCopy=min(ml,__BE__-di);
        for(var j=lx; j<matchCopy; j=j+__WG__){ wsh(di+j,rsh(di-off+(j%off))); }
        di=di+ml;
        workgroupBarrier();
        if(ci>=cend || di>=__BE__){ working=false; }
      }
    }
    workgroupBarrier();
    let pixBase = frm*cfg.w + blk*__BE__;
    let blockPix = min(__BE__, cfg.w - min(cfg.w, blk*__BE__));
    let groupsThis = (blockPix + 7u) >> 3u;
    let oBase = pixBase >> 2u;
    for(var lg=lx; lg<groupsThis; lg=lg+__WG__){
      var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
      for(var b:u32=0u;b<8u;b=b+1u){
        let byte=rsh(lg + b*__NPB__); let bit=1u<<b;
        if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
        if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
        if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
        if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
      }
      let o=oBase + lg*2u;
      stack[o]=v0|(v1<<8u)|(v2<<16u)|(v3<<24u);
      stack[o+1u]=v4|(v5<<8u)|(v6<<16u)|(v7<<24u);
    }
    workgroupBarrier();
  }
  }
}`;

// FUSED frame-cooperative low-8 decode with packed-word shared memory
// (EXPERIMENTAL V-J): each lane owns full u32 words while copying LZ4 literal
// and match bytes into shared memory. This avoids the byte-granular workgroup
// atomics used by FUSED_FRAME_COOP_LOW8_WGSL while keeping the same token-level
// barriers and the same LZ4 period identity. Off by default until measured.
const FUSED_FRAME_WORD_LOW8_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(2) var<storage,read_write> stack: array<u32>;  // uint8-packed (4/u32)
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;   // nFrames, gridX, nBlk, framePix
var<workgroup> sh: array<u32, __SH_WORDS__>;
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (sh[i>>2u]>>((i&3u)*8u))&0xffu;}
fn put_byte(word:u32, byteOffset:u32, v:u32)->u32{
  let shift=(byteOffset&3u)*8u;
  return (word&(~(0xffu<<shift)))|((v&0xffu)<<shift);
}
fn copy_literal_words(dst:u32, src:u32, n:u32, lx:u32){
  if(n==0u){return;}
  let first=dst>>2u;
  let stop=(dst+n+3u)>>2u;
  for(var w=first+lx; w<stop; w=w+__WG__){
    var word=sh[w];
    let base=w<<2u;
    for(var q=0u; q<4u; q=q+1u){
      let o=base+q;
      if(o>=dst && o<dst+n){ word=put_byte(word,q,rraw(src+o-dst)); }
    }
    sh[w]=word;
  }
}
fn copy_match_words(dst:u32, baseSrc:u32, off:u32, n:u32, lx:u32){
  if(n==0u){return;}
  let first=dst>>2u;
  let stop=(dst+n+3u)>>2u;
  for(var w=first+lx; w<stop; w=w+__WG__){
    var word=sh[w];
    let base=w<<2u;
    for(var q=0u; q<4u; q=q+1u){
      let o=base+q;
      if(o>=dst && o<dst+n){ word=put_byte(word,q,rsh(baseSrc+((o-dst)%off))); }
    }
    sh[w]=word;
  }
}
@compute @workgroup_size(__WG__)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let baseFrm = (wid.y*cfg.y + wid.x) * __FPW__; let lx = lid.x;
  if(baseFrm>=cfg.x){return;}
  for(var ff=0u; ff<__FPW__; ff=ff+1u){
  let frm = baseFrm + ff;
  if(frm>=cfg.x){break;}
  for(var blk=0u; blk<cfg.z; blk=blk+1u){
    for(var w=lx; w<__SH_WORDS__; w=w+__WG__){ sh[w]=0u; }
    workgroupBarrier();
    let g = frm*cfg.z + blk;
    let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
    var ci=coff; var di=0u; var working=true;
    loop{
      if(!working){break;}
      let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
      if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
      let litCopy=min(nlit,__BE__-di);
      copy_literal_words(di,ci,litCopy,lx);
      ci=ci+nlit; di=di+nlit;
      workgroupBarrier();
      if(ci>=cend || di>=__BE__){
        working=false;
      } else {
        let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
        if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
        let matchCopy=min(ml,__BE__-di);
        copy_match_words(di,di-off,off,matchCopy,lx);
        di=di+ml;
        workgroupBarrier();
        if(ci>=cend || di>=__BE__){ working=false; }
      }
    }
    workgroupBarrier();
    let pixBase = frm*cfg.w + blk*__BE__;
    let blockPix = min(__BE__, cfg.w - min(cfg.w, blk*__BE__));
    let groupsThis = (blockPix + 7u) >> 3u;
    let oBase = pixBase >> 2u;
    for(var lg=lx; lg<groupsThis; lg=lg+__WG__){
      var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
      for(var b:u32=0u;b<8u;b=b+1u){
        let byte=rsh(lg + b*__NPB__); let bit=1u<<b;
        if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
        if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
        if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
        if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
      }
      let o=oBase + lg*2u;
      stack[o]=v0|(v1<<8u)|(v2<<16u)|(v3<<24u);
      stack[o+1u]=v4|(v5<<8u)|(v6<<16u)|(v7<<24u);
    }
    workgroupBarrier();
  }
  }
}`;

// FUSED frame-serial low8 decode (EXPERIMENTAL V-Q): one workgroup per
// diffraction pattern, but lane 0 decodes each low-byte LZ4 block serially into
// non-atomic shared memory. For 4 KiB low-byte blocks this tests whether removing
// the cooperative decoder's per-token barriers is worth more than parallelizing
// the byte copy.
const FUSED_FRAME_SERIAL_LOW8_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(2) var<storage,read_write> stack: array<u32>;  // uint8-packed (4/u32)
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;   // nFrames, gridX, nBlk, framePix
var<workgroup> sh: array<u32, 2048>;
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (sh[i>>2u]>>((i&3u)*8u))&0xffu;}
fn wsh(i:u32,v:u32){let w=i>>2u;let s=(i&3u)*8u;sh[w]=(sh[w]&(~(0xffu<<s)))|((v&0xffu)<<s);}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let frm = wid.y*cfg.y + wid.x; let lx = lid.x;
  if(frm>=cfg.x){return;}
  for(var blk=0u; blk<cfg.z; blk=blk+1u){
    if(lx==0u){
      let g = frm*cfg.z + blk;
      let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
      var ci=coff; var di=0u;
      loop{
        if(ci>=cend || di>=__BE__){break;}
        let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
        if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
        let litCopy=min(nlit,__BE__-di);
        var k=0u; loop{if(k>=litCopy){break;} wsh(di+k,rraw(ci+k)); k=k+1u;}
        ci=ci+nlit; di=di+nlit;
        if(ci>=cend || di>=__BE__){break;}
        let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
        if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
        let matchCopy=min(ml,__BE__-di);
        var j=0u; loop{if(j>=matchCopy){break;} wsh(di+j,rsh(di+j-off)); j=j+1u;}
        di=di+ml;
      }
    }
    workgroupBarrier();
    let pixBase = frm*cfg.w + blk*__BE__;
    let blockPix = min(__BE__, cfg.w - min(cfg.w, blk*__BE__));
    let groupsThis = (blockPix + 7u) >> 3u;
    let oBase = pixBase >> 2u;
    for(var lg=lx; lg<groupsThis; lg=lg+64u){
      var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
      for(var b:u32=0u;b<8u;b=b+1u){
        let byte=rsh(lg + b*__NPB__); let bit=1u<<b;
        if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
        if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
        if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
        if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
      }
      let o=oBase + lg*2u;
      stack[o]=v0|(v1<<8u)|(v2<<16u)|(v3<<24u);
      stack[o+1u]=v4|(v5<<8u)|(v6<<16u)|(v7<<24u);
    }
    workgroupBarrier();
  }
}`;

// FUSED frame-cooperative low-8 decode with u32-per-byte shared memory
// (EXPERIMENTAL V-H): trades extra workgroup memory for non-atomic shared
// writes during LZ4 copy. This is valid for the low8 path because each decoded
// byte is written exactly once per token after applying the LZ4 period identity.
const FUSED_FRAME_U32_LOW8_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(2) var<storage,read_write> stack: array<u32>;  // uint8-packed (4/u32)
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;   // nFrames, gridX, nBlk, framePix
var<workgroup> sh: array<u32, __BE_ARRAY__>;
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return sh[i]&0xffu;}
fn wsh(i:u32,v:u32){sh[i]=v&0xffu;}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let frm = wid.y*cfg.y + wid.x; let lx = lid.x;
  if(frm>=cfg.x){return;}
  for(var blk=0u; blk<cfg.z; blk=blk+1u){
    for(var w=lx; w<__BE__; w=w+64u){ sh[w]=0u; }
    workgroupBarrier();
    let g = frm*cfg.z + blk;
    let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
    var ci=coff; var di=0u; var working=true;
    loop{
      if(!working){break;}
      let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
      if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
      let litCopy=min(nlit,__BE__-di);
      for(var k=lx; k<litCopy; k=k+64u){ wsh(di+k,rraw(ci+k)); }
      ci=ci+nlit; di=di+nlit;
      workgroupBarrier();
      if(ci>=cend || di>=__BE__){
        working=false;
      } else {
        let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
        if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
        let matchCopy=min(ml,__BE__-di);
        for(var j=lx; j<matchCopy; j=j+64u){ wsh(di+j,rsh(di-off+(j%off))); }
        di=di+ml;
        workgroupBarrier();
        if(ci>=cend || di>=__BE__){ working=false; }
      }
    }
    workgroupBarrier();
    let pixBase = frm*cfg.w + blk*__BE__;
    let blockPix = min(__BE__, cfg.w - min(cfg.w, blk*__BE__));
    let groupsThis = (blockPix + 7u) >> 3u;
    let oBase = pixBase >> 2u;
    for(var lg=lx; lg<groupsThis; lg=lg+64u){
      var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
      for(var b:u32=0u;b<8u;b=b+1u){
        let byte=rsh(lg + b*__NPB__); let bit=1u<<b;
        if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
        if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
        if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
        if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
      }
      let o=oBase + lg*2u;
      stack[o]=v0|(v1<<8u)|(v2<<16u)|(v3<<24u);
      stack[o+1u]=v4|(v5<<8u)|(v6<<16u)|(v7<<24u);
    }
    workgroupBarrier();
  }
}`;

// FUSED frame-cooperative low-8 decode with single-lane token parsing
// (EXPERIMENTAL V-I): lane 0 parses each LZ4 token once, then all lanes copy
// literals/matches from shared token metadata. This removes 64-way redundant
// token parsing at the cost of extra workgroup barriers per token.
const FUSED_FRAME_SINGLEPARSE_LOW8_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(2) var<storage,read_write> stack: array<u32>;  // uint8-packed (4/u32)
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;   // nFrames, gridX, nBlk, framePix
var<workgroup> sh: array<atomic<u32>, 2048>;
var<workgroup> ciState: atomic<u32>;
var<workgroup> diState: atomic<u32>;
var<workgroup> stopAfter: atomic<u32>;
var<workgroup> litSrc: atomic<u32>;
var<workgroup> litDst: atomic<u32>;
var<workgroup> litN: atomic<u32>;
var<workgroup> matchDst: atomic<u32>;
var<workgroup> matchBase: atomic<u32>;
var<workgroup> matchOff: atomic<u32>;
var<workgroup> matchN: atomic<u32>;
var<workgroup> activeState: atomic<u32>;
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (atomicLoad(&sh[i>>2u])>>((i&3u)*8u))&0xffu;}
fn wsh(i:u32,v:u32){atomicOr(&sh[i>>2u], (v&0xffu)<<((i&3u)*8u));}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let frm = wid.y*cfg.y + wid.x; let lx = lid.x;
  if(frm>=cfg.x){return;}
  for(var blk=0u; blk<cfg.z; blk=blk+1u){
    for(var w=lx; w<2048u; w=w+64u){ atomicStore(&sh[w], 0u); }
    if(lx==0u){
      let g = frm*cfg.z + blk;
      atomicStore(&ciState, blkMeta[g*2u]);
      atomicStore(&diState, 0u);
      atomicStore(&activeState, 1u);
    }
    workgroupBarrier();
    let g = frm*cfg.z + blk;
    let cend=blkMeta[g*2u]+blkMeta[g*2u+1u];
    // Fixed trip count keeps every workgroup barrier in uniform control flow.
    // The real full-stack audit for the current 192x192 uint16 HDF5 source maxed
    // at 201 low-byte tokens/block; 256 gives headroom while keeping this variant
    // dataset-scoped and default-off until broader validation.
    for(var tokenRound=0u; tokenRound<256u; tokenRound=tokenRound+1u){
      if(lx==0u){
        atomicStore(&litSrc,0u); atomicStore(&litDst,0u); atomicStore(&litN,0u); atomicStore(&matchDst,0u); atomicStore(&matchBase,0u); atomicStore(&matchOff,1u); atomicStore(&matchN,0u); atomicStore(&stopAfter,0u);
        var ci=atomicLoad(&ciState); var di=atomicLoad(&diState);
        if(atomicLoad(&activeState)==0u || ci>=cend || di>=__BE__){
          atomicStore(&stopAfter,1u);
          atomicStore(&activeState,0u);
        } else {
          let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
          if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
          atomicStore(&litSrc,ci); atomicStore(&litDst,di); atomicStore(&litN,min(nlit,__BE__-di));
          ci=ci+nlit; di=di+nlit;
          if(ci>=cend || di>=__BE__){
            atomicStore(&ciState,ci); atomicStore(&diState,di); atomicStore(&stopAfter,1u);
            atomicStore(&activeState,0u);
          } else {
            let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
            if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
            atomicStore(&matchDst,di); atomicStore(&matchBase,di-off); atomicStore(&matchOff,off); atomicStore(&matchN,min(ml,__BE__-di));
            atomicStore(&ciState,ci); atomicStore(&diState,di+ml);
            if(ci>=cend || di+ml>=__BE__){atomicStore(&stopAfter,1u); atomicStore(&activeState,0u);}
          }
        }
      }
      workgroupBarrier();
      let litCount=atomicLoad(&litN); let litOut=atomicLoad(&litDst); let litIn=atomicLoad(&litSrc);
      for(var k=lx; k<litCount; k=k+64u){ wsh(litOut+k,rraw(litIn+k)); }
      workgroupBarrier();
      let matchCount=atomicLoad(&matchN); let matchOut=atomicLoad(&matchDst); let matchIn=atomicLoad(&matchBase); let off=atomicLoad(&matchOff);
      for(var j=lx; j<matchCount; j=j+64u){ wsh(matchOut+j,rsh(matchIn+(j%off))); }
      workgroupBarrier();
    }
    workgroupBarrier();
    let pixBase = frm*cfg.w + blk*__BE__;
    let blockPix = min(__BE__, cfg.w - min(cfg.w, blk*__BE__));
    let groupsThis = (blockPix + 7u) >> 3u;
    let oBase = pixBase >> 2u;
    for(var lg=lx; lg<groupsThis; lg=lg+64u){
      var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
      for(var b:u32=0u;b<8u;b=b+1u){
        let byte=rsh(lg + b*__NPB__); let bit=1u<<b;
        if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
        if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
        if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
        if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
      }
      let o=oBase + lg*2u;
      stack[o]=v0|(v1<<8u)|(v2<<16u)|(v3<<24u);
      stack[o+1u]=v4|(v5<<8u)|(v6<<16u)|(v7<<24u);
    }
    workgroupBarrier();
  }
}`;

// FUSED decode (float32 source -> float32 output, full precision): ONE WORKGROUP per block.
// Thread 0 LZ4-decodes the block into workgroup-SHARED sh (the fast path, no interBuf global
// round-trip), barrier, then all 64 threads inverse-bitshuffle the blockElems/8 eight-pixel
// groups over ALL 32 planes, reconstruct the full 32-bit value and write it as 1 u32/pixel
// (the IEEE-754 bit pattern - reduce/frame shaders bitcast<f32>). __NPB__ = blockElems/8 =
// planeBytes = groups/block; __BE__ = blockElems.
const FUSED_F32_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(2) var<storage,read_write> stack: array<u32>;  // float32 bit pattern (1/pixel)
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;   // totalBlocks, gridX, nBlk, framePix
var<workgroup> sh: array<u32, 2048>;   // up to 8192 decoded (bitshuffled) bytes / block
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (sh[i>>2u]>>((i&3u)*8u))&0xffu;}
fn wsh(i:u32,v:u32){let w=i>>2u;let s=(i&3u)*8u;sh[w]=(sh[w]&(~(0xffu<<s)))|((v&0xffu)<<s);}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let g = wid.y*cfg.y + wid.x; let lx = lid.x;
  if(lx==0u && g<cfg.x){
    let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
    var ci=coff; var di=0u;
    loop{ if(ci>=cend){break;}
      let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
      if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
      var k=0u; loop{if(k>=nlit){break;} wsh(di+k,rraw(ci+k)); k=k+1u;} ci=ci+nlit; di=di+nlit;
      if(ci>=cend){break;}
      let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
      if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
      var j=0u; loop{if(j>=ml){break;} wsh(di+j,rsh(di+j-off)); j=j+1u;} di=di+ml;
    }
  }
  workgroupBarrier();
  if(g>=cfg.x){return;}
  let frm = g/cfg.z; let blk = g%cfg.z;
  let pixBase = frm*cfg.w + blk*__BE__;
  let oBase = pixBase;   // 1 u32 per pixel
  for(var lg=lx; lg<__NPB__; lg=lg+64u){
    var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
    for(var b:u32=0u;b<32u;b=b+1u){
      let byte=rsh(lg + b*__NPB__); let bit=1u<<b;
      if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
      if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
      if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
      if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
    }
    let o=oBase + lg*8u;
    stack[o]=v0; stack[o+1u]=v1; stack[o+2u]=v2; stack[o+3u]=v3;
    stack[o+4u]=v4; stack[o+5u]=v5; stack[o+6u]=v6; stack[o+7u]=v7;
  }
}`;

const FUSED_F32_PIPE_CACHE = new Map<string, GPUComputePipeline>();
function getFusedF32Pipe(device: GPUDevice, blockElems: number): GPUComputePipeline {
  const code = FUSED_F32_WGSL.replace(/__NPB__/g, `${blockElems / 8}u`).replace(/__BE__/g, `${blockElems}u`);
  let p = FUSED_F32_PIPE_CACHE.get(code);
  if (!p) { p = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } }); FUSED_F32_PIPE_CACHE.set(code, p); }
  return p;
}

// Fused float32 decode job: one workgroup/block, float32 stack out (1 u32/pixel, mode 2).
function buildFusedJobF32(device: GPUDevice, spec: Bslz4Spec, preRaw?: GPUBuffer | RawInput): DecodeJob {
  const { compressed, blockMeta, nFrames, nBlocksPerFrame, blockElems, detSize } = spec;
  const totalBlocks = nFrames * nBlocksPerFrame;
  const stackWords = nFrames * detSize;   // 1 u32/pixel
  let rawBuf: GPUBuffer | RawInput;
  if (preRaw) { rawBuf = preRaw; }
  else {
    const rawSize = Math.ceil(compressed.byteLength / 4) * 4;
    const buffer = device.createBuffer({ size: rawSize, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    copyWide(buffer.getMappedRange(), compressed);
    buffer.unmap();
    rawBuf = rawInput(buffer, rawSize);
  }
  const metaBuf = device.createBuffer({ size: blockMeta.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(metaBuf, 0, blockMeta.buffer as ArrayBuffer, blockMeta.byteOffset, blockMeta.byteLength);
  const stack = device.createBuffer({ size: stackWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const gx = Math.min(totalBlocks, MAX_WG), gy = Math.ceil(totalBlocks / MAX_WG);
  const cfg = uniform(device, [totalBlocks, gx, nBlocksPerFrame, detSize]);
  const pipe = getFusedF32Pipe(device, blockElems);
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: rawBinding(rawBuf) }, { binding: 1, resource: { buffer: metaBuf } },
    { binding: 2, resource: { buffer: stack } }, { binding: 3, resource: { buffer: cfg } } ] });
  return {
    stack, mode: 2,
    record(enc) { const pass = enc.beginComputePass(); pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(gx, gy); pass.end(); },
    releaseTemps() { releaseRaw(rawBuf); metaBuf.destroy(); cfg.destroy(); },
  };
}

// STRATEGY D (parallel LZ4 via round-based dataflow). PARITY-VERIFIED bit-exact vs the serial
// fused kernel (uint16 AND uint32, nDiff=0 on real gold04/gold06 via verifyFusedD) but it is a
// PERF REGRESSION, kept default-OFF (BSLZ4_PARALLEL) as a reference + harness for future work.
// WHY it loses: round-based dataflow does O(maxDepth x blockBytes) work (every round re-scans
// every match byte) while the serial thread-0 decode is O(blockBytes). With maxDepth ~60 that
// is ~60x more total work; spread over 64 lanes it is roughly break-even on raw ops, and the
// per-round atomics + ~60 workgroupBarriers push it 3-5x SLOWER than serial (measured: 348ms/
// 681ms uint16/uint32 per 10k-frame file vs 118ms/138ms serial). The ONLY structurally-faster
// LZ4 is warp-cooperative copy inside a SINGLE forward pass (silx-style: lanes split each match's
// byte copy as the serial cursor advances), which keeps O(blockBytes) total work - a different
// algorithm than this round-based one, not a tuning of it. Round-based cannot beat serial.
//
// All 64 lanes parse the SAME token stream redundantly (parsing is cheap: ~50 tokens/block,
// pointer chasing in registers) and then cooperate on the byte copies. The LZ4 output is a
// dataflow DAG: literal bytes have
// depth 0; a match byte at di+k reads source di-off+(k%off). Because k%off lands in the FIRST
// `off` bytes of the match's source region (NOT at di+k-off), a period-`off` run does NOT form
// a depth-ml chain - it flattens. Measured on real Arina uint16+uint32 (thousands of blocks):
// the longest dependency chain in any 8 KB block is ~60. So repeated rounds, each a full
// redundant copy of every literal+match byte guarded by "only write a byte whose every source
// is already final," reach the fixed point. The loop runs until a round resolves no byte (the
// `progress` atomic is workgroup-uniform after the post-round barrier, so the early break is in
// UNIFORM control flow) - self-terminating at maxDepth+1 for ANY depth. The per-round
// workgroupBarrier stays uniform; the trap that blanked the prior attempt (barrier inside the
// data-dependent token loop) is gone.
//
// FINALITY without a second buffer: a byte is "final" once it equals its dataflow value and
// will never change again. We track that with a per-byte DONE bitmap in shared memory (1 bit
// per output byte, 8192 bits = 256 u32). Round r: every lane walks the token list; for each
// match byte di+k it checks DONE[src]; if set, it writes the byte to sh and marks DONE[di+k].
// Literal bytes are written + marked DONE in round 0 (their source is the compressed input,
// always available). A byte already DONE is skipped. Once a whole round resolves nothing the
// fixed point is reached (max depth ~60). Each output byte is written EXACTLY ONCE into its
// known-zero shared slot via a single atomicOr (OR-with-0 is identity + commutative, so the
// only lock-free byte write in WGSL's u32-only shared memory); DONE + progress are atomics too.
//
// __NPB__ = blockElems/8 = planeBytes = groups/block; __BE__ = blockElems; __MAXROUNDS__ backstop.
const FUSED_D_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;
@group(0) @binding(2) var<storage,read_write> stack: array<u32>;
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;   // totalBlocks, gridX, nBlk, framePix
var<workgroup> sh: array<atomic<u32>, 2048>;   // decoded bytes, atomic (race-free byte writes)
var<workgroup> done: array<atomic<u32>, 256>;  // 1 bit per output byte: is it final?
var<workgroup> progress: atomic<u32>;          // did this round resolve any byte? (convergence)
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (atomicLoad(&sh[i>>2u])>>((i&3u)*8u))&0xffu;}
// each output byte is written EXACTLY once into its KNOWN-ZERO slot via a single atomicOr:
// OR-with-0 is identity + commutative, so concurrent different-byte-same-word writes are
// race-free AND order-independent (the only correct lock-free byte write in WGSL).
fn wsh(i:u32,v:u32){atomicOr(&sh[i>>2u], (v&0xffu)<<((i&3u)*8u));}
fn isDone(i:u32)->bool{return ((atomicLoad(&done[i>>5u])>>(i&31u))&1u)!=0u;}
fn setDone(i:u32){atomicOr(&done[i>>5u], 1u<<(i&31u));}
fn clip8(v:u32)->u32{return select(v,255u,v>255u);}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let g = wid.y*cfg.y + wid.x; let lx = lid.x;
  let live = g < cfg.x;
  let coff = select(0u, blkMeta[g*2u], live);
  let cend = select(0u, coff + blkMeta[g*2u+1u], live);
  // zero sh (so atomicOr-into-zero is exact) and the DONE bitmap. Uniform counted loops.
  for(var w=lx; w<2048u; w=w+64u){ atomicStore(&sh[w], 0u); }
  for(var w=lx; w<256u;  w=w+64u){ atomicStore(&done[w], 0u); }
  workgroupBarrier();
  // ---- ROUND 0: write + mark all LITERAL bytes (source = compressed input, always ready).
  // Owner lane (di+k)&63==lx writes each literal byte exactly once. Match bytes deferred.
  {
    var ci=coff; var di=0u;
    loop{ if(ci>=cend){break;}
      let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
      if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
      var k=0u; loop{ if(k>=nlit){break;}
        let o=di+k; if((o&63u)==lx){ wsh(o, rraw(ci+k)); setDone(o); }
        k=k+1u; }
      ci=ci+nlit; di=di+nlit;
      if(ci>=cend){break;}
      let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
      if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
      di=di+ml;   // skip match output this round
    }
  }
  workgroupBarrier();
  // ---- ROUNDS 1..MAXROUNDS: resolve match bytes whose source is DONE. FIXED trip count so the
  // two per-round workgroupBarriers sit in UNIFORM control flow (WGSL forbids a barrier whose
  // reachability depends on a non-uniform value - an atomic load is always deemed non-uniform,
  // so an early break on progress is illegal). Instead the expensive token walk is gated by
  // 'active' (a non-uniform branch with NO barrier inside, which IS legal): once a whole round
  // resolves nothing, progress stays 0 forever (resolution is monotonic), so every later round
  // is just two cheap barriers. MAXROUNDS bounds the deepest LZ4 chain (~60 on real Arina).
  var working = 1u;
  for(var r=0u; r<__MAXROUNDS__; r=r+1u){
    if(lx==0u){ atomicStore(&progress, 0u); }
    workgroupBarrier();
    if(working != 0u){
      var ci=coff; var di=0u;
      loop{ if(ci>=cend){break;}
        let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
        if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
        ci=ci+nlit; di=di+nlit;   // literals already done in round 0
        if(ci>=cend){break;}
        let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
        if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
        var k=0u; loop{ if(k>=ml){break;}
          let o=di+k;
          if((o&63u)==lx && !isDone(o)){
            let src=di-off+(k%off);
            if(isDone(src)){ wsh(o, rsh(src)); setDone(o); atomicStore(&progress, 1u); }
          }
          k=k+1u; }
        di=di+ml;
      }
    }
    workgroupBarrier();
    working = atomicLoad(&progress);   // 0 => converged; gate (not branch-to-barrier) next round
  }
  if(!live){return;}
  // ---- inverse bitshuffle + uint8 pack, identical to FUSED_U16U8 (OR-fold high planes).
  let frm = g/cfg.z; let blk = g%cfg.z;
  let pixBase = frm*cfg.w + blk*__BE__;
  let oBase = (pixBase>>3u)*2u;
  for(var lg=lx; lg<__NPB__; lg=lg+64u){
    var hi:u32=0u;
    for(var b:u32=8u;b<__NBITS__;b=b+1u){ hi = hi | rsh(lg + b*__NPB__); }
    var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
    for(var b:u32=0u;b<8u;b=b+1u){
      let byte=rsh(lg + b*__NPB__); let bit=1u<<b;
      if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
      if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
      if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
      if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
    }
    let o=oBase + lg*2u;
    stack[o]=select(v0,255u,(hi&1u)!=0u)|(select(v1,255u,(hi&2u)!=0u)<<8u)|(select(v2,255u,(hi&4u)!=0u)<<16u)|(select(v3,255u,(hi&8u)!=0u)<<24u);
    stack[o+1u]=select(v4,255u,(hi&16u)!=0u)|(select(v5,255u,(hi&32u)!=0u)<<8u)|(select(v6,255u,(hi&64u)!=0u)<<16u)|(select(v7,255u,(hi&128u)!=0u)<<24u);
  }
}`;

const FUSED_D_PIPE_CACHE = new Map<string, GPUComputePipeline>();
function getFusedDPipe(device: GPUDevice, blockElems: number, nbits: number): GPUComputePipeline {
  const npb = blockElems / 8;
  const code = FUSED_D_WGSL.replace(/__NPB__/g, `${npb}u`).replace(/__BE__/g, `${blockElems}u`).replace(/__NBITS__/g, `${nbits}u`).replace(/__MAXROUNDS__/g, `256u`);
  let p = FUSED_D_PIPE_CACHE.get(code);
  if (!p) { p = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } }); FUSED_D_PIPE_CACHE.set(code, p); }
  return p;
}

const FUSED_PIPE_CACHE = new Map<string, GPUComputePipeline>();
function getFusedPipe(device: GPUDevice, blockElems: number, nbits: number): GPUComputePipeline {
  const npb = blockElems / 8;
  const code = FUSED_U16U8_WGSL.replace(/__NPB__/g, `${npb}u`).replace(/__BE__/g, `${blockElems}u`).replace(/__NBITS__/g, `${nbits}u`);
  let p = FUSED_PIPE_CACHE.get(code);
  if (!p) { p = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } }); FUSED_PIPE_CACHE.set(code, p); }
  return p;
}

const FUSED_LOW8_PIPE_CACHE = new Map<string, GPUComputePipeline>();
function getFusedLow8Pipe(device: GPUDevice, blockElems: number): GPUComputePipeline {
  const npb = blockElems / 8;
  const template = bslz4CoopLow8() ? FUSED_COOP_LOW8_WGSL : FUSED_LOW8_WGSL;
  const code = template
    .replace(/__NPB__/g, `${npb}u`)
    .replace(/__SH_WORDS__/g, `${Math.ceil(blockElems / 4)}u`)
    .replace(/__BE__/g, `${blockElems}u`);
  let p = FUSED_LOW8_PIPE_CACHE.get(code);
  if (!p) { p = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } }); FUSED_LOW8_PIPE_CACHE.set(code, p); }
  return p;
}

const FUSED_FRAME_LOW8_PIPE_CACHE = new Map<string, GPUComputePipeline>();
function getFusedFrameLow8Pipe(device: GPUDevice, blockElems: number): GPUComputePipeline {
  const npb = blockElems / 8;
  const framesPerWg = bslz4SingleParseLow8() || bslz4Low8U32Shared() || bslz4FrameSerialLow8() || bslz4WordLow8() ? 1 : bslz4FramesPerWorkgroup();
  const wgSize = bslz4FrameWorkgroupSize();
  const template = bslz4SingleParseLow8()
    ? FUSED_FRAME_SINGLEPARSE_LOW8_WGSL
    : bslz4FrameSerialLow8()
      ? FUSED_FRAME_SERIAL_LOW8_WGSL
      : bslz4Low8U32Shared()
        ? FUSED_FRAME_U32_LOW8_WGSL
        : bslz4WordLow8()
          ? FUSED_FRAME_WORD_LOW8_WGSL
          : FUSED_FRAME_COOP_LOW8_WGSL;
  const code = template
    .replace(/__NPB__/g, `${npb}u`)
    .replace(/__SH_WORDS__/g, `${Math.ceil(blockElems / 4)}u`)
    .replace(/__FPW__/g, `${framesPerWg}u`)
    .replace(/__WG__/g, `${wgSize}u`)
    .replace(/__BE_ARRAY__/g, `${blockElems}`)
    .replace(/__BE__/g, `${blockElems}u`);
  let p = FUSED_FRAME_LOW8_PIPE_CACHE.get(code);
  if (!p) { p = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } }); FUSED_FRAME_LOW8_PIPE_CACHE.set(code, p); }
  return p;
}

type IntegralSrcDtype = "uint8" | "uint16" | "uint32";

// One fused decode job: upload the raw bytes + block table, dispatch one workgroup per
// block (2D grid for the >65535 case). No interBuf. Integer source -> uint8 output only.
function buildFusedJob(device: GPUDevice, spec: Bslz4Spec, srcDtype: IntegralSrcDtype, preRaw?: GPUBuffer | RawInput): DecodeJob {
  const { compressed, blockMeta, nFrames, nBlocksPerFrame, blockElems, detSize } = spec;
  const nbits = srcDtype === "uint32" ? 32 : srcDtype === "uint16" ? 16 : 8;
  const totalBlocks = nFrames * nBlocksPerFrame;
  const frameLow8 = bslz4FrameLow8();
  const stackWords = Math.ceil(nFrames * detSize / 4);
  // raw buffer: either pre-uploaded via the staging pool (batch path) or, for the single
  // decode path, mappedAtCreation here.
  let rawBuf: GPUBuffer | RawInput;
  if (preRaw) { rawBuf = preRaw; }
  else {
    const rawSize = Math.ceil(compressed.byteLength / 4) * 4;
    const buffer = device.createBuffer({ size: rawSize, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    copyWide(buffer.getMappedRange(), compressed);
    buffer.unmap();
    rawBuf = rawInput(buffer, rawSize);
  }
  const metaBuf = device.createBuffer({ size: blockMeta.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(metaBuf, 0, blockMeta.buffer as ArrayBuffer, blockMeta.byteOffset, blockMeta.byteLength);
  const stack = device.createBuffer({ size: stackWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const frameGroup = frameLow8 && !bslz4Low8U32Shared() && !bslz4SingleParseLow8() && !bslz4FrameSerialLow8() ? bslz4FramesPerWorkgroup() : 1;
  const dispatchUnits = frameLow8 ? Math.ceil(nFrames / frameGroup) : totalBlocks;
  const gx = Math.min(dispatchUnits, MAX_WG), gy = Math.ceil(dispatchUnits / MAX_WG);
  const cfg = uniform(device, [frameLow8 ? nFrames : dispatchUnits, gx, nBlocksPerFrame, detSize]);
  const pipe = frameLow8 ? getFusedFrameLow8Pipe(device, blockElems) : bslz4Low8Only() ? getFusedLow8Pipe(device, blockElems) : getFusedPipe(device, blockElems, nbits);
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: rawBinding(rawBuf) }, { binding: 1, resource: { buffer: metaBuf } },
    { binding: 2, resource: { buffer: stack } }, { binding: 3, resource: { buffer: cfg } } ] });
  return {
    stack, mode: 1,
    record(enc) { const pass = enc.beginComputePass(); pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(gx, gy); pass.end(); },
    releaseTemps() { releaseRaw(rawBuf); metaBuf.destroy(); cfg.destroy(); },
  };
}

// Strategy-D fused job: identical I/O contract to buildFusedJob (same bindings, same stack
// layout, same dispatch grid) but uses the round-based PARALLEL LZ4 kernel. The kernel runs
// until the per-round `progress` atomic reports no byte resolved (convergence early-exit), so
// it self-terminates at maxDepth+1 rounds for ANY LZ4 dependency depth. __MAXROUNDS__ (4096) is
// only a runaway backstop, far above the ~60 worst case on real Arina uint16+uint32 blocks.
function buildFusedJobD(device: GPUDevice, spec: Bslz4Spec, srcDtype: IntegralSrcDtype, preRaw?: GPUBuffer | RawInput): DecodeJob {
  const { compressed, blockMeta, nFrames, nBlocksPerFrame, blockElems, detSize } = spec;
  const nbits = srcDtype === "uint32" ? 32 : srcDtype === "uint16" ? 16 : 8;
  const totalBlocks = nFrames * nBlocksPerFrame;
  const stackWords = Math.ceil(nFrames * detSize / 4);
  let rawBuf: GPUBuffer | RawInput;
  if (preRaw) { rawBuf = preRaw; }
  else {
    const rawSize = Math.ceil(compressed.byteLength / 4) * 4;
    const buffer = device.createBuffer({ size: rawSize, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    copyWide(buffer.getMappedRange(), compressed);
    buffer.unmap();
    rawBuf = rawInput(buffer, rawSize);
  }
  const metaBuf = device.createBuffer({ size: blockMeta.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(metaBuf, 0, blockMeta.buffer as ArrayBuffer, blockMeta.byteOffset, blockMeta.byteLength);
  const stack = device.createBuffer({ size: stackWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const gx = Math.min(totalBlocks, MAX_WG), gy = Math.ceil(totalBlocks / MAX_WG);
  const cfg = uniform(device, [totalBlocks, gx, nBlocksPerFrame, detSize]);
  const pipe = getFusedDPipe(device, blockElems, nbits);
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: rawBinding(rawBuf) }, { binding: 1, resource: { buffer: metaBuf } },
    { binding: 2, resource: { buffer: stack } }, { binding: 3, resource: { buffer: cfg } } ] });
  return {
    stack, mode: 1,
    record(enc) { const pass = enc.beginComputePass(); pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(gx, gy); pass.end(); },
    releaseTemps() { releaseRaw(rawBuf); metaBuf.destroy(); cfg.destroy(); },
  };
}

// Bit-exact parity + GPU-time check for Strategy D vs the serial Fallback, on ONE spec.
// Runs each kernel in its OWN submit (so the wall time around onSubmittedWorkDone is the
// isolated kernel cost, upload excluded - raw is pre-uploaded here), reads back both packed
// uint8 stacks, and reports the exact byte diff. No tolerance: D must be byte-identical to
// Fallback. Returns null if WebGPU is unavailable. Console-driven verify only.
export async function verifyFusedD(spec: Bslz4Spec, srcDtype: IntegralSrcDtype): Promise<{
  nBytes: number; nDiff: number; maxDiff: number; firstDiffAt: number; meanFrame0: number;
  dErr: string | null; fGpuMs: number; dGpuMs: number;
} | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  const stackWords = Math.ceil(spec.nFrames * spec.detSize / 4);
  const runReadback = async (job: DecodeJob): Promise<{ data: Uint8Array; ms: number }> => {
    const warm = device.createCommandEncoder(); job.record(warm); device.queue.submit([warm.finish()]); await device.queue.onSubmittedWorkDone();
    const enc = device.createCommandEncoder(); job.record(enc);
    const rb = device.createBuffer({ size: stackWords * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(job.stack, 0, rb, 0, stackWords * 4);
    const t0 = performance.now(); device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
    const ms = performance.now() - t0;
    await rb.mapAsync(GPUMapMode.READ);
    const data = new Uint8Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy();
    job.releaseTemps(); job.stack.destroy();
    return { data, ms };
  };
  const fRes = await runReadback(buildFusedJob(device, spec, srcDtype));
  device.pushErrorScope("validation");
  const dJob = buildFusedJobD(device, spec, srcDtype);
  const dRes = await runReadback(dJob);
  const dErr = await device.popErrorScope();
  const fBytes = fRes.data, dBytes = dRes.data;
  const n = Math.min(fBytes.length, dBytes.length);
  let nDiff = 0, maxDiff = 0, firstDiffAt = -1;
  for (let i = 0; i < n; i++) { const d = Math.abs(fBytes[i] - dBytes[i]); if (d) { nDiff++; if (d > maxDiff) maxDiff = d; if (firstDiffAt < 0) firstDiffAt = i; } }
  let sum = 0; const f0 = Math.min(spec.detSize, n); for (let i = 0; i < f0; i++) sum += dBytes[i];
  return { nBytes: n, nDiff, maxDiff, firstDiffAt, meanFrame0: sum / f0, dErr: dErr ? dErr.message : null, fGpuMs: fRes.ms, dGpuMs: dRes.ms };
}

export interface Bslz4Spec {
  compressed: Uint8Array;        // concatenated per-frame bslz4 chunks (frame-padded to 4B)
  blockMeta: Uint32Array;        // [coff,clen] per (frame,block), absolute byte offsets
  nFrames: number;
  nBlocksPerFrame: number;
  blockElems: number;            // elements per bitshuffle block (e.g. 4096 for uint16/8192B)
  detSize: number;               // detector pixels per frame (e.g. 192*192)
}

// Compute pipelines are independent of the data (only of the pass2 template), so compile
// them ONCE and reuse across every chunk + dataset - recompiling per chunk was wasteful.
const PIPE_CACHE = new Map<string, { p1: GPUComputePipeline; p2: GPUComputePipeline }>();
function getPipes(device: GPUDevice, srcDtype: "uint8" | "uint16" | "float32", u8: boolean, nBlocksPerFrame: number, detSize: number) {
  const pass2tpl = srcDtype === "float32" ? PASS2_F32_WGSL : srcDtype === "uint8" ? PASS2_U8SRC_WGSL : (u8 ? PASS2_U8_WGSL : PASS2_WGSL);
  const pass2 = pass2tpl.replace("__NBLK__", `${nBlocksPerFrame}u`).replace("__FRAMEPIX__", `${detSize}u`);
  let pipes = PIPE_CACHE.get(pass2);
  if (!pipes) {
    pipes = {
      p1: device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: PASS1_WGSL }), entryPoint: "main" } }),
      p2: device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: pass2 }), entryPoint: "main" } }),
    };
    PIPE_CACHE.set(pass2, pipes);
  }
  return pipes;
}

interface RawInput {
  buffer: GPUBuffer;
  offset?: number;
  size?: number;
  owned?: boolean;
}

function rawInput(buffer: GPUBuffer, size?: number): RawInput {
  return { buffer, offset: 0, size, owned: true };
}

function isRawInput(raw: GPUBuffer | RawInput): raw is RawInput {
  return typeof (raw as RawInput).buffer !== "undefined";
}

function rawBinding(raw: GPUBuffer | RawInput): GPUBufferBinding {
  if (!isRawInput(raw)) return { buffer: raw };
  const binding: GPUBufferBinding = { buffer: raw.buffer };
  if (raw.offset) binding.offset = raw.offset;
  if (raw.size) binding.size = raw.size;
  return binding;
}

function releaseRaw(raw: GPUBuffer | RawInput): void {
  if (!isRawInput(raw)) {
    raw.destroy();
  } else if (raw.owned !== false) {
    raw.buffer.destroy();
  }
}

interface DecodeJob {
  stack: GPUBuffer;
  mode: number;
  record(enc: GPUCommandEncoder): void;
  releaseTemps(): void;
}
export interface Bslz4BatchProfile {
  variant: string;
  groups: number;
  specs: number;
  compressedMB: number;
  uploadMs: number;
  uploadCopyWaitMs?: number;
  buildMs: number;
  gpuWaitMs: number;
  decodeComputeWaitMs?: number;
  totalMs: number;
  profileSplit?: boolean;
}

export interface Bslz4MaskedSumSpec extends Bslz4Spec {
  // Output scan offset. For a full scan this is usually the same as the source
  // offset; for a scan-region product it is the compact crop output offset.
  startScan: number;
  // Optional source-global offset and contiguous source-frame window. These make
  // a row-major scan crop exact without decoding unrelated scan positions.
  sourceStartScan?: number;
  frameStart?: number;
  frameCount?: number;
  // Optional fast-source contract: compressed/blockMeta already contain exactly
  // these detector bitshuffle blocks, in this order, for every frame. Used by
  // selected-block HDF5/range/sidecar sources to skip the CPU pack pass.
  selectedBlockIds?: number[];
}

export interface Bslz4MaskedSumProfile extends Bslz4BatchProfile {
  packMs: number;
  selectedBlocks: number;
  selectedPixels: number;
  selectedGroups: number;
  selectedBlockIds: number[];
}

const MASKED_SUM_LOW8_PIXEL_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;     // compact selected-block coff,clen
@group(0) @binding(2) var<storage,read> idx: array<u32>;         // block-local detector pixels
@group(0) @binding(3) var<storage,read> blkTable: array<u32>;    // originalBlock, idxOffset, idxCount
@group(0) @binding(4) var<storage,read_write> sums: array<f32>;
@group(0) @binding(5) var<uniform> cfg: vec4<u32>;               // nFrames, gridX, selectedBlocks, startScan
var<workgroup> sh: array<atomic<u32>, 2048>;
var<workgroup> part: array<u32, __WG__>;
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (atomicLoad(&sh[i>>2u])>>((i&3u)*8u))&0xffu;}
fn wsh(i:u32,v:u32){atomicOr(&sh[i>>2u], (v&0xffu)<<((i&3u)*8u));}
@compute @workgroup_size(__WG__)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let frm = wid.y*cfg.y + wid.x; let lx = lid.x;
  if(frm >= cfg.x){return;}
  var frameSum = 0u;
  for(var bi=0u; bi<cfg.z; bi=bi+1u){
    for(var w=lx; w<__SH_WORDS__u; w=w+__WG__){ atomicStore(&sh[w], 0u); }
  workgroupBarrier();
  let g = frm*cfg.z + bi;
  let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
  var ci=coff; var di=0u; var working=true;
  loop{
    if(!working){break;}
    let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
    if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
    let litCopy=min(nlit,__BE__-di);
    for(var k=lx; k<litCopy; k=k+__WG__){ wsh(di+k,rraw(ci+k)); }
    ci=ci+nlit; di=di+nlit;
    workgroupBarrier();
    if(ci>=cend || di>=__BE__){
      working=false;
    } else {
      let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
      if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
      let matchCopy=min(ml,__BE__-di);
      for(var j=lx; j<matchCopy; j=j+__WG__){ wsh(di+j,rsh(di-off+(j%off))); }
      di=di+ml;
      workgroupBarrier();
      if(ci>=cend || di>=__BE__){ working=false; }
    }
  }
  workgroupBarrier();
  let row = bi*3u;
  let idxOff = blkTable[row + 1u];
  let idxN = blkTable[row + 2u];
  var sum = 0u;
  for(var j=lx; j<idxN; j=j+__WG__){
    let p = idx[idxOff + j];
    let groupByte = p >> 3u;
    let pixBit = 1u << (p & 7u);
    var v = 0u;
    for(var b=0u; b<8u; b=b+1u){
      if((rsh(groupByte + b*__NPB__) & pixBit) != 0u){ v = v | (1u << b); }
    }
    sum = sum + v;
  }
  part[lx] = sum;
  workgroupBarrier();
  for(var s=__WG_HALF__; s>0u; s=s>>1u){
    if(lx < s){ part[lx] = part[lx] + part[lx+s]; }
    workgroupBarrier();
  }
  if(lx == 0u){ frameSum = frameSum + part[0]; }
  workgroupBarrier();
  }
  if(lx == 0u){ sums[cfg.w + frm] = f32(frameSum); }
}`;

const MASKED_SUM_LOW8_GROUPMASK_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;     // compact selected-block coff,clen
@group(0) @binding(2) var<storage,read> grpByte: array<u32>;     // block-local pixel byte groups
@group(0) @binding(3) var<storage,read> grpMask: array<u32>;     // selected low bits inside each group
@group(0) @binding(4) var<storage,read> blkTable: array<u32>;    // originalBlock, groupOffset, groupCount
@group(0) @binding(5) var<storage,read_write> sums: array<f32>;
@group(0) @binding(6) var<uniform> cfg: vec4<u32>;               // nFrames, gridX, selectedBlocks, startScan
  var<workgroup> sh: array<atomic<u32>, __SH_WORDS__>;
var<workgroup> part: array<u32, __WG__>;
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (atomicLoad(&sh[i>>2u])>>((i&3u)*8u))&0xffu;}
fn wsh(i:u32,v:u32){atomicOr(&sh[i>>2u], (v&0xffu)<<((i&3u)*8u));}
@compute @workgroup_size(__WG__)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let frm = wid.y*cfg.y + wid.x; let lx = lid.x;
  if(frm >= cfg.x){return;}
  var frameSum = 0u;
  for(var bi=0u; bi<cfg.z; bi=bi+1u){
    for(var w=lx; w<__SH_WORDS__u; w=w+__WG__){ atomicStore(&sh[w], 0u); }
  workgroupBarrier();
  let g = frm*cfg.z + bi;
  let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
  var ci=coff; var di=0u; var working=true;
  loop{
    if(!working){break;}
    let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
    if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
    let litCopy=min(nlit,__BE__-di);
    for(var k=lx; k<litCopy; k=k+__WG__){ wsh(di+k,rraw(ci+k)); }
    ci=ci+nlit; di=di+nlit;
    workgroupBarrier();
    if(ci>=cend || di>=__BE__){
      working=false;
    } else {
      let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
      if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
      let matchCopy=min(ml,__BE__-di);
      for(var j=lx; j<matchCopy; j=j+__WG__){ wsh(di+j,rsh(di-off+(j%off))); }
      di=di+ml;
      workgroupBarrier();
      if(ci>=cend || di>=__BE__){ working=false; }
    }
  }
  workgroupBarrier();
  let row = bi*3u;
  let idxOff = blkTable[row + 1u];
  let idxN = blkTable[row + 2u];
  var sum = 0u;
  for(var j=lx; j<idxN; j=j+__WG__){
    let groupByte = grpByte[idxOff + j];
    let bits = grpMask[idxOff + j];
    for(var b=0u; b<8u; b=b+1u){
      sum = sum + (countOneBits(rsh(groupByte + b*__NPB__) & bits) << b);
    }
  }
  part[lx] = sum;
  workgroupBarrier();
  for(var s=__WG_HALF__; s>0u; s=s>>1u){
    if(lx < s){ part[lx] = part[lx] + part[lx+s]; }
    workgroupBarrier();
  }
  if(lx == 0u){ frameSum = frameSum + part[0]; }
  workgroupBarrier();
  }
	  if(lx == 0u){ sums[cfg.w + frm] = f32(frameSum); }
	}`;

const MASKED_SUM_LOW8_PIPE_CACHE = new Map<string, GPUComputePipeline>();
function useMaskedSumGroupMask(scanCount: number): boolean {
  const forced = (globalThis as { __QT_BSLZ4_MASKED_SUM_GROUPMASK?: unknown }).__QT_BSLZ4_MASKED_SUM_GROUPMASK;
  if (forced !== undefined) return forced !== false;
  return scanCount > 256 * 256;
}

function useMaskedSumCompactShared(scanCount: number): boolean {
  const forced = (globalThis as { __QT_BSLZ4_MASKED_SUM_COMPACT_SHARED?: unknown }).__QT_BSLZ4_MASKED_SUM_COMPACT_SHARED;
  if (forced !== undefined) return forced !== false;
  return scanCount > 512 * 512;
}

function useMaskedSumPipeline(): boolean {
  return (globalThis as { __QT_BSLZ4_MASKED_SUM_PIPELINE?: unknown }).__QT_BSLZ4_MASKED_SUM_PIPELINE !== false;
}

function maskedSumWorkgroupSize(_scanCount: number): 64 | 128 | 256 {
  const forced = Number((globalThis as { __QT_BSLZ4_MASKED_SUM_WG?: unknown }).__QT_BSLZ4_MASKED_SUM_WG);
  if (forced === 64 || forced === 128 || forced === 256) return forced;
  return 64;
}

function getMaskedSumLow8PixelPipe(device: GPUDevice, blockElems: number, wgSize: 64 | 128 | 256): GPUComputePipeline {
  const npb = blockElems / 8;
  const shWords = Math.ceil(blockElems / 2);
  const code = MASKED_SUM_LOW8_PIXEL_WGSL
    .replace(/__NPB__/g, `${npb}u`)
    .replace(/__BE__/g, `${blockElems}u`)
    .replace(/__SH_WORDS__/g, `${shWords}`)
    .replace(/__WG_HALF__/g, `${Math.floor(wgSize / 2)}u`)
    .replace(/__WG__/g, `${wgSize}u`);
  let p = MASKED_SUM_LOW8_PIPE_CACHE.get(code);
  if (!p) { p = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } }); MASKED_SUM_LOW8_PIPE_CACHE.set(code, p); }
  return p;
}

function getMaskedSumLow8GroupMaskPipe(device: GPUDevice, blockElems: number, wgSize: 64 | 128 | 256, compactShared: boolean): GPUComputePipeline {
  const npb = blockElems / 8;
  const shWords = compactShared ? Math.ceil(blockElems / 4) : Math.ceil(blockElems / 2);
  const code = MASKED_SUM_LOW8_GROUPMASK_WGSL
    .replace(/__NPB__/g, `${npb}u`)
    .replace(/__BE__/g, `${blockElems}u`)
    .replace(/__SH_WORDS__/g, `${shWords}`)
    .replace(/__WG_HALF__/g, `${Math.floor(wgSize / 2)}u`)
    .replace(/__WG__/g, `${wgSize}u`);
  let p = MASKED_SUM_LOW8_PIPE_CACHE.get(code);
  if (!p) { p = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } }); MASKED_SUM_LOW8_PIPE_CACHE.set(code, p); }
  return p;
}

function maskedSumBlocks(mask: Uint32Array, badPixels: Uint32Array | undefined, blockElems: number): {
  blockIds: number[];
  blockTable: Uint32Array;
  groupBlockTable: Uint32Array;
  localIdx: Uint32Array;
  groupIdx: Uint32Array;
  groupMask: Uint32Array;
  selectedPixels: number;
  selectedGroups: number;
} {
  const bad = badPixels && badPixels.length ? new Set(badPixels) : null;
  const byBlock = new Map<number, number[]>();
  const groupByBlock = new Map<number, Map<number, number>>();
  for (let k = 0; k < mask.length; k++) {
    if (mask[k] === 0 || bad?.has(k)) continue;
    const block = Math.floor(k / blockElems);
    let local = byBlock.get(block);
    if (!local) {
      local = [];
      byBlock.set(block, local);
    }
    const localPix = k - block * blockElems;
    local.push(localPix);
    let groups = groupByBlock.get(block);
    if (!groups) {
      groups = new Map<number, number>();
      groupByBlock.set(block, groups);
    }
    const groupByte = localPix >> 3;
    const bit = 1 << (localPix & 7);
    groups.set(groupByte, (groups.get(groupByte) || 0) | bit);
  }
  const blockIds = Array.from(byBlock.keys()).sort((a, b) => a - b);
  const table = new Uint32Array(Math.max(1, blockIds.length) * 3);
  const groupTable = new Uint32Array(Math.max(1, blockIds.length) * 3);
  const idxParts: number[] = [];
  const groupParts: number[] = [];
  const groupMaskParts: number[] = [];
  for (let i = 0; i < blockIds.length; i++) {
    const pixels = byBlock.get(blockIds[i]) || [];
    const groups = Array.from((groupByBlock.get(blockIds[i]) || new Map<number, number>()).entries()).sort((a, b) => a[0] - b[0]);
    table[i * 3] = blockIds[i];
    table[i * 3 + 1] = idxParts.length;
    table[i * 3 + 2] = pixels.length;
    groupTable[i * 3] = blockIds[i];
    groupTable[i * 3 + 1] = groupParts.length;
    groupTable[i * 3 + 2] = groups.length;
    idxParts.push(...pixels);
    for (const [groupByte, groupMask] of groups) {
      groupParts.push(groupByte);
      groupMaskParts.push(groupMask);
    }
  }
  return {
    blockIds,
    blockTable: table,
    groupBlockTable: groupTable,
    localIdx: new Uint32Array(idxParts.length ? idxParts : [0]),
    groupIdx: new Uint32Array(groupParts.length ? groupParts : [0]),
    groupMask: new Uint32Array(groupMaskParts.length ? groupMaskParts : [0]),
    selectedPixels: idxParts.length,
    selectedGroups: groupParts.length,
  };
}

export function maskedSumBlockIds(mask: Uint32Array, badPixels: Uint32Array | undefined, blockElems: number): number[] {
  return maskedSumBlocks(mask, badPixels, blockElems).blockIds;
}

export function selectedBlockIdsCover(available: number[] | undefined, requested: number[]): boolean {
  if (!available) return false;
  const have = new Set(available);
  return requested.every((block) => have.has(block));
}

export function sliceMaskedSumSpecsByScanRegion(
  specs: Bslz4MaskedSumSpec[],
  scanRows: number,
  scanCols: number,
  scanRegion?: readonly [number, number, number, number] | null,
): Bslz4MaskedSumSpec[] {
  if (!scanRegion) return specs;
  const rows = Math.max(1, Math.round(scanRows));
  const cols = Math.max(1, Math.round(scanCols));
  const r0 = Math.max(0, Math.min(rows, Math.round(scanRegion[0])));
  const r1 = Math.max(0, Math.min(rows, Math.round(scanRegion[1])));
  const c0 = Math.max(0, Math.min(cols, Math.round(scanRegion[2])));
  const c1 = Math.max(0, Math.min(cols, Math.round(scanRegion[3])));
  if (r1 <= r0 || c1 <= c0) {
    throw new Error(`Invalid scan_region (${scanRegion.join(", ")}); expected (row_start, row_stop, col_start, col_stop) with non-empty bounds.`);
  }
  const cropCols = c1 - c0;
  const out: Bslz4MaskedSumSpec[] = [];
  for (let row = r0; row < r1; row++) {
    const rowSourceStart = row * cols + c0;
    const rowSourceStop = row * cols + c1;
    const rowOutputStart = (row - r0) * cropCols;
    for (const spec of specs) {
      const sourceStart = spec.sourceStartScan ?? spec.startScan;
      const sourceCount = spec.frameCount ?? spec.nFrames;
      const sourceStop = sourceStart + sourceCount;
      const start = Math.max(rowSourceStart, sourceStart);
      const stop = Math.min(rowSourceStop, sourceStop);
      if (stop <= start) continue;
      out.push({
        ...spec,
        startScan: rowOutputStart + (start - rowSourceStart),
        sourceStartScan: start,
        frameStart: (spec.frameStart ?? 0) + (start - sourceStart),
        frameCount: stop - start,
      });
    }
  }
  return out;
}

function maskedFrameWindow(spec: Bslz4MaskedSumSpec): { frameStart: number; frameCount: number } {
  const frameStart = Math.max(0, Math.min(spec.nFrames, Math.round(spec.frameStart ?? 0)));
  const requested = Math.round(spec.frameCount ?? (spec.nFrames - frameStart));
  const frameCount = Math.max(0, Math.min(spec.nFrames - frameStart, requested));
  return { frameStart, frameCount };
}

function sameBlockIds(a: number[] | undefined, b: number[]): boolean {
  if (!a || a.length !== b.length) return false;
  for (let i = 0; i < b.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function sliceCompactedBslz4Frames(spec: Bslz4MaskedSumSpec, frameStart = 0, frameCount = spec.nFrames): Bslz4MaskedSumSpec {
  const selectedBlocks = spec.nBlocksPerFrame;
  const firstFrame = Math.max(0, Math.min(spec.nFrames, Math.round(frameStart)));
  const nFrames = Math.max(0, Math.min(spec.nFrames - firstFrame, Math.round(frameCount)));
  if (firstFrame === 0 && nFrames === spec.nFrames) return spec;
  if (nFrames === 0 || selectedBlocks === 0) {
    return { ...spec, compressed: new Uint8Array(0), blockMeta: new Uint32Array(2), nFrames: 0 };
  }
  const firstBlock = firstFrame * selectedBlocks;
  const lastBlock = firstBlock + nFrames * selectedBlocks - 1;
  const rangeStart = spec.blockMeta[firstBlock * 2];
  const rangeEnd = spec.blockMeta[lastBlock * 2] + spec.blockMeta[lastBlock * 2 + 1];
  const blockMeta = new Uint32Array(nFrames * selectedBlocks * 2);
  for (let i = 0; i < nFrames * selectedBlocks; i++) {
    const src = (firstBlock + i) * 2;
    blockMeta[i * 2] = spec.blockMeta[src] - rangeStart;
    blockMeta[i * 2 + 1] = spec.blockMeta[src + 1];
  }
  return {
    ...spec,
    compressed: spec.compressed.subarray(rangeStart, rangeEnd),
    blockMeta,
    nFrames,
    frameStart: 0,
    frameCount: nFrames,
  };
}

function compactSelectedBslz4Blocks(spec: Bslz4MaskedSumSpec, blockIds: number[], frameStart = 0, frameCount = spec.nFrames): Bslz4MaskedSumSpec {
  if (!spec.selectedBlockIds) {
    return { ...compactBslz4Blocks(spec, blockIds, frameStart, frameCount), startScan: spec.startScan };
  }
  if (sameBlockIds(spec.selectedBlockIds, blockIds)) {
    return sliceCompactedBslz4Frames(spec, frameStart, frameCount);
  }
  const positions = blockIds.map((block) => spec.selectedBlockIds!.indexOf(block));
  const missing = positions.findIndex((pos) => pos < 0);
  if (missing >= 0) {
    throw new Error(`Selected-block sidecar is missing detector block ${blockIds[missing]}; use native HDF5 or a sidecar that covers this mask.`);
  }
  const firstFrame = Math.max(0, Math.min(spec.nFrames, Math.round(frameStart)));
  const nFrames = Math.max(0, Math.min(spec.nFrames - firstFrame, Math.round(frameCount)));
  if (nFrames === 0 || blockIds.length === 0) {
    return { ...spec, compressed: new Uint8Array(0), blockMeta: new Uint32Array(2), nFrames: 0, nBlocksPerFrame: blockIds.length, selectedBlockIds: blockIds };
  }
  const lengths: number[] = [];
  let total = 0;
  for (let f = 0; f < nFrames; f++) {
    const sourceFrame = firstFrame + f;
    for (const pos of positions) {
      const src = (sourceFrame * spec.nBlocksPerFrame + pos) * 2;
      const clen = spec.blockMeta[src + 1];
      lengths.push(clen);
      total += clen;
    }
  }
  const compressed = new Uint8Array(total);
  const blockMeta = new Uint32Array(Math.max(1, nFrames * blockIds.length * 2));
  let dst = 0;
  let m = 0;
  for (let f = 0; f < nFrames; f++) {
    const sourceFrame = firstFrame + f;
    for (let bi = 0; bi < positions.length; bi++) {
      const src = (sourceFrame * spec.nBlocksPerFrame + positions[bi]) * 2;
      const coff = spec.blockMeta[src];
      const clen = lengths[m++];
      blockMeta[(f * blockIds.length + bi) * 2] = dst;
      blockMeta[(f * blockIds.length + bi) * 2 + 1] = clen;
      compressed.set(spec.compressed.subarray(coff, coff + clen), dst);
      dst += clen;
    }
  }
  return {
    ...spec,
    compressed,
    blockMeta,
    nFrames,
    nBlocksPerFrame: blockIds.length,
    selectedBlockIds: blockIds,
    frameStart: 0,
    frameCount: nFrames,
  };
}

function compactBslz4Blocks(spec: Bslz4Spec, blockIds: number[], frameStart = 0, frameCount = spec.nFrames): Bslz4Spec {
  if (blockIds.length === 0) return { ...spec, compressed: new Uint8Array(0), blockMeta: new Uint32Array(2), nBlocksPerFrame: 0 };
  const firstFrame = Math.max(0, Math.min(spec.nFrames, Math.round(frameStart)));
  const nFrames = Math.max(0, Math.min(spec.nFrames - firstFrame, Math.round(frameCount)));
  const lengths: number[] = [];
  let total = 0;
  for (let f = 0; f < nFrames; f++) {
    const sourceFrame = firstFrame + f;
    for (const block of blockIds) {
      const src = (sourceFrame * spec.nBlocksPerFrame + block) * 2;
      const clen = spec.blockMeta[src + 1];
      lengths.push(clen);
      total += clen;
    }
  }
  const compressed = new Uint8Array(total);
  const blockMeta = new Uint32Array(Math.max(1, nFrames * blockIds.length * 2));
  let dst = 0;
  let m = 0;
  for (let f = 0; f < nFrames; f++) {
    const sourceFrame = firstFrame + f;
    for (let bi = 0; bi < blockIds.length; bi++) {
      const block = blockIds[bi];
      const src = (sourceFrame * spec.nBlocksPerFrame + block) * 2;
      const coff = spec.blockMeta[src];
      const clen = lengths[m++];
      blockMeta[(f * blockIds.length + bi) * 2] = dst;
      blockMeta[(f * blockIds.length + bi) * 2 + 1] = clen;
      compressed.set(spec.compressed.subarray(coff, coff + clen), dst);
      dst += clen;
    }
  }
  return { ...spec, compressed, blockMeta, nFrames, nBlocksPerFrame: blockIds.length };
}

interface MaskedSumJob {
  record(enc: GPUCommandEncoder): void;
  releaseTemps(): void;
}

function buildMaskedSumLow8Job(
  device: GPUDevice,
  spec: Bslz4MaskedSumSpec,
  rawBuf: GPUBuffer,
  groupIdxBuf: GPUBuffer,
  groupMaskBuf: GPUBuffer | null,
  blockTableBuf: GPUBuffer,
  sums: GPUBuffer,
  selectedBlocks: number,
  wgSize: 64 | 128 | 256,
  useGroupMask: boolean,
  compactShared: boolean,
): MaskedSumJob {
  const metaBuf = device.createBuffer({ size: Math.max(8, spec.blockMeta.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(metaBuf, 0, spec.blockMeta.buffer as ArrayBuffer, spec.blockMeta.byteOffset, spec.blockMeta.byteLength);
  const dispatchUnits = spec.nFrames;
  const gx = Math.min(dispatchUnits, MAX_WG), gy = Math.ceil(dispatchUnits / MAX_WG);
  const cfg = uniform(device, [spec.nFrames, gx, selectedBlocks, spec.startScan]);
  const pipe = useGroupMask ? getMaskedSumLow8GroupMaskPipe(device, spec.blockElems, wgSize, compactShared) : getMaskedSumLow8PixelPipe(device, spec.blockElems, wgSize);
  const bg = useGroupMask
    ? device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: rawBuf } },
      { binding: 1, resource: { buffer: metaBuf } },
      { binding: 2, resource: { buffer: groupIdxBuf } },
      { binding: 3, resource: { buffer: groupMaskBuf! } },
      { binding: 4, resource: { buffer: blockTableBuf } },
      { binding: 5, resource: { buffer: sums } },
      { binding: 6, resource: { buffer: cfg } },
    ] })
    : device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: rawBuf } },
      { binding: 1, resource: { buffer: metaBuf } },
      { binding: 2, resource: { buffer: groupIdxBuf } },
      { binding: 3, resource: { buffer: blockTableBuf } },
      { binding: 4, resource: { buffer: sums } },
      { binding: 5, resource: { buffer: cfg } },
    ] });
  return {
    record(enc) {
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(gx, gy);
      pass.end();
    },
    releaseTemps() { rawBuf.destroy(); metaBuf.destroy(); cfg.destroy(); },
  };
}

export async function decodeBslz4MaskedSumLow8Batch(
  specs: Bslz4MaskedSumSpec[],
  mask: Uint32Array,
  scanCount: number,
  badPixels?: Uint32Array,
  groupSize = 8,
): Promise<{ device: GPUDevice; buffer: GPUBuffer; profile: Bslz4MaskedSumProfile } | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  const blockElems = specs[0]?.blockElems || 4096;
  const selected = maskedSumBlocks(mask, badPixels, blockElems);
  const groupMask = useMaskedSumGroupMask(scanCount);
  const compactShared = groupMask && useMaskedSumCompactShared(scanCount);
  const wgSize = maskedSumWorkgroupSize(scanCount);
  const profile: Bslz4MaskedSumProfile = {
    variant: specs.some((spec) => Boolean(spec.selectedBlockIds))
      ? `product-first-frame-masked-sum-low8/${groupMask ? `groupmask-wg${wgSize}${compactShared ? "-compactSh" : ""}` : `pixel-wg${wgSize}`}/selected-block-sidecar/direct-f32/${useMaskedSumPipeline() ? "stagingPipeline" : "staging"}`
      : `product-first-frame-masked-sum-low8/${groupMask ? `groupmask-wg${wgSize}${compactShared ? "-compactSh" : ""}` : `pixel-wg${wgSize}`}/direct-f32/${useMaskedSumPipeline() ? "stagingPipeline" : "staging"}`,
    groups: 0,
    specs: specs.length,
    compressedMB: 0,
    uploadMs: 0,
    buildMs: 0,
    gpuWaitMs: 0,
    totalMs: 0,
    packMs: 0,
    selectedBlocks: selected.blockIds.length,
    selectedPixels: selected.selectedPixels,
    selectedGroups: selected.selectedGroups,
    selectedBlockIds: selected.blockIds,
  };
  const outF32 = device.createBuffer({ size: Math.max(4, scanCount * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const tAll = performance.now();
  const clear = device.createCommandEncoder();
  clear.clearBuffer(outF32, 0, Math.max(4, scanCount * 4));
  device.queue.submit([clear.finish()]);
  if (selected.selectedPixels === 0 || selected.blockIds.length === 0) {
    await device.queue.onSubmittedWorkDone();
    profile.totalMs = performance.now() - tAll;
    return { device, buffer: outF32, profile };
  }
  const idxSource = groupMask ? selected.groupIdx : selected.localIdx;
  const groupIdxBuf = device.createBuffer({ size: Math.max(4, idxSource.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(groupIdxBuf, 0, idxSource.buffer as ArrayBuffer, idxSource.byteOffset, idxSource.byteLength);
  const groupMaskBuf = groupMask
    ? device.createBuffer({ size: Math.max(4, selected.groupMask.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    : null;
  if (groupMaskBuf) {
    device.queue.writeBuffer(groupMaskBuf, 0, selected.groupMask.buffer as ArrayBuffer, selected.groupMask.byteOffset, selected.groupMask.byteLength);
  }
  const tableSource = groupMask ? selected.groupBlockTable : selected.blockTable;
  const blockTableBuf = device.createBuffer({ size: Math.max(4, tableSource.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(blockTableBuf, 0, tableSource.buffer as ArrayBuffer, tableSource.byteOffset, tableSource.byteLength);
  type PreparedMaskedSumGroup = {
    jobs: MaskedSumJob[];
    compressedMB: number;
    packMs: number;
    uploadMs: number;
    buildMs: number;
  };
  const prepareGroup = async (groupSpecsInput: Bslz4MaskedSumSpec[]): Promise<PreparedMaskedSumGroup | null> => {
    const groupSpecs = groupSpecsInput.filter((spec) => maskedFrameWindow(spec).frameCount > 0);
    if (groupSpecs.length === 0) return null;
    const tPack = performance.now();
    const compacted = groupSpecs.map((spec) => {
      const { frameStart, frameCount } = maskedFrameWindow(spec);
      if (spec.selectedBlockIds) {
        return { ...compactSelectedBslz4Blocks(spec, selected.blockIds, frameStart, frameCount), startScan: spec.startScan };
      }
      return { ...compactBslz4Blocks(spec, selected.blockIds, frameStart, frameCount), startScan: spec.startScan };
    });
    const packMs = performance.now() - tPack;
    const compressedMB = compacted.reduce((n, s) => n + s.compressed.byteLength, 0) / 1e6;
    const tUpload = performance.now();
    const raws = await uploadViaStaging(device, compacted);
    const uploadMs = performance.now() - tUpload;
    const tBuild = performance.now();
    const jobs = compacted.map((spec, i) => buildMaskedSumLow8Job(device, spec, raws[i], groupIdxBuf, groupMaskBuf, blockTableBuf, outF32, selected.blockIds.length, wgSize, groupMask, compactShared));
    const buildMs = performance.now() - tBuild;
    return { jobs, compressedMB, packMs, uploadMs, buildMs };
  };
  const submitGroup = (prepared: PreparedMaskedSumGroup): Promise<number> => {
    const enc = device.createCommandEncoder();
    for (const job of prepared.jobs) job.record(enc);
    device.queue.submit([enc.finish()]);
    const tGpu = performance.now();
    return device.queue.onSubmittedWorkDone().then(() => performance.now() - tGpu);
  };
  const acceptGroup = (prepared: PreparedMaskedSumGroup, gpuMs: number): void => {
    profile.packMs += prepared.packMs;
    profile.compressedMB += prepared.compressedMB;
    profile.uploadMs += prepared.uploadMs;
    profile.buildMs += prepared.buildMs;
    profile.gpuWaitMs += gpuMs;
    for (const job of prepared.jobs) job.releaseTemps();
    profile.groups++;
  };
  if (useMaskedSumPipeline()) {
    let nextPrepared: Promise<PreparedMaskedSumGroup | null> | null = specs.length ? prepareGroup(specs.slice(0, groupSize)) : null;
    for (let g = 0; g < specs.length; g += groupSize) {
      const prepared = await nextPrepared;
      const nextStart = g + groupSize;
      if (!prepared) {
        nextPrepared = nextStart < specs.length ? prepareGroup(specs.slice(nextStart, nextStart + groupSize)) : null;
        continue;
      }
      const done = submitGroup(prepared);
      nextPrepared = nextStart < specs.length ? prepareGroup(specs.slice(nextStart, nextStart + groupSize)) : null;
      acceptGroup(prepared, await done);
    }
	  } else {
	    for (let g = 0; g < specs.length; g += groupSize) {
	      const prepared = await prepareGroup(specs.slice(g, g + groupSize));
	      if (!prepared) continue;
	      acceptGroup(prepared, await submitGroup(prepared));
	    }
	  }
	  groupIdxBuf.destroy();
  groupMaskBuf?.destroy();
  blockTableBuf.destroy();
  profile.compressedMB = +profile.compressedMB.toFixed(1);
  profile.totalMs = performance.now() - tAll;
  return { device, buffer: outF32, profile };
}

// Reusable MAP_WRITE staging pool. mappedAtCreation allocates + zero-inits host-visible
// memory every call (~3.9 GB/s, ~1.7s for a 6.6 GB dataset); a persistent pool pays that
// once, then map/unmap-reuses, so the per-load upload becomes just the (fast, wide) CPU copy
// + a GPU-internal copyBufferToBuffer. Cleared on device loss.
let STAGING: GPUBuffer[] = [];
let STAGING_BYTES = 0;
function ensureStaging(device: GPUDevice, count: number, bytes: number): void {
  if (STAGING.length >= count && STAGING_BYTES >= bytes) return;
  STAGING.forEach((b) => b.destroy());
  STAGING_BYTES = Math.max(bytes, STAGING_BYTES);
  STAGING = Array.from({ length: count }, () => device.createBuffer({ size: STAGING_BYTES, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC }));
}
onGPULost(() => { STAGING = []; STAGING_BYTES = 0; });

// Upload each spec's compressed bytes into a plain STORAGE buffer via the staging pool:
// map a pooled staging buffer (no per-load alloc), wide-copy the bytes in, then a GPU copy
// to the STORAGE buffer the decoder reads. Returns the raw buffers (caller destroys them).
async function uploadViaStagingWithProfile(device: GPUDevice, specs: Bslz4Spec[], waitForCopy = false): Promise<{
  rawBufs: GPUBuffer[];
  copyWaitMs: number;
}> {
  const align4 = (n: number) => Math.ceil(n / 4) * 4;
  const maxBytes = specs.reduce((m, s) => Math.max(m, align4(s.compressed.byteLength)), 0);
  ensureStaging(device, specs.length, maxBytes);
  const rawBufs = specs.map((s) => device.createBuffer({ size: align4(s.compressed.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }));
  await Promise.all(specs.map(async (s, i) => {
    const sz = align4(s.compressed.byteLength);
    await STAGING[i].mapAsync(GPUMapMode.WRITE, 0, sz);   // reused buffer: maps without re-alloc
    copyWide(STAGING[i].getMappedRange(0, sz), s.compressed);
    STAGING[i].unmap();
  }));
  const enc = device.createCommandEncoder();
  specs.forEach((s, i) => enc.copyBufferToBuffer(STAGING[i], 0, rawBufs[i], 0, align4(s.compressed.byteLength)));
  const tCopy = performance.now();
  device.queue.submit([enc.finish()]);   // ordered before the decode submit; next group's mapAsync waits on it
  const copyWaitMs = waitForCopy ? await device.queue.onSubmittedWorkDone().then(() => performance.now() - tCopy) : 0;
  return { rawBufs, copyWaitMs };
}

async function uploadViaStaging(device: GPUDevice, specs: Bslz4Spec[]): Promise<GPUBuffer[]> {
  return (await uploadViaStagingWithProfile(device, specs)).rawBufs;
}

async function stageUploadCopies(device: GPUDevice, specs: Bslz4Spec[]): Promise<{
  rawBufs: GPUBuffer[];
  recordCopies: (enc: GPUCommandEncoder) => void;
}> {
  const align4 = (n: number) => Math.ceil(n / 4) * 4;
  const maxBytes = specs.reduce((m, s) => Math.max(m, align4(s.compressed.byteLength)), 0);
  ensureStaging(device, specs.length, maxBytes);
  const rawBufs = specs.map((s) => device.createBuffer({ size: align4(s.compressed.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }));
  await Promise.all(specs.map(async (s, i) => {
    const sz = align4(s.compressed.byteLength);
    await STAGING[i].mapAsync(GPUMapMode.WRITE, 0, sz);
    copyWide(STAGING[i].getMappedRange(0, sz), s.compressed);
    STAGING[i].unmap();
  }));
  return {
    rawBufs,
    recordCopies(enc: GPUCommandEncoder) {
      specs.forEach((s, i) => enc.copyBufferToBuffer(STAGING[i], 0, rawBufs[i], 0, align4(s.compressed.byteLength)));
    },
  };
}

async function uploadViaCombinedStaging(device: GPUDevice, specs: Bslz4Spec[]): Promise<{
  raws: RawInput[];
  release: () => void;
}> {
  const align4 = (n: number) => Math.ceil(n / 4) * 4;
  const align256 = (n: number) => Math.ceil(n / 256) * 256;
  const offsets: number[] = [];
  const sizes = specs.map((s) => align4(s.compressed.byteLength));
  let totalBytes = 0;
  for (const size of sizes) {
    totalBytes = align256(totalBytes);
    offsets.push(totalBytes);
    totalBytes += size;
  }
  totalBytes = Math.max(4, align4(totalBytes));
  ensureStaging(device, 1, totalBytes);
  const rawBuf = device.createBuffer({ size: totalBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  await STAGING[0].mapAsync(GPUMapMode.WRITE, 0, totalBytes);
  const mapped = STAGING[0].getMappedRange(0, totalBytes);
  specs.forEach((s, i) => copyWide(mapped, s.compressed, offsets[i]));
  STAGING[0].unmap();
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(STAGING[0], 0, rawBuf, 0, totalBytes);
  device.queue.submit([enc.finish()]);
  const raws = specs.map((_, i) => ({
    buffer: rawBuf,
    offset: offsets[i],
    size: sizes[i],
    owned: false,
  }));
  return { raws, release: () => rawBuf.destroy() };
}

function uploadViaWriteBuffer(device: GPUDevice, specs: Bslz4Spec[]): GPUBuffer[] {
  const align4 = (n: number) => Math.ceil(n / 4) * 4;
  const chunkBytes = 64 * 1024 * 1024;
  return specs.map((s) => {
    const rawBuf = device.createBuffer({ size: align4(s.compressed.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    let offset = 0;
    while (offset < s.compressed.byteLength) {
      const remaining = s.compressed.byteLength - offset;
      const n = Math.min(chunkBytes, remaining);
      const aligned = n & ~3;
      if (aligned > 0) {
        device.queue.writeBuffer(
          rawBuf,
          offset,
          s.compressed.buffer as ArrayBuffer,
          s.compressed.byteOffset + offset,
          aligned,
        );
        offset += aligned;
      }
      const tail = s.compressed.byteLength - offset;
      if (tail > 0 && tail < 4) {
        const padded = new Uint8Array(4);
        padded.set(s.compressed.subarray(offset));
        device.queue.writeBuffer(rawBuf, offset, padded.buffer, 0, 4);
        offset = s.compressed.byteLength;
      }
    }
    return rawBuf;
  });
}

function uploadViaMapped(device: GPUDevice, specs: Bslz4Spec[]): GPUBuffer[] {
  const align4 = (n: number) => Math.ceil(n / 4) * 4;
  return specs.map((s) => {
    const rawBuf = device.createBuffer({
      size: align4(s.compressed.byteLength),
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    copyWide(rawBuf.getMappedRange(), s.compressed);
    rawBuf.unmap();
    return rawBuf;
  });
}

// Build (but don't submit) a decode job: allocates the GPU buffers, uploads the compressed
// bytes + block table, and returns a recorder that appends the two compute passes to a
// shared encoder. Batching N jobs into ONE submit + ONE await lets the GPU pipeline the
// chunks instead of draining between each (the per-chunk await was the decode bottleneck).
function buildDecodeJob(device: GPUDevice, spec: Bslz4Spec, dtype: "uint8" | "uint16" | "float32", srcDtype: "uint8" | "uint16" | "float32"): DecodeJob {
  const { compressed, blockMeta, nFrames, nBlocksPerFrame, blockElems, detSize } = spec;
  const f32 = srcDtype === "float32";
  const srcBytes = srcDtype === "uint8" ? 1 : f32 ? 4 : 2;
  const blockBytes = blockElems * srcBytes;
  const planeBytes = blockElems / 8;
  const totalBlocks = nFrames * nBlocksPerFrame;
  const totalElems = nFrames * detSize;
  const u8 = !f32 && (dtype === "uint8" || srcDtype === "uint8");
  // float32: 1 u32/pixel (the IEEE-754 bit pattern). uint16: 2 px/u32. uint8: 4 px/u32.
  const stackWords = f32 ? totalElems : u8 ? Math.ceil(totalElems / 4) : totalElems / 2;
  // Upload the compressed bytes via a mapped-at-creation buffer: writing straight into the
  // GPU-visible mapped range is a single copy, vs writeBuffer's internal CPU staging copy -
  // roughly 2x the host->device throughput, and uploading 7.5 GB/dataset is the decode floor.
  const rawSize = Math.ceil(compressed.byteLength / 4) * 4;
  const rawBuf = device.createBuffer({ size: rawSize, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
  new Uint8Array(rawBuf.getMappedRange()).set(compressed);
  rawBuf.unmap();
  const interBuf = device.createBuffer({ size: totalBlocks * blockBytes, usage: GPUBufferUsage.STORAGE });
  const metaBuf = device.createBuffer({ size: blockMeta.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(metaBuf, 0, blockMeta.buffer as ArrayBuffer, blockMeta.byteOffset, blockMeta.byteLength);
  const stack = device.createBuffer({ size: stackWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const cfg1 = uniform(device, [totalBlocks, blockBytes, 0, 0]);
  const nGroups = totalElems / 8;
  const p2wg = Math.ceil(nGroups / 64), gx = Math.min(p2wg, MAX_WG), gy = Math.ceil(p2wg / MAX_WG);
  const cfg2 = uniform(device, [nGroups, gx * 64, blockElems, planeBytes]);
  const { p1, p2 } = getPipes(device, srcDtype, u8, nBlocksPerFrame, detSize);
  const bg1 = device.createBindGroup({ layout: p1.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: rawBuf } }, { binding: 1, resource: { buffer: interBuf } },
    { binding: 2, resource: { buffer: metaBuf } }, { binding: 3, resource: { buffer: cfg1 } } ] });
  const bg2 = device.createBindGroup({ layout: p2.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: interBuf } }, { binding: 1, resource: { buffer: stack } },
    { binding: 2, resource: { buffer: cfg2 } } ] });
  return {
    stack, mode: f32 ? 2 : u8 ? 1 : 0,
    record(enc) {
      const pa = enc.beginComputePass(); pa.setPipeline(p1); pa.setBindGroup(0, bg1); pa.dispatchWorkgroups(Math.ceil(totalBlocks / 64)); pa.end();
      const pb = enc.beginComputePass(); pb.setPipeline(p2); pb.setBindGroup(0, bg2); pb.dispatchWorkgroups(gx, gy); pb.end();
    },
    releaseTemps() { rawBuf.destroy(); interBuf.destroy(); metaBuf.destroy(); cfg1.destroy(); cfg2.destroy(); },
  };
}

// Decode N bslz4 specs into N packed GPU stack buffers, batching the GPU work in groups so
// at most `groupSize` chunks' transient buffers (raw + intermediate) are live at once - one
// submit + one await per group lets the GPU overlap upload and compute across chunks.
export async function decodeBslz4Batch(specs: Bslz4Spec[], dtype: "uint8" | "uint16" | "float32" = "uint8", srcDtype: "uint8" | "uint16" | "uint32" | "float32" = "uint16", groupSize = 14): Promise<{ device: GPUDevice; buffers: GPUBuffer[]; mode: number; profile: Bslz4BatchProfile } | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  const f32 = srcDtype === "float32";
  const fused = dtype === "uint8" && srcDtype !== "float32";   // integer -> uint8 fast path
  const low8 = bslz4Low8Only();
  const coopLow8 = bslz4CoopLow8();
  const frameLow8 = bslz4FrameLow8();
  const u32Low8 = bslz4Low8U32Shared();
  const wordLow8 = bslz4WordLow8();
  const singleParseLow8 = bslz4SingleParseLow8();
  const frameSerialLow8 = bslz4FrameSerialLow8();
  const framesPerWg = frameLow8 && !u32Low8 && !wordLow8 && !singleParseLow8 && !frameSerialLow8 ? bslz4FramesPerWorkgroup() : 1;
  const frameWg = frameLow8 && !u32Low8 && !singleParseLow8 && !frameSerialLow8 ? bslz4FrameWorkgroupSize() : 64;
  const parallel = bslz4Parallel();
  const writeBufferUpload = bslz4UploadWriteBuffer();
  const mappedUpload = !writeBufferUpload && bslz4UploadMapped();
  const combinedUpload = !writeBufferUpload && !mappedUpload && bslz4UploadCombined();
  const pipelineStaging = (fused || f32) && !writeBufferUpload && !mappedUpload && bslz4PipelineStaging();
  const profileSplit = bslz4ProfileSplit();
  const uploadRoute = writeBufferUpload
    ? "writeBuffer"
    : mappedUpload
      ? "mapped"
      : combinedUpload
        ? (pipelineStaging ? "stagingCombinedPipeline" : "stagingCombined")
        : pipelineStaging
          ? "stagingPipeline"
          : "staging";
  const buffers: GPUBuffer[] = []; let mode = 0;
  const decodeVariant = f32 ? "fused-f32" : fused ? (low8 ? (singleParseLow8 ? "fused-frame-singleparse-low8-experimental" : frameSerialLow8 ? "fused-frame-serial-low8-experimental" : u32Low8 ? "fused-frame-u32-low8-experimental" : wordLow8 ? `fused-frame-word-low8-experimental-wg${frameWg}` : frameLow8 ? `fused-frame-coop-low8-experimental-fpw${framesPerWg}-wg${frameWg}` : coopLow8 ? "fused-coop-low8-experimental" : "fused-low8-experimental") : parallel ? "fused-parallel-experimental" : "fused-clip-u8") : "two-pass";
  const profile: Bslz4BatchProfile = {
    variant: `${decodeVariant}/${uploadRoute}`,
    groups: 0,
    specs: specs.length,
    compressedMB: +(specs.reduce((n, s) => n + s.compressed.byteLength, 0) / 1e6).toFixed(1),
    uploadMs: 0,
    uploadCopyWaitMs: 0,
    buildMs: 0,
    gpuWaitMs: 0,
    decodeComputeWaitMs: 0,
    totalMs: 0,
    profileSplit,
  };
  const tAll = performance.now();
  const fusedBuild = low8 ? buildFusedJob : parallel ? buildFusedJobD : buildFusedJob;
  type PreparedDecodeGroup = { specs: Bslz4Spec[]; jobs: DecodeJob[]; uploadMs: number; uploadCopyWaitMs: number; buildMs: number; releaseUpload?: () => void };
  const prepareStagingGroup = async (groupSpecs: Bslz4Spec[]): Promise<PreparedDecodeGroup> => {
    const tUpload = performance.now();
    const combined = combinedUpload ? await uploadViaCombinedStaging(device, groupSpecs) : null;
    const uploaded = combined ? null : await uploadViaStagingWithProfile(device, groupSpecs, profileSplit);
    const raws = combined ? combined.raws : uploaded!.rawBufs;
    const uploadMs = performance.now() - tUpload;
    const tBuild = performance.now();
    const jobs = groupSpecs.map((s, i) => f32 ? buildFusedJobF32(device, s, raws[i]) : fusedBuild(device, s, srcDtype as IntegralSrcDtype, raws[i]));
    return { specs: groupSpecs, jobs, uploadMs, uploadCopyWaitMs: uploaded?.copyWaitMs ?? 0, buildMs: performance.now() - tBuild, releaseUpload: combined?.release };
  };
  const submitPreparedGroup = (prepared: PreparedDecodeGroup): { done: Promise<number> } => {
    const enc = device.createCommandEncoder();
    for (const j of prepared.jobs) j.record(enc);
    device.queue.submit([enc.finish()]);
    const tGpu = performance.now();
    return { done: device.queue.onSubmittedWorkDone().then(() => performance.now() - tGpu) };
  };
  if (pipelineStaging) {
    if (profileSplit) {
      for (let g = 0; g < specs.length; g += groupSize) {
        const prepared = await prepareStagingGroup(specs.slice(g, g + groupSize));
        const submitted = submitPreparedGroup(prepared);
        const gpuMs = await submitted.done;
        profile.gpuWaitMs += gpuMs;
        profile.decodeComputeWaitMs = (profile.decodeComputeWaitMs ?? 0) + gpuMs;
        profile.uploadMs += prepared.uploadMs;
        profile.uploadCopyWaitMs = (profile.uploadCopyWaitMs ?? 0) + prepared.uploadCopyWaitMs;
        profile.buildMs += prepared.buildMs;
        for (const j of prepared.jobs) { j.releaseTemps(); buffers.push(j.stack); mode = j.mode; }
        prepared.releaseUpload?.();
        profile.groups++;
      }
      profile.totalMs = performance.now() - tAll;
      return { device, buffers, mode, profile };
    }
    let nextPrepared: Promise<PreparedDecodeGroup> | null = specs.length ? prepareStagingGroup(specs.slice(0, groupSize)) : null;
    for (let g = 0; g < specs.length; g += groupSize) {
      const prepared = await nextPrepared!;
      const submitted = submitPreparedGroup(prepared);
      const nextStart = g + groupSize;
      nextPrepared = nextStart < specs.length ? prepareStagingGroup(specs.slice(nextStart, nextStart + groupSize)) : null;
      const gpuMs = await submitted.done;
      profile.gpuWaitMs += gpuMs;
      if (profileSplit) profile.decodeComputeWaitMs = (profile.decodeComputeWaitMs ?? 0) + gpuMs;
      profile.uploadMs += prepared.uploadMs;
      profile.uploadCopyWaitMs = (profile.uploadCopyWaitMs ?? 0) + prepared.uploadCopyWaitMs;
      profile.buildMs += prepared.buildMs;
      for (const j of prepared.jobs) { j.releaseTemps(); buffers.push(j.stack); mode = j.mode; }
      prepared.releaseUpload?.();
      profile.groups++;
    }
    profile.totalMs = performance.now() - tAll;
    return { device, buffers, mode, profile };
  }
  for (let g = 0; g < specs.length; g += groupSize) {
    const groupSpecs = specs.slice(g, g + groupSize);
    // Fused paths (uint8 + float32) upload through the reused staging pool (no per-load
    // mappedAtCreation alloc); the non-fused path keeps its own mappedAtCreation upload.
    const tUpload = performance.now();
    let recordUploadCopies: ((enc: GPUCommandEncoder) => void) | null = null;
    let raws: GPUBuffer[] | null = null;
    if (fused || f32) {
      if (writeBufferUpload) raws = uploadViaWriteBuffer(device, groupSpecs);
      else if (mappedUpload) raws = uploadViaMapped(device, groupSpecs);
      else if (combinedUpload) {
        const staged = await stageUploadCopies(device, groupSpecs);
        raws = staged.rawBufs;
        recordUploadCopies = staged.recordCopies;
      } else {
        const uploaded = await uploadViaStagingWithProfile(device, groupSpecs, profileSplit);
        raws = uploaded.rawBufs;
        profile.uploadCopyWaitMs = (profile.uploadCopyWaitMs ?? 0) + uploaded.copyWaitMs;
      }
    }
    profile.uploadMs += performance.now() - tUpload;
    const tBuild = performance.now();
    const jobs = groupSpecs.map((s, i) => f32 ? buildFusedJobF32(device, s, raws![i]) : fused ? fusedBuild(device, s, srcDtype as IntegralSrcDtype, raws![i]) : buildDecodeJob(device, s, dtype, srcDtype as "uint8"|"uint16"|"float32"));
    const enc = device.createCommandEncoder();
    recordUploadCopies?.(enc);
    for (const j of jobs) j.record(enc);
    profile.buildMs += performance.now() - tBuild;
    device.queue.submit([enc.finish()]);
    const tGpu = performance.now();
    await device.queue.onSubmittedWorkDone();
    const gpuMs = performance.now() - tGpu;
    profile.gpuWaitMs += gpuMs;
    if (profileSplit) profile.decodeComputeWaitMs = (profile.decodeComputeWaitMs ?? 0) + gpuMs;
    for (const j of jobs) { j.releaseTemps(); buffers.push(j.stack); mode = j.mode; }
    profile.groups++;
  }
  profile.totalMs = performance.now() - tAll;
  return { device, buffers, mode, profile };
}

// Decode a bslz4 stack to a packed GPU buffer ([scanPos][detPixel]). dtype "uint8"
// (clip 0-255, 4 px/u32, offline default - half the memory) or "uint16" (lossless,
// 2 px/u32). Layout matches Show4DSTEMCompute.sample() for that mode exactly.
// Returns null if WebGPU is unavailable. Throws (validation) only on misuse.
export async function decodeBslz4ToStack(spec: Bslz4Spec, dtype: "uint8" | "uint16" | "float32" = "uint8", srcDtype: "uint8" | "uint16" | "uint32" | "float32" = "uint16"): Promise<{ device: GPUDevice; buffer: GPUBuffer; mode: number } | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  const fusedOk = dtype === "uint8" && srcDtype !== "float32";
  const job = srcDtype === "float32"
    ? buildFusedJobF32(device, spec)
    : fusedOk
    ? (bslz4Low8Only() ? buildFusedJob(device, spec, srcDtype as IntegralSrcDtype) : bslz4Parallel() ? buildFusedJobD(device, spec, srcDtype as IntegralSrcDtype, undefined) : buildFusedJob(device, spec, srcDtype as IntegralSrcDtype))
    : buildDecodeJob(device, spec, dtype, srcDtype as "uint8"|"uint16"|"float32");
  const enc = device.createCommandEncoder();
  job.record(enc);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  job.releaseTemps();
  return { device, buffer: job.stack, mode: job.mode };
}

function uniform(device: GPUDevice, vals: number[]): GPUBuffer {
  const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(b, 0, new Uint32Array(vals).buffer);
  return b;
}
