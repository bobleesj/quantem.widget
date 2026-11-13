import { expect, test } from "vitest";
import {
  THUMBNAIL_CELL_BYTES, THUMBNAIL_PAGE_FRAMES, THUMBNAIL_UNIFORM_BYTES, packThumbnailCells, packThumbnailUniforms,
  thumbnailLayerSize, thumbnailPagesFor,
} from "./thumbnails";

test("layer size: 128 on the long side, aspect kept, never upsampled", () => {
  expect(thumbnailLayerSize(1024, 1024)).toEqual({ width: 128, height: 128 });
  expect(thumbnailLayerSize(2048, 1024)).toEqual({ width: 128, height: 64 });
  expect(thumbnailLayerSize(512, 4096)).toEqual({ width: 16, height: 128 });
  expect(thumbnailLayerSize(64, 32)).toEqual({ width: 64, height: 32 });
  expect(thumbnailLayerSize(4000, 1)).toEqual({ width: 128, height: 1 });
});

test("pages grow in whole chunks", () => {
  expect(THUMBNAIL_PAGE_FRAMES).toBe(64);
  expect(thumbnailPagesFor(0)).toBe(0);
  expect(thumbnailPagesFor(1)).toBe(1);
  expect(thumbnailPagesFor(64)).toBe(1);
  expect(thumbnailPagesFor(65)).toBe(2);
  expect(thumbnailPagesFor(1000)).toBe(16);
  // a stream growing one frame at a time adds a page every 64 frames, not a buffer per frame
  let added = 0;
  for (let frames = 1; frames <= 300; frames++) if (thumbnailPagesFor(frames) !== thumbnailPagesFor(frames - 1)) added++;
  expect(added).toBe(5);
});

test("cell packing: rect, layer within its page, range; grouped by page; missing layers left out", () => {
  const cells = [
    { frame: 130, x: 0, y: 0, width: 131, height: 131, vmin: 1, vmax: 2 },
    { frame: 63, x: 131, y: 0, width: 131, height: 131, vmin: 3, vmax: 4 },
    { frame: 64, x: 0, y: 131, width: 131, height: 131, vmin: 5, vmax: 6 },
    { frame: 7, x: 131, y: 131, width: 131, height: 131, vmin: 7, vmax: 8 },
    { frame: 8, x: 0, y: 262, width: 131, height: 131, vmin: 9, vmax: 10 },
  ];
  const packed = packThumbnailCells(cells, frame => frame !== 8);
  expect(packed.count).toBe(4);
  expect(packed.data.byteLength).toBe(4 * THUMBNAIL_CELL_BYTES);
  expect(packed.groups).toEqual([{ page: 0, first: 0, count: 2 }, { page: 1, first: 2, count: 1 }, { page: 2, first: 3, count: 1 }]);
  const f32 = new Float32Array(packed.data);
  const u32 = new Uint32Array(packed.data);
  // first instance: frame 63 (page 0, layer 63)
  expect([f32[0], f32[1], f32[2], f32[3]]).toEqual([131, 0, 131, 131]);
  expect(u32[4]).toBe(63);
  expect([f32[6], f32[7]]).toEqual([3, 4]);
  // second: frame 7, third: frame 64 (page 1, layer 0), fourth: frame 130 (page 2, layer 2)
  expect(u32[8 + 4]).toBe(7);
  expect(u32[16 + 4]).toBe(0);
  expect(f32[16 + 6]).toBe(5);
  expect(u32[24 + 4]).toBe(2);
  expect(packThumbnailCells([], () => true).count).toBe(0);
});

test("uniform packing: canvas size, layer size, log scale", () => {
  const data = packThumbnailUniforms(524, 3600, 128, 64, true);
  expect(data.byteLength).toBe(THUMBNAIL_UNIFORM_BYTES);
  const f32 = new Float32Array(data);
  const u32 = new Uint32Array(data);
  expect([f32[0], f32[1]]).toEqual([524, 3600]);
  expect([u32[2], u32[3], u32[4]]).toEqual([128, 64, 1]);
  expect(new Uint32Array(packThumbnailUniforms(10, 10, 4, 4, false))[4]).toBe(0);
});
