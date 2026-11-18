import { describe, expect, it } from "vitest";
import { blankUnvisited, blankUnvisitedDetail, unvisitedMask, unvisitedPattern } from "./unvisited";

describe("unvisited positions", () => {
  const mask = unvisitedMask(new DataView(new Uint8Array([0, 1, 0, 1]).buffer), 2, 2);

  it("draws the positions a path never visited blank", () => {
    const image = new DataView(new Float32Array([1, 2, 3, 4]).buffer);
    const drawn = new Float32Array(blankUnvisited(image, mask, 2, 2)!.buffer);
    expect(drawn[0]).toBe(1);
    expect(Number.isNaN(drawn[1])).toBe(true);
    expect(drawn[2]).toBe(3);
    expect(Number.isNaN(drawn[3])).toBe(true);
    expect(new Float32Array(image.buffer)[1]).toBe(2);          // the received image is left as it was
  });

  it("leaves a scan measured everywhere as it came", () => {
    const image = new DataView(new Float32Array([1, 2, 3, 4]).buffer);
    expect(blankUnvisited(image, null, 2, 2)).toBe(image);
    expect(unvisitedMask(new DataView(new Uint8Array(3).buffer), 2, 2)).toBeNull();   // another size: no mask
  });

  it("blanks each unvisited position's pixels of an upsampled image", () => {
    const detail = { data: new Float32Array(16).fill(5), factor: 2, rows: 4, cols: 4 };
    const drawn = blankUnvisitedDetail(detail, mask, 2, 2)!.data;
    expect(drawn[0]).toBe(5);
    expect(Number.isNaN(drawn[2])).toBe(true);     // row 0, col 2: scan position (0, 1)
    expect(Number.isNaN(drawn[15])).toBe(true);    // row 3, col 3: scan position (1, 1)
    expect(drawn[8]).toBe(5);                      // row 2, col 0: scan position (1, 0)
  });

  it("shows no pattern at a position the path never visited", () => {
    const pattern = new DataView(new Float32Array([7, 7, 7]).buffer);
    expect(unvisitedPattern(pattern, mask, 0, 3)).toBe(pattern);
    expect([...new Float32Array(unvisitedPattern(pattern, mask, 1, 3)!.buffer)].every(Number.isNaN)).toBe(true);
  });
});
