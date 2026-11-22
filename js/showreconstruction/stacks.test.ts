import { describe, expect, it } from "vitest";
import { appendCheckpoint, EMPTY, floatsOf, indexAt, mergeStacks, stackFromTrait } from "./stacks";

// A run streams one checkpoint a message; the browser's stack must equal what flush() would sync as a trait.

function frame(value: number) { return new Float32Array([value, value + 0.5, value + 1, value + 1.5]); }

describe("checkpoint stacks", () => {
  it("appended checkpoints equal the flushed trait", () => {
    let stack = EMPTY;
    for (let epoch = 1; epoch <= 5; epoch++) stack = appendCheckpoint(stack, epoch, frame(epoch), [2, 2], Array.from({ length: epoch }, (_, i) => i + 1), [epoch / 10]);
    const flushed = new Float32Array([1, 2, 3, 4, 5].flatMap((value) => Array.from(frame(value))));
    const trait = stackFromTrait(new DataView(flushed.buffer), Array.from({ length: 5 }, () => [2, 2]), [1, 2, 3, 4, 5], [[0.1], [0.2], [0.3], [0.4], [0.5]]);
    expect(stack.epochs).toEqual(trait.epochs);
    expect(stack.frames.map((one) => Array.from(one))).toEqual(trait.frames.map((one) => Array.from(one)));
    expect(stack.extras).toEqual(trait.extras);
  });

  it("drops what Python thinned and keeps the newest", () => {
    let stack = EMPTY;
    for (let epoch = 1; epoch <= 4; epoch++) stack = appendCheckpoint(stack, epoch, frame(epoch), [2, 2], [1, 2, 3, 4].slice(0, epoch));
    stack = appendCheckpoint(stack, 5, frame(5), [2, 2], [1, 3, 5]);
    expect(stack.epochs).toEqual([1, 3, 5]);
    expect(stack.frames[1][0]).toBe(3);
  });

  it("keeps frames of another shape: a run that changes its probe's modes keeps its earlier draws", () => {
    let stack = appendCheckpoint(EMPTY, 1, frame(1), [2, 2], [1]);
    stack = appendCheckpoint(stack, 2, new Float32Array(9), [3, 3], [1, 2]);
    expect(stack.epochs).toEqual([1, 2]);
    expect(stack.shapes).toEqual([[2, 2], [3, 3]]);
    const flushed = new Float32Array([...Array.from(frame(1)), ...new Array(9).fill(0)]);
    const trait = stackFromTrait(new DataView(flushed.buffer), [[2, 2], [3, 3]], [1, 2]);
    expect(trait.frames.map((one) => one.length)).toEqual([4, 9]);
  });

  it("merges runs, a run's start frame giving way to the earlier run's last", () => {
    const first = appendCheckpoint(appendCheckpoint(EMPTY, 0, frame(0), [2, 2], [0]), 8, frame(8), [2, 2], [0, 8]);
    const second = appendCheckpoint(appendCheckpoint(EMPTY, 8, frame(80), [2, 2], [8]), 16, frame(16), [2, 2], [8, 16]);
    const merged = mergeStacks([first, second]);
    expect(merged.epochs).toEqual([0, 8, 16]);
    expect(merged.frames[1][0]).toBe(8);
    expect(indexAt(merged, 12)).toBe(1);
    expect(indexAt(merged, -3)).toBe(0);
    expect(indexAt(EMPTY, 3)).toBe(-1);
  });

  it("reads a buffer at an offset", () => {
    const backing = new Float32Array([9, 1, 2, 9]);
    expect(Array.from(floatsOf(new DataView(backing.buffer, 4, 8)))).toEqual([1, 2]);
  });
});
