// The checkpoint stacks of ShowReconstruction in the browser: the projections, volumes and probes kept for stepping
// back through a run. They start from the synced traits (a saved notebook, an exported page, a view that just opened)
// and grow one checkpoint a message during a run, so an epoch costs one checkpoint over the comm, not the stack.

/** One stack: a frame (one checkpoint's float32 values) per kept epoch, oldest first, each with its own shape. */
export type Stack = { epochs: number[]; frames: Float32Array[]; shapes: number[][]; extras: unknown[] };

export const EMPTY: Stack = { epochs: [], frames: [], shapes: [], extras: [] };

/** The stack a synced trait holds: `bytes` the frames back to back, `shapes` one a frame. */
export function stackFromTrait(bytes: DataView | null | undefined, shapes: number[][] | undefined, epochs: number[] | undefined, extras?: unknown[]): Stack {
  const kept = (shapes || []).length;
  if (!bytes || !bytes.byteLength || !kept) return EMPTY;
  const all = bytes.byteOffset % 4 === 0 ? new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)  // a view, no copy: every view of a model reads one buffer
    : new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const frames: Float32Array[] = [];
  let offset = 0;
  for (const shape of shapes!) {
    const size = shape.reduce((product, length) => product * length, 1);
    frames.push(all.subarray(offset, offset + size));
    offset += size;
  }
  return {
    epochs: (epochs || []).slice(0, kept), frames, shapes: shapes!.slice(),
    extras: extras ? extras.slice(0, kept) : Array.from({ length: kept }, () => null),
  };
}

/** The stack after one checkpoint arrives: the frames Python no longer keeps (`kept`, its epoch list) are dropped,
 *  and the new frame is added at `epoch` (replacing one already there). */
export function appendCheckpoint(stack: Stack, epoch: number, frame: Float32Array, shape: number[], kept: number[], extra: unknown = null): Stack {
  const keep = new Set(kept);
  const epochs: number[] = [], frames: Float32Array[] = [], shapes: number[][] = [], extras: unknown[] = [];
  stack.epochs.forEach((value, index) => {
    if (keep.has(value) && value !== epoch) { epochs.push(value); frames.push(stack.frames[index]); shapes.push(stack.shapes[index]); extras.push(stack.extras[index]); }
  });
  epochs.push(epoch); frames.push(frame); shapes.push(shape); extras.push(extra);
  return { epochs, frames, shapes, extras };
}

/** Several runs' stacks as one, in order; an epoch two runs hold (a run's start is the last epoch of the one before)
 *  keeps the earlier run's frame. */
export function mergeStacks(stacks: Stack[]): Stack {
  const seen = new Set<number>();
  const out: Stack = { epochs: [], frames: [], shapes: [], extras: [] };
  for (const stack of stacks) stack.epochs.forEach((epoch, index) => {
    if (seen.has(epoch)) return;
    seen.add(epoch);
    out.epochs.push(epoch); out.frames.push(stack.frames[index]); out.shapes.push(stack.shapes[index]); out.extras.push(stack.extras[index]);
  });
  return out;
}

/** The index of the kept frame at or before an epoch (the first when none is that early), -1 for an empty stack. */
export function indexAt(stack: Stack, epoch: number): number {
  if (!stack.epochs.length) return -1;
  return Math.max(0, stack.epochs.reduce((best, value, index) => (value <= epoch ? index : best), 0));
}

/** A message buffer (DataView or ArrayBuffer) as its own float32 array. */
export function floatsOf(buffer: DataView | ArrayBuffer): Float32Array {
  const view = buffer instanceof DataView ? buffer : new DataView(buffer);
  return new Float32Array(view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength));
}
