/** The detector image of the scan arriving, position by position.
 *
 *  The kernel sends each batch of newly summed scan positions as a `live_image_positions` message
 *  (raster order, the scan's own grid). They are written into one array that is drawn once per
 *  animation frame. Positions not received yet keep what was there: the scan before, as Velox shows it. */
export type PositionsMessage = {
  type?: string; scan: string; generation: number; start: number; count: number; total: number;
};

export function createLiveImage() {
  let scan: string | null = null;
  let generation = -1;
  let firstGeneration = -1;
  let data: Float32Array | null = null;
  let received = 0;
  let version = 0;
  let changedFrom = Infinity, changedTo = 0;
  return {
    /** Write one batch. `under` gives the image to start a new scan on when there is no scan before of
     *  this size (the rows received so far over the previous image), or null. False for a batch that cannot be used. */
    write(message: PositionsMessage, values: Float32Array, under: () => Float32Array | null = () => null): boolean {
      const total = Math.round(message.total);
      const start = Math.round(message.start);
      if (!(total > 0) || !(start >= 0) || start >= total || !message.scan) return false;
      if (message.scan !== scan || !data || data.length !== total) {
        if (!data || data.length !== total) {
          const seed = under();
          data = seed && seed.length === total ? Float32Array.from(seed) : new Float32Array(total).fill(NaN);
        }
        scan = message.scan;
        generation = firstGeneration = message.generation;
        received = 0;
      } else if (message.generation !== generation) {
        // The detector area moved: every position so far comes again, summed over the new area.
        generation = message.generation;
        received = 0;
      }
      const count = Math.min(values.length, Math.round(message.count), total - start);
      if (!(count > 0)) return false;
      data.set(count === values.length ? values : values.subarray(0, count), start);
      received = Math.max(received, start + count);
      changedFrom = Math.min(changedFrom, start);
      changedTo = Math.max(changedTo, start + count);
      version += 1;
      return true;
    },
    scan: () => scan,
    data: () => data,
    /** Positions of this scan (and detector area) written so far. */
    received: () => received,
    /** True once the detector area moved during this scan: the positions beyond `received` were summed over another area. */
    moved: () => generation !== firstGeneration,
    /** Counts every batch written: a drawing is current while this has not changed. */
    version: () => version,
    /** The detector area the positions are summed over, counted up each time it moves. */
    generation: () => generation,
    /** The rows of a `cols`-wide image written since the last take, [first, last), or null. */
    takeDirty(cols: number): [number, number] | null {
      if (!(changedTo > changedFrom) || !(cols > 0)) return null;
      const rows: [number, number] = [Math.floor(changedFrom / cols), Math.ceil(changedTo / cols)];
      changedFrom = Infinity;
      changedTo = 0;
      return rows;
    },
  };
}

/** The float32 values of a message buffer, copied (a comm buffer need not be aligned to four bytes). */
export function positionValues(buffer: DataView | ArrayBuffer | undefined, count: number): Float32Array | null {
  if (!buffer) return null;
  const bytes = buffer instanceof ArrayBuffer
    ? new Uint8Array(buffer) : new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const length = Math.min(Math.floor(bytes.byteLength / 4), Math.max(0, Math.round(count)));
  if (!length) return null;
  const values = new Float32Array(length);
  new Uint8Array(values.buffer).set(bytes.subarray(0, length * 4));
  return values;
}
