/** A scan arriving is drawn over the last image of its kind, as Velox and HAADF scanning do: the positions it has
 *  measured come from the new scan, every position not measured yet (NaN) shows the last image of the same kind and
 *  shape, never black. A kind names what the image is (the detector image of one detector, DPC, the Direct phase)
 *  and the scan and detector shape; an image of another kind or shape starts from nothing. */
export function createPaintOver() {
  const last = new Map<string, Float32Array>();
  return {
    /** The image to draw: `image` itself when it is whole or nothing of its kind was drawn before, else a new
     *  image with its unmeasured positions taken from the last one drawn. Remembers what it returns. */
    over(kind: string, rows: number, cols: number, image: DataView | null): DataView | null {
      const pixels = rows * cols;
      if (!image || !(pixels > 0) || image.byteLength !== pixels * 4) return image;
      const values = image.byteOffset % 4 === 0
        ? new Float32Array(image.buffer, image.byteOffset, pixels)
        : new Float32Array(image.buffer.slice(image.byteOffset, image.byteOffset + pixels * 4));
      const key = `${kind}|${rows}x${cols}`;
      const under = last.get(key);
      let missing = -1;
      for (let i = 0; i < pixels; i++) if (values[i] !== values[i]) { missing = i; break; }
      if (missing < 0 || !under) {
        last.set(key, values);
        return image;
      }
      const drawn = new Float32Array(values);
      for (let i = missing; i < pixels; i++) if (drawn[i] !== drawn[i]) drawn[i] = under[i];
      last.set(key, drawn);
      return new DataView(drawn.buffer);
    },
    /** Forget every image: the next scan starts from nothing. */
    clear() { last.clear(); },
  };
}

export type PaintOver = ReturnType<typeof createPaintOver>;

/** What a viewer draws: painted over the last image of its kind when it is embedded in a host panel, the image as
 *  it came otherwise (a notebook viewer draws unmeasured positions as before). */
export function paintedOver(embedded: boolean, paint: PaintOver, kind: string, rows: number, cols: number,
                            image: DataView | null): DataView | null {
  return embedded ? paint.over(kind, rows, cols, image) : image;
}
