/** The images a streaming viewer's kernel attaches to each scan (Direct ptycho, centre of mass, DPC), kept by the
 *  scan's identity, one at a time as they come.
 *
 *  The kernel sends `{type: "vi_product_image", label, source, rows, cols}` with one float32 buffer of `rows * cols`
 *  values whenever one scan's image changes. Before 11-17 every attach() resent every scan's images as one stack
 *  (19 MB for five 512 x 512 scans, 16 times that at 2048 x 2048), and the page spent 100 to 200 ms decoding each.
 *  A scan without an image of a streamed label shows blank (NaN), as its slot in the stack did. */
export type ProductImageMessage = { type?: string; label: string; source: string; rows: number; cols: number };

export function createStreamedProducts() {
  const images = new Map<string, Map<string, Float32Array>>();
  const blanks = new Map<number, Float32Array>();
  let version = 0;
  return {
    /** Keep one scan's image. False for a message that cannot be used. */
    write(message: ProductImageMessage, values: Float32Array | null): boolean {
      const rows = Math.round(message.rows), cols = Math.round(message.cols);
      if (!values || !message.label || !message.source || !(rows > 0) || !(cols > 0)
          || values.length < rows * cols) return false;
      let byScan = images.get(message.label);
      if (!byScan) { byScan = new Map(); images.set(message.label, byScan); }
      byScan.set(message.source, values.length === rows * cols ? values : values.subarray(0, rows * cols));
      version += 1;
      return true;
    },
    /** Forget the scans no longer kept. */
    keep(sources: string[]) {
      const kept = new Set(sources);
      let dropped = false;
      for (const byScan of images.values()) {
        for (const source of [...byScan.keys()]) {
          if (!kept.has(source)) { byScan.delete(source); dropped = true; }
        }
      }
      if (dropped) version += 1;
    },
    /** One scan's image of `label` at rows x cols, or a blank (NaN) image of that size when there is none. */
    image(label: string, source: string, rows: number, cols: number): Float32Array {
      const found = images.get(label)?.get(source);
      if (found && found.length === rows * cols) return found;
      const size = rows * cols;
      let blank = blanks.get(size);
      if (!blank) { blank = new Float32Array(size).fill(NaN); blanks.clear(); blanks.set(size, blank); }
      return blank;
    },
    has(label: string, source: string): boolean { return Boolean(images.get(label)?.has(source)); },
    /** Counts every change: a drawing is current while this has not changed. */
    version: () => version,
  };
}

export type StreamedProducts = ReturnType<typeof createStreamedProducts>;

const stores = new WeakMap<object, StreamedProducts>();

/** The one store of a widget model. */
export function streamedProductsOf(model: object): StreamedProducts {
  let store = stores.get(model);
  if (!store) { store = createStreamedProducts(); stores.set(model, store); }
  return store;
}
