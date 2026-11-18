/** The images of the scan arriving, kept outside React: each named image (the Direct phase, the detector image's
 *  rows, the centre of mass, DPC) is one array that new rows are written into as they come, and the rows that
 *  changed are remembered so a draw recolours only those.
 *
 *  Rows come as `live_image_rows` messages: `{type, scan, image, first_row, rows, cols, total, upsample}` with one
 *  float32 buffer of `rows * upsample` by `cols * upsample` values (`rows` scan rows starting at `first_row`).
 *  `under: true` marks rows drawn beneath the scan (the image before it), which do not count as measured.
 *  Rows not received yet keep what was there: a new scan paints over the last one. */
export type RowsMessage = {
  type?: string; scan: string; image: string; first_row: number; rows: number; cols: number; total: number;
  upsample?: number; under?: boolean;
};

export type LiveRowImage = {
  name: string; scan: string; rows: number; cols: number; factor: number;
  /** rows * factor by cols * factor */
  data: Float32Array;
  /** rows by cols: `data` itself at factor 1, else the mean over each factor x factor block */
  grid: Float32Array;
  /** scan rows of this scan written so far */
  measured: number;
};

/** The whole-image frame of the `live_phase_preview` trait, as sent before rows came as messages. */
export type LiveFrame = {
  rows?: number; cols?: number; total?: number; detRows?: number; detCols?: number; mode?: string;
  upsample?: number; scan?: string; com_unit?: string; images?: string[];
  diffraction_only?: boolean; source_id?: string;
  phase?: DataView; dp?: DataView; roi?: DataView; com_row?: DataView; com_col?: DataView; dpc?: DataView;
};

/** What the view needs to know about the scan arriving; it changes once per scan, not with every row. */
export type LiveInfo = {
  scan: string; total: number; cols: number; detRows: number; detCols: number; mode: string; comUnit: string; images: string[];
  diffractionOnly: boolean; sourceId: string;
};

/** The frame's description without its pixels; null when no scan is arriving. */
export function liveInfo(frame: LiveFrame | undefined | null): LiveInfo | null {
  if (!frame?.detRows || !frame.detCols || !frame.total || !frame.cols) return null;
  const images = FRAME_IMAGES.filter(([field, name]) => name !== "roi"
    && (frame[field] || (Array.isArray(frame.images) && frame.images.includes(name)))).map(([, name]) => name);
  return { scan: String(frame.scan ?? ""), total: Math.round(frame.total), cols: Math.round(frame.cols),
           detRows: Math.round(frame.detRows), detCols: Math.round(frame.detCols), mode: String(frame.mode ?? ""),
           comUnit: String(frame.com_unit ?? ""), images, diffractionOnly: Boolean(frame.diffraction_only),
           sourceId: frame.source_id ?? "" };
}

export function sameLiveInfo(a: LiveInfo | null, b: LiveInfo | null): boolean {
  if (!a || !b) return a === b;
  return a.scan === b.scan && a.total === b.total && a.cols === b.cols && a.detRows === b.detRows
    && a.detCols === b.detCols && a.mode === b.mode && a.comUnit === b.comUnit
    && a.diffractionOnly === b.diffractionOnly && a.sourceId === b.sourceId
    && a.images.length === b.images.length && a.images.every((name, index) => name === b.images[index]);
}

/** The float32 values of a comm buffer, copied only when it is not aligned to four bytes. */
export function floatValues(buffer: DataView | ArrayBuffer | undefined | null): Float32Array | null {
  if (!buffer) return null;
  if (buffer instanceof ArrayBuffer) return new Float32Array(buffer, 0, Math.floor(buffer.byteLength / 4));
  const length = Math.floor(buffer.byteLength / 4);
  if (buffer.byteOffset % 4 === 0) return new Float32Array(buffer.buffer, buffer.byteOffset, length);
  const values = new Float32Array(length);
  new Uint8Array(values.buffer).set(new Uint8Array(buffer.buffer, buffer.byteOffset, length * 4));
  return values;
}

/** The mean over each factor x factor block of scan rows [first, last), written into `grid`. */
export function binRows(data: Float32Array, grid: Float32Array, cols: number, factor: number, first: number, last: number) {
  const width = cols * factor;
  const scale = 1 / (factor * factor);
  for (let row = first; row < last; row++) {
    for (let col = 0; col < cols; col++) {
      let sum = 0;
      for (let r = 0; r < factor; r++) {
        const start = (row * factor + r) * width + col * factor;
        for (let c = 0; c < factor; c++) sum += data[start + c];
      }
      grid[row * cols + col] = sum * scale;
    }
  }
}

/** The names the frame's whole images are kept under. */
const FRAME_IMAGES: [keyof LiveFrame, string][] = [
  ["roi", "roi"], ["com_row", "COM_row"], ["com_col", "COM_col"], ["dpc", "DPC"],
];

export function createLiveRows() {
  const images = new Map<string, LiveRowImage>();
  const dirty = new Map<string, [number, number]>();
  let version = 0;
  const mark = (name: string, first: number, last: number) => {
    const rows = dirty.get(name);
    dirty.set(name, rows ? [Math.min(rows[0], first), Math.max(rows[1], last)] : [first, last]);
    version += 1;
  };
  const write = (message: RowsMessage, values: Float32Array | null): boolean => {
    const total = Math.round(message.total), cols = Math.round(message.cols);
    const first = Math.round(message.first_row), count = Math.round(message.rows);
    const factor = Math.max(1, Math.round(message.upsample ?? 1));
    if (!values || !message.image || !message.scan || !(total > 0) || !(cols > 0) || !(first >= 0) || !(count > 0)
        || first + count > total || values.length < count * factor * cols * factor) return false;
    let image = images.get(message.image);
    if (!image || image.rows !== total || image.cols !== cols || image.factor !== factor) {
      const data = new Float32Array(total * factor * cols * factor).fill(NaN);
      image = { name: message.image, scan: message.scan, rows: total, cols, factor, data,
                grid: factor === 1 ? data : new Float32Array(total * cols).fill(NaN), measured: 0 };
      images.set(message.image, image);
    } else if (image.scan !== message.scan) {
      image.scan = message.scan;
      image.measured = 0;
    }
    const width = cols * factor;
    const length = count * factor * width;
    image.data.set(values.length === length ? values : values.subarray(0, length), first * factor * width);
    if (factor > 1) binRows(image.data, image.grid, cols, factor, first, first + count);
    if (!message.under) image.measured = Math.max(image.measured, first + count);
    mark(message.image, first, first + count);
    return true;
  };
  return {
    /** Write the rows of one message. False for a message that cannot be used. */
    write,
    /** Write the whole images of a `live_phase_preview` frame that still carries its pixels. */
    writeFrame(frame: LiveFrame | undefined | null): boolean {
      const info = liveInfo(frame);
      if (!info || !frame || !frame.rows || frame.rows > info.total) return false;
      const rows = Math.round(frame.rows);
      const base = { scan: info.scan, first_row: 0, rows, cols: info.cols, total: info.total };
      let written = false;
      if (frame.mode === "Direct" && frame.phase) {
        const factor = Math.max(1, Math.round(frame.upsample ?? 1));
        if (frame.phase.byteLength === rows * factor * info.cols * factor * 4) {
          written = write({ ...base, image: "Direct", upsample: factor }, floatValues(frame.phase)) || written;
        }
      }
      for (const [field, name] of FRAME_IMAGES) {
        const bytes = frame[field] as DataView | undefined;
        if (bytes && bytes.byteLength === rows * info.cols * 4) {
          written = write({ ...base, image: name }, floatValues(bytes)) || written;
        }
      }
      return written;
    },
    /** The image kept under this name, when it is this scan's size. */
    get(name: string, rows?: number, cols?: number): LiveRowImage | null {
      const image = images.get(name);
      if (!image || (rows !== undefined && image.rows !== rows) || (cols !== undefined && image.cols !== cols)) return null;
      return image;
    },
    /** The scan rows of an image written since the last take, [first, last), or null. */
    takeDirty(name: string): [number, number] | null {
      const rows = dirty.get(name) ?? null;
      dirty.delete(name);
      return rows;
    },
    /** Forget every image: scans were cleared, and the next scan is not drawn over them. */
    clear() { images.clear(); dirty.clear(); version += 1; },
    /** Counts every write: a drawing is current while this has not changed. */
    version: () => version,
  };
}

export type LiveRows = ReturnType<typeof createLiveRows>;

/** The image the view shows while a scan arrives: the operator's choice when this scan has rows of it, else the
 *  image shown before while it has data (no black panel while the new choice has nothing yet), else the detector
 *  image. `has(name)` says whether this scan has rows of that image. */
export function liveShownImage(choice: string, shownBefore: string | null, has: (name: string) => boolean): string {
  if (has(choice)) return choice;
  if (shownBefore && has(shownBefore)) return shownBefore;
  return "roi";
}

/** At most `limit` evenly spaced values of [0, length), for a contrast range that follows a growing image without
 *  reading every pixel on every look. */
export function sampledValues(data: Float32Array, length: number, limit: number, into: Float32Array | null): Float32Array {
  const count = Math.min(length, data.length);
  if (count <= limit) return count === data.length ? data : data.subarray(0, count);
  const samples = into && into.length === limit ? into : new Float32Array(limit);
  const step = count / limit;
  for (let i = 0; i < limit; i++) samples[i] = data[Math.floor(i * step)];
  return samples;
}
