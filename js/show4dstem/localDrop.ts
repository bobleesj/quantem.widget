/**
 * Files of a folder dropped on a served-less Show4DSTEM viewer: the same
 * HDF5 masters, data files and index sidecars the "Open data folder" picker
 * grants. Only File handles are collected; no detector bytes are read here.
 */
interface DroppedEntry {
  isFile: boolean;
  isDirectory: boolean;
  file(success: (file: File) => void, failure: (error: DOMException) => void): void;
  createReader(): { readEntries(success: (entries: DroppedEntry[]) => void, failure: (error: DOMException) => void): void };
}

// "._" names are macOS AppleDouble metadata twins of real files, not HDF5.
const isH5Source = (file: File) => !file.name.startsWith("._") && /\.(h5|hdf5|qh5idx|qbslz4)$/i.test(file.name);

export async function collectDroppedH5Files(data: DataTransfer, maxFiles = 10000): Promise<File[]> {
  // Entries must be taken during the drop event: the browser protects the drag
  // data store once the handler returns.
  const entries = Array.from(data.items, item => item.webkitGetAsEntry() as DroppedEntry | null);
  if (!entries.some(Boolean)) return Array.from(data.files).filter(isH5Source);
  const files: File[] = [];
  const visit = async (entry: DroppedEntry): Promise<void> => {
    if (files.length >= maxFiles) throw new Error("Too many files. Drop only the acquisition folder.");
    if (entry.isFile) {
      const file = await new Promise<File>((resolve, reject) => entry.file(resolve, reject));
      if (isH5Source(file)) files.push(file);
    } else if (entry.isDirectory) {
      // readEntries returns a folder in batches until an empty one.
      const reader = entry.createReader();
      for (;;) {
        const children = await new Promise<DroppedEntry[]>((resolve, reject) => reader.readEntries(resolve, reject));
        if (!children.length) break;
        for (const child of children) await visit(child);
      }
    }
  };
  for (const entry of entries) if (entry) await visit(entry);
  return files;
}
