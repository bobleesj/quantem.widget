# Export and run ShowPtycho

This tutorial exports a ShowPtycho reconstruction as a **browser folder** and
opens it with no Python kernel and no Jupyter. You can run it by granting the
folder to Chrome, or by using the generated local launcher. The exported viewer
opens from exact bright-field detector columns by default, then runs SSB live as
you tune aberrations.

If you just want the interactive widget inside a notebook, see
[ShowPtycho in Jupyter](showptycho.md) instead. This page assumes you already
have an **optimized** `ssb` (built and solved with `ssb.optimize()` +
`ssb.refine()` — that solve is required; an un-optimized export produces a junk
phase).

## Export

```python
from quantem.widget import ShowPtycho

# ssb is already optimized: SSB(...).optimize(n_trials=200).refine()
w = ShowPtycho(ssb, source_file="scan_master.h5", save_dir="out/")
w.export("out/", title="my sample SSB")
```

This writes a folder:

- `index.html` — the viewer
- `ShowPtycho.command` — double-click launcher for Chrome on macOS
- `source/` — the exact BF-column browser source plus linked HDF5 evidence
- `snapshots/` — calibration, manifest, viewer snapshots, and review metadata

The export persists no expanded float32 images and no complex64 BF reducers.
By default, the browser range-reads `source/bf_columns.u8` or
`source/bf_columns.u16` and does not decode the compressed HDF5 stack on open.

### Export at native detector size

The WebGPU browser export **cannot bin the detector**. If the `ssb` was built
with `det_bin=2` (a 96x96 calibration) but the embedded HDF5 is native 192x192,
the browser decodes 192x192, mismatches the calibration, and shows

```
detector shape mismatch; HDF5 has 192x192, calibration has 96x96
```

with blank panels. Always build and export at native detector size
(`det_bin=1`, the default).

## Run it

The exported folder needs `source/` and `snapshots/` present next to
`index.html`. There are two ways to open it.

### A. Double-click (File System Access)

1. Double-click `index.html`.
2. Click **Open data folder** and grant the folder the HTML lives in.
3. It renders, starting at the embedded calibration snapshot.

One grant per session. This works fully offline.

### B. CLI (serves and opens, no grant click)

```bash
quantem show out/
```

`quantem showptycho` is a compatibility alias for `quantem ptycho`, and
`quantem show` auto-detects the same folder. Point either command at the
exported folder; it serves the folder over range-capable HTTP and opens it, so
the viewer loads without the manual folder grant. Use this when double-click +
grant is inconvenient, for example over a remote connection.

## What you can do in the viewer

- Drag **C10 / C12 / phi12 / rotation** — the browser rebuilds the BF-indexed
  `G(k)` reducers and re-runs SSB live; the phase and FFT update in tens of
  milliseconds on a real GPU.
- Toggle the **FFT** panel to watch Bragg spots sharpen as aberrations improve.
- Change colormap, contrast, and the amplitude/complex view.
- **Save** writes the current aberrations and preview JPEG into `snapshots/`.

## Verify WebGPU is on real hardware

Interactive speed requires a real GPU. If a browser falls back to a software
renderer (SwiftShader), the reconstruction still runs but slowly, and any timing
you read is meaningless. On a real adapter the stats bar names the hardware
(for example `nvidia ...` or `apple ...`); a software fallback will not. New GPUs
can be missing from a browser's allow-list — if WebGPU is unexpectedly absent,
launch the browser with GPU blocklisting ignored.

## Checklist

1. The `ssb` was optimized (`optimize()` + `refine()`) before export.
2. Native detector (`det_bin=1`) — the browser cannot bin.
3. Export writes a clean root: `index.html`, `ShowPtycho.command`, `source/`,
   and `snapshots/`.
4. Open by double-click + **Open data folder**, or `quantem show out/`.
5. On open, the stats bar shows a non-null `loss` and the phase renders.

## Privacy

Exports embed source HDF5 file basenames in the metadata under `snapshots/` and
in the viewer state. Keep exported folders local and do not publish them if the
dataset or filenames are not yours to share.
