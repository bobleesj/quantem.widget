# Experimental ANS sources in Show4DSTEM

ANS (asymmetric numeral systems) is the lossless count representation that
`quantem.gpu.io.load` uses for 4D-STEM acquisitions on CUDA and MPS. The codec,
container validation, CUDA/Metal/WebGPU decoding, and scientific GPU math
belong to **quantem.gpu**. Show4DSTEM provides the viewer and interaction
policy.

Use matching source revisions of both packages. This is not a PyPI release and
does not establish a stable container or private API compatibility promise.
The backend's `docs/api/qem-python.md` and `docs/api/qem-codecs.md` describe the
QEM file and its codecs; read them in the matching `quantem.gpu` checkout.

## Live viewers

An acquisition from `io.load` is already ANS encoded on the GPU, and
`Show4DSTEM` reads it directly:

```python
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

loaded = load("acquisition_master.h5")
Show4DSTEM(loaded)
```

See [Show4DSTEM](../api/show4dstem.md#encoded-acquisitions) for the live
behavior. These views need a kernel; the browser export below is the offline
path for encoded counts.

## QEM files in the browser

```python
from quantem.gpu import io
from quantem.widget.show4dstem_webgpu_export import export_show4dstem_rans_viewer

with io.load("acquisition_master.h5") as loaded:
    saved = io.save("acquisition.qem", loaded)
html = export_show4dstem_rans_viewer([saved.path], "qem-viewer")
```

`io.save` writes the encoded bytes of a loaded acquisition into one
self-contained `.qem` file without re-encoding, and never replaces an existing
file. A 4D NumPy uint8/uint16 array of shape
`(scan_row, scan_col, detector_row, detector_col)` can be encoded with the CPU
reference encoder instead: `io.save("acquisition.qem", counts, backend="cpu")`.
That encoder is bounded and not qualified for real-time full-acquisition
encoding.

The exporter verifies each file's header and body checksums, reads geometry
and native dtype from the header without decoding counts, and links the files
into the export folder (format token `qem-v1`). The browser decoder reads
uint8/uint16 counts only; a float32 or scaled file raises, and is viewed in a
live kernel with `Show4DSTEM(io.load(path))`. Open the generated viewer and
grant its linked `.qem` files with **Open QEM files**. For a series, pass an
ordered list of paths with matching shape and dtype; `tilts=` selects a subset
and `frame_labels=` names them. A detector-validity mask can be supplied
through `valid_pixels`; it does not overwrite raw sentinels in the source file.

The same exporter accepts a retained detector-rANS build manifest (the encoder
JSON) through an explicit compatibility adapter. The two formats share
implementation ownership, not an interchangeable bitstream.

This feature does **not** guarantee 120 displayed frames/s, general WebGPU
bitpacked-input support, or automatic multi-GPU placement. See the backend
guide for measured limits; rAF and render-submission counters are not
screen-presentation measurements.

## Building the shared backend

Install the matching `quantem.gpu` and widget checkouts in your development
environment, then build the widget against that exact backend:

```bash
python -m pip install -e /path/to/quantem.gpu -e /path/to/quantem.widget
cd /path/to/quantem.widget
npm ci
QUANTEM_GPU_SRC=/path/to/quantem.gpu/src PYTHON=python npm run build
```

`scripts/sync-gpu-webgpu.mjs` recreates the ignored generated engine tree
`js/.generated/engine/` from the file list in `quantem.gpu`'s
`webgpu/sources.json` manifest (for example `detector/webgpu/*`,
`io/hdf5/webgpu/*`, `ssb/webgpu/*`, `display/webgpu/*`, and `dpc/webgpu/*`).
Edit the package sources, not that generated tree. A widget built against an
unrelated backend checkout is not the validated experimental pair. Regenerate
exported HTML after rebuilding so an old viewer cannot retain old shader code.
