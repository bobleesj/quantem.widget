# ShowPtycho

Interactive ptychography aberration review for a prepared SSB reconstruction or
raw 4D-STEM data. `ShowPtycho` lets a microscopist tune defocus, astigmatism,
scan-detector rotation, phase flip, display contrast, and the FFT view while
watching the reconstructed phase update.

## What The Browser Computes

In a live notebook, Python owns the reconstruction state and uses the available
backend, usually CUDA on Linux or Metal/MPS on Apple silicon. The browser is the
viewer.

In a WebGPU folder export, the browser owns the interactive review. The default
folder contains a small `index.html` viewer, calibration metadata, and an exact
microscopy payload under `source/`. The default payload is the original
compressed HDF5 master/data source; the browser uses WebGPU to decompress the
selected BF evidence and build BF-indexed `G(k)` reducers transiently in GPU
memory. Moving
C10, C12, phi12, or scan rotation makes the browser build BF-indexed `G(k)`
reducers transiently in GPU memory and run the SSB phase reconstruction from
those transient buffers. The export does not persist expanded float32 images or
complex64 BF reducers by default.

FFT is a display analysis of the current reconstructed phase. When the FFT panel
is visible, the widget computes the FFT from the latest phase image and redraws
that FFT panel. It is not re-running the raw detector preprocessing.

## Real-Space Crop And SSB Refit

For a region-specific probe/aberration fit, open the prepared CUDA SSB with the
raw master path. ShowPtycho then exposes `Crop` in the top-right action group
beside `Export` and `Reset`. Enable `Crop`, drag a rectangle on the phase panel,
inspect the live row/column region readout, then use `Refit SSB`.
The widget reloads only that scan region from the original HDF5 source, runs 200
SSB optimization trials followed by refinement, and replaces the phase/FFT and
their calibration with the result.

```python
ssb = SSB(data, semiangle=20.0, scan_sampling=0.276, voltage_kV=300.0)
w = ShowPtycho(
    ssb,
    source_file="BTO_20_master.h5",
    fft_on=True,
)
```

The selection may be rectangular; each dimension must span at least `32` scan
positions. `Crop Reset` clears the selection without changing the current
reconstruction. This is a reconstruction operation, not a display crop.

The refit control is absent from standalone HTML/WebGPU exports and MPS-only
sessions. Those modes can inspect an existing result interactively, but only a
source-backed CUDA session has the raw detector data and SSB optimizer needed
to make a new scientific fit.

## Bright-Field Count

The `BF` count is the number of bright-field detector pixels used by the SSB
sum. It is a reconstruction quality/speed control, not a real-space crop and
not a display downsample. A `512 x 512` phase image stays `512 x 512` whether
the browser uses 30 percent of the BF disk or the full BF disk.

Use the full BF count for final microscopy claims:

- It averages information from the whole selected BF disk.
- It gives the lowest-noise, most stable phase and loss readout.
- It is the closest browser result to the backend reference reconstruction.
- It costs more first-use GPU time because every selected BF pixel contributes
  a complex `G(k)` image. Those reducers are transient in the default folder
  workflow, not saved as a persistent cache.

Use fewer BF pixels explicitly for exploration:

- Slider drag, playback, and sweep review can stay responsive.
- The large aberration trend is usually visible quickly.
- The phase can be noisier or slightly biased compared with full BF, especially
  for weak features, small BF fractions, or precise signoff.
- The UI status reports the active count as `used/total BF`, so a scientist can
  tell whether they are looking at a drag preview or a full-BF result.

For example, `drag_bf=0.3` is useful for rapidly finding the right defocus or
astigmatism neighborhood on memory-constrained browsers. The default is
`drag_bf=1.0`, so the first view and saved/signoff states use the full selected
BF disk.

## WebGPU Signoff Checklist

Use this checklist before claiming that a WebGPU folder export is ready for
interactive microscopy review:

- [ ] Record the WebGPU adapter. SwiftShader, llvmpipe, or any software adapter
  is not valid performance evidence.
- [ ] Record the native scan size. The browser kernels support square
  `128 x 128`, `256 x 256`, `512 x 512`, and `1024 x 1024` phase grids.
- [ ] Record the payload path: compressed HDF5 default or explicit BF-column companion.
  Do not treat a saved `g_bf.c64`/float32 cache as the default sharing path.
- [ ] Record the BF policy. Include both selected BF pixels and active aperture
  BF pixels, for example `542/1805 selected, 379 active`.
- [ ] Record first-use timing: bytes fetched, fetch time, unpack/decode time,
  FFT/reducer setup time, and total time.
- [ ] Drive C10, C12, phi12, scan rotation, BF, phase histogram, phase colormap,
  FFT toggle, FFT histogram, and flip controls in the browser.
- [ ] Report UI/GPU mean, p50, p95, and FPS-equivalent timing for repeated
  interactions. A visible image alone is not a performance signoff.
- [ ] Capture screenshots that show the phase image, FFT when enabled, BF
  status text, and performance readout.

Current implementation coverage:

| Native phase grid | Browser support | Current signoff status |
|---|---|---|
| `128 x 128` | Implemented by the shared WGSL path | Source/unit guard covered; use real headed data before paper claims |
| `256 x 256` | Implemented by the shared WGSL path | Source/unit guard covered; use real headed data before paper claims |
| `512 x 512` | Implemented by the shared WGSL path | Real experimental full-BF browser drive has reached about 24 FPS for C10 changes |
| `1024 x 1024` | Implemented by the shared WGSL path | Real Berk BF-column browser drive works, but full active-BF controls remain about 6 FPS on Phil Metal and fail the 30 FPS target |

## Folder Export

Use a WebGPU folder export when a colleague needs to open the same ptychography
review without the notebook kernel:

```python
w = ShowPtycho(ssb, fft_on=True)  # starts at full selected BF
w.export("logic013_512_review")
```

To start the viewer directly in the authoritative full-BF mode, pass
`drag_bf=1.0`:

```python
w = ShowPtycho(ssb, drag_bf=1.0, fft_on=True)
```

The default folder contains:

```text
logic013_512_review/
├── index.html
├── manifest.json
├── cal.json
├── README.md
└── source/
    ├── scan_master.h5
    ├── scan_data_000001.h5
    └── ...
```

Open the folder with the `quantem` CLI:

```bash
quantem ptycho /path/to/logic013_512_review
```

`quantem showptycho` remains a compatibility alias. The command validates the
folder, prints the compressed HDF5 source summary, starts the required local
HTTP server, and opens `index.html` in the browser. It stays alive until Ctrl-C.
Use `--port 8900` only when you need a stable URL, and use `--bind 0.0.0.0`
only when the viewer should be reachable from another device. Double-clicking
`index.html` is not the supported path because the browser still needs normal
HTTP fetches for the nearby HDF5 files.

The HTML file should stay small because it is only the viewer. The microscopy
payload is the original compressed HDF5 data under `source/`; the export avoids
writing `g_bf.c64`, `.f32` reference images, or detector-binned copies by
default. A legacy persistent BF cache is a compatibility/debug path, not the
normal sharing workflow.

## Reference

```{eval-rst}
.. autofunction:: quantem.widget.showptycho.ShowPtycho

.. autoclass:: quantem.widget.showptycho.PtychoCalibration
   :members:
```

## Interactive Controls

Each control should repaint the current phase image without requiring a
notebook round trip in WebGPU folder mode.

| Control | User effect | Expected behavior |
|---|---|---|
| C10 slider | Tune defocus | Phase updates; status reports GPU/UI time |
| C12 slider | Tune 2-fold astigmatism magnitude | Phase updates; loss shown for full-BF results |
| phi12 slider | Tune astigmatism angle | Phase updates; FFT follows when visible |
| Rotation slider | Tune scan-detector rotation | Reconstruction updates in the live backend path |
| FFT toggle | Show/hide reciprocal-space phase FFT | FFT computes only when visible |
| BF slider | Choose how much of the BF disk contributes to each interactive reconstruction | Starts at full BF; drag left for a smaller exploratory subset |
| Phase colormap / histogram | Change display mapping | Current image repaints; reconstruction data is unchanged |
| FFT colormap / histogram | Change FFT display mapping | FFT panel repaints; phase reconstruction is unchanged |
| Flip phase | Invert phase sign | Current phase and FFT update without recomputing BF data |
| Save/star | Mark a useful aberration state | Save only after recomputing with the intended BF count |

```{seealso}
The shared folder/single-file export language is documented in
[HTML export](html-export).
```
