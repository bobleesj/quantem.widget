# Load and I/O — FAQ

Every question here comes from a real user session. Pick the one that matches
what you're trying to do; each answer is a copy-pasteable snippet.

For a beginner-friendly walkthrough of `uint8`/`uint16`, memory estimates,
and the image readers, start with {doc}`IO/GPU <../tutorials/io_gpu>`; GPU
selection and cleanup are in {doc}`Memory management <../tutorials/memory_management>`.

For the full function reference, see `load` and the autodocs at the
bottom of this page.

---

## First-time walkthrough (no data of your own required)

If you don't have a 4D-STEM master.h5 on disk yet, use one of the reference
datasets on Hugging Face — the whole flow is four lines:

```python
from quantem.gpu.io import load
from quantem.widget import ShowFolder, Show4DSTEM
from quantem.gpu.io import discover
from quantem.widget.io import list_datasets, download

# 1. See what's available (returns names like '4dstem/gold_512' with prefix)
list_datasets()

# 2. Download by SHORT name (drop the '4dstem/' prefix). Returns a Path
#    under ~/.cache/huggingface/... — cached after first call.
path = download("gold_512")

# 3. Look before you load — cached thumbnails, metadata, selection
ShowFolder(path)

# 4. Discover the master.h5 files + load the first + open the viewer
masters = discover(path)             # sorted list of Path
loaded = load(masters[0])            # encoded on the GPU at native detector sampling
Show4DSTEM(loaded)
```

**Gotcha**: `list_datasets()` returns `4dstem/gold_512` (with prefix) but
`download()` takes the SHORT name `gold_512` (no prefix). This is a quirk of
the underlying `quantem.data` API and will be aligned in a future release —
for now, drop the prefix.

The gold reference scans (`gold_512` and friends) are 1-5 GB
compressed and load in seconds on any modern GPU or Mac. Once this works
end-to-end, swap `download(...)` for `Path("/data/session")` and everything
downstream is identical.

## `ShowFolder(path)` — what's in this folder before I load anything?

`ShowFolder` is the folder-level entry point for real microscope sessions. It
builds cached thumbnails for image files, shows acquisition metadata, lets you
star files for downstream analysis, saves that selection, and can open starred
images immediately as Show2D or Show3D from the embedded selection panel.

```python
from quantem.widget import ShowFolder

folder = ShowFolder("/data/session")
folder.paths("image")  # selected files after you star panels
```

For 4D-STEM master files, use `quantem.gpu.io.discover` to return sorted master paths
for a scripted load. Then inspect one file with `quantem.gpu.io.inspect` or load it with
`quantem.gpu.io.load`.

Prefer `discover` when you just want the sorted paths back for a
scripted load:

```python
from quantem.gpu.io import discover

masters = discover("/data/session")               # all
masters = discover("/data/session", scan_shape=(512, 512))  # filter by scan size
```

Use `inspect` when you want readiness, shape, dtype, and calibration metadata
without loading detector frames:

```python
from quantem.gpu.io import inspect

report = inspect("/data/session/scan_00_master.h5")
print(report.scan_shape, report.detector_shape, report.dtype)
# e.g. (512, 512) (192, 192)
```

## How do I read only a scan ROI?

`load` keeps the whole acquisition encoded on the GPU, so a reconstruction or
denoise workflow reads only the scan patch it needs. `read` decodes that region
into a Torch tensor on the acquisition's GPU:

```python
from quantem.gpu.io import load

with load("/data/session/scan_00_master.h5") as loaded:
    patch_t = loaded.read(scan_region=(160, 293, 234, 367))  # row_start, row_stop, col_start, col_stop
print(patch_t.shape)  # torch.Size([133, 133, 192, 192])
```

Stops are exclusive, and `detector_region=` bounds the detector pixels the same
way. For a drift-corrected time series, derive `scan_region` from the shared
specimen ROI, the frame shift, and a small scan halo, then sample the final ROI
from the local patch. The detector counts remain raw; drift stays as
scan-position metadata.

## Lightweight visual thumbnails

Use `quantem.widget.render.thumbnail` when you need compact visual previews for
folder reports, static dashboards, or quick review pages. WebP is the default
preview format because it gives small files for noisy microscopy images while
still showing the structure a human needs for browsing: particles, scan
artifacts, contrast, FFT-like texture, and bad frames. This matters when a
folder contains hundreds of images, when an HTML report is opened on a laptop or
phone, or when a CI/dashboard page needs many thumbnails without becoming a
large artifact.

Use WebP thumbnails for:

- folder-browser previews and cached visual review pages
- maintainer smoke dashboards and HTML reports
- static report previews where file size matters more than exact pixel values
- quick human decisions such as "which file should I open next?"

Do not use WebP thumbnails for scientific data storage, measurements,
publication figures, exact widget state, or HTML exports that need interaction.
WebP is a visual preview and may be lossy. Keep scientific arrays in array/HDF5
formats when values need to be reused.

`q85` means "quality 85" for a lossy image encoder. Higher values keep more
visual detail and make larger files; lower values make smaller files and can
show compression artifacts. It is only a preview setting, not a scientific data
type or measurement setting.

Use this policy when choosing a widget output image format:

| Surface | Preferred format | Why |
|---|---|---|
| Saved-notebook fallback for `Show2D` / `Show3D` | JPEG preview generated from the widget render | Very portable across JupyterLab, VS Code, Colab, GitHub previews, and nbconvert; much smaller than PNG for noisy microscopy images |
| Folder browser, ShowFolder reports, smoke dashboards | WebP thumbnail, usually quality 85 | Smallest practical preview for pages with many images |
| Publication-style static output from `save_image(...)` | PNG, PDF, or TIFF | Stable, lossless or publication-friendly output |
| Interactive sharing | HTML export | Keeps controls live without a Python kernel |
| Exact analysis data or reproducible widget state | Array/HDF5 data plus JSON view state, or `save_state=True` only for small widgets | Preserves values and interactivity instead of storing a lossy preview |

The default saved-notebook path should stay conservative: keep notebooks small
by omitting heavy widget buffers, but use a broadly supported JPEG preview so
the output remains visible when someone opens the notebook without rerunning
the kernel. If a local workflow values smaller files more than maximum
notebook-tool compatibility, choose WebP explicitly:

```python
Show2D(image, notebook_preview_format="webp", notebook_preview_quality=85)
Show3D(stack, notebook_preview_format="webp", notebook_preview_quality=85)
```

```python
from quantem.widget.render import save_thumbnail, thumbnail_webp

webp_bytes = thumbnail_webp(image, size=256, cmap="inferno")
save_thumbnail(image, "preview.webp", size=256, cmap="inferno")
```

Use `save_image(...)` on a widget when you want a publication-style figure,
`export_html(...)` when you want the interactive widget, and `io.save(...)` or a
domain file format when you want data that another analysis step will consume.

---

## I'm on a Linux workstation with an NVIDIA RTX GPU. How do I load a scan?

```python
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

loaded = load("scan_master.h5")
Show4DSTEM(loaded)
```

`load` auto-detects CUDA and decodes the HDF5 chunks on the GPU into ANS
encoded storage. No flag is needed; pass `device=1` to choose another visible
CUDA device. A 512 x 512 x 192 x 192 uint16 scan (18 GiB as a dense array)
occupies about 0.1 to 2 GiB depending on its counts, so full detector
resolution fits on every common workstation GPU: RTX PRO 6000 Blackwell
(96 GB), L40S / A100 (48 GB), RTX 4090 / A6000 (24 GB), and smaller cards.
The memory budget is set by what you read or reconstruct from it, not by the
loaded acquisition.

## I'm on a MacBook (Apple Silicon). How do I load a scan?

Same one-liner as CUDA:

```python
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

loaded = load("scan_master.h5")
Show4DSTEM(loaded)
```

`load` auto-detects Apple Metal (MPS) and keeps the acquisition encoded in the
same form as on CUDA. Unified memory means "VRAM" = "RAM": a 24 GB MacBook
shares that memory with macOS, the browser, and everything else running, and
an encoded 512 x 512 x 192 x 192 scan uses about 0.1 to 2 GiB of it.

For several scans on a Mac, `load([m1, m2, m3])` returns one acquisition per
file, and `Show4DSTEM(load([m1, m2, m3]))` opens them as a comparison grid.
`Show4DSTEM.from_folder(folder)` opens after the first master and loads the
rest in the background.

## I want to compare several scans in one viewer.

Pass a list. `load` returns one encoded acquisition per master, and
`Show4DSTEM` opens the list as a comparison grid with one shared detector ROI.
File names become panel labels:

```python
masters = [
    "/data/session/file_001_master.h5",
    "/data/session/file_002_master.h5",
    "/data/session/file_003_master.h5",
]
acquisitions = load(masters)
Show4DSTEM(acquisitions)
```

The acquisitions must share scan shape, detector shape, and device. They are
not stacked into one 5D array; each panel reads its own acquisition at full
detector resolution and native count dtype.

## I want a viewer to follow a growing folder.

Show2D, Show3D, and Show4DSTEM share one folder-watching lifecycle. Watching is
enabled by default, and every viewer can be paused, polled, resumed, and closed
without constructing a replacement widget.

| Viewer | What a new file becomes | Data and memory behavior |
|---|---|---|
| `Show2D.from_folder(...)` | One new gallery panel; visible pages default to 20 panels | Reads only the new full-resolution source file; preserves the existing widget and per-file panel state |
| `Show3D.from_folder(...)` | One new frame in a single unpaged stack | Reads only the new full-resolution source file; preserves the existing widget and frame state |
| `Show4DSTEM.from_folder(...)` | One new panel in the comparison grid | Loads the new master into encoded GPU storage at full detector resolution; existing acquisitions stay loaded |

```python
from quantem.widget import Show2D, Show3D, Show4DSTEM

images = Show2D.from_folder(
    "/data/session/images",
    pattern="*.tif",
    page_size=20,  # another positive integer, or None for one gallery
)
movie = Show3D.from_folder("/data/session/frames", pattern="frame_*.tif")
scans = Show4DSTEM.from_folder(
    "/data/session/4dstem",
    pattern="*_master.h5",
    page_size=5,
)
```

The common lifecycle is:

```python
added = images.poll_folder()          # one immediate scan
images.stop_folder_watch()            # idempotent pause
images.watch_folder(interval=1.0)     # resume
images.close()                        # stop background work and close the comm
```

The same methods apply to all three viewers. `poll_folder()` returns the
zero-based indices appended by that scan. Pass `watch=False` to any
`from_folder(...)` call for deterministic manual polling. Watching is
append-only: known files are not duplicated, transiently incomplete files wait
for a later poll, and deletions do not remove already displayed scientific data.

Show2D folder pages are sequential independent files, not the repeated-slot
comparison pages accepted by direct `Show2D(...)`. Show3D folder files never
cross a page threshold: they always extend one frame axis, even when the folder
contains hundreds of frames.

These APIs load source data for scientific display. `ShowFolder` serves a
different purpose: it uses cached WebP thumbnails and metadata so a large
session can be browsed and selected quickly. Thumbnail pixels must never be
substituted for the full-resolution arrays opened by Show2D or Show3D, or for
the source masters opened by Show4DSTEM.

## I want to load every master file in a folder.

Use `Show4DSTEM.from_folder(...)` when the folder can grow or when you want the
viewer to open before every master has loaded:

```python
from quantem.widget import Show4DSTEM

viewer = Show4DSTEM.from_folder(
    "/data/session",
    backend="cuda",
    device=0,
    columns=3,
)
viewer.wait_for_folder()   # optional: block until every opening master is loaded
```

Every ready master is loaded into encoded GPU storage at full detector
resolution. The viewer opens after the first one; the rest join the comparison
grid in the background, and new ready masters append through `poll_folder()` /
`watch_folder()` without rebuilding the widget. Watching starts by default.
`viewer.free()` closes the acquisitions `from_folder` loaded.

Use explicit discovery plus `load(...)` when the file list is fixed and you want
to control exactly which masters are compared:

```python
from quantem.gpu.io import discover, load
from quantem.widget import Show4DSTEM

masters = discover("/data/session")   # sorted, filters to *_master.h5
acquisitions = load(masters)
Show4DSTEM(acquisitions)
```

`discover` also accepts a `scan_shape=(512, 512)` filter to keep only
matching acquisitions when a folder mixes scan sizes.

## Before loading anything, how do I check what's in a folder?

```python
from quantem.widget import ShowFolder

folder = ShowFolder("/data/session")  # thumbnails, metadata, selection, cache
```

Use the embedded selection panel to open starred images as Show2D or Show3D.
For 4D-STEM master files, pair this with `discover` and `inspect`
before calling `load`.

## How do I inspect a single master's calibration + metadata without loading it?

```python
from quantem.gpu.io import inspect

report = inspect("scan_master.h5")
print(report.metadata)  # voltage, semiangle, sampling, and source metadata
```

## I have HAADF or a 2D image (Velox EMD, TIFF, PNG). How do I load that?

```python
from quantem.widget import Show2D, read_image

img = read_image("haadf.emd")   # Dataset2d with sampling + units
Show2D(img)
```

For a stack (multi-frame TIFF, sequence of PNGs):

```python
from quantem.widget import Show3D, read_image_stack

stack = read_image_stack("frames", pattern="frame_*.png")
Show3D(stack)
```

## I want the reference gold or MoS2 dataset from Hugging Face.

```python
from quantem.widget.io import list_datasets, download

list_datasets()                # what's shared
path = download("gold_drift_0deg")   # returns local path
data = load(path)
```

## I want to save a loaded acquisition to disk.

```python
from quantem.gpu import io

with io.load("scan_master.h5") as loaded:
    io.save("scan.qem", loaded)   # one self-contained file; the encoded bytes are written as they are

loaded = io.load("scan.qem")      # reopens encoded on CUDA or MPS
```

A `.qem` destination selects the standalone QuantEM format. `io.save` never
replaces an existing file. Integer `.qem` files also open in the browser
viewer described in [Experimental ANS sources](../developer/experimental-ans.md).

## Does `load` bin the detector or change the dtype?

No. `load` keeps every detector pixel and the native count dtype (`uint8` or
`uint16` for counting detectors, uint32 counts that fit in `uint16` stored as
`uint16`, and `float32` sources with their bits unchanged). The
acquisition stays ANS encoded on the GPU, so full detector resolution is the
only load mode. Reductions belong to explicit later steps:

- `loaded.read(scan_region=..., detector_region=...)` decodes a bounded region.
- `Show4DSTEM.export_html(det_bin=..., scan_bin=..., dtype=...)` mean-bins and
  packs an exported browser payload.
- `quantem show4dstem ... --html --bin N --dtype uint8` does the same from the
  terminal.

## Memory rule of thumb for a 512×512×192×192 scan

| form | GPU memory |
|---|---:|
| dense uint16 array (not created by `load`) | 18 GiB |
| encoded acquisition from `load` | about 0.1 to 2 GiB, depending on counts |
| one bounded `read` of 64 scan rows, uint16 | 2.25 GiB |

Check the encoded size of a loaded acquisition with `loaded.resident_bytes`
and the dense size with `loaded.logical_bytes`. `Show4DSTEM(loaded)` adds the
viewer's own buffers (colormap, virtual-image cache, diffraction buffer) on top
of the encoded storage.

Detector files are often integers, not floating-point images. If you are new to
dtype choices: `uint16` (`u16`) stores exact raw detector counts from 0 to
65535 in 2 bytes per pixel. `uint8` (`u8`) stores 0 to 255 in 1 byte per pixel,
so it is smaller and faster for display, but it can saturate real count data.
`load` keeps the detector's own dtype; choose `uint8` only for an explicitly
labelled export.

## How do I choose a specific NVIDIA GPU?

Pass `device=` to `load` to choose among the CUDA devices the kernel can see:

```python
loaded = load("scan_master.h5", device=1)
```

To restrict which physical GPUs the kernel sees at all, set
`CUDA_VISIBLE_DEVICES` before launching Jupyter:

```bash
CUDA_VISIBLE_DEVICES=0 jupyter lab --no-browser --ip=0.0.0.0
```

Use `1`, `2`, etc. for another physical GPU. This is a CUDA/NVIDIA control; it
is not used for Apple Silicon machines.

```bash
CUDA_VISIBLE_DEVICES=1 jupyter lab --no-browser --ip=0.0.0.0
```

Inside the notebook:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.mem_get_info())
```

To release memory from the current Python kernel, close the viewer and then
the acquisition it shows:

```python
viewer.close()
loaded.close()
```

If memory is still occupied, another object or another Jupyter kernel still
owns it. Shut down that kernel from JupyterLab or stop the Python process.

## I have data others should use. How do I upload it?

The shared Hugging Face dataset repo
([bobleesj/quantem-data](https://huggingface.co/datasets/bobleesj/quantem-data),
MIT license) is the one place tutorial and reference data lives. Upload and
download commands, including Hugging Face pull requests, are on that dataset
card. This section is the `quantem.widget.io` helper reference for
maintainers who already have write access. The upload steps are:

1. **Install the hub extra and log in once.** Uploading needs a Hugging Face
   account and a write token from
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens):

   ```bash
   pip install "quantem.widget[hub]"
   hf auth login   # paste a Write token; answer n to git credential
   ```

   Token steps for a Community pull request (no write access) are on the
   dataset card. A Read token cannot open a PR. Ignore any hint to run
   `git config --global credential.helper store`.

2. **Upload with the bucket + sidecar convention.** The repo has two trees:

   - `widget-tutorials/<widget>/<dataset>/<size>/` — the baseline tutorial
     bundles behind `quantem.widget.datasets` (sizes `small`/`medium`/
     `large`/`full`). Contributors add these with a Hugging Face pull
     request, not `quantem.widget.io.upload`. See
     [Contribute tutorial data](../tutorials/contribute_data.md).
     Datasets shared by several widgets (the gold HAADF feeds both Show2D
     and Show3D) live under `widget-tutorials/shared/`.
   - `4dstem/` and `haadf/` — full-size originals for power users. A folder
     of Arina `*_master.h5` files goes under `4dstem/`, a single image file
     under `haadf/` (those are also the defaults for a directory vs a file).

   For `4dstem/` and `haadf/`, pass `meta=` so the helper writes a
   `quantem_meta.json` sidecar. Tutorial loaders read `meta.json` from the
   staged `widget-tutorials/` folder instead:

   ```python
   from quantem.widget.io import upload

   upload(
       "/data/session/gold_512/",          # folder -> 4dstem/gold_512/*
       name="gold_512",
       folder="4dstem",
       meta={"sampling": [0.5, 0.5], "units": ["A", "A"],
             "voltage_kV": 300, "probe_mrad": 30, "camera_length_mm": 91},
   )
   ```

   Write access to `bobleesj/quantem-data` is for maintainers. Contributors
   open a Hugging Face pull request on the dataset card, then a GitHub pull
   request for the named loader
   ([Contribute tutorial data](../tutorials/contribute_data.md)).

3. **Verify like a user would.** List, download to a fresh path, and open it
   in the widget before announcing the dataset:

   ```python
   from quantem.widget.io import list_datasets, download, status

   list_datasets()            # '4dstem/gold_512' should appear
   folder = download("gold_512")
   status()                   # repo-wide file/size snapshot
   ```

Remove a mistake with `delete("name")` — it deletes every file under the
dataset's folder, so double-check `list_datasets()` first. Uploads to the
shared repo are published under its MIT license; only upload data you have
the right to share.

## Function reference

```{eval-rst}
.. autofunction:: quantem.gpu.io.load
```

### Discover + inspect

```{eval-rst}
.. autofunction:: quantem.gpu.io.discover
```
```{eval-rst}
.. autofunction:: quantem.gpu.io.inspect
```

### Images (2D / 3D)

```{eval-rst}
.. autofunction:: quantem.widget.io.image.read_image
```
```{eval-rst}
.. autofunction:: quantem.widget.io.image.read_image_stack
```

### Hugging Face datasets

```{eval-rst}
.. autofunction:: quantem.widget.io.hub.list_datasets
```
```{eval-rst}
.. autofunction:: quantem.widget.io.hub.download
```
```{eval-rst}
.. autofunction:: quantem.widget.io.hub.upload
```
```{eval-rst}
.. autofunction:: quantem.widget.io.hub.status
```
```{eval-rst}
.. autofunction:: quantem.widget.io.hub.delete
```

### Save

```{eval-rst}
.. autofunction:: quantem.gpu.io.save
```
