# Command line

Installing `quantem.widget` adds a `quantem` command (and a short `qw` alias).
Point it at a file or a folder and it renders the right viewer - no notebook,
no Python.

```bash
quantem show ./anything/                     # auto-detect content, pick the viewer
quantem show2d scan.png                       # an image            -> Show2D
quantem show3d ./frames/                       # a folder of frames -> Show3D scrub
quantem show2d ./frames/ --watch               # live folder        -> append new images
quantem show4dstem ./masters/                  # *_master.h5        -> live Show4DSTEM
quantem show4dstem a_master.h5 b_master.h5     # several masters    -> one 5D multi-tilt viewer
quantem show4dstem ./masters/ --html           # 4D-STEM            -> shareable offline HTML
quantem ptycho scan_master.h5                   # raw 4D-STEM master -> ptychography WebGPU review
quantem ptycho ./ptycho-export/                 # ShowPtycho folder  -> WebGPU browser review
quantem showfolder ./session/                  # microscopy folder  -> ShowFolder notebook/HTML
quantem data-transfer plan ./raw/ /ssd0/run /ssd1/run --manifest run.json
quantem data-transfer show4dstem --manifest run.json --gpus 0,1 --dtype u8 --bin 1
quantem html tutorial.ipynb                    # a notebook         -> standalone interactive HTML
quantem github tutorial_github.ipynb --no-execute # optional static copy for GitHub preview
```

## Subcommands

| Command | Input | Output |
|---|---|---|
| `quantem show <path>` | anything | auto-detects and dispatches to one of the below |
| `quantem show2d <image / folder>` | one image, or a folder | a Show2D HTML (a folder becomes a gallery); with `--watch`, a live ShowFolder notebook |
| `quantem show3d <folder>` | a folder of same-size frames | a Show3D scrub HTML; with `--watch`, a live ShowFolder notebook |
| `quantem show4dstem <master(s) / folder>` | one or more `*_master.h5` | a live Show4DSTEM notebook (or `--html`) |
| `quantem ptycho <master.h5 / folder>` | a raw `*_master.h5` or a ShowPtycho WebGPU folder export | builds/serves a ptychography browser review |
| `quantem showptycho <master.h5 / folder>` | same as `quantem ptycho` | compatibility alias |
| `quantem showfolder <folder>` | microscopy session folder | a ShowFolder notebook (or `--html`) |
| `quantem data-transfer plan/inspect/copy/update/masters/show4dstem` | `*_master.h5` folder plus target roots | manifest-backed transfer planning, state inspection, explicit copy, resume/update, ready-master listing, and Show4DSTEM notebook handoff |
| `quantem html <notebook.ipynb>` | a notebook you wrote | runs it, or with `--no-execute` exports saved outputs/state, into one standalone interactive HTML |
| `quantem github <notebook.ipynb>` | an optional static copy of a notebook | strips widget state and embeds compressed pictures for GitHub's notebook preview |
| `quantem jupyter` | nothing (run on the GPU box) | starts JupyterLab (`--env`, `--port`) and prints the SSH-tunnel line to paste on your laptop |

**Images** save a standalone HTML and open in your browser. **4D-STEM** opens a
live, kernel-backed notebook by default (full real-time interaction); `--html`
instead writes an **offline WebGPU browser export** - drag detectors, switch
BF/ABF/ADF, pan diffraction, all with no kernel. Compact single-file exports can
be opened directly; interactive raw 4D exports may include a local launcher when
Chrome must fetch a companion data payload over HTTP.

Several masters (a folder, or listed explicitly) stack into **one 5D viewer with a
Dataset slider** to flip between scans. `--combined --html` writes that as one
offline file (served locally, since a `file://` page can't fetch its companion).

Everything lands in `~/Downloads` (or the current directory on machines without
one) and opens automatically on a desktop.

## Show4DSTEM HTML export

Use the CLI when you want a quick browser artifact from raw masters:

```bash
quantem show4dstem scan_001_master.h5 --html --bin 8
quantem show4dstem scan_001_master.h5 --html --bin 1 --dtype uint16
quantem show4dstem ./session_masters --html --bin 8 --out ~/Downloads
quantem show4dstem scan_001_master.h5 scan_002_master.h5 --html --combined --bin 8
```

`--bin` is detector mean binning for the exported browser payload. The default
Show4DSTEM CLI export is an interactive raw-4D WebGPU viewer after that explicit
binning choice. It is meant for offline detector-ROI browsing, not as the
smallest possible report.

Use `--bin 1 --dtype uint16` when the user wants the full native detector
sampling path without opening Jupyter:

```bash
quantem show4dstem /data/session --html --bin 1 --dtype uint16 --out ~/Downloads
```

That command can produce very large browser artifacts. It is the right no-notebook
choice when native detector detail matters; for quick laptop browsing use
`--bin 8 --dtype uint8`, and for compact collaborator review use the Python
`export_kind="report"` path below.

For large lazy folders, curated review grids, or collaborator screening,
open a live viewer and export a compact report from Python instead:

```python
from quantem.widget import Show4DSTEM

viewer = Show4DSTEM.from_folder(
    "/data/session",
    gpus=[0, 1],
    det_bin=1,
    dtype="u8",
    view_mode="multiple",
    page_size=12,
)

viewer.export_html(
    "show4dstem_report.html",
    export_kind="report",
    dataset_scope="unhidden",
    scan_bin=2,
    det_bin=8,
    dtype="uint8",
)
```

Use `export_kind="interactive"` from Python when you want the same offline
browser interaction as the CLI but need finer control over real-space binning,
detector binning, or dtype:

```python
viewer.export_html(
    "show4dstem_interactive.html",
    export_kind="interactive",
    dtype="uint8",
    scan_bin=2,
    det_bin=4,
)
```

See [Show4DSTEM export recipes](tutorials/show4dstem_export) for the decision
table and LLM-friendly checklist.

## ShowPtycho folder review

ShowPtycho WebGPU review can start directly from one `*_master.h5`:

```bash
quantem ptycho BTO_18_master.h5
```

The command looks for a matching QuantEM calibration next to the master, for
example `quantem/screen/_calibrations.json`. When that file is not present, it
uses quick-start defaults and prints them before loading:

```text
semiangle=30 mrad, scan_sampling=0.5 A, voltage=300 kV
```

Those defaults are enough for fast local review and collaborator handoff. For
measurement, publication, or calibration signoff, provide the microscope
geometry explicitly:

```bash
quantem ptycho BTO_18_master.h5 \
  --semiangle 30 --scan-sampling 0.264 --voltage-kv 300
```

ShowPtycho defaults to native detector pixels (`--bin 1`) because ptychography
review should not silently downsample the bright-field disk. Use `--bin N` only
when you intentionally want a downsampled exploratory export. The generated
artifact is a folder containing `index.html`, `manifest.json`, calibration
metadata, and a `source/` directory with the original compressed HDF5 master and
data files linked or copied into the review folder. It does not save persistent
float32 reference images or a complex64 BF reducer by default. The browser
decodes HDF5 chunks on WebGPU and builds the BF-indexed reducers transiently.
The default interactive BF policy is full selected BF (`--drag-bf 1.0`) so the
first view uses all known BF evidence without loading non-BF detector pixels.
Use `--drag-bf 0.3` or another smaller fraction only when you intentionally want
a faster exploratory preview.

The browser source is the compressed HDF5 WebGPU path: the exported folder
carries the original compressed HDF5 under `source/`, and the browser
decompresses the selected BF evidence with WebGPU.

Existing ShowPtycho WebGPU exports are also folders because the microscopy
payload can be several gigabytes. Open them with the CLI:

```bash
quantem ptycho ./logic013_512_bfr24/
```

or let auto-detection choose the same path:

```bash
quantem show ./logic013_512_bfr24/
```

The command validates `manifest.json`, prints the compressed HDF5 source
summary, starts the required local HTTP server with byte-range support, opens
`index.html`, and stays alive until Ctrl-C. Use `--port 8900` for a fixed port
or `--bind 0.0.0.0` only when you intentionally want another device on the
network to reach the viewer. Share the whole folder with a colleague; sending
only `index.html` omits the HDF5 source files needed for WebGPU reconstruction.

## DataTransfer

Use `data-transfer` before heavy multi-GPU browsing or ptychography when a
session should be split across fast disks. It writes a durable manifest that the
CLI, Python utilities, and downstream tools can inspect later.

```bash
quantem data-transfer plan ./raw_session/ /nvme0/session /nvme1/session --manifest session.json
quantem data-transfer inspect --manifest session.json
quantem data-transfer copy --manifest session.json          # dry-run by default
quantem data-transfer copy --manifest session.json --execute
quantem data-transfer masters --manifest session.json
quantem data-transfer show4dstem --manifest session.json --gpus 0,1 --dtype u8 --bin 1
```

`copy` writes through `*.partial` files and refuses mismatched existing targets.
Default verification is by file size for speed; add `--hash sha256` at planning
time and `--verify hash` at inspect/copy time when the extra full-file reads are
worth the stronger guarantee.

`update` rescans the original source folder and appends new masters without
moving old target assignments:

```bash
quantem data-transfer update --manifest session.json
quantem data-transfer copy --manifest session.json --execute
```

`masters` prints only target masters whose full acquisition group is complete by
default. Use `--all-masters` when you want the planned target paths before the
copy has finished. `show4dstem` writes a live notebook from those ready target
masters. The command is GPU-friendly but still explicit: `--gpus 0,1` becomes
`load(masters, devices=[0, 1], ...)` in the generated notebook, `--dtype u8`
uses direct uint8 browse decoding for fast screening, and `--bin 1` keeps native
detector sampling.

Python equivalent:

```python
from quantem.widget import Show4DSTEM, load
from quantem.widget.io import read_data_transfer_manifest, target_masters

plan = read_data_transfer_manifest("session.json")
masters = [str(path) for path in target_masters(plan)]
data = load(masters, det_bin=1, dtype="u8", devices=[0, 1])
Show4DSTEM(data, page_budget="auto", page_device=[0, 1])
```

If all targets resolve to one physical disk, the CLI warns that cold load speed
is still disk-bound. Multiple GPUs help capacity, but fast cold flips need files
spread across independent disks.

For notebook sharing, keep the full-state `.ipynb` for collaborators and use
`quantem html --no-execute` for an interactive web artifact. Use `quantem github
--no-execute` only when you specifically need a non-interactive copy for
GitHub's native notebook renderer. GitHub blob/raw pages do not execute exported
HTML; serve HTML from GitHub Pages or another static host.

## Options

| Option | Effect |
|---|---|
| `--bin N` | detector mean-bin factor; Show4DSTEM defaults to 8, ShowPtycho and `data-transfer` default to 1 |
| `--dtype uint8/uint16` | Show4DSTEM HTML export/storage dtype; `uint8` is compact browse, `uint16` keeps the wider detector-count range |
| `--html` | 4D-STEM: write the offline-WebGPU HTML instead of a notebook |
| `--watch` | folder: write a live ShowFolder-watched notebook; Show2D/Show3D append new image files, Show4DSTEM opens lazy masters |
| `--gpus 0,1`, `--page-budget auto` | watched Show4DSTEM: pick CUDA cards and GPU-resident dataset cache policy |
| `--combined` | many masters -> one 5D HTML viewer (served locally) |
| `--out PATH` | output file or directory (default `~/Downloads`) |
| `--no-open` | write the file(s) without launching a browser or Jupyter |
| `--title`, `-v/--verbose` | page title; verbose progress |
| `--calibration`, `--semiangle`, `--scan-sampling`, `--voltage-kv` | ShowPtycho master generation geometry and calibration controls |
| `--drag-bf X` | ShowPtycho BF fraction or count; default `1.0` is full BF, `0.3` is 30 percent, values greater than 1 are explicit BF-pixel counts |

## Backends

The loader picks the backend automatically - **CUDA** on an NVIDIA box, **Apple
Metal (MPS)** on a Mac, **CPU** otherwise. No flag needed. On a MacBook:

```bash
quantem show4dstem ./masters/ --html --bin 8
```

loads on Metal, mean-bins the detector to fit the laptop, and writes a
double-clickable HTML in seconds. The detector is **mean-binned** (not summed) so
the bright field never clips at uint8, and binning happens at load so the full
multi-gigabyte stack never has to fit in memory. See [Load and I/O](api/io) for
the backend + binning details.
