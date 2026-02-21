# changelog

## Unreleased

### New widgets
- **Merge4DSTEM** — stack multiple 4D-STEM datasets along a time axis with GPU-accelerated merge, detector binning, source preview, and Zarr export
- **Show1D** — interactive 1D viewer for spectra, profiles, and time series with multi-trace overlay, calibrated axes, log scale, and figure export

### Show2D
- file loaders: `from_png`, `from_tiff`, `from_emd`, `from_path`, `from_folder(file_type=...)`
- stack reduction modes (`first`, `index`, `mean`, `max`, `sum`) for collapsing stacks to 2D

### Show3D
- multi-ROI: place, resize, duplicate, and delete multiple ROIs with per-ROI color and stats
- one-click export bundle (`.zip` with PNG + ROI timeseries CSV + state JSON)
- file loaders: `from_emd`, `from_tiff`, `from_png`, `from_folder(file_type=...)`

### Show3D, Show4D, Show4DSTEM
- quick-view presets: save/recall 3 display configurations via UI buttons or keyboard (`1/2/3`, `Shift+1/2/3`)

## v0.0.8 (2026-02-20)

## v0.0.7 (2026-02-19)

### Bin
- `Bin` widget for calibration-aware 4D-STEM binning with BF/ADF QC previews
- export: `save_image()` (single panel or grid PNG/PDF), `save_zip()` (all panels + metadata), `save_gif()` (original vs binned comparison)

### Batch preprocessing
- preset-driven folder batch runner: `python -m quantem.widget.bin_batch`
- supports `.npy` and `.npz`, plus streamed 5D `.npy` processing for `time_axis=0`
- torch device can be selected via preset or `--device`

## v0.0.6 (2026-02-19)

### Show4DSTEM
- `save_image()` for programmatic export to PNG/PDF with optional metadata sidecar JSON
- export views: `diffraction`, `virtual`, `fft`, `all`
- `save_image()` supports temporary overrides (`position=(row, col)`, `frame_idx`) with automatic state restoration
- exported images now match interactive display settings (colormaps, scale modes, percentile clipping)
- `state_dict()` now includes scan position (`pos_row`, `pos_col`) for exact state restore
- `ArrowUp` / `ArrowDown` scan-row navigation (with `Shift` step), `Esc` to release focus

### Show4D
- `ArrowUp` / `ArrowDown` row navigation (with `Shift` step), `Esc` to release focus

### All profile widgets (Show2D, Show3D, Show4D, Show4DSTEM, Mark2D)
- line-profile endpoints are now draggable after placement
- dragging the line body translates both endpoints together (preserves line shape)
- hover cursor changes near profile endpoints/line

## v0.0.5 (2026-02-18)

### All profile widgets (Show2D, Show3D, Show4D, Show4DSTEM, Mark2D)
- `set_profile` now takes two `(row, col)` tuples: `set_profile((row0, col0), (row1, col1))`

### Show4DSTEM
- 5D time-series/tilt-series support: accepts `(n_frames, scan_rows, scan_cols, det_rows, det_cols)` arrays with frame slider, play/pause controls, and `frame_dim_label` (e.g. `"Tilt"`, `"Time"`, `"Focus"`)
- frame playback: fps slider, loop, bounce, reverse, transport buttons, `[` / `]` keyboard shortcuts
- grab-and-drag ROI: clicking inside the ROI drags with an offset instead of teleporting the center
- theme-aware ROI colors: darker green overlays in light theme for better visibility
- fixed resize handle hit area: was ~70px due to pixel mismatch, now correctly sized

## v0.0.4 (2026-02-16)

### Show2D
- ROI with inner/outer resize handles and cursor feedback
- line profile with hover value readout and resizable height
- live FFT filtering with mask painting
- colorbar overlay, export dropdown, clipboard copy
- auto-contrast with percentile clipping
- gallery mode with keyboard navigation

### Show3D
- ROI and lens controls moved to toggle panels
- line profile with hover value readout and resizable height
- live FFT filtering with mask painting
- FFT panel aligned with main canvas
- colorbar overlay, export dropdown (figure, PNG, GIF, ZIP), clipboard copy
- auto-contrast with percentile clipping

### Show3DVolume
- three orthogonal slice panels (XY, XZ, YZ) with synchronized cursors
- export figure with all three slices
- colorbar overlay, GIF/ZIP export

### Show4DSTEM
- virtual imaging with BF, ABF, ADF, custom ROI presets
- diffraction pattern viewer with annular ROI
- export dropdown and clipboard copy for both panels
- colorbar overlay

### Show4D
- dual navigation/signal panel layout
- ROI masking on navigation image
- path animation with GIF/ZIP export
- export dropdown and clipboard copy

### Align2D
- two-image overlay with opacity blending
- FFT-based auto-alignment via phase correlation
- difference view mode
- export figure for both panels

### Mark2D
- interactive point picker with click-to-place
- ROI support (rectangle, circle, annulus)
- snap-to-peak for precise atomic column positioning
- undo, colorbar overlay, export figure with markers

### Shared
- light/dark theme detection across all widgets
- colormap LUTs (inferno, viridis, plasma, magma, hot, gray)
- WebGPU FFT with CPU fallback
- HiDPI scale bar with unit conversion
- publication figure export via `exportFigure`
- state persistence (`state_dict`, `save`, `load_state_dict`, `state` param)
- `set_image` for replacing data without recreating widget
- NumPy, PyTorch, CuPy array support
- quantem Dataset metadata auto-extraction
- (row, col) coordinate convention

## v0.0.3 (2025-12-01)

- initial release with Show2D, Show3D, Show3DVolume, Show4DSTEM
- demo notebooks
- sphinx docs with pydata theme
