# changelog

## Unreleased

### Merge4DSTEM
- added `Merge4DSTEM` widget for stacking multiple 4D-STEM datasets along a time axis
- torch-backed merge via `torch.stack` on GPU (MPS/CUDA/CPU auto-detect)
- accepts `Dataset4dstem`, file paths (Zarr zip), or raw arrays as input
- strict shape validation (hard fail) with calibration mismatch warnings
- `.merge()` produces 5D `(n_sources, scan_r, scan_c, det_r, det_c)` array
- `.result` property returns `Dataset4dstem`, `.result_array` returns raw numpy
- `.save_result(path)` saves merged data as Zarr zip archive
- `.to_show4dstem()` opens merged result in `Show4DSTEM`
- preview canvas shows mean diffraction pattern from first source
- interactive React UI with source table, merge button, status bar
- state persistence: `state_dict()`, `save()`, `load_state_dict()`, `summary()`
- tool lock/hide support via `disabled_tools`/`hidden_tools`

### Show1D
- added `Show1D` widget for interactive 1D data viewing (spectra, profiles, time series)
- single or multi-trace overlay with distinct colors and legend
- calibrated X/Y axes with labels and units
- log scale, grid lines, interactive zoom/pan, snap-to-nearest crosshair
- publication-quality figure export
- mutation methods: `set_data()`, `add_trace()`, `remove_trace()`, `clear()`
- state persistence: `state_dict()`, `save()`, `load_state_dict()`, `summary()`

### Show2D
- added explicit file/folder loaders for beginner workflows: `from_png`, `from_tiff`, `from_emd`, `from_png_folder`, `from_tiff_folder`, `from_emd_folder`
- added generic explicit dispatcher `from_path` / `from_folder(file_type=...)`
- EMD loaders now support `dataset_path` override for deterministic dataset selection
- added optional stack reduction modes (`first`, `index`, `mean`, `max`, `sum`) to collapse stacks into one 2D image when needed

### Show3D
- ported ROI model to multi-ROI state (`roi_list`, `roi_selected_idx`, `roi_stats`)
- added Show2D-style ROI UX: per-ROI color, focus highlight, selected ROI controls
- ROI resize handles are now edge-hover/drag driven (no always-visible circular handles)
- added ROI workflow polish: click empty canvas to add ROI at cursor, `Delete` to remove selected, duplicate selected ROI
- added ROI legend actions: show/hide, lock/unlock, and live per-ROI stats
- added adjustable ROI focus dim strength (`roi_focus_dim`)
- added one-click export bundle (`show3d_bundle.zip`) with current-frame PNG + `roi_timeseries.csv` + `state.json`
- added playback loop presets (`Full`, `Selection`, `Bookmarks`) and clearer scrub labeling
- added first-use ROI resize hint that auto-hides
- added explicit Show3D file loaders: `from_emd`, `from_tiff`, `from_png`, `from_emd_folder`, `from_tiff_folder`, `from_png_folder`; `from_folder(..., file_type=...)` now requires explicit type and EMD accepts `dataset_path`
- added persistent quick-view presets with 3 slots (save/load from UI buttons or keyboard `1/2/3`, `Shift+1/2/3`)
- presets now sync via `view_presets_json` and persist across `state_dict()`, `save()/load_state_dict()`, and `state=` init
- added preset reset action in the UI and saved-slot indicator text (`Saved: ...`)
- added Python quick-view preset helpers: `list_view_preset_slots()`, `get_view_preset()`, `set_view_preset()`, `clear_view_preset()`, `reset_view_presets()`

### Show4D
- added persistent quick-view presets with 3 slots (save/load from UI buttons or keyboard `1/2/3`, `Shift+1/2/3`)
- presets now sync via `view_presets_json` and persist across `state_dict()`, `save()/load_state_dict()`, and `state=` init
- added preset UX polish: `Clear 1/2/3` actions and transient saved/loaded status text
- added preset reset action in the UI (`Reset`)
- added Python quick-view preset helpers: `list_view_preset_slots()`, `get_view_preset()`, `set_view_preset()`, `clear_view_preset()`, `reset_view_presets()`

### Show4DSTEM
- added persistent quick-view presets with 3 slots (save/load from UI buttons or keyboard `1/2/3`, `Shift+1/2/3`)
- presets now sync via `view_presets_json` and persist across `state_dict()`, `save()/load_state_dict()`, and `state=` init
- added preset reset action in the UI and saved-slot indicator text (`Saved: ...`)
- added Python quick-view preset helpers: `list_view_preset_slots()`, `get_view_preset()`, `set_view_preset()`, `clear_view_preset()`, `reset_view_presets()`

### Docs
- updated `docs/widgets/show3d.rst` to document multi-ROI workflow and versioned state envelope saves
- added Show3D loading notebooks for PNG folders, TIFF, EMD, plus a format chooser hub (`show3d_load_hub`, `show3d_load_png_folder`, `show3d_load_tiff`, `show3d_load_emd`)
- added Show2D loading notebooks for PNG folders, TIFF, EMD, plus a format chooser hub (`show2d_load_hub`, `show2d_load_png_folder`, `show2d_load_tiff`, `show2d_load_emd`)
- updated state persistence docs with a quick-view preset example and all three restore paths (`load_state_dict`, `save/state file`, `state=` dict)

### Tests
- added focused smoke tests for Show3D/Show4D/Show4DSTEM quick-view preset save/load via keyboard shortcuts
- hardened Jupyter smoke fixture with startup retries, dynamic open-port selection, longer startup timeout, and failure log tail output

## v0.0.8 (2026-02-20)

## v0.0.7 (2026-02-19)

### Bin
- added `Bin` widget for calibration-aware 4D-STEM binning with BF/ADF QC previews
- strict torch-only compute mode enforced (`compute_backend="torch"`, `torch_enabled=True`)
- added export helpers:
  - `save_image()` (single panel or grid PNG/PDF + optional metadata JSON)
  - `save_zip()` (all preview panels + metadata, optional `.npy` arrays)
  - `save_gif()` (original vs binned BF/ADF comparison)
- added state payload hardening for torch-only restore semantics

### Batch preprocessing
- added preset-driven folder batch runner: `python -m quantem.widget.bin_batch`
- processes files one-at-a-time for large datasets/time series
- supports `.npy` and `.npz`, plus streamed 5D `.npy` processing for `time_axis=0`
- torch device can be selected via preset or `--device`

### Docs and examples
- added `docs/widgets/bin.rst` and `docs/api/bin.rst`
- added Bin example notebooks:
  - `notebooks/bin/bin_simple.ipynb`
  - `notebooks/bin/bin_all_features.ipynb`
  - mirrored docs examples under `docs/examples/bin/`
- wired Bin into docs toctrees and landing page references

### Tests
- added Bin unit tests for shape/calibration/state/export behavior
- extended E2E smoke coverage with Bin interactions/screenshots
- added dedicated screenshot capture script: `tests/capture_bin.py`

## v0.0.6 (2026-02-19)

### Show4DSTEM
- added `save_image()` for programmatic export to PNG/PDF with optional metadata sidecar JSON
- export views are strict: `diffraction`, `virtual`, `fft`, `all` (no aliases)
- `save_image()` supports temporary export overrides (`position=(row, col)`, `frame_idx`) with automatic state restoration
- synced UI display settings to Python traits for export parity with interactive state: colormaps, scale modes, power exponents, percentile clipping, FFT auto mode, FFT visibility, and DP colorbar toggle
- `state_dict()` now includes scan position (`pos_row`, `pos_col`) so saved state restores exact location
- focus-scoped keyboard handling (`tabIndex` + `onKeyDown`) instead of a global `window` listener
- `ArrowUp` / `ArrowDown` scan-row navigation (with `Shift` step)
- `Esc` releases widget keyboard focus

### Show4D
- focus-scoped keyboard handling (`tabIndex` + `onKeyDown`) instead of a global `window` listener
- `ArrowUp` / `ArrowDown` row navigation (with `Shift` step)
- `Esc` releases widget keyboard focus

### All profile widgets (Show2D, Show3D, Show4D, Show4DSTEM, Mark2D)
- line-profile endpoints are now draggable after placement
- dragging the line body translates both endpoints together (preserves line shape)
- Show4DSTEM applies this to both DP and VI profile lines
- hover now changes cursor near profile endpoints/line (`grab`) before dragging

### Notebooks
- `show4dstem_simple.ipynb`: MPS-safe fallback for `torch.poisson` (CPU fallback on Apple Silicon)

## v0.0.5 (2026-02-18)

### All profile widgets (Show2D, Show3D, Show4D, Show4DSTEM, Mark2D)
- `set_profile` signature changed to `set_profile((row0, col0), (row1, col1))` — takes two `(row, col)` tuples instead of four separate floats

### Show4DSTEM
- 5D time-series/tilt-series support: accepts `(n_frames, scan_rows, scan_cols, det_rows, det_cols)` arrays with frame slider, play/pause controls, and `frame_dim_label` parameter (e.g. `"Tilt"`, `"Time"`, `"Focus"`)
- Show3D-style frame playback: fps slider, loop, bounce (ping-pong), reverse, transport buttons (rewind/play/forward/stop)
- keyboard shortcuts `[` / `]` for prev/next frame navigation
- grab-and-drag ROI: clicking inside the detector ROI now drags it with an offset instead of teleporting the center, making it much easier to reposition
- theme-aware ROI colors: green overlays use darker shades in light theme for better visibility
- fixed resize handle hit area: handle was ~70px due to image-pixel vs screen-pixel mismatch, now correctly sized to match the visual handle dot
- same grab-and-drag fix applied to VI ROI on the virtual image panel

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
- tighter controls row fitting default canvas width

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

### shared
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
