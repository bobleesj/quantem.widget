# changelog

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
