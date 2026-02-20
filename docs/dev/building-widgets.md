# Building Widgets


## Anatomy of a widget file

Every widget follows the same structure. Here's `Show4DSTEM` — the most complex
widget (~4200 lines Python, ~4100 lines TSX). Simpler widgets have the same
sections, just shorter.


### Python file (7 sections, top-to-bottom)

```
┌─────────────────────────────────────────────────┐
│  1. Imports + constants                         │
│  2. Trait declarations (synced to JS)           │
│  3. __init__ (data loading, setup)              │
│  4. Observers (react to trait changes)          │
│  5. Public API (set_image, ROI presets, etc.)   │
│  6. State protocol (state_dict, save, summary)  │
│  7. Internal methods (_get_frame, _render, etc.)│
└─────────────────────────────────────────────────┘
```

**1. Imports + constants** — `anywidget`, `traitlets`, `numpy`, `torch`, shared
modules (`to_numpy`, `save_state_file`, tool parity).

**2. Trait declarations** — Every synced property, grouped by purpose:

```python
class Show4DSTEM(anywidget.AnyWidget):
    # Position & shape
    pos_row = traitlets.Int(0).tag(sync=True)
    pos_col = traitlets.Int(0).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)

    # Display settings
    dp_colormap = traitlets.Unicode("inferno").tag(sync=True)

    # ROI
    roi_mode = traitlets.Unicode("off").tag(sync=True)
    ...
```

**3. `__init__`** — Always follows this pattern:

```python
def __init__(self, data, ..., state=None, **kwargs):
    super().__init__(**kwargs)          # 1. Initialize anywidget
    self.widget_version = resolve_widget_version()
    data_np = to_numpy(data)           # 2. Convert input
    self._device = torch.device(...)   # 3. Move to GPU
    self._data = torch.from_numpy(...).to(self._device)
    self.shape_rows = ...              # 4. Set traits from data
    self.frame_bytes = ...             # 5. Send first frame to JS
    self.observe(self._update_frame,   # 6. Register observers
                 names=["pos_row", "pos_col"])
    if state is not None:              # 7. Restore saved state (last!)
        self.load_state_dict(state)
```

**4. Observers** — Registered in `__init__` via `self.observe()`. When JS changes
a trait, the callback runs in Python.

**5. Public API** — `set_image()`, `roi_circle()`, `set_path()`, `play()`, etc.
Return `self` for chaining.

**6. State protocol** — `state_dict()`, `save()`, `load_state_dict()`, `summary()`,
`__repr__()`. Same in every widget.

**7. Internal methods** — Prefixed with `_`. Data extraction, rendering, export.


### TypeScript file (7 sections, top-to-bottom)

```
┌─────────────────────────────────────────────────┐
│  1. Imports + constants + styles                │
│  2. Utility functions + helper components       │
│  3. Model state hooks (useModelState)           │
│  4. Local state + refs                          │
│  5. Effects (useEffect) — rendering + setup     │
│  6. Event handlers (mouse, keyboard, export)    │
│  7. JSX return (the actual UI layout)           │
└─────────────────────────────────────────────────┘
```

**1. Imports + constants** — React, MUI, shared modules. Constants defined
*outside* the component (won't recreate on render):

```tsx
const MIN_ZOOM = 0.5;
const CANVAS_SIZE = 360;
const compactButton = { fontSize: 10, minWidth: 0, px: "6px", py: "2px" };
```

**2. Utilities + helpers** — `Histogram`, `InfoTooltip`, `KeyboardShortcuts`,
canvas drawing functions (`drawRoiOverlayHiDPI`, `drawViPositionMarker`).

**3. Model state hooks** — Reads synced traits from Python:

```tsx
function Show4DSTEM({ model }: { model: AnyModel }) {
  const [posRow, setPosRow] = useModelState<number>("pos_row");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  // ... 40+ more
```

**4. Local state + refs** — UI-only state (drag tracking, cursor position,
canvas dimensions) and refs to canvas DOM elements.

**5. Effects** — The rendering engine. Each `useEffect` watches dependencies and
re-runs when they change:

```tsx
React.useEffect(() => {
  const raw = extractFloat32(frameBytes);
  if (!raw) return;
  // ... apply colormap, draw to canvas
}, [frameBytes, dpColormap, zoom, panX, panY]);
```

**6. Event handlers** — Mouse, keyboard, export. Each panel has its own set.

**7. JSX return** — The UI layout, built from nested MUI components and canvas
stacks.


### How the sections connect

When a user clicks on the virtual image:

```
JS: handleViMouseDown          → setPosRow(5), setPosCol(10)
                                         ↓
Python: self.observe() fires   → _update_frame(pos_row=5, pos_col=10)
                                         ↓
Python: _get_frame()           → self.frame_bytes = tensor[5, 10].tobytes()
                                         ↓
JS: useEffect [frameBytes]     → extractFloat32 → applyColormap → drawImage
                                         ↓
Browser: canvas shows new diffraction pattern
```

Every widget follows this cycle: **JS event → trait update → Python observer →
new bytes → JS effect → canvas render**.


## Patterns

### Compound traits for batched updates

When JS needs to send row and column at once, use a compound `List` trait:

```python
roi_center = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)
self.observe(self._on_roi_center_change, names=["roi_center"])
```

Now Python fires one observer per drag event, not two.

### Binary data: always raw float32

Image data goes through the `Bytes` trait as raw float32 — no JSON, no base64:

```python
# Python side: tensor → raw bytes
self.frame_bytes = frame.cpu().numpy().astype(np.float32).tobytes()
```

```tsx
// JS side: DataView → Float32Array
import { extractFloat32 } from "../format";
const raw = extractFloat32(frameBytes);  // handles alignment + null check
```

### Canvas rendering: layered architecture

Each widget stacks two or more `<canvas>` elements:
1. **Data canvas** — colormapped image at native resolution (e.g., 128×128 px)
2. **UI canvas** — crosshair, ROI shapes, scale bar, text overlays (drawn at
   device pixel ratio for crisp lines on HiDPI displays)

Some widgets add extra layers: Show3D has an overlay + lens canvas,
Show4DSTEM has separate data/overlay/UI sets per panel (DP, virtual image, FFT).

### Coordinate convention: (row, col), never (x, y)

All user-facing coordinates use `(row, col)`. This matches electron microscopy
convention and NumPy array indexing (`array[row, col]`).

- **Python API**: `roi_row`/`roi_col`, `{"row": r, "col": c}`, `(row, col)` tuples
- **JS/TS types**: `Point = { row, col }`, `ROI = { row, col }`
- **Display**: `(row, col)` in stats bars, tooltips, readouts

Internal drawing code (canvas pixels, DOM events, pan/zoom) can use `x`/`y` —
those are screen coordinates. But anything exposed to the user must use `(row, col)`.

```python
# Right
pos_row = traitlets.Int(0).tag(sync=True)
pos_col = traitlets.Int(0).tag(sync=True)

# Wrong
pos_x = traitlets.Int(0).tag(sync=True)
pos_y = traitlets.Int(0).tag(sync=True)
```

### Array compatibility

All widgets accept NumPy arrays, PyTorch tensors, CuPy arrays, and any
`np.asarray()`-compatible object via `array_utils.to_numpy()`.

Widgets auto-extract metadata from `quantem` Dataset objects using `hasattr`
checks. Explicit parameters always override:

```python
w = Show2D(numpy_array)
w = Show2D(torch_tensor)
w = Show2D(quantem_dataset)            # auto-extracts title, pixel_size
w = Show2D(quantem_dataset, title="Override")  # explicit wins
```

### Replacing data with `set_image()`

Every widget implements `set_image()` to swap data while preserving display settings:

```python
w = Show2D(image_a, cmap="viridis", log_scale=True)
w.set_image(image_b)  # keeps cmap and log_scale
```

### Scale bar: HiDPI rendering

Drawn on the **UI canvas** (not data canvas) for crisp text at any zoom:

- Device pixel ratio: `width={cssW * DPR}`, `style={{ width: cssW }}`
- Fixed CSS sizes: 60 px bar, 5 px thick, 16 px font
- White with drop shadow for contrast on any colormap
- Shared via `drawScaleBarHiDPI()` in `js/scalebar.ts`


## Adding a new widget

Say you want to add `ShowPDF`. Start from the UI, not the data.

### 1. Design first

Before writing code, figure out:
- What does the user see? (canvases, sliders, buttons, panels)
- What can the user click or drag?
- What happens when they do?

Pick the closest existing widget and copy it wholesale:
- `Show1D` (~450 lines Python, ~1200 lines TSX) — simplest
- `Show2D` (~1100 lines Python, ~3000 lines TSX) — good starting point
- `Show4DSTEM` (~4200 lines Python, ~4100 lines TSX) — most complex

> **Tip:** Copy the model widget's code into an LLM (Claude, ChatGPT), describe
> what you want to change, and let it modify the layout and rendering for you.

### 2. Create files

1. **`src/quantem/widget/showpdf.py`** — Python backend:
   - Extend `anywidget.AnyWidget`
   - Set `_esm` and `_css` pointing to `static/showpdf.js` and `static/showpdf.css`
   - Define traitlets with `.tag(sync=True)`

2. **`js/showpdf/index.tsx`** — JavaScript frontend:
   - `useModelState()` to read/write traitlets
   - Render to `<canvas>` elements

### 3. Register the build

3. **`package.json`** — add `js/showpdf/index.tsx` to the build entries
4. **`pyproject.toml`** — add `"src/quantem/widget/static/showpdf.js"` to `ensured-targets`
5. **`src/quantem/widget/__init__.py`** — export `ShowPDF`

### 4. Add tests and docs

6. **`tests/test_widget_showpdf.py`** — trait initialization, state persistence, data loading
7. **`docs/widgets/showpdf.rst`** and a notebook in **`docs/examples/showpdf/`**

### 5. Verify

```bash
npm run build
pip install -e .
python -c "from quantem.widget import ShowPDF; print('OK')"
pytest tests/test_widget_showpdf.py
```


## Design philosophy

**Send only what's on screen.** A 4D-STEM dataset is 4 GB. A single 128×128
diffraction pattern is 64 KB. Send 64 KB, not 4 GB.

**Python computes, JavaScript renders.** Python owns slicing tensors, computing
virtual images, running GPU kernels. JavaScript owns colormaps, canvas drawing,
zoom/pan, crosshairs, tooltips, and GPU-accelerated FFT (via WebGPU). The
boundary is the `Bytes` trait.

**Keep interactive feedback in JS.** Mouse tracking, hover coordinates, zoom/pan
run at 60fps with zero round-trips. Only cross to Python when the user does
something that requires recomputation.

**Test with real data from day one.** A 10×10×10×10 test array won't reveal
problems a 256×256×128×128 dataset will.


## Common mistakes

**Don't over-engineer the file structure.** Each widget is one Python file and
one TSX folder. `show4dstem.py` is ~4200 lines and that's fine. Don't split it
into `show4dstem_traits.py` + `show4dstem_compute.py` + `show4dstem_observers.py`.
One file means you can read top-to-bottom and feed the entire widget to an LLM.

**Don't create base classes.** No `BaseWidget` or `GPUWidget`. Each widget stands
alone. Share via utility functions (`to_numpy()`), not inheritance. When in doubt,
duplicate.

**Don't over-split JavaScript.** Shared JS modules (`colormaps.ts`, `theme.ts`,
etc.) exist because every widget needs them. Don't create a new shared module
unless multiple widgets already use the same logic. Discuss with @bobleesj first.

**Don't send full datasets to the browser.** Assigning the entire tensor to a
`Bytes` trait will freeze or crash the browser. Only send the current view.

**Don't reimplement computation.** Check if it already exists in `quantem` core
before implementing CoM, FFT, alignment, or masking.
