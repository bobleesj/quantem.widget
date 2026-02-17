# quantem.widget

Interactive Jupyter widgets for electron microscopy visualization. Works with NumPy, CuPy, and PyTorch arrays.

## Installation

```bash
pip install quantem-widget
```

## Quick Start

### Show2D - Static Image Viewer

```python
import numpy as np
from quantem.widget import Show2D

# Single image
image = np.random.rand(256, 256)
Show2D(image)

# Multiple images (gallery mode)
images = [img1, img2, img3]
Show2D(images, labels=["A", "B", "C"])
```

### Show3D - Stack Viewer with Playback

```python
import numpy as np
from quantem.widget import Show3D

# 3D stack (z-stack, time series, defocus series)
stack = np.random.rand(100, 256, 256)
Show3D(stack, title="My Stack", fps=5)
```

### Show3DVolume - Orthogonal Slice Viewer

```python
import numpy as np
from quantem.widget import Show3DVolume

volume = np.random.rand(64, 64, 64).astype(np.float32)
Show3DVolume(volume, title="My Volume", cmap="viridis")
```

### Show4DSTEM - 4D-STEM Viewer

```python
import numpy as np
from quantem.widget import Show4DSTEM

data = np.random.rand(64, 64, 128, 128)
Show4DSTEM(data, pixel_size=2.39, k_pixel_size=0.46)
```

### Mark2D - Interactive Image Annotation

```python
import numpy as np
from quantem.widget import Mark2D

image = np.random.rand(256, 256)
w = Mark2D(image, max_points=5)
w

# After clicking, retrieve selected points
w.selected_points  # [{'row': 83, 'col': 120}, ...]
```

## Array Compatibility

All widgets accept NumPy arrays, PyTorch tensors, CuPy arrays, and quantem Dataset objects via duck typing. No manual conversion needed.

| Widget | NumPy | PyTorch | CuPy | quantem Dataset |
|--------|-------|---------|------|-----------------|
| Show2D | yes | yes | yes | `Dataset2d` |
| Show3D | yes | yes | yes | `Dataset3d` |
| Show3DVolume | yes | yes | yes | `Dataset3d` |
| Show4DSTEM | yes | yes | yes | `Dataset4dstem` |
| Mark2D | yes | yes | yes | `Dataset2d` |
| Align2D | yes | yes | yes | `Dataset2d` |

When a quantem Dataset is passed, metadata (title, pixel size, units) is extracted automatically. Explicit parameters always override auto-extracted values.

```python
import torch
Show2D(torch.randn(256, 256))  # PyTorch tensor

from quantem.core.datastructures import Dataset3d
dataset = Dataset3d.from_array(stack, name="Focal Series", sampling=(1.0, 0.25, 0.25), units=("nm", "Å", "Å"))
Show3D(dataset)  # title and pixel_size extracted from dataset
```

## Documentation with Interactive Widgets

The Sphinx documentation renders anywidget-based widgets interactively in the browser — users can zoom, pan, change colormaps, toggle FFT, etc. directly on the docs page without a running kernel.

### How it works

1. Notebooks are executed locally in JupyterLab, which saves **widget state** (JS bundle + binary image data) into the notebook metadata
2. nbsphinx renders the pre-saved widget state as interactive HTML using `@jupyter-widgets/html-manager`
3. GitHub Actions deploys to GitHub Pages on every push to `main`

### Adding or updating docs notebooks

```bash
# 1. Run the notebook in JupyterLab (widget state is saved on File > Save)
jupyter lab docs/examples/show2d/show2d_simple.ipynb

# 2. Verify widget state is embedded
python -c "import json; nb=json.load(open('docs/examples/show2d/show2d_simple.ipynb')); print('Widget state:', bool(nb.get('metadata',{}).get('widgets',{})))"

# 3. Commit the notebook (with widget state)
git add docs/examples/show2d/show2d_simple.ipynb
```

Docs example notebooks in `docs/examples/` can be either real files with saved widget state, or symlinks to `notebooks/` (which must also have widget state saved).

### Limitations

- **JS-only interactivity works**: zoom, pan, colormap, log scale, FFT, auto-contrast, histogram
- **Python features don't work**: frame navigation (Show3D), export (GIF/ZIP), `set_image()`, trait observers
- Each widget embeds its full JS bundle (~600 KB) + image data, so pages can be several MB

## CI/CD

Two GitHub Actions workflows automate publishing and documentation:

### Docs deployment (`.github/workflows/docs.yml`)

Deploys Sphinx documentation to GitHub Pages on every push to `main`.

- **Trigger**: push to `main` branch (or manual `workflow_dispatch`)
- **What it does**: installs Node.js + Python deps, runs `npm install && pip install .` (hatch-jupyter-builder builds JS automatically), then `sphinx-build`
- **Prerequisite**: enable GitHub Pages in repo Settings → Pages → Source: **GitHub Actions**
- **URL**: https://bobleesj.github.io/quantem.widget/
- Notebooks are rendered with `nbsphinx_execute = "never"` — pre-saved outputs (including widget state) are used as-is, no execution on CI

### TestPyPI publishing (`.github/workflows/publish.yml`)

Publishes to TestPyPI when a version tag is pushed.

```bash
# 1. Bump version in pyproject.toml
# 2. Commit, tag, and push
git tag v0.0.5
git push origin main && git push origin v0.0.5
```

GitHub Actions compiles React/TypeScript, builds the Python wheel, and uploads to TestPyPI. Note: TestPyPI does not allow re-uploading the same version — always bump the version before tagging.

### Verify TestPyPI Release

After CI finishes, verify the published package in a clean environment:

```bash
./scripts/verify_testpypi.sh 0.0.5
```

This creates a fresh conda env, installs from TestPyPI, verifies all widget imports and JS bundles, then opens JupyterLab with a test notebook for visual inspection. When done, press Ctrl+C and clean up:

```bash
conda env remove -n test-widget-env -y
```

### TestPyPI Trusted Publisher Setup

1. Go to https://test.pypi.org/manage/account/publishing/
2. Add a new "pending publisher" with:
   - **PyPI project name**: `quantem-widget`
   - **Owner**: `bobleesj`
   - **Repository**: `quantem.widget`
   - **Workflow name**: `publish.yml`
   - **Environment**: leave blank

## Development

```bash
git clone https://github.com/bobleesj/quantem.widget.git
cd quantem.widget
npm install
npm run build
pip install -e .
```

## License

MIT
