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

### Clicker - Interactive Point Picker

```python
import numpy as np
from quantem.widget import Clicker

image = np.random.rand(256, 256)
w = Clicker(image, max_points=5)
w

# After clicking, retrieve selected points
w.selected_points  # [{'x': 120.5, 'y': 83.2}, ...]
```

## Array Compatibility

All widgets accept NumPy arrays, PyTorch tensors, CuPy arrays, and quantem Dataset objects via duck typing. No manual conversion needed.

| Widget | NumPy | PyTorch | CuPy | quantem Dataset |
|--------|-------|---------|------|-----------------|
| Show2D | yes | yes | yes | `Dataset2d` |
| Show3D | yes | yes | yes | `Dataset3d` |
| Show3DVolume | yes | yes | yes | `Dataset3d` |
| Show4DSTEM | yes | yes | yes | `Dataset4dstem` |
| Clicker | yes | yes | yes | `Dataset2d` |
| Align2D | yes | yes | yes | `Dataset2d` |

When a quantem Dataset is passed, metadata (title, pixel size, units) is extracted automatically. Explicit parameters always override auto-extracted values.

```python
import torch
Show2D(torch.randn(256, 256))  # PyTorch tensor

from quantem.core.datastructures import Dataset3d
dataset = Dataset3d.from_array(stack, name="Focal Series", sampling=(1.0, 0.25, 0.25), units=("nm", "Å", "Å"))
Show3D(dataset)  # title and pixel_size extracted from dataset
```

## Publishing

Push a tag to publish to TestPyPI via GitHub Actions:

```bash
git tag v0.0.1
git push origin v0.0.1
```

GitHub Actions automatically compiles the React/TypeScript, builds the Python wheel, and uploads to TestPyPI.

### Verify TestPyPI Release

After CI finishes, verify the published package in a clean environment:

```bash
./scripts/verify_testpypi.sh 0.0.3
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
