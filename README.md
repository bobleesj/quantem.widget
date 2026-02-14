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

## Publishing

Push a tag to publish to TestPyPI via GitHub Actions:

```bash
git tag v0.0.1
git push origin v0.0.1
```

GitHub Actions automatically compiles the React/TypeScript, builds the Python wheel, and uploads to TestPyPI.

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
