"""
clicker: Interactive point picker widget for 2D images.

Click on an image to select points (atom positions, features, etc.).
Shows raw and normalized intensity values on hover.
Supports gallery mode with multiple images.
"""

import pathlib
from typing import List, Optional

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.widget.show2d import _resize_image


class Clicker(anywidget.AnyWidget):
    """
    Interactive point picker for 2D images.

    Click on an image to select points. Useful for picking atom positions,
    selecting features, or defining lattice vectors.

    Parameters
    ----------
    data : array_like
        2D array (H, W), 3D array (N, H, W), or list of 2D arrays.
    scale : float, default 1.0
        Display scale factor.
    dot_size : int, default 12
        Size of point markers in pixels.
    max_points : int, default 3
        Maximum number of selected points per image.
    ncols : int, default 3
        Number of columns in gallery grid.
    labels : list of str, optional
        Per-image labels for gallery mode.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Clicker
    >>>
    >>> # Single image
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> w = Clicker(img, scale=1.5, max_points=5)
    >>> w  # display in notebook
    >>> points = w.selected_points  # [{"x": 100, "y": 200}, ...]
    >>>
    >>> # Gallery mode
    >>> imgs = [np.random.rand(64, 64) for _ in range(3)]
    >>> w = Clicker(imgs, ncols=3, labels=["A", "B", "C"])
    >>> w.selected_points  # [[{...}, ...], [{...}, ...], [{...}, ...]]
    """

    _esm = pathlib.Path(__file__).parent / "static" / "clicker.js"

    # Image data (gallery-capable, matching Show2D pattern)
    n_images = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(0).tag(sync=True)
    height = traitlets.Int(0).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    img_min = traitlets.List(traitlets.Float()).tag(sync=True)
    img_max = traitlets.List(traitlets.Float()).tag(sync=True)

    # Gallery controls
    selected_idx = traitlets.Int(0).tag(sync=True)
    ncols = traitlets.Int(3).tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)

    # UI controls
    scale = traitlets.Float(1.0).tag(sync=True)
    selected_points = traitlets.List().tag(sync=True)
    dot_size = traitlets.Int(12).tag(sync=True)
    max_points = traitlets.Int(10).tag(sync=True)

    def __init__(
        self,
        data,
        scale: float = 1.0,
        dot_size: int = 12,
        max_points: int = 10,
        ncols: int = 3,
        labels: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale = scale
        self.dot_size = dot_size
        self.max_points = max_points
        self.ncols = ncols
        self._set_data(data, labels)

    def _set_data(self, data, labels=None):
        if isinstance(data, list):
            images = [to_numpy(d) for d in data]
            for img in images:
                if img.ndim != 2:
                    raise ValueError("Each image in the list must be 2D (H, W).")
            shapes = [img.shape for img in images]
            if len(set(shapes)) > 1:
                max_h = max(s[0] for s in shapes)
                max_w = max(s[1] for s in shapes)
                images = [_resize_image(img, max_h, max_w) for img in images]
            arr = np.stack(images).astype(np.float32)
        else:
            arr = to_numpy(data).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 3:
                pass  # (N, H, W) gallery
            else:
                raise ValueError("Expected 2D (H,W) or 3D (N,H,W) array, or list of 2D arrays.")

        self._data = arr
        n, h, w = arr.shape
        self.n_images = n
        self.height = h
        self.width = w

        # Per-image min/max
        mins, maxs = [], []
        for i in range(n):
            mins.append(float(arr[i].min()))
            maxs.append(float(arr[i].max()))
        self.img_min = mins
        self.img_max = maxs

        # Labels
        if labels is not None:
            self.labels = list(labels)
        else:
            self.labels = [f"Image {i + 1}" for i in range(n)]

        # Frame bytes (raw float32, all images concatenated)
        self.frame_bytes = arr.tobytes()

        # Reset points
        if n == 1:
            self.selected_points = []
        else:
            self.selected_points = [[] for _ in range(n)]

        self.selected_idx = 0

    def set_image(self, data, labels=None):
        """
        Replace image(s). Accepts 2D, 3D, or list of 2D arrays.

        Parameters
        ----------
        data : array_like
            2D array (H, W), 3D array (N, H, W), or list of 2D arrays.
        labels : list of str, optional
            Per-image labels for gallery mode.
        """
        self._set_data(data, labels)
