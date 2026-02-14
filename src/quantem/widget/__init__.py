"""
quantem.widget: Interactive Jupyter widgets using anywidget + React.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("quantem-widget")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from quantem.widget.clicker import Clicker
from quantem.widget.show2d import Show2D
from quantem.widget.show3d import Show3D
from quantem.widget.show3dvolume import Show3DVolume
from quantem.widget.show4dstem import Show4DSTEM

__all__ = ["Clicker", "Show2D", "Show3D", "Show3DVolume", "Show4DSTEM"]
