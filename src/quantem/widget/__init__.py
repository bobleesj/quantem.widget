"""
quantem.widget: Interactive Jupyter widgets using anywidget + React.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("quantem-widget")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from quantem.widget.mark2d import Mark2D
from quantem.widget.edit2d import Edit2D
from quantem.widget.show2d import Show2D
from quantem.widget.show3d import Show3D
from quantem.widget.show3dvolume import Show3DVolume
from quantem.widget.show4d import Show4D
from quantem.widget.show4dstem import Show4DSTEM
from quantem.widget.align2d import Align2D
from quantem.widget.showcomplex import ShowComplex2D

__all__ = ["Align2D", "Edit2D", "Mark2D", "Show2D", "Show3D", "Show3DVolume", "Show4D", "Show4DSTEM", "ShowComplex2D"]
