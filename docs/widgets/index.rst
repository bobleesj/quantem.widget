Widgets
=======

quantem.widget provides nine interactive widgets for electron microscopy visualization.

All widgets support:

- **NumPy**, **CuPy**, and **PyTorch** arrays as input
- Automatic **light/dark theme** detection
- **Zoom and pan** with mouse scroll and drag
- **Colormaps**: inferno, viridis, plasma, magma, hot, gray
- **State persistence** — ``summary()``, ``state_dict()``, ``save(path)``, ``state=`` constructor param

.. toctree::
   :maxdepth: 1

   show2d
   show3d
   show3dvolume
   show4dstem
   show4d
   align2d
   mark2d
   edit2d
   showcomplex
