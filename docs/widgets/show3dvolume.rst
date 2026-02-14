Show3DVolume
============

Orthogonal slice viewer for 3D volumes with XY, XZ, and YZ planes.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Show3DVolume

   volume = np.random.rand(64, 64, 64).astype(np.float32)
   Show3DVolume(volume, title="My Volume", cmap="viridis")

Features
--------

- **Three orthogonal views** — XY, XZ, and YZ slice planes
- **Interactive slicing** — Click or drag to navigate through the volume
- **Crosshair overlay** — Shows intersection of slice planes
- **3D volume rendering** — WebGL ray-casting for 3D visualization
- **Per-axis playback** — Animate through slices along any axis
- **FFT** — Toggle Fourier transform for each slice plane

Examples
--------

- :doc:`Simple demo </examples/show3dvolume/show3dvolume_simple>`
- :doc:`All features </examples/show3dvolume/show3dvolume_all_features>`

API Reference
-------------

See :class:`quantem.widget.Show3DVolume` for full documentation.
