Show3D
======

Interactive 3D stack viewer with playback, ROI analysis, FFT, and export.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Show3D

   stack = np.random.rand(50, 256, 256)
   Show3D(stack, title="My Stack", fps=10, cmap="gray")

   # With timestamps
   defocus = np.linspace(-60, 60, 50)
   Show3D(
       stack,
       labels=[f"C10={d:.0f} nm" for d in defocus],
       pixel_size=0.25,
       timestamps=defocus.tolist(),
       timestamp_unit="nm",
   )

Features
--------

- **Playback** — Play/pause/stop with configurable FPS, loop, boomerang modes
- **ROI analysis** — Circle, square, and rectangle ROI with live stats
- **FFT** — Toggle Fourier transform display
- **Histogram** — Live intensity histogram
- **Scale bar** — Calibrated scale bar when ``pixel_size`` is set
- **Comparison mode** — Side-by-side comparison of two slices
- **Drift indicator** — Track sample drift across frames
- **Export** — Save current frame as PNG or full stack as GIF

Methods
-------

.. code-block:: python

   w = Show3D(stack)

   w.play()
   w.pause()
   w.stop()
   w.set_roi(x=128, y=128, radius=20)
   w.compare_with(idx=25)

Examples
--------

- :doc:`Simple demo </examples/show3d/show3d_simple>`
- :doc:`All features </examples/show3d/show3d_all_features>`

API Reference
-------------

See :class:`quantem.widget.Show3D` for full documentation.
