Show2D
======

Static 2D image viewer with optional FFT, histogram analysis, and gallery mode.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Show2D

   # Single image
   image = np.random.rand(256, 256)
   Show2D(image, cmap="inferno")

   # Gallery of images
   images = [np.random.rand(256, 256) for _ in range(6)]
   Show2D(images, labels=["A", "B", "C", "D", "E", "F"], ncols=3)

Features
--------

- **Gallery mode** — Display multiple images side-by-side with configurable columns
- **FFT** — Toggle Fourier transform display with ``show_fft=True``
- **Histogram** — Intensity histogram with adjustable contrast
- **Scale bar** — Calibrated scale bar when ``pixel_size_angstrom`` is set
- **Log scale** — Logarithmic intensity scaling with ``log_scale=True``
- **Auto contrast** — Percentile-based contrast with ``auto_contrast=True``

Examples
--------

- :doc:`Simple demo </examples/show2d/show2d_simple>`
- :doc:`All features </examples/show2d/show2d_all_features>`

API Reference
-------------

See :class:`quantem.widget.Show2D` for full documentation.
