Show4DSTEM
==========

Interactive 4D-STEM viewer with virtual imaging, ROI modes, and path animation.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Show4DSTEM

   data = np.random.rand(32, 32, 128, 128).astype(np.float32)
   w = Show4DSTEM(data, pixel_size=2.39, k_pixel_size=0.46)

Features
--------

- **Dual-panel viewer** — Diffraction pattern (left) and virtual image (right)
- **ROI modes** — Circle, square, rectangle, annular, and point ROIs
- **Auto-calibration** — Automatic BF disk center and radius detection
- **Path animation** — Animate scan position along custom paths or raster patterns
- **Virtual imaging** — Real-time BF, ABF, ADF virtual images
- **Scale bars** — Calibrated in angstroms (real-space) and mrad (k-space)
- **PyTorch acceleration** — GPU-accelerated virtual image computation

Methods
-------

.. code-block:: python

   w = Show4DSTEM(data)

   # ROI modes
   w.roi_circle(radius=10)
   w.roi_annular(inner_radius=5, outer_radius=15)
   w.roi_square(half_size=8)
   w.roi_rect(width=20, height=10)
   w.roi_point()

   # Calibration
   w.auto_detect_center()

   # Path animation
   w.raster(step=2, interval_ms=50)
   w.set_path([(0, 0), (1, 0), (2, 0)], interval_ms=100)
   w.play()
   w.pause()
   w.stop()

State Persistence
-----------------

.. code-block:: python

   w = Show4DSTEM(data, log_scale=True, center=(32, 32), bf_radius=9)

   w.summary()          # Print human-readable state
   w.state_dict()       # Get all settings as a dict
   w.save("state.json") # Save to JSON file

   # Restore from file or dict
   w2 = Show4DSTEM(data, state="state.json")

Examples
--------

- :doc:`Simple demo </examples/show4dstem/show4dstem_simple>`
- :doc:`All features </examples/show4dstem/show4dstem_all_features>`

API Reference
-------------

See :class:`quantem.widget.Show4DSTEM` for full documentation.
