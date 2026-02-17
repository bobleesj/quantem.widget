Show4D
======

Interactive 4D data viewer with dual-panel navigation and signal display.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Show4D

   data = np.random.rand(32, 32, 64, 64).astype(np.float32)
   Show4D(data, title="4D Data", nav_pixel_size=2.39, nav_pixel_unit="A")

   # Custom navigation image
   nav = data.std(axis=(2, 3))
   Show4D(data, nav_image=nav)

Features
--------

- **Dual-panel viewer** -- Navigation image (left) and signal frame (right)
- **ROI modes** -- Circle, square, and rectangle ROI on navigation image
- **ROI reduce** -- Mean, max, min, or sum over ROI positions
- **Scale bars** -- Calibrated for both navigation and signal panels
- **Snap-to-peak** -- Click near a feature and snap to the local maximum
- **Custom nav image** -- Override the default mean image

Methods
-------

.. code-block:: python

   w = Show4D(data)

   # Position control
   w.position = (10, 20)

   # ROI modes
   w.roi_mode = "circle"
   w.roi_center_row = 16.0
   w.roi_center_col = 16.0
   w.roi_radius = 5.0

State Persistence
-----------------

.. code-block:: python

   w = Show4D(data, cmap="viridis", log_scale=True)

   w.summary()          # Print human-readable state
   w.state_dict()       # Get all settings as a dict
   w.save("state.json") # Save to JSON file

   # Restore from file or dict
   w2 = Show4D(data, state="state.json")

Examples
--------

- :doc:`Simple demo </examples/show4d/show4d_simple>`
- :doc:`All features </examples/show4d/show4d_all_features>`

API Reference
-------------

See :class:`quantem.widget.Show4D` for full documentation.
