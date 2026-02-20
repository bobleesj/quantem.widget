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
- **Quick view presets** -- 3 save/load slots (UI buttons or keyboard ``1/2/3``, ``Shift+1/2/3``)

Control Groups
--------------

.. code-block:: python

   # Lock groups (visible but non-interactive)
   w = Show4D(
       data,
       disable_roi=True,
       disable_navigation=True,
       disable_playback=True,
       disable_fft=True,
   )

   # Hide groups entirely
   w = Show4D(
       data,
       hide_histogram=True,
       hide_profile=True,
       hidden_tools=["export"],
   )

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
   w.set_view_preset("1", {
       "navColormap": "viridis",
       "sigColormap": "plasma",
       "navScaleMode": "linear",
       "sigScaleMode": "log",
       "navVminPct": 5,
       "navVmaxPct": 95,
       "sigVminPct": 10,
       "sigVmaxPct": 90,
       "roiMode": "circle",
       "profileActive": True,
   })

   w.summary()          # Print human-readable state
   state = w.state_dict()  # Snapshot full state
   w.save("state.json") # Save versioned envelope JSON file

   # Preset helpers
   w.list_view_preset_slots()  # ('1',)
   w.get_view_preset("1")
   w.clear_view_preset("1")
   w.reset_view_presets()

   # Restore in three ways
   w.load_state_dict(state)              # 1) apply dict in-place
   w2 = Show4D(data, state="state.json") # 2) from saved file
   w3 = Show4D(data, state=state)        # 3) from dict at init

Examples
--------

- :doc:`Simple demo </examples/show4d/show4d_simple>`
- :doc:`All features </examples/show4d/show4d_all_features>`

API Reference
-------------

See :class:`quantem.widget.Show4D` for full documentation.
