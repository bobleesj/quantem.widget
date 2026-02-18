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
- **Export** — Save current frame as PNG or full stack as GIF

Methods
-------

.. code-block:: python

   w = Show3D(stack)

   # Playback
   w.play()
   w.pause()
   w.stop()
   w.goto(25)  # Jump to frame 25
   w.set_playback_path([0, 10, 20, 10, 0])  # Custom frame order

   # ROI analysis
   w.set_roi(row=128, col=128, radius=20)
   w.roi_circle(radius=15)
   w.roi_square(half_size=10)
   w.roi_rectangle(width=20, height=10)
   w.roi_annular(inner=5, outer=15)

   # Line profile
   w.set_profile((10, 10), (200, 200))
   w.clear_profile()

State Persistence
-----------------

.. code-block:: python

   w = Show3D(stack, cmap="gray", fps=12, boomerang=True)
   w.bookmarked_frames = [0, 15, 29]

   w.summary()          # Print human-readable state
   w.state_dict()       # Get all settings as a dict
   w.save("state.json") # Save to JSON file

   # Restore from file or dict
   w2 = Show3D(stack, state="state.json")

Examples
--------

- :doc:`Simple demo </examples/show3d/show3d_simple>`
- :doc:`All features </examples/show3d/show3d_all_features>`

API Reference
-------------

See :class:`quantem.widget.Show3D` for full documentation.
