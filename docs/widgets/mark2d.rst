Mark2D
=======

Interactive point picker for 2D images. Click to select atom positions, features, or lattice vectors.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Mark2D

   image = np.random.rand(256, 256)
   w = Mark2D(image, max_points=10)
   w  # display widget, click to pick points

   # Access selected points
   print(w.selected_points)

Features
--------

- **Point picking** — Click to place markers, click again to remove
- **Gallery mode** — Pick points on multiple images independently
- **Marker customization** — Shapes: circle, triangle, square, diamond, star
- **Undo/redo** — Undo and redo point selections
- **Max points** — Configurable maximum number of points per image

Methods
-------

.. code-block:: python

   w = Mark2D(image)
   w.set_image(new_image)
   w.set_image([img1, img2], labels=["A", "B"])

State Persistence
-----------------

.. code-block:: python

   w = Mark2D(image, snap_enabled=True, colormap="viridis")

   w.summary()          # Print human-readable state
   w.state_dict()       # Get all settings as a dict
   w.save("state.json") # Save to JSON file

   # Restore from file or dict — points, ROIs, and settings come back
   w2 = Mark2D(image, state="state.json")

Examples
--------

- :doc:`Simple demo </examples/mark2d/mark2d_simple>`
- :doc:`All features </examples/mark2d/mark2d_all_features>`

API Reference
-------------

See :class:`quantem.widget.Mark2D` for full documentation.
