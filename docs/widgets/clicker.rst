Clicker
=======

Interactive point picker for 2D images. Click to select atom positions, features, or lattice vectors.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Clicker

   image = np.random.rand(256, 256)
   w = Clicker(image, max_points=10)
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

   w = Clicker(image)
   w.set_image(new_image)
   w.set_image([img1, img2], labels=["A", "B"])

Examples
--------

- :doc:`Simple demo </examples/clicker/clicker_simple>`
- :doc:`All features </examples/clicker/clicker_all_features>`

API Reference
-------------

See :class:`quantem.widget.Clicker` for full documentation.
