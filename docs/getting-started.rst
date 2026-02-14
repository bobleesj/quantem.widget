Getting Started
===============

Installation
------------

.. code-block:: bash

   pip install quantem-widget

For development:

.. code-block:: bash

   git clone https://github.com/bobleesj/quantem.widget.git
   cd quantem.widget
   npm install
   npm run build
   pip install -e .

Quick Start
-----------

View a 2D image
~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from quantem.widget import Show2D

   image = np.random.rand(256, 256)
   Show2D(image)

View a gallery of images
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   images = [np.random.rand(256, 256) for _ in range(6)]
   Show2D(images, labels=["A", "B", "C", "D", "E", "F"], ncols=3)

View a 3D stack with playback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from quantem.widget import Show3D

   stack = np.random.rand(50, 256, 256)
   Show3D(stack, title="My Stack", fps=10, cmap="gray")

View orthogonal slices of a volume
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from quantem.widget import Show3DVolume

   volume = np.random.rand(64, 64, 64).astype(np.float32)
   Show3DVolume(volume, title="My Volume", cmap="viridis")

View 4D-STEM data
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from quantem.widget import Show4DSTEM

   data = np.random.rand(32, 32, 128, 128).astype(np.float32)
   Show4DSTEM(data, pixel_size=2.39, k_pixel_size=0.46)

Pick points on an image
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from quantem.widget import Clicker

   image = np.random.rand(256, 256)
   w = Clicker(image, max_points=5)
   w  # display widget, click to pick points

   # Access selected points
   print(w.selected_points)

PyTorch Support
----------------

All widgets accept PyTorch tensors directly:

.. code-block:: python

   import torch
   from quantem.widget import Show2D

   tensor = torch.randn(256, 256)
   Show2D(tensor)
