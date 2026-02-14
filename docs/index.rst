quantem.widget
==============

Interactive Jupyter widgets for electron microscopy visualization.
Works with NumPy, CuPy, and PyTorch arrays.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: getting-started
      :link-type: doc

      Installation and quick start guide.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      Full API documentation for all widgets.

   .. grid-item-card:: Examples
      :link: examples/index
      :link-type: doc

      Interactive Jupyter notebook examples.

   .. grid-item-card:: Widgets Guide
      :link: widgets/index
      :link-type: doc

      Detailed usage guide for each widget.

Widgets
-------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Widget
     - Description
   * - :doc:`Show2D <widgets/show2d>`
     - Static 2D image viewer with gallery support, FFT, histogram
   * - :doc:`Show3D <widgets/show3d>`
     - 3D stack viewer with playback, ROI, FFT, export (PNG/GIF)
   * - :doc:`Show3DVolume <widgets/show3dvolume>`
     - Orthogonal slice viewer (XY, XZ, YZ) with 3D volume rendering
   * - :doc:`Show4DSTEM <widgets/show4dstem>`
     - 4D-STEM diffraction pattern viewer with virtual imaging
   * - :doc:`Clicker <widgets/clicker>`
     - Interactive point picker for 2D images

Quick Install
-------------

.. code-block:: bash

   pip install quantem-widget

.. toctree::
   :maxdepth: 2
   :hidden:

   getting-started
   widgets/index
   examples/index
   api/index
