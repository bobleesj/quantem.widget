ShowComplex2D
=============

Interactive viewer for complex-valued 2D data from ptychography, holography,
and exit wave reconstruction.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import ShowComplex2D

   # From complex array
   data = np.exp(1j * phase) * amplitude
   ShowComplex2D(data, display_mode="hsv", title="Exit Wave")

   # From (real, imag) tuple
   ShowComplex2D((real_part, imag_part), display_mode="phase")

Features
--------

- **Five display modes** -- Amplitude, phase, HSV (hue=phase, brightness=amplitude), real, imaginary
- **Phase colorwheel** -- Inset showing phase-to-color mapping (visible in phase and HSV modes)
- **FFT** -- Toggle Fourier transform side panel
- **Colorbar** -- Vertical colorbar with data range labels
- **Log scale** -- Logarithmic intensity mapping for amplitude mode
- **Auto contrast** -- Percentile-based contrast stretching
- **Scale bar** -- Calibrated when ``pixel_size_angstrom`` is set
- **Figure export** -- Publication-quality PNG with title, scale bar, and colorbar

Display Modes
-------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Description
   * - ``amplitude``
     - Magnitude of complex values (default)
   * - ``phase``
     - Phase angle in radians (cyclic colormap)
   * - ``hsv``
     - Hue encodes phase, brightness encodes amplitude
   * - ``real``
     - Real part only
   * - ``imag``
     - Imaginary part only

State Persistence
-----------------

.. code-block:: python

   w = ShowComplex2D(data, display_mode="phase", cmap="viridis")

   w.summary()          # Print human-readable state
   w.state_dict()       # Get all settings as a dict
   w.save("state.json") # Save to JSON file

   # Restore from file or dict
   w2 = ShowComplex2D(data, state="state.json")

API Reference
-------------

See :class:`quantem.widget.ShowComplex2D` for full documentation.
