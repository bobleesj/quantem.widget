Show1D
======

Interactive 1D data viewer for spectra, line profiles, and time series.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Show1D

   # Single trace
   spectrum = np.random.rand(512)
   Show1D(spectrum, x_label="Energy", x_unit="eV", y_label="Counts")

   # Multiple traces with calibrated X axis
   energy = np.linspace(0, 800, 512).astype(np.float32)
   traces = [np.random.rand(512) for _ in range(3)]
   Show1D(traces, x=energy, labels=["A", "B", "C"])

Features
--------

- **Multi-trace overlay** — Display multiple 1D signals with distinct colors and legend
- **Calibrated axes** — X/Y axis labels and units (e.g. "Energy (eV)", "Counts")
- **Log scale** — Logarithmic Y axis with ``log_scale=True``
- **Grid lines** — Toggle grid with ``show_grid=True``
- **Interactive zoom/pan** — Scroll to zoom, drag to pan, R to reset
- **Crosshair** — Snap-to-nearest cursor readout with trace label
- **Export** — Publication PNG (white background) or screenshot PNG

State Persistence
-----------------

.. code-block:: python

   w = Show1D(spectrum, title="EELS", log_scale=True)

   w.summary()          # Print human-readable state
   w.state_dict()       # Get all settings as a dict
   w.save("state.json") # Save versioned envelope JSON file

   # Restore from file or dict
   w2 = Show1D(spectrum, state="state.json")

Mutation Methods
----------------

.. code-block:: python

   w = Show1D(data, title="Live Plot")

   # Replace data (preserves display settings)
   w.set_data(new_data, x=new_x)

   # Add/remove individual traces
   w.add_trace(trace, label="New")
   w.remove_trace(0)

   # Clear all traces
   w.clear()

Examples
--------

- :doc:`Simple demo </examples/show1d/show1d_simple>`
- :doc:`All features </examples/show1d/show1d_all_features>`

API Reference
-------------

See :class:`quantem.widget.Show1D` for full documentation.
