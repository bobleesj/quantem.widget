Tool Parity Matrix
==================

This matrix tracks lock/hide API parity and the shared controls customizer.
The repository CI gate validates this contract via ``scripts/check_tool_parity.py``.

Legend:

- ``yes``: implemented and CI-checked
- ``n/a``: not applicable for this widget category

.. list-table::
   :header-rows: 1

   * - Widget
     - Shared Tool Registry
     - Runtime API
     - Controls Dropdown
     - Presets (All/Compact/Mask/Crop)
   * - Show2D
     - yes
     - yes
     - yes
     - yes
   * - Show3D
     - yes
     - yes
     - yes
     - yes
   * - Show3DVolume
     - yes
     - yes
     - yes
     - yes
   * - Show4D
     - yes
     - yes
     - yes
     - yes
   * - Show4DSTEM
     - yes
     - yes
     - yes
     - yes
   * - ShowComplex2D
     - yes
     - yes
     - yes
     - yes
   * - Mark2D
     - yes
     - yes
     - n/a
     - n/a
   * - Edit2D
     - yes
     - yes
     - yes
     - yes
   * - Align2D
     - yes
     - yes
     - n/a
     - n/a
