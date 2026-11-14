# At the microscope

Things to run on a PC beside the microscope or detector. Nothing here connects to a microscope or detector.

**Browse ARINA scans:** double-click `BROWSE_ARINA.bat`. JupyterLab opens `browse_arina.ipynb`: set `SCAN` to a
scan (an ARINA `_master.h5` with its data files beside it, or a `.qem`), run the cells: the scan in `Show4DSTEM`,
then direct ptychography with `ShowPtycho.live` (fifth-order fit on the central 256 x 256, the whole phase; drag the
fit box to fit elsewhere; `ptycho.save()` writes `<scan>.json` beside the scan). A 4 GB card holds one 512 x 512 scan.

Set up the Python once with conda (`environments/README.md`, the `arina-pc` environment). `BROWSE_ARINA.bat` uses
`QUANTEM_PYTHON` if set, else the active conda environment, else conda's `arina-pc` environment.
