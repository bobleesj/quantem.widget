# At the microscope

Things to run on a PC beside the microscope or detector. Nothing here connects to a microscope or detector.

**Browse ARINA scans:** double-click `BROWSE_ARINA.bat`. JupyterLab opens `browse_arina.ipynb`: set `SCAN` to a
scan (an ARINA `_master.h5` with its data files beside it, or a `.qem`), run the cells: the scan in `Show4DSTEM`,
then direct ptychography with `ShowPtycho.live` (fifth-order fit on the central 256 x 256, or 128 x 128 on a GPU of under 6 GB, the whole phase; drag the
fit box to fit elsewhere; `ptycho.save()` writes `<scan>.json` beside the scan). A 4 GB card holds one 512 x 512 scan.

Set up the Python once with conda (`environments/README.md`, the `arina-pc` environment). `BROWSE_ARINA.bat` uses
`QUANTEM_PYTHON` if set, else the active conda environment, else conda's `arina-pc` environment.

**Measured** on a 4 GB RTX A400 (Windows): a 512 x 512 scan on a 192 x 192 detector (19.3 GB of counts) loads in
about 10 s as 2.4 GB on the card and fits fifth order in about 15 s. Close a scan before the next one:
`ptycho.close()`, `viewer.close()`, `quantem.gpu.device.release_cached_memory()`.

**On a Python that also has Intel MKL** (a vendor's Python, such as a microscope control PC's): MKL and PyTorch
each bring an OpenMP runtime, and the second one stops the kernel the moment a fit starts, with no message in the
notebook (in a terminal: `OMP: Error #15`). Set `KMP_DUPLICATE_LIB_OK=TRUE` before `BROWSE_ARINA.bat`, for example
in a two-line .bat that sets it and calls this one. A conda `arina-pc` environment has one runtime and needs nothing.
