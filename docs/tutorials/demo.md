# Interactive demo route

Start with a crystal, plan the experiment, then inspect a volume. These four
notebooks are the short route through the crystal and ptychography-planning
widgets. Run each notebook from the top in JupyterLab; the final expression in a
widget cell displays the interactive view.

| Step | Notebook | Try during the demo |
|---|---|---|
| 1 · Crystal and phase | [ShowCIF](showcif.ipynb) | Change unit-cell repeats, hide a species, tilt the object, and play through potential slabs |
| 2 · Acquisition geometry | [PlanPtycho](planptycho.ipynb) | Move the focus, compare 40/50/60 nm cell coverage, and switch native/2× virtual support |
| 3 · Depth playback | [Show3D](show3d.ipynb) | Scrub, play, compare Avg 1 with Avg 3, and inspect the FFT |
| 4 · Depth comparison | [Show3DSlices](show3dslices.ipynb) | Move the crosshair, rotate an oblique cut, and inspect the linked 3D view |

ShowCIF uses an explicit idealized BaTiO₃ cell; PlanPtycho uses an idealized
SrTiO₃ cell. Show3D downloads a public HAADF image and builds a moving-crop
stack. Show3DSlices uses a synthetic scalar volume. None is presented as an
experimental BTO reconstruction or a quantitative multislice simulation.

## Prepare once

Follow [Installation](../install.md) in the **same environment as the Jupyter
kernel**. Crystal examples need the `crystal` extra. Each notebook calls
`qw.profile()` so a presenter can record the actual environment.

The new ShowCIF and planning examples require a source build containing those
APIs. For a clone of this revision:

```bash
python -m pip install -e '.[crystal]'
npm ci
npm run build
python -m jupyter lab
```

Select that environment's Python kernel and restart it after changing the
installation. Use a WebGPU-capable browser on localhost or HTTPS. If WebGPU is
unavailable, check the browser's graphics support and secure origin before the
demo. An ordinary HTTP LAN/IP URL is not equivalent to localhost.

Run the notebooks once before presenting. Show3D's public example needs network
access on the first run; standalone widget HTML can also need a CDN widget
manager. Do not advertise the demo as fully offline without testing the intended
export and browser with the network disconnected.

## What the audience should take away

- A CIF supplies atomic positions and cell geometry. Species visibility and
  camera rotation help inspection; they do not refine a structure.
- ShowCIF's expected phase is the interaction constant times projected
  independent-atom potential. It is not a thick-specimen exit-wave simulation.
- PlanPtycho estimates geometry and margins. A green geometric check does not
  establish propagation convergence or reconstruction resolution.
- A moving average is a display mean across planes. It is distinct from a full
  depth projection and leaves the reconstruction unchanged.

## Other tools

| Task | Starting notebook or guide |
|---|---|
| Image, FFT, contrast, ROI | [Show2D](show2d.ipynb) |
| Maps with physical coordinates | [Plot2D](plot2d.ipynb) |
| Time traces and metrics | [Show1D](show1d.ipynb) |
| Scan positions and virtual detectors | [Show4DSTEM](show4dstem.ipynb) |
| SSB phase and aberrations | [ShowPtycho workflow guide](showptycho.md) |
| Diffraction spots and rings | [ShowDiffraction](showdiffraction.ipynb) |
| Spectra and elemental maps | [ShowEDS](showeds.ipynb) |
| Lattice basis selection | [ChooseLattice](choose_lattice.ipynb) |
| Session files | [ShowFolder](showfolder.ipynb) |

These are separate workflows, not additional prerequisites for the four-step
demo. ShowPtycho's linked page is a workflow guide, not a newly validated
standalone notebook. See the [API index](../api/index.md) for complete parameters
and [HTML export](widget_export.ipynb) for sharing an interactive result.
