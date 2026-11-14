"""Open browse_arina.ipynb in JupyterLab with this Python; nothing connects to a microscope or detector.

Run by BROWSE_ARINA.bat (or ``python microscope/browse_arina.py``); extra options go to JupyterLab
(``--no-browser``, ``--port=...``). JupyterLab runs on the Python that runs this file, and so does the notebook:
its kernel is the one already installed for this Python (with that kernel's own settings, such as a CUDA path),
else this Python plainly. A virtual environment made on another Python (a venv on a vendor's Python) keeps
JupyterLab's built page and its extensions in the parent's share/jupyter, which the venv does not search: that is
added here.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "browse_arina.ipynb"
# where the notebook's kernel is written (outside the checkout, so git sees nothing)
STATE = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "quantem-widget" / "browse-arina" / "jupyter"


def same_python(command: str) -> bool:
    """Whether a kernel's command is this Python."""
    try:
        return Path(command).resolve() == Path(sys.executable).resolve()
    except OSError:
        return False


def notebook_kernel(env: dict) -> dict:
    """The kernel the notebook runs on: the one installed for this Python, else this Python plainly."""
    from jupyter_client.kernelspec import KernelSpecManager

    saved = os.environ.copy()
    os.environ.update(env)                  # the kernel folders JupyterLab will search
    try:
        specs = KernelSpecManager().get_all_specs()
    finally:
        os.environ.clear()
        os.environ.update(saved)
    for name, found in sorted(specs.items()):
        spec = found["spec"]
        argv = spec.get("argv") or []
        # a kernel that runs here, on this Python (not one that hands the work to another machine)
        if argv and same_python(argv[0]) and any(part.startswith("ipykernel") for part in argv[1:]):
            print(f"The notebook runs on the kernel \"{spec.get('display_name', name)}\" ({name}).")
            return spec
    return {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python 3 (browse ARINA)", "language": "python"}


def main() -> int:
    env = os.environ.copy()
    options = []
    parent_share = Path(sys.base_prefix) / "share" / "jupyter"
    if (sys.prefix != sys.base_prefix and not (Path(sys.prefix) / "share/jupyter/lab/static").is_dir()
            and (parent_share / "lab/static").is_dir()):
        options.append(f"--LabApp.app_dir={parent_share / 'lab'}")
        env["JUPYTER_PATH"] = os.pathsep.join(filter(None, [env.get("JUPYTER_PATH"), str(parent_share)]))
    # the notebook names the kernel python3: here that is this Python's own kernel, found before anything else
    kernel = STATE / "kernels" / "python3"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(json.dumps(notebook_kernel(env), indent=1), encoding="utf-8")
    env["JUPYTER_PATH"] = os.pathsep.join(filter(None, [str(STATE), env.get("JUPYTER_PATH")]))
    print(f"Opening {NOTEBOOK.name} in JupyterLab with {sys.executable}")
    print("Close this window to stop JupyterLab. Nothing here connects to a microscope or detector.")
    return subprocess.call([sys.executable, "-m", "jupyterlab", str(NOTEBOOK), f"--ServerApp.root_dir={HERE}",
                            *options, *sys.argv[1:]], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
