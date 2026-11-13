@echo off
rem Sets up, repairs or updates the whole installation. Run it in a Miniforge Prompt, inside the folder that holds
rem (or will hold) the quantem.gpu and quantem.widget checkouts. Safe to run again: finished steps are skipped.
setlocal
set "RAW=https://raw.githubusercontent.com/bobleesj/quantem.widget/arina-pc/environments"

where conda >nul 2>nul || (echo conda was not found. Open a Miniforge Prompt: https://conda-forge.org/download/ & goto :failed)
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || (echo No NVIDIA driver answered. Install driver 580 or newer. & goto :failed)

echo.
echo [1/5] Environment arina-pc: Python, Git, Node.js, JupyterLab, PyTorch with CUDA
curl -fsSLO %RAW%/arina-pc.yml || (echo Could not download arina-pc.yml. Check the internet connection. & goto :failed)
call conda env list | findstr /b /c:"arina-pc " >nul
if errorlevel 1 (call conda env create -f arina-pc.yml) else (call conda env update -n arina-pc -f arina-pc.yml)
if errorlevel 1 goto :failed
call conda activate arina-pc
if errorlevel 1 goto :failed
rem every later step must run in this environment's Python, not another one on the PC
python -c "import sys; sys.exit(0 if sys.prefix.replace('\\', '/').rstrip('/').endswith('/arina-pc') else 1)" || (echo The arina-pc environment is not the active Python. Close this window, open a new Miniforge Prompt, run install.bat again. & goto :failed)

echo.
echo [2/5] Source checkouts
if not exist quantem.gpu\.git (git clone -b arina-pc https://github.com/bobleesj/quantem.gpu.git || goto :failed)
if not exist quantem.widget\.git (git clone -b arina-pc https://github.com/bobleesj/quantem.widget.git || goto :failed)

echo.
echo [3/5] quantem.gpu and quantem.widget
python -m pip install -e "quantem.gpu[cuda]" -e quantem.widget || goto :failed

echo.
echo [4/5] Widget JavaScript
call widget build || goto :failed

echo.
echo [5/5] Graphics card
python -c "import torch, cupy as cp; f, t = cp.cuda.runtime.memGetInfo(); assert torch.cuda.is_available(), 'this torch has no CUDA'; print('GPU ready:', torch.cuda.get_device_name(0), round(f / 2**30, 2), 'of', round(t / 2**30, 2), 'GiB free')" || (echo The card is not usable from Python. Copy the lines above and send them. & goto :failed)

echo.
echo Done. Next time: conda activate arina-pc ^&^& jupyter lab        New features: widget update
call jupyter lab quantem.widget\environments\browse.ipynb
goto :eof

:failed
echo.
echo Installation stopped at the step above. Nothing was removed; run install.bat again after fixing it.
exit /b 1
