@echo off
rem Browse ARINA 4D-STEM scans: opens browse_arina.ipynb in JupyterLab. Nothing connects to a microscope.
rem Which Python: QUANTEM_PYTHON if set; else the active conda environment; else conda's arina-pc environment.
setlocal
title Browse ARINA scans
set "PY="
if defined QUANTEM_PYTHON set "PY=%QUANTEM_PYTHON%"
if not defined PY if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PY=%CONDA_PREFIX%\python.exe"
if not defined PY for %%R in ("%USERPROFILE%\miniforge3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%ProgramData%\miniforge3") do (
  if not defined PY if exist "%%~R\envs\arina-pc\python.exe" set "PY=%%~R\envs\arina-pc\python.exe"
)
if not defined PY (
  echo No Python with quantem.widget found.
  echo Set it up once with conda: see environments\README.md in quantem.widget ^(the arina-pc environment^),
  echo or set QUANTEM_PYTHON to the python.exe that has quantem.widget.
  pause
  exit /b 1
)
"%PY%" "%~dp0browse_arina.py" %*
if errorlevel 1 pause
