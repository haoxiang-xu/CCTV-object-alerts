@echo off
REM Usage: switch-cuda-version.bat 12.1 or switch-cuda-version.bat 12.2 or switch-cuda-version.bat 12.3

SET CUDA_VERSION=%1
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION%\bin;%PATH%
SET CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION%
SET LD_LIBRARY_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION%\lib\x64;%LD_LIBRARY_PATH%
