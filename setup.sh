#!/bin/bash




echo "=== Checking NVIDIA GPU status ==="
nvidia-smi

echo "=== Uninstalling existing JAX and related packages ==="
pip uninstall -y jax jaxlib

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Uninstalling conflicting packages ==="
pip uninstall -y tensorflow ml-dtypes dopamine-rl thinc gym

echo "=== Installing specific versions of required packages ==="
pip install numpy==1.26.0 gym==0.26.0 ml-dtypes==0.4.0

echo "=== Installing JAX with CUDA 12 support ==="
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "=== Moving into Kinetix directory ==="
cd Kinetix || { echo "Kinetix folder not found!"; exit 1; }

echo "=== Setting Git Status ==="

git config --global user.email "preciousblessed1000@gmail.com"
git config --global user.name "preciousBlessed"

echo "=== Installing Python dependencies from requirements.txt ==="
pip install -r requirements.txt

echo "=== Installing Kinetix package in editable mode ==="
pip install -e .

echo "=== Installing and setting up pre-commit hooks ==="
pip install pre-commit
pre-commit install

echo "=== ✅ Installation completed successfully and git setups initialized ==="