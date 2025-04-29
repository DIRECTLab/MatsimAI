#!/bin/bash

# Clean build
mvn clean install -DskipTests

# Setup Conda environment
eval "$(conda shell.bash hook)" # Assuming Bash shell
conda create -n matsimAIenv python=3.10 -y
conda activate matsimAIenv

# Install core Python libraries
conda install -c conda-forge pandas numpy matplotlib tqdm bidict tensorboard rich osmnx cython seaborn bs4 bokeh -y

# Install local package
pip install -e .
pip install tbparse

# Check if a GPU is available using nvidia-smi
if nvidia-smi &> /dev/null; then
    echo "GPU detected: Installing GPU-enabled PyTorch"
    # Install GPU version of PyTorch
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
else
    echo "No GPU detected: Installing CPU-only PyTorch"
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

# Install PyG (PyTorch Geometric)
conda install pyg -c pyg -y

# Compile Cython extensions
python compile_cython.py build_ext --inplace
