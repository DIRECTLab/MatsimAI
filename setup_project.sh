#!/bin/bash

mvn clean install -DskipTests

# Setup Conda environment
eval "$(conda shell.bash hook)" # Assuming Bash shell
conda create -n matsimAIenv python=3.10 -y
conda activate matsimAIenv

# Install core Python libraries
conda install -c conda-forge pandas numpy matplotlib tqdm bidict tensorboard rich osmnx cython seaborn bs4 -y

# Install CPU-only PyTorch by default
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install local package
pip install -e .

# Check if CUDA is available
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    V=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9\.]*\).*/\1/p')
    echo "GPU detected: Installing CUDA version $V"
    conda install pytorch torchvision torchaudio pytorch-cuda=$V -c pytorch -c nvidia -y
fi

# Install PyG (PyTorch Geometric)
conda install pyg -c pyg -y

# Compile Cython extensions
python compile_cython.py build_ext --inplace
