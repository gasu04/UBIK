#!/bin/bash
# Ubik Environment Activation Script
# Uses existing pytorch_env

source ~/pytorch_env/bin/activate

# Set Ubik environment variables
export UBIK_HOME=~/ubik
export HF_HOME=~/ubik/data/cache/huggingface
export TORCH_HOME=~/ubik/data/cache/torch

# CUDA paths
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# WSL2 NVIDIA driver libraries
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Verify activation
echo "Ubik environment activated (using pytorch_env)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "UBIK_HOME: $UBIK_HOME"
echo "HF_HOME: $HF_HOME"
if [ -d "$CUDA_HOME" ]; then
    echo "CUDA_HOME: $CUDA_HOME"
    echo "NVCC: $(nvcc --version | grep release)"
else
    echo "CUDA_HOME: $CUDA_HOME (not found)"
fi
