# Ubik Environment Setup

## Python Environment

- **Python Version**: 3.12.3
- **Virtual Environment**: Uses existing `~/pytorch_env`
- **PyTorch Version**: 2.9.1+cu128
- **CUDA**: 12.8 (available and working)
- **NVIDIA Driver**: 575.64.04

## Activation

```bash
# Quick activation
ubik

# Or full path
source ~/ubik/scripts/activate_ubik.sh
```

## Environment Variables

When activated, the following are set:
- `UBIK_HOME`: ~/ubik
- `HF_HOME`: ~/ubik/data/cache/huggingface
- `TRANSFORMERS_CACHE`: ~/ubik/data/cache/huggingface
- `TORCH_HOME`: ~/ubik/data/cache/torch
- `VIRTUAL_ENV`: ~/pytorch_env

## Useful Aliases

Added to ~/.bashrc:
- `ubik` - Activate Ubik environment
- `ubik-logs` - Tail inference logs
- `ubik-gpu` - Watch GPU usage with nvidia-smi
- `ubik-home` - Change to Ubik directory

## Package Management

```bash
# Upgrade pip
pip install --upgrade pip

# Install ML packages
pip install transformers accelerate bitsandbytes

# Install training packages
pip install trl peft datasets tensorboard

# Install Ubik requirements (when available)
pip install -r ~/ubik/requirements.txt
```

## CUDA Toolkit

- **CUDA Toolkit Version**: 12.4.131
- **CUDA Home**: /usr/local/cuda-12.4
- **cuDNN Version**: 9.1.0.02 (bundled with PyTorch)
- **NVCC**: Available after sourcing ~/.bashrc or activating ubik
- **Compilation**: Tested and working with RTX 5090

### CUDA Environment Variables

Set automatically in ~/.bashrc:
```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

### Verify CUDA Installation

```bash
# Run comprehensive verification
~/ubik/scripts/verify_cuda.sh

# Quick checks
nvcc --version
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Verification

```bash
source ~/ubik/scripts/activate_ubik.sh
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```
