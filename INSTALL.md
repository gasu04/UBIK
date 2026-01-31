# Ubik Installation Guide

## System Requirements

- **OS**: Ubuntu 24.04 LTS (WSL2 or native)
- **Python**: 3.12+
- **GPU**: NVIDIA RTX 5090 (32GB VRAM)
- **CUDA**: 12.4+
- **RAM**: 128GB recommended
- **Storage**: 500GB+ for models

## Quick Start

### 1. Clone or Navigate to Ubik Directory

```bash
cd ~/ubik
```

### 2. Activate Environment

```bash
# Quick activation (if bashrc configured)
ubik

# Or manual activation
source ~/pytorch_env/bin/activate
```

### 3. Install Dependencies

#### Option A: Install from requirements.txt (Recommended)

```bash
pip install -r requirements.txt
```

This installs the latest compatible versions of all packages.

#### Option B: Install from frozen requirements (Exact versions)

```bash
pip install -r requirements-frozen.txt
```

This installs the exact versions tested on PowerSpec AI100.

### 4. Verify Installation

```bash
# Verify CUDA setup
~/ubik/scripts/verify_cuda.sh

# Verify PyTorch + RTX 5090
~/ubik/scripts/verify_pytorch.sh
```

## Core Dependencies

### ML Framework
- **PyTorch 2.9.1+cu128** - Deep learning framework with CUDA 12.8
- **Transformers 4.57.3** - HuggingFace transformers
- **Accelerate 1.12.0** - Distributed training utilities

### Quantization & Optimization
- **BitsAndBytes 0.49.1** - 4-bit/8-bit quantization
- **Optimum 2.1.0** - Hardware optimization
- **Safetensors** - Fast model serialization

### Training & Fine-tuning
- **PEFT 0.18.1** - Parameter-efficient fine-tuning (LoRA, QLoRA)
- **TRL 0.26.2** - Reinforcement learning (DPO, PPO)
- **Datasets 4.4.2** - Dataset management
- **SentencePiece** - Tokenization

### Infrastructure
- **FastAPI** - API server
- **Uvicorn** - ASGI server
- **WebSockets** - Real-time communication
- **Loguru** - Advanced logging

### Development
- **JupyterLab** - Interactive notebooks
- **TensorBoard** - Training visualization
- **Pytest** - Testing framework

## Manual Installation (From Scratch)

If you need to set up from scratch:

### 1. Create Virtual Environment

```bash
python3.12 -m venv ~/pytorch_env
source ~/pytorch_env/bin/activate
pip install --upgrade pip setuptools wheel
```

### 2. Install PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install Ubik Requirements

```bash
cd ~/ubik
pip install -r requirements.txt
```

## Optional Packages

### Experiment Tracking

```bash
pip install wandb  # Weights & Biases
```

### Flash Attention 2 (For even faster inference)

```bash
pip install flash-attn --no-build-isolation
```

Note: Flash Attention compilation can take 10-20 minutes.

## Verify Installation

After installation, verify everything works:

```bash
# Activate environment
source ~/pytorch_env/bin/activate

# Run verification
python << 'PYTHON'
import torch
import transformers
import accelerate
import bitsandbytes
import peft
import trl

print("✅ All packages imported successfully!")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
PYTHON
```

Expected output:
```
✅ All packages imported successfully!
PyTorch: 2.9.1+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 5090
```

## Troubleshooting

### CUDA Not Available

```bash
# Check CUDA toolkit
nvcc --version

# Check NVIDIA driver
nvidia-smi

# Verify PyTorch sees CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory Errors

- Use quantization (4-bit or 8-bit)
- Reduce batch size
- Enable gradient checkpointing
- Use CPU offloading with Accelerate

### Import Errors

```bash
# Reinstall package
pip install --force-reinstall package-name

# Clear cache
pip cache purge
```

## Updating Packages

```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade transformers

# Regenerate frozen requirements
pip freeze > requirements-frozen.txt
```

## Environment Variables

Set in `~/.bashrc` or `~/ubik/scripts/activate_ubik.sh`:

```bash
export UBIK_HOME=~/ubik
export HF_HOME=~/ubik/data/cache/huggingface
export TRANSFORMERS_CACHE=~/ubik/data/cache/huggingface
export TORCH_HOME=~/ubik/data/cache/torch
export CUDA_HOME=/usr/local/cuda-12.4
```

## Next Steps

After installation:

1. **Download models**: See `docs/models.md`
2. **Configure inference**: See `somatic/inference/README.md`
3. **Setup training**: See `training/README.md`
4. **Start API server**: See `docs/api.md`

## Support

- Issues: Check `~/ubik/logs/` for error logs
- Documentation: See `~/ubik/README.md`
- Verification: Run `~/ubik/scripts/verify_*.sh`
