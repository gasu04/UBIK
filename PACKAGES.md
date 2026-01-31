# Ubik Installed Packages

## Summary

**Total Packages**: 169 (frozen)  
**Python Version**: 3.12.3  
**Environment**: ~/pytorch_env  

## Core ML Stack

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.9.1+cu128 | Deep learning framework |
| Transformers | 4.57.3 | Transformer models (DeepSeek, Llama, etc.) |
| Accelerate | 1.12.0 | Distributed training & inference |
| BitsAndBytes | 0.49.1 | 4-bit/8-bit quantization |
| PEFT | 0.18.1 | LoRA, QLoRA adapters |
| TRL | 0.26.2 | DPO, PPO training |
| Datasets | 4.4.2 | Dataset management |
| Optimum | 2.1.0 | Hardware optimization |

## CUDA & GPU

| Component | Version |
|-----------|---------|
| CUDA Toolkit | 12.4.131 |
| PyTorch CUDA | 12.8 |
| cuDNN | 9.1.0.02 |
| Compute Capability | 12.0 (sm_120) |

## Quantization Support

- ✅ **8-bit quantization** (Int8)
- ✅ **4-bit quantization** (NF4, FP4)
- ✅ **QLoRA** (4-bit LoRA training)
- ✅ **GPTQ** (via Optimum)
- ✅ **AWQ** (via Transformers)

## Precision Support

| Precision | Status | Use Case |
|-----------|--------|----------|
| FP32 | ✅ | Full precision (8B models) |
| FP16 | ✅ | Half precision (~20B models) |
| BF16 | ✅ | Brain Float (Recommended for LLMs) |
| TF32 | ✅ | TensorFloat-32 (automatic on Blackwell) |
| Int8 | ✅ | 8-bit quantization (~40B models) |
| Int4 | ✅ | 4-bit quantization (~70B models) |

## Model Support

### Optimized for RTX 5090 (32GB VRAM)

**Full Precision (BF16)**:
- DeepSeek-V3 16B ✅
- Llama 3.1 8B ✅
- Mistral 7B/22B ✅
- Qwen 14B ✅

**4-bit Quantization**:
- Llama 3.1 70B ✅
- DeepSeek-V3 67B ✅
- Mixtral 8x7B ✅

**8-bit Quantization**:
- Larger models up to ~40B parameters

## Training Capabilities

### Parameter-Efficient Fine-Tuning (PEFT)
- ✅ LoRA (Low-Rank Adaptation)
- ✅ QLoRA (Quantized LoRA)
- ✅ Prefix Tuning
- ✅ P-Tuning
- ✅ IA3 (Infused Adapter)

### Reinforcement Learning (TRL)
- ✅ DPO (Direct Preference Optimization)
- ✅ PPO (Proximal Policy Optimization)
- ✅ SFT (Supervised Fine-Tuning)
- ✅ Reward Modeling

## Advanced Features

- ✅ **Flash Attention 2** (via PyTorch SDPA)
- ✅ **Gradient Checkpointing**
- ✅ **Mixed Precision Training**
- ✅ **CPU Offloading**
- ✅ **Model Parallelism**
- ✅ **Gradient Accumulation**

## API & Infrastructure

| Package | Version | Purpose |
|---------|---------|---------|
| FastAPI | (to be installed) | REST API server |
| Uvicorn | (to be installed) | ASGI server |
| WebSockets | (to be installed) | Real-time communication |
| HTTPX | 0.28.1 | Async HTTP client |
| Pydantic | (to be installed) | Data validation |

## Development Tools

| Package | Version | Purpose |
|---------|---------|---------|
| JupyterLab | 4.5.2 | Interactive notebooks |
| IPython | 9.1.1 | Enhanced Python shell |
| Matplotlib | 3.10.1 | Plotting |
| Pandas | 2.3.3 | Data analysis |
| NumPy | 2.4.1 | Numerical computing |

## Monitoring & Logging

| Package | Version | Purpose |
|---------|---------|---------|
| psutil | 7.2.1 | System monitoring |
| tqdm | 4.67.1 | Progress bars |
| TensorBoard | (to be installed) | Training visualization |

## Testing

| Package | Purpose |
|---------|---------|
| pytest | (to be installed) | Unit testing |
| pytest-asyncio | (to be installed) | Async testing |
| pytest-cov | (to be installed) | Coverage reporting |

## Installation Files

1. **requirements.txt** - Curated list with minimum versions
2. **requirements-frozen.txt** - Exact versions (169 packages)

## Disk Usage Estimate

- Python packages: ~5 GB
- PyTorch + CUDA libs: ~3 GB
- Model cache (HuggingFace): Will grow with downloads
- Recommended free space: 500GB+ for models

## Updates

To update packages:
```bash
pip install --upgrade -r requirements.txt
pip freeze > requirements-frozen.txt
```

## Verification

Run verification scripts:
```bash
~/ubik/scripts/verify_cuda.sh        # CUDA toolkit
~/ubik/scripts/verify_pytorch.sh     # PyTorch + GPU
```

## Notes

- PyTorch includes CUDA 12.8 libraries
- WSL2 uses Windows NVIDIA driver (577.00)
- CUDA toolkit 12.4 installed for compilation
- BitsAndBytes compiled for CUDA 12.x
- All packages tested on RTX 5090

## References

- PyTorch: https://pytorch.org/
- Transformers: https://huggingface.co/docs/transformers
- PEFT: https://huggingface.co/docs/peft
- TRL: https://huggingface.co/docs/trl
- BitsAndBytes: https://github.com/TimDettmers/bitsandbytes
