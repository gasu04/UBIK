#!/bin/bash
echo "=== CUDA Verification ==="
echo ""
echo "1. Environment Variables:"
echo "   CUDA_HOME: $CUDA_HOME"
echo ""

echo "2. NVCC Version:"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep release
else
    echo "   nvcc not found in PATH"
    echo "   Run: source ~/.bashrc"
fi
echo ""

echo "3. NVIDIA Driver & GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

echo "4. CUDA Libraries:"
echo "   Checking key libraries..."
ls -lh /usr/local/cuda-12.4/lib64/libcudart.so* 2>/dev/null | head -3 || echo "   libcudart not found"
ls -lh /usr/local/cuda-12.4/lib64/libcublas.so* 2>/dev/null | head -3 || echo "   libcublas not found"
echo ""

echo "5. PyTorch CUDA Support:"
if [ -d ~/pytorch_env ]; then
    source ~/pytorch_env/bin/activate
    python -c "
import torch
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
print(f'   CUDA version: {torch.version.cuda}')
print(f'   cuDNN version: {torch.backends.cudnn.version()}')
print(f'   cuDNN enabled: {torch.backends.cudnn.enabled}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
" 2>&1
    deactivate
else
    echo "   pytorch_env not found"
fi
echo ""

echo "6. CUDA Toolkit Installation:"
dpkg -l | grep "cuda-toolkit-12-4" | head -2
echo ""

echo "=== Verification Complete ==="
