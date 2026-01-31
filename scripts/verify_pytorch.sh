#!/bin/bash
# PyTorch with CUDA verification for RTX 5090

source ~/pytorch_env/bin/activate

python << 'PYTHON_SCRIPT'
import torch
import time

print("=" * 60)
print("PyTorch RTX 5090 Verification")
print("=" * 60)
print()

# 1. PyTorch Installation
print("1. PyTorch Installation")
print("-" * 40)
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   CUDA version: {torch.version.cuda}")
print(f"   cuDNN version: {torch.backends.cudnn.version()}")
print(f"   cuDNN enabled: {torch.backends.cudnn.enabled}")
print()

# 2. GPU Detection
if torch.cuda.is_available():
    print("2. GPU Detection")
    print("-" * 40)
    props = torch.cuda.get_device_properties(0)
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   Compute Capability: {props.major}.{props.minor}")
    print(f"   Total Memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"   Multi-Processors: {props.multi_processor_count}")
    print(f"   CUDA Cores (est): {props.multi_processor_count * 128}")
    
    if props.major == 12 and props.minor == 0:
        print("   ✅ RTX 5090 (Blackwell sm_120) detected")
    print()

    # 3. CUDA Arch Support
    print("3. CUDA Architecture Support")
    print("-" * 40)
    arch_list = torch.cuda.get_arch_list()
    if 'sm_120' in arch_list:
        print("   ✅ sm_120 (RTX 5090) supported")
    print(f"   Supported architectures: {', '.join(arch_list[-4:])}")
    print()

    # 4. Data Type Support
    print("4. Data Type Support")
    print("-" * 40)
    try:
        x = torch.randn(10, 10, dtype=torch.float16).cuda()
        print("   ✅ FP16 (float16)")
        del x
    except:
        print("   ❌ FP16")
    
    try:
        x = torch.randn(10, 10, dtype=torch.bfloat16).cuda()
        print("   ✅ BF16 (bfloat16)")
        del x
    except:
        print("   ❌ BF16")
    
    print(f"   TF32 (matmul): {torch.backends.cuda.matmul.allow_tf32}")
    print(f"   TF32 (cuDNN): {torch.backends.cudnn.allow_tf32}")
    print()

    # 5. CUDA Operations Benchmark
    print("5. CUDA Operations Benchmark")
    print("-" * 40)
    size = 5000
    x = torch.randn(size, size).cuda()
    y = torch.randn(size, size).cuda()
    
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"   Matrix mult ({size}x{size}): {(end-start)*1000:.2f} ms")
    print(f"   Memory used: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    del x, y, z
    torch.cuda.empty_cache()
    print()

    # 6. Advanced Features
    print("6. Advanced Features")
    print("-" * 40)
    
    # Flash Attention
    try:
        import torch.nn.functional as F
        q = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)
        k = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)
        v = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)
        out = F.scaled_dot_product_attention(q, k, v)
        print("   ✅ Flash Attention (SDPA)")
        del q, k, v, out
    except:
        print("   ❌ Flash Attention")
    
    # Transformers
    try:
        import transformers
        print(f"   ✅ Transformers {transformers.__version__}")
    except:
        print("   ⚠️  Transformers not installed")
    
    # Accelerate
    try:
        import accelerate
        print(f"   ✅ Accelerate {accelerate.__version__}")
    except:
        print("   ⚠️  Accelerate not installed")
    
    # BitsAndBytes
    try:
        import bitsandbytes
        print(f"   ✅ BitsAndBytes {bitsandbytes.__version__}")
    except:
        print("   ⚠️  BitsAndBytes not installed (optional)")
    
    torch.cuda.empty_cache()
    print()

print("=" * 60)
print("✅ Verification Complete")
print("=" * 60)

PYTHON_SCRIPT

deactivate
