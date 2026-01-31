# Ubik Project - Somatic Node (PowerSpec AI100)

## Directory Structure

- `somatic/` - Core somatic node code (inference, MCP client)
- `models/` - Downloaded and fine-tuned models
- `training/` - DPO training infrastructure
- `data/` - Training data and datasets
- `logs/` - Application and system logs
- `config/` - Configuration files
- `scripts/` - Utility scripts
- `tests/` - Test suite

## Quick Start

1. Activate environment: `source ~/ubik/scripts/activate_ubik.sh`
2. Start inference server: `~/ubik/scripts/start_inference.sh`
3. Check health: `~/ubik/scripts/check_inference.sh`

## Hardware

- CPU: AMD Threadripper 9960X
- GPU: NVIDIA RTX 5090 (32GB)
- RAM: 128GB
- Role: Inference and DPO training
