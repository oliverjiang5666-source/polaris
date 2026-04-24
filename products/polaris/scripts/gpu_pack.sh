#!/bin/bash
# Pack project for cloud GPU upload
# Usage: bash scripts/gpu_pack.sh
# Output: energy-storage-gpu.tar.gz (~120MB)

set -e
cd "$(dirname "$0")/.."

echo "=== Packing project for GPU training ==="

# Create tarball with only needed files
tar czf energy-storage-gpu.tar.gz \
    config.py \
    forecast/__init__.py \
    forecast/transformer_config.py \
    forecast/transformer_model.py \
    forecast/transformer_dataset.py \
    forecast/transformer_forecaster.py \
    forecast/lgbm_forecaster.py \
    forecast/mpc_controller.py \
    forecast/naive.py \
    oracle/__init__.py \
    oracle/lp_oracle.py \
    data/__init__.py \
    data/china/__init__.py \
    data/china/features.py \
    data/china/province_registry.py \
    data/china/processed/shandong_oracle.parquet \
    data/china/processed/shanxi_oracle.parquet \
    data/china/processed/guangdong_oracle.parquet \
    data/china/processed/gansu_oracle.parquet \
    scripts/11_transformer_train.py \
    2>/dev/null

SIZE=$(du -h energy-storage-gpu.tar.gz | cut -f1)
echo "=== Done: energy-storage-gpu.tar.gz ($SIZE) ==="
echo ""
echo "Upload to GPU server, then run:"
echo "  tar xzf energy-storage-gpu.tar.gz"
echo "  bash scripts/gpu_setup.sh"
echo "  bash scripts/gpu_run.sh"
