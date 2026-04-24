#!/bin/bash
# Pack V3/V4 PatchTST project for cloud GPU upload
# Usage: bash scripts/gpu_pack_v3.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Packing PatchTST V3 for GPU training ==="

tar czf energy-storage-gpu-v3.tar.gz \
    config.py \
    forecast/__init__.py \
    forecast/transformer_config.py \
    forecast/transformer_model.py \
    forecast/transformer_dataset.py \
    forecast/transformer_forecaster.py \
    forecast/patchtst_model.py \
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
    scripts/12_patchtst_train.py \
    scripts/14_patchtst_v4_train.py \
    scripts/15_patchtst_v5_train.py \
    scripts/16_validate_v5.py \
    2>/dev/null

SIZE=$(du -h energy-storage-gpu-v3.tar.gz | cut -f1)
echo "=== Done: energy-storage-gpu-v3.tar.gz ($SIZE) ==="
echo ""
echo "Upload & run:"
echo "  cd /root/autodl-tmp && tar xzf energy-storage-gpu-v3.tar.gz"
echo "  nohup python -u scripts/12_patchtst_train.py --all > gpu_patchtst.log 2>&1 &"
