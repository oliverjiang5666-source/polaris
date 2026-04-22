#!/bin/bash
# Run full training + evaluation on GPU for all 4 provinces
# Expected time on 5090: ~30-45 minutes total
# Expected time on 4090: ~45-60 minutes total

set -e

echo "============================================"
echo "  Transformer EPF Training — All Provinces"
echo "  $(date)"
echo "============================================"

PYTHONPATH=. python3 -u scripts/11_transformer_train.py --all 2>&1 | tee gpu_training.log

echo ""
echo "============================================"
echo "  Training complete! $(date)"
echo "  Results saved to data/china/processed/transformer_results.csv"
echo "  Full log: gpu_training.log"
echo "============================================"
