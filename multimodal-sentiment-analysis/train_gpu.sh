#!/bin/bash
# Anti-Overfitting V3 Training Script - Run this in a regular terminal (not IDE sandbox)
# Usage: bash train_gpu.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  Checking GPU availability..."
echo "=============================================="
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA is not available. Please check your GPU setup."
    exit 1
fi

echo ""
echo "=============================================="
echo "  Anti-Overfitting V3 - Multimodal Sentiment"
echo "  =============================================="
echo "  3 Datasets Merged (MVSA-Single + Twitter2015 + Twitter2017)"
echo ""
echo "  Image-Specific Anti-Overfitting:"
echo "    - Image dropout: 0.5 (vs text dropout: 0.2)"
echo "    - Image encoder LR: 1e-6 (vs text LR: 2e-6)"
echo "    - Image weight decay: 5e-4 (vs 1e-4)"
echo "    - Feature noise injection (scale=0.15, prob=0.5)"
echo "    - DropPath rate: 0.15"
echo ""
echo "  Enhanced Image Augmentation:"
echo "    - Cutout (40px, prob=0.4)"
echo "    - RandomErasing (tensor-level, prob=0.3)"
echo "    - MixUp (alpha=0.2, prob=0.3)"
echo "    - Horizontal flip (prob=0.5)"
echo "    - Color jitter (wider range)"
echo "    - Histogram equalization (prob=0.5)"
echo "    - Solarize & Posterize"
echo ""
echo "  Training Strategy:"
echo "    - Gradient accumulation: 2 steps"
echo "    - Weighted sampling for class imbalance"
echo "    - Label smoothing: 0.1"
echo "    - Contrastive weight: 0.2 (reduced from 0.3)"
echo "    - Early stop patience: 8"
echo "    - Max epochs: 30"
echo "=============================================="
echo ""

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 scripts/run_training.py --config config_gpu.json --model_name anti_overfit_v3

echo ""
echo "=============================================="
echo "  Training Complete!"
echo "=============================================="
