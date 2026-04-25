#!/bin/bash
# V5 Large Models Training Script - Run this in a regular terminal (not IDE sandbox)
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
echo "  V5 Large Models - Multimodal Sentiment"
echo "=============================================="
echo "  3 Datasets Merged (MVSA-Single + Twitter2015 + Twitter2017)"
echo ""
echo "  Model Upgrades:"
echo "    Text: RoBERTa-base (125M) -> RoBERTa-large (355M)"
echo "      - hidden_size: 768 -> 1024"
echo "      - num_layers: 12 -> 24"
echo "    Image: CLIP-ViT-Base-patch32 (86M) -> CLIP-ViT-Large-patch14 (307M)"
echo "      - num_layers: 12 -> 24"
echo "      - patch_size: 32 -> 14 (finer granularity)"
echo "      - tokens: 50 -> 257 (5x more visual tokens)"
echo ""
echo "  Architecture:"
echo "    projection_dim: 512 -> 768"
echo "    attention_dim: 512 -> 768"
echo "    num_attention_heads: 8 -> 12"
echo "    cross_attn_layers: 4 -> 6"
echo ""
echo "  Training Strategy (adjusted for large models):"
echo "    Cross-Attention:"
echo "      - encoder_lr=8e-7, image_encoder_lr=4e-7"
echo "      - warmup_ratio=0.2, epochs=50, patience=12"
echo "      - grad_accum=4, batch_size=8 (effective=32)"
echo "    Text-Only:"
echo "      - encoder_lr=2e-6"
echo "      - warmup_ratio=0.1, epochs=30, patience=8"
echo "      - grad_accum=2, batch_size=8 (effective=16)"
echo "    Image-Only:"
echo "      - encoder_lr=4e-7"
echo "      - warmup_ratio=0.15, epochs=40, patience=10"
echo "      - grad_accum=4, batch_size=8 (effective=32)"
echo ""
echo "  Regularization:"
echo "    text_dropout=0.15, image_dropout=0.4"
echo "    DropPath=0.2, contrastive_weight=0.15"
echo "    Reliability Gate: enabled"
echo "=============================================="
echo ""

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 scripts/run_training.py --config config_gpu.json --model_name v5_large

echo ""
echo "=============================================="
echo "  Training Complete!"
echo "=============================================="
