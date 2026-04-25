import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoModel
from src.config.config import Config
from src.training.models import TextEncoder, ImageEncoder

config = Config("config_gpu.json")

print("=" * 60)
print("验证 RoBERTa 预训练权重加载")
print("=" * 60)

# 加载官方预训练模型作为参考
print("\n1. 加载官方 roberta-base 预训练模型...")
official_roberta = AutoModel.from_pretrained(config.model_config["text_model"])

# 加载我们的 TextEncoder
print("\n2. 加载我们的 TextEncoder...")
text_encoder = TextEncoder(config)

# 比较核心参数
print("\n3. 比较核心参数是否一致：")
params_to_check = [
    "embeddings.word_embeddings.weight",
    "embeddings.position_embeddings.weight",
    "encoder.layer.0.attention.self.query.weight",
    "encoder.layer.0.attention.self.query.bias",
    "encoder.layer.11.output.LayerNorm.weight",
]

all_match = True
for param_name in params_to_check:
    official_param = official_roberta.state_dict()[param_name]
    our_param = text_encoder.roberta.state_dict()[param_name]
    
    is_close = torch.allclose(official_param, our_param, atol=1e-6)
    status = "✅ 匹配" if is_close else "❌ 不匹配"
    print(f"   {param_name}: {status}")
    
    if not is_close:
        all_match = False
        print(f"      差异: max_diff = {(official_param - our_param).abs().max().item()}")

print("\n" + "=" * 60)
print("验证 ViT 预训练权重加载")
print("=" * 60)

# 加载官方预训练模型作为参考
print("\n1. 加载官方 google/vit-base-patch16-224 预训练模型...")
official_vit = AutoModel.from_pretrained(config.model_config["vision_model"])

# 加载我们的 ImageEncoder
print("\n2. 加载我们的 ImageEncoder...")
image_encoder = ImageEncoder(config)

# 比较核心参数
print("\n3. 比较核心参数是否一致：")
vit_params_to_check = [
    "embeddings.patch_embeddings.projection.weight",
    "embeddings.position_embeddings",
    "encoder.layer.0.attention.attention.query.weight",
    "encoder.layer.0.attention.attention.query.bias",
    "encoder.layer.11.layernorm_after.weight",
]

vit_all_match = True
for param_name in vit_params_to_check:
    official_param = official_vit.state_dict()[param_name]
    our_param = image_encoder.vit.state_dict()[param_name]
    
    is_close = torch.allclose(official_param, our_param, atol=1e-6)
    status = "✅ 匹配" if is_close else "❌ 不匹配"
    print(f"   {param_name}: {status}")
    
    if not is_close:
        vit_all_match = False
        print(f"      差异: max_diff = {(official_param - our_param).abs().max().item()}")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
if all_match and vit_all_match:
    print("✅ 所有预训练权重都正确加载！")
    print(f"   - RoBERTa 参数总数: {sum(p.numel() for p in text_encoder.roberta.parameters()):,}")
    print(f"   - ViT 参数总数: {sum(p.numel() for p in image_encoder.vit.parameters()):,}")
else:
    print("❌ 存在预训练权重未正确加载的情况！")
    if not all_match:
        print("   - RoBERTa 部分参数不匹配")
    if not vit_all_match:
        print("   - ViT 部分参数不匹配")
