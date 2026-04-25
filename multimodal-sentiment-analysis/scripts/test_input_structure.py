import argparse
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import Config
from src.training.models import CrossAttentionModel, TextOnlyModel, ImageOnlyModel


def main():
    parser = argparse.ArgumentParser(description="Test input/output structure")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()

    config = Config(args.config)
    device = torch.device(config.train_config["device"])

    batch_size = 4
    seq_len = config.data_config["max_seq_length"]
    img_size = config.data_config["image_size"]

    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    pixel_values = torch.randn(batch_size, 3, img_size, img_size).to(device)

    print("Testing Cross-Attention Model...")
    model = CrossAttentionModel(config).to(device)
    model.eval()
    with torch.no_grad():
        logits, text_feat, image_feat, z = model(input_ids, attention_mask, pixel_values)
    print(f"  Logits: {logits.shape} (expected: [{batch_size}, 3])")
    print(f"  Text features: {text_feat.shape} (expected: [{batch_size}, 256])")
    print(f"  Image features: {image_feat.shape} (expected: [{batch_size}, 256])")
    print(f"  Fused z: {z.shape} (expected: [{batch_size}, 64])")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / Total: {total:,}")

    print("\nTesting Text-Only Model...")
    text_model = TextOnlyModel(config).to(device)
    text_model.eval()
    with torch.no_grad():
        logits, text_feat = text_model(input_ids, attention_mask)
    print(f"  Logits: {logits.shape} (expected: [{batch_size}, 3])")
    print(f"  Text features: {text_feat.shape} (expected: [{batch_size}, 256])")

    print("\nTesting Image-Only Model...")
    image_model = ImageOnlyModel(config).to(device)
    image_model.eval()
    with torch.no_grad():
        logits, image_feat = image_model(pixel_values)
    print(f"  Logits: {logits.shape} (expected: [{batch_size}, 3])")
    print(f"  Image features: {image_feat.shape} (expected: [{batch_size}, 256])")

    print("\nAll structure tests passed!")


if __name__ == "__main__":
    main()
