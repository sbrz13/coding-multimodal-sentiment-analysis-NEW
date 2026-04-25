import argparse
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import Config
from src.training.models import CrossAttentionModel, TextOnlyModel, ImageOnlyModel
from transformers import AutoTokenizer, AutoImageProcessor
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Test single sample inference")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()

    config = Config(args.config)
    device = torch.device(config.train_config["device"])

    cross_attn_model = CrossAttentionModel(config).to(device)
    text_model = TextOnlyModel(config).to(device)
    image_model = ImageOnlyModel(config).to(device)

    cross_attn_model.eval()
    text_model.eval()
    image_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_config["text_model"])
    image_processor = AutoImageProcessor.from_pretrained(config.model_config["vision_model"])

    test_text = "I love this product! It's amazing."
    test_image = Image.new("RGB", (224, 224), color="red")

    text_inputs = tokenizer(
        test_text,
        max_length=config.data_config["max_seq_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    image_inputs = image_processor(test_image, return_tensors="pt")

    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)
    pixel_values = image_inputs["pixel_values"].to(device)

    print("Testing Cross-Attention Model...")
    with torch.no_grad():
        logits, text_feat, image_feat, z = cross_attn_model(input_ids, attention_mask, pixel_values)
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=1)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Text feature shape: {text_feat.shape}")
    print(f"  Image feature shape: {image_feat.shape}")
    print(f"  Fused representation shape: {z.shape}")
    print(f"  Probabilities: {probs.cpu().numpy()}")
    print(f"  Prediction: {pred.item()}")

    print("\nTesting Text-Only Model...")
    with torch.no_grad():
        logits, text_feat = text_model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=1)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Probabilities: {probs.cpu().numpy()}")
    print(f"  Prediction: {pred.item()}")

    print("\nTesting Image-Only Model...")
    with torch.no_grad():
        logits, image_feat = image_model(pixel_values)
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=1)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Probabilities: {probs.cpu().numpy()}")
    print(f"  Prediction: {pred.item()}")

    print("\nAll model tests passed!")


if __name__ == "__main__":
    main()
