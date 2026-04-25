import argparse
import os
import sys
import numpy as np
import torch
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import Config
from src.data_processing.data_loader import get_data_loaders
from src.training.models import CrossAttentionModel, TextOnlyModel, ImageOnlyModel
from src.evaluation.metrics import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate multimodal sentiment analysis model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--model_dir", type=str, default="results/models", help="Directory containing model files")
    parser.add_argument("--model_name", type=str, default="multimodal_sentiment", help="Model name prefix")
    args = parser.parse_args()

    config = Config(args.config)
    device = torch.device(config.train_config["device"])

    print("=" * 60)
    print("  Multimodal Sentiment Analysis - Evaluation")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Model directory: {args.model_dir}")
    print("=" * 60)

    _, val_loader, test_loader, class_weights = get_data_loaders(config)
    evaluator = Evaluator(config)

    # Load models
    cross_attn_path = os.path.join(args.model_dir, f"{args.model_name}_cross_attn_best.pt")
    text_only_path = os.path.join(args.model_dir, f"{args.model_name}_text_only_best.pt")
    image_only_path = os.path.join(args.model_dir, f"{args.model_name}_image_only_best.pt")
    meta_learner_path = os.path.join(args.model_dir, f"{args.model_name}_meta_learner.pkl")

    cross_attn_model = CrossAttentionModel(config).to(device)
    cross_attn_model.load_state_dict(torch.load(cross_attn_path, map_location=device))
    cross_attn_model.eval()

    text_model = TextOnlyModel(config).to(device)
    text_model.load_state_dict(torch.load(text_only_path, map_location=device))
    text_model.eval()

    image_model = ImageOnlyModel(config).to(device)
    image_model.load_state_dict(torch.load(image_only_path, map_location=device))
    image_model.eval()

    meta_learner = joblib.load(meta_learner_path)

    # Extract test probabilities
    print("\nExtracting probabilities from test set...")
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"]

            p_joint = cross_attn_model.get_probabilities(input_ids, attention_mask, pixel_values)
            p_text = text_model.get_probabilities(input_ids, attention_mask)
            p_image = image_model.get_probabilities(pixel_values)

            p_all = torch.cat([p_joint, p_text, p_image], dim=1)
            all_probs.append(p_all.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)

    # Meta-learner predictions
    meta_predictions = meta_learner.predict(all_probs)
    meta_results = evaluator.evaluate(all_labels.tolist(), meta_predictions.tolist(), f"{args.model_name}_meta_eval")
    evaluator.print_results(meta_results, "Meta-Learner")

    # Individual model evaluations
    print("\n--- Individual Model Evaluations ---")

    y_true_ca, y_pred_ca = [], []
    y_true_text, y_pred_text = [], []
    y_true_image, y_pred_image = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"]

            logits_ca, _, _, _ = cross_attn_model(input_ids, attention_mask, pixel_values)
            logits_text, _ = text_model(input_ids, attention_mask)
            logits_image, _ = image_model(pixel_values)

            y_true_ca.extend(labels.numpy())
            y_pred_ca.extend(torch.argmax(logits_ca, dim=1).cpu().numpy())
            y_true_text.extend(labels.numpy())
            y_pred_text.extend(torch.argmax(logits_text, dim=1).cpu().numpy())
            y_true_image.extend(labels.numpy())
            y_pred_image.extend(torch.argmax(logits_image, dim=1).cpu().numpy())

    ca_results = evaluator.evaluate(y_true_ca, y_pred_ca, f"{args.model_name}_cross_attn_eval")
    evaluator.print_results(ca_results, "Cross-Attention Model")

    text_results = evaluator.evaluate(y_true_text, y_pred_text, f"{args.model_name}_text_only_eval")
    evaluator.print_results(text_results, "Text-Only Model")

    image_results = evaluator.evaluate(y_true_image, y_pred_image, f"{args.model_name}_image_only_eval")
    evaluator.print_results(image_results, "Image-Only Model")

    # Compare all models
    print("\n--- Model Comparison ---")
    model_comparison = {
        "Cross-Attention": ca_results,
        "Text-Only": text_results,
        "Image-Only": image_results,
        "Meta-Learner": meta_results,
    }
    comparison = evaluator.compare_models(model_comparison)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
