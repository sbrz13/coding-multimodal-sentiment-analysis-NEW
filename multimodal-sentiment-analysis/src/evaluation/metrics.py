import time
import json
import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.metrics = config.eval_config["metrics"]
        self.results_dir = config.eval_config["results_dir"]
        os.makedirs(self.results_dir, exist_ok=True)

    def evaluate(self, y_true, y_pred, model_name=None):
        results = {}

        if "accuracy" in self.metrics:
            results["accuracy"] = float(accuracy_score(y_true, y_pred))

        if "f1" in self.metrics:
            results["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            results["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
            results["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        if "precision" in self.metrics:
            results["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
            results["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))

        if "recall" in self.metrics:
            results["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
            results["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

        results["classification_report"] = classification_report(
            y_true, y_pred, target_names=["negative", "neutral", "positive"], output_dict=True, zero_division=0
        )

        if self.config.eval_config["save_results"] and model_name:
            self.save_results(results, model_name)

        return results

    def evaluate_with_inference_time(self, model, data_loader, device, model_type="cross_attention"):
        model.to(device)
        model.eval()

        y_true = []
        y_pred = []
        total_inference_time = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["label"]

                start_time = time.time()

                if model_type == "cross_attention":
                    logits, _, _, _ = model(input_ids, attention_mask, pixel_values)
                elif model_type == "text_only":
                    logits, _ = model(input_ids, attention_mask)
                elif model_type == "image_only":
                    logits, _ = model(pixel_values)

                end_time = time.time()
                total_inference_time += (end_time - start_time)

                predictions = torch.argmax(logits, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

        results = self.evaluate(y_true, y_pred)
        avg_inference_time = total_inference_time / len(data_loader)
        results["inference_time"] = {
            "total": total_inference_time,
            "average_per_batch": avg_inference_time,
            "per_sample": avg_inference_time / data_loader.batch_size,
        }

        return results, y_true, y_pred

    def save_results(self, results, model_name):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_name = f"{model_name}_eval_{timestamp}.json"
        file_path = os.path.join(self.results_dir, file_name)

        serializable = {}
        for k, v in results.items():
            if isinstance(v, np.floating):
                serializable[k] = float(v)
            elif isinstance(v, np.integer):
                serializable[k] = int(v)
            else:
                serializable[k] = v

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=4, ensure_ascii=False)

    def compare_models(self, model_results):
        comparison = {}
        metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

        for metric in metrics:
            if metric in list(model_results.values())[0]:
                metric_values = {model: results[metric] for model, results in model_results.items()}
                best_model = max(metric_values, key=metric_values.get)
                comparison[metric] = {
                    "values": metric_values,
                    "best_model": best_model,
                    "best_value": metric_values[best_model],
                }

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_name = f"model_comparison_{timestamp}.json"
        file_path = os.path.join(self.results_dir, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=4, ensure_ascii=False)

        return comparison

    def print_results(self, results, model_name="Model"):
        print(f"\n{'='*50}")
        print(f"  {model_name} Evaluation Results")
        print(f"{'='*50}")
        print(f"  Accuracy:  {results.get('accuracy', 0):.4f}")
        print(f"  F1 (macro): {results.get('f1_macro', 0):.4f}")
        print(f"  F1 (weighted): {results.get('f1_weighted', 0):.4f}")
        print(f"  Precision (macro): {results.get('precision_macro', 0):.4f}")
        print(f"  Recall (macro): {results.get('recall_macro', 0):.4f}")
        if "inference_time" in results:
            print(f"  Avg inference per sample: {results['inference_time']['per_sample']*1000:.2f} ms")
        print(f"{'='*50}")
