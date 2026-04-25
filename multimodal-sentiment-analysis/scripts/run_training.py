import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import Config
from src.data_processing.data_loader import get_data_loaders
from src.training.models import CrossAttentionModel, TextOnlyModel, ImageOnlyModel
from src.evaluation.metrics import Evaluator


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_optimizer(model, config, model_type="cross_attention"):
    encoder_lr = config.train_config.get("encoder_learning_rate", 2e-6)
    image_encoder_lr = config.train_config.get("image_encoder_learning_rate", encoder_lr)
    fusion_lr = config.train_config.get("fusion_learning_rate", 5e-5)
    classifier_lr = config.train_config.get("classifier_learning_rate", 1e-4)
    weight_decay = config.train_config["weight_decay"]
    image_weight_decay = config.train_config.get("image_weight_decay", weight_decay)

    text_encoder_params = []
    image_encoder_params = []
    fusion_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "roberta" in name:
            text_encoder_params.append(param)
        elif "vit" in name:
            image_encoder_params.append(param)
        elif "classifier" in name:
            classifier_params.append(param)
        else:
            fusion_params.append(param)

    param_groups = []
    if text_encoder_params:
        param_groups.append({"params": text_encoder_params, "lr": encoder_lr, "weight_decay": weight_decay})
    if image_encoder_params:
        param_groups.append({"params": image_encoder_params, "lr": image_encoder_lr, "weight_decay": image_weight_decay})
    if fusion_params:
        param_groups.append({"params": fusion_params, "lr": fusion_lr, "weight_decay": weight_decay})
    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": classifier_lr, "weight_decay": weight_decay})

    if not param_groups:
        param_groups = [{"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": config.train_config["learning_rate"], "weight_decay": weight_decay}]

    optimizer = AdamW(param_groups)
    return optimizer


def mixup_criterion(criterion, logits, labels, mix_lambda=None, label_secondary=None):
    if mix_lambda is not None and label_secondary is not None:
        ce_loss_no_smooth = nn.CrossEntropyLoss(weight=criterion.weight, reduction='none')
        loss_primary = ce_loss_no_smooth(logits, labels)
        loss_secondary = ce_loss_no_smooth(logits, label_secondary)
        if isinstance(mix_lambda, torch.Tensor):
            mix_lambda = mix_lambda.view(-1)
        loss = mix_lambda * loss_primary + (1 - mix_lambda) * loss_secondary
        return loss.mean()
    return criterion(logits, labels)


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, model_type="cross_attention", scaler=None, label_smoothing=0.0, grad_accum_steps=1):
    model.train()
    total_loss = 0
    ce_loss_total = 0
    cl_loss_total = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        has_mixup = "mix_lambda" in batch
        mix_lambda = batch.get("mix_lambda", None)
        label_secondary = batch.get("label_secondary", None)
        if mix_lambda is not None:
            mix_lambda = mix_lambda.to(device)
        if label_secondary is not None:
            label_secondary = label_secondary.to(device)

        use_amp = scaler is not None
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            if model_type == "cross_attention":
                logits, text_feat, image_feat, z_head = model(input_ids, attention_mask, pixel_values)
                ce_loss = mixup_criterion(criterion, logits, labels, mix_lambda, label_secondary)
                cl_loss = model.contrastive_loss(text_feat, image_feat)
                total_loss_batch = ce_loss + model.contrastive_weight * cl_loss
            elif model_type == "text_only":
                logits, _ = model(input_ids, attention_mask)
                ce_loss = mixup_criterion(criterion, logits, labels, mix_lambda, label_secondary)
                total_loss_batch = ce_loss
                cl_loss = torch.tensor(0.0, device=device)
            elif model_type == "image_only":
                logits, _ = model(pixel_values)
                ce_loss = mixup_criterion(criterion, logits, labels, mix_lambda, label_secondary)
                total_loss_batch = ce_loss
                cl_loss = torch.tensor(0.0, device=device)

        total_loss_batch = total_loss_batch / grad_accum_steps

        if use_amp:
            scaler.scale(total_loss_batch).backward()
            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        else:
            total_loss_batch.backward()
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        total_loss += total_loss_batch.item() * grad_accum_steps * labels.size(0)
        ce_loss_total += ce_loss.item() * labels.size(0)
        cl_loss_total += cl_loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    avg_ce = ce_loss_total / total
    avg_cl = cl_loss_total / total
    accuracy = correct / total
    return avg_loss, accuracy, avg_ce, avg_cl


@torch.no_grad()
def evaluate_model(model, data_loader, criterion, device, model_type="cross_attention"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        if model_type == "cross_attention":
            logits, _, _, _ = model(input_ids, attention_mask, pixel_values)
        elif model_type == "text_only":
            logits, _ = model(input_ids, attention_mask)
        elif model_type == "image_only":
            logits, _ = model(pixel_values)

        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, y_true, y_pred


@torch.no_grad()
def extract_probabilities(cross_attn_model, text_model, image_model, data_loader, device):
    cross_attn_model.eval()
    text_model.eval()
    image_model.eval()

    all_probs = []
    all_labels = []

    for batch in tqdm(data_loader, desc="Extracting probabilities", leave=False):
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
    return all_probs, all_labels


def train_base_model(model, model_name, model_type, train_loader, val_loader, criterion, device, config, model_dir, args):
    optimizer = build_optimizer(model, config, model_type)

    grad_accum_steps = config.train_config.get("gradient_accumulation_steps", 1)

    total_steps = len(train_loader) * config.train_config["num_epochs"] // grad_accum_steps
    warmup_steps = int(total_steps * config.train_config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    label_smoothing = config.train_config.get("label_smoothing", 0.0)

    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0
    patience = config.train_config.get("early_stop_patience", 7)

    for epoch in range(config.train_config["num_epochs"]):
        train_loss, train_acc, avg_ce, avg_cl = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, model_type, scaler, label_smoothing, grad_accum_steps
        )
        val_loss, val_acc, val_true, val_pred = evaluate_model(model, val_loader, criterion, device, model_type)

        val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)

        loss_info = f"ce={avg_ce:.4f}"
        if model_type == "cross_attention":
            loss_info += f", cl={avg_cl:.4f}"

        gap = train_acc - val_acc
        print(f"  Epoch {epoch+1}/{config.train_config['num_epochs']}: "
              f"train_loss={train_loss:.4f} ({loss_info}), train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, "
              f"gap={gap:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_dir, f"{args.model_name}_{model_name}_best.pt"))
            print(f"    -> Best model saved (val_f1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    -> Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    torch.save(model.state_dict(), os.path.join(model_dir, f"{args.model_name}_{model_name}_last.pt"))
    print(f"  Best val_acc={best_val_acc:.4f}, Best val_f1={best_val_f1:.4f}")
    return best_val_acc, best_val_f1


def main():
    parser = argparse.ArgumentParser(description="Two-stage training for multimodal sentiment analysis")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--model_name", type=str, default="multimodal_sentiment", help="Model name prefix")
    args = parser.parse_args()

    config = Config(args.config)
    device = torch.device(config.train_config["device"])
    set_seed(config.train_config["seed"])

    use_merged_data = config.data_config.get("use_merged_data", True)
    label_smoothing = config.train_config.get("label_smoothing", 0.0)
    use_weighted_sampling = config.data_config.get("use_weighted_sampling", False)
    use_mixup = config.data_config.get("use_mixup", False)
    grad_accum_steps = config.train_config.get("gradient_accumulation_steps", 1)
    image_encoder_lr = config.train_config.get("image_encoder_learning_rate", "N/A")
    image_weight_decay = config.train_config.get("image_weight_decay", "N/A")
    image_dropout = config.model_config.get("image_dropout_rate", "N/A")
    text_dropout = config.model_config.get("text_dropout_rate", "N/A")
    drop_path_rate = config.model_config.get("drop_path_rate", 0)
    image_noise_scale = config.model_config.get("image_noise_scale", 0)
    image_noise_prob = config.model_config.get("image_noise_prob", 0)

    print("=" * 60)
    print("  Multimodal Sentiment Analysis - Anti-Overfitting V3")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Text model: {config.model_config['text_model']}")
    print(f"  Vision model: {config.model_config['vision_model']}")
    print(f"  Projection dim: {config.model_config['projection_dim']}")
    print(f"  Attention dim: {config.model_config['attention_dim']}")
    print(f"  Num attention heads: {config.model_config['num_attention_heads']}")
    print(f"  Cross-attention layers: {config.model_config.get('cross_attn_layers', 4)}")
    print(f"  Freeze encoders: {config.model_config['freeze_encoders']}")
    print(f"  Contrastive weight: {config.model_config.get('contrastive_weight', 0.3)}")
    print(f"  Label smoothing: {label_smoothing}")
    print(f"  Use merged data: {use_merged_data}")
    print(f"  Data augmentation: {config.data_config.get('augment', True)}")
    print(f"  Weighted sampling: {use_weighted_sampling}")
    print(f"  MixUp augmentation: {use_mixup}")
    print(f"  Epochs: {config.train_config['num_epochs']}")
    print(f"  Text encoder LR: {config.train_config.get('encoder_learning_rate', 2e-6)}")
    print(f"  Image encoder LR: {image_encoder_lr}")
    print(f"  Fusion LR: {config.train_config.get('fusion_learning_rate', 5e-5)}")
    print(f"  Classifier LR: {config.train_config.get('classifier_learning_rate', 1e-4)}")
    print(f"  Image weight decay: {image_weight_decay}")
    print(f"  Gradient accumulation: {grad_accum_steps}")
    print(f"  Text dropout: {text_dropout}")
    print(f"  Image dropout: {image_dropout}")
    print(f"  Drop path rate: {drop_path_rate}")
    print(f"  Image noise scale: {image_noise_scale}")
    print(f"  Image noise prob: {image_noise_prob}")
    print(f"  Batch size: {config.data_config['batch_size']}")
    print(f"  Early stop patience: {config.train_config.get('early_stop_patience', 7)}")
    print(f"  Mixed precision (AMP): {device.type == 'cuda'}")
    print("=" * 60)

    train_loader, val_loader, test_loader, class_weights, per_dataset_test_loaders = get_data_loaders(config)
    class_weights = class_weights.to(device)

    if label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"\nDataset sizes: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    if per_dataset_test_loaders:
        for ds_name, ds_loader in per_dataset_test_loaders.items():
            print(f"  {ds_name} test: {len(ds_loader.dataset)}")
    print(f"Class weights: {class_weights.cpu().numpy()}")

    results_dir = config.eval_config["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    model_dir = os.path.join(results_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    evaluator = Evaluator(config)

    print("\n" + "=" * 60)
    print("  STAGE 1: Training Base Models")
    print("=" * 60)

    print("\n--- Training Cross-Attention Model ---")
    cross_attn_model = CrossAttentionModel(config).to(device)
    trainable_params = sum(p.numel() for p in cross_attn_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in cross_attn_model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / Total: {total_params:,}")
    ca_acc, ca_f1 = train_base_model(cross_attn_model, "cross_attn", "cross_attention",
                                      train_loader, val_loader, criterion, device, config, model_dir, args)

    print("\n--- Training Text-Only Model ---")
    text_model = TextOnlyModel(config).to(device)
    trainable_params = sum(p.numel() for p in text_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in text_model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / Total: {total_params:,}")
    text_acc, text_f1 = train_base_model(text_model, "text_only", "text_only",
                                          train_loader, val_loader, criterion, device, config, model_dir, args)

    print("\n--- Training Image-Only Model ---")
    image_model = ImageOnlyModel(config).to(device)
    trainable_params = sum(p.numel() for p in image_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in image_model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / Total: {total_params:,}")
    image_acc, image_f1 = train_base_model(image_model, "image_only", "image_only",
                                            train_loader, val_loader, criterion, device, config, model_dir, args)

    print("\n--- Loading best base models for Stage 2 ---")
    cross_attn_model.load_state_dict(torch.load(os.path.join(model_dir, f"{args.model_name}_cross_attn_best.pt"), map_location=device, weights_only=True))
    text_model.load_state_dict(torch.load(os.path.join(model_dir, f"{args.model_name}_text_only_best.pt"), map_location=device, weights_only=True))
    image_model.load_state_dict(torch.load(os.path.join(model_dir, f"{args.model_name}_image_only_best.pt"), map_location=device, weights_only=True))

    for param in cross_attn_model.parameters():
        param.requires_grad = False
    for param in text_model.parameters():
        param.requires_grad = False
    for param in image_model.parameters():
        param.requires_grad = False

    print("\n" + "=" * 60)
    print("  STAGE 2: Training Meta-Learner on Validation Set")
    print("=" * 60)

    val_probs, val_labels = extract_probabilities(cross_attn_model, text_model, image_model, val_loader, device)
    print(f"Validation probability matrix shape: {val_probs.shape}")

    meta_learner = LogisticRegression(
        C=config.meta_config["meta_learner_C"],
        multi_class=config.meta_config["meta_learner_multi_class"],
        max_iter=config.meta_config["meta_learner_max_iter"],
        random_state=config.train_config["seed"],
    )
    meta_learner.fit(val_probs, val_labels)

    meta_path = os.path.join(model_dir, f"{args.model_name}_meta_learner.pkl")
    joblib.dump(meta_learner, meta_path)
    print(f"Meta-learner saved to: {meta_path}")

    print("\n" + "=" * 60)
    print("  FINAL EVALUATION ON MERGED TEST SET")
    print("=" * 60)

    test_probs, test_labels = extract_probabilities(cross_attn_model, text_model, image_model, test_loader, device)
    meta_predictions = meta_learner.predict(test_probs)

    meta_results = evaluator.evaluate(test_labels.tolist(), meta_predictions.tolist(), f"{args.model_name}_meta_learner")
    evaluator.print_results(meta_results, "Meta-Learner (Final)")

    print("\n--- Individual Base Model Results (Merged Test) ---")

    ca_loss, ca_acc, ca_true, ca_pred = evaluate_model(cross_attn_model, test_loader, criterion, device, "cross_attention")
    ca_results = evaluator.evaluate(ca_true, ca_pred, f"{args.model_name}_cross_attention")
    evaluator.print_results(ca_results, "Cross-Attention Model")

    text_loss, text_acc, text_true, text_pred = evaluate_model(text_model, test_loader, criterion, device, "text_only")
    text_results = evaluator.evaluate(text_true, text_pred, f"{args.model_name}_text_only")
    evaluator.print_results(text_results, "Text-Only Model")

    image_loss, image_acc, image_true, image_pred = evaluate_model(image_model, test_loader, criterion, device, "image_only")
    image_results = evaluator.evaluate(image_true, image_pred, f"{args.model_name}_image_only")
    evaluator.print_results(image_results, "Image-Only Model")

    if per_dataset_test_loaders:
        print("\n" + "=" * 60)
        print("  PER-DATASET TEST EVALUATION")
        print("=" * 60)

        for ds_name, ds_loader in per_dataset_test_loaders.items():
            print(f"\n--- {ds_name} Test Set ---")
            ca_loss, ca_acc, ca_true, ca_pred = evaluate_model(cross_attn_model, ds_loader, criterion, device, "cross_attention")
            ca_results = evaluator.evaluate(ca_true, ca_pred, f"{ds_name}_cross_attention")
            evaluator.print_results(ca_results, f"Cross-Attention ({ds_name})")

            text_loss, text_acc, text_true, text_pred = evaluate_model(text_model, ds_loader, criterion, device, "text_only")
            text_results = evaluator.evaluate(text_true, text_pred, f"{ds_name}_text_only")
            evaluator.print_results(text_results, f"Text-Only ({ds_name})")

            image_loss, image_acc, image_true, image_pred = evaluate_model(image_model, ds_loader, criterion, device, "image_only")
            image_results = evaluator.evaluate(image_true, image_pred, f"{ds_name}_image_only")
            evaluator.print_results(image_results, f"Image-Only ({ds_name})")

            ds_probs, ds_labels = extract_probabilities(cross_attn_model, text_model, image_model, ds_loader, device)
            ds_meta_pred = meta_learner.predict(ds_probs)
            ds_meta_results = evaluator.evaluate(ds_labels.tolist(), ds_meta_pred.tolist(), f"{ds_name}_meta_learner")
            evaluator.print_results(ds_meta_results, f"Meta-Learner ({ds_name})")

    print("\n--- Model Comparison (Merged Test) ---")
    model_comparison = {
        "Cross-Attention": ca_results,
        "Text-Only": text_results,
        "Image-Only": image_results,
        "Meta-Learner": meta_results,
    }
    comparison = evaluator.compare_models(model_comparison)

    config_path = os.path.join(model_dir, f"{args.model_name}_config.json")
    config.save_config(config_path)
    print(f"\nConfig saved to: {config_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
