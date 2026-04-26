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


MODEL_STRATEGIES = {
    "cross_attention": {
        "warmup_ratio": 0.2,
        "learning_rate": 3e-6,
        "encoder_learning_rate": 5e-7,
        "image_encoder_learning_rate": 2e-7,
        "fusion_learning_rate": 2e-5,
        "classifier_learning_rate": 6e-5,
        "weight_decay": 1e-4,
        "image_weight_decay": 3e-4,
        "label_smoothing": 0.1,
        "num_epochs": 50,
        "early_stop_patience": 12,
        "gradient_accumulation_steps": 8,
        "use_weighted_sampling": True,
        "use_mixup": True,
    },
    "text_only": {
        "warmup_ratio": 0.1,
        "learning_rate": 8e-6,
        "encoder_learning_rate": 1e-6,
        "fusion_learning_rate": 5e-5,
        "classifier_learning_rate": 1.5e-4,
        "weight_decay": 0.01,
        "image_weight_decay": 0.01,
        "label_smoothing": 0.05,
        "num_epochs": 30,
        "early_stop_patience": 8,
        "gradient_accumulation_steps": 4,
        "use_weighted_sampling": False,
        "use_mixup": False,
    },
    "image_only": {
        "warmup_ratio": 0.15,
        "learning_rate": 3e-6,
        "encoder_learning_rate": 2e-7,
        "image_encoder_learning_rate": 2e-7,
        "fusion_learning_rate": 2e-5,
        "classifier_learning_rate": 6e-5,
        "weight_decay": 1e-4,
        "image_weight_decay": 3e-4,
        "label_smoothing": 0.1,
        "num_epochs": 40,
        "early_stop_patience": 10,
        "gradient_accumulation_steps": 8,
        "use_weighted_sampling": True,
        "use_mixup": True,
    },
}


def build_optimizer(model, config, model_type="cross_attention"):
    strategy = MODEL_STRATEGIES[model_type]

    encoder_lr = strategy["encoder_learning_rate"]
    image_encoder_lr = strategy.get("image_encoder_learning_rate", encoder_lr)
    fusion_lr = strategy["fusion_learning_rate"]
    classifier_lr = strategy["classifier_learning_rate"]
    weight_decay = strategy["weight_decay"]
    image_weight_decay = strategy["image_weight_decay"]

    text_encoder_params = []
    image_encoder_params = []
    fusion_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "roberta" in name or "text_encoder" in name:
            text_encoder_params.append(param)
        elif "vit" in name or "clip" in name or "image_encoder" in name:
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
        param_groups = [{"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": strategy["learning_rate"], "weight_decay": weight_decay}]

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


def get_model_data_loaders(config, model_type):
    strategy = MODEL_STRATEGIES[model_type]
    use_weighted_sampling = strategy["use_weighted_sampling"]
    use_mixup = strategy["use_mixup"]

    from src.data_processing.data_loader import (
        get_data_loaders as _get_data_loaders,
        MultimodalSentimentDataset,
        MixUpCollator,
        load_json_data,
        compute_class_weights_from_data,
        LABEL_MAP,
    )
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import AutoTokenizer, AutoImageProcessor, CLIPImageProcessor
    from collections import Counter

    data_dir = config.data_config["data_dir"]
    batch_size = config.data_config["batch_size"]
    num_workers = config.data_config.get("num_workers", 0)
    augment = config.data_config.get("augment", True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_config["text_model"])
    vision_model_name = config.model_config["vision_model"]
    if "clip" in vision_model_name.lower():
        image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
    else:
        image_processor = AutoImageProcessor.from_pretrained(vision_model_name)

    use_merged = config.data_config.get("use_merged_data", True)

    if use_merged:
        merged_train_path = os.path.join(data_dir, "merged_train.json")
        merged_val_path = os.path.join(data_dir, "merged_val.json")
        merged_test_path = os.path.join(data_dir, "merged_test.json")

        if not os.path.exists(merged_train_path):
            raise FileNotFoundError(f"Merged data not found: {merged_train_path}")

        train_data = load_json_data(merged_train_path)
        val_data = load_json_data(merged_val_path)
        test_data = load_json_data(merged_test_path)

        train_dataset = MultimodalSentimentDataset(train_data, config, tokenizer, image_processor, augment=augment)
        val_dataset = MultimodalSentimentDataset(val_data, config, tokenizer, image_processor, augment=False)
        test_dataset = MultimodalSentimentDataset(test_data, config, tokenizer, image_processor, augment=False)

        collate_fn = MixUpCollator(alpha=0.2, prob=0.3) if use_mixup else None

        if use_weighted_sampling:
            labels_list = [str(item["label"]).lower().strip() for item in train_data]
            label_indices = []
            for l in labels_list:
                if l in LABEL_MAP:
                    label_indices.append(LABEL_MAP[l])
                else:
                    try:
                        label_indices.append(int(l))
                    except ValueError:
                        label_indices.append(1)
            class_counts = Counter(label_indices)
            weights = [1.0 / class_counts[l] for l in label_indices]
            sampler = WeightedRandomSampler(weights, len(weights))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                      num_workers=num_workers, drop_last=False, collate_fn=collate_fn)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, drop_last=False, collate_fn=collate_fn)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

        class_weights = compute_class_weights_from_data(train_data)

        per_dataset_test_loaders = {}
        for ds_name in ["mvsa_single", "twitter2015", "twitter2017"]:
            test_path = os.path.join(data_dir, f"test_{ds_name}.json")
            if os.path.exists(test_path):
                ds_data = load_json_data(test_path)
                ds_dataset = MultimodalSentimentDataset(ds_data, config, tokenizer, image_processor, augment=False)
                ds_loader = DataLoader(ds_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
                per_dataset_test_loaders[ds_name] = ds_loader

        return train_loader, val_loader, test_loader, class_weights, per_dataset_test_loaders
    else:
        train_data = load_json_data(os.path.join(data_dir, "train.json"))
        val_data = load_json_data(os.path.join(data_dir, "val.json"))
        test_data = load_json_data(os.path.join(data_dir, "test.json"))

        train_dataset = MultimodalSentimentDataset(train_data, config, tokenizer, image_processor, augment=augment)
        val_dataset = MultimodalSentimentDataset(val_data, config, tokenizer, image_processor, augment=False)
        test_dataset = MultimodalSentimentDataset(test_data, config, tokenizer, image_processor, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

        class_weights = compute_class_weights_from_data(train_data)

        return train_loader, val_loader, test_loader, class_weights, {}


def train_base_model(model, model_name, model_type, train_loader, val_loader, criterion, device, config, model_dir, args):
    strategy = MODEL_STRATEGIES[model_type]

    optimizer = build_optimizer(model, config, model_type)

    grad_accum_steps = strategy["gradient_accumulation_steps"]

    total_steps = len(train_loader) * strategy["num_epochs"] // grad_accum_steps
    warmup_steps = int(total_steps * strategy["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    label_smoothing = strategy["label_smoothing"]

    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0
    patience = strategy["early_stop_patience"]

    print(f"\n  [{model_name}] Strategy:")
    print(f"    warmup_ratio={strategy['warmup_ratio']}, epochs={strategy['num_epochs']}, "
          f"patience={patience}, grad_accum={grad_accum_steps}")
    print(f"    label_smoothing={label_smoothing}, weighted_sampling={strategy['use_weighted_sampling']}, "
          f"mixup={strategy['use_mixup']}")
    print(f"    encoder_lr={strategy['encoder_learning_rate']}, "
          f"image_encoder_lr={strategy.get('image_encoder_learning_rate', 'N/A')}, "
          f"fusion_lr={strategy['fusion_learning_rate']}, classifier_lr={strategy['classifier_learning_rate']}")
    print(f"    weight_decay={strategy['weight_decay']}, "
          f"image_weight_decay={strategy['image_weight_decay']}")

    for epoch in range(strategy["num_epochs"]):
        train_loss, train_acc, avg_ce, avg_cl = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, model_type, scaler, label_smoothing, grad_accum_steps
        )
        val_loss, val_acc, val_true, val_pred = evaluate_model(model, val_loader, criterion, device, model_type)

        val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)

        loss_info = f"ce={avg_ce:.4f}"
        if model_type == "cross_attention":
            loss_info += f", cl={avg_cl:.4f}"

        gap = train_acc - val_acc
        print(f"  Epoch {epoch+1}/{strategy['num_epochs']}: "
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

    print("=" * 60)
    print("  Multimodal Sentiment Analysis - V5 Large Models")
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
    print(f"  Use reliability gate: {config.model_config.get('use_reliability_gate', True)}")
    print(f"  Batch size: {config.data_config['batch_size']}")
    print(f"  Mixed precision (AMP): {device.type == 'cuda'}")
    print()
    print("  Per-Model Strategies:")
    for mt, s in MODEL_STRATEGIES.items():
        print(f"    {mt}: warmup={s['warmup_ratio']}, epochs={s['num_epochs']}, "
              f"lr={s['learning_rate']}, smoothing={s['label_smoothing']}, "
              f"weighted={s['use_weighted_sampling']}, mixup={s['use_mixup']}")
    print("=" * 60)

    ca_train_loader, val_loader, test_loader, class_weights, per_dataset_test_loaders = get_model_data_loaders(config, "cross_attention")

    class_weights = class_weights.to(device)

    print(f"\nDataset sizes: train={len(ca_train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
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

    ca_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=MODEL_STRATEGIES["cross_attention"]["label_smoothing"])
    ca_acc, ca_f1 = train_base_model(cross_attn_model, "cross_attn", "cross_attention",
                                      ca_train_loader, val_loader, ca_criterion, device, config, model_dir, args)

    print("\n--- Training Text-Only Model ---")
    text_train_loader, _, _, text_class_weights, _ = get_model_data_loaders(config, "text_only")
    text_class_weights = text_class_weights.to(device)
    text_model = TextOnlyModel(config).to(device)
    trainable_params = sum(p.numel() for p in text_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in text_model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / Total: {total_params:,}")

    text_criterion = nn.CrossEntropyLoss(weight=text_class_weights, label_smoothing=MODEL_STRATEGIES["text_only"]["label_smoothing"])
    text_acc, text_f1 = train_base_model(text_model, "text_only", "text_only",
                                          text_train_loader, val_loader, text_criterion, device, config, model_dir, args)

    print("\n--- Training Image-Only Model ---")
    image_train_loader, _, _, image_class_weights, _ = get_model_data_loaders(config, "image_only")
    image_class_weights = image_class_weights.to(device)
    image_model = ImageOnlyModel(config).to(device)
    trainable_params = sum(p.numel() for p in image_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in image_model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / Total: {total_params:,}")

    image_criterion = nn.CrossEntropyLoss(weight=image_class_weights, label_smoothing=MODEL_STRATEGIES["image_only"]["label_smoothing"])
    image_acc, image_f1 = train_base_model(image_model, "image_only", "image_only",
                                            image_train_loader, val_loader, image_criterion, device, config, model_dir, args)

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

    eval_criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\n" + "=" * 60)
    print("  FINAL EVALUATION ON MERGED TEST SET")
    print("=" * 60)

    test_probs, test_labels = extract_probabilities(cross_attn_model, text_model, image_model, test_loader, device)
    meta_predictions = meta_learner.predict(test_probs)

    meta_results = evaluator.evaluate(test_labels.tolist(), meta_predictions.tolist(), f"{args.model_name}_meta_learner")
    evaluator.print_results(meta_results, "Meta-Learner (Final)")

    print("\n--- Individual Base Model Results (Merged Test) ---")

    ca_loss, ca_acc, ca_true, ca_pred = evaluate_model(cross_attn_model, test_loader, eval_criterion, device, "cross_attention")
    ca_results = evaluator.evaluate(ca_true, ca_pred, f"{args.model_name}_cross_attention")
    evaluator.print_results(ca_results, "Cross-Attention Model")

    text_loss, text_acc, text_true, text_pred = evaluate_model(text_model, test_loader, eval_criterion, device, "text_only")
    text_results = evaluator.evaluate(text_true, text_pred, f"{args.model_name}_text_only")
    evaluator.print_results(text_results, "Text-Only Model")

    image_loss, image_acc, image_true, image_pred = evaluate_model(image_model, test_loader, eval_criterion, device, "image_only")
    image_results = evaluator.evaluate(image_true, image_pred, f"{args.model_name}_image_only")
    evaluator.print_results(image_results, "Image-Only Model")

    if per_dataset_test_loaders:
        print("\n" + "=" * 60)
        print("  PER-DATASET TEST EVALUATION")
        print("=" * 60)

        for ds_name, ds_loader in per_dataset_test_loaders.items():
            print(f"\n--- {ds_name} Test Set ---")
            ca_loss, ca_acc, ca_true, ca_pred = evaluate_model(cross_attn_model, ds_loader, eval_criterion, device, "cross_attention")
            ca_results = evaluator.evaluate(ca_true, ca_pred, f"{ds_name}_cross_attention")
            evaluator.print_results(ca_results, f"Cross-Attention ({ds_name})")

            text_loss, text_acc, text_true, text_pred = evaluate_model(text_model, ds_loader, eval_criterion, device, "text_only")
            text_results = evaluator.evaluate(text_true, text_pred, f"{ds_name}_text_only")
            evaluator.print_results(text_results, f"Text-Only ({ds_name})")

            image_loss, image_acc, image_true, image_pred = evaluate_model(image_model, ds_loader, eval_criterion, device, "image_only")
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
