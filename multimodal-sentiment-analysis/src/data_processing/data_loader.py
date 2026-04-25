import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoImageProcessor
from collections import Counter


LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_NAMES = ["negative", "neutral", "positive"]


class Cutout:
    def __init__(self, n_holes=1, length=32, prob=0.5):
        self.n_holes = n_holes
        self.length = length
        self.prob = prob

    def __call__(self, image):
        if random.random() > self.prob:
            return image
        w, h = image.size
        img_array = np.array(image)
        for _ in range(self.n_holes):
            y = random.randint(0, h)
            x = random.randint(0, w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            img_array[y1:y2, x1:x2] = 128
        return Image.fromarray(img_array)


class RandomErasingTensor:
    def __init__(self, prob=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0):
        self.prob = prob
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img_tensor):
        if random.random() > self.prob:
            return img_tensor
        c, h, w = img_tensor.shape
        area = h * w
        for _ in range(10):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            eh = int(round(np.sqrt(target_area * aspect_ratio)))
            ew = int(round(np.sqrt(target_area / aspect_ratio)))
            if eh < h and ew < w:
                y = random.randint(0, h - eh)
                x = random.randint(0, w - ew)
                if self.value == 'random':
                    img_tensor[:, y:y+eh, x:x+ew] = torch.randn(c, eh, ew)
                else:
                    img_tensor[:, y:y+eh, x:x+ew] = self.value
                break
        return img_tensor


class ImageAugmentation:
    def __init__(self, enable=True, mode='train'):
        self.enable = enable
        self.mode = mode
        self.cutout = Cutout(n_holes=1, length=40, prob=0.4)

    def __call__(self, image):
        if not self.enable:
            return image

        if random.random() < 0.5:
            image = ImageOps.equalize(image)

        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        if random.random() < 0.4:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.7, 1.3))

        if random.random() < 0.4:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.7, 1.3))

        if random.random() < 0.4:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(random.uniform(0.6, 1.4))

        if random.random() < 0.3:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(random.uniform(0.5, 2.0))

        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            image = image.rotate(angle, fillcolor=(128, 128, 128), expand=False)

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.2:
            image = ImageOps.autocontrast(image)

        if random.random() < 0.15:
            image = ImageOps.solarize(image, threshold=random.randint(100, 200))

        if random.random() < 0.15:
            image = ImageOps.posterize(image, bits=random.randint(3, 7))

        image = self.cutout(image)

        return image


class MixUpCollator:
    def __init__(self, alpha=0.2, prob=0.3):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch):
        if random.random() > self.prob:
            return torch.utils.data.dataloader.default_collate(batch)

        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0

        batch_size = len(batch)
        index = torch.randperm(batch_size)

        mixed_batch = []
        for i in range(batch_size):
            j = index[i]
            item_i = batch[i]
            item_j = batch[j]

            mixed_item = {}
            mixed_item["input_ids"] = item_i["input_ids"]
            mixed_item["attention_mask"] = item_i["attention_mask"]

            if isinstance(item_i["pixel_values"], torch.Tensor) and isinstance(item_j["pixel_values"], torch.Tensor):
                mixed_item["pixel_values"] = lam * item_i["pixel_values"] + (1 - lam) * item_j["pixel_values"]
            else:
                mixed_item["pixel_values"] = item_i["pixel_values"]

            mixed_item["label"] = item_i["label"]
            mixed_item["label_secondary"] = item_j["label"]
            mixed_item["mix_lambda"] = lam

            mixed_batch.append(mixed_item)

        return torch.utils.data.dataloader.default_collate(mixed_batch)


class MultimodalSentimentDataset(Dataset):
    def __init__(self, data, config, tokenizer, image_processor, augment=False):
        self.data = data
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.augment = augment
        self.image_aug = ImageAugmentation(enable=augment, mode='train' if augment else 'val')
        self.random_erase = RandomErasingTensor(prob=0.3) if augment else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = str(sample["text"])
        image_path = sample["image_path"]

        if not os.path.exists(image_path):
            image = Image.new("RGB", (self.config.data_config["image_size"], self.config.data_config["image_size"]))
        else:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (self.config.data_config["image_size"], self.config.data_config["image_size"]))

        if self.augment:
            image = self.image_aug(image)

        text_inputs = self.tokenizer(
            text,
            max_length=self.config.data_config["max_seq_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        image_inputs = self.image_processor(image, return_tensors="pt")

        pixel_values = image_inputs["pixel_values"].squeeze(0)

        if self.random_erase is not None:
            pixel_values = self.random_erase(pixel_values)

        label_str = str(sample["label"]).lower().strip()
        if label_str in LABEL_MAP:
            label = LABEL_MAP[label_str]
        else:
            label = int(label_str)

        return {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "label": label,
        }


def load_json_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_class_weights_from_data(data):
    labels = []
    for item in data:
        label_str = str(item["label"]).lower().strip()
        if label_str in LABEL_MAP:
            labels.append(LABEL_MAP[label_str])
        else:
            labels.append(int(label_str))
    labels = np.array(labels)
    n_samples = len(labels)
    n_classes = len(LABEL_NAMES)
    class_counts = np.bincount(labels, minlength=n_classes).astype(float)
    class_counts[class_counts == 0] = 1.0
    weights = n_samples / (n_classes * class_counts)
    weights = weights / weights.sum() * n_classes
    return torch.FloatTensor(weights)


def get_data_loaders(config):
    data_dir = config.data_config["data_dir"]
    batch_size = config.data_config["batch_size"]
    num_workers = config.data_config.get("num_workers", 0)
    augment = config.data_config.get("augment", True)
    use_weighted_sampling = config.data_config.get("use_weighted_sampling", False)
    use_mixup = config.data_config.get("use_mixup", False)

    tokenizer = AutoTokenizer.from_pretrained(config.model_config["text_model"])
    image_processor = AutoImageProcessor.from_pretrained(config.model_config["vision_model"])

    use_merged = config.data_config.get("use_merged_data", True)

    if use_merged:
        merged_train_path = os.path.join(data_dir, "merged_train.json")
        merged_val_path = os.path.join(data_dir, "merged_val.json")
        merged_test_path = os.path.join(data_dir, "merged_test.json")

        if not os.path.exists(merged_train_path):
            raise FileNotFoundError(
                f"Merged data not found: {merged_train_path}\n"
                "Please run: python3 scripts/merge_datasets.py"
            )

        train_data = load_json_data(merged_train_path)
        val_data = load_json_data(merged_val_path)
        test_data = load_json_data(merged_test_path)

        train_dataset = MultimodalSentimentDataset(train_data, config, tokenizer, image_processor, augment=augment)
        val_dataset = MultimodalSentimentDataset(val_data, config, tokenizer, image_processor, augment=False)
        test_dataset = MultimodalSentimentDataset(test_data, config, tokenizer, image_processor, augment=False)

        collate_fn = MixUpCollator(alpha=0.2, prob=0.3) if use_mixup else None

        if use_weighted_sampling:
            labels = [str(item["label"]).lower().strip() for item in train_data]
            label_indices = []
            for l in labels:
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
