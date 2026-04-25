import json
import os
import torch


class Config:
    def __init__(self, config_file=None):
        self.data_config = {
            "dataset_name": "merged",
            "data_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"),
            "batch_size": 32,
            "image_size": 224,
            "max_seq_length": 128,
            "num_workers": 4,
            "use_merged_data": True,
            "augment": True,
            "use_weighted_sampling": True,
            "use_mixup": True,
        }

        self.model_config = {
            "text_model": "roberta-base",
            "vision_model": "openai/clip-vit-base-patch32",
            "text_hidden_size": 768,
            "vision_hidden_size": 768,
            "projection_dim": 512,
            "attention_dim": 512,
            "num_attention_heads": 8,
            "cross_attn_layers": 4,
            "num_classes": 3,
            "dropout_rate": 0.3,
            "text_dropout_rate": 0.2,
            "image_dropout_rate": 0.5,
            "drop_path_rate": 0.15,
            "image_noise_scale": 0.15,
            "image_noise_prob": 0.5,
            "freeze_encoders": False,
            "contrastive_temperature": 0.07,
            "contrastive_weight": 0.2,
            "use_reliability_gate": True,
        }

        self.train_config = {
            "learning_rate": 5e-6,
            "encoder_learning_rate": 2e-6,
            "image_encoder_learning_rate": 1e-6,
            "fusion_learning_rate": 5e-5,
            "classifier_learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "image_weight_decay": 5e-4,
            "num_epochs": 30,
            "warmup_ratio": 0.1,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "seed": 42,
            "early_stop_patience": 8,
            "label_smoothing": 0.1,
            "gradient_accumulation_steps": 2,
        }

        self.meta_config = {
            "meta_learner_C": 100,
            "meta_learner_multi_class": "ovr",
            "meta_learner_max_iter": 1000,
        }

        self.eval_config = {
            "metrics": ["accuracy", "f1", "precision", "recall"],
            "save_results": True,
            "results_dir": "results/",
        }

        if config_file and os.path.exists(config_file):
            self.load_config(config_file)

    def load_config(self, config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        if "data_config" in config:
            self.data_config.update(config["data_config"])
        if "model_config" in config:
            self.model_config.update(config["model_config"])
        if "train_config" in config:
            self.train_config.update(config["train_config"])
        if "meta_config" in config:
            self.meta_config.update(config["meta_config"])
        if "eval_config" in config:
            self.eval_config.update(config["eval_config"])

    def save_config(self, config_file):
        config = self.to_dict()
        os.makedirs(os.path.dirname(config_file) if os.path.dirname(config_file) else ".", exist_ok=True)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    def to_dict(self):
        return {
            "data_config": self.data_config,
            "model_config": self.model_config,
            "train_config": self.train_config,
            "meta_config": self.meta_config,
            "eval_config": self.eval_config,
        }
