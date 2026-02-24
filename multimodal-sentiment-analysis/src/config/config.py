import json
import os

class Config:
    def __init__(self, config_file=None):
        # 默认配置
        self.data_config = {
            "dataset_name": "mvsa",  # 数据集名称：mvsa, twitter
            "data_dir": "data/",  # 数据存储目录
            "batch_size": 32,  # 批次大小
            "image_size": 224,  # 图像大小
            "max_seq_length": 128,  # 文本最大长度
            "train_val_split": 0.8  # 训练验证集分割比例
        }
        
        self.model_config = {
            "text_model": "bert-base-uncased",  # 文本模型：bert-base-uncased, roberta-base, electra-base-discriminator
            "vision_model": "vit-base-patch16-224",  # 视觉模型：inception_v3, convnext-base, vit-base-patch16-224
            "fusion_strategy": "feature_level",  # 融合策略：feature_level, decision_level
            "num_classes": 3,  # 情感类别数：3（积极、消极、中性）
            "dropout_rate": 0.5  # Dropout率
        }
        
        self.train_config = {
            "learning_rate": 2e-5,  # 学习率
            "num_epochs": 10,  # 训练轮数
            "warmup_steps": 500,  # 预热步数
            "weight_decay": 0.01,  # 权重衰减
            "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"  # 设备
        }
        
        self.eval_config = {
            "metrics": ["accuracy", "f1", "precision", "recall"],  # 评估指标
            "save_results": True,  # 是否保存评估结果
            "results_dir": "results/"  # 结果保存目录
        }
        
        # 如果提供了配置文件，加载配置
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file):
        """从JSON文件加载配置"""
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 更新配置
        if "data_config" in config:
            self.data_config.update(config["data_config"])
        if "model_config" in config:
            self.model_config.update(config["model_config"])
        if "train_config" in config:
            self.train_config.update(config["train_config"])
        if "eval_config" in config:
            self.eval_config.update(config["eval_config"])
    
    def save_config(self, config_file):
        """保存配置到JSON文件"""
        config = {
            "data_config": self.data_config,
            "model_config": self.model_config,
            "train_config": self.train_config,
            "eval_config": self.eval_config
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    
    def get_config(self):
        """获取完整配置"""
        return {
            "data_config": self.data_config,
            "model_config": self.model_config,
            "train_config": self.train_config,
            "eval_config": self.eval_config
        }
