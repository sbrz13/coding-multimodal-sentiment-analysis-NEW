import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoImageProcessor

class MultimodalSentimentDataset(Dataset):
    def __init__(self, data_dir, split="train", config=None):
        """
        多模态情感分析数据集
        
        Args:
            data_dir: 数据目录
            split: 数据集分割（train, val, test）
            config: 配置对象
        """
        self.data_dir = data_dir
        self.split = split
        self.config = config
        
        # 加载数据集
        self.data = self._load_data()
        
        # 初始化文本和图像处理器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_config["text_model"])
        self.image_processor = AutoImageProcessor.from_pretrained(config.model_config["vision_model"])
    
    def _load_data(self):
        """加载数据集"""
        # 根据数据集名称加载不同的数据
        if self.config.data_config["dataset_name"] == "mvsa":
            return self._load_mvsa_data()
        elif self.config.data_config["dataset_name"] == "twitter":
            return self._load_twitter_data()
        else:
            raise ValueError(f"不支持的数据集: {self.config.data_config['dataset_name']}")
    
    def _load_mvsa_data(self):
        """加载MVSA数据集"""
        # MVSA数据集结构：
        # data/
        # ├── images/
        # ├── mvsa_train.csv
        # ├── mvsa_val.csv
        # └── mvsa_test.csv
        
        csv_file = os.path.join(self.data_dir, f"mvsa_{self.split}.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"文件不存在: {csv_file}")
        
        data = pd.read_csv(csv_file)
        return data
    
    def _load_twitter_data(self):
        """加载Twitter多模态情感分析数据集"""
        # Twitter数据集结构：
        # data/
        # ├── images/
        # ├── twitter_train.csv
        # ├── twitter_val.csv
        # └── twitter_test.csv
        
        csv_file = os.path.join(self.data_dir, f"twitter_{self.split}.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"文件不存在: {csv_file}")
        
        data = pd.read_csv(csv_file)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.data.iloc[idx]
        
        # 加载文本
        text = sample["text"]
        
        # 加载图像
        image_path = os.path.join(self.data_dir, "images", sample["image_id"])
        if not os.path.exists(image_path):
            # 如果图像不存在，使用空图像
            image = Image.new("RGB", (self.config.data_config["image_size"], self.config.data_config["image_size"]))
        else:
            image = Image.open(image_path).convert("RGB")
        
        # 预处理文本
        text_inputs = self.tokenizer(
            text,
            max_length=self.config.data_config["max_seq_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 预处理图像
        image_inputs = self.image_processor(
            image,
            return_tensors="pt"
        )
        
        # 获取标签
        label = sample["label"]
        
        return {
            "text_inputs": {
                "input_ids": text_inputs["input_ids"].squeeze(),
                "attention_mask": text_inputs["attention_mask"].squeeze()
            },
            "image_inputs": {
                "pixel_values": image_inputs["pixel_values"].squeeze()
            },
            "label": label
        }

def get_data_loaders(config):
    """
    获取数据加载器
    
    Args:
        config: 配置对象
    
    Returns:
        train_loader, val_loader, test_loader: 训练、验证和测试数据加载器
    """
    data_dir = config.data_config["data_dir"]
    batch_size = config.data_config["batch_size"]
    
    # 创建数据集
    train_dataset = MultimodalSentimentDataset(data_dir, "train", config)
    val_dataset = MultimodalSentimentDataset(data_dir, "val", config)
    test_dataset = MultimodalSentimentDataset(data_dir, "test", config)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader
