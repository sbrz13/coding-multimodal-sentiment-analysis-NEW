import argparse
import os
import sys
import torch
from PIL import Image

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import Config
from src.training.models import MultimodalSentimentModel
from transformers import AutoTokenizer, AutoImageProcessor


def main():
    """测试单个样本输入"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试单个样本输入")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()
    
    # 加载配置
    config = Config(args.config)
    device = config.train_config["device"]
    
    # 初始化模型
    model = MultimodalSentimentModel(config)
    model.to(device)
    model.eval()
    
    # 初始化文本和图像处理器
    tokenizer = AutoTokenizer.from_pretrained(config.model_config["text_model"])
    image_processor = AutoImageProcessor.from_pretrained(config.model_config["vision_model"])
    
    # 创建测试样本
    test_text = "I love this product! It's amazing."
    test_image = Image.new("RGB", (config.data_config["image_size"], config.data_config["image_size"]), color="red")
    
    # 预处理输入
    text_inputs = tokenizer(
        test_text,
        max_length=config.data_config["max_seq_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    image_inputs = image_processor(
        test_image,
        return_tensors="pt"
    )
    
    # 将输入移到设备上
    text_inputs = {
        "input_ids": text_inputs["input_ids"].to(device),
        "attention_mask": text_inputs["attention_mask"].to(device)
    }
    
    image_inputs = {
        "pixel_values": image_inputs["pixel_values"].to(device)
    }
    
    # 测试模型
    print("测试单个样本输入...")
    print(f"输入文本: {test_text}")
    print(f"输入图像: 红色测试图像 ({config.data_config['image_size']}x{config.data_config['image_size']})")
    
    with torch.no_grad():
        outputs = model(text_inputs, image_inputs)
        logits = outputs["logits"]
        predictions = torch.argmax(logits, dim=1)
    
    # 打印结果
    print(f"模型输出: {logits.cpu().numpy()}")
    print(f"预测标签: {predictions.item()}")
    
    # 测试批量输入
    print("\n测试批量输入...")
    
    # 创建批量输入
    batch_text_inputs = {
        "input_ids": torch.cat([text_inputs["input_ids"], text_inputs["input_ids"]], dim=0),
        "attention_mask": torch.cat([text_inputs["attention_mask"], text_inputs["attention_mask"]], dim=0)
    }
    
    batch_image_inputs = {
        "pixel_values": torch.cat([image_inputs["pixel_values"], image_inputs["pixel_values"]], dim=0)
    }
    
    with torch.no_grad():
        batch_outputs = model(batch_text_inputs, batch_image_inputs)
        batch_logits = batch_outputs["logits"]
        batch_predictions = torch.argmax(batch_logits, dim=1)
    
    print(f"批量输出形状: {batch_logits.shape}")
    print(f"批量预测标签: {batch_predictions.cpu().numpy()}")
    
    print("\n测试完成！模型可以正确处理单个样本输入和批量输入。")


if __name__ == "__main__":
    main()
