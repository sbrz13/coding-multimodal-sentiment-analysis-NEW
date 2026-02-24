import argparse
import os
import sys
import torch
import torch.nn as nn

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import Config


class DummyModel(nn.Module):
    """ dummy模型，用于测试输入结构 """
    def __init__(self, config):
        super(DummyModel, self).__init__()
        self.config = config
        
        # 模拟文本模型
        self.text_embedding = nn.Embedding(10000, 768)
        self.text_pooler = nn.Linear(768, 768)
        
        # 模拟视觉模型
        self.vision_embedding = nn.Linear(3*224*224, 768)
        self.vision_pooler = nn.Linear(768, 768)
        
        # 融合层
        self.fusion_layer = nn.Linear(768 + 768, 3)
    
    def forward(self, text_inputs, image_inputs):
        """前向传播"""
        # 文本特征提取
        text_emb = self.text_embedding(text_inputs["input_ids"])
        text_emb = text_emb.mean(dim=1)
        text_features = self.text_pooler(text_emb)
        
        # 视觉特征提取
        image_emb = self.vision_embedding(image_inputs["pixel_values"].flatten(1))
        vision_features = self.vision_pooler(image_emb)
        
        # 融合
        fused = torch.cat([text_features, vision_features], dim=1)
        logits = self.fusion_layer(fused)
        
        return {"logits": logits}


def main():
    """测试输入结构"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试输入结构")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()
    
    # 加载配置
    config = Config(args.config)
    device = config.train_config["device"]
    
    # 初始化模型
    model = DummyModel(config)
    model.to(device)
    model.eval()
    
    # 创建单个样本输入
    print("测试单个样本输入...")
    
    # 单个样本输入
    single_text_inputs = {
        "input_ids": torch.randint(0, 10000, (1, config.data_config["max_seq_length"])).to(device),
        "attention_mask": torch.ones(1, config.data_config["max_seq_length"]).to(device)
    }
    
    single_image_inputs = {
        "pixel_values": torch.randn(1, 3, config.data_config["image_size"], config.data_config["image_size"]).to(device)
    }
    
    # 测试单个样本
    with torch.no_grad():
        single_outputs = model(single_text_inputs, single_image_inputs)
        print(f"单个样本输出形状: {single_outputs['logits'].shape}")
    
    # 测试批量样本输入
    print("\n测试批量样本输入...")
    
    # 批量样本输入
    batch_size = 4
    batch_text_inputs = {
        "input_ids": torch.randint(0, 10000, (batch_size, config.data_config["max_seq_length"])).to(device),
        "attention_mask": torch.ones(batch_size, config.data_config["max_seq_length"]).to(device)
    }
    
    batch_image_inputs = {
        "pixel_values": torch.randn(batch_size, 3, config.data_config["image_size"], config.data_config["image_size"]).to(device)
    }
    
    # 测试批量样本
    with torch.no_grad():
        batch_outputs = model(batch_text_inputs, batch_image_inputs)
        print(f"批量样本输出形状: {batch_outputs['logits'].shape}")
    
    print("\n测试完成！模型可以正确处理单个样本输入和批量样本输入。")
    print("输入结构验证成功：")
    print(f"- 文本输入: input_ids (batch_size, seq_length), attention_mask (batch_size, seq_length)")
    print(f"- 图像输入: pixel_values (batch_size, channels, height, width)")
    print(f"- 输出: logits (batch_size, num_classes)")


if __name__ == "__main__":
    main()
