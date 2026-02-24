import argparse
import os
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.config.config import Config
from src.data_processing.data_loader import get_data_loaders
from src.training.models import MultimodalSentimentModel
from src.evaluation.metrics import Evaluator


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练多模态情感分析模型")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--model_name", type=str, default="multimodal_sentiment_model", help="模型名称")
    args = parser.parse_args()
    
    # 加载配置
    config = Config(args.config)
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # 初始化模型
    model = MultimodalSentimentModel(config)
    device = config.train_config["device"]
    model.to(device)
    
    # 初始化评估器
    evaluator = Evaluator(config)
    
    # 初始化优化器和学习率调度器
    optimizer = AdamW(
        model.parameters(),
        lr=config.train_config["learning_rate"],
        weight_decay=config.train_config["weight_decay"]
    )
    
    total_steps = len(train_loader) * config.train_config["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.train_config["warmup_steps"],
        num_training_steps=total_steps
    )
    
    # 训练模型
    print("训练流程已准备就绪")
    print(f"训练数据大小: {len(train_loader.dataset)}")
    print(f"验证数据大小: {len(val_loader.dataset)}")
    print(f"测试数据大小: {len(test_loader.dataset)}")
    print(f"使用设备: {device}")
    print(f"融合策略: {config.model_config['fusion_strategy']}")
    print(f"文本模型: {config.model_config['text_model']}")
    print(f"视觉模型: {config.model_config['vision_model']}")
    
    # 训练循环
    for epoch in range(config.train_config["num_epochs"]):
        print(f"\n========== 第 {epoch+1}/{config.train_config['num_epochs']} 轮 ==========")
        
        # 训练阶段
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc="训练"):
            # 准备输入
            text_inputs = {
                "input_ids": batch["text_inputs"]["input_ids"].to(device),
                "attention_mask": batch["text_inputs"]["attention_mask"].to(device)
            }
            
            image_inputs = {
                "pixel_values": batch["image_inputs"]["pixel_values"].to(device)
            }
            
            labels = batch["label"].to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 模型前向传播
            outputs = model(text_inputs, image_inputs)
            logits = outputs["logits"]
            
            # 计算损失
            loss = torch.nn.functional.cross_entropy(logits, labels)
            train_loss += loss.item()
            
            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        print(f"训练损失: {avg_train_loss:.4f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证"):
                # 准备输入
                text_inputs = {
                    "input_ids": batch["text_inputs"]["input_ids"].to(device),
                    "attention_mask": batch["text_inputs"]["attention_mask"].to(device)
                }
                
                image_inputs = {
                    "pixel_values": batch["image_inputs"]["pixel_values"].to(device)
                }
                
                labels = batch["label"].to(device)
                
                # 模型前向传播
                outputs = model(text_inputs, image_inputs)
                logits = outputs["logits"]
                
                # 计算损失
                loss = torch.nn.functional.cross_entropy(logits, labels)
                val_loss += loss.item()
                
                # 获取预测结果
                predictions = torch.argmax(logits, dim=1)
                
                # 收集真实标签和预测标签
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
        
        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        print(f"验证损失: {avg_val_loss:.4f}")
        
        # 评估模型性能
        eval_results = evaluator.evaluate(y_true, y_pred, f"{args.model_name}_epoch_{epoch+1}")
        print(f"验证准确率: {eval_results['accuracy']:.4f}")
        print(f"验证F1分数: {eval_results['f1_macro']:.4f}")
    
    # 测试阶段
    print("\n========== 测试阶段 ==========")
    model.eval()
    test_loss = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试"):
            # 准备输入
            text_inputs = {
                "input_ids": batch["text_inputs"]["input_ids"].to(device),
                "attention_mask": batch["text_inputs"]["attention_mask"].to(device)
            }
            
            image_inputs = {
                "pixel_values": batch["image_inputs"]["pixel_values"].to(device)
            }
            
            labels = batch["label"].to(device)
            
            # 模型前向传播
            outputs = model(text_inputs, image_inputs)
            logits = outputs["logits"]
            
            # 计算损失
            loss = torch.nn.functional.cross_entropy(logits, labels)
            test_loss += loss.item()
            
            # 获取预测结果
            predictions = torch.argmax(logits, dim=1)
            
            # 收集真实标签和预测标签
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    
    # 计算平均测试损失
    avg_test_loss = test_loss / len(test_loader)
    print(f"测试损失: {avg_test_loss:.4f}")
    
    # 评估模型性能
    test_results = evaluator.evaluate(y_true, y_pred, f"{args.model_name}_test")
    print(f"测试准确率: {test_results['accuracy']:.4f}")
    print(f"测试F1分数: {test_results['f1_macro']:.4f}")
    
    # 保存模型
    model_dir = os.path.join(config.eval_config["results_dir"], "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f"{args.model_name}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")
    
    # 保存配置
    config_path = os.path.join(model_dir, f"{args.model_name}_config.json")
    config.save_config(config_path)
    print(f"配置已保存到: {config_path}")


if __name__ == "__main__":
    main()
