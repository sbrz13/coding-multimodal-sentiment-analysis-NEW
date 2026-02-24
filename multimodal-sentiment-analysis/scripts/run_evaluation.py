import argparse
import os
import torch

from src.config.config import Config
from src.data_processing.data_loader import get_data_loaders
from src.training.models import MultimodalSentimentModel
from src.evaluation.metrics import Evaluator


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="评估多模态情感分析模型")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--model_name", type=str, default="multimodal_sentiment_model", help="模型名称")
    args = parser.parse_args()
    
    # 加载配置
    config = Config(args.config)
    
    # 获取数据加载器
    _, _, test_loader = get_data_loaders(config)
    
    # 初始化模型
    model = MultimodalSentimentModel(config)
    device = config.train_config["device"]
    model.to(device)
    
    # 加载模型权重
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"模型权重已加载: {args.model_path}")
    else:
        raise FileNotFoundError(f"模型权重文件不存在: {args.model_path}")
    
    # 初始化评估器
    evaluator = Evaluator(config)
    
    # 评估模型
    print("评估流程已准备就绪")
    print(f"测试数据大小: {len(test_loader.dataset)}")
    print(f"使用设备: {device}")
    print(f"融合策略: {config.model_config['fusion_strategy']}")
    print(f"文本模型: {config.model_config['text_model']}")
    print(f"视觉模型: {config.model_config['vision_model']}")
    
    # 评估模型性能
    print("\n开始评估...")
    
    # 准备评估数据
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 准备输入
            text_inputs = {
                "input_ids": batch["text_inputs"]["input_ids"].to(device),
                "attention_mask": batch["text_inputs"]["attention_mask"].to(device)
            }
            
            image_inputs = {
                "pixel_values": batch["image_inputs"]["pixel_values"].to(device)
            }
            
            labels = batch["label"].to(device)
            
            # 模型推理
            outputs = model(text_inputs, image_inputs)
            logits = outputs["logits"]
            
            # 获取预测结果
            predictions = torch.argmax(logits, dim=1)
            
            # 收集真实标签和预测标签
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    
    # 计算评估指标
    eval_results = evaluator.evaluate(y_true, y_pred, f"{args.model_name}_evaluation")
    
    # 打印评估结果
    print("\n评估结果:")
    print(f"准确率: {eval_results['accuracy']:.4f}")
    print(f"F1分数 (macro): {eval_results['f1_macro']:.4f}")
    print(f"F1分数 (micro): {eval_results['f1_micro']:.4f}")
    print(f"F1分数 (weighted): {eval_results['f1_weighted']:.4f}")
    print(f"精确率 (macro): {eval_results['precision_macro']:.4f}")
    print(f"召回率 (macro): {eval_results['recall_macro']:.4f}")
    
    # 比较不同模型的性能
    print("\n模型性能比较...")
    print("注意：这里可以添加多个模型的评估结果进行比较")
    print("例如：比较特征级融合和决策级融合的性能差异")
    print("或者比较不同预训练模型组合的性能差异")


if __name__ == "__main__":
    main()
