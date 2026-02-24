from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import time
import json
import os

class Evaluator:
    def __init__(self, config):
        """
        评估器类
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.metrics = config.eval_config["metrics"]
        self.results_dir = config.eval_config["results_dir"]
        
        # 确保结果目录存在
        os.makedirs(self.results_dir, exist_ok=True)
    
    def evaluate(self, y_true, y_pred, model_name=None):
        """
        评估模型性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            model_name: 模型名称（用于保存结果）
        
        Returns:
            评估结果字典
        """
        results = {}
        
        # 计算准确率
        if "accuracy" in self.metrics:
            results["accuracy"] = accuracy_score(y_true, y_pred)
        
        # 计算F1分数
        if "f1" in self.metrics:
            results["f1_macro"] = f1_score(y_true, y_pred, average="macro")
            results["f1_micro"] = f1_score(y_true, y_pred, average="micro")
            results["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")
        
        # 计算精确率
        if "precision" in self.metrics:
            results["precision_macro"] = precision_score(y_true, y_pred, average="macro")
            results["precision_micro"] = precision_score(y_true, y_pred, average="micro")
            results["precision_weighted"] = precision_score(y_true, y_pred, average="weighted")
        
        # 计算召回率
        if "recall" in self.metrics:
            results["recall_macro"] = recall_score(y_true, y_pred, average="macro")
            results["recall_micro"] = recall_score(y_true, y_pred, average="micro")
            results["recall_weighted"] = recall_score(y_true, y_pred, average="weighted")
        
        # 生成分类报告
        results["classification_report"] = classification_report(y_true, y_pred, output_dict=True)
        
        # 保存评估结果
        if self.config.eval_config["save_results"] and model_name:
            self.save_results(results, model_name)
        
        return results
    
    def evaluate_with_time(self, model, data_loader):
        """
        评估模型性能并计算推理时间
        
        Args:
            model: 模型对象
            data_loader: 数据加载器
        
        Returns:
            评估结果字典（包含推理时间）
        """
        import torch
        
        device = self.config.train_config["device"]
        model.to(device)
        model.eval()
        
        y_true = []
        y_pred = []
        total_inference_time = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # 准备输入
                text_inputs = {
                    "input_ids": batch["text_inputs"]["input_ids"].to(device),
                    "attention_mask": batch["text_inputs"]["attention_mask"].to(device)
                }
                
                image_inputs = {
                    "pixel_values": batch["image_inputs"]["pixel_values"].to(device)
                }
                
                labels = batch["label"].to(device)
                
                # 记录推理开始时间
                start_time = time.time()
                
                # 模型推理
                outputs = model(text_inputs, image_inputs)
                
                # 记录推理结束时间
                end_time = time.time()
                total_inference_time += (end_time - start_time)
                
                # 获取预测结果
                predictions = torch.argmax(outputs.logits, dim=1)
                
                # 收集真实标签和预测标签
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
        
        # 计算平均推理时间
        avg_inference_time = total_inference_time / len(data_loader)
        
        # 计算评估指标
        results = self.evaluate(y_true, y_pred)
        
        # 添加推理时间指标
        results["inference_time"] = {
            "total": total_inference_time,
            "average": avg_inference_time,
            "per_sample": avg_inference_time / data_loader.batch_size
        }
        
        return results
    
    def save_results(self, results, model_name):
        """
        保存评估结果到文件
        
        Args:
            results: 评估结果字典
            model_name: 模型名称
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_name = f"{model_name}_eval_{timestamp}.json"
        file_path = os.path.join(self.results_dir, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"评估结果已保存到: {file_path}")
    
    def compare_models(self, model_results):
        """
        比较多个模型的性能
        
        Args:
            model_results: 模型结果字典，格式为 {model_name: results}
        
        Returns:
            比较结果字典
        """
        comparison = {}
        
        # 对每个指标进行比较
        metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
        
        for metric in metrics:
            if metric in list(model_results.values())[0]:
                metric_values = {model: results[metric] for model, results in model_results.items()}
                best_model = max(metric_values, key=metric_values.get)
                
                comparison[metric] = {
                    "values": metric_values,
                    "best_model": best_model,
                    "best_value": metric_values[best_model]
                }
        
        # 保存比较结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_name = f"model_comparison_{timestamp}.json"
        file_path = os.path.join(self.results_dir, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=4, ensure_ascii=False)
        
        print(f"模型比较结果已保存到: {file_path}")
        
        return comparison
