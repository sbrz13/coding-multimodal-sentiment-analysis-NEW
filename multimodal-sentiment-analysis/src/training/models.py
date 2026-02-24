import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class MultimodalSentimentModel(nn.Module):
    def __init__(self, config):
        """
        多模态情感分析模型基类
        
        Args:
            config: 配置对象
        """
        super(MultimodalSentimentModel, self).__init__()
        self.config = config
        self.fusion_strategy = config.model_config["fusion_strategy"]
        
        # 根据融合策略选择具体模型
        if self.fusion_strategy == "feature_level":
            self.model = FeatureLevelFusionModel(config)
        elif self.fusion_strategy == "decision_level":
            self.model = DecisionLevelFusionModel(config)
        else:
            raise ValueError(f"不支持的融合策略: {self.fusion_strategy}")
    
    def forward(self, text_inputs, image_inputs):
        """
        前向传播
        
        Args:
            text_inputs: 文本输入
            image_inputs: 图像输入
        
        Returns:
            模型输出
        """
        return self.model(text_inputs, image_inputs)

class FeatureLevelFusionModel(nn.Module):
    def __init__(self, config):
        """
        特征级融合模型
        
        Args:
            config: 配置对象
        """
        super(FeatureLevelFusionModel, self).__init__()
        self.config = config
        
        # 加载文本模型
        self.text_model = AutoModel.from_pretrained(config.model_config["text_model"])
        
        # 加载视觉模型
        self.vision_model = AutoModel.from_pretrained(config.model_config["vision_model"])
        
        # 获取特征维度
        self.text_feature_dim = self.text_model.config.hidden_size
        self.vision_feature_dim = self.vision_model.config.hidden_size
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.text_feature_dim + self.vision_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(config.model_config["dropout_rate"]),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.model_config["dropout_rate"])
        )
        
        # 分类层
        self.classifier = nn.Linear(256, config.model_config["num_classes"])
    
    def forward(self, text_inputs, image_inputs):
        """
        前向传播
        
        Args:
            text_inputs: 文本输入，包含input_ids和attention_mask
            image_inputs: 图像输入，包含pixel_values
        
        Returns:
            模型输出
        """
        # 文本特征提取
        text_outputs = self.text_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # 视觉特征提取
        vision_outputs = self.vision_model(
            pixel_values=image_inputs["pixel_values"]
        )
        vision_features = vision_outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # 特征融合
        fused_features = torch.cat([text_features, vision_features], dim=1)
        fused_features = self.fusion_layer(fused_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return {
            "logits": logits,
            "text_features": text_features,
            "vision_features": vision_features,
            "fused_features": fused_features
        }

class DecisionLevelFusionModel(nn.Module):
    def __init__(self, config):
        """
        决策级融合模型
        
        Args:
            config: 配置对象
        """
        super(DecisionLevelFusionModel, self).__init__()
        self.config = config
        
        # 文本子模型
        self.text_submodel = TextSubModel(config)
        
        # 视觉子模型
        self.vision_submodel = VisionSubModel(config)
        
        # 融合权重
        self.text_weight = nn.Parameter(torch.tensor(0.5))
        self.vision_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, text_inputs, image_inputs):
        """
        前向传播
        
        Args:
            text_inputs: 文本输入，包含input_ids和attention_mask
            image_inputs: 图像输入，包含pixel_values
        
        Returns:
            模型输出
        """
        # 文本子模型预测
        text_outputs = self.text_submodel(text_inputs)
        text_logits = text_outputs["logits"]
        
        # 视觉子模型预测
        vision_outputs = self.vision_submodel(image_inputs)
        vision_logits = vision_outputs["logits"]
        
        # 归一化权重
        weights = torch.softmax(torch.stack([self.text_weight, self.vision_weight]), dim=0)
        
        # 决策融合
        fused_logits = weights[0] * text_logits + weights[1] * vision_logits
        
        return {
            "logits": fused_logits,
            "text_logits": text_logits,
            "vision_logits": vision_logits,
            "weights": weights
        }

class TextSubModel(nn.Module):
    def __init__(self, config):
        """
        文本子模型
        
        Args:
            config: 配置对象
        """
        super(TextSubModel, self).__init__()
        self.config = config
        
        # 加载文本模型
        self.text_model = AutoModel.from_pretrained(config.model_config["text_model"])
        
        # 分类层
        self.classifier = nn.Linear(self.text_model.config.hidden_size, config.model_config["num_classes"])
    
    def forward(self, text_inputs):
        """
        前向传播
        
        Args:
            text_inputs: 文本输入，包含input_ids和attention_mask
        
        Returns:
            模型输出
        """
        # 文本特征提取
        outputs = self.text_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )
        features = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # 分类
        logits = self.classifier(features)
        
        return {
            "logits": logits,
            "features": features
        }

class VisionSubModel(nn.Module):
    def __init__(self, config):
        """
        视觉子模型
        
        Args:
            config: 配置对象
        """
        super(VisionSubModel, self).__init__()
        self.config = config
        
        # 加载视觉模型
        self.vision_model = AutoModel.from_pretrained(config.model_config["vision_model"])
        
        # 分类层
        self.classifier = nn.Linear(self.vision_model.config.hidden_size, config.model_config["num_classes"])
    
    def forward(self, image_inputs):
        """
        前向传播
        
        Args:
            image_inputs: 图像输入，包含pixel_values
        
        Returns:
            模型输出
        """
        # 视觉特征提取
        outputs = self.vision_model(
            pixel_values=image_inputs["pixel_values"]
        )
        features = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # 分类
        logits = self.classifier(features)
        
        return {
            "logits": logits,
            "features": features
        }
