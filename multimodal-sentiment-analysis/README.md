# 多模态情感分析项目

## 项目概述

本项目旨在设计一个多层次特征融合框架，用于多模态情感分析，利用来自最先进预训练模型的迁移学习。项目将评估不同的视觉特征模型（如Inception、ConvNeXt和Vision Transformer）和文本特征模型（如BERT、RoBERTa或ELECTRA），并比较两种融合策略：决策级融合和特征级融合。

## 项目结构

```
multimodal-sentiment-analysis/
├── data/             # 数据目录
├── src/              # 源代码目录
│   ├── config/       # 配置模块
│   ├── data_processing/  # 数据处理模块
│   ├── evaluation/   # 评估模块
│   ├── training/     # 训练模块
├── scripts/          # 脚本目录
├── results/          # 结果目录
├── requirements.txt  # 依赖文件
├── setup.py          # 安装文件
└── README.md         # 项目说明
```

## 安装说明

1. 克隆项目到本地：

```bash
git clone <repository-url>
cd multimodal-sentiment-analysis
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 或者使用setup.py安装：

```bash
pip install -e .
```

## 数据预处理

1. 下载数据集（MVSA或Twitter多模态情感分析数据集）

2. 运行数据预处理脚本：

```bash
python scripts/preprocess_data.py --input_dir <原始数据目录> --output_dir data/
```

## 模型训练

运行训练脚本：

```bash
python scripts/run_training.py --config <配置文件路径> --model_name <模型名称>
```

## 模型评估

运行评估脚本：

```bash
python scripts/run_evaluation.py --config <配置文件路径> --model_path <模型权重路径> --model_name <模型名称>
```

## 配置说明

项目使用JSON格式的配置文件，主要配置项包括：

- **data_config**：数据相关配置，如数据集名称、批量大小、图像大小等
- **model_config**：模型相关配置，如文本模型、视觉模型、融合策略等
- **train_config**：训练相关配置，如学习率、训练轮数等
- **eval_config**：评估相关配置，如评估指标、结果保存目录等

## 支持的数据集

- **MVSA**：多视角情感分析数据集
- **Twitter**：Twitter多模态情感分析数据集

## 支持的模型

### 文本模型
- BERT (bert-base-uncased)
- RoBERTa (roberta-base)
- ELECTRA (electra-base-discriminator)

### 视觉模型
- Inception (inception_v3)
- ConvNeXt (convnext-base)
- Vision Transformer (vit-base-patch16-224)

### 融合策略
- **特征级融合** (feature_level)：在分类前融合两种模态的嵌入
- **决策级融合** (decision_level)：结合来自单独单模态分类器的输出

## 评估指标

- 准确率 (Accuracy)
- F1分数 (F1-score)
- 精确率 (Precision)
- 召回率 (Recall)
- 计算效率 (Computational Efficiency)

## 项目目标

本项目的目标是证明多模态融合显著优于单模态方法，为现实世界的社交媒体分析应用提供更 robust 和准确的情感分类。

## 注意事项

- 本项目只实现了非模型搭建部分，模型搭建需要根据具体需求实现
- 运行前请确保已安装所有依赖包
- 运行训练和评估前请确保已预处理数据集
