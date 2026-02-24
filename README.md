# **Multimodal Sentiment Analysis Project**

## 📋 **Table of Contents**
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Configuration](#configuration)
- [Supported Datasets](#supported-datasets)
- [Supported Models](#supported-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Objectives](#project-objectives)
- [Important Notes](#important-notes)

---

## 🎯 **Project Overview**

This project focuses on designing a **multi-level feature fusion framework** for multimodal sentiment analysis, leveraging **transfer learning** from state-of-the-art pre-trained models. 

### Key Features:
- **Visual Models**: Inception, ConvNeXt, Vision Transformer
- **Text Models**: BERT, RoBERTa, ELECTRA
- **Fusion Strategies**: Feature-level fusion and Decision-level fusion
- **Comprehensive Evaluation**: Accuracy, F1-score, Precision, Recall, Computational Efficiency

---

## 📂 **Project Structure**

```
multimodal-sentiment-analysis/
├── 📁 data/                          # Data directory
├── 📁 src/                           # Source code
│   ├── 📁 config/                    # Configuration module
│   ├── 📁 data_processing/           # Data processing module
│   ├── 📁 evaluation/                # Evaluation module
│   └── 📁 training/                   # Training module
├── 📁 scripts/                        # Scripts directory
├── 📁 results/                         # Results directory
├── 📄 requirements.txt                 # Dependencies file
├── 📄 setup.py                          # Installation file
└── 📄 README.md                         # Project documentation
```

---

## 🔧 **Installation**

### Option 1: Using pip and requirements.txt
```bash
# Clone the repository
git clone <repository-url>
cd multimodal-sentiment-analysis

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using setup.py
```bash
# Clone the repository
git clone <repository-url>
cd multimodal-sentiment-analysis

# Install the package in development mode
pip install -e .
```

---

## 🔄 **Data Preprocessing**

### Step 1: Download Dataset
Download either:
- **MVSA** (Multi-View Sentiment Analysis dataset)
- **Twitter Multimodal Sentiment Analysis dataset**

### Step 2: Run Preprocessing Script
```bash
python scripts/preprocess_data.py \
    --input_dir <raw_data_directory> \
    --output_dir data/
```

---

## 🚀 **Model Training**

### Basic Training Command
```bash
python scripts/run_training.py \
    --config <configuration_file_path> \
    --model_name <model_name>
```

### Example
```bash
python scripts/run_training.py \
    --config configs/bert_vit_fusion.json \
    --model_name bert_vit_feature_fusion
```

---

## 📊 **Model Evaluation**

### Basic Evaluation Command
```bash
python scripts/run_evaluation.py \
    --config <configuration_file_path> \
    --model_path <model_weights_path> \
    --model_name <model_name>
```

### Example
```bash
python scripts/run_evaluation.py \
    --config configs/bert_vit_fusion.json \
    --model_path results/best_model.pth \
    --model_name bert_vit_feature_fusion
```

---

## ⚙️ **Configuration**

The project uses **JSON configuration files** with the following structure:

```json
{
    "data_config": {
        "dataset_name": "mvsa",
        "batch_size": 32,
        "image_size": 224,
        "max_text_length": 128
    },
    "model_config": {
        "text_model": "bert-base-uncased",
        "vision_model": "vit-base-patch16-224",
        "fusion_strategy": "feature_level",
        "num_classes": 3
    },
    "train_config": {
        "learning_rate": 2e-5,
        "num_epochs": 10,
        "optimizer": "adamw",
        "scheduler": "linear"
    },
    "eval_config": {
        "metrics": ["accuracy", "f1", "precision", "recall"],
        "save_dir": "results/"
    }
}
```

---

## 📚 **Supported Datasets**

| Dataset | Description | Use Case |
|---------|-------------|----------|
| **MVSA** | Multi-View Sentiment Analysis dataset | General sentiment analysis |
| **Twitter** | Twitter multimodal sentiment dataset | Social media analysis |

---

## 🤖 **Supported Models**

### Text Models
| Model | Variant | Description |
|-------|---------|-------------|
| **BERT** | `bert-base-uncased` | Bidirectional Encoder Representations from Transformers |
| **RoBERTa** | `roberta-base` | Robustly optimized BERT approach |
| **ELECTRA** | `electra-base-discriminator` | Efficiently Learning an Encoder that Classifies Token Replacements |

### Vision Models
| Model | Variant | Description |
|-------|---------|-------------|
| **Inception** | `inception_v3` | GoogLeNet with inception modules |
| **ConvNeXt** | `convnext-base` | Modernized ConvNet architecture |
| **ViT** | `vit-base-patch16-224` | Vision Transformer |

### Fusion Strategies
| Strategy | Description | Advantages |
|----------|-------------|------------|
| **Feature-level Fusion** | Fuse embeddings before classification | End-to-end learning, joint representation |
| **Decision-level Fusion** | Combine unimodal classifier outputs | Model independence, flexible integration |

---

## 📈 **Evaluation Metrics**

| Metric | Description | Formula |
|--------|-------------|---------|
| **Accuracy** | Overall correctness | (TP + TN) / (TP + TN + FP + FN) |
| **F1-score** | Harmonic mean of precision and recall | 2 * (Precision * Recall) / (Precision + Recall) |
| **Precision** | Positive predictive value | TP / (TP + FP) |
| **Recall** | True positive rate | TP / (TP + FN) |
| **Computational Efficiency** | Inference time & memory usage | Measured empirically |

---

## 🎯 **Project Objectives**

1. **Demonstrate Superiority**: Prove that multimodal fusion significantly outperforms unimodal approaches
2. **Comprehensive Comparison**: Evaluate multiple visual and textual feature extractors
3. **Fusion Strategy Analysis**: Compare feature-level vs decision-level fusion
4. **Real-world Applicability**: Provide robust and accurate sentiment classification for social media analysis

---

## ⚠️ **Important Notes**

### Prerequisites
- ✅ Ensure all dependencies are installed (`pip install -r requirements.txt`)
- ✅ Verify dataset is properly preprocessed before training
- ✅ Check GPU availability for faster training

### Implementation Status
- ⚠️ This project implements only the **non-model building parts**
- 🔧 Model architecture implementation needs to be customized based on specific requirements
- 📝 Users need to implement their own model classes following the provided interfaces

### Best Practices
- 🔄 Use cross-validation for robust evaluation
- 📊 Monitor training curves to detect overfitting
- 💾 Save checkpoints regularly during training
- 📝 Log all experiments for reproducibility

---

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 **Contact**

For questions or support, please open an issue in the repository.

---

**Happy Coding!** 🚀
