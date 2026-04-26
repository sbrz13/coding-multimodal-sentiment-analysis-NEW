# Multimodal Sentiment Analysis Project

## 📋 Project Overview

This project implements a **multimodal sentiment analysis framework** supporting multiple model architectures and datasets. The project leverages **transfer learning** from pre-trained models and implements both **feature-level fusion** and **decision-level fusion** strategies.

### Key Features
- ✅ **Multi-Dataset Support**: MVSA-Single, Twitter2015, Twitter2017
- ✅ **Multiple Architectures**: Cross-Attention, Text-Only, Image-Only, Meta-Learner Ensemble
- ✅ **Advanced Attention Mechanisms**: Multi-Head Cross-Attention + Sequence-Level Attention + Modality Reliability Gate
- ✅ **Anti-Overfitting Strategies**: DropPath, FeatureNoiseInjection, MixUp, Data Augmentation
- ✅ **GPU-Accelerated Training**: Mixed Precision Training, Gradient Accumulation
- ✅ **Comprehensive Evaluation**: Accuracy, F1, Precision, Recall, and more

---

## 🚀 Quick Start: Step-by-Step Setup Guide

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd multimodal-sentiment-analysis
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Datasets

#### Option A: Use MVSA-Single Only
1. Download MVSA-Single from [MCR-lab](https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/)
2. Place the data in `data/MVSA_Single/`
3. Run the split script:
   ```bash
   python scripts/split_dataset.py
   ```

#### Option B: Use All Three Datasets (Recommended)
1. Download all three datasets (see [Supported Datasets](#-supported-datasets) section for links)
2. Place MVSA-Single in `data/MVSA_Single/`
3. Place Twitter2015/2017 in `data/data/IJCAI2019_data/`
4. Run the merge script:
   ```bash
   python scripts/merge_datasets.py
   ```

### Step 4: Verify Setup
```bash
python scripts/verify_pretrained_weights.py
```

### Step 5: Start Training
```bash
bash train_gpu.sh
```

### Step 6: View Results
```bash
python scripts/analysis.py
```
Generated figures will be saved to `results/analysis/`.

---

## 📦 Required Packages and Dependencies

### Core Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.24.0 | Numerical operations |
| pandas | >=2.0.0 | Data manipulation |
| scikit-learn | >=1.3.0 | Machine learning utilities |
| torch | >=2.0.0 | Deep learning framework |
| torchvision | >=0.15.0 | Computer vision utilities |
| transformers | >=4.30.0 | Pre-trained model library |
| nltk | >=3.8.0 | Natural language processing |
| Pillow | >=10.0.0 | Image processing |
| opencv-python | >=4.8.0 | Computer vision |
| datasets | >=2.14.0 | Dataset management |
| matplotlib | >=3.7.0 | Visualization |
| seaborn | >=0.12.0 | Statistical visualization |
| tqdm | >=4.65.0 | Progress bars |
| joblib | >=1.3.0 | Model serialization |
| scipy | >=1.10.0 | Scientific computing |

### System Requirements
- **Python**: 3.8+
- **CUDA**: 12.4+ (for GPU training)
- **GPU**: NVIDIA GPU with 12GB+ VRAM (V5 Large Models)
- **RAM**: 32GB+ recommended
- **Storage**: 10GB+ (for model cache and data)

### Install All Dependencies
```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
multimodal-sentiment-analysis/
├── data/                           # Data directory
│   ├── data/                       # Processed data
│   │   └── IJCAI2019_data/         # Twitter2015/2017 raw data
│   └── MVSA_Single/                # MVSA-Single raw data
├── src/                            # Source code
│   ├── config/                     # Configuration module
│   │   ├── __init__.py
│   │   └── config.py              # Configuration management
│   ├── data_processing/            # Data processing module
│   │   ├── __init__.py
│   │   ├── data_loader.py         # Dataset and DataLoader
│   │   └── preprocessor.py        # Data preprocessing
│   ├── evaluation/                 # Evaluation module
│   │   ├── __init__.py
│   │   └── metrics.py             # Evaluation metrics
│   └── training/                   # Training module
│       ├── __init__.py
│       └── models.py              # Model definitions
├── scripts/                        # Scripts directory
│   ├── merge_datasets.py          # Merge 3 datasets into unified format
│   ├── preprocess_data.py         # Data preprocessing
│   ├── run_evaluation.py          # Model evaluation
│   ├── run_training.py            # Two-stage training pipeline
│   ├── split_dataset.py           # Stratified train/val/test split
│   ├── analysis.py                # Results visualization
│   └── verify_pretrained_weights.py # Weight verification
├── results/                        # Results directory
│   ├── models/                    # Saved model weights
│   ├── analysis/                  # Generated figures
│   ├── v1_baseline/               # V1 baseline results
│   ├── v2_enhanced/               # V2 enhanced results
│   ├── v3_anti_overfit/           # V3 anti-overfit results
│   └── v4_optimized/              # V4 optimized results
├── config_cpu.json                 # CPU training configuration
├── config_gpu.json                 # GPU training configuration
├── requirements.txt                # Dependencies
├── setup.py                        # Installation configuration
├── train_gpu.sh                    # GPU training script
└── README.md                       # Project documentation
```

### Module Descriptions

| Module | File | Functionality |
|--------|------|---------------|
| **config** | `config.py` | Central configuration management, loads JSON configs |
| **data_processing** | `data_loader.py` | Dataset class, DataLoader, image augmentation, MixUp collator |
| **data_processing** | `preprocessor.py` | Text tokenization, image preprocessing |
| **evaluation** | `metrics.py` | Accuracy, F1, Precision, Recall, classification reports |
| **training** | `models.py` | CrossAttentionModel, TextOnlyModel, ImageOnlyModel, Meta-Learner |
| **scripts** | `run_training.py` | Two-stage training: base models + meta-learner ensemble |
| **scripts** | `merge_datasets.py` | Merges MVSA-Single + Twitter2015 + Twitter2017 |
| **scripts** | `split_dataset.py` | Stratified 6:2:2 train/val/test split |
| **scripts** | `analysis.py` | Generates performance comparison figures |

---

## 📚 Supported Datasets

| Dataset | Samples | Classes | Description | Download Link |
|---------|---------|---------|-------------|---------------|
| **MVSA-Single** | ~2,000 | 3 (Neg/Neu/Pos) | High-quality image-text pairs from social media | [MCR-lab](https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/) |
| **Twitter2015** | ~2,500 | 3 | Twitter posts with images, real-world noise | [IJCAI2019 Data](https://ieee-dataport.org/documents/twitterdata) |
| **Twitter2017** | ~3,000 | 3 | Twitter posts with images, real-world noise | [IJCAI2019 Data](https://ieee-dataport.org/documents/twitterdata) |

### Dataset Format

#### MVSA-Single (JSON format)
```json
{
  "id": "sample_id",
  "text": "Sample text content",
  "image_path": "/path/to/image.jpg",
  "label": "positive"
}
```

#### Twitter2015/2017 (TSV format)
```
id	label	image_id	text_masked	target
1	0	img001.jpg	I love $T$!	apple
```

### Data Processing Pipeline

This project uses a **multi-step data processing pipeline** to prepare datasets for training:

#### Step 1: MVSA-Single Stratified Split
The `split_dataset.py` script performs a **stratified 6:2:2 split** on MVSA-Single:
1. Reads `labelResultAll.txt` to get text and image labels
2. Fuses labels (text label takes priority when text and image labels differ)
3. Groups samples by label and shuffles within each group
4. Splits into train (60%), val (20%), test (20%) maintaining class distribution
5. Saves as `train.json`, `val.json`, `test.json`

#### Step 2: Multi-Dataset Merge
The `merge_datasets.py` script combines all three datasets:
1. **MVSA-Single**: Parses JSON format, filters by image existence
2. **Twitter2015/2017**: Parses TSV format, replaces `$T$` placeholder with target entity
3. **Merges** train and val sets from all three datasets
4. **Keeps test sets separate** for per-dataset evaluation
5. Saves as `merged_train.json`, `merged_val.json`, `test_mvsa_single.json`, `test_twitter2015.json`, `test_twitter2017.json`, `merged_test.json`

#### Output JSON Format
All processed data follows a unified format:
```json
[
  {
    "id": "dataset_sample_id",
    "text": "Processed text content",
    "image_path": "/absolute/path/to/image.jpg",
    "label": "positive",
    "dataset": "mvsa_single"
  }
]
```

### Dataset Statistics (After Merging)
| Split | Samples |
|-------|---------|
| Merged Train | ~5,500 |
| Merged Val | ~1,800 |
| MVSA-Single Test | ~400 |
| Twitter2015 Test | ~500 |
| Twitter2017 Test | ~600 |
| **Total** | **~8,800** |

---

## 🚀 Model Training

### GPU Training (Recommended)

```bash
bash train_gpu.sh
```

### Custom Training

```bash
python scripts/run_training.py \
    --config config_gpu.json \
    --model_name multimodal_sentiment
```

### Training Pipeline

The project uses a **two-stage training strategy**:

**Stage 1: Training Base Models**
1. Cross-Attention Model (multimodal fusion)
2. Text-Only Model (unimodal text)
3. Image-Only Model (unimodal image)

**Stage 2: Training Meta-Learner**
- Logistic Regression meta-learner to ensemble predictions from three base models
- Trained on validation set to improve overall performance

### Training Configuration

#### config_gpu.json Key Parameters

```json
{
    "data_config": {
        "batch_size": 8,
        "use_merged_data": true,      // Use merged datasets
        "augment": true,              // Enable data augmentation
        "use_weighted_sampling": true, // Weighted sampling (handle class imbalance)
        "use_mixup": true             // MixUp augmentation
    },
    "train_config": {
        "learning_rate": 3e-6,
        "encoder_learning_rate": 8e-7,     // Text encoder learning rate
        "image_encoder_learning_rate": 4e-7, // Image encoder learning rate (lower)
        "fusion_learning_rate": 2e-5,       // Fusion layer learning rate
        "classifier_learning_rate": 6e-5,   // Classifier learning rate
        "num_epochs": 50,
        "label_smoothing": 0.1,
        "gradient_accumulation_steps": 4,   // Gradient accumulation
        "early_stop_patience": 12,
        "warmup_ratio": 0.2
    },
    "model_config": {
        "text_model": "roberta-large",
        "vision_model": "openai/clip-vit-large-patch14",
        "projection_dim": 768,
        "num_attention_heads": 12,
        "cross_attn_layers": 6,
        "text_dropout_rate": 0.15,
        "image_dropout_rate": 0.4,          // Higher image dropout
        "drop_path_rate": 0.2,              // DropPath regularization
        "image_noise_scale": 0.1,           // Feature noise
        "contrastive_weight": 0.15,         // Contrastive learning weight
        "use_reliability_gate": true        // Modality reliability gate
    }
}
```

---

## 📈 Model Evaluation

### Evaluate All Models

```bash
python scripts/run_evaluation.py \
    --config config_gpu.json \
    --model_dir results/models/
```

### Evaluate Single Model

```bash
python scripts/run_evaluation.py \
    --config config_gpu.json \
    --model_path results/models/multimodal_sentiment_cross_attn_best.pt \
    --model_name cross_attention
```

### Evaluation Metrics

The project provides comprehensive evaluation metrics:
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Macro, Micro, Weighted F1
- **Precision**: Macro, Weighted Precision
- **Recall**: Macro, Weighted Recall
- **Classification Report**: Detailed metrics for each class

---

## 📊 Results Visualization

### Generate Analysis Figures

```bash
python scripts/analysis.py
```

Generated figures (saved to `results/analysis/`):
- `baseline_acc_f1.png` - Unimodal vs Multimodal performance
- `feature_level_acc.png` - Fusion strategy comparison
- `version_comparison_acc_f1.png` - Model version evolution
- `cross_attn_cm.png` - Cross-Attention confusion matrix
- `decision_level_acc.png` - Decision-level fusion comparison
- `cross_attn_auc.png` - Cross-Attention ROC curve
- `metalearner_auc.png` - Meta-Learner ROC curve
- `metalearner_cm.png` - Meta-Learner confusion matrix
- `per_dataset_comparison.png` - Per-dataset model comparison

---

## 🤖 Supported Model Architectures

### 1. Cross-Attention Model

**Key Features**:
- Multi-Head Cross-Attention mechanism (12 heads, 6 layers)
- Sequence-Level Attention fusion
- Bidirectional attention: Text→Image and Image→Text
- Modality Reliability Gate (dynamic text/image weighting)
- Contrastive learning loss (cross-modal alignment)

**Use Case**: Scenarios requiring full utilization of both text and image information

### 2. Text-Only Model

**Key Features**:
- RoBERTa-large text encoder (355M parameters)
- Three-layer classifier
- Text dropout regularization

**Use Case**: Pure text sentiment analysis

### 3. Image-Only Model

**Key Features**:
- CLIP-ViT-Large image encoder (307M parameters)
- Feature noise injection
- High dropout rate to prevent overfitting

**Use Case**: Pure image sentiment analysis

### 4. Meta-Learner (Ensemble Model)

**Key Features**:
- Logistic Regression meta-learner
- Ensembles prediction probabilities from three base models
- Trained on validation set

**Use Case**: Scenarios requiring best overall performance

---

## 🔬 Technical Principles

### Cross-Attention Mechanism

The core of this project is the **bidirectional cross-attention mechanism** that enables deep fusion between text and image modalities:

#### How It Works
1. **Text→Image Attention**: Text tokens query image patches, extracting visual information relevant to each word
2. **Image→Text Attention**: Image patches query text tokens, extracting textual context relevant to each visual region
3. **Multi-Head Processing**: 12 attention heads capture different types of cross-modal relationships simultaneously
4. **Sequence-Level Attention**: After cross-attention, a sequence-level attention mechanism weights the importance of each token/patch

#### Why Cross-Attention > Simple Concatenation
- **Dynamic Alignment**: Learns which text tokens correspond to which image regions
- **Context-Aware Fusion**: Text features are enhanced with relevant visual context, and vice versa
- **Fine-Grained Interaction**: Operates at token/patch level rather than global feature level

### Modality Reliability Gate

A learnable gating mechanism that dynamically adjusts the contribution of each modality:

```
gate_weight = sigmoid(MLP(text_features, image_features))
fused_features = gate_weight * text_features + (1 - gate_weight) * image_features
```

**Benefits**:
- Automatically downweights noisy or irrelevant images
- Handles cases where one modality is more informative than the other
- Prevents the model from being misled by poor-quality inputs

### Contrastive Learning

Adds a contrastive loss term to encourage cross-modal alignment:

- **Positive Pairs**: Text and image from the same sample should have similar representations
- **Negative Pairs**: Text and image from different samples should have dissimilar representations
- **Temperature Parameter**: Controls the sharpness of the similarity distribution

This helps the model learn a shared semantic space where related text-image pairs are close together.

### Two-Stage Training Strategy

**Stage 1: Base Model Training**
- Each model (Cross-Attention, Text-Only, Image-Only) is trained independently
- Allows each model to specialize in its modality
- Uses different hyperparameters optimized for each architecture

**Stage 2: Meta-Learner Training**
- A logistic regression model is trained on validation set predictions
- Learns optimal weights for combining predictions from base models
- Provides robust ensemble performance

### Anti-Overfitting Design

| Strategy | Purpose | Implementation |
|----------|---------|----------------|
| **DropPath** | Stochastic depth | Randomly drops attention blocks during training |
| **Feature Noise** | Robustness | Adds Gaussian noise to features |
| **MixUp** | Data augmentation | Interpolates samples and labels |
| **Label Smoothing** | Regularization | Softens target labels |
| **Weighted Sampling** | Class balance | Oversamples minority classes |
| **Hierarchical LR** | Preservation | Lower LR for pre-trained encoders |

---

## 🎯 Anti-Overfitting Strategies

The project implements multi-level anti-overfitting strategies:

### Data Level
- ✅ **Image Augmentation**: Cutout, RandomErasing, MixUp, Color Jitter, Rotation
- ✅ **Weighted Sampling**: Handle class imbalance
- ✅ **Label Smoothing**: Reduce model overfitting risk

### Model Level
- ✅ **DropPath**: Stochastic depth regularization
- ✅ **FeatureNoiseInjection**: Feature noise injection
- ✅ **Differentiated Dropout**: Image dropout (0.4) > Text dropout (0.15)
- ✅ **Modality Reliability Gate**: Auto-downweight noisy/irrelevant images

### Training Strategy
- ✅ **Hierarchical Learning Rates**: Different learning rates for different components
- ✅ **Gradient Accumulation**: More stable training
- ✅ **Early Stopping**: Prevent overfitting
- ✅ **Cosine Learning Rate Schedule**: Dynamic learning rate adjustment

---

## 🔍 Advanced Features

### Verify Pretrained Weights

Ensure pre-trained models are loaded correctly:

```bash
python scripts/verify_pretrained_weights.py
```

### Check Dataset Statistics

View class distribution, sampling weights, etc.:

```bash
python scripts/merge_datasets.py --check_stats
```

### Mixed Precision Training

Automatically enable AMP (Automatic Mixed Precision) for faster training:

```python
# Already enabled in config_gpu.json
"train_config": {
    "use_amp": true
}
```

---

## ⚠️ Important Notes

### Hardware Requirements
- **GPU**: NVIDIA GPU 12GB+ VRAM
- **CPU**: Can be used but training is slower
- **RAM**: 32GB+ recommended
- **Storage**: 10GB+ (model cache)

### Training Time
- **Base Models**: ~2-3 hours (RTX 3080 Ti)
- **Large Models**: ~5-8 hours (RTX 3080 Ti)

### Model Cache
Models are cached locally after first download:
- RoBERTa-large: ~1.4GB
- CLIP-ViT-Large: ~1.7GB
- **Total**: ~3.1GB

---

## 📄 License

This project is licensed under the MIT License

---

## 🤝 Contributing

Issues and Pull Requests are welcome!

---

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Happy Coding!** 🚀
