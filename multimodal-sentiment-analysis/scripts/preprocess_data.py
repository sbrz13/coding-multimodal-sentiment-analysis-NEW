import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.config import Config
from src.data_processing.preprocessor import TextPreprocessor, ImagePreprocessor, preprocess_dataset


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="预处理多模态情感分析数据集")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--input_dir", type=str, required=True, help="原始数据目录")
    parser.add_argument("--output_dir", type=str, default="data/", help="处理后数据保存目录")
    args = parser.parse_args()
    
    # 加载配置
    config = Config(args.config)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    
    # 初始化预处理器
    text_preprocessor = TextPreprocessor()
    image_preprocessor = ImagePreprocessor(config.data_config["image_size"])
    
    # 根据数据集名称进行预处理
    if config.data_config["dataset_name"] == "mvsa":
        preprocess_mvsa(args.input_dir, args.output_dir, text_preprocessor)
    elif config.data_config["dataset_name"] == "twitter":
        preprocess_twitter(args.input_dir, args.output_dir, text_preprocessor)
    else:
        raise ValueError(f"不支持的数据集: {config.data_config['dataset_name']}")
    
    print("数据预处理完成！")


def preprocess_mvsa(input_dir, output_dir, text_preprocessor):
    """预处理MVSA数据集"""
    # MVSA数据集原始结构：
    # input_dir/
    # ├── images/
    # └── mvsa.csv
    
    # 加载原始数据
    csv_file = os.path.join(input_dir, "mvsa.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"文件不存在: {csv_file}")
    
    data = pd.read_csv(csv_file)
    
    # 预处理文本
    data["text"] = data["text"].apply(text_preprocessor.preprocess)
    
    # 分割数据集
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # 保存处理后的数据
    train_data.to_csv(os.path.join(output_dir, "mvsa_train.csv"), index=False)
    val_data.to_csv(os.path.join(output_dir, "mvsa_val.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "mvsa_test.csv"), index=False)
    
    # 复制图像文件
    copy_images(os.path.join(input_dir, "images"), os.path.join(output_dir, "images"), data["image_id"])


def preprocess_twitter(input_dir, output_dir, text_preprocessor):
    """预处理Twitter多模态情感分析数据集"""
    # Twitter数据集原始结构：
    # input_dir/
    # ├── images/
    # └── twitter.csv
    
    # 加载原始数据
    csv_file = os.path.join(input_dir, "twitter.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"文件不存在: {csv_file}")
    
    data = pd.read_csv(csv_file)
    
    # 预处理文本
    data["text"] = data["text"].apply(text_preprocessor.preprocess)
    
    # 分割数据集
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # 保存处理后的数据
    train_data.to_csv(os.path.join(output_dir, "twitter_train.csv"), index=False)
    val_data.to_csv(os.path.join(output_dir, "twitter_val.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "twitter_test.csv"), index=False)
    
    # 复制图像文件
    copy_images(os.path.join(input_dir, "images"), os.path.join(output_dir, "images"), data["image_id"])


def copy_images(src_dir, dst_dir, image_ids):
    """复制图像文件"""
    import shutil
    
    for image_id in image_ids:
        src_path = os.path.join(src_dir, image_id)
        dst_path = os.path.join(dst_dir, image_id)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)


if __name__ == "__main__":
    main()
