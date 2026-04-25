import argparse
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.config import Config
from src.data_processing.preprocessor import TextPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess multimodal sentiment analysis dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--input_dir", type=str, required=True, help="Raw data directory")
    parser.add_argument("--output_dir", type=str, default="data/", help="Output directory")
    args = parser.parse_args()

    config = Config(args.config)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    text_preprocessor = TextPreprocessor()

    dataset_name = config.data_config["dataset_name"]
    if dataset_name == "mvsa":
        preprocess_mvsa(args.input_dir, args.output_dir, text_preprocessor)
    elif dataset_name in ("twitter", "twitter15", "twitter17"):
        preprocess_twitter(args.input_dir, args.output_dir, text_preprocessor, dataset_name)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    print("Data preprocessing complete!")


def preprocess_mvsa(input_dir, output_dir, text_preprocessor):
    csv_file = os.path.join(input_dir, "mvsa.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")

    data = pd.read_csv(csv_file)
    data["text"] = data["text"].apply(text_preprocessor.preprocess)
    data["label"] = data["label"].str.lower().str.strip()

    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data["label"])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data["label"])

    train_data.to_csv(os.path.join(output_dir, "mvsa_train.csv"), index=False)
    val_data.to_csv(os.path.join(output_dir, "mvsa_val.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "mvsa_test.csv"), index=False)

    copy_images(os.path.join(input_dir, "images"), os.path.join(output_dir, "images"), data["image_id"])

    print(f"MVSA dataset: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")


def preprocess_twitter(input_dir, output_dir, text_preprocessor, dataset_name="twitter"):
    csv_file = os.path.join(input_dir, "twitter.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")

    data = pd.read_csv(csv_file)
    data["text"] = data["text"].apply(text_preprocessor.preprocess)
    data["label"] = data["label"].str.lower().str.strip()

    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data["label"])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data["label"])

    train_data.to_csv(os.path.join(output_dir, f"{dataset_name}_train.csv"), index=False)
    val_data.to_csv(os.path.join(output_dir, f"{dataset_name}_val.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, f"{dataset_name}_test.csv"), index=False)

    copy_images(os.path.join(input_dir, "images"), os.path.join(output_dir, "images"), data["image_id"])

    print(f"Twitter dataset: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")


def copy_images(src_dir, dst_dir, image_ids):
    for image_id in image_ids:
        src_path = os.path.join(src_dir, str(image_id))
        dst_path = os.path.join(dst_dir, str(image_id))
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)


if __name__ == "__main__":
    main()
