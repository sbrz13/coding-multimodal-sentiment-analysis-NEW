import json
import os
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
IJCAI_DIR = os.path.join(DATA_DIR, "data", "IJCAI2019_data")
OUTPUT_DIR = DATA_DIR

LABEL_MAP_TSV = {"0": "negative", "1": "neutral", "2": "positive"}


def parse_twitter_tsv(tsv_path, images_dir, dataset_name):
    samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            idx = row[0].strip()
            label = row[1].strip()
            image_id = row[2].strip()
            text_masked = row[3].strip()
            target = row[4].strip() if len(row) > 4 else ""

            text = text_masked.replace("$T$", target).strip()
            if not text:
                text = target.strip()

            image_path = os.path.join(images_dir, image_id)
            if not os.path.exists(image_path):
                continue

            label_str = LABEL_MAP_TSV.get(label, "neutral")

            samples.append({
                "id": f"{dataset_name}_{idx}",
                "text": text,
                "image_path": image_path,
                "label": label_str,
                "dataset": dataset_name,
            })
    return samples


def parse_mvsa_json(json_path, dataset_name="mvsa_single"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for item in data:
        image_path = item.get("image_path", "")
        if not os.path.exists(image_path):
            continue
        samples.append({
            "id": f"{dataset_name}_{item['id']}",
            "text": item.get("text", ""),
            "image_path": image_path,
            "label": item.get("label", "neutral"),
            "dataset": dataset_name,
        })
    return samples


def main():
    print("=" * 60)
    print("  合并三数据集: MVSA-Single + Twitter2015 + Twitter2017")
    print("=" * 60)

    all_train = []
    all_val = []
    test_by_dataset = {}

    # MVSA-Single
    print("\n[1/3] 解析 MVSA-Single...")
    for split in ["train", "val", "test"]:
        json_path = os.path.join(DATA_DIR, f"{split}.json")
        if os.path.exists(json_path):
            samples = parse_mvsa_json(json_path, "mvsa_single")
            print(f"  {split}: {len(samples)} 条 (图片存在)")
            if split == "train":
                all_train.extend(samples)
            elif split == "val":
                all_val.extend(samples)
            else:
                test_by_dataset["mvsa_single"] = samples
        else:
            print(f"  {split}: 文件不存在")

    # Twitter2015
    print("\n[2/3] 解析 Twitter2015...")
    tw15_img_dir = os.path.join(IJCAI_DIR, "twitter2015_images")
    for split, tsv_name in [("train", "train"), ("val", "dev"), ("test", "test")]:
        tsv_path = os.path.join(IJCAI_DIR, "twitter2015", f"{tsv_name}.tsv")
        if os.path.exists(tsv_path):
            samples = parse_twitter_tsv(tsv_path, tw15_img_dir, "twitter2015")
            print(f"  {split}: {len(samples)} 条 (图片存在)")
            if split == "train":
                all_train.extend(samples)
            elif split == "val":
                all_val.extend(samples)
            else:
                test_by_dataset["twitter2015"] = samples
        else:
            print(f"  {split}: 文件不存在")

    # Twitter2017
    print("\n[3/3] 解析 Twitter2017...")
    tw17_img_dir = os.path.join(IJCAI_DIR, "twitter2017_images")
    for split, tsv_name in [("train", "train"), ("val", "dev"), ("test", "test")]:
        tsv_path = os.path.join(IJCAI_DIR, "twitter2017", f"{tsv_name}.tsv")
        if os.path.exists(tsv_path):
            samples = parse_twitter_tsv(tsv_path, tw17_img_dir, "twitter2017")
            print(f"  {split}: {len(samples)} 条 (图片存在)")
            if split == "train":
                all_train.extend(samples)
            elif split == "val":
                all_val.extend(samples)
            else:
                test_by_dataset["twitter2017"] = samples
        else:
            print(f"  {split}: 文件不存在")

    print("\n" + "=" * 60)
    print("  合并统计")
    print("=" * 60)
    print(f"  合并训练集: {len(all_train)} 条")
    print(f"  合并验证集: {len(all_val)} 条")
    for ds, samples in test_by_dataset.items():
        print(f"  {ds} 测试集: {len(samples)} 条")
    total = len(all_train) + len(all_val) + sum(len(s) for s in test_by_dataset.values())
    print(f"  总计: {total} 条")

    # 保存合并后的训练集和验证集
    merged_train_path = os.path.join(OUTPUT_DIR, "merged_train.json")
    merged_val_path = os.path.join(OUTPUT_DIR, "merged_val.json")
    with open(merged_train_path, "w", encoding="utf-8") as f:
        json.dump(all_train, f, ensure_ascii=False, indent=2)
    print(f"\n  已保存: {merged_train_path}")
    with open(merged_val_path, "w", encoding="utf-8") as f:
        json.dump(all_val, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {merged_val_path}")

    # 保存各数据集的测试集
    for ds, samples in test_by_dataset.items():
        test_path = os.path.join(OUTPUT_DIR, f"test_{ds}.json")
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"  已保存: {test_path}")

    # 保存合并测试集（用于整体评估）
    all_test = []
    for samples in test_by_dataset.values():
        all_test.extend(samples)
    merged_test_path = os.path.join(OUTPUT_DIR, "merged_test.json")
    with open(merged_test_path, "w", encoding="utf-8") as f:
        json.dump(all_test, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {merged_test_path}")

    # 标签分布统计
    from collections import Counter
    print("\n" + "=" * 60)
    print("  标签分布")
    print("=" * 60)
    for name, data in [("训练集", all_train), ("验证集", all_val)]:
        labels = [s["label"] for s in data]
        dist = Counter(labels)
        print(f"  {name}: {dict(dist)}")
    for ds, samples in test_by_dataset.items():
        labels = [s["label"] for s in samples]
        dist = Counter(labels)
        print(f"  {ds}测试集: {dict(dist)}")


if __name__ == "__main__":
    main()
