import os
import json
import random
from collections import defaultdict

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "MVSA_Single")
LABEL_FILE = os.path.join(DATA_ROOT, "labelResultAll.txt")
DATA_DIR = os.path.join(DATA_ROOT, "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2


def load_labels():
    samples = []
    with open(LABEL_FILE, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            sample_id = parts[0].strip()
            labels = parts[1].strip().split(",")
            if len(labels) < 2:
                continue
            text_label = labels[0].strip().lower()
            image_label = labels[1].strip().lower()

            txt_path = os.path.join(DATA_DIR, f"{sample_id}.txt")
            jpg_path = os.path.join(DATA_DIR, f"{sample_id}.jpg")
            if not os.path.exists(txt_path) or not os.path.exists(jpg_path):
                continue

            with open(txt_path, "r", encoding="utf-8", errors="replace") as tf:
                text = tf.read().strip()

            if text_label == image_label:
                fused_label = text_label
            else:
                fused_label = text_label

            valid_labels = {"positive", "neutral", "negative"}
            if fused_label not in valid_labels:
                continue

            samples.append({
                "id": sample_id,
                "text": text,
                "image_path": jpg_path,
                "text_label": text_label,
                "image_label": image_label,
                "label": fused_label,
            })
    return samples


def stratified_split(samples, train_ratio, val_ratio, test_ratio, seed):
    random.seed(seed)

    label_groups = defaultdict(list)
    for s in samples:
        label_groups[s["label"]].append(s)

    train, val, test = [], [], []
    for label, group in label_groups.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def save_split(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("Loading MVSA_Single data...")
    samples = load_labels()
    print(f"  Total valid samples: {len(samples)}")

    label_counts = defaultdict(int)
    for s in samples:
        label_counts[s["label"]] += 1
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count / len(samples) * 100:.1f}%)")

    print(f"\nSplitting {TRAIN_RATIO}:{VAL_RATIO}:{TEST_RATIO} (stratified)...")
    train, val, test = stratified_split(samples, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)

    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        lc = defaultdict(int)
        for s in split:
            lc[s["label"]] += 1
        print(f"  {name} label distribution: {dict(sorted(lc.items()))}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_split(train, os.path.join(OUTPUT_DIR, "train.json"))
    save_split(val, os.path.join(OUTPUT_DIR, "val.json"))
    save_split(test, os.path.join(OUTPUT_DIR, "test.json"))

    print(f"\nSaved to:")
    print(f"  {os.path.join(OUTPUT_DIR, 'train.json')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'val.json')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'test.json')}")


if __name__ == "__main__":
    main()
