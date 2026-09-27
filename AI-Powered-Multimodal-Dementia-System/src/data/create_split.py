from __future__ import annotations

from pathlib import Path
import csv
import random

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "mri" / "raw" / "Combined Dataset"
METADATA_DIR = ROOT / "data" / "mri" / "metadata"
SEED = 42

LABEL_MAP = {
    "No Impairment": "No-Demented",
    "Very Mild Impairment": "Very Mildly Demented",
    "Mild Impairment": "Mildly Demented",
    "Moderate Impairment": "Moderately Demented",
}


def write_split(csv_path: Path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "class"])
        for row in rows:
            writer.writerow([row[0], row[1]])


def build_metadata():
    train_root = RAW_ROOT / "train"
    test_root = RAW_ROOT / "test"
    split_rows = {"train": [], "validation": [], "test": []}
    rng = random.Random(SEED)

    for class_dir in sorted(train_root.iterdir()):
        if not class_dir.is_dir():
            continue

        files = sorted(class_dir.iterdir())
        label = LABEL_MAP.get(class_dir.name, class_dir.name)
        rng.shuffle(files)

        val_count = max(1, round(len(files) * 0.15))
        train_files = files[val_count:]
        val_files = files[:val_count]

        for img_path in train_files:
            rel_path = img_path.relative_to(ROOT).as_posix()
            split_rows["train"].append((rel_path, label))

        for img_path in val_files:
            rel_path = img_path.relative_to(ROOT).as_posix()
            split_rows["validation"].append((rel_path, label))

    for class_dir in sorted(test_root.iterdir()):
        if not class_dir.is_dir():
            continue

        label = LABEL_MAP.get(class_dir.name, class_dir.name)
        for img_path in sorted(class_dir.iterdir()):
            if img_path.is_file():
                rel_path = img_path.relative_to(ROOT).as_posix()
                split_rows["test"].append((rel_path, label))

    write_split(METADATA_DIR / "train.csv", split_rows["train"])
    write_split(METADATA_DIR / "validation.csv", split_rows["validation"])
    write_split(METADATA_DIR / "test.csv", split_rows["test"])

    print(f"Created train/validation/test CSV metadata under {METADATA_DIR}")
    for key, rows in split_rows.items():
        print(f"{key}: {len(rows)} rows")


if __name__ == "__main__":
    build_metadata()
