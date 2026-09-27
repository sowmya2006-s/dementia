from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "mri" / "raw" / "Combined Dataset"
REPORT_PATH = ROOT / "results" / "tables" / "dataset_report.csv"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def inventory_dataset():
    rows = []
    total_images = 0

    for split_name in ["train", "test"]:
        split_dir = RAW_ROOT / split_name
        if not split_dir.exists():
            continue

        class_counts = defaultdict(int)
        format_counts = Counter()
        dimension_counts = Counter()
        corrupted = 0

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            for img_path in sorted(class_dir.iterdir()):
                if not img_path.is_file() or img_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue

                class_counts[class_dir.name] += 1
                total_images += 1

                try:
                    with Image.open(img_path) as image:
                        width, height = image.size
                        format_counts[image.format or img_path.suffix.upper().lstrip(".")] += 1
                        dimension_counts[f"{width}x{height}"] += 1
                except Exception:
                    corrupted += 1

        for class_name, count in sorted(class_counts.items()):
            rows.append(
                {
                    "split": split_name,
                    "class_name": class_name,
                    "file_count": count,
                    "dominant_dimensions": dimension_counts.most_common(3),
                    "format_counts": dict(sorted(format_counts.items())),
                    "corrupted_files": corrupted,
                    "dataset_total": total_images,
                }
            )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "split",
            "class_name",
            "file_count",
            "dominant_dimensions",
            "format_counts",
            "corrupted_files",
            "dataset_total",
        ])
        for row in rows:
            writer.writerow([
                row["split"],
                row["class_name"],
                row["file_count"],
                str(row["dominant_dimensions"]),
                str(row["format_counts"]),
                row["corrupted_files"],
                row["dataset_total"],
            ])

    print(f"Verified dataset inventory at {REPORT_PATH}")
    print(f"Total image files discovered: {total_images}")
    for row in rows:
        print(f"{row['split']}: {row['class_name']} -> {row['file_count']} files")


if __name__ == "__main__":
    inventory_dataset()
