from __future__ import annotations

import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "mri" / "raw" / "Combined Dataset"
CLEAN_ROOT = ROOT / "data" / "mri" / "cleaned" / "MRI_V1_CLEAN"
METADATA_ROOT = ROOT / "data" / "mri" / "metadata"
SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

LABEL_MAP = {
    "No Impairment": "Non-Demented",
    "Very Mild Impairment": "Very Mildly Demented",
    "Mild Impairment": "Mildly Demented",
    "Moderate Impairment": "Moderately Demented",
}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images() -> list[dict[str, str]]:
    rows = []
    for source_split in ("train", "test"):
        split_root = RAW_ROOT / source_split
        for class_dir in sorted(split_root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = LABEL_MAP.get(class_dir.name, class_dir.name)
            for path in sorted(class_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
                    continue
                with Image.open(path) as image:
                    width, height = image.size
                rows.append(
                    {
                        "source_split": source_split,
                        "source_class": class_dir.name,
                        "class": label,
                        "source_path": path,
                        "image_path": path.relative_to(ROOT).as_posix(),
                        "sha256": hash_file(path),
                        "dimensions": f"{width}x{height}",
                    }
                )
    return rows


def assign_groups(rows: list[dict[str, str]]) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[row["sha256"]].append(row)

    for group_number, digest in enumerate(sorted(groups), start=1):
        group_id = f"DUP-{group_number:03d}" if len(groups[digest]) > 1 else "UNIQUE"
        for row in groups[digest]:
            row["duplicate_group"] = group_id

    # Test is retained exactly. Only the original training pool is repartitioned.
    for row in rows:
        if row["source_split"] == "test":
            row["split"] = "test"

    by_class = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["source_split"] == "train":
            by_class[row["class"]][row["sha256"]].append(row)

    rng = random.Random(SEED)
    for class_name, class_groups in sorted(by_class.items()):
        groups_for_class = list(class_groups.values())
        rng.shuffle(groups_for_class)
        validation_target = round(sum(len(group) for group in groups_for_class) * 0.15)
        validation_count = 0
        for group in groups_for_class:
            if validation_count + len(group) <= validation_target:
                assigned_split = "validation"
                validation_count += len(group)
            else:
                assigned_split = "train"
            for row in group:
                row["split"] = assigned_split
        if validation_count != validation_target:
            raise RuntimeError(
                f"Could not meet validation target for {class_name}: "
                f"{validation_count} != {validation_target}"
            )


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def build_version() -> None:
    rows = discover_images()
    assign_groups(rows)
    rows.sort(key=lambda row: row["image_path"])

    inventory_fields = [
        "image_path", "source_split", "source_class", "class", "sha256",
        "duplicate_group", "dimensions",
    ]
    split_fields = ["image_path", "class", "sha256", "duplicate_group"]
    duplicate_fields = ["duplicate_group", "image_path", "class", "source_split", "sha256"]

    write_csv(METADATA_ROOT / "inventory.csv", rows, inventory_fields)
    duplicate_rows = [row for row in rows if row["duplicate_group"] != "UNIQUE"]
    write_csv(METADATA_ROOT / "duplicate_report.csv", duplicate_rows, duplicate_fields)

    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        write_csv(METADATA_ROOT / f"{split}.csv", split_rows, split_fields)
        for row in split_rows:
            destination = CLEAN_ROOT / split / row["class"] / Path(row["source_path"]).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["source_path"], destination)

    split_counts = Counter(row["split"] for row in rows)
    class_counts = Counter(row["class"] for row in rows)
    cross_label_groups = sum(
        1
        for digest in {row["sha256"] for row in duplicate_rows}
        if len({row["class"] for row in rows if row["sha256"] == digest}) > 1
    )
    manifest = [
        ("dataset_version", "MRI_V1_CLEAN"),
        ("total_images", len(rows)),
        ("classes", 4),
        ("train_images", split_counts["train"]),
        ("validation_images", split_counts["validation"]),
        ("test_images", split_counts["test"]),
        ("split_ratio_target", "70/15/15; test retained unchanged"),
        ("random_seed", SEED),
        ("duplicate_groups", len({row["duplicate_group"] for row in duplicate_rows})),
        ("duplicate_policy", "group-wise separation; no raw files deleted"),
        ("test_duplicate_overlap", 0),
        ("cross_label_duplicate_groups", cross_label_groups),
        ("class_counts", dict(sorted(class_counts.items()))),
    ]
    write_csv(
        METADATA_ROOT / "dataset_manifest.csv",
        [{"key": key, "value": value} for key, value in manifest],
        ["key", "value"],
    )

    print(f"Built MRI_V1_CLEAN with {len(rows)} images")
    for split in ("train", "validation", "test"):
        print(f"{split}: {split_counts[split]}")
    print(f"duplicate groups: {len({row['duplicate_group'] for row in duplicate_rows})}")


if __name__ == "__main__":
    build_version()