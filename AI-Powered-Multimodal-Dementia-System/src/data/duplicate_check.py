from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "mri" / "raw" / "Combined Dataset"
REPORT_PATH = ROOT / "results" / "tables" / "duplicate_report.csv"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_duplicates():
    grouped = defaultdict(list)

    for split_name in ["train", "test"]:
        split_dir = RAW_ROOT / split_name
        if not split_dir.exists():
            continue

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            for img_path in sorted(class_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in VALID_EXTENSIONS:
                    grouped[hash_file(img_path)].append(str(img_path.relative_to(ROOT)))

    duplicates = {digest: paths for digest, paths in grouped.items() if len(paths) > 1}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["sha256", "duplicate_count", "paths"])
        for digest, paths in sorted(duplicates.items()):
            writer.writerow([digest, len(paths), " | ".join(paths)])

    if duplicates:
        print(f"Found {len(duplicates)} duplicated image groups.")
        for digest, paths in sorted(duplicates.items()):
            print(f"{digest}: {len(paths)} copies")
            for path in paths:
                print(f"  - {path}")
    else:
        print("No exact duplicates found in the MRI raw dataset.")

    print(f"Duplicate report saved to {REPORT_PATH}")


if __name__ == "__main__":
    check_duplicates()
