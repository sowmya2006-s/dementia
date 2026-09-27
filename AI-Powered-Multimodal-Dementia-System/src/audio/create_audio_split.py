from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLEAN_ROOT = ROOT / "data" / "audio" / "cleaned"
METADATA_ROOT = ROOT / "data" / "metadata"
TRAIN_CSV = METADATA_ROOT / "audio_train.csv"
VAL_CSV = METADATA_ROOT / "audio_validation.csv"
TEST_CSV = METADATA_ROOT / "audio_test.csv"
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def discover_records() -> list[dict[str, str]]:
    records = []
    for path in sorted(CLEAN_ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        relative = path.relative_to(CLEAN_ROOT)
        if len(relative.parts) < 3:
            raise ValueError(f"Expected class/person/file path, got: {relative}")
        records.append({
            "file_path": relative.as_posix(),
            "class": relative.parts[0],
            "person_id": relative.parts[1],
        })
    return records


def assign_people(records: list[dict[str, str]]) -> dict[str, str]:
    person_classes = defaultdict(set)
    person_files = Counter(record["person_id"] for record in records)
    for record in records:
        person_classes[record["person_id"]].add(record["class"])
    mixed_people = {
        person for person, classes in person_classes.items() if len(classes) > 1
    }
    if mixed_people:
        raise RuntimeError(f"People appear in multiple classes: {sorted(mixed_people)}")

    people_by_class = defaultdict(list)
    for person, classes in person_classes.items():
        people_by_class[next(iter(classes))].append(person)

    assignments = {}
    rng = random.Random(SEED)
    for class_name, people in sorted(people_by_class.items()):
        rng.shuffle(people)
        people.sort(key=lambda person: person_files[person], reverse=True)
        n_people = len(people)
        n_test = max(1, round(n_people * TEST_RATIO))
        n_val = max(1, round(n_people * VAL_RATIO))
        if n_test + n_val >= n_people:
            raise RuntimeError(f"Not enough people to create train split for {class_name}")
        for person in people[:n_test]:
            assignments[person] = "test"
        for person in people[n_test:n_test + n_val]:
            assignments[person] = "validation"
        for person in people[n_test + n_val:]:
            assignments[person] = "train"
    return assignments


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["file_path", "class", "person_id", "split"])
        writer.writeheader()
        writer.writerows(rows)


def create_audio_split() -> None:
    if not CLEAN_ROOT.exists():
        raise FileNotFoundError(f"Cleaned audio folder does not exist: {CLEAN_ROOT}")
    records = discover_records()
    if not records:
        raise FileNotFoundError(f"No cleaned audio files found under {CLEAN_ROOT}")
    assignments = assign_people(records)
    for record in records:
        record["split"] = assignments[record["person_id"]]

    write_csv(TRAIN_CSV, [r for r in records if r["split"] == "train"])
    write_csv(VAL_CSV, [r for r in records if r["split"] == "validation"])
    write_csv(TEST_CSV, [r for r in records if r["split"] == "test"])

    split_people = {
        split: {r["person_id"] for r in records if r["split"] == split}
        for split in ("train", "validation", "test")
    }
    split_files = {
        split: {r["file_path"] for r in records if r["split"] == split}
        for split in ("train", "validation", "test")
    }

    print("=" * 70)
    print("AUDIO GROUP-BASED TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)
    print(f"Total cleaned audio files: {len(records)}")
    print(f"Total people: {len({r['person_id'] for r in records})}")
    print("\nFILES")
    for split in ("train", "validation", "test"):
        print(f"{split.capitalize():11s}: {len(split_files[split])}")
    print("\nPERSONS")
    for split in ("train", "validation", "test"):
        print(f"{split.capitalize():11s}: {len(split_people[split])}")
    print("\nCLASS DISTRIBUTION")
    for split in ("train", "validation", "test"):
        counts = Counter(r["class"] for r in records if r["split"] == split)
        print(f"{split.capitalize()}: {dict(sorted(counts.items()))}")
    print("\nPERSON LEAKAGE CHECK")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        print(f"{left.capitalize()} INTERSECT {right}: {len(split_people[left] & split_people[right])}")
    print("\nFILE LEAKAGE CHECK")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        print(f"{left.capitalize()} INTERSECT {right}: {len(split_files[left] & split_files[right])}")
    print("\nREPRODUCIBILITY")
    print(f"Random seed: {SEED}")
    print(f"CSV files: {TRAIN_CSV}, {VAL_CSV}, {TEST_CSV}")
    print("Raw and cleaned audio files were not modified.")


if __name__ == "__main__":
    create_audio_split()