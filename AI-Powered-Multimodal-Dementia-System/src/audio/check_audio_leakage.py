from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "audio" / "raw"
CLEAN_ROOT = ROOT / "data" / "audio" / "cleaned"
METADATA_ROOT = ROOT / "data" / "metadata"
DUPLICATE_REPORT = METADATA_ROOT / "audio_duplicate_report.csv"
SPEAKER_REPORT = METADATA_ROOT / "audio_speaker_report.csv"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as audio_file:
        for chunk in iter(lambda: audio_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_audio_root() -> tuple[Path, str]:
    cleaned_files = [
        path for path in CLEAN_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ] if CLEAN_ROOT.exists() else []
    if cleaned_files:
        return CLEAN_ROOT, "cleaned"
    return RAW_ROOT, "raw"


def possible_speaker(relative_path: Path) -> str:
    return relative_path.parts[1] if len(relative_path.parts) >= 3 else "UNKNOWN"


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def check_audio_leakage() -> None:
    audio_root, source = choose_audio_root()
    audio_files = sorted(
        path for path in audio_root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        raise FileNotFoundError(f"No audio files found under {audio_root}")

    hash_to_records: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    records = []
    for audio_path in audio_files:
        relative_path = audio_path.relative_to(audio_root)
        file_hash = sha256_file(audio_path)
        record = {
            "file_path": relative_path.as_posix(),
            "class": relative_path.parts[0] if relative_path.parts else "UNKNOWN",
            "sha256": file_hash,
            "possible_speaker_id": possible_speaker(relative_path),
            "source": source,
        }
        records.append(record)
        hash_to_records[file_hash].append(record)

    duplicate_rows = []
    duplicate_group = 0
    for file_hash, group in sorted(hash_to_records.items()):
        if len(group) < 2:
            continue
        duplicate_group += 1
        for record in group:
            duplicate_rows.append({
                "duplicate_group": duplicate_group,
                "sha256": file_hash,
                "file_path": record["file_path"],
                "class": record["class"],
                "possible_speaker_id": record["possible_speaker_id"],
                "duplicate_count": len(group),
                "source": source,
            })

    write_csv(
        DUPLICATE_REPORT,
        duplicate_rows,
        ["duplicate_group", "sha256", "file_path", "class",
         "possible_speaker_id", "duplicate_count", "source"],
    )
    write_csv(
        SPEAKER_REPORT,
        records,
        ["file_path", "class", "sha256", "possible_speaker_id", "source"],
    )

    speaker_counts = Counter(
        record["possible_speaker_id"] for record in records
        if record["possible_speaker_id"] != "UNKNOWN"
    )
    cross_class_speakers = {
        speaker for speaker in speaker_counts
        if len({record["class"] for record in records
                if record["possible_speaker_id"] == speaker}) > 1
    }

    print("=" * 70)
    print("AUDIO DUPLICATE + SPEAKER LEAKAGE CHECK")
    print("=" * 70)
    print(f"Audio source checked : {audio_root}")
    print(f"Source status        : {source.upper()}")
    print(f"Audio files found    : {len(records)}")
    if source == "raw":
        print("WARNING: cleaned audio is empty; these are preliminary raw-data results.")
    print("\nEXACT DUPLICATE SUMMARY")
    print(f"Unique SHA-256 hashes : {len(hash_to_records)}")
    print(f"Duplicate groups      : {duplicate_group}")
    print(f"Files involved        : {len(duplicate_rows)}")
    print("\nSPEAKER ID ANALYSIS")
    print(f"Possible speakers detected : {len(speaker_counts)}")
    print(f"Speakers in multiple classes: {len(cross_class_speakers)}")
    if speaker_counts:
        for speaker, count in sorted(speaker_counts.items()):
            print(f"{speaker:30s} {count}")
    else:
        print("No possible speaker IDs detected from folder structure.")
    print("\nREPORTS WRITTEN")
    print(f"Duplicate report : {DUPLICATE_REPORT}")
    print(f"Speaker report   : {SPEAKER_REPORT}")
    print("No audio files were deleted, modified, or moved.")


if __name__ == "__main__":
    check_audio_leakage()