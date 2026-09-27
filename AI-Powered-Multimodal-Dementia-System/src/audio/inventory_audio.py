from __future__ import annotations

import contextlib
import csv
import wave
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIO_ROOT = ROOT / "data" / "audio" / "raw"
OUTPUT_CSV = ROOT / "data" / "metadata" / "audio_inventory.csv"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def inspect_wav(path: Path) -> dict[str, object]:
    with contextlib.closing(wave.open(str(path), "rb")) as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.getnframes()
        duration = frames / sample_rate if sample_rate > 0 else None
    return {
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "frames": frames,
        "status": "OK",
    }


def inventory_audio() -> None:
    if not AUDIO_ROOT.exists():
        raise FileNotFoundError(f"Audio dataset folder does not exist: {AUDIO_ROOT}")

    audio_files = sorted(
        path
        for path in AUDIO_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    records = []
    corrupt_files = []

    for audio_path in audio_files:
        relative_path = audio_path.relative_to(AUDIO_ROOT)
        class_name = relative_path.parts[0] if len(relative_path.parts) > 1 else "UNKNOWN"
        file_format = audio_path.suffix.lower()
        record = {
            "file_path": relative_path.as_posix(),
            "class": class_name,
            "format": file_format,
            "duration_seconds": None,
            "sample_rate": None,
            "channels": None,
            "sample_width": None,
            "frames": None,
            "status": "NOT_YET_DECODED",
        }

        if file_format == ".wav":
            try:
                record.update(inspect_wav(audio_path))
            except Exception as error:
                record["status"] = "CORRUPT"
                corrupt_files.append(f"{relative_path.as_posix()}: {error}")

        records.append(record)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "file_path", "class", "format", "duration_seconds", "sample_rate",
        "channels", "sample_width", "frames", "status",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    class_counts = Counter(record["class"] for record in records)
    format_counts = Counter(record["format"] for record in records)
    status_counts = Counter(record["status"] for record in records)
    durations = [
        record["duration_seconds"]
        for record in records
        if record["duration_seconds"] is not None
    ]

    print("=" * 60)
    print("AUDIO DATASET INVENTORY")
    print("=" * 60)
    print(f"Dataset folder          : {AUDIO_ROOT}")
    print(f"Total audio files found : {len(records)}")
    print("\nCLASS DISTRIBUTION")
    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name:20s} {count}")
    print("\nFILE FORMAT DISTRIBUTION")
    for file_format, count in sorted(format_counts.items()):
        print(f"{file_format:10s} {count}")
    print("\nFILE STATUS")
    for status, count in sorted(status_counts.items()):
        print(f"{status:20s} {count}")
    if durations:
        print("\nWAV DURATION STATISTICS")
        print(f"Minimum duration : {min(durations):.2f} sec")
        print(f"Maximum duration : {max(durations):.2f} sec")
        print(f"Average duration : {sum(durations) / len(durations):.2f} sec")
        print(f"WAV files measured : {len(durations)}")
    print("\nCORRUPT FILES")
    if corrupt_files:
        print(f"Corrupt WAV files : {len(corrupt_files)}")
        for file_name in corrupt_files:
            print(file_name)
    else:
        print("No corrupt WAV files detected.")
    print("\nINVENTORY COMPLETED")
    print(f"Inventory CSV : {OUTPUT_CSV}")
    print("No audio files were modified, deleted, or moved.")


if __name__ == "__main__":
    inventory_audio()