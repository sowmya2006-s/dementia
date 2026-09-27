from __future__ import annotations

import csv
import json
from pathlib import Path

import librosa
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CLEAN_ROOT = ROOT / "data" / "audio" / "cleaned"
METADATA_ROOT = ROOT / "data" / "metadata"
OUTPUT_ROOT = ROOT / "outputs" / "audio" / "mfcc_v1"

SAMPLE_RATE = 16_000
N_MFCC = 40
N_MELS = 40
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 512
FMIN = 20
FMAX = SAMPLE_RATE // 2
CHUNK_SECONDS = 5
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
SEED = 42


def load_split(split: str) -> list[dict[str, str]]:
    path = METADATA_ROOT / f"audio_{split}.csv"
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def extract_chunks(audio: np.ndarray) -> list[np.ndarray]:
    if audio.size == 0:
        return []
    chunks = []
    for start in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[start:start + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        chunks.append(chunk.astype(np.float32, copy=False))
    return chunks


def extract_mfcc(chunk: np.ndarray) -> np.ndarray:
    return librosa.feature.mfcc(
        y=chunk,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
    ).astype(np.float32)


def build_split(split: str) -> tuple[np.ndarray, list[dict[str, object]]]:
    rows = load_split(split)
    features = []
    metadata = []
    for row in rows:
        audio_path = CLEAN_ROOT / row["file_path"]
        audio, sample_rate = librosa.load(
            str(audio_path), sr=SAMPLE_RATE, mono=True
        )
        for chunk_index, chunk in enumerate(extract_chunks(audio)):
            features.append(extract_mfcc(chunk))
            metadata.append({
                "chunk_index": chunk_index,
                "file_path": row["file_path"],
                "class": row["class"],
                "person_id": row["person_id"],
                "split": split,
                "sample_rate": SAMPLE_RATE,
                "chunk_seconds": CHUNK_SECONDS,
                "padded_final_chunk": len(audio) < (chunk_index + 1) * CHUNK_SAMPLES,
            })
    if not features:
        raise RuntimeError(f"No MFCC chunks generated for {split}")
    return np.stack(features), metadata


def write_metadata(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_person_leakage(metadata_by_split: dict[str, list[dict[str, object]]]) -> None:
    people = {
        split: {str(row["person_id"]) for row in rows}
        for split, rows in metadata_by_split.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = people[left] & people[right]
        if overlap:
            raise RuntimeError(f"MFCC person leakage between {left} and {right}: {sorted(overlap)}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    metadata_by_split = {}
    summary = {}
    for split in ("train", "validation", "test"):
        features, metadata = build_split(split)
        metadata_by_split[split] = metadata
        np.savez_compressed(OUTPUT_ROOT / f"{split}.npz", mfcc=features)
        write_metadata(OUTPUT_ROOT / f"{split}_metadata.csv", metadata)
        summary[split] = {
            "chunks": int(features.shape[0]),
            "mfcc": int(features.shape[1]),
            "frames": int(features.shape[2]),
            "recordings": len({row["file_path"] for row in metadata}),
            "persons": len({row["person_id"] for row in metadata}),
        }

    verify_person_leakage(metadata_by_split)
    config = {
        "feature_version": "MFCC_V1",
        "sample_rate": SAMPLE_RATE,
        "channels": "mono",
        "n_mfcc": N_MFCC,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
        "fmin": FMIN,
        "fmax": FMAX,
        "chunk_seconds": CHUNK_SECONDS,
        "chunk_policy": "non-overlapping; zero-pad final chunk",
        "seed": SEED,
        "summary": summary,
    }
    (OUTPUT_ROOT / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print("MFCC extraction completed")
    for split, values in summary.items():
        print(f"{split}: {values}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print("Person leakage check: PASS")


if __name__ == "__main__":
    main()