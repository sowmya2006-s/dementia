from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
AUDIO_ROOT = ROOT / "data" / "audio" / "raw"
CLEAN_ROOT = ROOT / "data" / "audio" / "cleaned"
REPORT_PATH = ROOT / "data" / "metadata" / "audio_cleaning_report.csv"
TARGET_SR = 16_000
MIN_DURATION = 0.5
MAX_DURATION = 60.0
SILENCE_RMS_THRESHOLD = 1e-4
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def decode_audio(path: Path) -> tuple[np.ndarray, int, str, str]:
    try:
        audio, sample_rate = sf.read(str(path), always_2d=False)
        return np.asarray(audio, dtype=np.float32), int(sample_rate), "soundfile", ""
    except Exception as soundfile_error:
        try:
            audio, sample_rate = librosa.load(str(path), sr=None, mono=False)
            return (
                np.asarray(audio, dtype=np.float32),
                int(sample_rate),
                "librosa",
                f"soundfile: {soundfile_error}",
            )
        except Exception as librosa_error:
            raise RuntimeError(
                f"soundfile: {soundfile_error}; librosa: {librosa_error}"
            ) from librosa_error


def to_mono(audio: np.ndarray) -> tuple[np.ndarray, int]:
    if audio.ndim == 1:
        return audio, 1
    if audio.ndim != 2:
        raise ValueError(f"unsupported audio dimensions: {audio.shape}")
    # soundfile uses samples x channels; librosa uses channels x samples.
    if audio.shape[0] > audio.shape[1]:
        return np.mean(audio, axis=1), audio.shape[1]
    return np.mean(audio, axis=0), audio.shape[0]


def clean_audio() -> None:
    if not AUDIO_ROOT.exists():
        raise FileNotFoundError(f"Raw audio folder does not exist: {AUDIO_ROOT}")
    CLEAN_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        path for path in AUDIO_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    records = []

    print("=" * 70)
    print("AUDIO CLEANING AND VALIDATION")
    print("=" * 70)
    print(f"Raw audio files found: {len(audio_files)}")

    for index, audio_path in enumerate(audio_files, start=1):
        relative_path = audio_path.relative_to(AUDIO_ROOT)
        class_name = relative_path.parts[0] if relative_path.parts else "UNKNOWN"
        output_path = CLEAN_ROOT / relative_path.with_suffix(".wav")
        record = {
            "file_path": relative_path.as_posix(),
            "class": class_name,
            "original_format": audio_path.suffix.lower(),
            "decoder": "",
            "decode_error": "",
            "duration_seconds": np.nan,
            "original_sample_rate": np.nan,
            "output_sample_rate": TARGET_SR,
            "channels": np.nan,
            "rms": np.nan,
            "peak": np.nan,
            "status": "INVALID",
            "reason": "",
            "cleaned_file": "",
        }
        print(f"[{index}/{len(audio_files)}] {relative_path.as_posix()}")

        try:
            audio, original_sr, decoder, decode_error = decode_audio(audio_path)
            record["decoder"] = decoder
            record["decode_error"] = decode_error
            record["original_sample_rate"] = original_sr
            audio, channels = to_mono(audio)
            record["channels"] = channels

            if not np.isfinite(audio).all():
                record["reason"] = "NaN_or_Inf_values"
            elif original_sr <= 0 or len(audio) == 0:
                record["reason"] = "Empty_or_invalid_sample_rate"
            else:
                if original_sr != TARGET_SR:
                    audio = librosa.resample(
                        audio, orig_sr=original_sr, target_sr=TARGET_SR
                    )
                duration = len(audio) / TARGET_SR
                rms = float(np.sqrt(np.mean(np.square(audio))))
                peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
                record["duration_seconds"] = duration
                record["rms"] = rms
                record["peak"] = peak

                if duration < MIN_DURATION:
                    record["status"] = "INVALID"
                    record["reason"] = "Too_short"
                elif duration > MAX_DURATION:
                    record["status"] = "REVIEW"
                    record["reason"] = "Too_long"
                elif rms < SILENCE_RMS_THRESHOLD:
                    record["status"] = "REVIEW"
                    record["reason"] = "Near_silent"
                else:
                    record["status"] = "VALID"
                    record["reason"] = "Passed_all_checks"

                if record["status"] in {"VALID", "REVIEW"}:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(output_path), audio, TARGET_SR, subtype="PCM_16")
                    record["cleaned_file"] = output_path.relative_to(CLEAN_ROOT).as_posix()
        except Exception as error:
            record["reason"] = "Decoding_or_processing_error"
            record["decode_error"] = str(error)

        records.append(record)

    dataframe = pd.DataFrame(records)
    dataframe.to_csv(REPORT_PATH, index=False, encoding="utf-8")
    print("\n" + "=" * 70)
    print("AUDIO CLEANING SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(dataframe)}")
    print("\nStatus:")
    print(dataframe["status"].value_counts(dropna=False).to_string())
    print("\nReasons:")
    print(dataframe["reason"].value_counts(dropna=False).to_string())
    print("\nClass distribution:")
    print(dataframe["class"].value_counts().sort_index().to_string())
    valid_durations = dataframe.loc[
        dataframe["status"] == "VALID", "duration_seconds"
    ].dropna()
    if not valid_durations.empty:
        print("\nVALID AUDIO DURATION STATISTICS")
        print(f"Minimum: {valid_durations.min():.2f} seconds")
        print(f"Maximum: {valid_durations.max():.2f} seconds")
        print(f"Mean: {valid_durations.mean():.2f} seconds")
        print(f"Median: {valid_durations.median():.2f} seconds")
    print(f"\nCleaning report saved to: {REPORT_PATH}")
    print(f"Cleaned audio saved to: {CLEAN_ROOT}")
    print("Raw audio was NOT modified.")


if __name__ == "__main__":
    clean_audio()