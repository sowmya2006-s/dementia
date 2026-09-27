from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import soundfile as sf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

OUTPUT_DIR = PROJECT_ROOT / "data" / "audio" / "features" / "mfcc"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# AUDIO PARAMETERS
# ============================================================

SAMPLE_RATE = 16000

N_MFCC = 40

N_FFT = 1024

HOP_LENGTH = 512

WIN_LENGTH = 1024

N_MELS = 40

FMIN = 20

FMAX = 8000

CHUNK_SECONDS = 10

CHUNK_HOP_SECONDS = 5

CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS

CHUNK_HOP_SAMPLES = SAMPLE_RATE * CHUNK_HOP_SECONDS

RANDOM_SEED = 42


# ============================================================
# SPLITS
# ============================================================

SPLITS = {
    "train": METADATA_DIR / "audio_train.csv",
    "validation": METADATA_DIR / "audio_validation.csv",
    "test": METADATA_DIR / "audio_test.csv"
}


# ============================================================
# MFCC FUNCTION
# ============================================================

def extract_mfcc(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )

    return mfcc.astype(np.float32)


# ============================================================
# CREATE AUDIO CHUNKS
# ============================================================

def create_chunks(audio):

    chunks = []

    audio_length = len(audio)

    # If recording is shorter than 10 seconds
    if audio_length <= CHUNK_SAMPLES:

        padded = np.pad(
            audio,
            (0, CHUNK_SAMPLES - audio_length)
        )

        chunks.append(
            (padded, True)
        )

        return chunks

    start = 0

    while start < audio_length:

        end = start + CHUNK_SAMPLES

        chunk = audio[start:end]

        padded = False

        if len(chunk) < CHUNK_SAMPLES:

            chunk = np.pad(
                chunk,
                (0, CHUNK_SAMPLES - len(chunk))
            )

            padded = True

        chunks.append(
            (chunk, padded)
        )

        start += CHUNK_HOP_SAMPLES

        # Stop when the next chunk would start
        # beyond the recording
        if start >= audio_length:
            break

    return chunks


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split_name, csv_path):

    print("\n" + "=" * 60)
    print(f"PROCESSING: {split_name.upper()}")
    print("=" * 60)

    df = pd.read_csv(csv_path)

    print(f"Recordings: {len(df)}")

    split_output = OUTPUT_DIR / split_name

    split_output.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []
    errors = []

    total_chunks = 0

    for index, row in df.iterrows():

        audio_path = PROJECT_ROOT / "data" / "audio" / "cleaned" / row["file_path"]

        label = row["class"]

        person_id = row["person_id"]

        recording_name = audio_path.stem

        try:

            # ------------------------------------------------
            # Load audio
            # ------------------------------------------------

            audio, sr = librosa.load(
                audio_path,
                sr=SAMPLE_RATE,
                mono=True
            )

            # ------------------------------------------------
            # Create chunks
            # ------------------------------------------------

            chunks = create_chunks(audio)

            for chunk_index, (chunk, padded) in enumerate(chunks):

                # --------------------------------------------
                # MFCC
                # --------------------------------------------

                mfcc = extract_mfcc(chunk)

                # --------------------------------------------
                # Save feature
                # --------------------------------------------

                person_folder = split_output / str(person_id)

                person_folder.mkdir(
                    parents=True,
                    exist_ok=True
                )

                feature_name = (
                    f"{recording_name}"
                    f"_chunk_{chunk_index + 1:03d}.npy"
                )

                feature_path = person_folder / feature_name

                np.save(
                    feature_path,
                    mfcc
                )

                # --------------------------------------------
                # Metadata
                # --------------------------------------------

                records.append({

                    "split": split_name,

                    "person_id": person_id,

                    "label": label,

                    "source_file": row["file_path"],

                    "recording_name": recording_name,

                    "chunk_index": chunk_index + 1,

                    "feature_path": str(feature_path),

                    "mfcc_height": mfcc.shape[0],

                    "mfcc_width": mfcc.shape[1],

                    "padded": padded

                })

                total_chunks += 1

        except Exception as e:

            print(
                f"ERROR: {audio_path}"
            )

            print(e)
            errors.append({"file_path": row["file_path"], "error": str(e)})

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    manifest = pd.DataFrame(records)

    manifest_path = (
        METADATA_DIR /
        f"audio_mfcc_{split_name}.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False
    )

    extracted_sources = set(manifest["source_file"]) if not manifest.empty else set()
    expected_sources = set(df["file_path"])
    missing_sources = sorted(expected_sources - extracted_sources)
    if missing_sources or errors:
        error_path = METADATA_DIR / f"audio_mfcc_{split_name}_errors.csv"
        pd.DataFrame(errors).to_csv(error_path, index=False)
        raise RuntimeError(
            f"MFCC extraction incomplete for {split_name}: "
            f"missing_recordings={missing_sources}; errors={errors}. "
            f"Details: {error_path}"
        )

    print(f"\nRecordings: {len(df)}")

    print(f"Chunks: {total_chunks}")

    print(
        f"Manifest saved: {manifest_path}"
    )

    return manifest


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MFCC FEATURE EXTRACTION")
    print("=" * 60)

    all_manifests = []

    for split_name, csv_path in SPLITS.items():

        manifest = process_split(
            split_name,
            csv_path
        )

        all_manifests.append(manifest)

    # --------------------------------------------------------
    # Combined manifest
    # --------------------------------------------------------

    combined = pd.concat(
        all_manifests,
        ignore_index=True
    )

    combined_path = (
        METADATA_DIR /
        "audio_mfcc_manifest.csv"
    )

    combined.to_csv(
        combined_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("MFCC EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        f"Total chunks: {len(combined)}"
    )

    print(
        f"Combined manifest: {combined_path}"
    )

    print("\nSplit distribution:")

    print(
        combined.groupby(
            ["split", "label"]
        ).size()
    )

    print("\nMFCC shape:")

    print(
        combined[
            ["mfcc_height", "mfcc_width"]
        ].drop_duplicates()
    )


if __name__ == "__main__":
    main()