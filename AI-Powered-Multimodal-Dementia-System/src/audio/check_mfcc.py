from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "audio_mfcc_manifest.csv"
)


df = pd.read_csv(MANIFEST)

print("=" * 60)
print("MFCC QUALITY CHECK")
print("=" * 60)


# ------------------------------------------------------------
# Basic information
# ------------------------------------------------------------

print("\nTotal MFCC chunks:")
print(len(df))


print("\nSplit counts:")
print(df["split"].value_counts())


print("\nClass counts:")
print(df["label"].value_counts())


# ------------------------------------------------------------
# Check missing paths
# ------------------------------------------------------------

missing = 0

for path in df["feature_path"]:

    if not (PROJECT_ROOT / path).exists() and not Path(path).exists():
        missing += 1

print("\nMissing feature files:")
print(missing)


# ------------------------------------------------------------
# Load random features
# ------------------------------------------------------------

print("\nChecking first 5 features...")

for i in range(min(5, len(df))):

    path = PROJECT_ROOT / df.iloc[i]["feature_path"]

    feature = np.load(path)

    print(
        f"{path.name} -> "
        f"shape={feature.shape}, "
        f"min={feature.min():.4f}, "
        f"max={feature.max():.4f}"
    )

    if np.isnan(feature).any():
        print("WARNING: NaN detected")

    if np.isinf(feature).any():
        print("WARNING: Inf detected")


# ------------------------------------------------------------
# Person leakage
# ------------------------------------------------------------

train_people = set(
    df[df["split"] == "train"]["person_id"]
)

val_people = set(
    df[df["split"] == "validation"]["person_id"]
)

test_people = set(
    df[df["split"] == "test"]["person_id"]
)


print("\nPerson leakage:")

print(
    "Train ∩ Validation:",
    len(train_people & val_people)
)

print(
    "Train ∩ Test:",
    len(train_people & test_people)
)

print(
    "Validation ∩ Test:",
    len(val_people & test_people)
)


print("\nMFCC CHECK COMPLETE")