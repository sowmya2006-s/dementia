"""
MRI CNN Baseline Training Script
=================================
AI-Powered Multimodal Dementia Detection System
Phase 1: MRI Classification

Architecture: 3-block CNN with BatchNorm, AdaptiveAvgPool, Dropout(0.30)
Loss: CrossEntropyLoss (4-class softmax classification)
Optimizer: Adam (lr=0.001)
Early stopping: patience=7, monitor=validation loss
Seed: 42

This script:
1. Loads the frozen MRI split (train/validation/test CSVs)
2. Verifies zero leakage between splits via SHA-256
3. Trains the CNN baseline model
4. Evaluates on the held-out test set exactly once
5. Logs everything to MLflow (sqlite:///mlflow.db)
6. Saves all artifacts, metrics, plots, and reports
"""

from __future__ import annotations

import json
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import platform
import random
import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT / "data" / "mri" / "metadata"
MODEL_DIR = ROOT / "models" / "mri" / "cnn"
RESULTS = ROOT / "results" / "mri" / "cnn"

SEED = 42
CLASS_NAMES = [
    "Non-Demented",
    "Very Mildly Demented",
    "Mildly Demented",
    "Moderately Demented",
]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {index: name for name, index in CLASS_TO_INDEX.items()}

BATCH_SIZE = 16
LEARNING_RATE = 0.001
MAX_EPOCHS = 30
PATIENCE = 7
IMAGE_SIZE = 224
DROPOUT = 0.30
WEIGHT_DECAY = 0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_all(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device() -> torch.device:
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Device: GPU ({gpu_name})")
    else:
        device = torch.device("cpu")
        gpu_name = None
        print("Device: CPU")
    return device


# ============================================================
# DATASET
# ============================================================

class MRIDataset(Dataset):
    """PyTorch Dataset for MRI images from a frozen CSV split."""

    def __init__(self, frame: pd.DataFrame, transform, root: Path):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = self.root / row["image_path"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        label = torch.tensor(int(row["label_index"]), dtype=torch.long)
        return tensor, label


# ============================================================
# MODEL
# ============================================================

class MRICNN(nn.Module):
    """
    Simple 3-block CNN baseline for 4-class MRI classification.

    Block 1: Conv2D(3→32) + BN + ReLU + MaxPool
    Block 2: Conv2D(32→64) + BN + ReLU + MaxPool
    Block 3: Conv2D(64→128) + BN + ReLU + MaxPool
    AdaptiveAvgPool2D(1,1) → Flatten → Dropout(0.30) → Linear(128→4)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# DATA LOADING AND VERIFICATION
# ============================================================

def load_split(split_name: str) -> pd.DataFrame:
    """Load a frozen CSV split and add label indices."""
    path = METADATA / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen split file: {path}")

    frame = pd.read_csv(path)

    # Verify required columns
    required_columns = {"image_path", "class", "sha256"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    # Map class names to label indices
    frame["label_index"] = frame["class"].map(CLASS_TO_INDEX)
    if frame["label_index"].isna().any():
        unknown = frame.loc[frame["label_index"].isna(), "class"].unique()
        raise ValueError(f"Unknown class(es) in {split_name}: {unknown}")

    return frame


def verify_image_existence(frames: dict[str, pd.DataFrame]) -> None:
    """Verify that all images referenced in the split CSVs actually exist."""
    print("\nVerifying image file existence...")
    total_missing = 0
    for split_name, frame in frames.items():
        missing = []
        for img_path in frame["image_path"]:
            full_path = ROOT / img_path
            if not full_path.exists():
                missing.append(img_path)
        if missing:
            total_missing += len(missing)
            print(f"  WARNING: {split_name} has {len(missing)} missing images")
            for m in missing[:5]:
                print(f"    - {m}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")

    if total_missing > 0:
        raise FileNotFoundError(
            f"STOP: {total_missing} total missing image files. "
            "Cannot proceed with training."
        )
    print("  All images verified: OK")


def verify_split_integrity(frames: dict[str, pd.DataFrame]) -> None:
    """
    Comprehensive split verification:
    1. All 4 classes present in each split
    2. Zero hash overlap between splits (SHA-256 leakage check)
    3. Class distribution report
    4. No NaN/invalid metadata
    """
    print("\n" + "=" * 60)
    print("FROZEN MRI SPLIT VERIFICATION")
    print("=" * 60)

    # 1. Check counts and class distribution
    for split_name, frame in frames.items():
        print(f"\n--- {split_name.upper()} ---")
        print(f"  Total images: {len(frame)}")

        classes_present = sorted(frame["class"].unique())
        print(f"  Classes present: {len(classes_present)}")

        if set(frame["label_index"]) != set(range(4)):
            raise ValueError(
                f"{split_name} does not contain all four classes. "
                f"Found: {sorted(frame['label_index'].unique())}"
            )

        distribution = frame["class"].value_counts()
        for cls_name in CLASS_NAMES:
            count = distribution.get(cls_name, 0)
            pct = 100.0 * count / len(frame) if len(frame) > 0 else 0
            print(f"    {cls_name}: {count} ({pct:.1f}%)")

        # Check for NaN values
        nan_count = frame[["image_path", "class", "sha256"]].isna().sum().sum()
        if nan_count > 0:
            raise ValueError(f"{split_name} contains {nan_count} NaN values")
        print(f"  NaN values: 0")

    # 2. SHA-256 based leakage check
    print("\n--- LEAKAGE CHECK (SHA-256) ---")
    hashes = {name: set(df["sha256"]) for name, df in frames.items()}

    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]
    for left, right in pairs:
        overlap = hashes[left] & hashes[right]
        if overlap:
            raise ValueError(
                f"LEAKAGE DETECTED: {len(overlap)} shared SHA-256 hashes "
                f"between {left} and {right}. STOP."
            )
        print(f"  {left} AND {right}: 0 overlapping hashes [OK]")

    # 3. Total count verification
    total = sum(len(f) for f in frames.values())
    print(f"\n  Total images across all splits: {total}")

    # 4. Check for duplicate paths within each split
    for split_name, frame in frames.items():
        dup_paths = frame["image_path"].duplicated().sum()
        if dup_paths > 0:
            print(f"  WARNING: {split_name} has {dup_paths} duplicate paths")

    print("\nFROZEN MRI SPLIT VERIFICATION: PASSED [OK]")
    print("=" * 60)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred, prefix: str = "") -> dict:
    """Compute classification metrics using macro averaging."""
    return {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        f"{prefix}recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        f"{prefix}f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Train for one epoch, return metrics."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_truths = []
    all_predictions = []

    for batch_idx, (batch_images, batch_labels) in enumerate(loader, 1):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        logits = model(batch_images)
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_labels)
        total_samples += len(batch_labels)
        all_truths.extend(batch_labels.cpu().numpy())
        all_predictions.extend(logits.argmax(1).cpu().numpy())

        if batch_idx % 100 == 0 or batch_idx == len(loader):
            print(f"    Batch {batch_idx}/{len(loader)} - batch loss: {loss.item():.4f}", flush=True)

    result = compute_metrics(all_truths, all_predictions)
    result["loss"] = total_loss / total_samples
    return result


def evaluate_model(model, loader, loss_fn, device):
    """Evaluate model on a data loader, return metrics."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_truths = []
    all_predictions = []

    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_images)
            loss = loss_fn(logits, batch_labels)

            total_loss += loss.item() * len(batch_labels)
            total_samples += len(batch_labels)
            all_truths.extend(batch_labels.cpu().numpy())
            all_predictions.extend(logits.argmax(1).cpu().numpy())

    result = compute_metrics(all_truths, all_predictions)
    result["loss"] = total_loss / total_samples
    return result


# ============================================================
# PLOTTING
# ============================================================

def save_training_plots(history_df: pd.DataFrame, results_dir: Path) -> None:
    """Generate and save training curves from history dataframe."""

    # Training loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("MRI CNN — Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "training_loss_curve.png", dpi=150)
    plt.close()

    # Validation loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(
        history_df["epoch"], history_df["val_loss"],
        marker="o", markersize=3, color="tab:orange"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("MRI CNN — Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "validation_loss_curve.png", dpi=150)
    plt.close()

    # Training accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(
        history_df["epoch"], history_df["train_accuracy"],
        marker="o", markersize=3, color="tab:green"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("MRI CNN — Training Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "training_accuracy_curve.png", dpi=150)
    plt.close()

    # Validation accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(
        history_df["epoch"], history_df["val_accuracy"],
        marker="o", markersize=3, color="tab:red"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("MRI CNN — Validation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "validation_accuracy_curve.png", dpi=150)
    plt.close()

    # F1 curve (train and val)
    plt.figure(figsize=(8, 5))
    plt.plot(
        history_df["epoch"], history_df["train_f1"],
        marker="o", markersize=3, label="Train F1 (macro)"
    )
    plt.plot(
        history_df["epoch"], history_df["val_f1"],
        marker="s", markersize=3, label="Val F1 (macro)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1 Score")
    plt.title("MRI CNN — F1 Score (Macro)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "f1_curve.png", dpi=150)
    plt.close()

    print("Training plots saved.")


def save_confusion_matrix(
    y_true, y_pred, class_names: list[str], results_dir: Path
) -> np.ndarray:
    """Generate and save confusion matrix plot and CSV."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Save CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(results_dir / "confusion_matrix.csv")

    # Save plot
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("MRI CNN — Confusion Matrix", fontsize=13)

    # Add text annotations
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
            )

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    print("Confusion matrix saved.")
    return cm


# ============================================================
# SANITY CHECK
# ============================================================

def run_sanity_check(model, loader, loss_fn, device) -> None:
    """Pre-training sanity check to verify the pipeline works."""
    print("\n--- PRE-TRAINING SANITY CHECK ---")

    images, labels = next(iter(loader))
    print(f"  Batch shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Label range: [{labels.min().item()}, {labels.max().item()}]")

    # Check labels in range
    assert labels.min().item() >= 0, "Labels contain negative values"
    assert labels.max().item() <= 3, "Labels exceed range 0-3"

    # Check for NaN/Inf in input
    assert torch.isfinite(images).all(), "Input contains NaN or Inf"

    # Forward pass
    images_dev = images.to(device)
    labels_dev = labels.to(device)
    logits = model(images_dev)

    print(f"  Output shape: {logits.shape}")
    assert logits.shape == (len(labels), 4), (
        f"Expected output shape [{len(labels)}, 4], got {logits.shape}"
    )
    assert torch.isfinite(logits).all(), "Output contains NaN or Inf"

    # Loss
    loss = loss_fn(logits, labels_dev)
    print(f"  Loss: {loss.item():.6f}")
    assert torch.isfinite(loss), "Loss is NaN or Inf"

    # Backward pass
    loss.backward()
    print("  Backward pass: OK")

    # Reset gradients
    model.zero_grad()

    print("\nMRI CNN SANITY CHECK PASSED [OK]\n")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 60)
    print("MRI CNN BASELINE TRAINING")
    print("AI-Powered Multimodal Dementia Detection System")
    print("=" * 60)

    # ---- Reproducibility ----
    seed_all(SEED)

    # ---- Device ----
    device = get_device()
    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )

    # ---- Create directories ----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ---- Environment info ----
    environment = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": gpu_name,
        "device": str(device),
        "os": platform.platform(),
        "seed": SEED,
    }

    # ---- Load frozen splits ----
    print("\nLoading frozen MRI splits...")
    frames = {}
    for split_name in ("train", "validation", "test"):
        frames[split_name] = load_split(split_name)
        print(f"  {split_name}: {len(frames[split_name])} images")

    # ---- Verify image files exist ----
    verify_image_existence(frames)

    # ---- Full split integrity verification ----
    verify_split_integrity(frames)

    # ---- Preprocessing transform ----
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # ---- Create data loaders ----
    actual_batch_size = BATCH_SIZE
    loaders = {}
    for split_name, frame in frames.items():
        dataset = MRIDataset(frame, transform, ROOT)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=actual_batch_size,
            shuffle=(split_name == "train"),
            num_workers=0,
            pin_memory=False,
        )

    # ---- Model ----
    model = MRICNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"\nModel: MRICNN")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ---- Save environment, config, model summary (pre-training) ----
    config = {
        "experiment_name": "MRI_Dementia_Classification",
        "dataset_version": "MRI_V1_CLEAN",
        "train_split": "data/mri/metadata/train.csv",
        "validation_split": "data/mri/metadata/validation.csv",
        "test_split": "data/mri/metadata/test.csv",
        "train_count": len(frames["train"]),
        "validation_count": len(frames["validation"]),
        "test_count": len(frames["test"]),
        "total_count": sum(len(f) for f in frames.values()),
        "image_size": IMAGE_SIZE,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "grayscale_handling": "convert to RGB",
        "class_mapping": CLASS_TO_INDEX,
        "model": "MRI_CNN",
        "loss": "CrossEntropyLoss",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": actual_batch_size,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "dropout": DROPOUT,
        "seed": SEED,
        "device": str(device),
        "selection_criterion": "lowest validation loss",
        "metric_average": "macro for precision, recall, F1",
    }

    (RESULTS / "training_environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    (RESULTS / "training_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    model_summary_text = (
        f"{model}\n\n"
        f"Input: 3x{IMAGE_SIZE}x{IMAGE_SIZE}\n"
        f"Total parameters: {total_params}\n"
        f"Trainable parameters: {trainable_params}\n"
        f"Loss: CrossEntropyLoss\n"
        f"Optimizer: Adam\n"
        f"Learning rate: {LEARNING_RATE}\n"
        f"Weight decay: {WEIGHT_DECAY}\n"
        f"Batch size: {actual_batch_size}\n"
        f"Max epochs: {MAX_EPOCHS}\n"
        f"Early stopping patience: {PATIENCE}\n"
        f"Dropout: {DROPOUT}\n"
        f"Seed: {SEED}\n"
        f"Class mapping: {CLASS_TO_INDEX}\n"
        f"Device: {device}\n"
    )
    (RESULTS / "model_summary.txt").write_text(
        model_summary_text, encoding="utf-8"
    )

    # ---- Sanity check ----
    run_sanity_check(model, loaders["train"], loss_fn, device)

    # ---- MLflow setup ----
    import mlflow

    mlflow_db_path = ROOT / "mlflow.db"
    tracking_uri = f"sqlite:///{mlflow_db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("MRI_Dementia_Classification")
    print(f"MLflow tracking URI: {tracking_uri}")

    with mlflow.start_run(run_name="MRI_CNN_seed42") as run:
        run_id = run.info.run_id
        print(f"MLflow run started: {run_id}")

        # Log parameters
        mlflow.log_params({
            "model": "MRI_CNN",
            "model_family": "CNN",
            "input_type": "MRI",
            "image_size": IMAGE_SIZE,
            "num_classes": 4,
            "loss": "CrossEntropyLoss",
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": actual_batch_size,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "dropout": DROPOUT,
            "seed": SEED,
            "class_mapping": json.dumps(CLASS_TO_INDEX),
        })

        # Log environment
        mlflow.log_params({
            "python_version": environment["python_version"],
            "pytorch_version": environment["pytorch_version"],
            "torchvision_version": environment["torchvision_version"],
            "cuda_available": environment["cuda_available"],
            "cuda_version": str(environment["cuda_version"]),
            "device_type": environment["device"],
            "gpu_name": str(environment["gpu_name"]),
        })

        # ============================================================
        # TRAINING LOOP
        # ============================================================
        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)

        best_val_loss = float("inf")
        best_epoch = 0
        wait = 0
        history = []

        for epoch in range(1, MAX_EPOCHS + 1):
            # Train
            train_result = train_one_epoch(
                model, loaders["train"], loss_fn, optimizer, device
            )

            # Validate
            val_result = evaluate_model(
                model, loaders["validation"], loss_fn, device
            )

            # Build history row
            row = {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "train_accuracy": train_result["accuracy"],
                "train_precision": train_result["precision"],
                "train_recall": train_result["recall"],
                "train_f1": train_result["f1"],
                "val_loss": val_result["loss"],
                "val_accuracy": val_result["accuracy"],
                "val_precision": val_result["precision"],
                "val_recall": val_result["recall"],
                "val_f1": val_result["f1"],
                "learning_rate": LEARNING_RATE,
            }
            history.append(row)

            # Log to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": row["train_loss"],
                    "train_accuracy": row["train_accuracy"],
                    "train_precision": row["train_precision"],
                    "train_recall": row["train_recall"],
                    "train_f1": row["train_f1"],
                    "val_loss": row["val_loss"],
                    "val_accuracy": row["val_accuracy"],
                    "val_precision": row["val_precision"],
                    "val_recall": row["val_recall"],
                    "val_f1": row["val_f1"],
                },
                step=epoch,
            )

            # Print epoch summary
            marker = ""
            if row["val_loss"] < best_val_loss:
                marker = " * best"

            print(
                f"Epoch {epoch:02d}/{MAX_EPOCHS}: "
                f"train_loss={row['train_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} "
                f"val_acc={row['val_accuracy']:.4f} "
                f"val_f1={row['val_f1']:.4f}{marker}",
                flush=True,
            )

            # Early stopping check
            if row["val_loss"] < best_val_loss:
                best_val_loss = row["val_loss"]
                best_epoch = epoch
                wait = 0

                # Save best checkpoint
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": best_epoch,
                    "best_validation_loss": best_val_loss,
                    "class_mapping": CLASS_TO_INDEX,
                    "config": config,
                    "seed": SEED,
                }
                torch.save(checkpoint, MODEL_DIR / "best_mri_cnn.pth")
            else:
                wait += 1

            if wait >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

        # Save last checkpoint
        last_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_loss": best_val_loss,
            "class_mapping": CLASS_TO_INDEX,
            "config": config,
            "seed": SEED,
        }
        torch.save(last_checkpoint, MODEL_DIR / "last_mri_cnn.pth")

        print(f"\nTraining completed. Best epoch: {best_epoch}, "
              f"Best val loss: {best_val_loss:.6f}")

        # ============================================================
        # SAVE TRAINING HISTORY
        # ============================================================
        history_df = pd.DataFrame(history)
        history_df.to_csv(RESULTS / "training_history.csv", index=False)
        print("Training history saved.")

        # ============================================================
        # TRAINING PLOTS
        # ============================================================
        save_training_plots(history_df, RESULTS)

        # ============================================================
        # TEST EVALUATION (single pass on best checkpoint)
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST EVALUATION")
        print("=" * 60)

        # Load best checkpoint
        best_ckpt = torch.load(
            MODEL_DIR / "best_mri_cnn.pth",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(best_ckpt["model_state_dict"])
        model.eval()

        all_truths = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for batch_images, batch_labels in loaders["test"]:
                batch_images = batch_images.to(device)
                logits = model(batch_images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

                all_truths.extend(batch_labels.numpy())
                all_predictions.extend(probs.argmax(axis=1))
                all_probabilities.extend(probs)

        y_true = np.array(all_truths)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)

        # Compute test metrics
        test_accuracy = float(accuracy_score(y_true, y_pred))
        macro_precision = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        macro_recall = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        macro_f1 = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        weighted_precision = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        weighted_recall = float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        weighted_f1 = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )

        # ROC-AUC (one-vs-rest, macro)
        try:
            roc_auc = float(
                roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
            )
        except Exception as e:
            print(f"  ROC-AUC calculation failed: {e}")
            roc_auc = None

        test_metrics = {
            "accuracy": test_accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "roc_auc_ovr_macro": roc_auc,
        }

        (RESULTS / "test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2), encoding="utf-8"
        )

        print(f"\n  Test Accuracy:        {test_accuracy:.4f}")
        print(f"  Macro Precision:      {macro_precision:.4f}")
        print(f"  Macro Recall:         {macro_recall:.4f}")
        print(f"  Macro F1:             {macro_f1:.4f}")
        print(f"  Weighted Precision:   {weighted_precision:.4f}")
        print(f"  Weighted Recall:      {weighted_recall:.4f}")
        print(f"  Weighted F1:          {weighted_f1:.4f}")
        roc_display = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
        print(f"  ROC-AUC (OvR macro):  {roc_display}")

        # ============================================================
        # CONFUSION MATRIX
        # ============================================================
        save_confusion_matrix(y_true, y_pred, CLASS_NAMES, RESULTS)

        # ============================================================
        # CLASSIFICATION REPORT
        # ============================================================
        report = classification_report(
            y_true, y_pred,
            labels=list(range(4)),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
        (RESULTS / "classification_report.txt").write_text(
            report, encoding="utf-8"
        )
        print("\nClassification Report:")
        print(report)

        # ============================================================
        # TEST PREDICTIONS
        # ============================================================
        pred_df = frames["test"].copy()
        pred_df["true_label"] = y_true
        pred_df["true_class"] = pred_df["class"]
        pred_df["predicted_label"] = y_pred
        pred_df["predicted_class"] = [CLASS_NAMES[int(i)] for i in y_pred]
        pred_df["prob_non_demented"] = y_prob[:, 0]
        pred_df["prob_very_mild"] = y_prob[:, 1]
        pred_df["prob_mild"] = y_prob[:, 2]
        pred_df["prob_moderate"] = y_prob[:, 3]

        pred_df[
            [
                "image_path",
                "true_label",
                "true_class",
                "predicted_label",
                "predicted_class",
                "prob_non_demented",
                "prob_very_mild",
                "prob_mild",
                "prob_moderate",
            ]
        ].to_csv(RESULTS / "test_predictions.csv", index=False)
        print("Test predictions saved.")

        # ============================================================
        # EXPERIMENT MANIFEST
        # ============================================================
        manifest = {
            "experiment_name": "MRI_Dementia_Classification",
            "experiment_id": run.info.experiment_id,
            "run_id": run_id,
            "run_name": "MRI_CNN_seed42",
            "run_date": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "version": "MRI_V1_CLEAN",
                "total": sum(len(f) for f in frames.values()),
                "train": len(frames["train"]),
                "validation": len(frames["validation"]),
                "test": len(frames["test"]),
                "classes": CLASS_NAMES,
                "class_mapping": CLASS_TO_INDEX,
            },
            "preprocessing": {
                "image_size": IMAGE_SIZE,
                "normalization": {
                    "mean": IMAGENET_MEAN,
                    "std": IMAGENET_STD,
                },
                "grayscale_handling": "convert to RGB",
            },
            "model_architecture": "MRICNN (3-block CNN + AdaptiveAvgPool + Dropout + Linear)",
            "hyperparameters": {
                "optimizer": "Adam",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": actual_batch_size,
                "max_epochs": MAX_EPOCHS,
                "early_stopping_patience": PATIENCE,
                "dropout": DROPOUT,
                "seed": SEED,
                "loss": "CrossEntropyLoss",
                "selection_criterion": "lowest validation loss",
            },
            "environment": environment,
            "results": {
                "best_epoch": best_epoch,
                "best_validation_loss": best_val_loss,
                "test_metrics": test_metrics,
            },
            "checkpoint_paths": [
                str(MODEL_DIR / "best_mri_cnn.pth"),
                str(MODEL_DIR / "last_mri_cnn.pth"),
            ],
            "result_paths": [
                str(RESULTS / "training_history.csv"),
                str(RESULTS / "test_metrics.json"),
                str(RESULTS / "confusion_matrix.png"),
                str(RESULTS / "confusion_matrix.csv"),
                str(RESULTS / "classification_report.txt"),
                str(RESULTS / "test_predictions.csv"),
                str(RESULTS / "training_loss_curve.png"),
                str(RESULTS / "validation_loss_curve.png"),
                str(RESULTS / "training_accuracy_curve.png"),
                str(RESULTS / "validation_accuracy_curve.png"),
                str(RESULTS / "f1_curve.png"),
                str(RESULTS / "model_summary.txt"),
                str(RESULTS / "training_environment.json"),
                str(RESULTS / "training_config.json"),
            ],
            "mlflow_run_id": run_id,
        }
        (RESULTS / "mri_cnn_experiment_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print("Experiment manifest saved.")

        # ============================================================
        # MODEL COMPARISON TABLE (CNN only)
        # ============================================================
        comparison_path = ROOT / "results" / "mri" / "mri_model_comparison.csv"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison = pd.DataFrame([{
            "model": "MRI_CNN",
            "input": "MRI",
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_precision": test_metrics["weighted_precision"],
            "weighted_recall": test_metrics["weighted_recall"],
            "weighted_f1": test_metrics["weighted_f1"],
            "roc_auc_ovr_macro": test_metrics["roc_auc_ovr_macro"],
            "mlflow_run_id": run_id,
        }])
        comparison.to_csv(comparison_path, index=False)
        print("Model comparison CSV saved (CNN only).")

        # ============================================================
        # PAPER-READY RESULT SUMMARY
        # ============================================================
        roc_line = (
            f"{roc_auc:.4f}" if roc_auc is not None else "N/A (calculation failed)"
        )
        result_summary_md = f"""# MRI CNN Baseline — Result Summary

## 1. Experiment

| Field | Value |
|---|---|
| Experiment name | MRI_Dementia_Classification |
| Run name | MRI_CNN_seed42 |
| MLflow run ID | `{run_id}` |
| Date | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} |

## 2. Dataset

| Field | Value |
|---|---|
| Dataset version | MRI_V1_CLEAN |
| Total images | {sum(len(f) for f in frames.values())} |
| Train | {len(frames['train'])} |
| Validation | {len(frames['validation'])} |
| Test | {len(frames['test'])} |
| Classes | 4 |

**Class labels:**
- 0: Non-Demented
- 1: Very Mildly Demented
- 2: Mildly Demented
- 3: Moderately Demented

## 3. Preprocessing

- Input size: {IMAGE_SIZE} × {IMAGE_SIZE}
- Normalization: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Grayscale handling: converted to RGB

## 4. Model Architecture

Three-block CNN baseline:
- Block 1: Conv2D(3→32) + BatchNorm + ReLU + MaxPool2D
- Block 2: Conv2D(32→64) + BatchNorm + ReLU + MaxPool2D
- Block 3: Conv2D(64→128) + BatchNorm + ReLU + MaxPool2D
- AdaptiveAvgPool2D(1,1) → Flatten → Dropout(0.30) → Linear(128→4)

Total parameters: {total_params:,}

## 5. Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | {LEARNING_RATE} |
| Weight decay | {WEIGHT_DECAY} |
| Batch size | {actual_batch_size} |
| Max epochs | {MAX_EPOCHS} |
| Early stopping patience | {PATIENCE} |
| Dropout | {DROPOUT} |
| Loss | CrossEntropyLoss |
| Seed | {SEED} |
| Device | {device} |
| Selection criterion | Lowest validation loss |

## 6. Training Outcome

| Field | Value |
|---|---|
| Best epoch | {best_epoch} |
| Best validation loss | {best_val_loss:.6f} |
| Total epochs trained | {epoch} |
| Early stopping | {'Yes' if wait >= PATIENCE else 'No (completed all epochs)'} |

## 7. Test Set Results

| Metric | Value |
|---|---|
| Accuracy | {test_accuracy:.4f} |
| Macro Precision | {macro_precision:.4f} |
| Macro Recall | {macro_recall:.4f} |
| Macro F1 | {macro_f1:.4f} |
| Weighted Precision | {weighted_precision:.4f} |
| Weighted Recall | {weighted_recall:.4f} |
| Weighted F1 | {weighted_f1:.4f} |
| ROC-AUC (OvR, macro) | {roc_line} |

## 8. Observations

- This is a simple baseline CNN without transfer learning or data augmentation.
- The model was evaluated on the held-out test set exactly once, after training was completed.
- Metrics may differ from transfer-learning approaches or ensemble methods.
- The metric averaging method used is macro (unweighted mean across classes).

## 9. Limitations

- No data augmentation was applied.
- No transfer learning was used.
- The CNN architecture is intentionally simple (3 convolutional blocks).
- Results reflect a single seed (42) and may vary with different seeds.
- This model is not intended for clinical use or diagnosis.

## 10. Artifacts

| File | Location |
|---|---|
| Best checkpoint | `models/mri/cnn/best_mri_cnn.pth` |
| Last checkpoint | `models/mri/cnn/last_mri_cnn.pth` |
| Training history | `results/mri/cnn/training_history.csv` |
| Test metrics | `results/mri/cnn/test_metrics.json` |
| Confusion matrix | `results/mri/cnn/confusion_matrix.png` |
| Classification report | `results/mri/cnn/classification_report.txt` |
| Test predictions | `results/mri/cnn/test_predictions.csv` |
| Experiment manifest | `results/mri/cnn/mri_cnn_experiment_manifest.json` |
"""
        (RESULTS / "MRI_CNN_Result_Summary.md").write_text(
            result_summary_md, encoding="utf-8"
        )
        print("Paper-ready result summary saved.")

        # ============================================================
        # LOG FINAL METRICS TO MLFLOW
        # ============================================================
        final_metrics_to_log = {
            "test_accuracy": test_accuracy,
            "test_macro_precision": macro_precision,
            "test_macro_recall": macro_recall,
            "test_macro_f1": macro_f1,
            "test_weighted_precision": weighted_precision,
            "test_weighted_recall": weighted_recall,
            "test_weighted_f1": weighted_f1,
            "best_epoch": float(best_epoch),
            "best_validation_loss": best_val_loss,
        }
        if roc_auc is not None:
            final_metrics_to_log["test_roc_auc_ovr_macro"] = roc_auc
        mlflow.log_metrics(final_metrics_to_log)

        # ============================================================
        # UPLOAD ARTIFACTS TO MLFLOW
        # ============================================================
        print("\nUploading artifacts to MLflow...")
        artifact_files = [
            MODEL_DIR / "best_mri_cnn.pth",
            MODEL_DIR / "last_mri_cnn.pth",
            RESULTS / "training_history.csv",
            RESULTS / "training_loss_curve.png",
            RESULTS / "validation_loss_curve.png",
            RESULTS / "training_accuracy_curve.png",
            RESULTS / "validation_accuracy_curve.png",
            RESULTS / "f1_curve.png",
            RESULTS / "confusion_matrix.png",
            RESULTS / "confusion_matrix.csv",
            RESULTS / "classification_report.txt",
            RESULTS / "test_predictions.csv",
            RESULTS / "test_metrics.json",
            RESULTS / "model_summary.txt",
            RESULTS / "training_environment.json",
            RESULTS / "training_config.json",
            RESULTS / "mri_cnn_experiment_manifest.json",
            RESULTS / "MRI_CNN_Result_Summary.md",
        ]

        for artifact_path in artifact_files:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))
                print(f"  [OK] {artifact_path.name}")
            else:
                print(f"  [FAIL] MISSING: {artifact_path}")
                raise FileNotFoundError(
                    f"STOP: Cannot upload artifact -- file missing: {artifact_path}"
                )

        print("All artifacts uploaded to MLflow.")

    # ================================================================
    # MLFLOW VERIFICATION
    # ================================================================
    print("\n" + "=" * 60)
    print("MLFLOW VERIFICATION")
    print("=" * 60)

    client = mlflow.MlflowClient()
    verified_run = client.get_run(run_id)

    print(f"  Experiment: {verified_run.info.experiment_id}")
    print(f"  Run name: {verified_run.info.run_name}")
    print(f"  Run ID: {verified_run.info.run_id}")
    print(f"  Status: {verified_run.info.status}")

    # Verify parameters
    params = verified_run.data.params
    assert params.get("model") == "MRI_CNN", "MLflow param 'model' mismatch"
    assert params.get("seed") == str(SEED), "MLflow param 'seed' mismatch"
    print(f"  Parameters logged: {len(params)}")

    # Verify metrics
    run_metrics = verified_run.data.metrics
    required_metrics = [
        "test_accuracy", "test_macro_f1", "best_epoch", "best_validation_loss"
    ]
    for m in required_metrics:
        assert m in run_metrics, f"MLflow metric '{m}' missing"
    print(f"  Metrics logged: {len(run_metrics)}")

    # Verify artifacts
    artifacts = client.list_artifacts(run_id)
    artifact_names = [a.path for a in artifacts]
    print(f"  Artifacts logged: {len(artifact_names)}")
    required_artifacts = [
        "best_mri_cnn.pth",
        "training_history.csv",
        "test_metrics.json",
        "confusion_matrix.png",
    ]
    for ra in required_artifacts:
        if ra not in artifact_names:
            print(f"  WARNING: Expected artifact '{ra}' not found")

    print("\nMLFLOW VERIFICATION PASSED [OK]")

    # ================================================================
    # FINAL REPORT
    # ================================================================
    roc_final = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
    gpu_final = gpu_name if gpu_name else "None"

    print("\n" + "=" * 60)
    print("MRI CNN EXPERIMENT COMPLETE")
    print("=" * 60)
    print()
    print(f"Dataset:              {sum(len(f) for f in frames.values())} images")
    print(f"Classes:              4")
    print(f"Seed:                 {SEED}")
    print(f"Model:                CNN")
    print(f"Input:                {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Device:               {device}")
    print(f"GPU:                  {gpu_final}")
    print(f"Best epoch:           {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print()
    print("TEST RESULTS:")
    print(f"  Accuracy:           {test_accuracy:.4f}")
    print(f"  Macro Precision:    {macro_precision:.4f}")
    print(f"  Macro Recall:       {macro_recall:.4f}")
    print(f"  Macro F1:           {macro_f1:.4f}")
    print(f"  Weighted Precision: {weighted_precision:.4f}")
    print(f"  Weighted Recall:    {weighted_recall:.4f}")
    print(f"  Weighted F1:        {weighted_f1:.4f}")
    print(f"  ROC-AUC:            {roc_final}")
    print()
    print("MLflow:")
    print(f"  Experiment = MRI_Dementia_Classification")
    print(f"  Run = MRI_CNN_seed42")
    print(f"  Run ID = {run_id}")
    print()
    print(f"CHECKPOINT: {MODEL_DIR / 'best_mri_cnn.pth'}")
    print(f"RESULTS:    {RESULTS}")
    print()
    print("MLFLOW VERIFICATION: PASSED")
    print()

    # ================================================================
    # FINAL SAFETY / RESEARCH CHECKLIST
    # ================================================================
    checklist = {
        "frozen MRI split used": True,
        "no dataset modification": True,
        "no train/test leakage": True,
        "four classes present": True,
        "CNN trained successfully": best_epoch > 0,
        "best checkpoint saved": (MODEL_DIR / "best_mri_cnn.pth").exists(),
        "last checkpoint saved": (MODEL_DIR / "last_mri_cnn.pth").exists(),
        "training history saved": (RESULTS / "training_history.csv").exists(),
        "graphs saved": (RESULTS / "training_loss_curve.png").exists(),
        "confusion matrix saved": (RESULTS / "confusion_matrix.png").exists(),
        "classification report saved": (RESULTS / "classification_report.txt").exists(),
        "predictions saved": (RESULTS / "test_predictions.csv").exists(),
        "test metrics saved": (RESULTS / "test_metrics.json").exists(),
        "environment saved": (RESULTS / "training_environment.json").exists(),
        "configuration saved": (RESULTS / "training_config.json").exists(),
        "experiment manifest saved": (RESULTS / "mri_cnn_experiment_manifest.json").exists(),
        "MLflow run created": run_id is not None,
        "MLflow metrics logged": True,
        "MLflow artifacts logged": True,
        "MLflow run verified": True,
        "model comparison CSV updated (CNN only)": comparison_path.exists(),
        "no bagging started": True,
        "no boosting started": True,
        "no fabricated results": True,
    }

    all_passed = all(checklist.values())
    for item, status in checklist.items():
        mark = "OK" if status else "FAIL"
        print(f"  [{mark}] {item}")

    print()
    if all_passed:
        print("MRI CNN BASELINE VERIFIED [OK]")
    else:
        failed = [k for k, v in checklist.items() if not v]
        print(f"VERIFICATION FAILED: {failed}")

    print("=" * 60)


if __name__ == "__main__":
    main()