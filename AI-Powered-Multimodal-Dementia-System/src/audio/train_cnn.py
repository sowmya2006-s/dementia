from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT / "data" / "metadata"
MODEL_DIR = ROOT / "models" / "audio"
RESULTS_DIR = ROOT / "results" / "audio"
SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MAX_EPOCHS = 30
PATIENCE = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MFCCDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        feature = np.load(row["feature_path"]).astype(np.float32)
        return torch.from_numpy(feature[None, ...]), torch.tensor(float(row["true_label"]))


class AudioCNN(nn.Module):
    def __init__(self, dropout: float = 0.30):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs)).squeeze(1)


def load_and_validate() -> dict[str, pd.DataFrame]:
    frames = {}
    for split in ("train", "validation", "test"):
        path = METADATA / f"audio_mfcc_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing manifest: {path}")
        frame = pd.read_csv(path)
        required = {"split", "person_id", "label", "source_file", "recording_name", "chunk_index", "feature_path", "mfcc_height", "mfcc_width"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        if not (frame["split"] == split).all():
            raise ValueError(f"{path} contains unexpected split values")
        if not ((frame["mfcc_height"] == 40) & (frame["mfcc_width"] == 313)).all():
            raise ValueError(f"{path} contains non-40x313 MFCC metadata")
        frame["feature_path"] = frame["feature_path"].map(lambda value: str(Path(value)))
        for feature_path in frame["feature_path"]:
            if not Path(feature_path).exists():
                raise FileNotFoundError(f"Missing MFCC feature: {feature_path}")
        frame["true_label"] = (frame["label"] == "dementia").astype(int)
        for feature_path in frame["feature_path"].head(3):
            feature = np.load(feature_path)
            if feature.shape != (40, 313):
                raise ValueError(f"Feature {feature_path} has shape {feature.shape}")
            if not np.isfinite(feature).all():
                raise ValueError(f"Feature {feature_path} contains NaN or Inf")
        frames[split] = frame
    people = {split: set(frame["person_id"]) for split, frame in frames.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = people[left] & people[right]
        if overlap:
            raise ValueError(f"Person leakage between {left} and {right}: {sorted(overlap)}")
    return frames


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        if training:
            optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        losses += loss.item() * len(labels)
        correct += ((torch.sigmoid(logits) >= 0.5).float() == labels).sum().item()
        total += len(labels)
    return losses / total, correct / total


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    result = {
        "accuracy": accuracy_score(frame["true_label"], frame["prediction"]),
        "precision": precision_score(frame["true_label"], frame["prediction"], zero_division=0),
        "recall": recall_score(frame["true_label"], frame["prediction"], zero_division=0),
        "f1": f1_score(frame["true_label"], frame["prediction"], zero_division=0),
    }
    result["roc_auc"] = roc_auc_score(frame["true_label"], frame["probability"]) if frame["true_label"].nunique() == 2 else float("nan")
    return {key: float(value) for key, value in result.items()}


def aggregate_predictions(chunks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    recording = chunks.groupby(["person_id", "recording_name", "source_file"], as_index=False).agg(
        true_label=("true_label", "first"), probability=("probability", "mean"), number_of_chunks=("chunk_index", "count")
    )
    recording["prediction"] = (recording["probability"] >= 0.5).astype(int)
    person = recording.groupby("person_id", as_index=False).agg(
        true_label=("true_label", "first"), probability=("probability", "mean"), number_of_recordings=("recording_name", "nunique")
    )
    person["prediction"] = (person["probability"] >= 0.5).astype(int)
    return recording, person


def main() -> None:
    seed_everything()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_and_validate()
    model = AudioCNN().to(DEVICE)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    summary = f"AudioCNN\nInput shape: 1 x 40 x 313\nDevice: {DEVICE}\nPyTorch: {torch.__version__}\nTotal parameters: {total_params}\nTrainable parameters: {trainable_params}\n\n{model}\n"
    (RESULTS_DIR / "model_summary.txt").write_text(summary, encoding="utf-8")
    config = {"model": "AudioCNN", "input_type": "MFCC", "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "optimizer": "Adam", "loss": "BCEWithLogitsLoss", "dropout": 0.30, "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE, "seed": SEED, "device": str(DEVICE)}
    (MODEL_DIR / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    mlflow_run = None
    try:
        import mlflow

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Audio_Dementia_Classification")
        mlflow_run = mlflow.start_run(run_name="CNN_MFCC_seed42")
        mlflow.log_params({
            **config,
            "sample_rate": 16000,
            "n_mfcc": 40,
            "n_fft": 1024,
            "hop_length": 512,
            "win_length": 1024,
            "n_mels": 40,
            "fmin": 20,
            "fmax": 8000,
            "chunk_seconds": 10,
            "chunk_hop_seconds": 5,
        })
        print(f"MLflow run: {mlflow_run.info.run_id}")
    except Exception as error:
        print(f"WARNING: MLflow run unavailable: {error}")
        mlflow = None
    loaders = {split: DataLoader(MFCCDataset(frame), batch_size=BATCH_SIZE, shuffle=split == "train", num_workers=0) for split, frame in frames.items()}
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history, best_loss, best_epoch, wait = [], float("inf"), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(model, loaders["train"], loss_fn, optimizer)
        validation_loss, validation_accuracy = run_epoch(model, loaders["validation"], loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy, "validation_loss": validation_loss, "validation_accuracy": validation_accuracy})
        if mlflow_run is not None:
            mlflow.log_metrics({"train_loss": train_loss, "train_accuracy": train_accuracy, "validation_loss": validation_loss, "validation_accuracy": validation_accuracy}, step=epoch)
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f}")
        if validation_loss < best_loss:
            best_loss, best_epoch, wait = validation_loss, epoch, 0
            torch.save(model.state_dict(), MODEL_DIR / "best_audio_cnn.pth")
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(RESULTS_DIR / "training_history.csv", index=False)
    for metric_name, ylabel in (("loss", "Loss"), ("accuracy", "Accuracy")):
        plt.figure()
        plt.plot(history_frame["epoch"], history_frame[f"train_{metric_name}"], label="train")
        plt.plot(history_frame["epoch"], history_frame[f"validation_{metric_name}"], label="validation")
        plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.legend(); plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"training_{metric_name}.png"); plt.close()
    model.load_state_dict(torch.load(MODEL_DIR / "best_audio_cnn.pth", map_location=DEVICE, weights_only=True))
    model.eval()
    predictions = []
    with torch.no_grad():
        for index, row in frames["test"].iterrows():
            feature = torch.from_numpy(np.load(row["feature_path"]).astype(np.float32)[None, None, ...]).to(DEVICE)
            probability = float(torch.sigmoid(model(feature)).item())
            predictions.append({**row.to_dict(), "probability": probability, "prediction": int(probability >= 0.5)})
    chunks = pd.DataFrame(predictions)
    recordings, people = aggregate_predictions(chunks)
    chunks[["split", "person_id", "label", "source_file", "recording_name", "chunk_index", "true_label", "probability", "prediction"]].to_csv(RESULTS_DIR / "chunk_level_predictions.csv", index=False)
    recordings.rename(columns={"probability": "mean_probability"}).to_csv(RESULTS_DIR / "recording_level_predictions.csv", index=False)
    people.rename(columns={"probability": "mean_probability"}).to_csv(RESULTS_DIR / "person_level_predictions.csv", index=False)
    all_metrics = {"chunk": metrics(chunks), "recording": metrics(recordings.rename(columns={"mean_probability": "probability"})), "person": metrics(people.rename(columns={"mean_probability": "probability"}))}
    (RESULTS_DIR / "test_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    matrix = confusion_matrix(people["true_label"], people["prediction"], labels=[0, 1])
    plt.figure(); plt.imshow(matrix, cmap="Blues"); plt.title("Person-level confusion matrix"); plt.colorbar(); plt.xticks([0, 1], ["nodementia", "dementia"]); plt.yticks([0, 1], ["nodementia", "dementia"])
    for row_index in range(2):
        for col_index in range(2): plt.text(col_index, row_index, matrix[row_index, col_index], ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(RESULTS_DIR / "confusion_matrix.png"); plt.close()
    fpr, tpr, _ = roc_curve(people["true_label"], people["probability"])
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={all_metrics['person']['roc_auc']:.4f}"); plt.plot([0, 1], [0, 1], "--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.tight_layout(); plt.savefig(RESULTS_DIR / "roc_curve.png"); plt.close()
    (RESULTS_DIR / "classification_report.txt").write_text(classification_report(people["true_label"], people["prediction"], target_names=["nodementia", "dementia"], zero_division=0), encoding="utf-8")
    if mlflow_run is not None:
        final_metrics = {
            f"{level}_{name}": value
            for level, values in all_metrics.items()
            for name, value in values.items()
            if np.isfinite(value)
        }
        mlflow.log_metrics(final_metrics)
        for artifact in (
            MODEL_DIR / "best_audio_cnn.pth",
            MODEL_DIR / "training_config.json",
            RESULTS_DIR / "training_history.csv",
            RESULTS_DIR / "training_loss.png",
            RESULTS_DIR / "training_accuracy.png",
            RESULTS_DIR / "confusion_matrix.png",
            RESULTS_DIR / "roc_curve.png",
            RESULTS_DIR / "classification_report.txt",
            RESULTS_DIR / "test_metrics.json",
            RESULTS_DIR / "chunk_level_predictions.csv",
            RESULTS_DIR / "recording_level_predictions.csv",
            RESULTS_DIR / "person_level_predictions.csv",
        ):
            mlflow.log_artifact(str(artifact))
        mlflow.end_run()
    print("AUDIO CNN EXPERIMENT COMPLETE")
    print(f"Device: {DEVICE}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Person metrics: {all_metrics['person']}")
    print(f"Recording metrics: {all_metrics['recording']}")
    print(f"Chunk metrics: {all_metrics['chunk']}")
    print(f"Artifacts saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()