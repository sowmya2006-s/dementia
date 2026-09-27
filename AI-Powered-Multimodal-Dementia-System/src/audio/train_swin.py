
from __future__ import annotations

import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT / "data" / "metadata"
MODEL_DIR = ROOT / "models" / "audio_swin"
RESULTS = ROOT / "results" / "audio_swin"
SEED = 42
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 20
PATIENCE = 5
CLIP = 1.0
THRESHOLD = 0.5
INPUT_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class MFCCDataset(Dataset):
    def __init__(self, frame): self.frame = frame.reset_index(drop=True)
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        x = np.load(row.feature_path).astype(np.float32)
        if x.shape != (40, 313) or not np.isfinite(x).all(): raise ValueError(f"Invalid feature: {row.feature_path}")
        x = (x - x.mean()) / (x.std() + 1e-6)
        x = torch.from_numpy(x[None, None, ...])
        x = torch.nn.functional.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False).squeeze(0).repeat(3, 1, 1)
        return x, torch.tensor(float(row.true_label))


def validate_data():
    frames = {}
    for split in ("train", "validation", "test"):
        path = METADATA / f"audio_mfcc_{split}.csv"
        if not path.exists(): raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {"split", "person_id", "label", "source_file", "recording_name", "chunk_index", "feature_path", "mfcc_height", "mfcc_width"}
        if required - set(frame.columns): raise ValueError(f"{path} missing {required - set(frame.columns)}")
        if not (frame.split == split).all(): raise ValueError(f"Unexpected split values in {path}")
        if not ((frame.mfcc_height == 40) & (frame.mfcc_width == 313)).all(): raise ValueError(f"Bad metadata shape in {path}")
        if frame.feature_path.map(lambda p: not Path(p).exists()).any(): raise FileNotFoundError(f"Missing feature in {path}")
        frame["true_label"] = (frame.label == "dementia").astype(int)
        for feature_path in frame.feature_path:
            feature = np.load(feature_path)
            if feature.shape != (40, 313): raise ValueError(f"Bad feature shape: {feature_path}")
            if not np.isfinite(feature).all(): raise ValueError(f"NaN/Inf feature: {feature_path}")
        frames[split] = frame
    people = {key: set(value.person_id) for key, value in frames.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if people[left] & people[right]: raise ValueError(f"Person leakage: {left}/{right}")
    return frames


def make_model():
    weights = torchvision.models.Swin_T_Weights.DEFAULT
    model = torchvision.models.swin_t(weights=weights)
    model.head = nn.Linear(model.head.in_features, 1)
    return model, True


def epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None; model.train(training); total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if training: optimizer.zero_grad()
        logits = model(x).squeeze(1); loss = loss_fn(logits, y)
        if training:
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); optimizer.step()
        total_loss += loss.item() * len(y); correct += ((torch.sigmoid(logits) >= THRESHOLD).float() == y).sum().item(); total += len(y)
    return total_loss / total, correct / total


def calc(frame):
    values = {"accuracy": accuracy_score(frame.true_label, frame.prediction), "precision": precision_score(frame.true_label, frame.prediction, zero_division=0), "recall": recall_score(frame.true_label, frame.prediction, zero_division=0), "f1": f1_score(frame.true_label, frame.prediction, zero_division=0)}
    values["roc_auc"] = roc_auc_score(frame.true_label, frame.probability) if frame.true_label.nunique() == 2 else float("nan")
    return {key: float(value) for key, value in values.items()}


def aggregate(chunks):
    recording = chunks.groupby(["person_id", "recording_name", "source_file"], as_index=False).agg(true_label=("true_label", "first"), probability=("probability", "mean"), number_of_chunks=("chunk_index", "count"))
    recording["prediction"] = (recording.probability >= THRESHOLD).astype(int)
    person = recording.groupby("person_id", as_index=False).agg(true_label=("true_label", "first"), probability=("probability", "mean"), number_of_recordings=("recording_name", "nunique"))
    person["prediction"] = (person.probability >= THRESHOLD).astype(int)
    return recording, person


def main():
    seed_all(); MODEL_DIR.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    frames = validate_data()
    model, pretrained = make_model(); model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters()); trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    environment = {"python": platform.python_version(), "pytorch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda, "device": str(DEVICE), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    config = {"model": "Swin-Tiny", "pretrained": pretrained, "seed": SEED, "batch_size": BATCH_SIZE, "learning_rate": LR, "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "gradient_clip": CLIP, "loss": "BCEWithLogitsLoss", "threshold": THRESHOLD, "mfcc_shape": "40x313", "swin_input_size": INPUT_SIZE, "normalization": "per-MFCC mean/std", "channel_conversion": "replicate single channel to RGB", "resize": "bilinear 40x313 to 224x224", "device": str(DEVICE)}
    (RESULTS / "training_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (RESULTS / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (RESULTS / "model_summary.txt").write_text(f"{model}\n\nInput: 3x224x224\nPretrained: {pretrained}\nDevice: {DEVICE}\nTotal parameters: {total_params}\nTrainable parameters: {trainable}\n", encoding="utf-8")
    loaders = {split: DataLoader(MFCCDataset(frame), batch_size=BATCH_SIZE, shuffle=split == "train", num_workers=0) for split, frame in frames.items()}
    loss_fn = nn.BCEWithLogitsLoss(); optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000"); mlflow.set_experiment("Audio_Dementia_Classification")
    with mlflow.start_run(run_name="SwinTiny_MFCC_seed42") as run:
        mlflow.log_params({**config, **environment, "model_family": "Transformer", "sample_rate": 16000, "n_mfcc": 40, "n_fft": 1024, "hop_length": 512, "win_length": 1024, "n_mels": 40, "fmin": 20, "fmax": 8000, "chunk_seconds": 10, "chunk_hop_seconds": 5})
        best_loss = float("inf"); best_epoch = 0; wait = 0; history = []
        for current_epoch in range(1, MAX_EPOCHS + 1):
            train_loss, train_acc = epoch(model, loaders["train"], loss_fn, optimizer); val_loss, val_acc = epoch(model, loaders["validation"], loss_fn)
            history.append({"epoch": current_epoch, "train_loss": train_loss, "train_accuracy": train_acc, "validation_loss": val_loss, "validation_accuracy": val_acc, "validation_precision": np.nan, "validation_recall": np.nan, "validation_f1": np.nan})
            mlflow.log_metrics({"train_loss": train_loss, "train_accuracy": train_acc, "validation_loss": val_loss, "validation_accuracy": val_acc}, step=current_epoch)
            print(f"Epoch {current_epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss; best_epoch = current_epoch; wait = 0; torch.save(model.state_dict(), MODEL_DIR / "best_swin_tiny_mfcc.pth")
            else:
                wait += 1
                if wait >= PATIENCE: print(f"Early stopping at epoch {current_epoch}"); break
        torch.save(model.state_dict(), MODEL_DIR / "last_swin_tiny_mfcc.pth")
        history_frame = pd.DataFrame(history); history_frame.to_csv(RESULTS / "training_history.csv", index=False)
        for metric, label in (("loss", "Loss"), ("accuracy", "Accuracy")):
            for split_name in ("train", "validation"):
                plt.figure(); plt.plot(history_frame.epoch, history_frame[f"{split_name}_{metric}"], label=split_name); plt.xlabel("Epoch"); plt.ylabel(label); plt.legend(); plt.tight_layout(); plt.savefig(RESULTS / f"{split_name}_{metric}_curve.png"); plt.close()
        model.load_state_dict(torch.load(MODEL_DIR / "best_swin_tiny_mfcc.pth", map_location=DEVICE, weights_only=True)); model.eval(); predictions = []
        with torch.no_grad():
            for _, row in frames["test"].iterrows():
                x, _ = MFCCDataset(pd.DataFrame([row]))[0]; probability = float(torch.sigmoid(model(x.unsqueeze(0).to(DEVICE))).item()); predictions.append({**row.to_dict(), "probability": probability, "prediction": int(probability >= THRESHOLD)})
        chunks = pd.DataFrame(predictions); recordings, people = aggregate(chunks); all_metrics = {"chunk_level": calc(chunks), "recording_level": calc(recordings), "person_level": calc(people), "best_epoch": best_epoch, "best_validation_loss": best_loss}
        chunks[["split", "person_id", "label", "source_file", "recording_name", "chunk_index", "true_label", "probability", "prediction"]].to_csv(RESULTS / "chunk_level_predictions.csv", index=False); recordings.rename(columns={"probability": "mean_probability"}).to_csv(RESULTS / "recording_level_predictions.csv", index=False); people.rename(columns={"probability": "mean_probability"}).to_csv(RESULTS / "person_level_predictions.csv", index=False)
        (RESULTS / "test_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
        matrix = confusion_matrix(people.true_label, people.prediction, labels=[0, 1]); plt.figure(); plt.imshow(matrix, cmap="Blues"); plt.xticks([0, 1], ["nodementia", "dementia"]); plt.yticks([0, 1], ["nodementia", "dementia"]); plt.title("Swin-Tiny person confusion matrix"); plt.colorbar(); plt.tight_layout(); plt.savefig(RESULTS / "confusion_matrix_person.png"); plt.close()
        fpr, tpr, _ = roc_curve(people.true_label, people.probability); plt.figure(); plt.plot(fpr, tpr, label=f"AUC={all_metrics['person_level']['roc_auc']:.4f}"); plt.plot([0, 1], [0, 1], "--"); plt.legend(); plt.tight_layout(); plt.savefig(RESULTS / "roc_curve_person.png"); plt.close()
        (RESULTS / "classification_report_person.txt").write_text(classification_report(people.true_label, people.prediction, labels=[0, 1], target_names=["nodementia", "dementia"], zero_division=0), encoding="utf-8")
        mlflow.log_metrics({f"{level}_{name}": value for level, values in all_metrics.items() if isinstance(values, dict) for name, value in values.items() if np.isfinite(value)}); mlflow.log_metric("best_epoch", best_epoch); mlflow.log_metric("best_validation_loss", best_loss)
        for path in list(MODEL_DIR.glob("best_swin_tiny_mfcc.pth")) + list(MODEL_DIR.glob("last_swin_tiny_mfcc.pth")) + list(RESULTS.iterdir()):
            if path.is_file(): mlflow.log_artifact(str(path))
        manifest = {"experiment_name": "Audio_Dementia_Classification", "run_name": "SwinTiny_MFCC_seed42", "run_id": run.info.run_id, "model": "Swin-Tiny", "dataset": {"train_chunks": len(frames["train"]), "validation_chunks": len(frames["validation"]), "test_chunks": len(frames["test"]), "train_persons": frames["train"].person_id.nunique(), "validation_persons": frames["validation"].person_id.nunique(), "test_persons": frames["test"].person_id.nunique()}, "config": config, "environment": environment, "best_checkpoint": str(MODEL_DIR / "best_swin_tiny_mfcc.pth"), "timestamp": datetime.now(timezone.utc).isoformat()}
        (RESULTS / "swin_experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8"); comparison = pd.DataFrame([{ "model": "CNN", "run_id": "5a6b44b53e3748d198a59932edc9ca0e", "input": "MFCC", "accuracy": 0.42857142857142855, "precision": 0.6666666666666666, "recall": 0.15384615384615385, "f1": 0.25, "roc_auc": 0.5480769230769231 }, {"model": "Swin-Tiny", "run_id": run.info.run_id, "input": "MFCC", **all_metrics["person_level"]}]); comparison.to_csv(RESULTS / "model_comparison.csv", index=False)
        print(f"SWIN RUN ID: {run.info.run_id}"); print(f"PERSON METRICS: {all_metrics['person_level']}")


if __name__ == "__main__": main()