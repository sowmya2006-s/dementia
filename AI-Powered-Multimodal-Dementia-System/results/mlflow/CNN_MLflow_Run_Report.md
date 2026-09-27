# CNN MLflow Run Report

## Run identity

- Experiment: `Audio_Dementia_Classification`
- Run ID: `5a6b44b53e3748d198a59932edc9ca0e`
- Run name: `CNN_MFCC_seed42`
- Model: `AudioCNN`
- Creation timestamp: `2026-09-25T10:42:50.077000+00:00`

## Dataset

- Total recordings: 228
- Total persons: 135
- Classes: dementia (131 recordings), nodementia (97 recordings)
- Train: 120 recordings, 93 persons
- Validation: 48 recordings, 21 persons
- Test: 60 recordings, 21 persons
- Split seed: 42
- MFCC chunks: train 1,560; validation 564; test 689
- MFCC shape: 40 x 313
- Person leakage checks: all train/validation/test intersections were 0

## MFCC configuration

- Sample rate: 16,000 Hz
- Mono audio: true
- MFCC coefficients: 40
- FFT size: 1,024
- Hop length: 512
- Window length: 1,024
- Mel filters: 40
- Frequency range: 20-8,000 Hz
- Chunk duration: 10 seconds
- Chunk hop: 5 seconds

## Training configuration

- Batch size: 16
- Learning rate: 0.001
- Optimizer: Adam
- Loss: BCEWithLogitsLoss
- Dropout: 0.3
- Maximum epochs: 30
- Early stopping patience: 7
- Device: CPU
- Best epoch: 1
- Best validation loss: 0.69234694480896

## Final test metrics

### Person level, primary

- Accuracy: 0.42857142857142855
- Precision: 0.6666666666666666
- Recall: 0.15384615384615385
- F1: 0.25
- ROC-AUC: 0.5480769230769231

### Recording level

- Accuracy: 0.4666666666666667
- Precision: 0.7
- Recall: 0.19444444444444445
- F1: 0.30434782608695654
- ROC-AUC: 0.5671296296296297

### Chunk level

- Accuracy: 0.4833091436865022
- Precision: 0.7468354430379747
- Recall: 0.14936708860759493
- F1: 0.2489451476793249
- ROC-AUC: 0.6508567984155688

These are experimental classification results on the frozen test split, not clinical diagnostic performance.

## Artifact list

The target MLflow run contains:

- `best_audio_cnn.pth`
- `training_config.json`
- `training_history.csv`
- `training_loss.png`
- `training_accuracy.png`
- `confusion_matrix.png`
- `roc_curve.png`
- `classification_report.txt`
- `test_metrics.json`
- `chunk_level_predictions.csv`
- `recording_level_predictions.csv`
- `person_level_predictions.csv`

The corresponding project-level outputs are under `models/audio/` and `results/audio/`.

## MLflow storage

- Tracking URI: `sqlite:///C:/Users/sricb/OneDrive/Desktop/AI dementia dection/mlruns.db`
- Backend: SQLite
- Tracking database: `C:\Users\sricb\OneDrive\Desktop\AI dementia dection\mlruns.db`
- Project tracking backup: `mlflow/mlruns.db`
- Run artifact location: `file:///C:/Users/sricb/OneDrive/Desktop/AI%20dementia%20dection/AI-Powered-Multimodal-Dementia-System/mlruns/1/5a6b44b53e3748d198a59932edc9ca0e/artifacts`
