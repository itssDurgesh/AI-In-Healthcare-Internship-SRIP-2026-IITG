# Sleep Apnea Detection — AI for Health (SRIP 2026)

An end-to-end pipeline for detecting breathing irregularities (Hypopnea and Obstructive Apnea) during sleep using physiological signals and deep learning.

---

## Project Overview

Sleep apnea is a condition where breathing repeatedly stops and starts during sleep. This project builds a complete machine learning pipeline to automatically detect these events from overnight physiological recordings.

**Signals Used:**
- Nasal Airflow (32 Hz) — measures airflow through the nose
- Thoracic/Abdominal Movement (32 Hz) — measures chest expansion
- SpO₂ / Oxygen Saturation (4 Hz) — measures blood oxygen level

**Dataset:** 5 participants (AP01–AP05), ~7.5–8 hours of overnight sleep recording each.

---

## Directory Structure

```
internship/
│
├── Data/
│   ├── AP01/
│   │   ├── Flow - 30-05-2024.txt             ← Nasal airflow (32 Hz)
│   │   ├── Thorac - 30-05-2024.txt           ← Thoracic movement (32 Hz)
│   │   ├── SPO2 - 30-05-2024.txt             ← Oxygen saturation (4 Hz)
│   │   ├── Flow Events - 30-05-2024.txt      ← Annotated breathing events
│   │   └── Sleep profile - 30-05-2024.txt    ← Sleep stage annotations
│   ├── AP02/ ── AP05/                         ← Same structure, different dates
│
├── Dataset/
│   ├── breathing_dataset.csv                 ← 8800 labeled 30s windows
│   └── sleep_stage_dataset.csv              ← Windows with sleep stage labels
│
├── Visualizations/
│   ├── AP01_visualization.pdf               ← 8hr signal plot with event overlays
│   ├── AP01_confusion_matrix_with_sampler.png
│   ├── AP02_confusion_matrix_with_sampler.png
│   ├── AP03_confusion_matrix_with_sampler.png
│   ├── AP04_confusion_matrix_with_sampler.png
│   ├── AP05_confusion_matrix_with_sampler.png
│   └── Result_of_Simple1D_CNN.png           ← Baseline CNN results
│
├── models/
│   ├── cnn_model.py                         ← 1D CNN classifier
│   └── conv_lstm_model.py                   ← CNN + LSTM classifier
│
├── scripts/
│   ├── vis_new.py                           ← Task 1: Signal visualization
│   ├── create_dataset_new.py               ← Task 2: Preprocessing + windowing
│   ├── create_sleep_dataset.py             ← Sleep stage dataset creation
│   ├── train_model.py                      ← Task 3: Baseline 1D CNN training
│   └── train_model_weighted_sampler.py     ← Task 3: Class-balanced training
│
├── utils/
│   └── data_loader.py                      ← Shared signal parsing helpers
│
├── README.md
├── requirements.txt
└── report.pdf
```

---

## Installation

```bash
git clone https://github.com/itssDurgesh/AI-In-Healthcare-Internship-SRIP-2026-IITG.git
cd sleep-apnea-detection
pip install -r requirements.txt
```

**Requirements:**
```
numpy
pandas
scipy
matplotlib
seaborn
scikit-learn
torch
```

---

## How to Run

### Task 1 — Visualize Signals
```bash
python scripts/vis_new.py -name "Data/AP01"
```
- Plots Nasal Flow, Thoracic Movement, SpO₂ over full 8-hour recording
- Overlays annotated events as colored bands (orange = Hypopnea, red = Obstructive Apnea)
- Saves PDF to `Visualizations/AP01_visualization.pdf`

---

### Task 2 — Create Dataset
```bash
python scripts/create_dataset_new.py -in_dir "Data" -out_dir "Dataset"
```
- Applies Butterworth bandpass filter (0.17–0.40 Hz) to isolate breathing frequencies
- Aligns SpO₂ (4 Hz) to Flow/Thorac (32 Hz) timeline using timestamp-based merge
- Splits into 30-second windows with 50% overlap → 960 samples per window
- Labels each window: >50% overlap with event → event label, else → Normal
- Saves `Dataset/breathing_dataset.csv`

```bash
python scripts/create_sleep_dataset.py -in_dir "Data" -dataset "Dataset/breathing_dataset.csv" -out_dir "Dataset"
```
- Matches each window to its sleep stage from the Sleep Profile file
- Saves `Dataset/sleep_stage_dataset.csv`

---

### Task 3 — Train & Evaluate Models
```bash
# Baseline — simple 1D CNN (no class balancing)
python scripts/train_model.py

# Improved — weighted sampler to handle class imbalance
python scripts/train_model_weighted_sampler.py
```
- Uses Leave-One-Participant-Out (LOPO) Cross Validation
- Train on 4 participants, test on 1 — repeat for all 5
- Reports Accuracy, Precision, Recall (macro), Confusion Matrix per fold
- Saves confusion matrix heatmaps to `Visualizations/`

---

## Dataset Details

### Signal File Format
Each file has metadata header rows followed by timestamped data:
```
Signal Type: Flow_TH_Type
Start Time: 5/30/2024 8:59:00 PM
Sample Rate: 32
Length: 875184
Unit:
Data:
30.05.2024 20:59:00,000; 120
30.05.2024 20:59:00,031; 120
...
```

### Event File Format
```
30.05.2024 23:48:45,119-23:49:01,408; 16;Hypopnea; N1
↑start datetime          ↑end       ↑dur ↑label    ↑sleep stage
```

### Final Dataset Stats
| Label | Windows | Percentage |
|---|---|---|
| Normal | 8,038 | 91.3% |
| Hypopnea | 593 | 6.7% |
| Obstructive Apnea | 164 | 1.9% |
| **Total** | **8,800** | **100%** |

---

## Signal Processing Pipeline

```
Raw Signals (32 Hz / 4 Hz)
        ↓
Timestamp-based alignment (merge_asof)
        ↓
Butterworth Bandpass Filter (0.17–0.40 Hz, order=4)
        ↓
30-second windows, 50% overlap → (960, 3) per window
        ↓
Label assignment (>50% event overlap rule)
        ↓
1D CNN / ConvLSTM classification
        ↓
LOPO Cross Validation → Metrics
```

---

## Models

### 1D CNN (`models/cnn_model.py`)
Convolutional neural network that slides a kernel along the time axis to detect local patterns in physiological signals.

```
Input: (batch, 3, 960)
→ Conv1d(3→32, k=7) + ReLU + MaxPool   → (batch, 32, 480)
→ Conv1d(32→64, k=5) + ReLU + MaxPool  → (batch, 64, 240)
→ Conv1d(64→128, k=3) + ReLU           → (batch, 128, 240)
→ AdaptiveAvgPool1d(1)                  → (batch, 128, 1)
→ Flatten → Linear(128→64) → Linear(64→3)
Output: (batch, 3)
```

### ConvLSTM (`models/conv_lstm_model.py`)
CNN extracts local features, then LSTM learns how those features evolve over time.

```
Input: (batch, 3, 960)
→ CNN blocks                            → (batch, 128, 120)
→ Permute                               → (batch, 120, 128)
→ 2-layer LSTM(128→128)                → (batch, 120, 128)
→ Take last timestep                    → (batch, 128)
→ Linear(128→64) → Linear(64→3)
Output: (batch, 3)
```

---

## Class Imbalance Handling

The dataset is heavily imbalanced (91% Normal). Without correction the model predicts Normal for everything and achieves 91% accuracy while detecting zero apnea events.

**Fix — two techniques combined:**
1. **WeightedRandomSampler** — minority class windows sampled more often per batch
2. **Weighted CrossEntropyLoss** — missing an Apnea is penalized much more than missing a Normal

---

## Sleep Stage Analysis

Apnea events cluster in **N2** and **REM** sleep — consistent with clinical literature.

| Sleep Stage | Hypopnea | Obstructive Apnea |
|---|---|---|
| N1 (Light Sleep) | 123 | 31 |
| N2 (Core Sleep) | 287 | 72 |
| N3 (Deep Sleep) | 60 | 1 |
| REM | 113 | 59 |
| Wake | 10 | 1 |

Key finding: N3 (deep sleep) has almost zero Obstructive Apnea — breathing is most stable in deep sleep. REM has disproportionately high Apnea relative to its window count.

---

## Evaluation Strategy

**Leave-One-Participant-Out Cross Validation (LOPO):**

Standard random splits cause data leakage — the model memorizes person-specific breathing patterns. LOPO ensures no participant appears in both train and test sets.

```
Fold 1: Train = AP02, AP03, AP04, AP05  |  Test = AP01
Fold 2: Train = AP01, AP03, AP04, AP05  |  Test = AP02
Fold 3: Train = AP01, AP02, AP04, AP05  |  Test = AP03
Fold 4: Train = AP01, AP02, AP03, AP05  |  Test = AP04
Fold 5: Train = AP01, AP02, AP03, AP04  |  Test = AP05
```

Recall is the most critical metric — missing a real apnea event is clinically dangerous.

---

## Important Notes

- File naming is inconsistent across participants (different date formats, different signal names). `data_loader.py` handles all variations using flexible keyword-based file search.
- SpO₂ (4 Hz) is aligned to 32 Hz timeline using `pandas.merge_asof` on timestamps. Index-based alignment would be wrong — same index represents different time points across signals.
- `Body event` and `Mixed Apnea` labels (5 total windows) are merged into `Obstructive Apnea` before training due to insufficient sample count.

---

## Acknowledgements

Dataset provided as part of the SRIP 2026 internship program.
