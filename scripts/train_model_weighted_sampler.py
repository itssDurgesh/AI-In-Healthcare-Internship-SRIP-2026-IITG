import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             confusion_matrix)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

from models.cnn_model import CNN1D

# ── Settings ──────────────────────────────────────────────────────────────────
DATASET_PATH  = 'Dataset/breathing_dataset.csv'
WINDOW_SIZE   = 960
EPOCHS        = 20
BATCH_SIZE    = 32
LEARNING_RATE = 0.001

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Load dataset ──────────────────────────────────────────────────────────────
print("\nLoading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"Total windows: {len(df)}")
print(f"\nLabel distribution:\n{df['label'].value_counts()}\n")

# ── Merge rare classes ────────────────────────────────────────────────────────
df['label'] = df['label'].replace({
    'Body event'  : 'Obstructive Apnea',
    'Mixed Apnea' : 'Obstructive Apnea'
})

# ── Encode labels ─────────────────────────────────────────────────────────────
label_encoder        = LabelEncoder()
df['label_encoded']  = label_encoder.fit_transform(df['label'])
num_classes          = len(label_encoder.classes_)

print(f"Final classes: {label_encoder.classes_}")
print(f"Encoded as:    {list(range(num_classes))}\n")

# ── Column names ──────────────────────────────────────────────────────────────
flow_cols   = [f'flow_{i}'   for i in range(WINDOW_SIZE)]
thorac_cols = [f'thorac_{i}' for i in range(WINDOW_SIZE)]
spo2_cols   = [f'spo2_{i}'   for i in range(WINDOW_SIZE)]

# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class BreathingDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        flow   = row[flow_cols].values.astype(np.float32)    # (960,)
        thorac = row[thorac_cols].values.astype(np.float32)  # (960,)
        spo2   = row[spo2_cols].values.astype(np.float32)    # (960,)

        # Stack → shape (3, 960) for Conv1d
        signal = np.stack([flow, thorac, spo2], axis=0)

        label  = int(row['label_encoded'])

        return torch.tensor(signal), torch.tensor(label)


# ── Training function ─────────────────────────────────────────────────────────
def train_model(model, train_loader, train_df):
    """
    Trains model using class-weighted loss to handle imbalance.
    """
    # Compute class weights — minority classes get higher penalty
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['label_encoded'].values),
        y=train_df['label_encoded'].values
    )
    print(f"  Class weights: { {label_encoder.classes_[i]: round(class_weights[i],2) for i in range(num_classes)} }")

    weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    criterion      = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer      = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()

    for epoch in range(EPOCHS):
        total_loss    = 0
        total_correct = 0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)   # (batch, 3, 960)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            output      = model(X_batch)
            loss        = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss    += loss.item()
            predictions    = torch.argmax(output, dim=1)
            total_correct += (predictions == y_batch).sum().item()
            total_samples += len(y_batch)

        if (epoch + 1) % 5 == 0:
            acc = total_correct / total_samples
            print(f"    Epoch {epoch+1}/{EPOCHS} | "
                  f"Loss: {total_loss/len(train_loader):.4f} | "
                  f"Accuracy: {acc:.3f}")

    return model


# ── Evaluation function ───────────────────────────────────────────────────────
def evaluate_model(model, test_loader):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch     = X_batch.to(DEVICE)
            output      = model(X_batch)
            predictions = torch.argmax(output, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    return np.array(all_labels), np.array(all_preds)


# ── Save confusion matrix heatmap ─────────────────────────────────────────────
def save_confusion_matrix(cm, participant):
    os.makedirs('Visualizations', exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title(f'Confusion Matrix — Test: {participant}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    path = f'Visualizations/{participant}_confusion_matrix.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Heatmap saved → {path}")


# ── Leave-One-Participant-Out Cross Validation ────────────────────────────────
participants = sorted(df['participant'].unique())
print(f"Participants: {participants}")
print(f"Starting LOPO Cross Validation...\n")

fold_results = []

for test_participant in participants:
    print(f"── Fold: Test = {test_participant} ──────────────────────────────")

    train_df = df[df['participant'] != test_participant].copy()
    test_df  = df[df['participant'] == test_participant].copy()

    print(f"  Train: {len(train_df)} windows | Test: {len(test_df)} windows")
    print(f"  Train label distribution:")
    print(f"  {train_df['label'].value_counts().to_dict()}")

    # ── Weighted Sampler ──────────────────────────────────────────────────────
    # Each sample gets weight = 1 / count_of_its_class
    # So minority class samples are picked more often
    label_counts  = np.bincount(train_df['label_encoded'].values)
    sample_weights = [1.0 / label_counts[label]
                      for label in train_df['label_encoded'].values]

    sampler = WeightedRandomSampler(
        weights=torch.FloatTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True   # allows picking same sample multiple times
    )

    # Datasets
    train_dataset = BreathingDataset(train_df)
    test_dataset  = BreathingDataset(test_df)

    # DataLoaders
    # Note: sampler replaces shuffle=True
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              sampler=sampler)
    test_loader  = DataLoader(test_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

    # Train
    model = CNN1D(num_classes=num_classes).to(DEVICE)
    print(f"  Training...")
    model = train_model(model, train_loader, train_df)

    # Evaluate
    print(f"  Evaluating...")
    y_true, y_pred = evaluate_model(model, test_loader)

    # Metrics
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred,    average='macro', zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n  Results for {test_participant}:")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}  (macro)")
    print(f"  Recall:    {rec:.3f}   (macro)")
    print(f"  Confusion Matrix:")
    print(f"  Classes: {list(label_encoder.classes_)}")
    print(f"  {cm}\n")

    # Save heatmap
    save_confusion_matrix(cm, test_participant)

    fold_results.append({
        'test_participant': test_participant,
        'accuracy'        : acc,
        'precision'       : prec,
        'recall'          : rec,
        'confusion_matrix': cm
    })

# ── Final summary ─────────────────────────────────────────────────────────────
print("=" * 60)
print("FINAL SUMMARY — Average across all folds")
print("=" * 60)

mean_acc  = np.mean([r['accuracy']  for r in fold_results])
mean_prec = np.mean([r['precision'] for r in fold_results])
mean_rec  = np.mean([r['recall']    for r in fold_results])

print(f"Mean Accuracy:  {mean_acc:.3f}")
print(f"Mean Precision: {mean_prec:.3f}")
print(f"Mean Recall:    {mean_rec:.3f}")

print("\nPer-fold breakdown:")
for r in fold_results:
    print(f"  {r['test_participant']} → "
          f"Acc={r['accuracy']:.3f} | "
          f"Prec={r['precision']:.3f} | "
          f"Rec={r['recall']:.3f}")