import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             confusion_matrix)
from sklearn.preprocessing import LabelEncoder

# Add parent folder to path so we can import CNN1D from models/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_model import CNN1D

# ── Settings ──────────────────────────────────────────────────────────────────
DATASET_PATH  = 'Dataset/breathing_dataset.csv'
WINDOW_SIZE   = 960      # samples per window (30s × 32Hz)
NUM_SIGNALS   = 3        # airflow, thoracic, spo2
EPOCHS        = 20       # how many times to go through training data
BATCH_SIZE    = 32       # how many windows per training step
LEARNING_RATE = 0.001    # how fast the model learns

# Use GPU if available, otherwise CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Step 1: Load the dataset ──────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"Total windows: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}\n")

label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])

num_classes = len(label_encoder.classes_)
# Merge rare classes first
df['label'] = df['label'].replace({
    'Body event'  : 'Obstructive Apnea',
    'Mixed Apnea' : 'Obstructive Apnea'
})

# Encode labels as numbers
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])
num_classes = len(label_encoder.classes_)

print(f"Final classes: {label_encoder.classes_}")
print(f"Encoded as: {list(range(num_classes))}\n")

airflow_cols  = [f'flow_{i}'   for i in range(WINDOW_SIZE)]
thoracic_cols = [f'thorac_{i}' for i in range(WINDOW_SIZE)]
spo2_cols     = [f'spo2_{i}'   for i in range(WINDOW_SIZE)]


class BreathingDataset(Dataset):
    def __init__(self, dataframe):
        """
        dataframe: subset of the main df for one fold (train or test)
        """
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        # how many windows in this dataset
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Extract the 3 signals and stack them: shape (3, 960)
        airflow  = row[airflow_cols].values.astype(np.float32)   # (960,)
        thoracic = row[thoracic_cols].values.astype(np.float32)  # (960,)
        spo2     = row[spo2_cols].values.astype(np.float32)      # (960,)

        # Stack into shape (3, 960) — PyTorch Conv1d expects (channels, length)
        signal = np.stack([airflow, thoracic, spo2], axis=0)    # (3, 960)

        label  = int(row['label_encoded'])

        return torch.tensor(signal), torch.tensor(label)


# ── Step 5: Training function ─────────────────────────────────────────────────

def train_model(model, train_loader):
    """
    Trains the model for EPOCHS number of epochs.
    Returns the trained model.
    """
    # CrossEntropyLoss is standard for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer adjusts weights to minimize the loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()  # set model to training mode

    for epoch in range(EPOCHS):
        total_loss    = 0
        total_correct = 0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            # Move data to GPU if available
            X_batch = X_batch.to(DEVICE)  # shape: (batch, 3, 960)
            y_batch = y_batch.to(DEVICE)  # shape: (batch,)

            # Zero out gradients from previous step
            optimizer.zero_grad()

            # Forward pass: get predictions
            output = model(X_batch)        # shape: (batch, num_classes)

            # Calculate loss
            loss = criterion(output, y_batch)

            # Backward pass: calculate gradients
            loss.backward()

            # Update model weights
            optimizer.step()

            # Track stats
            total_loss    += loss.item()
            predictions    = torch.argmax(output, dim=1)
            total_correct += (predictions == y_batch).sum().item()
            total_samples += len(y_batch)

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            acc = total_correct / total_samples
            print(f"    Epoch {epoch+1}/{EPOCHS} | "
                  f"Loss: {total_loss/len(train_loader):.4f} | "
                  f"Accuracy: {acc:.3f}")

    return model


# ── Step 6: Evaluation function ───────────────────────────────────────────────

def evaluate_model(model, test_loader):
    """
    Evaluates the model on the test set.
    Returns true labels and predicted labels.
    """
    model.eval()  # set model to evaluation mode (disables dropout etc.)

    all_preds  = []
    all_labels = []

    with torch.no_grad():  # no need to calculate gradients during evaluation
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)

            output      = model(X_batch)
            predictions = torch.argmax(output, dim=1)  # pick class with highest score

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    return np.array(all_labels), np.array(all_preds)


# ── Step 7: Leave-One-Participant-Out Cross Validation ────────────────────────

participants = sorted(df['participant'].unique())
print(f"Participants: {participants}")
print(f"Starting Leave-One-Participant-Out Cross Validation...\n")

# Store results from each fold
fold_results = []

for test_participant in participants:
    print(f"── Fold: Test = {test_participant} ──────────────────────────────")

    # Split data: everyone except test_participant is training data
    train_df = df[df['participant'] != test_participant]
    test_df  = df[df['participant'] == test_participant]

    print(f"  Train windows: {len(train_df)} | Test windows: {len(test_df)}")

    # Create PyTorch Datasets
    train_dataset = BreathingDataset(train_df)
    test_dataset  = BreathingDataset(test_df)

    # Create DataLoaders (handle batching automatically)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # Create a fresh model for each fold (start from scratch each time)
    model = CNN1D(num_classes=num_classes).to(DEVICE)

    # Train the model
    print(f"  Training...")
    model = train_model(model, train_loader)

    # Evaluate the model
    print(f"  Evaluating...")
    y_true, y_pred = evaluate_model(model, test_loader)

    # Calculate metrics
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n  Results for {test_participant}:")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}  (macro)")
    print(f"  Recall:    {rec:.3f}   (macro)")
    print(f"  Confusion Matrix:")
    print(f"  Classes: {list(label_encoder.classes_)}")
    print(f"  {cm}\n")

    fold_results.append({
        'test_participant': test_participant,
        'accuracy'        : acc,
        'precision'       : prec,
        'recall'          : rec,
        'confusion_matrix': cm
    })

# ── Step 8: Print final summary ───────────────────────────────────────────────
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
