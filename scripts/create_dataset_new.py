import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from utils.data_loader import load_participant

# ── Command line arguments ────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('-in_dir',  type=str, required=True)
parser.add_argument('-out_dir', type=str, required=True)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── Settings ──────────────────────────────────────────────────────────────────
LOWCUT     = 0.17   # Hz — lower breathing frequency
HIGHCUT    = 0.40   # Hz — upper breathing frequency
WINDOW_SEC = 30     # seconds per window
OVERLAP    = 0.5    # 50% overlap
FS         = 32     # Hz — we work at 32Hz (SpO2 aligned to this)

WINDOW_SIZE = int(WINDOW_SEC * FS)          # 30 * 32 = 960 samples
STRIDE      = int(WINDOW_SIZE * OVERLAP)    # 960 * 0.5 = 480 samples

# ── Bandpass filter ───────────────────────────────────────────────────────────
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq  = fs / 2.0
    low  = lowcut  / nyq
    high = highcut / nyq
    sos  = butter(order, [low, high], btype='bandpass', output='sos')
    return sosfilt(sos, signal)

# ── Windowing ─────────────────────────────────────────────────────────────────
def create_windows(signal, window_size, stride):
    windows = []
    start   = 0
    while start + window_size <= len(signal):
        windows.append(signal[start : start + window_size])
        start += stride
    return np.array(windows)

# ── Label each window ─────────────────────────────────────────────────────────
def get_label(window_start_sec, window_end_sec, events_df):
    """
    Returns event label if >50% overlap with any event, else 'Normal'
    """
    window_duration = window_end_sec - window_start_sec  # 30 seconds

    for _, event in events_df.iterrows():
        overlap_start    = max(window_start_sec, event['start_sec'])
        overlap_end      = min(window_end_sec,   event['end_sec'])
        overlap_duration = overlap_end - overlap_start

        if overlap_duration > 0:
            if (overlap_duration / window_duration) > 0.5:
                return event['label']

    return 'Normal'

# ── Process all participants ──────────────────────────────────────────────────
participants = sorted([
    p for p in os.listdir(args.in_dir)
    if os.path.isdir(os.path.join(args.in_dir, p))
])

print(f"Found participants: {participants}\n")

all_rows = []

for participant in participants:
    print(f"Processing {participant}...")
    folder = os.path.join(args.in_dir, participant)

    # Load signals and events
    merged_df, events_df, fs_actual = load_participant(folder)

    # Apply bandpass filter to all 3 signals
    flow_filtered   = bandpass_filter(merged_df['flow'].values,   LOWCUT, HIGHCUT, FS)
    thorac_filtered = bandpass_filter(merged_df['thorac'].values, LOWCUT, HIGHCUT, FS)
    spo2_filtered   = bandpass_filter(merged_df['spo2'].values,   LOWCUT, HIGHCUT, FS)

    timestamps = merged_df['timestamp_sec'].values

    # Create windows for each signal
    flow_windows   = create_windows(flow_filtered,   WINDOW_SIZE, STRIDE)
    thorac_windows = create_windows(thorac_filtered, WINDOW_SIZE, STRIDE)
    spo2_windows   = create_windows(spo2_filtered,   WINDOW_SIZE, STRIDE)

    num_windows = len(flow_windows)
    print(f"  Created {num_windows} windows")

    # Label each window
    for i in range(num_windows):
        start_idx = i * STRIDE
        end_idx   = start_idx + WINDOW_SIZE - 1

        # Get actual time in seconds for this window
        start_sec = timestamps[start_idx]
        end_sec   = timestamps[min(end_idx, len(timestamps) - 1)]

        label = get_label(start_sec, end_sec, events_df)

        # Build one row: metadata + all signal values flattened
        row = {
            'participant'    : participant,
            'window_index'   : i,
            'start_time_sec' : start_sec,
            'end_time_sec'   : end_sec,
            'label'          : label
        }

        for j in range(WINDOW_SIZE):
            row[f'flow_{j}']   = flow_windows[i][j]
            row[f'thorac_{j}'] = thorac_windows[i][j]
            row[f'spo2_{j}']   = spo2_windows[i][j]

        all_rows.append(row)

    # Show label distribution for this participant
    p_labels = [r['label'] for r in all_rows if r['participant'] == participant]
    unique, counts = np.unique(p_labels, return_counts=True)
    print(f"  Labels: {dict(zip(unique, counts))}\n")

# ── Save dataset ──────────────────────────────────────────────────────────────
print("Saving dataset...")
dataset_df  = pd.DataFrame(all_rows)
output_path = os.path.join(args.out_dir, 'breathing_dataset.csv')
dataset_df.to_csv(output_path, index=False)

print(f"Saved → {output_path}")
print(f"Total windows: {len(dataset_df)}")
print(f"\nOverall label distribution:")
print(dataset_df['label'].value_counts())
