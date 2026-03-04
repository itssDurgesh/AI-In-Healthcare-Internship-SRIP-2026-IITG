import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.data_loader import find_file

# ── Command line arguments ────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('-in_dir',  type=str, required=True,
                    help='Data folder e.g. Data')
parser.add_argument('-dataset', type=str, required=True,
                    help='Existing breathing dataset e.g. Dataset/breathing_dataset.csv')
parser.add_argument('-out_dir', type=str, required=True,
                    help='Output folder e.g. Dataset')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── Helper: Load sleep profile file ──────────────────────────────────────────
def load_sleep_profile(filepath):
    rows         = []
    first_dt     = None
    data_started = False

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Data starts after blank line
        if line == '':
            data_started = True
            continue

        if not data_started:
            continue

        try:
            # Format: "30.05.2024 20:59:00,000; Wake"
            parts         = line.split('; ')
            timestamp_str = parts[0].replace(',', '.')
            stage         = parts[1].strip()

            # Parse datetime
            dt = datetime.strptime(timestamp_str, '%d.%m.%Y %H:%M:%S.%f')

            if first_dt is None:
                first_dt = dt

            elapsed_sec = (dt - first_dt).total_seconds()

            rows.append({
                'timestamp_sec': elapsed_sec,
                'stage'        : stage
            })

        except Exception:
            continue

    return pd.DataFrame(rows)


# ── Step 1: Load existing breathing dataset ───────────────────────────────────
print("Loading breathing dataset...")
df = pd.read_csv(args.dataset)
print(f"Total windows: {len(df)}")
print(f"Participants: {df['participant'].unique()}")

# ── Step 2: Load sleep profiles for all participants ──────────────────────────
# Build a dictionary: participant → sleep profile DataFrame
sleep_profiles = {}

participants = sorted(df['participant'].unique())

for participant in participants:
    folder = os.path.join(args.in_dir, participant)
    print(f"\nLoading sleep profile for {participant}...")

    try:
        sleep_file = find_file(folder, 'sleep profile', exclude=None)
        print(f"  Found: {os.path.basename(sleep_file)}")

        profile_df = load_sleep_profile(sleep_file)
        print(f"  Loaded {len(profile_df)} sleep stage entries")
        print(f"  Stage distribution: {profile_df['stage'].value_counts().to_dict()}")

        sleep_profiles[participant] = profile_df

    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        sleep_profiles[participant] = None


# ── Step 3: Match each window to its sleep stage ──────────────────────────────
print("\nMatching windows to sleep stages...")

def get_sleep_stage(window_start_sec, profile_df):
    """
    Finds the sleep stage for a given window start time.
    Sleep profile has one entry every 30 seconds.
    We find the nearest entry using merge_asof logic.
    """
    if profile_df is None or len(profile_df) == 0:
        return 'Unknown'

    # Find the profile entry closest to window start time
    # Since profile is every 30s, just find nearest timestamp
    idx   = (profile_df['timestamp_sec'] - window_start_sec).abs().idxmin()
    stage = profile_df.loc[idx, 'stage']
    return stage


# Add sleep stage column to the dataframe
sleep_stages = []

for _, row in df.iterrows():
    participant   = row['participant']
    start_sec     = row['start_time_sec']
    profile_df    = sleep_profiles.get(participant)
    stage         = get_sleep_stage(start_sec, profile_df)
    sleep_stages.append(stage)

df['sleep_stage'] = sleep_stages

# ── Step 4: Show combined distribution ───────────────────────────────────────
print("\n=== Sleep Stage Distribution ===")
print(df['sleep_stage'].value_counts())

print("\n=== Label × Sleep Stage Crosstab ===")
print(pd.crosstab(df['label'], df['sleep_stage']))

output_path = os.path.join(args.out_dir, 'sleep_stage_dataset.csv')

meta_cols = ['participant', 'window_index', 'start_time_sec',
             'end_time_sec', 'label', 'sleep_stage']

sleep_df = df[meta_cols].copy()
sleep_df.to_csv(output_path, index=False)

print(f"\nSaved → {output_path}")
print(f"Total rows: {len(sleep_df)}")
print("\nFirst 10 rows:")
print(sleep_df.head(10))
