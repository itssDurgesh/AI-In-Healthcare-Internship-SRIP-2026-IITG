import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def find_file(folder, keyword, exclude=None):
    keyword_clean = keyword.replace(' ', '').lower()
    for filename in os.listdir(folder):
        filename_clean = filename.replace(' ', '').lower()
        if exclude and exclude.replace(' ', '').lower() in filename_clean:
            continue
        if keyword_clean in filename_clean:
            return os.path.join(folder, filename)
    raise FileNotFoundError(f"No file with keyword '{keyword}' in {folder}")

# ── Parse a signal file ───────────────────────────────────────────────────────

def load_signal(filepath):
    rows         = []
    first_dt     = None
    sample_rate  = None
    data_started = False

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Extract sample rate from header
        if line.startswith('Sample Rate:'):
            sample_rate = int(line.split(':')[1].strip())
            continue

        # Mark where actual data begins
        if line == 'Data:':
            data_started = True
            continue

        if not data_started or not line:
            continue

        try:
            # Format: "30.05.2024 20:59:00,031; 120"
            parts         = line.split('; ')
            timestamp_str = parts[0].replace(',', '.')   # fix decimal separator
            value         = float(parts[1])

            # Parse datetime: DD.MM.YYYY HH:MM:SS.mmm
            dt = datetime.strptime(timestamp_str, '%d.%m.%Y %H:%M:%S.%f')

            if first_dt is None:
                first_dt = dt

            elapsed_sec = (dt - first_dt).total_seconds()
            rows.append({'timestamp_sec': elapsed_sec, 'value': value})

        except Exception:
            continue

    df = pd.DataFrame(rows)
    return df, first_dt, sample_rate


# ── Parse the events file ─────────────────────────────────────────────────────

def load_events(filepath, recording_start_dt):
    rows         = []
    data_started = False

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Data starts after the blank line
        if line == '':
            data_started = True
            continue

        if not data_started:
            continue

        try:
            # Format: "30.05.2024 23:48:45,119-23:49:01,408; 16;Hypopnea; N1"
            parts      = line.split(';')
            time_range = parts[0].strip()
            duration   = float(parts[1].strip())
            label      = parts[2].strip()
            stage      = parts[3].strip()

            # time_range = "30.05.2024 23:48:45,119-23:49:01,408"
            date_part  = time_range[:10]    # "30.05.2024"
            times_part = time_range[11:]    # "23:48:45,119-23:49:01,408"

            start_time_str, end_time_str = times_part.split('-')

            # Parse start datetime
            start_str = f"{date_part} {start_time_str}".replace(',', '.')
            start_dt  = datetime.strptime(start_str, '%d.%m.%Y %H:%M:%S.%f')

            # Parse end datetime
            end_str = f"{date_part} {end_time_str}".replace(',', '.')
            end_dt  = datetime.strptime(end_str, '%d.%m.%Y %H:%M:%S.%f')

            # Handle midnight crossing
            if end_dt < start_dt:
                end_dt += timedelta(days=1)

            # Convert to elapsed seconds from recording start
            start_sec = (start_dt - recording_start_dt).total_seconds()
            end_sec   = (end_dt   - recording_start_dt).total_seconds()

            rows.append({
                'start_sec' : start_sec,
                'end_sec'   : end_sec,
                'duration'  : duration,
                'label'     : label,
                'stage'     : stage
            })

        except Exception as e:
            continue

    return pd.DataFrame(rows)


# ── Load all signals for one participant ──────────────────────────────────────

def load_participant(folder):
    # Find files by keyword (handles date in filename automatically)
    flow_file   = find_file(folder, 'flow',   exclude='events')
    thorac_file = find_file(folder, 'thorac', exclude='events')
    spo2_file   = find_file(folder, 'spo2',   exclude='events')
    events_file = find_file(folder, 'events', exclude=None)

    print(f"  Loading Flow:   {os.path.basename(flow_file)}")
    print(f"  Loading Thorac: {os.path.basename(thorac_file)}")
    print(f"  Loading SPO2:   {os.path.basename(spo2_file)}")
    print(f"  Loading Events: {os.path.basename(events_file)}")

    # Load signals
    flow_df,   recording_start, fs_flow   = load_signal(flow_file)
    thorac_df, _,               fs_thorac = load_signal(thorac_file)
    spo2_df,   _,               fs_spo2   = load_signal(spo2_file)

    print(f"  Sample rates → Flow: {fs_flow}Hz | Thorac: {fs_thorac}Hz | SpO2: {fs_spo2}Hz")
    print(f"  Recording start: {recording_start}")

    # Rename value columns
    flow_df   = flow_df.rename(columns={'value': 'flow'})
    thorac_df = thorac_df.rename(columns={'value': 'thorac'})
    spo2_df   = spo2_df.rename(columns={'value': 'spo2'})

    # Sort by timestamp (required for merge_asof)
    flow_df   = flow_df.sort_values('timestamp_sec').reset_index(drop=True)
    thorac_df = thorac_df.sort_values('timestamp_sec').reset_index(drop=True)
    spo2_df   = spo2_df.sort_values('timestamp_sec').reset_index(drop=True)

    # Align thorac (32Hz) to flow (32Hz) — same rate, just merge on timestamp
    merged = pd.merge_asof(
        flow_df,
        thorac_df,
        on='timestamp_sec',
        direction='nearest'
    )

    # Align spo2 (4Hz) to flow (32Hz) using nearest timestamp
    merged = pd.merge_asof(
        merged,
        spo2_df,
        on='timestamp_sec',
        direction='nearest'
    )

    print(f"  Aligned signals → shape: {merged.shape}")

    # Load events (convert to elapsed seconds)
    events_df = load_events(events_file, recording_start)
    print(f"  Events loaded: {len(events_df)}")
    print(f"  Event types: {events_df['label'].unique()}")

    return merged, events_df, fs_flow
