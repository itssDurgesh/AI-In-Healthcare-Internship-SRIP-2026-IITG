# scripts/vis.py
# Loads signals for one participant, plots all 3 signals,
# overlays breathing events, and saves as PDF.
#
# How to run (from project root):
#   python scripts/vis.py -name "Data/AP01"

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from utils.data_loader import load_participant

# ── Read command line arguments ───────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('-name', type=str, required=True,
                    help='Path to participant folder e.g. Data/AP01')
args = parser.parse_args()

participant_folder = args.name
participant_name   = os.path.basename(participant_folder)

print(f"\nLoading data for: {participant_name}")

# ── Load all signals and events ───────────────────────────────────────────────
merged_df, events_df, fs = load_participant(participant_folder)

# Convert timestamps to minutes for easier reading on plot
merged_df['time_min']  = merged_df['timestamp_sec'] / 60
events_df['start_min'] = events_df['start_sec'] / 60
events_df['end_min']   = events_df['end_sec']   / 60

print(f"\nTime range: 0 to {merged_df['time_min'].max():.1f} minutes")
print(f"Total events to overlay: {len(events_df)}")

# ── Event colors ──────────────────────────────────────────────────────────────
event_colors = {
    'Hypopnea'          : 'orange',
    'Obstructive Apnea' : 'red',
}

# ── Create figure with 3 subplots sharing same x-axis ────────────────────────
fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(22, 10),
    sharex=True
)

fig.suptitle(
    f'Sleep Study — Participant {participant_name}',
    fontsize=16,
    fontweight='bold'
)

# ── Plot 1: Nasal Flow ────────────────────────────────────────────────────────
axes[0].plot(
    merged_df['time_min'],
    merged_df['flow'],
    color='steelblue',
    linewidth=0.4
)
axes[0].set_ylabel('Nasal Flow\n(L/min)', fontsize=9)
axes[0].set_title('Nasal Flow (32 Hz)', fontsize=10)

# ── Plot 2: Thoracic Movement ─────────────────────────────────────────────────
axes[1].plot(
    merged_df['time_min'],
    merged_df['thorac'],
    color='darkorange',
    linewidth=0.4
)
axes[1].set_ylabel('Thoracic\nAmplitude', fontsize=9)
axes[1].set_title('Thoracic/Abdominal Resp. (32 Hz)', fontsize=10)

# ── Plot 3: SpO2 ──────────────────────────────────────────────────────────────
axes[2].plot(
    merged_df['time_min'],
    merged_df['spo2'],
    color='grey',
    linewidth=0.6
)
axes[2].set_ylabel('SpO2 (%)', fontsize=9)
axes[2].set_title('Oxygen Saturation — SpO2 (4 Hz)', fontsize=10)
axes[2].set_xlabel('Time (minutes)', fontsize=10)

# ── Overlay events as colored bands on all 3 subplots ────────────────────────
for _, event_row in events_df.iterrows():
    start = event_row['start_min']
    end   = event_row['end_min']
    label = event_row['label']
    color = event_colors.get(label, 'purple')

    for ax in axes:
        ax.axvspan(start, end, alpha=0.3, color=color)

# ── Build clean legend ────────────────────────────────────────────────────────
legend_patches = []
for label, color in event_colors.items():
    if label in events_df['label'].values:
        patch = mpatches.Patch(color=color, alpha=0.5, label=label)
        legend_patches.append(patch)

axes[0].legend(handles=legend_patches, loc='upper right', fontsize=9)

# Add grid for readability
for ax in axes:
    ax.grid(True, alpha=0.3, linewidth=0.5)

plt.tight_layout()

# ── Save as PDF ───────────────────────────────────────────────────────────────
os.makedirs('Visualizations', exist_ok=True)
output_path = os.path.join('Visualizations', f'{participant_name}_visualization.pdf')

with PdfPages(output_path) as pdf:
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print(f"\nVisualization saved → {output_path}")
