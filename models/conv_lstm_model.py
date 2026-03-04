import torch
import torch.nn as nn


class ConvLSTM(nn.Module):
    def __init__(self, num_classes=3):
        """
        num_classes: number of output classes
                     e.g. 3 = Normal, Hypopnea, Obstructive Apnea
        """
        super(ConvLSTM, self).__init__()
        self.conv_block = nn.Sequential(

            # Block 1: (batch, 3, 960) → (batch, 32, 480)
            nn.Conv1d(in_channels=3, out_channels=32,
                      kernel_size=7, padding=3),
            nn.BatchNorm1d(32),   # normalizes across batch for stable training
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # 960 → 480

            # Block 2: (batch, 32, 480) → (batch, 64, 240)
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # 480 → 240

            # Block 3: (batch, 64, 240) → (batch, 128, 120)
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # 240 → 120
        )

        self.lstm = nn.LSTM(
            input_size=128,     # features per time step (from CNN output)
            hidden_size=128,    # size of LSTM hidden state
            num_layers=2,       # stack 2 LSTM layers for more capacity
            batch_first=True,   # input shape: (batch, seq, features)
            dropout=0.3,        # dropout between LSTM layers
            bidirectional=False # reads sequence forward only
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        # ── CNN feature extraction ────────────────────────────────────────────
        # x shape: (batch, 3, 960)
        x = self.conv_block(x)
        # x shape: (batch, 128, 120)

        # ── Reshape for LSTM ──────────────────────────────────────────────────
        # Conv1d output: (batch, channels, time)
        # LSTM expects:  (batch, time, channels)
        # So we permute: (batch, 128, 120) → (batch, 120, 128)
        x = x.permute(0, 2, 1)
        # x shape: (batch, 120, 128)

        # ── LSTM temporal learning ────────────────────────────────────────────
        # lstm_out contains output at every time step: (batch, 120, 128)
        # h_n is the final hidden state: (num_layers, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # We only need the LAST time step output
        # lstm_out[:, -1, :] = output at the final time step
        # shape: (batch, 128)
        last_output = lstm_out[:, -1, :]

        # ── Classification ────────────────────────────────────────────────────
        out = self.classifier(last_output)
        # shape: (batch, num_classes)

        return out


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    fake_input = torch.randn(8, 3, 960)   # batch=8, 3 signals, 960 timesteps

    model = ConvLSTM(num_classes=3)
    output = model(fake_input)

    print(f"Input shape:  {fake_input.shape}")   # (8, 3, 960)
    print(f"Output shape: {output.shape}")        # (8, 3)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("ConvLSTM model works correctly!")
