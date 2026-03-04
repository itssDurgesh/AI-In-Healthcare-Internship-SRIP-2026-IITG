
import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, num_classes=3):
        """
        num_classes: number of output classes
                     e.g. 3 = Normal, Hypopnea, Obstructive Apnea
        """
        super(CNN1D, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # length: 960 → 480

            # Block 2
            # in_channels=32 because block 1 output has 32 channels
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # length: 480 → 240

            # Block 3
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),             # (batch, 128, 1) → (batch, 128)
            nn.Linear(128, 64),       # compress 128 features to 64
            nn.ReLU(),
            nn.Dropout(p=0.3),        # dropout randomly turns off 30% of neurons
                                      # this prevents overfitting
            nn.Linear(64, num_classes) # final layer: 64 → number of classes
        )

    def forward(self, x):
        """
        x: input tensor of shape (batch_size, 3, 960)
        returns: output tensor of shape (batch_size, num_classes)
        """
        x = self.conv_blocks(x)    # extract features
        x = self.classifier(x)     # classify
        return x


# ── Quick test to check the model works ───────────────────────────────────────
if __name__ == '__main__':
    # Create a fake batch of 8 windows, 3 signals, 960 time steps
    fake_input = torch.randn(8, 3, 960)

    model = CNN1D(num_classes=3)
    output = model(fake_input)

    print(f"Input shape:  {fake_input.shape}")   # (8, 3, 960)
    print(f"Output shape: {output.shape}")        # (8, 3)
    print("Model works correctly!")
