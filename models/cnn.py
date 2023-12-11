import torch
import torch.nn as nn
import numpy as np

class CNN_XS(nn.Module):
    def __init__(self, input_dim, output_dim, lookback, num_filters=16, kernel_size=3, **params):
        super(CNN_XS, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_filters * (lookback - kernel_size + 1), output_dim)
        self.to_device()

    def forward(self, x):
        x = self.conv1(x.view(x.size(0), 1, -1))
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def to_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

class CNN_M(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, lookback=10, num_filters=16, kernel_size=3, **params):
        super(CNN_M, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim, num_filters, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(num_filters, 2*num_filters, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(2*num_filters, 4*num_filters, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self._calc_fc_input_size(lookback), 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)
        self.to_device()

    def forward(self, x):
        x = self.layer1(x.view(x.size(0), 1, -1))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def _calc_fc_input_size(self, lookback):
        dummy_input = torch.zeros(1, 1, lookback)
        dummy_output = self.layer3(self.layer2(self.layer1(dummy_input)))
        return int(np.prod(dummy_output.size()))

    def to_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)