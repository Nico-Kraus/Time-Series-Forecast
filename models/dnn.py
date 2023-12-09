import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    device: str = "cpu"

    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, **params):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = []
        for _ in range(n_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x.squeeze(2)))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.fc3(x)
        return x

    def to_device(self, device: str):
        self.to(device)
        self.device = device
