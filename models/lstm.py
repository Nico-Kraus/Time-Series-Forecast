import torch
import torch.nn as nn


class LSTM(nn.Module):
    device: str = "cpu"

    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = (
            torch.zeros(self.n_layers, x.size(0), self.hidden_dim)
            .requires_grad_()
            .to(self.device)
        )
        c0 = (
            torch.zeros(self.n_layers, x.size(0), self.hidden_dim)
            .requires_grad_()
            .to(self.device)
        )
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

    def to_device(self, device: str):
        self.to(device)
        self.device = device
