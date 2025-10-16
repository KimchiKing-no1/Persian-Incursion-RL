# model.py
from typing import Optional
import torch
import torch.nn as nn

class ValueNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,)

def load_value_model(path: Optional[str], in_dim: int) -> ValueNet:
    model = ValueNet(in_dim)
    if path:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    return model
