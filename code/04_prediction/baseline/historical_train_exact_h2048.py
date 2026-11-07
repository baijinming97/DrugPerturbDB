"""Frozen architecture and correlation definitions for baseline parity checks."""
import torch
from torch import nn

class ExactH2048_978(nn.Module):
    """Three-hidden-layer 2048-unit control-expression baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(978, 2048), nn.ReLU(), nn.Dropout(0.1), nn.Linear(2048, 2048), nn.ReLU(), nn.Dropout(0.1), nn.Linear(2048, 2048), nn.ReLU(), nn.Dropout(0.1), nn.Linear(2048, 978))

    def forward(self, x_ctrl: torch.Tensor) -> torch.Tensor:
        return self.net(x_ctrl)

def sample_pcc(predicted_delta: torch.Tensor, true_delta: torch.Tensor) -> torch.Tensor:
    predicted = predicted_delta - predicted_delta.mean(dim=1, keepdim=True)
    target = true_delta - true_delta.mean(dim=1, keepdim=True)
    numerator = (predicted * target).sum(dim=1)
    denominator = torch.sqrt((predicted.square().sum(dim=1) * target.square().sum(dim=1)).clamp_min(1e-12))
    return numerator / denominator
