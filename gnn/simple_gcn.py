import torch
import torch.nn as nn
import warnings

class LogisticRegression(nn.Module):    
    def __init__(self, *, in_channels: int, out_channels: int, **kw) -> None:
        if kw:
            warnings.warn(f"Keywords passed to {self.__class__.__name__} were ignored: {kw}")
        super(LogisticRegression, self).__init__()
        self.W = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
        return self.W(x)
    
    def reset_parameters(self) -> None:
        self.W.reset_parameters()
