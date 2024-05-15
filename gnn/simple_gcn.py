import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import sys

class LogisticRegression(nn.Module):    
    def __init__(self, *, in_channels: int, out_channels: int, dropout: float, **kw) -> None:
        if kw:
            warnings.warn(f"Keyworks passed to {self.__class__.__name__} were ignored: {kw}", stacklevel=0)
        super(LogisticRegression, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)
        # skip applying softmax as this is done via cross-entropy loss
    
    def reset_parameters(self) -> None:
        self.lin.reset_parameters()