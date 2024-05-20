import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionMLP(nn.Module):

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        num_operators: int,
        **kw,
    ) -> None:
        super(InceptionMLP, self).__init__()
        if kw:
            warnings.warn(f"Keywords passed to {self.__class__.__name__} were ignored: {kw}")

        self.dropout = dropout
        self.in_channels_per_operator = in_channels // num_operators
        self.num_operators = num_operators

        self.inception_lins = nn.ModuleList(
            [nn.Linear(self.in_channels_per_operator, hidden_channels) for _ in range(num_operators)]
        )
        self.inception_bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(num_operators)]
        )
        self.classifier_lins = nn.ModuleList(
            # the inception layer is an extra hidden layer
            [nn.Linear(hidden_channels * self.num_operators, hidden_channels)] +
            [nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers - 3)] +
            [nn.Linear(hidden_channels, out_channels)]
        )
        self.classifier_bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(len(self.classifier_lins) - 1)]
        )

    def reset_parameters(self) -> None:
        for lin in self.inception_lins:
            lin.reset_parameters()
        for bn in self.inception_bns:
            bn.reset_parameters()
        for lin in self.classifier_lins:
            lin.reset_parameters()
        for bn in self.classifier_bns:
            bn.reset_parameters()

    def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
        start, incr = 0, self.in_channels_per_operator

        xs = []
        for lin, bn in zip(self.inception_lins, self.inception_bns):
            xi = x[:, start:start + incr]
            xi = lin(xi)
            xi = bn(xi)
            start += incr
            xs.append(xi)
        x = torch.cat(xs, dim=1)
        # paper applies prelu on ogbn-products
        x = F.relu(x) 
        x = F.dropout(x, p=self.dropout, training=self.training)

        for lin, bn in zip(self.classifier_lins[:-1], self.classifier_bns):
            x = lin(x)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
    
        x = self.classifier_lins[-1](x)
        # skip applying softmax as this is done via cross-entropy loss instead of nll
        return x
