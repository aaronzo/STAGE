# SPDX-License-Identifier: Apache-2.0

import argparse
from typing import Iterable, Optional, Any, Dict
from tqdm import tqdm
import warnings
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ogb.nodeproppred import Evaluator



class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        assert self.x.size(0) == self.y.size(0)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MLP(nn.Module):
    convs: Iterable[nn.Linear]
    piecewise_convs: Optional[Iterable[nn.Linear]]
    bns: Iterable[nn.BatchNorm1d]
    piecewise_bns: Optional[Iterable[nn.BatchNorm1d]]

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        piecewise = None,
        **kw
    ) -> None:
        super(MLP, self).__init__()
        if kw:
            warnings.warn(f"Keyworks passed to {self.__class__.__name__} were ignored: {kw}", stacklevel=0)
        self.piecewise_convs = self.piecewise_bns = None
        self.is_piecewise = piecewise is not None and piecewise > 0
        if self.is_piecewise:
            _in_channels = in_channels // piecewise  # for now this has to perfectly divide. Maybe multiply up instead.
            _hidden_channels = hidden_channels // piecewise
            self.piecewise_convs = nn.ModuleList(
                [torch.nn.Linear(_in_channels, _hidden_channels) for _ in range(piecewise)]
            )
            self.piecewise_bns = nn.ModuleList(
                [torch.nn.BatchNorm1d(_hidden_channels) for _ in range(piecewise)]
            )
        self.convs = nn.ModuleList(
            ([] if piecewise else [nn.Linear(in_channels, hidden_channels)]) + 
            [nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers - 2)] +
            [nn.Linear(hidden_channels, out_channels)]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(len(self.convs) - 1)]
        )
        self.dropout = dropout
        self.piecewise = piecewise

    def reset_parameters(self) -> None:
        for lin in self.convs:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.is_piecewise:
            for lin in self.piecewise_convs:
                lin.reset_parameters()
            for bn in self.piecewise_bns:
                bn.reset_parameters()

    def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
        if self.is_piecewise:
            pieces = []
            for i, (lin, bn) in zip(self.piecewise_convs, self.piecewise_bns):
                start = i * self.piecewise
                x_piece = lin(x[start : start + self.piecewise])
                x_piece = bn(x_piece)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                pieces.append(x_piece)
            x = torch.cat(pieces)

        for lin, bn in zip(self.convs[:-1], self.bns):
            x = lin(x)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x)
        return x
         # skip applying softmax as this is done via cross-entropy loss


def train(model: nn.Module, device: Any, train_loader: DataLoader, optimizer: optim.Optimizer) -> float:
    model.train()

    total_loss = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y.squeeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model: nn.Module, device: Any, loader: DataLoader, evaluator: Evaluator) -> Dict[str, float]:
    model.eval()

    y_pred, y_true = [], []
    for x, y in tqdm(loader):
        x = x.to(device)
        out = model(x)

        y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        y_true.append(y)

    return evaluator.eval({
        "y_true": torch.cat(y_true, dim=0),
        "y_pred": torch.cat(y_pred, dim=0),
    })['acc']

# run sign
# python3 sign_training.py --device 0 --dropout 0.3 --lr 0.00005 --hidden_channels 512 --num_layers 3 --embeddings_file_name sign_333_embeddings.pt --result_file_name sign_results.txt

# run sign-xl
# python3 sign_training.py --device 1 --dropout 0.5 --lr 0.00005 --hidden_channels 2048 --num_layers 3 --embeddings_file_name sign_333_embeddings.pt --result_file_name sign-xl_results.txt


def main():
    parser = argparse.ArgumentParser(description='OGBN-papers100M (SIGN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--embeddings_file_name', type=str, default='op_dict.pt')
    parser.add_argument('--result_file_name', type=str, default='results.txt')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    try:
        op_dict = torch.load(args.embeddings_file_name)
    except:
        raise RuntimeError('File {} not found. Need to run python preprocessing.py first'.format(args.embeddings_file_name))

    split_idx = op_dict['split_idx']
    x = torch.cat(op_dict['op_embedding'], dim=1)
    y = op_dict['label'].to(torch.long)
    num_classes = 172
    print('Input feature dimension: {}'.format(x.shape[-1]))
    print('Total number of nodes: {}'.format(x.shape[0]))

    train_dataset = SimpleDataset(x[split_idx['train']], y[split_idx['train']])
    valid_dataset = SimpleDataset(x[split_idx['valid']], y[split_idx['valid']])
    test_dataset = SimpleDataset(x[split_idx['test']], y[split_idx['test']])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = MLP(
        x.size(-1),
        args.hidden_channels,
        num_classes,
        args.num_layers,
        args.dropout).to(device)

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of parameters: {}.'.format(num_trainable_parameters))
    
    evaluator = Evaluator(name='ogbn-papers100M')

    for run in range(args.runs):
        
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            
            train(model, device, train_loader, optimizer)
            train_acc = test(model, device, train_loader, evaluator)
            valid_acc = test(model, device, valid_loader, evaluator)
            test_acc = test(model, device, test_loader, evaluator)

            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%')

if __name__=='__main__':
    main()