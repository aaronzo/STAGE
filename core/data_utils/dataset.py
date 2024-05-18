
import dgl
import torch
from torch.utils.data import Dataset as TorchDataset, Subset
# convert PyG dataset to DGL dataset


class CustomDGLDataset(TorchDataset):
    def __init__(self, name, pyg_data):
        self.name = name
        self.pyg_data = pyg_data

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        data = self.pyg_data
        g = dgl.DGLGraph()
        if self.name == 'ogbn-arxiv' or self.name == 'ogbn-products':
            edge_index = data.edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
        else:
            edge_index = data.edge_index
        g.add_nodes(data.num_nodes)
        g.add_edges(edge_index[0], edge_index[1])

        if data.edge_attr is not None:
            g.edata['feat'] = torch.FloatTensor(data.edge_attr)
        if self.name == 'ogbn-arxiv' or self.name == 'ogbn-products':
            g = dgl.to_bidirected(g)
            print(
                f"Using GAT based methods,total edges before adding self-loop {g.number_of_edges()}")
            g = g.remove_self_loop().add_self_loop()
            print(f"Total edges after adding self-loop {g.number_of_edges()}")
        if data.x is not None:
            g.ndata['feat'] = torch.FloatTensor(data.x)
        g.ndata['label'] = torch.LongTensor(data.y)
        return g

    @property
    def train_mask(self):
        return self.pyg_data.train_mask

    @property
    def val_mask(self):
        return self.pyg_data.val_mask

    @property
    def test_mask(self):
        return self.pyg_data.test_mask


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        encodings,
        *, 
        edge_index=None,
        adj_t=None,
        num_nodes=None,
        num_classes=None,
        labels=None,
        train_mask=None,
        val_mask=None,
        test_mask=None,
    ):
        self.encodings = encodings
        if labels is not None:
            self.labels = labels
        if edge_index is not None:
            self.edge_index = edge_index
        if adj_t is not None:
            self.adj_t = adj_t
        if train_mask is not None:
            self.train_mask = train_mask 
        if val_mask is not None:
            self.val_mask = val_mask
        if test_mask is not None:
            self.test_mask = test_mask
        if num_classes is not None:
            self.num_classes = num_classes
        self.num_nodes = len(self) if num_nodes is None else num_nodes
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).clone().detach()   # val[idx].clone().detach()   # torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['node_id'] = idx
        if hasattr(self, "labels"):
            item["labels"] = torch.tensor(self.labels[idx])
        if hasattr(self, "edge_index"):
            item["edge_index"] = self.edge_index
        if hasattr(self, "adj_t"):
            item["adj_t"] = self.adj_t

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

    def train_subset(self) -> Subset:
        assert hasattr(self, "train_mask")
        return Subset(self, self.train_mask.nonzero().squeeze().tolist())

    def val_subset(self) -> Subset:
        assert hasattr(self, "val_mask")
        return Subset(self, self.val_mask.nonzero().squeeze().tolist())
    
    def test_subset(self) -> Subset:
        assert hasattr(self, "test_mask")
        return Subset(self, self.test_mask.nonzero().squeeze().tolist())
        
