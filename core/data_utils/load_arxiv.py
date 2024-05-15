from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd


def get_raw_text_arxiv(use_text=False, seed=0):
    '''
    It's worth noting that for the ogbn-arxiv dataset, the data split isn't influenced by the seed value.
    --> https://github.com/XiaoxinHe/TAPE/issues/7#issuecomment-1755821345
    '''

    # dataset = PygNodePropPredDataset(
    #     name='ogbn-arxiv', transform=T.ToSparseTensor())
    # https://github.com/pytorch/pytorch/issues/111359#issuecomment-1818013629
    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv', transform=T.Compose([T.ToUndirected(),T.ToSparseTensor()]))
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # data.edge_index = data.adj_t.to_symmetric()
    data.edge_index = data.adj_t
    if not use_text:
        return data, None

    nodeidx2paperid = pd.read_csv(
        'dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')
    nodeidx2paperid['paper id'] = nodeidx2paperid['paper id'].astype(int)

    raw_text = pd.read_csv('dataset/ogbn_arxiv_orig/titleabs.tsv',
                        sep='\t', 
                        # header=None, 
                        skiprows=1,
                        names=['paper id', 'title', 'abs'],
    )
    raw_text['paper id'] = raw_text['paper id'].astype(int, errors='ignore')
    
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')
    text = []
    for ti, ab in zip(df['title'], df['abs']):
        t = 'Title: ' + ti + '\n' + 'Abstract: ' + ab
        text.append(t)
    return data, text
