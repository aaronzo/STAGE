set +e

# Simple GCN for cora reproducing paper results (https://arxiv.org/pdf/1902.07153)

python -m core.trainGNN \
    dataset cora \
    gnn.model.name SimpleGCN \
    gnn.train.feature_type ogb \
    gnn.train.wd 0.000005 \
    gnn.train.early_stop 0 \
    runs 1 \


# SIGN params for ogbn-products reproducing paper results (https://arxiv.org/pdf/2004.11198)
# !! Note that this was not done on a subset. Hyperparams in appendix B of paper

python -m core.trainGNN \
    dataset ogbn-arxiv \
    gnn.model.name SIGN \
    gnn.model.hidden_dim 512 \
    gnn.model.num_layers 3 \
    gnn.diffusion.s 0 \
    gnn.diffusion.p 0 \
    gnn.diffusion.t 1 \
    gnn.train.feature_type ogb \
    gnn.train.wd 0.0001 \
    gnn.train.dropout 0.5 \
    gnn.train.lr 0.0001 \
    gnn.train.early_stop 0 \
    runs 1
