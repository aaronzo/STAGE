set + e
dataset=ogbn-arxiv
python -m core.trainEnsemble dataset $dataset gnn.model.name SimpleGCN seed 0
python -m core.trainEnsemble dataset $dataset gnn.model.name SIGN seed 0