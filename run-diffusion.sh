dataset=ogbn-arxiv
python -m core.trainEnsemble dataset $dataset gnn.model.name SimpleGCN >> ${dataset}_simplegcn.out