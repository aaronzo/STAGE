datasets="${1:-'ogbn-arxiv' 'arxiv_2023' 'ogbn-products'}" 
seeds="${2:-'0 1 2 3'}"

# for dataset in 'cora' 'pubmed' 'ogbn-arxiv' 'arxiv_2023' 'ogbn-products'
for dataset in $datasets
do
    echo "---------------------------------------"
    echo "Running pipeline for dataset '$dataset'"
    echo "---------------------------------------"
    for seed in $seeds
    do
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM dataset $dataset seed $seed >> ${dataset}_lm.out
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM dataset $dataset seed $seed lm.train.use_gpt True  >> ${dataset}_lm2.out
    done
    python -m core.trainEnsemble dataset $dataset gnn.model.name MLP >> ${dataset}_mlp.out
    python -m core.trainEnsemble dataset $dataset gnn.model.name GCN >> ${dataset}_gcn.out
    python -m core.trainEnsemble dataset $dataset gnn.model.name SAGE >> ${dataset}_sage.out
    python -m core.trainEnsemble dataset $dataset gnn.model.name RevGAT gnn.train.lr 0.002 gnn.train.dropout 0.5 >> ${dataset}_revgat.out
done