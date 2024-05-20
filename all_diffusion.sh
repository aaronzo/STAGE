for dataset in 'ogbn-arxiv' 'ogbn-products' 'cora' 'pubmed' 'arxiv_2023'
do
    bash ./sign.sh ${dataset}
    bash ./sgc.sh ${dataset}
done