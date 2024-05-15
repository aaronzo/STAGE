for dataset in 'ogbn-arxiv' 'ogbn-products' 'cora' 'pubmed' 'arxiv_2023'
do
    for lm_model in 'Alibaba-NLP/gte-Qwen1.5-7B-instruct' 'Salesforce/SFR-Embedding-Mistral'
    do
        python -m core.trainEnsemble gnn.train.feature_type TA dataset ${dataset} seed 42 lm.model.name ${lm_model} #| tee ${dataset}_${lm_model}_ta.out
    done
done
