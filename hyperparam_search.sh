for dataset in 'ogbn-products' 'cora' 'pubmed' 'arxiv_2023' 'ogbn-arxiv'
do
    for lm_model in 'Alibaba-NLP/gte-Qwen1.5-7B-instruct' 'Salesforce/SFR-Embedding-Mistral'
    do
        for num_layers in 4 6 8
        do
            for hidden_dim in 128 256 512
            do
                for dropout in 0.1 0.3 0.5
                do
                    python -m core.trainEnsemble gnn.train.feature_type TA_P dataset ${dataset} seed 42 lm.model.name ${lm_model} gnn.model.num_layers ${num_layers} gnn.model.hidden_dim ${hidden_dim} gnn.train.dropout ${dropout} #>> ${dataset}_${lm_model}_layers${num_layers}_dim${hidden_dim}_dropout${dropout}.out
                done
            done
        done
    done
done