for seed in 3 12 42 69
do
    for dataset in 'cora' 'pubmed' 'arxiv_2023' 'ogbn-products' 'ogbn-arxiv' 
    do
        for lm_model in 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp' 'Alibaba-NLP/gte-Qwen1.5-7B-instruct' 'Salesforce/SFR-Embedding-Mistral'
        do
            echo "---------------------------------------------------------------------------------"
            echo "Running pipeline for dataset '$dataset' seed '$seed' LM '$lm_model'"
            echo "---------------------------------------------------------------------------------"

            python -m core.trainEnsemble gnn.train.feature_type TA_P dataset ${dataset} seed ${seed} lm.model.name ${lm_model} #| tee ${dataset}_${lm_model}_ta.out
        done
    done
done