dataset=$1

for lm_model in 'Salesforce/SFR-Embedding-Mistral' 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp' 'Alibaba-NLP/gte-Qwen1.5-7B-instruct'
do
    results_file="./results/$dataset/sign-$lm_model.jsonl"
    mkdir -p `dirname $results_file`

    for spt in "300" "301" "330" "421" "530"
    do
        for seed in "3" "12" "42" "69"
        do  
        echo "---------------------------------------------------------------------------------"
        echo "Running pipeline for dataset '$dataset' seed '$seed' LM '$lm_model'"
        echo "---------------------------------------------------------------------------------"

        python -m core.trainGNN \
            dataset ${dataset} \
            seed ${seed} \
            lm.model.name ${lm_model} \
            gnn.model.name SIGN \
            gnn.model.num_layers 3 \
            gnn.diffusion.spt ${spt} \
            gnn.train.wd 0.0001 \
            gnn.train.dropout 0.5 \
            gnn.train.lr 0.0001 \
            gnn.train.early_stop 0 \
            results_file ${results_file}

        python -m core.trainGNN \
            dataset ${dataset} \
            seed ${seed} \
            lm.model.name ${lm_model} \
            gnn.model.name SIGN \
            gnn.model.num_layers 3 \
            gnn.diffusion.spt ${spt} \
            gnn.diffusion.sym True \
            gnn.train.wd 0.0001 \
            gnn.train.dropout 0.5 \
            gnn.train.lr 0.0001 \
            gnn.train.early_stop 0 \
            results_file ${results_file}
        
        done
    done
done
