

if [ ! -d "dataset/cora_orig" ]; then
    # ogbn_arxiv_orig
    gdown 1hxE0OPR7VLEHesr48WisynuoNMhXJbpl
    unzip cora_orig.zip
    mv cora_orig dataset/
    rm -rf cora_orig.zip
fi


if [ ! -d "gpt_responses/cora_orig" ]; then
    # ogbn_arxiv_orig GPT responses
    gdown 1tSepgcztiNNth4kkSR-jyGkNnN7QDYax
    unzip Cora.zip
    mkdir -p gpt_responses
    mv Cora gpt_responses/
    rm -rf Cora.zip
fi

