

if [ ! -d "dataset/ogbn_arxiv_orig" ]; then
    # ogbn_arxiv_orig
    curl https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz -o titleabs.tsv.gz
    gunzip titleabs.tsv.gz
    mkdir -p dataset/ogbn_arxiv_orig
    mv titleabs.tsv dataset/ogbn_arxiv_orig/
    rm -rf titleabs.tsv.gz
fi


if [ ! -d "gpt_responses/arxiv" ]; then
    # ogbn_arxiv_orig GPT responses
    gdown 1A6mZSFzDIhJU795497R6mAAM2Y9qutI5
    unzip ogbn-arxiv.zip
    mkdir -p gpt_responses
    mv ogbn-arxiv gpt_responses/
    rm -rf ogbn-arxiv.zip
fi

