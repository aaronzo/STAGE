

if [ ! -d "dataset/PubMed_orig" ]; then
    # ogbn_arxiv_orig
    gdown 1sYZX-jP6H8OkopVa9cp8-KXdEti5ki_W
    unzip PubMed_orig.zip
    mv PubMed_orig dataset/
    rm -rf PubMed_orig.zip
fi


if [ ! -d "gpt_responses/PubMed" ]; then
    # ogbn_arxiv_orig GPT responses
    gdown 166waPAjUwu7EWEvMJ0heflfp0-4EvrZS
    unzip PubMed.zip
    mkdir -p gpt_responses
    mv PubMed gpt_responses/
    rm -rf PubMed.zip
fi

