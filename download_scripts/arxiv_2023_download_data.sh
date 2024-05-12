


if [ ! -d "gpt_responses/arxiv_2023" ]; then
    # ogbn_arxiv_orig GPT responses
    curl -L -o arxiv_2023.zip "https://dl.dropboxusercontent.com/scl/fi/cpy9m3mu6jasxr18scsoc/arxiv_2023.zip?rlkey=4wwgw1pgtrl8fo308v7zpyk59&dl=1"
    unzip arxiv_2023.zip
    mkdir -p gpt_responses
    mv arxiv_2023 gpt_responses/
    rm -rf arxiv_2023.zip
fi

