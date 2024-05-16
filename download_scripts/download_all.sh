cwd=$(pwd)
cd `dirname $0`/..

bash ./download_scripts/arxiv_2023_download_data.sh
bash ./download_scripts/cora_download_data.sh
bash ./download_scripts/ogbn_arxiv_orig_download_data.sh
bash ./download_scripts/ogbn_products_download_data.sh
bash ./download_scripts/pubmed_download_data.sh
bash ./download_scripts/download_llm_embeddings.sh

cd $cwd