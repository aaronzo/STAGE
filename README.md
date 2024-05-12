
# PLACEHOLDER README

<img src="./overview.svg">

## Python environment setup with Conda
```bash
conda create -n cellotape python=3.8 -y && conda activate cellotape
pip install -r requirements.txt
# if you install a different version of torch, you'll need to modify the below cmds
# check version by running `import torch; print(torch.__version__)`
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.2.1+cu121.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.2.1+cu121.html
```


## 1. Download TAG datasets

Get (A) and (B) by running script:

- ogbn-arxiv  | `bash download_scripts/ogbn_arxiv_orig_download_data.sh`
- ogbn-products (subset) | `bash download_scripts/ogbn_products_download_data.sh`
- arxiv_2023 | `bash download_scripts/arxiv_2023_download_data.sh`
- Cora | `bash download_scripts/cora_download_data.sh`
- PubMed | `bash download_scripts/pubmed_download_data.sh`

### A. Original text attributes

| Dataset | Description |
| ----- |  ---- |
| ogbn-arxiv  | The [OGB](https://ogb.stanford.edu/docs/nodeprop/) provides the mapping from MAG paper IDs into the raw texts of titles and abstracts. <br/>Download the dataset [here](https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz), unzip and move it to `dataset/ogbn_arxiv_orig`.|
| ogbn-products (subset) |  The dataset is located under `dataset/ogbn_products_orig`.|
| arxiv_2023 |  Download the dataset [here](https://drive.google.com/file/d/1-s1Hf_2koa1DYp_TQvYetAaivK9YDerv/view?usp=sharing), unzip and move it to `dataset/arxiv_2023_orig`.|
|Cora| Download the dataset [here](https://drive.google.com/file/d/1hxE0OPR7VLEHesr48WisynuoNMhXJbpl/view?usp=share_link), unzip and move it to `dataset/cora_orig`.|
PubMed | Download the dataset [here](https://drive.google.com/file/d/1sYZX-jP6H8OkopVa9cp8-KXdEti5ki_W/view?usp=sharing), unzip and move it to `dataset/PubMed_orig`.|


### B. LLM responses
| Dataset | Description |
| ----- |  ---- |
| ogbn-arxiv  | Download the dataset [here](https://drive.google.com/file/d/1A6mZSFzDIhJU795497R6mAAM2Y9qutI5/view?usp=sharing), unzip and move it to `gpt_responses/ogbn-arxiv`.|
| ogbn-products (subset)  | Download the dataset [here](https://drive.google.com/file/d/1C769tlhd8pT0s7I3vXIEUI-PK7A4BB1p/view?usp=sharing), unzip and move it to `gpt_responses/ogbn-products`.|
| arxiv_2023 | Download the dataset [here](https://www.dropbox.com/scl/fi/cpy9m3mu6jasxr18scsoc/arxiv_2023.zip?rlkey=4wwgw1pgtrl8fo308v7zpyk59&dl=0), unzip and move it to `gpt_responses/arxiv_2023`.|
|Cora| Download the dataset [here](https://drive.google.com/file/d/1tSepgcztiNNth4kkSR-jyGkNnN7QDYax/view?usp=sharing), unzip and move it to `gpt_responses/cora`.|
PubMed | Download the dataset [here](https://drive.google.com/file/d/166waPAjUwu7EWEvMJ0heflfp0-4EvrZS/view?usp=sharing), unzip and move it to `gpt_responses/PubMed`.|


## 2. LM Stage / Generate Embeddings

### To just generate and save embeddings
```bash
# one of ['cora' 'pubmed' 'ogbn-arxiv' 'arxiv_2023' 'ogbn-products']
python -m core.LMs.generate_embeddings \
--dataset_name ogbn-arxiv \
--seed 42
```


### To fine-tune using the orginal text attributes
```
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM dataset ogbn-arxiv
```

### To fine-tune using the GPT responses
```
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM dataset ogbn-arxiv lm.train.use_gpt True
```


## 3. Training the GNNs

### To use different GNN models
```
python -m core.trainEnsemble gnn.model.name MLP
python -m core.trainEnsemble gnn.model.name GCN
python -m core.trainEnsemble gnn.model.name SAGE
python -m core.trainEnsemble gnn.model.name RevGAT gnn.train.lr 0.002 gnn.train.dropout 0.75
```

### To use different types of features
```
# Our enriched features
python -m core.trainEnsemble gnn.train.feature_type TA_P_E

# Our individual features
python -m core.trainGNN gnn.train.feature_type TA
python -m core.trainGNN gnn.train.feature_type E
python -m core.trainGNN gnn.train.feature_type P

# OGB features
python -m core.trainGNN gnn.train.feature_type ogb
```

### (Example) use only TA embeddings from LLM embedding model
```
python -m core.trainEnsemble gnn.train.feature_type TA dataset arxiv_2023 seed 42 gnn.model.name SAGE
```

## 4. Reproducibility
Use `run.sh` to run the codes and reproduce the published results.

This repository also provides the checkpoints for all trained models `(*.ckpt)` and the TAPE features `(*.emb)` used in the project. Please donwload them [here](https://drive.google.com/drive/folders/1nF8NDGObIqU0kCkzVaisWooGEQlcNSIN?usp=sharing).

### arxiv-2023 dataset
The codes for constructing and processing the `arxiv-2023` dataset are provided [here](https://github.com/XiaoxinHe/arxiv_2023).
