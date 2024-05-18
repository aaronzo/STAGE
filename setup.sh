torchVersion=${TORCH_VERSION:-2.2.1}
cudaVersion=${CUDA_VERSION:-cu121}

# conda create -n cellotape python=3.8 -y#
# conda activate cellotape
pip install packaging wheel
pip install -r requirements.txt
pip install flash_attn
pip install torch_geometric==2.5.3
# if you install a different version of torch and/or CUDA, 
# you'll need to modify the below cmds
# check torch version by running `import torch; print(torch.__version__)`
# check CUDA version by running `nvidia-smi`
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${torchVersion}+${cudaVersion}.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${torchVersion}+${cudaVersion}.html
pip install dgl -f https://data.dgl.ai/wheels/${cudaVersion}/repo.html
pip install -i https://pypi.org/simple/ bitsandbytes