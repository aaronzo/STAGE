
from gnn.diffusion import SIGNDiffusion, torch_to_graphblas, SimpleGCNDiffusion
from core.data_utils.load import load_data
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from core.LMs.utils import free_memory

REDUCED_DIM = 512
FULL_DIM = 4096
DIRECTED = False

TAG = "" if DIRECTED else "-undirected"

datasets = [
    "ogbn-arxiv",
    # "cora",
    # "pubmed",
    # "arxiv_2023",
    # "ogbn-products",
]

models = {
    "sfr": "Salesforce/SFR-Embedding-Mistral",
    "llm2vec": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    "qwen": "Alibaba-NLP/gte-Qwen1.5-7B-instruct",
}

sign_configurations = [
    # (3, 0, 0),
    # (3, 0, 1),
    (3, 3, 0),
    (4, 2, 1),
    # (5, 3, 0),
]

sgc_configurations = [
    2,
    # 3,
    # 4
]


def to_undirected(A):
    return ((A + A.T)/2).new()

def load_edges(dataset, directed=True):
    data, *_ = load_data(dataset)
    edges = getattr(data, "edge_index", getattr(data, "adj_t", None))
    num_nodes = data.x.size(0)
    adj = torch_to_graphblas(edges)
    return adj if directed else to_undirected(adj), num_nodes


def load_embedding(dataset, model, num_nodes, embedding_dim=4096):
    path = f"prt_lm/{dataset}/{model}-seedNone-dim{embedding_dim}.emb"
    data = np.array(
        np.memmap(path, mode='r', dtype=np.float16, shape=(num_nodes, embedding_dim))
    ).astype(np.float32)
    return data


scaler = StandardScaler(copy=True)
svd = Pipeline([
    ("scale", scaler),
    ("svd", TruncatedSVD(REDUCED_DIM, algorithm="arpack"))
])


for dataset in datasets:
    A, num_nodes = load_edges(dataset)
    print("processing", dataset)
    for model in models.values():
        print("processing", model)
        path = Path(f"svd/{dataset}/{model}-dim{REDUCED_DIM}.emb")
        if path.exists() and path.is_file():
            X = np.array(np.memmap(
                path,
                dtype=np.float32,
                mode='r+',
                shape=(num_nodes, REDUCED_DIM),
            ))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            svd_emb = np.memmap(
                path,
                dtype=np.float32,
                mode='w+',
                shape=(num_nodes, REDUCED_DIM),
            )
            X_full = load_embedding(dataset, model, num_nodes)
            X = svd.fit_transform(X_full)
            svd_emb[:] = X
            svd_emb.flush()
            print(f"saved svd of {REDUCED_DIM} dimensions")
            free_memory(svd_emb)

        for k in sgc_configurations:
            diffu = SimpleGCNDiffusion(k)
            path = Path(f"sgc/{dataset}/{model}-SGC_{k}-dim{REDUCED_DIM}{TAG}.emb")
            path.parent.mkdir(parents=True, exist_ok=True)
            sgc = np.memmap(
                path,
                dtype=np.float32,
                mode='w+',
                shape=(num_nodes, REDUCED_DIM),
            )
            sgc[:] = diffu.propagate(A, X)
            sgc.flush()
            free_memory(sgc)
            print(f"saved configuration of SGC({k}) embedding")

        for s, p, t in sign_configurations:
            diffu = SIGNDiffusion(s, p, t)
            dim = REDUCED_DIM * diffu.r
            path = Path(f"sign/{dataset}/{model}-SIGN_{diffu.s}{diffu.p}{diffu.t}-dim{dim}{TAG}.emb")
            path.parent.mkdir(parents=True, exist_ok=True)
            sign = np.memmap(
                path,
                dtype=np.float32, mode='w+',
                shape=(num_nodes, dim),
            )
            sign[:] = diffu.propagate(A, X)
            sign.flush()
            free_memory(sign)
            print(f"saved configuration of SIGN({s}, {p}, {t}) embedding")

            diffu = SIGNDiffusion(s, p, t, s_norm="gcn", p_norm="gcn", t_norm="gcn")
            dim = REDUCED_DIM * diffu.r
            path = Path(f"sign/{dataset}/{model}-SIGN_{diffu.s}{diffu.p}{diffu.t}sym-dim{dim}{TAG}.emb")
            path.parent.mkdir(parents=True, exist_ok=True)
            sign = np.memmap(
                path,
                dtype=np.float32, mode='w+',
                shape=(num_nodes, dim),
            )
            sign[:] = diffu.propagate(A, X)
            sign.flush()
            free_memory(sign)
            print(f"saved configuration of SIGN({s}, {p}, {t}) [gcn norm] embedding")




# for dataset in datasets:
#     A, num_nodes = load_edges(dataset)
#     print("processing", dataset)
#     for model in models.values():
#         print("processing", model)
#         X = load_embedding(dataset, model, num_nodes)
#         X = scaler.fit_transform(X)

#         for k in sgc_configurations:
#             diffu = SimpleGCNDiffusion(k)
#             path = Path(f"sgc/{dataset}/{model}-SGC_{k}-dim{FULL_DIM}.emb")
#             path.parent.mkdir(parents=True, exist_ok=True)
#             sgc = np.memmap(
#                 path,
#                 dtype=np.float32,
#                 mode='w+',
#                 shape=(num_nodes, FULL_DIM),
#             )
#             sgc[:] = diffu.propagate(A, X)
#             sgc.flush()
#             free_memory(sgc)
#             print(f"saved configuration of SGC({k}) full embedding")
