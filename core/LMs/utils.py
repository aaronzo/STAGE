import torch
import numpy as np
import gc
from core.data_utils.dataset import Dataset
from core.data_utils.load import load_data
from transformers import AutoTokenizer
import logging
from textwrap import dedent
from typing import Literal, Union
from core.GNNs.gnn_utils import Evaluator
from ogb.nodeproppred import Evaluator as OgbEvaluator

_default_task_descriptions = {
    'ogbn-arxiv': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts',
    'arxiv_2023': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts',
    'cora': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts',
    'pubmed': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts',
    'ogbn-products': 'Identify the main and secondary category of this product based on the titles and description',
}

_graph_aware_task_descriptions = {
    'ogbn-arxiv': dedent("""\
        Identify the main and secondary category of Arxiv papers based on the titles and abstracts.
        Your predictions will be used in a downstream graph-based prediction that for each paper can learn from
        your predictions of neighboring papers in a graph as well as the predictions for the paper
        in question. Papers in the graph are connected if one cites the other.
    """),
    'arxiv_2023': dedent("""\
        Identify the main and secondary category of Arxiv papers based on the titles and abstracts.
        Your predictions will be used in a downstream graph-based prediction that for each paper can learn from
        your predictions of neighboring papers in a graph as well as the predictions for the paper
        in question. Papers in the graph are connected if one cites the other.
    """),
    'cora': dedent("""\
        Identify the main and secondary category of Arxiv papers based on the titles and abstracts.
        Your predictions will be used in a downstream graph-based prediction that for each paper can learn from
        your predictions of neighboring papers in a graph as well as the predictions for the paper
        in question. Papers in the graph are connected if one cites the other.
    """),
    'pubmed': dedent("""\
        Identify the main and secondary category of Arxiv papers based on the titles and abstracts.
        Your predictions will be used in a downstream graph-based prediction that for each paper can learn from
        your predictions of neighboring papers in a graph as well as the predictions for the paper
        in question. Papers in the graph are connected if one cites the other.
    """),
    'ogbn-products': dedent("""\
        Identify the main and secondary category of this product based on the titles and description.
        Your predictions will be used in a downstream graph-based prediction that for each product can learn from
        your predictions of neighboring products in a graph as well as the predictions for the paper in question.
        Products in the graph are connected if they are purchased together.
    """),
}

def clear_cuda_cache():
    torch.cuda.empty_cache()

def free_memory(*args):
    for arg in args:
        del arg

def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def get_evaluator(dataset_name: str, preds: np.ndarray, labels: np.ndarray):
    cls = OgbEvaluator if dataset_name.startswith("ogbn") else Evaluator
    evaluator = cls(name=dataset_name)

    def _eval_fn(idx):
        out = evaluator.eval(
            {"y_true": torch.tensor(labels[idx]).view(-1, 1),
            "y_pred": torch.tensor(np.argmax(preds[idx], -1)).view(-1, 1),}
        )
        return out["acc"]
    
    return _eval_fn


def get_task_description(
    dataset_name: str,
    task_type: Union[Literal["default"], Literal["no-task"], Literal["graph-aware"]] = "default",
) -> str:
    task_type = task_type.lower().strip()
    if task_type == "default":
        return _default_task_descriptions[dataset_name]
    elif task_type == "no-task":
        return ""
    elif task_type == "graph-aware":
        return _graph_aware_task_descriptions[dataset_name]
    else:
        raise ValueError(f"Unknown task type '{task_type}', using defaults.")


def count_trainable_parameters(model: torch.nn.Module):
    return sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)