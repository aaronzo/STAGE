import torch
import numpy as np

_task_descriptions = {
    'ogbn-arxiv': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts',
    'arxiv_2023': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts',
    'cora': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts',
    'pubmed': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts',
    'ogbn-products': 'Identify the main and secondary category of this product based on the titles and description',
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


def get_evaluator(dataset_name: str):
    from core.GNNs.gnn_utils import Evaluator

    _evaluator = Evaluator(dataset_name)
    def evaluator(self, pred, labels):
        return _evaluator.eval({
            "y_pred": pred.argmax(dim=-1, keepdim=True),
            "y_true": labels.view(-1, 1)
        })["acc"]
    
    return evaluator


def get_task_description(dataset_name: str):
    return _task_descriptions[dataset_name]

def count_trainable_parameters(model: torch.nn.Module):
    return sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)