import torch
import torch.nn.functional as F
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModel,
)
from core.data_utils.load import load_data

from tqdm import tqdm
import argparse
import os
import logging
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)    

def clear_cuda_cache():
    torch.cuda.empty_cache()

def free_memory(*args):
    for arg in args:
        del arg


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings for input texts and save to disk.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility.")
    parser.add_argument("--lm_model_name", type=str, required=False, default='Salesforce/SFR-Embedding-Mistral', help="Model to use for embedding generation.")
    return parser.parse_args()

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


def generate_sfr_embedding_mistral(text, emb_path):

    BATCH_SIZE = 64
    max_length = 2048
    EMBED_DIM = 4096
    num_nodes = len(text)  # 169343

    emb = np.memmap(emb_path, dtype=np.float16, mode='w+', shape=(num_nodes, EMBED_DIM))

    # load model and tokenizer
    # ref --> https://huggingface.co/Salesforce/SFR-Embedding-Mistral#transformers
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
    model = AutoModel.from_pretrained(
        'Salesforce/SFR-Embedding-Mistral',
        device_map='auto',
        torch_dtype=torch.float16
        )
    
    task_description = 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts'
    logger.info(f"Using instruction <<{task_description}>>")
    text = [get_detailed_instruct(task_description, t) for t in text]


    for i in tqdm(range(0, len(text), BATCH_SIZE)):
        input_texts = text[i: i+BATCH_SIZE]
        batch_dict = tokenizer(
            input_texts, 
            max_length=max_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        outputs = model(**batch_dict)
        embeddings = last_token_pool(
            outputs.last_hidden_state, batch_dict['attention_mask']
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        if i < BATCH_SIZE:
            logger.info(f"embeddings shape: {embeddings.shape}")

        # Save embeddings to memmap
        emb[i:i+BATCH_SIZE] = embeddings.detach().cpu().numpy().astype(np.float16)

        free_memory(embeddings, outputs, batch_dict)
        clear_cuda_cache()
        gc.collect()

    emb.flush()
    return


@torch.no_grad()
def generate_embeddings_and_save(args):

    emb_dir = f"prt_lm/{args.dataset_name}"
    emb_path = f"{emb_dir}/{args.lm_model_name}-seed{args.seed}.emb"

    os.makedirs('/'.join(emb_path.split('/')[:-1]), exist_ok=True)

    logger.info(f"Generating embeddings for {args.dataset_name} using model: {args.lm_model_name}")
    logger.info(f"EMbeddings will be saved to {emb_path}")


    data, num_classes, text = load_data(
        dataset=args.dataset_name, use_text=True, use_gpt=False, seed=args.seed
    )

    if args.lm_model_name == 'Salesforce/SFR-Embedding-Mistral':
        generate_sfr_embedding_mistral(text, emb_path)

    print("Embeddings generated and saved successfully.")

    return

    
if __name__ == '__main__':
    args = parse_args()
    generate_embeddings_and_save(args)
