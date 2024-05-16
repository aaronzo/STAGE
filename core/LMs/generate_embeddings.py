import torch
import torch.nn.functional as F
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModel,
)
import pandas as pd
from core.data_utils.load import load_data
from core.LMs.utils import get_task_description

from tqdm import tqdm
import argparse
import os
import logging
import gc
import huggingface_hub
from core.LMs.utils import *

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
    parser.add_argument("--lm_model_name", type=str, required=False, default='Salesforce/SFR-Embedding-Mistral', help="Model to use for embedding generation.")
    parser.add_argument("--add_instruction", action='store_true', help="Whether to add instruction to the text.")
    return parser.parse_args()


def generate_gte_qwen_7b_instruct(text, emb_path, task_description):

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


    BATCH_SIZE = 8
    max_length = 2048
    EMBED_DIM = 4096
    num_nodes = len(text)  # 169343

    emb = np.memmap(emb_path, dtype=np.float16, mode='w+', shape=(num_nodes, EMBED_DIM))

    # load model and tokenizer
    # ref --> https://huggingface.co/Alibaba-NLP/gte-Qwen1.5-7B-instruct#transformers
    tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen1.5-7B-instruct', trust_remote_code=True)
    model = AutoModel.from_pretrained(
        'Alibaba-NLP/gte-Qwen1.5-7B-instruct', 
        trust_remote_code=True,
        device_map='auto',
        torch_dtype=torch.float16
        )

    text = [get_detailed_instruct(task_description, t) for t in text]


    for i in tqdm(range(0, len(text), BATCH_SIZE)):
        input_texts = text[i: i+BATCH_SIZE]
        batch_dict = tokenizer(
            input_texts, 
            max_length=max_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to('cuda')
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


def generate_sfr_embedding_mistral(text, emb_path, task_description):

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
    
    text = [get_detailed_instruct(task_description, t) for t in text]

    for i in tqdm(range(0, len(text), BATCH_SIZE)):
        input_texts = text[i: i+BATCH_SIZE]
        batch_dict = tokenizer(
            input_texts, 
            max_length=max_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to('cuda')
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

def generate_llm2vec_llama3(text, emb_path, task_description):
    '''
    ref --> https://github.com/McGill-NLP/llm2vec/blob/main/examples/classification.py
    '''
    from llm2vec import LLM2Vec

    BATCH_SIZE = 128
    max_length = 2048
    EMBED_DIM = 4096
    num_nodes = len(text)  # 169343

    print("Loading model...")
    model = LLM2Vec.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        max_length=max_length,
    )

    def append_instruction(instruction, sentences):
        new_sentences = []
        for s in sentences:
            new_sentences.append([instruction, s, 0])
        return new_sentences

    print(f"Encoding {len(text)} texts...")
    texts = [t if type(t) == str else '' for t in text] # some nan values in text
    nan_values = [t for t in text if type(t) != str]
    if len(nan_values) > 0:
        print(f"Found {len(nan_values)} nan values in text.")

    if task_description:
        texts = append_instruction(task_description, texts)
            
    emb = np.memmap(emb_path, dtype=np.float16, mode='w+', shape=(num_nodes, EMBED_DIM))
    
    emb[:] = np.asarray(model.encode(texts, batch_size=BATCH_SIZE))

    emb.flush()
    return


EMBEDDING_DIM = {
    'Salesforce/SFR-Embedding-Mistral': 4096,
    'Alibaba-NLP/gte-Qwen1.5-7B-instruct': 4096,
    'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp': 4096,
}

@torch.no_grad()
def generate_embeddings_and_save(args):

    args.seed = None  # not needed for pretrained model embeddings as we aren't training anything

    embedding_dim = EMBEDDING_DIM[args.lm_model_name]

    if args.add_instruction:
        # to differentiate between saved embeddings
        add_instruction_tag = ''
    else:
        add_instruction_tag = '-no-instruction' 

    emb_dir = f"prt_lm/{args.dataset_name}"
    emb_path = f"{emb_dir}/{args.lm_model_name}{add_instruction_tag}-seed{args.seed}-dim{embedding_dim}.emb"

    os.makedirs('/'.join(emb_path.split('/')[:-1]), exist_ok=True)

    logger.info(f"Generating embeddings for {args.dataset_name} using model: {args.lm_model_name}")
    logger.info(f"EMbeddings will be saved to {emb_path}")

    if args.dataset_name == 'ogbn-products':
        df = pd.read_csv('dataset/ogbn_products_orig/ogbn-products_subset.csv')
        df['text'] = "Title:\n" + df['title'] + "\nContent:\n " + df['content']
        text = df['text'].tolist()
        print(f"ogbn-products example: {text[0]}")
    else:
        data, num_classes, text = load_data(
            dataset=args.dataset_name, use_text=True, use_gpt=False, seed=42
        )

    # https://github.com/microsoft/unilm/blob/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5/utils.py#L142
    if args.add_instruction:
        task_description = get_task_description(args.dataset_name)
        print(f"<<Using instruction: {task_description}>>")
    else:
        print("<<No instruction added>>")
        task_description = None


    if args.lm_model_name == 'Salesforce/SFR-Embedding-Mistral':
        generate_sfr_embedding_mistral(text, emb_path, task_description)
    elif args.lm_model_name == 'Alibaba-NLP/gte-Qwen1.5-7B-instruct':
        huggingface_hub.login(os.environ['HF_TOKEN'])
        generate_gte_qwen_7b_instruct(text, emb_path, task_description)
    elif args.lm_model_name == 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp':
        generate_llm2vec_llama3(text, emb_path, task_description)
    else:
        raise ValueError(f"Model {args.lm_model_name} not supported")

    print("Embeddings generated and saved successfully.")

    return

    
if __name__ == '__main__':
    args = parse_args()
    generate_embeddings_and_save(args)
