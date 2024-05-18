import torch
import torch.nn as nn
from typing import *
from core.data_utils.load import load_data
import numpy as np
from core.data_utils.dataset import Dataset
from core.data_utils.dataloader import Collater
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments, Trainer, IntervalStrategy
from transformers.modeling_outputs import TokenClassifierOutput
from core.utils import time_logger, init_path
from core.LMs.utils import *
import gc
import logging
from core.LMs.model import SalesforceEmbeddingMistralClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)



class JointLmmGnnModel(nn.Module):
    # Need an inference version
    def __init__(self, llm_model, prediction_gnn_model, num_classes, loss_func) -> None:
        super(JointLmmGnnModel, self).__init__()
        self.llm_model = llm_model
        self.gnn_model = prediction_gnn_model
        self.num_classes = num_classes
        self.loss_func = loss_func
        self.pred = None

    def forward(self, input_ids, attention_mask=None, labels=None, edge_index=None, adj_t=None, node_id=None):
        _, embedding = self.llm_model(input_ids, attention_mask=attention_mask, return_hidden=True)
        # adj_t vs edge_index resolve later
        # embedding = embedding.float()
        if edge_index is not None:
            logits = self.gnn_model(embedding, edge_index)
        else:
            logits = self.gnn_model(embedding, adj_t)

        if node_id is not None:
            batch_nodes = node_id.cpu().numpy()

        if self.pred is not None:
            # Save prediction to disk (memmap)
            self.pred[batch_nodes] = logits.cpu().float().numpy().astype(np.float16)

        if labels is not None:
            # if labels.shape[-1] == 1:
            #     labels = labels.squeeze()
            loss = self.loss_func(logits.view(-1, self.num_classes), labels.view(-1))
            return TokenClassifierOutput(loss=loss, logits=logits)
        else:
            return dict(logits=logits)


class JointTrainer:
    def __init__(self, cfg) -> None:
        "Joint Trainer."
        # --------- General-------------

        self.dataset_name = cfg.dataset
        self.device = cfg.device
        self.seed = cfg.seed
        self.batch_size = cfg.lm.train.batch_size
        self.epochs = cfg.lm.train.epochs
        self.warmup_epochs = cfg.lm.train.warmup_epochs
        self.lr = cfg.lm.train.lr
        self.eval_patience = cfg.lm.train.eval_patience
        self.grad_acc_steps = cfg.lm.train.grad_acc_steps
        self.weight_decay = cfg.lm.train.weight_decay
        self.max_steps = cfg.lm.train.max_steps
        self.logging_steps = cfg.logging_steps

        # -------- LM Settings --------
        # ! Use LM settings where they are shared, e.g. lm.train.lr, lm.train.epochs applies for both !
        self.lm_model_name = cfg.lm.model.name
        assert self.lm_model_name == 'Salesforce/SFR-Embedding-Mistral'

        self.lm_dropout = cfg.lm.train.dropout
        self.lm_att_dropout = cfg.lm.train.att_dropout
        self.lm_cla_dropout = cfg.lm.train.cla_dropout
        self.lm_output_dir = f'output/{self.dataset_name}/{self.lm_model_name}-seed{self.seed}'
        
        self.tokenizer_max_length = cfg.tokenizer.max_length
        self.use_peft = cfg.use_peft
        self.peft_r = cfg.peft.r
        self.peft_lora_alpha = cfg.peft.lora_alpha
        self.peft_lora_dropout = cfg.peft.lora_dropout
        self.embedding_dim = cfg.embedding_dim

        # ---------- GNN Settings ----------

        self.gnn_model_name = cfg.gnn.model.name
        self.gnn_hidden_dim = cfg.gnn.model.hidden_dim
        self.gnn_num_layers = cfg.gnn.model.num_layers
        self.gnn_dropout = cfg.gnn.train.dropout

        # ---------- Preprocesss -----------

        self.task_description = cfg.lm.task.descriptions
        self.tokenizer = AutoTokenizer.from_pretrained(self.lm_model_name)
        self.task_description = get_task_description(self.dataset_name, task_type=self.task_description)
        self.dataset = preprocess_for_training(
            dataset_name=self.dataset_name,
            tokenizer=self.tokenizer,
            task_description=self.task_description,
            max_length=self.tokenizer_max_length,
            seed=self.seed,
        )
        self.num_nodes = self.dataset.num_nodes
        self.num_classes = self.dataset.num_classes
        self.inf_dataset = self.dataset

        self.train_dataset = self.dataset.train_subset()
        self.val_dataset = self.dataset.val_subset()
        self.test_dataset = self.dataset.test_subset()
        
        # ---------- Init Sub Models -----------
        
        print("Initializing LM and GNN models...")
        self._init_lm()
        self._init_gnn()

        # ---------- Init Joint Model ----------

        self.model = JointLmmGnnModel(
            llm_model=self.lm_model,
            prediction_gnn_model=self.gnn_model,
            num_classes=self.num_classes,
            loss_func=nn.CrossEntropyLoss()
        ) # .to(self.device) # OOM?

        self.model_name = f"joint-{self.lm_model_name}-{self.gnn_model_name}"
        self.output_dir = f'output/{self.dataset_name}/{self.model_name}-seed{self.seed}'
        self.ckpt_dir = f'joint_ckpt/{self.dataset_name}/{self.model_name}-seed{self.seed}'
        
        trainable_params = count_trainable_parameters(self.model)
        print(f"\nNumber of Total (Joint) parameters: {trainable_params}")

    def _init_lm(self) -> None:
        self.lm_model = SalesforceEmbeddingMistralClassifier(
            num_labels=self.num_classes,
            header_dropout_prob=self.lm_cla_dropout,
            output_dir=self.lm_output_dir,
            use_peft=True,
            peft_r=self.peft_r,
            peft_lora_alpha=self.peft_lora_alpha,
            peft_lora_dropout=self.peft_lora_dropout,
        )
        self.lm_model.config.dropout = self.lm_dropout
        self.lm_model.config.attention_dropout = self.lm_att_dropout

        trainable_params = count_trainable_parameters(self.lm_model)
        print(f"\nNumber of LLM parameters: {trainable_params}")

    def _init_gnn(self):
        if self.gnn_model_name == "GCN":
            from core.GNNs.GCN.model import GCN as GNN
        elif self.gnn_model_name == "SAGE":
            from core.GNNs.SAGE.model import SAGE as GNN
        elif self.gnn_model_name == "MLP":
            from core.GNNs.MLP.model import MLP as GNN
        else:
            print(f"Model {self.gnn_model_name} is not supported! Loading MLP ...")
            from core.GNNs.MLP.model import MLP as GNN
        print(f"Loading model {self.gnn_model_name}...")

        self.gnn_model = GNN(
            in_channels=self.embedding_dim,
            hidden_channels=self.gnn_hidden_dim,
            out_channels=self.num_classes,
            num_layers=self.gnn_num_layers,
            dropout=self.gnn_dropout,
            use_pred=False,
        )

        trainable_params = count_trainable_parameters(self.gnn_model)
        print(f"\nNumber of GNN parameters: {trainable_params}")

    @time_logger
    def train(self):
        # Define training parameters
        self.model.train()
        eq_batch_size = self.batch_size * 4
        train_steps = self.max_steps if self.max_steps else self.num_nodes // eq_batch_size + 1
        eval_steps = self.eval_patience // eq_batch_size
        warmup_steps = int(self.warmup_epochs * train_steps)

        print(f"Max steps: {train_steps}, Warmup steps: {warmup_steps}")

        # Define Trainer
        args = TrainingArguments(
            max_steps=train_steps,
            output_dir=self.output_dir,
            do_train=True,
            do_eval=True,
            eval_steps=eval_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_steps=eval_steps,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            save_total_limit=1,
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.grad_acc_steps, 
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size*8,
            warmup_steps=warmup_steps,
            num_train_epochs=self.epochs,
            dataloader_num_workers=1,
            # fp16=True,
            bf16=True,
            dataloader_drop_last=True,
            logging_steps=self.logging_steps,
        )
        self.trainer = CustomTrainer(  # CustomTrainer
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train pre-trained model
        self.trainer.train()
        # print('Saving model to disk...')
        # torch.save(self.model.state_dict(), init_path(f"{self.ckpt_dir}.ckpt"))
        # print(f'Joint Model saved to {self.ckpt_dir}.ckpt')

    @time_logger
    @torch.no_grad()
    def eval_and_save(self):
        pred = np.memmap(init_path(f"{self.ckpt_dir}.pred"),
                dtype=np.float16,
                mode='w+',
                shape=(self.num_nodes, self.num_classes))
    
        self.model.eval()  # this should be enough, if not:
        # self.model.forward = torch.no_grad(self.model.forward)
        self.model.pred = pred

        inference_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.batch_size*8,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            # fp16_full_eval=True,
        )

        trainer = Trainer(model=self.model, args=inference_args)
        print(f"Running inference...")
        prediction_outputs = trainer.predict(self.inf_dataset)
        labels = np.array(self.dataset.labels)
        eval = get_evaluator(self.dataset_name, pred, labels)

        train_acc = eval(self.dataset.train_mask)
        val_acc = eval(self.dataset.val_mask)
        test_acc = eval(self.dataset.test_mask)
        print(
            f'[LM] TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        
        print(f"Saved predictions to {self.ckpt_dir}.pred")
        
        return {'TrainAcc': train_acc, 'ValAcc': val_acc, 'TestAcc': test_acc}



def compute_metrics(p):
    from sklearn.metrics import accuracy_score
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}


@torch.no_grad
def preprocess_for_training(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    task_description: str,
    max_length: int,
    seed: int = None
) -> Dataset:
    clear_cuda_cache()

    data, num_classes, corpus = load_data(
        dataset=dataset_name, use_text=True, use_gpt=False, seed=seed
    )
    logger.info(f"Using instruction <<{task_description}>>")
    prompts = [get_detailed_instruct(task_description, t) for t in corpus]

    X = tokenizer(
        prompts, 
        max_length=max_length, 
        padding=True, 
        truncation=True, 
        return_tensors="pt",
    )

    del data.x
    gc.collect()
    num_nodes = data.y.shape[0]
    print(f"Num nodes: {num_nodes}")

    dataset = Dataset(
        X,
        edge_index=data.get("edge_index"),
        adj_t=data.get("adj_t"),
        labels=data.y.tolist(),
        num_nodes=num_nodes,
        num_classes=num_classes,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
    )
    return dataset


class CustomTrainer(Trainer):
    _exclude_keys = {"edge_index", "adj_t"}

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=Collater(exclude_keys=self._exclude_keys),
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=Collater(exclude_keys=self._exclude_keys),
        )
