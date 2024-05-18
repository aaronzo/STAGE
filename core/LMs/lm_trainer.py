import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
from core.LMs.model import BertClassifier, BertClaInfModel, SalesforceEmbeddingMistralClassifier
from core.data_utils.dataset import Dataset
from core.data_utils.load import load_data
from core.utils import init_path, time_logger
from core.LMs.utils import get_task_description, get_detailed_instruct, get_evaluator

LLMS = {
    'Salesforce/SFR-Embedding-Mistral': SalesforceEmbeddingMistralClassifier,
    }


def compute_metrics(p):
    from sklearn.metrics import accuracy_score
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}


class LMTrainer():
    def __init__(self, cfg):
        self.dataset_name = cfg.dataset
        self.seed = cfg.seed

        self.model_name = cfg.lm.model.name
        self.feat_shrink = cfg.lm.model.feat_shrink

        self.weight_decay = cfg.lm.train.weight_decay
        self.dropout = cfg.lm.train.dropout
        self.att_dropout = cfg.lm.train.att_dropout
        self.cla_dropout = cfg.lm.train.cla_dropout
        self.batch_size = cfg.lm.train.batch_size
        self.epochs = cfg.lm.train.epochs
        self.warmup_epochs = cfg.lm.train.warmup_epochs
        self.eval_patience = cfg.lm.train.eval_patience
        self.grad_acc_steps = cfg.lm.train.grad_acc_steps
        self.lr = cfg.lm.train.lr
        self.embedding_dim = cfg.embedding_dim
        self.max_steps = cfg.lm.train.max_steps

        self.use_gpt_str = "2" if cfg.lm.train.use_gpt else ""
        self.output_dir = f'output/{self.dataset_name}{self.use_gpt_str}/{self.model_name}-seed{self.seed}'
        self.ckpt_dir = f'prt_lm_finetuned/{self.dataset_name}{self.use_gpt_str}/{self.model_name}-seed{self.seed}'
        self.logging_steps = cfg.logging_steps

        # PEFT settings
        self.use_llm = self.model_name in LLMS
        self.task_descriptions = cfg.lm.task.descriptions
        self.use_peft = cfg.use_peft
        self.peft_r = cfg.peft.r
        self.peft_lora_alpha = cfg.peft.lora_alpha
        self.peft_lora_dropout = cfg.peft.lora_dropout

        # Preprocess data
        data, num_classes, text = load_data(
            dataset=self.dataset_name, use_text=True, use_gpt=cfg.lm.train.use_gpt, seed=self.seed)
        self.data = data
        self.num_nodes = data.y.size(0)
        self.n_labels = num_classes

        if self.use_llm:
            task_description = get_task_description(self.dataset_name, task_type=self.task_descriptions)
            text = [get_detailed_instruct(task_description, t) for t in text]            
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        X = tokenizer(text, padding=True, truncation=True, max_length=512)

        self.dataset = Dataset(X, labels=data.y.tolist())
        self.inf_dataset = self.dataset

        self.train_dataset = torch.utils.data.Subset(
            self.dataset, self.data.train_mask.nonzero().squeeze().tolist())
        self.val_dataset = torch.utils.data.Subset(
            self.dataset, self.data.val_mask.nonzero().squeeze().tolist())
        self.test_dataset = torch.utils.data.Subset(
            self.dataset, self.data.test_mask.nonzero().squeeze().tolist())
        

        # Define pretrained tokenizer and model
        if self.model_name in LLMS:
            self.model = LLMS[self.model_name](
                num_labels=self.n_labels,
                header_dropout_prob=self.cla_dropout,
                output_dir=self.output_dir,
                use_peft=True,
                peft_r=self.peft_r,
                peft_lora_alpha=self.peft_lora_alpha,
                peft_lora_dropout=self.peft_lora_dropout,
            )
        else:
            bert_model = AutoModel.from_pretrained(self.model_name)
            self.model = BertClassifier(bert_model,
                                        n_labels=self.n_labels,
                                        feat_shrink=self.feat_shrink)

        # prev_ckpt = f'prt_lm/{self.dataset_name}/{self.model_name}.ckpt'
        # if self.use_gpt_str and os.path.exists(prev_ckpt):
        #     print("Initialize using previous ckpt...")
        #     self.model.load_state_dict(torch.load(prev_ckpt))

        self.model.config.dropout = self.dropout
        self.model.config.attention_dropout = self.att_dropout

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of trainable parameters: {trainable_params}")

    @time_logger
    def train(self):

        # Define training parameters
        # eq_batch_size = self.batch_size * 4
        train_steps = self.max_steps if self.max_steps else self.num_nodes // self.batch_size
        eval_steps = self.eval_patience // self.batch_size
        warmup_steps = int(self.warmup_epochs * train_steps)

        print(f"Max steps: {train_steps}, Warmup steps: {warmup_steps}")
        print(f"Logging every {self.logging_steps} steps\n")

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
            # fp16=True,    # NOTE: this will cause OOM TODO: learn why?
            bf16=True,      # TODO: learn why this + model loaded in bfloat16 uses less memory than model loaded in 8bit?
            dataloader_drop_last=True,
            logging_steps=self.logging_steps,
        )
        self.trainer = Trainer(
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
        # print(f'\nLM saved to {self.ckpt_dir}.ckpt\n')

    @time_logger
    @torch.no_grad()
    def eval_and_save(self):
        pred = np.memmap(init_path(f"{self.ckpt_dir}.pred"),
                        dtype=np.float16,
                        mode='w+',
                        shape=(self.num_nodes, self.n_labels))
        emb = np.memmap(init_path(f"{self.ckpt_dir}.emb"),
                        dtype=np.float16,
                        mode='w+',
                        shape=(self.num_nodes, self.feat_shrink if self.feat_shrink else self.embedding_dim))

        if not self.use_llm:
            inf_model = BertClaInfModel(
                self.model, emb, pred, feat_shrink=self.feat_shrink)
            
        else:
            self.model.eval()  # this should be enough, if not
            # self.model.forward = torch.no_grad(self.model.forward)
            self.model.emb = emb
            self.model.pred = pred
            inf_model = self.model

        inference_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.batch_size*8,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            fp16_full_eval=True,
        )

        trainer = Trainer(model=inf_model, args=inference_args)
        print(f"Running inference...")
        prediction_output = trainer.predict(self.inf_dataset)
        # pred[:] = prediction_output.predictions
        labels = np.array(self.dataset.labels)
        eval = get_evaluator(self.dataset_name, pred, labels)

        train_acc = eval(self.data.train_mask)
        val_acc = eval(self.data.val_mask)
        test_acc = eval(self.data.test_mask)

        print(
            f'[LM] TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        
        print(f"Saved embeddings to {self.ckpt_dir}.ckpt")
        print(f"Saved predictions to {self.ckpt_dir}.pred")
        return {'TrainAcc': train_acc, 'ValAcc': val_acc, 'TestAcc': test_acc}
