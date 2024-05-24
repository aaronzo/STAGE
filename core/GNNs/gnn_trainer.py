import torch
from time import time, perf_counter
import numpy as np

from core.GNNs.gnn_utils import EarlyStopping
from core.data_utils.load import load_data, load_gpt_preds
from core.utils import time_logger
from core.utils import partially_initialized
import gc

LOG_FREQ = 10


class GNNTrainer():

    def __init__(self, cfg, feature_type):
        self.results_file = cfg.results_file
        self.INFO = {}

        self.seed = cfg.seed
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn.model.name
        self.lm_model_name = cfg.lm.model.name
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers
        self.dropout = cfg.gnn.train.dropout
        self.lr = cfg.gnn.train.lr
        self.feature_type = feature_type
        self.epochs = cfg.gnn.train.epochs

        self.embedding_dim = cfg.embedding_dim

        self.diffusion = self.diffusion_repr = None
        if self.gnn_model_name in {"SimpleGCN", "SIGN"}:
            self.diffusion = self.gnn_model_name

        # ! Load data
        data, num_classes = load_data(
            self.dataset_name, use_dgl=False, use_text=False, seed=self.seed)

        self.num_nodes = data.y.shape[0]
        self.num_classes = num_classes
        data.y = data.y.squeeze()

        # ! Init gnn feature
        topk = 3 if self.dataset_name == 'pubmed' else 5
        if self.feature_type == 'ogb':
            print("Loading OGB features...")
            features = data.x
        elif self.feature_type == 'TA':
            print("Loading pretrained LM features (title and abstract) ...")
            # NOTE: seed=None only applies to pretrained model embeddings. For fine-tuned models this would be a data leak
            if cfg.gnn.train.use_finetuned_embeddings:
                LM_emb_path = f'prt_lm_finetuned/{self.dataset_name}/Salesforce/SFR-Embedding-Mistral-seed{self.seed}.emb'
                print(f"Using fine-tuned LM embeddings!!")
            elif cfg.lm.task.descriptions !='default':
                LM_emb_path = f"prt_lm/{self.dataset_name}/{self.lm_model_name}-{cfg.lm.task.descriptions}-seedNone-dim{self.embedding_dim}.emb"
            elif cfg.lm.task.descriptions =='default':
                LM_emb_path = f"prt_lm/{self.dataset_name}/{self.lm_model_name}-seedNone-dim{self.embedding_dim}.emb"
            else:
                raise ValueError(f"Invalid LM task: {cfg.lm.task.descriptions}")
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, self.embedding_dim)))
            ).to(torch.float32)
            print(f"Embeddings shape: {features.shape}")
        elif self.feature_type == 'E':
            print("Loading pretrained LM features (explanations) ...")
            # NOTE: seed=None only applies to pretrained model embeddings. For fine-tuned models this would be a data leak
            LM_emb_path = f"prt_lm/{self.dataset_name}2/{self.lm_model_name}-seedNone-dim{self.embedding_dim}.emb" 
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, self.embedding_dim)))
            ).to(torch.float32)
        elif self.feature_type == 'P':
            print("Loading top-k prediction features ...")
            features = load_gpt_preds(self.dataset_name, topk)
        elif self.diffusion is not None:
            self.feature_type = "diffusion"
        else:
            print(
                f'Feature type {self.feature_type} not supported. Loading OGB features...')
            self.feature_type = 'ogb'
            features = data.x

        if self.diffusion is not None:
            if cfg.gnn.diffusion.svd:
                self.embedding_dim = 512
            tag = cfg.gnn.diffusion.tag
            self.INFO["tag"] = tag

            if self.diffusion == "SimpleGCN":
                k = cfg.gnn.diffusion.k
                sgc_path = f"sgc/{self.dataset_name}/{self.lm_model_name}-SGC_{k}-dim{self.embedding_dim}{tag}.emb"
                features = torch.from_numpy(np.array(
                    np.memmap(sgc_path, mode='r',
                            dtype=np.float32,
                            shape=(self.num_nodes, self.embedding_dim)))
                ).to(torch.float32)
                self.diffusion_repr = self.diffusion + f"_{k}"
                   
            elif self.diffusion == "SIGN":
                s, p, t, *_ = [int(c) for c in str(cfg.gnn.diffusion.spt)]
                dim = self.embedding_dim * (s+p+t)
                sym = "sym" if cfg.gnn.diffusion.sym else ""
                sign_path = f"sign/{self.dataset_name}/{self.lm_model_name}-SIGN_{s}{p}{t}{sym}-dim{dim}{tag}.emb"
                features = torch.from_numpy(np.array(
                    np.memmap(sign_path, mode='r',
                            dtype=np.float32,
                            shape=(self.num_nodes, dim)))
                ).to(torch.float32)
                self.diffusion_repr = self.diffusion + f"_{s}{p}{t}{sym}"

        num_features = features.shape[1]
        self.features = features.to(self.device)
        self.data = data.to(self.device)

        if self.diffusion is not None:
            del self.data.edge_index
            gc.collect()

        # ! Trainer init
        use_pred = self.feature_type == 'P'

        if self.diffusion is None:
            if self.gnn_model_name == "GCN":
                from core.GNNs.GCN.model import GCN as GNN
            elif self.gnn_model_name == "SAGE":
                from core.GNNs.SAGE.model import SAGE as GNN
            elif self.gnn_model_name == "MLP":
                from core.GNNs.MLP.model import MLP as GNN
            else:
                raise ValueError(f"Invalid Model provided: {self.gnn_model_name}")
        else:
            if self.diffusion == "SimpleGCN":
                from gnn.simple_gcn import LogisticRegression as GNN
            elif self.diffusion == "SIGN":
                from gnn.sign import InceptionMLP
                GNN = partially_initialized(InceptionMLP, num_operators=s+p+t)
            else:
                raise ValueError(f"Invalid Diffusion provided: {self.diffusion}")


        self.model = GNN(in_channels=self.hidden_dim*topk if use_pred else num_features,
                         hidden_channels=self.hidden_dim,
                         out_channels=self.num_classes,
                         num_layers=self.num_layers,
                         dropout=self.dropout,
                         use_pred=use_pred).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.0)

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f"\nNumber of parameters: {trainable_params}")
        self.ckpt = f"output/{self.dataset_name}/{self.gnn_model_name}.pt"
        self.stopper = EarlyStopping(
            patience=cfg.gnn.train.early_stop, path=self.ckpt) if cfg.gnn.train.early_stop > 0 else None
        self.loss_func = torch.nn.CrossEntropyLoss()

        from core.GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset_name)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )["acc"]

        self.INFO["num_params"] = trainable_params
        self.INFO["dataset"] = self.dataset_name
        self.INFO["gnn"] = self.gnn_model_name if not self.diffusion else self.diffusion_repr
        self.INFO["lm"] = self.lm_model_name
        self.INFO["seed"] = self.seed

    def _forward(self, x, edge_index):
        logits = self.model(x, edge_index)  # small-graph
        return logits

    def _train(self):
        # ! Shared
        self.model.train()
        self.optimizer.zero_grad()
        # ! Specific
        logits = self._forward(self.features, self.data.edge_index)
        loss = self.loss_func(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        train_acc = self.evaluator(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()

        return loss.item(), train_acc

    @ torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        logits = self._forward(self.features, self.data.edge_index)
        val_acc = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc, logits

    @time_logger
    def train(self):
        # ! Training
        start_time = perf_counter()
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            loss, train_acc = self._train()
            val_acc, test_acc, _ = self._evaluate()
            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(
                        f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
            if epoch % LOG_FREQ == 0:
                print(
                    f'Epoch: {epoch}, Time: {time()-t0:.4f}, Loss: {loss:.4f}, TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, ES: {es_str}')

        # ! Finished training, load checkpoints
        if self.stopper is not None:
            self.model.load_state_dict(torch.load(self.stopper.path))

        self.training_time = perf_counter() - start_time
        return self.model

    @ torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model.state_dict(), self.ckpt)
        val_acc, test_acc, logits = self._evaluate()
        print(
            f'GNN [{self.gnn_model_name} + {self.feature_type}] ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        self.INFO["val_acc"] = val_acc
        self.INFO["test_acc"] = test_acc
        self.INFO["training_time"] = self.training_time
        import json
        print(">>>EXPERIMENT SUMMARY<<<:", json.dumps(self.INFO))
        if self.results_file:
            with open(self.results_file, 'a') as f:
                f.write(json.dumps(self.INFO) + "\n")

        return logits, res
