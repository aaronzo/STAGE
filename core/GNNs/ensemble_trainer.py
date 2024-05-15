import torch
from copy import deepcopy

from core.GNNs.gnn_trainer import GNNTrainer
from core.GNNs.dgl_gnn_trainer import DGLGNNTrainer
from core.data_utils.load import load_data

LOG_FREQ = 10


class EnsembleTrainer():
    def __init__(self, cfg):
        self.cfg = cfg         
        self.original_cfg = deepcopy(self.cfg)  # Make a deep copy of the original configuration
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn.model.name
        self.gnn_ensemble_models = cfg.gnn.ensemble_models
        self.lm_model_name = cfg.lm.model.name
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers

        self.dropout = cfg.gnn.train.dropout
        self.lr = cfg.gnn.train.lr
        self.feature_type = cfg.gnn.train.feature_type
        self.epochs = cfg.gnn.train.epochs
        self.weight_decay = cfg.gnn.train.weight_decay

        # ! Load data
        data, _ = load_data(self.dataset_name, use_dgl=False, use_text=False, seed=cfg.seed)
        print(f"loaded dataset: {self.dataset_name}")

        data.y = data.y.squeeze()
        self.data = data.to(self.device)

        from core.GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset_name)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )["acc"]


    @ torch.no_grad()
    def _evaluate(self, logits):
        val_acc = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc

    @ torch.no_grad()
    def eval(self, logits, feature_type):
        val_acc, test_acc = self._evaluate(logits)
        print(
            f'({feature_type}) ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        return res

    def ensemble_eval(self, all_pred, feature_type):
        pred_ensemble = sum(all_pred)/len(all_pred)
        acc_ensemble = self.eval(pred_ensemble, feature_type)
        return acc_ensemble

    def choose_trainer(self, model_type):
        if model_type == 'RevGAT':
            print(f"{model_type} model chosen -- using DGLGNNTrainer...")
            self.cfg.lr = 0.002
            self.cfg.dropout = 0.5
            self.TRAINER = DGLGNNTrainer
        else:
            self.cfg = deepcopy(self.original_cfg)  # Restore the original configuration
            self.TRAINER = GNNTrainer
        
        self.cfg.gnn.model.name = model_type

    def train(self):
        all_pred = []
        all_acc = {}
        feature_types = self.feature_type.split('_')
        model_types = self.gnn_ensemble_models
        print(f"Training ensemble with models: {model_types}")
        for feature_type in feature_types:
            feature_pred = []
            for model_type in model_types:
                print(f"\n\nTraining model: {model_type} with feature type: {feature_type}\n\n")
                self.choose_trainer(model_type)
                self.cfg.gnn.train.feature_type = feature_type
                print(f"Config used: \n{self.cfg.gnn}\ntrainer: {self.TRAINER}\n")
                trainer = self.TRAINER(self.cfg, feature_type)
                trainer.train()
                pred, acc = trainer.eval_and_save()
                all_pred.append(pred)
                feature_pred.append(pred)
                # all_acc[feature_type] = acc
            all_acc[f"{feature_type}_ensemble"] = self.ensemble_eval(feature_pred, feature_type=f"{feature_type}_ensemble")
        all_acc[f'{self.feature_type}_ensemble'] = self.ensemble_eval(all_pred, feature_type=f'{self.feature_type}_ensemble')
        return all_acc
