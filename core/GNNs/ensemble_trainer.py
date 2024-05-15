import torch

from core.GNNs.gnn_trainer import GNNTrainer
from core.GNNs.dgl_gnn_trainer import DGLGNNTrainer
from core.data_utils.load import load_data

LOG_FREQ = 10


class EnsembleTrainer():
    def __init__(self, cfg):
        self.cfg = cfg
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

        if cfg.gnn.model.name == 'RevGAT':
            self.lr = 0.002
            self.dropout = 0.5

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

        if cfg.gnn.model.name == 'RevGAT':
            self.TRAINER = DGLGNNTrainer
        else:
            self.TRAINER = GNNTrainer

    @ torch.no_grad()
    def _evaluate(self, logits):
        val_acc = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc

    @ torch.no_grad()
    def eval(self, logits):
        val_acc, test_acc = self._evaluate(logits)
        print(
            f'({self.feature_type}) ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        return res

    def ensemble_eval(self, all_pred):
        pred_ensemble = sum(all_pred)/len(all_pred)
        acc_ensemble = self.eval(pred_ensemble)
        return acc_ensemble

    def train(self):
        all_pred = []
        all_acc = {}
        feature_types = self.feature_type.split('_')
        model_types = self.gnn_ensemble_models
        for feature_type in feature_types:
            for model_type in model_types:
                self.cfg.gnn.model.name = model_type
                trainer = self.TRAINER(self.cfg, feature_type)
                trainer.train()
                pred, acc = trainer.eval_and_save()
                all_pred.append(pred)
                all_acc[feature_type] = acc
            all_acc[f"{feature_type}_ensemble"] = self.ensemble_eval(all_pred)
        all_acc['ensemble'] = self.ensemble_eval(all_pred)
        return all_acc
