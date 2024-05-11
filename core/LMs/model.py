import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedModel, AutoConfig, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput
from core.utils import init_random_state
from peft import LoraConfig, PeftModel, TaskType
from transformers import logging as transformers_logging
import logging

logger = logging.getLogger(__name__)

transformers_logging.set_verbosity_error()


class SentenceClsHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.header_dropout_prob if config.header_dropout_prob is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, feature):
        x = self.dropout(feature)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    

class SalesforceEmbeddingMistralClassifier(nn.Module):
    def __init__(
            self, 
            num_labels,
            header_dropout_prob,
            output_dir,
            use_peft,
            peft_r,
            peft_lora_alpha,
            peft_lora_dropout
        ):
        super(SalesforceEmbeddingMistralClassifier, self).__init__()
        pretrained_repo = 'Salesforce/SFR-Embedding-Mistral'
        transformers_logging.set_verbosity_error()
        logger.warning(f"inherit model weights from {pretrained_repo}")
        config = AutoConfig.from_pretrained(pretrained_repo)
        config.num_labels = num_labels
        config.header_dropout_prob = header_dropout_prob
        config.save_pretrained(save_directory=output_dir)
        # init modules
        self.bert_model = AutoModel.from_pretrained(pretrained_repo, config=config, add_pooling_layer=False)
        self.head = SentenceClsHead(config)
        if use_peft:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=peft_r,
                lora_alpha=peft_lora_alpha,
                lora_dropout=peft_lora_dropout,
            )
            self.bert_model = PeftModel(self.bert_model, lora_config)
            self.bert_model.print_trainable_parameters()

    def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def forward(self, input_ids, att_mask, labels=None, return_hidden=False):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        sentence_embeddings = self.average_pool(bert_out.last_hidden_state, att_mask)
        out = self.head(sentence_embeddings)

        if return_hidden:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            return out, sentence_embeddings
        else:
            return out
        

class BertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, dropout=0.0, seed=0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        self.loss_func = nn.CrossEntropyLoss(
            label_smoothing=0.3, reduction='mean')

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        init_random_state(seed)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                preds=None):

        outputs = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=return_dict,
                                    output_hidden_states=True)
        # outputs[0]=last hidden state
        emb = self.dropout(outputs['hidden_states'][-1])
        import ipdb;ipdb.set_trace()
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        logits = self.classifier(cls_token_emb)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits)


class BertClaInfModel(PreTrainedModel):
    def __init__(self, model, emb, pred, feat_shrink=''):
        super().__init__(model.config)
        self.bert_classifier = model
        self.emb, self.pred = emb, pred
        self.feat_shrink = feat_shrink
        self.loss_func = nn.CrossEntropyLoss(
            label_smoothing=0.3, reduction='mean')

    @torch.no_grad()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                node_id=None):

        # Extract outputs from the model
        bert_outputs = self.bert_classifier.bert_encoder(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         return_dict=return_dict,
                                                         output_hidden_states=True)
        # outputs[0]=last hidden state
        emb = bert_outputs['hidden_states'][-1]
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.bert_classifier.feat_shrink_layer(
                cls_token_emb)
        logits = self.bert_classifier.classifier(cls_token_emb)

        # Save prediction and embeddings to disk (memmap)
        batch_nodes = node_id.cpu().numpy()
        self.emb[batch_nodes] = cls_token_emb.cpu().numpy().astype(np.float16)
        self.pred[batch_nodes] = logits.cpu().numpy().astype(np.float16)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits)
