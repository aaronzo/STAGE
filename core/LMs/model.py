import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedModel, AutoConfig, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput
from core.utils import init_random_state
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import logging as transformers_logging
import logging

logger = logging.getLogger(__name__)

transformers_logging.set_verbosity_error()


class SentenceClsHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, (config.hidden_size // 2))
        classifier_dropout = (
            config.header_dropout_prob if config.header_dropout_prob is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear((config.hidden_size // 2), config.num_labels)

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
            peft_r=None,
            peft_lora_alpha=None,
            peft_lora_dropout=None
        ):
        super(SalesforceEmbeddingMistralClassifier, self).__init__()
        pretrained_repo = 'Salesforce/SFR-Embedding-Mistral'
        transformers_logging.set_verbosity_error()
        logger.warning(f"inherit model weights from {pretrained_repo}")
        self.config = AutoConfig.from_pretrained(pretrained_repo)
        self.config.num_labels = num_labels
        self.config.header_dropout_prob = header_dropout_prob
        self.config.save_pretrained(save_directory=output_dir)
        self.emb = None
        self.pred = None
        # init modules
        self.head = SentenceClsHead(self.config)

        self.model = AutoModel.from_pretrained(
            pretrained_repo, 
            config=self.config, 
            device_map='auto', 
            # load_in_8bit=True,
            torch_dtype=torch.bfloat16
        )
        logger.info(f"Model dtype --> {self.model.dtype}")
        if use_peft:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,  # https://github.com/huggingface/peft/blob/v0.8.2/src/peft/utils/peft_types.py#L68-L73
                inference_mode=False,
                r=peft_r,
                lora_alpha=peft_lora_alpha,
                lora_dropout=peft_lora_dropout,
            )
            logger.info('Initialising PEFT Model...')
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of parameters in `self.model`: {trainable_params}")

        trainable_params = sum(p.numel()
                               for p in self.head.parameters() if p.requires_grad)
        print(f"\nNumber of trainable parameters in `self.head`: {trainable_params}")

        self.loss_func = nn.CrossEntropyLoss(
            label_smoothing=0.3, reduction='mean')



    def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(self, input_ids, attention_mask=None, labels=None, return_hidden=False, preds=None, node_id=None):
        base_out = self.model(input_ids=input_ids, attention_mask=attention_mask)  # torch.Size([9, 512, 4096])
        # print(f"base_out.last_hidden_state.shape --> {base_out.last_hidden_state.shape}")
        sentence_embeddings = self.average_pool(base_out.last_hidden_state, attention_mask)  # torch.Size([9, 4096])
        # print(f"sentence_embeddings.shape --> {sentence_embeddings.shape}")
        logits = self.head(sentence_embeddings)  # torch.Size([9, 7])
        # print(f"logits.shape --> {logits.shape}")

        if node_id is not None:
            batch_nodes = node_id.cpu().numpy()

        if self.emb is not None:
            # Save embeddings to disk (memmap)
            # upcast `bfloat16` tensor using `.float()` -> https://stackoverflow.com/questions/78128662/converting-pytorch-bfloat16-tensors-to-numpy-throws-typeerror
            self.emb[batch_nodes] = sentence_embeddings.cpu().float().numpy().astype(np.float16)
            
        if self.pred is not None:
            # Save prediction to disk (memmap)
            self.pred[batch_nodes] = logits.cpu().float().numpy().astype(np.float16)

        if labels is not None:
            if labels.shape[-1] == 1:
                labels = labels.squeeze()
            loss = self.loss_func(logits, labels)
        else:
            loss = None

        output = TokenClassifierOutput(loss=loss, logits=logits)

        if return_hidden:
            return output, sentence_embeddings

        return output


    @torch.no_grad()
    def emb(self, input_ids, attention_mask=None):
        base_out = self.model(input_ids=input_ids, attention_mask=attention_mask)  # torch.Size([9, 512, 4096])
        # print(f"base_out.last_hidden_state.shape --> {base_out.last_hidden_state.shape}")
        sentence_embeddings = self.average_pool(base_out.last_hidden_state, attention_mask)  # torch.Size([9, 4096])
        return sentence_embeddings.detach().cpu().numpy()


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
        emb = self.dropout(outputs['hidden_states'][-1])  # torch.Size([9, 512, 768])
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]  # torch.Size([9, 768])
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        logits = self.classifier(cls_token_emb) # torch.Size([9, 40])

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
