import sys

sys.path += ['../']
import torch
from torch import nn
import numpy as np
from transformers import (RobertaConfig, RobertaModel,
                          RobertaForSequenceClassification, RobertaTokenizer,
                          T5Tokenizer, T5Model, T5ForConditionalGeneration, T5Config, T5EncoderModel,
                          GPT2Config, GPT2Model, GPT2Tokenizer, GPT2LMHeadModel,
                          BertModel, BertTokenizer, BertConfig, BertForSequenceClassification, BertForTokenClassification,
                          DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer,
                          DPRContextEncoder, DPRQuestionEncoder)

import torch.nn.functional as F
from IPython import embed
import time


# ANCE model
class ANCE(RobertaForSequenceClassification):
    def __init__(self, config):
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768) # ANCE has
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)
        self.use_mean = False
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        outputs1 = outputs1.last_hidden_state
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def masked_mean_or_first(self, emb_all, mask):
        if self.use_mean:
            return self.masked_mean(emb_all, mask)
        else:
            return emb_all[:, 0]
    
    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    
    def forward(self, input_ids, attention_mask, wrap_pooler=False):
        return self.query_emb(input_ids, attention_mask)
