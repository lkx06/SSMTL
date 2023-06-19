import copy
import math
import sys

import torch
from torch import nn, reshape
import torch.nn.functional as F

import torch.nn.functional as F

from transformers import  RobertaModel, AutoConfig

from models.subNets.mm_modeling import BertSelfEncoder,BertCrossEncoder
from models.subNets.modeling import ADDBertEncoder,BertPooler

import logging
logger = logging.getLogger(__name__)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias
    
class BertForSequenceClassification(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained('roberta-base')
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.senti_selfattn = BertSelfEncoder(config, layer_num=1)
        #self.first_pooler = BertPooler(config)
        self.init_weight()

    def init_weight(self):
        ''' bert init
        '''
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)) and ('roberta' not in name ): #linear/embedding
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, BertLayerNorm) and ('roberta' not in name ):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None and ('roberta' not in name ):
                module.bias.data.zero_()

    def forward(self, text):
        input_ids, input_mask = text[:,0,:].long(), text[:,1,:].float()
        roberta_output=self.roberta(input_ids,input_mask)
        sentence_output=roberta_output.last_hidden_state
        '''
        extended_senti_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_senti_mask = extended_senti_mask.to(dtype=next(self.parameters()).dtype)
        extended_senti_mask = (1.0 - extended_senti_mask) * -10000.0
        senti_mixed_output = self.senti_selfattn(sentence_output, extended_senti_mask)  # [N, L+1, 768]
        senti_mixed_output = senti_mixed_output[-1]
        
        senti_comb_img_output = self.first_pooler(senti_mixed_output)
        '''
        text_pooled_output=roberta_output.pooler_output
        pooled_output = self.dropout(text_pooled_output)
        return pooled_output

class VisBert(nn.Module):
   
    def __init__(self,in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        super().__init__()
        config = AutoConfig.from_pretrained('roberta-base')
        self.roberta = RobertaModel.from_pretrained('roberta-base')

        self.img_self_attn = BertSelfEncoder(config, layer_num=1)
        self.ent2img_attention = BertCrossEncoder(config)
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.feat_linear = nn.Linear(2048, 768)
        self.linear_1 = nn.Linear(hidden_size, out_size)
        self.init_weight()

    def init_weight(self):
        ''' bert init
        '''
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)) and ('roberta' not in name ): #linear/embedding
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, BertLayerNorm) and ('roberta' not in name ):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None and ('roberta' not in name ):
                module.bias.data.zero_()

    def forward(self, img_feat,text2,image_mask):
        input_ids, input_mask = text2[:,0,:].long(), text2[:,1,:].float()
        roberta_output=self.roberta(input_ids,input_mask)
        sentence_output=roberta_output.last_hidden_state

        img_feat_ = self.feat_linear(img_feat)  # [N*n, 100, 2048] ->[N*n, 100, 768] 
        #image_mask = torch.ones((32, 100)).to(device)
        #image_mask = torch.ones((batch_size, roi_num)).to(device)
        extended_image_mask = image_mask.unsqueeze(1).unsqueeze(2)
        extended_image_mask = extended_image_mask.to(dtype=next(self.parameters()).dtype)
        extended_image_mask = (1.0 - extended_image_mask) * -10000.0  
        visual_output = self.img_self_attn(img_feat_, extended_image_mask)          #image self atttention
        visual_output = visual_output[-1] # 100*768

        s2_cross_encoder = self.ent2img_attention(sentence_output, visual_output, extended_image_mask)
        s2_cross_output_layer = s2_cross_encoder[-1]
        
        #packed_sequence = pack_padded_sequence(visual_output, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(s2_cross_output_layer)
        h = self.dropout(final_states[0].squeeze())
        
        y_1 = self.linear_1(h)
        
        #pooled_output = self.dropout(final_text_output)

        return y_1,h
    
class BertForSequenceClassification2(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained('roberta-base')
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.senti_selfattn = BertSelfEncoder(config, layer_num=1)
        #self.first_pooler = BertPooler(config)
        self.add_bert_attention = ADDBertEncoder(config)
            # #######self.img_attention_l2 = BertLayer(config)
        self.add_bert_pooler = BertPooler(config)
        self.init_weight()

    def init_weight(self):
        ''' bert init
        '''
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)) and ('roberta' not in name ): #linear/embedding
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, BertLayerNorm) and ('roberta' not in name ):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None and ('roberta' not in name ):
                module.bias.data.zero_()

    def forward(self, text):
        input_ids, input_mask = text[:,0,:].long(), text[:,1,:].float()
        roberta_output=self.roberta(input_ids,input_mask)
        sentence_output=roberta_output.last_hidden_state
        '''
        extended_senti_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_senti_mask = extended_senti_mask.to(dtype=next(self.parameters()).dtype)
        extended_senti_mask = (1.0 - extended_senti_mask) * -10000.0
        senti_mixed_output = self.senti_selfattn(sentence_output, extended_senti_mask)  # [N, L+1, 768]
        senti_mixed_output = senti_mixed_output[-1]
        
        senti_comb_img_output = self.first_pooler(senti_mixed_output)
        '''
        extended_attention_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            #img_att_text_output_layer = self.img_attention(sequence_output, extended_attention_mask)
        add_bert_encoder = self.add_bert_attention(sentence_output, extended_attention_mask)
        add_bert_text_output_layer = add_bert_encoder[-1]
            # ########img_att_text_output_layer_l2 = self.img_attention_l2(img_att_text_output_layer, extended_attention_mask)
            # ########img_att_text_output = self.img_pooler(img_att_text_output_layer_l2)
        text_pooled_output = self.add_bert_pooler(add_bert_text_output_layer)
        #text_pooled_output=roberta_output.pooler_output
        pooled_output = self.dropout(text_pooled_output)
        return pooled_output
