# self supervised multimodal multi-task learning network
import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from transformers import BertTokenizer, BertModel
from models.subNets.BertTextEncoder import BertTextEncoder
from models.subNets.robert import BertForSequenceClassification
from models.subNets.robert import VisBert
from models.subNets.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
__all__ = ['SELF_MM']

class SELF_MM(nn.Module):
    def __init__(self, args):
        super(SELF_MM, self).__init__()
        # text subnets
        self.aligned = args.need_data_aligned
        #self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)

        self.text_model = BertForSequenceClassification()
        
        # audio-vision subnets
        audio_in, video_in = args.feature_dims[1:]
        self.audio_model = AuViSubNet(audio_in, args.a_lstm_hidden_size, args.audio_out, \
                            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        #self.video_model = TViBertTextEncoder(video_in, args.v_lstm_hidden_size, args.video_out, \
                            #num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout, language=args.language, use_finetune=args.use_finetune)
        '''
        self.video_model = VisBert.from_pretrained(args.bert_model,
                            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                            in_size=video_in, hidden_size=args.v_lstm_hidden_size, out_size=args.video_out, \
                            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)
        '''
        self.video_model = VisBert(in_size=video_in, hidden_size=args.v_lstm_hidden_size, out_size=args.video_out, \
                            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)
        self.linear1=nn.Linear(1536,768)
        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(args.text_out + args.video_out, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(args.text_out, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)
        self.post_text_layer_3 = nn.Linear(args.post_text_dim, 1)

        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(args.audio_out, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim, 1)

        # the classify layer for video
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(args.video_out, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim, 1)

    def forward(self, text, audio, video):
        #print("开始")
        audio, audio_lengths = audio
        #video, video_lengths = video
        video, text2,image_mask= video
        #text, text2=text


        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach().cpu()
        
        text = self.text_model(text)
        #print('text的维度:')
        #print(text.shape)
        #（32，768）

        if self.aligned:
            audio = self.audio_model(audio, text_lengths)
            video= self.video_model(video, text_lengths)
        else:
            audio= self.audio_model(audio, audio_lengths)
            video,text_= self.video_model(video, text2,image_mask)
        '''
        text=torch.cat([text, text_], dim=-1)
        text = self.post_text_dropout(text)
        text= F.relu(self.linear1(text), inplace=False)
        '''
        # fusion
        fusion_h = torch.cat([text,video], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        # # text
        #text_h = self.post_text_dropout(text)
        text_h = F.relu(self.post_text_layer_1(text), inplace=False)
        # audio
        audio_h = self.post_audio_dropout(audio)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # vision
        video_h = self.post_video_dropout(video)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)

        # classifier-fusion
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)

        # classifier-text
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)

        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)

        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)

        res = {
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
        }
        return res

class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1

class ViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(ViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size+768, out_size)

    def forward(self, x, lengths, text2):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        text_img_output = torch.cat((h, text2), dim=1)
        y_1 = self.linear_1(text_img_output)
        return y_1
class ViBertTextEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False, language='en', use_finetune=False):
        """
        language: en / cn
        """
        super(ViBertTextEncoder, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = BertTokenizer
        model_class = BertModel
        # directory is fine
        # pretrained_weights = '/home/sharing/disk3/pretrained_embedding/Chinese/bert/pytorch'
        if language == 'en':
            self.tokenizer = tokenizer_class.from_pretrained('pretrained_model/bert_en', do_lower_case=True)
            self.model = model_class.from_pretrained('pretrained_model/bert_en')
            self.model2 = model_class.from_pretrained('pretrained_model/bert_en')
        elif language == 'cn':
            self.tokenizer = tokenizer_class.from_pretrained('pretrained_model/bert_cn')
            self.model = model_class.from_pretrained('pretrained_model/bert_cn')
        
        self.use_finetune = use_finetune
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size+768, out_size)

    
    def get_tokenizer(self):
        return self.tokenizer
    
    def from_text(self, text):
        """
        text: raw data
        """
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze()
    
    def forward(self, x, lengths, text2):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text2[:,0,:].long(), text2[:,1,:].float(), text2[:,2,:].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        #print(last_hidden_states.shape)
        #（32，64，768）
        text_out=last_hidden_states[:,0,:]
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        text_img_output = torch.cat((h, text_out), dim=1)
        y_1 = self.linear_1(text_img_output)
        return y_1
class TViBertTextEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False, language='en', use_finetune=False):
        """
        language: en / cn
        """
        super(TViBertTextEncoder, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = BertTokenizer
        model_class = BertModel
        # directory is fine
        # pretrained_weights = '/home/sharing/disk3/pretrained_embedding/Chinese/bert/pytorch'
        if language == 'en':
            self.tokenizer = tokenizer_class.from_pretrained('pretrained_model/bert_en', do_lower_case=True)
            self.model = model_class.from_pretrained('pretrained_model/bert_en')
        elif language == 'cn':
            self.tokenizer = tokenizer_class.from_pretrained('pretrained_model/bert_cn')
            self.model = model_class.from_pretrained('pretrained_model/bert_cn')
        
        self.use_finetune = use_finetune
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size+768, out_size)
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def from_text(self, text):
        """
        text: raw data
        """
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze()
    
    def forward(self, x, text2):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text2[:,0,:].long(), text2[:,1,:].float(), text2[:,2,:].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        #print(last_hidden_states.shape)
        #（32，64，768）
        text_out=last_hidden_states[:,0,:]
        #packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        text_img_output = torch.cat((h, text_out), dim=1)
        y_1 = self.linear_1(text_img_output)
        return y_1,text_out
