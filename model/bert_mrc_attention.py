#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Author: Xiaoy LI
# Description:
# Bert Model for MRC-Based NER Task


import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from layer.classifier import MultiNonLinearClassifier
from layer.bert_basic_model import BertModel, BertConfig


class BertQueryNER(nn.Module):
    def __init__(self, config):
        super(BertQueryNER, self).__init__()
        bert_config = BertConfig.from_dict(config.bert_config.to_dict())
        self.bert = BertModel(bert_config)

        self.start_outputs = nn.Linear(config.hidden_size, 2)
        self.label_attn = multihead_attention(self.hidden_size, num_heads=config.num_attention_head,
                                              dropout_rate=config.label_dropout)
        self.end_outputs = nn.Linear(config.hidden_size, 2)

        self.span_embedding = MultiNonLinearClassifier(config.hidden_size * 2, 1, config.dropout)
        self.hidden_size = config.hidden_size
        self.bert = self.bert.from_pretrained(config.bert_model)
        self.loss_wb = config.weight_start
        self.loss_we = config.weight_end
        self.loss_ws = config.weight_span

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                start_positions=None, end_positions=None, span_positions=None):
        """
        Args:
            start_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]]
            end_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]]
            span_positions: (batch x max_len x max_len)
                span_positions[k][i][j] is one of [0, 1],
                span_positions[k][i][j] represents whether or not from start_pos{i} to end_pos{j} of the K-th sentence in the batch is an entity.
        """

        sequence_output, pooled_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                                      output_all_encoded_layers=False)

        sequence_heatmap = sequence_output  # batch x seq_len x hidden
        batch_size, seq_len, hid_size = sequence_heatmap.size()

        start_logits = self.start_outputs(sequence_heatmap)  # batch x seq_len x 2
        end_logits = self.end_outputs(sequence_heatmap)  # batch x seq_len x 2

        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # the shape of start_end_concat[0] is : batch x 1 x seq_len x 2*hidden

        span_matrix = torch.cat([start_extend, end_extend], 3)  # batch x seq_len x seq_len x 2*hidden

        span_logits = self.span_embedding(span_matrix)  # batch x seq_len x seq_len x 1
        span_logits = torch.squeeze(span_logits)  # batch x seq_len x seq_len

        if start_positions is not None and end_positions is not None:
            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits.view(-1, 2), start_positions.view(-1))
            end_loss = loss_fct(end_logits.view(-1, 2), end_positions.view(-1))
            span_loss_fct = nn.BCEWithLogitsLoss()
            span_loss = span_loss_fct(span_logits.view(batch_size, -1), span_positions.view(batch_size, -1).float())
            total_loss = self.loss_wb * start_loss + self.loss_we * end_loss + self.loss_ws * span_loss
            return total_loss
        else:
            span_logits = torch.sigmoid(span_logits)  # batch x seq_len x seq_len
            start_logits = torch.argmax(start_logits, dim=-1)
            end_logits = torch.argmax(end_logits, dim=-1)

            return start_logits, end_logits, span_logits

class multihead_attention(nn.Module):

    def __init__(self, num_units, num_heads=1, dropout_rate=0, gpu=True, causality=False):
        '''Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        '''
        super(multihead_attention, self).__init__()
        self.gpu = gpu
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Linear(self.num_units, self.num_units), nn.ReLU()
        self.K_proj = nn.Linear(self.num_units, self.num_units), nn.ReLU()
        self.V_proj = nn.Linear(self.num_units, self.num_units), nn.ReLU()
        if self.gpu:
            self.Q_proj = self.Q_proj.cuda()
            self.K_proj = self.K_proj.cuda()
            self.V_proj = self.V_proj.cuda()


        self.output_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, queries, keys, values,last_layer = False):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]
        # Linear projections
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)
        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)
        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Activation
        if last_layer == False:
            outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)
        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks
        # Dropouts
        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)
        if last_layer == True:
            return outputs
        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)
        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        # Residual connection
        #outputs += queries

        return outputs

