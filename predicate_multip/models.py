#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : models.py
# @Author: Betafringe
# @Date  : 2019-03-29
# @Desc  : 
# @Contact : betafringe@foxmail.com

from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform
from torch.autograd import Variable
import numpy as np

from math import floor
import random
import sys
import time


class BaseModel(nn.Module):

    def __init__(self, Y, embed_file, dicts, lmbda=0, dropout=0.5, gpu=True, word_embed_size=200, pos_embed_size=10):
        super(BaseModel, self).__init__()
        torch.manual_seed(1337)
        self.gpu = gpu
        self.Y = Y
        self.word_embed_size = word_embed_size
        self.pos_embed_size = pos_embed_size
        self.embed_drop = nn.Dropout(p=dropout)
        self.lmbda = lmbda

        # add 2 to include UNK and PAD
        vocab_size = len(dicts[0])
        pos_size = 24
        self.embed_word = nn.Embedding(vocab_size + 2, word_embed_size, padding_idx=0)
        self.embed_pos = nn.Embedding(pos_size, pos_embed_size, padding_idx=0)

    def _get_loss(self, yhat, target, diffs=None):
        # calculate the BCE
        loss = F.binary_cross_entropy_with_logits(yhat, target)

        # add description regularization loss if relevant
        if self.lmbda > 0 and diffs is not None:
            diff = torch.stack(diffs).mean()
            loss = loss + diff
        return loss


class VanillaRNN(BaseModel):
    """
        General RNN - can be LSTM or GRU, uni/bi-directional
    """

    def __init__(self, Y, embed_file, dicts, rnn_dim, cell_type, num_layers, gpu, word_embed_size=200,  bidirectional=False):
        super(VanillaRNN, self).__init__(Y, embed_file, dicts, word_embed_size=word_embed_size, gpu=gpu)
        self.gpu = gpu
        self.rnn_dim = rnn_dim
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # recurrent unit
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(self.word_embed_size + self.pos_embed_size,
                               floor(self.rnn_dim/self.num_directions), self.num_layers, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(self.word_embed_size + self.pos_embed_size,
                              floor(self.rnn_dim/self.num_directions), self.num_layers, bidirectional=bidirectional)
        # linear output
        self.final = nn.Linear(self.rnn_dim, Y)

        # arbitrary initialization
        self.batch_size = 16
        self.hidden = self.init_hidden()

    def forward(self, x, target, desc_data=None, get_attention=False):
        # clear hidden state, reset batch size at the start of each batch
        self.refresh(x[0].size()[0])
        self.refresh(x[1].size()[0])
        # embed
        x_word = x[0]
        x_pos = x[1]

        embeds_word = self.embed_word(x_word).transpose(0, 2)
        embeds_pos = self.embed_pos(x_pos).transpose(0, 2)

        embeds = torch.cat((embeds_word, embeds_pos), 0).transpose(0, 2).transpose(0, 1)
        out, self.hidden = self.rnn(embeds, self.hidden)

        # get final hidden state in the appropriate way
        last_hidden = self.hidden[0] if self.cell_type == 'lstm' else self.hidden
        last_hidden = last_hidden[-1] if self.num_directions == 1 else last_hidden[-2:].transpose(0, 1).contiguous().view(
            self.batch_size, -1)
        # apply linear layer and sigmoid to get predictions
        yhat = F.sigmoid(self.final(last_hidden))
        loss = self._get_loss(yhat, target)
        return yhat, loss, None

    def init_hidden(self):
        if self.gpu:
            h_0 = Variable(torch.cuda.FloatTensor(self.num_directions * self.num_layers, self.batch_size,
                                                  floor(self.rnn_dim / self.num_directions)).zero_())
            if self.cell_type == 'lstm':
                c_0 = Variable(torch.cuda.FloatTensor(self.num_directions * self.num_layers, self.batch_size,
                                                      floor(self.rnn_dim / self.num_directions)).zero_())
                return (h_0, c_0)
            else:
                return h_0
        else:
            h_0 = Variable(torch.zeros(self.num_directions * self.num_layers, self.batch_size,
                                       floor(self.rnn_dim / self.num_directions)))
            if self.cell_type == 'lstm':
                c_0 = Variable(torch.zeros(self.num_directions * self.num_layers, self.batch_size,
                                           floor(self.rnn_dim / self.num_directions)))
                return (h_0, c_0)
            else:
                return h_0

    def refresh(self, batch_size):
        self.batch_size = batch_size
        self.hidden = self.init_hidden()


class Conv_RNN(BaseModel):
    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, rnn_dim, cell_type, num_layers,
                 bidirectional, gpu=True, dicts=None, word_embed_size=200, pos_embed_size=10, dropout=0.2):
        super(Conv_RNN, self).__init__(Y, embed_file, dicts, dropout=dropout, word_embed_size=200, pos_embed_size=10, gpu=gpu)
        # initialize conv layer as in 2.1

        self.conv = nn.Conv1d(word_embed_size+pos_embed_size, 100, kernel_size=kernel_size, padding=floor(kernel_size/2))
        xavier_uniform(self.conv.weight)

        # linear output
        # self.fc = nn.Linear(num_filter_maps, Y)
        # xavier_uniform(self.fc.weight)
        self.gpu = gpu
        self.rnn_dim = rnn_dim
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # recurrent unit
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(128, floor(self.rnn_dim / self.num_directions), self.num_layers,
                               bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(128, floor(self.rnn_dim / self.num_directions), self.num_layers,
                              bidirectional=bidirectional)
        # linear output
        self.final = nn.Linear(self.rnn_dim, Y)

        # arbitrary initialization
        self.batch_size = 32
        self.hidden = self.init_hidden()

    def forward(self, x, target, get_attention=False):
        # embed
        self.refresh(x[0].size()[0])
        self.refresh(x[1].size()[0])
        # embed
        x_word = x[0]
        x_pos = x[1]

        embeds_word = self.embed_word(x_word).transpose(0, 2)
        embeds_pos = self.embed_pos(x_pos).transpose(0, 2)

        embeds = torch.cat((embeds_word, embeds_pos), 0).transpose(0, 2).transpose(0, 1)

        embeds_drop = self.embed_drop(embeds)

        # conv/max-pooling
        c = self.conv(embeds_drop.transpose(1, 2))
        print(c.size())
        self.refresh(c.size()[0])

        cat = torch.cat((c, embeds_pos), 0)
        # print(c.transpose(1,2).transpose(0,1).size())
        # x = F.max_pool1d(F.tanh(c), kernel_size=floor(c.size()[2]/self.batch_size))
        # apply RNN
        out, self.hidden = self.rnn(cat.transpose(1, 2).transpose(0, 1), self.hidden)

        # get final hidden state in the appropriate way
        last_hidden = self.hidden[0] if self.cell_type == 'lstm' else self.hidden
        last_hidden = last_hidden[-1] if self.num_directions == 1 else \
            last_hidden[-2:].transpose(0, 1).contiguous().view(self.batch_size, -1)
        # apply linear layer and sigmoid to get predictions
        yhat = F.sigmoid(self.final(last_hidden))
        loss = self._get_loss(yhat, target)
        return yhat, loss, None

    def init_hidden(self):
        if self.gpu:
            h_0 = Variable(torch.cuda.FloatTensor(self.num_directions*self.num_layers, self.batch_size,
                                                  floor(self.rnn_dim/self.num_directions)).zero_())
            if self.cell_type == 'lstm':
                c_0 = Variable(torch.cuda.FloatTensor(self.num_directions*self.num_layers, self.batch_size,
                                                      floor(self.rnn_dim/self.num_directions)).zero_())
                return (h_0, c_0)
            else:
                return h_0
        else:
            h_0 = Variable(torch.zeros(self.num_directions*self.num_layers, self.batch_size, floor(self.rnn_dim/self.num_directions)))
            if self.cell_type == 'lstm':
                c_0 = Variable(torch.zeros(self.num_directions*self.num_layers, self.batch_size, floor(self.rnn_dim/self.num_directions)))
                return (h_0, c_0)
            else:
                return h_0

    def refresh(self, batch_size):
        self.batch_size = batch_size
        self.hidden = self.init_hidden()

