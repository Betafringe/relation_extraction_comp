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
from constant import *

class BaseModel(nn.Module):

    def __init__(self, Y, embed_file, dicts, lmbda=0, dropout=0.5, gpu=True, embed_size=100):
        super(BaseModel, self).__init__()
        torch.manual_seed(1337)
        self.gpu = gpu
        self.Y = Y
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)
        self.lmbda = lmbda

        #make embedding layer
        # if embed_file:
        #     print("loading pretrained embeddings...")
        #     W = torch.Tensor(extract_wvs.load_embeddings(embed_file))
        #
        #     self.embed = nn.Embedding(W.size()[0], W.size()[1])
        #     self.embed.weight.data = W.clone()
        # else:
        #add 2 to include UNK and PAD

        vocab_size = len(dicts[0])
        pos_size = 24
        p_size = 50
        self.embed_word = nn.Embedding(vocab_size + 2, embed_size, padding_idx=0)
        self.embed_pos = nn.Embedding(pos_size, embed_size, padding_idx=0)
        self.embed_pid = nn.Embedding(p_size, embed_size, padding_idx=0)


    def _get_loss(self, yhat, target, diffs=None):
        #calculate the BCE
        loss = F.binary_cross_entropy(yhat, target)

        #add description regularization loss if relevant
        if self.lmbda > 0 and diffs is not None:
            diff = torch.stack(diffs).mean()
            loss = loss + diff
        return loss

    def embed_descriptions(self, desc_data, gpu):
        #label description embedding via convolutional layer
        #number of labels is inconsistent across instances, so have to iterate over the batch
        b_batch = []
        for inst in desc_data:
            if len(inst) > 0:
                if gpu:
                    lt = Variable(torch.cuda.LongTensor(inst))
                else:
                    lt = Variable(torch.LongTensor(inst))
                d = self.desc_embedding(lt)
                d = d.transpose(1,2)
                d = self.label_conv(d)
                d = F.max_pool1d(F.tanh(d), kernel_size=d.size()[2])
                d = d.squeeze(2)
                b_inst = self.label_fc1(d)
                b_batch.append(b_inst)
            else:
                b_batch.append([])
        return b_batch

    def _compare_label_embeddings(self, target, b_batch, desc_data):
        #description regularization loss
        #b is the embedding from description conv
        #iterate over batch because each instance has different # labels
        diffs = []
        for i,bi in enumerate(b_batch):
            ti = target[i]
            inds = torch.nonzero(ti.data).squeeze().cpu().numpy()

            zi = self.final.weight[inds,:]
            diff = (zi - bi).mul(zi - bi).mean()

            #multiply by number of labels to make sure overall mean is balanced with regard to number of labels
            diffs.append(self.lmbda*diff*bi.size()[0])
        return diffs

class VanillaRNN(BaseModel):
    """
        General RNN - can be LSTM or GRU, uni/bi-directional
    """

    def __init__(self, Y, embed_file, dicts, rnn_dim, cell_type, num_layers, gpu, embed_size=200, bidirectional=False):
        super(VanillaRNN, self).__init__(Y, embed_file, dicts, embed_size=embed_size, gpu=gpu)
        self.gpu = gpu
        self.rnn_dim = rnn_dim
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # recurrent unit
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(self.embed_size*2, floor(self.rnn_dim/self.num_directions), self.num_layers, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(self.embed_size*2, floor(self.rnn_dim/self.num_directions), self.num_layers, bidirectional=bidirectional)
        # linear output
        self.final = nn.Linear(self.rnn_dim, Y)

        # arbitrary initialization
        self.batch_size = 16
        self.hidden = self.init_hidden()

    def forward(self, x, target, desc_data=None, get_attention=False):
        # clear hidden state, reset batch size at the start of each batch
        self.refresh(x[0].size()[0])
        self.refresh(x[1].size()[0])
        self.refresh(x[2].size()[0])
        # embed
        x_word = x[0]
        x_pos = x[1]
        x_pid = x[2]

        embeds_word = self.embed_word(x_word).transpose(0, 2)
        embeds_pos = self.embed_pos(x_pos).transpose(0, 2)
        #embeds_pid = self.embed_pid(x_pid).transpose(0, 2)
        print("***********************")
        print(embeds_word.size())
        embeds = torch.cat((embeds_word, embeds_pos), 0).transpose(0, 2).transpose(0,1)

        print(embeds.size())

        out, self.hidden = self.rnn(embeds, self.hidden)

        # get final hidden state in the appropriate way
        last_hidden = self.hidden[0] if self.cell_type == 'lstm' else self.hidden
        last_hidden = last_hidden[-1] if self.num_directions == 1 else last_hidden[-2:].transpose(0,
                                                                                                  1).contiguous().view(
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