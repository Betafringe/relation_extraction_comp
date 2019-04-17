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
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 200
POS_EMBEDDING_DIM = 200
HIDDEN_DIM = 4

class BaseModel(nn.Module):

    def __init__(self, Y, dicts, embed_file = '', lmbda=0, dropout=0.5, gpu=True, embed_size=100, embed_pos = 20):
        super(BaseModel, self).__init__()
        torch.manual_seed(1337)
        self.gpu = gpu
        self.Y = Y
        self.embed_size = embed_size
        self.embed_pos = embed_pos
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
        self.embed_word = nn.Embedding(vocab_size + 2, embed_size)
        self.embed_pos = nn.Embedding(pos_size, embed_pos)
        self.embed_pid = nn.Embedding(p_size, embed_size)


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

    def __init__(self, Y, embed_file, dicts, rnn_dim, cell_type, num_layers, gpu, embed_size=200, embed_pos = 20, bidirectional=False):
        super(VanillaRNN, self).__init__(Y, embed_file, dicts, embed_size=embed_size, gpu=gpu)
        self.gpu = gpu
        self.rnn_dim = rnn_dim
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # recurrent unit
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM((self.embed_size+self.embed_pos)*2, floor(self.rnn_dim/self.num_directions), self.num_layers, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU((self.embed_size+self.embed_pos)*2, floor(self.rnn_dim/self.num_directions), self.num_layers, bidirectional=bidirectional)
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
        embeds_pid = self.embed_pid(x_pid).transpose(0, 2)
        print("***********************")
        print(embeds_word.size())
        embeds = torch.cat((embeds_word, embeds_pos, embeds_pid), 0).transpose(0, 2).transpose(0,1)

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


def to_scalar(var):  # var是Variable,维度是１
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):  # vec是1*5, type是Variable

    max_score = vec[0, argmax(vec)]
    # max_score维度是１，　max_score.view(1,-1)维度是１＊１，max_score.view(1, -1).expand(1, vec.size()[1])的维度是１＊５
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])  # vec.size()维度是1*5
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))  # 为什么指数之后再求和，而后才log呢

# class BiLSTM_CRF(nn.Module):
#
#     def __init__(self, Y, dicts, vocab_size, tag_to_ix, embedding_dim, pos_dim, gpu, hidden_dim):
#         super(BiLSTM_CRF, self).__init__()
#         super(BiLSTM_CRF, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.pos_dim = pos_dim
#         self.hidden_dim = hidden_dim
#         self.vocab_size = len(dicts[0])
#         self.tag_to_ix = tag_to_ix
#         self.tagset_size = len(tag_to_ix)
#         self.gpu = gpu
#
#         #self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
#         pos_size = 24
#         self.embed_word = nn.Embedding(vocab_size + 2, embedding_dim)
#         self.embed_pos = nn.Embedding(pos_size, pos_dim)
#
#         self.lstm = nn.LSTM(embedding_dim+pos_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
#
#         # Maps the output of the LSTM into tag space.
#         self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
#
#         # Matrix of transition parameters.  Entry i,j is the score of
#         # transitioning *to* i *from* j. 居然是随机初始化的！！！！！！！！！！！！！！！之后的使用也是用这随机初始化的值进行操作！！
#         self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
#
#         # These two statements enforce the constraint that we never transfer
#         # to the start tag and we never transfer from the stop tag
#         self.transitions.data[tag_to_ix[START_TAG], :] = -10000
#         self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
#
#         self.hidden = self.init_hidden()
#
#     def init_hidden(self):
#         return (torch.randn(2, 1, self.hidden_dim // 2),
#                 torch.randn(2, 1, self.hidden_dim // 2))
#
#     def _forward_alg(self, feats):
#         init_alphas = torch.cuda.FloatTensor(1, self.tagset_size).fill_(-10000.)  # 1*5 而且全是-10000
#
#         # Do the forward algorithm to compute the partition function
#         init_alphas = torch.full((1, self.tagset_size), -10000.)
#         # START_TAG has all of the score.
#         init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
#
#         # Wrap in a variable so that we will get automatic backprop
#         forward_var = init_alphas
#
#         # Iterate through the sentence
#         for feat in feats:
#             alphas_t = []  # The forward tensors at this timestep
#             for next_tag in range(self.tagset_size):
#                 # broadcast the emission score: it is the same regardless of
#                 # the previous tag
#                 emit_score = feat[next_tag].view(
#                     1, -1).expand(1, self.tagset_size)
#                 # the ith entry of trans_score is the score of transitioning to
#                 # next_tag from i
#                 trans_score = self.transitions[next_tag].view(1, -1)
#                 # The ith entry of next_tag_var is the value for the
#                 # edge (i -> next_tag) before we do log-sum-exp
#                 next_tag_var = forward_var + trans_score + emit_score
#                 # The forward variable for this tag is log-sum-exp of all the
#                 # scores.
#                 alphas_t.append(log_sum_exp(next_tag_var).view(1))
#             forward_var = torch.cat(alphas_t).view(1, -1)
#         terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
#         alpha = log_sum_exp(terminal_var)
#         return alpha
#
#     def _get_lstm_features(self, data):
#         self.hidden = self.init_hidden()
#         embeds_word = self.embed_word(data[0]).view(len(data[0]), 1, -1).transpose(0, 2) #length*1*embedding_size
#         embeds_pos = self.embed_pos(data[1]).view(len(data[1]), 1, -1).transpose(0, 2) #length*1*pos_size
#
#         # embeds_word = self.embed_word(data[0]).transpose(0, 2)
#         # embeds_pos = self.embed_pos(data[1]).transpose(0, 2)
#         embeds = torch.cat((embeds_word, embeds_pos), 0).transpose(0, 2) #length*1*(embedding_size+pos_size)
#
#         lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # length*1*hidden_size
#         lstm_out = lstm_out.view(len(data[0]), self.hidden_dim)  # length*hidden_size
#         lstm_feats = self.hidden2tag(lstm_out)  # length*hidden_size is a linear layer
#         # self.hidden = self.init_hidden()
#         # embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
#         # lstm_out, self.hidden = self.lstm(embeds, self.hidden)
#         # lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
#         # lstm_feats = self.hidden2tag(lstm_out)
#         # return lstm_feats
#         return lstm_feats
#
#
#     def _score_sentence(self, feats, tags):
#         # Gives the score of a provided tag sequence
#         score = torch.zeros(1)
#         tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
#         for i, feat in enumerate(feats):
#             score = score + \
#                     self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
#         score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
#         return score
#
#     def _viterbi_decode(self, feats):
#         backpointers = []
#
#         # Initialize the viterbi variables in log space
#         init_vvars = torch.full((1, self.tagset_size), -10000.)
#         init_vvars[0][self.tag_to_ix[START_TAG]] = 0
#
#         # forward_var at step i holds the viterbi variables for step i-1
#         forward_var = init_vvars
#         for feat in feats:
#             bptrs_t = []  # holds the backpointers for this step
#             viterbivars_t = []  # holds the viterbi variables for this step
#
#             for next_tag in range(self.tagset_size):
#                 # next_tag_var[i] holds the viterbi variable for tag i at the
#                 # previous step, plus the score of transitioning
#                 # from tag i to next_tag.
#                 # We don't include the emission scores here because the max
#                 # does not depend on them (we add them in below)
#                 next_tag_var = forward_var + self.transitions[next_tag]
#                 best_tag_id = argmax(next_tag_var)
#                 bptrs_t.append(best_tag_id)
#                 viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
#             # Now add in the emission scores, and assign forward_var to the set
#             # of viterbi variables we just computed
#             forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
#             backpointers.append(bptrs_t)
#
#         # Transition to STOP_TAG
#         terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
#         best_tag_id = argmax(terminal_var)
#         path_score = terminal_var[0][best_tag_id]
#
#         # Follow the back pointers to decode the best path.
#         best_path = [best_tag_id]
#         for bptrs_t in reversed(backpointers):
#             best_tag_id = bptrs_t[best_tag_id]
#             best_path.append(best_tag_id)
#         # Pop off the start tag (we dont want to return that to the caller)
#         start = best_path.pop()
#         assert start == self.tag_to_ix[START_TAG]  # Sanity check
#         best_path.reverse()
#         return path_score, best_path
#
#     def neg_log_likelihood(self, sentence, tags):
#         feats = self._get_lstm_features(sentence)
#         forward_score = self._forward_alg(feats)
#         gold_score = self._score_sentence(feats, tags)
#         return forward_score - gold_score
#
#     def forward(self, sentence):  # dont confuse this with _forward_alg above.
#         # Get the emission scores from the BiLSTM
#         lstm_feats = self._get_lstm_features(sentence)
#
#         # Find the best path, given the features.
#         score, tag_seq = self._viterbi_decode(lstm_feats)
#         return score, tag_seq
class BiLSTM_CRF(nn.Module):

    def __init__(self, Y, dicts, vocab_size, tag_to_ix, embedding_dim, pos_dim, gpu, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(dicts[0])
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.gpu = gpu

        #self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        pos_size = 24
        self.embed_word = nn.Embedding(vocab_size + 2, embedding_dim)
        self.embed_pos = nn.Embedding(pos_size, pos_dim)

        self.lstm = nn.LSTM(embedding_dim+pos_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j. 居然是随机初始化的！！！！！！！！！！！！！！！之后的使用也是用这随机初始化的值进行操作！！
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.cuda.FloatTensor(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.cuda.FloatTensor(2, 1, self.hidden_dim // 2)))

    def _forward_alg(self, feats):
        init_alphas = torch.cuda.FloatTensor(1, self.tagset_size).fill_(-10000.)  # 1*5 而且全是-10000

        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.  # 因为start tag是4，所以tensor([[-10000., -10000., -10000.,      0., -10000.]])，将start的值为零，表示开始进行网络的传播，
        forward_var = init_alphas
        # # Wrap in a variable so that we will get automatic backprop
        # forward_var = autograd.Variable(torch.cuda.FloatTensor(init_alphas))  # 初始状态的forward_var，随着step t变化
        #
        # # Do the forward algorithm to compute the partition function
        # init_alphas = torch.full((1, self.tagset_size), -10000.)
        # # START_TAG has all of the score.
        # init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        #
        # # Wrap in a variable so that we will get automatic backprop
        # forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, data):
        self.hidden = self.init_hidden()
        embeds_word = self.embed_word(data[0]).view(len(data[0]), 1, -1).transpose(0, 2) #length*1*embedding_size
        embeds_pos = self.embed_pos(data[1]).view(len(data[1]), 1, -1).transpose(0, 2) #length*1*pos_size

        # embeds_word = self.embed_word(data[0]).transpose(0, 2)
        # embeds_pos = self.embed_pos(data[1]).transpose(0, 2)
        embeds = torch.cat((embeds_word, embeds_pos), 0).transpose(0, 2) #length*1*(embedding_size+pos_size)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # length*1*hidden_size
        lstm_out = lstm_out.view(len(data[0]), self.hidden_dim)  # length*hidden_size
        lstm_feats = self.hidden2tag(lstm_out)  # length*hidden_size is a linear layer
        # self.hidden = self.init_hidden()
        # embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        # lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        # lstm_feats = self.hidden2tag(lstm_out)
        # return lstm_feats
        return lstm_feats


    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.cuda.FloatTensor(1)
        #tags = torch.cat([torch.cuda.FloatTensor([self.tag_to_ix[START_TAG] for _ in range(16)]),tags])  # 将START_TAG的标签３拼接到tag序列最前面，这样tag就是12个了
        #
        tags = torch.cat([torch.cuda.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


# class BiLSTM_CRF(nn.Module):
#     def __init__(self, Y, dicts, vocab_size, tag_to_ix, embedding_dim, pos_dim, gpu, hidden_dim, batch_size=16):
#         super(BiLSTM_CRF, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.pos_dim = pos_dim
#         self.hidden_dim = hidden_dim
#         self.vocab_size = len(dicts[0])
#         self.tag_to_ix = tag_to_ix
#         self.tagset_size = len(tag_to_ix)
#         self.gpu = gpu
#         self.batch_size = batch_size
#
#         #self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
#         pos_size = 24
#         self.embed_word = nn.Embedding(vocab_size + 2, embedding_dim, padding_idx=0)
#         self.embed_pos = nn.Embedding(pos_size, pos_dim, padding_idx=0)
#
#         self.lstm = nn.LSTM(embedding_dim+pos_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
#
#         # Maps the output of the LSTM into tag space.
#         self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
#
#         # Matrix of transition parameters.  Entry i,j is the score of
#         # transitioning *to* i *from* j. 居然是随机初始化的！！！！！！！！！！！！！！！之后的使用也是用这随机初始化的值进行操作！！
#         self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
#
#         # These two statements enforce the constraint that we never transfer
#         # to the start tag and we never transfer from the stop tag
#         self.transitions.data[tag_to_ix[START_TAG], :] = -10000
#         self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
#
#         self.hidden = self.init_hidden()
#
#     def init_hidden(self):
#         if self.gpu:
#             h_0 = autograd.Variable(torch.cuda.FloatTensor(2, self.batch_size,
#                                                   floor(self.hidden_dim / 2)).zero_())
#             c_0 = autograd.Variable(torch.cuda.FloatTensor(2, self.batch_size,
#                                                   floor(self.hidden_dim / 2)).zero_())
#             return (h_0, c_0)
#         else:
#             h_0 = autograd.Variable(torch.randn(self.num_directions * self.num_layers, self.batch_size,
#                                        floor(self.hidden_dim / self.num_directions)))
#             c_0 = autograd.Variable(torch.randn(self.num_directions * self.num_layers, self.batch_size,
#                                        floor(self.hidden_dim / self.num_directions)))
#             return (h_0, c_0)
#         # return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
#         #         autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))
#
#     # 预测序列的得分
#     def _forward_alg(self, feats):
#         # Do the forward algorithm to compute the partition function
#         init_alphas = torch.cuda.FloatTensor(1, self.tagset_size).fill_(-10000.)  # 1*5 而且全是-10000
#
#         # START_TAG has all of the score.
#         init_alphas[0][self.tag_to_ix[
#             START_TAG]] = 0.  # 因为start tag是4，所以tensor([[-10000., -10000., -10000.,      0., -10000.]])，将start的值为零，表示开始进行网络的传播，
#
#         # Wrap in a variable so that we will get automatic backprop
#         forward_var = autograd.Variable(torch.cuda.FloatTensor(init_alphas))  # 初始状态的forward_var，随着step t变化
#
#         # Iterate through the sentence 会迭代feats的行数次，
#         for feat in feats:  # feat的维度是５ 依次把每一行取出来~
#             alphas_t = []  # The forward variables at this timestep
#             for next_tag in range(self.tagset_size):  # next tag 就是简单 i，从0到len
#                 # broadcast the emission score: it is the same regardless of
#                 # the previous tag
#                 emit_score = feat[next_tag].view(1, -1).expand(1,
#                                                                self.tagset_size)  # 维度是1*5 噢噢！原来，LSTM后的那个矩阵，就被当做是emit score了
#
#                 # the ith entry of trans_score is the score of transitioning to
#                 # next_tag from i
#                 trans_score = self.transitions[next_tag].view(1, -1)  # 维度是１＊５
#                 # The ith entry of next_tag_var is the value for the
#                 # edge (i -> next_tag) before we do log-sum-exp
#                 # 第一次迭代时理解：
#                 # trans_score所有其他标签到Ｂ标签的概率
#                 # 由lstm运行进入隐层再到输出层得到标签Ｂ的概率，emit_score维度是１＊５，5个值是相同的
#
#                 next_tag_var = forward_var + trans_score + emit_score
#                 # The forward variable for this tag is log-sum-exp of all the
#                 # scores.
#                 alphas_t.append(log_sum_exp(next_tag_var).unsqueeze(0))
#             # 此时的alphas t 是一个长度为5，例如<class 'list'>: [tensor(0.8259), tensor(2.1739), tensor(1.3526), tensor(-9999.7168), tensor(-0.7102)]
#             forward_var = torch.cat(alphas_t).view(1, -1)  # 到第（t-1）step时５个标签的各自分数
#         terminal_var = forward_var + self.transitions[self.tag_to_ix[
#             STOP_TAG]]  # 最后只将最后一个单词的forward var与转移 stop tag的概率相加 tensor([[   21.1036,    18.8673,    20.7906, -9982.2734, -9980.3135]])
#         alpha = log_sum_exp(terminal_var)  # alpha是一个0维的tensor
#
#         return alpha
#
#     # 得到feats
#     def _get_lstm_features(self, data):
#         self.hidden = self.init_hidden()
#         # embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
#
#         embeds_word = self.embed_word(data[0]).transpose(0, 2)
#         embeds_pos = self.embed_pos(data[1]).transpose(0, 2)
#         embeds = torch.cat((embeds_word, embeds_pos), 0).transpose(0, 2).transpose(0,1)
#
#         lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # 11*1*4
#         #lstm_out = lstm_out.view(len(data[0]), self.hidden_dim)  # 11*4
#
#         lstm_feats = self.hidden2tag(lstm_out)  # 11*5 is a linear layer
#
#         return lstm_feats
#
#     # 得到gold_seq tag的score 即根据真实的label 来计算一个score，但是因为转移矩阵是随机生成的，故算出来的score不是最理想的值
#     def _score_sentence(self, feats, tags):
#         # Gives the score of a provided tag sequence #feats 11*5  tag 11 维
#         score = autograd.Variable(torch.FloatTensor([0]))
#         print(self.tag_to_ix[START_TAG])
#         print(tags)
#         print(tags.size())
#         tags = torch.cat([torch.cuda.FloatTensor([self.tag_to_ix[START_TAG] for _ in range(16)]), tags])  # 将START_TAG的标签３拼接到tag序列最前面，这样tag就是12个了
#
#         res = []
#         for b in range(0, self.batch_size):
#             for i, feat in enumerate(feats):
#                 # self.transitions[tags[i + 1], tags[i]] 实际得到的是从标签i到标签i+1的转移概率
#                 # feat[tags[i+1]], feat是step i 的输出结果，有５个值，对应B, I, E, START_TAG, END_TAG, 取对应标签的值
#                 # transition【j,i】 就是从i ->j 的转移概率值
#                 score = score + self.transitions[tags[b][i + 1], tags[i]] + feat[tags[b][i + 1]]
#             score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[b][-1]]
#             res.append(score)
#         return res
#
#     # 解码，得到预测的序列，以及预测序列的得分
#     def _viterbi_decode(self, feats):
#         backpointers = []
#
#         # Initialize the viterbi variables in log space
#         init_vvars = torch.cuda.FloatTensor(1, self.tagset_size).fill_(-10000.)
#         init_vvars[0][self.tag_to_ix[START_TAG]] = 0
#
#         # forward_var at step i holds the viterbi variables for step i-1
#         forward_var = autograd.Variable(init_vvars)
#         for feat in feats:
#             bptrs_t = []  # holds the backpointers for this step
#             viterbivars_t = []  # holds the viterbi variables for this step
#
#             for next_tag in range(self.tagset_size):
#                 # next_tag_var[i] holds the viterbi variable for tag i at the
#                 # previous step, plus the score of transitioning
#                 # from tag i to next_tag.
#                 # We don't include the emission scores here because the max
#                 # does not depend on them (we add them in below)
#                 next_tag_var = forward_var + self.transitions[next_tag]  # 其他标签（B,I,E,Start,End）到标签next_tag的概率
#                 best_tag_id = argmax(next_tag_var)
#                 bptrs_t.append(best_tag_id)
#                 viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
#             # Now add in the emission scores, and assign forward_var to the set
#             # of viterbi variables we just computed
#             forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)  # 从step0到step(i-1)时5个序列中每个序列的最大score
#             backpointers.append(bptrs_t)  # bptrs_t有５个元素
#
#         # Transition to STOP_TAG
#         terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]  # 其他标签到STOP_TAG的转移概率
#         best_tag_id = argmax(terminal_var)
#         path_score = terminal_var[0][best_tag_id]
#
#         # Follow the back pointers to decode the best path.
#         best_path = [best_tag_id]
#         for bptrs_t in reversed(backpointers):  # 从后向前走，找到一个best路径
#             best_tag_id = bptrs_t[best_tag_id]
#             best_path.append(best_tag_id)
#         # Pop off the start tag (we dont want to return that to the caller)
#         start = best_path.pop()
#         assert start == self.tag_to_ix[START_TAG]  # Sanity check
#         best_path.reverse()  # 把从后向前的路径正过来
#         return path_score, best_path
#
#     def neg_log_likelihood(self, sentence, tags):
#         print(sentence[0].size())
#         feats = self._get_lstm_features(sentence)  # 11*5 经过了LSTM+Linear矩阵后的输出，之后作为CRF的输入。
#         forward_score = self._forward_alg(feats)  # 0维的一个得分，20.*来着
#         gold_score = self._score_sentence(feats, tags)  # tensor([ 4.5836])
#         print(gold_score)
#         loss = self._get_loss(forward_score, gold_score)
#         return  forward_score - gold_score
#
#     def forward(self, sentence):  # dont confuse this with _forward_alg above.
#         # Get the emission scores from the BiLSTM
#         lstm_feats = self._get_lstm_features(sentence)
#
#         # Find the best path, given the features.
#         score, tag_seq = self._viterbi_decode(lstm_feats)
#         return score, tag_seq