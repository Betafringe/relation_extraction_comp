#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : tools.py
# @Author: Betafringe
# @Date  : 2019-04-02
# @Desc  :
# @Contact : betafringe@foxmail.com


import csv
import json
import math
import os
import pickle

import torch
from torch.autograd import Variable

import so_labeling.models as models
import numpy as np


def pick_model(args, dicts):
    """
        Use args to initialize the appropriate model
    """
    Y = 8
    if args.model == "rnn":
        model = models.VanillaRNN(Y, args.embed_file, dicts, args.rnn_dim, args.cell_type, args.rnn_layers, args.gpu, args.embed_size,
                                  args.bidirectional)
    elif args.model == "cnn_vanilla":
        filter_size = int(args.filter_size)
        model = models.VanillaConv(Y, args.embed_file, filter_size, args.num_filter_maps, args.gpu, dicts, args.embed_size, args.dropout)
    elif args.model == "conv_attn":
        filter_size = int(args.filter_size)
        model = models.ConvAttnPool(Y, args.embed_file, filter_size, args.num_filter_maps, args.lmbda, args.gpu, dicts,
                                    embed_size=args.embed_size, dropout=args.dropout, multi=args.multi, multi_times=args.multi_times)
    elif args.model == "conv_mhattn":
        filter_size = int(args.filter_size)
        model = models.ConvMultiHeadAttn(Y, args.embed_file, filter_size, args.num_filter_maps, args.lmbda, args.gpu, dicts,
                                    embed_size=args.embed_size, dropout=args.dropout)
    elif args.model == "conv_lstm":
        filter_size = int(args.filter_size)
        model = models.conv_LSTM(Y, args.embed_file, dicts, args.input_size, args.input_dim, args.hidden_dim, args.kernel_size, args.num_layers,
                            args.gpu, args.dropout, args.embed_size, args.batch_first, args.bias, args.return_all_layers)
    elif args.model == "conv_rnn":
        filter_size = int(args.filter_size)
        model = models.Conv_RNN(Y, args.embed_file, filter_size, args.num_filter_maps, args.rnn_dim, args.cell_type, args.rnn_layers,
                            args.bidirectional, args.gpu, dicts, args.embed_size, args.dropout)
    elif args.model == "bilstm_crf":
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        with open('../../data/dict/word_idx') as f:
            size = len(f.readlines())
        model = models.BiLSTM_CRF(Y, dicts, len(dicts[0]), {"B - SUB": 0, "I - SUB": 1, "E - SUB": 2, "B - OBJ": 3, "I - OBJ": 4, "E - OBJ": 5, "O": 6, START_TAG: 7, STOP_TAG: 8}, embedding_dim=200, pos_dim=20,gpu=args.gpu, hidden_dim=128)

    if args.test_model:
        sd = torch.load(args.test_model)
        model.load_state_dict(sd)
    if args.gpu:
        model.cuda()
    return model


def make_param_dict(args):
    """
        Make a list of parameters to save for future reference
    """
    param_vals = [args.Y, args.filter_size, args.dropout, args.num_filter_maps, args.rnn_dim, args.cell_type, args.rnn_layers,
                  args.command, args.weight_decay, args.data_path, args.embed_file, args.lr]
    param_names = ["Y", "filter_size", "dropout", "num_filter_maps", "rnn_dim", "cell_type", "rnn_layers", "command",
                   "weight_decay", "data_path", "embed_file", "lr"]
    params = {name: val for name, val in zip(param_names, param_vals) if val is not None}
    return params