#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : run.py
# @Author: Soleil
# @Date  : 2019-04-04

import sys
sys.path.append('../')
sys.path.append('../../')

import argparse
import sys
from datetime import time

from collections import defaultdict
import os
import numpy as np
import csv
from torch.autograd import Variable

import torch
import torch.optim as optim
import torch.nn.functional as F

import so_labeling.spo_data_reader as reader
import so_labeling.tools as tools
import so_labeling.models as models
from constant import *
import persistence

data_generator = reader.DataReader(
        wordemb_dict_path='../../data/dict/word_idx',
        postag_dict_path='../../data/dict/postag_dict',
        label_dict_path='../../data/dict/label_dict',
        p_eng_dict_path='../../data/dict/p_eng',
        train_data_list_path='../../data/train_data.p',
        test_data_list_path='../../data/dev_data.p'
    )

def main(args):
    args, model, optimizer, params, dicts = init(args)
    #model = model.cuda()
    epochs_trained = train_epoches(args, model, optimizer, params, dicts)


def init(args):
    wordemb_dicts = reader.data_generator.get_dict('wordemb_dict')
    label_dicts = reader.data_generator.get_dict('label_dict')
    print(label_dicts)
    dicts = wordemb_dicts, label_dicts
    model = tools.pick_model(args, dicts)

    if not args.test_model:
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    else:
        optimizer = None
    params = tools.make_param_dict(args)
    return args, model, optimizer, params, dicts


def train_epoches(args, model, optimizer, params, dicts):
    """
            Main loop. does train and test
        """
    metrics_hist = defaultdict(lambda: [])
    metrics_hist_te = defaultdict(lambda: [])
    metrics_hist_tr = defaultdict(lambda: [])

    test_only = args.test_model is not None
    evaluate = args.test_model is not None
    # train for n_epochs unless criterion metric does not improve for [patience] epochs
    for epoch in range(args.n_epochs):
        # only test on train/test set on very last epoch
        if epoch == 0 and not args.test_model:
            model_dir = os.path.join(SO_MODEL_DIR, '_'.join([args.model, time.strftime('%b_%d_%H:%M', time.localtime())]))
            os.mkdir(model_dir)
        elif args.test_model:
            model_dir = os.path.dirname(os.path.abspath(args.test_model))
        metrics_all = one_epoch(model, optimizer, args.Y, epoch, args.n_epochs, args.batch_size, args.data_path,
                                test_only, dicts, model_dir,
                                args.samples, args.gpu, args.quiet)
        for name in metrics_all[0].keys():
            metrics_hist[name].append(metrics_all[0][name])
        for name in metrics_all[1].keys():
            metrics_hist_te[name].append(metrics_all[1][name])
        for name in metrics_all[2].keys():
            metrics_hist_tr[name].append(metrics_all[2][name])
        metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr)

        # save metrics, model, params
        persistence.save_everything(args, metrics_hist_all, model, model_dir, params, args.criterion, evaluate)

        if test_only:
            # we're done
            break

        if args.criterion in metrics_hist.keys():
            if early_stop(metrics_hist, args.criterion, args.patience):
                # stop training, do tests on test and train sets, and then stop the script
                print("%s hasn't improved in %d epochs, early stopping..." % (args.criterion, args.patience))
                test_only = True
                args.test_model = '%s/model_best_%s.pth' % (model_dir, args.criterion)
                model = tools.pick_model(args, dicts)
    return epoch + 1

def one_epoch(model, optimizer, Y, epoch, n_epochs, batch_size, data_path, version, testing, dicts, model_dir,
              samples, gpu, quiet):
    """
        Wrapper to do a training epoch and test on dev
    """
    if not testing:
        losses, unseen_code_inds = train(model, optimizer, Y, epoch, batch_size, data_path, gpu, dicts, quiet)
        loss = np.mean(losses)
        print("epoch loss: " + str(loss))
    else:
        loss = np.nan
        # if model.lmbda > 0:
        #     # still need to get unseen code inds
        #     print("getting set of codes not in training set")
        #     c2ind = dicts['c2ind']
        #     unseen_code_inds = set(dicts['ind2c'].keys())
        #     num_labels = len(dicts['ind2c'])
        #     with open(data_path, 'r') as f:
        #         r = csv.reader(f)
        #         # header
        #         next(r)
        #         for row in r:
        #             unseen_code_inds = unseen_code_inds.difference(
        #                 set([c2ind[c] for c in row[3].split(';') if c != '']))
        #     print("num codes not in train set: %d" % len(unseen_code_inds))
        # else:
        #     unseen_code_inds = set()

    # fold = 'test' if version == 'mimic2' else 'dev'
    fold = 'dev'
    if epoch == n_epochs - 1:
        print("last epoch: testing on test and train sets")
        testing = True
        quiet = False

    # test on dev
    metrics = test(model, Y, epoch, data_path, fold, gpu, version, unseen_code_inds, dicts, samples, model_dir,
                   testing)
    if testing or epoch == n_epochs - 1:
        print("\nevaluating on test")
        metrics_te = test(model, Y, epoch, data_path, "test", gpu, version, unseen_code_inds, dicts, samples,
                          model_dir, True)
    else:
        metrics_te = defaultdict(float)
        fpr_te = defaultdict(lambda: [])
        tpr_te = defaultdict(lambda: [])
    metrics_tr = {'loss': loss}
    metrics_all = (metrics, metrics_te, metrics_tr)
    return metrics_all

def test(args, model, optimizer, params, dicts):
    pass

def early_stop(metrics_hist, criterion, patience):
    if not np.all(np.isnan(metrics_hist[criterion])):
        if criterion == 'loss-dev':
            return np.nanargmin(metrics_hist[criterion]) > len(metrics_hist[criterion]) - patience
        else:
            return np.nanargmax(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
    else:
        #keep training if criterion results have all been nan so far
        return False

def train(model, optimizer, Y, epoch, batch_size, data_path, gpu, dicts):
    print("EPOCH %d" % epoch)

    #######################TODO######################
    num_labels = len(dicts['label_dicts'])

    losses = []
    # how often to print some info to stdout
    print_every = 25
    model.train()

    data_generator = reader.DataReader(
        wordemb_dict_path='../../data/dict/word_idx',
        postag_dict_path='../../data/dict/postag_dict',
        label_dict_path='../../data/dict/label_dict',
        p_eng_dict_path='../../data/dict/p_eng',
        train_data_list_path='../../data/train_data.p',
        test_data_list_path='../../data/dev_data.p'
    )
    gen = data_generator.get_train_reader()

    for batch_idx, features in enumerate(gen()):
        input_sent, word_idx_list, postag_list, label_list = features
        data = word_idx_list, postag_list
        target = label_list
        data, target = Variable(torch.LongTensor(data)), Variable(torch.FloatTensor(target))
        if gpu:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output, loss, _ = model(data, target)

        loss.backward()
        optimizer.step()

        losses.append(loss.data[0])
            # print the average loss of the last 10 batches
        print("Train epoch: {} [batch #{}, batch_size {}, seq length {}]\tLoss: {:.6f}".format(
            epoch, batch_idx, data.size()[0], data.size()[1], np.mean(losses[-10:])))
    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train on relation extraction')
    parser.add_argument("--data_path", type=str, required=False, default="../data/seg_text_comma.txt",
                        help="path to a file containing sorted train data")
    parser.add_argument("vocab", type=str, help="path to a file holding vocab word list for discretizing words")
    parser.add_argument("Y", type=str, help="size of label space")
    parser.add_argument("model", type=str, choices=["cnn_vanilla", "rnn", "conv_attn", "multi_conv_attn", "saved"], help="model")
    parser.add_argument("n_epochs", type=int, help="number of epochs to train")
    parser.add_argument("--embed-file", type=str, required=False, dest="embed_file",
                        help="path to a file holding pre-trained embeddings")
    parser.add_argument("--cell-type", type=str, choices=["lstm", "gru"], help="what kind of RNN to use (default: GRU)", dest='cell_type',
                        default='gru')
    parser.add_argument("--rnn-dim", type=int, required=False, dest="rnn_dim", default=128,
                        help="size of rnn hidden layer (default: 128)")
    parser.add_argument("--bidirectional", dest="bidirectional", action="store_const", required=False, const=True,
                        help="optional flag for rnn to use a bidirectional predicate_multip")
    parser.add_argument("--rnn-layers", type=int, required=False, dest="rnn_layers", default=1,
                        help="number of layers for RNN models (default: 1)")
    parser.add_argument("--embed-size", type=int, required=False, dest="embed_size", default=100,
                        help="size of embedding dimension. (default: 100)")
    parser.add_argument("--filter-size", type=str, required=False, dest="filter_size", default=4,
                        help="size of convolution filter to use. (default: 3) For multi_conv_attn, "
                             "give comma separated integers, e.g. 3,4,5")
    parser.add_argument("--num-filter-maps", type=int, required=False, dest="num_filter_maps", default=50,
                        help="size of conv output (default: 50)")
    parser.add_argument("--pool", choices=['max', 'avg'], required=False, dest="pool", help="which type of pooling to do (logreg predicate_multip only)")
    parser.add_argument("--weight-decay", type=float, required=False, dest="weight_decay", default=0,
                        help="coefficient for penalizing l2 norm of predicate_multip weights (default: 0)")
    parser.add_argument("--lr", type=float, required=False, dest="lr", default=1e-3,
                        help="learning rate for Adam optimizer (default=1e-3)")
    parser.add_argument("--batch-size", type=int, required=False, dest="batch_size", default=16,
                        help="size of training batches")
    parser.add_argument("--dropout", dest="dropout", type=float, required=False, default=0.5,
                        help="optional specification of dropout (default: 0.5)")
    parser.add_argument("--test-predicate_multip", type=str, dest="test_model", required=False, help="path to a saved predicate_multip to load and evaluate")
    parser.add_argument("--criterion", type=str, default='f1_micro', required=False, dest="criterion",
                        help="which metric to use for early stopping (default: f1_micro)")
    parser.add_argument("--patience", type=int, default=3, required=False,
                        help="how many epochs to wait for improved criterion metric before early stopping (default: 3)")
    parser.add_argument("--gpu", dest="gpu", action="store_const", required=False, const=True,
                        help="optional flag to use GPU if available")
    parser.add_argument("--stack-filters", dest="stack_filters", action="store_const", required=False, const=True,
                        help="optional flag for multi_conv_attn to instead use concatenated filter outputs, rather than pooling over them")
    parser.add_argument("--samples", dest="samples", action="store_const", required=False, const=True,
                        help="optional flag to save samples of good / bad predictions")
    parser.add_argument("--quiet", dest="quiet", action="store_const", required=False, const=True,
                        help="optional flag not to print so much during training")
    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)