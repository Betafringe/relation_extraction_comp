#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : run.py
# @Author: Betafringe
# @Date  : 2019-03-29
# @Desc  : 
# @Contact : betafringe@foxmail.com
import sys
sys.path.append('../')
sys.path.append('../../')

import argparse
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


from multip_data_reader import *
import tools as tools

MODEL_DIR = 'MODEL_CHECKPOINT'
data_generator = RcDataReader(
    wordemb_dict_path='../../data/dict/word_idx',
    postag_dict_path='../../data/dict/postag_dict',
    label_dict_path='../../data/dict/p_eng',
    train_data_list_path='../../data/train_data.json',
    test_data_list_path='../../data//dev_data.json')


def main(args):
    start = time.time()
    args, model, optimizer, params, dicts = init(args)
    if(args.gpu):
        model = model.cuda()
    epochs_trained = train_epoches(args, model, optimizer, params, dicts)
    print("TOTAL ELAPSED TIME FOR %s MODEL AND %d EPOCHS: %f" % (args.model, epochs_trained, time.time() - start))
    print("TOTAL ELAPSED TIME FOR %s MODEL: %f" % (args.model, time.time() - start))


def init(args):
    wordemb_dicts = data_generator.get_dict('wordemb_dict')
    label_dicts = data_generator.get_dict('label_dict')
    dicts = wordemb_dicts, label_dicts
    print(len(dicts))
    model = tools.pick_model(args, dicts)
    print(model)

    if not args.test_model:
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    else:
        optimizer = None
    params = tools.make_param_dict(args)
    return args, model, optimizer, params, dicts


def train_epoches(args, model, optimizer, params, dicts):
    is_train = True
    for epoch in range(args.n_epochs):
        if epoch == 0:
            model_dir = os.path.join(MODEL_DIR, '_'.join([args.model, time.strftime('%b_%d_%H:%M:%S', time.localtime())]))
            os.mkdir(model_dir)
        metrics = one_epoch(model,
                            optimizer,
                            args.Y, epoch,
                            args.n_epochs,
                            args.batch_size,
                            is_train, dicts,
                            args.gpu, model_dir=model_dir)

    return epoch+1


def one_epoch(model, optimizer, Y, epoch, n_epochs, batch_size, is_train, dicts, gpu, model_dir):
    if is_train:
        losses = train(model, optimizer, Y, epoch, n_epochs, batch_size, is_train, dicts, gpu)
        loss = np.mean(losses)
        print("epoch loss:" + str(loss))

    return loss


def test(model, optimizer, Y, epoch, n_epochs, batch_size, is_train, dicts, gpu):
    pass


def train(model, optimizer, Y, epoch, n_epochs, batch_size, is_train, dicts, gpu):
    print("EPOCH %d" % epoch)
    # num_labels = len(dicts[1])
    losses = []
    # how often to print some info to stdout
    print_every = 50
    model.train()
    gen = DataGen(is_train=True, batch_size=args.batch_size)
    for batch_idx, features in enumerate(gen):
        word_idx_list, postag_list, label_list = features
        target = label_list
        data = Variable(torch.LongTensor(word_idx_list)), Variable(torch.LongTensor(postag_list))
        target = Variable(torch.FloatTensor(target))
        if gpu:
            data = data[0].cuda(), data[1].cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output, loss, _ = model(data, target)

        loss.backward()
        optimizer.step()

        losses.append(loss.data[0])
        # print the average loss of the last 10 batches
        if batch_idx % print_every == 0:
            print("Train epoch: {} [batch #{}, batch_size {}, seq length {}]\tLoss: {:.6f}".format(
                epoch, batch_idx, data[0].size()[0], data[1].size()[1], np.mean(losses[-10:])))
    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train on relation extraction')
    parser.add_argument("Y", type=str, help="size of label space")
    parser.add_argument("model", type=str, choices=["cnn_vanilla", "rnn", "conv_attn", "multi_conv_attn", "saved", "conv_rnn"], help="predicate_multip")
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
    parser.add_argument("--word_embed_size", type=int, required=False, dest="word_embed_size", default=200,
                        help="size of embedding dimension. (default: 200)")
    parser.add_argument("--pos-embed-size", type=int, required=False, dest="pos_embed_size", default=10,
                      help="size of embedding dimension. (default: 10)")
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
    parser.add_argument("--batch-size", type=int, required=False, dest="batch_size", default=32,
                        help="size of training batches")
    parser.add_argument("--dropout", dest="dropout", type=float, required=False, default=0.5,
                        help="optional specification of dropout (default: 0.5)")
    parser.add_argument("--lmbda", type=float, required=False, dest="lmbda", default=0,
                        help="hyperparameter to tradeoff BCE loss and similarity embedding loss. defaults to 0")
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