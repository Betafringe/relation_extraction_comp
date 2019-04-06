#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : multip_data_reader.py
# @Author: Betafringe
# @Date  : 2019-03-29
# @Desc  : 
# @Contact : betafringe@foxmail.com
import os
import numpy as np
import json

import json
import os
import sys
import torch
sys.getdefaultencoding()
'utf-8'


class RcDataReader(object):
    """
    class for p classfication data reader
    """

    def __init__(self,
                 wordemb_dict_path,
                 postag_dict_path,
                 label_dict_path,
                 train_data_list_path='',
                 test_data_list_path=''):
        self._wordemb_dict_path = wordemb_dict_path
        self._postag_dict_path = postag_dict_path
        self._label_dict_path = label_dict_path
        self.train_data_list_path = train_data_list_path
        self.test_data_list_path = test_data_list_path
        self._p_map_eng_dict = {}
        # load dictionary
        self._dict_path_dict = {'wordemb_dict': self._wordemb_dict_path,
                                'postag_dict': self._postag_dict_path,
                                'label_dict': self._label_dict_path}
        # check if the file exists
        for input_dict in [wordemb_dict_path,
                           postag_dict_path,
                           label_dict_path,
                           train_data_list_path,
                           test_data_list_path]:
            if not os.path.exists(input_dict):
                raise ValueError("%s not found." % (input_dict))
                return

        self._feature_dict = {}
        self._feature_dict['postag_dict'] = self._load_dict_from_file(self._dict_path_dict['postag_dict'])
        self._feature_dict['wordemb_dict'] = self._load_dict_from_file(self._dict_path_dict['wordemb_dict'])
        self._feature_dict['label_dict'] = self._load_label_dict(self._dict_path_dict['label_dict'])
        self._reverse_dict = {name: self._get_reverse_dict(name) for name in self._dict_path_dict.keys()}
        self._reverse_dict['eng_map_p_dict'] = self._reverse_p_eng(self._p_map_eng_dict)
        self._UNK_IDX = 0

    def _load_label_dict(self, dict_name):
        """load label dict from file"""
        label_dict = {}
        with open(dict_name, 'r', encoding='utf-8') as fr:
            for idx, line in enumerate(fr):
                p, p_eng = line.strip().split('\t')
                label_dict[p_eng] = idx
                self._p_map_eng_dict[p] = p_eng
        return label_dict

    def _load_dict_from_file(self, dict_name, bias=0):
        """
        Load vocabulary from file.
        """
        dict_result = {}
        with open(dict_name, 'r', encoding='utf-8') as f_dict:
            for idx, line in enumerate(f_dict):
                line = line.strip()
                dict_result[line] = idx + bias
        return dict_result

    def _cal_mark_slot(self, spo_list, sentence):
        """
        Calculate the value of the label
        """
        mark_list = [0] * len(self._feature_dict['label_dict'])
        for spo in spo_list:
            predicate = spo['predicate']
            p_idx = self._feature_dict['label_dict'][self._p_map_eng_dict[predicate]]
            mark_list[p_idx] = 1
        return mark_list

    def _is_valid_input_data(self, input_data):
        """is the input data valid"""
        try:
            dic = json.loads(input_data)
        except:
            return False
        if "text" not in dic or "postag" not in dic or type(dic["postag"]) is not list:
            return False
        for item in dic['postag']:
            if "word" not in item or "pos" not in item:
                return False
        return True

    def _get_feed_iterator(self, line, need_input=False, need_label=True):
        # verify that the input format of each line meets the format
        if not self._is_valid_input_data(line):
            print(sys.stderr, 'Format is error')
            return None
        dic = json.loads(line)
        sentence = dic['text']
        sentence_term_list = [item['word'] for item in dic['postag']]
        sentence_pos_list = [item['pos'] for item in dic['postag']]
        sentence_emb_slot = [self._feature_dict['wordemb_dict'].get(w, self._UNK_IDX) for w in sentence_term_list]
        sentence_pos_slot = [self._feature_dict['postag_dict'].get(pos, self._UNK_IDX) for pos in sentence_pos_list]
        if 'spo_list' not in dic:
            label_slot = [0] * len(self._feature_dict['label_dict'])
        else:
            label_slot = self._cal_mark_slot(dic['spo_list'], sentence)
        # verify that the feature is valid
        if len(sentence_emb_slot) == 0 or len(sentence_pos_slot) == 0 or len(label_slot) == 0:
            return None
        feature_slot = [sentence_emb_slot, sentence_pos_slot]
        input_fields = json.dumps(dic)
        output_slot = feature_slot
        if need_input:
            output_slot = [input_fields] + output_slot
        if need_label:
            output_slot = output_slot + [label_slot]
        return output_slot

    def path_reader(self, data_path, need_input=False, need_label=True):
        """Read data from data_path"""
        self._feature_dict['data_keylist'] = []

        def reader():
            """Generator"""
            if os.path.isdir(data_path):
                input_files = os.listdir(data_path)
                for data_file in input_files:
                    data_file_path = os.path.join(data_path, data_file)
                    for line in open(data_file_path.strip()):
                        sample_result = self._get_feed_iterator(line.strip(), need_input, need_label)
                        if sample_result is None:
                            continue
                        yield tuple(sample_result)
            elif os.path.isfile(data_path):
                for line in open(data_path.strip(), encoding='utf-8'):
                    sample_result = self._get_feed_iterator(line.strip(), need_input, need_label)
                    if sample_result is None:
                        continue
                    yield tuple(sample_result)

        return reader

    def get_train_reader(self, need_input=False, need_label=True):
        """Data reader during training"""
        return self.path_reader(self.train_data_list_path, need_input, need_label)

    def get_test_reader(self, need_input=True, need_label=True):
        """Data reader during test"""
        return self.path_reader(self.test_data_list_path, need_input, need_label)

    def get_predict_reader(self, predict_file_path='', need_input=True, need_label=False):
        """Data reader during predict"""
        return self.path_reader(predict_file_path, need_input, need_label)

    def get_dict(self, dict_name):
        """Return dict"""
        if dict_name not in self._feature_dict:
            raise ValueError("dict name %s not found." % (dict_name))
        return self._feature_dict[dict_name]

    def get_all_dict_name(self):
        """Get name of all dict"""
        return self._feature_dict.keys()

    def get_dict_size(self, dict_name):
        """Return dict length"""
        if dict_name not in self._feature_dict:
            raise ValueError("dict name %s not found." % (dict_name))
        return len(self._feature_dict[dict_name])

    def _get_reverse_dict(self, dict_name):
        dict_reverse = {}
        for key, value in self._feature_dict[dict_name].items():
            dict_reverse[value] = key
        return dict_reverse

    def _reverse_p_eng(self, dic):
        dict_reverse = {}
        for key, value in dic.items():
            dict_reverse[value] = key
        return dict_reverse

    def get_label_output(self, label_idx):
        """Output final label, used during predict and test"""
        dict_name = 'label_dict'
        if len(self._reverse_dict[dict_name]) == 0:
            self._get_reverse_dict(dict_name)
        p_eng = self._reverse_dict[dict_name][label_idx]
        return self._reverse_dict['eng_map_p_dict'][p_eng]


class DataGen(object):
    def __init__(self, is_train, batch_size):
        self.batch_size = batch_size
        self._train = is_train
        self.reader = RcDataReader(wordemb_dict_path='../../data/dict/word_idx',
                                   postag_dict_path='../../data/dict/postag_dict',
                                   label_dict_path='../../data/dict/p_eng',
                                   train_data_list_path='../../data/train_data.json',
                                   test_data_list_path='../../data//dev_data.json')

        self.train_data = self.reader.get_train_reader()
        self.test_data = self.reader.get_test_reader()
        self.predict_data = self.reader.get_test_reader()

    def __iter__(self):
        if self._train is True:
            w = []
            p = []
            l = []
            for data in self.train_data():
                word_idx_list, postag_list, label_list = data
                if(len(word_idx_list) < 200):
                    word_idx_list.extend([0 for _ in range(200 - len(word_idx_list))])
                    postag_list.extend([0 for _ in range(200 - len(postag_list))])
                w.append(word_idx_list)
                p.append(postag_list)
                l.append(label_list)
            for n_batches in range(int(len(w) / self.batch_size)):
                yield w[n_batches * self.batch_size: (n_batches + 1) * self.batch_size], \
                      p[n_batches * self.batch_size: (n_batches + 1) * self.batch_size], \
                      l[n_batches * self.batch_size: (n_batches + 1) * self.batch_size]


    def __len__(self):
        return len(self.batch_size)


if __name__ == '__main__':
    #initialize data generator
    data_generator = RcDataReader(
        wordemb_dict_path='../../data/dict/word_idx',
        postag_dict_path='../../data/dict/postag_dict',
        label_dict_path='../../data/dict/p_eng',
        train_data_list_path='../../data/train_data.json',
        test_data_list_path='../../data//dev_data.json')

    # prepare data reader
    # total_len = 0
    # max_len = 0
    #
    # ttt = data_generator.get_train_reader()
    # for index, features in enumerate(ttt()):
    #     word_idx_list, postag_list, label_list = features
    #     if(len(word_idx_list)>max_len):
    #         max_len = len(word_idx_list)
    #     total_len += len(word_idx_list)
    #     if(index%1000==0):
    #         print(index)
    #     # print(input_sent)
    #     # print('1st features:', len(word_idx_list), word_idx_list)
    #     # print('2nd features:', len(postag_list), postag_list)
    #     # print('3rd features:', len(label_list), '\t', label_list)
    # avg_len = total_len/index
    # print(avg_len)
    # print(max_len)
    #
    gen = DataGen(is_train=True, batch_size=32)
    for i in gen:
        continue



