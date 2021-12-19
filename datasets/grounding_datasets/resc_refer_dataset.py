# -*- coding: utf-8 -*-

"""
Copied from https://github.com/zyang-ur/ReSC/blob/e4022f87bfd11200b67c4509bb9746640834ceae/utils/transforms.py

ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.
Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import sys
import cv2
import json
import uuid
import tqdm
import math
import torch
import random
# import h5py
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
from collections import OrderedDict
sys.path.append('.')
import operator
# import util
from util.word_utils import Corpus

import argparse
import logging
import re


from transformers import BertTokenizerFast, RobertaTokenizerFast
# from transformers import BertTokenizer,BertModel
from util.transforms import letterbox, random_affine
from datasets.lang_utils import convert_examples_to_features, read_examples
# sys.modules['utils'] = utils

def build_bert_tokenizer(bert_model):
    if bert_model.split('-')[0] == 'roberta':
        lang_backbone = RobertaTokenizerFast.from_pretrained(bert_model, do_lower_case=True, do_basic_tokenize=False)
    else:
        lang_backbone = BertTokenizerFast.from_pretrained(bert_model, do_lower_case=True, do_basic_tokenize=False)
    return lang_backbone

cv2.setNumThreads(0)


class DatasetNotFoundError(Exception):
    pass


class ReferDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {'splits': ('train', 'val', 'trainval', 'test')},
        'vg': {'splits': ('all')}
    }

    def __init__(self, data_root, im_dir, dataset='referit', 
                 split='train', max_query_len=128, lstm=False, bert_model='bert-base-uncased'):
        self.images = []
        self.data_root = data_root
        self.im_dir = im_dir
        self.dataset = dataset
        self.query_len = max_query_len
        self.lstm = lstm
        self.split = split
        self.tokenizer = build_bert_tokenizer(bert_model)

        if not self.exists_dataset():
            # self.process_dataset()
            print('Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ')
            exit(0)

        annotation_path = osp.join(data_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if self.lstm:
            self.corpus = Corpus()
            corpus_path = osp.join(annotation_path, 'corpus.pth')
            self.corpus = torch.load(corpus_path)

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(annotation_path, imgset_file)
            self.images += torch.load(imgset_path)

    def exists_dataset(self):
        return osp.exists(osp.join(self.data_root, self.dataset))

    def pull_item(self, idx):
        if self.dataset in ['flickr', 'vg']:
            img_file, bbox, phrase = self.images[idx]
        else:
            img_file, _, bbox, phrase, attri = self.images[idx]
        ## box format: to x1y1x2y2
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)

        img_path = osp.join(self.im_dir, img_file)
        img = cv2.imread(img_path)
        ## duplicate channel if gray image
        if img.shape[-1] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.stack([img] * 3)
        return img, phrase, bbox, img_file

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox, img_file = self.pull_item(idx)
        # phrase = phrase.decode("utf-8").encode().lower()
        phrase = phrase.lower()

        # encode phrase to bert input
        # Enable truncation in this case
        tokenized_sentence = self.tokenizer(
            phrase,
            padding='max_length',
            max_length=self.query_len,
            truncation=True,
            return_tensors='pt',
        )
        word_id = tokenized_sentence['input_ids'][0]
        word_mask = tokenized_sentence['attention_mask'][0]

        h, w, c = img.shape

        samples = {
            "img": img,
            "sentence": np.array(word_id, dtype=int),
            "sentence_mask": np.array(word_mask, dtype=int)
        }

        image_id = int(img_file.split('.')[0].split('_')[-1])
        target = {
            "image_id": image_id,
            "boxes": np.array([bbox], dtype=np.float32),
            "labels": [0],
            'dataset_id': idx,
            "orig_size": np.array([h, w], dtype=np.int)
        }
        return samples, target