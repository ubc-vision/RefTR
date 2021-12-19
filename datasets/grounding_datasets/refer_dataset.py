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
import argparse
import collections
import logging
import re
import operator
# import h5py
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
from collections import OrderedDict
sys.path.append('.')
# import util
from util.word_utils import Corpus

from transformers import BertTokenizerFast, RobertaTokenizerFast
from util.transforms import letterbox, random_affine
from datasets.lang_utils import convert_examples_to_features, read_examples
# sys.modules['utils'] = utils

cv2.setNumThreads(0)

def build_bert_tokenizer(bert_model):
    if bert_model.split('-')[0] == 'roberta':
        lang_backbone = RobertaTokenizerFast.from_pretrained(bert_model, do_lower_case=True, do_basic_tokenize=False)
    else:
        lang_backbone = BertTokenizerFast.from_pretrained(bert_model, do_lower_case=True, do_basic_tokenize=False)
    return lang_backbone

class DatasetNotFoundError(Exception):
    pass

class FlickrMultiPhraseDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'flickr': {'splits': ('train', 'val', 'test', 'trainval')}
    }

    def __init__(
            self, data_root, im_dir, dataset='referit', split='train', max_seq_len=88,
            max_num_phrases=16, max_phrase_len=22, bert_model='bert-base-uncased', lstm=False):
        self.images = []
        self.data_root = data_root
        self.im_dir = im_dir
        self.dataset = dataset
        self.seq_len = max_seq_len
        self.phrase_seq_len = max_phrase_len
        self.num_phrases = max_num_phrases
        self.split = split

        print("Using tokenizer from:", bert_model)
        self.tokenizer = build_bert_tokenizer(bert_model)
        # self.tokenizer.add_special_tokens({'cls_phrase': '[CLS_P]', 'sperator_phrase': '[SEP_P]'})

        annotation_path = osp.join(data_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(annotation_path, imgset_file)
            self.images += torch.load(imgset_path)

    def exists_dataset(self):
        return osp.exists(osp.join(self.data_root, self.dataset))

    def pull_item(self, idx):
        if self.dataset == 'flickr':
            img_file, phrase_pos, bbox, phrases, _, sentence = self.images[idx]
        else:
            img_file, _, bbox, phrase, sentence = self.images[idx]
            phrases = [sentence]
            phrase_pos = [0]
        ## box format: to x1y1x2y2
        bbox = np.array(bbox, dtype=int)

        img_path = osp.join(self.im_dir, img_file)
        img = cv2.imread(img_path)
        ## duplicate channel if gray image
        if img.shape[-1] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.stack([img] * 3)
        return img, phrases, phrase_pos, sentence, bbox, img_file

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        def phrase_pos_to_mask(pos_start, sentence_token, phrase_token, seq_len):
            phrase_len = len(phrase_token) - 2
            pos_start = pos_start + 1
            assert phrase_len >= 0
            assert sentence_token[pos_start:phrase_len+pos_start] == phrase_token[1:-1]

            mask = np.zeros(seq_len, dtype=np.bool)
            if phrase_len == 0:
                mask[0] = True
            else:
                mask[pos_start:phrase_len+pos_start] = True
            return mask

        img, phrases, phrase_char_pos_l, sentence, bbox, img_file = self.pull_item(idx)
        # phrase = phrase.decode("utf-8").encode().lower()

        # encode phrase to bert input
        tokenized_sentence = self.tokenizer(
            sentence,
            padding='max_length',
            max_length=self.seq_len,
            return_tensors='pt',
        )
        word_id = tokenized_sentence['input_ids'][0]
        word_mask = tokenized_sentence['attention_mask'][0]

        # examples = read_examples(sentence, idx)
        # sentence_features = convert_examples_to_features(
        #     examples=examples, seq_length=self.seq_len, tokenizer=self.tokenizer)
        # word_id = sentence_features[0].input_ids
        # word_mask = sentence_features[0].input_mask

        phrase_masks = []
        phrase_context_masks = []
        tokenized_phrases = []
        phrase_pos_l = []
        phrase_pos_r = []
        for p, char_pos_l in zip(phrases, phrase_char_pos_l):
            tokenized_phrase = self.tokenizer(
                p,
                padding='max_length',
                max_length=self.phrase_seq_len,
                return_tensors='np',
            )
            tokenized_phrases.append(tokenized_phrase['input_ids'][0])
            phrase_masks.append(tokenized_phrase['attention_mask'][0])

            # set up phrase_pos
            phrase_char_len = p.__len__()
            pos_l = tokenized_sentence.char_to_token(char_pos_l)
            pos_r = tokenized_sentence.char_to_token(char_pos_l + phrase_char_len - 1) 
            assert pos_l is not None and pos_r is not None
            # Tips for roberta: Ġ means the end of a new token 
            # So assert from the second character
            # assert tokenized_sentence.tokens()[pos_l+1:pos_r] == tokenized_phrase.tokens()[2:1+pos_r-pos_l],\
            #     (tokenized_sentence.tokens()[pos_l:pos_r], tokenized_phrase.tokens(), pos_l, pos_r)
            phrase_pos_l.append(pos_l)
            phrase_pos_r.append(pos_r+1)

        for i in range(len(phrases), self.num_phrases):
            tokenized_phrase = self.tokenizer(
                "",
                padding='max_length',
                max_length=self.phrase_seq_len,
                return_tensors='np',
            )
            tokenized_phrases.append(tokenized_phrase['input_ids'][0])
            phrase_masks.append(tokenized_phrase['attention_mask'][0])
            phrase_pos_l.append(0)
            phrase_pos_r.append(1)

        h, w, c = img.shape
        samples = {
            "img": img,
            "sentence": np.array(word_id, dtype=int),
            "sentence_mask": np.array(word_mask, dtype=bool),
            "phrase": np.array(tokenized_phrases, dtype=int),
            "phrase_mask": np.array(phrase_masks, dtype=bool),
            "phrase_pos_l": np.array(phrase_pos_l, dtype=int),
            "phrase_pos_r": np.array(phrase_pos_r, dtype=int)
        }

        image_id = int(img_file.split('.')[0].split('_')[-1])
        target = {
            "image_id": image_id,
            "boxes": np.array(bbox, dtype=np.float32),
            "labels": [0],
            'dataset_id': idx,
            "orig_size": np.array([h, w], dtype=np.int)
        }
        return samples, target
        # if self.testmode:
        #     return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
        #         np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
        #         np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0]
        # else:
        #     return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
        #     np.array(bbox, dtype=np.float32)

class ReferSegDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'refcoco_unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'refcoco+_unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'refcocog_google': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'refcocog_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        }
    }

    def __init__(self, data_root, im_dir, seg_dir, dataset='refcoco_unc', 
                 split='train', max_query_len=40, bert_model='bert-base-uncased'):
        self.images = []
        self.data_root = data_root
        self.im_dir = im_dir
        self.dataset = dataset
        self.query_len = max_query_len
        self.split = split
        self.tokenizer = build_bert_tokenizer(bert_model)

        dataset_dir = self.dataset.split('_')[0]
        annotation_path = osp.join(data_root, dataset_dir)
        self.seg_dir = osp.join(seg_dir, dataset_dir)

        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']
        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]
        splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(annotation_path, imgset_file)
            self.images += torch.load(imgset_path)

    def pull_item(self, idx):
        img_file, seg_file, bbox, phrase = self.images[idx]
        ## box format: x1y1x2y2
        bbox = np.array(bbox, dtype=int)
        img = cv2.imread(osp.join(self.im_dir, img_file))
        mask = np.load(osp.join(self.seg_dir, seg_file))
        assert img.shape[:2] == mask.shape[:2]
        ## duplicate channel if gray image
        if img.shape[-1] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.stack([img] * 3)
        return img, mask, phrase, bbox, img_file

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, mask, phrase, bbox, img_file = self.pull_item(idx)
        # phrase = phrase.decode("utf-8").encode().lower()
        phrase = phrase.lower()

        # encode phrase to bert input
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

        mask = mask[None, :, :]
        image_id = int(img_file.split('.')[0].split('_')[-1])
        target = {
            "image_id": image_id,
            'dataset_id': idx,
            "boxes": np.array([bbox], dtype=np.float32),
            "labels": [0],
            "masks": mask,
            "orig_size": np.array([h, w], dtype=np.int)
        }
        return samples, target

