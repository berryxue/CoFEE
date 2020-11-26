+#!/usr/bin/env python3
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# 

import os 
import torch
import sys
sys.path.append("../data_loader")
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler  
from mrc_utils import convert_examples_to_features
import numpy as np



class MRCNERDataLoader(object):
    def __init__(self, config, data_processor, label_list, tokenizer, mode="train", allow_impossible=True):

        self.data_dir = config.data_dir 
        self.max_seq_length= config.max_seq_length 

        if mode == "train":
            self.train_batch_size = config.train_batch_size 
            self.dev_batch_size = config.dev_batch_size 
            self.test_batch_size = config.test_batch_size 
            self.num_train_epochs = config.num_train_epochs 
        elif mode == "test":
            self.test_batch_size = config.test_batch_size 

        self.data_processor = data_processor 
        self.label_list = label_list 
        self.allow_impossible = allow_impossible
        self.tokenizer = tokenizer
        self.max_seq_len = config.max_seq_length 
        self.data_cache = config.data_cache 

        self.num_train_instances = 0 
        self.num_dev_instances = 0 
        self.num_test_instances = 0

    def convert_examples_to_features(self, data_sign="train",):

        print("=*="*10)
        print("loading {} data ... ...".format(data_sign))

        allow_impossible=self.allow_impossible

        if data_sign == "train":
            examples = self.data_processor.get_train_examples(self.data_dir)
            self.num_train_instances = len(examples)
            #allow_impossible=False
        elif data_sign == "dev":
            examples = self.data_processor.get_dev_examples(self.data_dir)
            self.num_dev_instances = len(examples)
        elif data_sign == "test":
            examples = self.data_processor.get_test_examples(self.data_dir)
            self.num_test_instances = len(examples)
        else:
            raise ValueError("please notice that the data_sign can only be train/dev/test !!")

        #cache_path = os.path.join(self.data_dir, "mrc-ner.{}.cache.{}.allow.{}".format(data_sign, str(self.max_seq_len), str(allow_impossible)))
        cache_path = os.path.join(self.data_dir, "mrc-ner.{}.cache.{}".format(data_sign, str(self.max_seq_len)))
        if os.path.exists(cache_path) and self.data_cache:
            features = torch.load(cache_path)
        else:

            features = convert_examples_to_features(examples, self.tokenizer, self.label_list, self.max_seq_length, allow_impossible=allow_impossible)
            if self.data_cache:
                torch.save(features, cache_path)
        return features


    def get_dataloader(self, data_sign="train", saved_dir=None):
        features = self.convert_examples_to_features(data_sign=data_sign)
        print(f"{len(features)} {data_sign} data loaded")
        if saved_dir:
            input_ids = torch.tensor(np.load(saved_dir + "train_input_ids.npy"), dtype=torch.long)
            input_mask = torch.tensor(np.load(saved_dir + "train_input_mask.npy"), dtype=torch.long)
            segment_ids = torch.tensor(np.load(saved_dir + "train_segment_ids.npy"), dtype=torch.long)
            ner_cate = torch.tensor(np.load(saved_dir + "train_ner_cate.npy"), dtype=torch.long)
            start_pos = torch.tensor(np.load(saved_dir + "train_start_pos.npy"), dtype=torch.long)
            end_pos = torch.tensor(np.load(saved_dir + "train_end_pos.npy"), dtype=torch.long)

        else:
            input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.short)
            segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            start_pos = torch.tensor([f.start_position for f in features], dtype=torch.short)
            end_pos = torch.tensor([f.end_position for f in features], dtype=torch.short)
            ner_cate = torch.tensor([f.ner_cate for f in features], dtype=torch.short)
        span_pos = torch.tensor([f.span_position for f in features], dtype=torch.short)
        dataset = TensorDataset(input_ids, input_mask, segment_ids, start_pos, end_pos, span_pos, ner_cate)
        
        if data_sign == "train":
            if saved_dir:
                datasampler = SequentialSampler(dataset)
            else:
                datasampler = RandomSampler(dataset) # RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size)
        elif data_sign == "dev":
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.dev_batch_size)
        elif data_sign == "test":
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size)

        return dataloader 


    def get_num_train_epochs(self, ):
        return int((self.num_train_instances / self.train_batch_size) * self.num_train_epochs) 








