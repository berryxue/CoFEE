#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Author: Xiaoy LI
# Last update: 2019.03.23
# First create: 2019.03.23
# Description:
# run_machine_comprehension.py
# Please Notice that the data should contain
# multi answers
# need pay MORE attention when loading data


import os
import sys
import time
import random
root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)
sys.path.append("./data_loader")
sys.path.append("./layer")
sys.path.append("./model")
sys.path.append("./metric")

import csv
import json
import argparse
import numpy as np
from tqdm import tqdm
from scipy.special import softmax

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from scripts import AverageMeter



from bert_mrc_ner_cluster import BertMRCNER, BertMRCNER_CLUSTER
import clustering as clustering
from model_config import Config
from mrc_data_loader import MRCNERDataLoader
from mrc_data_processor import *
from optim import AdamW, lr_linear_decay
from bert_mrc import BertQueryNER
from bert_tokenizer import BertTokenizer4Tagger
from mrc_ner_evaluate  import flat_ner_performance, nested_ner_performance

from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

def args_parser():
    # start parser
    parser = argparse.ArgumentParser()

    # requires parameters
    parser.add_argument("--config_path", default="configs/zh_bert.json", type=str)
    parser.add_argument("--data_dir", default="data/ner/MI/", type=str)
    parser.add_argument("--bert_model", default="/data/xuemengge/bert/data/CrossTag/bert-base-chinese-pytorch/", type=str, )
    parser.add_argument("--task_name", default=None, type=str)
    parser.add_argument("--max_seq_length", default=150, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--dev_batch_size", default=128, type=int)
    parser.add_argument("--test_batch_size", default=128, type=int)
    parser.add_argument("--checkpoint", default=100, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--warmup", default=0.1, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3006)
    parser.add_argument("--export_model", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="/home/lixiaoya/output")
    parser.add_argument("--data_sign", type=str, default="msra_ner")
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--clustering", type=str, default="Kmeans")
    parser.add_argument("--num_clusters", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--gama", type=float, default=0.9)
    parser.add_argument("--regenerate_rate", type=float, default=0.1)
    parser.add_argument("--pca_dim", type=int, default=256)
    parser.add_argument("--clus_niter", type=int, default=20)
    parser.add_argument("--cluster_layer", type=int, default=11)
    parser.add_argument("--view_number", type=int, default=20)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--data_cache", type=bool, default=True)
    parser.add_argument("--entity_threshold", type=float, default=0.0)
    parser.add_argument("--entity_sign", type=str, default="flat")
    parser.add_argument("--pool_mode", type=str, default="sum")

    args = parser.parse_args()

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed_all(args.seed)

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def load_data(config):
    print("-*-" * 10)
    print("current data_sign: {}".format(config.data_sign))

    if config.data_sign == "conll03":
        data_processor = Conll03Processor()
    elif config.data_sign == "zh_msra":
        data_processor = MSRAProcessor()
    elif config.data_sign == "zh_onto":
        data_processor = Onto4ZhProcessor()
    elif config.data_sign == "en_onto":
        data_processor = Onto5EngProcessor()
    elif config.data_sign == "genia":
        data_processor = GeniaProcessor()
    elif config.data_sign == "ace2004":
        data_processor = ACE2004Processor()
    elif config.data_sign == "ace2005":
        data_processor = ACE2005Processor()
    elif config.data_sign == "resume":
        data_processor = ResumeZhProcessor()
    elif config.data_sign == "wiki":
        data_processor = WikiProcessor()
    elif config.data_sign == "HP":
        data_processor = HPProcessor()
    elif config.data_sign == "HC":
        data_processor = HCProcessor()
    elif config.data_sign == "ecommerce":
        data_processor = EcommerceProcessor()
    elif config.data_sign == "twitter":
        data_processor = TwitterProcessor()
    else:
        raise ValueError("Please Notice that your data_sign DO NOT exits !!!!!")

    label_list = data_processor.get_labels()
    tokenizer = BertTokenizer4Tagger.from_pretrained(config.bert_model, do_lower_case=True)
    dataset_loaders = MRCNERDataLoader(config, data_processor, label_list, tokenizer, mode="train",
                                       allow_impossible=True)
    if config.data_sign == "HP":
        train_dataloader = dataset_loaders.get_dataloader(data_sign="train")
    else:
        train_dataloader = dataset_loaders.get_dataloader(data_sign="train", saved_dir=config.data_dir)
    dev_dataloader = dataset_loaders.get_dataloader(data_sign="dev")
    test_dataloader = dataset_loaders.get_dataloader(data_sign="test")
    num_train_steps = dataset_loaders.get_num_train_epochs()

    return train_dataloader, dev_dataloader, test_dataloader, num_train_steps, label_list




def load_model(config, num_train_steps, label_list, pretrain=None):
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    # for name,parameters in model.named_parameters():
    # print(name,':',parameters.size())
    if pretrain:
        """
        for idx in range(4):
            model = BertMRCNER(config)
            pretrained_dict = torch.load(pretrain+str(idx)+"epoch_bert_finetune_model.bin")
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            model.to(device)
            train_sequential_loader, train_dataloader = train_regenerate(model, train_dataloader, device,
                                                                         config, label_list,
                                                                         config.regenerate_rate * (idx + 1))
        """
        model = BertMRCNER_CLUSTER(config)
        pretrained_dict = torch.load(pretrain + "bert_finetune_model.bin")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # model.load_state_dict(torch.load(pretrain))
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # prepare optimzier
    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    # optimizer = Adam(optimizer_grouped_parameters, lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate, warmup=config.warmup,
                         t_total=num_train_steps, max_grad_norm=config.clip_grad)
    return model, optimizer, device, n_gpu


def eval_checkpoint(model_object, eval_dataloader, config, \
                    device, n_gpu, label_list, eval_sign="dev"):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader
    model_object.eval()

    eval_loss = 0
    start_pred_lst = []
    end_pred_lst = []
    span_pred_lst = []
    mask_lst = []
    start_gold_lst = []
    span_gold_lst = []
    end_gold_lst = []
    eval_steps = 0
    ner_cate_lst = []

    for input_ids, input_mask, segment_ids, start_pos, end_pos, span_pos, ner_cate in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        start_pos = start_pos.to(device)
        end_pos = end_pos.to(device)
        span_pos = span_pos.to(device)

        with torch.no_grad():
            tmp_eval_loss = model_object(input_ids, segment_ids, input_mask, start_pos, end_pos, span_pos)
            start_logits, end_logits, span_logits = model_object(input_ids, segment_ids, input_mask)
            start_logits = torch.argmax(start_logits, dim=-1)
            end_logits = torch.argmax(end_logits, dim=-1)

        start_pos = start_pos.to("cpu").numpy().tolist()
        end_pos = end_pos.to("cpu").numpy().tolist()
        span_pos = span_pos.to("cpu").numpy().tolist()

        start_label = start_logits.detach().cpu().numpy().tolist()
        end_label = end_logits.detach().cpu().numpy().tolist()
        span_logits = span_logits.detach().cpu().numpy().tolist()
        span_label = span_logits

        input_mask = input_mask.to("cpu").detach().numpy().tolist()

        ner_cate_lst += ner_cate.numpy().tolist()
        eval_loss += tmp_eval_loss.mean().item()
        mask_lst += input_mask
        eval_steps += 1

        start_pred_lst += start_label
        end_pred_lst += end_label
        span_pred_lst += span_label

        start_gold_lst += start_pos
        end_gold_lst += end_pos
        span_gold_lst += span_pos


    eval_accuracy, eval_precision, eval_recall, eval_f1 = flat_ner_performance(start_pred_lst, end_pred_lst,span_pred_lst, start_gold_lst,end_gold_lst, span_gold_lst,ner_cate_lst, label_list, threshold=config.entity_threshold, dims=2)

    average_loss = round(eval_loss / eval_steps, 4)
    eval_f1 = round(eval_f1, 4)
    eval_precision = round(eval_precision, 4)
    eval_recall = round(eval_recall, 4)
    eval_accuracy = round(eval_accuracy, 4)

    return average_loss, eval_accuracy, eval_precision, eval_recall, eval_f1


def merge_config(args_config):
    model_config_path = args_config.config_path
    model_config = Config.from_json_file(model_config_path)
    model_config.update_args(args_config)
    model_config.print_config()
    return model_config


def train_cluster(model, optimizer, train_dataloader, dev_dataloader, test_dataloader,
                  dev_best_acc, dev_best_f1, test_best_acc, test_best_f1,
                  config, device, n_gpu, label_list, cluster_dict, cluster_var, epoch):
    global_step = 0
    model.train()

    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    cluster_var = torch.tensor(cluster_var).to(device)
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, start_pos, end_pos, span_pos, ner_cate, cluster_pos = batch
        cluster_pos = cluster_pos.cpu().numpy()

        input_truth = []

        for i in cluster_pos:
            for k in range(i[0], i[1]):
                input_truth.append(cluster_dict[k])

        input_truth = torch.tensor(input_truth, dtype=torch.long).to(device)
        loss = model(input_ids, segment_ids, input_mask, start_pos, end_pos, span_pos, input_truth, cluster_var)


        if n_gpu > 1:
            loss = loss.mean()

        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)

        tr_loss += loss.item()

        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        if (step + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            global_step += 1

        if nb_tr_steps % config.checkpoint == 0 or nb_tr_steps==config.num_train_step:
            print("-*-" * 15)
            print("current instances is : ", nb_tr_steps)
            print("current training loss is : ")
            print(loss.item())
            # continue
            tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1 = eval_checkpoint(model, dev_dataloader, config, device, n_gpu, label_list, eval_sign="dev")
            print("......" * 10)
            print("DEV: loss, acc, precision, recall, f1")
            print(tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1)

            if tmp_dev_f1 > dev_best_f1 or tmp_dev_acc > dev_best_acc:
                dev_best_acc = tmp_dev_acc
                dev_best_f1 = tmp_dev_f1

                tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1 = eval_checkpoint(model,
                                                                                                        test_dataloader,
                                                                                                        config, device,
                                                                                                        n_gpu,
                                                                                                        label_list,
                                                                                                        eval_sign="test")
                print("......" * 10)
                print("TEST: loss, acc, precision, recall, f1")
                print(tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1)

                if tmp_test_f1 > test_best_f1 or tmp_test_acc > test_best_acc:
                    test_best_acc = tmp_test_acc
                    test_best_f1 = tmp_test_f1

                    # export model
                    if config.export_model:
                        model_to_save = model.module if hasattr(model, "module") else model
                        output_model_file = os.path.join(config.output_dir, str(epoch)+"-"+str(step)+"-"+"bert_finetune_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)

            print("-*-" * 15)

    return model, dev_best_acc, dev_best_f1, test_best_acc, test_best_f1


def compute_features(dataloader, model, N, config, device):
    """
    fuctions:
    get features
    reshape features
    the time of this function
    """
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    train_input_ids = []
    train_input_mask = []
    train_segment_ids = []
    train_ner_cate = []
    train_start_pos = []
    train_end_pos = []
    features=[]

    # discard the label information in the dataloader
    for i, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, start_pos, end_pos, span_true, ner_cate = batch

        if len(features)==0:
            features = model.get_features(input_ids, segment_ids, input_mask, start_pos, end_pos)
            if len(features)>0:
                features=features.data.cpu().numpy()
        else:
            aux = model.get_features(input_ids, segment_ids, input_mask, start_pos, end_pos)
            if len(aux)>0:
                aux=aux.data.cpu().numpy()
                features = np.concatenate((features, aux), axis=0)

        """
        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * config.train_batch_size: (i + 1) * config.train_batch_size] = aux
        else:
            # special treatment for final batch
            features[i * config.train_batch_size:] = aux
        """

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 1000 == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
        input_ids = input_ids.to("cpu").numpy().tolist()
        input_mask = input_mask.to("cpu").numpy().tolist()
        segment_ids = segment_ids.to("cpu").numpy().tolist()
        ner_cate = ner_cate.to("cpu").numpy().tolist()
        start_pos = start_pos.to("cpu").numpy().tolist()
        end_pos = end_pos.to("cpu").numpy().tolist()
        train_input_ids.extend(input_ids)
        train_input_mask.extend(input_mask)
        train_segment_ids.extend(segment_ids)
        train_ner_cate.extend(ner_cate)
        train_start_pos.extend(start_pos)
        train_end_pos.extend(end_pos)



    return features, train_input_ids, train_input_mask, train_segment_ids, train_ner_cate, train_start_pos, train_end_pos


def get_cluster_dataloader(config, train_input_ids, train_input_mask, train_segment_ids, train_ner_cate,
                           train_start_pos, train_end_pos):
    instance = len(train_input_ids)
    cluster_pos = [0] * instance
    start = 0
    for i in range(instance):
        start_pos = np.argwhere(np.array(train_start_pos[i]) == 1)
        end_pos = np.argwhere(np.array(train_end_pos[i]) == 1)
        cluster_pos[i] = [start, start + len(end_pos)]
        start = start + len(end_pos)

    train_input_ids = torch.tensor(train_input_ids, dtype=torch.long)
    train_input_mask = torch.tensor(train_input_mask, dtype=torch.long)
    train_segment_ids = torch.tensor(train_segment_ids, dtype=torch.long)
    train_ner_cate = torch.tensor(train_ner_cate, dtype=torch.long)
    train_start_pos = torch.tensor(train_start_pos, dtype=torch.long)
    train_end_pos = torch.tensor(train_end_pos, dtype=torch.long)
    cluster_pos = torch.tensor(cluster_pos, dtype=torch.long)

    train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_start_pos, train_end_pos, torch.tensor(np.zeros((train_input_ids.size(0),1,1), dtype=int)), train_ner_cate, cluster_pos)
    train_sampler = SequentialSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=config.train_batch_size)

    return train_loader


def main():
    args_config = args_parser()
    config = merge_config(args_config)
    train_loader, dev_loader, test_loader, num_train_steps, label_list = load_data(config)
    config.num_train_step=num_train_steps
    # reload the model
    model, optimizer, device, n_gpu = load_model(config, num_train_steps, label_list, args_config.pretrain)

    deepcluster = clustering.__dict__[config.clustering](config.num_clusters)

    dev_best_acc=0
    dev_best_f1=0
    test_best_acc=0
    test_best_f1=0

    for epoch in range(int(config.num_train_epochs)):
        print("#######" * 10)
        print("EPOCH: ", str(epoch))
        features, train_input_ids, train_input_mask, train_segment_ids, train_ner_cate, train_start_pos, train_end_pos \
            = compute_features(train_loader, model, len(train_loader), config, device)
        clustering_loss, cluster_var = deepcluster.cluster(features, config.view_number, config.cluster_layer, config.pca_dim,
                                              config.clus_niter, epoch, verbose=True)
        train_cluster_loader = get_cluster_dataloader(config, train_input_ids, train_input_mask,
                                                      train_segment_ids, train_ner_cate,
                                                      train_start_pos,train_end_pos)

        cluster_dict = clustering.cluster_assign(deepcluster.images_lists)

        model, dev_best_acc, dev_best_f1, test_best_acc, test_best_f1 = train_cluster(model, optimizer,
                                                         train_cluster_loader, dev_loader, test_loader,
                                                         dev_best_acc, dev_best_f1,
                                                         test_best_acc, test_best_f1,
                                                         config, device, n_gpu, label_list,
                                                         cluster_dict, cluster_var,
                                                         epoch)
    print("=&=" * 15)
    print("DEV: current best f1, acc")
    print(dev_best_f1, dev_best_acc)
    print("TEST: current bes f1, acc")
    print(test_best_f1, test_best_acc)
    print("=&=" * 15)


if __name__ == "__main__":
    main()
