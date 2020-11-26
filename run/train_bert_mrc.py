#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Description:
# run_machine_comprehension.py 
# Please Notice that the data should contain 
# multi answers 
# need pay MORE attention when loading data 



import os 
import argparse 
import numpy as np 
import random
from scipy.special import softmax
import sys
sys.path.append("./data_loader")
sys.path.append("./layer")
sys.path.append("./model")
sys.path.append("./metric")

import torch 
from torch import nn 

from model_config import Config
from mrc_data_loader import MRCNERDataLoader
from mrc_data_processor import *
from optim import AdamW, lr_linear_decay
from bert_mrc import BertQueryNER
from bert_tokenizer import BertTokenizer4Tagger
from mrc_ner_evaluate  import flat_ner_performance, nested_ner_performance
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear




def args_parser():
    # start parser 
    parser = argparse.ArgumentParser()

    # requires parameters 
    parser.add_argument("--config_path", default="configs/zh_bert.json", type=str)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--bert_model", default="/home/xuemengge/data/CrossTag/bert-base-chinese-pytorch/", type=str,)
    parser.add_argument("--task_name", default=None, type=str)
    parser.add_argument("--max_seq_length", default=150, type=int)
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--dev_batch_size", default=128, type=int)
    parser.add_argument("--test_batch_size", default=128, type=int)
    parser.add_argument("--checkpoint", default=100, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3006)
    parser.add_argument("--export_model", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="/home/lixiaoya/output")
    parser.add_argument("--data_sign", type=str, default="msra_ner")
    parser.add_argument("--weight_start", type=float, default=1.0) 
    parser.add_argument("--weight_end", type=float, default=1.0) 
    parser.add_argument("--weight_span", type=float, default=0.0)
    parser.add_argument("--entity_sign", type=str, default="flat")
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--entity_threshold", type=float, default=0.0)
    parser.add_argument("--data_cache", type=bool, default=True)

    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--regenerate_rate", type=float, default=0.1)
    parser.add_argument("--STrain", type=int, default=1)
    parser.add_argument("--perepoch", type=int, default=0)

    args = parser.parse_args()

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps 

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
 
    return args


def load_data(config):

    print("-*-"*10)
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
    elif config.data_sign == "project":
        data_processor = ProjectProcessor()

    else:
        raise ValueError("Please Notice that your data_sign DO NOT exits !!!!!")


    label_list = data_processor.get_labels()
    tokenizer = BertTokenizer4Tagger.from_pretrained(config.bert_model, do_lower_case=True)

    dataset_loaders = MRCNERDataLoader(config, data_processor, label_list, tokenizer, mode="train", allow_impossible=True)
    train_dataloader = dataset_loaders.get_dataloader(data_sign="train") 
    dev_dataloader = dataset_loaders.get_dataloader(data_sign="dev")
    test_dataloader = dataset_loaders.get_dataloader(data_sign="test")
    num_train_steps = dataset_loaders.get_num_train_epochs()


    return train_dataloader, dev_dataloader, test_dataloader, num_train_steps, label_list 



def load_model(config, num_train_steps, label_list, pretrain=None):
    device = torch.device("cuda") 
    n_gpu = config.n_gpu
    model = BertQueryNER(config, )
    if pretrain:
        print("loading pretrain........")
        pretrained_dict = torch.load(pretrain)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # model.load_state_dict(torch.load(pretrain))
    model.to(device)
    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # prepare optimzier 
    param_optimizer = list(model.named_parameters())

    """
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
    {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
    {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=10e-8)
    """
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
    {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
    {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    # optimizer = Adam(optimizer_grouped_parameters, lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate, warmup=config.warmup_proportion, t_total=num_train_steps, max_grad_norm=config.clip_grad)

    sheduler = None

    return model, optimizer, sheduler, device, n_gpu



def train(model, optimizer, sheduler,  train_dataloader, dev_dataloader, test_dataloader, config, \
    device, n_gpu, label_list):
    nb_tr_steps = 0 
    tr_loss = 0 

    dev_best_acc = 0 
    dev_best_precision = 0 
    dev_best_recall = 0 
    dev_best_f1 = 0 
    dev_best_loss = 10000000000000


    test_acc_when_dev_best = 0 
    test_pre_when_dev_best = 0 
    test_rec_when_dev_best = 0 
    test_f1_when_dev_best = 0 
    test_loss_when_dev_best = 1000000000000000

    model.train()

    for idx in range(int(config.num_train_epochs)):
        tr_loss = 0 
        nb_tr_examples, nb_tr_steps = 0, 0 
        print("#######"*10)
        print("EPOCH: ", str(idx))
        print("steps: ", len(train_dataloader))
        """
        if idx != 0:
            lr_linear_decay(optimizer)
        """
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch) 
            input_ids, input_mask, segment_ids, start_pos, end_pos, span_pos, ner_cate = batch 
            loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, \
                start_positions=start_pos, end_positions=end_pos, span_positions=span_pos)
            if n_gpu > 1:
                loss = loss.mean()

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad) 
            optimizer.step()

            tr_loss += loss.item()

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1 


            if nb_tr_steps % config.checkpoint == 0:
                print("-*-"*15)
                print("current training loss is : ")
                print(loss.item())
                tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1 = eval_checkpoint(model, dev_dataloader, config, device, n_gpu, label_list, eval_sign="dev")
                print("......"*10)
                print("DEV: loss, acc, precision, recall, f1")
                print(tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1)

                if tmp_dev_f1 > dev_best_f1 :
                    dev_best_acc = tmp_dev_acc 
                    dev_best_loss = tmp_dev_loss 
                    dev_best_precision = tmp_dev_prec 
                    dev_best_recall = tmp_dev_rec 
                    dev_best_f1 = tmp_dev_f1 

                    # export model 
                    if config.export_model:
                        model_to_save = model.module if hasattr(model, "module") else model 
                        output_model_file = os.path.join(config.output_dir, "bert_finetune_model_{}_{}.bin".format(str(idx),str(nb_tr_steps)))
                        torch.save(model_to_save.state_dict(), output_model_file)
                        print("SAVED model path is :") 
                        print(output_model_file)

                    tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1 = eval_checkpoint(model, test_dataloader, config, device, n_gpu, label_list, eval_sign="test")
                    print("......"*10)
                    print("TEST: loss, acc, precision, recall, f1")
                    print(tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1)


                    test_acc_when_dev_best = tmp_test_acc 
                    test_pre_when_dev_best = tmp_test_prec
                    test_rec_when_dev_best = tmp_test_rec
                    test_f1_when_dev_best = tmp_test_f1 
                    test_loss_when_dev_best = tmp_test_loss

                print("-*-"*15)



        if config.STrain and idx < (int(config.num_train_epochs) - 1):
            if config.perepoch:
                regenerate = config.regenerate_rate * (1 + idx)
            else:
                regenerate = config.regenerate_rate
                
            print("regenerate:", regenerate)
            train_dataloader = train_regenerate_nospan(model, train_dataloader,
                                            device, config, label_list,
                                            regenerate,
                                            str(idx))

    print("=&="*15)
    print("Best DEV : overall best loss, acc, precision, recall, f1 ")
    print(dev_best_loss, dev_best_acc, dev_best_precision, dev_best_recall, dev_best_f1)
    print("scores on TEST when Best DEV:loss, acc, precision, recall, f1 ")
    print(test_loss_when_dev_best, test_acc_when_dev_best, test_pre_when_dev_best, test_rec_when_dev_best, test_f1_when_dev_best)
    print("=&="*15)



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

    
    if config.entity_sign == "flat":
        eval_accuracy, eval_precision, eval_recall, eval_f1 = flat_ner_performance(start_pred_lst, end_pred_lst, span_pred_lst, start_gold_lst, end_gold_lst, span_gold_lst, ner_cate_lst, label_list, threshold=config.entity_threshold, dims=2)
    else:
        eval_accuracy, eval_precision, eval_recall, eval_f1 = nested_ner_performance(start_pred_lst, end_pred_lst, span_pred_lst, start_gold_lst, end_gold_lst, span_gold_lst, ner_cate_lst, label_list, threshold=config.entity_threshold, dims=2)

    average_loss = round(eval_loss / eval_steps, 4)  
    eval_f1 = round(eval_f1 , 4)
    eval_precision = round(eval_precision , 4)
    eval_recall = round(eval_recall , 4) 
    eval_accuracy = round(eval_accuracy , 4) 

    return average_loss, eval_accuracy, eval_precision, eval_recall, eval_f1

def train_regenerate_nospan(model_object, eval_dataloader, device, config, label_list, gama, saveddata_flag):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader
    if gama>1:
        gama=config.regenerate_rate*int(1/config.regenerate_rate)
    model_object.eval()
    train_input_ids = []
    train_input_mask = []
    train_segment_ids = []
    train_start_pos = []
    train_end_pos = []
    train_ner_cate = []
    start_pred_lst = []
    end_pred_lst = []
    mask_lst = []
    start_gold_lst = []
    end_gold_lst = []
    span_gold_lst= []
    eval_steps = 0
    ner_cate_lst = []
    examples=0

    for input_ids, input_mask, segment_ids, start_true, end_true, span_true, ner_cate in eval_dataloader:
        #examples+=len(input_ids)
        #print("Loading......",str(examples))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        start_true = start_true.to(device)
        end_true = end_true.to(device)
        span_true =span_true.to(device)

        with torch.no_grad():
            start_logits, end_logits, _ = model_object(input_ids, segment_ids, input_mask)

        start_logits = start_logits.detach().cpu().numpy()
        end_logits = end_logits.detach().cpu().numpy()
        start_true = start_true.to("cpu").numpy()
        end_true = end_true.to("cpu").numpy()
        span_true = span_true.to("cpu").numpy()
        reshape_lst = start_true.shape

        start_logits = np.reshape(start_logits, (reshape_lst[0], reshape_lst[1], 2)).tolist()
        end_logits = np.reshape(end_logits, (reshape_lst[0], reshape_lst[1], 2)).tolist()

        start_pos = np.zeros([reshape_lst[0], reshape_lst[1]], int)
        end_pos = np.zeros([reshape_lst[0], reshape_lst[1]], int)

        start_index = [[idx for idx, tmp in enumerate(softmax(np.array(j), axis=-1)) if tmp[-1]>gama] for j in start_logits]
        end_index = [[idx for idx, tmp in enumerate(softmax(np.array(j), axis=-1)) if tmp[-1]>gama] for j in end_logits]

        for batch_dim in range(len(start_index)):
            for tmp_start in start_index[batch_dim]:
                tmp_end = [tmp for tmp in end_index[batch_dim] if tmp >= tmp_start]
                if len(tmp_end) == 0:
                    continue
                else:
                    tmp_end = min(tmp_end)
                start_pos[batch_dim][tmp_start]=1
                end_pos[batch_dim][tmp_end]=1


        start_pos = start_pos.tolist()
        end_pos = end_pos.tolist()
        start_true = start_true.tolist()
        end_true = end_true.tolist()
        span_true = span_true.tolist()
        end_pred_lst += end_pos
        start_pred_lst += start_pos
        start_gold_lst += start_true
        end_gold_lst += end_true
        span_gold_lst += span_true
        ner_cate_lst += ner_cate.numpy().tolist()
        mask_lst += input_mask.to("cpu").numpy().tolist()
        start_pos = torch.tensor(start_pos, dtype=torch.short)
        end_pos = torch.tensor(end_pos, dtype=torch.short)
        train_input_ids.append(input_ids)
        train_input_mask.append(input_mask)
        train_segment_ids.append(segment_ids)
        train_start_pos.append(start_pos)
        train_end_pos.append(end_pos)
        train_ner_cate.append(ner_cate)
    train_input_ids = torch.cat(train_input_ids, 0)
    train_input_mask = torch.cat(train_input_mask, 0)
    train_segment_ids = torch.cat(train_segment_ids, 0)
    train_start_pos = torch.cat(train_start_pos, 0)
    train_end_pos = torch.cat(train_end_pos, 0)
    train_ner_cate = torch.cat(train_ner_cate, 0)
    #train_loss_mask = torch.cat(train_loss_mask, 0)

    np.save(config.output_dir + saveddata_flag + "-train_input_ids", train_input_ids.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_input_mask", train_input_mask.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_segment_ids", train_segment_ids.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_start_pos", train_start_pos.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_end_pos", train_end_pos.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_ner_cate", train_ner_cate.cpu().numpy())
    #np.save(config.output_dir + saveddata_flag + "-train_loss_mask", train_loss_mask.cpu().numpy())
    train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_start_pos, train_end_pos, torch.tensor(np.zeros((train_input_ids.size(0),1,1), dtype=int)), train_ner_cate)
    train_sampler = SequentialSampler(train_data)  # RandomSampler(dataset)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.train_batch_size)

    print("######Regenerate Over#########")
    return train_dataloader



def train_regenerate(model_object, eval_dataloader, device, config, label_list, gama, saveddata_flag):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader
    if gama>1:
        gama=config.regenerate_rate*int(1/config.regenerate_rate)
    model_object.eval()
    train_input_ids = []
    train_input_mask = []
    train_segment_ids = []
    train_start_pos = []
    train_end_pos = []
    train_span = []
    train_ner_cate = []
    start_pred_lst = []
    end_pred_lst = []
    span_pred_lst = []
    mask_lst = []
    start_gold_lst = []
    end_gold_lst = []
    span_gold_lst=[]
    eval_steps = 0
    ner_cate_lst = []

    examples=0

    for input_ids, input_mask, segment_ids, start_true, end_true, span_true, ner_cate in eval_dataloader:

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        start_true = start_true.to(device)
        end_true = end_true.to(device)
        span_true = span_true.to(device)

        with torch.no_grad():
            start_logits, end_logits, span_logits = model_object(input_ids, segment_ids, input_mask)

        start_logits = start_logits.detach().cpu().numpy()
        end_logits = end_logits.detach().cpu().numpy()
        span_logits = span_logits.detach().cpu().numpy()
        start_true = start_true.to("cpu").numpy()
        end_true = end_true.to("cpu").numpy()
        span_true = span_true.to("cpu").numpy()
        reshape_lst = start_true.shape

        start_logits = np.reshape(start_logits, (reshape_lst[0], reshape_lst[1], 2)).tolist()
        end_logits = np.reshape(end_logits, (reshape_lst[0], reshape_lst[1], 2)).tolist()
        span_logits = np.reshape(span_logits, (reshape_lst[0], reshape_lst[1], reshape_lst[1], 1)).tolist()

        start_pos = np.zeros([reshape_lst[0], reshape_lst[1]], int)
        end_pos = np.zeros([reshape_lst[0], reshape_lst[1]], int)
        span_pred = np.zeros([reshape_lst[0], reshape_lst[1], reshape_lst[1]], int)

        start_index = [[idx for idx, tmp in enumerate(j) if tmp[-1]>gama] for j in softmax(np.array(start_logits))]
        end_index = [[idx for idx, tmp in enumerate(j) if tmp[-1]>gama] for j in softmax(np.array(end_logits))]
        for batch_dim in range(len(start_index)):
            for tmp_start in start_index[batch_dim]:
                tmp_end = [tmp for tmp in end_index[batch_dim] if tmp >= tmp_start]
                if len(tmp_end) == 0:
                    continue
                else:
                    tmp_end = min(tmp_end)
                if span_logits[batch_dim][tmp_start][tmp_end] >= gama:
                    start_pos[batch_dim][tmp_start]=1
                    end_pos[batch_dim][tmp_start]=1
                    span_pred[batch_dim][tmp_start][tmp_end]=1

        start_pos = start_pos.tolist()
        end_pos = end_pos.tolist()
        span_pred=span_pred.tolist()
        start_true = start_true.tolist()
        end_true = end_true.tolist()
        span_true = span_true.tolist()
        end_pred_lst += end_pos
        start_pred_lst += start_pos
        span_pred_lst +=span_pred
        start_gold_lst += start_true
        end_gold_lst += end_true
        span_gold_lst += span_true
        ner_cate_lst += ner_cate.numpy().tolist()
        mask_lst += input_mask.to("cpu").numpy().tolist()
        start_pos = torch.tensor(start_pos, dtype=torch.short)
        end_pos = torch.tensor(end_pos, dtype=torch.short)
        span_pred = torch.tensor(span_pred, dtype=torch.short)
        train_input_ids.append(input_ids)
        train_input_mask.append(input_mask)
        train_segment_ids.append(segment_ids)
        train_start_pos.append(start_pos)
        train_end_pos.append(end_pos)
        train_span.append(span_pred)
        train_ner_cate.append(ner_cate)
    #eval_accuracy, eval_precision, eval_recall, eval_f1 = query_ner_compute_performance(start_pred_lst, end_pred_lst, start_gold_lst, end_gold_lst, ner_cate_lst, label_list, mask_lst, dims=2)
    eval_accuracy, eval_precision, eval_recall, eval_f1 = flat_ner_performance(start_pred_lst, end_pred_lst,
                                                                               span_pred_lst, start_gold_lst,
                                                                               end_gold_lst, span_gold_lst,
                                                                               ner_cate_lst, label_list,
                                                                               threshold=config.entity_threshold,
                                                                               dims=2)

    # eval_accuracy, eval_precision, eval_recall, eval_f1 = compute_performance(pred_lst, gold_lst, mask_lst, label_list, dims=2)

    eval_f1 = round(eval_f1, 4)
    eval_precision = round(eval_precision , 4)
    eval_recall = round(eval_recall , 4)
    eval_accuracy = round(eval_accuracy , 4)
    print("f1: precision:  recall: accuracy")
    print(eval_f1, eval_precision, eval_recall, eval_accuracy)
    train_input_ids = torch.cat(train_input_ids, 0)
    train_input_mask = torch.cat(train_input_mask, 0)
    train_segment_ids = torch.cat(train_segment_ids, 0)
    train_start_pos = torch.cat(train_start_pos, 0)
    train_end_pos = torch.cat(train_end_pos, 0)
    train_span = torch.cat(train_span, 0)
    train_ner_cate = torch.cat(train_ner_cate, 0)
    #train_loss_mask = torch.cat(train_loss_mask, 0)

    np.save(config.output_dir + saveddata_flag + "-train_input_ids", train_input_ids.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_input_mask", train_input_mask.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_segment_ids", train_segment_ids.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_start_pos", train_start_pos.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_end_pos", train_end_pos.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_span", train_span.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_ner_cate", train_ner_cate.cpu().numpy())
    #np.save(config.output_dir + saveddata_flag + "-train_loss_mask", train_loss_mask.cpu().numpy())
    train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_start_pos, train_end_pos, train_span, train_ner_cate)
    train_sampler = SequentialSampler(train_data)  # RandomSampler(dataset)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.train_batch_size)

    print("######Regenerate Over#########")
    return train_dataloader



def merge_config(args_config):
    model_config_path = args_config.config_path 
    model_config = Config.from_json_file(model_config_path)
    model_config.update_args(args_config)
    model_config.print_config()
    return model_config



def main():
    args_config = args_parser()
    config = merge_config(args_config)
    train_loader, dev_loader, test_loader, num_train_steps, label_list = load_data(config)
    model, optimizer, sheduler, device, n_gpu = load_model(config, num_train_steps, label_list, config.pretrain)
    train(model, optimizer, sheduler, train_loader, dev_loader, test_loader, config, device, n_gpu, label_list)
    

if __name__ == "__main__":
    main() 
