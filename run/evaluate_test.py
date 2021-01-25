# -*- coding: utf-8 -*-



# author: xiaoy li
# description:
# evaluate
import sys
sys.path.append("./data_loader")
sys.path.append("./layer")
sys.path.append("./model")
sys.path.append("./metric")



import random
import argparse
import numpy as np
from load_data import load_data


import torch

from bert_mrc import BertQueryNER
from model_config import Config

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", default="configs/zh_bert.json", type=str)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--bert_model", default="bert-base-chinese-pytorch/", type=str)
    parser.add_argument("--saved_model", type=str, default="saved_model/bert_finetune_model.bin")
    parser.add_argument("--max_seq_length", default=100, type=int)
    parser.add_argument("--test_batch_size", default=16, type=int)
    parser.add_argument("--data_sign", type=str, default="project")
    parser.add_argument("--entity_sign", type=str, default="flat")
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3006)
    parser.add_argument("--weight_start", type=float, default=1.0)
    parser.add_argument("--weight_end", type=float, default=1.0)
    parser.add_argument("--weight_span", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--entity_threshold", type=int, default=0)
    parser.add_argument("--data_cache", type=bool, default=True)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.cuda.manual_seed_all(args.seed)
    return args



def merge_config(args_config):
    model_config_path = args_config.config_path
    model_config = Config.from_json_file(model_config_path)
    model_config.update_args(args_config)
    model_config.print_config()
    return model_config





def eval_checkpoint(device,input_s, input_ids, segment_ids, input_mask, model_object):
    result=[]


    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    input_mask = torch.tensor([input_mask], dtype=torch.short).to(device)
    segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(device)


    with torch.no_grad():
        start_logits, end_logits, span_logits = model_object(input_ids, segment_ids, input_mask)
        print(start_logits)
        print(end_logits)
        start_logits = torch.argmax(start_logits, dim=-1)
        end_logits = torch.argmax(end_logits, dim=-1)
        start_label = start_logits.detach().cpu().numpy().tolist()
        end_label = end_logits.detach().cpu().numpy().tolist()

    print(start_label)
    start_labels = [idx for idx, tmp in enumerate(start_label) if tmp!= 0]
    end_labels = [idx for idx, tmp in enumerate(end_label) if tmp != 0]
    print(start_labels)

    for tmp_start in start_labels:
        tmp_end = [tmp for tmp in end_labels if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end)
        result.append(input_s[tmp_start:tmp_end+1])
    print(result)

    return result

def main():
    args_config = args_parser()
    config = merge_config(args_config)
    device = torch.device("cuda")
    model = BertQueryNER(config,)
    pretrained_dict = torch.load(config.saved_model)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.to(device)
    model.eval()

    input_s, input_ids, input_mask, segment_ids=load_data("明天9点召开互联网SXM隐患信息发现的研讨会，请大家准时参加。", config.bert_model)

    print(input_s)
    result=eval_checkpoint(device, input_s, input_ids, segment_ids, input_mask, model)

    return result

if __name__ == "__main__":
    main()
