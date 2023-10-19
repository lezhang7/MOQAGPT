from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
import pandas as pd

import argparse
import torch
import toml
#from utils import check_dir_exist_or_build, pstore, pload, split_and_padding_neighbor, set_seed, load_collection
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
import json
from tqdm import tqdm, trange
import random
from itertools import combinations
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class MMCoQA_Text(Dataset):
    def __init__(self, args, tokenizer, filename, collection=None):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        if collection:
            with open(collection , 'r', encoding="utf-8") as f:
                passages = f.readlines()

            passages_dict = {}
            for line in passages:
                line = json.loads(line.strip())
                passages_dict[line['id']] = line["contents"]

        question_prefix = 'question: '
        passage_prefix = 'context: '

        for line in tqdm(data, disable=args.disable_tqdm):
            record = json.loads(line)
            question_type = record["question_type"]
            if question_type != "text":
                continue
            qid = record["qid"]
            question_text = record["question"]
            oracle_text = record["gold_question"]
            answer = record["answer"][0]
            assert answer["modality"] == "text"
            pos_pid = answer["text_instances"][0]["doc_id"]
            if collection:
                pos_p_text = passages_dict[pos_pid]   
            else:
                pos_p_text = ""             
            if "text" in answer["text_instances"][0]:
                answer_span = answer["text_instances"][0]["text"]
                start_pos = answer["text_instances"][0]["start_byte"]
                end_pos = start_pos + len(answer_span.split()) - 1
            else:
                answer_span = answer["answer"]
                start_pos = -1
                end_pos = -1
                continue
            history_context = []
            history = record["history"]
            for turn in history:
                history_context.append(turn["question"])
                history_context.append(turn["answer"][0]["answer"])

            pos_p_ids = tokenizer.encode(pos_p_text, add_special_tokens = True, max_length=args.max_doc_length)
            cur_q_ids = tokenizer.encode(question_text, add_special_tokens = True, max_length=args.max_query_length)
            ora_q_ids = tokenizer.encode(oracle_text, add_special_tokens = True, max_length=args.max_query_length)
            q_pos_p_ids = cur_q_ids + pos_p_ids
            
            flat_concat = []
            flat_concat.extend(ora_q_ids)
            for j in range(len(history_context) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                #if args.use_prefix and first_context:
                #    ctx_utts_text[j] = "context: " + ctx_utts_text[j]
                #    first_context = False
                utt = tokenizer.encode(history_context[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)
            cur_q_ids, cur_q_ids_mask = padding_seq_to_same_length(cur_q_ids, max_pad_length = args.max_query_length)
            ora_q_ids, ora_q_ids_mask = padding_seq_to_same_length(ora_q_ids, max_pad_length = args.max_query_length)
            q_pos_p_ids, q_pos_p_ids_mask = padding_seq_to_same_length(q_pos_p_ids, max_pad_length = args.max_concat_length)
            self.examples.append([qid, 
                                  pos_pid,
                                  cur_q_ids,
                                  cur_q_ids_mask,
                                  ora_q_ids,
                                  ora_q_ids_mask,
                                  flat_concat,
                                  flat_concat_mask,
                                  start_pos,
                                  end_pos,
                                  answer_span,
                                  q_pos_p_ids,
                                  q_pos_p_ids_mask,
                                  pos_p_text])
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_qid": [],
                             "bt_pos_pid": [],
                             "bt_cur_q_ids": [],
                             "bt_cur_q_ids_mask": [],
                             "bt_ora_q_ids": [],
                             "bt_ora_q_ids_mask": [],
                             "bt_flat_concat": [],
                             "bt_flat_concat_mask": [],
                             "bt_start_pos": [],
                             "bt_end_pos": [],
                             "bt_answer_span": [],
                             "bt_q_pos_p_ids": [],
                             "bt_q_pos_p_ids_mask": [],
                             "bt_pos_p_text": []}

            for example in batch:
                collated_dict["bt_qid"].append(example[0])
                collated_dict["bt_pos_pid"].append(example[1])
                collated_dict["bt_cur_q_ids"].append(example[2]) 
                collated_dict["bt_cur_q_ids_mask"].append(example[3]) 
                collated_dict["bt_ora_q_ids"].append(example[4]) 
                collated_dict["bt_ora_q_ids_mask"].append(example[5]) 
                collated_dict["bt_flat_concat"].append(example[6]) 
                collated_dict["bt_flat_concat_mask"].append(example[7])
                collated_dict["bt_start_pos"].append(example[8])
                collated_dict["bt_end_pos"].append(example[9])
                collated_dict["bt_answer_span"].append(example[10])
                collated_dict["bt_q_pos_p_ids"].append(example[11])
                collated_dict["bt_q_pos_p_ids_mask"].append(example[12])
                collated_dict["bt_pos_p_text"].append(example[13])

            not_need_to_tensor_keys = set(["bt_qid", "bt_pos_pid", "bt_start_pos", "bt_end_pos", "bt_answer_span", "bt_pos_p_text"])
            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class MMCoQA_all(Dataset):
    def __init__(self, args, tokenizer, filename, collection=None):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        if collection:
            with open(collection , 'r', encoding="utf-8") as f:
                passages = f.readlines()

            passages_dict = {}
            for line in passages:
                line = json.loads(line.strip())
                passages_dict[line['id']] = line["contents"]

        question_prefix = 'question: '
        passage_prefix = 'context: '

        for line in tqdm(data, disable=args.disable_tqdm):
            record = json.loads(line)
            question_type = record["question_type"]
            qid = record["qid"]
            question_text = record["question"]
            oracle_text = record["gold_question"]
            history_context = []
            history = record["history"]
            for turn in history:
                history_context.append(turn["question"])
                history_context.append(turn["answer"][0]["answer"])

            cur_q_ids = tokenizer.encode(question_text, add_special_tokens = True, max_length=args.max_query_length)
            ora_q_ids = tokenizer.encode(oracle_text, add_special_tokens = True, max_length=args.max_query_length)
            
            flat_concat = []
            flat_concat.extend(cur_q_ids)
            for j in range(len(history_context) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                utt = tokenizer.encode(history_context[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)
            cur_q_ids, cur_q_ids_mask = padding_seq_to_same_length(cur_q_ids, max_pad_length = args.max_query_length)
            ora_q_ids, ora_q_ids_mask = padding_seq_to_same_length(ora_q_ids, max_pad_length = args.max_query_length)
            self.examples.append([qid, 
                                  cur_q_ids,
                                  cur_q_ids_mask,
                                  ora_q_ids,
                                  ora_q_ids_mask,
                                  flat_concat,
                                  flat_concat_mask,
                                  ])
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_qid": [],
                             "bt_cur_q_ids": [],
                             "bt_cur_q_ids_mask": [],
                             "bt_ora_q_ids": [],
                             "bt_ora_q_ids_mask": [],
                             "bt_flat_concat": [],
                             "bt_flat_concat_mask": [],
                             }

            for example in batch:
                collated_dict["bt_qid"].append(example[0])
                collated_dict["bt_cur_q_ids"].append(example[1]) 
                collated_dict["bt_cur_q_ids_mask"].append(example[2]) 
                collated_dict["bt_ora_q_ids"].append(example[3]) 
                collated_dict["bt_ora_q_ids_mask"].append(example[4]) 
                collated_dict["bt_flat_concat"].append(example[5]) 
                collated_dict["bt_flat_concat_mask"].append(example[6])

            not_need_to_tensor_keys = set(["bt_qid", "bt_pos_pid", "bt_start_pos", "bt_end_pos", "bt_answer_span", "bt_pos_p_text"])
            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class MMCoQA_Text_retrieval(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        for line in tqdm(data, disable=args.disable_tqdm):
            record = json.loads(line)
            question_type = record["question_type"]
            qid = record["qid"]
            question_text = record["question"]
            oracle_text = record["gold_question"]
            answer = record["answer"][0]
            if args.modality == "text":
                if question_type != "text":
                    continue
                assert answer["modality"] == "text"
                pos_pid = answer["text_instances"][0]["doc_id"]
                if "text" in answer["text_instances"][0]:
                    answer_span = answer["text_instances"][0]["text"]
                    start_pos = answer["text_instances"][0]["start_byte"]
                    end_pos = start_pos + len(answer_span.split()) - 1
                else:
                    answer_span = answer["answer"]
                    start_pos = -1
                    end_pos = -1
                    continue
            elif args.modality == "all":
                answer_span = answer["answer"]
                start_pos = -1
                end_pos = -1

            history_context = []
            history = record["history"]
            for turn in history:
                history_context.append(turn["question"])
                history_context.append(turn["answer"][0]["answer"])

            cur_q_ids = tokenizer.encode(question_text, add_special_tokens = True, max_length=args.max_query_length)
            ora_q_ids = tokenizer.encode(oracle_text, add_special_tokens = True, max_length=args.max_query_length)
            
            flat_concat = []
            flat_concat.extend(ora_q_ids)
            for j in range(len(history_context) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                utt = tokenizer.encode(history_context[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)
            cur_q_ids, cur_q_ids_mask = padding_seq_to_same_length(cur_q_ids, max_pad_length = args.max_query_length)
            ora_q_ids, ora_q_ids_mask = padding_seq_to_same_length(ora_q_ids, max_pad_length = args.max_query_length)
            self.examples.append([qid, 
                                  cur_q_ids,
                                  cur_q_ids_mask,
                                  ora_q_ids,
                                  ora_q_ids_mask,
                                  flat_concat,
                                  flat_concat_mask,
                                  start_pos,
                                  end_pos,
                                  answer_span])
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_qid": [],
                             "bt_cur_q_ids": [],
                             "bt_cur_q_ids_mask": [],
                             "bt_ora_q_ids": [],
                             "bt_ora_q_ids_mask": [],
                             "bt_flat_concat": [],
                             "bt_flat_concat_mask": [],
                             "bt_start_pos": [],
                             "bt_end_pos": [],
                             "bt_answer_span": []}

            for example in batch:
                collated_dict["bt_qid"].append(example[0])
                collated_dict["bt_cur_q_ids"].append(example[1]) 
                collated_dict["bt_cur_q_ids_mask"].append(example[2]) 
                collated_dict["bt_ora_q_ids"].append(example[3]) 
                collated_dict["bt_ora_q_ids_mask"].append(example[4]) 
                collated_dict["bt_flat_concat"].append(example[5]) 
                collated_dict["bt_flat_concat_mask"].append(example[6])
                collated_dict["bt_start_pos"].append(example[7])
                collated_dict["bt_end_pos"].append(example[8])
                collated_dict["bt_answer_span"].append(example[9])

            not_need_to_tensor_keys = set(["bt_qid", "bt_pos_pid", "bt_start_pos", "bt_end_pos", "bt_answer_span", "bt_pos_p_text"])
            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn



def padding_seq_to_same_length(input_ids, max_pad_length, pad_token = 0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids
            
    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length
  
    return input_ids, attention_mask