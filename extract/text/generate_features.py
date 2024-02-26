import torch
import numpy as np
import json
import linecache
from tqdm import tqdm
import os
import random
import pickle
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from models import ANCE
from MMCoQA_data import MMCoQA_Text, padding_seq_to_same_length


def pstore(x, path):
	with open(path, 'wb') as f:
		pickle.dump(x, f)
	print('store object in path = {} ok'.format(path))

def load_passage_collection(passages_file):
    #idx_id_list = []
    passages_dict = {}
    with open(passages_file,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            #passages_dict[line['id']] = line["contents"]
            passages_dict[line['id']] = line["title"] + ' ' + line["text"]
            #idx_id_list.append((line['id'],'text'))
    return passages_dict

def save_text_features(model, passages_file_path, pid2feature_path, feature_idx_to_id_path=None):
    '''
    - model = passage encoder
    - passages_file_path = "MMCoQA_text_collection.jsonl"
    - pid2feature_path = "pid2feature.pkl"
    - feature_idx_to_id_path = "feature_idx_to_id.pkl"
    '''

    # save a dict - pid: feature by pstore
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #all_text_features = [] # 218285 passages
    passage_dict = load_passage_collection(passages_file_path)
    pid2feature = {} # pid: feature
    #feature_idx_to_id = [] # feature_idx_to_id[idx] = pid
    #text_list = list(passages_dict.items()) # passages_dict - pid: text / text_list - (pid, text)
    model.eval()
    model.to(device)
    with torch.no_grad():
        p_num = 0
        bt_passage = []
        bt_passage_mask = []
        bt_pid = []  
        batch_size = 1024      
        for pid, passage in tqdm(passage_dict.items()):
            # text -> ids
            passage, passage_mask = padding_seq_to_same_length(tokenizer.encode(passage, add_special_tokens=True), 512)
            # construct batch
            bt_passage.append(passage)
            bt_passage_mask.append(passage_mask)
            bt_pid.append(pid)
            # feature_idx_to_id.append(pid)
            p_num += 1
            if p_num % batch_size == 0 or p_num == len(passage_dict):
                # ids -> emb
                bt_passage = torch.tensor(bt_passage, dtype=torch.long).to(device)
                bt_passage_mask = torch.tensor(bt_passage_mask, dtype=torch.long).to(device)
                passage_embs = model(bt_passage, bt_passage_mask) # B * dim
                passage_embs_cpu=passage_embs.cpu()
                del passage_embs_cpu
                torch.cuda.empty_cache()
                #all_text_features.append(passage_embs)
                for idx in range(len(bt_pid)):
                    pid2feature[bt_pid[idx]] = passage_embs[idx]
                bt_passage = []
                bt_passage_mask = []
                bt_pid = []
    torch.save(pid2feature, pid2feature_path)
    #pstore(pid2feature, pid2feature_path)
    #pstore(feature_idx_to_id, feature_idx_to_id_path)


if __name__ == '__main__':
    pretrained_encoder_path = "~/scratch/hub/ad-hoc-ance-msmarco"
    #passages_file_path = "datasets/MMCoQA/bm25_collection/MMCoQA_text_collection.jsonl"
    os.makedirs("~/scratch/MOQA/stored_features/mmqa/text/ance_features/",exist_ok=True)
    passages_file_path = "~/scratch/datasets/MMQA/MMQA_texts.jsonl"
    pid2feature_path = "~/scratch/MOQA/stored_features/mmqa/text/ance_features/ReferenceEmbedding.pt"
    config = RobertaConfig.from_pretrained(pretrained_encoder_path)
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_encoder_path, do_lower_case=True)
    model = ANCE.from_pretrained(pretrained_encoder_path, config=config)
    save_text_features(model, passages_file_path, pid2feature_path)