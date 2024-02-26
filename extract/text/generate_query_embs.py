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

def save_text_features(model, query_file_path, qid2feature_path):
    '''
    - model = passage encoder
    - query_file_path = ""
    - qid2feature_path = "qid2feature.pkl"
    '''

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(query_file_path, 'r') as f:
        data = f.readlines()
    print(len(data))
    qid2feature = {} # qid: emb
    model.eval()
    model.to(device)

    with torch.no_grad():
        for line in data:
            record = json.loads(line)
            qid = record["qid"]
            try:
                gold_question = record["gold_question"]
            except:
                gold_question = record["question"]
            query, query_mask = padding_seq_to_same_length(tokenizer.encode(gold_question, add_special_tokens=True), 32)
            query, query_mask = torch.tensor(query, dtype=torch.long).reshape(1, -1).to(device), torch.tensor(query_mask, dtype=torch.long).reshape(1, -1).to(device)
            query_embs = model(query, query_mask)
            qid2feature[qid] = query_embs.cpu()
    torch.save(qid2feature, qid2feature_path)
    # pstore(qid2feature, qid2feature_path)


if __name__ == '__main__':
    pretrained_encoder_path = "~/scratch/hub/ad-hoc-ance-msmarco"
    query_file_path = "~/scratch/datasets/MMQA/MMQA_dev.jsonl"
    os.makedirs("~/scratch/MOQA/stored_features/mmqa/text/ance_features/",exist_ok=True)
    qid2feature_path = "~/scratch/MOQA/stored_features/mmqa/text/ance_features/QuestionEmbedding.pt"
    os.makedirs("~/scratch/MOQA/stored_features/mmqa/text/ance_features/",exist_ok=True)
    #qid2feature_path = "datasets/MMCoQA/qid2feature.pt"
    config = RobertaConfig.from_pretrained(pretrained_model_name_or_path=pretrained_encoder_path)
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_encoder_path, do_lower_case=True)
    model = ANCE.from_pretrained(pretrained_encoder_path, config=config)
    save_text_features(model, query_file_path, qid2feature_path)