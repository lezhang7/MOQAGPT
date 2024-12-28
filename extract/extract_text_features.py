import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
import os
import argparse
from model_zoo import get_embedding_model

def parse_args():
    parser = argparse.ArgumentParser(description='extract image features')
    parser.add_argument('--passages_file_path', type=str, default="multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl")
    parser.add_argument('--query_file_path', type=str, default="MMCoQA_test.txt")
    parser.add_argument('--save_dir', type=str, default="stored_features/mmcoqa/text")
    parser.add_argument('--model', type=str, default="Alibaba-NLP/gte-base-en-v1.5")
    parser.add_argument('--batch_size', type=int, default=2048)
    args = parser.parse_args()
    return args


def load_passage_collection(passages_file_path):
    #idx_id_list = []
    passages_dict = {}
    with open(passages_file_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            passages_dict[line['id']] = line["title"] + ' ' + line["text"]
    return passages_dict

def load_query_collection(query_file_path):
    query_dict = {}
    with open(query_file_path,'r') as f:
        if query_file_path.endswith('.txt'):
            lines = f.readlines()
            for line in lines:
                line = json.loads(line.strip())
                query_dict[line['qid']] = line["question"]
        else:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line.strip())
                query_dict[line['qid']] = line["question"]
    return query_dict

def extract_passage_features(args, model):
    model_name = args.model.split('/')[-1] if '/' in args.model else args.model
    save_dir=os.path.join(args.save_dir,model_name+'_features')
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    
    print(f"extract text features by {args.model}, save to directory: {save_dir}")

    if not os.path.exists(os.path.join(save_dir,"ReferenceEmbedding.pt")):
        print(f"extract ReferenceEmbedding")
        passage_dict = load_passage_collection(args.passages_file_path)
        qid_ref_list=list(passage_dict.items())
        model.extract_text_features(qid_ref_list,args.batch_size,save_path=os.path.join(save_dir,"ReferenceEmbedding.pt"))
    else:
        print(f"ReferenceEmbedding of model {model_name} already exists")

    if not os.path.exists(os.path.join(save_dir,"QuestionEmbedding.pt")):
        print(f"extract QuestionEmbedding")
        query_dict = load_query_collection(args.query_file_path)
        qid_query_list=list(query_dict.items())
        model.extract_text_features(qid_query_list,args.batch_size,save_path=os.path.join(save_dir,"QuestionEmbedding.pt"))
    else:
        print(f"QuestionEmbedding of model {model_name} already exists")

    
if __name__ == '__main__':

    args=parse_args()
    model = get_embedding_model(args.model)
    model.eval()
    
    extract_passage_features(args, model)

    

