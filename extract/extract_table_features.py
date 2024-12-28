import sys
sys.path.append('../../')
sys.path.append('..')
from utils.utils import *
import sys
import torch
import numpy as np
import os
from tqdm import tqdm
from dataset_zoo import get_dataset
from model_zoo import get_embedding_model
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='extract image features')
    parser.add_argument('--dataset', type=str, default="mmqa")
    parser.add_argument('--model', type=str, default="ada")
    parser.add_argument('--batch_size', type=int, default=2048)
    args = parser.parse_args()
    return args

def extract_table_features(args):
    save_path=os.path.join(config["PROJECT_DIR"],'stored_features',args.dataset,"table",args.model+'_features')
    print(f"extract {args.dataset} image features by {args.model}, save to directory: {save_path}")
    dataset=get_dataset(args.dataset)
    model=get_embedding_model(args.model)
    images_ref=list(dataset._images_dict.items())
    queries=[(x['qid'],x['question']) for x in dataset._query_list]
    if not os.path.exists(os.path.join(save_path,"ReferenceEmbedding.pt")):
        print(f"extract ReferenceEmbedding")
        model.extract_text_features(dataset,images_ref,args.batch_size,save_path=os.path.join(save_path,"ReferenceEmbedding.pt"))
    if not os.path.exists(os.path.join(save_path,"QuestionEmbedding.pt")):
        print(f"extract QuestionEmbedding")
        model.extract_text_features(queries,args.batch_size,save_path=os.path.join(save_path,"QuestionEmbedding.pt"))



if __name__ == '__main__':
    args=parse_args()
    extract_table_features(args)