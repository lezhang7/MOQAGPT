import sys
sys.path.append('../../')
sys.path.append('..')
from utils.utils import *
import sys
import os
from dataset_zoo import get_dataset
from model_zoo import get_embedding_model
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='extract image features')
    parser.add_argument('--dataset', type=str, default="mmcoqa")
    parser.add_argument('--save_dir', type=str, default="stored_features/mmcoqa/image")
    parser.add_argument('--model', type=str, default="clip")
    parser.add_argument('--batch_size', type=int, default=2048)
    args = parser.parse_args()
    return args

def extract_image_features(args):
    model_name = args.model.split('/')[-1] if '/' in args.model else args.model
    save_dir=os.path.join(args.save_dir,model_name+'_features')
    print(f"extract {args.dataset} image features by {args.model}, save to directory: {args.save_dir}")
    dataset=get_dataset(args.dataset)
    model=get_embedding_model(args.model)
    images_ref=list(dataset._images_dict.items())
    queries=[(x['qid'],x['question']) for x in dataset._query_list]
    if not os.path.exists(os.path.join(save_dir,"ReferenceEmbedding.pt")):
        model.extract_image_features(dataset,images_ref,args.batch_size,save_path=save_dir)
    if not os.path.exists(os.path.join(save_dir,"QuestionEmbedding.pt")):
        model.extract_text_features(queries,args.batch_size,save_path=save_dir)



if __name__ == '__main__':
    config=read_config()
    args=parse_args()
    extract_image_features(args)