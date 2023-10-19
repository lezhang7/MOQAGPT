import json, string, re
import numpy as np
from collections import Counter
import argparse
import os
import sys
sys.path.insert(0, '..')
from utils.utils import *
config=read_config()

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
      return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
      return ' '.join(text.split())
    def remove_punc(text):
      exclude = set(string.punctuation)
      return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
      return str(text).lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
      return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def evaluate(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        
        f1_queue, em_queue = [], []
        for qid in data:
            if 'generated_answer' in data[qid]:
                gt = data[qid]['answer']
                try:
                    pred = data[qid]['generated_answer']
                    if isinstance(pred, list):
                        pred=pred[0]
                except Exception as e:
                    print(e)
                    print(data[qid])
                    exit()
            elif 'final_answer' in data[qid]:
                gt = data[qid]['gold_answer']
                pred = data[qid]['final_answer']
                print(qid)
                if isinstance(pred, list):
                    if len(pred)==0:
                        pred=''
                    else:
                        pred=pred[0]    
            elif 'gpt_answer' in data[qid]:
                gt = data[qid]['answer']
                pred = data[qid]['gpt_answer']
                if isinstance(pred, list):
                        pred=pred[0]

            f1 = f1_score(pred, gt)
            em = exact_match_score(pred, gt)
            f1_queue.append(f1)
            em_queue.append(em)
        
        f1_queue, em_queue = np.array(f1_queue), np.array(em_queue)
        return np.mean(f1_queue), np.mean(em_queue)

def modality_eval_mmcoqa(file_path, aux_file_path):
    with open(aux_file_path, 'r') as f, open(file_path) as g:
        lines = f.readlines()
        data = json.load(g)
        img_f1, img_em = [], []
        text_f1, text_em = [], []
        table_f1, table_em = [], []
        for line, qid in zip(lines, data):
            jdata = json.loads(line)
            modality = jdata['question_type']
            if 'generated_answer' in data[qid]:
                gt = data[qid]['answer']
                try:
                    pred = data[qid]['generated_answer']
                    if isinstance(pred, list):
                        pred=pred[0]
                except Exception as e:
                    print(e)
                    print(data[qid])
                    exit()
            elif 'final_answer' in data[qid]:
                gt = data[qid]['gold_answer']
                pred = data[qid]['final_answer']
                if len(pred)==0:
                    pred=''
                else:
                    pred=pred[0]       
            elif 'gpt_answer' in data[qid]:
                gt = data[qid]['answer']
                pred = data[qid]['gpt_answer']
                if isinstance(pred, list):
                        pred=pred[0]

            f1 = f1_score(pred, gt)
            em = exact_match_score(pred, gt)
            if modality == 'image':
                img_f1.append(f1), img_em.append(em)
            elif modality == 'text':
                text_f1.append(f1), text_em.append(em)
            else:
                table_f1.append(f1), table_em.append(em)
        img_f1, img_em = np.array(img_f1), np.array(img_em)
        text_f1, text_em = np.array(text_f1), np.array(text_em)
        table_f1, table_em = np.array(table_f1), np.array(table_em)

        return np.mean(img_f1), np.mean(img_em), \
            np.mean(text_f1), np.mean(text_em), \
            np.mean(table_f1), np.mean(table_em)
def modality_eval_mmqa(file_path, aux_file_path):
    with open(aux_file_path, 'r') as f:
        lines = f.readlines()
    data = load_json(file_path)
    single_modality_f1, single_modality_em = [], []
    multi_modality_f1, multi_modality_em = [], []
    for line, qid in zip(lines, data):
        jdata = json.loads(line)
        modality = jdata['metadata']['modalities']
    
        if 'generated_answer' in data[qid]:
            gt = data[qid]['answer']
            pred = data[qid]['generated_answer']
            if isinstance(pred, list):
                pred=pred[0]
        elif 'final_answer' in data[qid]:
            gt = data[qid]['gold_answer']
            pred = data[qid]['final_answer']
            if isinstance(pred, list):
                if len(pred)==0:
                    pred=''
                else:
                    pred=pred[0]    
        elif 'gpt_answer' in data[qid]:
            gt = data[qid]['answer']
            pred = data[qid]['gpt_answer']

        f1 = f1_score(pred, gt)
        em = exact_match_score(pred, gt)
        if len(modality) == 1:
            single_modality_f1.append(f1), single_modality_em.append(em)
        else:
            multi_modality_f1.append(f1), multi_modality_em.append(em)
    single_modality_f1, single_modality_em = np.array(single_modality_f1), np.array(single_modality_em)
    multi_modality_f1, multi_modality_em = np.array(multi_modality_f1), np.array(multi_modality_em)
    return np.mean(single_modality_f1), np.mean(single_modality_em), \
        np.mean(multi_modality_f1), np.mean(multi_modality_em)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_file",
        type=str,
        help="Path to file(s) with training data",
    )
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args=parse_args()
    f_path = args.target_file
    if "mmcoqa" in f_path:
        args.dataset="mmcoqa"
        aux_path = os.path.join(config["DATASET_DIR"],config["DATASET_CONFIG"]["MMCOQA"]["test"])
    elif "mmqa" in f_path:
        args.dataset="mmqa"
        aux_path = os.path.join(config["DATASET_DIR"],config["DATASET_CONFIG"]["MMQA"]["dev"])
        
    f1, em = evaluate(f_path)
    print(f'Overall results for {f_path}')
    print(f"F1 score: {f1:.4}")
    print(f"EM score: {em:.4}")
    print('--------------------')
    if args.dataset=="mmcoqa":
        img_f1, img_em, text_f1, text_em, table_f1, table_em =\
            modality_eval_mmcoqa(f_path, aux_path)
        print('Img:')
        print(f"F1 score: {img_f1:.4}")
        print(f"EM score: {img_em:.4}")
        print('--------------------')
        print('Text:')
        print(f"F1 score: {text_f1:.4}")
        print(f"EM score: {text_em:.4}")
        print('--------------------')
        print('Table:')
        print(f"F1 score: {table_f1:.4}")
        print(f"EM score: {table_em:.4}")
        # print(f"Latex output: {round(img_f1,2)*100:.4} & {  :.4} & {table_f1:.4} & {table_em:.4} &  {text_f1:.4} & {text_em:.4} & {f1:.4} & {em:.4}")
        print(f"Latex output: {round(img_f1,3)*100:.4} & {round(img_em,3)*100:.4} & {round(table_f1,3)*100:.4} & {round(table_em,3)*100:.4} & {round(text_f1,3)*100:.4} & {round(text_em,3)*100:.4} & {round(f1,3)*100:.4} & {round(em,3)*100:.4}")
    elif args.dataset=="mmqa":
        single_modality_f1, single_modality_em, multi_modality_f1, multi_modality_em =\
            modality_eval_mmqa(f_path, aux_path)
        print('Single modality:')
        print(f"F1 score: {single_modality_f1:.4}")
        print(f"EM score: {single_modality_em:.4}")
        print('--------------------')
        print('Multi modality:')
        print(f"F1 score: {multi_modality_f1:.4}")
        print(f"EM score: {multi_modality_em:.4}")
        print(f"Latex output: {round(single_modality_f1,3)*100:.4} & {round(single_modality_em,3)*100:.4} & {round(multi_modality_f1,3)*100:.4} & {round(multi_modality_em,3)*100:.4} & {round(f1,3)*100:.4} & {round(em,3)*100:.4}")
    