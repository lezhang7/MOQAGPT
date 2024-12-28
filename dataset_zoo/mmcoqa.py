import sys
sys.path.append('../')
import json
import os
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


import logging
import numpy as np
from io import open


from PIL import Image


# Required by XLNet evaluation method to compute optimal threshold (see write_predictions_extended() method)
# from utils_squad_evaluate import find_all_best_thresh_v2, make_qid_to_has_ans, get_raw_scores

logger = logging.getLogger(__name__)
DATA_SPLIT=["train","test"]

from utils.utils import *

class MmcoqaDataset(Dataset):
    dataset_name="MMCoQA"
    def __init__(self, datasplit:Optional[str] = "test",
                 history_num=0, prepend_history_questions=False, 
                 prepend_history_answers=False,
                 include_first_for_retriever=False,load_small=False,only_modality=None):
        if only_modality is not None:
            assert only_modality in ["text","image","table"],f"only_modality must a text in ['text','image','table']"
        self.only_modality=only_modality
        self.config=read_config()
        self.dataset_config=self.config["DATASET_CONFIG"][self.dataset_name]
        assert datasplit in DATA_SPLIT, f"filename must in {DATA_SPLIT}"
        self._filename=os.path.join(self.config["DATASET_DIR"],self.dataset_config[datasplit])
        self.load_small=load_small
        self._history_num = history_num
        self._prepend_history_questions = prepend_history_questions
        self._prepend_history_answers = prepend_history_answers    
        self._include_first_for_retriever = include_first_for_retriever
           
        self._query_list=self.read_query_list()
        self._images_dict=self.read_image_dict()
        self._text_dict=self.read_text_dict()
        self._tables_dict=self.read_table_dict()
        self.info()
                
    def __len__(self):
        return len(self._query_list)
                
    def __getitem__(self, idx):
        """read a line of preprocessed open-retrieval quac file into a quac example"""
        
        entry = self._query_list[idx]
        qas_id = entry["qid"]
        
        return_feature_dict = {}
        
    
        
        orig_question_text = entry["question"]
        history = entry['history']
        question_text_list = []
        if self._history_num > 0:
            for turn in history[- self._history_num :]:
                if self._prepend_history_questions:
                    question_text_list.append(turn['question'])
                if self._prepend_history_answers:
                    question_text_list.append(turn['answer'][0]['answer'])
        question_text_list.append(orig_question_text)
        question_text = ' [SEP] '.join(question_text_list)
        question_text_for_retriever = question_text

        # include the first question in addition to history_num for retriever (not reader)
        if self._include_first_for_retriever and len(history) > 0:
            first_question = history[0]['question']
            if first_question != question_text_list[0]:                    
                question_text_for_retriever = first_question + ' [SEP] ' + question_text
                
        query_feature_dict = {'qid': qas_id }
        query_feature_dict['question_text_for_retriever'] = question_text_for_retriever
        query_feature_dict['golden_question_text'] = entry["gold_question"]
        query_feature_dict['original_question_text'] = entry["question"]
        query_feature_dict['answer_text'] = entry['answer'][0]['answer']
        return_feature_dict.update(query_feature_dict)
        return_feature_dict.update({'reference_modality':entry['question_type']})
        if entry['question_type']=="text":  
            text_id=entry['answer'][0]['text_instances'][0]['doc_id']
            text = self._text_dict[text_id]
            example_id = '{}_{}'.format(qas_id, text_id)
            text_feature_dict = {
                                'example_id': example_id,
                                'reference_id':text_id,
                                'reference': text,
                                }
            return_feature_dict.update(text_feature_dict)
        elif entry['question_type']=="image":               
            image_id=entry['answer'][0]['image_instances'][0]['doc_id']
            image_path = self._images_dict[image_id]
            img=self.read_image(image_path)
            example_id = '{}_{}'.format(qas_id, image_id)
            text_feature_dict = {
                                'example_id': example_id,
                                'reference_id':image_id,
                                'reference':img}
            return_feature_dict.update(text_feature_dict)

        else:
            table_id=entry['table_id']
            table = self._tables_dict[table_id]

            example_id = '{}_{}'.format(qas_id, table_id)
            text_feature_dict = {
                                'example_id': example_id,
                                'reference_id':table_id,
                                'reference':table}
            return_feature_dict.update(text_feature_dict)

        return return_feature_dict
    def info(self):
        dataset_information={
            "dataset_name":self.dataset_name,
            "#query":len(self._query_list),
            "#text refences":len(self._text_dict),
            "#text average length":int(np.mean([len(text) for text in self._text_dict.values()])),
            "#images refences":len(self._images_dict),
            "#tables refences":len(self._tables_dict),
            "#tables average length":int(np.mean([len(text) for text in self._tables_dict.values()]))
        }
        print("-"*15,"BASIC DATASET INFO","-"*15)
        for key,value in dataset_information.items():
            print("{:25}| {:>23}".format(key,value))
        print("-"*50)
    def read_query_list(self):
        query_list = []
        with open(self._filename, 'r') as f:
            for line in f:
                content=json.loads(line)
                if self.only_modality is not None and content['answer'][0]["modality"] != self.only_modality:
                    continue
                query_list.append(content)
                if self.load_small and len(query_list) > 1000:
                    break
        return query_list
    def read_image(self, path):  ###下面还有一个read_image方法，要改的话不要忘记统一修改
        if not os.path.exists(path):
            image_id=path.split('.')[0]
            path=self._images_dict[image_id]
        img = Image.open(path).convert("RGB")
        if 1 in img.size:
            raise ValueError("Image size is 1: {}".format(path))
        return img
    def read_table_dict(self):
        tables_dict={}
        table_path=os.path.join(self.config["DATASET_DIR"],self.dataset_config["table"])
        with open(table_path,'r') as f:
            lines=f.readlines()
            for line in lines:
                line=json.loads(line.strip())
                table_context = ''
                for row_data in line['table']["table_rows"]:
                    for cell in row_data:
                        table_context=table_context+" "+cell['text']
                tables_dict[line['id']]=table_context
        return tables_dict
    def read_text_dict(self):
        text_dict={}
        text_path=os.path.join(self.config["DATASET_DIR"],self.dataset_config["text"])
        with open(text_path,'r') as f:
            lines=f.readlines()
            for line in lines:
                line=json.loads(line.strip())
                text_dict[line['id']]=line['text']
        return text_dict
    def read_image_dict(self):
        images_dict={}
        image_path=os.path.join(self.config["DATASET_DIR"],self.dataset_config["image"])
        with open(image_path,'r') as f:
            lines=f.readlines()
            for line in lines:
                line=json.loads(line.strip())
                images_dict[line['id']]=os.path.join(self.config["DATASET_DIR"],self.dataset_config["image_dir"],line['path'])
        return images_dict

        
    

        
if __name__=="__main__":
    mmcoqa=MmcoqaDataset('test')