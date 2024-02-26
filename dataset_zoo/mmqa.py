import sys
sys.path.append('../')
import json
import os
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


import logging
import numpy as np
from io import open


from PIL import Image
from utils.utils import *

DATA_SPLIT=["train","test","dev"]
DATA_SPLIT=["train","test","dev"]
class MMQA(Dataset):
    dataset_name="MMQA"
    
    def __init__(self, datasplit:Optional[str] = "test",load_small=False,only_modality=None):
        if only_modality is not None:
            assert only_modality in ["text","image","table"],f"only_modality must a text in ['text','image','table']"
        self.only_modality=only_modality
        self.datasplit=datasplit
        self.config=read_config()
        self.dataset_config=self.config["DATASET_CONFIG"][self.dataset_name]
        assert datasplit in DATA_SPLIT, f"filename must in {DATA_SPLIT}"
        self._filename=os.path.join(self.config["DATASET_DIR"],self.dataset_config[datasplit])
        self.load_small=load_small
        self._query_list=self.read_query_list()
        self._images_dict=self.read_image_dict()
        self._text_dict=self.read_text_dict()
        self._tables_dict=self.read_table_dict()
        self.info()
    def __len__(self):
        return len(self._query_list)
    def __getitem__(self, idx):
        """get example 
        {qid:qid,question:question,answer:answer,question_type:type,
        image:[img_path1,...],text:[text1,...],table:[table1,...]}}
        """
        
        entry = self._query_list[idx]
        qas_id = entry["qid"]
        
        return_feature_dict = {}
        return_feature_dict["qid"]=qas_id
        return_feature_dict["golden_question_text"]=entry["question"]
        return_feature_dict['answer_text'] = entry['answers'][0]['answer']
        return_feature_dict['reference_modality'] = [x["doc_part"] for x in entry["supporting_context"]]
        return_feature_dict['reference_id'] = entry['supporting_context'][0]['doc_id']
        
        meta_data = entry['metadata']
        if "image_doc_ids" in meta_data:
            doc_dict={}
            image_id=meta_data["image_doc_ids"]
            if isinstance(image_id,str):
                image_id=[image_id]
            for doc_id in image_id:
                doc_dict[doc_id]=self._images_dict[doc_id]
            return_feature_dict["image"]=doc_dict
        if "text_doc_ids" in meta_data:
            doc_dict={}
            text_id=meta_data["text_doc_ids"]
            if isinstance(text_id,str):
                text_id=[text_id]
            for doc_id in text_id:
                doc_dict[doc_id]=self._text_dict[doc_id]
            return_feature_dict["text"]=doc_dict
        if "table_id" in meta_data:
            doc_dict={}
            table_id=meta_data["table_id"]
            if isinstance(table_id,str):
                table_id=[table_id]
            for doc_id in table_id:
                doc_dict[doc_id]=self._tables_dict[doc_id]
            return_feature_dict["table"]=doc_dict
        return return_feature_dict
    def info(self):
        dataset_information={
            "dataset_name":self.dataset_name+"_"+self.datasplit,
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
                text_dict[line['id']]=line['title']+line['text']
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
    
    