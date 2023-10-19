import sys
sys.path.insert(0, '..')
import torch
import numpy as np
import json
import linecache
from tqdm import tqdm
import os
from PIL import Image
from typing import List,Tuple,Dict,Union
from utils.utils import *
import pytrec_eval
from dataset_zoo import *

class MultimodalRetriever():
    """
    Given a question id or tensor embedding, retrieve topk resources of each modality
    """
    def __init__(self,dataset_name:str,features_model_dict):
        """
        dataset_name MMCOQA
        features_model_dict {"image":"clip","table":"ada":"passage":"bm25"}
        """
        dataset_name=dataset_name.upper()
        self.config=read_config()
        assert dataset_name in self.config["DATASET_CONFIG"].keys(),f"dataset_name {dataset_name} not found in config"
        self.dataset_name=dataset_name
       
        # sanity check if selected model have its features stored
        dataset_features_model=[]
        for modality in self.config['FEATURES_CONFIG'][dataset_name]:
            dataset_features_model.extend(list(self.config['FEATURES_CONFIG'][dataset_name][modality].keys()))
        if any([model_name not in dataset_features_model for model_name in list(features_model_dict.values()) ]):
            raise ValueError(f"selected models {list(features_model_dict.values())} must have its features stored, however features from model {[model for model in list(features_model_dict.values()) if model not in dataset_features_model]} is found")
        self.qrels=self.load_qrels()
        self.features_model_dict=features_model_dict
        self.ref_features=self.load_model_ref_features()  
        self.query_features=self.load_model_query_features()
        print("load feautres done")
        self.references=self.load_references()
        print("load references done")
    def load_qrels(self):
        qrel_file_path=os.path.join(self.config["DATASET_DIR"],self.config['DATASET_CONFIG']["MMCOQA"]["qrel"])
        with open(qrel_file_path, 'r') as g:
            qrel = g.readlines()
        qrel = json.loads(qrel[0])
        return qrel
    def load_model_ref_features(self):
        # loading Reference Embedding of each modality from specified models
        features_dict={}
        for modality in list(self.features_model_dict.keys()):
            ReferenceEmbedding_path=os.path.join(self.config['PROJECT_DIR'],self.config['FEATURES_CONFIG'][self.dataset_name][modality][self.features_model_dict[modality]]["ReferenceEmbedding"])
            ReferenceEmbedding=torch.load(ReferenceEmbedding_path)
            features_dict[modality]=ReferenceEmbedding
        return features_dict
    def load_model_query_features(self):
        # loading Reference Embedding of each modality from specified models
        features_dict={}
        for modality in list(self.features_model_dict.keys()):
            ReferenceEmbedding_path=os.path.join(self.config['PROJECT_DIR'],self.config['FEATURES_CONFIG'][self.dataset_name][modality][self.features_model_dict[modality]]["QuestionEmbedding"])
            ReferenceEmbedding=torch.load(ReferenceEmbedding_path)
            features_dict[modality]=ReferenceEmbedding
        return features_dict

    def load_references(self):
        # loading Reference for each modality, including passage, table, raw image (.jsonl file)
        references={}
        for modality in list(self.features_model_dict.keys()):
            reference={}
            reference_path=os.path.join(self.config["DATASET_DIR"],self.config['DATASET_CONFIG']["MMCOQA"][modality])
            with open(os.path.join(reference_path),'r') as f:
                lines=f.readlines()
                for line in lines:
                    line=json.loads(line.strip())
                    if 'path' in line:
                        reference[line['id']]=line['path']
                    elif 'table' in line:
                        table_context = ''
                        for row_data in line['table']["table_rows"]:
                            for cell in row_data:
                                table_context=table_context+" "+cell['text']
                        reference[line['id']]=table_context
                    elif 'text' in line:
                        reference[line['id']]=line['text']
                    # idx_id_list.append((line['id'],'image'))
                references[modality]=reference
        return references
    @property
    def num_features(self):
        count={}
        for modality in self.features:
            count[modality]=len(self.features[modality]["features"])
        return count
    @property
    def num_evidence(self):
        count={}
        for modality in self.references:
            count[modality]=len(self.references[modality])
        return count
      
    def retrieve_from_single_modality(self,query_features,qids,ref_features,ref_ids,topk):
        sim_matrix = query_features @ ref_features.T
        rels, indices = torch.topk(sim_matrix, topk)
        res = {}
        for i, qid in enumerate(qids):
            topk_res = {}
            for rel, idx in zip(rels[i], indices[i]):
                data_id = ref_ids[idx]
                topk_res[data_id] = rel.item()
            res[qid] = topk_res
        return res
    def evaluate_single_modality(self, retrieved_results, valid_id: dict):
        """

            valid_id: used to filter retrival results; if set to none, no filter
        """
        def filter(retrieved_results: dict, valid_id: dict) -> dict:
            output = {}
            for i in valid_id:
                if i in retrieved_results: output[i] = retrieved_results[i]
            return output
    
        if valid_id:
            retrieved_results = filter(retrieved_results, valid_id)
        evaluator = pytrec_eval.RelevanceEvaluator(self.qrels, {'recall', 'ndcg', 'ndcg_cut.3', 'recip_rank'})
        res = evaluator.evaluate(retrieved_results)

        mrr_list = [v['recip_rank'] for v in res.values()]
        ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]
        ndcg_list = [v['ndcg'] for v in res.values()]
        recall_5_list = [v['recall_5'] for v in res.values()]
        recall_10_list = [v['recall_10'] for v in res.values()]

        final_res = {
            "MRR": np.average(mrr_list),
            "NDCG@3": np.average(ndcg_3_list),
            "NDCG": np.average(ndcg_list),
            "Recall@5": np.average(recall_5_list),
            "Recall@10": np.average(recall_10_list)
                    }
        return final_res
    def evaluate(self, retrieved_results):
        with open(os.path.join(self.config["DATASET_DIR"],self.config["DATASET_CONFIG"][self.dataset_name]["test" if self.dataset_name=="MMCOQA" else "dev"]),"r") as f:
            lines=f.readlines()
        modality_corresponding_id={}
        for line in lines:
            line=json.loads(line.strip())
            if line['question_type'] not in modality_corresponding_id:
                modality_corresponding_id[line['question_type']]={}
            modality_corresponding_id[line['question_type']][line['qid']]=0
        for modality in retrieved_results:
            retrieved_result=retrieved_results[modality]
            all_modality_qids=list(modality_corresponding_id[modality].keys())
            modality_id={qid:0 for qid in list(retrieved_results[modality].keys()) if qid in all_modality_qids}
            if modality_id:
                evaluation_result=self.evaluate_single_modality(retrieved_result,modality_id)
                print(f"Modality: {modality}: #{len(modality_id)}")
             
                print(evaluation_result)
    def retrieve(self,query_id: List[str],topk=50) -> List[List[str]] :
        """
        Given a query, retrieve its topk resource from each modalities and return document ids, the query must in id_to_feature_idx
        Input
            - query: list of input document id
        Output
            - retrieved_reference
                {
                    qid: { 
                        modality: [reference1,reference2,...]}
                }
            - retrieved_results
                {
                    modality: {
                        qid:    {
                            ref_id: score}
                            }
                }
                    
        """
        retrieved_results={}
        retrieved_reference={}
        assert isinstance(query_id,list), "query must be a list of string"
        for modality in list(self.query_features.keys()):
            query_features=torch.stack([self.query_features[modality][q] for q in query_id])
            ref=self.ref_features[modality]
            ref_features=torch.stack(list(ref.values()))
            ref_ids=list(ref.keys())
            retrieved_results[modality]=self.retrieve_from_single_modality(query_features,query_id,ref_features,ref_ids,topk)

        for q in query_id:
            retrieved_reference[q]={}
            for modality in list(self.query_features.keys()):
                retrieved_ids=list(retrieved_results[modality][q].keys())
                retrieved_reference[q][modality]=[self.references[modality][rid] for rid in retrieved_ids]
        return retrieved_results,retrieved_reference
    def retrieve_from_given_list(self,dataset,query_id: List[str],topk=5):
        dataset_dict={x["qid"]:x for x in dataset}
        retrieved_reference={}
        assert isinstance(query_id,list), "query must be a list of string"
        for qid in tqdm(query_id):
            retrieved_reference[qid]={}
            for modality in list(self.query_features.keys()):
                query_features=self.query_features[modality][qid]
                if len(dataset_dict[qid][modality]):
                    ref_features=torch.stack([self.ref_features[modality][ref_id] for ref_id in dataset_dict[qid][modality]])               
                    ref_ids=list(dataset_dict[qid][modality].keys())
                    rels, indices = torch.topk(query_features@ref_features.T, min(topk,ref_features.shape[0]))
                    retrieved_reference[qid][modality]=[self.references[modality][ref_ids[idx]] for idx in indices]
        return retrieved_reference




# if __name__=="__main__":
    # features_model_dict={"image":"clip","table":"ada"}
    # retriever=MultimodalRetriever("MMCOQA",features_model_dict)
    # result,re=retriever.retrieve(['C_123_2',"C_154_2"],3)
    # print(retriever.evaluate(result))