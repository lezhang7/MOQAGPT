import sys
sys.path.insert(0, '..')
from utils.utils import *
from dataset_zoo import get_dataset
from model_zoo import get_answer_model
from pipeline.retriever import MultimodalRetriever
from tqdm import tqdm
from typing import List,Dict,Any
import argparse


class Answerer():
    def __init__(self,dataset_name:str,modality:Dict[str,str],with_reasoning=False):
        print(f" Multimodal Answerer: {modality}")
        self.dataset_name=dataset_name
        self.with_reasoning=with_reasoning
        if with_reasoning:
            self.file_name=f"{'I'+modality['image']if modality['image'] else ''}{'T'+modality['text'] if modality['text'] else ''}{'Tab'+modality['table'] if modality['table'] else ''}_qar.json"
        else:
            self.file_name=f"{'I'+modality['image']if modality['image'] else ''}{'T'+modality['text'] if modality['text'] else ''}{'Tab'+modality['table'] if modality['table'] else ''}.json"
        self.modality=modality
        self.image_answerer = get_answer_model(modality["image"]) if modality["image"] is not None else None     
        self.table_answerer = get_answer_model(modality["table"]) if modality["table"] is not None else None  
        self.text_answerer = self.table_answerer if modality["text"] is not None else None
        self.text_max_length= 4000 if self.modality['text']== "chatgpt" else 1000
        self.query_dict = get_query_dict(dataset_name)
        self.config= read_config()
    def get_answers(self,retrieved_references,write_to_local=None,save_every_modality_answer=False):
        """
        retrieved_references: dict of dict of list, results from retriever
        {
            qid:
            {
                "passage":[pid1,pid2,...],
                "image":[image_id1,image_id2,...],
                "table":[table_id1,table_id2,...]
            }
        }
        """
        final_answers={}
        if write_to_local:
            output_path=os.path.join(config['PROJECT_DIR'],'output',self.dataset_name,self.file_name)
            if os.path.exists(output_path):
                final_answers=load_json(output_path) 

        for qid in tqdm(retrieved_references):
            if qid in final_answers:
                print(f"qid {qid} already in final_answers")
                continue
            final_answers[qid]={}
            query=self.query_dict[qid]['query']
            final_answers[qid]['question']=query
            final_answers[qid]['gold_answer']=self.query_dict[qid]['gold_answer']
            final_answers[qid]['gold_answer_modality']=self.query_dict[qid]['gold_answer_modality']

            for ref_modality in retrieved_references[qid]:
                if ref_modality == "text" and self.text_answerer is not None:
                    texts=retrieved_references[qid][ref_modality]
                    texts=[t[:self.text_max_length] for t in texts]
                    queries=[query]*len(texts)
                    gpt_results,intermediate_reasoning=self.text_answerer.get_answer_batched(queries,texts,self.with_reasoning)
                    final_answers[qid][ref_modality] = gpt_results
                    if self.with_reasoning:
                        final_answers[qid][f'intermediate_reasoning_{ref_modality}']=intermediate_reasoning
                elif ref_modality == "image" and self.image_answerer is not None:
                    images=[dataset.read_image(image_id) for image_id in retrieved_references[qid][ref_modality]]
                    queries=[query]*len(images)
                    if self.with_reasoning:
                        answers= self.image_answerer.get_answer_batched_cot(queries,images)
                    else:
                        answers=self.image_answerer.get_answer_batched(queries,images)
        
                    final_answers[qid][ref_modality] = answers
                elif ref_modality == "table" and self.table_answerer is not None:
                    texts=retrieved_references[qid][ref_modality]
                    texts=[t[:self.text_max_length] for t in texts]
                    queries=[query]*len(texts)
                    gpt_results,intermediate_reasoning=self.text_answerer.get_answer_batched(queries,texts,self.with_reasoning)
                    final_answers[qid][ref_modality] = gpt_results
                    if self.with_reasoning:
                        final_answers[qid][f'intermediate_reasoning_{ref_modality}']=intermediate_reasoning
            if write_to_local:
                read_then_save_json(output_path,final_answers)
       

        return final_answers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name",
        choices=["mmqa","mmcoqa"]
    )
    parser.add_argument(
        "--text_qa",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--table_qa",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image_qa",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--cot_qa",
        action="store_true",
        help="Whether to use cot for question answering"

    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args=parse_args()
    dataset=get_dataset(args.dataset)
    features_model_dict={"image":"clip","table":"ada","text":"ance"}
    retriever=MultimodalRetriever(args.dataset,features_model_dict)
    if args.dataset=="mmqa":
        retrieved_references=retriever.retrieve_from_given_list(dataset,[x['qid'] for x in dataset],5)
    elif args.dataset=="mmcoqa":
        result,retrieved_references=retriever.retrieve(list(get_query_dict(args.dataset).keys()),5)
    a=Answerer(args.dataset,{"text":args.text_qa,"table":args.table_qa,"image":args.image_qa},with_reasoning=args.cot_qa)
    final_answers=a.get_answers(retrieved_references,write_to_local=True,save_every_modality_answer=True)
   


