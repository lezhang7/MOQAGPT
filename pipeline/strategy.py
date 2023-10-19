import sys
sys.path.insert(0, '..')
from utils.utils import *
from model_zoo import get_answer_model
from tqdm import tqdm
from typing import List,Dict,Any
import argparse
class Strategy():
   def __init__(self,dataset_name,model_name="chatgpt"):
       self.model_name=model_name
       self.answerer=get_answer_model(model_name)
       self.config= read_config()
       self.dataset_name=dataset_name
   def clean(self,answers:List[str]):
      if isinstance(answers,str):
         answers=[answers]
      answers=[normalize_answer(answer) for answer in answers]
      answers=[answer for answer in answers if 'unk' not in answer and "none" not in answer]
      if not answers:
         return []
      if len(Counter(answers))!=len(answers):
         answers=[list(Counter(answers).keys())[0]]
      else:
         answers=[answers[0]]
      return answers
   def answer_type(self,answer):
     if len(answer)>10:
         return "string"
     else: 
         try:
            int(answer)
            return "int"
         except:
            pass
         try:
            float(answer)
            return "float"
         except:
            pass
         return "string"

   def get_candidates(self,raw_results,output_file,write_to_local=False):
      gpt_answer_type="gpt_answer"
      cleaned_results={}
      os.makedirs(os.path.dirname(output_file),exist_ok=True)
      if os.path.exists(output_file):
         cleaned_results=load_json(output_file)
      for qid in tqdm(raw_results):
         if qid in cleaned_results:
               print(f"qid {qid} already in final_answers")
               continue
         gpt_answer=raw_results[qid][gpt_answer_type]
         answer_type=self.answer_type(gpt_answer)
         cleaned_results[qid]={}
         question=raw_results[qid]["question"]
         cleaned_results[qid]["question"]=question
         final_answers=[]
         if self.dataset_name=="mmqa":
            if 'image' in raw_results[qid]:
               final_answers.extend(self.clean(raw_results[qid]['image'][:1]))
            final_answers.extend(self.clean(raw_results[qid]['table'][:5]))
            final_answers.extend(self.clean(raw_results[qid]['text'][:5]))
         elif self.dataset_name=="mmcoqa":
            final_answers.extend(self.clean(raw_results[qid]['image'][:5]))
            final_answers.extend(self.clean(raw_results[qid]['table'][:5]))
            final_answers.extend(self.clean(raw_results[qid]['text'][:5]))
         cleaned_gpt_answer=self.clean([gpt_answer])
         for i in final_answers:
            if len(cleaned_gpt_answer) and cleaned_gpt_answer[0] in i:
               cleaned_results[qid]["final_answer"]=cleaned_gpt_answer
         if answer_type!="float" :
            
            if len(cleaned_gpt_answer) and 'un' not in cleaned_gpt_answer[0]:
               final_answers.extend(cleaned_gpt_answer)
         cleaned_results[qid]["gold_answer"]=normalize_answer(raw_results[qid]['gold_answer'])
         cleaned_results[qid]["answer_candidates"]=final_answers
         cleaned_results[qid][gpt_answer_type]=self.clean([gpt_answer])
         prompt=f"Given question '{question}', please select the best answer from the following candidates: '{final_answers}', do not give additional sentence or explanations, just answer word: "
         cleaned_results[qid]["prompt1"]=prompt
         final_answer=self.answerer.get_answer(prompt)
         if len(s.clean(final_answer)) and any([x in s.clean(final_answer)[0] for x in ["none","sorry","unable","can't"]]):
            final_answer="Can't answer"
         if len(final_answer.split(" "))>3:
            prompt=f"Given question '{question}', please extract answer span from '{final_answer}', do not give additional sentence or explanations, just answer word:"
            cleaned_results[qid]["response1"]=final_answer
            final_answer=self.answerer.get_answer(prompt)  
            cleaned_results[qid]["prompt2"]=prompt
         cleaned_results[qid]["final_answer"]=self.clean([final_answer])
         if write_to_local:
            read_then_save_json(output_file,cleaned_results)
      return cleaned_results

         
def parse_args():
   parser = argparse.ArgumentParser()

   parser.add_argument(
        "--reasoner",
        type=str,
        required=True,
        choices=["chatgpt","gpt4","vicuna"]
    )
   parser.add_argument(
         "--direct_qa",
         type=str,
         required=True,
         help="Path to raw results",
   )
   parser.add_argument(
         "--textual_qa",
         type=str,
         required=True,
         help="Path to raw results",
   )
   parser.add_argument(
         "--visual_qa",
         type=str,
         required=True,
         help="Path to raw results",
   )
   args = parser.parse_args()
   if 'mmcoqa' in args.visual_qa:
      args.dataset="mmcoqa"
   elif 'mmqa' in args.visual_qa:
      args.dataset="mmqa"
   return args
       
            


         
        
       

if __name__ == "__main__":
   args=parse_args()
   s=Strategy(dataset_name=args.dataset,model_name=args.reasoner)
   image_json=load_json(args.visual_qa)
   text_table_json=load_json(args.textual_qa)
   direct_json=load_json(args.direct_qa)
   candidates=merge_json(image_json,text_table_json)
   candidates=merge_json(candidates,direct_json)
   file_name=f"{args.visual_qa.split('/')[-1].replace('.json','')}_{args.textual_qa.split('/')[-1].replace('.json','')}_{args.direct_qa.split('/')[-1].replace('.json','')}.json"
 
   output_file=os.path.join(config['PROJECT_DIR'],'output',args.dataset,"candidates",args.reasoner,file_name)
   print(output_file)
   a=s.get_candidates(candidates,output_file,write_to_local=True)