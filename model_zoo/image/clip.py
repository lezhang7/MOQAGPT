from transformers import CLIPProcessor, CLIPModel
import torch
from typing import List,Dict,Any,Union,Tuple
from tqdm import tqdm
import os
class CLIP_FOR_FEATURE_EXTRACTION():
    def __init__(self,model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    def extract_image_features(self,dataset,images_ref:List[Tuple[str, str]],bs=2048,save_path=None):
        """
        images_ref: List of tuples (image_id, image_path)
        """
        image_dict={}
        for i in tqdm(range(len(images_ref)//bs+1)):
            images_ref_batch=images_ref[i*bs:(i+1)*bs]
            images_batch=[]
            image_ids=[]
            for image_id, image_path in images_ref_batch:
                try:
                    images_batch.append(dataset.read_image(image_path))
                    image_ids.append(image_id)
                except:
                    print(f"{image_id} is corrupted")
                    continue
            inputs = self.processor(images=images_batch, return_tensors="pt", padding=True).to(self.device)
            # model.get_image_features(**inputs)
            
            with torch.no_grad():
                features_image = self.model.get_image_features(**inputs)
                features_image=features_image/features_image.norm(p=2, dim=-1, keepdim=True)
                features_image_cpu=features_image.cpu()
                del features_image
                torch.cuda.empty_cache()
            for j in range(len(image_ids)):
                image_dict[image_ids[j]]=features_image_cpu[j]
        if save_path is not None:
            os.makedirs(save_path,exist_ok=True)
            torch.save(image_dict,os.path.join(save_path,"ReferenceEmbedding.pt"))
        return image_dict
    def extract_text_features(self,queries:List[Tuple[str, str]],bs=2048,save_path=None):
        query_dict={}
        bs=128
        for i in tqdm(range(len(queries)//bs+1)):
            queries_batch=queries[i*bs:(i+1)*bs]
            
            qids=[x[0] for x in queries_batch]
            query=[x[1] for x in queries_batch]
            
            inputs = self.processor(text=query, return_tensors="pt", padding=True,truncation=True).to(self.device)
            # model.get_image_features(**inputs)
            
            with torch.no_grad():
                features_text = self.model.get_text_features(**inputs)
                features_text=features_text / features_text.norm(p=2, dim=-1, keepdim=True)
                features_text_cpu=features_text.cpu()
                del features_text
                torch.cuda.empty_cache()
            for i in range(len(qids)):
                query_dict[qids[i]]=features_text_cpu[i]
        if save_path is not None:
            os.makedirs(save_path,exist_ok=True)
            torch.save(query_dict,os.path.join(save_path,"QuestionEmbedding.pt"))
        return query_dict
        


    