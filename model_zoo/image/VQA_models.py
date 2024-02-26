from PIL import Image
import requests
import torch
from lavis.models import load_model_and_preprocess
from typing import List
# loads BLIP-2 pre-trained model

# prepare the image

import inflect
import string
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Initialize



class BLIP2_FOR_QA():
    def __init__(self,model_name="Salesforce/blip2-flan-t5-xl"):
        # assert model_name in ["Salesforce/blip2-flan-t5-xl","blip2_vicuna_instruct"], f"wrong model_name {model_name}"
        print(f"Loading model {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_name=="blip2":
            self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=self.device)
        elif model_name=="instructblip":
            self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=self.device)
        else:
            raise NameError(f"model_name {model_name} not implemented")
        self.model.to(self.device)
        self.model.eval()
        # self.p = inflect.engine()
        self.lemmatizer = WordNetLemmatizer()
    def get_answer(self,question,image):
        prompt = f"Question: {question} Answer:"
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output=self.model.generate({"image": image, "prompt": prompt},num_beams=5,length_penalty=-1,max_length=10)[0]
        cleaned_outpout=self.clean_answer(output)
        del output
        torch.cuda.empty_cache()
        return cleaned_outpout
    def get_answer_batched(self,questions:List[str],images):
        questions=[f"Question: {q} Answer:" for q in questions]
        images=torch.cat([self.vis_processors["eval"](image).unsqueeze(0).to(self.device) for image in images])
        with torch.no_grad():
            outputs=self.model.generate({"image": images, "prompt": questions},num_beams=5,length_penalty=-1,max_length=10)
        cleaned_outputs=[self.clean_answer(output) for output in outputs]
        del outputs
        torch.cuda.empty_cache()
        return cleaned_outputs
    def get_answer_batched_cot(self,questions:List[str],images):
        # questions=[f"Question: {q} Answer:" for q in questions]
        caption_prompt=["Describe the content in photo with detail: "]*len(images)
        

        images=torch.cat([self.vis_processors["eval"](image).unsqueeze(0).to(self.device) for image in images])
        with torch.no_grad():
            try:
                captions=self.model.generate({"image": images, "prompt": caption_prompt},num_beams=3,length_penalty=-1,max_length=100)
            except Exception as e:
                print(e)
                print(caption_prompt)
                
        cot_prompt=[f"Caption:{caption}\n\nQuestion:{question}\n\n Let's think step by step." for caption,question in zip(captions,questions) ]
        with torch.no_grad():
            inter_meditate_reasonings=self.model.generate({"image": images, "prompt": cot_prompt},num_beams=3,length_penalty=-1,max_length=150)
        with torch.no_grad():
            prompt=[f"Reference: {reasoning}\n\nQuestion: {question}\n\nGive me a very short answer, in one or two words." for reasoning,question in zip(inter_meditate_reasonings,questions)]
            outputs=self.model.generate({"image": images, "prompt":prompt},length_penalty=-1)
        cleaned_outputs=[self.clean_answer(output) for output in outputs]
        del outputs
        torch.cuda.empty_cache()
        return {"caption":captions,"intermediate_reasoning":inter_meditate_reasonings,"answer":cleaned_outputs}
    def clean_answer(self,answer):
        # Convert to lower case
        answer = answer.lower()

        # Remove punctuation
        answer = answer.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize the answer
        tokens = word_tokenize(answer)

        cleaned_tokens = []

        for token in tokens:
            # Apply lemmatization
            lemmatized_token = self.lemmatizer.lemmatize(token)
            cleaned_tokens.append(lemmatized_token)
        
        # Join the tokens back into a string
        cleaned_answer = ' '.join(cleaned_tokens)

        return cleaned_answer

