import sys
sys.path.append('../')
sys.path.append('../../')



def get_embedding_model(model_name):
    if model_name=="clip":
        from model_zoo.image.clip import CLIP_FOR_FEATURE_EXTRACTION
        model=CLIP_FOR_FEATURE_EXTRACTION()
        return model

    elif model_name=="ada":
        from model_zoo.text.ada import ADA_FOR_FEATURE_EXTRACTION
        model=ADA_FOR_FEATURE_EXTRACTION()
        return model
    else:
        from model_zoo.text.text_encoder import SentenceEmbedding
        model=SentenceEmbedding(model_name)
        return model

def get_answer_model(model_name):
    if  model_name=="blip2":
        from model_zoo.image.VQA_models import BLIP2_FOR_QA
        model=BLIP2_FOR_QA(model_name)
        return model
    elif model_name=="instructblip":
        from model_zoo.image.VQA_models import BLIP2_FOR_QA
        model=BLIP2_FOR_QA(model_name)
        return model
    elif model_name=="chatgpt":
        from model_zoo.text.QA_models import ChatGPT
        model=ChatGPT(model_name="gpt-3.5-turbo")
        return model
    elif model_name=="gpt4":
        from model_zoo.text.QA_models import ChatGPT
        model=ChatGPT(model_name="gpt-4")
        return model
    elif model_name=="vicuna":
        from model_zoo.text.QA_models import Llama
        model=Llama("eachadea/vicuna-7b-1.1")
        return model
    elif model_name=="llama2":
        from model_zoo.text.QA_models import Llama
        model=Llama("meta-llama/Llama-2-13b-hf")
        return model
    elif model_name=="llama2chat":
        from model_zoo.text.QA_models import Llama
        model=Llama("meta-llama/Llama-2-13b-chat-hf")
        return model
    elif model_name=="openchat":
        from model_zoo.text.QA_models import Llama
        model=Llama("openchat/openchat_v2_w")
        return model

    else:
        raise NameError(f"model_name {model_name} not implemented")

        
    