# **MoqaGPT: Zero-Shot Multi-modal Open-domain Question Answering with Large Language Models (EMNLP2023 Findings)**

<div style="text-align: center; margin-top: 2rem;">
    <a href="https://aclanthology.org/2023.findings-emnlp.85/" target="_blank" style="margin: 0 10px;">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-SAIL-red?logo=arxiv" height="30" />
    </a>
</div>

***TL;DR***: We propose a framework leveraging Large Language Models to solve multi-modal open-domian question answerings (moqa) in a **zero-shot manner**, this is very different from previous supervised paradigm, instead we do not require any domain-specific annotations and modules, posing our framework as a general paradigm for moqa tasks. 

![paradigmdifference](https://p.ipic.vip/hfn7wj.png)

The method framework tackles each modality seperately, and fuse results by Large Language models, the '*devide-and-conquer'* design circumvent complex multimodal information retrieval obstacle and flexible enough to incorporate any new modalities and stronger models into the framework to achieve better performance.

![](https://p.ipic.vip/kj67o3.png)

## Preparation

```
pip install -r requirements.txt
```

### Extract features

1. Download features at :hugs:https://huggingface.co/datasets/le723z/MOQAGPT_FEAT

2. Generate features by customized models from huggingface transformers, such as `[Alibaba-NLP/gte-large-en-v1.5/Alibaba-NLP/gte-Qwen2-1.5B-instruct]`, please refer to 3 files in `extract/`.  

   ``````bash
   # extract table and queries emebddings using CLIP model --dataset mmcoqa
   python extract/extract_image_features.py 
   
   # extract table and queries emebddings
   python extract/extract_table_features.py --dataset mmcoqa --model [Huggingface Model]
   
   # extract passages and queries emebddings
   python extract/extract_text_features.py --passages_file_path [] --query_file_path --model [Huggingface Model]
   
   ``````

:warning: Please make sure that `FEATURES_CONFIG` in `utils/config.yaml`  points to the correct path 

``````yaml
FEATURES_CONFIG:
  MMCOQA:
    image:
      clip:
        QuestionEmbedding: stored_features/mmcoqa/image/clip_features/QuestionEmbedding.pt
        ReferenceEmbedding: stored_features/mmcoqa/image/clip_features/ReferenceEmbedding.pt
``````

## Quick Retrieval Example

```python
from pipeline.retriever import MultimodalRetriever
features_model_dict={"image":"clip","table":"ada","text":"gte-base-en-v1.5"}
retriever=MultimodalRetriever("MMCOQA",features_model_dict)
question_ids=["C_226_0","C_226_1"]
retrieved_results,retrieved_reference=retriever.retrieve(question_ids) 

# retrieved_results {qid: {modality: [ref1_id: score, ref2_id: score]}}
# retrieved_reference {qid: {modality: [ref1_content, ref2_content]}}

```

## Run MOQAGPT 

1. Get `direct_QA` using LLM in `['openchat', 'gpt-4', 'llama2chat']` for DATA in `['mmqa', 'mmcoqa']`, the answers will be saved at for example  `MOQAGPT/output/$DATA/direct_chatgpt.json`

   ``````bash
   python pipeline/direct_qa.py --dataset $DATA --direct_qa_model $LLM
   ``````

2. Get answers for query of references from various modality. The answers will be saved at for example  `MOQAGPT/output/$DATA/direct_chatgpt.json`

   ```bash
   python pipeline/answerer.py --dataset $DATA --text_qa $LLM --table_qa $LLM
   ```

3. Ensemble multiple answers from various sources and reason the final answer for the query. The results will be saved at for example `output/mmqa/candidates/chatgpt/Iblip2_Tllama2chatTabllama2chat_direct_chatgpt.json`

   ```bash
   python pipeline/strategy.py --reasoner $LLM \
           --textual_qa ~/scratch/MOQA/output/mmqa/Tllama2chatTabllama2chat.json \
           --visual_qa ~/scratch/MOQA/output/mmqa/Iblip2.json \
           --direct_qa ~/scratch/MOQA/output/mmqa/direct_chatgpt.json
   ```

4. Evaluation results

   ```bash
   python pipeline/evaluation.py --target_file output/mmqa/candidates/chatgpt/Iblip2_Tllama2chatTabllama2chat_direct_chatgpt.json
   ```

# Citation

```
@inproceedings{zhang-etal-2023-moqagpt,
    title = "{M}oqa{GPT} : Zero-Shot Multi-modal Open-domain Question Answering with Large Language Model",
    author = "Zhang, Le  and
      Wu, Yihong  and
      Mo, Fengran  and
      Nie, Jian-Yun  and
      Agrawal, Aishwarya",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.85",
    doi = "10.18653/v1/2023.findings-emnlp.85",
    pages = "1195--1210",
    abstract = "Multi-modal open-domain question answering typically requires evidence retrieval from databases across diverse modalities, such as images, tables, passages, etc. Even Large Language Models (LLMs) like GPT-4 fall short in this task. To enable LLMs to tackle the task in a zero-shot manner, we introduce MoqaGPT, a straightforward and flexible framework. Using a divide-and-conquer strategy that bypasses intricate multi-modality ranking, our framework can accommodate new modalities and seamlessly transition to new models for the task. Built upon LLMs, MoqaGPT retrieves and extracts answers from each modality separately, then fuses this multi-modal information using LLMs to produce a final answer. Our methodology boosts performance on the MMCoQA dataset, improving F1 by +37.91 points and EM by +34.07 points over the supervised baseline. On the MultiModalQA dataset, MoqaGPT surpasses the zero-shot baseline, improving F1 by 9.5 points and EM by 10.1 points, and significantly closes the gap with supervised methods. Our codebase is available at https://github.com/lezhang7/MOQAGPT.",
}
```