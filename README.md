# **MoqaGPT: Zero-Shot Multi-modal Open-domain Question Answering with Large Language Models (EMNLP2023 Findings)**

## Introduction

***TL;DR***: We propose a framework leveraging Large Language Models to solve multi-modal open-domian question answerings (moqa) in a **zero-shot manner**, this is very different from previous supervised paradigm, instead we do not require any domain-specific annotations and modules, posing our framework as a general paradigm for moqa tasks. [The work has been accepted in to EMNLP 2023 Findings.](https://aclanthology.org/2023.findings-emnlp.85/)

![paradigmdifference](https://p.ipic.vip/hfn7wj.png)

The method framework tackles each modality seperately, and fuse results by Large Language models, the '*devide-and-conquer'* design circumvent complex multimodal information retrieval obstacle and flexible enough to incorporate any new modalities and stronger models into the framework to achieve better performance.

![](https://p.ipic.vip/kj67o3.png)

## Code

### Preparation

```
pip install -r requirements.txt
bash scripts/download_features.sh # download features
```

Please make sure that `FEATURES_CONFIG` in `utils/global.py`  points to the correct path (it should be correct by default)

``````yaml
FEATURES_CONFIG:
  MMCOQA:
    image:
      clip:
        QuestionEmbedding: stored_features/mmcoqa/image/clip_features/QuestionEmbedding.pt
        ReferenceEmbedding: stored_features/mmcoqa/image/clip_features/ReferenceEmbedding.pt
``````

### Retrieve quick example

```
from model_zoo import retriever
features_model_dict={"image":"clip","table":"gpt3","passage":"gpt3"}
retriever=MultimodalRetriever("MMCOQA",features_model_dict)
question_ids=["85a99525aa24b266108b41a22bf1e21c","dc80ff16b65023c74711518f4a46a732"]
retrieved_documents=retriever.retrieve(question_ids) #retrieved_documents is a dict {"image":[doc_id1,doc_id2],"table":[doc_id1,doc_id2]}
```

### Passage Feature Generation 

To download the ANCE dense retriever

```
mkdir checkpoints
wget https://drive.google.com/file/d/1aQquB0ZbeSkhTiwNU-XNtjbll-ZgocV6/view?usp=sharing
```

To generate passage collection indexing, 

  1.run preprocess_MMCoQA.py to transform "multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl" to new collection file "MMCoQA_text_collection.jsonl" for bm25 and dense indexing.

  2.run generate_feature.py to save embedding as "pid2feature.pkl".

```
python preprocess_MMCoQA.py
python generate_feature.py
```

### Run MoQA

1. Get `direct_QA` using LLM in `['openchat', 'gpt-4', 'llama2chat']` for data in `['mmqa', 'mmcoqa']`, the answers will be saved at for example  `MOQA/output/mmqa/direct_chatgpt.json`

   ``````bash
   cd ~/MOQA/pipeline
   python direct_qa.py --dataset $data --direct_qa_model $LLM
   ``````

2. Get answers for query of references from various modality. The answers will be saved at for example  `MOQA/output/mmqa/direct_chatgpt.json`

   ```bash
   python answerer.py --dataset $data --text_qa $LLM --table_qa $LLM
   ```

3. Ensemble multiple answers from various sources and reason the final answer for the qeury. The results will be saved at for example `~/scratch/MOQA/output/mmqa/candidates/chatgpt/Iblip2_Tllama2chatTabllama2chat_direct_chatgpt.json`

   ```bash
   python strategy.py --reasoner $LLM \
           --textual_qa ~/scratch/MOQA/output/mmqa/Tllama2chatTabllama2chat.json \
           --visual_qa ~/scratch/MOQA/output/mmqa/Iblip2.json \
           --direct_qa ~/scratch/MOQA/output/mmqa/direct_chatgpt.json
   ```

4. Evaluation results

   ```
   python evaluation.py --target_file ~/scratch/MOQA/output/mmqa/candidates/chatgpt/Iblip2_Tllama2chatTabllama2chat_direct_chatgpt.json
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

