# **MoqaGPT: Zero-Shot Multi-modal Open-domain Question Answering with Large Language Models (EMNLP2023 Findings)**

### Introduction

***TL;DR***: We propose a framework leveraging Large Language Models to solve multi-modal open-domian question answerings (moqa) in a **zero-shot manner**, this is very different from previous supervised paradigm, instead we do not require any domain-specific annotations and modules, posing our framework as a general paradigm for moqa tasks. The work has been accepted in to EMNLP 2023 Findings.

![paradigmdifference](https://p.ipic.vip/hfn7wj.png)

The method framework tackles each modality seperately, and fuse results by Large Language models, the '*devide-and-conquer'* design circumvent complex multimodal information retrieval obstacle and flexible enough to incorporate any new modalities and stronger models into the framework to achieve better performance.

![](https://p.ipic.vip/kj67o3.png)

### Code

The proposed framwork do not involve any training, to produce results, firstly install depent libraries.

`pip install -r requirements.txt` 

To perfrom moqa task, one need to firstly download **datasets** and **features** for information retrieval.

`bash scripts/download_features.sh` 

*Be sure to change FEATURES_CONFIG in utils/global.py file*

### Features Retrieve

```
from model_zoo import retriever
features_model_dict={"image":"clip","table":"gpt3","passage":"gpt3"}
retriever=MultimodalRetriever("MMCOQA",features_model_dict)
question_ids=["85a99525aa24b266108b41a22bf1e21c","dc80ff16b65023c74711518f4a46a732"]
retrieved_documents=retriever.retrieve(question_ids) #retrieved_documents is a dict {"image":[doc_id1,doc_id2],"table":[doc_id1,doc_id2]}
```

### passage retriever

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

### passage retrieval