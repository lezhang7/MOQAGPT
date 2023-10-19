<!-- # EMNLP 2023 Chain-of-Thought Promp Mmethod for Multimodal Conversational Question Answering
### how is multimodal conversational question answering differs from textual converstaional question answer? and what is our contribution?
  - mmcoqa requires multimodal retrieval, and rank among different modalities, thus we introduce a multimodal ranking network to rank relevence for the questino across diffrent modalities.
  - we propose to use in-context learning to (1) rewrite current questions given history for co-reference resovling. (2) extract answers from passages for question answering using chain-of-thought prompting.
  - given candidate answers from multiple and multimodal evidence sources, we need to decide the final answer of the question, we could use (1) heuristic method, which select most frequent answers and take the score of different evidcen source into consideration. (2) use in-context learning to decide the final anser.

*IN ALL, our contrbution are:*
  1. we propose a general framework for multimodal conversation question answering which can handle questions requring evidence of any modality (by simply plug extra encoder into the framework).
  2. we introduce multimodal ranking network which can yield confidence score for evidence from multimodalities. 
  3. we propose extractive in-context question answering 
  4. we propose answer decision strategy to decide gold answer from multimodal candidates.  -->


## Usage

### Dataset Doanloading
download dataset by runing
```
pip install gdown
bash scritps/download_dataset.sh
```

### Features Doanloading
run the scripts to download processed features for both queries and resources
```
pip install gdown
bash scripts/download_features.sh
```
**Be sure to change FEATURES_CONFIG in utils/global.py file**

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

```
import passage_retriever

# retrieve topk passage_ids related to the question, return {'qid':[pid1, pid2, pid3,...]} and {'qid':[fea1, fea2, fea3,...]}
q_rank_list, q_fea_list = passage_retriever.passage_retrieval(passage_retriever_path, pid2feature_path, test_file_path, top_k)
```
