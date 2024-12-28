# read from pickle file
import pickle
import os
import yaml
import re
import json, string, re
import numpy as np
from collections import Counter
from nltk.stem import PorterStemmer
word_num_dict = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 
                     'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 
                     'ten': 10}

def read_config():
    with open('utils/config.yaml', 'r') as file:
    # Parse the YAML content
        config = yaml.safe_load(file)
    return config


def get_sentence_before_first_newline(text):
    match = re.match(r'^(.*?)(?=\n)', text)
    if match:
        return match.group(0)
    else:
        return text
def word2num(s):
    if s in word_num_dict:
        return str(word_num_dict[s])
    else:
        return s
def stemize_text(text):
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in text.split()]
    stemmed_words=[word2num(word) for word in stemmed_words ]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
      return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
      return ' '.join(text.split())
    def remove_punc(text):
      exclude = set(string.punctuation)
      return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
      return str(text).lower()
    if isinstance(s, list):
        return [normalize_answer(x) for x in s]
    else:
        return stemize_text(white_space_fix(remove_articles(remove_punc(lower(s)))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
      return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def prompt_qa_llama2(reference:str, question:str):
    return f"You are doing extractive question answering. Given a document: {reference}. \
          Extract a short answer to the question: {question} from the document. If there is not \
            enough information to answer this question, just answer me with 'Unkown'.  \
                Please provide a concise response, limited to one or two words, No explanation and further question.  Answer: "
def prompt_qa(reference:str, question:str):
    return f"You are doing extractive question answering. Given the document: {reference}. Extract a short answer to the question: \
        {question} from the document. If insufficient information is available to answer the question, respond with 'Unknown'. The \
        answer should be one or two words long, Answer: "

def prompt_direct_qa(questions:str):
    return f"Question: '{questions}'. Please provide a concise response, limited to one or two words, No explanation and further question.  Answer:"


def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
# write to pickle file  
def write_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
# load json file
import json
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
# save json file
def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f,indent=4)

def read_then_save_json(filename, data):
    if os.path.exists(filename):
        json_data = load_json(filename)
        json_data.update(data)
    else:
        json_data = data
    save_json(filename, json_data)


config=read_config()
def get_query_dict(dataset_name):
    dataset_name=dataset_name.upper()
    assert dataset_name in list(config["DATASET_CONFIG"].keys()), f"dataset_name must be in {config['DATASET_CONFIG'].keys()}"
    query_list = []
    if dataset_name=="MMCOQA":
        with open(os.path.join(config["DATASET_DIR"],config["DATASET_CONFIG"][dataset_name]["test"]), 'r') as f:
            for line in f:
                content=json.loads(line)
                query_list.append(content)
        return {x["qid"]:{"query":x["gold_question"],"gold_answer":x["answer"][0]["answer"],"gold_answer_modality":x["answer"][0]["modality"]} for x in query_list}
    elif dataset_name=="MMQA":
        with open(os.path.join(config["DATASET_DIR"],config["DATASET_CONFIG"][dataset_name]["dev"]), 'r') as f:
            for line in f:
                content=json.loads(line)
                query_list.append(content)
        return {x["qid"]:{"query":x["question"],"gold_answer":x["answers"][0]["answer"],"gold_answer_modality":x['metadata']['modalities']} for x in query_list}

negative_words = ['can\'t', 'cannot', 'don\'t', 'do not', 'won\'t', 'will not','does not','doesn\'t',
                  'unable', 'impossible', 'unavailable', 'dislike', 'hate', 'unhappy',"no information","no answer","no mention"
                  'unfortunate', 'terrible', 'awful', 'horrible', 'never',"ai language model",'sorry','unknown','no information','not possible','not provided']
def check_negative_words(sentence):
    if not isinstance(sentence, str):
        sentence=sentence[0]
    sentence = sentence.lower()
    for word in negative_words:
        if word in sentence:
            return True
    return False

def merge_json(json1,json2):
    for key in json1:
        json1[key].update(json2[key])
    return json1