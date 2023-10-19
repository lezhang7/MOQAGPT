#!/bin/bash

module load miniconda/3
conda init
conda activate openflamingo

cd MOQA/pipeline
python strategy.py --reasoner chatgpt \
        --textual_qa /home/mila/l/le.zhang/scratch/MOQA/output/mmqa/Tllama2chatTabllama2chat.json \
        --visual_qa /home/mila/l/le.zhang/scratch/MOQA/output/mmqa/Iblip2.json \
        --direct_qa /home/mila/l/le.zhang/scratch/MOQA/output/mmqa/direct_chatgpt.json