#!/bin/bash

module load miniconda/3
conda init
conda activate flava
cd /home/mila/l/le.zhang/scratch/MOQA/pipeline
python evaluation.py \
    --target_file /home/mila/l/le.zhang/scratch/MOQA/output/mmqa/candidates/chatgpt/Iblip2_Tllama2chatTabllama2chat_direct_chatgpt.json