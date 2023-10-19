#!/bin/bash
#SBATCH --job-name=qa_mmqa
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                              # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --ntasks-per-node=1                               # Ask for 1 GPU
#SBATCH --mem=128G                                        # Ask for 10 GB of RAM
#SBATCH --time=20:00:00                                   
#SBATCH --output=/home/mila/l/le.zhang/scratch/slurm_logs/job_output-%j.txt
#SBATCH --error=/home/mila/l/le.zhang/scratch/slurm_logs/job_error-%j.txt 

module load miniconda/3
conda init
conda activate openflamingo

cd /home/mila/l/le.zhang/scratch/MOQA/pipeline
# python direct_gpt.py
# python answerer.py --dataset mmcoqa --text_qa llama2chat --table_qa llama2chat
python answerer.py --dataset mmcoqa --text_qa openchat --table_qa openchat