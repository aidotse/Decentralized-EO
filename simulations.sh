#!/bin/bash

#SBATCH --gres=gpu:3               
#SBATCH --partition=orchid          
#SBATCH --account=orchid
#SBATCH -o %j.out                   
#SBATCH -e %j.err                   
#SBATCH --time=3:00:00             
#SBATCH --ntasks=1                  
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000

## executables
conda activate decentralizedEO
nvidia-smi

python test_main.py