#!/bin/bash

SBATCH --gres=gpu:3               # Request 3 GPUs
#SBATCH --partition=orchid          
#SBATCH --account=orchid
#SBATCH -o %j.out                   
#SBATCH -e %j.err                   
#SBATCH --time=3:00:00             
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000

## executables
conda activate decentralizedEO
nvidia-smi

python devices.py

cd licos
export CUDA_VISIBLE_DEVICES=0,1,2
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

mpiexec --oversubscribe -n 3 python test_main.py