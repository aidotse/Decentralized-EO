#!/bin/bash

#SBATCH --gres=gpu:1              
#SBATCH --partition=orchid          
#SBATCH --account=orchid
#SBATCH -o %j.out                   
#SBATCH -e %j.err                   
#SBATCH --time=0:30:00             
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=1
#SBATCH --mem=64000

## executables
conda activate decentralizedEO
nvidia-smi

cd licos
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES=0,1,2
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
mpiexec --oversubscribe -n 2 python test_main.py
nvidia-smi > gpu_usage_$SLURM_JOB_ID.log
