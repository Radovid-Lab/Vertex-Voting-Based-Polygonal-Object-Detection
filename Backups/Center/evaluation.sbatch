#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=10240
#SBATCH --mail-type=END

srun python3 ./test.py -p ./Data/