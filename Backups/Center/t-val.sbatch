#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=4096
#SBATCH --mail-type=END
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
#SBATCH --gres=gpu:turing:1

srun python3 ./train.py -e 50