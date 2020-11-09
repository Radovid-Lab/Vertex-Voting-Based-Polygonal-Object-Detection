#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=07:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000
#SBATCH --mail-type=END
#SBATCH --gres=gpu:turing:2
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn

srun nvidia-smi
srun python3 ./train.py -d 0,1 --identifier baseline config/wireframe.yaml
