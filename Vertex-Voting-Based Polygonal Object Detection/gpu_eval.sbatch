#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:08:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2048
#SBATCH --mail-type=END
#SBATCH --gres=gpu:1
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn

srun nvidia-smi
srun python3 eval_mAP.py config/wireframe.yaml ./checkpoint_best.pth ../../../datasets/validation/for_lcnn/base3 logs/test/  

