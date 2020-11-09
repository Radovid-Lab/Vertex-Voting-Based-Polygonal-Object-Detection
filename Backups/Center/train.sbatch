#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20480
#SBATCH --mail-type=END
#SBATCH --gres=gpu:p100:2
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun nvidia-smi
srun python3 ./train.py -e 50 -p /tudelft.net/staff-bulk/ewi/insy/VisionLab/klang/experiments/Center/Data/ -b 32