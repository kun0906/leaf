#!/bin/bash

#SBATCH --job-name='sbatch'         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
## SBATCH --mem-per-cpu=40G         # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)\
#SBATCH --priority=10             # Only Slurm operators and administrators can set the priority of a job.
# # SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --output=out_sbatch.txt
#SBATCH --error=err_sbatch.txt
# #SBATCH --mail-user=kun.bj@cloud.com # not work \
# #SBATCH --mail-user=<YourNetID>@princeton.edu
#SBATCH --mail-user=ky8517@princeton.edu

#module purge
module load anaconda3/5.0.1
source activate tf1-gpu

# check cuda and cudnn version for tensorflow_gpu==1.13.1
# https://www.tensorflow.org/install/source#linux
module load cudatoolkit/10.0
module load cudnn/cuda-10.0/7.5.0

whereis nvcc
which nvcc
nvcc --version

cd /scratch/gpfs/ky8517/leaf-torch/data/femnist
pwd
python3 -V
#python vgg16_zero.py
./preprocess.sh

#
#module load anaconda3/2021.5
#cd /scratch/gpfs/ky8517/fkm/afkm
#PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main.py
