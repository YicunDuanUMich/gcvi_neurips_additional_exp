#!/bin/bash

#SBATCH --mail-user=pduan@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --account=regier0
#SBATCH --partition=spgpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=16GB
#SBATCH --gpu_cmode=shared

## wall time hours:minutes:seconds
#SBATCH --time=00:30:00

module load python/3.10.4 poetry/1.6.1 cuda/12.1.1 cudnn/12.1-v8.9.0
module list

echo "running on: $SLURM_JOB_NODELIST"

cd ~/gcvi_neurips_additional_exp
source .venv/bin/activate

python pyro_cases/new_generate_sample_dict.py \
    --save-path /scratch/regier_root/regier0/pduan/gcvi_refer_test_sample_dict
