#!/bin/bash

#SBATCH --mail-user=pduan@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --account=regier0
#SBATCH --partition=spgpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a40:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=40GB
#SBATCH --gpu_cmode=shared

## wall time hours:minutes:seconds
#SBATCH --time=120:00:00

module load python/3.10.4 poetry/1.6.1 cuda/12.1.1 cudnn/12.1-v8.9.0
module list

echo "running on: $SLURM_JOB_NODELIST"

cd ~/gcvi_neurips_additional_exp
source .venv/bin/activate

python pyro_cases/run_multi_amortized_favi.py \
    --save-path /scratch/regier_root/regier0/pduan/gcvi_08-02_deep_set_favi_with_fixed_design_output \
    --nn-type deep_set \
    --repeat-times 96 \
    --use-fixed-design
