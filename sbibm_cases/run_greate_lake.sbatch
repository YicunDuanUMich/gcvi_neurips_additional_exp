#!/bin/bash

#SBATCH --mail-user=pduan@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --account=regier0
#SBATCH --partition=spgpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=256GB

## wall time hours:minutes:seconds
#SBATCH --time=24:00:00

module load python/3.10.4 poetry/1.6.1 cuda/12.1.1 cudnn/12.1-v8.9.0
module list

echo "running on: $SLURM_JOB_NODELIST"

cd ~/gcvi_neurips_additional_exp
source .venv/bin/activate

export OMP_NUM_THREADS=4 && \
export OPENBLAS_NUM_THREADS=4 && \
export MKL_NUM_THREADS=4 && \
export VECLIB_MAXIMUM_THREADS=4 && \
export NUMEXPR_NUM_THREADS=4 && \
python ./smibm/run.py --task-name two_moons --model-num 1000 --cuda-idx 0 ; \
    python ./smibm/run.py --task-name slcp --model-num 1000 --cuda-idx 0 ; \
    python ./smibm/run.py --task-name gaussian_mixture --model-num 1000 --cuda-idx 0 ; \
    python ./smibm/run.py --task-name gaussian_linear --model-num 1000 --cuda-idx 0 ; \
    python ./smibm/run.py --task-name gaussian_linear_uniform --model-num 1000 --cuda-idx 0 ; \
    python ./smibm/run.py --task-name bernoulli_glm --model-num 1000 --cuda-idx 0 ; \
    python ./smibm/run.py --task-name slcp_distractors --model-num 1000 --cuda-idx 0 ; \
    python ./smibm/run.py --task-name bernoulli_glm_raw --model-num 1000 --cuda-idx 0
