#!/bin/bash

export OMP_NUM_THREADS=4 && \
export OPENBLAS_NUM_THREADS=4 && \
export MKL_NUM_THREADS=4 && \
export VECLIB_MAXIMUM_THREADS=4 && \
export NUMEXPR_NUM_THREADS=4

python pyro_cases/run_multi.py --save-path /data/scratch/pduan/gcvi_05-28_output --cuda-idx 4,5,6 --repeat-times 36


export OMP_NUM_THREADS=4 && \
export OPENBLAS_NUM_THREADS=4 && \
export MKL_NUM_THREADS=4 && \
export VECLIB_MAXIMUM_THREADS=4 && \
export NUMEXPR_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=5,6,7 python pyro_cases/run_multi_non_amortized_vae.py \
                           --save-path /data/scratch/pduan/gcvi_06-30_non_amortized_vae_output \
                           --repeat-times 36


export OMP_NUM_THREADS=4 && \
export OPENBLAS_NUM_THREADS=4 && \
export MKL_NUM_THREADS=4 && \
export VECLIB_MAXIMUM_THREADS=4 && \
export NUMEXPR_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=5,6,7 python pyro_cases/run_multi_set_transformer_favi.py \
                           --save-path /data/scratch/pduan/gcvi_06-30_set_transformer_favi_output \
                           --repeat-times 36