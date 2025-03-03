export OMP_NUM_THREADS=4 && \
export OPENBLAS_NUM_THREADS=4 && \
export MKL_NUM_THREADS=4 && \
export VECLIB_MAXIMUM_THREADS=4 && \
export NUMEXPR_NUM_THREADS=4 && \
python ./sbibm_cases/run.py --task-name two_moons --model-num 1000 --cuda-idx 4 ; \
    python ./sbibm_cases/run.py --task-name slcp --model-num 1000 --cuda-idx 4 ; \
    python ./sbibm_cases/run.py --task-name gaussian_mixture --model-num 1000 --cuda-idx 4 ; \
    python ./sbibm_cases/run.py --task-name gaussian_linear --model-num 1000 --cuda-idx 4

export OMP_NUM_THREADS=4 && \
export OPENBLAS_NUM_THREADS=4 && \
export MKL_NUM_THREADS=4 && \
export VECLIB_MAXIMUM_THREADS=4 && \
export NUMEXPR_NUM_THREADS=4 && \
python ./sbibm_cases/run.py --task-name gaussian_linear_uniform --model-num 1000 --cuda-idx 5 ; \
    python ./sbibm_cases/run.py --task-name bernoulli_glm --model-num 1000 --cuda-idx 5 ; \
    python ./sbibm_cases/run.py --task-name slcp_distractors --model-num 1000 --cuda-idx 5 ; \
    python ./sbibm_cases/run.py --task-name bernoulli_glm_raw --model-num 1000 --cuda-idx 5


export OMP_NUM_THREADS=4 && \
export OPENBLAS_NUM_THREADS=4 && \
export MKL_NUM_THREADS=4 && \
export VECLIB_MAXIMUM_THREADS=4 && \
export NUMEXPR_NUM_THREADS=4 && \
python ./sbibm_cases/run.py --task-name two_moons --model-num 30 --cuda-idx 4 --use-elbo ; \
    python ./sbibm_cases/run.py --task-name slcp --model-num 30 --cuda-idx 4 --use-elbo ; \
    python ./sbibm_cases/run.py --task-name gaussian_mixture --model-num 30 --cuda-idx 4 --use-elbo ; \
    python ./sbibm_cases/run.py --task-name gaussian_linear --model-num 30 --cuda-idx 4 --use-elbo

export OMP_NUM_THREADS=4 && \
export OPENBLAS_NUM_THREADS=4 && \
export MKL_NUM_THREADS=4 && \
export VECLIB_MAXIMUM_THREADS=4 && \
export NUMEXPR_NUM_THREADS=4 && \
python ./sbibm_cases/run.py --task-name gaussian_linear_uniform --model-num 30 --cuda-idx 5 --use-elbo ; \
    python ./sbibm_cases/run.py --task-name bernoulli_glm --model-num 30 --cuda-idx 5 --use-elbo ; \
    python ./sbibm_cases/run.py --task-name slcp_distractors --model-num 30 --cuda-idx 5 --use-elbo ; \
    python ./sbibm_cases/run.py --task-name bernoulli_glm_raw --model-num 30 --cuda-idx 5 --use-elbo