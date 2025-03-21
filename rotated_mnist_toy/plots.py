import os
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4"  
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import torch
import math
import matplotlib.pyplot as plt
import numpy as np 
# -- plotting -- 
import matplotlib
import numpy as np
import hydra
from omegaconf import DictConfig
import random
import matplotlib as mpl
from rotated_mnist_toy.setup import setup
plt.style.use('science')
mpl.rc('lines', linewidth=4.0)
matplotlib.rcParams.update({'font.size': 60}) 


@hydra.main(version_base=None, config_path="../config", config_name="mnist_toy")
def main(cfg : DictConfig) -> None:
    # initialize(config_path="../config")
    # cfg = compose(config_name="mnist")
    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dir = cfg.dir
    os.chdir(dir)

    (
        z,
        xs,
        prior,
        epochs,
        device,
        mb_size,
        encoder,
        angle_encoder,
        optimizer_encoder,
        optimizer_encoder2,
        scheduler_encoder,
        scheduler_encoder2,
        logger_string,
        gan,
        kwargs
    ) = setup(cfg)


    path1 = "{}/{}/iwbo,0.001,64,32,init=0,true=260".format(cfg.log_dir, "rotated_mnist_toy")
    path2 = "{}/{}/favi,0.001,64,32,init=0,true=260".format(cfg.log_dir, "rotated_mnist_toy")
    ts = np.arange(0, 10000, 20)

    # fKL plot
    fkls_favi = np.load(path2 + '/fkls.npy')
    fkls_iwbo = np.load(path1 + '/fkls.npy')

    # fkls_favi = gaussian_filter1d(fkls_favi, 1.0)
    # fkls_iwbo = gaussian_filter1d(fkls_iwbo, 1.0)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(50,25))
    ax[0,0].plot(ts, fkls_favi[:ts.shape[0]], label='Expected Forward KL')
    ax[0,0].plot(ts, fkls_iwbo[:ts.shape[0]], label='-ELBO')
    ax[0,0].set_xlabel('Gradient Step')
    # ax[0,0].legend()
    ax[0,0].set_title('Forward KL: $\\textrm{KL}(p(\\theta \mid x_{\\textrm{true}}) \mid \mid q(\\theta \mid x_{\\textrm{true}}))$')
    #plt.savefig('./rotated_mnist_toy/figs/fkl.png')

    # rKL plot
    rkls_favi = np.load(path2 + '/rkls.npy')
    rkls_iwbo = np.load(path1 + '/rkls.npy')

    # rkls_favi = gaussian_filter1d(rkls_favi, 2.0)
    # rkls_iwbo = gaussian_filter1d(rkls_iwbo, 2.0)

    # fig, ax = plt.subplots(figsize=(20,10))
    ax[1,0].plot(ts, rkls_favi[:ts.shape[0]], label='Expected Forward KL')
    ax[1,0].plot(ts, rkls_iwbo[:ts.shape[0]], label='-ELBO')
    ax[1,0].set_title('Reverse KL: $\\textrm{KL}(q(\\theta \mid x_{\\textrm{true}})) \mid \mid p(\\theta \mid x_{\\textrm{true}})$')
    ax[1,0].set_xlabel('Gradient Step')
    # ax[1,0].legend()
    #ax.set_yscale('log')
    # plt.savefig('./rotated_mnist_toy/figs/iwbo.png')


    # NLL plot
    nlls_favi = np.load(path2 + '/nlls.npy')
    nlls_iwbo = np.load(path1 + '/nlls.npy')

    # nlls_favi = gaussian_filter1d(nlls_favi, 4.0)
    # nlls_iwbo = gaussian_filter1d(nlls_iwbo, 4.0)

    #fig, ax = plt.subplots(figsize=(20,10))
    ax[0,1].plot(ts, nlls_favi[:ts.shape[0]], label='Expected Forward KL')
    ax[0,1].plot(ts, nlls_iwbo[:ts.shape[0]], label='-ELBO')
    ax[0,1].set_xlabel('Gradient Step')
    # ax[0,1].legend()
    ax[0,1].set_title('Negative Log Likelihood ($-\\log q(\\theta_{\\textrm{true}} \mid x_{\\textrm{true}}))$')
    #ax.set_yscale('log')
    #plt.savefig('./rotated_mnist_toy/figs/nll.png')


    # Angle plot
    angles_favi = np.load(path2 + '/angle_estimates.npy')
    angles_iwbo = np.load(path1 + '/angle_estimates.npy')

    angles_favi = (angles_favi * (180/math.pi)) % 360
    angles_iwbo = (angles_iwbo * (180/math.pi)) % 360


    #fig, ax = plt.subplots(figsize=(20,10))
    trunc = ts[-1]
    ax[1,1].axhline(260., c='r', label='Truth')
    ax[1,1].plot(np.arange(trunc), angles_favi[:trunc], label='Expected Forward KL')
    ax[1,1].plot(np.arange(trunc), angles_iwbo[:trunc], label='-ELBO')
    ax[1,1].set_xlabel('Gradient Step')
    # ax[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[1,1].set_title('Angle Estimate (Variational Mode)')
    #ax.set_yscale('log')


    handles, labels = ax[1, 1].get_legend_handles_labels()

    # Create a single legend under the whole subplot grid
    fig.legend(handles, labels, loc="lower center", ncol=4)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('./rotated_mnist_toy/figs/joint.png')


if __name__ == "__main__":
    main()  
