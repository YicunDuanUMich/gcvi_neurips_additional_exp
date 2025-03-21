import os
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4"  
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import torch
import matplotlib.pyplot as plt
import numpy as np 
# -- plotting -- 
import matplotlib
import hydra
import pandas as pd
from omegaconf import DictConfig
import random
from amortized_clustering.setup import setup
import seaborn as sns
plt.style.use('science')
matplotlib.rcParams.update({'font.size': 30}) 

def found_correct_centers(cfg, encoder, loss, width, seeds, parameterizations, **kwargs):
    logger_string = 'loss={},width={},nsets=1'.format(loss, width)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    
    def etas_to_musigma2(eta1, eta2):
        sigma2 = -1/(2*eta2)
        mu = -1*eta1/(2*eta2)
        return mu, sigma2

    
    results = {}
    for parameterization in parameterizations:
        num_correct = 0
        num_total = 0
        for seed in seeds:
            try:
                encoder.load_state_dict(torch.load('{}/amortized_clustering/{}/weights1_{}_{}.pth'.format(cfg.log_dir, logger_string, seed, parameterization)))
                
            except:
                continue
            device = encoder.dec[1].bias.device
            this_seed_X = torch.load('{}/amortized_clustering/{}/X_{}.pt'.format(cfg.log_dir, logger_string, seed)).to(device)
            this_seed_Z = torch.load('{}/amortized_clustering/{}/Z_{}.pt'.format(cfg.log_dir, logger_string, seed))

            # Get predicted centers based on parameterization
            if parameterization != 'natural':
                pred_centers = encoder(this_seed_X)[...,0].reshape(-1).detach().cpu()
            else:
                eta = encoder(this_seed_X)
                eta1 = eta[:,0]
                eta2 = eta[:,1].clamp(min=-1000., max=-0.1)
                pred_centers, _ = etas_to_musigma2(eta1, eta2)
            true_centers = this_seed_Z.reshape(-1).detach().cpu()

            # To nearest
            nearest = [find_nearest(true_centers, this_pred_center.item()) for this_pred_center in pred_centers]

            if nearest == true_centers.tolist():
                num_correct += 1

            num_total += 1
            
        results[parameterization] = num_correct/num_total if num_total != 0 else 0
    
    return results

def dists_to_center(cfg, encoder, loss, width, seeds, parameterizations, **kwargs):
    logger_string = 'loss={},width={},nsets=1'.format(loss, width)
    
    def etas_to_musigma2(eta1, eta2):
        sigma2 = -1/(2*eta2)
        mu = -1*eta1/(2*eta2)
        return mu, sigma2

    results = {}
    results_std = {}
    for parameterization in parameterizations:
        dists = []
        for seed in seeds:
            try:
                encoder.load_state_dict(torch.load('{}/amortized_clustering/{}/weights1_{}_{}.pth'.format(cfg.log_dir, logger_string, seed, parameterization)))
                
            except:
                continue
            device = encoder.dec[1].bias.device
            this_seed_X = torch.load('{}/amortized_clustering/{}/X_{}.pt'.format(cfg.log_dir, logger_string, seed)).to(device)
            this_seed_Z = torch.load('{}/amortized_clustering/{}/Z_{}.pt'.format(cfg.log_dir, logger_string, seed))

            # Get predicted centers based on parameterization
            if parameterization != 'natural':
                pred_centers = encoder(this_seed_X)[...,0].reshape(-1).detach().cpu()
            else:
                eta = encoder(this_seed_X)
                eta1 = eta[:,0]
                eta2 = eta[:,1].clamp(min=-1000., max=-0.1)
                pred_centers, _ = etas_to_musigma2(eta1, eta2)
            true_centers = this_seed_Z.reshape(-1).detach().cpu()

            dist = torch.abs(pred_centers.cpu() - true_centers.cpu()).sum()
            dists.append(dist.item())

        results[parameterization] = torch.tensor(dists).mean().item()
        results_std[parameterization] = torch.tensor(dists).std().item()
    

    return results, results_std

def centers_in_order(cfg, encoder, loss, width, seeds, parameterizations, **kwargs):
    logger_string = 'loss={},width={},nsets=1'.format(loss, width)
    
    def etas_to_musigma2(eta1, eta2):
        sigma2 = -1/(2*eta2)
        mu = -1*eta1/(2*eta2)
        return mu, sigma2

    results = {}
    for parameterization in parameterizations:
        num_in_order = 0
        num_total = 0
        for seed in seeds:
            try:
                encoder.load_state_dict(torch.load('{}/amortized_clustering/{}/weights1_{}_{}.pth'.format(cfg.log_dir, logger_string, seed, parameterization)))
                
            except:
                continue
            device = encoder.dec[1].bias.device
            this_seed_X = torch.load('{}/amortized_clustering/{}/X_{}.pt'.format(cfg.log_dir, logger_string, seed)).to(device)
            this_seed_Z = torch.load('{}/amortized_clustering/{}/Z_{}.pt'.format(cfg.log_dir, logger_string, seed))

            # Get predicted centers based on parameterization
            if parameterization != 'natural':
                pred_centers = encoder(this_seed_X)[...,0].reshape(-1).detach().cpu()
            else:
                eta = encoder(this_seed_X)
                eta1 = eta[:,0]
                eta2 = eta[:,1].clamp(min=-1000., max=-0.1)
                pred_centers, _ = etas_to_musigma2(eta1, eta2)
            
            if (pred_centers.sort()[0] == pred_centers).all():
                num_in_order += 1

            num_total += 1
        results[parameterization] = num_in_order/num_total if num_total != 0 else 0

    return results


def plot_shift_estimates(cfg, encoder, loss, width, seeds, parameterizations, **kwargs):
    logger_string = 'loss={},width={},nsets=1'.format(loss, width)
    
    def etas_to_musigma2(eta1, eta2):
        sigma2 = -1/(2*eta2)
        mu = -1*eta1/(2*eta2)
        return mu, sigma2

    
    results = {}
    for parameterization in parameterizations:
        shift_predictions = []
        for seed in seeds:
            try:
                encoder.load_state_dict(torch.load('{}/amortized_clustering/{}/weights2_{}_{}.pth'.format(cfg.log_dir, logger_string, seed, parameterization)))
            except:
                continue
            device = encoder.dec[1].bias.device
            this_seed_X = torch.load('{}/amortized_clustering/{}/X_{}.pt'.format(cfg.log_dir, logger_string, seed)).to(device)
            this_seed_Z = torch.load('{}/amortized_clustering/{}/Z_{}.pt'.format(cfg.log_dir, logger_string, seed))

            # Get predicted centers based on parameterization
            if parameterization != 'natural':
                pred_shift = encoder(this_seed_X)[0,0].item()      
            else:
                eta = encoder(this_seed_X)
                eta1 = eta[:,0]
                eta2 = eta[:,1].clamp(min=-1000., max=-0.1)
                pred_shift, _ = etas_to_musigma2(eta1, eta2)
                pred_shift = pred_shift.item()

            shift_predictions.append(pred_shift)
    
        results[parameterization] = shift_predictions
    

    return results


@hydra.main(version_base=None, config_path="../config", config_name="amortized_clustering")
def main(cfg : DictConfig) -> None:
    # initialize(config_path="../config")
    # cfg = compose(config_name="amortized_clustering")

    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dir = cfg.dir
    os.chdir(dir)

    (Z, 
    X, 
    logger_string,
    encoder1,
    encoder2,
    optimizer1,
    optimizer2,
    scheduler1,
    scheduler2,
    kwargs) = setup(cfg)

    losses = ['elbo', 'favi']
    
    # How often are we essentially correct?
    width_to_use = cfg.plots.width_to_use
    seeds = list(range(cfg.plots.seed_start, cfg.plots.seed_end))
    results = {}
    for loss in losses:
        # Select parameterizations to use
        ps = ['natural', 'mean']
        prop_correct = found_correct_centers(cfg, encoder1, loss, width_to_use, seeds, ps)
        results[loss] = prop_correct
    results = pd.DataFrame(results)
    results.to_latex('./amortized_clustering/figs/correct_centers.tex')

    # How often are predicted centers in order?
    results = {}
    for loss in losses:
        # Select parameterizations to use
        ps = ['natural', 'mean']
        prop_correct = centers_in_order(cfg, encoder1, loss, width_to_use, seeds, ps)
        results[loss] = prop_correct
    results = pd.DataFrame(results)
    results.to_latex('./amortized_clustering/figs/centers_in_order.tex')

    # L1 Distance Quantification
    results = {}
    results_std = {}
    for loss in losses:
        # Select parameterizations to use
        ps = ['natural', 'mean']
        prop_correct, stds = dists_to_center(cfg, encoder1, loss, width_to_use, seeds, ps)
        results[loss] = prop_correct
        results_std[loss] = stds
    results = pd.DataFrame(results)
    results_std = pd.DataFrame(results_std)
    results.to_latex('./amortized_clustering/figs/center_l1s.tex')
    results_std.to_latex('./amortized_clustering/figs/center_l1s_std.tex')

    # Summary of shift estimates
    shift_results = {}
    for loss in losses:
        # Select parameterizations to use
        ps = ['natural', 'mean']
        shift_ests = plot_shift_estimates(cfg, encoder2, loss, width_to_use, seeds, ps)
        shift_results[loss] = shift_ests

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(90,60), sharex=True)
    bw_adjust=1
    linewidth=15

    sns.kdeplot(shift_results['elbo']['natural'], bw_adjust=bw_adjust, ax=ax[0,0], linewidth=linewidth)
    sns.kdeplot(shift_results['favi']['natural'], bw_adjust=bw_adjust, ax=ax[0,1], linewidth=linewidth, c='g')

    sns.kdeplot(shift_results['elbo']['mean'], bw_adjust=bw_adjust, ax=ax[1,0], linewidth=linewidth)
    sns.kdeplot(shift_results['favi']['mean'], bw_adjust=bw_adjust, ax=ax[1,1], linewidth=linewidth, c='g')

    custom_xlim = (80, 120)

    #Setting the values for all axes.
    plt.setp(ax, xlim=custom_xlim)

    ax[0,0].set_ylabel('Natural Parameterization', fontsize=140)
    ax[0,0].set_yticks([])
    ax[0,0].tick_params(labelsize=120)
    #ax[0,1].set_ylabel('Natural Parameterization', fontsize=100)
    ax[0,1].set_yticks([])
    ax[0,1].yaxis.set_visible(False)
    ax[0,1].tick_params(labelsize=120)
    ax[1,0].set_ylabel('Mean Parameterization', fontsize=140)
    ax[1,0].set_yticks([])
    ax[1,0].tick_params(labelsize=120)
    ax[1,1].yaxis.set_visible(False)
    ax[1,1].tick_params(labelsize=120)
    
    ax[0,0].set_title('ELBO', fontsize=160)
    ax[0,1].set_title('$L_P$', fontsize=160)

    fig.tight_layout()

    plt.savefig('./amortized_clustering/figs/all_shift_estimates.png')

if __name__ == "__main__":
   main()
