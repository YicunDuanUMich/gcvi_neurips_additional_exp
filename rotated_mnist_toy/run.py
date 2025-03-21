import os
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4"  
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import torch
import hydra
import numpy as np 
# -- plotting -- 
from os.path import exists
from omegaconf import DictConfig
import random
from rotated_mnist_toy.losses import favi_loss, elbo, log_evidence
from rotated_mnist_toy.setup import setup
from rotated_mnist_toy.divs import fKL, nll

def loss_choice(loss_name, z, true_x, **kwargs):
    if loss_name == 'favi':
        return favi_loss(z, true_x, **kwargs)
    elif loss_name == 'iwbo':
        return -1*elbo(z, true_x, **kwargs)
    else:
        raise ValueError('Specify an appropriate loss name string.')

@hydra.main(version_base=None, config_path="../config", config_name="mnist_toy")
def main(cfg : DictConfig) -> None:
    # initialize(config_path="../config", job_name="test_app")
    # cfg = compose(config_name="mnist_toy")

    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dir = cfg.dir
    os.chdir(dir)
    log_dir = cfg.log_dir

    (
        z,
        xs,
        prior,
        steps,
        device,
        mb_size,
        _, 
        angle_encoder,
        _,
        optimizer_encoder,
        _, 
        scheduler_encoder,
        logger_string,
        gan,
        kwargs
    ) = setup(cfg)

    if not exists('{}'.format(cfg.log_dir)):
        os.mkdir('{}'.format(cfg.log_dir))
    if not exists('{}/rotated_mnist_toy'.format(cfg.log_dir)):
        os.mkdir('{}/rotated_mnist_toy'.format(cfg.log_dir))
    if not exists('{}/rotated_mnist_toy/{}'.format(cfg.log_dir, logger_string)):
        os.mkdir('{}/rotated_mnist_toy/{}'.format(cfg.log_dir, logger_string))

    loss_name = kwargs['loss_name']
    kwargs.pop('loss_name')

    angle_estimates = []
    favi_losses = []
    fkls = []
    rkls = []
    nlls = []
    for j in range(steps):
        optimizer_encoder.zero_grad()
        loss = loss_choice(loss_name, z, xs, **kwargs)

        if torch.isnan(loss).any():
            continue
        loss.backward()
        scheduler_encoder.step_and_update_lr()

        print('Loss iteration {} is {}'.format(j, loss.item()))

        # Angle logging
        angle_estimate = angle_encoder.get_q(xs).loc.item()
        angle_estimates.append(angle_estimate)

        if (j) % 20 == 0:
            # Log divs
            favi_losses.append(favi_loss(z, xs, **kwargs).item())
            rkls.append((log_evidence(z, xs, **kwargs) - elbo(z, xs, **kwargs)).item())
            fkls.append(fKL(z, xs, kwargs).item())
            nlls.append(nll(z, xs, kwargs).item())
    
        if (j+1) % 1000 == 0:
            torch.save(gan.state_dict(), '{}/rotated_mnist_toy/{}/gan.pth'.format(log_dir, logger_string))
            torch.save(angle_encoder.state_dict(), '{}/rotated_mnist_toy/{}/angle_encoder.pth'.format(log_dir, logger_string))
            np.save('{}/rotated_mnist_toy/{}/angle_estimates.npy'.format(log_dir, logger_string), np.array(angle_estimates))
            np.save('{}/rotated_mnist_toy/{}/fkls.npy'.format(log_dir, logger_string), np.array(fkls))
            np.save('{}/rotated_mnist_toy/{}/favi_losses.npy'.format(log_dir, logger_string), np.array(favi_losses))
            np.save('{}/rotated_mnist_toy/{}/rkls.npy'.format(log_dir, logger_string), np.array(rkls))
            np.save('{}/rotated_mnist_toy/{}/nlls.npy'.format(log_dir, logger_string), np.array(nlls))

if __name__ == "__main__":
    main()  
