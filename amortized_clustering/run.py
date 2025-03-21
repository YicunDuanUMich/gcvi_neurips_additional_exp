import os
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4"  
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import torch
import numpy as np 
# -- plotting -- 
from os.path import exists
import hydra
from omegaconf import DictConfig
import random
from amortized_clustering.setup import setup
from amortized_clustering.losses import batch_favi_loss,  batch_elbo_loss

def loss_choice(loss_name, **kwargs):
    if loss_name == 'favi':
        return batch_favi_loss(**kwargs)
    elif loss_name == 'elbo':
        return batch_elbo_loss(**kwargs)
    else:
        raise ValueError('Specify an appropriate loss name string.')


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

    # Setup experiment
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

    if not exists('{}'.format(cfg.log_dir)):
        os.mkdir('{}'.format(cfg.log_dir))
    if not exists('{}/amortized_clustering'.format(cfg.log_dir)):
        os.mkdir('{}/amortized_clustering'.format(cfg.log_dir))
    if not exists('{}/amortized_clustering/{}'.format(cfg.log_dir, logger_string)):
        os.mkdir('{}/amortized_clustering/{}'.format(cfg.log_dir, logger_string))

    # We want to save this particular Z and X for evaluation later
    torch.save(Z, '{}/amortized_clustering/{}/Z_{}.pt'.format(cfg.log_dir, logger_string, cfg.seed))
    torch.save(X, '{}/amortized_clustering/{}/X_{}.pt'.format(cfg.log_dir, logger_string, cfg.seed))

    loss_name = kwargs['loss']

    for j in range(kwargs['steps']):
        scheduler1.zero_grad()
        scheduler2.zero_grad()
        loss = loss_choice(loss_name, **kwargs)
        print('Loss iter {} is {} seed is {}'.format(j, loss, seed))
        loss.backward()
        scheduler1.step_and_update_lr()
        scheduler2.step_and_update_lr()

        # Log weights periodically
        if (j+1) % 5000 == 0:
            torch.save(encoder1.state_dict(), '{}/amortized_clustering/{}/weights1_{}_{}.pth'.format(cfg.log_dir, logger_string, cfg.seed, cfg.encoder.parameterization))
            torch.save(encoder2.state_dict(), '{}/amortized_clustering/{}/weights2_{}_{}.pth'.format(cfg.log_dir, logger_string, cfg.seed, cfg.encoder.parameterization))
            

    
if __name__ == "__main__":
   main()
