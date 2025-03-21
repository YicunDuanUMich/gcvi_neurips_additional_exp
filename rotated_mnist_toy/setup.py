import os
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4"  
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import torch.distributions as D
import torch
import math

from omegaconf import DictConfig
from modules.mnist_modules import MNISTGenerator, MNISTAngleEncoder
from modules.custom_lr_scheduler import CustomOptim2

def setup(cfg : DictConfig):
    latent_dim = cfg.data.latent_dim
    device = cfg.training.device
    initial_angle = cfg.training.favi_offset
    weight_path = cfg.weight_path
    gan = MNISTGenerator(cfg, initial_angle, device, weight_path=weight_path)
    gan.load()
    gan.decoder = gan.decoder.to(device)

    # Generate dataset for angle 90 degrees counterclockwise.
    n_pts = cfg.data.n_pts
    true = cfg.data.true_angle
    z = gan.prior.sample((n_pts,latent_dim)).to(device)
    angle = torch.tensor(true*math.pi/180).to(device)
    x = gan.decoder(z)
    x = gan.rotate(x, angle)
    gen_imgs = gan.denorm(x.reshape((-1, 28,28)).detach())
    final_x = gen_imgs.reshape(n_pts, -1)
    distr = D.Normal(final_x, gan.noise)
    final_x = distr.sample()
    
    
    loss_name = cfg.training.loss
    prior = D.Normal(torch.zeros(latent_dim).to(device), 1.)
    angle_prior = D.Uniform(0, 2*math.pi)
    favi_mb_size = cfg.training.favi_mb_size
    favi_offset = cfg.training.favi_offset
    K = cfg.training.K

    kwargs = {'loss_name': loss_name,
            'latent_dim': latent_dim,
            'prior': prior,
            'angle_prior': angle_prior,
            'gan': gan,
            'favi_mb_size': favi_mb_size,
            'favi_offset': favi_offset,
            'K': K,
            'angle': angle}

    steps = cfg.training.steps
    device=cfg.training.device
    mb_size = cfg.training.mb_size
    parameterization = cfg.training.parameterization

    kwargs['mb_size'] = mb_size
    kwargs['device'] = device

    angle_encoder = MNISTAngleEncoder(favi_offset=initial_angle, device=device, parameterization=parameterization)
    
    kwargs['angle_encoder'] = angle_encoder
    logger_string = '{},{},{},{},init={},true={}'.format(cfg.training.loss, cfg.training.lr, latent_dim, mb_size, initial_angle, true)
    angle_encoder.to(device)
    optimizer_encoder = torch.optim.Adam(angle_encoder.parameters(), lr=cfg.training.lr)
    scheduler_encoder = CustomOptim2(optimizer_encoder, start_lr=cfg.training.lr)

    return (
        z,
        final_x,
        prior,
        steps,
        device,
        mb_size,
        None, # placeholder for encoder on z
        angle_encoder,
        None, # placeholder for optimizer on z
        optimizer_encoder,
        None, # placeholder for scheduler on z
        scheduler_encoder,
        logger_string,
        gan,
        kwargs
    )