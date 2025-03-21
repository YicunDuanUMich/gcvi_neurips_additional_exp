import torch
import torch.distributions as D
import math
from rotated_mnist_toy.losses import get_imp_weights_amortized_angle_only

def fKL(zs, xs, kwargs):
    angle_encoder = kwargs['angle_encoder']
    angle_prior = kwargs['angle_prior']
    K = kwargs['K']

    latent_particles = zs
    angle_particles = angle_prior.sample((K,)).to(xs.device)
    weights, log_weights = get_imp_weights_amortized_angle_only(angle_particles, latent_particles, xs, K_for_imp_weights=K, log=True, **kwargs)

    values = -1*angle_encoder.get_log_prob(angle_particles, xs)
    return torch.dot(weights, values)


def nll(zs, xs, kwargs):
    theta = kwargs['angle']
    angle_encoder = kwargs['angle_encoder']

    q = angle_encoder.get_q(xs)
    new_loc = q.loc.item() % (2 * math.pi)
    new_std = q.scale.item()

    q_to_use = D.Normal(new_loc, new_std)

    return -1*q_to_use.log_prob(theta)

    
