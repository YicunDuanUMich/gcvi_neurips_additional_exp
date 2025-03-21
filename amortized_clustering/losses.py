import torch
from amortized_clustering.generate import generate_data_favi
import torch.distributions as D
import math
from einops import rearrange, reduce, repeat


def gaussian_log_density_natural(eta1, eta2, x):
    first_term = eta1*x
    second_term = eta2*(x**2)
    Aeta = (-1/4)*torch.div(eta1**2, eta2) - (1/2)*torch.log(-2*eta2)
    log_dens = torch.log(torch.tensor(1/math.sqrt(2*math.pi))) + first_term + second_term - Aeta

    return log_dens


def batch_favi_loss(**kwargs):
    encoder1 = kwargs['encoder1']
    encoder2 = kwargs['encoder2']
    parameterization = kwargs['parameterization']

    shift, z, x = generate_data_favi(**kwargs)

    if parameterization != 'natural':
        params1 = encoder1(x)
        q1 = D.Normal(params1[:,0], params1[:,1])
        params2 = encoder2(x)
        q2 = D.Normal(params2[:,0], params2[:,1]) #change maybe

        reshaped_z = z.reshape(-1)

        return -1*(q1.log_prob(reshaped_z).sum(-1).mean() + q2.log_prob(shift).mean())

    else:
        eta1 = encoder1(x)
        eta11 = eta1[:,0]
        eta12 = eta1[:,1].clamp(min=-1000., max=-0.1)
        log_dens1 = gaussian_log_density_natural(eta11, eta12, z.reshape(-1))

        eta2 = encoder2(x)
        eta21 = eta2[:,0]
        eta22 = eta2[:,1].clamp(min=-1000., max=-0.1)
        log_dens2 = gaussian_log_density_natural(eta21, eta22, shift.reshape(-1))
        return -1*log_dens1.sum(-1) + -1*log_dens2.sum(-1)

def batch_elbo_loss(**kwargs):
    K = kwargs['K']
    true_x = kwargs['X']
    device = kwargs['device']
    encoder1 = kwargs['encoder1']
    encoder2 = kwargs['encoder2']
    true_centers = kwargs['true_centers']
    sigma = kwargs['sigma']
    tau = kwargs['tau']
    n_sets = kwargs['n_sets']
    n_centers = kwargs['n_centers']
    proportions = kwargs['proportions']
    parameterization = kwargs['parameterization']

    def etas_to_musigma2(eta1, eta2):
        sigma2 = -1/(2*eta2)
        mu = -1*eta1/(2*eta2)
        return mu, sigma2

    if parameterization != 'natural':
        params1 = encoder1(true_x)
        q1 = D.Normal(params1[:,0], params1[:,1])
        params2 = encoder2(true_x)
        q2 = D.Normal(params2[:,0], params2[:,1]) #change maybe
    else:
        eta1 = encoder1(true_x)
        eta11 = eta1[:,0]
        eta12 = eta1[:,1].clamp(min=-1000., max=-0.1)
        mu1, sigma21 = etas_to_musigma2(eta11, eta12)
        q1 = D.Normal(mu1, torch.sqrt(sigma21))

        eta2 = encoder2(true_x)
        eta21 = eta2[:,0]
        eta22 = eta2[:,1].clamp(min=-1000., max=-0.1)
        mu2, sigma22 = etas_to_musigma2(eta21, eta22)
        q2 = D.Normal(mu2, torch.sqrt(sigma22))
    
    
    draws_z = q1.rsample((K,))
    draws_s = q2.rsample((K,))#.clamp(-200.+1e-6, 200.-1e-6)

    # Get numerators
    log_ps = D.Normal(0., 100.).log_prob(draws_s).reshape(-1).to(device) # K x 1
    mean = repeat(true_centers, 'n_cent 1 -> K n_cent', K=K) + repeat(draws_s, 'K 1 -> K n_cent', n_cent=n_centers)
    log_pzs = reduce(D.Normal(mean, sigma).log_prob(draws_z), 'K n_cent -> K', 'sum') #sum dimensions
    mix = D.Categorical((torch.ones((K, n_centers))*proportions).to(device))
    comp = D.Normal(draws_z, tau)

    mix_models = D.MixtureSameFamily(mix, comp)
    to_plug_in = rearrange(true_x, '1 n_samp 1 -> n_samp')
    to_plug_in = repeat(to_plug_in, 'n_samp -> n_samp K', K=K)
    log_pxzs = mix_models.log_prob(to_plug_in).sum(0)

    nums = log_ps + log_pzs + log_pxzs
    denoms = q1.log_prob(draws_z).sum(-1) + q2.log_prob(draws_s).sum(-1)

    log_weights = nums-denoms # K b
    weights = log_weights.softmax(0) # K b
    dots = weights.T @ log_weights # b 
    loss = -1*dots
    return loss

