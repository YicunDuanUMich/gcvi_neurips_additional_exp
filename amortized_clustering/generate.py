import torch
from torch import distributions as D
from einops import rearrange

def generate_data(true_centers, n_sets, n_obs, sigma, tau, proportions, true_shift, prior=True):
    '''Given a tensor of d true centers,
    generate n_batches of Z, X. Each Z will have the 
    dimension d, each X will have dimension n_obs. Each
    observation X_i is drawn from a mixture of normals with
    probabilties given by proportions. The true_shift is the deviation
    from zero either in the prior (if prior=True) or in the decoder (if
    prior=False).
    '''
    n_centers = true_centers.size()[0]
    prior_shift = torch.tensor(true_shift) # this hard codes the known shift
    noisy_centers = D.Normal(true_centers + prior_shift.item(), sigma)
    Z = noisy_centers.sample((n_sets,))
    mix = D.Categorical(torch.ones((n_sets, n_centers))*proportions)
    comp = D.Independent(D.Normal(Z, tau),1)
    mix_models = D.MixtureSameFamily(mix, comp)
    X = mix_models.sample((n_obs,))
    X = rearrange(X, 'n_samp b d -> b n_samp d')
    return prior_shift, Z, X

def generate_data_favi(true_centers, n_sets, n_obs, sigma, tau, proportions, **kwargs):
    '''Given a tensor of d true centers,
    generate n_sets of Z, X. Each Z will have the 
    dimension d, each X will have dimension n_obs. Each
    observation X_i is drawn from a mixture of normals with
    probabilties given by proportions. The true_shift is the deviation
    from zero either in the prior (if prior=True) or in the decoder (if
    prior=False).
    '''
    device = kwargs['device']
    n_centers = true_centers.size()[0]
    prior_shift = D.Normal(0., 100.).sample() #unlike the above, we must sample from the prior on the shift
    noisy_centers = D.Normal(true_centers + prior_shift.item(), sigma)
    Z = noisy_centers.sample((n_sets,)) 
    mix = D.Categorical((torch.ones((n_sets, n_centers))*proportions).to(device))
    comp = D.Independent(D.Normal(Z, tau), 1)
    mix_models = D.MixtureSameFamily(mix, comp)
    X = mix_models.sample((n_obs,))
    X = rearrange(X, 'n_samp b d -> b n_samp d')
    return prior_shift.to(device), Z.to(device), X.to(device)