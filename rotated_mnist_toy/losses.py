import torch
import torch.distributions as D
import torch.nn as nn
from einops import rearrange, reduce, repeat
import torch.nn.functional as F

#------HELPER FUNCTIONS------#
def rotate_batch_images(batch_tensor, rotation_angles):
    """
    """
    batch_tensor = rearrange(batch_tensor, 'K b (d1 d2) -> K b d1 d2', d1=28, d2=28)

    # Batch size and number of images per batch
    batch_size = batch_tensor.size(0)
    num_images_per_batch = batch_tensor.size(1)

    # Create the rotation matrices
    cos_a = torch.cos(rotation_angles)
    sin_a = torch.sin(rotation_angles)

    # Create 2x3 rotation matrices for each angle
    rotation_matrices = torch.zeros((batch_size, 2, 3), device=batch_tensor.device)
    rotation_matrices[:, 0, 0] = cos_a
    rotation_matrices[:, 0, 1] = -sin_a
    rotation_matrices[:, 1, 0] = sin_a
    rotation_matrices[:, 1, 1] = cos_a

    # Create the affine grids for each image in the batch
    # Output shape of affine_grid will be (B*N, H, W, 2)
    affine_grids = F.affine_grid(rotation_matrices, (batch_size, 1, 28, 28), align_corners=False)
    affine_grids = affine_grids.repeat(1, num_images_per_batch, 1, 1)
    affine_grids = affine_grids.reshape(batch_size * num_images_per_batch, 28, 28, 2)

    # Reshape the input tensor to match the grid
    batch_tensor = batch_tensor.reshape(batch_size * num_images_per_batch, 1, 28, 28)

    # Apply the grid sample
    rotated_images = F.grid_sample(batch_tensor, affine_grids, align_corners=False)

    # Reshape the output back to (B, N, 28, 28)
    rotated_images = rotated_images.reshape(batch_size, num_images_per_batch, 28, 28)

    # Flatten again
    rotated_images = rearrange(rotated_images, 'K b d1 d2 -> K b (d1 d2)')

    return rotated_images
    
def get_imp_weights_amortized_angle_only(angle_particles, latent_particles, pts, K_for_imp_weights, log=False, **kwargs):
    '''
    Given a set of particles, and data poins pts, uses encoder and model to construct
    importance weights.
    '''
    device = kwargs['device']
    angle_encoder = kwargs['angle_encoder']
    gan = kwargs['gan']
    angle_prior = kwargs['angle_prior']
    n_obs = pts.shape[0]
    K = K_for_imp_weights


    # mb_size = pts.shape[0]
    # copy_pts = torch.clone(pts).to(pts.device)#.cpu()
    log_q_angle = angle_encoder.get_q(pts).log_prob(angle_particles)#.cpu()
    log_prior_angle = angle_prior.log_prob(angle_prior.sample((K,))).to(pts.device)#.cpu()

    # Now go get likelihood p(x | z, \theta)
    # image_means = gan.decoder(latent_particles.cpu())
    r_latents = repeat(latent_particles, 'b d -> K b d', K=K)
    image_means = gan.decoder(r_latents)
    rot_images = rotate_batch_images(image_means, angle_particles)
    gen_imgs = gan.denorm(rot_images.reshape((K, n_obs, 28,28)))
    gen_imgs = rearrange(gen_imgs, 'K n_obs d1 d2 -> K n_obs (d1 d2)')
    distr = D.Normal(gen_imgs, gan.noise)
    log_p = reduce(distr.log_prob(pts), 'K nobs 784 -> K', 'sum')#.sum(-1)

    log_weights = log_prior_angle + log_p - log_q_angle
    weights = nn.Softmax(0)(log_weights)

    if log:
        return weights.to(device), log_weights.to(device)
    else:
        return weights.to(device)
    

def log_evidence(zs, xs, **kwargs):
    # Choose data points
    K = kwargs['K']
    angle_prior = kwargs['angle_prior']

    # Get samples, weights
    latent_particles = zs
    angle_particles = angle_prior.sample((K,)).to(xs.device)
    weights, log_weights = get_imp_weights_amortized_angle_only(angle_particles, latent_particles, xs, K_for_imp_weights=K, log=True, **kwargs)
    result = torch.logsumexp(log_weights.detach(), 0) + torch.log(torch.tensor(1/K))

    # Return loss
    return result

def elbo(zs, xs, **kwargs):
    # Choose data points
    K = kwargs['K']
    angle_encoder = kwargs['angle_encoder']

    # Get samples, weights
    latent_particles = zs
    angle_particles = angle_encoder.get_q(xs).rsample((K,))
    weights, log_weights = get_imp_weights_amortized_angle_only(angle_particles, latent_particles, xs, K_for_imp_weights=K, log=True, **kwargs)
    result = log_weights.mean() # a K-sample Monte Carlo estimate of the ELBO, avg. of K different log(p(z,x)/q(z)) values. Not an IWBO. 
    return result

def favi_loss(zs, xs, **kwargs):
    # Generate synethetic data batch
    angle_prior = kwargs['angle_prior']
    gan = kwargs['gan']
    device = kwargs['device']
    angle_encoder = kwargs['angle_encoder']
    K = 1

    n_obs = xs.shape[0]

    sim_angle = angle_prior.sample((K,)).to(device)
    r_latents = repeat(zs, 'b d -> K b d', K=K)
    image_means = gan.decoder(r_latents)
    rot_images = rotate_batch_images(image_means, sim_angle)
    gen_imgs = gan.denorm(rot_images.reshape((K, n_obs, 28,28)))
    gen_imgs = rearrange(gen_imgs, 'K n_obs d1 d2 -> K n_obs (d1 d2)')
    distr = D.Normal(gen_imgs, gan.noise)
    sim_x = distr.sample()

    return -1*angle_encoder.get_log_prob(sim_angle, sim_x).mean()