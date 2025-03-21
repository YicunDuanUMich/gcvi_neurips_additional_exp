import torch
import math
from torch import distributions as D
import torch.nn as nn
import torch.nn.functional as F

class MNISTGenerator(nn.Module):
    def __init__(self, cfg, init, device, latent_size=64, hidden_size=256, image_size=784, weight_path=None):
        super(MNISTGenerator, self).__init__()
        self.latent_size = latent_size
        self.prior = D.Normal(0., 1.)
        self.rotation_prior = D.Uniform(0, 2*math.pi)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh())
        self.cfg = cfg
        self.rotation = nn.Parameter(torch.tensor(init*math.pi/180).to(device), requires_grad=True)
        self.noise = 1e-2
        self.device = device
        self.weight_path = weight_path

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def load(self):
        assert isinstance(self.weight_path, str)
        self.decoder.load_state_dict(torch.load(self.weight_path))
        return

    def rotate(self, x, angle, return_flat=False):
        x = x.view(-1, 28, 28).float()
        n_samples = x.size()[0]
        w, h = (28,28)
        x = x.unsqueeze(1)
        filler_zero = torch.tensor(0., requires_grad=True).to(self.device)
        row1 = torch.stack([torch.cos(angle)*w/h, -torch.sin(angle), filler_zero])
        row2 = torch.stack([torch.sin(angle), torch.cos(angle)*w/h, filler_zero])
        theta = torch.stack([row1, row2]).unsqueeze(0)
        grid = F.affine_grid(theta, x[0].unsqueeze(0).size(), align_corners=False).to(self.device)
        x = F.grid_sample(x, grid.repeat(n_samples, 1, 1, 1), padding_mode='border', align_corners=False)
        x = x.squeeze(1)
        return x

    def sample(self, n_obs=1):
        z = self.prior.sample((n_obs, self.latent_size)).to(self.device)
        angle = self.rotation_prior.sample().to(self.device)
        x = self.decoder(z)
        x = self.rotate(x, angle)
        gen_imgs = self.denorm(x.reshape((-1, 28,28)).detach())
        return x, gen_imgs, z, angle

    def draw_new(self, n_obs=100):
        '''
        Draw latent r.v z from prior,
        create new instance of decoder, and sample batch of
        X from the decoder. 
        '''
        z = self.prior.sample((n_obs,self.latent_size)).to(self.device)
        theta = self.rotation_prior.sample().to(self.device)
        x = self.decoder(z)
        x = self.rotate(x, theta)
        dist = D.Normal(x, self.noise)
        dream = dist.sample()
        return theta, z, dream

    def get_p(self, zs):
        batch_size, K = zs.size()[0], zs.size()[1]
        img_means = self.decoder(zs)
        rot_means = self.rotate(img_means, angle=self.rotation).view(batch_size, K, -1)
        decoder = D.Normal(rot_means, self.noise)
        return decoder

    def log_p_x_given_z_new(self, zs, x):
        '''
        Returns log p(x|z) for a single
        z and x.
        '''
        batch_size, K = zs.size()[0], zs.size()[1]
        decoder = self.get_p(zs)
        reshaped = x.view(batch_size, -1).unsqueeze(1).repeat(1,K,1)
        return decoder.log_prob(reshaped).sum(-1)

    def log_p_z_new(self, zs):
        '''
        Return log p(z) for an instance of z.
        Sum twice; once across latent dimension, once
        across independent draws k=1, \dots, K.
        '''
        return self.prior.log_prob(zs).sum(-1)

    def log_p_x_and_z_new(self, zs, x):
        '''
        Return log p(x,z) for an instance of x,z.
        '''
        return self.log_p_z_new(zs) + self.log_p_x_given_z_new(zs, x)

class InferenceModel(nn.Module):
    '''
    Inference model object endowed with
        - latents, the true (batches of) latent rv's that generated 
        (batches of) data
        - data, the (batches of) real data given
        - processor, a function mapping paramters phi to useful scale
        - encoder, a module object with trainable parameters
        - device, GPU/CPU
    '''
    
    def __init__(self, cfg, latents, data, encoder, encoder2, device):
        super(InferenceModel, self).__init__()
        self.cfg = cfg
        self.latents = latents
        self.data = data.to(device)
        self.encoder = encoder
        self.encoder2 = encoder2
        self.device = device

    def processor(self, q_params):
        latent_size = q_params.size()[1]//2
        means = q_params[:,:latent_size]
        log_sds = q_params[:,latent_size:]
        return means, torch.exp(log_sds)+.0001
    
    def get_batch(self, batch_size=16, all=False):
        '''
        If given_index is None,
        pick a random sample from the data. The true latents are unknown.
        Otherwise, return the ith batch if given_index=i. 
        Reshape batches of x for use in the encoder.
        '''
        if all:
            return self.data
        else:
            n_obs = self.data.size()[0]
            indices = torch.randint(low=0, high=n_obs, size=(batch_size,))
            batch_x = self.data[indices,:]
            return batch_x

    def get_q(self, x, verbose=False):
        '''
        Given (batch of) x, fetch paramters phi and
        construct Distribution object q_phi(z|x)
        for sampling from, log probs, etc.
        '''
        phi = self.encoder(x)
        params = self.processor(phi)
        q_phi = D.Normal(*params)
        # if verbose:
        #     print(q_phi)
        #     print(q_phi.loc)
        #     print(q_phi.scale)
        return q_phi

    def get_q2(self, x):
        params = self.encoder2(x)
        out = params[0]
        up = params[1]
        log_conc = params[2]
        mean = torch.atan2(out, up) + self.cfg.experiment.bass_offset
        q2 = D.VonMises(mean, torch.exp(log_conc)+1e-8)
        return q2

    def print_params(self):
        '''For debugging.'''
        batch_x = self.get_batch(all=True)
        q2 = self.get_q2(batch_x)
        mc_mean = q2.mean
        print('Monte Carlo mean estimate is {}'.format(mc_mean))
        print('Rotation estimate for true data is {}'.format((mc_mean*180/math.pi)%360))
        return mc_mean

    def draw_K_new(self, K, grad=True):
        '''Draw a (batch of) x at random, construct
        encoder q_phi(z|x), draw an K instances z (e.g., for IWAE),
        return all these draws along with data instance x.'''
        x = self.get_batch(batch_size=1)
        batch_size = x.size()[0]
        q_phi = self.get_q(x)
        zs = q_phi.rsample((K,))
        zs = zs.transpose(0,1)
        return (zs, x) if grad else (zs.detach(), x.detach())

    def log_qzx_new(self, zs, x, grad=True):
        '''Given (batch of) x, and (batch of) z,
        compute log q_phi(z|x), the log density.'''
        q_phi = self.get_q(x)
        return q_phi.log_prob(zs).sum(-1) if grad else q_phi.log_prob(zs).detach().sum(-1)

    def log_hyper(self, theta, z, x):
        q2 = self.get_q2(x)
        return q2.log_prob(theta)


class MNISTLatentEncoder(nn.Module):
    def __init__(self):
        super(MNISTLatentEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.encoder(x)
        return x

class MNISTAngleEncoder(nn.Module):
    def __init__(self, favi_offset=0, device='cpu', parameterization='vonmises'):
        super(MNISTAngleEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.dense = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 2)
        )
        self.favi_offset = favi_offset*math.pi/180
        self.device = device
        self.parameterization = parameterization
        assert self.parameterization in ['vonmises', 'gaussian'], "Must parameterize by either Gaussian or Von Mises exponential family."

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        #x = x.view(-1, 784)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = x.mean(dim=0)
        return x

    def get_q(self, x):
        if self.parameterization == 'gaussian':
            eta = self.forward(x)#.clamp(-10., 10.)
            sigma = 0.5
            mu = eta[...,0] * sigma
            return D.Normal(mu, sigma)
        elif self.parameterization == 'vonmises':
            eta = self.forward(x)
            kappa = torch.sqrt(torch.sum(torch.square(eta), dim=-1))
            mu = torch.atan2(eta[...,1], eta[..., 0]) % (2*math.pi)
            return D.VonMises(mu, kappa)
        else:
            raise ValueError('Proper variational distribution not specified.')

    def get_log_prob(self, theta, x):
        offset_theta = theta + self.favi_offset
        q = self.get_q(x)
        return q.log_prob(offset_theta)