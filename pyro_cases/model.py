import torch
import math
from torch import nn
import pyro
import copy
import pyro.distributions as dist

from einops import repeat, rearrange
from typing import Dict

import sbibm
from pathlib import Path

SBIBM_INSTALL_PATH = Path(sbibm.__file__).parent


class DenseEncoderGaussian(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        assert self.out_dim % 2 == 0
        self.hidden_dim = hidden_dim
        self.scale = torch.tensor(1 / math.sqrt(hidden_dim))
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.relu = nn.ReLU()

        # Follow proper initialization from our paper
        torch.nn.init.normal_(self.linear1.weight)
        torch.nn.init.zeros_(self.linear2.weight)

    @classmethod
    def eta_to_mu_sigma2(cls, eta1, eta2):
        sigma2 = -1 / (2 * eta2)
        mu = eta1 * sigma2
        return mu, sigma2
    
    @classmethod
    def gaussian_log_density_natural(cls, eta1, eta2, x):
        return eta1 * x + \
               eta2 * (x ** 2) + \
               (eta1 ** 2) / (4 * eta2) + \
               0.5 * torch.log(-eta2 / torch.pi)
    
    def get_eta(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # loc, scale = torch.mul(x, self.scale).chunk(2, dim=-1)
        # # return loc, torch.exp(scale).clamp(min=0.01, max=10.0)
        # return loc, torch.ones_like(scale)
        eta1, eta2 = torch.mul(x, self.scale).chunk(2, dim=-1)
        eta2 = eta2 - 1.0
        assert eta1.shape == eta2.shape
        eta2 = eta2.clamp(min=-1000.0, max=-0.1)
        return eta1, eta2

    def forward(self, x):
        eta1, eta2 = self.get_eta(x)
        mu, sigma2 = self.eta_to_mu_sigma2(eta1, eta2)
        return mu, sigma2.sqrt()
    
    def batch_favi_loss(self, theta, x):
        # loc, scale = self(x)
        # sigma2 = scale ** 2
        # return (0.5 * torch.log(sigma2) + (theta - loc) ** 2 / (2 * sigma2)).sum(dim=-1)
        eta1, eta2 = self.get_eta(x)
        log_dens = self.gaussian_log_density_natural(eta1, eta2, theta)
        return -(log_dens.sum(dim=-1))


class BaseVAE(nn.Module):
    x_dim = None
    theta_dim = None
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.register_buffer("dummy_param", torch.zeros(0))
        self.init_network(hidden_dim)

    def init_network(self, hidden_dim):
        self.encoder = DenseEncoderGaussian(self.x_dim, 
                                            self.theta_dim * 2, 
                                            hidden_dim)
    
    @property
    def device(self):
        return self.dummy_param.device

    def model(self, batch_size, sample_dict: Dict[str, torch.Tensor]):
        raise NotImplementedError()

    def guide(self, batch_size, sample_dict: Dict[str, torch.Tensor]):
        pyro.module("encoder", self.encoder)
        x = self.extract_x(sample_dict)
        assert batch_size == x.shape[0]

        theta_loc, theta_scale = self.encoder(x)
        with pyro.plate("plate_batch", batch_size):    
            pyro.sample("latent", dist.Normal(theta_loc, theta_scale).to_event(1))

    def get_obs_sample_dict(self, obs_seed):
        pyro.set_rng_seed(obs_seed)
        return self.generate_sample_dict(batch_size=1)

    def get_observation(self, obs_seed):
        sample_dict = self.get_obs_sample_dict(obs_seed)
        return self.extract_x(sample_dict), self.extract_theta(sample_dict)

    def generate_sample_dict(self, batch_size):
        return self.model(batch_size=batch_size, sample_dict=None)
    
    def _extract_x_func(self, sample_dict):
        return sample_dict["x"]
    
    def extract_x(self, sample_dict):
        x = self._extract_x_func(sample_dict)
        assert x.shape[-1] == self.x_dim
        return x
    
    def _extract_theta_func(self, sample_dict):
        return sample_dict["theta"]
    
    def extract_theta(self, sample_dict):
        theta = self._extract_theta_func(sample_dict)
        assert theta.shape[-1] == self.theta_dim
        return theta


class GaussianLinearVAE(BaseVAE):
    x_dim = 10
    theta_dim = 10

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
        else:
            sample_dict = copy.copy(sample_dict)

        with pyro.plate("plate_batch", batch_size):
            theta_loc = torch.zeros((self.theta_dim, ), device=self.device)
            theta_cov = 0.1 * torch.eye(self.theta_dim, device=self.device)
            sample_dict["theta"] = pyro.sample("latent", dist.MultivariateNormal(loc=theta_loc,
                                                                                 covariance_matrix=theta_cov))
            sample_dict["x"] = pyro.sample("obs", 
                                            dist.MultivariateNormal(loc=sample_dict["theta"], covariance_matrix=theta_cov), 
                                            obs=sample_dict["x"] if "x" in sample_dict else None)
        return sample_dict
            

class GaussianLinearUniformVAE(BaseVAE):
    x_dim = 10
    theta_dim = 10

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
        else:
            sample_dict = copy.copy(sample_dict)

        with pyro.plate("plate_batch", batch_size):
            theta_loc = torch.zeros((self.theta_dim, ), device=self.device)
            theta_cov = torch.eye(self.theta_dim, device=self.device)
            sample_dict["theta"] = pyro.sample("latent", dist.MultivariateNormal(loc=theta_loc,
                                                                                 covariance_matrix=theta_cov))

            cov_m = 0.1 * torch.eye(self.theta_dim, device=self.device)
            sample_dict["x"] = pyro.sample("obs",
                                           dist.MultivariateNormal(loc=sample_dict["theta"], covariance_matrix=cov_m),
                                           obs=sample_dict["x"] if "x" in sample_dict else None)
        return sample_dict
                    

class SLCPVAE(BaseVAE):
    x_dim = 8
    theta_dim = 5

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
        else:
            sample_dict = copy.copy(sample_dict)
            sample_dict["x"] = rearrange(sample_dict["x"], "b (r two) -> r b two", two=2)
        
        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        with plate_batch:
            theta_loc = torch.zeros((self.theta_dim, ), device=self.device)
            theta_cov = 0.25 * torch.eye(self.theta_dim, device=self.device)
            sample_dict["theta"] = pyro.sample("latent", dist.MultivariateNormal(loc=theta_loc,
                                                                                 covariance_matrix=theta_cov))
        plate_r = pyro.plate("plate_r", 4, dim=-2)
        with plate_batch, plate_r:
            m = torch.stack((sample_dict["theta"][..., 0], 
                             sample_dict["theta"][..., 1]), 
                             dim=-1)  # (..., b, 2)
            m = repeat(m, "... b k -> ... r b k", r=4)
            s1 = repeat(sample_dict["theta"][..., 2].abs(), 
                        "... b -> ... r b", r=4)
            s2 = repeat(sample_dict["theta"][..., 3].abs(), 
                        "... b -> ... r b", r=4)
            rho = repeat(torch.tanh(sample_dict["theta"][..., 4]), 
                         "... b -> ... r b", r=4)
            S = torch.zeros((*m.shape[:-1], 2, 2), device=self.device)  # (..., r, b, 2, 2)
            S[..., 0, 0] = s1 ** 2
            S[..., 0, 1] = rho * s1 * s2
            S[..., 1, 0] = rho * s1 * s2
            S[..., 1, 1] = s2 ** 2
            S[..., 0, 0] += 1e-4
            S[..., 1, 1] += 1e-4

            sample_dict["x"] = pyro.sample("obs",
                                            dist.MultivariateNormal(loc=m, covariance_matrix=S),
                                            obs=sample_dict["x"] if "x" in sample_dict else None)
            
        sample_dict["x"] = rearrange(sample_dict["x"], "r b two -> b (r two)")
        return sample_dict
            

class SLCPwDistractorVAE(SLCPVAE):
    x_dim = 100
    theta_dim = 5

    def __init__(self, hidden_dim):
        super().__init__(hidden_dim)

        # from sbibm
        self.register_buffer("permutation_idx",
                             torch.load(SBIBM_INSTALL_PATH / "tasks/slcp/files/permutation_idx.torch"))
        self.gmm = torch.load(SBIBM_INSTALL_PATH / "tasks/slcp/files/gmm.torch")
        
    def model(self, batch_size, sample_dict):
        if sample_dict is not None:
            sample_dict = copy.copy(sample_dict)
            assert sample_dict["x"].shape[-1] == self.permutation_idx.shape[0]
            sample_dict["x"] = torch.scatter(torch.zeros_like(sample_dict["x"]), 
                                             dim=1, 
                                             index=repeat(self.permutation_idx, 
                                                            "d -> b d", 
                                                            b=batch_size), 
                                             src=sample_dict["x"])[:, :8]
        sample_dict = super().model(batch_size, sample_dict)

        # noise = self.gmm.sample((batch_size, )).to(dtype=sample_dict["x"].dtype,
        #                                            device=sample_dict["x"].device)
        noise = torch.randn_like(sample_dict["x"][:, 0:1].repeat(1, 92))
        sample_dict["x"] = torch.cat([sample_dict["x"], noise], dim=1)[:, self.permutation_idx]
        return sample_dict
    

class BeroulliGLMRAWVAE(BaseVAE):
    x_dim = 100
    theta_dim = 10

    def __init__(self, hidden_dim):
        super().__init__(hidden_dim)

        # from sbibm
        self.register_buffer("stimulus_I",
                             torch.load(SBIBM_INSTALL_PATH / "tasks/bernoulli_glm/files/stimulus_I.pt"))
        self.register_buffer("design_matrix",
                             torch.load(SBIBM_INSTALL_PATH / "tasks/bernoulli_glm/files/design_matrix.pt"))
        
    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
        else:
            sample_dict = copy.copy(sample_dict)

        with pyro.plate("plate_batch", batch_size):
            M = self.theta_dim - 1
            m_D = torch.diag(torch.ones(M)) - torch.diag(torch.ones(M - 1), -1)
            m_F = torch.matmul(m_D, m_D) + torch.diag(1.0 * torch.arange(M) / (M)) ** 0.5
            Binv = torch.zeros(size=(M + 1, M + 1))
            Binv[0, 0] = 0.5  # offset
            Binv[1:, 1:] = torch.matmul(m_F.T, m_F)  # filter
            Binv = Binv.to(device=self.device)
            sample_dict["theta"] = pyro.sample("latent", 
                                               dist.MultivariateNormal(loc=torch.zeros((M + 1, ), device=self.device),
                                                                       precision_matrix=Binv))
        
            # Simulate GLM
            psi = torch.matmul(rearrange(self.design_matrix, "k1 k2 -> 1 k1 k2"),
                               rearrange(sample_dict["theta"], "b k -> b k 1")).squeeze(-1)
            z = 1 / (1 + torch.exp(-psi))  # (b, 100)
            sample_dict["x"] = pyro.sample("obs",
                                            dist.Bernoulli(z).to_event(1),
                                            obs=sample_dict["x"] if "x" in sample_dict else None)  # (b, 100)
        return sample_dict
    

class BernoulliGLMVAE(BeroulliGLMRAWVAE):
    x_dim = 10
    theta_dim = 10

    def model(self, batch_size, sample_dict):
        if sample_dict is not None:
            sample_dict = copy.copy(sample_dict)
        sample_dict = super().model(batch_size, sample_dict)  # (b, 100)
        num_spikes = torch.sum(sample_dict["x"], dim=-1, keepdim=True)  # (b, 1)
        sta = torch.nn.functional.conv1d(
                    sample_dict["x"].view(batch_size, 1, -1), 
                    self.stimulus_I.view(1, 1, -1), 
                    padding=8
                ).squeeze(-2)[:, -9:]  # (n, 9)
        sample_dict["stat"] = torch.cat([num_spikes, sta], dim=-1)
        return sample_dict
    
    def _extract_x_func(self, sample_dict):
        return sample_dict["stat"]


class GaussianMixtureVAE(BaseVAE):
    x_dim = 2
    theta_dim = 2

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
        else:
            sample_dict = copy.copy(sample_dict)

        with pyro.plate("plate_batch", batch_size):
            theta_loc = torch.zeros((self.theta_dim, ), device=self.device)
            theta_cov = 9 * torch.eye(self.theta_dim, device=self.device)
            sample_dict["theta"] = pyro.sample("latent", dist.MultivariateNormal(loc=theta_loc,
                                                                                 covariance_matrix=theta_cov))
            
            idx = (torch.rand((batch_size, 1), device=self.device) > 0.5).long()

            # Select loc and scales according to mixture index
            loc = torch.tensor([1.0, 1.0], device=self.device)[idx] * sample_dict["theta"]
            scale = torch.tensor([1.0, 0.1], device=self.device)[idx]

            sample_dict["x"] = pyro.sample("obs", 
                                           dist.Normal(loc=loc, scale=scale).to_event(1),
                                           obs=sample_dict["x"] if "x" in sample_dict else None)
        return sample_dict


class TwoMoonsVAE(BaseVAE):
    x_dim = 2
    theta_dim = 2

    def model(self, batch_size, sample_dict):
        ori_sample_dict_is_none = False
        if sample_dict is None:
            sample_dict = {}
            ori_sample_dict_is_none = True
        else:
            sample_dict = copy.copy(sample_dict)

        with pyro.plate("plate_batch", batch_size):
            theta_loc = torch.zeros((self.theta_dim, ), device=self.device)
            theta_cov = 0.25 * torch.eye(self.theta_dim, device=self.device)
            sample_dict["theta"] = pyro.sample("latent", dist.MultivariateNormal(loc=theta_loc,
                                                                                 covariance_matrix=theta_cov))
            if ori_sample_dict_is_none:
                r = torch.randn_like(sample_dict["theta"][:, 0]) * 0.01 + 0.1
                alpha = (torch.rand_like(sample_dict["theta"][:, 0]) - 0.5) * torch.pi
                x1 = r * torch.cos(alpha) + 0.25 - torch.abs(sample_dict["theta"].sum(dim=-1)) / math.sqrt(2)
                x2 = r * torch.sin(alpha) + (sample_dict["theta"][:, 1] - sample_dict["theta"][:, 0]) / math.sqrt(2)
                sample_dict["x"] = torch.stack([x1, x2], dim=-1)
                return sample_dict
            
            r2 = (sample_dict["x"][:, 0] - 0.25 + torch.abs(sample_dict["theta"].sum(dim=-1)) / math.sqrt(2)) ** 2 + \
                 (sample_dict["x"][:, 1] - (sample_dict["theta"][:, 1] - sample_dict["theta"][:, 0]) / math.sqrt(2)) ** 2
            r = torch.sqrt(r2)
            pyro.sample("r", 
                        dist.Normal(loc=0.1 * torch.ones_like(r),
                                    scale=0.01 * torch.ones_like(r)),
                        obs=r)
        return sample_dict


class ARM_anova_randon_nopred(BaseVAE):
    x_dim = 52
    theta_dim = 6
    
    def _extract_x_func(self, sample_dict):
        y = rearrange(sample_dict["y"], "n b -> b n")
        sigma_a = sample_dict["sigma_a"].view(-1, 1)
        sigma_y = sample_dict["sigma_y"].view(-1, 1)
        return torch.cat([y, sigma_a, sigma_y], dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        mu_a = sample_dict["mu_a"].view(-1, 1)
        a = rearrange(sample_dict["a"], "j b -> b j")
        return torch.cat([mu_a, a], dim=-1)
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_j = pyro.plate("plate_j", sample_dict["J"], dim=-2)

        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)
        with plate_batch:    
            pyro.sample("mu_a", dist.Normal(theta_loc[:, 0], theta_scale[:, 0]))
            with plate_j:
                pyro.sample("a", dist.Normal(rearrange(theta_loc[:, 1:], "b j -> j b"), 
                                             rearrange(theta_scale[:, 1:], "b j -> j b")))

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 50
            sample_dict["J"] = 5
            sample_dict["sigma_a"] = torch.ones(batch_size, device=self.device) * 10
            sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
            sample_dict["county"] = repeat(torch.arange(sample_dict["J"], device=self.device), 
                                           "j -> (j r)", r=sample_dict["N"] // sample_dict["J"])
        else:
            sample_dict = copy.copy(sample_dict)
        
        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_j = pyro.plate("plate_j", sample_dict["J"], dim=-2)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)

        with plate_batch:
            sample_dict["mu_a"] = pyro.sample("mu_a", dist.Normal(torch.tensor(0.0, device=self.device), 
                                                                  torch.tensor(1.0, device=self.device)))
            with plate_j:
                sample_dict["a"] = pyro.sample("a", dist.Normal(10 * sample_dict["mu_a"],
                                                                sample_dict["sigma_a"]))  # (b, j)
            with plate_n:
                y_hat = sample_dict["a"][..., sample_dict["county"], :]
                sample_dict["y"] = pyro.sample("y",
                                                dist.Normal(y_hat,
                                                            repeat(sample_dict["sigma_y"], 
                                                                   "b -> n b", 
                                                                   n=sample_dict["N"])),
                                                obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict

class ARM_anova_randon_nopred_chr(BaseVAE):
    x_dim = 52
    theta_dim = 6
    
    def _extract_x_func(self, sample_dict):
        y = rearrange(sample_dict["y"], "n b -> b n")
        sigma_a = sample_dict["sigma_a"].view(-1, 1)
        sigma_y = sample_dict["sigma_y"].view(-1, 1)
        return torch.cat([y, sigma_a, sigma_y], dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        mu_a = sample_dict["mu_a"].view(-1, 1)
        eta = rearrange(sample_dict["eta"], "j b -> b j")
        return torch.cat([mu_a, eta], dim=-1)
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_j = pyro.plate("plate_j", sample_dict["J"], dim=-2)

        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)
        with plate_batch:    
            pyro.sample("mu_a", dist.Normal(theta_loc[:, 0], theta_scale[:, 0]))
            with plate_j:
                pyro.sample("eta", dist.Normal(rearrange(theta_loc[:, 1:], "b j -> j b"), 
                                               rearrange(theta_scale[:, 1:], "b j -> j b")))

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 50
            sample_dict["J"] = 5
            sample_dict["sigma_a"] = torch.ones(batch_size, device=self.device) * 10
            sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
            sample_dict["county"] = repeat(torch.arange(sample_dict["J"], device=self.device), 
                                           "j -> (j r)", r=sample_dict["N"] // sample_dict["J"])
        else:
            sample_dict = copy.copy(sample_dict)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_j = pyro.plate("plate_j", sample_dict["J"], dim=-2)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)

        with plate_batch:
            sample_dict["mu_a"] = pyro.sample("mu_a", dist.Normal(torch.tensor(0.0, device=self.device), 
                                                                  torch.tensor(1.0, device=self.device)))
            with plate_j:
                sample_dict["eta"] = pyro.sample("eta", dist.Normal(torch.tensor(0.0, device=self.device), 
                                                                    torch.tensor(1.0, device=self.device)))
            a = sample_dict["mu_a"] + sample_dict["sigma_a"] * sample_dict["eta"]  # (j, b)
            with plate_n:
                y_hat = a[..., sample_dict["county"], :]
                sample_dict["y"] = pyro.sample("y",
                                                dist.Normal(y_hat,
                                                            repeat(sample_dict["sigma_y"], 
                                                                "b -> n b", 
                                                                n=sample_dict["N"])),
                                                obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict

class ARM_congress(BaseVAE):
    x_dim = 151
    theta_dim = 3
    
    def _extract_x_func(self, sample_dict):
        vote_88 = rearrange(sample_dict["vote_88"], "n b -> b n")
        vote_86 = rearrange(sample_dict["vote_86"], "n b -> b n")
        incumbency_88 = rearrange(sample_dict["incumbency_88"], "n b -> b n")
        sigma = sample_dict["sigma"].view(-1, 1)
        return torch.cat([vote_88, vote_86, incumbency_88, sigma], dim=-1)  # (b, 151)
    
    def _extract_theta_func(self, sample_dict):
        return sample_dict["beta"]
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)

        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)
        with plate_batch:    
            pyro.sample("beta", dist.Normal(theta_loc, theta_scale).to_event(1))

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 50
            sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
            sample_dict["incumbency_88"] = torch.rand(sample_dict["N"], batch_size, device=self.device) * 100 + 10
            sample_dict["vote_86"] = torch.rand(sample_dict["N"], batch_size, device=self.device) * 100 + 10
        else:
            sample_dict = copy.copy(sample_dict)
            sample_dict["sigma"] = sample_dict["sigma_y"] * 100

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)

        with plate_batch:
            sample_dict["beta"] = pyro.sample("beta", dist.Normal(torch.zeros((3, ), device=self.device),
                                                                  0.2 * torch.ones((3, ), device=self.device)).to_event(1))  # (b, 3)
            with plate_n:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["vote_86"] + \
                        sample_dict["beta"][..., 2] * sample_dict["incumbency_88"]  # (n, b)
                sample_dict["vote_88"] = pyro.sample("vote_88", 
                                                    dist.Normal(y_hat,
                                                                scale=repeat(sample_dict["sigma"], 
                                                                            "b -> n b", 
                                                                            n=sample_dict["N"])),
                                                    obs=sample_dict["vote_88"] if "vote_88" in sample_dict else None)
        return sample_dict


class ARM_earnings1(BaseVAE):
    x_dim = 150
    theta_dim = 3
    
    def _extract_x_func(self, sample_dict):
        earn_pos = rearrange(sample_dict["earn_pos"], "n b -> b n")
        height = rearrange(sample_dict["height"], "n b -> b n")
        male = rearrange(sample_dict["male"], "n b -> b n")
        return torch.cat([earn_pos, height, male], dim=-1)  # (b, 151)
    
    def _extract_theta_func(self, sample_dict):
        return sample_dict["beta"]
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)

        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)
        with plate_batch:    
            pyro.sample("beta", dist.Normal(theta_loc, theta_scale).to_event(1))

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 50
            sample_dict["height"] = torch.randint(low=150, high=190, size=(sample_dict["N"], batch_size),
                                                  device=self.device)
            sample_dict["male"] = torch.randint(low=0, high=2, size=(sample_dict["N"], batch_size),
                                                device=self.device)
        else:
            sample_dict = copy.copy(sample_dict)
        
        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)

        with plate_batch:
            sample_dict["beta"] = pyro.sample("beta", dist.Normal(torch.zeros((3, ), device=self.device),
                                                                  0.2 * torch.ones((3, ), device=self.device)).to_event(1))
            logits = sample_dict["beta"][:, 0] + \
                     sample_dict["beta"][:, 1] * sample_dict["height"] + \
                     sample_dict["beta"][:, 2] * sample_dict["male"]
            with plate_n:
                sample_dict["earn_pos"] = pyro.sample("earn_pos", 
                                                    dist.Bernoulli(logits=logits),
                                                    obs=sample_dict["earn_pos"] if "earn_pos" in sample_dict else None)
        return sample_dict


class ARM_earnings2(BaseVAE):
    x_dim = 151
    theta_dim = 3
    
    def _extract_x_func(self, sample_dict):
        log_earnings = rearrange(sample_dict["log_earnings"], "n b -> b n")
        height = rearrange(sample_dict["height"], "n b -> b n")
        male = rearrange(sample_dict["male"], "n b -> b n")
        sigma = sample_dict["sigma"].view(-1, 1)
        return torch.cat([log_earnings, height, male, sigma], dim=-1)  # (b, 151)
    
    def _extract_theta_func(self, sample_dict):
        return sample_dict["beta"]
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)
        with plate_batch:    
            pyro.sample("beta", dist.Normal(theta_loc, theta_scale).to_event(1))

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 50
            sample_dict["height"] = torch.randint(low=150, high=190, size=(sample_dict["N"], batch_size),
                                                  device=self.device)
            sample_dict["male"] = torch.randint(low=0, high=2, size=(sample_dict["N"], batch_size),
                                                device=self.device)
            sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)

        with plate_batch:
            sample_dict["beta"] = pyro.sample("beta", dist.Normal(torch.zeros((3, ), device=self.device),
                                                                  0.2 * torch.ones((3, ), device=self.device)).to_event(1))
            log_earnings = sample_dict["beta"][:, 0] + \
                           sample_dict["beta"][:, 1] * sample_dict["height"] + \
                           sample_dict["beta"][:, 2] * sample_dict["male"]
            with plate_n:
                sample_dict["log_earnings"] = pyro.sample("log_earnings", 
                                                        dist.Normal(loc=log_earnings,
                                                                    scale=repeat(sample_dict["sigma"],
                                                                                "b -> n b",
                                                                                n=sample_dict["N"])),
                                                        obs=sample_dict["log_earnings"] if "log_earnings" in sample_dict else None)
        return sample_dict


class ARM_earnings_latin_square(BaseVAE):
    x_dim = 607
    theta_dim = 76
    def __init__(self, hidden_dim):
        super().__init__(hidden_dim)

        self.x_format = [("x", 300), ("y", 300), 
                        ("sigma_a1", 1), ("sigma_a2", 1), 
                        ("sigma_b1", 1), ("sigma_b2", 1), 
                        ("sigma_c", 1), ("sigma_d", 1), 
                        ("sigma_y", 1)]
        self.theta_format = [("mu_a1", ()), ("mu_a2", ()), 
                            ("mu_b1", ()), ("mu_b2", ()), 
                            ("mu_c", ()), ("mu_d", ()),
                            ("a1", (5, 1)), ("a2", (5, 1)), 
                            ("b1", (5, )), ("b2", (5, )), 
                            ("c", (5, 5)), ("d", (5, 5))]
        
    def _extract_x_func(self, sample_dict):
        x_list = []
        for x_name, x_dim in self.x_format:
            x = rearrange(sample_dict[x_name], "... b -> b (...)")
            assert x.shape[-1] == x_dim
            x_list.append(x)
        return torch.cat(x_list, dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        theta_list = []
        for theta_name, theta_dim in self.theta_format:
            theta = sample_dict[theta_name]
            assert theta.shape[:-1] == theta_dim
            theta_list.append(rearrange(theta, "... b -> b (...)"))
        return torch.cat(theta_list, dim=-1)
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_a = pyro.plate("plate_a", sample_dict["n_eth"], dim=-3)
        plate_b = pyro.plate("plate_b", sample_dict["n_age"], dim=-2)

        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)

        with plate_batch:
            pyro.sample("mu_a1", dist.Normal(theta_loc[:, 0], theta_scale[:, 0]))
            pyro.sample("mu_a2", dist.Normal(theta_loc[:, 1], theta_scale[:, 1]))
            pyro.sample("mu_b1", dist.Normal(theta_loc[:, 2], theta_scale[:, 2]))
            pyro.sample("mu_b2", dist.Normal(theta_loc[:, 3], theta_scale[:, 3]))
            pyro.sample("mu_c", dist.Normal(theta_loc[:, 4], theta_scale[:, 4]))
            pyro.sample("mu_d", dist.Normal(theta_loc[:, 5], theta_scale[:, 5]))
            with plate_a:
                pyro.sample("a1", dist.Normal(
                    rearrange(theta_loc[:, 6:11], "b n_eth -> n_eth 1 b"),
                    rearrange(theta_scale[:, 6:11], "b n_eth -> n_eth 1 b")
                ))
                pyro.sample("a2", dist.Normal(
                    rearrange(theta_loc[:, 11:16], "b n_eth -> n_eth 1 b"),
                    rearrange(theta_scale[:, 11:16], "b n_eth -> n_eth 1 b")
                ))
            with plate_b:
                pyro.sample("b1", dist.Normal(
                    rearrange(theta_loc[:, 16:21], "b n_age -> n_age b"),
                    rearrange(theta_scale[:, 16:21], "b n_age -> n_age b")
                ))
                pyro.sample("b2", dist.Normal(
                    rearrange(theta_loc[:, 21:26], "b n_age -> n_age b"),
                    rearrange(theta_scale[:, 21:26], "b n_age -> n_age b")
                ))
            with plate_a, plate_b:
                pyro.sample("c", dist.Normal(
                    rearrange(theta_loc[:, 26:51], "b (n_eth n_age) -> n_eth n_age b", n_eth=5, n_age=5),
                    rearrange(theta_scale[:, 26:51], "b (n_eth n_age) -> n_eth n_age b", n_eth=5, n_age=5)
                ))
                pyro.sample("d", dist.Normal(
                    rearrange(theta_loc[:, 51:], "b (n_eth n_age) -> n_eth n_age b", n_eth=5, n_age=5),
                    rearrange(theta_scale[:, 51:], "b (n_eth n_age) -> n_eth n_age b", n_eth=5, n_age=5)
                ))

    @classmethod
    def _add_sample(cls, 
                    sample_dict, 
                    name: str, 
                    dist):
        sample_dict[name] = pyro.sample(name, dist)

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 300
            sample_dict["n_age"] = 5
            sample_dict["n_eth"] = 5
            sample_dict["sigma_a1"] = torch.ones(batch_size, device=self.device) * 100
            sample_dict["sigma_a2"] = torch.ones(batch_size, device=self.device) * 100
            sample_dict["sigma_b1"] = torch.ones(batch_size, device=self.device) * 100
            sample_dict["sigma_b2"] = torch.ones(batch_size, device=self.device) * 10
            sample_dict["sigma_c"] = torch.ones(batch_size, device=self.device) * 100
            sample_dict["sigma_d"] = torch.ones(batch_size, device=self.device) * 10
            sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
            assert sample_dict["N"] % (sample_dict["n_eth"] * sample_dict["n_age"]) == 0
            eth_age_pair = torch.stack(torch.meshgrid(torch.arange(sample_dict["n_eth"], device=self.device), 
                                                      torch.arange(sample_dict["n_age"], device=self.device),
                                                      indexing="ij"), dim=-1)
            sample_dict["eth"] = repeat(eth_age_pair[..., 0],
                                        "k1 k2 -> (r k1 k2)", 
                                        r=sample_dict["N"] // (sample_dict["n_eth"] * sample_dict["n_age"]))
            sample_dict["age"] = repeat(eth_age_pair[..., 1],
                                        "k1 k2 -> (r k1 k2)", 
                                        r=sample_dict["N"] // (sample_dict["n_eth"] * sample_dict["n_age"]))
            sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 10
        else:
            sample_dict = copy.copy(sample_dict)
        
        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_a = pyro.plate("plate_a", sample_dict["n_eth"], dim=-3)
        plate_b = pyro.plate("plate_b", sample_dict["n_age"], dim=-2)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)
           
        with plate_batch:
            for param_name in ["mu_a1", "mu_a2", "mu_b1", "mu_b2", "mu_c", "mu_d"]:
                self._add_sample(sample_dict, 
                                 param_name,
                                 dist.Normal(torch.tensor(0.0, device=self.device),
                                             torch.tensor(1.0, device=self.device)))
            with plate_a:
                self._add_sample(sample_dict,
                                 "a1",
                                 dist.Normal(10 * sample_dict["mu_a1"],
                                             sample_dict["sigma_a1"]))
                self._add_sample(sample_dict,
                                 "a2",
                                 dist.Normal(10 * sample_dict["mu_a2"],
                                             sample_dict["sigma_a2"]))
                
            with plate_b:
                self._add_sample(sample_dict,
                                 "b1",
                                 dist.Normal(10 * sample_dict["mu_b1"],
                                             sample_dict["sigma_b1"]))
                self._add_sample(sample_dict,
                                 "b2",
                                 dist.Normal(0.1 * sample_dict["mu_b2"],
                                             sample_dict["sigma_b2"]))
                
            with plate_a, plate_b:
                self._add_sample(sample_dict,
                                 "c",
                                 dist.Normal(10 * sample_dict["mu_c"],
                                             sample_dict["sigma_c"]))
                self._add_sample(sample_dict,
                                 "d",
                                 dist.Normal(0.1 * sample_dict["mu_d"],
                                             sample_dict["sigma_d"]))
                
            with plate_n:
                c = sample_dict["c"][sample_dict["eth"], sample_dict["age"], :]  # (n, b)
                d = sample_dict["d"][sample_dict["eth"], sample_dict["age"], :]  # (n, b)
                
                y_hat = sample_dict["a1"].squeeze(-2)[sample_dict["eth"], :] + \
                        sample_dict["a2"].squeeze(-2)[sample_dict["eth"], :] * sample_dict["x"] + \
                        sample_dict["b1"][sample_dict["age"], :] + \
                        sample_dict["b2"][sample_dict["age"], :] * sample_dict["x"] + \
                        c + d * sample_dict["x"]
                sample_dict["y"] = pyro.sample("y",
                                               dist.Normal(y_hat, 
                                                           repeat(sample_dict["sigma_y"], 
                                                                  "b -> n b", 
                                                                  n=sample_dict["N"])),
                                               obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_earnings_latin_square_chr(BaseVAE):
    x_dim = 607
    theta_dim = 76
    def __init__(self, hidden_dim):
        super().__init__(hidden_dim)

        self.x_format = [("x", 300), ("y", 300), 
                        ("sigma_a1", 1), ("sigma_a2", 1), 
                        ("sigma_b1", 1), ("sigma_b2", 1), 
                        ("sigma_c", 1), ("sigma_d", 1), 
                        ("sigma_y", 1)]
        self.theta_format = [("mu_a1", ()), ("mu_a2", ()), 
                            ("mu_b1", ()), ("mu_b2", ()), 
                            ("mu_c", ()), ("mu_d", ()),
                            ("eta_a1", (5, 1)), ("eta_a2", (5, 1)), 
                            ("eta_b1", (5, )), ("eta_b2", (5, )), 
                            ("eta_c", (5, 5)), ("eta_d", (5, 5))]
        
    def _extract_x_func(self, sample_dict):
        x_list = []
        for x_name, x_dim in self.x_format:
            x = rearrange(sample_dict[x_name], "... b -> b (...)")
            assert x.shape[-1] == x_dim
            x_list.append(x)
        return torch.cat(x_list, dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        theta_list = []
        for theta_name, theta_dim in self.theta_format:
            theta = sample_dict[theta_name]
            assert theta.shape[:-1] == theta_dim
            theta_list.append(rearrange(theta, "... b -> b (...)"))
        return torch.cat(theta_list, dim=-1)
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_a = pyro.plate("plate_a", sample_dict["n_eth"], dim=-3)
        plate_b = pyro.plate("plate_b", sample_dict["n_age"], dim=-2)

        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)

        with plate_batch:
            pyro.sample("mu_a1", dist.Normal(theta_loc[:, 0], theta_scale[:, 0]))
            pyro.sample("mu_a2", dist.Normal(theta_loc[:, 1], theta_scale[:, 1]))
            pyro.sample("mu_b1", dist.Normal(theta_loc[:, 2], theta_scale[:, 2]))
            pyro.sample("mu_b2", dist.Normal(theta_loc[:, 3], theta_scale[:, 3]))
            pyro.sample("mu_c", dist.Normal(theta_loc[:, 4], theta_scale[:, 4]))
            pyro.sample("mu_d", dist.Normal(theta_loc[:, 5], theta_scale[:, 5]))
            with plate_a:
                pyro.sample("eta_a1", dist.Normal(
                    rearrange(theta_loc[:, 6:11], "b n_eth -> n_eth 1 b"),
                    rearrange(theta_scale[:, 6:11], "b n_eth -> n_eth 1 b")
                ))
                pyro.sample("eta_a2", dist.Normal(
                    rearrange(theta_loc[:, 11:16], "b n_eth -> n_eth 1 b"),
                    rearrange(theta_scale[:, 11:16], "b n_eth -> n_eth 1 b")
                ))
            with plate_b:
                pyro.sample("eta_b1", dist.Normal(
                    rearrange(theta_loc[:, 16:21], "b n_age -> n_age b"),
                    rearrange(theta_scale[:, 16:21], "b n_age -> n_age b")
                ))
                pyro.sample("eta_b2", dist.Normal(
                    rearrange(theta_loc[:, 21:26], "b n_age -> n_age b"),
                    rearrange(theta_scale[:, 21:26], "b n_age -> n_age b")
                ))
            with plate_a, plate_b:
                pyro.sample("eta_c", dist.Normal(
                    rearrange(theta_loc[:, 26:51], "b (n_eth n_age) -> n_eth n_age b", n_eth=5, n_age=5),
                    rearrange(theta_scale[:, 26:51], "b (n_eth n_age) -> n_eth n_age b", n_eth=5, n_age=5)
                ))
                pyro.sample("eta_d", dist.Normal(
                    rearrange(theta_loc[:, 51:], "b (n_eth n_age) -> n_eth n_age b", n_eth=5, n_age=5),
                    rearrange(theta_scale[:, 51:], "b (n_eth n_age) -> n_eth n_age b", n_eth=5, n_age=5)
                ))

    @classmethod
    def _add_sample(cls, 
                    sample_dict, 
                    name: str, 
                    dist):
        sample_dict[name] = pyro.sample(name, dist)

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 300
            sample_dict["n_age"] = 5
            sample_dict["n_eth"] = 5
            sample_dict["sigma_a1"] = torch.ones(batch_size, device=self.device) * 5
            sample_dict["sigma_a2"] = torch.ones(batch_size, device=self.device) * 1
            sample_dict["sigma_b1"] = torch.ones(batch_size, device=self.device) * 5
            sample_dict["sigma_b2"] = torch.ones(batch_size, device=self.device) * 0.1
            sample_dict["sigma_c"] = torch.ones(batch_size, device=self.device) * 0.1
            sample_dict["sigma_d"] = torch.ones(batch_size, device=self.device) * 0.01
            sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
            assert sample_dict["N"] % (sample_dict["n_eth"] * sample_dict["n_age"]) == 0
            eth_age_pair = torch.stack(torch.meshgrid(torch.arange(sample_dict["n_eth"], device=self.device), 
                                                      torch.arange(sample_dict["n_age"], device=self.device),
                                                      indexing="ij"), dim=-1)
            sample_dict["eth"] = repeat(eth_age_pair[..., 0],
                                        "k1 k2 -> (r k1 k2)", 
                                        r=sample_dict["N"] // (sample_dict["n_eth"] * sample_dict["n_age"]))
            sample_dict["age"] = repeat(eth_age_pair[..., 1],
                                        "k1 k2 -> (r k1 k2)", 
                                        r=sample_dict["N"] // (sample_dict["n_eth"] * sample_dict["n_age"]))
            sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 30 + 200
        else:
            sample_dict = copy.copy(sample_dict)
        
        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_a = pyro.plate("plate_a", sample_dict["n_eth"], dim=-3)
        plate_b = pyro.plate("plate_b", sample_dict["n_age"], dim=-2)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)
           
        with plate_batch:
            for param_name in ["mu_a1", "mu_a2", "mu_b1", "mu_b2", "mu_c", "mu_d"]:
                self._add_sample(sample_dict, 
                                 param_name,
                                 dist.Normal(torch.tensor(0.0, device=self.device),
                                             torch.tensor(1.0, device=self.device)))
            with plate_a:
                self._add_sample(sample_dict,
                                 "eta_a1",
                                 dist.Normal(torch.tensor(0.0, device=self.device),
                                             torch.tensor(1.0, device=self.device)))
                self._add_sample(sample_dict,
                                 "eta_a2",
                                 dist.Normal(torch.tensor(0.0, device=self.device),
                                             torch.tensor(1.0, device=self.device)))
                
            with plate_b:
                self._add_sample(sample_dict,
                                 "eta_b1",
                                 dist.Normal(torch.tensor(0.0, device=self.device),
                                             torch.tensor(1.0, device=self.device)))
                self._add_sample(sample_dict,
                                 "eta_b2",
                                 dist.Normal(torch.tensor(0.0, device=self.device),
                                             torch.tensor(1.0, device=self.device)))
                
            with plate_a, plate_b:
                self._add_sample(sample_dict,
                                 "eta_c",
                                 dist.Normal(torch.tensor(0.0, device=self.device),
                                             torch.tensor(1.0, device=self.device)))
                self._add_sample(sample_dict,
                                 "eta_d",
                                 dist.Normal(torch.tensor(0.0, device=self.device),
                                             torch.tensor(1.0, device=self.device)))
            
            a1 = 5 * sample_dict["mu_a1"] + sample_dict["sigma_a1"] * sample_dict["eta_a1"]
            a2 = sample_dict["mu_a2"] + sample_dict["sigma_a2"] * sample_dict["eta_a2"]
            b1 = 5 * sample_dict["mu_b1"] + sample_dict["sigma_b1"] * sample_dict["eta_b1"]
            b2 = 0.1 * sample_dict["mu_b2"] + sample_dict["sigma_b2"] * sample_dict["eta_b2"]
            c = 0.1 * sample_dict["mu_c"] + sample_dict["sigma_c"] * sample_dict["eta_c"]
            d = 0.01 * sample_dict["mu_d"] + sample_dict["sigma_d"] * sample_dict["eta_d"]

            with plate_n:
                c = c[sample_dict["eth"], sample_dict["age"], :]  # (n, b)
                d = d[sample_dict["eth"], sample_dict["age"], :]  # (n, b)
                
                y_hat = a1.squeeze(-2)[sample_dict["eth"], :] + \
                        a2.squeeze(-2)[sample_dict["eth"], :] * sample_dict["x"] + \
                        b1[sample_dict["age"], :] + \
                        b2[sample_dict["age"], :] * sample_dict["x"] + \
                        c + d * sample_dict["x"]
                sample_dict["y"] = pyro.sample("y",
                                               dist.Normal(y_hat, 
                                                           repeat(sample_dict["sigma_y"], 
                                                                  "b -> n b", 
                                                                  n=sample_dict["N"])),
                                               obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict
    

class ARM_earnings_vary_si(BaseVAE):
    x_dim = 103
    theta_dim = 12
    
    def _extract_x_func(self, sample_dict):
        log_earnings = rearrange(sample_dict["log_earnings"], "n b -> b n")
        height = rearrange(sample_dict["height"], "n b -> b n")
        sigma_a1 = sample_dict["sigma_a1"].view(-1, 1)
        sigma_a2 = sample_dict["sigma_a2"].view(-1, 1)
        sigma_y = sample_dict["sigma_y"].view(-1, 1)
        return torch.cat([log_earnings, height, sigma_a1, sigma_a2, sigma_y], dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        mu_a1 = sample_dict["mu_a1"].view(-1, 1)
        mu_a2 = sample_dict["mu_a2"].view(-1, 1)
        a1 = rearrange(sample_dict["a1"], "n b -> b n")
        a2 = rearrange(sample_dict["a2"], "n b -> b n")
        return torch.cat([mu_a1, mu_a2, a1, a2], dim=-1)
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_a = pyro.plate("plate_a", sample_dict["n_eth"], dim=-2)

        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)
        with plate_batch:    
            pyro.sample("mu_a1", dist.Normal(theta_loc[:, 0], theta_scale[:, 0]))
            pyro.sample("mu_a2", dist.Normal(theta_loc[:, 1], theta_scale[:, 1]))
            with plate_a:
                pyro.sample("a1", dist.Normal(rearrange(theta_loc[:, 2:7], "b k -> k b"), 
                                              rearrange(theta_scale[:, 2:7], "b k -> k b")))
                pyro.sample("a2", dist.Normal(rearrange(theta_loc[:, 7:12], "b k -> k b"), 
                                              rearrange(theta_scale[:, 7:12], "b k -> k b")))

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 50
            sample_dict["n_eth"] = 5
            assert sample_dict["N"] % sample_dict["n_eth"] == 0
            sample_dict["height"] = torch.randint(low=150, high=190, size=(sample_dict["N"], batch_size),
                                                  device=self.device)
            sample_dict["eth"] = repeat(torch.arange(sample_dict["n_eth"], device=self.device),
                                        "k -> (r k)", r=sample_dict["N"] // sample_dict["n_eth"])
            sample_dict["sigma_a1"] = torch.ones(batch_size, device=self.device) * 10
            sample_dict["sigma_a2"] = torch.ones(batch_size, device=self.device) * 0.01
            sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_a = pyro.plate("plate_a", sample_dict["n_eth"], dim=-2)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)

        with plate_batch:
            sample_dict["mu_a1"] = pyro.sample("mu_a1", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                    torch.tensor(1.0, device=self.device)))
            sample_dict["mu_a2"] = pyro.sample("mu_a2", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                    torch.tensor(1.0, device=self.device)))
            with plate_a:
                sample_dict["a1"] = pyro.sample("a1",
                                                dist.Normal(10 * sample_dict["mu_a1"],
                                                            sample_dict["sigma_a1"]))
                sample_dict["a2"] = pyro.sample("a2",
                                                dist.Normal(0.01 * sample_dict["mu_a2"],
                                                            sample_dict["sigma_a2"]))
            with plate_n:
                y_hat = sample_dict["a1"][..., sample_dict["eth"], :] + \
                        sample_dict["a2"][..., sample_dict["eth"], :] * sample_dict["height"]
                sample_dict["log_earnings"] = pyro.sample("log_earnings", 
                                                        dist.Normal(loc=y_hat,
                                                                    scale=repeat(sample_dict["sigma_y"],
                                                                                "b -> n b",
                                                                                n=sample_dict["N"])),
                                                        obs=sample_dict["log_earnings"] if "log_earnings" in sample_dict else None)
        return sample_dict


class ARM_earnings_vary_si_chr(BaseVAE):
    x_dim = 103
    theta_dim = 12
    
    def _extract_x_func(self, sample_dict):
        log_earnings = rearrange(sample_dict["log_earnings"], "n b -> b n")
        height = rearrange(sample_dict["height"], "n b -> b n")
        sigma_a1 = sample_dict["sigma_a1"].view(-1, 1)
        sigma_a2 = sample_dict["sigma_a2"].view(-1, 1)
        sigma_y = sample_dict["sigma_y"].view(-1, 1)
        return torch.cat([log_earnings, height, sigma_a1, sigma_a2, sigma_y], dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        mu_a1 = sample_dict["mu_a1"].view(-1, 1)
        mu_a2 = sample_dict["mu_a2"].view(-1, 1)
        eta1 = rearrange(sample_dict["eta1"], "n b -> b n")
        eta2 = rearrange(sample_dict["eta2"], "n b -> b n")
        return torch.cat([mu_a1, mu_a2, eta1, eta2], dim=-1)
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_a = pyro.plate("plate_a", sample_dict["n_eth"], dim=-2)
        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)
        with plate_batch:    
            pyro.sample("mu_a1", dist.Normal(theta_loc[:, 0], theta_scale[:, 0]))
            pyro.sample("mu_a2", dist.Normal(theta_loc[:, 1], theta_scale[:, 1]))
            with plate_a:
                pyro.sample("eta1", dist.Normal(rearrange(theta_loc[:, 2:7], "b k -> k b"), 
                                                rearrange(theta_scale[:, 2:7], "b k -> k b")))
                pyro.sample("eta2", dist.Normal(rearrange(theta_loc[:, 7:12], "b k -> k b"), 
                                                rearrange(theta_scale[:, 7:12], "b k -> k b")))

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 50
            sample_dict["n_eth"] = 5
            assert sample_dict["N"] % sample_dict["n_eth"] == 0
            sample_dict["height"] = torch.randint(low=150, high=190, size=(sample_dict["N"], batch_size),
                                                  device=self.device)
            sample_dict["eth"] = repeat(torch.arange(sample_dict["n_eth"], device=self.device),
                                        "k -> (r k)", r=sample_dict["N"] // sample_dict["n_eth"])
            sample_dict["sigma_a1"] = torch.ones(batch_size, device=self.device) * 10
            sample_dict["sigma_a2"] = torch.ones(batch_size, device=self.device) * 0.1
            sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_a = pyro.plate("plate_a", sample_dict["n_eth"], dim=-2)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)

        with plate_batch:
            sample_dict["mu_a1"] = pyro.sample("mu_a1", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                    torch.tensor(1.0, device=self.device)))
            sample_dict["mu_a2"] = pyro.sample("mu_a2", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                    torch.tensor(1.0, device=self.device)))
            with plate_a:
                sample_dict["eta1"] = pyro.sample("eta1", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                      torch.tensor(1.0, device=self.device)))
                sample_dict["eta2"] = pyro.sample("eta2", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                      torch.tensor(1.0, device=self.device)))
            a1 = 10 * sample_dict["mu_a1"] + sample_dict["sigma_a1"] * sample_dict["eta1"]
            a2 = 0.1 * sample_dict["mu_a2"] + sample_dict["sigma_a2"] * sample_dict["eta2"]
            with plate_n:
                y_hat = a1[..., sample_dict["eth"], :] + \
                        a2[..., sample_dict["eth"], :] * sample_dict["height"]
                sample_dict["log_earnings"] = pyro.sample("log_earnings", 
                                                        dist.Normal(loc=y_hat,
                                                                    scale=repeat(sample_dict["sigma_y"],
                                                                                "b -> n b",
                                                                                n=sample_dict["N"])),
                                                        obs=sample_dict["log_earnings"] if "log_earnings" in sample_dict else None)
        return sample_dict
    

class ARM_election88_ch14(BaseVAE):
    x_dim = 301
    theta_dim = 13
    
    def _extract_x_func(self, sample_dict):
        black = rearrange(sample_dict["black"], "n b -> b n")
        female = rearrange(sample_dict["female"], "n b -> b n")
        y = rearrange(sample_dict["y"], "n b -> b n")
        sigma_a = sample_dict["sigma_a"].view(-1, 1)
        return torch.cat([black, female, y, sigma_a], dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        mu_a = sample_dict["mu_a"].view(-1, 1)
        a = rearrange(sample_dict["a"], "k b -> b k")
        b = rearrange(sample_dict["b"], "k b -> b k")
        return torch.cat([mu_a, a, b], dim=-1)
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_a = pyro.plate("plate_a", sample_dict["n_state"], dim=-2)
        plate_b = pyro.plate("plate_b", 2, dim=-2)

        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)
        with plate_batch:    
            pyro.sample("mu_a", dist.Normal(theta_loc[:, 0], theta_scale[:, 0]))
            with plate_a:
                pyro.sample("a", dist.Normal(rearrange(theta_loc[:, 1:11], "b k -> k b"),
                                             rearrange(theta_scale[:, 1:11], "b k -> k b")))
            with plate_b:
                pyro.sample("b", dist.Normal(rearrange(theta_loc[:, 11:13], "b k -> k b"),
                                             rearrange(theta_scale[:, 11:13], "b k -> k b")))

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 100
            sample_dict["n_state"] = 10
            assert sample_dict["N"] % sample_dict["n_state"] == 0
            sample_dict["state"] = repeat(torch.arange(sample_dict["n_state"], device=self.device),
                                          "k -> (r k)", r=sample_dict["N"] // sample_dict["n_state"])
            sample_dict["female"] = torch.randint(low=0, high=2, size=(sample_dict["N"], batch_size),
                                                  device=self.device)
            sample_dict["black"] = torch.randint(low=0, high=2, size=(sample_dict["N"], batch_size),
                                                 device=self.device)
            sample_dict["sigma_a"] = torch.ones(batch_size, device=self.device) * 1
        else:
            sample_dict = copy.copy(sample_dict)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_a = pyro.plate("plate_a", sample_dict["n_state"], dim=-2)
        plate_b = pyro.plate("plate_b", 2, dim=-2)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)

        with plate_batch:
            sample_dict["mu_a"] = pyro.sample("mu_a", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                  torch.tensor(1.0, device=self.device)))
            with plate_a:
                sample_dict["a"] = pyro.sample("a", dist.Normal(sample_dict["mu_a"],
                                                                sample_dict["sigma_a"]))
            with plate_b:
                sample_dict["b"] = pyro.sample("b", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                torch.tensor(100.0, device=self.device)))
            with plate_n:
                y_hat = sample_dict["b"][..., 0, :].unsqueeze(-2) * sample_dict["black"] + \
                        sample_dict["b"][..., 1, :].unsqueeze(-2) * sample_dict["female"] + \
                        sample_dict["a"][..., sample_dict["state"], :]
                sample_dict["y"] = pyro.sample("y", 
                                               dist.Bernoulli(logits=y_hat),
                                              obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_election88_ch19(BaseVAE):
    x_dim = 244
    theta_dim = 18
    
    def _extract_x_func(self, sample_dict):
        black = rearrange(sample_dict["black"], "n b -> b n")
        female = rearrange(sample_dict["female"], "n b -> b n")
        y = rearrange(sample_dict["y"], "n b -> b n")
        sigma_age = sample_dict["sigma_age"].view(-1, 1)
        sigma_edu = sample_dict["sigma_edu"].view(-1, 1)
        sigma_age_edu = sample_dict["sigma_age_edu"].view(-1, 1)
        sigma_state = sample_dict["sigma_state"].view(-1, 1)
        return torch.cat([black, female, y, sigma_age, sigma_edu, sigma_age_edu, sigma_state], dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        mu_age = sample_dict["mu_age"].view(-1, 1)
        mu_edu = sample_dict["mu_edu"].view(-1, 1)
        mu_age_edu = sample_dict["mu_age_edu"].view(-1, 1)
        mu_state = sample_dict["mu_state"].view(-1, 1)
        beta = rearrange(sample_dict["beta"], "... b -> b (...)")
        b_age = rearrange(sample_dict["b_age"], "... b -> b (...)")
        b_edu = rearrange(sample_dict["b_edu"], "... b -> b (...)")
        b_age_edu = rearrange(sample_dict["b_age_edu"], "... b -> b (...)")
        b_state = rearrange(sample_dict["b_state"], "... b -> b (...)")
        return torch.cat([mu_age, mu_edu, mu_age_edu, mu_state,
                          beta, b_age, b_edu, b_age_edu, b_state], dim=-1)
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_beta = pyro.plate("plate_beta", 4, dim=-2)
        plate_age = pyro.plate("plate_age", sample_dict["n_age"], dim=-3)
        plate_edu = pyro.plate("plate_edu", sample_dict["n_edu"], dim=-2)
        plate_state = pyro.plate("plate_state", sample_dict["n_state"], dim=-2)

        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)
        with plate_batch:    
            pyro.sample("mu_age", dist.Normal(theta_loc[:, 0], theta_scale[:, 0]))
            pyro.sample("mu_edu", dist.Normal(theta_loc[:, 1], theta_scale[:, 1]))
            pyro.sample("mu_age_edu", dist.Normal(theta_loc[:, 2], theta_scale[:, 2]))
            pyro.sample("mu_state", dist.Normal(theta_loc[:, 3], theta_scale[:, 3]))
            with plate_beta:
                pyro.sample("beta", dist.Normal(rearrange(theta_loc[:, 4:8], "b k -> k b"),
                                                rearrange(theta_scale[:, 4:8], "b k -> k b")))
            with plate_age:
                pyro.sample("b_age", dist.Normal(rearrange(theta_loc[:, 8:10], "b k -> k 1 b"),
                                                 rearrange(theta_scale[:, 8:10], "b k -> k 1 b")))
            with plate_edu:
                pyro.sample("b_edu", dist.Normal(rearrange(theta_loc[:, 10:12], "b k -> k b"),
                                                 rearrange(theta_scale[:, 10:12], "b k -> k b")))
            with plate_age, plate_edu:
                pyro.sample("b_age_edu",
                            dist.Normal(rearrange(theta_loc[:, 12:16], 
                                                  "b (k1 k2) -> k1 k2 b", 
                                                  k1=sample_dict["n_age"], 
                                                  k2=sample_dict["n_edu"]),
                                        rearrange(theta_scale[:, 12:16],
                                                  "b (k1 k2) -> k1 k2 b", 
                                                  k1=sample_dict["n_age"], 
                                                  k2=sample_dict["n_edu"])
                                        )
                            )
            with plate_state:
                pyro.sample("b_state", dist.Normal(rearrange(theta_loc[:, 16:18], "b k -> k b"),
                                                   rearrange(theta_scale[:, 16:18], "b k -> k b")))

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 80
            sample_dict["n_state"] = 2
            sample_dict["n_age"] = 2
            sample_dict["n_edu"] = 2
            total_permutate = sample_dict["n_state"] * sample_dict["n_age"] * sample_dict["n_edu"]
            assert sample_dict["N"] % total_permutate == 0
            state_age_edu_region_mesh = torch.stack(torch.meshgrid(
                torch.arange(sample_dict["n_state"], device=self.device),
                torch.arange(sample_dict["n_age"], device=self.device),
                torch.arange(sample_dict["n_edu"], device=self.device),
            indexing="ij"), dim=-1)  # (2, 2, 2, 2, 3)
            sample_dict["state"] = repeat(state_age_edu_region_mesh[..., 0].reshape(-1),
                                          "k -> (r k)", r=sample_dict["N"] // total_permutate)
            sample_dict["age"] = repeat(state_age_edu_region_mesh[..., 1].reshape(-1),
                                        "k -> (r k)", r=sample_dict["N"] // total_permutate)
            sample_dict["edu"] = repeat(state_age_edu_region_mesh[..., 2].reshape(-1),
                                        "k -> (r k)", r=sample_dict["N"] // total_permutate)
            sample_dict["female"] = torch.randint(low=0, high=2, size=(sample_dict["N"], batch_size),
                                                  device=self.device)
            sample_dict["black"] = torch.randint(low=0, high=2, size=(sample_dict["N"], batch_size),
                                                 device=self.device)
            sample_dict["sigma_age"] = torch.ones(batch_size, device=self.device) * 10
            sample_dict["sigma_edu"] = torch.ones(batch_size, device=self.device) * 10
            sample_dict["sigma_age_edu"] = torch.ones(batch_size, device=self.device) * 10
            sample_dict["sigma_state"] = torch.ones(batch_size, device=self.device) * 10
        else:
            sample_dict = copy.copy(sample_dict)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_beta = pyro.plate("plate_beta", 4, dim=-2)
        plate_age = pyro.plate("plate_age", sample_dict["n_age"], dim=-3)
        plate_edu = pyro.plate("plate_edu", sample_dict["n_edu"], dim=-2)
        plate_state = pyro.plate("plate_state", sample_dict["n_state"], dim=-2)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)

        with plate_batch:
            sample_dict["mu_age"] = pyro.sample("mu_age", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                      torch.tensor(1.0, device=self.device)))
            sample_dict["mu_edu"] = pyro.sample("mu_edu", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                      torch.tensor(1.0, device=self.device)))
            sample_dict["mu_age_edu"] = pyro.sample("mu_age_edu", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                              torch.tensor(1.0, device=self.device)))
            sample_dict["mu_state"] = pyro.sample("mu_state", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                          torch.tensor(1.0, device=self.device)))
            with plate_beta:
                sample_dict["beta"] = pyro.sample("beta", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                      torch.tensor(100.0, device=self.device)))  #(4, b)
            with plate_age:
                sample_dict["b_age"] = pyro.sample("b_age", dist.Normal(100 * sample_dict["mu_age"],
                                                                        sample_dict["sigma_age"]))  # (n_age, 1, b)
            with plate_edu:
                sample_dict["b_edu"] = pyro.sample("b_edu", dist.Normal(100 * sample_dict["mu_edu"],
                                                                        sample_dict["sigma_edu"]))  # (n_edu, b)
            with plate_age, plate_edu:
                sample_dict["b_age_edu"] = pyro.sample("b_age_edu", dist.Normal(100 * sample_dict["mu_age_edu"],
                                                                                sample_dict["sigma_age_edu"]))  # (n_age, n_edu, b)
            with plate_state:
                sample_dict["b_state"] = pyro.sample("b_state", dist.Normal(10 * sample_dict["mu_state"],
                                                                            sample_dict["sigma_state"]))  # (n_state, b)
            with plate_n:
                Xbeta = sample_dict["beta"][..., 0, :].unsqueeze(-2) + \
                        sample_dict["beta"][..., 1, :].unsqueeze(-2) * sample_dict["female"] + \
                        sample_dict["beta"][..., 2, :].unsqueeze(-2) * sample_dict["black"] + \
                        sample_dict["beta"][..., 3, :].unsqueeze(-2) * sample_dict["female"] * sample_dict["black"] + \
                        sample_dict["b_age"][..., sample_dict["age"], :, :].squeeze(-2) + \
                        sample_dict["b_edu"][..., sample_dict["edu"], :] + \
                        sample_dict["b_age_edu"][..., sample_dict["age"], sample_dict["edu"], :] + \
                        sample_dict["b_state"][..., sample_dict["state"], :]
                sample_dict["y"] = pyro.sample("y",
                                               dist.Bernoulli(logits=Xbeta),
                                               obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_electric(BaseVAE):
    x_dim = 102
    theta_dim = 7
    
    def _extract_x_func(self, sample_dict):
        treatment = rearrange(sample_dict["treatment"], "n b -> b n")
        y = rearrange(sample_dict["y"], "n b -> b n")
        sigma_a = sample_dict["sigma_a"].view(-1, 1)
        sigma_y = sample_dict["sigma_y"].view(-1, 1)
        return torch.cat([treatment, y, sigma_a, sigma_y], dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        mu_a = sample_dict["mu_a"].view(-1, 1)
        a = rearrange(sample_dict["a"], "k b -> b k")
        beta = sample_dict["beta"].view(-1, 1)
        return torch.cat([mu_a, a, beta], dim=-1)
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_pair = pyro.plate("plate_pair", sample_dict["n_pair"], dim=-2)

        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)
        with plate_batch:    
            pyro.sample("mu_a", dist.Normal(theta_loc[:, 0], theta_scale[:, 0]))
            with plate_pair:
                pyro.sample("a", dist.Normal(rearrange(theta_loc[:, 1:6], "b k -> k b"),
                                             rearrange(theta_scale[:, 1:6], "b k -> k b")))
            pyro.sample("beta", dist.Normal(theta_loc[:, 6], theta_scale[:, 6]))

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 50
            sample_dict["n_pair"] = 5
            assert sample_dict["N"] % sample_dict["n_pair"] == 0
            sample_dict["pair"] = repeat(torch.arange(sample_dict["n_pair"], device=self.device),
                                        "k -> (r k)", r=sample_dict["N"] // sample_dict["n_pair"])
            sample_dict["treatment"] = torch.randint(low=0, high=10,
                                                     size=(sample_dict["N"], batch_size),
                                                     device=self.device)
            sample_dict["sigma_a"] = torch.ones(batch_size, device=self.device) * 10
            sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_pair = pyro.plate("plate_pair", sample_dict["n_pair"], dim=-2)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)

        with plate_batch:
            sample_dict["mu_a"] = pyro.sample("mu_a", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                  torch.tensor(1.0, device=self.device)))
            with plate_pair:
                sample_dict["a"] = pyro.sample("a", dist.Normal(100 * sample_dict["mu_a"],
                                                                sample_dict["sigma_a"]))
            sample_dict["beta"] = pyro.sample("beta", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                  torch.tensor(1.0, device=self.device)))
            with plate_n:
                y_hat = sample_dict["a"][..., sample_dict["pair"], :] + sample_dict["beta"] * sample_dict["treatment"]
                sample_dict["y"] = pyro.sample("y",
                                               dist.Normal(y_hat, 
                                                           repeat(sample_dict["sigma_y"],
                                                                  "b -> n b", n=sample_dict["N"])),
                                               obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_electric_1a(BaseVAE):
    x_dim = 158
    theta_dim = 13
    
    def _extract_x_func(self, sample_dict):
        treatment = rearrange(sample_dict["treatment"], "n b -> b n")
        y = rearrange(sample_dict["y"], "n b -> b n")
        sigma_a = rearrange(sample_dict["sigma_a"], "k b -> b k")
        sigma_y = rearrange(sample_dict["sigma_y"], "k b -> b k")
        return torch.cat([treatment, y, sigma_a, sigma_y], dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        mu_a = rearrange(sample_dict["mu_a"], "k b -> b k")
        a = rearrange(sample_dict["a"], "k b -> b k")
        b = rearrange(sample_dict["b"], "k b -> b k")
        return torch.cat([mu_a, a, b], dim=-1)
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_pair = pyro.plate("plate_pair", sample_dict["n_pair"], dim=-2)
        plate_grade = pyro.plate("plate_grade", sample_dict["n_grade"], dim=-2)
        plate_grade_pair = pyro.plate("plate_grade_pair", sample_dict["n_pair"], dim=-2)

        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)
        with plate_batch:
            with plate_grade_pair:    
                pyro.sample("mu_a", dist.Normal(rearrange(theta_loc[:, 0:5], "b k -> k b"), 
                                                rearrange(theta_scale[:, 0:5], "b k -> k b")))
            with plate_pair:
                pyro.sample("a", dist.Normal(rearrange(theta_loc[:, 5:10], "b k -> k b"),
                                             rearrange(theta_scale[:, 5:10], "b k -> k b")))
            with plate_grade:
                pyro.sample("b", dist.Normal(rearrange(theta_loc[:, 10:13], "b k -> k b"),
                                             rearrange(theta_scale[:, 10:13], "b k -> k b")))

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 75
            sample_dict["n_pair"] = 5
            sample_dict["n_grade"] = 3
            assert sample_dict["N"] % (sample_dict["n_pair"] * sample_dict["n_grade"]) == 0
            grade_pair_mesh = torch.stack(torch.meshgrid(
                torch.arange(sample_dict["n_grade"], device=self.device),
                torch.arange(sample_dict["n_pair"], device=self.device),
                indexing="ij"
            ), dim=-1)  # (3, 5, 2)
            sample_dict["grade"] = repeat(grade_pair_mesh[..., 0],
                                          "k1 k2 -> (r k1 k2)",
                                          r=sample_dict["N"] // (sample_dict["n_pair"] * sample_dict["n_grade"]))
            sample_dict["pair"] = repeat(grade_pair_mesh[..., 1],
                                          "k1 k2 -> (r k1 k2)",
                                          r=sample_dict["N"] // (sample_dict["n_pair"] * sample_dict["n_grade"]))
            sample_dict["grade_pair"] = torch.randperm(sample_dict["n_pair"],
                                                        generator=torch.Generator(device=self.device).manual_seed(1234),
                                                        device=self.device)
            sample_dict["treatment"] = torch.randint(low=0, high=10,
                                                     size=(sample_dict["N"], batch_size),
                                                     device=self.device)
            sample_dict["sigma_a"] = repeat(torch.arange(sample_dict["n_pair"], device=self.device) + 1,
                                            "n_pair -> n_pair b",
                                            b=batch_size)
            sample_dict["sigma_y"] = repeat((torch.arange(sample_dict["n_grade"], device=self.device) + 1) * 0.1,
                                            "n_grade -> n_grade b",
                                             b=batch_size)
        else:
            sample_dict = copy.copy(sample_dict)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_pair = pyro.plate("plate_pair", sample_dict["n_pair"], dim=-2)
        plate_grade = pyro.plate("plate_grade", sample_dict["n_grade"], dim=-2)
        plate_grade_pair = pyro.plate("plate_grade_pair", sample_dict["n_pair"], dim=-2)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)

        with plate_batch:
            with plate_grade_pair:
                sample_dict["mu_a"] = pyro.sample("mu_a", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                      torch.tensor(1.0, device=self.device)))
            sigma_a_hat = sample_dict["sigma_a"][..., sample_dict["grade_pair"], :]
            mu_a_hat = 100 * sample_dict["mu_a"][..., sample_dict["grade_pair"], :]
            with plate_pair:
                sample_dict["a"] = pyro.sample("a", dist.Normal(mu_a_hat, sigma_a_hat))
            with plate_grade:
                sample_dict["b"] = pyro.sample("b", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                torch.tensor(100.0, device=self.device)))
            sigma_y_hat = sample_dict["sigma_y"][..., sample_dict["grade"], :]
            with plate_n:
                y_hat = sample_dict["a"][..., sample_dict["pair"], :] + \
                        sample_dict["b"][..., sample_dict["grade"], :] * sample_dict["treatment"]
                sample_dict["y"] = pyro.sample("y", 
                                               dist.Normal(y_hat, sigma_y_hat),
                                               obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_electric_1a_chr(BaseVAE):
    x_dim = 158
    theta_dim = 13
    
    def _extract_x_func(self, sample_dict):
        treatment = rearrange(sample_dict["treatment"], "n b -> b n")
        y = rearrange(sample_dict["y"], "n b -> b n")
        sigma_a = rearrange(sample_dict["sigma_a"], "k b -> b k")
        sigma_y = rearrange(sample_dict["sigma_y"], "k b -> b k")
        return torch.cat([treatment, y, sigma_a, sigma_y], dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        mu_a = rearrange(sample_dict["mu_a"], "k b -> b k")
        eta_a = rearrange(sample_dict["eta_a"], "k b -> b k")
        b = rearrange(sample_dict["b"], "k b -> b k")
        return torch.cat([mu_a, eta_a, b], dim=-1)
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_pair = pyro.plate("plate_pair", sample_dict["n_pair"], dim=-2)
        plate_grade = pyro.plate("plate_grade", sample_dict["n_grade"], dim=-2)
        plate_grade_pair = pyro.plate("plate_grade_pair", sample_dict["n_pair"], dim=-2)

        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)
        with plate_batch:
            with plate_grade_pair:    
                pyro.sample("mu_a", dist.Normal(rearrange(theta_loc[:, 0:5], "b k -> k b"), 
                                                rearrange(theta_scale[:, 0:5], "b k -> k b")))
            with plate_pair:
                pyro.sample("eta_a", dist.Normal(rearrange(theta_loc[:, 5:10], "b k -> k b"),
                                                 rearrange(theta_scale[:, 5:10], "b k -> k b")))
            with plate_grade:
                pyro.sample("b", dist.Normal(rearrange(theta_loc[:, 10:13], "b k -> k b"),
                                             rearrange(theta_scale[:, 10:13], "b k -> k b")))

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 75
            sample_dict["n_pair"] = 5
            sample_dict["n_grade"] = 3
            assert sample_dict["N"] % (sample_dict["n_pair"] * sample_dict["n_grade"]) == 0
            grade_pair_mesh = torch.stack(torch.meshgrid(
                torch.arange(sample_dict["n_grade"], device=self.device),
                torch.arange(sample_dict["n_pair"], device=self.device),
                indexing="ij"
            ), dim=-1)  # (3, 5, 2)
            sample_dict["grade"] = repeat(grade_pair_mesh[..., 0],
                                          "k1 k2 -> (r k1 k2)",
                                          r=sample_dict["N"] // (sample_dict["n_pair"] * sample_dict["n_grade"]))
            sample_dict["pair"] = repeat(grade_pair_mesh[..., 1],
                                          "k1 k2 -> (r k1 k2)",
                                          r=sample_dict["N"] // (sample_dict["n_pair"] * sample_dict["n_grade"]))
            sample_dict["grade_pair"] = torch.randperm(sample_dict["n_pair"],
                                                        generator=torch.Generator(device=self.device).manual_seed(1234),
                                                        device=self.device)
            sample_dict["treatment"] = torch.randint(low=0, high=10,
                                                     size=(sample_dict["N"], batch_size),
                                                     device=self.device)
            sample_dict["sigma_a"] = repeat(torch.arange(sample_dict["n_pair"], device=self.device) + 1,
                                            "n_pair -> n_pair b",
                                            b=batch_size)
            sample_dict["sigma_y"] = repeat((torch.arange(sample_dict["n_grade"], device=self.device) + 1) * 0.1,
                                            "n_grade -> n_grade b",
                                             b=batch_size)
        else:
            sample_dict = copy.copy(sample_dict)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_pair = pyro.plate("plate_pair", sample_dict["n_pair"], dim=-2)
        plate_grade = pyro.plate("plate_grade", sample_dict["n_grade"], dim=-2)
        plate_grade_pair = pyro.plate("plate_grade_pair", sample_dict["n_pair"], dim=-2)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)

        with plate_batch:
            with plate_grade_pair:
                sample_dict["mu_a"] = pyro.sample("mu_a", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                      torch.tensor(1.0, device=self.device)))
            sigma_a_hat = sample_dict["sigma_a"][..., sample_dict["grade_pair"], :]
            mu_a_hat = 100 * sample_dict["mu_a"][..., sample_dict["grade_pair"], :]
            with plate_pair:
                sample_dict["eta_a"] = pyro.sample("eta_a", dist.Normal(torch.tensor(0.0, device=self.device), 
                                                                        torch.tensor(1.0, device=self.device)))
            a = mu_a_hat + sigma_a_hat * sample_dict["eta_a"]
            with plate_grade:
                sample_dict["b"] = pyro.sample("b", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                torch.tensor(100.0, device=self.device)))
            sigma_y_hat = sample_dict["sigma_y"][..., sample_dict["grade"], :]
            with plate_n:
                y_hat = a[..., sample_dict["pair"], :] + \
                        sample_dict["b"][..., sample_dict["grade"], :] * sample_dict["treatment"]
                sample_dict["y"] = pyro.sample("y", 
                                               dist.Normal(y_hat, sigma_y_hat),
                                               obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_electric_1b(BaseVAE):
    x_dim = 152
    theta_dim = 8
    
    def _extract_x_func(self, sample_dict):
        treatment = rearrange(sample_dict["treatment"], "n b -> b n")
        pre_test = rearrange(sample_dict["pre_test"], "n b -> b n")
        y = rearrange(sample_dict["y"], "n b -> b n")
        sigma_a = rearrange(sample_dict["sigma_a"], "b -> b 1")
        sigma_y = rearrange(sample_dict["sigma_y"], "b -> b 1")
        return torch.cat([treatment, pre_test, y, sigma_a, sigma_y], dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        mu_a = rearrange(sample_dict["mu_a"], "b -> b 1")
        a = rearrange(sample_dict["a"], "k b -> b k")
        beta = rearrange(sample_dict["beta"], "k b -> b k")
        return torch.cat([mu_a, a, beta], dim=-1)
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_pair = pyro.plate("plate_pair", sample_dict["n_pair"], dim=-2)
        plate_beta = pyro.plate("plate_beta", 2, dim=-2)
        
        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)
        with plate_batch:
            pyro.sample("mu_a", dist.Normal(rearrange(theta_loc[:, 0:1], "b k -> k b"), 
                                            rearrange(theta_scale[:, 0:1], "b k -> k b")))
            with plate_pair:
                pyro.sample("a", dist.Normal(rearrange(theta_loc[:, 1:6], "b k -> k b"),
                                             rearrange(theta_scale[:, 1:6], "b k -> k b")))
            with plate_beta:
                pyro.sample("beta", dist.Normal(rearrange(theta_loc[:, 6:8], "b k -> k b"),
                                             rearrange(theta_scale[:, 6:8], "b k -> k b")))

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 50
            sample_dict["n_pair"] = 5
            assert sample_dict["N"] % sample_dict["n_pair"] == 0
            sample_dict["pair"] = repeat(torch.arange(sample_dict["n_pair"]),
                                          "k -> (r k)",
                                          r=sample_dict["N"] // sample_dict["n_pair"])
            sample_dict["treatment"] = torch.randint(low=0, high=10,
                                                     size=(sample_dict["N"], batch_size),
                                                     device=self.device)
            sample_dict["pre_test"] = torch.randint(low=20, high=40,
                                                     size=(sample_dict["N"], batch_size),
                                                     device=self.device)
            sample_dict["sigma_a"] = torch.ones(batch_size, device=self.device) * 10
            sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_pair = pyro.plate("plate_pair", sample_dict["n_pair"], dim=-2)
        plate_beta = pyro.plate("plate_beta", 2, dim=-2)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)

        with plate_batch:
            sample_dict["mu_a"] = pyro.sample("mu_a", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                  torch.tensor(1.0, device=self.device)))
            with plate_pair:
                sample_dict["a"] = pyro.sample("a", dist.Normal(100 * sample_dict["mu_a"], sample_dict["sigma_a"]))
            with plate_beta:
                sample_dict["beta"] = pyro.sample("beta", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                      torch.tensor(100.0, device=self.device)))
            with plate_n:
                y_hat = sample_dict["a"][..., sample_dict["pair"], :] + \
                        sample_dict["beta"][..., 0, :].unsqueeze(-2) * sample_dict["treatment"] + \
                        sample_dict["beta"][..., 1, :].unsqueeze(-2) * sample_dict["pre_test"]
                sample_dict["y"] = pyro.sample("y",
                                               dist.Normal(y_hat, 
                                                           repeat(sample_dict["sigma_y"],
                                                                  "b -> n b", n=sample_dict["N"])),
                                                obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_electric_1b_chr(BaseVAE):
    x_dim = 152
    theta_dim = 8
    
    def _extract_x_func(self, sample_dict):
        treatment = rearrange(sample_dict["treatment"], "n b -> b n")
        pre_test = rearrange(sample_dict["pre_test"], "n b -> b n")
        y = rearrange(sample_dict["y"], "n b -> b n")
        sigma_a = rearrange(sample_dict["sigma_a"], "b -> b 1")
        sigma_y = rearrange(sample_dict["sigma_y"], "b -> b 1")
        return torch.cat([treatment, pre_test, y, sigma_a, sigma_y], dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        mu_a = rearrange(sample_dict["mu_a"], "b -> b 1")
        eta = rearrange(sample_dict["eta"], "k b -> b k")
        beta = rearrange(sample_dict["beta"], "k b -> b k")
        return torch.cat([mu_a, eta, beta], dim=-1)
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_pair = pyro.plate("plate_pair", sample_dict["n_pair"], dim=-2)
        plate_beta = pyro.plate("plate_beta", 2, dim=-2)
        
        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)
        with plate_batch:
            pyro.sample("mu_a", dist.Normal(rearrange(theta_loc[:, 0:1], "b k -> k b"), 
                                            rearrange(theta_scale[:, 0:1], "b k -> k b")))
            with plate_pair:
                pyro.sample("eta", dist.Normal(rearrange(theta_loc[:, 1:6], "b k -> k b"),
                                               rearrange(theta_scale[:, 1:6], "b k -> k b")))
            with plate_beta:
                pyro.sample("beta", dist.Normal(rearrange(theta_loc[:, 6:8], "b k -> k b"),
                                             rearrange(theta_scale[:, 6:8], "b k -> k b")))

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 50
            sample_dict["n_pair"] = 5
            assert sample_dict["N"] % sample_dict["n_pair"] == 0
            sample_dict["pair"] = repeat(torch.arange(sample_dict["n_pair"]),
                                          "k -> (r k)",
                                          r=sample_dict["N"] // sample_dict["n_pair"])
            sample_dict["treatment"] = torch.randint(low=0, high=10,
                                                     size=(sample_dict["N"], batch_size),
                                                     device=self.device)
            sample_dict["pre_test"] = torch.randint(low=20, high=40,
                                                     size=(sample_dict["N"], batch_size),
                                                     device=self.device)
            sample_dict["sigma_a"] = torch.ones(batch_size, device=self.device) * 10
            sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_pair = pyro.plate("plate_pair", sample_dict["n_pair"], dim=-2)
        plate_beta = pyro.plate("plate_beta", 2, dim=-2)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)

        with plate_batch:
            sample_dict["mu_a"] = pyro.sample("mu_a", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                  torch.tensor(1.0, device=self.device)))
            with plate_pair:
                sample_dict["eta"] = pyro.sample("eta", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                    torch.tensor(1.0, device=self.device)))
            a = 100 * sample_dict["mu_a"] + sample_dict["sigma_a"] * sample_dict["eta"]
            with plate_beta:
                sample_dict["beta"] = pyro.sample("beta", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                      torch.tensor(100.0, device=self.device)))
            with plate_n:
                y_hat = a[..., sample_dict["pair"], :] + \
                        sample_dict["beta"][..., 0, :].unsqueeze(-2) * sample_dict["treatment"] + \
                        sample_dict["beta"][..., 1, :].unsqueeze(-2) * sample_dict["pre_test"]
                sample_dict["y"] = pyro.sample("y",
                                               dist.Normal(y_hat, 
                                                           repeat(sample_dict["sigma_y"],
                                                                  "b -> n b", n=sample_dict["N"])),
                                                obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_electric_1c(BaseVAE):
    x_dim = 233
    theta_dim = 16
    
    def _extract_x_func(self, sample_dict):
        treatment = rearrange(sample_dict["treatment"], "n b -> b n")
        pre_test = rearrange(sample_dict["pre_test"], "n b -> b n")
        y = rearrange(sample_dict["y"], "n b -> b n")
        sigma_a = rearrange(sample_dict["sigma_a"], "k b -> b k")
        sigma_y = rearrange(sample_dict["sigma_y"], "k b -> b k")
        return torch.cat([treatment, pre_test, y, sigma_a, sigma_y], dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        mu_a = rearrange(sample_dict["mu_a"], "k b -> b k")
        a = rearrange(sample_dict["a"], "k b -> b k")
        b = rearrange(sample_dict["b"], "k b -> b k")
        c = rearrange(sample_dict["c"], "k b -> b k")
        return torch.cat([mu_a, a, b, c], dim=-1)
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_pair = pyro.plate("plate_pair", sample_dict["n_pair"], dim=-2)
        plate_grade = pyro.plate("plate_grade", sample_dict["n_grade"], dim=-2)
        plate_grade_pair = pyro.plate("plate_grade_pair", sample_dict["n_pair"], dim=-2)

        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)
        with plate_batch:
            with plate_grade_pair:    
                pyro.sample("mu_a", dist.Normal(rearrange(theta_loc[:, 0:5], "b k -> k b"), 
                                                rearrange(theta_scale[:, 0:5], "b k -> k b")))
            with plate_pair:
                pyro.sample("a", dist.Normal(rearrange(theta_loc[:, 5:10], "b k -> k b"),
                                             rearrange(theta_scale[:, 5:10], "b k -> k b")))
            with plate_grade:
                pyro.sample("b", dist.Normal(rearrange(theta_loc[:, 10:13], "b k -> k b"),
                                             rearrange(theta_scale[:, 10:13], "b k -> k b")))
                pyro.sample("c", dist.Normal(rearrange(theta_loc[:, 13:16], "b k -> k b"),
                                             rearrange(theta_scale[:, 13:16], "b k -> k b")))

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 75
            sample_dict["n_pair"] = 5
            sample_dict["n_grade"] = 3
            assert sample_dict["N"] % (sample_dict["n_pair"] * sample_dict["n_grade"]) == 0
            grade_pair_mesh = torch.stack(torch.meshgrid(
                torch.arange(sample_dict["n_grade"], device=self.device),
                torch.arange(sample_dict["n_pair"], device=self.device),
                indexing="ij"
            ), dim=-1)  # (3, 5, 2)
            sample_dict["grade"] = repeat(grade_pair_mesh[..., 0],
                                          "k1 k2 -> (r k1 k2)",
                                          r=sample_dict["N"] // (sample_dict["n_pair"] * sample_dict["n_grade"]))
            sample_dict["pair"] = repeat(grade_pair_mesh[..., 1],
                                          "k1 k2 -> (r k1 k2)",
                                          r=sample_dict["N"] // (sample_dict["n_pair"] * sample_dict["n_grade"]))
            sample_dict["grade_pair"] = torch.randperm(sample_dict["n_pair"],
                                                        generator=torch.Generator(device=self.device).manual_seed(1234),
                                                        device=self.device)
            sample_dict["treatment"] = torch.randint(low=0, high=10,
                                                     size=(sample_dict["N"], batch_size),
                                                     device=self.device)
            sample_dict["pre_test"] = torch.randint(low=20, high=40,
                                                     size=(sample_dict["N"], batch_size),
                                                     device=self.device)
            sample_dict["sigma_a"] = repeat(torch.arange(sample_dict["n_pair"], device=self.device) + 1,
                                            "n_pair -> n_pair b",
                                            b=batch_size)
            sample_dict["sigma_y"] = repeat((torch.arange(sample_dict["n_grade"], device=self.device) + 1) * 0.1,
                                            "n_grade -> n_grade b",
                                             b=batch_size)
        else:
            sample_dict = copy.copy(sample_dict)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_pair = pyro.plate("plate_pair", sample_dict["n_pair"], dim=-2)
        plate_grade = pyro.plate("plate_grade", sample_dict["n_grade"], dim=-2)
        plate_grade_pair = pyro.plate("plate_grade_pair", sample_dict["n_pair"], dim=-2)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)

        with plate_batch:
            with plate_grade_pair:
                sample_dict["mu_a"] = pyro.sample("mu_a", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                      torch.tensor(1.0, device=self.device)))
            sigma_a_hat = sample_dict["sigma_a"][..., sample_dict["grade_pair"], :]
            mu_a_hat = 40 * sample_dict["mu_a"][..., sample_dict["grade_pair"], :]
            with plate_pair:
                sample_dict["a"] = pyro.sample("a", dist.Normal(mu_a_hat, sigma_a_hat))
            with plate_grade:
                sample_dict["b"] = pyro.sample("b", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                torch.tensor(100.0, device=self.device)))
                sample_dict["c"] = pyro.sample("c", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                torch.tensor(100.0, device=self.device)))
            sigma_y_hat = sample_dict["sigma_y"][..., sample_dict["grade"], :]
            with plate_n:
                y_hat = sample_dict["a"][..., sample_dict["pair"], :] + \
                        sample_dict["b"][..., sample_dict["grade"], :] * sample_dict["treatment"] + \
                        sample_dict["c"][..., sample_dict["grade"], :] * sample_dict["pre_test"]
                sample_dict["y"] = pyro.sample("y", 
                                               dist.Normal(y_hat, sigma_y_hat),
                                               obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_electric_1c_chr(BaseVAE):
    x_dim = 233
    theta_dim = 16
    
    def _extract_x_func(self, sample_dict):
        treatment = rearrange(sample_dict["treatment"], "n b -> b n")
        pre_test = rearrange(sample_dict["pre_test"], "n b -> b n")
        y = rearrange(sample_dict["y"], "n b -> b n")
        sigma_a = rearrange(sample_dict["sigma_a"], "k b -> b k")
        sigma_y = rearrange(sample_dict["sigma_y"], "k b -> b k")
        return torch.cat([treatment, pre_test, y, sigma_a, sigma_y], dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        mu_a = rearrange(sample_dict["mu_a"], "k b -> b k")
        eta_a = rearrange(sample_dict["eta_a"], "k b -> b k")
        b = rearrange(sample_dict["b"], "k b -> b k")
        c = rearrange(sample_dict["c"], "k b -> b k")
        return torch.cat([mu_a, eta_a, b, c], dim=-1)
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_pair = pyro.plate("plate_pair", sample_dict["n_pair"], dim=-2)
        plate_grade = pyro.plate("plate_grade", sample_dict["n_grade"], dim=-2)
        plate_grade_pair = pyro.plate("plate_grade_pair", sample_dict["n_pair"], dim=-2)

        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)
        with plate_batch:
            with plate_grade_pair:    
                pyro.sample("mu_a", dist.Normal(rearrange(theta_loc[:, 0:5], "b k -> k b"), 
                                                rearrange(theta_scale[:, 0:5], "b k -> k b")))
            with plate_pair:
                pyro.sample("eta_a", dist.Normal(rearrange(theta_loc[:, 5:10], "b k -> k b"),
                                                 rearrange(theta_scale[:, 5:10], "b k -> k b")))
            with plate_grade:
                pyro.sample("b", dist.Normal(rearrange(theta_loc[:, 10:13], "b k -> k b"),
                                             rearrange(theta_scale[:, 10:13], "b k -> k b")))
                pyro.sample("c", dist.Normal(rearrange(theta_loc[:, 13:16], "b k -> k b"),
                                             rearrange(theta_scale[:, 13:16], "b k -> k b")))

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 75
            sample_dict["n_pair"] = 5
            sample_dict["n_grade"] = 3
            assert sample_dict["N"] % (sample_dict["n_pair"] * sample_dict["n_grade"]) == 0
            grade_pair_mesh = torch.stack(torch.meshgrid(
                torch.arange(sample_dict["n_grade"], device=self.device),
                torch.arange(sample_dict["n_pair"], device=self.device),
                indexing="ij"
            ), dim=-1)  # (3, 5, 2)
            sample_dict["grade"] = repeat(grade_pair_mesh[..., 0],
                                          "k1 k2 -> (r k1 k2)",
                                          r=sample_dict["N"] // (sample_dict["n_pair"] * sample_dict["n_grade"]))
            sample_dict["pair"] = repeat(grade_pair_mesh[..., 1],
                                          "k1 k2 -> (r k1 k2)",
                                          r=sample_dict["N"] // (sample_dict["n_pair"] * sample_dict["n_grade"]))
            sample_dict["grade_pair"] = torch.randperm(sample_dict["n_pair"],
                                                        generator=torch.Generator(device=self.device).manual_seed(1234),
                                                        device=self.device)
            sample_dict["treatment"] = torch.randint(low=0, high=10,
                                                     size=(sample_dict["N"], batch_size),
                                                     device=self.device)
            sample_dict["pre_test"] = torch.randint(low=20, high=40,
                                                     size=(sample_dict["N"], batch_size),
                                                     device=self.device)
            sample_dict["sigma_a"] = repeat(torch.arange(sample_dict["n_pair"], device=self.device) + 1,
                                            "n_pair -> n_pair b",
                                            b=batch_size)
            sample_dict["sigma_y"] = repeat((torch.arange(sample_dict["n_grade"], device=self.device) + 1) * 0.1,
                                            "n_grade -> n_grade b",
                                             b=batch_size)
        else:
            sample_dict = copy.copy(sample_dict)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_pair = pyro.plate("plate_pair", sample_dict["n_pair"], dim=-2)
        plate_grade = pyro.plate("plate_grade", sample_dict["n_grade"], dim=-2)
        plate_grade_pair = pyro.plate("plate_grade_pair", sample_dict["n_pair"], dim=-2)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)

        with plate_batch:
            with plate_grade_pair:
                sample_dict["mu_a"] = pyro.sample("mu_a", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                      torch.tensor(1.0, device=self.device)))
            with plate_pair:
                sample_dict["eta_a"] = pyro.sample("eta_a", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                        torch.tensor(1.0, device=self.device)))
            a = 50 * sample_dict["mu_a"][..., sample_dict["grade_pair"], :] + \
                sample_dict["sigma_a"][..., sample_dict["grade_pair"], :] * sample_dict["eta_a"]
            with plate_grade:
                sample_dict["b"] = pyro.sample("b", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                torch.tensor(100.0, device=self.device)))
                sample_dict["c"] = pyro.sample("c", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                torch.tensor(100.0, device=self.device)))
            sigma_y_hat = sample_dict["sigma_y"][..., sample_dict["grade"], :]
            with plate_n:
                y_hat = a[..., sample_dict["pair"], :] + \
                        sample_dict["b"][..., sample_dict["grade"], :] * sample_dict["treatment"] + \
                        sample_dict["c"][..., sample_dict["grade"], :] * sample_dict["pre_test"]
                sample_dict["y"] = pyro.sample("y", 
                                               dist.Normal(y_hat, sigma_y_hat),
                                               obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_electric_chr(BaseVAE):
    x_dim = 102
    theta_dim = 7
    
    def _extract_x_func(self, sample_dict):
        treatment = rearrange(sample_dict["treatment"], "n b -> b n")
        y = rearrange(sample_dict["y"], "n b -> b n")
        sigma_a = rearrange(sample_dict["sigma_a"], "b -> b 1")
        sigma_y = rearrange(sample_dict["sigma_y"], "b -> b 1")
        return torch.cat([treatment, y, sigma_a, sigma_y], dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        mu_a = rearrange(sample_dict["mu_a"], "b -> b 1")
        eta = rearrange(sample_dict["eta"], "k b -> b k")
        beta = rearrange(sample_dict["beta"], "b -> b 1")
        return torch.cat([mu_a, eta, beta], dim=-1)
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_pair = pyro.plate("plate_pair", sample_dict["n_pair"], dim=-2)

        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)
        with plate_batch:
            pyro.sample("mu_a", dist.Normal(theta_loc[:, 0], theta_scale[:, 0]))
            with plate_pair:
                pyro.sample("eta", dist.Normal(rearrange(theta_loc[:, 1:6], "k b -> b k"), 
                                               rearrange(theta_scale[:, 1:6], "k b -> b k")))
            pyro.sample("beta", dist.Normal(theta_loc[:, 6], theta_scale[:, 6]))

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 50
            sample_dict["n_pair"] = 5
            assert sample_dict["N"] % sample_dict["n_pair"] == 0
            sample_dict["pair"] = repeat(torch.arange(sample_dict["n_pair"], device=self.device),
                                          "k -> (r k)",
                                          r=sample_dict["N"] // sample_dict["n_pair"])
            sample_dict["treatment"] = torch.randint(low=0, high=10,
                                                     size=(sample_dict["N"], batch_size),
                                                     device=self.device)
            sample_dict["sigma_a"] = torch.ones(batch_size, device=self.device) * 10
            sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)
            sample_dict["sigma_y"] = sample_dict["sigma_y"] * 100
        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_pair = pyro.plate("plate_pair", sample_dict["n_pair"], dim=-2)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)

        with plate_batch:
            sample_dict["mu_a"] = pyro.sample("mu_a", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                  torch.tensor(1.0, device=self.device)))
            with plate_pair:
                sample_dict["eta"] = pyro.sample("eta", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                    torch.tensor(1.0, device=self.device)))
            sample_dict["beta"] = pyro.sample("beta", dist.Normal(torch.tensor(0.0, device=self.device),
                                                                  torch.tensor(1.0, device=self.device)))
            a = 100 * sample_dict["mu_a"] + sample_dict["sigma_a"] * sample_dict["eta"]
            with plate_n:
                y_hat = a[..., sample_dict["pair"], :] + sample_dict["beta"] * sample_dict["treatment"]
                sample_dict["y"] = pyro.sample("y",
                                               dist.Normal(y_hat, 
                                                           repeat(sample_dict["sigma_y"],
                                                                  "b -> n b", n=sample_dict["N"])),
                                                obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_electric_inter(BaseVAE):
    x_dim = 201
    theta_dim = 4
    
    def _extract_x_func(self, sample_dict):
        treatment = rearrange(sample_dict["treatment"], "n b -> b n")
        pre_test = rearrange(sample_dict["pre_test"], "n b -> b n")
        inter = rearrange(sample_dict["inter"], "n b -> b n")
        post_test = rearrange(sample_dict["post_test"], "n b -> b n")
        sigma = rearrange(sample_dict["sigma"], "b -> b 1")
        return torch.cat([treatment, post_test, sigma, pre_test, inter], dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        return sample_dict["beta"]
    
    def guide(self, batch_size, sample_dict):
        pyro.module("encoder", self.encoder)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)

        x = self.extract_x(sample_dict)
        theta_loc, theta_scale = self.encoder(x)
        with plate_batch:
            pyro.sample("beta", dist.Normal(theta_loc, theta_scale).to_event(1))

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = {}
            sample_dict["N"] = 50
            sample_dict["treatment"] = torch.randint(low=0, high=10,
                                                     size=(sample_dict["N"], batch_size),
                                                     device=self.device)
            sample_dict["pre_test"] = torch.randint(low=20, high=30,
                                                    size=(sample_dict["N"], batch_size),
                                                    device=self.device)
            sample_dict["inter"] = sample_dict["treatment"] * sample_dict["pre_test"]
            sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plate_batch = pyro.plate("plate_batch", batch_size, dim=-1)
        plate_n = pyro.plate("plate_n", sample_dict["N"], dim=-2)

        with plate_batch:
            sample_dict["beta"] = pyro.sample("beta",
                                              dist.Normal(torch.zeros(batch_size, 4, device=self.device),
                                                          torch.ones(batch_size, 4, device=self.device) * 0.5).to_event(1))
            with plate_n:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["treatment"] + \
                        sample_dict["beta"][..., 2] * sample_dict["pre_test"] + \
                        sample_dict["beta"][..., 3] * sample_dict["inter"]
                sample_dict["post_test"] = pyro.sample("post_test",
                                                       dist.Normal(y_hat, 
                                                                   repeat(sample_dict["sigma"], "b -> n b", n=sample_dict["N"])),
                                                       obs=sample_dict["post_test"] if "post_test" in sample_dict else None)
        return sample_dict
