import torch
import math
import pyro
import copy
import pyro.distributions as dist

from einops import repeat, rearrange

from pyro_cases.base_vae import (BaseVAE, 
                                 BaseVAEwRegister, 
                                 SampleDict, 
                                 MetaDataContext, 
                                 DataContext)

import sbibm
from pathlib import Path

SBIBM_INSTALL_PATH = Path(sbibm.__file__).parent


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
                                            obs=sample_dict.get("x", None))
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
                                           obs=sample_dict.get("x", None))
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
                                            obs=sample_dict.get("x", None))
            
        sample_dict["x"] = rearrange(sample_dict["x"], "r b two -> b (r two)")
        return sample_dict
            

class SLCPwDistractorVAE(SLCPVAE):
    x_dim = 100
    theta_dim = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                                            obs=sample_dict.get("x", None))  # (b, 100)
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
                                           obs=sample_dict.get("x", None))
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


class ARM_anova_randon_nopred(BaseVAEwRegister):
    x_dim = 52
    theta_dim = 6

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_j": self.plate("plate_j", sample_dict["J"], dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["J"] = 5
                sample_dict["county"] = repeat(torch.arange(sample_dict["J"], device=self.device), 
                                               "j -> (j r)", r=sample_dict["N"] // sample_dict["J"])
            with DataContext(sample_dict, self):
                sample_dict["sigma_a"] = torch.ones(batch_size, device=self.device) * 10
                sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)
        
        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_j"]:
                sample_dict["a"] = self.r_sample("a", dist.Normal(10 * sample_dict["mu_a"],
                                                                         sample_dict["sigma_a"]))  # (b, j)
            with plates["plate_n"]:
                y_hat = sample_dict["a"][..., sample_dict["county"], :]
                sample_dict["y"] = self.r_obs("y",
                                                dist.Normal(y_hat,
                                                            repeat(sample_dict["sigma_y"], 
                                                                "b -> n b", 
                                                                n=sample_dict["N"])),
                                                obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict

class ARM_anova_randon_nopred_chr(BaseVAEwRegister):
    x_dim = 52
    theta_dim = 6
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_j": self.plate("plate_j", sample_dict["J"], dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2)
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["J"] = 5
                sample_dict["county"] = repeat(torch.arange(sample_dict["J"], device=self.device), 
                                               "j -> (j r)", r=sample_dict["N"] // sample_dict["J"])
            with DataContext(sample_dict, self):
                sample_dict["sigma_a"] = torch.ones(batch_size, device=self.device) * 10
                sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
            
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_j"]:
                sample_dict["eta"] = self.r_sample("eta", self.scalar_normal_dist(0.0, 1.0))
            a = sample_dict["mu_a"] + sample_dict["sigma_a"] * sample_dict["eta"]  # (j, b)
            with plates["plate_n"]:
                y_hat = a[..., sample_dict["county"], :]
                sample_dict["y"] = self.r_obs("y",
                                            dist.Normal(y_hat,
                                                        repeat(sample_dict["sigma_y"], 
                                                            "b -> n b", 
                                                            n=sample_dict["N"])),
                                            obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict

class ARM_congress(BaseVAEwRegister):
    x_dim = 151
    theta_dim = 3
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
                sample_dict["incumbency_88"] = torch.rand(sample_dict["N"], batch_size, device=self.device) * 100 + 10
                sample_dict["vote_86"] = torch.rand(sample_dict["N"], batch_size, device=self.device) * 100 + 10
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((3, ), device=self.device),
                                                                    0.2 * torch.ones((3, ), device=self.device)).to_event(1))  # (b, 3)
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["vote_86"] + \
                        sample_dict["beta"][..., 2] * sample_dict["incumbency_88"]  # (n, b)
                sample_dict["vote_88"] = self.r_obs("vote_88", 
                                                            dist.Normal(y_hat,
                                                                        scale=repeat(sample_dict["sigma"], 
                                                                                    "b -> n b", 
                                                                                    n=sample_dict["N"])),
                                                            obs=sample_dict["vote_88"] if "vote_88" in sample_dict else None)
        return sample_dict


class ARM_earnings1(BaseVAEwRegister):
    x_dim = 150
    theta_dim = 3
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["height"] = torch.randint(low=150, high=190, size=(sample_dict["N"], batch_size),
                                                      device=self.device)
                sample_dict["male"] = torch.randint(low=0, high=2, size=(sample_dict["N"], batch_size),
                                                    device=self.device)
        else:
            sample_dict = copy.copy(sample_dict)
        
        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((3, ), device=self.device),
                                                                    0.2 * torch.ones((3, ), device=self.device)).to_event(1))
            logits = sample_dict["beta"][:, 0] + \
                     sample_dict["beta"][:, 1] * sample_dict["height"] + \
                     sample_dict["beta"][:, 2] * sample_dict["male"]
            with plates["plate_n"]:
                sample_dict["earn_pos"] = self.r_obs("earn_pos", 
                                                    dist.Bernoulli(logits=logits),
                                                    obs=sample_dict["earn_pos"] if "earn_pos" in sample_dict else None)
        return sample_dict


class ARM_earnings2(BaseVAEwRegister):
    x_dim = 151
    theta_dim = 3

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["height"] = torch.randint(low=150, high=190, size=(sample_dict["N"], batch_size),
                                                      device=self.device)
                sample_dict["male"] = torch.randint(low=0, high=2, size=(sample_dict["N"], batch_size),
                                                    device=self.device)
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((3, ), device=self.device),
                                                                     0.2 * torch.ones((3, ), device=self.device)).to_event(1))
            log_earnings = sample_dict["beta"][:, 0] + \
                           sample_dict["beta"][:, 1] * sample_dict["height"] + \
                           sample_dict["beta"][:, 2] * sample_dict["male"]
            with plates["plate_n"]:
                sample_dict["log_earnings"] = self.r_obs("log_earnings", 
                                                        dist.Normal(loc=log_earnings,
                                                                    scale=repeat(sample_dict["sigma"],
                                                                                "b -> n b",
                                                                                n=sample_dict["N"])),
                                                        obs=sample_dict["log_earnings"] if "log_earnings" in sample_dict else None)
        return sample_dict


class ARM_earnings_latin_square(BaseVAEwRegister):
    x_dim = 607
    theta_dim = 76

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_a": self.plate("plate_a", sample_dict["n_eth"], dim=-3),
            "plate_b": self.plate("plate_b", sample_dict["n_age"], dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }
    

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 300
                sample_dict["n_age"] = 5
                sample_dict["n_eth"] = 5
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
            with DataContext(sample_dict, self):
                sample_dict["sigma_a1"] = torch.ones(batch_size, device=self.device) * 100
                sample_dict["sigma_a2"] = torch.ones(batch_size, device=self.device) * 100
                sample_dict["sigma_b1"] = torch.ones(batch_size, device=self.device) * 100
                sample_dict["sigma_b2"] = torch.ones(batch_size, device=self.device) * 10
                sample_dict["sigma_c"] = torch.ones(batch_size, device=self.device) * 100
                sample_dict["sigma_d"] = torch.ones(batch_size, device=self.device) * 10
                sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
                sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 10
        else:
            sample_dict = copy.copy(sample_dict)
        
        plates = self.get_plates(batch_size, sample_dict)   
        with plates["plate_batch"]:
            for param_name in ["mu_a1", "mu_a2", "mu_b1", "mu_b2", "mu_c", "mu_d"]:
                sample_dict[param_name] = self.r_sample(param_name, self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_a"]:
                sample_dict["a1"] = self.r_sample("a1",
                                                    dist.Normal(10 * sample_dict["mu_a1"],
                                                                sample_dict["sigma_a1"]))
                sample_dict["a2"] = self.r_sample("a2",
                                                    dist.Normal(10 * sample_dict["mu_a2"],
                                                                sample_dict["sigma_a2"]))
                
            with plates["plate_b"]:
                sample_dict["b1"] = self.r_sample("b1",
                                                    dist.Normal(10 * sample_dict["mu_b1"],
                                                                sample_dict["sigma_b1"]))
                sample_dict["b2"] = self.r_sample("b2",
                                                    dist.Normal(0.1 * sample_dict["mu_b2"],
                                                                sample_dict["sigma_b2"]))
                
            with plates["plate_a"], plates["plate_b"]:
                sample_dict["c"] = self.r_sample("c",
                                                    dist.Normal(10 * sample_dict["mu_c"],
                                                                sample_dict["sigma_c"]))
                sample_dict["d"] = self.r_sample("d",
                                                    dist.Normal(0.1 * sample_dict["mu_d"],
                                                                sample_dict["sigma_d"]))
                
            with plates["plate_n"]:
                c = sample_dict["c"][sample_dict["eth"], sample_dict["age"], :]  # (n, b)
                d = sample_dict["d"][sample_dict["eth"], sample_dict["age"], :]  # (n, b)
                
                y_hat = sample_dict["a1"].squeeze(-2)[sample_dict["eth"], :] + \
                        sample_dict["a2"].squeeze(-2)[sample_dict["eth"], :] * sample_dict["x"] + \
                        sample_dict["b1"][sample_dict["age"], :] + \
                        sample_dict["b2"][sample_dict["age"], :] * sample_dict["x"] + \
                        c + d * sample_dict["x"]
                sample_dict["y"] = self.r_obs("y",
                                               dist.Normal(y_hat, 
                                                           repeat(sample_dict["sigma_y"], 
                                                                  "b -> n b", 
                                                                  n=sample_dict["N"])),
                                               obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_earnings_latin_square_chr(BaseVAEwRegister):
    x_dim = 607
    theta_dim = 76
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_a": self.plate("plate_a", sample_dict["n_eth"], dim=-3),
            "plate_b": self.plate("plate_b", sample_dict["n_age"], dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 300
                sample_dict["n_age"] = 5
                sample_dict["n_eth"] = 5
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
            with DataContext(sample_dict, self):
                sample_dict["sigma_a1"] = torch.ones(batch_size, device=self.device) * 5
                sample_dict["sigma_a2"] = torch.ones(batch_size, device=self.device) * 1
                sample_dict["sigma_b1"] = torch.ones(batch_size, device=self.device) * 5
                sample_dict["sigma_b2"] = torch.ones(batch_size, device=self.device) * 0.1
                sample_dict["sigma_c"] = torch.ones(batch_size, device=self.device) * 0.1
                sample_dict["sigma_d"] = torch.ones(batch_size, device=self.device) * 0.01
                sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
                sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 30 + 200
        else:
            sample_dict = copy.copy(sample_dict)
        
        plates = self.get_plates(batch_size, sample_dict)   
        with plates["plate_batch"]:
            for param_name in ["mu_a1", "mu_a2", "mu_b1", "mu_b2", "mu_c", "mu_d"]:
                sample_dict[param_name] = self.r_sample(param_name, self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_a"]:
                sample_dict["eta_a1"] = self.r_sample("eta_a1", self.scalar_normal_dist(0.0, 1.0))
                sample_dict["eta_a2"] = self.r_sample("eta_a2", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_b"]:
                sample_dict["eta_b1"] = self.r_sample("eta_b1", self.scalar_normal_dist(0.0, 1.0))
                sample_dict["eta_b2"] = self.r_sample("eta_b2", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_a"], plates["plate_b"]:
                sample_dict["eta_c"] = self.r_sample("eta_c", self.scalar_normal_dist(0.0, 1.0))
                sample_dict["eta_d"] = self.r_sample("eta_d", self.scalar_normal_dist(0.0, 1.0))
            
            a1 = 5 * sample_dict["mu_a1"] + sample_dict["sigma_a1"] * sample_dict["eta_a1"]
            a2 = sample_dict["mu_a2"] + sample_dict["sigma_a2"] * sample_dict["eta_a2"]
            b1 = 5 * sample_dict["mu_b1"] + sample_dict["sigma_b1"] * sample_dict["eta_b1"]
            b2 = 0.1 * sample_dict["mu_b2"] + sample_dict["sigma_b2"] * sample_dict["eta_b2"]
            c = 0.1 * sample_dict["mu_c"] + sample_dict["sigma_c"] * sample_dict["eta_c"]
            d = 0.01 * sample_dict["mu_d"] + sample_dict["sigma_d"] * sample_dict["eta_d"]

            with plates["plate_n"]:
                c = c[sample_dict["eth"], sample_dict["age"], :]  # (n, b)
                d = d[sample_dict["eth"], sample_dict["age"], :]  # (n, b)
                
                y_hat = a1.squeeze(-2)[sample_dict["eth"], :] + \
                        a2.squeeze(-2)[sample_dict["eth"], :] * sample_dict["x"] + \
                        b1[sample_dict["age"], :] + \
                        b2[sample_dict["age"], :] * sample_dict["x"] + \
                        c + d * sample_dict["x"]
                sample_dict["y"] = self.r_obs("y",
                                               dist.Normal(y_hat, 
                                                           repeat(sample_dict["sigma_y"], 
                                                                  "b -> n b", 
                                                                  n=sample_dict["N"])),
                                               obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict
    

class ARM_earnings_vary_si(BaseVAEwRegister):
    x_dim = 103
    theta_dim = 12
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_a": self.plate("plate_a", sample_dict["n_eth"], dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_eth"] = 5
                assert sample_dict["N"] % sample_dict["n_eth"] == 0
                sample_dict["eth"] = repeat(torch.arange(sample_dict["n_eth"], device=self.device),
                                            "k -> (r k)", r=sample_dict["N"] // sample_dict["n_eth"])
            with DataContext(sample_dict, self):
                sample_dict["height"] = torch.randint(low=150, high=190, size=(sample_dict["N"], batch_size),
                                                      device=self.device)
                sample_dict["sigma_a1"] = torch.ones(batch_size, device=self.device) * 10
                sample_dict["sigma_a2"] = torch.ones(batch_size, device=self.device) * 0.01
                sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a1"] = self.r_sample("mu_a1", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["mu_a2"] = self.r_sample("mu_a2", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_a"]:
                sample_dict["a1"] = self.r_sample("a1",
                                                    dist.Normal(10 * sample_dict["mu_a1"],
                                                                sample_dict["sigma_a1"]))
                sample_dict["a2"] = self.r_sample("a2",
                                                    dist.Normal(0.01 * sample_dict["mu_a2"],
                                                                sample_dict["sigma_a2"]))
            with plates["plate_n"]:
                y_hat = sample_dict["a1"][..., sample_dict["eth"], :] + \
                        sample_dict["a2"][..., sample_dict["eth"], :] * sample_dict["height"]
                sample_dict["log_earnings"] = self.r_obs("log_earnings", 
                                                        dist.Normal(loc=y_hat,
                                                                    scale=repeat(sample_dict["sigma_y"],
                                                                                "b -> n b",
                                                                                n=sample_dict["N"])),
                                                        obs=sample_dict["log_earnings"] if "log_earnings" in sample_dict else None)
        return sample_dict


class ARM_earnings_vary_si_chr(BaseVAEwRegister):
    x_dim = 103
    theta_dim = 12
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_a": self.plate("plate_a", sample_dict["n_eth"], dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_eth"] = 5
                assert sample_dict["N"] % sample_dict["n_eth"] == 0
                sample_dict["eth"] = repeat(torch.arange(sample_dict["n_eth"], device=self.device),
                                            "k -> (r k)", r=sample_dict["N"] // sample_dict["n_eth"])
            with DataContext(sample_dict, self):
                sample_dict["height"] = torch.randint(low=150, high=190, size=(sample_dict["N"], batch_size),
                                                      device=self.device)
                sample_dict["sigma_a1"] = torch.ones(batch_size, device=self.device) * 10
                sample_dict["sigma_a2"] = torch.ones(batch_size, device=self.device) * 0.01
                sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a1"] = self.r_sample("mu_a1", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["mu_a2"] = self.r_sample("mu_a2", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_a"]:
                sample_dict["eta1"] = self.r_sample("eta1", self.scalar_normal_dist(0.0, 1.0))
                sample_dict["eta2"] = self.r_sample("eta2", self.scalar_normal_dist(0.0, 1.0))
            a1 = 10 * sample_dict["mu_a1"] + sample_dict["sigma_a1"] * sample_dict["eta1"]
            a2 = 0.1 * sample_dict["mu_a2"] + sample_dict["sigma_a2"] * sample_dict["eta2"]
            with plates["plate_n"]:
                y_hat = a1[..., sample_dict["eth"], :] + \
                        a2[..., sample_dict["eth"], :] * sample_dict["height"]
                sample_dict["log_earnings"] = self.r_obs("log_earnings", 
                                                        dist.Normal(loc=y_hat,
                                                                    scale=repeat(sample_dict["sigma_y"],
                                                                                "b -> n b",
                                                                                n=sample_dict["N"])),
                                                        obs=sample_dict["log_earnings"] if "log_earnings" in sample_dict else None)
        return sample_dict
    

class ARM_election88_ch14(BaseVAEwRegister):
    x_dim = 301
    theta_dim = 13

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_a": self.plate("plate_a", sample_dict["n_state"], dim=-2),
            "plate_b": self.plate("plate_b", 2, dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 100
                sample_dict["n_state"] = 10
                assert sample_dict["N"] % sample_dict["n_state"] == 0
                sample_dict["state"] = repeat(torch.arange(sample_dict["n_state"], device=self.device),
                                              "k -> (r k)", r=sample_dict["N"] // sample_dict["n_state"])
            with DataContext(sample_dict, self):
                sample_dict["female"] = torch.randint(low=0, high=2, size=(sample_dict["N"], batch_size),
                                                      device=self.device)
                sample_dict["black"] = torch.randint(low=0, high=2, size=(sample_dict["N"], batch_size),
                                                     device=self.device)
                sample_dict["sigma_a"] = torch.ones(batch_size, device=self.device) * 1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_a"]:
                sample_dict["a"] = self.r_sample("a", dist.Normal(sample_dict["mu_a"],
                                                                sample_dict["sigma_a"]))
            with plates["plate_b"]:
                sample_dict["b"] = self.r_sample("b", self.scalar_normal_dist(0.0, 100.0))
            with plates["plate_n"]:
                y_hat = sample_dict["b"][..., 0, :].unsqueeze(-2) * sample_dict["black"] + \
                        sample_dict["b"][..., 1, :].unsqueeze(-2) * sample_dict["female"] + \
                        sample_dict["a"][..., sample_dict["state"], :]
                sample_dict["y"] = self.r_obs("y", 
                                               dist.Bernoulli(logits=y_hat),
                                              obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_election88_ch19(BaseVAEwRegister):
    x_dim = 244
    theta_dim = 18
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_beta": self.plate("plate_beta", 4, dim=-2),
            "plate_age": self.plate("plate_age", sample_dict["n_age"], dim=-3),
            "plate_edu": self.plate("plate_edu", sample_dict["n_edu"], dim=-2),
            "plate_state": self.plate("plate_state", sample_dict["n_state"], dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
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
                indexing="ij"), dim=-1)  # (2, 2, 2, 3)
                sample_dict["state"] = repeat(state_age_edu_region_mesh[..., 0].reshape(-1),
                                              "k -> (r k)", r=sample_dict["N"] // total_permutate)
                sample_dict["age"] = repeat(state_age_edu_region_mesh[..., 1].reshape(-1),
                                            "k -> (r k)", r=sample_dict["N"] // total_permutate)
                sample_dict["edu"] = repeat(state_age_edu_region_mesh[..., 2].reshape(-1),
                                            "k -> (r k)", r=sample_dict["N"] // total_permutate)
            with DataContext(sample_dict, self):
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

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_age"] = self.r_sample("mu_age", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["mu_edu"] = self.r_sample("mu_edu", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["mu_age_edu"] = self.r_sample("mu_age_edu", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["mu_state"] = self.r_sample("mu_state", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_beta"]:
                sample_dict["beta"] = self.r_sample("beta", self.scalar_normal_dist(0.0, 100.0))  #(4, b)
            with plates["plate_age"]:
                sample_dict["b_age"] = self.r_sample("b_age", dist.Normal(100 * sample_dict["mu_age"],
                                                                            sample_dict["sigma_age"]))  # (n_age, 1, b)
            with plates["plate_edu"]:
                sample_dict["b_edu"] = self.r_sample("b_edu", dist.Normal(100 * sample_dict["mu_edu"],
                                                                            sample_dict["sigma_edu"]))  # (n_edu, b)
            with plates["plate_age"], plates["plate_edu"]:
                sample_dict["b_age_edu"] = self.r_sample("b_age_edu", dist.Normal(100 * sample_dict["mu_age_edu"],
                                                                                sample_dict["sigma_age_edu"]))  # (n_age, n_edu, b)
            with plates["plate_state"]:
                sample_dict["b_state"] = self.r_sample("b_state", dist.Normal(10 * sample_dict["mu_state"],
                                                                            sample_dict["sigma_state"]))  # (n_state, b)
            with plates["plate_n"]:
                Xbeta = sample_dict["beta"][..., 0, :].unsqueeze(-2) + \
                        sample_dict["beta"][..., 1, :].unsqueeze(-2) * sample_dict["female"] + \
                        sample_dict["beta"][..., 2, :].unsqueeze(-2) * sample_dict["black"] + \
                        sample_dict["beta"][..., 3, :].unsqueeze(-2) * sample_dict["female"] * sample_dict["black"] + \
                        sample_dict["b_age"][..., sample_dict["age"], :, :].squeeze(-2) + \
                        sample_dict["b_edu"][..., sample_dict["edu"], :] + \
                        sample_dict["b_age_edu"][..., sample_dict["age"], sample_dict["edu"], :] + \
                        sample_dict["b_state"][..., sample_dict["state"], :]
                sample_dict["y"] = self.r_obs("y",
                                               dist.Bernoulli(logits=Xbeta),
                                               obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_electric(BaseVAEwRegister):
    x_dim = 102
    theta_dim = 7
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_pair": self.plate("plate_pair", sample_dict["n_pair"], dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_pair"] = 5
                assert sample_dict["N"] % sample_dict["n_pair"] == 0
                sample_dict["pair"] = repeat(torch.arange(sample_dict["n_pair"], device=self.device),
                                             "k -> (r k)", r=sample_dict["N"] // sample_dict["n_pair"])
            with DataContext(sample_dict, self):
                sample_dict["treatment"] = torch.randint(low=0, high=10,
                                                        size=(sample_dict["N"], batch_size),
                                                        device=self.device)
                sample_dict["sigma_a"] = torch.ones(batch_size, device=self.device) * 10
                sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_pair"]:
                sample_dict["a"] = self.r_sample("a", dist.Normal(100 * sample_dict["mu_a"],
                                                                sample_dict["sigma_a"]))
            sample_dict["beta"] = self.r_sample("beta", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_n"]:
                y_hat = sample_dict["a"][..., sample_dict["pair"], :] + sample_dict["beta"] * sample_dict["treatment"]
                sample_dict["y"] = self.r_obs("y",
                                               dist.Normal(y_hat, 
                                                           repeat(sample_dict["sigma_y"],
                                                                  "b -> n b", n=sample_dict["N"])),
                                               obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_electric_1a(BaseVAEwRegister):
    x_dim = 158
    theta_dim = 13
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_pair": self.plate("plate_pair", sample_dict["n_pair"], dim=-2),
            "plate_grade": self.plate("plate_grade", sample_dict["n_grade"], dim=-2),
            "plate_grade_pair": self.plate("plate_grade_pair", sample_dict["n_pair"], dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
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
            with DataContext(sample_dict, self):
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

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            with plates["plate_grade_pair"]:
                sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            sigma_a_hat = sample_dict["sigma_a"][..., sample_dict["grade_pair"], :]
            mu_a_hat = 100 * sample_dict["mu_a"][..., sample_dict["grade_pair"], :]
            with plates["plate_pair"]:
                sample_dict["a"] = self.r_sample("a", dist.Normal(mu_a_hat, sigma_a_hat))
            with plates["plate_grade"]:
                sample_dict["b"] = self.r_sample("b", self.scalar_normal_dist(0.0, 100.0))
            sigma_y_hat = sample_dict["sigma_y"][..., sample_dict["grade"], :]
            with plates["plate_n"]:
                y_hat = sample_dict["a"][..., sample_dict["pair"], :] + \
                        sample_dict["b"][..., sample_dict["grade"], :] * sample_dict["treatment"]
                sample_dict["y"] = self.r_obs("y", 
                                               dist.Normal(y_hat, sigma_y_hat),
                                               obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_electric_1a_chr(BaseVAEwRegister):
    x_dim = 158
    theta_dim = 13
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_pair": self.plate("plate_pair", sample_dict["n_pair"], dim=-2),
            "plate_grade": self.plate("plate_grade", sample_dict["n_grade"], dim=-2),
            "plate_grade_pair": self.plate("plate_grade_pair", sample_dict["n_pair"], dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
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
            with DataContext(sample_dict, self):
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

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            with plates["plate_grade_pair"]:
                sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            sigma_a_hat = sample_dict["sigma_a"][..., sample_dict["grade_pair"], :]
            mu_a_hat = 100 * sample_dict["mu_a"][..., sample_dict["grade_pair"], :]
            with plates["plate_pair"]:
                sample_dict["eta_a"] = self.r_sample("eta_a", self.scalar_normal_dist(0.0, 1.0))
            a = mu_a_hat + sigma_a_hat * sample_dict["eta_a"]
            with plates["plate_grade"]:
                sample_dict["b"] = self.r_sample("b", self.scalar_normal_dist(0.0, 100.0))
            sigma_y_hat = sample_dict["sigma_y"][..., sample_dict["grade"], :]
            with plates["plate_n"]:
                y_hat = a[..., sample_dict["pair"], :] + \
                        sample_dict["b"][..., sample_dict["grade"], :] * sample_dict["treatment"]
                sample_dict["y"] = self.r_obs("y", 
                                               dist.Normal(y_hat, sigma_y_hat),
                                               obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_electric_1b(BaseVAEwRegister):
    x_dim = 152
    theta_dim = 8
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_pair": self.plate("plate_pair", sample_dict["n_pair"], dim=-2),
            "plate_beta": self.plate("plate_beta", 2, dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_pair"] = 5
                assert sample_dict["N"] % sample_dict["n_pair"] == 0
                sample_dict["pair"] = repeat(torch.arange(sample_dict["n_pair"], device=self.device),
                                            "k -> (r k)",
                                            r=sample_dict["N"] // sample_dict["n_pair"])
            with DataContext(sample_dict, self):
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

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_pair"]:
                sample_dict["a"] = self.r_sample("a", dist.Normal(100 * sample_dict["mu_a"], sample_dict["sigma_a"]))
            with plates["plate_beta"]:
                sample_dict["beta"] = self.r_sample("beta", self.scalar_normal_dist(0.0, 100.0))
            with plates["plate_n"]:
                y_hat = sample_dict["a"][..., sample_dict["pair"], :] + \
                        sample_dict["beta"][..., 0, :].unsqueeze(-2) * sample_dict["treatment"] + \
                        sample_dict["beta"][..., 1, :].unsqueeze(-2) * sample_dict["pre_test"]
                sample_dict["y"] = self.r_obs("y",
                                               dist.Normal(y_hat, 
                                                           repeat(sample_dict["sigma_y"],
                                                                  "b -> n b", n=sample_dict["N"])),
                                                obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_electric_1b_chr(BaseVAEwRegister):
    x_dim = 152
    theta_dim = 8

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_pair": self.plate("plate_pair", sample_dict["n_pair"], dim=-2),
            "plate_beta": self.plate("plate_beta", 2, dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_pair"] = 5
                assert sample_dict["N"] % sample_dict["n_pair"] == 0
                sample_dict["pair"] = repeat(torch.arange(sample_dict["n_pair"], device=self.device),
                                            "k -> (r k)",
                                            r=sample_dict["N"] // sample_dict["n_pair"])
            with DataContext(sample_dict, self):
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

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_pair"]:
                sample_dict["eta"] = self.r_sample("eta", self.scalar_normal_dist(0.0, 1.0))
            a = 100 * sample_dict["mu_a"] + sample_dict["sigma_a"] * sample_dict["eta"]
            with plates["plate_beta"]:
                sample_dict["beta"] = self.r_sample("beta", self.scalar_normal_dist(0.0, 100.0))
            with plates["plate_n"]:
                y_hat = a[..., sample_dict["pair"], :] + \
                        sample_dict["beta"][..., 0, :].unsqueeze(-2) * sample_dict["treatment"] + \
                        sample_dict["beta"][..., 1, :].unsqueeze(-2) * sample_dict["pre_test"]
                sample_dict["y"] = self.r_obs("y",
                                               dist.Normal(y_hat, 
                                                           repeat(sample_dict["sigma_y"],
                                                                  "b -> n b", n=sample_dict["N"])),
                                                obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_electric_1c(BaseVAEwRegister):
    x_dim = 233
    theta_dim = 16
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_pair": self.plate("plate_pair", sample_dict["n_pair"], dim=-2),
            "plate_grade": self.plate("plate_grade", sample_dict["n_grade"], dim=-2),
            "plate_grade_pair": self.plate("plate_grade_pair", sample_dict["n_pair"], dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
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
            with DataContext(sample_dict, self):
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

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            with plates["plate_grade_pair"]:
                sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            sigma_a_hat = sample_dict["sigma_a"][..., sample_dict["grade_pair"], :]
            mu_a_hat = 40 * sample_dict["mu_a"][..., sample_dict["grade_pair"], :]
            with plates["plate_pair"]:
                sample_dict["a"] = self.r_sample("a", dist.Normal(mu_a_hat, sigma_a_hat))
            with plates["plate_grade"]:
                sample_dict["b"] = self.r_sample("b", self.scalar_normal_dist(0.0, 100.0))
                sample_dict["c"] = self.r_sample("c", self.scalar_normal_dist(0.0, 100.0))
            sigma_y_hat = sample_dict["sigma_y"][..., sample_dict["grade"], :]
            with plates["plate_n"]:
                y_hat = sample_dict["a"][..., sample_dict["pair"], :] + \
                        sample_dict["b"][..., sample_dict["grade"], :] * sample_dict["treatment"] + \
                        sample_dict["c"][..., sample_dict["grade"], :] * sample_dict["pre_test"]
                sample_dict["y"] = self.r_obs("y", 
                                               dist.Normal(y_hat, sigma_y_hat),
                                               obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_electric_1c_chr(BaseVAEwRegister):
    x_dim = 233
    theta_dim = 16
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_pair": self.plate("plate_pair", sample_dict["n_pair"], dim=-2),
            "plate_grade": self.plate("plate_grade", sample_dict["n_grade"], dim=-2),
            "plate_grade_pair": self.plate("plate_grade_pair", sample_dict["n_pair"], dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
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
            with DataContext(sample_dict, self):
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

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            with plates["plate_grade_pair"]:
                sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_pair"]:
                sample_dict["eta_a"] = self.r_sample("eta_a", self.scalar_normal_dist(0.0, 1.0))
            a = 50 * sample_dict["mu_a"][..., sample_dict["grade_pair"], :] + \
                sample_dict["sigma_a"][..., sample_dict["grade_pair"], :] * sample_dict["eta_a"]
            with plates["plate_grade"]:
                sample_dict["b"] = self.r_sample("b", self.scalar_normal_dist(0.0, 100.0))
                sample_dict["c"] = self.r_sample("c", self.scalar_normal_dist(0.0, 100.0))
            sigma_y_hat = sample_dict["sigma_y"][..., sample_dict["grade"], :]
            with plates["plate_n"]:
                y_hat = a[..., sample_dict["pair"], :] + \
                        sample_dict["b"][..., sample_dict["grade"], :] * sample_dict["treatment"] + \
                        sample_dict["c"][..., sample_dict["grade"], :] * sample_dict["pre_test"]
                sample_dict["y"] = self.r_obs("y", 
                                               dist.Normal(y_hat, sigma_y_hat),
                                               obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_electric_chr(BaseVAEwRegister):
    x_dim = 102
    theta_dim = 7
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_pair": self.plate("plate_pair", sample_dict["n_pair"], dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_pair"] = 5
                assert sample_dict["N"] % sample_dict["n_pair"] == 0
                sample_dict["pair"] = repeat(torch.arange(sample_dict["n_pair"], device=self.device),
                                            "k -> (r k)",
                                            r=sample_dict["N"] // sample_dict["n_pair"])
            with DataContext(sample_dict, self):
                sample_dict["treatment"] = torch.randint(low=0, high=10,
                                                        size=(sample_dict["N"], batch_size),
                                                        device=self.device)
                sample_dict["sigma_a"] = torch.ones(batch_size, device=self.device) * 10
                sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)
        
        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_pair"]:
                sample_dict["eta"] = self.r_sample("eta", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["beta"] = self.r_sample("beta", self.scalar_normal_dist(0.0, 1.0))
            a = 100 * sample_dict["mu_a"] + sample_dict["sigma_a"] * sample_dict["eta"]
            with plates["plate_n"]:
                y_hat = a[..., sample_dict["pair"], :] + sample_dict["beta"] * sample_dict["treatment"]
                sample_dict["y"] = self.r_obs("y",
                                               dist.Normal(y_hat, 
                                                           repeat(sample_dict["sigma_y"],
                                                                  "b -> n b", n=sample_dict["N"])),
                                                obs=sample_dict["y"] if "y" in sample_dict else None)
        return sample_dict


class ARM_electric_inter(BaseVAEwRegister):
    x_dim = 201
    theta_dim = 4
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
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

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta",
                                                dist.Normal(torch.zeros(batch_size, 4, device=self.device),
                                                            torch.ones(batch_size, 4, device=self.device) * 0.5).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["treatment"] + \
                        sample_dict["beta"][..., 2] * sample_dict["pre_test"] + \
                        sample_dict["beta"][..., 3] * sample_dict["inter"]
                sample_dict["post_test"] = self.r_obs("post_test",
                                                       dist.Normal(y_hat, 
                                                                   repeat(sample_dict["sigma"], "b -> n b", n=sample_dict["N"])),
                                                       obs=sample_dict["post_test"] if "post_test" in sample_dict else None)
        return sample_dict


class ARM_electric_multi_preds(BaseVAEwRegister):
    x_dim = 151
    theta_dim = 3
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["treatment"] = torch.randint(low=0, high=10,
                                                        size=(sample_dict["N"], batch_size),
                                                        device=self.device)
                sample_dict["pre_test"] = torch.randint(low=20, high=30,
                                                        size=(sample_dict["N"], batch_size),
                                                        device=self.device)
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta",
                                                dist.Normal(torch.zeros(batch_size, 3, device=self.device),
                                                            torch.ones(batch_size, 3, device=self.device) * 0.5).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["treatment"] + \
                        sample_dict["beta"][..., 2] * sample_dict["pre_test"]
                sample_dict["post_test"] = self.r_obs("post_test",
                                                       dist.Normal(y_hat, 
                                                                   repeat(sample_dict["sigma"], "b -> n b", n=sample_dict["N"])),
                                                       obs=sample_dict.get("post_test", None))
        return sample_dict


class ARM_electric_one_pred(BaseVAEwRegister):
    x_dim = 101
    theta_dim = 2
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["treatment"] = torch.randint(low=0, high=10,
                                                        size=(sample_dict["N"], batch_size),
                                                        device=self.device)
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta",
                                                dist.Normal(torch.zeros(batch_size, 2, device=self.device),
                                                            torch.ones(batch_size, 2, device=self.device) * 0.5).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["treatment"]
                sample_dict["post_test"] = self.r_obs("post_test",
                                                       dist.Normal(y_hat, 
                                                                   repeat(sample_dict["sigma"], "b -> n b", n=sample_dict["N"])),
                                                       obs=sample_dict.get("post_test", None))
        return sample_dict
    

class ARM_electric_supp(BaseVAEwRegister):
    x_dim = 151
    theta_dim = 3
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["supp"] = torch.randint(low=0, high=5,
                                                    size=(sample_dict["N"], batch_size),
                                                    device=self.device)
                sample_dict["pre_test"] = torch.randint(low=20, high=30,
                                                        size=(sample_dict["N"], batch_size),
                                                        device=self.device)
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta",
                                                dist.Normal(torch.zeros(batch_size, 3, device=self.device),
                                                            torch.ones(batch_size, 3, device=self.device) * 0.5).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["supp"] + \
                        sample_dict["beta"][..., 2] * sample_dict["pre_test"]
                sample_dict["post_test"] = self.r_obs("post_test",
                                                       dist.Normal(y_hat, 
                                                                   repeat(sample_dict["sigma"], "b -> n b", n=sample_dict["N"])),
                                                       obs=sample_dict.get("post_test", None))
        return sample_dict
    

class ARM_electric_tr(BaseVAEwRegister):
    x_dim = 101
    theta_dim = 2
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["treatment"] = torch.randint(low=0, high=10,
                                                        size=(sample_dict["N"], batch_size),
                                                        device=self.device)
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta",
                                                dist.Normal(torch.zeros(batch_size, 2, device=self.device),
                                                            torch.ones(batch_size, 2, device=self.device) * 0.5).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["treatment"]
                sample_dict["post_test"] = self.r_obs("post_test",
                                                       dist.Normal(y_hat, 
                                                                   repeat(sample_dict["sigma"], "b -> n b", n=sample_dict["N"])),
                                                       obs=sample_dict.get("post_test", None))
        return sample_dict
    

class ARM_electric_trpre(BaseVAEwRegister):
    x_dim = 151
    theta_dim = 3
    
    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["treatment"] = torch.randint(low=0, high=10,
                                                        size=(sample_dict["N"], batch_size),
                                                        device=self.device)
                sample_dict["pre_test"] = torch.randint(low=20, high=30,
                                                        size=(sample_dict["N"], batch_size),
                                                        device=self.device)
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta",
                                                dist.Normal(torch.zeros(batch_size, 3, device=self.device),
                                                            torch.ones(batch_size, 3, device=self.device) * 0.5).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["treatment"] + \
                        sample_dict["beta"][..., 2] * sample_dict["pre_test"]
                sample_dict["post_test"] = self.r_obs("post_test",
                                                       dist.Normal(y_hat, 
                                                                   repeat(sample_dict["sigma"], "b -> n b", n=sample_dict["N"])),
                                                       obs=sample_dict.get("post_test", None))
        return sample_dict
    

class ARM_grades(BaseVAEwRegister):
    x_dim = 101
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }
    
    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["midterm"] = torch.randint(low=60, high=100,
                                                        size=(sample_dict["N"], batch_size),
                                                        device=self.device)
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta",
                                                dist.Normal(torch.zeros(batch_size, 2, device=self.device),
                                                            torch.ones(batch_size, 2, device=self.device) * 0.5).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["midterm"]
                sample_dict["final"] = self.r_obs("final",
                                                    dist.Normal(y_hat, 
                                                                repeat(sample_dict["sigma"], "b -> n b", n=sample_dict["N"])),
                                                    obs=sample_dict.get("final", None))
        return sample_dict


class ARM_hiv_chr(BaseVAEwRegister):
    x_dim = 103
    theta_dim = 12

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_person": self.plate("plate_person", sample_dict["J"], dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }
    
    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["J"] = 5
                assert sample_dict["N"] % sample_dict["J"] == 0
                sample_dict["person"] = repeat(torch.arange(sample_dict["J"], device=self.device),
                                               "k -> (r k)", r=sample_dict["N"] // sample_dict["J"])
            with DataContext(sample_dict, self):
                sample_dict["time"] = dist.Uniform(10.0, 100.0).sample((sample_dict["N"], batch_size)).to(device=self.device)
                sample_dict["sigma_a1"] = torch.ones(batch_size, device=self.device) * 1.0
                sample_dict["sigma_a2"] = torch.ones(batch_size, device=self.device) * 0.1
                sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a1"] = self.r_sample("mu_a1", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["mu_a2"] = self.r_sample("mu_a2", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_person"]:
                sample_dict["eta1"] = self.r_sample("eta1", self.scalar_normal_dist(0.0, 1.0))
                sample_dict["eta2"] = self.r_sample("eta2", self.scalar_normal_dist(0.0, 1.0))
                a1 = sample_dict["mu_a1"] + sample_dict["sigma_a1"] * sample_dict["eta1"]
                a2 = 0.1 * sample_dict["mu_a2"] + sample_dict["sigma_a2"] * sample_dict["eta2"]
            with plates["plate_n"]:
                y_hat = a1[..., sample_dict["person"], :] + \
                        a2[..., sample_dict["person"], :] * sample_dict["time"]
                sample_dict["y"] = self.r_obs("y",
                                            dist.Normal(y_hat, 
                                                        repeat(sample_dict["sigma_y"], "b -> n b", n=sample_dict["N"])),
                                            obs=sample_dict.get("y", None))
        return sample_dict


class ARM_hiv_inter_chr(BaseVAEwRegister):
    x_dim = 153
    theta_dim = 13

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_person": self.plate("plate_person", sample_dict["J"], dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }
    
    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["J"] = 5
                assert sample_dict["N"] % sample_dict["J"] == 0
                sample_dict["person"] = repeat(torch.arange(sample_dict["J"], device=self.device),
                                               "k -> (r k)", r=sample_dict["N"] // sample_dict["J"])
            with DataContext(sample_dict, self):
                sample_dict["time"] = dist.Uniform(10.0, 100.0).sample((sample_dict["N"], batch_size)).to(device=self.device)
                sample_dict["treatment"] = torch.randint(low=0, high=10,
                                                        size=(sample_dict["N"], batch_size),
                                                        device=self.device)
                sample_dict["sigma_a1"] = torch.ones(batch_size, device=self.device) * 10.0
                sample_dict["sigma_a2"] = torch.ones(batch_size, device=self.device) * 0.1
                sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a1"] = self.r_sample("mu_a1", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["mu_a2"] = self.r_sample("mu_a2", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["beta"] = self.r_sample("beta", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_person"]:
                sample_dict["eta1"] = self.r_sample("eta1", self.scalar_normal_dist(0.0, 1.0))
                sample_dict["eta2"] = self.r_sample("eta2", self.scalar_normal_dist(0.0, 1.0))
            a1 = sample_dict["mu_a1"] + sample_dict["sigma_a1"] * sample_dict["eta1"]
            a2 = 0.1 * sample_dict["mu_a2"] + sample_dict["sigma_a2"] * sample_dict["eta2"]
            with plates["plate_n"]:
                y_hat = a1[..., sample_dict["person"], :] + \
                        a2[..., sample_dict["person"], :] * sample_dict["time"] + \
                        sample_dict["beta"] * sample_dict["time"] * sample_dict["treatment"]
                sample_dict["y"] = self.r_obs("y",
                                            dist.Normal(y_hat, 
                                                        repeat(sample_dict["sigma_y"], "b -> n b", n=sample_dict["N"])),
                                            obs=sample_dict.get("y", None))
        return sample_dict
    

class ARM_hiv_inter(BaseVAEwRegister):
    x_dim = 153
    theta_dim = 13

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_person": self.plate("plate_person", sample_dict["J"], dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }
    
    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["J"] = 5
                assert sample_dict["N"] % sample_dict["J"] == 0
                sample_dict["person"] = repeat(torch.arange(sample_dict["J"], device=self.device),
                                               "k -> (r k)", r=sample_dict["N"] // sample_dict["J"])
            with DataContext(sample_dict, self):
                sample_dict["time"] = dist.Uniform(10.0, 100.0).sample((sample_dict["N"], batch_size)).to(device=self.device)
                sample_dict["treatment"] = torch.randint(low=0, high=10,
                                                        size=(sample_dict["N"], batch_size),
                                                        device=self.device)
                sample_dict["sigma_a1"] = torch.ones(batch_size, device=self.device) * 10.0
                sample_dict["sigma_a2"] = torch.ones(batch_size, device=self.device) * 0.1
                sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a1"] = self.r_sample("mu_a1", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["mu_a2"] = self.r_sample("mu_a2", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["beta"] = self.r_sample("beta", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_person"]:
                sample_dict["a1"] = self.r_sample("a1", dist.Normal(10 * sample_dict["mu_a1"], sample_dict["sigma_a1"]))
                sample_dict["a2"] = self.r_sample("a2", dist.Normal(0.1 * sample_dict["mu_a2"], sample_dict["sigma_a2"]))
            with plates["plate_n"]:
                y_hat = sample_dict["a1"][..., sample_dict["person"], :] + \
                        sample_dict["a2"][..., sample_dict["person"], :] * sample_dict["time"] + \
                        sample_dict["beta"] * sample_dict["time"] * sample_dict["treatment"]
                sample_dict["y"] = self.r_obs("y",
                                            dist.Normal(y_hat, 
                                                        repeat(sample_dict["sigma_y"], "b -> n b", n=sample_dict["N"])),
                                            obs=sample_dict.get("y", None))
        return sample_dict


class ARM_hiv(BaseVAEwRegister):
    x_dim = 103
    theta_dim = 12

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_person": self.plate("plate_person", sample_dict["J"], dim=-2),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }
    
    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["J"] = 5
                assert sample_dict["N"] % sample_dict["J"] == 0
                sample_dict["person"] = repeat(torch.arange(sample_dict["J"], device=self.device),
                                               "k -> (r k)", r=sample_dict["N"] // sample_dict["J"])
            with DataContext(sample_dict, self):
                sample_dict["time"] = dist.Uniform(10.0, 100.0).sample((sample_dict["N"], batch_size)).to(device=self.device)
                sample_dict["sigma_a1"] = torch.ones(batch_size, device=self.device) * 1.0
                sample_dict["sigma_a2"] = torch.ones(batch_size, device=self.device) * 0.1
                sample_dict["sigma_y"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a1"] = self.r_sample("mu_a1", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["mu_a2"] = self.r_sample("mu_a2", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_person"]:
                sample_dict["a1"] = self.r_sample("a1", dist.Normal(sample_dict["mu_a1"], sample_dict["sigma_a1"]))
                sample_dict["a2"] = self.r_sample("a2", dist.Normal(0.1 * sample_dict["mu_a2"], sample_dict["sigma_a2"]))
            with plates["plate_n"]:
                y_hat = sample_dict["a1"][..., sample_dict["person"], :] + \
                        sample_dict["a2"][..., sample_dict["person"], :] * sample_dict["time"]
                sample_dict["y"] = self.r_obs("y",
                                            dist.Normal(y_hat, 
                                                        repeat(sample_dict["sigma_y"], "b -> n b", n=sample_dict["N"])),
                                            obs=sample_dict.get("y", None))
        return sample_dict


class ARM_ideo_interactions(BaseVAEwRegister):
    x_dim = 201
    theta_dim = 4

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["party"] = torch.randint(low=0, high=2,
                                                     size=(sample_dict["N"], batch_size), 
                                                     device=self.device)
                sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 10 + 50
                sample_dict["inter"] = sample_dict["x"] * sample_dict["party"]
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 4, device=self.device),
                                                                    torch.ones(batch_size, 4, device=self.device) * 10).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["party"] + \
                        sample_dict["beta"][..., 2] * sample_dict["x"] + \
                        sample_dict["beta"][..., 3] * sample_dict["inter"]
                sample_dict["score1"] = self.r_obs("score1",
                                                   dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                             "b -> n b", n=sample_dict["N"])),
                                                    obs=sample_dict.get("score1", None))
        return sample_dict


class ARM_ideo_reparam(BaseVAEwRegister):
    x_dim = 201
    theta_dim = 4

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["party"] = torch.randint(low=0, high=2,
                                                     size=(sample_dict["N"], batch_size), 
                                                     device=self.device)
                sample_dict["z1"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 20 + 200
                sample_dict["z2"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 10 + 500
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 4, device=self.device),
                                                                    torch.ones(batch_size, 4, device=self.device) * 10).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["party"] + \
                        sample_dict["beta"][..., 2] * sample_dict["z1"] + \
                        sample_dict["beta"][..., 3] * sample_dict["z2"]
                sample_dict["score1"] = self.r_obs("score1",
                                                   dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                             "b -> n b", n=sample_dict["N"])),
                                                    obs=sample_dict.get("score1", None))
        return sample_dict


class ARM_kidiq_interaction_c(BaseVAEwRegister):
    x_dim = 201
    theta_dim = 4

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                mom_hs = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 0.5
                mom_iq = torch.randint(low=80, high=120,
                                        size=(sample_dict["N"], batch_size),
                                        device=self.device).float()
                sample_dict["c_mom_hs"] = mom_hs - torch.mean(mom_hs, dim=0)
                sample_dict["c_mom_iq"] = mom_iq - torch.mean(mom_iq, dim=0)
                sample_dict["inter"] = sample_dict["c_mom_hs"] * sample_dict["c_mom_iq"]
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 4, device=self.device),
                                                                    torch.ones(batch_size, 4, device=self.device) * 10).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["c_mom_hs"] + \
                        sample_dict["beta"][..., 2] * sample_dict["c_mom_iq"] + \
                        sample_dict["beta"][..., 3] * sample_dict["inter"]
                sample_dict["kid_score"] = self.r_obs("kid_score",
                                                   dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                             "b -> n b", n=sample_dict["N"])),
                                                    obs=sample_dict.get("kid_score", None))
        return sample_dict
    

class ARM_kidiq_interaction_c2(BaseVAEwRegister):
    x_dim = 201
    theta_dim = 4

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                mom_hs = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 0.5
                mom_iq = torch.randint(low=80, high=120,
                                        size=(sample_dict["N"], batch_size),
                                        device=self.device).float()
                sample_dict["c2_mom_hs"] = mom_hs - 0.5
                sample_dict["c2_mom_iq"] = mom_iq - 100
                sample_dict["inter"] = sample_dict["c2_mom_hs"] * sample_dict["c2_mom_iq"]
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 4, device=self.device),
                                                                    torch.ones(batch_size, 4, device=self.device) * 10).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["c2_mom_hs"] + \
                        sample_dict["beta"][..., 2] * sample_dict["c2_mom_iq"] + \
                        sample_dict["beta"][..., 3] * sample_dict["inter"]
                sample_dict["kid_score"] = self.r_obs("kid_score",
                                                   dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                             "b -> n b", n=sample_dict["N"])),
                                                    obs=sample_dict.get("kid_score", None))
        return sample_dict


class ARM_kidiq_interaction_z(BaseVAEwRegister):
    x_dim = 201
    theta_dim = 4

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                mom_hs = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 0.5
                mom_iq = torch.randint(low=80, high=120,
                                        size=(sample_dict["N"], batch_size),
                                        device=self.device).float()
                sample_dict["z_mom_hs"] = (mom_hs - torch.mean(mom_hs, dim=0)) / (2 * torch.std(mom_hs, dim=0))
                sample_dict["z_mom_iq"] = (mom_iq - torch.mean(mom_iq, dim=0)) / (2 * torch.std(mom_iq, dim=0))
                sample_dict["inter"] = sample_dict["z_mom_hs"] * sample_dict["z_mom_iq"]
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 4, device=self.device),
                                                                    torch.ones(batch_size, 4, device=self.device) * 10).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["z_mom_hs"] + \
                        sample_dict["beta"][..., 2] * sample_dict["z_mom_iq"] + \
                        sample_dict["beta"][..., 3] * sample_dict["inter"]
                sample_dict["kid_score"] = self.r_obs("kid_score",
                                                   dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                             "b -> n b", n=sample_dict["N"])),
                                                    obs=sample_dict.get("kid_score", None))
        return sample_dict


class ARM_kidiq_interaction(BaseVAEwRegister):
    x_dim = 201
    theta_dim = 4

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["mom_hs"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 0.5
                sample_dict["mom_iq"] = torch.randint(low=80, high=120,
                                                        size=(sample_dict["N"], batch_size),
                                                        device=self.device).float()
                sample_dict["inter"] = sample_dict["mom_hs"] * sample_dict["mom_iq"]
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 4, device=self.device),
                                                                    torch.ones(batch_size, 4, device=self.device) * 10).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["mom_hs"] + \
                        sample_dict["beta"][..., 2] * sample_dict["mom_iq"] + \
                        sample_dict["beta"][..., 3] * sample_dict["inter"]
                sample_dict["kid_score"] = self.r_obs("kid_score",
                                                   dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                             "b -> n b", n=sample_dict["N"])),
                                                    obs=sample_dict.get("kid_score", None))
        return sample_dict
    

class ARM_kidiq_multi_preds(BaseVAEwRegister):
    x_dim = 151
    theta_dim = 3

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["mom_hs"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 0.5
                sample_dict["mom_iq"] = torch.randint(low=80, high=120,
                                                        size=(sample_dict["N"], batch_size),
                                                        device=self.device).float()
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 3, device=self.device),
                                                                    torch.ones(batch_size, 3, device=self.device) * 10).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["mom_hs"] + \
                        sample_dict["beta"][..., 2] * sample_dict["mom_iq"]
                sample_dict["kid_score"] = self.r_obs("kid_score",
                                                   dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                             "b -> n b", n=sample_dict["N"])),
                                                    obs=sample_dict.get("kid_score", None))
        return sample_dict


class ARM_kidscore_momhs(BaseVAEwRegister):
    x_dim = 101
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["mom_hs"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 0.5
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 2, device=self.device),
                                                                    torch.ones(batch_size, 2, device=self.device) * 10).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["mom_hs"]
                sample_dict["kid_score"] = self.r_obs("kid_score",
                                                   dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                             "b -> n b", n=sample_dict["N"])),
                                                    obs=sample_dict.get("kid_score", None))
        return sample_dict
    

class ARM_kidscore_momiq(BaseVAEwRegister):
    x_dim = 101
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["mom_iq"] = torch.randint(low=80, high=120,
                                                        size=(sample_dict["N"], batch_size),
                                                        device=self.device).float()
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 2, device=self.device),
                                                                    torch.ones(batch_size, 2, device=self.device) * 10).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["mom_iq"]
                sample_dict["kid_score"] = self.r_obs("kid_score",
                                                   dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                             "b -> n b", n=sample_dict["N"])),
                                                    obs=sample_dict.get("kid_score", None))
        return sample_dict


class ARM_kidscore_momwork(BaseVAEwRegister):
    x_dim = 201
    theta_dim = 4

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                mom_work = torch.randint(low=2, high=5,
                                        size=(sample_dict["N"], batch_size),
                                        device=self.device)
                sample_dict["work2"] = (mom_work == 2).float()
                sample_dict["work3"] = (mom_work == 3).float()
                sample_dict["work4"] = (mom_work == 4).float()
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 4, device=self.device),
                                                                    torch.ones(batch_size, 4, device=self.device) * 10).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["work2"] + \
                        sample_dict["beta"][..., 2] * sample_dict["work3"] + \
                        sample_dict["beta"][..., 3] * sample_dict["work4"]
                sample_dict["kid_score"] = self.r_obs("kid_score",
                                                   dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                             "b -> n b", n=sample_dict["N"])),
                                                    obs=sample_dict.get("kid_score", None))
        return sample_dict


class ARM_latent_glm(BaseVAEwRegister):
    x_dim = 720
    theta_dim = 22

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_age": self.plate("plate_age", sample_dict["n_age"], dim=-3),
            "plate_edu": self.plate("plate_edu", sample_dict["n_edu"], dim=-2),
            "plate_region": self.plate("plate_region", sample_dict["n_region"], dim=-2),
            "plate_state": self.plate("plate_state", sample_dict["n_state"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 240
                sample_dict["n_age"] = 3
                sample_dict["n_edu"] = 2
                sample_dict["n_region"] = 2
                sample_dict["n_state"] = 4
                total_permutate = sample_dict["n_state"] * sample_dict["n_age"] * sample_dict["n_edu"]
                assert sample_dict["N"] % total_permutate == 0
                age_edu_state_mesh = torch.stack(torch.meshgrid(
                    torch.arange(sample_dict["n_age"], device=self.device),
                    torch.arange(sample_dict["n_edu"], device=self.device),
                    torch.arange(sample_dict["n_state"], device=self.device),
                indexing="ij"), dim=-1)  # (3, 2, 4, 3)
                r = sample_dict["N"] // total_permutate
                sample_dict["age"] = repeat(age_edu_state_mesh[..., 0].reshape(-1), "k -> (r k)", r=r)
                sample_dict["edu"] = repeat(age_edu_state_mesh[..., 1].reshape(-1), "k -> (r k)", r=r)
                sample_dict["state"] = repeat(age_edu_state_mesh[..., 2].reshape(-1), "k -> (r k)", r=r)
                sample_dict["region"] = torch.randint(low=0, high=sample_dict["n_region"], 
                                                      size=(sample_dict["n_state"], ), 
                                                      generator=torch.Generator(device=self.device).manual_seed(1234),
                                                      device=self.device)
                sample_dict["v_prev"] = torch.arange(sample_dict["n_state"], device=self.device) + 1
            with DataContext(sample_dict, self):
                sample_dict["black"] = torch.randint(low=0, high=2,
                                                     size=(sample_dict["N"], batch_size),
                                                     device=self.device)
                sample_dict["female"] = torch.randint(low=0, high=2,
                                                      size=(sample_dict["N"], batch_size),
                                                      device=self.device)
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["b_0"] = self.r_sample("b_0", self.scalar_normal_dist(0.0, 100.0))
            sample_dict["b_female"] = self.r_sample("b_female", self.scalar_normal_dist(0.0, 100.0))
            sample_dict["b_black"] = self.r_sample("b_black", self.scalar_normal_dist(0.0, 100.0))
            sample_dict["b_female_black"] = self.r_sample("b_female_black", self.scalar_normal_dist(0.0, 100.0))
            sample_dict["b_v_prev"] = self.r_sample("b_v_prev", self.scalar_normal_dist(0.0, 100.0))
            with plates["plate_age"]:
                sample_dict["b_age"] = self.r_sample("b_age", self.scalar_normal_dist(0.0, 100.0))
                assert sample_dict["b_age"].shape[-2] == 1
            with plates["plate_edu"]:
                sample_dict["b_edu"] = self.r_sample("b_edu", self.scalar_normal_dist(0.0, 100.0))
            with plates["plate_region"]:
                sample_dict["b_region"] = self.r_sample("b_region", self.scalar_normal_dist(0.0, 100.0))
            with plates["plate_age"], plates["plate_edu"]:
                sample_dict["b_age_edu"] = self.r_sample("b_age_edu", self.scalar_normal_dist(0.0, 100.0))
            
            b_state_hat = sample_dict["b_region"][..., sample_dict["region"], :] + \
                        sample_dict["b_v_prev"] * repeat(sample_dict["v_prev"], "k -> k b", b=batch_size)  # (n_state, b)
            with plates["plate_state"]:
                sample_dict["b_state"] = self.r_sample("b_state", 
                                                       dist.Normal(b_state_hat, 
                                                                   torch.ones_like(b_state_hat) * 100.0))
            
            with plates["plate_n"]:
                Xbeta = sample_dict["b_0"] + \
                        sample_dict["b_female"] * sample_dict["female"] + \
                        sample_dict["b_black"] * sample_dict["black"] + \
                        sample_dict["b_female_black"] * sample_dict["female"] * sample_dict["black"] + \
                        sample_dict["b_age"].squeeze(-2)[..., sample_dict["age"], :] + \
                        sample_dict["b_edu"][..., sample_dict["edu"], :] + \
                        sample_dict["b_age_edu"][..., sample_dict["age"], sample_dict["edu"], :] + \
                        sample_dict["b_state"][..., sample_dict["state"], :]
                p = (1 / (1 + torch.exp(-Xbeta))).clamp(min=0.0, max=1.0)
                sample_dict["y"] = self.r_obs("y",
                                            dist.Bernoulli(p),
                                            obs=sample_dict.get("y", None))
        return sample_dict


class ARM_log10earn_height(BaseVAEwRegister):
    x_dim = 101
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["height"] = torch.randint(low=150, high=190,
                                                      size=(sample_dict["N"], batch_size),
                                                      device=self.device)
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 2, device=self.device),
                                                                    torch.ones(batch_size, 2, device=self.device) * 10).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["height"]
                sample_dict["log10_earn"] = self.r_obs("log10_earn",
                                                        dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                                    "b -> n b", n=sample_dict["N"])),
                                                        obs=sample_dict.get("log10_earn", None))
        return sample_dict


class ARM_logearn_height_male(BaseVAEwRegister):
    x_dim = 151
    theta_dim = 3

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["height"] = torch.randint(low=150, high=190,
                                                      size=(sample_dict["N"], batch_size),
                                                      device=self.device)
                sample_dict["male"] = torch.randint(low=0, high=2,
                                                    size=(sample_dict["N"], batch_size),
                                                    device=self.device)
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 3, device=self.device),
                                                                    torch.ones(batch_size, 3, device=self.device) * 10).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["height"] + \
                        sample_dict["beta"][..., 2] * sample_dict["male"]
                sample_dict["log_earn"] = self.r_obs("log_earn",
                                                        dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                                    "b -> n b", n=sample_dict["N"])),
                                                        obs=sample_dict.get("log_earn", None))
        return sample_dict


class ARM_logearn_height(BaseVAEwRegister):
    x_dim = 101
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["height"] = torch.randint(low=150, high=190,
                                                      size=(sample_dict["N"], batch_size),
                                                      device=self.device)
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 2, device=self.device),
                                                                    torch.ones(batch_size, 2, device=self.device) * 0.1).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["height"]
                sample_dict["log_earn"] = self.r_obs("log_earn",
                                                        dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                                    "b -> n b", n=sample_dict["N"])),
                                                        obs=sample_dict.get("log_earn", None))
        return sample_dict


class ARM_logearn_interaction_z(BaseVAEwRegister):
    x_dim = 201
    theta_dim = 4

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                height = torch.randint(low=150, high=190,
                                        size=(sample_dict["N"], batch_size),
                                        device=self.device).float()
                sample_dict["z_height"] = (height - torch.mean(height, dim=0)) / torch.std(height, dim=0)
                sample_dict["male"] = torch.randint(low=0, high=2,
                                                    size=(sample_dict["N"], batch_size),
                                                    device=self.device)
                sample_dict["inter"] = sample_dict["male"] * sample_dict["z_height"]
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 4, device=self.device),
                                                                    torch.ones(batch_size, 4, device=self.device) * 10).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["z_height"] + \
                        sample_dict["beta"][..., 2] * sample_dict["male"] + \
                        sample_dict["beta"][..., 3] * sample_dict["inter"]
                sample_dict["log_earn"] = self.r_obs("log_earn",
                                                        dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                                    "b -> n b", n=sample_dict["N"])),
                                                        obs=sample_dict.get("log_earn", None))
        return sample_dict


class ARM_logearn_interaction(BaseVAEwRegister):
    x_dim = 201
    theta_dim = 4

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["height"] = torch.randint(low=150, high=190,
                                                    size=(sample_dict["N"], batch_size),
                                                    device=self.device)
                sample_dict["male"] = torch.randint(low=0, high=2,
                                                    size=(sample_dict["N"], batch_size),
                                                    device=self.device)
                sample_dict["inter"] = sample_dict["male"] * sample_dict["height"]
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 4, device=self.device),
                                                                    torch.ones(batch_size, 4, device=self.device) * 10).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["height"] + \
                        sample_dict["beta"][..., 2] * sample_dict["male"] + \
                        sample_dict["beta"][..., 3] * sample_dict["inter"]
                sample_dict["log_earn"] = self.r_obs("log_earn",
                                                        dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                                    "b -> n b", n=sample_dict["N"])),
                                                        obs=sample_dict.get("log_earn", None))
        return sample_dict


class ARM_logearn_logheight(BaseVAEwRegister):
    x_dim = 151
    theta_dim = 3

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                height = torch.randint(low=150, high=190,
                                        size=(sample_dict["N"], batch_size),
                                        device=self.device)
                sample_dict["log_height"] = torch.log(height)
                sample_dict["male"] = torch.randint(low=0, high=2,
                                                    size=(sample_dict["N"], batch_size),
                                                    device=self.device)
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 3, device=self.device),
                                                                    torch.ones(batch_size, 3, device=self.device) * 0.1).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["log_height"] + \
                        sample_dict["beta"][..., 2] * sample_dict["male"]
                sample_dict["log_earn"] = self.r_obs("log_earn",
                                                        dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                                    "b -> n b", n=sample_dict["N"])),
                                                        obs=sample_dict.get("log_earn", None))
        return sample_dict


class ARM_mesquite_log(BaseVAEwRegister):
    x_dim = 351
    theta_dim = 7

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                diam1 = torch.randint(low=100, high=200,
                                      size=(sample_dict["N"], batch_size),
                                      device=self.device)
                diam2 = torch.randint(low=100, high=200,
                                      size=(sample_dict["N"], batch_size),
                                      device=self.device)
                canopy_height = torch.randint(low=50, high=100,
                                      size=(sample_dict["N"], batch_size),
                                      device=self.device)
                total_height = canopy_height * (6 + torch.randn(sample_dict["N"], batch_size, device=self.device) * 0.1)
                density = torch.randint(low=10, high=200,
                                        size=(sample_dict["N"], batch_size),
                                        device=self.device)
                sample_dict["log_diam1"] = torch.log(diam1)
                sample_dict["log_diam2"] = torch.log(diam2)
                sample_dict["log_canopy_height"] = torch.log(canopy_height)
                sample_dict["log_total_height"] = torch.log(total_height)
                sample_dict["log_density"] = torch.log(density)
                sample_dict["group"] = torch.randint(low=0, high=2,
                                                    size=(sample_dict["N"], batch_size),
                                                    device=self.device)
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 7, device=self.device),
                                                                    torch.ones(batch_size, 7, device=self.device) * 10.0).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["log_diam1"] + \
                        sample_dict["beta"][..., 2] * sample_dict["log_diam2"] + \
                        sample_dict["beta"][..., 3] * sample_dict["log_canopy_height"] + \
                        sample_dict["beta"][..., 4] * sample_dict["log_total_height"] + \
                        sample_dict["beta"][..., 5] * sample_dict["log_density"] + \
                        sample_dict["beta"][..., 6] * sample_dict["group"]
                sample_dict["log_weight"] = self.r_obs("log_weight",
                                                        dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                                    "b -> n b", n=sample_dict["N"])),
                                                        obs=sample_dict.get("log_weight", None))
        return sample_dict


class ARM_mesquite_va(BaseVAEwRegister):
    x_dim = 201
    theta_dim = 4

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                diam1 = torch.randint(low=100, high=200,
                                      size=(sample_dict["N"], batch_size),
                                      device=self.device)
                diam2 = torch.randint(low=100, high=200,
                                      size=(sample_dict["N"], batch_size),
                                      device=self.device)
                canopy_height = torch.randint(low=50, high=100,
                                      size=(sample_dict["N"], batch_size),
                                      device=self.device)
                sample_dict["log_canopy_volume"] = torch.log(diam1 * diam2 * canopy_height)
                sample_dict["log_canopy_area"] = torch.log(diam1 * diam2)
                sample_dict["group"] = torch.randint(low=0, high=2,
                                                    size=(sample_dict["N"], batch_size),
                                                    device=self.device)
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 4, device=self.device),
                                                                    torch.ones(batch_size, 4, device=self.device) * 10.0).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["log_canopy_volume"] + \
                        sample_dict["beta"][..., 2] * sample_dict["log_canopy_area"] + \
                        sample_dict["beta"][..., 3] * sample_dict["group"]
                sample_dict["log_weight"] = self.r_obs("log_weight",
                                                        dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                                    "b -> n b", n=sample_dict["N"])),
                                                        obs=sample_dict.get("log_weight", None))
        return sample_dict


class ARM_mesquite_vas(BaseVAEwRegister):
    x_dim = 351
    theta_dim = 7

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                diam1 = torch.randint(low=100, high=200,
                                      size=(sample_dict["N"], batch_size),
                                      device=self.device)
                diam2 = torch.randint(low=100, high=200,
                                      size=(sample_dict["N"], batch_size),
                                      device=self.device)
                canopy_height = torch.randint(low=50, high=100,
                                      size=(sample_dict["N"], batch_size),
                                      device=self.device)
                total_height = canopy_height * (6 + torch.randn(sample_dict["N"], batch_size, device=self.device) * 0.1)
                density = torch.randint(low=10, high=200,
                                        size=(sample_dict["N"], batch_size),
                                        device=self.device)
                sample_dict["log_canopy_volume"] = torch.log(diam1 * diam2 * canopy_height)
                sample_dict["log_canopy_area"] = torch.log(diam1 * diam2)
                sample_dict["log_canopy_shape"] = torch.log(diam1 / diam2)
                sample_dict["log_total_height"] = torch.log(total_height)
                sample_dict["log_density"] = torch.log(density)
                sample_dict["group"] = torch.randint(low=0, high=2,
                                                    size=(sample_dict["N"], batch_size),
                                                    device=self.device)
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 7, device=self.device),
                                                                    torch.ones(batch_size, 7, device=self.device) * 10.0).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["log_canopy_volume"] + \
                        sample_dict["beta"][..., 2] * sample_dict["log_canopy_area"] + \
                        sample_dict["beta"][..., 3] * sample_dict["log_canopy_shape"] + \
                        sample_dict["beta"][..., 4] * sample_dict["log_total_height"] + \
                        sample_dict["beta"][..., 5] * sample_dict["log_density"] + \
                        sample_dict["beta"][..., 6] * sample_dict["group"]
                sample_dict["log_weight"] = self.r_obs("log_weight",
                                                        dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                                    "b -> n b", n=sample_dict["N"])),
                                                        obs=sample_dict.get("log_weight", None))
        return sample_dict


class ARM_mesquite_vash(BaseVAEwRegister):
    x_dim = 301
    theta_dim = 6

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                diam1 = torch.randint(low=100, high=200,
                                      size=(sample_dict["N"], batch_size),
                                      device=self.device)
                diam2 = torch.randint(low=100, high=200,
                                      size=(sample_dict["N"], batch_size),
                                      device=self.device)
                canopy_height = torch.randint(low=50, high=100,
                                      size=(sample_dict["N"], batch_size),
                                      device=self.device)
                total_height = canopy_height * (6 + torch.randn(sample_dict["N"], batch_size, device=self.device) * 0.1)
                sample_dict["log_canopy_volume"] = torch.log(diam1 * diam2 * canopy_height)
                sample_dict["log_canopy_area"] = torch.log(diam1 * diam2)
                sample_dict["log_canopy_shape"] = torch.log(diam1 / diam2)
                sample_dict["log_total_height"] = torch.log(total_height)
                sample_dict["group"] = torch.randint(low=0, high=2,
                                                    size=(sample_dict["N"], batch_size),
                                                    device=self.device)
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 6, device=self.device),
                                                                    torch.ones(batch_size, 6, device=self.device) * 10.0).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["log_canopy_volume"] + \
                        sample_dict["beta"][..., 2] * sample_dict["log_canopy_area"] + \
                        sample_dict["beta"][..., 3] * sample_dict["log_canopy_shape"] + \
                        sample_dict["beta"][..., 4] * sample_dict["log_total_height"] + \
                        sample_dict["beta"][..., 5] * sample_dict["group"]
                sample_dict["log_weight"] = self.r_obs("log_weight",
                                                        dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                                    "b -> n b", n=sample_dict["N"])),
                                                        obs=sample_dict.get("log_weight", None))
        return sample_dict


class ARM_mesquite_volume(BaseVAEwRegister):
    x_dim = 101
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                diam1 = torch.randint(low=100, high=200,
                                      size=(sample_dict["N"], batch_size),
                                      device=self.device)
                diam2 = torch.randint(low=100, high=200,
                                      size=(sample_dict["N"], batch_size),
                                      device=self.device)
                canopy_height = torch.randint(low=50, high=100,
                                      size=(sample_dict["N"], batch_size),
                                      device=self.device)
                sample_dict["log_canopy_volume"] = torch.log(diam1 * diam2 * canopy_height)
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 2, device=self.device),
                                                                    torch.ones(batch_size, 2, device=self.device) * 10.0).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["log_canopy_volume"]
                sample_dict["log_weight"] = self.r_obs("log_weight",
                                                        dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                                    "b -> n b", n=sample_dict["N"])),
                                                        obs=sample_dict.get("log_weight", None))
        return sample_dict


class ARM_mesquite(BaseVAEwRegister):
    x_dim = 351
    theta_dim = 7

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["diam1"] = torch.randint(low=100, high=200,
                                                    size=(sample_dict["N"], batch_size),
                                                    device=self.device)
                sample_dict["diam2"] = torch.randint(low=100, high=200,
                                                    size=(sample_dict["N"], batch_size),
                                                    device=self.device)
                sample_dict["canopy_height"] = torch.randint(low=50, high=100,
                                                            size=(sample_dict["N"], batch_size),
                                                            device=self.device)
                sample_dict["total_height"] = sample_dict["canopy_height"] * (6 + torch.randn(sample_dict["N"], batch_size, device=self.device) * 0.1)
                sample_dict["density"] = torch.randint(low=10, high=200,
                                                        size=(sample_dict["N"], batch_size),
                                                        device=self.device)
                sample_dict["group"] = torch.randint(low=0, high=2,
                                                    size=(sample_dict["N"], batch_size),
                                                    device=self.device)
                sample_dict["sigma"] = torch.ones(batch_size, device=self.device) * 0.1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros(batch_size, 7, device=self.device),
                                                                    torch.ones(batch_size, 7, device=self.device) * 10.0).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["diam1"] + \
                        sample_dict["beta"][..., 2] * sample_dict["diam2"] + \
                        sample_dict["beta"][..., 3] * sample_dict["canopy_height"] + \
                        sample_dict["beta"][..., 4] * sample_dict["total_height"] + \
                        sample_dict["beta"][..., 5] * sample_dict["density"] + \
                        sample_dict["beta"][..., 6] * sample_dict["group"]
                sample_dict["weight"] = self.r_obs("weight",
                                                    dist.Normal(y_hat, repeat(sample_dict["sigma"],
                                                                                "b -> n b", n=sample_dict["N"])),
                                                    obs=sample_dict.get("weight", None))
        return sample_dict


class ARM_multilevel_logistic(ARM_latent_glm):
    pass


class ARM_pilots_ch13(BaseVAEwRegister):
    x_dim = 75
    theta_dim = 9

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_group": self.plate("plate_group", sample_dict["n_groups"], dim=-2),
            "plate_scenario": self.plate("plate_scenario", sample_dict["n_scenarios"], dim=-2)
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 75
                sample_dict["n_groups"] = 3
                sample_dict["n_scenarios"] = 5
                total_permutate = sample_dict["n_groups"] * sample_dict["n_scenarios"]
                assert sample_dict["N"] % total_permutate == 0
                group_scenario_mesh = torch.stack(
                    torch.meshgrid(
                        torch.arange(sample_dict["n_groups"], device=self.device),
                        torch.arange(sample_dict["n_scenarios"], device=self.device),
                        indexing="ij"
                    ), 
                dim=-1)  # (3, 5, 2)
                r = sample_dict["N"] // total_permutate
                sample_dict["group_id"] = repeat(group_scenario_mesh[..., 0].reshape(-1),
                                                 "k -> (r k)", r=r)
                sample_dict["scenario_id"] = repeat(group_scenario_mesh[..., 1].reshape(-1),
                                                    "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                pass
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu"] = self.r_sample("mu", dist.Uniform(torch.tensor(-100.0, device=self.device), 
                                                                 torch.tensor(100.0, device=self.device)))
            with plates["plate_group"]:
                sample_dict["gamma"] = self.r_sample("gamma", self.scalar_normal_dist(0.0, 100.0))
            with plates["plate_scenario"]:
                sample_dict["delta"] = self.r_sample("delta", self.scalar_normal_dist(0.0, 100.0))
            with plates["plate_n"]:
                y_hat = sample_dict["mu"] + \
                        sample_dict["gamma"][..., sample_dict["group_id"], :] + \
                        sample_dict["delta"][..., sample_dict["scenario_id"], :]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_pilots_ch14(BaseVAEwRegister):
    x_dim = 75
    theta_dim = 10

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_group": self.plate("plate_group", sample_dict["n_groups"], dim=-2),
            "plate_scenario": self.plate("plate_scenario", sample_dict["n_scenarios"], dim=-2)
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 75
                sample_dict["n_groups"] = 3
                sample_dict["n_scenarios"] = 5
                total_permutate = sample_dict["n_groups"] * sample_dict["n_scenarios"]
                assert sample_dict["N"] % total_permutate == 0
                group_scenario_mesh = torch.stack(
                    torch.meshgrid(
                        torch.arange(sample_dict["n_groups"], device=self.device),
                        torch.arange(sample_dict["n_scenarios"], device=self.device),
                        indexing="ij"
                    ), 
                dim=-1)  # (3, 5, 2)
                r = sample_dict["N"] // total_permutate
                sample_dict["group_id"] = repeat(group_scenario_mesh[..., 0].reshape(-1),
                                                 "k -> (r k)", r=r)
                sample_dict["scenario_id"] = repeat(group_scenario_mesh[..., 1].reshape(-1),
                                                    "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                pass
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["mu_b"] = self.r_sample("mu_b", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_group"]:
                sample_dict["a"] = self.r_sample("a", dist.Normal(10 * sample_dict["mu_a"],
                                                                  torch.ones_like(sample_dict["mu_a"]) * 10))
            with plates["plate_scenario"]:
                sample_dict["b"] = self.r_sample("b", dist.Normal(10 * sample_dict["mu_b"],
                                                                  torch.ones_like(sample_dict["mu_b"]) * 10))
            with plates["plate_n"]:
                y_hat = sample_dict["a"][..., sample_dict["group_id"], :] + \
                        sample_dict["b"][..., sample_dict["scenario_id"], :]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_pilots_chr_ch13(BaseVAEwRegister):
    x_dim = 75
    theta_dim = 10

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 75
                sample_dict["n_groups"] = 3
                sample_dict["n_scenarios"] = 5
                total_permutate = sample_dict["n_groups"] * sample_dict["n_scenarios"]
                assert sample_dict["N"] % total_permutate == 0
                group_scenario_mesh = torch.stack(
                    torch.meshgrid(
                        torch.arange(sample_dict["n_groups"], device=self.device),
                        torch.arange(sample_dict["n_scenarios"], device=self.device),
                        indexing="ij"
                    ), 
                dim=-1)  # (3, 5, 2)
                r = sample_dict["N"] // total_permutate
                sample_dict["group_id"] = repeat(group_scenario_mesh[..., 0].reshape(-1),
                                                 "k -> (r k)", r=r)
                sample_dict["scenario_id"] = repeat(group_scenario_mesh[..., 1].reshape(-1),
                                                    "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                pass
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["mu_b"] = self.r_sample("mu_b", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["eta_a"] = self.r_sample("eta_a", dist.Normal(torch.zeros(sample_dict["n_groups"], device=self.device),
                                                                      torch.ones(sample_dict["n_groups"], device=self.device)).to_event(1))
            sample_dict["eta_b"] = self.r_sample("eta_b", dist.Normal(torch.zeros(sample_dict["n_scenarios"], device=self.device),
                                                                      torch.ones(sample_dict["n_scenarios"], device=self.device)).to_event(1))
            with plates["plate_n"]:
                a = 0.1 * sample_dict["mu_a"] + rearrange(sample_dict["eta_a"], "b k -> k b") * 0.05
                b = 0.1 * sample_dict["mu_b"] + rearrange(sample_dict["eta_b"], "b k -> k b") * 0.05
                y_hat = a[..., sample_dict["group_id"], :] + \
                        b[..., sample_dict["scenario_id"], :]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_pilots_chr_ch14(BaseVAEwRegister):
    x_dim = 75
    theta_dim = 10

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_group": self.plate("plate_group", sample_dict["n_groups"], dim=-2),
            "plate_scenario": self.plate("plate_scenario", sample_dict["n_scenarios"], dim=-2)
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 75
                sample_dict["n_groups"] = 3
                sample_dict["n_scenarios"] = 5
                total_permutate = sample_dict["n_groups"] * sample_dict["n_scenarios"]
                assert sample_dict["N"] % total_permutate == 0
                group_scenario_mesh = torch.stack(
                    torch.meshgrid(
                        torch.arange(sample_dict["n_groups"], device=self.device),
                        torch.arange(sample_dict["n_scenarios"], device=self.device),
                        indexing="ij"
                    ), 
                dim=-1)  # (3, 5, 2)
                r = sample_dict["N"] // total_permutate
                sample_dict["group_id"] = repeat(group_scenario_mesh[..., 0].reshape(-1),
                                                 "k -> (r k)", r=r)
                sample_dict["scenario_id"] = repeat(group_scenario_mesh[..., 1].reshape(-1),
                                                    "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                pass
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["mu_b"] = self.r_sample("mu_b", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_group"]:
                sample_dict["eta_a"] = self.r_sample("eta_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_scenario"]:
                sample_dict["eta_b"] = self.r_sample("eta_b", self.scalar_normal_dist(0.0, 1.0))
            a = 10 * sample_dict["mu_a"] + sample_dict["eta_a"] * 3
            b = 10 * sample_dict["mu_b"] + sample_dict["eta_b"] * 3
            with plates["plate_n"]:
                y_hat = a[..., sample_dict["group_id"], :] + \
                        b[..., sample_dict["scenario_id"], :]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_chr(BaseVAEwRegister):
    x_dim = 50
    theta_dim = 6

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2)
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                pass
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_eta"] = self.r_sample("mu_eta", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["et"] = self.r_sample("et", self.scalar_normal_dist(0.0, 1.0))
                eta = 0.1 * sample_dict["mu_eta"] + 0.05 * sample_dict["et"]
            with plates["plate_n"]:
                y_hat = eta[..., sample_dict["county"], :]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_complete_pool(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 10 
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((2, ), device=self.device),
                                                                    torch.ones((2, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + sample_dict["beta"][..., 1] * sample_dict["x"]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_group_chr(BaseVAEwRegister):
    x_dim = 150
    theta_dim = 12

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
            "plate_beta": self.plate("plate_beta", 2, dim=-2)
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 10 
                sample_dict["u"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 10 + 100
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            with plates["plate_county"]:
                sample_dict["mu_b"] = self.r_sample("mu_b", self.scalar_normal_dist(0.0, 1.0))
                sample_dict["eta"] = self.r_sample("eta", self.scalar_normal_dist(0.0, 1.0))
            b = sample_dict["mu_b"] + 0.5 * sample_dict["eta"]
            with plates["plate_beta"]:
                sample_dict["beta"] = self.r_sample("beta", self.scalar_normal_dist(0.0, 100.0))
            with plates["plate_n"]:
                y_hat = b[..., sample_dict["county"], :] + \
                        sample_dict["x"] * sample_dict["beta"][..., 0, :] + \
                        sample_dict["u"] * sample_dict["beta"][..., 1, :]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_group(BaseVAEwRegister):
    x_dim = 150
    theta_dim = 9

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
            "plate_beta": self.plate("plate_beta", 2, dim=-2)
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 10 
                sample_dict["u"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 10 + 100
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_alpha"] = self.r_sample("mu_alpha", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["mu_beta"] = self.r_sample("mu_beta", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["alpha"] = self.r_sample("alpha", dist.Normal(sample_dict["mu_alpha"],
                                                                          torch.ones_like(sample_dict["mu_alpha"]) * 0.5))
            with plates["plate_beta"]:
                sample_dict["beta"] = self.r_sample("beta", dist.Normal(sample_dict["mu_beta"],
                                                                        torch.ones_like(sample_dict["mu_beta"]) * 0.5))
            with plates["plate_n"]:
                y_hat = sample_dict["alpha"][..., sample_dict["county"], :] + \
                        sample_dict["x"] * sample_dict["beta"][..., 0, :] + \
                        sample_dict["u"] * sample_dict["beta"][..., 1, :]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_inter_vary_chr(BaseVAEwRegister):
    x_dim = 200
    theta_dim = 14

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
            "plate_beta": self.plate("plate_beta", 2, dim=-2)
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 10 
                sample_dict["u"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 10 + 100
                sample_dict["inter"] = sample_dict["x"] * sample_dict["u"]
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            with plates["plate_beta"]:
                sample_dict["beta"] = self.r_sample("beta", self.scalar_normal_dist(0.0, 100.0))
            sample_dict["mu_a1"] = self.r_sample("mu_a1", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["mu_a2"] = self.r_sample("mu_a2", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["eta1"] = self.r_sample("eta1", self.scalar_normal_dist(0.0, 1.0))
                sample_dict["eta2"] = self.r_sample("eta2", self.scalar_normal_dist(0.0, 1.0))
            a1 = sample_dict["mu_a1"] + 0.5 * sample_dict["eta1"]
            a2 = 0.1 * sample_dict["mu_a2"] + 0.05 * sample_dict["eta2"]
            with plates["plate_n"]:
                y_hat = a1[..., sample_dict["county"], :] + \
                        a2[..., sample_dict["county"], :] * sample_dict["x"] + \
                        sample_dict["u"] * sample_dict["beta"][..., 0, :] + \
                        sample_dict["inter"] * sample_dict["beta"][..., 1, :]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_inter_vary(BaseVAEwRegister):
    x_dim = 200
    theta_dim = 15

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
            "plate_beta": self.plate("plate_beta", 2, dim=-2)
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 10 
                sample_dict["u"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 10 + 100
                sample_dict["inter"] = sample_dict["x"] * sample_dict["u"]
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_beta"] = self.r_sample("mu_beta", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_beta"]:
                sample_dict["beta"] = self.r_sample("beta", dist.Normal(100 * sample_dict["mu_beta"],
                                                                        torch.ones_like(sample_dict["mu_beta"]) * 10))
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["mu_b"] = self.r_sample("mu_b", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["a"] = self.r_sample("a", dist.Normal(sample_dict["mu_a"], 
                                                                  torch.ones_like(sample_dict["mu_a"]) * 0.5))
                sample_dict["b"] = self.r_sample("b", dist.Normal(0.1 * sample_dict["mu_b"], 
                                                                  torch.ones_like(sample_dict["mu_b"]) * 0.05))
            with plates["plate_n"]:
                y_hat = sample_dict["a"][..., sample_dict["county"], :] + \
                        sample_dict["b"][..., sample_dict["county"], :] * sample_dict["x"] + \
                        sample_dict["u"] * sample_dict["beta"][..., 0, :] + \
                        sample_dict["inter"] * sample_dict["beta"][..., 1, :]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_intercept_chr(BaseVAEwRegister):
    x_dim = 50
    theta_dim = 6

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                pass
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["eta"] = self.r_sample("eta", self.scalar_normal_dist(0.0, 1.0))
            a = 10 * sample_dict["mu_a"] + 2 * sample_dict["eta"]
            with plates["plate_n"]:
                y_hat = a[..., sample_dict["county"], :]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_intercept(BaseVAEwRegister):
    x_dim = 50
    theta_dim = 6

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                pass
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["a"] = self.r_sample("a", dist.Normal(sample_dict["mu_a"],
                                                                  torch.ones_like(sample_dict["mu_a"]) * 0.5))
            with plates["plate_n"]:
                y_hat = sample_dict["a"][..., sample_dict["county"], :]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_no_pool_chr(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 7

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 10
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["eta"] = self.r_sample("eta", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["beta"] = self.r_sample("beta", self.scalar_normal_dist(0.0, 1.0))
            a = sample_dict["mu_a"] + 0.5 * sample_dict["eta"]
            with plates["plate_n"]:
                y_hat = a[..., sample_dict["county"], :] + \
                        sample_dict["beta"] * sample_dict["x"]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_no_pool(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 7

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 10
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["a"] = self.r_sample("a", dist.Normal(sample_dict["mu_a"],
                                                                  torch.ones_like(sample_dict["mu_a"]) * 0.5))
            sample_dict["beta"] = self.r_sample("beta", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_n"]:
                y_hat = sample_dict["a"][..., sample_dict["county"], :] + \
                        sample_dict["beta"] * sample_dict["x"]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_redundant_chr(BaseVAEwRegister):
    x_dim = 50
    theta_dim = 6

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                pass
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_eta"] = self.r_sample("mu_eta", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["eta"] = self.r_sample("eta", self.scalar_normal_dist(0.0, 3.0))
                a = 100 * sample_dict["mu_eta"] + 3 * sample_dict["eta"]
            with plates["plate_n"]:
                y_hat = a[..., sample_dict["county"], :] + \
                        100 * sample_dict["mu_eta"]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_redundant(BaseVAEwRegister):
    x_dim = 50
    theta_dim = 6

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                pass
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu"] = self.r_sample("mu", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["eta"] = self.r_sample("eta", self.scalar_normal_dist(0.0, 10.0))
            with plates["plate_n"]:
                y_hat = sample_dict["eta"][..., sample_dict["county"], :] + \
                        100 * sample_dict["mu"]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_vary_intercept_floor_chr(BaseVAEwRegister):
    x_dim = 150
    theta_dim = 8

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
            "plate_b": self.plate("plate_b", 2, dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 10 
                sample_dict["u"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 10 + 100
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["eta"] = self.r_sample("eta", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_b"]:
                sample_dict["b"] = self.r_sample("b", self.scalar_normal_dist(0.0, 1.0))
            a = sample_dict["mu_a"] + 0.5 * sample_dict["eta"]
            with plates["plate_n"]:
                y_hat = a[..., sample_dict["county"], :] + \
                        sample_dict["u"] * sample_dict["b"][..., 0, :] + \
                        sample_dict["x"] * sample_dict["b"][..., 1, :]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_vary_intercept_floor(BaseVAEwRegister):
    x_dim = 150
    theta_dim = 8

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
            "plate_b": self.plate("plate_b", 2, dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 10 
                sample_dict["u"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 10 + 100
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["a"] = self.r_sample("a", dist.Normal(sample_dict["mu_a"],
                                                                  torch.ones_like(sample_dict["mu_a"]) * 0.5))
            with plates["plate_b"]:
                sample_dict["b"] = self.r_sample("b", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_n"]:
                y_hat = sample_dict["a"][..., sample_dict["county"], :] + \
                        sample_dict["u"] * sample_dict["b"][..., 0, :] + \
                        sample_dict["x"] * sample_dict["b"][..., 1, :]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict
    

class ARM_radon_vary_intercept_floor2_chr(BaseVAEwRegister):
    x_dim = 151
    theta_dim = 9

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
            "plate_b": self.plate("plate_b", 3, dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 10 
                sample_dict["x_mean"] = torch.mean(sample_dict["x"], dim=-2)
                sample_dict["u"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 10 + 100
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["eta"] = self.r_sample("eta", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_b"]:
                sample_dict["b"] = self.r_sample("b", self.scalar_normal_dist(0.0, 1.0))
            a = sample_dict["mu_a"] + 0.5 * sample_dict["eta"]
            with plates["plate_n"]:
                y_hat = a[..., sample_dict["county"], :] + \
                        sample_dict["u"] * sample_dict["b"][..., 0, :] + \
                        sample_dict["x"] * sample_dict["b"][..., 1, :] + \
                        sample_dict["x_mean"] * sample_dict["b"][..., 2, :]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_vary_intercept_floor2(BaseVAEwRegister):
    x_dim = 151
    theta_dim = 9

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
            "plate_b": self.plate("plate_b", 3, dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 10 
                sample_dict["x_mean"] = torch.mean(sample_dict["x"], dim=-2)
                sample_dict["u"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 10 + 100
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["a"] = self.r_sample("a", dist.Normal(sample_dict["mu_a"],
                                                                  torch.ones_like(sample_dict["mu_a"]) * 0.5))
            with plates["plate_b"]:
                sample_dict["b"] = self.r_sample("b", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_n"]:
                y_hat = sample_dict["a"][..., sample_dict["county"], :] + \
                        sample_dict["u"] * sample_dict["b"][..., 0, :] + \
                        sample_dict["x"] * sample_dict["b"][..., 1, :] + \
                        sample_dict["x_mean"] * sample_dict["b"][..., 2, :]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_vary_intercept_nofloor_chr(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 7

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                sample_dict["u"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 10 + 100
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["eta"] = self.r_sample("eta", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["b"] = self.r_sample("b", self.scalar_normal_dist(0.0, 1.0))
            a = sample_dict["mu_a"] + 0.5 * sample_dict["eta"]
            with plates["plate_n"]:
                y_hat = a[..., sample_dict["county"], :] + \
                        sample_dict["u"] * sample_dict["b"] * 0.1
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_vary_intercept_nofloor(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 7

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                sample_dict["u"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 10 + 100
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a"] = self.r_sample("mu_a", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["a"] = self.r_sample("a", dist.Normal(sample_dict["mu_a"],
                                                                  torch.ones_like(sample_dict["mu_a"]) * 0.5))
            sample_dict["b"] = self.r_sample("b", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_n"]:
                y_hat = sample_dict["a"][..., sample_dict["county"], :] + \
                        sample_dict["u"] * sample_dict["b"] * 0.1
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_vary_si_chr(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 12

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 10 
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a1"] = self.r_sample("mu_a1", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["mu_a2"] = self.r_sample("mu_a2", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["eta1"] = self.r_sample("eta1", self.scalar_normal_dist(0.0, 1.0))
                sample_dict["eta2"] = self.r_sample("eta2", self.scalar_normal_dist(0.0, 1.0))
            a1 = sample_dict["mu_a1"] + 0.4 * sample_dict["eta1"]
            a2 = 0.1 * sample_dict["mu_a2"] + 0.05 * sample_dict["eta2"]
            with plates["plate_n"]:
                y_hat = a1[..., sample_dict["county"], :] + \
                        a2[..., sample_dict["county"], :] * sample_dict["x"]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon_vary_si(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 12

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 10 
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu_a1"] = self.r_sample("mu_a1", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["mu_a2"] = self.r_sample("mu_a2", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["a1"] = self.r_sample("a1", dist.Normal(sample_dict["mu_a1"],
                                                                    torch.ones_like(sample_dict["mu_a1"]) * 0.5))
                sample_dict["a2"] = self.r_sample("a2", dist.Normal(sample_dict["mu_a2"],
                                                                    torch.ones_like(sample_dict["mu_a2"]) * 0.5))
            with plates["plate_n"]:
                y_hat = sample_dict["a1"][..., sample_dict["county"], :] + \
                        sample_dict["a2"][..., sample_dict["county"], :] * sample_dict["x"]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_radon(BaseVAEwRegister):
    x_dim = 50
    theta_dim = 6

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_county": self.plate("plate_county", sample_dict["n_county"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_county"] = 5
                assert sample_dict["N"] % sample_dict["n_county"] == 0
                r = sample_dict["N"] // sample_dict["n_county"]
                sample_dict["county"] = repeat(torch.arange(sample_dict["n_county"], device=self.device),
                                                 "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                pass
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["mu"] = self.r_sample("mu", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_county"]:
                sample_dict["eta"] = self.r_sample("eta", self.scalar_normal_dist(0.0, 3.0))
            with plates["plate_n"]:
                y_hat = sample_dict["eta"][..., sample_dict["county"], :] + \
                        0.1 * sample_dict["mu"]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_roaches_overdispersion(BaseVAEwRegister):
    x_dim = 250
    theta_dim = 54

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["roach1"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 5 + 20
                sample_dict["treatment"] = torch.randint(low=0, high=2, 
                                                         size=(sample_dict["N"], batch_size),
                                                         device=self.device)
                sample_dict["senior"] = torch.randint(low=0, high=2, 
                                                    size=(sample_dict["N"], batch_size),
                                                    device=self.device)
                sample_dict["log_expo"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 0.25
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((4, ), device=self.device),
                                                                    torch.ones((4, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                sample_dict["lambda"] = self.r_sample("lambda", self.scalar_normal_dist(0.0, 0.1))
                log_rate = sample_dict["lambda"] + \
                           sample_dict["log_expo"] + \
                           sample_dict["beta"][..., 0] + \
                           sample_dict["beta"][..., 1] * sample_dict["roach1"] + \
                           sample_dict["beta"][..., 2] * sample_dict["senior"] + \
                           sample_dict["beta"][..., 3] * sample_dict["treatment"]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Poisson(rate=log_rate.exp() + 1e-8),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_separation(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 5 + 30
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((2, ), device=self.device),
                                                                    torch.ones((2, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                logits = sample_dict["beta"][..., 0] + \
                         sample_dict["beta"][..., 1] * sample_dict["x"]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Bernoulli(logits=logits),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_sesame_multi_preds_3a(BaseVAEwRegister):
    x_dim = 400
    theta_dim = 8

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["encouraged"] = torch.randint(low=0, high=1,
                                                          size=(sample_dict["N"], batch_size),
                                                          device=self.device)
                sample_dict["setting"] = torch.randint(low=0, high=1,
                                                          size=(sample_dict["N"], batch_size),
                                                          device=self.device)
                sample_dict["pretest"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 2 + 10
                site = torch.randint(low=2, high=6,
                                    size=(sample_dict["N"], batch_size),
                                    device=self.device)
                sample_dict["site2"] = (site == 2).float()
                sample_dict["site3"] = (site == 3).float()
                sample_dict["site4"] = (site == 4).float()
                sample_dict["site5"] = (site == 5).float()
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((8, ), device=self.device),
                                                                    torch.ones((8, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["encouraged"] + \
                        sample_dict["beta"][..., 2] * sample_dict["pretest"] + \
                        sample_dict["beta"][..., 3] * sample_dict["site2"] + \
                        sample_dict["beta"][..., 4] * sample_dict["site3"] + \
                        sample_dict["beta"][..., 5] * sample_dict["site4"] + \
                        sample_dict["beta"][..., 6] * sample_dict["site5"] + \
                        sample_dict["beta"][..., 7] * sample_dict["setting"]
                sample_dict["watched"] = self.r_obs("watched",
                                                    dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                                    obs=sample_dict.get("watched", None))
        return sample_dict


class ARM_sesame_multi_preds_3b(BaseVAEwRegister):
    x_dim = 400
    theta_dim = 8

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["watched_hat"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 20
                sample_dict["setting"] = torch.randint(low=0, high=1,
                                                          size=(sample_dict["N"], batch_size),
                                                          device=self.device)
                sample_dict["pretest"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 2 + 10
                site = torch.randint(low=2, high=6,
                                    size=(sample_dict["N"], batch_size),
                                    device=self.device)
                sample_dict["site2"] = (site == 2).float()
                sample_dict["site3"] = (site == 3).float()
                sample_dict["site4"] = (site == 4).float()
                sample_dict["site5"] = (site == 5).float()
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((8, ), device=self.device),
                                                                    torch.ones((8, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["watched_hat"] + \
                        sample_dict["beta"][..., 2] * sample_dict["pretest"] + \
                        sample_dict["beta"][..., 3] * sample_dict["site2"] + \
                        sample_dict["beta"][..., 4] * sample_dict["site3"] + \
                        sample_dict["beta"][..., 5] * sample_dict["site4"] + \
                        sample_dict["beta"][..., 6] * sample_dict["site5"] + \
                        sample_dict["beta"][..., 7] * sample_dict["setting"]
                sample_dict["y"] = self.r_obs("y",
                                            dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                            obs=sample_dict.get("y", None))
        return sample_dict


class ARM_sesame_one_pred_2b(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["watched_hat"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 20
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((2, ), device=self.device),
                                                                    torch.ones((2, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["watched_hat"]
                sample_dict["y"] = self.r_obs("y",
                                            dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                            obs=sample_dict.get("y", None))
        return sample_dict


class ARM_sesame_one_pred_a(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["encouraged"] = torch.randint(low=0, high=1,
                                                          size=(sample_dict["N"], batch_size),
                                                          device=self.device)
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((2, ), device=self.device),
                                                                    torch.ones((2, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["encouraged"]
                sample_dict["y"] = self.r_obs("y",
                                            dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                            obs=sample_dict.get("y", None))
        return sample_dict


class ARM_unemployment(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["y_lag"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 3 + 10
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((2, ), device=self.device),
                                                                    torch.ones((2, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["y_lag"]
                sample_dict["y"] = self.r_obs("y",
                                            dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                            obs=sample_dict.get("y", None))
        return sample_dict


class ARM_weight(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                height = torch.randn(sample_dict["N"], batch_size, device=self.device) * 10 + 180
                sample_dict["c_height"] = height - torch.mean(height, dim=0)
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["a"] = self.r_sample("a", self.scalar_normal_dist(70.0, 5.0))
            sample_dict["b"] = self.r_sample("b", self.scalar_normal_dist(0.0, 0.05))
            with plates["plate_n"]:
                y_hat = sample_dict["a"] + \
                        sample_dict["b"] * sample_dict["c_height"]
                sample_dict["y"] = self.r_obs("y",
                                              dist.Normal(y_hat, torch.ones_like(y_hat) * 0.1),
                                              obs=sample_dict.get("y", None))
        return sample_dict


class ARM_wells_d100ars(BaseVAEwRegister):
    x_dim = 150
    theta_dim = 3

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                dist_ = torch.rand(sample_dict["N"], batch_size, device=self.device) * 100
                sample_dict["dist100"] = dist_ / 100
                sample_dict["arsenic"] = torch.rand(sample_dict["N"], batch_size, device=self.device) * 10 + 1
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((3, ), device=self.device),
                                                                    torch.ones((3, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                logits = sample_dict["beta"][..., 0] + \
                         sample_dict["beta"][..., 1] * sample_dict["dist100"] + \
                         sample_dict["beta"][..., 2] * sample_dict["arsenic"]
                sample_dict["switched"] = self.r_obs("switched",
                                                    dist.Bernoulli(logits=logits),
                                                    obs=sample_dict.get("switched", None))
        return sample_dict


class ARM_wells_daae_c(BaseVAEwRegister):
    x_dim = 300
    theta_dim = 6

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                dist_ = torch.rand(sample_dict["N"], batch_size, device=self.device) * 100
                arsenic = torch.rand(sample_dict["N"], batch_size, device=self.device) * 10 + 1
                educ = torch.randint(low=0, high=5,
                                     size=(sample_dict["N"], batch_size),
                                     device=self.device)
                sample_dict["c_dist100"] = (dist_ - torch.mean(dist_, dim=0)) / 100
                sample_dict["c_arsenic"] = arsenic - torch.mean(arsenic, dim=0)
                sample_dict["da_inter"] = sample_dict["c_dist100"] * sample_dict["c_arsenic"]
                sample_dict["assoc"] = torch.randn(sample_dict["N"], batch_size, device=self.device) * 2
                sample_dict["educ4"] = educ / 4
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((6, ), device=self.device),
                                                                    torch.ones((6, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                logits = sample_dict["beta"][..., 0] + \
                         sample_dict["beta"][..., 1] * sample_dict["c_dist100"] + \
                         sample_dict["beta"][..., 2] * sample_dict["c_arsenic"] + \
                         sample_dict["beta"][..., 3] * sample_dict["da_inter"] + \
                         sample_dict["beta"][..., 4] * sample_dict["assoc"] + \
                         sample_dict["beta"][..., 5] * sample_dict["educ4"]
                sample_dict["switched"] = self.r_obs("switched",
                                                    dist.Bernoulli(logits=logits),
                                                    obs=sample_dict.get("switched", None))
        return sample_dict


class ARM_wells_dae_c(BaseVAEwRegister):
    x_dim = 250
    theta_dim = 5

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                dist_ = torch.rand(sample_dict["N"], batch_size, device=self.device) * 100
                arsenic = torch.rand(sample_dict["N"], batch_size, device=self.device) * 10 + 1
                educ = torch.randint(low=0, high=5,
                                     size=(sample_dict["N"], batch_size),
                                     device=self.device)
                sample_dict["c_dist100"] = (dist_ - torch.mean(dist_, dim=0)) / 100
                sample_dict["c_arsenic"] = arsenic - torch.mean(arsenic, dim=0)
                sample_dict["da_inter"] = sample_dict["c_dist100"] * sample_dict["c_arsenic"]
                sample_dict["educ4"] = educ / 4
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((5, ), device=self.device),
                                                                    torch.ones((5, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                logits = sample_dict["beta"][..., 0] + \
                         sample_dict["beta"][..., 1] * sample_dict["c_dist100"] + \
                         sample_dict["beta"][..., 2] * sample_dict["c_arsenic"] + \
                         sample_dict["beta"][..., 3] * sample_dict["da_inter"] + \
                         sample_dict["beta"][..., 4] * sample_dict["educ4"]
                sample_dict["switched"] = self.r_obs("switched",
                                                    dist.Bernoulli(logits=logits),
                                                    obs=sample_dict.get("switched", None))
        return sample_dict


class ARM_wells_dae_inter_c(BaseVAEwRegister):
    x_dim = 350
    theta_dim = 7

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                dist_ = torch.rand(sample_dict["N"], batch_size, device=self.device) * 100
                arsenic = torch.rand(sample_dict["N"], batch_size, device=self.device) * 10 + 1
                educ = torch.randint(low=0, high=5,
                                     size=(sample_dict["N"], batch_size),
                                     device=self.device).float()
                sample_dict["c_dist100"] = (dist_ - torch.mean(dist_, dim=0)) / 100
                sample_dict["c_arsenic"] = arsenic - torch.mean(arsenic, dim=0)
                sample_dict["c_educ4"] = (educ - torch.mean(educ, dim=0)) / 4
                sample_dict["da_inter"] = sample_dict["c_dist100"] * sample_dict["c_arsenic"]
                sample_dict["de_inter"] = sample_dict["c_dist100"] * sample_dict["c_educ4"]
                sample_dict["ae_inter"] = sample_dict["c_arsenic"] * sample_dict["c_educ4"]
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((7, ), device=self.device),
                                                                    torch.ones((7, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                logits = sample_dict["beta"][..., 0] + \
                         sample_dict["beta"][..., 1] * sample_dict["c_dist100"] + \
                         sample_dict["beta"][..., 2] * sample_dict["c_arsenic"] + \
                         sample_dict["beta"][..., 3] * sample_dict["c_educ4"] + \
                         sample_dict["beta"][..., 4] * sample_dict["da_inter"] + \
                         sample_dict["beta"][..., 5] * sample_dict["de_inter"] + \
                         sample_dict["beta"][..., 6] * sample_dict["ae_inter"]
                sample_dict["switched"] = self.r_obs("switched",
                                                    dist.Bernoulli(logits=logits),
                                                    obs=sample_dict.get("switched", None))
        return sample_dict


class ARM_wells_dae_inter(BaseVAEwRegister):
    x_dim = 250
    theta_dim = 5

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                dist_ = torch.rand(sample_dict["N"], batch_size, device=self.device) * 100
                arsenic = torch.rand(sample_dict["N"], batch_size, device=self.device) * 10 + 1
                educ = torch.randint(low=0, high=5,
                                     size=(sample_dict["N"], batch_size),
                                     device=self.device)
                sample_dict["dist100"] = dist_ / 100
                sample_dict["arsenic"] = arsenic
                sample_dict["educ4"] = educ / 4
                sample_dict["inter"] = sample_dict["dist100"] * sample_dict["arsenic"]
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((5, ), device=self.device),
                                                                    torch.ones((5, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                logits = sample_dict["beta"][..., 0] + \
                         sample_dict["beta"][..., 1] * sample_dict["dist100"] + \
                         sample_dict["beta"][..., 2] * sample_dict["arsenic"] + \
                         sample_dict["beta"][..., 3] * sample_dict["educ4"] + \
                         sample_dict["beta"][..., 4] * sample_dict["inter"]
                sample_dict["switched"] = self.r_obs("switched",
                                                    dist.Bernoulli(logits=logits),
                                                    obs=sample_dict.get("switched", None))
        return sample_dict


class ARM_wells_dae(BaseVAEwRegister):
    x_dim = 200
    theta_dim = 4

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                dist_ = torch.rand(sample_dict["N"], batch_size, device=self.device) * 100
                arsenic = torch.rand(sample_dict["N"], batch_size, device=self.device) * 10 + 1
                educ = torch.randint(low=0, high=5,
                                     size=(sample_dict["N"], batch_size),
                                     device=self.device)
                sample_dict["dist100"] = dist_ / 100
                sample_dict["arsenic"] = arsenic
                sample_dict["educ4"] = educ / 4
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((4, ), device=self.device),
                                                                    torch.ones((4, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                logits = sample_dict["beta"][..., 0] + \
                         sample_dict["beta"][..., 1] * sample_dict["dist100"] + \
                         sample_dict["beta"][..., 2] * sample_dict["arsenic"] + \
                         sample_dict["beta"][..., 3] * sample_dict["educ4"]
                sample_dict["switched"] = self.r_obs("switched",
                                                    dist.Bernoulli(logits=logits),
                                                    obs=sample_dict.get("switched", None))
        return sample_dict


class ARM_wells_dist(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["dist_"] = torch.rand(sample_dict["N"], batch_size, device=self.device) * 100
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((2, ), device=self.device),
                                                                    torch.ones((2, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                logits = sample_dict["beta"][..., 0] + \
                         sample_dict["beta"][..., 1] * sample_dict["dist_"]
                sample_dict["switched"] = self.r_obs("switched",
                                                    dist.Bernoulli(logits=logits),
                                                    obs=sample_dict.get("switched", None))
        return sample_dict


class ARM_wells_dist100(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                dist_ = torch.rand(sample_dict["N"], batch_size, device=self.device) * 100
                sample_dict["dist100"] = dist_ / 100
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((2, ), device=self.device),
                                                                    torch.ones((2, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                logits = sample_dict["beta"][..., 0] + \
                         sample_dict["beta"][..., 1] * sample_dict["dist100"]
                sample_dict["switched"] = self.r_obs("switched",
                                                    dist.Bernoulli(logits=logits),
                                                    obs=sample_dict.get("switched", None))
        return sample_dict


class ARM_wells_interaction_c(BaseVAEwRegister):
    x_dim = 200
    theta_dim = 4

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                dist_ = torch.rand(sample_dict["N"], batch_size, device=self.device) * 100
                arsenic = torch.rand(sample_dict["N"], batch_size, device=self.device) * 10 + 1
                sample_dict["c_dist100"] = (dist_ - torch.mean(dist_, dim=0)) / 100
                sample_dict["c_arsenic"] = arsenic - torch.mean(arsenic, dim=0)
                sample_dict["inter"] = sample_dict["c_dist100"] * sample_dict["c_arsenic"]
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((4, ), device=self.device),
                                                                    torch.ones((4, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                logits = sample_dict["beta"][..., 0] + \
                         sample_dict["beta"][..., 1] * sample_dict["c_dist100"] + \
                         sample_dict["beta"][..., 2] * sample_dict["c_arsenic"] + \
                         sample_dict["beta"][..., 3] * sample_dict["inter"]
                sample_dict["switched"] = self.r_obs("switched",
                                                    dist.Bernoulli(logits=logits),
                                                    obs=sample_dict.get("switched", None))
        return sample_dict


class ARM_wells_interaction(BaseVAEwRegister):
    x_dim = 200
    theta_dim = 4

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["dist_"] = torch.rand(sample_dict["N"], batch_size, device=self.device) * 100
                sample_dict["arsenic"] = torch.rand(sample_dict["N"], batch_size, device=self.device) * 10 + 1
                sample_dict["inter"] = sample_dict["dist_"] * sample_dict["arsenic"]
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((4, ), device=self.device),
                                                                    torch.ones((4, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                logits = sample_dict["beta"][..., 0] + \
                         sample_dict["beta"][..., 1] * sample_dict["dist_"] + \
                         sample_dict["beta"][..., 2] * sample_dict["arsenic"] + \
                         sample_dict["beta"][..., 3] * sample_dict["inter"]
                sample_dict["switched"] = self.r_obs("switched",
                                                    dist.Bernoulli(logits=logits),
                                                    obs=sample_dict.get("switched", None))
        return sample_dict


class ARM_wells_logit(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                dist_ = torch.rand(sample_dict["N"], batch_size, device=self.device) * 100
                sample_dict["dist100"] = dist_ / 100
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((2, ), device=self.device),
                                                                    torch.ones((2, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                logits = sample_dict["beta"][..., 0] + \
                         sample_dict["beta"][..., 1] * sample_dict["dist100"]
                sample_dict["switched"] = self.r_obs("switched",
                                                    dist.Bernoulli(logits=logits),
                                                    obs=sample_dict.get("switched", None))
        return sample_dict


class ARM_wells_predicted(ARM_wells_dae_inter_c):
    pass


class ARM_wells_probit(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                dist_ = torch.rand(sample_dict["N"], batch_size, device=self.device) * 100
                sample_dict["dist100"] = dist_ / 100
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((2, ), device=self.device),
                                                                    torch.ones((2, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                phi = self.scalar_normal_dist(0.0, 1.0).cdf
                logits = phi(sample_dict["beta"][..., 0] + \
                             sample_dict["beta"][..., 1] * sample_dict["dist100"])
                sample_dict["switched"] = self.r_obs("switched",
                                                    dist.Bernoulli(logits=logits),
                                                    obs=sample_dict.get("switched", None))
        return sample_dict


class ARM_wells(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["dist_"] = torch.rand(sample_dict["N"], batch_size, device=self.device) * 100
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((2, ), device=self.device),
                                                                    torch.ones((2, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                logits = sample_dict["beta"][..., 0] + \
                         sample_dict["beta"][..., 1] * sample_dict["dist_"]
                sample_dict["switched"] = self.r_obs("switched",
                                                    dist.Bernoulli(logits=logits),
                                                    obs=sample_dict.get("switched", None))
        return sample_dict


class ARM_y_x(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                sample_dict["x"] = torch.randn(sample_dict["N"], batch_size, device=self.device)
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["beta"] = self.r_sample("beta", dist.Normal(torch.zeros((2, ), device=self.device),
                                                                    torch.ones((2, ), device=self.device) * 0.2).to_event(1))
            with plates["plate_n"]:
                y_hat = sample_dict["beta"][..., 0] + \
                        sample_dict["beta"][..., 1] * sample_dict["x"]
                sample_dict["y"] = self.r_obs("y",
                                            dist.Normal(y_hat,
                                                        torch.ones_like(y_hat) * 0.1),
                                            obs=sample_dict.get("y", None))
        return sample_dict


class BUGS_beatles_probit(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n"] = 100
            with DataContext(sample_dict, self):
                sample_dict["centered_x"] = torch.randn(sample_dict["N"], batch_size, device=self.device)
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["alpha_star"] = self.r_sample("alpha_star", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["beta"] = self.r_sample("beta", self.scalar_normal_dist(0.0, 10000.0))
            with plates["plate_n"]:
                p = self.scalar_normal_dist(0.0, 1.0).cdf(sample_dict["alpha_star"] + sample_dict["beta"] * sample_dict["centered_x"])
                sample_dict["r"] = self.r_obs("r",
                                            dist.Binomial(sample_dict["n"], p),
                                            obs=sample_dict.get("r", None))
        return sample_dict


class BUGS_dyes(BaseVAEwRegister):
    x_dim = 50
    theta_dim = 2

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
            with DataContext(sample_dict, self):
                pass
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["theta"] = self.r_sample("theta", self.scalar_normal_dist(0.0, 1e5))
            sample_dict["mu"] = self.r_sample("mu", dist.Normal(sample_dict["theta"],
                                                                torch.ones_like(sample_dict["theta"]) * 1000))
            with plates["plate_n"]:
                sample_dict["y"] = self.r_obs("y",
                                            dist.Normal(sample_dict["mu"],
                                                        torch.ones_like(sample_dict["mu"]) * 20),
                                            obs=sample_dict.get("y", None))
        return sample_dict


class BUGS_lsat(BaseVAEwRegister):
    x_dim = 100
    theta_dim = 3

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 100
            with DataContext(sample_dict, self):
                pass
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            sample_dict["alpha"] = self.r_sample("alpha", self.scalar_normal_dist(0.0, 100.0))
            sample_dict["theta"] = self.r_sample("theta", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["beta"] = self.r_sample("beta", self.scalar_normal_dist(0.0, 100.0))
            with plates["plate_n"]:
                sample_dict["r"] = self.r_obs("r",
                                            dist.Bernoulli(logits=sample_dict["beta"] * sample_dict["theta"] - sample_dict["alpha"]),
                                            obs=sample_dict.get("r", None))
        return sample_dict


class MISC_irt_multilevel(BaseVAEwRegister):
    x_dim = 50
    theta_dim = 8

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_alpha": self.plate("plate_alpha", sample_dict["n_alpha"], dim=-2),
            "plate_beta": self.plate("plate_beta", sample_dict["n_beta"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_alpha"] = 2
                sample_dict["n_beta"] = 5
                total_permutate = sample_dict["n_alpha"] * sample_dict["n_beta"]
                assert sample_dict["N"] % total_permutate == 0
                r = sample_dict["N"] // total_permutate
                alpha_beta_mesh = torch.stack(
                    torch.meshgrid(torch.arange(sample_dict["n_alpha"], device=self.device),
                                   torch.arange(sample_dict["n_beta"], device=self.device),
                                   indexing="ij"),
                dim=-1)
                sample_dict["alpha_order"] = repeat(alpha_beta_mesh[..., 0].reshape(-1),
                                                    "k -> (r k)", r=r)
                sample_dict["beta_order"] = repeat(alpha_beta_mesh[..., 1].reshape(-1),
                                                   "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                pass
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            with plates["plate_alpha"]:
                sample_dict["alpha"] = self.r_sample("alpha", self.scalar_normal_dist(0.0, 10.0))
            with plates["plate_beta"]:
                sample_dict["beta"] = self.r_sample("beta", self.scalar_normal_dist(0.0, 5.0))
            sample_dict["delta"] = self.r_sample("delta", self.scalar_normal_dist(0.75, 1.0))
            with plates["plate_n"]:
                logits = sample_dict["alpha"][..., sample_dict["alpha_order"], :] - \
                         sample_dict["beta"][..., sample_dict["beta_order"], :] + \
                         sample_dict["delta"]
                sample_dict["y"] = self.r_obs("y",
                                            dist.Bernoulli(logits=logits),
                                            obs=sample_dict.get("y", None))
        return sample_dict


class MISC_irt(BaseVAEwRegister):
    x_dim = 50
    theta_dim = 8

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_alpha": self.plate("plate_alpha", sample_dict["n_alpha"], dim=-2),
            "plate_beta": self.plate("plate_beta", sample_dict["n_beta"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_alpha"] = 2
                sample_dict["n_beta"] = 5
                total_permutate = sample_dict["n_alpha"] * sample_dict["n_beta"]
                assert sample_dict["N"] % total_permutate == 0
                r = sample_dict["N"] // total_permutate
                alpha_beta_mesh = torch.stack(
                    torch.meshgrid(torch.arange(sample_dict["n_alpha"], device=self.device),
                                   torch.arange(sample_dict["n_beta"], device=self.device),
                                   indexing="ij"),
                dim=-1)
                sample_dict["alpha_order"] = repeat(alpha_beta_mesh[..., 0].reshape(-1),
                                                    "k -> (r k)", r=r)
                sample_dict["beta_order"] = repeat(alpha_beta_mesh[..., 1].reshape(-1),
                                                   "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                pass
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            with plates["plate_alpha"]:
                sample_dict["alpha"] = self.r_sample("alpha", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_beta"]:
                sample_dict["beta"] = self.r_sample("beta", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["delta"] = self.r_sample("delta", self.scalar_normal_dist(0.75, 1.0))
            with plates["plate_n"]:
                logits = sample_dict["alpha"][..., sample_dict["alpha_order"], :] - \
                         sample_dict["beta"][..., sample_dict["beta_order"], :] + \
                         sample_dict["delta"]
                sample_dict["y"] = self.r_obs("y",
                                            dist.Bernoulli(logits=logits),
                                            obs=sample_dict.get("y", None))
        return sample_dict


class MISC_irt2_multilevel(BaseVAEwRegister):
    x_dim = 50
    theta_dim = 13

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_alpha": self.plate("plate_alpha", sample_dict["n_alpha"], dim=-2),
            "plate_beta": self.plate("plate_beta", sample_dict["n_beta"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_alpha"] = 2
                sample_dict["n_beta"] = 5
                total_permutate = sample_dict["n_alpha"] * sample_dict["n_beta"]
                assert sample_dict["N"] % total_permutate == 0
                r = sample_dict["N"] // total_permutate
                alpha_beta_mesh = torch.stack(
                    torch.meshgrid(torch.arange(sample_dict["n_alpha"], device=self.device),
                                   torch.arange(sample_dict["n_beta"], device=self.device),
                                   indexing="ij"),
                dim=-1)
                sample_dict["alpha_order"] = repeat(alpha_beta_mesh[..., 0].reshape(-1),
                                                    "k -> (r k)", r=r)
                sample_dict["beta_order"] = repeat(alpha_beta_mesh[..., 1].reshape(-1),
                                                   "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                pass
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            with plates["plate_alpha"]:
                sample_dict["alpha"] = self.r_sample("alpha", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_beta"]:
                sample_dict["beta"] = self.r_sample("beta", self.scalar_normal_dist(0.0, 1.0))
                sample_dict["log_gamma"] = self.r_sample("log_gamma", self.scalar_normal_dist(0.0, 0.5))
            sample_dict["delta"] = self.r_sample("delta", self.scalar_normal_dist(0.75, 1.0))
            with plates["plate_n"]:
                logits = sample_dict["log_gamma"][..., sample_dict["beta_order"], :].exp() * \
                        (sample_dict["alpha"][..., sample_dict["alpha_order"], :] - \
                         sample_dict["beta"][..., sample_dict["beta_order"], :] + \
                         sample_dict["delta"])
                sample_dict["y"] = self.r_obs("y",
                                            dist.Bernoulli(logits=logits),
                                            obs=sample_dict.get("y", None))
        return sample_dict


class MISC_irt2(BaseVAEwRegister):
    x_dim = 50
    theta_dim = 13

    def get_plates(self, batch_size, sample_dict):
        return {
            "plate_batch": self.plate("plate_batch", batch_size, dim=-1),
            "plate_n": self.plate("plate_n", sample_dict["N"], dim=-2),
            "plate_alpha": self.plate("plate_alpha", sample_dict["n_alpha"], dim=-2),
            "plate_beta": self.plate("plate_beta", sample_dict["n_beta"], dim=-2),
        }

    def model(self, batch_size, sample_dict):
        if sample_dict is None:
            sample_dict = SampleDict()
            with MetaDataContext(sample_dict, self):
                sample_dict["N"] = 50
                sample_dict["n_alpha"] = 2
                sample_dict["n_beta"] = 5
                total_permutate = sample_dict["n_alpha"] * sample_dict["n_beta"]
                assert sample_dict["N"] % total_permutate == 0
                r = sample_dict["N"] // total_permutate
                alpha_beta_mesh = torch.stack(
                    torch.meshgrid(torch.arange(sample_dict["n_alpha"], device=self.device),
                                   torch.arange(sample_dict["n_beta"], device=self.device),
                                   indexing="ij"),
                dim=-1)
                sample_dict["alpha_order"] = repeat(alpha_beta_mesh[..., 0].reshape(-1),
                                                    "k -> (r k)", r=r)
                sample_dict["beta_order"] = repeat(alpha_beta_mesh[..., 1].reshape(-1),
                                                   "k -> (r k)", r=r)
            with DataContext(sample_dict, self):
                pass
        else:
            sample_dict = copy.copy(sample_dict)

        plates = self.get_plates(batch_size, sample_dict)
        with plates["plate_batch"]:
            with plates["plate_alpha"]:
                sample_dict["alpha"] = self.r_sample("alpha", self.scalar_normal_dist(0.0, 1.0))
            with plates["plate_beta"]:
                sample_dict["beta"] = self.r_sample("beta", self.scalar_normal_dist(0.0, 1.0))
                sample_dict["log_gamma"] = self.r_sample("log_gamma", self.scalar_normal_dist(0.0, 1.0))
            sample_dict["delta"] = self.r_sample("delta", self.scalar_normal_dist(0.75, 1.0))
            with plates["plate_n"]:
                logits = sample_dict["log_gamma"][..., sample_dict["beta_order"], :].exp() * \
                        (sample_dict["alpha"][..., sample_dict["alpha_order"], :] - \
                         sample_dict["beta"][..., sample_dict["beta_order"], :] + \
                         sample_dict["delta"])
                sample_dict["y"] = self.r_obs("y",
                                            dist.Bernoulli(logits=logits),
                                            obs=sample_dict.get("y", None))
        return sample_dict
