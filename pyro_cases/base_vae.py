import torch
import math
from torch import nn
import pyro
import copy
import pyro.distributions as dist
import contextlib

from einops import rearrange
from typing import Dict
from collections import OrderedDict, UserDict


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
    

class PlateTracker:
    def __init__(self):
        self.plate_stack = []

    def enter_plate(self, name):
        self.plate_stack.append(name)

    def exit_plate(self, name):
        popped = self.plate_stack.pop()
        assert popped == name, f"plate stack mismatch: expected {name}, got {popped}"

    def get_plate_stack(self):
        return copy.deepcopy(self.plate_stack)
    

class WrappedPlate:
    def __init__(self, 
                 tracker: PlateTracker, 
                 plate_name: str, 
                 plate_size: int, 
                 plate_dim: int, 
                 **plate_kwargs):
        self.tracker = tracker
        self.plate = pyro.plate(plate_name, plate_size, dim=plate_dim, **plate_kwargs)
        self._cm = None

    def __enter__(self):
        self.tracker.enter_plate(self.plate.name)
        self._cm = self.plate.__enter__()
        return self._cm

    def __exit__(self, exc_type, exc_val, traceback):
        self.plate.__exit__(exc_type, exc_val, traceback)
        self.tracker.exit_plate(self.plate.name)


class BaseVAEwRegister(BaseVAE):    
    def __init__(self, hidden_dim):
        super().__init__(hidden_dim)
        self.sample_dict_instr = {
            "meta_data": [],
            "data": [],
            "latent": OrderedDict(),
            "obs": OrderedDict(),
        }
        self.plate_tracker = PlateTracker()
        self.already_registered = False

    def plate(self, name, size, dim, **plate_kwargs):
        return WrappedPlate(self.plate_tracker, 
                            name,
                            size,
                            dim,
                            **plate_kwargs)

    def get_plates(self, batch_size, sample_dict) -> Dict[str, pyro.plate]:
        raise NotImplementedError()

    def guide(self, batch_size, sample_dict: Dict[str, torch.Tensor]):
        pyro.module("encoder", self.encoder)
        x = self.extract_x(sample_dict)
        assert batch_size == x.shape[0]

        theta_loc, theta_scale = self.encoder(x)

        plates = self.get_plates(batch_size, sample_dict)
        for kp, p in plates.items():
            assert isinstance(p, WrappedPlate), f"plate {kp} is not WrappedPlate"
        cur_size = 0
        for latent_name, latent_details in self.sample_dict_instr["latent"].items():
            cur_plates = [plates[p] for p in latent_details["plates"]]
            with contextlib.ExitStack() as s:
                cms = [s.enter_context(p) for p in cur_plates]
                event_shape = OrderedDict([(f"e{i}", e) for i, e in enumerate(latent_details["event_shape"])])
                event_tag = " ".join(list(event_shape.keys()))
                batch_shape = OrderedDict([(f"b{i}", b) for i, b in enumerate(latent_details["batch_shape"][:-1])])
                batch_tag = " ".join(list(batch_shape.keys()))
                if event_tag == "" and batch_tag == "":
                    distri = dist.Normal(theta_loc[:, cur_size], theta_scale[:, cur_size])
                else:
                    distri = dist.Normal(
                        loc=rearrange(theta_loc[:, cur_size:(cur_size + latent_details["latent_size"])],
                                    f"b ({batch_tag} {event_tag}) -> {batch_tag} b {event_tag}",
                                    **event_shape, **batch_shape),
                        scale=rearrange(theta_scale[:, cur_size:(cur_size + latent_details["latent_size"])],
                                        f"b ({batch_tag} {event_tag}) -> {batch_tag} b {event_tag}",
                                        **event_shape, **batch_shape),
                    )
                if len(latent_details["event_shape"]) > 0:
                    distri = distri.to_event(len(latent_details["event_shape"]))
                pyro.sample(latent_name, distri)
                cur_size += latent_details["latent_size"]
    
    def _extract_x_func(self, sample_dict):
        x = []
        for data_name in self.sample_dict_instr["data"]:
            x.append(rearrange(sample_dict[data_name], "... b -> b (...)"))
        for obs_name, obs_details in self.sample_dict_instr["obs"].items():
            event_shape = " ".join([f"e{i}" for i in range(len(obs_details["event_shape"]))])
            x.append(rearrange(sample_dict[obs_name], f"... b {event_shape} -> b (... {event_shape})"))
        return torch.cat(x, dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        theta = []
        for latent_name, latent_details in self.sample_dict_instr["latent"].items():
            event_shape = " ".join([f"e{i}" for i in range(len(latent_details["event_shape"]))])
            cur_theta = rearrange(sample_dict[latent_name], f"... b {event_shape} -> b (... {event_shape})")
            theta.append(cur_theta)
            if not self.already_registered:
                self.sample_dict_instr["latent"][latent_name]["latent_size"] = cur_theta.shape[-1]
            else:
                assert self.sample_dict_instr["latent"][latent_name]["latent_size"] is not None
        return torch.cat(theta, dim=-1)

    def r_data(self, name, data):
        if not self.already_registered:
            assert name not in self.sample_dict_instr["data"]
            self.sample_dict_instr["data"].append(name)
        else:
            assert name in self.sample_dict_instr["data"]
        return data
    
    def r_meta_data(self, name, meta_data):
        if not self.already_registered:
            assert name not in self.sample_dict_instr["meta_data"]
            self.sample_dict_instr["meta_data"].append(name)
        else:
            assert name in self.sample_dict_instr["meta_data"]
        return meta_data
    
    def r_sample(self, name, sample_dist):
        if not self.already_registered:
            plates = self.plate_tracker.get_plate_stack()
            event_shape = list(sample_dist.event_shape)
            sample = pyro.sample(name, sample_dist)
            batch_shape = list(sample.shape) if len(event_shape) == 0 else list(sample.shape[:-len(event_shape)])
            assert name not in self.sample_dict_instr["latent"]
            self.sample_dict_instr["latent"][name] = {
                "dist": type(sample_dist),
                "batch_shape": batch_shape,
                "event_shape": event_shape,
                "latent_size": None,
                "plates": plates
            }
        else:
            sample = pyro.sample(name, sample_dist)
            assert name in self.sample_dict_instr["latent"]
        return sample
    
    def r_obs(self, name, sample_dist, obs):
        if not self.already_registered:
            plates = self.plate_tracker.get_plate_stack()
            event_shape = list(sample_dist.event_shape)
            sample = pyro.sample(name, sample_dist, obs=obs)
            batch_shape = list(sample.shape) if len(event_shape) == 0 else list(sample.shape[:-len(event_shape)])
            assert name not in self.sample_dict_instr["obs"]
            self.sample_dict_instr["obs"][name] = {
                "dist": type(sample_dist),
                "batch_shape": batch_shape,
                "event_shape": event_shape,
                "plates": plates
            }
        else:
            sample = pyro.sample(name, sample_dist, obs=obs)
            assert name in self.sample_dict_instr["obs"]
        return sample

    def do_register(self, batch_size):
        sample_dict = self.generate_sample_dict(batch_size)
        self.extract_theta(sample_dict)
        self.already_registered = True
    
    def scalar_normal_dist(self, loc, scale):
        return dist.Normal(loc=torch.tensor(loc, device=self.device),
                           scale=torch.tensor(scale, device=self.device))
    

class SampleDict(UserDict):
    def __init__(self, vae: BaseVAEwRegister):
        super().__init__()
        self.vae = vae
        assert isinstance(self.vae, BaseVAEwRegister)
        self.context_flag = "no_context"

    def __setitem__(self, key, item):
        match self.context_flag:
            case "meta_data":
                self.vae.r_meta_data(key, item)
            case "data":
                self.vae.r_data(key, item)
            case "no_context":
                pass
            case _:
                raise ValueError("context flag is invalid")
        return super().__setitem__(key, item)
    

class MetaDataContext:
    def __init__(self, sample_dict: SampleDict):
        self.sample_dict = sample_dict
        assert isinstance(self.sample_dict, SampleDict)

    def __enter__(self):
        assert self.sample_dict.context_flag == "no_context"
        self.sample_dict.context_flag = "meta_data"
        return self.sample_dict
    
    def __exit__(self, exc_type, exc_val, traceback):
        assert self.sample_dict.context_flag == "meta_data"
        self.sample_dict.context_flag = "no_context"


class DataContext:
    def __init__(self, sample_dict: SampleDict):
        self.sample_dict = sample_dict
        assert isinstance(self.sample_dict, SampleDict)

    def __enter__(self):
        assert self.sample_dict.context_flag == "no_context"
        self.sample_dict.context_flag = "data"
        return self.sample_dict
    
    def __exit__(self, exc_type, exc_val, traceback):
        assert self.sample_dict.context_flag == "data"
        self.sample_dict.context_flag = "no_context"
